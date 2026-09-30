"""Ordered, bounded CPU preprocessing for AIRBOT training.

Workers are fresh CPU-only interpreters, never forks of a live JAX/CUDA process.
Only small task descriptors travel over pipes; arrays use parent-owned shared
memory. A slot remains immutable until the consumer explicitly releases it.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import dataclasses
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import threading

import numpy as np


def validate_loader_settings(runtime, batch_size):
    workers = runtime.get("data_workers", 0)
    prefetch = runtime.get("prefetch_batches", 2)
    if type(workers) is not int or workers < 0:
        raise ValueError("runtime.data_workers must be a nonnegative integer")
    if workers:
        if workers > batch_size:
            raise ValueError("data_workers must not exceed batch_size")
        if type(prefetch) is not int or not 1 <= prefetch <= 4:
            raise ValueError("prefetch_batches must be an integer in [1, 4]")
        cpus = runtime.get("data_worker_cpus", [])
        if not cpus or any(type(c) is not int or c < 0 for c in cpus) or len(set(cpus)) != len(cpus):
            raise ValueError("data_worker_cpus must contain unique CPU indices")
    return workers, prefetch


class ParallelAirbotLoader:
    """Prepare batches in parallel, preserving exactly the caller's sample order.

    Call start(), then get(step)/release(step) in order. The consumer must finish
    reading all returned arrays (including asynchronous device transfers) before
    release(). close() terminates workers and frees all temporary resources.
    """

    def __init__(self, replay, model_config, actor_norm_stats, critic_norm_stats,
                 config, template, sample_step, directory, *, timeout=180):
        import jax

        self.workers, self.depth = validate_loader_settings(config["runtime"], config["batch_size"])
        if not self.workers:
            raise ValueError("ParallelAirbotLoader requires at least one worker")
        self.batch_size = config["batch_size"]
        self.sample_step = sample_step
        self.timeout = timeout
        self.pending = {}
        self.borrowed = None
        self.processes = []
        self.handles = []
        self.views = []
        self.logs = []
        self.pool = None
        self.temp = None
        self.closed = False
        self.started = False
        self.worker_pids = []
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.temp = tempfile.TemporaryDirectory(prefix=".loader-", dir=directory)
            work = Path(self.temp.name)
            snapshot = copy.copy(replay)
            # Remove the parent-local LRU closure and decoded image/video handles.
            snapshot.__dict__.pop("images", None)
            snapshot._image_cache = OrderedDict()  # noqa: SLF001
            snapshot._video_cache = OrderedDict()  # noqa: SLF001
            with (work / "replay.pkl").open("wb") as f:
                pickle.dump({"replay": snapshot, "model_config": model_config,
                    "actor_norm_stats": actor_norm_stats, "critic_norm_stats": critic_norm_stats,
                    "config": config}, f, protocol=5)
            arrays, self.treedef = jax.tree.flatten(template)
            specs = []
            size = 0
            for value in arrays:
                array = np.asarray(value)
                shape = (self.batch_size, *array.shape[1:])
                size = (size + 63) // 64 * 64
                specs.append({"shape": shape, "dtype": array.dtype.str, "offset": size})
                size += int(np.prod(shape)) * array.dtype.itemsize
            slots = []
            for _ in range(self.depth):
                handle = shared_memory.SharedMemory(create=True, size=size)
                self.handles.append(handle)
                slots.append(handle.name)
                self.views.append([np.ndarray(s["shape"], dtype=s["dtype"],
                    buffer=handle.buf, offset=s["offset"]) for s in specs])
            self.buffer_bytes = size * self.depth
            (work / "buffers.json").write_text(json.dumps({"specs": specs, "slots": slots}))
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
            self.locks = [threading.Lock() for _ in range(self.workers)]
            cpus = ",".join(map(str, config["runtime"]["data_worker_cpus"]))
            for index in range(self.workers):
                log = (directory / f"loader_worker_{index}.log").open("a", buffering=1)
                self.logs.append(log)
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    "--worker", str(work), cpus], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=log, text=True, bufsize=1, env=env)
                self.processes.append(process)
                self.worker_pids.append(process.pid)
            self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="airbot-loader")
            ready = [self.pool.submit(self._read, i) for i in range(self.workers)]
            for f in ready:
                result = f.result(timeout=self.timeout)
                if result.get("status") != "ready" or result.get("backend") != "cpu":
                    raise RuntimeError(f"Loader worker failed CPU initialization: {result}")
        except BaseException:
            self.close()
            raise

    def _read(self, index):
        line = self.processes[index].stdout.readline()
        if not line:
            raise RuntimeError(f"Data worker {index} exited; inspect loader_worker_{index}.log")
        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(f"Data worker {index}: {result['error']}")
        return result

    def _request(self, index, request):
        with self.locks[index]:
            process = self.processes[index]
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            result = self._read(index)
            if result.get("step") != request["step"] or result.get("slot") != request["slot"]:
                raise RuntimeError("Data worker response does not match the requested batch")
            return result

    def _submit(self, step, slot):
        chunks = self.sample_step(step)
        if len(chunks) != self.batch_size:
            raise ValueError("Sampled batch size differs from configured batch_size")
        cuts = np.linspace(0, self.batch_size, self.workers + 1, dtype=int)
        futures = []
        for worker in range(self.workers):
            begin, end = map(int, cuts[worker:worker+2])
            futures.append(self.pool.submit(self._request, worker, {
                "step": step, "slot": slot, "begin": begin,
                "chunks": [dataclasses.asdict(c) for c in chunks[begin:end]],
            }))
        self.pending[step] = (slot, futures)

    def start(self, first_step, last_step):
        if self.started:
            raise RuntimeError("Loader has already been started")
        self.started = True
        self.next_get = first_step
        self.last_step = last_step
        self.next_submit = first_step
        for slot in range(self.depth):
            if self.next_submit <= last_step:
                self._submit(self.next_submit, slot)
                self.next_submit += 1

    def get(self, step):
        if self.borrowed is not None or step != self.next_get:
            raise RuntimeError("Release each batch before requesting the next step")
        slot, futures = self.pending[step]
        for f in futures:
            f.result(timeout=self.timeout)
        self.borrowed = step
        return self.treedef.unflatten(self.views[slot])

    def release(self, step):
        if self.borrowed != step:
            raise RuntimeError("Only the currently borrowed batch can be released")
        slot, _ = self.pending.pop(step)
        self.borrowed = None
        self.next_get += 1
        if self.next_submit <= self.last_step:
            self._submit(self.next_submit, slot)
            self.next_submit += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Terminate before waiting for pipe reader threads, including failed jobs.
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
        for p in self.processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        for p in self.processes:
            p.stdin.close()
            p.stdout.close()
        self.pending.clear()
        self.views.clear()
        for handle in self.handles:
            handle.close()
            handle.unlink()
        for log in self.logs:
            log.close()
        if self.temp is not None:
            self.temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _worker_main(work, cpus):
    os.sched_setaffinity(0, set(map(int, cpus.split(","))))
    import functools
    from multiprocessing import resource_tracker
    import traceback

    import jax
    import torch

    from openpi.training.airbot_lwd_data import Chunk
    from openpi.training.airbot_lwd_data import make_batch_builder

    handles = []
    try:
        torch.set_num_threads(1)
        if jax.default_backend() != "cpu":
            raise RuntimeError("Data workers must never initialize GPU execution")
        work = Path(work)
        with (work / "replay.pkl").open("rb") as f:
            state = pickle.load(f)
        replay, model, config = state["replay"], state["model_config"], state["config"]
        replay.images = functools.lru_cache(maxsize=4096)(replay.images)
        builder = make_batch_builder(replay, model, state["actor_norm_stats"], state["critic_norm_stats"], config)
        schema = json.loads((work / "buffers.json").read_text())
        views = []
        for name in schema["slots"]:
            handle = shared_memory.SharedMemory(name=name)
            handles.append(handle)
            # Each standalone interpreter has its own tracker. Only the owning
            # parent may unlink these buffers; a worker exit must not delete them.
            resource_tracker.unregister(handle._name, "shared_memory")  # noqa: SLF001
            views.append([np.ndarray(s["shape"], dtype=s["dtype"], buffer=handle.buf,
                offset=s["offset"]) for s in schema["specs"]])
        print(json.dumps({"status": "ready", "backend": "cpu", "pid": os.getpid()}), flush=True)
        for line in sys.stdin:
            task = json.loads(line)
            chunks = [Chunk(**c) for c in task["chunks"]]
            batch = builder(chunks)
            values = jax.tree.leaves(batch)
            for destination, source in zip(views[task["slot"]], values, strict=True):
                destination[task["begin"]:task["begin"] + len(chunks)] = np.asarray(source)
            print(json.dumps({"step": task["step"], "slot": task["slot"]}), flush=True)
    except BaseException as error:
        traceback.print_exc()
        print(json.dumps({"error": repr(error)}), flush=True)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("This module is launched by ParallelAirbotLoader")
    _worker_main(sys.argv[2], sys.argv[3])
