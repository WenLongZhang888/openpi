"""Unified pure BC / offline LWD for AIRBOT; see examples/airbot/lwd_offline.md."""

import argparse
import contextlib
import dataclasses
import fcntl
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

# Establish runtime before importing JAX or torch's shared libraries.
if __name__ == "__main__":
    import yaml as _yaml

    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--config", type=Path, required=True)
    _early, _ = _parser.parse_known_args()
    _config_path = _early.config.resolve()
    _runtime = _yaml.safe_load(_config_path.read_text())["runtime"]
    os.chdir(Path(__file__).resolve().parents[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, _runtime["gpus"]))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(_runtime.get("omp_num_threads", 8))
    os.environ["OPENPI_DATA_HOME"] = str(Path.cwd() / ".cache")
    os.environ["HF_HUB_CACHE"] = str(Path.cwd() / ".cache/huggingface/hub")
    os.environ["LD_LIBRARY_PATH"] = ":".join(_runtime["library_paths"])
    os.environ["PYTHONPATH"] = ":".join(_runtime["python_paths"])
    if "--check" in sys.argv or "--prepare-only" in sys.argv:
        os.environ["JAX_PLATFORMS"] = "cpu"
    if os.environ.get("OPENPI_AIRBOT_ENV_READY") != str(_config_path):
        os.environ["OPENPI_AIRBOT_ENV_READY"] = str(_config_path)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])

# Limit host compilation/thread pools before importing JAX.
if __name__ == "__main__" and hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: _runtime.get("cpu_threads", 16)])

import flax.nnx as nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import airbot_lwd_run
from openpi.training import divl_lwd_training
from openpi.training import optimizer as optimizer_lib
from openpi.training.airbot_bc_training import FrameSampler
from openpi.training.airbot_bc_training import bc_train_step
from openpi.training.airbot_lwd_data import AirbotDataConfig
from openpi.training.airbot_lwd_data import AirbotReplay
from openpi.training.airbot_lwd_data import make_batch_builder
from openpi.training.airbot_parallel_loader import ParallelAirbotLoader
from openpi.training.airbot_parallel_loader import validate_loader_settings
from openpi.training.airbot_wandb import WandbLogger
from openpi.training.qam_cached_training import actor_train_step as cached_actor_train_step
from openpi.training.qam_training import DIVLTrainingConfig
from openpi.training.qam_training import QAMTrainingConfig
from openpi.training.qam_training import initialize_actor_training
from openpi.training.qam_training import offline_train_step


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


IMPLEMENTATION_SOURCES = {
    "entrypoint": "scripts/train_airbot_lwd.py",
    "bc_training": "src/openpi/training/airbot_bc_training.py",
    "wandb_logging": "src/openpi/training/airbot_wandb.py",
    "data": "src/openpi/training/airbot_lwd_data.py",
    "parallel_loader": "src/openpi/training/airbot_parallel_loader.py",
    "run_state": "src/openpi/training/airbot_lwd_run.py",
    "critic": "src/openpi/training/divl_critic.py",
    "divl": "src/openpi/training/divl_lwd_training.py",
    "qam": "src/openpi/training/qam.py",
    "qam_training": "src/openpi/training/qam_training.py",
    "qam_cached": "src/openpi/training/qam_cached_training.py",
    "siglip": "src/openpi/models/siglip.py",
    "critic_vision": "src/openpi/models/divl_vision.py",
    "critic_encoder_config": "src/openpi/models/divl_config.py",
}


def build_critic(config, encoder_config, action_dim):
    from gemma.gm.ckpts import _compat

    from openpi.models.divl_heads import DistributionalVHead
    from openpi.models.divl_heads import TwinQHead
    from openpi.models.divl_state_encoder import DIVLStateEncoder
    from openpi.training.divl_weight_loaders import load_siglip_weights
    from openpi.training.divl_weight_loaders import resolve_checkpoint_dirs

    seed = config["seed"]
    critic = nnx.Dict(
        encoder=DIVLStateEncoder(encoder_config, rngs=nnx.Rngs(seed)),
        v_head=DistributionalVHead(rngs=nnx.Rngs(seed + 1)),
        q_head=TwinQHead(action_dim, config["horizon"], rngs=nnx.Rngs(seed + 2)),
    )
    if config.get("init_critic_checkpoint"):
        # Every critic parameter is restored later; do not load unrelated base weights.
        return critic
    gemma_checkpoint, siglip_checkpoint = resolve_checkpoint_dirs(encoder_config)
    with ocp.StandardCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(gemma_checkpoint)
        metadata = getattr(metadata, "item_metadata", metadata)
        spec = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(
                x.shape, x.dtype, sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])
            ),
            metadata.tree,
        )
        restored = checkpointer.restore(gemma_checkpoint, target=spec)
    params = _compat.nest_params(_compat.param_remapper(restored))["transformer"]
    state = nnx.state(critic.encoder.gemma.model, nnx.Param)
    actual, expected = flatten_dict(params), flatten_dict(state.to_pure_dict())
    if actual.keys() != expected.keys() or any(actual[k].shape != expected[k].shape for k in actual):
        raise ValueError("Gemma backbone checkpoint shape mismatch")
    state.replace_by_pure_dict(params)
    nnx.update(critic.encoder.gemma.model, state)
    load_siglip_weights(critic.encoder.vision, siglip_checkpoint)
    return critic


def initialize_bundle(actor, config, encoder_config=None, action_dim=14):
    """BC owns actor/EMA only; offline alone constructs critic and V/Q optimizers."""
    mode = config["mode"]
    qam_config = QAMTrainingConfig(**config["qam"]) if mode == "offline" else None
    if mode == "bc":
        actor.eval()
        nnx.update(actor, jax.tree.map(lambda x: x.astype(jnp.float32), nnx.state(actor, nnx.Param)))
        reference = nnx.clone(actor)  # EMA during BC; exported as deployable BC actor.
        actor_optimizer = nnx.Optimizer(
            actor,
            optimizer_lib.create_optimizer(
                optimizer_lib.AdamW(**config["bc"]["optimizer"]),
                optimizer_lib.CosineDecaySchedule(**config["bc"]["lr_schedule"]),
            ),
            wrt=nnx.Param,
        )
    else:
        reference, actor_optimizer = initialize_actor_training(actor, qam_config, total_steps=config["num_train_steps"])
    bundle = nnx.Dict(actor=actor, reference=reference, actor_optimizer=actor_optimizer)
    critic = target = value_optimizer = q_optimizer = None
    if mode == "offline":
        log("Building critic (Gemma 270M + SigLIP So400M; FP32 attention by default)")
        critic = build_critic(config, encoder_config, action_dim)
        critic.eval()
        target, value_optimizer, q_optimizer = divl_lwd_training.initialize_training(
            critic, config["num_train_steps"], learning_rate=config["critic_learning_rate"]
        )
        bundle.critic, bundle.target = critic, target
        bundle.value_optimizer, bundle.q_optimizer = value_optimizer, q_optimizer
    return bundle


def update_bundle(bundle, batch, action_stats, rng, config, qam_config=None, divl_config=None):
    mode, freeze_critic = config["mode"], config.get("freeze_critic", False)
    actor, reference, actor_optimizer = bundle.actor, bundle.reference, bundle.actor_optimizer
    if mode == "offline":
        critic, target = bundle.critic, bundle.target
        value_optimizer, q_optimizer = bundle.value_optimizer, bundle.q_optimizer
    if mode == "bc":
        observation, actions = batch
        rng, metrics = bc_train_step(
            actor, reference, actor_optimizer, observation, actions, rng, ema_decay=config["bc"]["ema_decay"]
        )
    elif freeze_critic:
        rng, metrics = cached_actor_train_step(
            actor,
            reference,
            critic,
            actor_optimizer,
            batch.actor_observation,
            batch.critic["obs"],
            action_stats,
            rng,
            config=qam_config,
        )
        metrics = {**metrics, "critic/frozen": jnp.asarray(1.0)}
    else:
        rng, metrics = offline_train_step(
            actor,
            reference,
            critic,
            target,
            actor_optimizer,
            value_optimizer,
            q_optimizer,
            batch,
            action_stats,
            rng,
            qam_config=qam_config,
            divl_config=divl_config,
            actor_update_fn=cached_actor_train_step,
        )
    return rng, metrics


def _run(cleanup):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/airbot/lwd_offline.yaml"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="CPU-only strict config/data/input checks")
    parser.add_argument("--batch-size", type=int, help="global batch size across all visible GPUs")
    parser.add_argument(
        "--prepare-only", action="store_true", help="validate data and build sample batches, without loading models"
    )
    parser.add_argument(
        "--smoke-steps", type=int, default=0, help="stop after N additional real updates, save, then verify restore"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = yaml.safe_load(args.config.read_text())
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    mode = config["mode"]
    if mode not in ("bc", "offline"):
        raise ValueError(
            "mode must be bc or offline; legacy bc_critic/qam configs must be migrated to a new run directory"
        )
    freeze_critic = config.setdefault("freeze_critic", False)
    if type(freeze_critic) is not bool:
        raise ValueError("freeze_critic must be a boolean")
    if mode == "bc" and any(
        key in config
        for key in (
            "qam",
            "divl",
            "critic_learning_rate",
            "gemma_checkpoint",
            "gemma_tokenizer",
            "siglip_checkpoint",
            "critic_siglip_attention_dtype",
        )
    ):
        raise ValueError("Remove critic/QAM settings from the pure BC configuration")
    if mode == "bc" and (freeze_critic or config.get("init_critic_checkpoint")):
        raise ValueError("BC does not construct or initialize a critic")
    if freeze_critic and not config.get("init_critic_checkpoint"):
        raise ValueError("freeze_critic requires a trained init_critic_checkpoint")
    gpus = config["runtime"]["gpus"]
    if not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("runtime.gpus must contain unique nonnegative GPU indices")
    if config["per_device_batch_size"] < 1:
        raise ValueError("per_device_batch_size must be positive")
    if args.batch_size is None:
        config["batch_size"] = len(gpus) * config["per_device_batch_size"]
    workers, prefetch = validate_loader_settings(config["runtime"], config["batch_size"])
    args.run_dir = args.run_dir or Path(config["run_dir"])
    if not 0 <= config.get("bc", {}).get("ema_decay", 0.99) < 1:
        raise ValueError("Invalid BC EMA decay")
    data_config = AirbotDataConfig.from_dict(config["airbot"])
    qam_config = divl_config = None
    if mode == "offline":
        qam_config = QAMTrainingConfig(**config["qam"])
        divl_config = DIVLTrainingConfig(**config["divl"])
        if config["td_steps"] != 1:
            raise ValueError("AIRBOT implements 1-step chunk-level TD")
        if (
            qam_config.action_dim != data_config.action_dim
            or qam_config.action_clip is not None
            or not qam_config.sample_from_reference
        ):
            raise ValueError("AIRBOT LWD requires matching unclipped actions and a fixed reference sampler")
        if config["horizon"] != qam_config.replan_steps or config["horizon"] != qam_config.critic_horizon:
            raise ValueError("Execution and critic horizons must match")
    for name in (
        "batch_size",
        "num_train_steps",
        "log_interval",
        "save_interval",
        "keep_checkpoints",
    ):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if args.smoke_steps < 0:
        raise ValueError("smoke-steps must be nonnegative")
    model_config = pi0_config.Pi0Config(**config["actor_model"])
    if model_config.action_horizon != config["horizon"]:
        raise ValueError("Actor model horizon must match execution horizon")
    checkpoint = Path(config["bc_checkpoint"]).resolve()
    norm_dir = (
        Path(config["norm_stats_dir"]).resolve() if mode == "bc" else checkpoint / "assets" / data_config.norm_asset_id
    )
    replay = AirbotReplay(
        config["datasets"],
        horizon=config["horizon"],
        gamma=config.get("gamma", 0.9999),
        exclude_episodes=config.get("exclude_episodes", ()),
        data_config=data_config,
    )
    frame_sampler = FrameSampler(replay) if mode == "bc" else None
    if args.check:
        required = [checkpoint / "params/_METADATA", norm_dir / "norm_stats.json"]
        if mode == "offline":
            required.append(Path(config["gemma_tokenizer"]))
            if config.get("init_critic_checkpoint"):
                required.extend(
                    Path(config["init_critic_checkpoint"]) / name
                    for name in ("complete.json", "training_state/_METADATA")
                )
            else:
                required += [
                    Path(config["gemma_checkpoint"]) / "_METADATA",
                    Path(config["siglip_checkpoint"]) / "model.safetensors",
                ]
        for required_path in required:
            if not required_path.is_file():
                raise FileNotFoundError(required_path)
        normalize.load(norm_dir)
        if mode == "bc":
            optimizer_lib.CosineDecaySchedule(**config["bc"]["lr_schedule"]).create()
        log(
            f"Config/data checked: mode={mode}, GPUs={gpus}, global batch={config['batch_size']}, steps={config['num_train_steps']}"
        )
        return
    norm_stats = normalize.load(norm_dir)
    config["actor_norm_sha256"] = hashlib.sha256((norm_dir / "norm_stats.json").read_bytes()).hexdigest()
    critic_norm_stats = norm_stats if mode == "offline" else None
    if config.get("init_critic_checkpoint"):
        critic_checkpoint = Path(config["init_critic_checkpoint"]).resolve()
        if not (critic_checkpoint / "complete.json").is_file():
            raise ValueError(f"Incomplete critic checkpoint: {critic_checkpoint}")
        critic_config = json.loads((critic_checkpoint / "config.json").read_text())
        if critic_config.get("mode") == "bc":
            raise ValueError(
                "A pure BC checkpoint contains no critic; omit init_critic_checkpoint for fresh offline training"
            )
        # Legacy checkpoint schema is used ONLY to recover normalization, never to label replay.
        old_schema = {k: v for k, v in critic_config.get("airbot", {}).items() if k != "all_success"}
        old_airbot = AirbotDataConfig.from_dict(old_schema)
        if (
            critic_config["horizon"] != config["horizon"]
            or old_airbot.action_dim != data_config.action_dim
            or old_airbot.camera_mapping != data_config.camera_mapping
            or old_airbot.delta_action_indices != data_config.delta_action_indices
        ):
            raise ValueError("Critic checkpoint robot/action schema does not match this run")
        critic_norm_dir = (
            critic_checkpoint / "assets" / critic_config.get("critic_norm_asset_id", old_airbot.norm_asset_id)
        )
        critic_norm_stats = normalize.load(critic_norm_dir)
        config["critic_norm_sha256"] = hashlib.sha256((critic_norm_dir / "norm_stats.json").read_bytes()).hexdigest()
        config["init_critic_identity"] = {
            name: hashlib.sha256((critic_checkpoint / name).read_bytes()).hexdigest()
            for name in ("config.json", "complete.json", "training_state/_METADATA")
        }
    config["model_config"] = dataclasses.asdict(model_config)
    encoder_config = None
    if mode == "offline":
        from openpi.models.divl_config import DIVLStateEncoderConfig

        config["critic_norm_asset_id"] = "critic_norm_stats"
        config.setdefault("critic_siglip_attention_dtype", "float32")
        if config["critic_siglip_attention_dtype"] not in ("bfloat16", "float32"):
            raise ValueError("critic_siglip_attention_dtype must be bfloat16 or float32")
        encoder_config = DIVLStateEncoderConfig(
            image_keys=data_config.camera_keys,
            state_dim=data_config.action_dim,
            gemma_checkpoint=config["gemma_checkpoint"],
            gemma_tokenizer=config["gemma_tokenizer"],
            siglip_checkpoint=config["siglip_checkpoint"],
            siglip_attention_dtype=config["critic_siglip_attention_dtype"],
        )
    # The replay is immutable for a run. Cache only decoded images; batch
    # construction, normalization and tokenization retain the original semantics.
    replay.images = functools.lru_cache(maxsize=4096)(replay.images)
    log("On-demand decoded image cache enabled (4096 states); no full-dataset warmup")
    manifest = replay.manifest()
    builder = make_batch_builder(replay, model_config, norm_stats, critic_norm_stats, config)
    log(
        f"Replay: {manifest['composition']}; excluded: {manifest['excluded_episodes']}; "
        f"recording gaps: {sum(e.gap is not None for e in replay.episodes)}"
    )
    if mode == "bc":
        probe_chunks = frame_sampler.sample(2, np.random.default_rng(config["seed"]))
        probe = builder(probe_chunks)
        if probe[1].shape != (len(probe_chunks), config["horizon"], model_config.action_dim):
            raise AssertionError("BC action shape mismatch")
        if not np.isfinite(probe[1]).all():
            raise ValueError("Nonfinite BC actions")
    else:
        # Validate every terminal outcome present in this replay. Some datasets, such as
        # a success-only labeled replay has no failure probe.
        terminal_outcomes = sorted({replay.episodes[c.episode].outcome for c in replay.chunks if c.terminal})
        if "success" not in terminal_outcomes:
            raise ValueError("Replay must contain at least one successful terminal episode")
        terminals = {
            outcome: next(c for c in replay.chunks if c.terminal and replay.episodes[c.episode].outcome == outcome)
            for outcome in terminal_outcomes
        }
        probe_chunks = [terminals[outcome] for outcome in terminal_outcomes]
        probe = builder(probe_chunks)
        rewards = np.asarray(probe.critic["rewards"])
        if not np.all(np.asarray(probe.critic["discounts"]) == 0):
            raise ValueError("Terminal reward/discount validation failed")
        for i, outcome in enumerate(terminal_outcomes):
            expected = 1.0 if outcome == "success" else 0.0
            if (rewards[i] > 0) != (expected > 0):
                raise ValueError("Terminal reward/discount validation failed")
        for i, chunk in enumerate(probe_chunks):
            recovered = transforms.Unnormalize({"actions": critic_norm_stats["actions"]}, use_quantiles=True)(
                {"actions": np.asarray(probe.critic["actions"])[i]}
            )["actions"]
            physical, _, _, _ = replay.transition(chunk)
            np.testing.assert_allclose(recovered[: chunk.length], physical[: chunk.length], atol=1e-5)
        for camera in data_config.camera_keys:
            np.testing.assert_allclose(
                probe.actor_observation.images[camera], probe.critic["obs"]["images"][camera], atol=1e-6
            )
    if args.prepare_only:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / "data_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
        log(f"Data preparation passed: mode={mode}; report={args.run_dir / 'data_audit.json'}")
        return
    devices = jax.devices()
    if any(device.platform != "gpu" for device in devices):
        raise ValueError("Training requires GPUs; select them with CUDA_VISIBLE_DEVICES")
    if config["batch_size"] % len(devices):
        raise ValueError("Global batch_size must be divisible by the number of visible GPUs")
    if len(devices) != len(gpus):
        raise ValueError("Visible GPU count differs from configured runtime.gpus")
    mesh = jax.sharding.Mesh(np.asarray(devices), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    config["num_devices"] = len(devices)
    log(
        f"Data parallelism: {len(devices)} GPUs, global batch={config['batch_size']}, local batch={config['batch_size'] // len(devices)}"
    )
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repo_root = Path(__file__).resolve().parents[1]
    implementation = airbot_lwd_run.implementation_manifest(repo_root, IMPLEMENTATION_SOURCES)
    airbot_lwd_run.prepare_run_directory(run, config, manifest, implementation, resume=args.resume)
    wandb_logger = WandbLogger(run, config, resume=args.resume)
    if wandb_logger.run is not None:
        log(f"W&B run: {wandb_logger.run.url}")
    log(f"Loading BC actor from {checkpoint}; devices={devices}")
    actor = model_config.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16, sharding=replicated))
    bundle = initialize_bundle(actor, config, encoder_config, data_config.action_dim)
    reference, actor_optimizer = bundle.reference, bundle.actor_optimizer
    critic = bundle.critic if mode == "offline" else None
    target = bundle.target if mode == "offline" else None
    value_optimizer = bundle.value_optimizer if mode == "offline" else None
    q_optimizer = bundle.q_optimizer if mode == "offline" else None
    # GSPMD partitions the batch and synchronizes parameter gradients. Model,
    # target and optimizer arrays are replicated; this is one shared learner.
    nnx.update(bundle, jax.device_put(nnx.state(bundle), replicated))
    action_stats = jax.device_put(builder.action_stats, replicated) if mode == "offline" else None
    rng = jax.device_put(jax.random.PRNGKey(config["seed"]), replicated)
    start = 0
    if config.get("init_critic_checkpoint") and not args.resume:
        actor_signature = airbot_lwd_run.signature(actor)
        reference_signature = airbot_lwd_run.signature(reference)
        airbot_lwd_run.initialize_critic_from_checkpoint(config["init_critic_checkpoint"], critic, target)
        np.testing.assert_array_equal(actor_signature, airbot_lwd_run.signature(actor))
        np.testing.assert_array_equal(reference_signature, airbot_lwd_run.signature(reference))
        for optimizer in (actor_optimizer, value_optimizer, q_optimizer):
            if int(optimizer.step.value) != 0:
                raise AssertionError("Warm start must retain fresh optimizer states")
        log(
            f"Initialized only critic/target from {config['init_critic_checkpoint']}; BC actor/reference retained, optimizers reset"
        )
    if args.resume:
        paths = [p for p in (run / "checkpoints").iterdir() if p.name.isdigit() and (p / "complete.json").exists()]
        if not paths:
            raise ValueError("No complete checkpoint available for resume")
        path = max(paths, key=lambda p: int(p.name))
        start, rng = airbot_lwd_run.restore_checkpoint(path, bundle, rng)
        log(f"Restored {mode} models, optimizers and RNG at step {start}")
    stop = min(config["num_train_steps"], start + args.smoke_steps) if args.smoke_steps else config["num_train_steps"]
    if stop <= start:
        log("Training already complete")
        return
    reference_before = airbot_lwd_run.signature(reference)
    actor_before = airbot_lwd_run.signature(actor)
    critic_before = airbot_lwd_run.signature(critic) if critic is not None else None
    frozen_modules = {"critic": critic, "target": target, "reference": reference} if freeze_critic else {}
    frozen_before = airbot_lwd_run.frozen_parameter_hashes(frozen_modules)
    metrics_file = cleanup.enter_context((run / "metrics.jsonl").open("a", buffering=1))
    bc_lr = optimizer_lib.CosineDecaySchedule(**config["bc"]["lr_schedule"]).create() if mode == "bc" else None
    log(f"{mode} training steps {start + 1}..{stop}; first update includes JAX compilation")

    def sample_step(step):
        sample_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], step]))
        return (
            frame_sampler.sample(config["batch_size"], sample_rng)
            if frame_sampler
            else replay.sample(config["batch_size"], sample_rng)
        )

    loader = None
    if workers:
        log(f"Starting {workers} CPU data workers with {prefetch} shared-memory buffers")
        template = probe
        loader = cleanup.enter_context(
            ParallelAirbotLoader(
                replay,
                model_config,
                norm_stats,
                critic_norm_stats,
                config,
                template,
                sample_step,
                run,
            )
        )
        loader.start(start + 1, stop)
        log(f"Data workers ready: PIDs={loader.worker_pids}; shared memory={loader.buffer_bytes / 2**30:.2f} GiB")
    started = time.monotonic()
    for step in range(start + 1, stop + 1):
        # Step-addressable sampling is identical in serial and parallel paths.
        wait_started = time.monotonic()
        if loader is not None:
            batch = jax.device_put(loader.get(step), data_sharding)
        else:
            chunks = sample_step(step)
            batch = jax.device_put(builder(chunks), data_sharding)
        data_wait_seconds = time.monotonic() - wait_started
        rng, metrics = update_bundle(bundle, batch, action_stats, rng, config, qam_config, divl_config)
        values = {k: float(v) for k, v in jax.device_get(metrics).items()}
        if loader is not None:
            # Finish all host-buffer reads before workers can overwrite this slot.
            jax.block_until_ready(batch)
            loader.release(step)
        values["data/wait_seconds"] = data_wait_seconds
        if step == start + 1:
            if any(not x.sharding.is_fully_replicated for x in jax.tree.leaves(nnx.state(bundle))):
                raise AssertionError("Data-parallel model/optimizer state must remain replicated")
            log(f"Synchronized model/optimizer state verified across {len(devices)} GPUs")
        if values.get("critic/update_applied", 1.0) == 0 or not all(np.isfinite(v) for v in values.values()):
            (run / "nonfinite.json").write_text(json.dumps({"step": step, **values}, indent=2))
            raise FloatingPointError(f"Nonfinite update at step {step}: {values}")
        values.update(step=step, elapsed_seconds=time.monotonic() - started)
        cosine_factor = 0.5 * (1 + np.cos(np.pi * (step - 1) / config["num_train_steps"]))
        if mode == "offline":
            values["critic/learning_rate"] = (
                0.0 if freeze_critic else float(config["critic_learning_rate"] * cosine_factor)
            )
        values["actor/learning_rate"] = (
            float(bc_lr(step - 1)) if bc_lr else float(qam_config.learning_rate * cosine_factor)
        )
        metrics_file.write(json.dumps(values) + "\n")
        if step == start + 1 or step % config["log_interval"] == 0 or step == stop:
            wandb_logger.log(values)
            if mode == "bc":
                log(
                    f"step={step} bc={values['actor/bc_loss']:.5g} "
                    f"actor_grad={values['actor/grad_norm']:.5g} elapsed={values['elapsed_seconds']:.1f}s"
                )
            else:
                ratio_valid = values["actor/regularization_ratio_valid"]
                guide_ratio = f"{values['actor/guidance_to_regularization_ratio']:.5g}" if ratio_valid else "n/a"
                qam_ratio = f"{values['actor/qam_to_regularization_ratio']:.5g}" if ratio_valid else "n/a"
                log(
                    f"step={step} critic={values.get('critic/loss', 'frozen')} qam={values['actor/qam_loss']:.5g} "
                    f"reg={values['actor/regularization_loss']:.5g} guide={values['actor/guidance_loss']:.5g} "
                    f"guide_const={values['actor/guidance_constant']:.5g} "
                    f"qam/reg={qam_ratio} |guide|/reg={guide_ratio} "
                    f"actor_grad={values['actor/grad_norm']:.5g} elapsed={values['elapsed_seconds']:.1f}s"
                )
        if step % config["save_interval"] == 0 or step == stop:
            if freeze_critic and airbot_lwd_run.frozen_parameter_hashes(frozen_modules) != frozen_before:
                raise AssertionError("Frozen critic/target/reference parameters changed")
            if mode == "offline" and not np.array_equal(reference_before, airbot_lwd_run.signature(reference)):
                raise AssertionError("Fixed BC reference was modified")
            path = airbot_lwd_run.save_checkpoint(
                run,
                step,
                bundle,
                rng,
                config,
                norm_stats,
                norm_asset_id=data_config.norm_asset_id,
                log=log,
                critic_norm_stats=critic_norm_stats,
            )
            wandb_logger.checkpoint(step, path)
    if args.smoke_steps:
        modules = {"actor": actor, "reference": reference}
        if critic is not None:
            modules["critic"] = critic
        expected = {name: airbot_lwd_run.signature(module) for name, module in modules.items()}
        if np.array_equal(actor_before, expected["actor"]):
            raise AssertionError("Actor parameter probes did not change")
        if critic is not None and (np.array_equal(critic_before, expected["critic"]) != freeze_critic):
            raise AssertionError("Critic updates do not match freeze_critic")
        state_step, restored_rng = airbot_lwd_run.restore_checkpoint(path, bundle, rng)
        if state_step != stop or not np.array_equal(restored_rng, rng):
            raise AssertionError("Checkpoint step/RNG mismatch")
        for name, module in modules.items():
            np.testing.assert_array_equal(airbot_lwd_run.signature(module), expected[name])
        report = {
            "passed": True,
            "mode": mode,
            "freeze_critic": freeze_critic,
            "start_step": start,
            "end_step": stop,
            "checkpoint": str(path),
            "last_metrics": values,
        }
        (run / f"smoke_report_{stop}.json").write_text(json.dumps(report, indent=2) + "\n")
        log("Smoke validation passed: finite updates, mode-correct parameter changes, checkpoint restored")
    if stop == config["num_train_steps"]:
        wandb_logger.complete(stop)
    metrics_file.close()
    log(f"Finished at step {stop}; metrics: {run / 'metrics.jsonl'}")


def main():
    with contextlib.ExitStack() as cleanup:
        return _run(cleanup)


if __name__ == "__main__":
    import wandb

    def stop_training(signum, frame):
        raise KeyboardInterrupt(f"Training stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop_training)

    try:
        main()
    except BaseException:
        wandb.finish(exit_code=1)
        raise
    else:
        wandb.finish()
