"""Synchronous data-parallel offline LWD for AIRBOT; see examples/airbot/lwd_offline.md."""

import argparse
import dataclasses
import fcntl
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

# Limit host compilation/thread pools on this large machine, before importing JAX.
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])

import flax.nnx as nnx
from flax.traverse_util import flatten_dict
from gemma.gm.ckpts import _compat
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.models.divl_heads import DistributionalVHead
from openpi.models.divl_heads import TwinQHead
from openpi.models.divl_state_encoder import DIVLStateEncoder
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training import config as config_lib
from openpi.training import divl_lwd_training
from openpi.training.airbot_lwd_data import CAMERAS
from openpi.training.airbot_lwd_data import AirbotBatchBuilder
from openpi.training.airbot_lwd_data import AirbotReplay
from openpi.training.divl_critic import critic_loss
from openpi.training.divl_weight_loaders import load_siglip_weights
from openpi.training.qam_cached_training import actor_train_step as cached_actor_train_step
from openpi.training.qam_training import DIVLTrainingConfig
from openpi.training.qam_training import QAMTrainingConfig
from openpi.training.qam_training import initialize_actor_training
from openpi.training.qam_training import offline_train_step


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_critic(config, encoder_config):
    seed = config["seed"]
    critic = nnx.Dict(
        encoder=DIVLStateEncoder(encoder_config, rngs=nnx.Rngs(seed)),
        v_head=DistributionalVHead(rngs=nnx.Rngs(seed + 1)),
        q_head=TwinQHead(14, config["horizon"], rngs=nnx.Rngs(seed + 2)),
    )
    with ocp.StandardCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(Path(config["gemma_checkpoint"]).resolve())
        metadata = getattr(metadata, "item_metadata", metadata)
        spec = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(
                x.shape, x.dtype, sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])
            ),
            metadata.tree,
        )
        restored = checkpointer.restore(Path(config["gemma_checkpoint"]).resolve(), target=spec)
    params = _compat.nest_params(_compat.param_remapper(restored))["transformer"]
    state = nnx.state(critic.encoder.gemma.model, nnx.Param)
    actual, expected = flatten_dict(params), flatten_dict(state.to_pure_dict())
    if actual.keys() != expected.keys() or any(actual[k].shape != expected[k].shape for k in actual):
        raise ValueError("Gemma backbone checkpoint shape mismatch")
    state.replace_by_pure_dict(params)
    nnx.update(critic.encoder.gemma.model, state)
    load_siglip_weights(critic.encoder.vision, Path(config["siglip_checkpoint"]))
    return critic


@nnx.jit
def evaluate(critic, target, batch, tau_base=0.6, alpha=0.3, tau_min=0.0, tau_max=1.0):
    critic.eval()
    target.eval()
    loss, metrics = critic_loss(critic, target, batch, tau_base, alpha, tau_min, tau_max)
    z = critic.encoder(**batch["obs"], train=False)
    q = jnp.minimum(*critic.q_head(z, batch["actions"], batch["action_mask"]))
    return {**metrics, "loss": loss, "q": q}


def signature(module):
    """Small parameter probe, checked alongside full losses and gradients."""
    return np.asarray([float(x.reshape(-1)[0]) for x in jax.tree.leaves(nnx.state(module, nnx.Param))])


def save_checkpoint(run, step, bundle, rng, config, norm_stats):
    destination = run / "checkpoints" / str(step)
    staging = run / "checkpoints" / f"{step}.incomplete"
    if destination.exists():
        raise FileExistsError(f"Complete checkpoint already exists: {destination}")
    if staging.exists():
        abandoned = staging.with_name(f"{staging.name}.abandoned-{time.time_ns()}")
        staging.rename(abandoned)
        log(f"Preserved interrupted checkpoint at {abandoned}")
    staging.mkdir(parents=True)
    payload = {
        "models_and_optimizers": nnx.state(bundle).to_pure_dict(),
        "rng": rng,
        "step": np.asarray(step, dtype=np.int64),
    }
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(staging / "training_state", payload)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(staging / "params", {"params": nnx.state(bundle.actor, nnx.Param).to_pure_dict()})
    normalize.save(staging / "assets" / "airbot_cube_200", norm_stats)
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (staging / "complete.json").write_text(json.dumps({"step": step}) + "\n")
    staging.rename(destination)
    checkpoints = sorted(
        (p for p in destination.parent.iterdir() if p.name.isdigit() and (p / "complete.json").exists()),
        key=lambda p: int(p.name),
    )
    for old in checkpoints[: -config["keep_checkpoints"]]:
        shutil.rmtree(old)
    log(f"Saved full training state and deployable actor: {destination}")
    return destination


def restore_checkpoint(path, bundle, rng):
    target = {
        "models_and_optimizers": nnx.state(bundle).to_pure_dict(),
        "rng": rng,
        "step": np.asarray(0, dtype=np.int64),
    }
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(path / "training_state", target=target)
    state = nnx.state(bundle)
    state.replace_by_pure_dict(restored["models_and_optimizers"])
    nnx.update(bundle, state)
    step = int(restored["step"])
    for optimizer in (bundle.actor_optimizer, bundle.value_optimizer, bundle.q_optimizer):
        if int(optimizer.step.value) != step:
            raise ValueError("Restored optimizer step does not match checkpoint step")
    if bundle.actor_optimizer.model is not bundle.actor:
        raise ValueError("Restored actor optimizer lost its model binding")
    if bundle.value_optimizer.model is not bundle.critic or bundle.q_optimizer.model is not bundle.critic:
        raise ValueError("Restored critic optimizers lost their model bindings")
    return step, restored["rng"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/airbot/lwd_offline.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/lwd_airbot_cube/offline"))
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
    qam_config = QAMTrainingConfig(**config["qam"])
    divl_config = DIVLTrainingConfig(**config["divl"])
    if config["td_steps"] != 1:
        raise ValueError("AIRBOT cube adapter implements 1-step chunk-level TD")
    if qam_config.action_dim != 14 or qam_config.action_clip is not None or not qam_config.sample_from_reference:
        raise ValueError("AIRBOT LWD requires 14D unclipped delta actions and a fixed reference sampler")
    if config["horizon"] != qam_config.replan_steps or config["horizon"] != qam_config.critic_horizon:
        raise ValueError("Execution and critic horizons must match")
    for name in (
        "batch_size",
        "num_train_steps",
        "log_interval",
        "eval_interval",
        "eval_batches",
        "save_interval",
        "keep_checkpoints",
    ):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if args.smoke_steps < 0:
        raise ValueError("smoke-steps must be nonnegative")
    train_config = config_lib.get_config(config["actor_config"])
    if train_config.model.action_horizon != config["horizon"]:
        raise ValueError("BC model horizon must match execution horizon")
    checkpoint = Path(config["bc_checkpoint"]).resolve()
    norm_dir = checkpoint / "assets" / "airbot_cube_200"
    norm_stats = normalize.load(norm_dir)
    config["actor_norm_sha256"] = hashlib.sha256((norm_dir / "norm_stats.json").read_bytes()).hexdigest()
    config["model_config"] = dataclasses.asdict(train_config.model)
    encoder_config = DIVLStateEncoderConfig(image_keys=CAMERAS, state_dim=14, gemma_tokenizer=config["gemma_tokenizer"])
    replay = AirbotReplay(
        config["datasets"],
        horizon=config["horizon"],
        gamma=config["gamma"],
        validation_fraction=config["validation_fraction"],
        seed=config["seed"],
        exclude_episodes=config.get("exclude_episodes", ()),
    )
    # The replay is immutable for a run. Cache only decoded images; batch
    # construction, normalization and tokenization retain the original semantics.
    replay.images = functools.lru_cache(maxsize=4096)(replay.images)
    image_keys = sorted(
        {(chunk.episode, index) for chunk in replay.chunks["train"] for index in (chunk.start, chunk.next_index)}
    )
    cache_started = time.monotonic()
    for episode_id, index in image_keys:
        replay.images(episode_id, index)
    log(
        f"Optimized QAM enabled; decoded image cache: {len(image_keys)} states in {time.monotonic() - cache_started:.1f}s"
    )
    manifest = replay.manifest()
    builder = AirbotBatchBuilder(
        replay, train_config.model, norm_stats, DIVLTokenizer(encoder_config), prompt=config["prompt"]
    )
    log(
        f"Replay: {manifest['composition']}; excluded: {manifest['excluded_episodes']}; "
        f"recording gaps: {sum(e.gap is not None for e in replay.episodes)}"
    )
    # Always validate a successful and failed terminal plus a nonterminal batch.
    terminals = {
        outcome: next(c for c in replay.chunks["train"] if c.terminal and replay.episodes[c.episode].outcome == outcome)
        for outcome in ("success", "failure")
    }
    probe_chunks = [terminals["success"], terminals["failure"]]
    probe = builder(probe_chunks)
    if not (
        float(probe.critic["rewards"][0]) > 0
        and float(probe.critic["rewards"][1]) == 0
        and np.all(np.asarray(probe.critic["discounts"]) == 0)
    ):
        raise ValueError("Terminal reward/discount validation failed")
    for i, chunk in enumerate(probe_chunks):
        recovered = transforms.Unnormalize({"actions": norm_stats["actions"]}, use_quantiles=True)(
            {"actions": np.asarray(probe.critic["actions"])[i]}
        )["actions"]
        physical, _, _, _ = replay.transition(chunk)
        np.testing.assert_allclose(recovered[: chunk.length], physical[: chunk.length], atol=1e-5)
    for camera in CAMERAS:
        np.testing.assert_allclose(
            probe.actor_observation.images[camera], probe.critic["obs"]["images"][camera], atol=1e-6
        )
    if args.prepare_only:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / "data_audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
        log(
            f"Data preparation passed: actor={probe.actor_observation.state.shape}, critic={probe.critic['actions'].shape}; report={args.run_dir / 'data_audit.json'}"
        )
        return
    devices = jax.devices()
    if any(device.platform != "gpu" for device in devices):
        raise ValueError("Training requires GPUs; select them with CUDA_VISIBLE_DEVICES")
    if config["batch_size"] % len(devices):
        raise ValueError("Global batch_size must be divisible by the number of visible GPUs")
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
    config_path, manifest_path = run / "config.json", run / "replay_manifest.json"
    if args.resume:
        if (
            json.loads(config_path.read_text()) != json.loads(json.dumps(config))
            or json.loads(manifest_path.read_text()) != manifest
        ):
            raise ValueError("Resume requires unchanged training config and replay snapshot")
    else:
        if config_path.exists() or (run / "checkpoints").exists():
            raise FileExistsError("Run already exists; use --resume or choose a new --run-dir")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"Loading BC actor from {checkpoint}; devices={devices}")
    actor = train_config.model.load(
        model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16, sharding=replicated)
    )
    reference, actor_optimizer = initialize_actor_training(actor, qam_config, total_steps=config["num_train_steps"])
    log("Loading pretrained critic backbones (Gemma 270M + SigLIP So400M)")
    critic = build_critic(config, encoder_config)
    # Match the graph mode at the end of QAM. DIVL enables training internally.
    critic.eval()
    target, value_optimizer, q_optimizer = divl_lwd_training.initialize_training(
        critic, config["num_train_steps"], learning_rate=config["critic_learning_rate"]
    )
    bundle = nnx.Dict(
        actor=actor,
        reference=reference,
        critic=critic,
        target=target,
        actor_optimizer=actor_optimizer,
        value_optimizer=value_optimizer,
        q_optimizer=q_optimizer,
    )
    # GSPMD partitions the batch and synchronizes parameter gradients. Model,
    # target and optimizer arrays are replicated; this is one shared learner.
    nnx.update(bundle, jax.device_put(nnx.state(bundle), replicated))
    (run / "implementation.json").write_text(
        json.dumps({"actor_update": "cached_prefix_v1", "decoded_image_cache": 4096}, indent=2) + "\n"
    )
    action_stats = jax.device_put(builder.action_stats, replicated)
    rng = jax.device_put(jax.random.PRNGKey(config["seed"]), replicated)
    start = 0
    if args.resume:
        paths = [p for p in (run / "checkpoints").iterdir() if p.name.isdigit() and (p / "complete.json").exists()]
        if not paths:
            raise ValueError("No complete checkpoint available for resume")
        path = max(paths, key=lambda p: int(p.name))
        start, rng = restore_checkpoint(path, bundle, rng)
        log(f"Restored actor, fixed reference, critic/target, all optimizers and RNG at step {start}")
    stop = min(config["num_train_steps"], start + args.smoke_steps) if args.smoke_steps else config["num_train_steps"]
    if stop <= start:
        log("Training already complete")
        return
    reference_before, actor_before, critic_before = signature(reference), signature(actor), signature(critic)
    metrics_file = (run / "metrics.jsonl").open("a", buffering=1)
    log(f"Joint training steps {start + 1}..{stop}; first update includes JAX compilation")
    started = time.monotonic()
    for step in range(start + 1, stop + 1):
        # Step-addressable sampling makes resumes deterministic without a mutable host RNG.
        chunks = replay.sample(
            "train", config["batch_size"], np.random.default_rng(np.random.SeedSequence([config["seed"], step]))
        )
        batch = jax.device_put(builder(chunks), data_sharding)
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
        values = {k: float(v) for k, v in jax.device_get(metrics).items()}
        if step == start + 1:
            if any(not x.sharding.is_fully_replicated for x in jax.tree.leaves(nnx.state(bundle))):
                raise AssertionError("Data-parallel model/optimizer state must remain replicated")
            log(f"Synchronized model/optimizer state verified across {len(devices)} GPUs")
        if not all(np.isfinite(v) for v in values.values()):
            (run / "nonfinite.json").write_text(json.dumps({"step": step, **values}, indent=2))
            raise FloatingPointError(f"Nonfinite update at step {step}: {values}")
        values.update(step=step, elapsed_seconds=time.monotonic() - started)
        metrics_file.write(json.dumps(values) + "\n")
        if step == start + 1 or step % config["log_interval"] == 0 or step == stop:
            log(
                f"step={step} critic={values['critic/loss']:.5g} qam={values['actor/qam_loss']:.5g} actor_grad={values['actor/grad_norm']:.5g} elapsed={values['elapsed_seconds']:.1f}s"
            )
        if step % config["eval_interval"] == 0 or step == stop:
            results = []
            for offset in range(config["eval_batches"]):
                val_batch = jax.device_put(
                    builder(
                        replay.sample(
                            "val", config["batch_size"], np.random.default_rng(config["seed"] + 100000 + offset)
                        )
                    ),
                    data_sharding,
                )
                result = evaluate(
                    critic,
                    target,
                    val_batch.critic,
                    divl_config.quantile_level,
                    divl_config.alpha,
                    divl_config.tau_min,
                    divl_config.tau_max,
                )
                results.append({k: float(np.mean(v)) for k, v in jax.device_get(result).items()})
            validation = {f"val/{k}": float(np.mean([r[k] for r in results])) for k in results[0]}
            if not all(np.isfinite(v) for v in validation.values()):
                raise FloatingPointError(f"Nonfinite validation: {validation}")
            metrics_file.write(json.dumps({"step": step, **validation}) + "\n")
            log(f"Validation: {validation}")
        if step % config["save_interval"] == 0 or step == stop:
            if not np.array_equal(reference_before, signature(reference)):
                raise AssertionError("Fixed BC reference was modified")
            path = save_checkpoint(run, step, bundle, rng, config, norm_stats)
    if args.smoke_steps:
        actor_after, critic_after = signature(actor), signature(critic)
        if np.array_equal(actor_before, actor_after) or np.array_equal(critic_before, critic_after):
            raise AssertionError("Actor or critic parameter probes did not change")
        state_step, restored_rng = restore_checkpoint(path, bundle, rng)
        if state_step != stop or not np.array_equal(restored_rng, rng):
            raise AssertionError("Checkpoint step/RNG mismatch")
        for module, expected in ((actor, actor_after), (critic, critic_after), (reference, reference_before)):
            if not np.array_equal(signature(module), expected):
                raise AssertionError("Restored parameter probes differ")
        report = {
            "passed": True,
            "actor_implementation": "cached_prefix_v1",
            "start_step": start,
            "end_step": stop,
            "actor_changed": True,
            "critic_changed": True,
            "reference_fixed": True,
            "restore_parameter_probes_match": True,
            "num_devices": len(devices),
            "global_batch_size": config["batch_size"],
            "last_metrics": values,
            "checkpoint": str(path),
        }
        (run / f"smoke_report_{stop}.json").write_text(json.dumps(report, indent=2) + "\n")
        log("Smoke validation passed: finite joint updates, actor/critic changed, fixed reference, checkpoint restored")
    log(f"Finished at step {stop}; metrics: {run / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
