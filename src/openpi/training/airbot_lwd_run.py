"""Run identity and checkpoint lifecycle for AIRBOT BC and offline LWD."""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
import time

import flax.nnx as nnx
import jax
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import normalize


def implementation_manifest(repo_root: Path, sources: Mapping[str, str]) -> dict:
    """Fingerprint training semantics that must not change across a resume."""
    hashes = {}
    for name, relative_path in sources.items():
        source = repo_root / relative_path
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "schema_version": 3,
        "actor_update": "bc_ema_then_offline_cached_qam_v2",
        "decoded_image_cache": 4096,
        "source_sha256": hashes,
    }


def prepare_run_directory(run: Path, config: dict, replay_manifest: dict, implementation: dict, *, resume: bool):
    """Create a new run identity or verify every identity file before resume."""
    config_path = run / "config.json"
    replay_path = run / "replay_manifest.json"
    implementation_path = run / "implementation.json"
    expected = {
        config_path: json.loads(json.dumps(config)),
        replay_path: replay_manifest,
        implementation_path: implementation,
    }
    if resume:
        for path, value in expected.items():
            if not path.is_file() or json.loads(path.read_text()) != value:
                raise ValueError(f"Resume requires unchanged {path.name}")
        return
    if any(path.exists() for path in expected) or (run / "checkpoints").exists():
        raise FileExistsError("Run already exists; use --resume or choose a new --run-dir")
    for path, value in expected.items():
        path.write_text(json.dumps(value, indent=2) + "\n")


def signature(module):
    """Small parameter probe, checked alongside full losses and gradients."""
    return np.asarray([float(x.reshape(-1)[0]) for x in jax.tree.leaves(nnx.state(module, nnx.Param))])


def save_checkpoint(run, step, bundle, rng, config, norm_stats, *, norm_asset_id, log, critic_norm_stats=None):
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
    export_actor = bundle.reference if config["mode"] == "bc" else bundle.actor
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(staging / "params", {"params": nnx.state(export_actor, nnx.Param).to_pure_dict()})
    normalize.save(staging / "assets" / norm_asset_id, norm_stats)
    if critic_norm_stats is not None:
        normalize.save(staging / "assets" / config["critic_norm_asset_id"], critic_norm_stats)
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
    config = json.loads((path / "config.json").read_text())
    critic_step = 0 if config.get("freeze_critic") else step
    optimizers = [(bundle.actor_optimizer, step, bundle.actor)]
    if config["mode"] == "offline":
        optimizers += [(bundle.value_optimizer, critic_step, bundle.critic), (bundle.q_optimizer, critic_step, bundle.critic)]
    for optimizer, expected, model in optimizers:
        if int(optimizer.step.value) != expected:
            raise ValueError("Restored optimizer step does not match training mode")
        if optimizer.model is not model:
            raise ValueError("Restored optimizer lost its model binding")
    return step, restored["rng"]


def initialize_critic_from_checkpoint(path, critic, target):
    """Load only critic/target parameters; retain new actor and optimizer state."""
    path = Path(path).resolve()
    if not (path / "complete.json").is_file():
        raise ValueError(f"Expected a complete LWD checkpoint: {path}")
    modules = {"critic": critic, "target": target}
    item = {"models_and_optimizers": {
        name: nnx.state(module, nnx.Param).to_pure_dict() for name, module in modules.items()
    }}
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            path / "training_state",
            args=ocp.args.PyTreeRestore(
                item=item, transforms={},
                restore_args=ocp.checkpoint_utils.construct_restore_args(item),
            ),
        )
    for name, module in modules.items():
        state = nnx.state(module, nnx.Param)
        state.replace_by_pure_dict(restored["models_and_optimizers"][name])
        nnx.update(module, state)


def frozen_parameter_hashes(modules):
    """Hash every frozen parameter, using one local replica to limit transfer size."""
    result = {}
    for name, module in modules.items():
        digest = hashlib.sha256()
        for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(module, nnx.Param))[0]:
            local_leaf = leaf.addressable_shards[0].data if (
                hasattr(leaf, "is_fully_replicated") and leaf.is_fully_replicated
            ) else leaf
            array = np.asarray(jax.device_get(local_leaf))
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Nonfinite frozen parameter: {name}/{path}")
            digest.update(str(path).encode())
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes(order="C"))
        result[name] = digest.hexdigest()
    return result
