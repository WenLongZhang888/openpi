"""Offline QAM updates for an already loaded BC actor and DIVL critic."""

import dataclasses
import functools
import math

from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _model
from openpi.models import pi0
from openpi.training import divl_lwd_training as divl
from openpi.training import qam


@dataclasses.dataclass(frozen=True)
class QAMTrainingConfig:
    num_steps: int = 10
    epsilon: float = 0.1
    temperature: float = 2.0
    replan_steps: int = 5
    critic_horizon: int = 30
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in ("num_steps", "replan_steps", "critic_horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_steps < 2:
            raise ValueError("QAM training needs at least one current-actor SDE step and one reference ODE step")
        if not 0 < self.epsilon < 1 or (1 - self.epsilon) / self.num_steps > self.epsilon:
            raise ValueError("epsilon must be in (0, 1) and resolve the SDE step size")
        if self.replan_steps > self.critic_horizon:
            raise ValueError("replan_steps cannot exceed critic_horizon")
        for name in ("temperature", "learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclasses.dataclass(frozen=True)
class DIVLTrainingConfig:
    quantile_level: float = 0.6
    alpha: float = 0.3
    tau_min: float = 0.0
    tau_max: float = 1.0
    ema_rate: float = 0.005

    def __post_init__(self):
        if not 0 <= self.tau_min <= self.quantile_level <= self.tau_max <= 1:
            raise ValueError("DIVL quantile bounds must satisfy 0 <= min <= level <= max <= 1")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("DIVL alpha must be finite and nonnegative")
        if not 0 <= self.ema_rate <= 1:
            raise ValueError("DIVL ema_rate must be in [0, 1]")


DEFAULT_QAM_CONFIG = QAMTrainingConfig()
DEFAULT_DIVL_CONFIG = DIVLTrainingConfig()


def quantile_pair(norm_stats, key, dimensions):
    """Validate statistics on the host before using them in a compiled update."""
    if key not in norm_stats:
        raise ValueError(f"Missing {key} normalization statistics")
    stats = norm_stats[key]
    if stats.q01 is None or stats.q99 is None:
        raise ValueError(f"{key} requires q01 and q99 statistics")
    low, high = np.asarray(stats.q01, dtype=np.float32), np.asarray(stats.q99, dtype=np.float32)
    if low.ndim != 1 or high.shape != low.shape or low.size < dimensions:
        raise ValueError(f"{key} quantiles must be matching vectors with at least {dimensions} entries")
    low, high = low[:dimensions], high[:dimensions]
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
        raise ValueError(f"{key} quantiles must be finite and ordered")
    return low, high


@struct.dataclass
class QAMActionStats:
    actor_q01: jax.Array
    actor_q99: jax.Array
    critic_q01: jax.Array
    critic_q99: jax.Array

    @classmethod
    def from_norm_stats(cls, actor_norm_stats, critic_norm_stats):
        actor_low, actor_high = quantile_pair(actor_norm_stats, "actions", 7)
        critic_low, critic_high = quantile_pair(critic_norm_stats, "actions", 7)
        return cls(*map(jnp.asarray, (actor_low, actor_high, critic_low, critic_high)))


@struct.dataclass
class QAMBatch:
    actor_observation: _model.Observation
    critic: dict


def initialize_actor_training(
    bc_actor: pi0.Pi0, config: QAMTrainingConfig = DEFAULT_QAM_CONFIG
) -> tuple[pi0.Pi0, nnx.Optimizer]:
    """Initialize ONCE, immediately after loading the BC checkpoint.

    Clone the independent reference before changing actor parameter dtypes.
    Reference keeps the loaded precision; actor gets FP32 master parameters.
    All actor parameters train offline. This is not a resume function: a
    resumed run must restore its original reference and optimizer state.
    """
    if bc_actor.action_dim < 7 or bc_actor.action_horizon < config.replan_steps:
        raise ValueError("BC actor cannot supply the configured LIBERO execution prefix")
    bc_actor.eval()
    reference = nnx.clone(bc_actor)
    parameters = nnx.state(bc_actor, nnx.Param)
    nnx.update(bc_actor, jax.tree.map(lambda value: value.astype(jnp.float32), parameters))
    optimizer = nnx.Optimizer(
        bc_actor,
        optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
        ),
        wrt=nnx.Param,
    )
    return reference, optimizer


def _validate_actor_update(actor, reference, optimizer, config):
    if actor is reference:
        raise ValueError("reference must be an independent BC model")
    if optimizer.model is not actor:
        raise ValueError("actor optimizer belongs to a different model")
    if actor.action_dim < 7 or actor.action_horizon < config.replan_steps:
        raise ValueError("actor cannot supply the configured LIBERO execution prefix")
    if (actor.action_horizon, actor.action_dim) != (reference.action_horizon, reference.action_dim):
        raise ValueError("actor and reference must use the same action shape")


@functools.partial(nnx.jit, static_argnames=("config",))
def actor_train_step(
    actor: pi0.Pi0,
    reference: pi0.Pi0,
    critic: nnx.Dict,
    optimizer: nnx.Optimizer,
    actor_observation: _model.Observation,
    critic_observation: dict,
    action_stats: QAMActionStats,
    rng: jax.Array,
    *,
    config: QAMTrainingConfig = DEFAULT_QAM_CONFIG,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Build fresh supervision using the current critic, then update actor.

    Both observations must describe the SAME raw states in the same order.
    No image augmentation: actor and critic see the collected images. eval()
    controls stochastic layers; it does not disable actor parameter gradients.
    The caller must retain the returned RNG for the next update.
    """
    _validate_actor_update(actor, reference, optimizer, config)
    actor.eval()
    reference.eval()
    critic.eval()
    observation = _model.preprocess_observation(None, actor_observation, train=False)
    next_rng, sample_rng = jax.random.split(rng)

    # State encoding is independent of generated actions, so compute it once.
    z = jax.lax.stop_gradient(critic.encoder(**critic_observation, train=False))
    times, path = qam.sample_qam_path(
        actor, observation, sample_rng, reference=reference, num_steps=config.num_steps, epsilon=config.epsilon
    )
    terminal, terminal_metrics = qam.qam_terminal_adjoint(
        critic,
        z,
        path[-1],
        action_stats.actor_q01,
        action_stats.actor_q99,
        action_stats.critic_q01,
        action_stats.critic_q99,
        temperature=config.temperature,
        replan_steps=config.replan_steps,
        critic_horizon=config.critic_horizon,
    )
    adjoints = qam.qam_backward_adjoint(reference, observation, times, path, terminal)
    # Supervision was constructed outside the differentiated actor loss.
    (_, loss_metrics), grads = nnx.value_and_grad(
        qam.qam_actor_loss, argnums=nnx.DiffState(0, optimizer.wrt), has_aux=True
    )(actor, reference, observation, times, path, adjoints)
    grad_norm = optax.global_norm(grads)
    optimizer.update(grads)
    metrics = {
        **loss_metrics,
        **terminal_metrics,
        "grad_norm": grad_norm,
        "path_max_abs": jnp.max(jnp.abs(path)),
    }
    return next_rng, {f"actor/{key}": jax.lax.stop_gradient(value) for key, value in metrics.items()}


def offline_train_step(
    actor: pi0.Pi0,
    reference: pi0.Pi0,
    critic: nnx.Dict,
    target_critic: nnx.Dict,
    actor_optimizer: nnx.Optimizer,
    value_optimizer: nnx.Optimizer,
    critic_optimizer: nnx.Optimizer,
    batch: QAMBatch,
    action_stats: QAMActionStats,
    rng: jax.Array,
    *,
    qam_config: QAMTrainingConfig = DEFAULT_QAM_CONFIG,
    divl_config: DIVLTrainingConfig = DEFAULT_DIVL_CONFIG,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """LWD Algorithm 2: V -> Q -> EMA -> QAM with the UPDATED online Q.

    The two stages are separately JIT-compiled. Replay actions belong only
    to the DIVL update; the QAM endpoint always comes from fresh generation.
    Initialize target/value/Q optimizers with divl_lwd_training.initialize_training
    after loading critic backbones. Offline RL updates all modules from step 1;
    a separate critic-only warmup is not required by LWD.
    """
    _validate_actor_update(actor, reference, actor_optimizer, qam_config)
    if critic is target_critic or critic_optimizer.model is not critic or value_optimizer.model is not critic:
        raise ValueError("critic needs an independent target and its own optimizer")
    if batch.critic["actions"].shape[1:] != (qam_config.critic_horizon, 7):
        raise ValueError("replay action shape does not match the QAM critic configuration")
    if batch.actor_observation.state.shape[0] != batch.critic["actions"].shape[0]:
        raise ValueError("actor and critic batch sizes differ")
    critic_metrics = divl.train_step(
        critic,
        target_critic,
        value_optimizer,
        critic_optimizer,
        batch.critic,
        tau_base=divl_config.quantile_level,
        alpha=divl_config.alpha,
        tau_min=divl_config.tau_min,
        tau_max=divl_config.tau_max,
        ema_rate=divl_config.ema_rate,
    )
    next_rng, actor_metrics = actor_train_step(
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
    return next_rng, {**actor_metrics, **{f"critic/{key}": value for key, value in critic_metrics.items()}}
