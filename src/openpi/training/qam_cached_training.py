"""Cached-prefix QAM for fixed-reference AIRBOT training.

Actor prefix caches remain inside differentiation; reference velocities are
reused only at the same states and times used during path generation.
"""

import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.training import qam
from openpi.training.qam_training import _validate_actor_update


def prefix_cache(model, observation):
    tokens, mask, ar_mask = model.embed_prefix(observation)
    positions = jnp.cumsum(mask, axis=1) - 1
    _, kv = model.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask), positions=positions)
    return mask, kv


class CachedVelocity:
    def __init__(self, model, cache):
        self.model = model
        self.mask, self.kv = cache
        self.action_horizon = model.action_horizon
        self.action_dim = model.action_dim

    def predict_velocity(self, observation, noisy_actions, time):
        suffix, mask, ar_mask, cond = self.model.embed_suffix(observation, noisy_actions, time)
        suffix_attention = make_attn_mask(mask, ar_mask)
        prefix_attention = jnp.broadcast_to(self.mask[:, None, :], (mask.shape[0], mask.shape[1], self.mask.shape[1]))
        attention = jnp.concatenate([prefix_attention, suffix_attention], axis=-1)
        positions = jnp.sum(self.mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        (_, out), _ = self.model.PaliGemma.llm(
            [None, suffix], mask=attention, positions=positions, kv_cache=self.kv, adarms_cond=[None, cond]
        )
        return self.model.action_out_proj(out[:, -self.action_horizon :])


def sample_with_velocities(reference, observation, rng, *, config):
    times = jnp.linspace(config.epsilon, 1.0, config.num_steps + 1, dtype=jnp.float32)
    init_rng, path_rng = jax.random.split(rng)
    shape = (observation.state.shape[0], reference.action_horizon, reference.action_dim)
    initial = jax.random.normal(init_rng, shape, dtype=jnp.float32)
    keys = jax.random.split(path_rng, config.num_steps)

    def step(actions, inputs):
        w, dw, key = inputs
        velocity = qam.qam_velocity(reference, observation, actions, w)
        drift = 2 * velocity - actions / w
        sigma2 = 2 * (1 - w) / w
        nxt = actions + dw * drift + jnp.sqrt(dw * sigma2) * jax.random.normal(key, shape, dtype=jnp.float32)
        nxt = jax.lax.stop_gradient(nxt)
        return nxt, (nxt, jax.lax.stop_gradient(velocity))

    penultimate, (following, velocities) = jax.lax.scan(step, initial, (times[:-2], jnp.diff(times)[:-1], keys[:-1]))
    last_v = qam.qam_velocity(reference, observation, penultimate, times[-2])
    endpoint = penultimate + (times[-1] - times[-2]) * last_v
    path = jnp.concatenate([initial[None], following, endpoint[None]], axis=0)
    velocities = jnp.concatenate([velocities, last_v[None]], axis=0)
    return times, jax.lax.stop_gradient(path), jax.lax.stop_gradient(velocities)


def cached_actor_loss(actor, observation, times, path, adjoints, reference_velocities):
    # Construct INSIDE differentiation: image/prefix parameters retain gradients.
    cached_actor = CachedVelocity(actor, prefix_cache(actor, observation))

    def step(_, inputs):
        w, actions, adjoint, reference_velocity = inputs
        sigma2 = 2 * (1 - w) / w
        target = jax.lax.stop_gradient(reference_velocity - 0.5 * sigma2 * adjoint)
        velocity = qam.qam_velocity(cached_actor, observation, actions, w)
        loss = jnp.mean(4 / sigma2 * jnp.sum((velocity - target) ** 2, axis=(-2, -1)))
        deviation = jnp.mean(jnp.sum((velocity - reference_velocity) ** 2, axis=(-2, -1)))
        return None, (loss, deviation)

    _, (losses, deviations) = jax.lax.scan(
        jax.checkpoint(step), None, (times[:-1], path[:-1], adjoints[:-1], reference_velocities)
    )
    widths = jnp.diff(times)
    loss = jnp.sum(widths * losses)
    return loss, {"qam_loss": loss, "velocity_deviation_integral": jnp.sum(widths * deviations)}


@functools.partial(nnx.jit, static_argnames=("config",))
def actor_train_step(
    actor, reference, critic, optimizer, actor_observation, critic_observation, action_stats, rng, *, config
):
    _validate_actor_update(actor, reference, optimizer, config)
    if not config.sample_from_reference:
        raise ValueError("Velocity reuse is valid only for fixed-reference sampling")
    actor.eval()
    reference.eval()
    critic.eval()
    observation = model_lib.preprocess_observation(None, actor_observation, train=False)
    next_rng, sample_rng = jax.random.split(rng)
    z = jax.lax.stop_gradient(critic.encoder(**critic_observation, train=False))
    cached_reference = CachedVelocity(
        reference, jax.tree.map(jax.lax.stop_gradient, prefix_cache(reference, observation))
    )
    times, path, reference_velocities = sample_with_velocities(cached_reference, observation, sample_rng, config=config)
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
        action_dim=config.action_dim,
        action_clip=config.action_clip,
    )
    adjoints = qam.qam_backward_adjoint(cached_reference, observation, times, path, terminal)
    (_, metrics), grads = nnx.value_and_grad(cached_actor_loss, argnums=nnx.DiffState(0, optimizer.wrt), has_aux=True)(
        actor, observation, times, path, adjoints, reference_velocities
    )
    norm = optax.global_norm(grads)
    optimizer.update(grads)
    result = {**metrics, **terminal_metrics, "grad_norm": norm, "path_max_abs": jnp.max(jnp.abs(path))}
    return next_rng, {f"actor/{k}": jax.lax.stop_gradient(v) for k, v in result.items()}
