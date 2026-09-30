"""BC updates and uniform-frame sampling for the staged AIRBOT learner."""

import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training.airbot_lwd_data import Chunk


class FrameSampler:
    """Uniform over recorded training frames; never cross an episode boundary."""

    def __init__(self, replay):
        self.replay = replay
        self.ids = [i for i, e in enumerate(replay.episodes) if e.outcome == "success"]
        if not self.ids:
            raise ValueError("BC needs explicitly labeled successful episodes")
        if any(replay.episodes[i].gap is not None for i in self.ids):
            raise ValueError("Uniform frame BC currently requires recordings without gaps")
        self.ends = np.cumsum([len(replay.episodes[i].state) for i in self.ids])

    def sample(self, batch_size, rng):
        positions = rng.integers(int(self.ends[-1]), size=batch_size)
        rows = np.searchsorted(self.ends, positions, side="right")
        starts = positions - np.where(rows > 0, self.ends[np.maximum(rows - 1, 0)], 0)
        chunks = []
        for row, start in zip(rows, starts, strict=True):
            episode_id = self.ids[int(row)]
            total = len(self.replay.episodes[episode_id].state)
            count = min(self.replay.horizon, total - int(start))
            chunks.append(
                Chunk(episode_id, int(start), count, min(int(start) + count, total - 1), int(start) + count == total)
            )
        return chunks


def bc_actions(replay, chunks, normalize, model_action_dim):
    """LeRobot-style repeated terminal action, delta joints, then normalize/pad."""
    rows = []
    for chunk in chunks:
        physical, _, _, _ = replay.transition(chunk)
        physical[chunk.length :] = physical[chunk.length - 1]
        rows.append(physical)
    actions = normalize({"actions": np.stack(rows)})["actions"]
    return np.pad(actions, ((0, 0), (0, 0), (0, model_action_dim - actions.shape[-1]))).astype(np.float32)


@functools.partial(nnx.jit, static_argnames=("ema_decay",))
def bc_train_step(actor, ema_actor, optimizer, observation, actions, rng, *, ema_decay=0.99):
    if actor is ema_actor or optimizer.model is not actor:
        raise ValueError("BC requires an independent EMA and an optimizer bound to actor")
    actor.train()
    next_rng, loss_rng = jax.random.split(rng)

    def loss_fn(model):
        return jnp.mean(model.compute_loss(loss_rng, observation, actions, train=True))

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, optimizer.wrt))(actor)
    optimizer.update(grads)
    nnx.update(
        ema_actor,
        jax.tree.map(
            lambda old, new: ema_decay * old + (1 - ema_decay) * new,
            nnx.state(ema_actor, nnx.Param),
            nnx.state(actor, nnx.Param),
        ),
    )
    actor.eval()
    ema_actor.eval()
    return next_rng, {"actor/bc_loss": loss, "actor/grad_norm": optax.global_norm(grads)}
