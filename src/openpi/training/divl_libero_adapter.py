from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi import transforms
from openpi.training.divl_critic import critic_loss


class EpisodeReplay:
    """Small, in-memory replay for initial tests; samples chunks uniformly."""

    def __init__(self, directory, split="train", seed=0):
        self.directory = Path(directory)
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.seen = set()
        self.episodes = []
        self.indices = []

    def refresh(self):
        for path in sorted(self.directory.glob("*.npz")):
            if path in self.seen:
                continue
            with np.load(path, allow_pickle=False) as data:
                if str(data["split"]) == self.split:
                    episode = {key: data[key] for key in data.files}
                    episode_id = len(self.episodes)
                    self.episodes.append(episode)
                    self.indices.extend((episode_id, i) for i in range(len(episode["rewards"])))
            self.seen.add(path)

    def __len__(self):
        return len(self.indices)

    def sample(self, batch_size):
        chosen = self.rng.integers(len(self.indices), size=batch_size)
        return [(self.episodes[e], i) for e, i in (self.indices[j] for j in chosen)]

    def composition(self):
        successes = sum(bool(e["success"]) for e in self.episodes)
        return {
            "episodes": len(self.episodes),
            "successes": successes,
            "failures": len(self.episodes) - successes,
            "chunks": len(self),
        }


def make_batch(samples, tokenizer, norm_stats):
    """Raw physical states/actions -> critic inputs. No second delta transform."""
    normalize = transforms.Normalize(norm_stats, use_quantiles=True)

    def observation(offset):
        state = np.stack([e["state"][i + offset] for e, i in samples])
        state = normalize({"state": state})["state"]
        tokens = [tokenizer.tokenize(str(e["prompt"]), s) for (e, _), s in zip(samples, state, strict=True)]
        images = {
            key: np.stack([e[source][i + offset] for e, i in samples]).astype(np.float32) / 127.5 - 1.0
            for key, source in [("base_0_rgb", "base"), ("left_wrist_0_rgb", "wrist")]
        }
        return {
            "images": images,
            "image_masks": {key: np.ones(len(samples), dtype=bool) for key in images},
            "token_ids": np.stack([x[0] for x in tokens]),
            "token_mask": np.stack([x[1] for x in tokens]),
        }

    actions = np.stack([e["actions"][i] for e, i in samples])
    mask = np.stack([e["action_mask"][i] for e, i in samples])
    actions = normalize({"actions": actions})["actions"]
    actions = np.where(mask[..., None], actions, 0.0).astype(np.float32)
    batch = {
        "obs": observation(0),
        "next_obs": observation(1),
        "actions": actions,
        "action_mask": mask,
        "rewards": np.asarray([e["rewards"][i] for e, i in samples], dtype=np.float32),
        "discounts": np.asarray([e["discounts"][i] for e, i in samples], dtype=np.float32),
    }
    return jax.tree.map(jnp.asarray, batch)


def initialize_training(critic, total_steps, learning_rate=5e-4):
    """Call AFTER loading both pretrained backbones, once per fresh training run.

    Keep FP32 master parameters for Adam and EMA; backbone compute dtype is
    still controlled by its model configuration. All critic parameters train.
    """
    parameters = nnx.state(critic, nnx.Param)
    nnx.update(critic, jax.tree.map(lambda x: x.astype(jnp.float32), parameters))
    target_critic = nnx.clone(critic)
    schedule = optax.cosine_decay_schedule(learning_rate, total_steps)
    optimizer = nnx.Optimizer(critic, optax.adam(schedule), wrt=nnx.Param)
    return target_critic, optimizer


@nnx.jit
def train_step(
    critic, target_critic, optimizer, batch, tau_base=0.6, alpha=0.3, tau_min=0.0, tau_max=1.0, ema_rate=0.005
):
    (loss, metrics), grads = nnx.value_and_grad(critic_loss, has_aux=True)(
        critic,
        target_critic,
        batch,
        tau_base,
        alpha,
        tau_min,
        tau_max,
    )
    optimizer.update(grads)
    current = nnx.state(critic, nnx.Param)
    target = nnx.state(target_critic, nnx.Param)
    nnx.update(
        target_critic,
        jax.tree.map(
            lambda old, new: (1.0 - ema_rate) * old + ema_rate * new,
            target,
            current,
        ),
    )
    return {**metrics, "loss": loss, "grad_norm": optax.global_norm(grads)}


@nnx.jit
def evaluate(critic, target_critic, batch, mc_returns, tau_base=0.6, alpha=0.3, tau_min=0.0, tau_max=1.0):
    loss, metrics = critic_loss(critic, target_critic, batch, tau_base, alpha, tau_min, tau_max)
    z = critic.encoder(**batch["obs"], train=False)
    q = jnp.minimum(*critic.q_head(z, batch["actions"], batch["action_mask"]))
    return {**metrics, "loss": loss, "q_mc_mse": jnp.mean((q - mc_returns) ** 2)}
