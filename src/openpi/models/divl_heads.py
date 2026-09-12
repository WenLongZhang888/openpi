import flax.nnx as nnx
import jax
import jax.numpy as jnp


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 512,
        rngs: nnx.Rngs,
    ):
        self.net = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, dtype=jnp.float32, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim, hidden_dim, dtype=jnp.float32, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim, output_dim, dtype=jnp.float32, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)


class DistributionalVHead(nnx.Module):
    def __init__(
        self,
        input_dim: int = 640,
        *,
        hidden_dim: int = 512,
        num_atoms: int = 201,
        v_min: float = -0.1,
        v_max: float = 1.1,
        rngs: nnx.Rngs,
    ):
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.net = MLP(input_dim, num_atoms, hidden_dim=hidden_dim, rngs=rngs)

    @property
    def atoms(self) -> jax.Array:
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms, dtype=jnp.float32)

    def __call__(self, z: jax.Array) -> jax.Array:
        # [B, state_dim] -> [B, num_atoms], returns logits.
        return self.net(z)


class TemporalActionPool(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        *,
        embed_dim: int = 256,
        rngs: nnx.Rngs,
    ):
        self.action_proj = nnx.Linear(
            action_dim,
            embed_dim,
            dtype=jnp.float32,
            rngs=rngs,
        )

        self.position_embedding = nnx.Param(0.02 * jax.random.normal(rngs.params(), (action_horizon, embed_dim)))
        self.score = nnx.Linear(
            embed_dim,
            1,
            use_bias=False,
            dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(
        self,
        actions: jax.Array,
        action_mask: jax.Array | None = None,
    ) -> jax.Array:
        # actions: [B, H, action_dim], 已经归一化的连续动作
        horizon = actions.shape[1]
        tokens = self.action_proj(actions)
        tokens = jnp.tanh(tokens + self.position_embedding.value[None, :horizon, :])

        scores = self.score(tokens)[..., 0]  # [B, H]
        if action_mask is not None:
            scores = jnp.where(action_mask, scores, -jnp.inf)

        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.sum(weights[..., None] * tokens, axis=1)


class TwinQHead(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        *,
        state_dim: int = 640,
        action_embed_dim: int = 256,
        hidden_dim: int = 512,
        rngs: nnx.Rngs,
    ):
        self.action_pool = TemporalActionPool(
            action_dim,
            action_horizon,
            embed_dim=action_embed_dim,
            rngs=rngs,
        )

        input_dim = state_dim + action_embed_dim
        self.q1 = MLP(
            input_dim,
            1,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.q2 = MLP(
            input_dim,
            1,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        z: jax.Array,
        actions: jax.Array,
        action_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        action_features = self.action_pool(actions, action_mask)
        features = jnp.concatenate([z.astype(jnp.float32), action_features], axis=-1)
        return self.q1(features)[..., 0], self.q2(features)[..., 0]
