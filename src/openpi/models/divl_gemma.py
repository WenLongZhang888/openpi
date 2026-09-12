import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from gemma import gm
import jax
import jax.numpy as jnp

from openpi.models.divl_config import DIVLStateEncoderConfig


class _GemmaForCritic(gm.nn.Gemma3_270M):
    def embed_tokens(self, token_ids: jax.Array) -> jax.Array:
        # Reuse the official embedding table and embedding scaling
        return self.embedder.encode(token_ids)

    def encode_embeddings(
        self,
        embeddings: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
        x = embeddings

        for block in self.blocks:
            # No KV cache is needed for this full-sequence forward
            _, x = block(x, positions, None, attention_mask)

        return self.final_norm(x)


class DIVLGemmaBackbone(nnx.Module):
    def __init__(self, config: DIVLStateEncoderConfig, *, rngs: nnx.Rngs):
        self.compute_dtype = jnp.dtype(config.compute_dtype)

        self.model = nnx_bridge.ToNNX(_GemmaForCritic(dtype=self.compute_dtype))

        self.model.lazy_init(
            jnp.full((1, 1), 2, dtype=jnp.int32),
            return_last_only=True,
            rngs=rngs,
        )

    def embed_tokens(self, token_ids: jax.Array) -> jax.Array:
        return self.model(token_ids, method="embed_tokens").astype(self.compute_dtype)

    def __call__(
        self,
        embeddings: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
        return self.model(
            embeddings,
            positions,
            attention_mask,
            method="encode_embeddings",
        ).astype(self.compute_dtype)
