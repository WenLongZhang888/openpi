import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.models.divl_gemma import DIVLGemmaBackbone
from openpi.models.divl_vision import DIVLVisionEncoder


class DIVLReadoutInput(nnx.Module):
    def __init__(self, config: DIVLStateEncoderConfig, *, rngs: nnx.Rngs):
        self.compute_dtype = jnp.dtype(config.compute_dtype)

        # 解释含义
        self.readout = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (1, 1, config.hidden_dim),
            )
        )

    def __call__(
        self,
        vision_tokens: jax.Array,
        vision_mask: jax.Array,
        text_embeddings: jax.Array,
        text_mask: jax.Array,
    ):
        # vision_tokens: [B, N, D]; vision_mask: [B, N].
        # text_embeddings: [B, T, D]; text_mask: [B, T].
        batch_size = vision_tokens.shape[0]

        # 解释含义
        readout = jnp.broadcast_to(
            self.readout.value,
            (batch_size, 1, self.readout.value.shape[-1]),
        )

        embeddings = jnp.concatenate(
            [vision_tokens, text_embeddings, readout],
            axis=1,
        ).astype(self.compute_dtype)

        valid = jnp.concatenate(
            [vision_mask, text_mask, jnp.ones((batch_size, 1), dtype=jnp.bool_)],
            axis=1,
        )

        # Padding does not consume RoPE positions.
        positions = jnp.maximum(
            jnp.cumsum(valid, axis=1, dtype=jnp.int32) - 1,
            0,
        )

        # Prefix belongs to block 0, and the readout token belongs to the last block.
        length = embeddings.shape[1]
        blocks = jnp.zeros((length,), dtype=jnp.int32)
        blocks = blocks.at[length - 1].set(1)

        # Rows are queries; columns are keys.
        allowed = blocks[:, None] >= blocks[None, :]

        attention_mask = allowed[None, :, :] & valid[:, :, None] & valid[:, None, :]

        embeddings = jnp.where(
            valid[..., None],
            embeddings,
            jnp.zeros_like(embeddings),
        )

        return embeddings, positions, attention_mask


class DIVLStateEncoder(nnx.Module):
    def __init__(self, config: DIVLStateEncoderConfig, *, rngs: nnx.Rngs):
        self.vision = DIVLVisionEncoder(config, rngs=rngs)
        self.gemma = DIVLGemmaBackbone(config, rngs=rngs)
        self.readout_input = DIVLReadoutInput(config, rngs=rngs)

    def __call__(
        self,
        images: dict[str, jax.Array],
        image_masks: dict[str, jax.Array],
        token_ids: jax.Array,
        token_mask: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        vision_tokens, vision_mask = self.vision.encode_views(
            images,
            image_masks,
            train=train,
        )

        text_embeddings = self.gemma.embed_tokens(token_ids)

        embeddings, positions, attention_mask = self.readout_input(
            vision_tokens,
            vision_mask,
            text_embeddings,
            token_mask,
        )

        hidden_states = self.gemma(
            embeddings,
            positions,
            attention_mask,
        )

        return hidden_states[:, -1, :]
