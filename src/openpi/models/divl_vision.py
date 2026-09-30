import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import siglip
from openpi.models.divl_config import DIVLStateEncoderConfig


class DIVLVisionEncoder(nnx.Module):
    def __init__(self, config: DIVLStateEncoderConfig, *, rngs: nnx.Rngs):
        self.siglip = nnx_bridge.ToNNX(
            siglip.Module(
                variant="So400m/14",
                num_classes=None,
                pool_type="none",
                scan=True,
                dtype_mm=config.compute_dtype,
                attention_dtype=config.siglip_attention_dtype,
            )
        )

        self.image_keys = config.image_keys

        dummy_image = jnp.zeros((1, config.image_size, config.image_size, 3), dtype=jnp.float32)
        self.siglip.lazy_init(
            dummy_image,
            train=False,
            rngs=rngs,
        )

        self.projection = nnx.Linear(
            config.vision_hidden_dim,
            config.hidden_dim,
            dtype=jnp.dtype(config.compute_dtype),
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(
        self,
        image: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        features, _ = self.siglip(image, train=train)
        return self.projection(features)

    def encode_views(
        self,
        images: dict[str, jax.Array],
        image_masks: dict[str, jax.Array],
        *,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        tokens_per_camera = []
        masks_per_camera = []

        for name in self.image_keys:
            # Reuse the same SigLIP and projection for each camera view.
            tokens = self(images[name], train=train)

            # Camera validity [B] -> patch validity [B, P]
            mask = jnp.broadcast_to(
                image_masks[name][:, None],
                tokens.shape[:2],
            )

            tokens_per_camera.append(tokens)
            masks_per_camera.append(mask)

        return (
            jnp.concatenate(tokens_per_camera, axis=1),
            jnp.concatenate(masks_per_camera, axis=1),
        )
