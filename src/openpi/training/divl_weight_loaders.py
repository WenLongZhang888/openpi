from pathlib import Path

import flax.nnx as nnx
from flax.traverse_util import unflatten_dict
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.models.divl_vision import DIVLVisionEncoder
from openpi.shared import download

_SIGLIP_REVISION = "d04cf29fca7b6374f74d8bea1969314492266b5e"


def resolve_checkpoint_dirs(
    config: DIVLStateEncoderConfig,
) -> tuple[Path, Path]:
    gemma_dir = download.maybe_download(
        config.gemma_checkpoint,
    )

    siglip_ref = Path(config.siglip_checkpoint).expanduser()
    if siglip_ref.is_dir():
        siglip_dir = siglip_ref
    else:
        siglip_dir = Path(snapshot_download(
            repo_id=config.siglip_checkpoint,
            revision=_SIGLIP_REVISION,
            allow_patterns=[
                "model.safetensors",
                "config.json",
                "preprocessor_config.json",
            ],
        ))

    return gemma_dir, Path(siglip_dir)


def load_siglip_weights(
    encoder: DIVLVisionEncoder,
    checkpoint_dir: Path,
):
    state = nnx.state(encoder.siglip, nnx.Param)
    shapes = {"/".join(path): variable.value.shape for path, variable in state.flat_state().items()}

    with safe_open(
        checkpoint_dir / "model.safetensors",
        framework="np",
    ) as checkpoint:

        def read(name):
            return checkpoint.get_tensor("vision_model." + name)

        # Patch embedding、位置编码和最终 LayerNorm。
        params = {
            "embedding/kernel": read("embeddings.patch_embedding.weight").transpose(2, 3, 1, 0),
            "embedding/bias": read("embeddings.patch_embedding.bias"),
            "pos_embedding": read("embeddings.position_embedding.weight")[None],
            "Transformer/encoder_norm/scale": read("post_layernorm.weight"),
            "Transformer/encoder_norm/bias": read("post_layernorm.bias"),
        }

        # Transformer block 内的参数名称对应关系。
        layer_names = {
            "LayerNorm_0/scale": "layer_norm1.weight",
            "LayerNorm_0/bias": "layer_norm1.bias",
            "LayerNorm_1/scale": "layer_norm2.weight",
            "LayerNorm_1/bias": "layer_norm2.bias",
            "MlpBlock_0/Dense_0/kernel": "mlp.fc1.weight",
            "MlpBlock_0/Dense_0/bias": "mlp.fc1.bias",
            "MlpBlock_0/Dense_1/kernel": "mlp.fc2.weight",
            "MlpBlock_0/Dense_1/bias": "mlp.fc2.bias",
        }

        for flax_name, hf_name in (
            ("query", "q_proj"),
            ("key", "k_proj"),
            ("value", "v_proj"),
            ("out", "out_proj"),
        ):
            prefix = f"MultiHeadDotProductAttention_0/{flax_name}"
            layer_names[f"{prefix}/kernel"] = f"self_attn.{hf_name}.weight"
            layer_names[f"{prefix}/bias"] = f"self_attn.{hf_name}.bias"

        # OpenPI 使用 scan,将各层参数堆叠在第一个维度。
        for target_name, source_name in layer_names.items():
            key = f"Transformer/encoderblock/{target_name}"
            shape = shapes[key]
            layers = []

            for index in range(shape[0]):
                value = read(f"encoder.layers.{index}.{source_name}")
                if target_name.endswith("/kernel"):
                    value = value.T

                layers.append(value.reshape(shape[1:]))

            params[key] = np.stack(layers)

    # 必要检查:预训练参数不能遗漏或形状不匹配。
    if {name: value.shape for name, value in params.items()} != shapes:
        raise ValueError("SigLIP checkpoint does not match the encoder parameters.")

    params = unflatten_dict(params, sep="/")
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, params))
    nnx.update(encoder.siglip, state)
