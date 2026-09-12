from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DIVLStateEncoderConfig:
    # 论文指定的 VLM backbone: Gemma 3-270M+SigLIP-So400M
    gemma_checkpoint: str = "gs://gemma-data/checkpoints/gemma3-270m-it"
    siglip_checkpoint: str = "google/siglip-so400m-patch14-224"
    gemma_tokenizer: str = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

    image_keys: tuple[str, ...] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
    )

    state_dim: int = 8
    state_bins: int = 256

    max_text_tokens: int = 200

    compute_dtype: Literal["bfloat16", "float32"] = "bfloat16"

    @property
    def image_size(self) -> int:
        return 224

    @property
    def vision_hidden_dim(self) -> int:
        return 1152

    @property
    def hidden_dim(self) -> int:
        return 640
