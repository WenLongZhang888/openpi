import numpy as np
import sentencepiece

from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.shared import download


class DIVLTokenizer:
    def __init__(
        self,
        config: DIVLStateEncoderConfig,
    ):
        self.max_len = config.max_text_tokens
        self.state_bins = config.state_bins

        path = download.maybe_download(config.gemma_tokenizer)
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(path),
        )

    def tokenize(
        self,
        prompt: str,
        normalized_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(normalized_state, dtype=np.float32)

        bins = np.linspace(-1, 1, self.state_bins + 1)[:-1]

        state_ids = np.digitize(state, bins=bins) - 1
        state_ids = np.clip(state_ids, 0, self.state_bins - 1)

        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ").replace("\t", " ")
        state_text = " ".join(map(str, state_ids))
        text = f"Task: {cleaned_prompt}, State: {state_text};\n"

        ids = self.tokenizer.encode(text, add_bos=True)
        if len(ids) > self.max_len:
            raise ValueError(f"DIVL text has {len(ids)} tokens, exceeding max_text_tokens={self.max_len}.")

        token_ids = np.full(
            self.max_len,
            self.tokenizer.pad_id(),
            dtype=np.int32,
        )
        token_ids[: len(ids)] = ids

        token_mask = np.arange(self.max_len) < len(ids)

        return token_ids, token_mask
