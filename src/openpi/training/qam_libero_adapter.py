"""Build aligned actor/DIVL inputs from the existing LIBERO episode replay."""

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import libero_policy
from openpi.training import divl_libero_adapter as divl
from openpi.training.qam_training import DEFAULT_QAM_CONFIG
from openpi.training.qam_training import QAMActionStats
from openpi.training.qam_training import QAMBatch
from openpi.training.qam_training import QAMTrainingConfig
from openpi.training.qam_training import quantile_pair


class LiberoQAMBatchBuilder:
    def __init__(
        self,
        model_config: pi0_config.Pi0Config,
        actor_norm_stats,
        critic_norm_stats,
        critic_tokenizer,
        *,
        actor_tokenizer=None,
        training_config: QAMTrainingConfig = DEFAULT_QAM_CONFIG,
        td_steps: int = 10,
    ):
        """Use statistics from each model's own checkpoint/training setup.

        This adapter matches pi05_libero with no extra delta-action transform.
        Pass the BC model's exact Pi0Config, including discrete_state_input.
        Tokenizers are separate; the optional actor tokenizer supports reuse.
        """
        if not model_config.pi05:
            raise ValueError("This adapter supports pi05_libero without an extra delta-action transform")
        if isinstance(td_steps, bool) or not isinstance(td_steps, int) or td_steps < 1:
            raise ValueError("td_steps must be a positive integer")
        self.td_steps = td_steps
        if model_config.action_dim < 8 or model_config.action_horizon < training_config.replan_steps:
            raise ValueError("model cannot represent the LIBERO state and execution prefix")
        quantile_pair(actor_norm_stats, "state", 8)
        quantile_pair(critic_norm_stats, "state", 8)
        self.action_stats = QAMActionStats.from_norm_stats(actor_norm_stats, critic_norm_stats)
        self.config = training_config
        self.critic_norm_stats = critic_norm_stats
        self.critic_tokenizer = critic_tokenizer
        if actor_tokenizer is None:
            actor_tokenizer = _tokenizer.PaligemmaTokenizer(model_config.max_token_len)
        # Same inference preprocessing order as policy_config.create_trained_policy.
        self.actor_transform = transforms.compose(
            [
                libero_policy.LiberoInputs(model_type=model_config.model_type),
                transforms.Normalize(actor_norm_stats, use_quantiles=True),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(actor_tokenizer, discrete_state_input=model_config.discrete_state_input),
                transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )

    def __call__(self, samples) -> QAMBatch:
        samples = list(samples)
        if not samples:
            raise ValueError("Cannot build an empty QAM batch")
        actor_inputs = []
        for episode, index in samples:
            if not 0 <= index < len(episode["rewards"]):
                raise ValueError("Replay index must identify a current-state transition")
            actions = episode["actions"][index]
            mask = np.asarray(episode["action_mask"][index], dtype=bool)
            if actions.shape != (self.config.critic_horizon, 7) or mask.shape != (self.config.critic_horizon,):
                raise ValueError("Replay action shape does not match the critic configuration")
            length = int(mask.sum())
            if not 1 <= length <= self.config.replan_steps or not np.array_equal(mask, np.arange(len(mask)) < length):
                raise ValueError("Replay mask must describe a nonempty executed prefix within replan_steps")
            # Stored images are already rotated/resized by policy_observation.
            # Use the CURRENT state, preserving the same sample order as DIVL.
            raw = {
                "observation/image": episode["base"][index],
                "observation/wrist_image": episode["wrist"][index],
                "observation/state": np.asarray(episode["state"][index], dtype=np.float32).copy(),
                "prompt": str(episode["prompt"]),
            }
            actor_inputs.append(self.actor_transform(raw))
        stacked = jax.tree.map(lambda *values: np.stack(values), *actor_inputs)
        observation = jax.tree.map(jnp.asarray, _model.Observation.from_dict(stacked))
        critic_batch = divl.make_batch(samples, self.critic_tokenizer, self.critic_norm_stats, td_steps=self.td_steps)
        return QAMBatch(actor_observation=observation, critic=critic_batch)
