"""Offline AIRBOT replay from LeRobot v2.1, with episode labels and gap boundaries."""

from collections import Counter
from collections import OrderedDict
import dataclasses
import hashlib
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import aloha_policy
from openpi.training.qam_training import QAMActionStats
from openpi.training.qam_training import QAMBatch


@dataclasses.dataclass(frozen=True)
class AirbotDataConfig:
    """Dataset/model boundary for one AIRBOT training configuration."""

    fps: int = 25
    action_dim: int = 14
    norm_asset_id: str = "airbot_cube_200"
    camera_mapping: tuple[tuple[str, str], ...] = (
        ("cam_high", "base_0_rgb"),
        ("cam_left_wrist", "left_wrist_0_rgb"),
        ("cam_right_wrist", "right_wrist_0_rgb"),
    )
    delta_action_indices: tuple[int, ...] = (*range(6), *range(7, 13))
    state_key: str = "observation.state"
    action_key: str = "action"
    image_storage: str = "embedded"

    def __post_init__(self):
        if self.fps < 1 or self.action_dim < 1:
            raise ValueError("fps and action_dim must be positive")
        if not self.norm_asset_id.strip():
            raise ValueError("norm_asset_id must not be empty")
        if not self.camera_mapping:
            raise ValueError("camera_mapping must not be empty")
        if self.image_storage not in ("embedded", "external_video"):
            raise ValueError("image_storage must be embedded or external_video")
        if not self.state_key.strip() or not self.action_key.strip():
            raise ValueError("state_key and action_key must not be empty")
        policy_names, dataset_names = zip(*self.camera_mapping, strict=True)
        if len(set(policy_names)) != len(policy_names) or len(set(dataset_names)) != len(dataset_names):
            raise ValueError("camera_mapping needs unique policy and dataset camera names")
        if len(set(self.delta_action_indices)) != len(self.delta_action_indices) or any(
            not 0 <= index < self.action_dim for index in self.delta_action_indices
        ):
            raise ValueError("delta_action_indices must be unique valid action dimensions")

    @classmethod
    def from_dict(cls, payload):
        if "all_success" in payload:
            raise ValueError(
                "airbot.all_success was removed; supply explicit outcome labels in episode metadata or conversion manifest"
            )
        camera_mapping = payload.get("camera_mapping")
        return cls(
            fps=int(payload.get("fps", 25)),
            action_dim=int(payload.get("action_dim", 14)),
            norm_asset_id=str(payload.get("norm_asset_id", "airbot_cube_200")),
            camera_mapping=cls.camera_mapping
            if camera_mapping is None
            else tuple((str(policy), str(dataset)) for policy, dataset in camera_mapping.items()),
            delta_action_indices=tuple(
                int(index) for index in payload.get("delta_action_indices", cls.delta_action_indices)
            ),
            state_key=str(payload.get("state_key", "observation.state")),
            action_key=str(payload.get("action_key", "action")),
            image_storage=str(payload.get("image_storage", "embedded")),
        )

    @property
    def camera_keys(self):
        return tuple(dataset for _, dataset in self.camera_mapping)

    @property
    def joint_mask(self):
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[np.asarray(self.delta_action_indices)] = True
        return mask


DEFAULT_AIRBOT_DATA_CONFIG = AirbotDataConfig()
# Compatibility constants for callers that only need the default robot schema.
CAMERAS = DEFAULT_AIRBOT_DATA_CONFIG.camera_keys
JOINT_MASK = DEFAULT_AIRBOT_DATA_CONFIG.joint_mask


@dataclasses.dataclass
class Episode:
    dataset: str
    index: int
    path: Path
    source_file: str
    source_sha256: str
    outcome: str
    state: np.ndarray
    actions: np.ndarray
    gap: int | None


@dataclasses.dataclass(frozen=True)
class Chunk:
    episode: int
    start: int
    length: int
    next_index: int
    terminal: bool


def episode_chunks(episode_id: int, length: int, horizon: int, gap: int | None) -> list[Chunk]:
    """Drop the action at a recording gap; bootstrap from the last pre-gap state.

    A segment ending at frame g uses actions before g and next observation g.
    No transition from g to g+1 is invented. The final logged observation is
    used as a placeholder next observation only at a true terminal (discount=0).
    """
    if length < 2 or horizon < 1:
        raise ValueError("Need at least two frames and a positive horizon")
    if gap is not None and not 0 <= gap < length - 1:
        raise ValueError("Gap must lie between two recorded frames")
    segments = [(0, length, True)] if gap is None else [(0, gap, False), (gap + 1, length, True)]
    chunks = []
    for begin, end, final_segment in segments:
        for start in range(begin, end, horizon):
            count = min(horizon, end - start)
            terminal = final_segment and start + count == length
            chunks.append(Chunk(episode_id, start, count, min(start + count, length - 1), terminal))
    return chunks


def resolve_episode_label(row, manifest_row=None, dataset="dataset"):
    """Resolve explicit labels; never infer success from data origin or file names."""
    identity = f"{dataset}/{row['episode_index']}"
    labels = []
    for source in (row, manifest_row):
        if source is None:
            continue
        if "outcome" in source:
            if source["outcome"] not in ("success", "failure"):
                raise ValueError(f"Invalid outcome for {identity}: {source['outcome']!r}")
            labels.append(source["outcome"])
    # Older conversion manifests do not carry a status field; their presence
    # means the conversion record itself is complete.  Reject an explicit
    # non-complete status so partially written conversions cannot enter replay.
    if manifest_row is not None and "status" in manifest_row and manifest_row["status"] != "complete":
        raise ValueError(f"Incomplete conversion for {identity}")
    if not labels:
        raise ValueError(
            f"Missing outcome for {identity}; annotate success/failure in meta/episodes.jsonl or meta/conversion_manifest.jsonl"
        )
    if len(set(labels)) != 1:
        raise ValueError(f"Episode and conversion labels disagree: {identity}")
    for key in ("source_file", "source_sha256", "alignment_gap_after_step"):
        if manifest_row is not None and key in row and key in manifest_row and row[key] != manifest_row[key]:
            raise ValueError(f"Episode and conversion {key} disagree: {identity}")
    merged = {**row, **(manifest_row or {})}
    return (
        labels[0],
        merged.get("source_file", f"episode_{row['episode_index']:06d}.parquet"),
        merged.get("source_sha256", ""),
        merged.get("alignment_gap_after_step"),
    )


class AirbotReplay:
    def __init__(
        self,
        datasets,
        *,
        horizon=32,
        gamma=0.9999,
        exclude_episodes=(),
        data_config=DEFAULT_AIRBOT_DATA_CONFIG,
    ):
        if not 0 < gamma <= 1:
            raise ValueError("Invalid discount")
        exclusions = list(exclude_episodes)
        try:
            exclusion_keys = {(str(row["dataset"]), str(row["source_file"])) for row in exclusions}
        except (KeyError, TypeError) as exc:
            raise ValueError("Each exclusion needs dataset and source_file") from exc
        if len(exclusion_keys) != len(exclusions):
            raise ValueError("Duplicate episode exclusion")
        self.horizon, self.gamma = horizon, gamma
        self.data_config = data_config
        self.episodes = []
        self.excluded_episodes = []
        self._image_cache = OrderedDict()
        self.metadata_hashes = {}
        self._dataset_info = {}
        self._video_cache = OrderedDict()
        matched_exclusions = set()
        for directory in datasets:
            root = Path(directory).resolve()
            info = json.loads((root / "meta/info.json").read_text())
            state_key, action_key = data_config.state_key, data_config.action_key
            if info["fps"] != data_config.fps or any(
                info["features"].get(key, {}).get("shape") != [data_config.action_dim]
                for key in (action_key, state_key)
            ):
                raise ValueError(f"Expected {data_config.fps} Hz, {data_config.action_dim}D AIRBOT data: {root}")
            self._dataset_info[root.name] = (root, info)
            episodes_path = root / "meta/episodes.jsonl"
            episodes_raw = episodes_path.read_bytes()
            self.metadata_hashes[str(episodes_path)] = hashlib.sha256(episodes_raw).hexdigest()
            metadata = {"episodes.jsonl": [json.loads(line) for line in episodes_raw.splitlines() if line.strip()]}
            manifest_path = root / "meta/conversion_manifest.jsonl"
            manifest_by_index = {}
            if manifest_path.exists():
                manifest_raw = manifest_path.read_bytes()
                self.metadata_hashes[str(manifest_path)] = hashlib.sha256(manifest_raw).hexdigest()
                manifest = [json.loads(line) for line in manifest_raw.splitlines() if line.strip()]
                latest = {row["source_file"]: row for row in manifest}
                manifest_by_index = {row["episode_index"]: row for row in latest.values()}
                if len(latest) != len(manifest_by_index):
                    raise ValueError(f"Ambiguous manifest: {root}")
            episode_rows = metadata["episodes.jsonl"]
            indices = [row["episode_index"] for row in episode_rows]
            if len(indices) != info["total_episodes"] or len(set(indices)) != len(indices):
                raise ValueError(f"Incomplete/duplicate episode metadata: {root}")
            if set(manifest_by_index) - set(indices):
                raise ValueError(f"Manifest refers to unknown episodes: {root}")
            # Validate ALL labels before reading any parquet (including excluded episodes).
            resolved = [
                resolve_episode_label(row, manifest_by_index.get(row["episode_index"]), root.name)
                for row in episode_rows
            ]
            for row, (outcome, source_file, source_sha256, alignment_gap) in zip(episode_rows, resolved, strict=True):
                index = row["episode_index"]
                exclusion_key = (root.name, source_file)
                if exclusion_key in exclusion_keys:
                    matched_exclusions.add(exclusion_key)
                    self.excluded_episodes.append(
                        {
                            "dataset": root.name,
                            "episode_index": index,
                            "source_file": source_file,
                            "source_sha256": source_sha256,
                            "outcome": outcome,
                            "frames": row["length"],
                        }
                    )
                    continue
                path = root / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
                table = pq.read_table(path, columns=[state_key, action_key, "frame_index", "episode_index"])
                state = np.asarray(table[state_key].to_pylist(), dtype=np.float32)
                actions = np.asarray(table[action_key].to_pylist(), dtype=np.float32)
                if state.shape != (row["length"], data_config.action_dim) or actions.shape != state.shape:
                    raise ValueError(f"Invalid state/action shape: {path}")
                if not np.isfinite(state).all() or not np.isfinite(actions).all():
                    raise ValueError(f"Nonfinite state/action: {path}")
                if not np.array_equal(table["frame_index"].to_numpy(), np.arange(len(state))):
                    raise ValueError(f"Noncontiguous frames: {path}")
                if not np.all(table["episode_index"].to_numpy() == index):
                    raise ValueError(f"Wrong episode index: {path}")
                self.episodes.append(
                    Episode(
                        root.name,
                        index,
                        path,
                        source_file,
                        source_sha256,
                        outcome,
                        state,
                        actions,
                        alignment_gap,
                    )
                )
        unmatched = exclusion_keys - matched_exclusions
        if unmatched:
            raise ValueError(f"Episode exclusions did not match complete data: {sorted(unmatched)}")
        self.chunks = []
        for i, episode in enumerate(self.episodes):
            self.chunks.extend(episode_chunks(i, len(episode.state), horizon, episode.gap))

    def manifest(self):
        return {
            "metadata_sha256": self.metadata_hashes,
            "horizon": self.horizon,
            "gamma": self.gamma,
            "excluded_episodes": self.excluded_episodes,
            "episodes": [
                {
                    "dataset": e.dataset,
                    "episode_index": e.index,
                    "source_file": e.source_file,
                    "source_sha256": e.source_sha256,
                    "outcome": e.outcome,
                    "frames": len(e.state),
                    "gap_after_frame": e.gap,
                }
                for e in self.episodes
            ],
            "composition": {
                "episodes": len(self.episodes),
                "outcomes": dict(Counter(e.outcome for e in self.episodes)),
                "chunks": len(self.chunks),
            },
        }

    def sample(self, batch_size, rng):
        return [self.chunks[i] for i in rng.integers(len(self.chunks), size=batch_size)]

    def images(self, episode_id, index):
        episode = self.episodes[episode_id]
        if self.data_config.image_storage == "external_video":
            root, info = self._dataset_info[episode.dataset]
            result = {}
            for camera in self.data_config.camera_keys:
                key = (episode.dataset, episode.index, camera)
                if key not in self._video_cache:
                    from torchcodec.decoders import VideoDecoder

                    video_path = root / info["video_path"].format(
                        episode_chunk=episode.index // info["chunks_size"],
                        video_key=camera,
                        episode_index=episode.index,
                    )
                    decoder = VideoDecoder(str(video_path))
                    if len(decoder) != len(episode.state):
                        raise ValueError(
                            f"Video/data length mismatch for {video_path}: {len(decoder)} != {len(episode.state)}"
                        )
                    self._video_cache[key] = decoder
                decoder = self._video_cache[key]
                self._video_cache.move_to_end(key)
                frame = decoder[index].permute(1, 2, 0).numpy()
                if frame.shape != (224, 224, 3):
                    raise ValueError("Expected pre-resized 224x224 RGB videos")
                result[camera] = frame
            while len(self._video_cache) > 24:
                self._video_cache.popitem(last=False)
            return result
        if episode_id not in self._image_cache:
            table = pq.read_table(
                self.episodes[episode_id].path,
                columns=[f"observation.images.{key}" for key in self.data_config.camera_keys],
            )
            self._image_cache[episode_id] = table
            if len(self._image_cache) > 8:
                self._image_cache.popitem(last=False)
        self._image_cache.move_to_end(episode_id)
        table = self._image_cache[episode_id]
        result = {}
        for camera in self.data_config.camera_keys:
            cell = table[f"observation.images.{camera}"][index].as_py()
            if cell["bytes"] is None:
                raise ValueError("AIRBOT replay expects embedded LeRobot images")
            with Image.open(io.BytesIO(cell["bytes"])) as image:
                result[camera] = np.asarray(image.convert("RGB"))
            if result[camera].shape != (224, 224, 3):
                raise ValueError("Expected pre-resized 224x224 RGB images")
        return result

    def transition(self, chunk):
        episode = self.episodes[chunk.episode]
        actions = episode.actions[chunk.start : chunk.start + chunk.length].copy()
        joint_mask = self.data_config.joint_mask
        actions[:, joint_mask] -= episode.state[chunk.start, joint_mask]
        padded = np.zeros((self.horizon, self.data_config.action_dim), dtype=np.float32)
        padded[: chunk.length] = actions
        mask = np.arange(self.horizon) < chunk.length
        reward = self.gamma ** (chunk.length - 1) if chunk.terminal and episode.outcome == "success" else 0.0
        discount = 0.0 if chunk.terminal else self.gamma**chunk.length
        return padded, mask, reward, discount


class AirbotBatchBuilder:
    def __init__(
        self,
        replay,
        model_config,
        actor_norm_stats,
        critic_tokenizer=None,
        *,
        prompt="pick and place cube",
        critic_norm_stats=None,
    ):
        self.replay = replay
        self.prompt = prompt
        # Warm-started critics retain their original state/action coordinates.
        self.normalize = transforms.Normalize(actor_norm_stats, use_quantiles=True)
        self.critic_tokenizer = critic_tokenizer
        self.model_action_dim = model_config.action_dim
        if critic_tokenizer is not None:
            critic_norm_stats = actor_norm_stats if critic_norm_stats is None else critic_norm_stats
            self.critic_normalize = transforms.Normalize(critic_norm_stats, use_quantiles=True)
            self.action_stats = QAMActionStats.from_norm_stats(
                actor_norm_stats, critic_norm_stats, dimensions=replay.data_config.action_dim
            )
        self.actor_transform = transforms.compose(
            [
                aloha_policy.AlohaInputs(adapt_to_pi=False),
                self.normalize,
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(
                    tokenizer_lib.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )

    def __call__(self, chunks):
        actor_inputs, obs, next_obs, transitions = [], [], [], []
        for chunk in chunks:
            episode = self.replay.episodes[chunk.episode]
            current_images = self.replay.images(chunk.episode, chunk.start)
            actor_inputs.append(
                self.actor_transform(
                    {
                        "images": {
                            name: current_images[key].transpose(2, 0, 1)
                            for name, key in self.replay.data_config.camera_mapping
                        },
                        "state": episode.state[chunk.start].copy(),
                        "prompt": self.prompt,
                    }
                )
            )
            if self.critic_tokenizer is None:
                continue
            for storage, index, images in (
                (obs, chunk.start, current_images),
                (next_obs, chunk.next_index, self.replay.images(chunk.episode, chunk.next_index)),
            ):
                state = self.critic_normalize({"state": episode.state[index]})["state"]
                tokens, mask = self.critic_tokenizer.tokenize(self.prompt, state)
                storage.append(
                    {
                        "images": {k: v.astype(np.float32) / 127.5 - 1 for k, v in images.items()},
                        "image_masks": dict.fromkeys(images, np.True_),
                        "token_ids": tokens,
                        "token_mask": mask,
                    }
                )
            transitions.append(self.replay.transition(chunk))

        def stack(rows):
            return jax.tree.map(lambda *values: np.stack(values), *rows)

        observation = jax.tree.map(jnp.asarray, model_lib.Observation.from_dict(stack(actor_inputs)))
        if self.critic_tokenizer is None:
            from openpi.training.airbot_bc_training import bc_actions

            return observation, bc_actions(self.replay, chunks, self.normalize, self.model_action_dim)
        actions, masks, rewards, discounts = map(np.stack, zip(*transitions, strict=True))
        actions = self.critic_normalize({"actions": actions})["actions"]
        batch = {
            "obs": stack(obs),
            "next_obs": stack(next_obs),
            "actions": np.where(masks[..., None], actions, 0).astype(np.float32),
            "action_mask": masks,
            "rewards": rewards.astype(np.float32),
            "discounts": discounts.astype(np.float32),
        }
        return QAMBatch(
            observation,
            jax.tree.map(jnp.asarray, batch),
        )


def make_batch_builder(replay, model_config, actor_norm_stats, critic_norm_stats, config):
    """Shared serial/worker construction; BC never loads a critic tokenizer."""
    tokenizer = None
    if config["mode"] == "offline":
        from openpi.models.divl_config import DIVLStateEncoderConfig
        from openpi.models.divl_tokenizer import DIVLTokenizer

        tokenizer = DIVLTokenizer(
            DIVLStateEncoderConfig(
                image_keys=replay.data_config.camera_keys,
                state_dim=replay.data_config.action_dim,
                gemma_tokenizer=config["gemma_tokenizer"],
            )
        )
    elif config["mode"] != "bc":
        raise ValueError("mode must be bc or offline")
    return AirbotBatchBuilder(
        replay, model_config, actor_norm_stats, tokenizer, prompt=config["prompt"], critic_norm_stats=critic_norm_stats
    )
