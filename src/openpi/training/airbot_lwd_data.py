"""Offline AIRBOT replay from LeRobot v2.1, with episode labels and gap boundaries."""

from collections import Counter
from collections import OrderedDict
from collections import defaultdict
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

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
JOINT_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False])


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
    split: str = "train"


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


class AirbotReplay:
    def __init__(
        self,
        datasets,
        *,
        horizon=32,
        gamma=0.9999,
        validation_fraction=0.1,
        seed=42,
        exclude_episodes=(),
    ):
        if not 0 < gamma <= 1 or not 0 < validation_fraction < 1:
            raise ValueError("Invalid discount or validation fraction")
        exclusions = list(exclude_episodes)
        try:
            exclusion_keys = {(str(row["dataset"]), str(row["source_file"])) for row in exclusions}
        except (KeyError, TypeError) as exc:
            raise ValueError("Each exclusion needs dataset and source_file") from exc
        if len(exclusion_keys) != len(exclusions):
            raise ValueError("Duplicate episode exclusion")
        self.horizon, self.gamma = horizon, gamma
        self.episodes = []
        self.excluded_episodes = []
        self._image_cache = OrderedDict()
        self.metadata_hashes = {}
        matched_exclusions = set()
        for directory in datasets:
            root = Path(directory).resolve()
            info = json.loads((root / "meta/info.json").read_text())
            if info["fps"] != 25 or any(
                info["features"][key]["shape"] != [14] for key in ("action", "observation.state")
            ):
                raise ValueError(f"Expected 25 Hz, 14D AIRBOT data: {root}")
            metadata = {}
            for name in ("episodes.jsonl", "conversion_manifest.jsonl"):
                raw = (root / "meta" / name).read_bytes()
                self.metadata_hashes[str(root / "meta" / name)] = hashlib.sha256(raw).hexdigest()
                metadata[name] = [json.loads(line) for line in raw.splitlines() if line.strip()]
            latest = {row["source_file"]: row for row in metadata["conversion_manifest.jsonl"]}
            by_index = {row["episode_index"]: row for row in latest.values()}
            if len(by_index) != info["total_episodes"] or len(latest) != len(by_index):
                raise ValueError(f"Incomplete/ambiguous manifest: {root}")
            for row in metadata["episodes.jsonl"]:
                index = row["episode_index"]
                manifest = by_index[index]
                outcome = manifest["outcome"]
                if manifest["status"] != "complete" or outcome not in ("success", "failure"):
                    raise ValueError(f"Need explicit terminal success/failure: {root.name}/{index}")
                if row.get("outcome", outcome) != outcome:
                    raise ValueError("Episode and conversion labels disagree")
                exclusion_key = (root.name, manifest["source_file"])
                if exclusion_key in exclusion_keys:
                    matched_exclusions.add(exclusion_key)
                    self.excluded_episodes.append(
                        {
                            "dataset": root.name,
                            "episode_index": index,
                            "source_file": manifest["source_file"],
                            "source_sha256": manifest["source_sha256"],
                            "outcome": outcome,
                            "frames": row["length"],
                        }
                    )
                    continue
                path = root / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
                table = pq.read_table(path, columns=["observation.state", "action", "frame_index", "episode_index"])
                state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
                actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
                if state.shape != (row["length"], 14) or actions.shape != state.shape:
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
                        manifest["source_file"],
                        manifest["source_sha256"],
                        outcome,
                        state,
                        actions,
                        manifest.get("alignment_gap_after_step"),
                    )
                )
        unmatched = exclusion_keys - matched_exclusions
        if unmatched:
            raise ValueError(f"Episode exclusions did not match complete data: {sorted(unmatched)}")
        groups = defaultdict(list)
        for i, episode in enumerate(self.episodes):
            groups[(episode.dataset, episode.outcome)].append(i)
        rng = np.random.default_rng(seed)
        for indices in groups.values():
            if len(indices) < 2:
                raise ValueError("Each dataset/outcome group needs at least two episodes for a held-out split")
            count = min(len(indices) - 1, max(1, int(np.ceil(len(indices) * validation_fraction))))
            for i in rng.permutation(indices)[:count]:
                self.episodes[i].split = "val"
        self.chunks = {"train": [], "val": []}
        for i, episode in enumerate(self.episodes):
            self.chunks[episode.split].extend(episode_chunks(i, len(episode.state), horizon, episode.gap))

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
                    "split": e.split,
                    "frames": len(e.state),
                    "gap_after_frame": e.gap,
                }
                for e in self.episodes
            ],
            "composition": {
                split: {
                    "episodes": sum(e.split == split for e in self.episodes),
                    "outcomes": dict(Counter(e.outcome for e in self.episodes if e.split == split)),
                    "chunks": len(self.chunks[split]),
                }
                for split in self.chunks
            },
        }

    def sample(self, split, batch_size, rng):
        chunks = self.chunks[split]
        return [chunks[i] for i in rng.integers(len(chunks), size=batch_size)]

    def images(self, episode_id, index):
        if episode_id not in self._image_cache:
            table = pq.read_table(self.episodes[episode_id].path, columns=[f"observation.images.{k}" for k in CAMERAS])
            self._image_cache[episode_id] = table
            if len(self._image_cache) > 8:
                self._image_cache.popitem(last=False)
        self._image_cache.move_to_end(episode_id)
        table = self._image_cache[episode_id]
        result = {}
        for camera in CAMERAS:
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
        actions[:, JOINT_MASK] -= episode.state[chunk.start, JOINT_MASK]
        padded = np.zeros((self.horizon, 14), dtype=np.float32)
        padded[: chunk.length] = actions
        mask = np.arange(self.horizon) < chunk.length
        reward = self.gamma ** (chunk.length - 1) if chunk.terminal and episode.outcome == "success" else 0.0
        discount = 0.0 if chunk.terminal else self.gamma**chunk.length
        return padded, mask, reward, discount


class AirbotBatchBuilder:
    def __init__(self, replay, model_config, actor_norm_stats, critic_tokenizer, *, prompt="pick and place cube"):
        self.replay = replay
        self.prompt = prompt
        # Both models use the BC checkpoint's immutable state/delta-action stats.
        self.normalize = transforms.Normalize(actor_norm_stats, use_quantiles=True)
        self.critic_tokenizer = critic_tokenizer
        self.action_stats = QAMActionStats.from_norm_stats(actor_norm_stats, actor_norm_stats, dimensions=14)
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
                            for name, key in zip(
                                ("cam_high", "cam_left_wrist", "cam_right_wrist"), CAMERAS, strict=True
                            )
                        },
                        "state": episode.state[chunk.start].copy(),
                        "prompt": self.prompt,
                    }
                )
            )
            for storage, index, images in (
                (obs, chunk.start, current_images),
                (next_obs, chunk.next_index, self.replay.images(chunk.episode, chunk.next_index)),
            ):
                state = self.normalize({"state": episode.state[index]})["state"]
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

        actions, masks, rewards, discounts = map(np.stack, zip(*transitions, strict=True))
        actions = self.normalize({"actions": actions})["actions"]
        batch = {
            "obs": stack(obs),
            "next_obs": stack(next_obs),
            "actions": np.where(masks[..., None], actions, 0).astype(np.float32),
            "action_mask": masks,
            "rewards": rewards.astype(np.float32),
            "discounts": discounts.astype(np.float32),
        }
        return QAMBatch(
            jax.tree.map(jnp.asarray, model_lib.Observation.from_dict(stack(actor_inputs))),
            jax.tree.map(jnp.asarray, batch),
        )
