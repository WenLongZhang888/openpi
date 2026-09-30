"""Compute normalization statistics for an OpenPI config or AIRBOT BC YAML.

OpenPI configs are selected with ``--config-name``. AIRBOT's unified trainer uses
``--config`` and writes the statistics to that YAML's ``norm_stats_dir``.
"""

from pathlib import Path

import numpy as np
import tqdm
import tyro
import yaml

import openpi.models.model as _model
import openpi.shared.normalize as normalize
from openpi.training.airbot_lwd_data import AirbotDataConfig
from openpi.training.airbot_lwd_data import AirbotReplay
from openpi.training.airbot_lwd_data import Chunk
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def compute_openpi_stats(config_name: str, max_frames: int | None = None) -> Path:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    return Path(output_path)


def compute_airbot_stats(config_path: Path, max_frames: int | None = None) -> Path:
    """Compute stats over the exact state/action representation used by AIRBOT BC."""
    config = yaml.safe_load(config_path.read_text())
    if config.get("mode") != "bc":
        raise ValueError("AIRBOT normalization statistics must be computed from a mode: bc YAML")
    if "norm_stats_dir" not in config:
        raise ValueError("AIRBOT BC YAML must define norm_stats_dir")
    if max_frames is not None and max_frames < 2:
        raise ValueError("max_frames must be at least 2")

    data_config = AirbotDataConfig.from_dict(config["airbot"])
    replay = AirbotReplay(
        config["datasets"],
        horizon=config["horizon"],
        gamma=config.get("gamma", 0.9999),
        exclude_episodes=config.get("exclude_episodes", ()),
        data_config=data_config,
    )
    episode_ids = [i for i, episode in enumerate(replay.episodes) if episode.outcome == "success"]
    if not episode_ids:
        raise ValueError("AIRBOT BC normalization requires explicitly labeled successful episodes")
    if any(replay.episodes[i].gap is not None for i in episode_ids):
        raise ValueError("AIRBOT BC normalization does not support successful recordings with gaps")

    ends = np.cumsum([len(replay.episodes[i].state) for i in episode_ids])
    total_frames = int(ends[-1])
    frame_count = total_frames if max_frames is None else min(total_frames, max_frames)
    if frame_count < 2:
        raise ValueError("AIRBOT BC normalization requires at least two successful frames")
    if frame_count == total_frames:
        positions = np.arange(total_frames)
    else:
        positions = np.random.default_rng(config["seed"]).choice(total_frames, size=frame_count, replace=False)

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    batch_size = 256
    for offset in tqdm.tqdm(range(0, frame_count, batch_size), desc="Computing AIRBOT stats"):
        batch_positions = positions[offset : offset + batch_size]
        rows = np.searchsorted(ends, batch_positions, side="right")
        previous_ends = np.where(rows > 0, ends[np.maximum(rows - 1, 0)], 0)
        starts = batch_positions - previous_ends
        states = []
        actions = []
        for row, start_value in zip(rows, starts, strict=True):
            episode_id = episode_ids[int(row)]
            episode = replay.episodes[episode_id]
            start = int(start_value)
            length = min(replay.horizon, len(episode.state) - start)
            chunk = Chunk(
                episode=episode_id,
                start=start,
                length=length,
                next_index=min(start + length, len(episode.state) - 1),
                terminal=start + length == len(episode.state),
            )
            physical_actions, _, _, _ = replay.transition(chunk)
            physical_actions[length:] = physical_actions[length - 1]
            states.append(episode.state[start])
            actions.append(physical_actions)
        stats["state"].update(np.stack(states))
        stats["actions"].update(np.stack(actions))

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}
    output_path = Path(config["norm_stats_dir"]).resolve()
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    return output_path


def main(
    config_name: str | None = None,
    config: Path | None = None,
    max_frames: int | None = None,
):
    """Compute stats using exactly one of --config-name or --config."""
    if (config_name is None) == (config is None):
        raise ValueError("Specify exactly one of --config-name or --config")
    if config is not None:
        compute_airbot_stats(config.resolve(), max_frames)
    else:
        compute_openpi_stats(config_name, max_frames)


if __name__ == "__main__":
    tyro.cli(main)
