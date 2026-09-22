"""Convert synchronized AIRBOT MCAP episodes to OpenPI-compatible LeRobot data.

The converter is adapted from FastWAM-Airbot's MCAP converter. Unlike the
original script, image resizing uses OpenPI's ``resize_with_pad`` implementation
so the stored frames exactly match the 224 x 224 model input preprocessing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any

import av
import flatbuffers
import flatbuffers.number_types
import flatbuffers.table
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
import numpy as np
from openpi_client import image_tools
import yaml


@dataclasses.dataclass(frozen=True, slots=True)
class ConversionConfig:
    input_dir: Path
    output_dir: Path
    repo_id: str
    source_task: str
    task: str
    robot_type: str
    fps: int
    image_height: int
    image_width: int
    resize_images: bool
    state_topics: tuple[str, ...]
    action_topics: tuple[str, ...]
    camera_topics: dict[str, str]
    intervention_topic: str
    recursive: bool = True
    require_outcome: bool = False
    min_frames: int = 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConversionConfig:
        config_path = Path(path).resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("conversion config must be a YAML object")

        base_dir = config_path.parents[2] if len(config_path.parents) >= 3 else config_path.parent

        def resolve_path(value: Any) -> Path:
            result = Path(str(value)).expanduser()
            return result if result.is_absolute() else (base_dir / result).resolve()

        images = payload.get("images", {})
        topics = payload.get("topics", {})
        resize_mode = str(images.get("resize", "openpi")).lower()
        if resize_mode not in {"openpi", "none"}:
            raise ValueError("images.resize must be 'openpi' or 'none'")

        config = cls(
            input_dir=resolve_path(payload["input_dir"]),
            output_dir=resolve_path(payload["output_dir"]),
            repo_id=str(payload["repo_id"]),
            source_task=str(payload["source_task"]),
            task=str(payload["task"]),
            robot_type=str(payload.get("robot_type", "airbot")),
            fps=int(payload.get("fps", 25)),
            image_height=int(images.get("height", 224)),
            image_width=int(images.get("width", 224)),
            resize_images=resize_mode == "openpi",
            state_topics=tuple(str(topic) for topic in topics["state"]),
            action_topics=tuple(str(topic) for topic in topics["action"]),
            camera_topics={str(name): str(topic) for name, topic in topics["cameras"].items()},
            intervention_topic=str(topics.get("intervention", "/dagger/intervention")),
            recursive=bool(payload.get("recursive", True)),
            require_outcome=bool(payload.get("require_outcome", False)),
            min_frames=int(payload.get("min_frames", 2)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.input_dir.is_dir():
            raise ValueError(f"input_dir is not a directory: {self.input_dir}")
        if not self.repo_id.strip() or not self.task.strip() or not self.source_task.strip():
            raise ValueError("repo_id, task and source_task must not be empty")
        if self.fps <= 0 or self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("fps and image dimensions must be positive")
        if self.min_frames <= 0:
            raise ValueError("min_frames must be positive")
        if not self.state_topics or not self.action_topics or not self.camera_topics:
            raise ValueError("state, action and camera topics must not be empty")
        required = [*self.state_topics, *self.action_topics, self.intervention_topic]
        if len(set(required)) != len(required):
            raise ValueError("state, action and intervention topics must be unique")
        if len(set(self.camera_topics.values())) != len(self.camera_topics):
            raise ValueError("camera attachment topics must be unique")


@dataclasses.dataclass(frozen=True, slots=True)
class VideoAttachment:
    name: str
    data: bytes
    frames: int
    width: int
    height: int
    codec: str


@dataclasses.dataclass(frozen=True, slots=True)
class McapEpisode:
    path: Path
    state: np.ndarray
    action: np.ndarray
    intervention: np.ndarray
    videos: dict[str, VideoAttachment]
    source_task: str
    outcome: str
    source_log_timestamps_ns: np.ndarray
    alignment_gap_after_step: int | None
    alignment_gap_ms: float | None

    @property
    def num_frames(self) -> int:
        return int(self.state.shape[0])


class ConversionError(RuntimeError):
    pass


def _read_episode_outcome(path: Path, *, required: bool) -> str:
    """Read sidecar and embedded labels, rejecting contradictory annotations."""
    metadata_path = path.with_suffix(".episode.json")
    labels = []
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(f"invalid episode outcome metadata: {metadata_path.name}") from exc
        labels.append((metadata_path.name, payload))
    with path.open("rb") as file:
        for attachment in make_reader(file).iter_attachments():
            if attachment.name != "component_info":
                continue
            try:
                component = json.loads(attachment.data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ConversionError(f"invalid component_info JSON: {path.name}") from exc
            if not isinstance(component, dict):
                raise ConversionError(f"invalid component_info object: {path.name}")
            if "episode_info" in component:
                labels.append((f"{path.name}:component_info/episode_info", component["episode_info"]))
    if not labels:
        if required:
            raise ConversionError(f"missing sidecar and embedded episode outcome metadata: {path.name}")
        return "unknown"
    outcomes = set()
    for source, payload in labels:
        outcome = payload.get("outcome") if isinstance(payload, dict) else None
        if outcome not in ("success", "failure", "timeout"):
            raise ConversionError(f"invalid episode outcome in {source}: {outcome!r}")
        if "success" in payload and payload["success"] != int(outcome == "success"):
            raise ConversionError(f"inconsistent success/outcome in {source}")
        outcomes.add(outcome)
    if len(outcomes) != 1:
        raise ConversionError(f"conflicting sidecar/embedded episode outcomes: {path.name}")
    return outcomes.pop()


def decode_float_array(data: bytes) -> np.ndarray:
    root_offset = flatbuffers.packer.uoffset.unpack_from(data, 0)[0]
    table = flatbuffers.table.Table(bytearray(data), root_offset)
    vector_offset = flatbuffers.number_types.UOffsetTFlags.py_type(table.Offset(4))
    if vector_offset == 0:
        return np.empty((0,), dtype=np.float32)
    values = table.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, vector_offset)
    return np.asarray(values, dtype=np.float32).copy()


def natural_key(value: str | Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def discover_mcap_files(config: ConversionConfig) -> list[Path]:
    files = config.input_dir.rglob("*.mcap") if config.recursive else config.input_dir.glob("*.mcap")
    return sorted(files, key=lambda path: natural_key(path.relative_to(config.input_dir)))


def source_id(path: Path, config: ConversionConfig) -> str:
    return path.relative_to(config.input_dir).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_metadata_string(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _inspect_video(name: str, data: bytes) -> VideoAttachment:
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.video:
            raise ConversionError(f"attachment {name!r} contains no video stream")
        stream = container.streams.video[0]
        decoded_frames = sum(1 for _ in container.decode(stream))
        declared_frames = int(stream.frames or 0)
        if declared_frames and declared_frames != decoded_frames:
            raise ConversionError(f"attachment {name!r} declares {declared_frames} frames but decodes {decoded_frames}")
        return VideoAttachment(
            name=name,
            data=data,
            frames=decoded_frames,
            width=int(stream.width),
            height=int(stream.height),
            codec=str(stream.codec_context.name),
        )


def _stack_topic_values(values: dict[str, list[np.ndarray]], topics: tuple[str, ...], kind: str) -> np.ndarray:
    missing = [topic for topic in topics if not values[topic]]
    if missing:
        raise ConversionError(f"missing {kind} topics: {missing}")
    counts = {topic: len(values[topic]) for topic in topics}
    if len(set(counts.values())) != 1:
        raise ConversionError(f"{kind} topic counts differ: {counts}")
    dimensions = {topic: sorted({len(item) for item in values[topic]}) for topic in topics}
    bad_dimensions = {topic: dims for topic, dims in dimensions.items() if len(dims) != 1 or dims == [0]}
    if bad_dimensions:
        raise ConversionError(f"inconsistent {kind} dimensions: {bad_dimensions}")
    row_count = next(iter(counts.values()))
    rows = [np.concatenate([values[topic][index] for topic in topics]) for index in range(row_count)]
    array = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ConversionError(f"{kind} contains NaN or Inf")
    return array


def _find_alignment_gap(log_timestamps_ns: np.ndarray, intervention: np.ndarray) -> tuple[int | None, float | None]:
    if len(log_timestamps_ns) < 2:
        return None, None
    transitions = np.flatnonzero(np.diff(intervention.astype(np.int64)) == 1)
    if len(transitions) != 1:
        return None, None
    index = int(transitions[0])
    gap_ms = float((log_timestamps_ns[index + 1] - log_timestamps_ns[index]) / 1_000_000)
    if gap_ms < 200:
        return None, None
    return index, gap_ms


def read_mcap_episode(path: str | Path, config: ConversionConfig) -> McapEpisode:
    mcap_path = Path(path)
    outcome = _read_episode_outcome(mcap_path, required=config.require_outcome)
    required_message_topics = (*config.state_topics, *config.action_topics, config.intervention_topic)
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    log_timestamps: dict[str, list[int]] = defaultdict(list)
    videos_by_name: dict[str, VideoAttachment] = {}
    source_task = ""

    try:
        with mcap_path.open("rb") as file:
            reader = make_reader(file)
            summary = reader.get_summary()
            if summary is None:
                raise ConversionError("MCAP has no summary")
            available_topics = {channel.topic for channel in summary.channels.values()}
            missing_topics = sorted(set(required_message_topics) - available_topics)
            if missing_topics:
                raise ConversionError(f"missing required topics: {missing_topics}")

            for metadata in reader.iter_metadata():
                if metadata.name == "task_info":
                    source_task = _clean_metadata_string(metadata.metadata.get("task_name", ""))

            attachment_data: dict[str, bytes] = {}
            for attachment in reader.iter_attachments():
                if attachment.media_type == "video/mp4" and attachment.name in config.camera_topics.values():
                    if attachment.name in attachment_data:
                        raise ConversionError(f"duplicate video attachment: {attachment.name}")
                    attachment_data[attachment.name] = bytes(attachment.data)

            missing_cameras = [name for name, topic in config.camera_topics.items() if topic not in attachment_data]
            if missing_cameras:
                raise ConversionError(f"missing camera attachments: {missing_cameras}")
            for camera_name, topic in config.camera_topics.items():
                videos_by_name[camera_name] = _inspect_video(topic, attachment_data[topic])

            for schema, channel, message in reader.iter_messages(topics=list(required_message_topics)):
                if schema is None or schema.name != "airbot_fbs.FloatArray":
                    schema_name = None if schema is None else schema.name
                    raise ConversionError(f"unsupported schema {schema_name!r} on topic {channel.topic!r}")
                values[channel.topic].append(decode_float_array(message.data))
                log_timestamps[channel.topic].append(int(message.log_time))
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"failed to read {mcap_path}: {type(exc).__name__}: {exc}") from exc

    if source_task != config.source_task:
        raise ConversionError(f"source task mismatch: expected {config.source_task!r}, found {source_task!r}")

    state = _stack_topic_values(values, config.state_topics, "state")
    action = _stack_topic_values(values, config.action_topics, "action")
    intervention_values = values[config.intervention_topic]
    if not intervention_values or any(len(item) != 1 for item in intervention_values):
        raise ConversionError("intervention must contain one value per step")
    intervention = np.rint(np.asarray([item[0] for item in intervention_values])).astype(np.int64)
    if not np.isin(intervention, [0, 1]).all():
        raise ConversionError("intervention contains values other than 0 or 1")

    counts = {
        "state": len(state),
        "action": len(action),
        "intervention": len(intervention),
        **{f"camera:{name}": video.frames for name, video in videos_by_name.items()},
    }
    if len(set(counts.values())) != 1:
        raise ConversionError(f"step counts differ: {counts}")
    if len(state) < config.min_frames:
        raise ConversionError(f"episode has only {len(state)} frames; minimum is {config.min_frames}")

    reference_topic = config.action_topics[0]
    source_log_timestamps_ns = np.asarray(log_timestamps[reference_topic], dtype=np.int64)
    if len(source_log_timestamps_ns) != len(state):
        raise ConversionError("reference timestamp count does not match episode frames")
    if np.any(np.diff(source_log_timestamps_ns) <= 0):
        raise ConversionError("source log timestamps are not strictly increasing")
    alignment_gap_after_step, alignment_gap_ms = _find_alignment_gap(source_log_timestamps_ns, intervention)
    return McapEpisode(
        path=mcap_path,
        state=state,
        action=action,
        intervention=intervention,
        videos=videos_by_name,
        source_task=source_task,
        outcome=outcome,
        source_log_timestamps_ns=source_log_timestamps_ns,
        alignment_gap_after_step=alignment_gap_after_step,
        alignment_gap_ms=alignment_gap_ms,
    )


def iter_video_rgb(video: VideoAttachment, config: ConversionConfig) -> Iterator[np.ndarray]:
    with av.open(io.BytesIO(video.data)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            if config.resize_images:
                rgb = image_tools.resize_with_pad(rgb, config.image_height, config.image_width)
            yield np.asarray(rgb, dtype=np.uint8)


def _image_key(camera_name: str) -> str:
    return f"observation.images.{camera_name}"


def _image_shapes(config: ConversionConfig, episode: McapEpisode) -> dict[str, tuple[int, int, int]]:
    if config.resize_images:
        shape = (config.image_height, config.image_width, 3)
        return dict.fromkeys(config.camera_topics, shape)
    return {name: (video.height, video.width, 3) for name, video in episode.videos.items()}


def _features(
    config: ConversionConfig,
    state_dim: int,
    action_dim: int,
    image_shapes: dict[str, tuple[int, int, int]],
) -> dict[str, dict[str, Any]]:
    features = {
        _image_key(camera_name): {
            "dtype": "image",
            "shape": image_shapes[camera_name],
            "names": ["height", "width", "channel"],
        }
        for camera_name in config.camera_topics
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["state"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["action"],
    }
    features["intervention"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["intervention"],
    }
    return features


def _create_dataset(
    config: ConversionConfig,
    state_dim: int,
    action_dim: int,
    image_shapes: dict[str, tuple[int, int, int]],
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=config.repo_id,
        root=config.output_dir,
        robot_type=config.robot_type,
        fps=config.fps,
        features=_features(config, state_dim, action_dim, image_shapes),
        use_videos=False,
    )


def _open_dataset_for_append(config: ConversionConfig) -> LeRobotDataset:
    dataset = LeRobotDataset(repo_id=config.repo_id, root=config.output_dir)
    dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset


def _normalized_schema(features: dict[str, dict[str, Any]]) -> dict[str, tuple[str, tuple[int, ...]]]:
    internal = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return {key: (str(value["dtype"]), tuple(value["shape"])) for key, value in features.items() if key not in internal}


def _verify_dataset_schema(
    dataset: LeRobotDataset,
    config: ConversionConfig,
    state_dim: int,
    action_dim: int,
    image_shapes: dict[str, tuple[int, int, int]],
) -> None:
    expected = _normalized_schema(_features(config, state_dim, action_dim, image_shapes))
    found = _normalized_schema(dataset.meta.features)
    if expected != found:
        raise ConversionError(f"existing LeRobot schema differs: expected {expected}, found {found}")
    if int(dataset.fps) != config.fps:
        raise ConversionError(f"existing dataset fps is {dataset.fps}, expected {config.fps}")


def _remove_current_episode_images(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        return
    image_dir = dataset.root / "images"
    if image_dir.is_dir():
        shutil.rmtree(image_dir)


def write_episode(dataset: LeRobotDataset, episode: McapEpisode, config: ConversionConfig) -> int:
    camera_iterators = {name: iter_video_rgb(video, config) for name, video in episode.videos.items()}
    episode_index = int(dataset.meta.total_episodes)
    try:
        for frame_index in range(episode.num_frames):
            images: dict[str, np.ndarray] = {}
            for camera_name, iterator in camera_iterators.items():
                try:
                    images[camera_name] = next(iterator)
                except StopIteration as exc:
                    raise ConversionError(f"camera {camera_name!r} ended before frame {frame_index}") from exc

            frame: dict[str, Any] = {
                "observation.state": episode.state[frame_index],
                "action": episode.action[frame_index],
                "intervention": np.asarray([episode.intervention[frame_index]], dtype=np.int64),
                "task": config.task,
            }
            frame.update({_image_key(name): image for name, image in images.items()})
            dataset.add_frame(frame)

        for camera_name, iterator in camera_iterators.items():
            try:
                next(iterator)
            except StopIteration:
                continue
            raise ConversionError(f"camera {camera_name!r} contains extra frames")
        dataset.save_episode()
    except Exception:
        _remove_current_episode_images(dataset)
        dataset.episode_buffer = dataset.create_episode_buffer()
        raise
    return episode_index


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"invalid manifest line {line_number}: {exc}") from exc
        rows[str(row["source_file"])] = row
    return rows


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()


def _validate_episode_shape(
    episode: McapEpisode,
    state_dim: int,
    action_dim: int,
    image_shapes: dict[str, tuple[int, int, int]],
    config: ConversionConfig,
) -> None:
    if episode.state.shape[1] != state_dim or episode.action.shape[1] != action_dim:
        raise ConversionError(
            f"{episode.path}: dimensions differ from first episode: "
            f"state={episode.state.shape[1]}, action={episode.action.shape[1]}"
        )
    found_shapes = _image_shapes(config, episode)
    if found_shapes != image_shapes:
        raise ConversionError(f"{episode.path}: image shapes differ: expected {image_shapes}, found {found_shapes}")


def verify_output(
    config: ConversionConfig,
    expected_episodes: int,
    image_shapes: dict[str, tuple[int, int, int]],
) -> None:
    dataset = LeRobotDataset(repo_id=config.repo_id, root=config.output_dir)
    if dataset.num_episodes != expected_episodes:
        raise ConversionError(f"reload found {dataset.num_episodes} episodes, expected {expected_episodes}")
    if dataset.num_frames <= 0:
        raise ConversionError("reload verification found no frames")
    for sample_index in sorted({0, dataset.num_frames - 1}):
        sample = dataset[sample_index]
        if tuple(sample["observation.state"].shape) != tuple(dataset.features["observation.state"]["shape"]):
            raise ConversionError(f"state shape mismatch at sample {sample_index}")
        if tuple(sample["action"].shape) != tuple(dataset.features["action"]["shape"]):
            raise ConversionError(f"action shape mismatch at sample {sample_index}")
        for camera_name, expected_shape in image_shapes.items():
            image = sample[_image_key(camera_name)]
            if tuple(image.shape[-2:]) != expected_shape[:2]:
                raise ConversionError(f"camera {camera_name} shape mismatch at sample {sample_index}")


def convert_dataset(
    config: ConversionConfig,
    *,
    resume: bool = False,
    force: bool = False,
    limit: int | None = None,
) -> Path:
    if resume and force:
        raise ValueError("resume and force cannot be used together")
    files = discover_mcap_files(config)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        files = files[:limit]
    if not files:
        raise ConversionError(f"no MCAP files found in {config.input_dir}")

    if config.output_dir.exists():
        if force:
            shutil.rmtree(config.output_dir)
        elif not resume:
            raise ConversionError(f"output already exists: {config.output_dir}; use --resume or --force")

    first_episode = read_mcap_episode(files[0], config)
    state_dim = int(first_episode.state.shape[1])
    action_dim = int(first_episode.action.shape[1])
    image_shapes = _image_shapes(config, first_episode)
    if config.output_dir.exists():
        dataset = _open_dataset_for_append(config)
        _verify_dataset_schema(dataset, config, state_dim, action_dim, image_shapes)
    else:
        config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset = _create_dataset(config, state_dim, action_dim, image_shapes)

    manifest_path = config.output_dir / "meta" / "conversion_manifest.jsonl"
    manifest = _load_manifest(manifest_path)
    completed_count = sum(row.get("status") == "complete" for row in manifest.values())
    pending_rows = [row for row in manifest.values() if row.get("status") == "pending"]
    if pending_rows and dataset.meta.total_episodes == completed_count + 1:
        pending = max(pending_rows, key=lambda row: int(row["episode_index"]))
        pending["status"] = "complete"
        _append_manifest(manifest_path, pending)
        manifest[str(pending["source_file"])] = pending
        completed_count += 1
    if dataset.meta.total_episodes != completed_count:
        raise ConversionError(
            "dataset episode count does not match completed conversion manifest; "
            "use a clean output or restore the manifest"
        )

    resize_label = f"OpenPI {config.image_height}x{config.image_width}" if config.resize_images else "native"
    print(
        f"Converting {len(files)} MCAP files to {config.output_dir} at {config.fps} FPS "
        f"({state_dim}D state, {action_dim}D action, images={resize_label})"
    )

    for file_index, path in enumerate(files, start=1):
        source_file = source_id(path, config)
        old_row = manifest.get(source_file)
        if old_row is not None and old_row.get("status") == "complete":
            if old_row.get("source_sha256") != sha256_file(path):
                raise ConversionError(f"{source_file}: source content changed since previous conversion")
            print(f"[{file_index}/{len(files)}] skip completed {source_file}")
            continue

        started = time.perf_counter()
        episode = first_episode if path == files[0] else read_mcap_episode(path, config)
        _validate_episode_shape(episode, state_dim, action_dim, image_shapes, config)
        source_hash = sha256_file(path)
        if old_row is not None and old_row.get("source_sha256") != source_hash:
            raise ConversionError(f"{source_file}: source content changed since previous conversion")

        row = {
            "episode_index": int(dataset.meta.total_episodes),
            "source_file": source_file,
            "source_sha256": source_hash,
            "frames": episode.num_frames,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "fps": config.fps,
            "image_shapes": {name: list(shape) for name, shape in image_shapes.items()},
            "image_resize": "openpi_resize_with_pad" if config.resize_images else "none",
            "source_task": episode.source_task,
            "task": config.task,
            "outcome": episode.outcome,
            "alignment_gap_after_step": episode.alignment_gap_after_step,
            "alignment_gap_ms": episode.alignment_gap_ms,
            "status": "pending",
        }
        _append_manifest(manifest_path, row)
        episode_index = write_episode(dataset, episode, config)
        if episode_index != row["episode_index"]:
            raise ConversionError(f"unexpected episode index {episode_index}, expected {row['episode_index']}")
        row["status"] = "complete"
        _append_manifest(manifest_path, row)
        manifest[source_file] = row

        elapsed = time.perf_counter() - started
        gap = f", removed alignment gap {episode.alignment_gap_ms:.1f} ms" if episode.alignment_gap_ms else ""
        print(
            f"[{file_index}/{len(files)}] wrote episode {episode_index}: "
            f"{source_file}, {episode.num_frames} frames{gap}, outcome={episode.outcome}, {elapsed:.1f}s"
        )

    verify_output(config, int(dataset.meta.total_episodes), image_shapes)
    print(
        f"Done: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames, "
        f"verified at {config.output_dir}"
    )
    return config.output_dir


def inspect_dataset(config: ConversionConfig, limit: int | None = None) -> None:
    files = discover_mcap_files(config)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ConversionError(f"no MCAP files found in {config.input_dir}")
    for index, path in enumerate(files, start=1):
        episode = read_mcap_episode(path, config)
        video_shapes = {name: (video.height, video.width) for name, video in episode.videos.items()}
        print(
            f"[{index}/{len(files)}] {source_id(path, config)}: {episode.num_frames} frames, "
            f"state={episode.state.shape[1]}, action={episode.action.shape[1]}, "
            f"images={video_shapes}, intervention={int(episode.intervention.sum())}, outcome={episode.outcome}"
        )
    print(f"Validated {len(files)} MCAP files")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert synchronized AIRBOT MCAP episodes to LeRobot v2.1.")
    parser.add_argument("--config", type=Path, required=True, help="YAML conversion config")
    parser.add_argument("--resume", action="store_true", help="continue an existing output dataset")
    parser.add_argument("--force", action="store_true", help="delete and recreate the output dataset")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N source files")
    parser.add_argument("--inspect-only", action="store_true", help="validate MCAP inputs without writing LeRobot")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ConversionConfig.from_yaml(args.config)
    if args.inspect_only:
        inspect_dataset(config, args.limit)
        return
    convert_dataset(config, resume=args.resume, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
