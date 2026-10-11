"""Convert Robotwin v3 into a piper-only LeRobot v2.1 dataset.

The converter keeps only episodes whose task metadata is prefixed with
[piper]. It rewrites the padded 16-D Robotwin state/action vectors to the
canonical 14-D two-arm layout and cuts the v3 concatenated videos into v2
per-episode MP4 files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
CAMERAS = {
    "observation.images.head": "observation.images.head",
    "observation.images.left_wrist": "observation.images.left_wrist",
    "observation.images.right_wrist": "observation.images.right_wrist",
}
PIPER_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=(Path(__file__).resolve().parents[1] / "training_data" / "Robotwin2.0-lerobot"))
    parser.add_argument("--output", type=Path, default=(Path(__file__).resolve().parents[1] / "training_data" / "Robotwin2.0-lerobot-piper-v2"))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--camera", choices=tuple(CAMERAS), default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_piper_episodes(source: Path, max_episodes: int | None) -> list[dict]:
    records = []
    for path in sorted((source / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        for row in table.to_pylist():
            tasks = row.get("tasks") or []
            if not tasks or not str(tasks[0]).startswith("[piper]"):
                continue
            records.append(row)
            if max_episodes is not None and len(records) >= max_episodes:
                return records
    records.sort(key=lambda row: int(row["episode_index"]))
    return records


def camera_source_path(source: Path, camera_key: str, record: dict) -> Path:
    chunk = int(record[f"videos/{camera_key}/chunk_index"])
    file_index = int(record[f"videos/{camera_key}/file_index"])
    return source / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def camera_output_path(output: Path, camera_key: str, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return output / "videos" / f"chunk-{chunk:03d}" / camera_key / f"episode_{episode_index:06d}.mp4"


def extract_video_segment(source_path: Path, output_path: Path, start: float, end: float) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    input_container = av.open(str(source_path))
    input_stream = input_container.streams.video[0]
    output_container = av.open(str(output_path), mode="w")
    output_stream = None
    frames_written = 0
    try:
        input_container.seek(max(0, int(start / float(input_stream.time_base))), stream=input_stream, backward=True)
        for input_frame in input_container.decode(input_stream):
            timestamp = float(input_frame.pts * input_frame.time_base) if input_frame.pts is not None else None
            if timestamp is None:
                continue
            if timestamp + 0.5 / FPS < start:
                continue
            if timestamp >= end - 0.5 / FPS:
                break
            if output_stream is None:
                output_stream = output_container.add_stream("libx264", rate=FPS)
                output_stream.time_base = Fraction(1, FPS)
                output_stream.width = input_frame.width
                output_stream.height = input_frame.height
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"preset": "veryfast", "crf": "23", "bf": "0"}
                output_stream.codec_context.max_b_frames = 0
            frame = input_frame.reformat(format="yuv420p")
            frame.pts = frames_written
            frame.time_base = Fraction(1, FPS)
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
            frames_written += 1
        if output_stream is None:
            raise RuntimeError(f"No frames found in {source_path} for [{start}, {end})")
        for packet in output_stream.encode():
            output_container.mux(packet)
    finally:
        output_container.close()
        input_container.close()
    if frames_written == 0:
        raise RuntimeError(f"Empty video segment: {source_path} [{start}, {end})")
    return frames_written


def make_features() -> dict:
    flat_names = [
        "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
        "left_joint_5", "left_joint_6", "left_gripper",
        "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
        "right_joint_5", "right_joint_6", "right_gripper",
    ]
    return {
        "observation.state": {"dtype": "float32", "shape": [14], "names": [flat_names]},
        "action": {"dtype": "float32", "shape": [14], "names": [flat_names]},
        "observation.images.head": {
            "dtype": "video", "shape": [3, 240, 320], "names": ["channels", "height", "width"],
        },
        "observation.images.left_wrist": {
            "dtype": "video", "shape": [3, 240, 320], "names": ["channels", "height", "width"],
        },
        "observation.images.right_wrist": {
            "dtype": "video", "shape": [3, 240, 320], "names": ["channels", "height", "width"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def write_metadata(output: Path, records: list[dict], task_indices: dict[str, int]) -> None:
    total_frames = sum(int(row["length"]) for row in records)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "piper",
        "total_episodes": len(records),
        "total_frames": total_frames,
        "total_tasks": len(task_indices),
        "total_videos": len(records) * len(CAMERAS),
        "total_chunks": (len(records) + 999) // 1000,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": FPS,
        "splits": {"train": f"0:{len(records)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": make_features(),
        "robotwin_conversion": {
            "source": "Robotwin2.0-lerobot",
            "source_format": "v3",
            "filter": "task label starts with [piper]",
            "state_action_indices": PIPER_INDICES.tolist(),
            "rgb_timestamps": "original observation timestamps",
        },
    }
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    tasks = [{"task_index": index, "task": task} for task, index in sorted(task_indices.items(), key=lambda x: x[1])]
    (output / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in tasks), encoding="utf-8"
    )
    (output / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "tasks": [row["tasks"][0]], "length": int(row["length"])}) + "\n"
            for index, row in enumerate(records)
        ),
        encoding="utf-8",
    )
    (output / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps({"episode_index": index, "stats": {}}) + "\n" for index in range(len(records))),
        encoding="utf-8",
    )
    (output / "meta" / "stats.json").write_text("{}\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Robotwin2.0 piper subset (LeRobot v2.1)\n\n"
        "Converted from the local Robotwin2.0 LeRobot v3 dataset. Only piper episodes are included.\n",
        encoding="utf-8",
    )


def convert_data(source: Path, output: Path, records: list[dict], task_indices: dict[str, int]) -> None:
    by_file: dict[tuple[int, int], list[tuple[int, dict]]] = defaultdict(list)
    for output_index, record in enumerate(records):
        by_file[(int(record["data/chunk_index"]), int(record["data/file_index"]))].append((output_index, record))
    cumulative = 0
    episode_offsets = {}
    for output_index, record in enumerate(records):
        episode_offsets[output_index] = cumulative
        cumulative += int(record["length"])

    for (chunk, file_index), file_records in sorted(by_file.items()):
        path = source / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
        table = pq.read_table(path)
        first_source_index = int(table["index"][0].as_py())
        for output_index, record in file_records:
            start = int(record["dataset_from_index"]) - first_source_index
            length = int(record["length"])
            source_rows = table.slice(start, length)
            state = np.asarray(source_rows["observation.state"].to_pylist(), dtype=np.float32)[:, PIPER_INDICES]
            action = np.asarray(source_rows["action"].to_pylist(), dtype=np.float32)[:, PIPER_INDICES]
            task_index = task_indices[str(record["tasks"][0])]
            frame_index = np.arange(length, dtype=np.int64)
            data = pa.table({
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
                "timestamp": pa.array(np.arange(length, dtype=np.float32) / FPS),
                "frame_index": pa.array(frame_index),
                "episode_index": pa.array(np.full(length, output_index, dtype=np.int64)),
                "index": pa.array(np.arange(length, dtype=np.int64) + episode_offsets[output_index]),
                "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
            })
            destination = output / "data" / f"chunk-{output_index // 1000:03d}" / f"episode_{output_index:06d}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                pq.write_table(data, destination, compression="zstd")


def convert_videos(source: Path, output: Path, records: list[dict], camera: str | None = None) -> None:
    cameras = (camera,) if camera is not None else tuple(CAMERAS)
    for output_index, record in enumerate(records):
        for camera_key in cameras:
            start = float(record[f"videos/{camera_key}/from_timestamp"])
            end = float(record[f"videos/{camera_key}/to_timestamp"])
            source_path = camera_source_path(source, camera_key, record)
            destination = camera_output_path(output, camera_key, output_index)
            if destination.exists() and destination.stat().st_size > 0:
                continue
            frames = extract_video_segment(source_path, destination, start, end)
            expected = int(record["length"])
            if abs(frames - expected) > 1:
                raise RuntimeError(
                    f"{camera_key} episode {record['episode_index']} has {frames} video frames, expected {expected}"
                )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        marker = args.output / "meta" / "conversion_complete.json"
        if not args.overwrite and marker.exists():
            raise SystemExit(f"{args.output} already converted; use --overwrite")
        if args.overwrite:
            shutil.rmtree(args.output)
    records = read_piper_episodes(args.source, args.max_episodes)
    if not records:
        raise SystemExit("No piper episodes found")
    task_indices = {}
    for row in records:
        task = str(row["tasks"][0])
        task_indices.setdefault(task, len(task_indices))
    args.output.mkdir(parents=True, exist_ok=True)
    write_metadata(args.output, records, task_indices)
    if not args.skip_data:
        convert_data(args.source, args.output, records, task_indices)
    if not args.skip_videos:
        convert_videos(args.source, args.output, records, args.camera)
    marker = {
        "source": str(args.source),
        "source_format": "Robotwin v3",
        "target": str(args.output),
        "target_format": "LeRobot v2.1",
        "episodes": len(records),
        "frames": sum(int(row["length"]) for row in records),
    }
    if args.max_episodes is None:
        (args.output / "meta" / "conversion_complete.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8"
        )
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()



