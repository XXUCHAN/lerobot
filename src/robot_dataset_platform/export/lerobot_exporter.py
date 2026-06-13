from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from robot_dataset_platform.manifest.resolver import ManifestResolver, ManifestSample, read_manifest_samples


class LeRobotManifestExporter:
    def __init__(
        self,
        snapshot_dir: Path | str,
        manifest_path: Path | str,
        export_dir: Path | str,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.manifest_path = Path(manifest_path)
        self.export_dir = Path(export_dir)
        self.resolver = ManifestResolver(self.snapshot_dir)

    def export(self) -> dict[str, Any]:
        samples = read_manifest_samples(self.manifest_path)
        if not samples:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        if self.export_dir.exists():
            shutil.rmtree(self.export_dir)

        data_dir = self.export_dir / "data" / "chunk-000"
        episodes_dir = self.export_dir / "meta" / "episodes" / "chunk-000"
        meta_dir = self.export_dir / "meta"
        data_dir.mkdir(parents=True, exist_ok=True)
        episodes_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        sample_refs: list[dict[str, Any]] = []
        task_ids: dict[str, int] = {}

        for sample_index, sample in enumerate(samples):
            resolved = self.resolver.resolve(sample)
            instruction = sample.instruction_text or ""
            task_index = task_ids.setdefault(instruction, len(task_ids))
            row = self._build_data_row(sample_index, task_index, sample, resolved)
            rows.append(row)
            episodes.append(self._build_episode_row(sample_index, task_index, sample, instruction))
            sample_refs.append(self._build_sample_ref(sample, resolved))

        pq.write_table(pa.Table.from_pylist(rows), data_dir / "file-000.parquet", compression="snappy")
        pq.write_table(
            pa.Table.from_pylist(episodes),
            episodes_dir / "file-000.parquet",
            compression="snappy",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"task_index": task_index, "task": task}
                    for task, task_index in sorted(task_ids.items(), key=lambda item: item[1])
                ]
            ),
            meta_dir / "tasks.parquet",
            compression="snappy",
        )

        source_info = self._read_json(self.snapshot_dir / "meta" / "info.json")
        source_stats = self._read_json(self.snapshot_dir / "meta" / "stats.json")
        info = self._build_info(source_info, rows, len(task_ids))
        stats = self._build_stats(source_stats, rows)
        (meta_dir / "info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (meta_dir / "stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        sample_refs_path = meta_dir / "sample_refs.jsonl"
        sample_refs_path.write_text(
            "\n".join(json.dumps(ref, ensure_ascii=False) for ref in sample_refs) + "\n",
            encoding="utf-8",
        )

        return {
            "export_dir": str(self.export_dir),
            "samples": len(rows),
            "tasks": len(task_ids),
            "data_path": str(data_dir / "file-000.parquet"),
            "episodes_path": str(episodes_dir / "file-000.parquet"),
            "sample_refs_path": str(sample_refs_path),
        }

    @staticmethod
    def _build_data_row(
        sample_index: int,
        task_index: int,
        sample: ManifestSample,
        resolved: dict[str, Any],
    ) -> dict[str, Any]:
        observation_records = resolved["observation"]["records"]
        action_records = resolved["action"]["records"]
        if not observation_records:
            raise ValueError(f"Observation window is empty: {sample.sample_id}")
        if not action_records:
            raise ValueError(f"Action window is empty: {sample.sample_id}")

        anchor_observation = observation_records[-1]
        next_action = action_records[0]
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "source_episode_id": sample.episode_id,
            "source_instruction_id": sample.instruction_id,
            "source_anchor_frame": sample.anchor_frame,
            "observation_start_frame": sample.observation_start_frame,
            "observation_end_frame": sample.observation_end_frame,
            "action_start_frame": sample.action_start_frame,
            "action_end_frame": sample.action_end_frame,
            "timestamp": anchor_observation.get("timestamp"),
            "frame_index": 0,
            "episode_index": sample_index,
            "index": sample_index,
            "task_index": task_index,
            "task.instructions": sample.instruction_text,
        }

        for key, value in anchor_observation.items():
            if key.startswith("observation."):
                row[key] = value
        for key, value in next_action.items():
            if key.startswith("action."):
                row[key] = value

        return row

    @staticmethod
    def _build_episode_row(
        sample_index: int,
        task_index: int,
        sample: ManifestSample,
        instruction: str,
    ) -> dict[str, Any]:
        return {
            "episode_index": sample_index,
            "tasks": [instruction],
            "length": 1,
            "dataset_from_index": sample_index,
            "dataset_to_index": sample_index + 1,
            "task_index": task_index,
            "source_episode_id": sample.episode_id,
            "source_instruction_id": sample.instruction_id,
            "source_anchor_frame": sample.anchor_frame,
            "observation_start_frame": sample.observation_start_frame,
            "observation_end_frame": sample.observation_end_frame,
            "action_start_frame": sample.action_start_frame,
            "action_end_frame": sample.action_end_frame,
        }

    @staticmethod
    def _build_sample_ref(sample: ManifestSample, resolved: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "episode_id": sample.episode_id,
            "instruction_id": sample.instruction_id,
            "instruction_text": sample.instruction_text,
            "observation_frame_range": resolved["observation"]["frame_range"],
            "action_frame_range": resolved["action"]["frame_range"],
            "video_refs": resolved["video_refs"],
        }

    @staticmethod
    def _build_info(source_info: dict[str, Any], rows: list[dict[str, Any]], total_tasks: int) -> dict[str, Any]:
        source_features = source_info.get("features", {})
        output_features: dict[str, Any] = {}
        for key in rows[0].keys():
            if key in source_features:
                output_features[key] = source_features[key]
            elif key in {"timestamp"}:
                output_features[key] = {"dtype": "float32", "shape": [1], "names": None}
            elif key in {"frame_index", "episode_index", "index", "task_index"}:
                output_features[key] = {"dtype": "int64", "shape": [1], "names": None}
            elif key.endswith("_frame") or key == "source_anchor_frame":
                output_features[key] = {"dtype": "int64", "shape": [1], "names": None}
            else:
                output_features[key] = {"dtype": "string", "shape": [1], "names": None}

        return {
            "codebase_version": source_info.get("codebase_version", "v3.0"),
            "robot_type": source_info.get("robot_type"),
            "total_episodes": len(rows),
            "total_frames": len(rows),
            "total_tasks": total_tasks,
            "fps": source_info.get("fps"),
            "splits": {"train": f"0:{len(rows)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": source_info.get("video_path"),
            "features": output_features,
            "export_note": (
                "Manifest-based tabular export. Large source videos are referenced in "
                "meta/sample_refs.jsonl and are not copied into this export."
            ),
        }

    @staticmethod
    def _build_stats(source_stats: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "export_total_samples": len(rows),
            "source_stats": source_stats,
            "note": "Stats are inherited from source snapshot; recomputation is deferred for the MVP export.",
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
