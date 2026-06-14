from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
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
        registry_dir: Path | str | None = None,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.manifest_path = Path(manifest_path)
        self.export_dir = Path(export_dir)
        self.registry_dir = Path(registry_dir) if registry_dir else None
        self.resolver = ManifestResolver(self.snapshot_dir)

    def export(self) -> dict[str, Any]:
        samples = read_manifest_samples(self.manifest_path)
        if not samples:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        manifest_sha256 = self._file_sha256(self.manifest_path)
        export_version = manifest_sha256[:16]
        created_at_utc = datetime.now(timezone.utc).isoformat()

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
        annotation_type_counts: Counter[str] = Counter()
        annotation_version_counts: Counter[str] = Counter()
        source_sample_ids: set[str] = set()
        source_episode_ids: set[str] = set()
        missing_video_paths: set[str] = set()

        for sample_index, sample in enumerate(samples):
            resolved = self.resolver.resolve(sample)
            instruction = sample.instruction_text or ""
            task_index = task_ids.setdefault(instruction, len(task_ids))

            rows.append(self._build_data_row(sample_index, task_index, sample, resolved))
            episodes.append(self._build_episode_row(sample_index, task_index, sample, instruction))
            sample_refs.append(self._build_sample_ref(sample, resolved))

            annotation_type_counts[str(sample.get("annotation_type", "unknown"))] += 1
            annotation_version_counts[str(sample.get("annotation_version", "unknown"))] += 1
            source_sample_ids.add(str(sample.get("source_sample_id", sample.sample_id)))
            source_episode_ids.add(sample.episode_id)
            for video_ref in resolved["video_refs"]:
                if not video_ref.get("exists"):
                    missing_video_paths.add(str(video_ref.get("path")))

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
        info = self._build_info(source_info, rows, len(task_ids), export_version, manifest_sha256)
        stats = self._build_stats(
            source_stats,
            rows,
            task_ids,
            annotation_type_counts,
            annotation_version_counts,
            source_sample_ids,
            source_episode_ids,
            missing_video_paths,
        )
        lineage = self._build_lineage(
            export_version=export_version,
            manifest_sha256=manifest_sha256,
            created_at_utc=created_at_utc,
            samples=samples,
        )

        self._write_json(meta_dir / "info.json", info)
        self._write_json(meta_dir / "stats.json", stats)
        self._write_json(meta_dir / "export_lineage.json", lineage)
        sample_refs_path = meta_dir / "sample_refs.jsonl"
        sample_refs_path.write_text(
            "\n".join(json.dumps(ref, ensure_ascii=False) for ref in sample_refs) + "\n",
            encoding="utf-8",
        )

        if self.registry_dir:
            self.registry_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(self.registry_dir / "metadata.json", info)
            self._write_json(self.registry_dir / "stats.json", stats)
            self._write_json(self.registry_dir / "lineage.json", lineage)

        return {
            "export_dir": str(self.export_dir),
            "export_version": export_version,
            "samples": len(rows),
            "source_samples": len(source_sample_ids),
            "source_episodes": len(source_episode_ids),
            "tasks": len(task_ids),
            "missing_video_refs": len(missing_video_paths),
            "data_path": str(data_dir / "file-000.parquet"),
            "episodes_path": str(episodes_dir / "file-000.parquet"),
            "sample_refs_path": str(sample_refs_path),
            "registry_dir": str(self.registry_dir) if self.registry_dir else None,
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
            "source_sample_id": sample.get("source_sample_id", sample.sample_id),
            "source_episode_id": sample.episode_id,
            "source_instruction_id": sample.get("source_instruction_id", sample.instruction_id),
            "source_instruction_text": sample.get("source_instruction_text"),
            "instruction_id": sample.instruction_id,
            "annotation_type": sample.get("annotation_type"),
            "annotation_version": sample.get("annotation_version"),
            "annotation_policy": sample.get("annotation_policy"),
            "annotation_language": sample.get("annotation_language"),
            "annotation_table": sample.get("annotation_table"),
            "annotation_snapshot_id": sample.get("annotation_snapshot_id"),
            "source_anchor_frame": sample.anchor_frame,
            "observation_start_frame": sample.observation_start_frame,
            "observation_end_frame": sample.observation_end_frame,
            "action_start_frame": sample.action_start_frame,
            "action_end_frame": sample.action_end_frame,
            "observation_rows": sample.get("observation_rows"),
            "action_rows": sample.get("action_rows"),
            "anchor_timestamp": sample.get("anchor_timestamp"),
            "max_timestamp_gap_seconds": sample.get("max_timestamp_gap_seconds"),
            "sync_status": sample.get("sync_status"),
            "sync_rule": sample.get("sync_rule"),
            "source_frames_table": sample.get("source_frames_table"),
            "source_frames_snapshot_id": sample.get("source_frames_snapshot_id"),
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
            "sample_id": sample.sample_id,
            "source_sample_id": sample.get("source_sample_id", sample.sample_id),
            "source_episode_id": sample.episode_id,
            "source_instruction_id": sample.get("source_instruction_id", sample.instruction_id),
            "instruction_id": sample.instruction_id,
            "annotation_type": sample.get("annotation_type"),
            "annotation_version": sample.get("annotation_version"),
            "annotation_policy": sample.get("annotation_policy"),
            "source_anchor_frame": sample.anchor_frame,
            "observation_start_frame": sample.observation_start_frame,
            "observation_end_frame": sample.observation_end_frame,
            "action_start_frame": sample.action_start_frame,
            "action_end_frame": sample.action_end_frame,
        }

    @staticmethod
    def _build_sample_ref(sample: ManifestSample, resolved: dict[str, Any]) -> dict[str, Any]:
        missing_videos = [ref for ref in resolved["video_refs"] if not ref.get("exists")]
        return {
            "sample_id": sample.sample_id,
            "source_sample_id": sample.get("source_sample_id", sample.sample_id),
            "episode_id": sample.episode_id,
            "instruction_id": sample.instruction_id,
            "instruction_text": sample.instruction_text,
            "source_instruction_id": sample.get("source_instruction_id"),
            "source_instruction_text": sample.get("source_instruction_text"),
            "annotation": {
                "type": sample.get("annotation_type"),
                "version": sample.get("annotation_version"),
                "policy": sample.get("annotation_policy"),
                "language": sample.get("annotation_language"),
                "table": sample.get("annotation_table"),
                "snapshot_id": sample.get("annotation_snapshot_id"),
            },
            "sync": {
                "status": sample.get("sync_status"),
                "rule": sample.get("sync_rule"),
                "max_timestamp_gap_seconds": sample.get("max_timestamp_gap_seconds"),
            },
            "source_frames": {
                "table": sample.get("source_frames_table"),
                "snapshot_id": sample.get("source_frames_snapshot_id"),
            },
            "observation_frame_range": resolved["observation"]["frame_range"],
            "action_frame_range": resolved["action"]["frame_range"],
            "video_refs": resolved["video_refs"],
            "missing_video_count": len(missing_videos),
        }

    @staticmethod
    def _build_info(
        source_info: dict[str, Any],
        rows: list[dict[str, Any]],
        total_tasks: int,
        export_version: str,
        manifest_sha256: str,
    ) -> dict[str, Any]:
        source_features = source_info.get("features", {})
        output_features: dict[str, Any] = {}
        for key, value in rows[0].items():
            if key in source_features:
                output_features[key] = source_features[key]
            elif key == "timestamp" or isinstance(value, float):
                output_features[key] = {"dtype": "float32", "shape": [1], "names": None}
            elif key in {"frame_index", "episode_index", "index", "task_index"}:
                output_features[key] = {"dtype": "int64", "shape": [1], "names": None}
            elif key.endswith("_frame") or key.endswith("_rows") or key.endswith("_snapshot_id"):
                output_features[key] = {"dtype": "int64", "shape": [1], "names": None}
            elif isinstance(value, list):
                output_features[key] = {"dtype": "list", "shape": [len(value)], "names": None}
            else:
                output_features[key] = {"dtype": "string", "shape": [1], "names": None}

        return {
            "format": "lerobot_manifest_export_v0",
            "export_version": export_version,
            "manifest_sha256": manifest_sha256,
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
    def _build_stats(
        source_stats: dict[str, Any],
        rows: list[dict[str, Any]],
        task_ids: dict[str, int],
        annotation_type_counts: Counter[str],
        annotation_version_counts: Counter[str],
        source_sample_ids: set[str],
        source_episode_ids: set[str],
        missing_video_paths: set[str],
    ) -> dict[str, Any]:
        return {
            "export_total_samples": len(rows),
            "export_total_tasks": len(task_ids),
            "source_sample_count": len(source_sample_ids),
            "source_episode_count": len(source_episode_ids),
            "annotation_type_counts": dict(sorted(annotation_type_counts.items())),
            "annotation_version_counts": dict(sorted(annotation_version_counts.items())),
            "missing_video_ref_count": len(missing_video_paths),
            "missing_video_paths": sorted(missing_video_paths),
            "source_stats": source_stats,
            "note": "Stats are inherited from source snapshot; recomputation is deferred for the MVP export.",
        }

    def _build_lineage(
        self,
        export_version: str,
        manifest_sha256: str,
        created_at_utc: str,
        samples: list[ManifestSample],
    ) -> dict[str, Any]:
        first = samples[0]
        return {
            "format": "lerobot_manifest_export_lineage_v0",
            "export_version": export_version,
            "created_at_utc": created_at_utc,
            "snapshot_dir": str(self.snapshot_dir),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": manifest_sha256,
            "manifest_row_count": len(samples),
            "source_frames_table": first.get("source_frames_table"),
            "source_frames_snapshot_id": first.get("source_frames_snapshot_id"),
            "annotation_table": first.get("annotation_table"),
            "annotation_snapshot_id": first.get("annotation_snapshot_id"),
            "annotation_version": first.get("annotation_version"),
            "sync_rule": first.get("sync_rule"),
            "export_dir": str(self.export_dir),
            "data_path": "data/chunk-000/file-000.parquet",
            "episodes_path": "meta/episodes/chunk-000/file-000.parquet",
            "tasks_path": "meta/tasks.parquet",
            "sample_refs_path": "meta/sample_refs.jsonl",
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
