from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REQUIRED_DATA_COLUMNS = {
    "sample_id",
    "source_sample_id",
    "source_episode_id",
    "instruction_id",
    "task.instructions",
    "observation_start_frame",
    "observation_end_frame",
    "action_start_frame",
    "action_end_frame",
    "annotation_version",
    "source_frames_snapshot_id",
    "sync_status",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_export(export_dir: Path, manifest_path: Path, registry_dir: Path | None) -> dict[str, Any]:
    data_path = export_dir / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = export_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_path = export_dir / "meta" / "tasks.parquet"
    info_path = export_dir / "meta" / "info.json"
    stats_path = export_dir / "meta" / "stats.json"
    lineage_path = export_dir / "meta" / "export_lineage.json"
    refs_path = export_dir / "meta" / "sample_refs.jsonl"

    required_files = [
        data_path,
        episodes_path,
        tasks_path,
        info_path,
        stats_path,
        lineage_path,
        refs_path,
        manifest_path,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        return {"status": "failed", "missing_files": missing_files}

    data_table = pq.read_table(data_path)
    episodes_table = pq.read_table(episodes_path)
    tasks_table = pq.read_table(tasks_path)
    info = read_json(info_path)
    stats = read_json(stats_path)
    lineage = read_json(lineage_path)
    sample_refs = read_jsonl(refs_path)

    manifest_rows = count_jsonl(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    data_rows = data_table.num_rows
    episode_rows = episodes_table.num_rows
    task_rows = tasks_table.num_rows
    sample_ref_rows = len(sample_refs)

    data_columns = set(data_table.column_names)
    missing_required_columns = sorted(REQUIRED_DATA_COLUMNS - data_columns)
    data_sample_ids = data_table.column("sample_id").to_pylist() if "sample_id" in data_columns else []
    ref_sample_ids = [str(row.get("sample_id")) for row in sample_refs]

    checks = {
        "manifest_rows_match_data_rows": manifest_rows == data_rows,
        "episode_rows_match_data_rows": episode_rows == data_rows,
        "sample_refs_match_data_rows": sample_ref_rows == data_rows,
        "info_total_frames_match_data_rows": info.get("total_frames") == data_rows,
        "stats_total_samples_match_data_rows": stats.get("export_total_samples") == data_rows,
        "lineage_manifest_rows_match": lineage.get("manifest_row_count") == manifest_rows,
        "lineage_manifest_sha256_match": lineage.get("manifest_sha256") == manifest_sha256,
        "info_manifest_sha256_match": info.get("manifest_sha256") == manifest_sha256,
        "sample_ids_are_unique": len(data_sample_ids) == len(set(data_sample_ids)),
        "sample_refs_are_unique": len(ref_sample_ids) == len(set(ref_sample_ids)),
        "required_columns_present": not missing_required_columns,
    }

    registry_checks: dict[str, bool] = {}
    if registry_dir:
        registry_metadata = registry_dir / "metadata.json"
        registry_stats = registry_dir / "stats.json"
        registry_lineage = registry_dir / "lineage.json"
        registry_checks = {
            "registry_metadata_exists": registry_metadata.exists(),
            "registry_stats_exists": registry_stats.exists(),
            "registry_lineage_exists": registry_lineage.exists(),
        }
        if registry_lineage.exists():
            registry_checks["registry_lineage_sha256_match"] = (
                read_json(registry_lineage).get("manifest_sha256") == manifest_sha256
            )

    status = "passed" if all(checks.values()) and all(registry_checks.values()) else "failed"
    report = {
        "status": status,
        "export_dir": str(export_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "export_version": info.get("export_version"),
        "counts": {
            "manifest_rows": manifest_rows,
            "data_rows": data_rows,
            "episode_rows": episode_rows,
            "task_rows": task_rows,
            "sample_ref_rows": sample_ref_rows,
        },
        "checks": checks,
        "registry_checks": registry_checks,
        "missing_required_columns": missing_required_columns,
        "annotation_type_counts": stats.get("annotation_type_counts", {}),
        "annotation_version_counts": stats.get("annotation_version_counts", {}),
        "missing_video_ref_count": stats.get("missing_video_ref_count"),
    }

    write_json(export_dir / "meta" / "validation_report.json", report)
    if registry_dir:
        registry_dir.mkdir(parents=True, exist_ok=True)
        write_json(registry_dir / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a LeRobot-style manifest export.")
    parser.add_argument(
        "--export-dir",
        default="data/exports/lerobot/l2d_v3_synced_manifest_export",
        help="Export directory to validate.",
    )
    parser.add_argument(
        "--manifest-path",
        default="data/manifests/l2d_v3_synced_sample/manifest.jsonl",
        help="Source manifest JSONL path.",
    )
    parser.add_argument(
        "--registry-dir",
        default="registry/datasets/l2d_v3_synced_sample/exports/lerobot_manifest_export",
        help="Registry export directory to validate.",
    )
    args = parser.parse_args()

    registry_dir = Path(args.registry_dir) if args.registry_dir else None
    report = validate_export(Path(args.export_dir), Path(args.manifest_path), registry_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
