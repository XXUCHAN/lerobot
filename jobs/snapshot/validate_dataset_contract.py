from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


CONTRACT_NAME = "robot_dataset_contract_v1"
DEFAULT_SNAPSHOT_DIR = Path("data/external/RobotisSW_omy_PickAndPlace_RedBlock2")
DEFAULT_OUTPUT = Path("registry/datasets/robotis_omy_pick_place_redblock/contract_report.json")
REQUIRED_FRAME_COLUMNS = {"episode_index", "frame_index", "timestamp", "task_index"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    ok: bool,
    details: dict[str, Any] | None = None,
    level: str = "error",
) -> None:
    checks.append(
        {
            "name": name,
            "status": "ok" if ok else level,
            "details": details or {},
        }
    )


def parquet_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.parquet")) if path.exists() else []


def load_frame_table(snapshot_dir: Path) -> pa.Table:
    files = parquet_files(snapshot_dir / "data")
    if not files:
        return pa.table({})
    return ds.dataset([str(file) for file in files], format="parquet").to_table()


def load_episode_rows(snapshot_dir: Path) -> list[dict[str, Any]]:
    episodes_jsonl = snapshot_dir / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        return read_jsonl(episodes_jsonl)

    files = parquet_files(snapshot_dir / "meta" / "episodes")
    if files:
        return pq.read_table(files).to_pylist()
    return []


def load_task_rows(snapshot_dir: Path) -> list[dict[str, Any]]:
    tasks_jsonl = snapshot_dir / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        return read_jsonl(tasks_jsonl)

    tasks_parquet = snapshot_dir / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        return pq.read_table(tasks_parquet).to_pylist()
    return []


def feature_keys_by_dtype(info: dict[str, Any], dtype: str) -> list[str]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    return sorted(
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == dtype
    )


def observation_feature_keys(info: dict[str, Any], frame_columns: set[str]) -> list[str]:
    features = info.get("features", {})
    info_keys = [
        key
        for key, feature in features.items()
        if key.startswith("observation.") and isinstance(feature, dict) and feature.get("dtype") != "video"
    ] if isinstance(features, dict) else []
    frame_keys = [key for key in frame_columns if key.startswith("observation.")]
    return sorted(set(info_keys + frame_keys))


def validate_monotonic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_index"])].append(row)

    timestamp_errors: list[int] = []
    frame_errors: list[int] = []
    duplicate_frame_errors: list[int] = []
    frame_gap_episodes: list[int] = []

    for episode_id, episode_rows in by_episode.items():
        ordered = sorted(episode_rows, key=lambda item: int(item["frame_index"]))
        frames = [int(item["frame_index"]) for item in ordered]
        timestamps = [float(item["timestamp"]) for item in ordered]

        if any(current < previous for previous, current in zip(frames, frames[1:])):
            frame_errors.append(episode_id)
        if len(set(frames)) != len(frames):
            duplicate_frame_errors.append(episode_id)
        if frames and frames != list(range(frames[0], frames[0] + len(frames))):
            frame_gap_episodes.append(episode_id)
        if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
            timestamp_errors.append(episode_id)

    return {
        "episodes": len(by_episode),
        "timestamp_non_monotonic_episodes": timestamp_errors[:20],
        "frame_non_monotonic_episodes": frame_errors[:20],
        "duplicate_frame_episodes": duplicate_frame_errors[:20],
        "frame_gap_episodes": frame_gap_episodes[:20],
        "timestamp_monotonic": not timestamp_errors,
        "frame_index_monotonic": not frame_errors,
        "frame_index_unique": not duplicate_frame_errors,
        "frame_index_contiguous": not frame_gap_episodes,
    }


def validate_episode_lengths(frame_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame_counts = Counter(int(row["episode_index"]) for row in frame_rows)
    mismatches: list[dict[str, Any]] = []
    for row in episode_rows:
        if "episode_index" not in row or "length" not in row:
            continue
        episode_id = int(row["episode_index"])
        expected = int(row["length"])
        actual = int(frame_counts.get(episode_id, 0))
        if expected != actual:
            mismatches.append({"episode_index": episode_id, "metadata_length": expected, "frame_rows": actual})
    return {
        "checked_episodes": len(episode_rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def validate_task_refs(frame_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame_task_ids = {int(row["task_index"]) for row in frame_rows if row.get("task_index") is not None}
    task_ids = {int(row["task_index"]) for row in task_rows if row.get("task_index") is not None}
    missing = sorted(frame_task_ids - task_ids)
    return {
        "frame_task_ids": sorted(frame_task_ids),
        "task_metadata_ids": sorted(task_ids),
        "missing_task_ids": missing,
    }


def validate_video_refs(info: dict[str, Any], snapshot_dir: Path, episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    video_keys = feature_keys_by_dtype(info, "video")
    template = info.get("video_path")
    chunks_size = int(info.get("chunks_size") or 1000)
    if not video_keys or not template:
        return {
            "video_keys": video_keys,
            "expected_refs": 0,
            "missing_refs": 0,
            "missing_paths": [],
        }

    missing_paths: list[str] = []
    expected_refs = 0
    for row in episode_rows:
        if "episode_index" not in row:
            continue
        episode_index = int(row["episode_index"])
        episode_chunk = episode_index // chunks_size
        for video_key in video_keys:
            expected_refs += 1
            relative_path = template.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=video_key,
            )
            if not (snapshot_dir / relative_path).exists():
                missing_paths.append(relative_path)

    return {
        "video_keys": video_keys,
        "expected_refs": expected_refs,
        "missing_refs": len(missing_paths),
        "missing_paths": missing_paths[:20],
    }


def validate_shape_consistency(info: dict[str, Any], frame_table: pa.Table, keys: list[str]) -> dict[str, Any]:
    features = info.get("features", {})
    results: dict[str, Any] = {}
    for key in keys:
        if key not in frame_table.column_names:
            results[key] = {"status": "missing_from_frame_data"}
            continue
        expected_shape = features.get(key, {}).get("shape") if isinstance(features, dict) else None
        expected_len = expected_shape[0] if isinstance(expected_shape, list) and expected_shape else None
        if expected_len is None:
            results[key] = {"status": "skipped", "reason": "missing_shape_in_info"}
            continue

        values = frame_table[key].combine_chunks().to_pylist()
        bad_count = sum(1 for value in values if not isinstance(value, list) or len(value) != expected_len)
        results[key] = {
            "status": "ok" if bad_count == 0 else "error",
            "expected_length": expected_len,
            "bad_rows": bad_count,
        }
    return results


def validate_contract(snapshot_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    info_path = snapshot_dir / "meta" / "info.json"
    info = read_json(info_path)
    frame_files = parquet_files(snapshot_dir / "data")
    episode_rows = load_episode_rows(snapshot_dir)
    task_rows = load_task_rows(snapshot_dir)

    add_check(checks, "snapshot_dir_exists", snapshot_dir.exists(), {"path": str(snapshot_dir)})
    add_check(checks, "info_json_exists", info_path.exists(), {"path": str(info_path)})
    add_check(checks, "frame_parquet_exists", bool(frame_files), {"files": len(frame_files)})
    add_check(checks, "episode_metadata_exists", bool(episode_rows), {"rows": len(episode_rows)})
    add_check(checks, "task_metadata_exists", bool(task_rows), {"rows": len(task_rows)})

    frame_table = load_frame_table(snapshot_dir)
    frame_columns = set(frame_table.column_names)
    missing_required = sorted(REQUIRED_FRAME_COLUMNS - frame_columns)
    action_keys = [key for key in ["action", "action.continuous", "action.discrete"] if key in frame_columns]
    observation_keys = observation_feature_keys(info, frame_columns)
    video_report = validate_video_refs(info, snapshot_dir, episode_rows)

    add_check(
        checks,
        "required_frame_columns_present",
        not missing_required,
        {"required": sorted(REQUIRED_FRAME_COLUMNS), "missing": missing_required},
    )
    add_check(checks, "action_column_exists", bool(action_keys), {"action_columns": action_keys})
    add_check(checks, "observation_column_exists", bool(observation_keys), {"observation_columns": observation_keys})
    add_check(
        checks,
        "video_path_template_exists",
        bool(info.get("video_path")),
        {"video_path": info.get("video_path")},
    )
    add_check(
        checks,
        "video_refs_resolvable",
        video_report["expected_refs"] > 0 and video_report["missing_refs"] == 0,
        video_report,
    )

    frame_rows: list[dict[str, Any]] = []
    if not missing_required and frame_table.num_rows:
        frame_rows = frame_table.select(sorted(REQUIRED_FRAME_COLUMNS)).to_pylist()
        monotonic = validate_monotonic(frame_rows)
        length_report = validate_episode_lengths(frame_rows, episode_rows)
        task_report = validate_task_refs(frame_rows, task_rows)
        add_check(checks, "timestamp_monotonic_by_episode", monotonic["timestamp_monotonic"], monotonic)
        add_check(checks, "frame_index_monotonic_by_episode", monotonic["frame_index_monotonic"], monotonic)
        add_check(checks, "frame_index_unique_by_episode", monotonic["frame_index_unique"], monotonic)
        add_check(checks, "frame_index_contiguous_by_episode", monotonic["frame_index_contiguous"], monotonic)
        add_check(checks, "episode_lengths_match_frames", length_report["mismatch_count"] == 0, length_report)
        add_check(checks, "task_metadata_covers_frame_tasks", not task_report["missing_task_ids"], task_report)
    else:
        add_check(checks, "frame_quality_checks_runnable", False, {"reason": "missing required frame columns"})

    shape_report = validate_shape_consistency(info, frame_table, [*action_keys, *observation_keys])
    shape_errors = {key: value for key, value in shape_report.items() if value.get("status") == "error"}
    add_check(checks, "action_observation_shapes_match_info", not shape_errors, {"features": shape_report})

    errors = [check for check in checks if check["status"] == "error"]
    warnings = [check for check in checks if check["status"] == "warning"]
    return {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": CONTRACT_NAME,
        "dataset": snapshot_dir.name,
        "snapshot_dir": str(snapshot_dir),
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "summary": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "declared_episodes": info.get("total_episodes"),
            "declared_frames": info.get("total_frames"),
            "declared_tasks": info.get("total_tasks"),
            "frame_parquet_files": len(frame_files),
            "episode_metadata_rows": len(episode_rows),
            "task_metadata_rows": len(task_rows),
            "frame_rows_loaded": frame_table.num_rows,
            "video_expected_refs": video_report["expected_refs"],
            "video_missing_refs": video_report["missing_refs"],
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a source robot dataset against the platform contract.")
    parser.add_argument(
        "--snapshot-dir",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help="Local source dataset snapshot directory.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Contract validation report path.",
    )
    args = parser.parse_args()

    report = validate_contract(Path(args.snapshot_dir))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
