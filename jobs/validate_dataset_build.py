from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from robot_dataset_platform.lakehouse.iceberg import (
    build_iceberg_spark,
    latest_snapshot_id,
    load_yaml_config,
)


def full_table_name(catalog: str, namespace: str, table: str) -> str:
    return f"{catalog}.{namespace}.{table}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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


def table_count(spark: SparkSession, table_name: str) -> int:
    return int(spark.sql(f"SELECT COUNT(*) AS rows FROM {table_name}").collect()[0]["rows"])


def duplicate_count(df: DataFrame, column: str) -> int:
    total = df.count()
    distinct = df.select(column).distinct().count()
    return int(total - distinct)


def duplicate_group_count(df: DataFrame, columns: list[str]) -> int:
    return int(df.groupBy(*columns).count().where(F.col("count") > 1).count())


def missing_required_count(df: DataFrame, columns: list[str]) -> int:
    condition = None
    for column in columns:
        current = F.col(column).isNull()
        condition = current if condition is None else condition | current
    if condition is None:
        return 0
    return int(df.where(condition).count())


def validate_raw_tables(
    spark: SparkSession,
    config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = config.get("catalog_name", "robot_lakehouse")
    source = config["source"]
    raw_namespace = source["namespace"]
    raw_tables = {
        "frames": full_table_name(catalog, raw_namespace, source["frames_table"]),
        "episodes": full_table_name(catalog, raw_namespace, source.get("episodes_table", "episodes")),
        "tasks": full_table_name(catalog, raw_namespace, source.get("tasks_table", "tasks")),
    }

    result: dict[str, Any] = {}
    for key, table_name in raw_tables.items():
        rows = table_count(spark, table_name)
        snapshot_id = latest_snapshot_id(spark, table_name)
        result[key] = {"table": table_name, "rows": rows, "snapshot_id": snapshot_id}
        add_check(
            checks,
            f"raw_{key}_table_not_empty",
            rows > 0,
            {"table": table_name, "rows": rows, "snapshot_id": snapshot_id},
        )
        add_check(
            checks,
            f"raw_{key}_snapshot_exists",
            snapshot_id is not None,
            {"table": table_name, "snapshot_id": snapshot_id},
        )

    return result


def validate_synced_table(
    spark: SparkSession,
    sync_config: dict[str, Any],
    manifest_config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = sync_config.get("catalog_name", "robot_lakehouse")
    target = sync_config["target"]
    synced_table = full_table_name(catalog, target["namespace"], target["samples_table"])
    synced = spark.table(synced_table).cache()
    rows = synced.count()
    snapshot_id = latest_snapshot_id(spark, synced_table)
    required_status = manifest_config.get("require_sync_status", "ok")

    status_counts = {
        row["sync_status"]: int(row["count"])
        for row in synced.groupBy("sync_status").count().collect()
    }
    required_status_rows = int(status_counts.get(required_status, 0))
    non_required_rows = rows - required_status_rows
    duplicate_samples = duplicate_count(synced, "sample_id")

    quality_failed = 0
    for column in ["is_window_complete", "is_timestamp_monotonic", "is_gap_within_threshold"]:
        if column in synced.columns:
            quality_failed += synced.where(~F.col(column)).count()

    required_columns = [
        "sample_id",
        "episode_id",
        "instruction_id",
        "instruction_text",
        "anchor_frame",
        "observation_start_frame",
        "observation_end_frame",
        "action_start_frame",
        "action_end_frame",
        "source_frames_snapshot_id",
    ]
    missing_required_rows = missing_required_count(
        synced,
        [column for column in required_columns if column in synced.columns],
    )

    add_check(
        checks,
        "synced_table_not_empty",
        rows > 0,
        {"table": synced_table, "rows": rows, "snapshot_id": snapshot_id},
    )
    add_check(
        checks,
        "synced_table_snapshot_exists",
        snapshot_id is not None,
        {"table": synced_table, "snapshot_id": snapshot_id},
    )
    add_check(
        checks,
        "synced_status_all_required",
        rows > 0 and non_required_rows == 0,
        {"required_status": required_status, "status_counts": status_counts},
    )
    add_check(
        checks,
        "synced_sample_id_unique",
        duplicate_samples == 0,
        {"duplicate_samples": duplicate_samples},
    )
    add_check(
        checks,
        "synced_required_fields_present",
        missing_required_rows == 0,
        {"missing_required_rows": missing_required_rows},
    )
    add_check(
        checks,
        "synced_quality_flags_pass",
        quality_failed == 0,
        {"failed_quality_flag_count": int(quality_failed)},
    )

    lineage_path = Path(sync_config["registry_dir"]) / "sync_lineage.json"
    stats_path = Path(sync_config["registry_dir"]) / "sync_stats.json"
    lineage = read_json(lineage_path)
    stats = read_json(stats_path)
    add_check(
        checks,
        "sync_lineage_snapshot_matches_table",
        lineage.get("target_snapshot_id") == snapshot_id,
        {
            "lineage_path": str(lineage_path),
            "lineage_target_snapshot_id": lineage.get("target_snapshot_id"),
            "actual_snapshot_id": snapshot_id,
        },
    )
    add_check(
        checks,
        "sync_stats_count_matches_table",
        stats.get("synced_samples") == rows,
        {
            "stats_path": str(stats_path),
            "stats_synced_samples": stats.get("synced_samples"),
            "actual_rows": rows,
        },
    )

    result = {
        "table": synced_table,
        "rows": rows,
        "snapshot_id": snapshot_id,
        "required_status_rows": required_status_rows,
        "status_counts": status_counts,
        "duplicate_samples": duplicate_samples,
    }
    synced.unpersist()
    return result


def filtered_annotations(spark: SparkSession, manifest_config: dict[str, Any]) -> DataFrame | None:
    annotation_table = manifest_config.get("annotation_table")
    if not annotation_table:
        return None

    annotations = spark.table(annotation_table)
    if manifest_config.get("annotation_active_only", True) and "is_active" in annotations.columns:
        annotations = annotations.where(F.col("is_active") == F.lit(True))
    if manifest_config.get("annotation_version") and "annotation_version" in annotations.columns:
        annotations = annotations.where(
            F.col("annotation_version") == F.lit(manifest_config["annotation_version"])
        )
    return annotations


def validate_annotation_table(
    spark: SparkSession,
    manifest_config: dict[str, Any],
    annotation_config: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    annotation_table = manifest_config.get("annotation_table")
    if not annotation_table:
        return {"enabled": False}

    annotations = filtered_annotations(spark, manifest_config)
    if annotations is None:
        return {"enabled": False}

    annotations = annotations.cache()
    rows = annotations.count()
    snapshot_id = latest_snapshot_id(spark, annotation_table)
    duplicate_annotations = duplicate_group_count(annotations, ["episode_id", "instruction_id"])
    required_columns = [
        "instruction_id",
        "episode_id",
        "source_instruction_id",
        "text",
        "annotation_type",
        "annotation_version",
        "annotation_policy",
        "is_active",
    ]
    missing_required_rows = missing_required_count(annotations, required_columns)
    type_counts = {
        row["annotation_type"]: int(row["count"])
        for row in annotations.groupBy("annotation_type").count().collect()
    }

    add_check(
        checks,
        "annotation_table_not_empty",
        rows > 0,
        {"table": annotation_table, "rows": rows, "snapshot_id": snapshot_id},
    )
    add_check(
        checks,
        "annotation_table_snapshot_exists",
        snapshot_id is not None,
        {"table": annotation_table, "snapshot_id": snapshot_id},
    )
    add_check(
        checks,
        "annotation_episode_instruction_unique",
        duplicate_annotations == 0,
        {"duplicate_annotations": duplicate_annotations},
    )
    add_check(
        checks,
        "annotation_required_fields_present",
        missing_required_rows == 0,
        {"missing_required_rows": missing_required_rows},
    )

    registry_dir = Path(annotation_config.get("registry_dir", ""))
    lineage_path = registry_dir / "annotation_lineage.json"
    stats_path = registry_dir / "annotation_stats.json"
    lineage = read_json(lineage_path)
    stats = read_json(stats_path)
    add_check(
        checks,
        "annotation_lineage_snapshot_matches_table",
        not registry_dir.name
        or lineage.get("target_snapshot_id") == snapshot_id,
        {
            "lineage_path": str(lineage_path),
            "lineage_target_snapshot_id": lineage.get("target_snapshot_id"),
            "actual_snapshot_id": snapshot_id,
        },
    )
    add_check(
        checks,
        "annotation_stats_count_matches_table",
        not registry_dir.name or stats.get("annotations") == rows,
        {
            "stats_path": str(stats_path),
            "stats_annotations": stats.get("annotations"),
            "actual_rows": rows,
        },
    )

    result = {
        "enabled": True,
        "table": annotation_table,
        "rows": rows,
        "snapshot_id": snapshot_id,
        "annotation_type_counts": type_counts,
        "duplicate_annotations": duplicate_annotations,
    }
    annotations.unpersist()
    return result


def expected_manifest_rows(
    spark: SparkSession,
    manifest_config: dict[str, Any],
    synced_summary: dict[str, Any],
) -> int:
    annotation_table = manifest_config.get("annotation_table")
    if not annotation_table:
        return int(synced_summary["required_status_rows"])

    source = spark.table(manifest_config["source_table"])
    if manifest_config.get("require_sync_status"):
        source = source.where(F.col("sync_status") == F.lit(manifest_config["require_sync_status"]))

    annotations = filtered_annotations(spark, manifest_config)
    if annotations is None:
        return int(synced_summary["required_status_rows"])

    source_alias = source.alias("source")
    annotation_alias = annotations.alias("annotation")
    join_conditions = [F.col("source.episode_id") == F.col("annotation.episode_id")]
    join_columns = manifest_config.get("annotation_join_columns", ["episode_id"])
    if "source_instruction_id" in join_columns:
        join_conditions.append(
            F.col("source.instruction_id") == F.col("annotation.source_instruction_id")
        )

    join_condition = join_conditions[0]
    for condition in join_conditions[1:]:
        join_condition = join_condition & condition

    return int(source_alias.join(annotation_alias, join_condition, "inner").count())


def validate_manifest(
    spark: SparkSession,
    manifest_config: dict[str, Any],
    synced_summary: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_dir = Path(manifest_config["manifest_dir"])
    manifest_jsonl = manifest_dir / "manifest.jsonl"
    manifest_parquet = manifest_dir / "manifest.parquet"
    registry_dir = Path(manifest_config["registry_dir"])
    metadata_path = registry_dir / "metadata.json"
    lineage_path = registry_dir / "lineage.json"
    stats_path = registry_dir / "stats.json"

    manifest = spark.read.parquet(str(manifest_parquet)).cache()
    parquet_rows = manifest.count()
    jsonl_rows = jsonl_line_count(manifest_jsonl)
    manifest_sha256 = file_sha256(manifest_jsonl)
    manifest_version = manifest_sha256[:16]
    duplicate_samples = duplicate_count(manifest, "sample_id")

    required_columns = [
        "sample_id",
        "episode_id",
        "instruction_id",
        "anchor_frame",
        "observation_start_frame",
        "observation_end_frame",
        "action_start_frame",
        "action_end_frame",
    ]
    if manifest_config.get("annotation_table"):
        required_columns.extend(
            [
                "source_sample_id",
                "source_instruction_id",
                "source_instruction_text",
                "annotation_type",
                "annotation_version",
                "annotation_policy",
                "annotation_table",
                "annotation_snapshot_id",
            ]
        )
    missing_required_rows = missing_required_count(manifest, required_columns)
    invalid_windows = manifest.where(
        (F.col("observation_start_frame") > F.col("observation_end_frame"))
        | (F.col("action_start_frame") > F.col("action_end_frame"))
        | (F.col("observation_end_frame") >= F.col("action_start_frame"))
    ).count()

    metadata = read_json(metadata_path)
    lineage = read_json(lineage_path)
    stats = read_json(stats_path)
    expected_rows = expected_manifest_rows(spark, manifest_config, synced_summary)

    add_check(
        checks,
        "manifest_parquet_not_empty",
        parquet_rows > 0,
        {"manifest_parquet": str(manifest_parquet), "rows": parquet_rows},
    )
    add_check(
        checks,
        "manifest_jsonl_count_matches_parquet",
        jsonl_rows == parquet_rows,
        {"manifest_jsonl": str(manifest_jsonl), "jsonl_rows": jsonl_rows, "parquet_rows": parquet_rows},
    )
    add_check(
        checks,
        "manifest_count_matches_source_view",
        parquet_rows == expected_rows,
        {
            "manifest_rows": parquet_rows,
            "expected_manifest_rows": expected_rows,
            "annotation_table": manifest_config.get("annotation_table"),
        },
    )
    add_check(
        checks,
        "manifest_sample_id_unique",
        duplicate_samples == 0,
        {"duplicate_samples": duplicate_samples},
    )
    add_check(
        checks,
        "manifest_required_fields_present",
        missing_required_rows == 0,
        {"missing_required_rows": missing_required_rows},
    )
    add_check(
        checks,
        "manifest_windows_valid",
        invalid_windows == 0,
        {"invalid_windows": int(invalid_windows)},
    )
    add_check(
        checks,
        "manifest_hash_matches_registry",
        stats.get("manifest_sha256") == manifest_sha256
        and lineage.get("manifest_sha256") == manifest_sha256
        and metadata.get("manifest_version") == manifest_version,
        {
            "manifest_sha256": manifest_sha256,
            "manifest_version": manifest_version,
            "stats_manifest_sha256": stats.get("manifest_sha256"),
            "lineage_manifest_sha256": lineage.get("manifest_sha256"),
            "metadata_manifest_version": metadata.get("manifest_version"),
        },
    )
    add_check(
        checks,
        "manifest_source_snapshot_matches_synced_table",
        lineage.get("source_snapshot_id") == synced_summary["snapshot_id"],
        {
            "lineage_source_snapshot_id": lineage.get("source_snapshot_id"),
            "synced_table_snapshot_id": synced_summary["snapshot_id"],
        },
    )
    if manifest_config.get("annotation_table"):
        annotation_snapshot_id = manifest.select("annotation_snapshot_id").dropDuplicates().collect()
        annotation_snapshot_ids = [
            int(row["annotation_snapshot_id"])
            for row in annotation_snapshot_id
            if row["annotation_snapshot_id"] is not None
        ]
        add_check(
            checks,
            "manifest_annotation_snapshot_matches_lineage",
            len(set(annotation_snapshot_ids)) == 1
            and lineage.get("annotation_snapshot_id") == annotation_snapshot_ids[0],
            {
                "lineage_annotation_snapshot_id": lineage.get("annotation_snapshot_id"),
                "manifest_annotation_snapshot_ids": annotation_snapshot_ids,
            },
        )

    result = {
        "manifest_jsonl": str(manifest_jsonl),
        "manifest_parquet": str(manifest_parquet),
        "rows": parquet_rows,
        "jsonl_rows": jsonl_rows,
        "manifest_sha256": manifest_sha256,
        "manifest_version": manifest_version,
        "duplicate_samples": duplicate_samples,
    }
    manifest.unpersist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local robot dataset build artifacts.")
    parser.add_argument(
        "--sync-config",
        default="configs/sync_builder.yaml",
        help="Sensor sync config path.",
    )
    parser.add_argument(
        "--manifest-config",
        default="configs/manifest_from_synced.yaml",
        help="Synced manifest config path.",
    )
    parser.add_argument(
        "--annotation-config",
        default="configs/annotation_builder.yaml",
        help="Instruction annotation builder config path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Validation report output path.",
    )
    args = parser.parse_args()

    sync_config = load_yaml_config(Path(args.sync_config))
    manifest_config = load_yaml_config(Path(args.manifest_config))
    annotation_config_path = Path(args.annotation_config)
    annotation_config = load_yaml_config(annotation_config_path) if annotation_config_path.exists() else {}
    output_path = Path(args.output) if args.output else Path(manifest_config["registry_dir"]) / "validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spark = build_iceberg_spark(sync_config, "robot-dataset-build-validator")
    checks: list[dict[str, Any]] = []
    raw_summary = validate_raw_tables(spark, sync_config, checks)
    synced_summary = validate_synced_table(spark, sync_config, manifest_config, checks)
    annotation_summary = validate_annotation_table(spark, manifest_config, annotation_config, checks)
    manifest_summary = validate_manifest(spark, manifest_config, synced_summary, checks)

    errors = [check for check in checks if check["status"] == "error"]
    warnings = [check for check in checks if check["status"] == "warning"]
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if errors else "passed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "raw": raw_summary,
        "synced": synced_summary,
        "annotation": annotation_summary,
        "manifest": manifest_summary,
        "checks": checks,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    spark.stop()

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
