from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from robot_dataset_platform.lakehouse.iceberg import (
    build_iceberg_spark,
    latest_snapshot_id,
    load_yaml_config,
    write_iceberg_table,
)


def full_table_name(catalog: str, namespace: str, table: str) -> str:
    return f"{catalog}.{namespace}.{table}"


def require_columns(df: DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def select_frame_columns(frames: DataFrame, config: dict[str, Any]) -> DataFrame:
    columns = config["columns"]
    required = [
        columns["episode"],
        columns["frame"],
        columns["timestamp"],
        columns["task"],
        columns["instruction"],
    ]
    sensor_timestamp = columns.get("sensor_timestamp")
    if sensor_timestamp:
        required.append(sensor_timestamp)
    require_columns(frames, required)

    selected = frames.select(
        F.col(columns["episode"]).cast("long").alias("raw_episode_id"),
        F.col(columns["frame"]).cast("long").alias("raw_frame_index"),
        F.col(columns["timestamp"]).cast("double").alias("raw_timestamp"),
        F.col(columns["task"]).cast("long").alias("raw_task_index"),
        F.col(columns["instruction"]).cast("string").alias("instruction_text"),
        *(
            [F.col(sensor_timestamp).cast("long").alias("raw_sensor_timestamp")]
            if sensor_timestamp
            else [F.lit(None).cast("long").alias("raw_sensor_timestamp")]
        ),
    )

    episode_filter = config.get("episode_filter") or []
    if episode_filter:
        selected = selected.where(
            F.col("raw_episode_id").isin([int(episode) for episode in episode_filter])
        )

    return selected


def build_anchor_samples(
    frames: DataFrame,
    config: dict[str, Any],
    source_table: str,
    source_snapshot_id: int | None,
) -> DataFrame:
    frame_stride = int(config.get("frame_stride", 30))
    window = config.get("window", {})
    quality = config.get("quality", {})
    sync_rule = quality.get("sync_rule", "frame_window_timestamp_validation_v1")

    base = (
        frames.where((F.col("raw_frame_index") % F.lit(frame_stride)) == 0)
        .select(
            F.col("raw_episode_id"),
            F.col("raw_episode_id").cast("string").alias("episode_id"),
            F.col("raw_frame_index").alias("anchor_frame"),
            F.col("raw_timestamp").alias("anchor_timestamp"),
            F.col("raw_sensor_timestamp").alias("anchor_sensor_timestamp"),
            F.concat(F.lit("task_"), F.col("raw_task_index").cast("string")).alias(
                "instruction_id"
            ),
            F.col("instruction_text"),
        )
        .withColumn(
            "sample_id",
            F.concat_ws(
                "_",
                F.lit("sample"),
                F.col("episode_id"),
                F.col("instruction_id"),
                F.col("anchor_frame").cast("string"),
            ),
        )
        .withColumn(
            "observation_start_frame",
            F.col("anchor_frame") + F.lit(int(window.get("observation_start_offset", -10))),
        )
        .withColumn(
            "observation_end_frame",
            F.col("anchor_frame") + F.lit(int(window.get("observation_end_offset", 0))),
        )
        .withColumn(
            "action_start_frame",
            F.col("anchor_frame") + F.lit(int(window.get("action_start_offset", 1))),
        )
        .withColumn(
            "action_end_frame",
            F.col("anchor_frame") + F.lit(int(window.get("action_end_offset", 10))),
        )
        .where(F.col("observation_start_frame") >= 0)
        .withColumn(
            "expected_observation_rows",
            F.col("observation_end_frame") - F.col("observation_start_frame") + F.lit(1),
        )
        .withColumn(
            "expected_action_rows",
            F.col("action_end_frame") - F.col("action_start_frame") + F.lit(1),
        )
        .withColumn("source_frames_table", F.lit(source_table))
        .withColumn("source_frames_snapshot_id", F.lit(source_snapshot_id).cast("long"))
        .withColumn("sync_rule", F.lit(sync_rule))
    )
    return base


def build_window_stats(
    samples: DataFrame,
    frames: DataFrame,
    prefix: str,
    start_col: str,
    end_col: str,
) -> DataFrame:
    joined = (
        samples.select("sample_id", "raw_episode_id", start_col, end_col)
        .alias("s")
        .join(
            frames.alias("f"),
            (F.col("s.raw_episode_id") == F.col("f.raw_episode_id"))
            & (F.col("f.raw_frame_index") >= F.col(f"s.{start_col}"))
            & (F.col("f.raw_frame_index") <= F.col(f"s.{end_col}")),
            "left",
        )
        .select(
            F.col("s.sample_id"),
            F.col("f.raw_frame_index").alias("window_frame"),
            F.col("f.raw_timestamp").alias("window_timestamp"),
        )
    )

    order = Window.partitionBy("sample_id").orderBy("window_frame")
    with_gaps = (
        joined.withColumn("prev_timestamp", F.lag("window_timestamp").over(order))
        .withColumn(
            "timestamp_gap_seconds",
            F.when(
                F.col("prev_timestamp").isNotNull(),
                F.col("window_timestamp") - F.col("prev_timestamp"),
            ),
        )
        .withColumn(
            "abs_timestamp_gap_seconds",
            F.abs(F.col("timestamp_gap_seconds")),
        )
    )

    return with_gaps.groupBy("sample_id").agg(
        F.count("window_frame").cast("long").alias(f"{prefix}_rows"),
        F.min("window_frame").cast("long").alias(f"{prefix}_actual_start_frame"),
        F.max("window_frame").cast("long").alias(f"{prefix}_actual_end_frame"),
        F.min("window_timestamp").alias(f"{prefix}_start_timestamp"),
        F.max("window_timestamp").alias(f"{prefix}_end_timestamp"),
        (
            F.max("window_timestamp") - F.min("window_timestamp")
        ).alias(f"{prefix}_duration_seconds"),
        F.max("abs_timestamp_gap_seconds").alias(f"{prefix}_max_timestamp_gap_seconds"),
        F.min("timestamp_gap_seconds").alias(f"{prefix}_min_timestamp_gap_seconds"),
    )


def add_quality_columns(samples: DataFrame, config: dict[str, Any]) -> DataFrame:
    max_gap = float(config.get("quality", {}).get("max_timestamp_gap_seconds", 0.25))

    enriched = (
        samples.withColumn(
            "observation_rows", F.coalesce(F.col("observation_rows"), F.lit(0)).cast("long")
        )
        .withColumn("action_rows", F.coalesce(F.col("action_rows"), F.lit(0)).cast("long"))
        .withColumn(
            "observation_missing_rows",
            F.col("expected_observation_rows") - F.col("observation_rows"),
        )
        .withColumn(
            "action_missing_rows",
            F.col("expected_action_rows") - F.col("action_rows"),
        )
        .withColumn(
            "max_timestamp_gap_seconds",
            F.greatest(
                F.coalesce(F.col("observation_max_timestamp_gap_seconds"), F.lit(0.0)),
                F.coalesce(F.col("action_max_timestamp_gap_seconds"), F.lit(0.0)),
            ),
        )
        .withColumn(
            "min_timestamp_gap_seconds",
            F.least(
                F.coalesce(F.col("observation_min_timestamp_gap_seconds"), F.lit(0.0)),
                F.coalesce(F.col("action_min_timestamp_gap_seconds"), F.lit(0.0)),
            ),
        )
        .withColumn(
            "sync_status",
            F.when(
                (F.col("observation_missing_rows") > 0) | (F.col("action_missing_rows") > 0),
                F.lit("missing_window_rows"),
            )
            .when(F.col("min_timestamp_gap_seconds") < 0, F.lit("non_monotonic_timestamp"))
            .when(F.col("max_timestamp_gap_seconds") > F.lit(max_gap), F.lit("timestamp_gap_exceeded"))
            .otherwise(F.lit("ok")),
        )
        .withColumn("max_allowed_timestamp_gap_seconds", F.lit(max_gap))
        .withColumn(
            "is_window_complete",
            (F.col("observation_missing_rows") == 0) & (F.col("action_missing_rows") == 0),
        )
        .withColumn("is_timestamp_monotonic", F.col("min_timestamp_gap_seconds") >= 0)
        .withColumn("is_gap_within_threshold", F.col("max_timestamp_gap_seconds") <= F.lit(max_gap))
    )
    return enriched


def stable_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_synced_samples(
    frames: DataFrame,
    config: dict[str, Any],
    source_table: str,
    source_snapshot_id: int | None,
) -> DataFrame:
    selected_frames = select_frame_columns(frames, config).cache()
    anchors = build_anchor_samples(selected_frames, config, source_table, source_snapshot_id)

    observation_stats = build_window_stats(
        anchors,
        selected_frames,
        "observation",
        "observation_start_frame",
        "observation_end_frame",
    )
    action_stats = build_window_stats(
        anchors,
        selected_frames,
        "action",
        "action_start_frame",
        "action_end_frame",
    )

    samples = (
        anchors.join(observation_stats, "sample_id", "left")
        .join(action_stats, "sample_id", "left")
        .drop("raw_episode_id")
    )
    return add_quality_columns(samples, config)


def status_counts(samples: DataFrame) -> dict[str, int]:
    return {
        row["sync_status"]: int(row["count"])
        for row in samples.groupBy("sync_status").count().collect()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build timestamp-aware synced sample table from raw Iceberg frames."
    )
    parser.add_argument(
        "--config",
        default="configs/sync_builder.yaml",
        help="Sensor sync builder config path.",
    )
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    catalog_name = config.get("catalog_name", "robot_lakehouse")
    source = config["source"]
    target = config["target"]
    registry_dir = Path(config["registry_dir"])
    parquet_compression = config.get("parquet_compression", "snappy")

    source_table = full_table_name(catalog_name, source["namespace"], source["frames_table"])
    target_table = full_table_name(catalog_name, target["namespace"], target["samples_table"])

    registry_dir.mkdir(parents=True, exist_ok=True)

    spark = build_iceberg_spark(config, "robot-sensor-sync-builder")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog_name}.{target['namespace']}")

    source_snapshot_id = latest_snapshot_id(spark, source_table)
    frames = spark.table(source_table)
    synced_samples = build_synced_samples(frames, config, source_table, source_snapshot_id)

    sample_count = write_iceberg_table(synced_samples, target_table, parquet_compression)
    target_snapshot_id = latest_snapshot_id(spark, target_table)
    counts = status_counts(spark.table(target_table))
    max_gap = spark.table(target_table).agg(F.max("max_timestamp_gap_seconds")).collect()[0][0]

    lineage = {
        "dataset_name": config["dataset_name"],
        "sync_config_sha256": stable_config_hash(config),
        "source_table": source_table,
        "source_snapshot_id": source_snapshot_id,
        "target_table": target_table,
        "target_snapshot_id": target_snapshot_id,
        "episode_filter": config.get("episode_filter", []),
        "frame_stride": config.get("frame_stride"),
        "window": config.get("window", {}),
        "quality": config.get("quality", {}),
    }
    (registry_dir / "sync_lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats = {
        "synced_samples": sample_count,
        "sync_status_counts": counts,
        "max_observed_timestamp_gap_seconds": max_gap,
    }
    (registry_dir / "sync_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result = {
        "target_table": target_table,
        "rows": sample_count,
        "target_snapshot_id": target_snapshot_id,
        "sync_status_counts": counts,
        "max_observed_timestamp_gap_seconds": max_gap,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main()
