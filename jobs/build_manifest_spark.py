from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from robot_dataset_platform.lakehouse.iceberg import build_iceberg_spark, latest_snapshot_id


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_info(snapshot_dir: Path) -> dict:
    info_path = snapshot_dir / "meta" / "info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8"))


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def spark_col(name: str):
    return F.col(f"`{name}`") if "." in name else F.col(name)


def build_plain_spark(app_name: str, spark_master: str) -> SparkSession:
    return SparkSession.builder.appName(app_name).master(spark_master).getOrCreate()


def build_manifest_from_synced_table(spark: SparkSession, config: dict):
    source_table = config["source_table"]
    require_sync_status = config.get("require_sync_status")
    source = spark.table(source_table)
    if require_sync_status:
        source = source.where(F.col("sync_status") == F.lit(require_sync_status))

    manifest_columns = [
        "sample_id",
        "episode_id",
        "instruction_id",
        "instruction_text",
        "anchor_frame",
        "observation_start_frame",
        "observation_end_frame",
        "action_start_frame",
        "action_end_frame",
        "anchor_timestamp",
        "observation_start_timestamp",
        "observation_end_timestamp",
        "action_start_timestamp",
        "action_end_timestamp",
        "observation_rows",
        "action_rows",
        "max_timestamp_gap_seconds",
        "sync_status",
        "sync_rule",
        "source_frames_table",
        "source_frames_snapshot_id",
    ]
    return source.select(*[column for column in manifest_columns if column in source.columns])


def build_manifest_from_snapshot(spark: SparkSession, config: dict):
    snapshot_dir = Path(config["snapshot_dir"])
    frame_stride = int(config.get("frame_stride", 30))
    episode_filter = config.get("episode_filter", [])
    window = config.get("window", {})

    data_path = snapshot_dir / "data"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing frame data directory: {data_path}")

    frames = spark.read.option("recursiveFileLookup", "true").parquet(str(data_path))
    columns = frames.columns
    episode_col = pick_column(columns, ["episode_index", "episode_id", "episode"])
    frame_col = pick_column(columns, ["frame_index", "index", "timestamp"])
    task_col = pick_column(columns, ["task_index", "task_id"])
    instruction_text_col = pick_column(columns, ["task.instructions", "task.policy"])

    if episode_col is None:
        frames = frames.withColumn("episode_id", F.lit("ep_unknown"))
        episode_col = "episode_id"

    if frame_col is None:
        frames = frames.withColumn("frame_index", F.monotonically_increasing_id())
        frame_col = "frame_index"

    if task_col is None:
        frames = frames.withColumn("instruction_id", F.lit("inst_unknown"))
        task_col = "instruction_id"

    if episode_filter:
        frames = frames.where(spark_col(episode_col).isin([int(episode) for episode in episode_filter]))

    if instruction_text_col is None:
        frames = frames.withColumn("instruction_text", F.lit(None).cast("string"))
        instruction_text_col = "instruction_text"

    base = (
        frames.select(
            spark_col(episode_col).cast("string").alias("episode_id"),
            spark_col(frame_col).cast("long").alias("anchor_frame"),
            F.concat(F.lit("task_"), spark_col(task_col).cast("string")).alias("instruction_id"),
            spark_col(instruction_text_col).cast("string").alias("instruction_text"),
        )
        .where((F.col("anchor_frame") % F.lit(frame_stride)) == 0)
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
    )

    return (
        base.withColumn(
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
        .select(
            "sample_id",
            "episode_id",
            "instruction_id",
            "instruction_text",
            "anchor_frame",
            "observation_start_frame",
            "observation_end_frame",
            "action_start_frame",
            "action_end_frame",
        )
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sample dataset manifest with Spark.")
    parser.add_argument(
        "--config",
        default="configs/manifest_builder.yaml",
        help="Manifest builder config path.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    manifest_dir = Path(config["manifest_dir"])
    registry_dir = Path(config["registry_dir"])
    spark_master = config.get("spark_master", "local[*]")
    parquet_compression = config.get("parquet_compression", "snappy")
    source_table = config.get("source_table")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    if source_table:
        spark = build_iceberg_spark(config, "robot-dataset-manifest-builder")
        manifest = build_manifest_from_synced_table(spark, config)
        source_snapshot_id = latest_snapshot_id(spark, source_table)
    else:
        spark = build_plain_spark("robot-dataset-manifest-builder", spark_master)
        manifest = build_manifest_from_snapshot(spark, config)
        source_snapshot_id = None

    order_columns = [
        column
        for column in ["episode_id", "anchor_frame", "instruction_id"]
        if column in manifest.columns
    ]
    if order_columns:
        manifest = manifest.orderBy(*order_columns)
    manifest = manifest.cache()
    manifest_rows = manifest.count()

    output_path = manifest_dir / "manifest.jsonl"
    parquet_path = manifest_dir / "manifest.parquet"
    temp_path = manifest_dir / "_manifest_json"
    manifest.coalesce(1).write.mode("overwrite").json(str(temp_path))
    manifest.write.mode("overwrite").option("compression", parquet_compression).parquet(
        str(parquet_path)
    )

    part_files = sorted(temp_path.glob("part-*.json"))
    if part_files:
        output_path.write_text(part_files[0].read_text(encoding="utf-8"), encoding="utf-8")
    else:
        output_path.write_text("", encoding="utf-8")

    manifest_sha256 = file_sha256(output_path)
    manifest_version = manifest_sha256[:16]

    info = read_info(Path(config["snapshot_dir"])) if config.get("snapshot_dir") else {}
    lineage = {
        "dataset_name": config["dataset_name"],
        "manifest_version": manifest_version,
        "manifest_sha256": manifest_sha256,
        "source_type": "iceberg_synced_table" if source_table else "snapshot_parquet",
        "snapshot_dir": config.get("snapshot_dir"),
        "source_table": source_table,
        "source_snapshot_id": source_snapshot_id,
        "source_codebase_version": info.get("codebase_version"),
        "source_total_episodes": info.get("total_episodes"),
        "source_total_frames": info.get("total_frames"),
        "episode_filter": config.get("episode_filter", []),
        "frame_stride": config.get("frame_stride"),
        "window": config.get("window", {}),
        "require_sync_status": config.get("require_sync_status"),
        "manifest_path": str(output_path),
        "manifest_parquet_path": str(parquet_path),
        "parquet_compression": parquet_compression,
    }
    (registry_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats = {
        "manifest_rows": manifest_rows,
        "manifest_version": manifest_version,
        "manifest_sha256": manifest_sha256,
    }
    (registry_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "dataset_name": config["dataset_name"],
        "format": "robot_dataset_manifest_v0",
        "manifest_version": manifest_version,
        "manifest_path": str(output_path),
        "manifest_parquet_path": str(parquet_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (registry_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote manifest: {output_path}")
    print(f"Manifest version: {manifest_version}")
    print(f"Wrote snappy parquet manifest: {parquet_path}")
    print(f"Wrote registry: {registry_dir}")
    manifest.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
