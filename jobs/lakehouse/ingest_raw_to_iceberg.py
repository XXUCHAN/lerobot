from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from robot_dataset_platform.lakehouse.iceberg import (
    build_iceberg_spark,
    load_yaml_config,
    write_iceberg_table,
)


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    if not normalized:
        return "col"
    if normalized[0].isdigit():
        return f"col_{normalized}"
    return normalized


def normalize_columns(df: DataFrame) -> DataFrame:
    used: set[str] = set()
    expressions = []
    for original in df.columns:
        normalized = normalize_column_name(original)
        candidate = normalized
        suffix = 2
        while candidate in used:
            candidate = f"{normalized}_{suffix}"
            suffix += 1
        used.add(candidate)
        expressions.append(F.col(f"`{original}`").alias(candidate))
    return df.select(*expressions)


def read_parquet_dir(spark: SparkSession, path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet directory: {path}")
    return spark.read.option("recursiveFileLookup", "true").parquet(str(path))


def read_jsonl_file(spark: SparkSession, path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    return spark.read.json(str(path))


def build_episodes_from_frames(frames: DataFrame) -> DataFrame:
    if "episode_index" not in frames.columns:
        raise ValueError("Cannot derive episodes from frames without episode_index column.")

    aggregations = [F.count("*").cast("long").alias("length")]
    if "timestamp" in frames.columns:
        aggregations.extend(
            [
                F.min("timestamp").alias("start_timestamp"),
                F.max("timestamp").alias("end_timestamp"),
            ]
        )
    if "frame_index" in frames.columns:
        aggregations.extend(
            [
                F.min("frame_index").cast("long").alias("start_frame"),
                F.max("frame_index").cast("long").alias("end_frame"),
            ]
        )
    if "task_index" in frames.columns:
        aggregations.append(F.collect_set("task_index").alias("task_indices"))

    return frames.groupBy("episode_index").agg(*aggregations).orderBy("episode_index")


def read_or_build_episodes(spark: SparkSession, snapshot_dir: Path, frames: DataFrame) -> tuple[DataFrame, str]:
    episodes_dir = snapshot_dir / "meta" / "episodes"
    episodes_jsonl = snapshot_dir / "meta" / "episodes.jsonl"

    if episodes_dir.exists():
        return normalize_columns(read_parquet_dir(spark, episodes_dir)), "meta/episodes"
    if episodes_jsonl.exists():
        return normalize_columns(read_jsonl_file(spark, episodes_jsonl)), "meta/episodes.jsonl"
    return normalize_columns(build_episodes_from_frames(frames)), "derived_from_frames"


def read_tasks(spark: SparkSession, snapshot_dir: Path) -> tuple[DataFrame | None, str | None]:
    tasks_parquet = snapshot_dir / "meta" / "tasks.parquet"
    tasks_jsonl = snapshot_dir / "meta" / "tasks.jsonl"

    if tasks_parquet.exists():
        return normalize_columns(spark.read.parquet(str(tasks_parquet))), "meta/tasks.parquet"
    if tasks_jsonl.exists():
        return normalize_columns(read_jsonl_file(spark, tasks_jsonl)), "meta/tasks.jsonl"
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw LeRobot parquet files into Iceberg tables.")
    parser.add_argument(
        "--config",
        default="configs/lakehouse_ingest.yaml",
        help="Lakehouse ingest config path.",
    )
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    snapshot_dir = Path(config["snapshot_dir"])
    catalog_name = config.get("catalog_name", "robot_lakehouse")
    namespace = config.get("namespace", "raw")
    tables = config.get("tables", {})
    parquet_compression = config.get("parquet_compression", "snappy")

    spark = build_iceberg_spark(config, "robot-lakehouse-iceberg-ingest")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog_name}.{namespace}")

    frames = normalize_columns(read_parquet_dir(spark, snapshot_dir / "data"))
    frames_table = f"{catalog_name}.{namespace}.{tables.get('frames', 'frames')}"
    frame_rows = write_iceberg_table(frames, frames_table, parquet_compression)

    episodes, episodes_source = read_or_build_episodes(spark, snapshot_dir, frames)
    episodes_table = f"{catalog_name}.{namespace}.{tables.get('episodes', 'episodes')}"
    episode_rows = write_iceberg_table(episodes, episodes_table, parquet_compression)

    task_rows = 0
    tasks_table = f"{catalog_name}.{namespace}.{tables.get('tasks', 'tasks')}"
    tasks, tasks_source = read_tasks(spark, snapshot_dir)
    if tasks is not None:
        task_rows = write_iceberg_table(tasks, tasks_table, parquet_compression)

    result = {
        "catalog": catalog_name,
        "namespace": namespace,
        "warehouse_dir": config["warehouse_dir"],
        "tables": {
            frames_table: {"rows": frame_rows, "source": "data"},
            episodes_table: {"rows": episode_rows, "source": episodes_source},
            tasks_table: {"rows": task_rows, "source": tasks_source},
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main()
