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

    episodes = normalize_columns(read_parquet_dir(spark, snapshot_dir / "meta" / "episodes"))
    episodes_table = f"{catalog_name}.{namespace}.{tables.get('episodes', 'episodes')}"
    episode_rows = write_iceberg_table(episodes, episodes_table, parquet_compression)

    tasks_path = snapshot_dir / "meta" / "tasks.parquet"
    task_rows = 0
    tasks_table = f"{catalog_name}.{namespace}.{tables.get('tasks', 'tasks')}"
    if tasks_path.exists():
        tasks = normalize_columns(spark.read.parquet(str(tasks_path)))
        task_rows = write_iceberg_table(tasks, tasks_table, parquet_compression)

    result = {
        "catalog": catalog_name,
        "namespace": namespace,
        "warehouse_dir": config["warehouse_dir"],
        "tables": {
            frames_table: {"rows": frame_rows},
            episodes_table: {"rows": episode_rows},
            tasks_table: {"rows": task_rows},
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main()
