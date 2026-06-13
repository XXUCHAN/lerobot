from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_dataset_platform.lakehouse.iceberg import build_iceberg_spark, load_yaml_config


def table_summary(spark, table_name: str) -> dict[str, Any]:
    rows = spark.sql(f"SELECT COUNT(*) AS rows FROM {table_name}").collect()[0]["rows"]
    snapshots = spark.sql(
        f"""
        SELECT snapshot_id, committed_at, operation
        FROM {table_name}.snapshots
        ORDER BY committed_at DESC
        LIMIT 5
        """
    ).toPandas()

    return {
        "rows": rows,
        "snapshots": snapshots.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local Iceberg raw tables.")
    parser.add_argument(
        "--config",
        default="configs/lakehouse_ingest.yaml",
        help="Lakehouse ingest config path.",
    )
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    catalog_name = config.get("catalog_name", "robot_lakehouse")
    namespace = config.get("namespace", "raw")
    tables = config.get("tables", {})

    spark = build_iceberg_spark(config, "robot-lakehouse-iceberg-inspect")
    result = {}
    for key in ["frames", "episodes", "tasks"]:
        table = f"{catalog_name}.{namespace}.{tables.get(key, key)}"
        result[table] = table_summary(spark, table)

    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main()
