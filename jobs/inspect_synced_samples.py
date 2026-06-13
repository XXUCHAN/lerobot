from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_dataset_platform.lakehouse.iceberg import (
    build_iceberg_spark,
    latest_snapshot_id,
    load_yaml_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect synced Iceberg sample table.")
    parser.add_argument(
        "--config",
        default="configs/sync_builder.yaml",
        help="Sensor sync builder config path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of sample rows to print.",
    )
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    catalog_name = config.get("catalog_name", "robot_lakehouse")
    target = config["target"]
    table = f"{catalog_name}.{target['namespace']}.{target['samples_table']}"

    spark = build_iceberg_spark(config, "robot-synced-samples-inspect")
    rows = spark.sql(f"SELECT COUNT(*) AS rows FROM {table}").collect()[0]["rows"]
    snapshot_id = latest_snapshot_id(spark, table)
    status_counts = {
        row["sync_status"]: int(row["count"])
        for row in spark.table(table).groupBy("sync_status").count().collect()
    }
    sample_rows = (
        spark.table(table)
        .select(
            "sample_id",
            "episode_id",
            "instruction_id",
            "anchor_frame",
            "anchor_timestamp",
            "observation_start_frame",
            "observation_end_frame",
            "action_start_frame",
            "action_end_frame",
            "observation_rows",
            "action_rows",
            "max_timestamp_gap_seconds",
            "sync_status",
        )
        .orderBy("episode_id", "anchor_frame")
        .limit(args.limit)
        .toPandas()
        .to_dict(orient="records")
    )

    result = {
        "table": table,
        "rows": rows,
        "snapshot_id": snapshot_id,
        "sync_status_counts": status_counts,
        "samples": sample_rows,
    }
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main()
