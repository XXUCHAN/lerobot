from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import functions as F

from robot_dataset_platform.lakehouse.iceberg import (
    build_iceberg_spark,
    latest_snapshot_id,
    load_yaml_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect instruction annotation Iceberg table.")
    parser.add_argument(
        "--config",
        default="configs/annotation_builder.yaml",
        help="Instruction annotation builder config path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of annotation rows to print.",
    )
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    catalog_name = config.get("catalog_name", "robot_lakehouse")
    target = config["target"]
    table = f"{catalog_name}.{target['namespace']}.{target['table']}"

    spark = build_iceberg_spark(config, "robot-instruction-annotation-inspect")
    annotations = spark.table(table)
    rows = annotations.count()
    snapshot_id = latest_snapshot_id(spark, table)
    type_counts = {
        row["annotation_type"]: int(row["count"])
        for row in annotations.groupBy("annotation_type").count().collect()
    }
    per_episode = [
        {"episode_id": str(row["episode_id"]), "annotations": int(row["count"])}
        for row in annotations.groupBy("episode_id").count().orderBy("episode_id").collect()
    ]
    sample_rows = (
        annotations.select(
            "episode_id",
            "source_instruction_id",
            "instruction_id",
            "text",
            "annotation_type",
            "annotation_version",
            "annotation_policy",
            "is_active",
        )
        .orderBy("episode_id", "annotation_type", "instruction_id")
        .limit(args.limit)
        .toPandas()
        .to_dict(orient="records")
    )

    result = {
        "table": table,
        "rows": rows,
        "snapshot_id": snapshot_id,
        "annotation_type_counts": type_counts,
        "annotations_per_episode": per_episode,
        "samples": sample_rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main()
