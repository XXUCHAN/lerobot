from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

from robot_dataset_platform.lakehouse.iceberg import (
    build_iceberg_spark,
    latest_snapshot_id,
    load_yaml_config,
    write_iceberg_table,
)


def full_table_name(catalog: str, namespace: str, table: str) -> str:
    return f"{catalog}.{namespace}.{table}"


def stable_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_annotations(source: DataFrame, config: dict[str, Any]) -> DataFrame:
    return (
        source.select(
            F.col("episode_id").cast("string").alias("episode_id"),
            F.col("instruction_id").cast("string").alias("source_instruction_id"),
            F.col("instruction_text").cast("string").alias("source_instruction_text"),
        )
        .where(F.col("source_instruction_text").isNotNull())
        .dropDuplicates(["episode_id", "source_instruction_id", "source_instruction_text"])
        .withColumn("text", F.col("source_instruction_text"))
        .withColumn("annotation_type", F.lit(config.get("source_annotation_type", "source_task")))
        .withColumn("task_label", F.lit(config.get("default_task_label")).cast("string"))
        .withColumn("object_label", F.lit(None).cast("string"))
        .withColumn("scene_label", F.lit(None).cast("string"))
        .withColumn("is_active", F.lit(True))
    )


def additional_annotation_rules(spark: SparkSession, config: dict[str, Any]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for rule in config.get("additional_annotations", []):
        match_text = rule["match_text"]
        annotation_type = rule.get("annotation_type", "paraphrase")
        for text in rule.get("texts", []):
            rows.append(
                {
                    "match_text": match_text,
                    "text": text,
                    "annotation_type": annotation_type,
                    "task_label": rule.get("task_label", config.get("default_task_label")),
                    "object_label": rule.get("object_label"),
                    "scene_label": rule.get("scene_label"),
                    "is_active": bool(rule.get("is_active", True)),
                }
            )

    schema = StructType(
        [
            StructField("match_text", StringType(), False),
            StructField("text", StringType(), False),
            StructField("annotation_type", StringType(), False),
            StructField("task_label", StringType(), True),
            StructField("object_label", StringType(), True),
            StructField("scene_label", StringType(), True),
            StructField("is_active", BooleanType(), False),
        ]
    )
    return spark.createDataFrame(rows, schema=schema)


def build_additional_annotations(
    spark: SparkSession,
    source_annotations: DataFrame,
    config: dict[str, Any],
) -> DataFrame:
    rules = additional_annotation_rules(spark, config)
    if not rules.head(1):
        return spark.createDataFrame([], source_annotations.schema)

    return (
        source_annotations.alias("source")
        .join(
            rules.alias("rule"),
            F.col("source.source_instruction_text") == F.col("rule.match_text"),
            "inner",
        )
        .select(
            F.col("source.episode_id"),
            F.col("source.source_instruction_id"),
            F.col("source.source_instruction_text"),
            F.col("rule.text"),
            F.col("rule.annotation_type"),
            F.col("rule.task_label"),
            F.col("rule.object_label"),
            F.col("rule.scene_label"),
            F.col("rule.is_active"),
        )
    )


def finalize_annotations(annotations: DataFrame, config: dict[str, Any]) -> DataFrame:
    annotation_version = config["annotation_version"]
    annotation_policy = config["annotation_policy"]
    language = config.get("language", "en")
    created_by = config.get("created_by", "annotation_builder")

    id_seed = F.concat_ws(
        "||",
        F.col("episode_id"),
        F.col("source_instruction_id"),
        F.col("text"),
        F.col("annotation_type"),
        F.lit(annotation_version),
        F.lit(annotation_policy),
    )

    return (
        annotations.withColumn("annotation_version", F.lit(annotation_version))
        .withColumn("annotation_policy", F.lit(annotation_policy))
        .withColumn("language", F.lit(language))
        .withColumn("created_by", F.lit(created_by))
        .withColumn("created_at_utc", F.lit(datetime.now(timezone.utc).isoformat()))
        .withColumn("is_active", F.coalesce(F.col("is_active"), F.lit(True)))
        .withColumn("instruction_id", F.concat(F.lit("inst_"), F.substring(F.sha2(id_seed, 256), 1, 16)))
        .select(
            "instruction_id",
            "episode_id",
            "source_instruction_id",
            "source_instruction_text",
            "text",
            "annotation_type",
            "annotation_version",
            "annotation_policy",
            "language",
            "task_label",
            "object_label",
            "scene_label",
            "created_by",
            "created_at_utc",
            "is_active",
        )
        .dropDuplicates(["episode_id", "instruction_id"])
    )


def annotation_type_counts(df: DataFrame) -> dict[str, int]:
    return {
        row["annotation_type"]: int(row["count"])
        for row in df.groupBy("annotation_type").count().collect()
    }


def annotations_per_episode(df: DataFrame) -> list[dict[str, Any]]:
    return [
        {"episode_id": str(row["episode_id"]), "annotations": int(row["count"])}
        for row in df.groupBy("episode_id").count().orderBy("episode_id").collect()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the instruction annotation Iceberg table from synced samples."
    )
    parser.add_argument(
        "--config",
        default="configs/annotation_builder.yaml",
        help="Instruction annotation builder config path.",
    )
    args = parser.parse_args()

    config = load_yaml_config(Path(args.config))
    catalog_name = config.get("catalog_name", "robot_lakehouse")
    source_table = config["source_table"]
    target = config["target"]
    target_table = full_table_name(catalog_name, target["namespace"], target["table"])
    registry_dir = Path(config["registry_dir"])
    parquet_compression = config.get("parquet_compression", "snappy")

    registry_dir.mkdir(parents=True, exist_ok=True)

    spark = build_iceberg_spark(config, "robot-instruction-annotation-builder")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog_name}.{target['namespace']}")

    source = spark.table(source_table)
    if config.get("require_sync_status"):
        source = source.where(F.col("sync_status") == F.lit(config["require_sync_status"]))

    source_snapshot_id = latest_snapshot_id(spark, source_table)
    source_annotations = build_source_annotations(source, config)
    additional_annotations = build_additional_annotations(spark, source_annotations, config)
    annotations = finalize_annotations(
        source_annotations.unionByName(additional_annotations),
        config,
    ).cache()

    annotation_count = write_iceberg_table(annotations, target_table, parquet_compression)
    target_snapshot_id = latest_snapshot_id(spark, target_table)
    type_counts = annotation_type_counts(annotations)
    per_episode = annotations_per_episode(annotations)

    lineage = {
        "dataset_name": config["dataset_name"],
        "annotation_config_sha256": stable_config_hash(config),
        "source_table": source_table,
        "source_snapshot_id": source_snapshot_id,
        "target_table": target_table,
        "target_snapshot_id": target_snapshot_id,
        "annotation_version": config["annotation_version"],
        "annotation_policy": config["annotation_policy"],
        "require_sync_status": config.get("require_sync_status"),
    }
    (registry_dir / "annotation_lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats = {
        "annotations": annotation_count,
        "annotation_type_counts": type_counts,
        "annotations_per_episode": per_episode,
    }
    (registry_dir / "annotation_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result = {
        "target_table": target_table,
        "rows": annotation_count,
        "target_snapshot_id": target_snapshot_id,
        "annotation_type_counts": type_counts,
        "annotations_per_episode": per_episode,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    annotations.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
