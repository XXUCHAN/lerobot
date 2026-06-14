from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pyspark.sql import DataFrame, SparkSession


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_iceberg_spark(config: dict[str, Any], app_name: str) -> SparkSession:
    catalog_name = config.get("catalog_name", "robot_lakehouse")
    warehouse_dir = config.get("warehouse_dir", "warehouse/robot_lakehouse")
    spark_master = config.get("spark_master", "local[*]")
    hadoop_user = config.get("hadoop_user_name", "spark")

    return (
        SparkSession.builder.appName(app_name)
        .master(spark_master)
        .config("spark.hadoop.hadoop.job.ugi", hadoop_user)
        .config("spark.driver.extraJavaOptions", f"-Duser.name={hadoop_user}")
        .config("spark.executor.extraJavaOptions", f"-Duser.name={hadoop_user}")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(f"spark.sql.catalog.{catalog_name}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog_name}.type", "hadoop")
        .config(f"spark.sql.catalog.{catalog_name}.warehouse", warehouse_dir)
        .config("spark.sql.parquet.compression.codec", config.get("parquet_compression", "snappy"))
        .getOrCreate()
    )


def write_iceberg_table(df: DataFrame, table_name: str, compression: str = "snappy") -> int:
    df.writeTo(table_name).using("iceberg").tableProperty(
        "write.parquet.compression-codec",
        compression,
    ).createOrReplace()
    return df.count()


def latest_snapshot_id(spark: SparkSession, table_name: str) -> int | None:
    try:
        rows = spark.sql(
            f"""
            SELECT snapshot_id
            FROM {table_name}.snapshots
            ORDER BY committed_at DESC
            LIMIT 1
            """
        ).collect()
    except Exception:
        return None

    if not rows:
        return None
    return int(rows[0]["snapshot_id"])
