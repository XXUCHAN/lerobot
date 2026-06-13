# Robot Dataset Production Platform

Robot learning dataset platform for versioned raw episodes, sensor synchronization,
manifest-based dataset construction, and reproducible LeRobot training exports.

## Architecture Diagram

![Robot Dataset Production Platform Architecture](docs/architecture.png)

Open the HTML version:

```text
docs/architecture.html
```

The diagram summarizes the platform as four layers:

```text
Robot / Edge Source
  -> Versioned Raw Data Lakehouse
  -> Spark Sensor Sync / Manifest Builder
  -> Dataset Registry / Training
```

It highlights four production bottlenecks:

- raw data lineage is hard to trace
- camera, joint, and action streams are not naturally synchronized
- adding language instructions can duplicate large video datasets
- model training is hard to reproduce without fixed dataset versions

## Core Idea

Raw robot data is stored once as immutable episodes. Training datasets are built as
logical views through dataset manifests:

```text
sample = episode + instruction + frame window
```

This avoids copying large video/sensor files whenever instructions, sync rules, or
dataset versions change.

## Project Structure

```text
configs/                         # Lakehouse, sync, builder, export configs
warehouse/robot_lakehouse/        # Local Iceberg/lakehouse warehouse
data/raw/episodes/                # Raw robot episode files
data/annotations/                 # Language instructions and labels
data/manifests/                   # Dataset manifest JSONL files
data/exports/lerobot/             # LeRobot/HuggingFace export output
registry/datasets/                # Dataset metadata, lineage, stats, versions
src/robot_dataset_platform/       # Platform source modules
jobs/                             # Runnable ingestion/build/export jobs
tests/                            # Unit tests and fixtures
notebooks/                        # Dataset inspection notebooks
```

## Initial MVP Flow

```text
Raw Episode
  -> Versioned Lakehouse Snapshot
  -> Spark Sensor Sync / Window Builder
  -> Dataset Manifest
  -> Dataset Registry
  -> LeRobot Export
```

## Docker Compose MVP

The local MVP uses Docker Compose so Spark, Java, and Python dependencies stay
inside containers.

Build the environment:

```bash
docker compose build
```

Start Spark:

```bash
docker compose up -d spark-master spark-worker
```

Download a LeRobot v3 sample snapshot instead of the full dataset. The full
`yaak-ai/L2D-v3` dataset is about 556GB, so the MVP downloads an
episode-aligned sample under 10GB. It keeps every camera modality for episodes
0 and 1, plus the frame parquet and metadata needed to inspect the dataset:

```bash
docker compose run --rm app -lc "python jobs/download_lerobot_sample.py"
```

Inspect the downloaded snapshot:

```bash
docker compose run --rm app -lc "python jobs/inspect_lerobot_snapshot.py"
```

Build a first dataset manifest with Spark:

```bash
docker compose run --rm app -lc "spark-submit --master spark://spark-master:7077 jobs/build_manifest_spark.py"
```

The sample downloader currently fetches about 6.75 GiB. The selected video
shards are the files referenced by episodes 0 and 1 in `meta/episodes`, so the
manifest does not point at missing camera files:

```text
README.md
meta/**
data/chunk-000/file-000.parquet
data/chunk-000/file-001.parquet
data/chunk-000/file-002.parquet
data/chunk-000/file-003.parquet
videos/observation.images.front_left/chunk-000/file-024.mp4
videos/observation.images.front_left/chunk-000/file-092.mp4
videos/observation.images.left_backward/chunk-000/file-021.mp4
videos/observation.images.left_backward/chunk-000/file-074.mp4
videos/observation.images.left_forward/chunk-000/file-025.mp4
videos/observation.images.left_forward/chunk-000/file-084.mp4
videos/observation.images.map/chunk-000/file-000.mp4
videos/observation.images.map/chunk-000/file-002.mp4
videos/observation.images.rear/chunk-000/file-020.mp4
videos/observation.images.rear/chunk-000/file-071.mp4
videos/observation.images.right_backward/chunk-000/file-024.mp4
videos/observation.images.right_backward/chunk-000/file-081.mp4
videos/observation.images.right_forward/chunk-000/file-025.mp4
videos/observation.images.right_forward/chunk-000/file-087.mp4
```

Manifest output is written as JSONL and Snappy-compressed Parquet:

```text
data/manifests/l2d_v3_sample/manifest.jsonl
data/manifests/l2d_v3_sample/manifest.parquet/
```

Generated data is ignored by Git:

```text
data/
warehouse/
registry/datasets/
.cache/
```

## Key Components

- `lakehouse`: raw episode tables, Iceberg snapshots, time travel metadata
- `sync`: timestamp alignment, interpolation, observation/action windows
- `annotations`: language instruction and label metadata
- `manifest`: sample index generation without copying raw data
- `registry`: dataset version, lineage, stats, and reproducibility records
- `export`: LeRobot/HuggingFace dataset export
