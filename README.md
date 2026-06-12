# Robot Dataset Production Platform

Robot learning dataset platform for versioned raw episodes, sensor synchronization,
manifest-based dataset construction, and reproducible LeRobot training exports.

## Architecture Diagram

Open the HTML architecture diagram:

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

## Key Components

- `lakehouse`: raw episode tables, Iceberg snapshots, time travel metadata
- `sync`: timestamp alignment, interpolation, observation/action windows
- `annotations`: language instruction and label metadata
- `manifest`: sample index generation without copying raw data
- `registry`: dataset version, lineage, stats, and reproducibility records
- `export`: LeRobot/HuggingFace dataset export
