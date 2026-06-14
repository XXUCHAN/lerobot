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
jobs/                             # Runnable pipeline jobs grouped by stage
jobs/snapshot/                    # HuggingFace snapshot download and inspection
jobs/lakehouse/                   # Raw parquet -> Iceberg lakehouse ingest
jobs/sync/                        # Sensor sync and observation/action windows
jobs/annotations/                 # Instruction annotation metadata layer
jobs/manifest/                    # Logical dataset manifest and validation
jobs/export/                      # LeRobot-style export and export validation
tests/                            # Unit tests and fixtures
notebooks/                        # Dataset inspection notebooks
```

## Initial MVP Flow

```text
Raw Episode
  -> Versioned Lakehouse Snapshot
  -> Spark Sensor Sync / Window Builder
  -> Instruction Annotation Layer
  -> Dataset Manifest
  -> Dataset Registry
  -> LeRobot Export
```

## Docker Compose MVP

The local MVP uses Docker Compose so Spark, Java, and Python dependencies stay
inside containers.

On Linux servers, run containers with the host user id so bind-mounted generated
data can be written by both Docker and the shell:

```bash
cp .env.example .env
sed -i "s/HOST_UID=.*/HOST_UID=$(id -u)/" .env
sed -i "s/HOST_GID=.*/HOST_GID=$(id -g)/" .env
```

If a previous container created local artifact directories with another user,
fix ownership before downloading large samples:

```bash
sudo chown -R $(id -u):$(id -g) .cache data warehouse registry 2>/dev/null || true
mkdir -p .cache data warehouse registry
```

Spark/Hadoop jobs run with `HADOOP_USER_NAME=spark` and `-Duser.name=spark` in
Docker Compose so Hadoop can resolve a stable user even when the container runs
with the host UID.

Build the environment:

```bash
docker compose build
```

Start Spark:

```bash
docker compose up -d spark-master spark-worker
```

Download the Robotis Pick & Place source snapshot:

```bash
docker compose run --rm app -lc "python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='RobotisSW/omy_PickAndPlace_RedBlock2',
    repo_type='dataset',
    local_dir='data/external/RobotisSW_omy_PickAndPlace_RedBlock2',
    allow_patterns=['README.md', 'meta/**', 'data/**', 'videos/**'],
)
PY"
```

Inspect the downloaded snapshot:

```bash
docker compose run --rm app -lc "python jobs/snapshot/inspect_lerobot_snapshot.py"
```

The current Robotis working set is LeRobot v2.1 style data:

```text
episodes: 100
frames: 47,761
tasks: 2
videos: 200
robot: aiworker
```

Ingest raw LeRobot parquet files into local Iceberg tables:

```bash
docker compose run --rm app -lc "python jobs/lakehouse/ingest_raw_to_iceberg.py"
```

This creates local Hadoop-catalog Iceberg tables under
`warehouse/robot_lakehouse`:

```text
robot_lakehouse.raw.frames
robot_lakehouse.raw.episodes
robot_lakehouse.raw.tasks
```

Inspect Iceberg row counts and snapshots:

```bash
docker compose run --rm app -lc "python jobs/lakehouse/inspect_iceberg_tables.py"
```

Build timestamp-aware synced samples from the raw Iceberg frame table:

```bash
docker compose run --rm app -lc "python jobs/sync/build_synced_samples.py"
```

This creates a validated sync table:

```text
robot_lakehouse.synced.samples
```

Each row is a logical sample with observation/action windows, timestamp
coverage, row-count checks, source snapshot id, and `sync_status`.

Inspect synced samples:

```bash
docker compose run --rm app -lc "python jobs/sync/inspect_synced_samples.py"
```

Build instruction annotations from synced samples:

```bash
docker compose run --rm app -lc "python jobs/annotations/build_instruction_annotations.py"
```

This creates a separate annotation table:

```text
robot_lakehouse.annotations.instructions
```

Each annotation row links text to an existing episode and source instruction.
Adding paraphrases creates small metadata rows only; videos and raw frame parquet
are not copied.

Inspect annotations:

```bash
docker compose run --rm app -lc "python jobs/annotations/inspect_instruction_annotations.py"
```

Build a first dataset manifest with Spark:

```bash
docker compose run --rm app -lc "python jobs/manifest/build_manifest_spark.py"
```

Build a manifest from validated synced Iceberg samples:

```bash
docker compose run --rm app -lc "python jobs/manifest/build_manifest_spark.py --config configs/manifest_from_synced.yaml"
```

Validate the end-to-end build artifacts:

```bash
docker compose run --rm app -lc "python jobs/manifest/validate_dataset_build.py"
```

The validator checks that raw Iceberg tables have snapshots, manifest source rows
pass the required `sync_status`, manifest row counts match the annotation-expanded
source view, sample ids are unique, frame windows are valid, and registry
hashes/snapshot ids match the generated artifacts.

Resolve one manifest row back to the source episode windows:

```bash
docker compose run --rm app -lc "python jobs/manifest/resolve_manifest_sample.py"
```

This proves the manifest is a logical dataset index, not a copied dataset. A
single row resolves to:

```text
instruction text
episode id
observation frame window
action frame window
episode video references
```

Export the annotation-aware synced manifest into a LeRobot-style tabular dataset:

```bash
docker compose run --rm app -lc "python jobs/export/export_lerobot_manifest.py"
```

The export materializes one logical training sample per manifest row. It copies
tabular observation/action values into a LeRobot-style directory, keeps large
source videos as references in `meta/sample_refs.jsonl`, and records manifest,
annotation, sync, and source snapshot lineage in both the export metadata and the
dataset registry.

```text
data/exports/lerobot/robotis_omy_pick_place_redblock_synced_export/
  meta/info.json
  meta/stats.json
  meta/export_lineage.json
  meta/validation_report.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  meta/sample_refs.jsonl
  data/chunk-000/file-000.parquet
```

Validate the export artifacts:

```bash
docker compose run --rm app -lc "python jobs/export/validate_lerobot_export.py"
```

The export validator checks that manifest rows, exported data rows, episode rows,
sample refs, manifest hashes, required columns, and registry records all match.

The completed Robotis MVP produces:

```text
raw.frames: 47,761
raw.episodes: 100
raw.tasks: 2
synced.samples: 1,534
synced ok samples: 1,508
annotations: 400
manifest rows: 6,032
export rows: 6,032
missing video refs: 0
```

If `validate_lerobot_export.py` reports non-zero `missing_video_ref_count`, rerun
the downloader so all referenced camera shards exist locally.

Manifest output is written as JSONL and Snappy-compressed Parquet:

```text
data/manifests/robotis_omy_pick_place_redblock_synced/manifest.jsonl
data/manifests/robotis_omy_pick_place_redblock_synced/manifest.parquet/
```

The synced manifest also writes a content-derived version:

```text
registry/datasets/robotis_omy_pick_place_redblock_synced/metadata.json
registry/datasets/robotis_omy_pick_place_redblock_synced/lineage.json
registry/datasets/robotis_omy_pick_place_redblock_synced/stats.json
registry/datasets/robotis_omy_pick_place_redblock_synced/validation_report.json
```

With the annotation layer enabled, the synced manifest expands one physical
episode window into multiple logical samples:

```text
sample = source_sample + annotation instruction
```

For example, one synced window can produce separate manifest rows for the source
instruction and its paraphrases while all rows still reference the same episode,
frame window, and Iceberg snapshots.

Generated data is ignored by Git:

```text
data/
warehouse/
registry/datasets/
registry/annotations/
.cache/
```

## Key Components

- `lakehouse`: raw episode tables, Iceberg snapshots, time travel metadata
- `sync`: timestamp alignment, interpolation, observation/action windows
- `annotations`: language instruction and label metadata
- `manifest`: sample index generation without copying raw data
- `manifest/resolver`: manifest row to source episode/frame windows
- `registry`: dataset version, lineage, stats, and reproducibility records
- `export`: manifest-based LeRobot/HuggingFace dataset export
