# Pipeline Jobs

Runnable jobs are grouped by pipeline stage so the architecture flow is visible from the file tree.

```text
snapshot/       Download and inspect source LeRobot snapshots
lakehouse/      Ingest raw parquet and metadata into Iceberg tables
sync/           Build sensor-synced observation/action window samples
annotations/    Build and inspect instruction annotation metadata
manifest/       Build, resolve, and validate logical dataset manifests
export/         Export and validate LeRobot-style training datasets
pipeline/       Run the Robotis MVP pipeline end-to-end
```

Run jobs from the repository root through Docker Compose, for example:

```bash
docker compose run --rm app -lc "python jobs/lakehouse/ingest_raw_to_iceberg.py"
```

Or run the complete Robotis MVP pipeline:

```bash
docker compose run --rm app -lc "python jobs/pipeline/run_robotis_mvp.py"
```
