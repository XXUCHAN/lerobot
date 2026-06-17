from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SNAPSHOT_DIR = Path("data/external/RobotisSW_omy_PickAndPlace_RedBlock2")
DEFAULT_MANIFEST_CONFIG = Path("configs/manifest_from_synced.yaml")
DEFAULT_MANIFEST_REGISTRY = Path("registry/datasets/robotis_omy_pick_place_redblock_synced")
DEFAULT_EXPORT_DIR = Path("data/exports/lerobot/robotis_omy_pick_place_redblock_synced_export")
DEFAULT_EXPORT_REGISTRY = DEFAULT_MANIFEST_REGISTRY / "exports" / "lerobot_manifest_export"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]
    inspect: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_step(step: PipelineStep, cwd: Path) -> None:
    print(f"\n==> {step.name}", flush=True)
    print("$ " + " ".join(step.command), flush=True)
    subprocess.run(step.command, cwd=cwd, check=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_snapshot(snapshot_dir: Path, download: bool) -> None:
    info_path = snapshot_dir / "meta" / "info.json"
    data_dir = snapshot_dir / "data"
    if info_path.exists() and data_dir.exists():
        return

    if not download:
        raise FileNotFoundError(
            f"Missing Robotis snapshot under {snapshot_dir}. "
            "Run with --download or download it before running the pipeline."
        )

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="RobotisSW/omy_PickAndPlace_RedBlock2",
        repo_type="dataset",
        local_dir=str(snapshot_dir),
        allow_patterns=["README.md", "meta/**", "data/**", "videos/**"],
    )


def pipeline_steps(py: str, manifest_config: Path, include_inspect: bool) -> list[PipelineStep]:
    steps = [
        PipelineStep("Validate source dataset contract", [py, "jobs/snapshot/validate_dataset_contract.py"]),
        PipelineStep("Inspect source snapshot", [py, "jobs/snapshot/inspect_lerobot_snapshot.py"], inspect=True),
        PipelineStep("Ingest raw data to Iceberg", [py, "jobs/lakehouse/ingest_raw_to_iceberg.py"]),
        PipelineStep("Inspect raw Iceberg tables", [py, "jobs/lakehouse/inspect_iceberg_tables.py"], inspect=True),
        PipelineStep("Build synced samples", [py, "jobs/sync/build_synced_samples.py"]),
        PipelineStep("Inspect synced samples", [py, "jobs/sync/inspect_synced_samples.py", "--limit", "3"], inspect=True),
        PipelineStep("Build instruction annotations", [py, "jobs/annotations/build_instruction_annotations.py"]),
        PipelineStep("Inspect instruction annotations", [py, "jobs/annotations/inspect_instruction_annotations.py", "--limit", "6"], inspect=True),
        PipelineStep(
            "Build annotation-aware manifest",
            [py, "jobs/manifest/build_manifest_spark.py", "--config", str(manifest_config)],
        ),
        PipelineStep("Validate dataset build", [py, "jobs/manifest/validate_dataset_build.py"]),
        PipelineStep("Resolve one manifest sample", [py, "jobs/manifest/resolve_manifest_sample.py", "--preview-records", "1"], inspect=True),
        PipelineStep("Export LeRobot-style dataset", [py, "jobs/export/export_lerobot_manifest.py"]),
        PipelineStep("Validate LeRobot export", [py, "jobs/export/validate_lerobot_export.py"]),
    ]
    if include_inspect:
        return steps
    return [step for step in steps if not step.inspect]


def print_summary(manifest_registry: Path, export_dir: Path, export_registry: Path) -> None:
    build_report = read_json(manifest_registry / "validation_report.json")
    export_report = read_json(export_dir / "meta" / "validation_report.json")
    manifest_stats = read_json(manifest_registry / "stats.json")
    export_stats = read_json(export_dir / "meta" / "stats.json")
    export_lineage = read_json(export_registry / "lineage.json")

    summary = {
        "dataset_build_status": build_report.get("status"),
        "export_status": export_report.get("status"),
        "raw_frames": build_report.get("raw", {}).get("frames", {}).get("rows"),
        "raw_episodes": build_report.get("raw", {}).get("episodes", {}).get("rows"),
        "synced_samples": build_report.get("synced", {}).get("rows"),
        "synced_ok_samples": build_report.get("synced", {}).get("required_status_rows"),
        "annotations": build_report.get("annotation", {}).get("rows"),
        "manifest_rows": manifest_stats.get("manifest_rows"),
        "export_rows": export_stats.get("export_total_samples"),
        "source_samples": export_stats.get("source_sample_count"),
        "source_episodes": export_stats.get("source_episode_count"),
        "missing_video_refs": export_stats.get("missing_video_ref_count"),
        "manifest_version": manifest_stats.get("manifest_version"),
        "export_version": export_lineage.get("export_version"),
        "export_dir": str(export_dir),
    }
    print("\n==> Pipeline summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Robotis Pick & Place MVP pipeline end-to-end.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the Robotis source snapshot if it is missing.",
    )
    parser.add_argument(
        "--skip-inspect",
        action="store_true",
        help="Skip inspect/preview steps and run only build/validation steps.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=str(DEFAULT_SNAPSHOT_DIR),
        help="Robotis source snapshot directory.",
    )
    parser.add_argument(
        "--manifest-config",
        default=str(DEFAULT_MANIFEST_CONFIG),
        help="Annotation-aware manifest builder config path.",
    )
    args = parser.parse_args()

    root = repo_root()
    snapshot_dir = Path(args.snapshot_dir)
    manifest_config = Path(args.manifest_config)

    ensure_snapshot(root / snapshot_dir, args.download)

    for step in pipeline_steps(sys.executable, manifest_config, include_inspect=not args.skip_inspect):
        run_step(step, root)

    print_summary(root / DEFAULT_MANIFEST_REGISTRY, root / DEFAULT_EXPORT_DIR, root / DEFAULT_EXPORT_REGISTRY)


if __name__ == "__main__":
    main()
