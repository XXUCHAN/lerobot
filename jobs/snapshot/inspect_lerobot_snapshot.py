from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table


console = Console()


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def print_json_summary(path: Path) -> None:
    if not path.exists():
        console.print(f"[yellow]Missing[/yellow] {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    console.print(f"\n[bold]{path}[/bold]")
    for key in [
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "fps",
        "data_path",
        "video_path",
    ]:
        if key in data:
            console.print(f"  {key}: {data[key]}")


def print_parquet_summary(label: str, path: Path) -> None:
    files = sorted(path.rglob("*.parquet")) if path.exists() else []
    table = Table(title=label)
    table.add_column("file")
    table.add_column("rows", justify="right")
    table.add_column("size", justify="right")

    for file in files[:10]:
        meta = pq.read_metadata(file)
        table.add_row(str(file), str(meta.num_rows), human_size(file.stat().st_size))

    console.print(table)
    if len(files) > 10:
        console.print(f"... {len(files) - 10} more parquet files")


def print_video_summary(root: Path) -> None:
    files = sorted(root.rglob("*.mp4")) if root.exists() else []
    table = Table(title="Video files")
    table.add_column("file")
    table.add_column("size", justify="right")

    for file in files[:10]:
        table.add_row(str(file), human_size(file.stat().st_size))

    console.print(table)
    if len(files) > 10:
        console.print(f"... {len(files) - 10} more video files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local LeRobot v3 snapshot.")
    parser.add_argument(
        "--snapshot-dir",
        default="data/external/RobotisSW_omy_PickAndPlace_RedBlock2",
        help="Local snapshot directory.",
    )
    args = parser.parse_args()

    root = Path(args.snapshot_dir)
    console.print(f"[bold]Inspecting snapshot[/bold]: {root}")

    print_json_summary(root / "meta" / "info.json")
    print_json_summary(root / "meta" / "stats.json")
    print_parquet_summary("Episode metadata", root / "meta" / "episodes")
    print_parquet_summary("Frame data", root / "data")
    print_video_summary(root / "videos")

    data_files = sorted((root / "data").rglob("*.parquet")) if (root / "data").exists() else []
    if data_files:
        dataset = ds.dataset([str(file) for file in data_files], format="parquet")
        console.print("\n[bold]Frame data schema[/bold]")
        console.print(dataset.schema)


if __name__ == "__main__":
    main()
