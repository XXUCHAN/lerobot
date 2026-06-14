from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, snapshot_download
from rich.console import Console
from rich.table import Table


console = Console()


@dataclass(frozen=True)
class DownloadPlan:
    repo_id: str
    target_dir: Path
    allow_patterns: list[str]
    selected_episodes: list[int]
    selected_video_patterns: list[str]
    estimated_bytes: int | None
    missing_size_patterns: list[str]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def download_metadata(repo_id: str, target_dir: Path, metadata_patterns: list[str]) -> Path:
    console.print("[bold]Step 1/2: downloading metadata[/bold]")
    ensure_writable_directory(target_dir)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        ensure_writable_directory(Path(hf_home))
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=metadata_patterns,
    )
    return Path(local_dir)


def ensure_writable_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to {path}. On Linux bind mounts, run containers with the host UID/GID, "
            "for example: HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm app -lc \"python jobs/snapshot/download_lerobot_sample.py\""
        ) from exc


def read_episode_table(snapshot_dir: Path) -> pa.Table:
    episode_files = sorted((snapshot_dir / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"Missing episode metadata under {snapshot_dir / 'meta' / 'episodes'}")
    return pq.read_table(episode_files)


def build_episode_filter(config: dict[str, Any], available_episodes: list[int]) -> list[int]:
    explicit_filter = config.get("episode_filter")
    if explicit_filter:
        return [int(episode) for episode in explicit_filter]

    start = int(config.get("episode_filter_start", 0))
    count = int(config.get("episode_limit", config.get("episode_filter_count", 0)))
    if count <= 0:
        raise ValueError("Set episode_limit or episode_filter_count to build an episode-aligned sample.")

    selected = [episode for episode in available_episodes if episode >= start]
    return selected[:count]


def filter_episode_rows(episodes: pa.Table, selected_episode_ids: list[int]) -> list[dict[str, Any]]:
    episode_col = pick_column(episodes.column_names, ["episode_index", "episode_id", "episode"])
    if episode_col is None:
        raise ValueError("Episode metadata must contain episode_index, episode_id, or episode.")

    mask = pc.is_in(episodes[episode_col], value_set=pa.array(selected_episode_ids))
    return episodes.filter(mask).to_pylist()


def available_episode_ids(episodes: pa.Table) -> list[int]:
    episode_col = pick_column(episodes.column_names, ["episode_index", "episode_id", "episode"])
    if episode_col is None:
        raise ValueError("Episode metadata must contain episode_index, episode_id, or episode.")
    return sorted({int(value.as_py()) for value in episodes[episode_col] if value.as_py() is not None})


def extract_video_patterns(
    episode_rows: list[dict[str, Any]],
    episode_columns: list[str],
    video_keys: list[str] | None,
) -> list[str]:
    patterns: set[str] = set()
    prefix = "videos/"
    suffix = "/file_index"

    for name in episode_columns:
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        video_key = name[len(prefix) : -len(suffix)]
        if video_keys and video_key not in video_keys:
            continue

        chunk_key = f"videos/{video_key}/chunk_index"
        for row in episode_rows:
            file_index = row.get(name)
            chunk_index = row.get(chunk_key)
            if file_index is None or chunk_index is None:
                continue
            patterns.add(f"videos/{video_key}/chunk-{int(chunk_index):03d}/file-{int(file_index):03d}.mp4")

    return sorted(patterns)


def repo_file_sizes(repo_id: str) -> dict[str, int]:
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    sizes: dict[str, int] = {}
    for sibling in info.siblings:
        path = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if path and size is not None:
            sizes[path] = int(size)
    return sizes


def estimate_pattern_bytes(patterns: list[str], sizes: dict[str, int]) -> tuple[int | None, list[str]]:
    total = 0
    missing: list[str] = []
    for pattern in patterns:
        if "*" in pattern or pattern.endswith("/**"):
            prefix = pattern[:-3] if pattern.endswith("/**") else pattern.split("*")[0]
            matches = [size for path, size in sizes.items() if path.startswith(prefix)]
            if matches:
                total += sum(matches)
            else:
                missing.append(pattern)
            continue
        if pattern in sizes:
            total += sizes[pattern]
        else:
            missing.append(pattern)
    return (None if missing else total), missing


def build_download_plan(
    repo_id: str,
    target_dir: Path,
    config: dict[str, Any],
    file_sizes: dict[str, int],
) -> DownloadPlan:
    episodes = read_episode_table(target_dir)
    selected_episodes = build_episode_filter(config, available_episode_ids(episodes))
    episode_rows = filter_episode_rows(episodes, selected_episodes)

    if not episode_rows:
        raise ValueError(f"No episode metadata rows matched: {selected_episodes}")

    video_keys = None if config.get("include_all_camera_views", True) else config.get("video_keys", [])
    video_patterns = extract_video_patterns(episode_rows, episodes.column_names, video_keys)

    base_patterns = list(config.get("base_patterns", ["README.md", "meta/**", "data/**"]))
    allow_patterns = sorted(dict.fromkeys([*base_patterns, *video_patterns]))
    estimated_bytes, missing_size_patterns = estimate_pattern_bytes(allow_patterns, file_sizes)

    max_download_gb = config.get("max_download_gb")
    if estimated_bytes is not None and max_download_gb is not None:
        max_bytes = int(float(max_download_gb) * 1024**3)
        if estimated_bytes > max_bytes:
            raise ValueError(
                f"Download plan is {human_bytes(estimated_bytes)}, which exceeds max_download_gb={max_download_gb}. "
                "Lower episode_limit or choose fewer video_keys."
            )

    return DownloadPlan(
        repo_id=repo_id,
        target_dir=target_dir,
        allow_patterns=allow_patterns,
        selected_episodes=selected_episodes,
        selected_video_patterns=video_patterns,
        estimated_bytes=estimated_bytes,
        missing_size_patterns=missing_size_patterns,
    )


def print_plan(plan: DownloadPlan) -> None:
    table = Table(title="Episode-aligned download plan")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("repo", plan.repo_id)
    table.add_row("target", str(plan.target_dir))
    table.add_row("episodes", f"{plan.selected_episodes[0]}..{plan.selected_episodes[-1]} ({len(plan.selected_episodes)})")
    table.add_row("video shards", str(len(plan.selected_video_patterns)))
    table.add_row("allow patterns", str(len(plan.allow_patterns)))
    table.add_row("estimated size", human_bytes(plan.estimated_bytes))
    table.add_row("patterns without size", str(len(plan.missing_size_patterns)))
    console.print(table)

    console.print("[bold]Patterns:[/bold]")
    for pattern in plan.allow_patterns:
        console.print(f"  - {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an episode-aligned LeRobot v3 snapshot sample.")
    parser.add_argument(
        "--config",
        default="configs/sample_snapshot.yaml",
        help="YAML config with repo_id, target_dir, episode_limit, and download limits.",
    )
    parser.add_argument("--repo-id", help="Override Hugging Face dataset repo id.")
    parser.add_argument("--target-dir", help="Override local download directory.")
    parser.add_argument("--episode-limit", type=int, help="Override number of episodes to include.")
    parser.add_argument("--max-download-gb", type=float, help="Override max planned download size in GiB.")
    parser.add_argument("--dry-run", action="store_true", help="Build and print the download plan only.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.episode_limit is not None:
        config["episode_limit"] = args.episode_limit
    if args.max_download_gb is not None:
        config["max_download_gb"] = args.max_download_gb

    repo_id = args.repo_id or config["repo_id"]
    target_dir = Path(args.target_dir or config["target_dir"])
    metadata_patterns = list(config.get("metadata_patterns", ["README.md", "meta/**"]))

    console.print(f"[bold]Downloading sample snapshot[/bold]: {repo_id}")
    console.print(f"Target: {target_dir}")

    download_metadata(repo_id, target_dir, metadata_patterns)

    try:
        file_sizes = repo_file_sizes(repo_id)
    except Exception as exc:  # pragma: no cover - depends on HF API availability.
        console.print(f"[yellow]Could not fetch remote file sizes:[/yellow] {exc}")
        file_sizes = {}

    plan = build_download_plan(repo_id, target_dir, config, file_sizes)
    print_plan(plan)

    if args.dry_run:
        console.print("[yellow]Dry run enabled; skipping data/video download.[/yellow]")
        return

    console.print("[bold]Step 2/2: downloading selected frame/video shards[/bold]")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=plan.allow_patterns,
    )

    console.print(f"[green]Downloaded to[/green] {local_dir}")


if __name__ == "__main__":
    main()
