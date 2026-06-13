from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from rich.console import Console


console = Console()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small LeRobot v3 snapshot sample.")
    parser.add_argument(
        "--config",
        default="configs/sample_snapshot.yaml",
        help="YAML config with repo_id, target_dir, and allow_patterns.",
    )
    parser.add_argument("--repo-id", help="Override Hugging Face dataset repo id.")
    parser.add_argument("--target-dir", help="Override local download directory.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    repo_id = args.repo_id or config["repo_id"]
    target_dir = Path(args.target_dir or config["target_dir"])
    allow_patterns = config.get("allow_patterns", [])

    console.print(f"[bold]Downloading sample snapshot[/bold]: {repo_id}")
    console.print(f"Target: {target_dir}")
    console.print("Patterns:")
    for pattern in allow_patterns:
        console.print(f"  - {pattern}")

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=allow_patterns,
    )

    console.print(f"[green]Downloaded to[/green] {local_dir}")


if __name__ == "__main__":
    main()

