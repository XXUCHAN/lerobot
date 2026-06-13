from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_dataset_platform.manifest.resolver import ManifestResolver, read_manifest_samples


def compact_records(records: list[dict[str, Any]], max_records: int) -> list[dict[str, Any]]:
    return records[:max_records]


def compact_resolved_sample(sample: dict[str, Any], max_records: int) -> dict[str, Any]:
    observation = sample["observation"]
    action = sample["action"]
    missing_videos = [ref for ref in sample["video_refs"] if not ref["exists"]]

    return {
        "sample_id": sample["sample_id"],
        "episode_id": sample["episode_id"],
        "instruction": sample["instruction"],
        "observation": {
            "frame_range": observation["frame_range"],
            "rows": observation["rows"],
            "columns": observation["columns"],
            "preview_records": compact_records(observation["records"], max_records),
        },
        "action": {
            "frame_range": action["frame_range"],
            "rows": action["rows"],
            "columns": action["columns"],
            "preview_records": compact_records(action["records"], max_records),
        },
        "video_refs": sample["video_refs"],
        "missing_video_count": len(missing_videos),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve one manifest row into source data windows.")
    parser.add_argument(
        "--snapshot-dir",
        default="data/external/yaak-ai_L2D-v3_sample",
        help="Local LeRobot snapshot directory.",
    )
    parser.add_argument(
        "--manifest-path",
        default="data/manifests/l2d_v3_sample/manifest.jsonl",
        help="Manifest JSONL path.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Sample id to resolve. Defaults to the first manifest row.",
    )
    parser.add_argument(
        "--preview-records",
        type=int,
        default=1,
        help="Number of observation/action records to print.",
    )
    args = parser.parse_args()

    samples = read_manifest_samples(Path(args.manifest_path))
    if not samples:
        raise ValueError(f"Manifest is empty: {args.manifest_path}")

    sample = samples[0]
    if args.sample_id:
        sample = next((item for item in samples if item.sample_id == args.sample_id), None)
        if sample is None:
            raise ValueError(f"Sample id not found: {args.sample_id}")

    resolver = ManifestResolver(Path(args.snapshot_dir))
    resolved = resolver.resolve(sample)
    compact = compact_resolved_sample(resolved, args.preview_records)
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
