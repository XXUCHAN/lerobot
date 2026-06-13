from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_dataset_platform.export.lerobot_exporter import LeRobotManifestExporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manifest samples into a LeRobot-style dataset.")
    parser.add_argument(
        "--snapshot-dir",
        default="data/external/yaak-ai_L2D-v3_sample",
        help="Local LeRobot source snapshot directory.",
    )
    parser.add_argument(
        "--manifest-path",
        default="data/manifests/l2d_v3_sample/manifest.jsonl",
        help="Manifest JSONL path.",
    )
    parser.add_argument(
        "--export-dir",
        default="data/exports/lerobot/l2d_v3_manifest_export",
        help="Output LeRobot-style dataset directory.",
    )
    args = parser.parse_args()

    exporter = LeRobotManifestExporter(
        snapshot_dir=Path(args.snapshot_dir),
        manifest_path=Path(args.manifest_path),
        export_dir=Path(args.export_dir),
    )
    result = exporter.export()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
