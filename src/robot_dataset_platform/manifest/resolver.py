from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    episode_id: str
    instruction_id: str
    instruction_text: str | None
    anchor_frame: int
    observation_start_frame: int
    observation_end_frame: int
    action_start_frame: int
    action_end_frame: int

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ManifestSample":
        return cls(
            sample_id=str(row["sample_id"]),
            episode_id=str(row["episode_id"]),
            instruction_id=str(row["instruction_id"]),
            instruction_text=row.get("instruction_text"),
            anchor_frame=int(row["anchor_frame"]),
            observation_start_frame=int(row["observation_start_frame"]),
            observation_end_frame=int(row["observation_end_frame"]),
            action_start_frame=int(row["action_start_frame"]),
            action_end_frame=int(row["action_end_frame"]),
        )


class ManifestResolver:
    def __init__(self, snapshot_dir: Path | str):
        self.snapshot_dir = Path(snapshot_dir)
        self._frame_dataset: ds.Dataset | None = None
        self._episode_table: pa.Table | None = None

    @property
    def frame_dataset(self) -> ds.Dataset:
        if self._frame_dataset is None:
            files = sorted((self.snapshot_dir / "data").rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"Missing frame parquet files under {self.snapshot_dir / 'data'}")
            self._frame_dataset = ds.dataset([str(file) for file in files], format="parquet")
        return self._frame_dataset

    @property
    def episode_table(self) -> pa.Table:
        if self._episode_table is None:
            files = sorted((self.snapshot_dir / "meta" / "episodes").rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"Missing episode metadata parquet files under {self.snapshot_dir / 'meta' / 'episodes'}"
                )
            self._episode_table = pq.read_table(files)
        return self._episode_table

    def resolve(self, sample: ManifestSample) -> dict[str, Any]:
        observation = self._project_observation(
            self._read_frame_window(
                sample.episode_id,
                sample.observation_start_frame,
                sample.observation_end_frame,
            )
        )
        action = self._project_action(
            self._read_frame_window(
                sample.episode_id,
                sample.action_start_frame,
                sample.action_end_frame,
            )
        )
        video_refs = self._episode_video_refs(sample.episode_id)

        return {
            "sample_id": sample.sample_id,
            "episode_id": sample.episode_id,
            "instruction": {
                "instruction_id": sample.instruction_id,
                "text": sample.instruction_text,
            },
            "observation": {
                "frame_range": [
                    sample.observation_start_frame,
                    sample.observation_end_frame,
                ],
                "rows": observation.num_rows,
                "columns": observation.column_names,
                "records": observation.to_pylist(),
            },
            "action": {
                "frame_range": [
                    sample.action_start_frame,
                    sample.action_end_frame,
                ],
                "rows": action.num_rows,
                "columns": action.column_names,
                "records": action.to_pylist(),
            },
            "video_refs": video_refs,
        }

    def resolve_raw(self, sample: ManifestSample) -> dict[str, Any]:
        observation = self._read_frame_window(
            sample.episode_id,
            sample.observation_start_frame,
            sample.observation_end_frame,
        )
        action = self._read_frame_window(
            sample.episode_id,
            sample.action_start_frame,
            sample.action_end_frame,
        )
        video_refs = self._episode_video_refs(sample.episode_id)

        return {
            "sample_id": sample.sample_id,
            "episode_id": sample.episode_id,
            "instruction": {
                "instruction_id": sample.instruction_id,
                "text": sample.instruction_text,
            },
            "observation": {
                "frame_range": [
                    sample.observation_start_frame,
                    sample.observation_end_frame,
                ],
                "rows": observation.num_rows,
                "columns": observation.column_names,
                "records": observation.to_pylist(),
            },
            "action": {
                "frame_range": [
                    sample.action_start_frame,
                    sample.action_end_frame,
                ],
                "rows": action.num_rows,
                "columns": action.column_names,
                "records": action.to_pylist(),
            },
            "video_refs": video_refs,
        }

    @staticmethod
    def _project_observation(table: pa.Table) -> pa.Table:
        return table.select(
            [
                name
                for name in table.column_names
                if name in {"episode_index", "frame_index", "timestamp"}
                or name.startswith("observation.")
            ]
        )

    @staticmethod
    def _project_action(table: pa.Table) -> pa.Table:
        return table.select(
            [
                name
                for name in table.column_names
                if name in {"episode_index", "frame_index", "timestamp"}
                or name.startswith("action.")
            ]
        )

    def _read_frame_window(self, episode_id: str, start_frame: int, end_frame: int) -> pa.Table:
        columns = self.frame_dataset.schema.names
        episode_col = self._pick_column(columns, ["episode_index", "episode_id", "episode"])
        frame_col = self._pick_column(columns, ["frame_index", "index"])
        if episode_col is None or frame_col is None:
            raise ValueError("Frame data must include episode and frame columns.")

        expression = (
            (ds.field(episode_col) == self._coerce_episode_id(episode_id))
            & (ds.field(frame_col) >= start_frame)
            & (ds.field(frame_col) <= end_frame)
        )
        return self.frame_dataset.to_table(filter=expression)

    def _episode_video_refs(self, episode_id: str) -> list[dict[str, Any]]:
        episode_col = self._pick_column(self.episode_table.column_names, ["episode_index", "episode_id", "episode"])
        if episode_col is None:
            return []

        mask = pc.equal(self.episode_table[episode_col], self._coerce_episode_id(episode_id))
        rows = self.episode_table.filter(mask).to_pylist()
        if not rows:
            return []

        row = rows[0]
        refs: list[dict[str, Any]] = []
        for name in self.episode_table.column_names:
            prefix = "videos/"
            suffix = "/file_index"
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue

            video_key = name[len(prefix) : -len(suffix)]
            chunk_key = f"videos/{video_key}/chunk_index"
            file_index = row.get(name)
            chunk_index = row.get(chunk_key)
            if file_index is None or chunk_index is None:
                continue

            relative_path = (
                f"videos/{video_key}/chunk-{int(chunk_index):03d}/file-{int(file_index):03d}.mp4"
            )
            local_path = self.snapshot_dir / relative_path
            refs.append(
                {
                    "video_key": video_key,
                    "path": relative_path,
                    "exists": local_path.exists(),
                }
            )

        return refs

    @staticmethod
    def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    @staticmethod
    def _coerce_episode_id(episode_id: str) -> int | str:
        try:
            return int(episode_id)
        except ValueError:
            return episode_id


def read_manifest_samples(path: Path | str) -> list[ManifestSample]:
    manifest_path = Path(path)
    samples: list[ManifestSample] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(ManifestSample.from_dict(json.loads(line)))
    return samples
