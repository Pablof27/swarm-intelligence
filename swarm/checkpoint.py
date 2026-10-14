"""Checkpoints for the command-line interface.

A checkpoint is a directory with:

- ``checkpoint.json``: the settings that determine the results, checked when resuming;
- ``results.jsonl``: one line per finished run, appended as soon as the run ends;
- ``runs/<instance>.pkl``: the latest snapshot of every unfinished run (pickle; only load your own).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
from pathlib import Path
from typing import Any

from swarm.io import ResultInfo, SolutionInfo

FORMAT_VERSION = 1


class CheckpointMismatch(Exception):
    """The checkpoint directory belongs to a run with different settings."""


class Checkpoint:
    def __init__(self, directory: Path):
        self.directory = directory
        self.settings_file = directory / "checkpoint.json"
        self.results_file = directory / "results.jsonl"
        self.runs_dir = directory / "runs"

    @staticmethod
    def stored_settings(directory: Path) -> dict[str, Any] | None:
        settings_file = directory / "checkpoint.json"
        if not settings_file.exists():
            return None
        data = json.loads(settings_file.read_text())
        if data.get("version") != FORMAT_VERSION:
            raise CheckpointMismatch(f"{directory} was written by an incompatible version of this program")
        return data["settings"]

    @classmethod
    def open(cls, directory: Path, settings: dict[str, Any]) -> Checkpoint:
        """Create the checkpoint, or reopen it if its stored settings match ``settings``."""
        checkpoint = cls(directory)
        stored = cls.stored_settings(directory)
        if stored is None:
            checkpoint.runs_dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(
                checkpoint.settings_file,
                json.dumps({"version": FORMAT_VERSION, "settings": settings}, indent=2).encode(),
            )
            return checkpoint
        differences = [f"{key}: {stored.get(key)!r} → {value!r}" for key, value in settings.items() if stored.get(key) != value]
        if differences:
            raise CheckpointMismatch(
                f"{directory} was created with different settings ({', '.join(differences)}); "
                "use the same options to resume it or choose another checkpoint directory"
            )
        checkpoint.runs_dir.mkdir(exist_ok=True)
        return checkpoint

    def finished_runs(self) -> dict[str, dict[str, Any]]:
        """Outcomes of the finished runs by instance name."""
        runs: dict[str, dict[str, Any]] = {}
        if self.results_file.exists():
            for line in self.results_file.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by a crash
                result = record.pop("result")
                record["result"] = ResultInfo(**{**result, "best": SolutionInfo(**result["best"])})
                runs[record.pop("name")] = record
        return runs

    def record(self, name: str, outcome: dict[str, Any]) -> None:
        """Durably store a finished run, then drop its snapshot."""
        record = {"name": name, **outcome, "result": dataclasses.asdict(outcome["result"])}
        with open(self.results_file, "a") as file:
            file.write(json.dumps(record) + "\n")
            file.flush()
            os.fsync(file.fileno())
        self.snapshot_path(name).unlink(missing_ok=True)

    def snapshot_path(self, name: str) -> Path:
        return self.runs_dir / f"{name}.pkl"


def save_snapshot(path: str | Path, state: dict[str, Any]) -> None:
    _write_atomic(Path(path), pickle.dumps({"version": FORMAT_VERSION, **state}, protocol=pickle.HIGHEST_PROTOCOL))


def load_snapshot(path: str | Path) -> dict[str, Any] | None:
    """The saved state, or None when there is no usable snapshot."""
    try:
        state = pickle.loads(Path(path).read_bytes())
    except FileNotFoundError:
        return None
    except Exception:  # truncated, or written by an incompatible version of the code
        return None
    return state if isinstance(state, dict) and state.get("version") == FORMAT_VERSION else None


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
