"""Generic event bus for the live dashboard.

Phases publish state by calling EventWriter.publish(kind, payload). Each
publish() atomically writes <live_dir>/<phase>_state.json and updates
<live_dir>/state.json (the global state file the dashboard polls first).

`kind` is a free-form discriminator (e.g. "training", "diagnostic",
"eval", "aggregate"); the dashboard renders per-kind layouts but is
otherwise content-agnostic. Adding a new phase later requires no
dashboard code change as long as its `kind` is one of the rendered
ones, or it falls through to the generic key/value renderer.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically by writing to a temp file and renaming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


@dataclass
class EventWriter:
    """Publishes phase state files into a single live/ directory.

    The dashboard auto-discovers <phase>_state.json files and renders
    them based on the `kind` field in each payload.
    """

    live_dir: Path
    run_name: str

    def publish(self, phase: str, kind: str, payload: dict) -> None:
        """Write/overwrite this phase's state file.

        The dashboard polls these files at ~2 Hz. Atomic-replace ensures
        the reader never sees a half-written file.
        """
        envelope = {
            "phase": phase,
            "kind": kind,
            "run": self.run_name,
            "ts": time.time(),
            **payload,
        }
        path = self.live_dir / f"{phase}_state.json"
        atomic_write_json(path, envelope)

    def set_global(self, active_phase: str, status: str, **extra: Any) -> None:
        """Update <live_dir>/state.json — the global pointer.

        The dashboard JS reads this first to decide which tab to show.
        """
        envelope = {
            "active_phase": active_phase,
            "status": status,
            "run": self.run_name,
            "ts": time.time(),
            **extra,
        }
        atomic_write_json(self.live_dir / "state.json", envelope)

    def clear_phase(self, phase: str) -> None:
        """Remove a stale phase state file (used by --force restart)."""
        path = self.live_dir / f"{phase}_state.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def clear_all(self) -> None:
        """Remove every <phase>_state.json file plus the global state."""
        if not self.live_dir.is_dir():
            return
        for f in self.live_dir.glob("*_state.json"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        try:
            (self.live_dir / "state.json").unlink()
        except FileNotFoundError:
            pass
