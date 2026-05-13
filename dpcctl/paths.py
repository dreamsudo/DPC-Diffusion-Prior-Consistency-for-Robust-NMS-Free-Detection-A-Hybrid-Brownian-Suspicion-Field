"""Directory layout helpers for the orchestrator.

Run tree:

    <runs_root>/<name>/
      _shared/                  # phases that run once across all seeds
        prep/
        aggregate/
      seed_<N>/                 # per-seed phases
        train_p1/
        diagnose_p1/
        train_p2/
        diagnose_p2/
        eval_p3/
          alpha_<int(alpha*10)>/eval/
        eval_negative_control/
      live/                     # generic event-bus state files for dashboard
        state.json              # global state (active_phase, etc.)
        <phase>_state.json      # one per phase, payload-agnostic
      config_resolved.json
      run.log
"""

from __future__ import annotations

from pathlib import Path


SHARED_PHASES = {"prep", "aggregate"}


def run_dir(runs_root: Path, name: str) -> Path:
    return runs_root / name


def shared_phase_dir(runs_root: Path, name: str, phase: str) -> Path:
    return run_dir(runs_root, name) / "_shared" / phase


def seed_phase_dir(runs_root: Path, name: str, seed: int, phase: str) -> Path:
    return run_dir(runs_root, name) / f"seed_{seed}" / phase


def phase_dir(runs_root: Path, name: str, seed: int, phase: str) -> Path:
    """Route a phase to either _shared/<phase> or seed_<N>/<phase>."""
    if phase in SHARED_PHASES:
        return shared_phase_dir(runs_root, name, phase)
    return seed_phase_dir(runs_root, name, seed, phase)


def live_dir(runs_root: Path, name: str) -> Path:
    return run_dir(runs_root, name) / "live"


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
