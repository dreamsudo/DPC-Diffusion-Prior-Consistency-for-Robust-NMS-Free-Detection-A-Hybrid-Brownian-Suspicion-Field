"""Phase orchestrator.

Resolves the user's requested phase list into a dependency-ordered run
plan, executes phases in order, and writes status.json files per
(seed, phase) so reruns can resume.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from dpcctl import paths
from dpcctl.config import OrchestratorConfig
from dpcctl.events import EventWriter
from dpcctl.phases import (
    Phase,
    PhaseContext,
    PhaseStatus,
    get_phase_class,
    make_context,
    all_phase_names,
)


# Canonical phase order. If the user asks for "all", we run this list.
CANONICAL_ORDER = [
    "prep",
    "train_p1",
    "diagnose_p1",
    "train_p2",
    "diagnose_p2",
    "eval_p3",
    "eval_negative_control",
    "aggregate",
]


def resolve_phases(requested: list[str]) -> list[str]:
    """Return CANONICAL_ORDER if 'all' is requested, else filter it."""
    if not requested or requested == ["all"]:
        return list(CANONICAL_ORDER)
    return [p for p in CANONICAL_ORDER if p in requested]


def _ts(t: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts(time.time())}] {msg}", flush=True)


def run_orchestrator(
    cfg: OrchestratorConfig,
    requested_phases: list[str],
    force: bool = False,
) -> bool:
    """Execute the requested phases. Returns True if everything succeeded."""
    phases = resolve_phases(requested_phases)

    _log(f"DPC-YOLO26 orchestrator (run: '{cfg.name}')")
    _log(f"config:  {cfg.config_path}")
    _log(f"phases:  {phases}")
    _log(f"force:   {force}")
    _log(f"seeds:   {cfg.seeds}")

    paths.ensure(cfg.run_dir)
    paths.ensure(cfg.run_dir / "live")
    (cfg.run_dir / "config_resolved.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, default=str)
    )

    live = paths.live_dir(cfg.run_dir.parent, cfg.name)
    writer = EventWriter(live_dir=live, run_name=cfg.name)
    if force:
        writer.clear_all()
    writer.set_global("starting", "running", seeds=cfg.seeds, phases=phases)

    t_start = time.time()
    failures: list[str] = []

    for phase_name in phases:
        cls = get_phase_class(phase_name)

        if cls.is_shared:
            iterations = [(None, paths.shared_phase_dir(
                cfg.run_dir.parent, cfg.name, phase_name))]
            _log(f"== phase: {phase_name} (shared) ==")
        else:
            iterations = []
            for seed in cfg.seeds:
                iterations.append((seed, paths.seed_phase_dir(
                    cfg.run_dir.parent, cfg.name, seed, phase_name)))

        for seed, out_dir in iterations:
            status_path = out_dir / "status.json"
            if not force and status_path.is_file():
                try:
                    s = json.loads(status_path.read_text())
                    if s.get("status") == "complete":
                        suffix = f"(shared)" if cls.is_shared else f"(seed {seed})"
                        _log(f"   [skip] {phase_name} {suffix}: already complete")
                        continue
                except json.JSONDecodeError:
                    pass

            if not cls.is_shared:
                _log(f"== phase: {phase_name} (seed {seed}) ==")

            ctx = make_context(cfg, phase_name, seed)
            phase = cls(ctx)
            t_phase = time.time()
            try:
                phase.run()
            except Exception as e:
                elapsed = (time.time() - t_phase) / 60.0
                _log(f"   [fail] phase '{phase_name}' FAILED after {elapsed:.1f} min: {e}")
                traceback.print_exc()
                failures.append(phase_name)
                writer.set_global(phase_name, "failed", error=str(e))
                break
            elapsed = (time.time() - t_phase) / 60.0
            _log(f"   [ok]   phase '{phase_name}' complete in {elapsed:.1f} min")

        if failures:
            break

    total_min = (time.time() - t_start) / 60.0
    if failures:
        _log(f"   [fail] orchestrator finished with FAILURES after {total_min:.1f} min")
        writer.set_global("done", "failed", failures=failures)
        return False
    else:
        _log(f"   [ok]   orchestrator finished successfully in {total_min:.1f} min")
        writer.set_global("done", "complete")
        return True
