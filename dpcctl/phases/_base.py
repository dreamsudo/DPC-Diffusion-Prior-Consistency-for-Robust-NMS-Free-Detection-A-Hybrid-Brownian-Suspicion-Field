"""Base classes for phase modules.

Each concrete phase subclasses `Phase` and implements `run()`. The
orchestrator creates a `PhaseContext` per (seed, phase) pair, instantiates
the phase, and calls `run()`.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dpcctl import paths
from dpcctl.config import OrchestratorConfig
from dpcctl.events import EventWriter, atomic_write_json


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


# Registered phase classes by name. Populated when each phase module is
# imported. The orchestrator looks up entries here.
_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: register a Phase subclass under `name`."""
    def deco(cls):
        cls.phase_name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_phase_class(name: str):
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown phase '{name}'. registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def all_phase_names() -> list[str]:
    return list(_REGISTRY.keys())


@dataclass
class PhaseContext:
    """Per-phase execution context.

    Holds paths, the seed (or None for shared phases), the resolved
    config, and the event writer the phase uses to publish state.
    """

    cfg: OrchestratorConfig
    phase_name: str
    seed: Optional[int]
    out_dir: Path
    live_dir: Path
    events: EventWriter

    @property
    def is_shared(self) -> bool:
        return self.seed is None


def make_context(
    cfg: OrchestratorConfig,
    phase_name: str,
    seed: Optional[int],
) -> PhaseContext:
    runs_root = cfg.run_dir.parent
    if seed is None:
        out_dir = paths.shared_phase_dir(runs_root, cfg.name, phase_name)
    else:
        out_dir = paths.seed_phase_dir(runs_root, cfg.name, seed, phase_name)
    paths.ensure(out_dir)

    live = paths.live_dir(runs_root, cfg.name)
    paths.ensure(live)

    events = EventWriter(live_dir=live, run_name=cfg.name)
    return PhaseContext(
        cfg=cfg,
        phase_name=phase_name,
        seed=seed,
        out_dir=out_dir,
        live_dir=live,
        events=events,
    )


class Phase(ABC):
    """Abstract base for all pipeline phases.

    Each subclass must:
      - Declare `phase_name` via `@register("...")` decoration
      - Declare `is_shared` (True if it runs once across all seeds)
      - Declare `depends_on` (list of phase names that must complete first)
      - Implement `run(ctx)`
    """

    phase_name: str = ""
    is_shared: bool = False
    depends_on: tuple = ()

    def __init__(self, ctx: PhaseContext):
        self.ctx = ctx

    @abstractmethod
    def run(self) -> dict:
        """Execute the phase. Returns a dict of summary metrics."""
        ...

    def write_status(self, status: PhaseStatus, **extra: Any) -> None:
        """Write status.json into the phase out_dir."""
        atomic_write_json(self.ctx.out_dir / "status.json", {
            "phase": self.phase_name,
            "seed": self.ctx.seed,
            "status": status.value,
            "ts": time.time(),
            **extra,
        })
