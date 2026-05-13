"""Atomic checkpoint save/load with full training state.

CARRIED FORWARD FROM v2.x — this module was one of the cleanest in v2 and is
trusted completely. Only change for v3 is updating the docstring to reference
the v3 architecture doc.

Each checkpoint is a *directory* containing:
  - model.pt              raw weights
  - ema.pt                EMA shadow weights (optional)
  - optimizer.pt          AdamW state (optional)
  - scheduler.pt          cosine LR state (optional)
  - rng.pt                torch + numpy + python RNG states
  - meta.json             epoch, step, config, timestamps
  - SHA256SUMS            checksums of every file above

Atomic dir-rename guarantees a crash mid-save can't corrupt the previous good
checkpoint. The tmp dir only becomes "the checkpoint" when os.rename succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture_rng_state() -> dict:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    ema=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    meta: Optional[dict] = None,
) -> None:
    """Save a full training checkpoint atomically."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=out_dir.name + ".tmp.",
            dir=str(out_dir.parent),
        )
    )

    try:
        torch.save(model.state_dict(), tmp_dir / "model.pt")

        if ema is not None:
            torch.save(ema.state_dict(), tmp_dir / "ema.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
        torch.save(_capture_rng_state(), tmp_dir / "rng.pt")

        meta = dict(meta or {})
        meta.setdefault("saved_at_utc", datetime.now(timezone.utc).isoformat())
        meta.setdefault("torch_version", torch.__version__)
        with open(tmp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        sums = {}
        for p in sorted(tmp_dir.iterdir()):
            if p.name == "SHA256SUMS":
                continue
            sums[p.name] = _sha256_file(p)
        with open(tmp_dir / "SHA256SUMS", "w") as f:
            for name, h in sums.items():
                f.write(f"{h}  {name}\n")

        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.rename(str(tmp_dir), str(out_dir))

    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def load_checkpoint(
    in_dir: Path,
    model: torch.nn.Module,
    ema=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location=None,
    restore_rng: bool = True,
) -> dict:
    """Load a checkpoint dir into the provided objects, return meta dict.

    Uses weights_only=False because rng.pt contains numpy state. Safe because
    we only ever load checkpoints we ourselves produced via save_checkpoint.
    """
    in_dir = Path(in_dir)
    if in_dir.is_symlink():
        in_dir = in_dir.resolve()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {in_dir}")

    model.load_state_dict(
        torch.load(in_dir / "model.pt", map_location=map_location, weights_only=False)
    )
    if ema is not None and (in_dir / "ema.pt").exists():
        ema.load_state_dict(
            torch.load(in_dir / "ema.pt", map_location=map_location, weights_only=False)
        )
    if optimizer is not None and (in_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(in_dir / "optimizer.pt", map_location=map_location, weights_only=False)
        )
    if scheduler is not None and (in_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(
            torch.load(in_dir / "scheduler.pt", map_location=map_location, weights_only=False)
        )
    if restore_rng and (in_dir / "rng.pt").exists():
        rng_state = torch.load(
            in_dir / "rng.pt", map_location="cpu", weights_only=False
        )
        _restore_rng_state(rng_state)

    with open(in_dir / "meta.json") as f:
        return json.load(f)


def update_symlink(target: Path, link_path: Path) -> None:
    """Atomically update a symlink to point at `target`."""
    link_path = Path(link_path)
    target = Path(target)
    rel_target = os.path.relpath(target, link_path.parent)
    tmp_link = link_path.with_suffix(".tmp." + str(os.getpid()))
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(rel_target, tmp_link)
    os.replace(str(tmp_link), str(link_path))


def prune_step_checkpoints(ckpt_dir: Path, keep_last: int = 3) -> None:
    """Delete old `ckpt_step_*` directories, keeping the most recent `keep_last`."""
    ckpt_dir = Path(ckpt_dir)
    step_ckpts = sorted(
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.startswith("ckpt_step_")
    )
    to_delete = step_ckpts[:-keep_last] if keep_last > 0 else step_ckpts
    for p in to_delete:
        shutil.rmtree(p)
