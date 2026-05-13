"""diagnose_p1 and diagnose_p2 — residual ratio diagnostics.

Both phases wrap tools/diagnose_residuals.py. diagnose_p2 additionally
runs tools/compare_phases.py to compute Phase 1 vs Phase 2 deltas.

v3.3.0 fixes:
  - Flag name corrected: --denoiser-checkpoint -> --checkpoint.
  - Checkpoint resolution: walks the v3.3.0 directory layout
    (checkpoints/{latest,final}) instead of a flat final.pt file.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from dpcctl.phases._base import Phase, PhaseStatus, register


def _resolve_phase_checkpoint(seed_dir: Path, phase_name: str) -> Path:
    """Find the checkpoint directory written by tools/train_phase{1,2}.py.

    Checks in order: checkpoints/latest, checkpoints/final, legacy final.pt.
    """
    base = seed_dir / phase_name
    candidates = [
        base / "checkpoints" / "latest",
        base / "checkpoints" / "final",
        base / "final.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"{phase_name} checkpoint not found; tried: {[str(c) for c in candidates]}"
    )


def _run_diagnose(ctx, ckpt_path: Path, label: str) -> dict:
    cfg = ctx.cfg
    events = ctx.events
    events.set_global(label, "running", seed=ctx.seed)

    out_dir = ctx.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    staging = Path(__file__).resolve().parents[2]
    tool = staging / "tools" / "diagnose_residuals.py"

    cmd = [
        "python", str(tool),
        "--checkpoint", str(ckpt_path),
        "--apricot-eval-cache", str(
            cfg.cache_dir / f"apricot_eval_{cfg.prep.eval_resolution}.pt"
        ),
        "--output-dir", str(out_dir),
        "--seed", str(ctx.seed),
        "--device", cfg.device,
        "--use-ema",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed_min = (time.time() - t0) / 60.0
    if result.returncode != 0:
        raise RuntimeError(f"{label}: diagnose_residuals exited {result.returncode}")

    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}

    events.publish(label, "diagnostic", {
        "seed": ctx.seed,
        "elapsed_min": round(elapsed_min, 2),
        "aggregate": summary.get("aggregate", {}),
    })
    return {"elapsed_min": elapsed_min, "summary": summary}


@register("diagnose_p1")
class DiagnoseP1Phase(Phase):
    is_shared = False
    depends_on = ("train_p1",)

    def run(self) -> dict:
        self.write_status(PhaseStatus.RUNNING)
        seed_dir = self.ctx.out_dir.parent
        ckpt = _resolve_phase_checkpoint(seed_dir, "train_p1")
        out = _run_diagnose(self.ctx, ckpt, "diagnose_p1")
        self.write_status(PhaseStatus.COMPLETE, elapsed_min=out["elapsed_min"])
        return out


@register("diagnose_p2")
class DiagnoseP2Phase(Phase):
    is_shared = False
    depends_on = ("train_p2",)

    def run(self) -> dict:
        self.write_status(PhaseStatus.RUNNING)
        seed_dir = self.ctx.out_dir.parent
        ckpt = _resolve_phase_checkpoint(seed_dir, "train_p2")
        out = _run_diagnose(self.ctx, ckpt, "diagnose_p2")

        # Compare to Phase 1 if both summaries exist
        p1_summary = seed_dir / "diagnose_p1" / "summary.json"
        p2_summary = self.ctx.out_dir / "summary.json"
        if p1_summary.is_file() and p2_summary.is_file():
            staging = Path(__file__).resolve().parents[2]
            cmp_tool = staging / "tools" / "compare_phases.py"
            cmp_out = self.ctx.out_dir / "comparison_to_phase1.json"
            cmd = [
                "python", str(cmp_tool),
                "--phase1-summary", str(p1_summary),
                "--phase2-summary", str(p2_summary),
                "--output", str(cmp_out),
            ]
            subprocess.run(cmd, check=False)

        self.write_status(PhaseStatus.COMPLETE, elapsed_min=out["elapsed_min"])
        return out
