"""eval_negative_control — DPC vs baseline on clean COCO (no patches).

The failure mode this phase catches: DPC overzealously suppresses
detections even when no adversarial patch is present. Healthy DPC
behavior is "approximately no change" on clean scenes.

Outputs per-image detection-count deltas (baseline vs DPC) and a
distribution summary.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from dpcctl.phases._base import Phase, PhaseStatus, register


@register("eval_negative_control")
class EvalNegativeControlPhase(Phase):
    is_shared = False
    depends_on = ("train_p2",)

    def run(self) -> dict:
        cfg = self.ctx.cfg
        events = self.ctx.events
        events.clear_phase("eval_negative_control")
        events.set_global("eval_negative_control", "running",
                          seed=self.ctx.seed)
        self.write_status(PhaseStatus.RUNNING)

        seed_dir = self.ctx.out_dir.parent
        p2_dir = seed_dir / "train_p2"
        ckpt = None
        for candidate in (p2_dir / "checkpoints" / "latest",
                          p2_dir / "checkpoints" / "final",
                          p2_dir / "final.pt"):
            if candidate.exists():
                ckpt = candidate
                break
        if ckpt is None:
            self.write_status(PhaseStatus.FAILED,
                              error=f"Phase 2 checkpoint not found under {p2_dir}")
            raise RuntimeError(
                f"eval_negative_control: no Phase 2 checkpoint under {p2_dir}"
            )

        yolo_head_finetuned = p2_dir / "yolo26_head_finetuned.pt"

        # We reuse the COCO probe cache as the clean-scene source.
        coco_cache = cfg.cache_dir / f"coco_train2017_{cfg.prep.probe_resolution}.pt"
        if not coco_cache.is_file():
            self.write_status(PhaseStatus.FAILED,
                              error=f"COCO cache missing: {coco_cache}")
            raise RuntimeError(
                "eval_negative_control: COCO cache missing; run prep first"
            )

        staging = Path(__file__).resolve().parents[2]
        tool = staging / "tools" / "evaluate_phase3.py"

        out_dir = self.ctx.out_dir / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluate_phase3 in negative-control mode (clean cache, no
        # APRICOT patches). The eval tool treats missing patch boxes as
        # "every box is off-patch", which is the right behavior here.
        cmd = [
            "python", str(tool),
            "--denoiser-checkpoint", str(ckpt),
            "--yolo-weights", str(cfg.yolo_weights_path),
            "--apricot-eval-cache", str(coco_cache),
            "--output-dir", str(out_dir),
            "--cls-alpha", "50.0",
            "--obj-alpha", "50.0",
            "--score-threshold", str(cfg.eval_negative_control.score_threshold),
            "--n-images", str(cfg.eval_negative_control.n_images),
            "--seed", str(self.ctx.seed),
            "--negative-control",
        ]
        if yolo_head_finetuned.is_file():
            cmd += ["--yolo-head-finetuned", str(yolo_head_finetuned)]
        if cfg.eval_negative_control.use_phase2_ema:
            cmd.append("--use-ema")

        t0 = time.time()
        events.publish("eval_negative_control", "eval", {
            "seed": self.ctx.seed,
            "status": "running",
            "n_images": cfg.eval_negative_control.n_images,
        })
        result = subprocess.run(cmd, check=False)
        elapsed_min = (time.time() - t0) / 60.0

        if result.returncode != 0:
            self.write_status(PhaseStatus.FAILED,
                              error=f"evaluate_phase3 (negative-control) exited {result.returncode}")
            raise RuntimeError(
                f"eval_negative_control: evaluate_phase3 failed (exit {result.returncode})"
            )

        agg_path = out_dir / "aggregate.json"
        summary = {}
        if agg_path.is_file():
            try:
                summary = json.loads(agg_path.read_text())
            except json.JSONDecodeError:
                summary = {}

        events.publish("eval_negative_control", "eval", {
            "seed": self.ctx.seed,
            "status": "complete",
            "elapsed_min": round(elapsed_min, 2),
            "aggregate": summary,
        })
        self.write_status(PhaseStatus.COMPLETE, elapsed_min=elapsed_min)
        return {"elapsed_min": elapsed_min, "summary": summary}
