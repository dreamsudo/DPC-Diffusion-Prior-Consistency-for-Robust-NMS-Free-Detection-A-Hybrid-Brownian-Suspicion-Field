"""eval_p3 — Phase 3 end-to-end evaluation.

For each alpha in cfg.eval_p3.alphas, runs tools/evaluate_phase3.py
against the Phase 2 checkpoint. Aggregates per-alpha results into
alpha_sweep.json.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from dpcctl.phases._base import Phase, PhaseStatus, register


@register("eval_p3")
class EvalP3Phase(Phase):
    is_shared = False
    depends_on = ("train_p2",)

    def run(self) -> dict:
        cfg = self.ctx.cfg
        events = self.ctx.events
        events.clear_phase("eval_p3")
        events.set_global("eval_p3", "running", seed=self.ctx.seed)
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
                f"eval_p3: no Phase 2 checkpoint under {p2_dir}"
            )

        # Optional fine-tuned head weights produced by train_phase2.py
        yolo_head_finetuned = p2_dir / "yolo26_head_finetuned.pt"

        staging = Path(__file__).resolve().parents[2]
        tool = staging / "tools" / "evaluate_phase3.py"
        apricot_eval = cfg.cache_dir / f"apricot_eval_{cfg.prep.eval_resolution}.pt"

        results = []
        t0 = time.time()
        for alpha in cfg.eval_p3.alphas:
            alpha_dir = self.ctx.out_dir / f"alpha_{int(alpha * 10):03d}"
            eval_subdir = alpha_dir / "eval"
            eval_subdir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python", str(tool),
                "--denoiser-checkpoint", str(ckpt),
                "--yolo-weights", str(cfg.yolo_weights_path),
                "--apricot-eval-cache", str(apricot_eval),
                "--output-dir", str(eval_subdir),
                "--cls-alpha", str(alpha),
                "--obj-alpha", str(alpha),
                "--score-threshold", str(cfg.eval_p3.score_threshold),
                "--on-patch-iou", str(cfg.eval_p3.on_patch_iou),
                "--device", cfg.device,
                "--seed", str(self.ctx.seed),
            ]
            if yolo_head_finetuned.is_file():
                cmd += ["--yolo-head-finetuned", str(yolo_head_finetuned)]
            if cfg.eval_p3.n_images is not None:
                cmd += ["--n-images", str(cfg.eval_p3.n_images)]
            if cfg.eval_p3.use_phase2_ema:
                cmd.append("--use-ema")

            events.publish("eval_p3", "eval", {
                "seed": self.ctx.seed,
                "alpha": alpha,
                "status": "running",
                "alphas_total": len(cfg.eval_p3.alphas),
            })

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                self.write_status(PhaseStatus.FAILED,
                                  error=f"evaluate_phase3 alpha={alpha} exited {result.returncode}")
                raise RuntimeError(
                    f"eval_p3: evaluate_phase3 failed for alpha={alpha}"
                )

            agg_path = eval_subdir / "aggregate.json"
            if agg_path.is_file():
                try:
                    agg = json.loads(agg_path.read_text())
                    agg["alpha"] = alpha
                    results.append(agg)
                except json.JSONDecodeError:
                    pass

        elapsed_min = (time.time() - t0) / 60.0
        sweep_path = self.ctx.out_dir / "alpha_sweep.json"
        sweep_path.write_text(json.dumps({
            "alphas": cfg.eval_p3.alphas,
            "n_images": cfg.eval_p3.n_images,
            "results": results,
            "elapsed_min": elapsed_min,
        }, indent=2, default=str))

        events.publish("eval_p3", "eval", {
            "seed": self.ctx.seed,
            "alphas_total": len(cfg.eval_p3.alphas),
            "alphas_complete": len(results),
            "status": "complete",
            "elapsed_min": round(elapsed_min, 2),
        })
        self.write_status(PhaseStatus.COMPLETE, elapsed_min=elapsed_min)
        return {"elapsed_min": elapsed_min, "n_alphas": len(results)}
