"""train_p2 phase — Phase 2 JOINT denoiser + YOLO26 head training.

Wraps tools/train_phase2.py. Full v3.3.0 rewrite:
  - Passes --yolo-weights (required for joint training)
  - Passes split LRs (--lr-denoiser, --lr-yolo-head)
  - Uses canonical mixture-proportion names (--p-coco, --p-apricot, --p-synthetic)
  - Passes all paper hyperparameters from cfg.dpc (lambda_match, lambda_entropy,
    lambda_locstab, lambda_anchor, lambda_loc, lambda_conf, lambda_mse)
  - Resolves Phase 1 checkpoint as the 'final' directory under train_p1's
    checkpoints, not a flat .pt file
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from dpcctl.phases._base import Phase, PhaseStatus, register
from dpcctl.phases.train_p1 import _dashboard_tail_train


def _resolve_phase1_checkpoint(seed_dir: Path) -> Path:
    """Return the Phase 1 checkpoint directory the v3.3.0 trainer expects.

    The Phase 1 trainer saves checkpoints under
      <out_dir>/checkpoints/{step_<N>, final, latest}
    where 'latest' symlinks to either 'final' or the most recent step.
    """
    p1_out = seed_dir / "train_p1"
    candidates = [
        p1_out / "checkpoints" / "latest",
        p1_out / "checkpoints" / "final",
        p1_out / "final",  # legacy single-file layout
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Phase 1 checkpoint not found; tried: {[str(c) for c in candidates]}"
    )


@register("train_p2")
class TrainP2Phase(Phase):
    is_shared = False
    depends_on = ("train_p1",)

    def run(self) -> dict:
        cfg = self.ctx.cfg
        events = self.ctx.events
        events.clear_phase("train_p2")
        events.set_global("train_p2", "running", seed=self.ctx.seed)
        self.write_status(PhaseStatus.RUNNING)

        out_dir = self.ctx.out_dir
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metrics_dir / "per_step.csv"

        seed_dir = self.ctx.out_dir.parent
        p1_ckpt = _resolve_phase1_checkpoint(seed_dir)

        staging = Path(__file__).resolve().parents[2]
        tool = staging / "tools" / "train_phase2.py"

        cmd = [
            "python", str(tool),
            "--phase1-checkpoint", str(p1_ckpt),
            "--yolo-weights", str(cfg.yolo_weights_path),
            "--coco-cache", str(
                cfg.cache_dir / f"coco_train2017_{cfg.prep.probe_resolution}.pt"
            ),
            "--apricot-cache", str(
                cfg.cache_dir / f"apricot_train_{cfg.prep.probe_resolution}.pt"
            ),
            "--apricot-val-cache", str(
                cfg.cache_dir / f"apricot_val_{cfg.prep.probe_resolution}.pt"
            ),
            "--color-dist", str(cfg.cache_dir / "color_distribution.json"),
            "--output-dir", str(out_dir),
            "--seed", str(self.ctx.seed),
            "--device", cfg.device,
            "--epochs", str(cfg.train_p2.additional_epochs),
            "--steps-per-epoch", str(cfg.train_p2.steps_per_epoch),
            "--batch-size", str(cfg.train_p2.batch_size),
            "--lr-denoiser", str(cfg.train_p2.denoiser_lr),
            "--lr-yolo-head", str(cfg.train_p2.head_lr),
            "--warmup-steps", str(cfg.train_p2.warmup_steps),
            "--log-every-steps", str(cfg.train_p2.log_every_steps),
            "--save-ckpt-every-steps", str(cfg.train_p2.save_ckpt_every_steps),
            "--num-workers", str(cfg.train_p2.num_workers),
            "--ema-decay", str(cfg.train_p2.ema_decay),
            "--early-stop-patience", str(cfg.train_p2.early_stop_patience),
            "--p-coco", str(cfg.train_p2.p_coco),
            "--p-apricot", str(cfg.train_p2.p_apricot),
            "--p-synthetic", str(cfg.train_p2.p_synthetic),
            # Paper hyperparameters (Eq. 22, Eq. 19, Eq. 20, etc.)
            "--lambda-mse", str(cfg.dpc.lambda_mse),
            "--lambda-match", str(cfg.dpc.lambda_match),
            "--lambda-entropy", str(cfg.dpc.lambda_entropy),
            "--lambda-locstab", str(cfg.dpc.lambda_locstab),
            "--lambda-anchor", str(cfg.dpc.lambda_anchor),
            "--lambda-loc", str(cfg.dpc.lambda_loc),
            "--lambda-conf", str(cfg.dpc.lambda_conf),
        ]

        t0 = time.time()
        stop_event = threading.Event()
        tail = threading.Thread(
            target=_dashboard_tail_train,
            args=(stop_event, csv_path, self.ctx.live_dir, "train_p2",
                  self.ctx.seed, cfg.train_p2.log_every_steps,
                  cfg.train_p2.additional_epochs),
            daemon=True,
        )
        if cfg.viz.live_dashboard:
            tail.start()

        try:
            result = subprocess.run(cmd, check=False)
        finally:
            stop_event.set()
            if tail.is_alive():
                tail.join(timeout=3.0)

        if result.returncode != 0:
            self.write_status(
                PhaseStatus.FAILED,
                error=f"train_phase2 exited {result.returncode}",
            )
            raise RuntimeError(
                f"train_p2: train_phase2.py failed (exit {result.returncode})"
            )

        elapsed_min = (time.time() - t0) / 60.0
        events.publish("train_p2", "training", {
            "seed": self.ctx.seed,
            "epochs_total": cfg.train_p2.additional_epochs,
            "status": "complete",
            "elapsed_min": round(elapsed_min, 2),
        })
        self.write_status(PhaseStatus.COMPLETE, elapsed_min=elapsed_min)
        return {"elapsed_min": elapsed_min}
