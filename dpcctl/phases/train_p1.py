"""train_p1 phase — Phase 1 denoiser pretraining.

Wraps tools/train_phase1.py. v3.3.0: flag names brought into line with
the trainer's actual CLI; passes --skip-data-gate/--skip-loss-gate
because the orchestrator's prep phase already validates data integrity.
"""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path

from dpcctl.events import EventWriter
from dpcctl.phases._base import Phase, PhaseStatus, register


def _dashboard_tail_train(
    stop_event: threading.Event,
    csv_path: Path,
    live_dir: Path,
    phase: str,
    seed: int,
    log_every: int,
    epochs_total: int,
) -> None:
    """Tail per_step.csv, publish state to the dashboard event bus."""
    last_size = -1
    last_global_step = 0
    loss_recent: list[float] = []
    writer = EventWriter(live_dir=live_dir, run_name="")
    poll_interval = 1.5

    while not stop_event.is_set():
        try:
            if csv_path.is_file():
                size = csv_path.stat().st_size
                if size != last_size:
                    last_size = size
                    rows = []
                    try:
                        with open(csv_path, "r", newline="") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                rows.append(row)
                    except (OSError, csv.Error):
                        rows = []

                    if rows:
                        last = rows[-1]
                        try:
                            last_global_step = int(last.get("global_step", 0))
                            current_loss = float(
                                last.get("loss_total", last.get("loss", 0.0))
                            )
                        except (TypeError, ValueError):
                            current_loss = 0.0
                        loss_recent = []
                        for r in rows[-30:]:
                            try:
                                loss_recent.append(
                                    float(r.get("loss_total", r.get("loss", 0.0)))
                                )
                            except (TypeError, ValueError):
                                pass

                        payload = {
                            "seed": seed,
                            "epochs_total": epochs_total,
                            "global_step": last_global_step,
                            "current_loss": current_loss,
                            "loss_recent": loss_recent,
                            "log_every": log_every,
                            "n_rows": len(rows),
                        }
                        writer.publish(phase, "training", payload)
                        writer.set_global(phase, "running")
        except Exception:
            pass

        stop_event.wait(timeout=poll_interval)


@register("train_p1")
class TrainP1Phase(Phase):
    is_shared = False
    depends_on = ("prep",)

    def run(self) -> dict:
        cfg = self.ctx.cfg
        events = self.ctx.events
        events.clear_phase("train_p1")
        events.set_global("train_p1", "running", seed=self.ctx.seed)
        self.write_status(PhaseStatus.RUNNING)

        out_dir = self.ctx.out_dir
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metrics_dir / "per_step.csv"

        staging = Path(__file__).resolve().parents[2]
        tool = staging / "tools" / "train_phase1.py"

        cmd = [
            "python", str(tool),
            "--coco-cache", str(
                cfg.cache_dir / f"coco_train2017_{cfg.prep.probe_resolution}.pt"
            ),
            "--apricot-val-cache", str(
                cfg.cache_dir / f"apricot_val_{cfg.prep.probe_resolution}.pt"
            ),
            "--output-dir", str(out_dir),
            "--seed", str(self.ctx.seed),
            "--device", cfg.device,
            "--epochs", str(cfg.train_p1.epochs),
            "--batch-size", str(cfg.train_p1.batch_size),
            "--lr", str(cfg.train_p1.learning_rate),
            "--warmup-steps", str(cfg.train_p1.warmup_steps),
            "--log-every-steps", str(cfg.train_p1.log_every_steps),
            "--save-ckpt-every-steps", str(cfg.train_p1.save_ckpt_every_steps),
            "--num-workers", str(cfg.train_p1.num_workers),
            "--max-steps-per-epoch", str(cfg.train_p1.max_steps_per_epoch),
            "--ema-decay", str(cfg.train_p1.ema_decay),
            "--early-stop-patience", str(cfg.train_p1.early_stop_patience),
            # Orchestrator's prep phase already validated data integrity;
            # the trainer's own gates would be a redundant re-check.
            "--skip-data-gate",
            "--skip-loss-gate",
        ]

        t0 = time.time()
        stop_event = threading.Event()
        tail = threading.Thread(
            target=_dashboard_tail_train,
            args=(stop_event, csv_path, self.ctx.live_dir, "train_p1",
                  self.ctx.seed, cfg.train_p1.log_every_steps,
                  cfg.train_p1.epochs),
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
                error=f"train_phase1 exited {result.returncode}",
            )
            raise RuntimeError(
                f"train_p1: train_phase1.py failed (exit {result.returncode})"
            )

        elapsed_min = (time.time() - t0) / 60.0
        events.publish("train_p1", "training", {
            "seed": self.ctx.seed,
            "epochs_total": cfg.train_p1.epochs,
            "status": "complete",
            "elapsed_min": round(elapsed_min, 2),
        })
        self.write_status(PhaseStatus.COMPLETE, elapsed_min=elapsed_min)
        return {"elapsed_min": elapsed_min}
