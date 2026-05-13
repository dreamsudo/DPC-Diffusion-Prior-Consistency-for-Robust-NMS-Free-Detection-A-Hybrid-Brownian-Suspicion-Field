#!/usr/bin/env python3
"""Phase 1 — Natural-prior pretraining.

Trains TinyUNet on COCO only. No APRICOT, no synthetic patches, no auxiliary
losses. Just diffusion MSE on natural images. The output is a denoiser that
(per S1) produces residuals with median inside-over-outside ratio ≥ 1.2 on
APRICOT patch regions, as an emergent property of natural-manifold consistency.

Usage:
    python tools/train_phase1.py \
        --coco-cache caches/coco_train2017_128.pt \
        --epochs 8 \
        --batch-size 16 \
        --lr 1e-4 \
        --seed 42 \
        --output-dir runs/phase1_seed42

Pre-launch gates (enforced):
  - sanity_check_data report exists and has zero failures
  - sanity_check_loss --phase 1 report exists and has zero failures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dpc._version import __version__
from dpc.checkpoint import (
    load_checkpoint,
    prune_step_checkpoints,
    save_checkpoint,
    update_symlink,
)
from dpc.config import DPCConfig
from dpc.data import (
    NormalImageDataset,
    TensorAugment,
    collate_dpc_batch,
    make_coco_split,
)
from dpc.data_cache import TensorCache
from dpc.denoiser import TinyUNetDenoiser
from dpc.diffusion import (
    add_noise,
    get_alpha_bars,
    make_beta_schedule,
    sample_timesteps,
)
from dpc.ema import EMA
from dpc.field import DPCField
from dpc.manifest import fingerprint_environment, write_manifest
from dpc.metrics import (
    aggregate_residual_distribution,
    residual_ratio_per_image,
)
from dpc.seeding import make_generator, set_global_seed


_SHOULD_STOP = {"flag": False}


def _handle_sigint(signum, frame):
    print("\n[!] SIGINT received - finishing current step then saving...")
    _SHOULD_STOP["flag"] = True


@dataclass
class Phase1Config:
    coco_cache: str
    apricot_val_cache: Optional[str] = None  # if present, residual diagnostic per epoch
    output_dir: str = "runs/phase1_seed42"
    seed: int = 42
    device: str = "auto"

    # Training schedule
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    num_workers: int = -1

    # Early stop
    early_stop_patience: int = 3

    # EMA
    ema_decay: float = 0.999

    # Validation
    val_frac: float = 0.01
    val_max_batches: int = 32

    # Logging
    log_every_steps: int = 50
    save_sample_every_steps: int = 250
    save_ckpt_every_steps: int = 1000
    keep_last_step_ckpts: int = 3
    csv_flush_every_steps: int = 100

    # Smoke-test step cap (0 = no cap)
    max_steps_per_epoch: int = 0

    # Pre-launch gates
    skip_data_gate: bool = False
    skip_loss_gate: bool = False
    data_gate_path: str = "runs/sanity_check_data_phase1/report.json"
    loss_gate_path: str = "runs/sanity_check_loss_phase1/report.json"


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Cosine schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def check_gate(report_path: Path, name: str, log) -> bool:
    """Verify a sanity-check report exists and has no failures."""
    if not report_path.is_file():
        log(f"  [gate] {name} report MISSING at {report_path}")
        log(f"  [gate] run: python tools/{name}.py [...]")
        return False
    with open(report_path) as f:
        report = json.load(f)
    n_failed = len(report.get("checks_failed", [])) or report.get("n_failed", 0)
    if n_failed > 0:
        log(f"  [gate] {name} has {n_failed} failed checks")
        return False
    log(f"  [gate] {name}: PASS ({len(report.get('checks_passed', []))} checks)")
    return True


def compute_val_loss(denoiser, val_loader, device, alpha_bars, t_min, t_max,
                     gen, max_batches: int = 32) -> Optional[float]:
    """Diffusion MSE on the validation set."""
    if val_loader is None:
        return None
    denoiser.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x0 = batch["images"].to(device)
            b = x0.shape[0]
            t = sample_timesteps(b, t_min, t_max, device, gen)
            eps = torch.randn(x0.shape, device=device, generator=gen, dtype=x0.dtype)
            x_t = add_noise(x0, t, eps, alpha_bars)
            eps_hat = denoiser(x_t, t)
            losses.append(((eps_hat - eps) ** 2).mean().item())
    denoiser.train()
    return float(np.mean(losses)) if losses else None


def compute_val_ratio_distribution(
    field: DPCField,
    val_apricot_cache: Optional[TensorCache],
    device: torch.device,
    seed: int,
    max_images: int = 32,
) -> Optional[dict]:
    """Compute residual ratio distribution on a sample of APRICOT val images.

    Used per-epoch in place of the v2 single concentration scalar (mistake #27).
    Returns the aggregate distribution dict, or None if no apricot val cache.
    """
    if val_apricot_cache is None:
        return None
    field.eval()
    n = min(len(val_apricot_cache), max_images)
    per_image = []
    gen = make_generator(seed + 100, device=device)
    with torch.no_grad():
        for i in range(n):
            item = val_apricot_cache[i]
            img = item["image"].unsqueeze(0).to(device)
            bboxes = item["metadata"].get("bboxes_xyxy", [])
            if not bboxes:
                continue
            # Build mask at probe res (apricot val cache is at probe res already)
            H, W = img.shape[-2:]
            mask = torch.zeros((H, W), dtype=torch.float32, device=device)
            for x1, y1, x2, y2 in bboxes:
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(W, int(x2)), min(H, int(y2))
                if ix2 > ix1 and iy2 > iy1:
                    mask[iy1:iy2, ix1:ix2] = 1.0
            if mask.sum() < 1:
                continue
            raw = field.compute_raw_signal(img, generator=gen)
            hybrid = raw["hybrid"][0, 0]  # [H, W]
            stats = residual_ratio_per_image(hybrid, mask)
            per_image.append(stats)
    field.train()
    return aggregate_residual_distribution(per_image)


def main(cfg: Phase1Config):
    # ─── Reproducibility ───────────────────────────────────────────────────
    set_global_seed(cfg.seed)
    signal.signal(signal.SIGINT, _handle_sigint)

    # ─── Paths ─────────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir).resolve()
    dirs = {
        "base": output_dir,
        "samples": output_dir / "samples",
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "logs": output_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    log_path = dirs["logs"] / "train.log"
    log_file = open(log_path, "a")

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line)
        except BrokenPipeError:
            pass
        log_file.write(line + "\n")
        log_file.flush()

    log(f"=== DPC Phase 1 — Natural-Prior Pretraining v{__version__} ===")
    log(f"output: {output_dir}")
    log(f"seed:   {cfg.seed}")

    # ─── Pre-launch gates ─────────────────────────────────────────────────
    log("")
    log("[gates] verifying pre-launch sanity checks")
    if not cfg.skip_data_gate:
        if not check_gate(Path(cfg.data_gate_path), "sanity_check_data", log):
            log("[gates] FAIL — refusing to launch. Pass --skip-data-gate to override.")
            sys.exit(1)
    else:
        log("  [gate] sanity_check_data SKIPPED (override)")
    if not cfg.skip_loss_gate:
        if not check_gate(Path(cfg.loss_gate_path), "sanity_check_loss", log):
            log("[gates] FAIL — refusing to launch. Pass --skip-loss-gate to override.")
            sys.exit(1)
    else:
        log("  [gate] sanity_check_loss SKIPPED (override)")
    log("[gates] all pre-launch gates passed")

    # ─── DPC config ────────────────────────────────────────────────────────
    dpc_cfg = DPCConfig(device=cfg.device)
    device = dpc_cfg.get_device()
    log(f"device: {device}")

    # Save config
    with open(dirs["base"] / "config.json", "w") as f:
        cfg_dict = {
            "version": __version__,
            "tool": "train_phase1",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase1_config": asdict(cfg),
            "dpc_config": dpc_cfg.asdict(),
            "environment": fingerprint_environment(),
        }
        json.dump(cfg_dict, f, indent=2, default=str)

    # ─── Caches ────────────────────────────────────────────────────────────
    log("")
    log("[data] loading caches")
    coco = TensorCache(Path(cfg.coco_cache).resolve())
    log(f"  coco: {len(coco)} images at {coco.resolution}")
    train_idx, val_idx = make_coco_split(coco, val_frac=cfg.val_frac, seed=cfg.seed)
    log(f"  coco train/val: {len(train_idx)}/{len(val_idx)}")

    apricot_val = None
    if cfg.apricot_val_cache:
        try:
            apricot_val = TensorCache(Path(cfg.apricot_val_cache).resolve())
            log(f"  apricot val: {len(apricot_val)} images (used for ratio diagnostic)")
        except FileNotFoundError:
            log(f"  apricot val cache not found, ratio diagnostic disabled")

    # ─── Datasets/loaders ──────────────────────────────────────────────────
    train_aug = TensorAugment(seed=cfg.seed)
    train_ds = NormalImageDataset(coco, indices=train_idx, transform=train_aug)
    val_ds = NormalImageDataset(coco, indices=val_idx, transform=None)

    if cfg.num_workers < 0:
        num_workers = max(1, (os.cpu_count() or 4) - 2)
    else:
        num_workers = cfg.num_workers

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_dpc_batch, drop_last=True,
        pin_memory=False,  # pin_memory mostly helps CUDA
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_dpc_batch,
    )
    log(f"  workers: {num_workers}")

    # ─── Model + optimizer ────────────────────────────────────────────────
    log("")
    log("[model] building")
    denoiser = TinyUNetDenoiser(use_attention=dpc_cfg.use_attention).to(device)
    n_params = sum(p.numel() for p in denoiser.parameters())
    log(f"  denoiser params: {n_params:,}")

    field = DPCField(denoiser, dpc_cfg).to(device)
    ema = EMA(denoiser, decay=cfg.ema_decay).to(device)

    optimizer = optim.AdamW(
        denoiser.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = build_lr_scheduler(optimizer, total_steps, cfg.warmup_steps)
    log(f"  steps/epoch: {steps_per_epoch}  total: {total_steps}")

    # Pre-allocate device buffers for diffusion
    betas = make_beta_schedule(dpc_cfg.diffusion_steps, dpc_cfg.beta_start, dpc_cfg.beta_end).to(device)
    alpha_bars = get_alpha_bars(betas)

    train_gen = make_generator(cfg.seed, device=device)

    # ─── CSV writer ────────────────────────────────────────────────────────
    per_step_csv = dirs["metrics"] / "per_step.csv"
    csv_new = not per_step_csv.exists()
    csv_f = open(per_step_csv, "a", newline="")
    csv_w = csv.writer(csv_f)
    if csv_new:
        csv_w.writerow([
            "epoch", "step", "global_step",
            "loss_total", "loss_mse",
            "grad_norm", "lr",
            "step_time_ms",
        ])

    # ─── Train loop ────────────────────────────────────────────────────────
    log("")
    log("[train] entering training loop")
    denoiser.train()
    field.train()

    global_step = 0
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    early_stopped = False

    try:
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()

            for step_in_epoch, batch in enumerate(train_loader, 1):
                if _SHOULD_STOP["flag"]:
                    log("[stop] graceful shutdown triggered")
                    break
                if cfg.max_steps_per_epoch > 0 and step_in_epoch > cfg.max_steps_per_epoch:
                    log(f"[cap] reached max_steps_per_epoch={cfg.max_steps_per_epoch}")
                    break

                global_step += 1
                t_start = time.perf_counter()

                x0 = batch["images"].to(device)
                b = x0.shape[0]

                optimizer.zero_grad()

                t = sample_timesteps(b, dpc_cfg.timestep_min, dpc_cfg.timestep_max, device, train_gen)
                eps = torch.randn(x0.shape, device=device, generator=train_gen, dtype=x0.dtype)
                x_t = add_noise(x0, t, eps, alpha_bars)

                eps_hat = denoiser(x_t, t)
                loss_mse = ((eps_hat - eps) ** 2).mean()
                loss_total = loss_mse  # Phase 1 has only MSE

                loss_total.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    denoiser.parameters(), max_norm=cfg.grad_clip_norm
                ).item()
                optimizer.step()
                scheduler.step()
                ema.update(denoiser)

                step_ms = (time.perf_counter() - t_start) * 1000
                lr_now = optimizer.param_groups[0]["lr"]

                csv_w.writerow([
                    epoch, step_in_epoch, global_step,
                    float(loss_total.detach()), float(loss_mse.detach()),
                    grad_norm, lr_now, step_ms,
                ])

                if global_step % cfg.csv_flush_every_steps == 0:
                    csv_f.flush()

                if global_step % cfg.log_every_steps == 0:
                    log(f"epoch [{epoch}/{cfg.epochs}] step [{step_in_epoch}/{steps_per_epoch}] "
                        f"loss={float(loss_total):.4f} gn={grad_norm:.2f} "
                        f"lr={lr_now:.2e} t={step_ms:.0f}ms")

                # Step checkpoint
                if global_step % cfg.save_ckpt_every_steps == 0:
                    ckpt_dir = dirs["checkpoints"] / f"ckpt_step_{global_step:08d}"
                    save_checkpoint(
                        ckpt_dir, denoiser, ema=ema, optimizer=optimizer, scheduler=scheduler,
                        meta={
                            "version": __version__,
                            "epoch": epoch,
                            "step": step_in_epoch,
                            "global_step": global_step,
                            "trigger": "step_interval",
                            "best_val_loss": best_val_loss,
                        },
                    )
                    update_symlink(ckpt_dir, dirs["checkpoints"] / "latest")
                    prune_step_checkpoints(dirs["checkpoints"], keep_last=cfg.keep_last_step_ckpts)

            if _SHOULD_STOP["flag"]:
                break

            # ── End of epoch ────────────────────────────────────────────
            epoch_time = time.perf_counter() - epoch_start
            log(f"[epoch {epoch}] training time: {epoch_time/60:.1f} min")

            # Validation
            val_gen = make_generator(cfg.seed + epoch * 1000, device=device)
            # Use EMA weights for validation (standard DDPM practice)
            ema_denoiser = ema.shadow
            val_loss = compute_val_loss(
                ema_denoiser, val_loader, device, alpha_bars,
                dpc_cfg.timestep_min, dpc_cfg.timestep_max, val_gen,
                max_batches=cfg.val_max_batches,
            )
            log(f"[epoch {epoch}] val_loss_normal: {val_loss}")

            # Distributional ratio summary (replaces single concentration scalar)
            ema_field = DPCField(ema_denoiser, dpc_cfg).to(device)
            ratio_dist = compute_val_ratio_distribution(
                ema_field, apricot_val, device, cfg.seed + epoch * 2000,
                max_images=64,
            )
            if ratio_dist:
                log(f"[epoch {epoch}] ratio_dist: median={ratio_dist['median_ratio']} "
                    f"n_pos={ratio_dist['n_positive']}/{ratio_dist['n_valid']} "
                    f"≥5x={ratio_dist['bins'].get('greater_than_5x', 0)}")

            is_best = (val_loss is not None) and (val_loss < best_val_loss)
            if is_best:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Per-epoch summary
            epoch_summary = {
                "version": __version__,
                "epoch": epoch,
                "global_step": global_step,
                "epoch_time_seconds": epoch_time,
                "val_loss_normal": val_loss,
                "ratio_distribution_on_val": ratio_dist,
                "best_val_loss": best_val_loss,
                "is_best_val": is_best,
                "lr_at_epoch_end": optimizer.param_groups[0]["lr"],
                "epochs_since_improvement": epochs_since_improvement,
            }
            with open(dirs["metrics"] / f"per_epoch_{epoch:02d}.json", "w") as f:
                json.dump(epoch_summary, f, indent=2, default=str)

            # Epoch checkpoint
            ckpt_dir = dirs["checkpoints"] / f"ckpt_epoch_{epoch:02d}"
            save_checkpoint(
                ckpt_dir, denoiser, ema=ema, optimizer=optimizer, scheduler=scheduler,
                meta={
                    "version": __version__,
                    "epoch": epoch,
                    "step": steps_per_epoch,
                    "global_step": global_step,
                    "trigger": "epoch_boundary",
                    "val_loss_normal": val_loss,
                    "best_val_loss": best_val_loss,
                },
            )
            update_symlink(ckpt_dir, dirs["checkpoints"] / "latest")
            if is_best:
                update_symlink(ckpt_dir, dirs["checkpoints"] / "best")
                log(f"[epoch {epoch}] BEST val={val_loss:.4f}")

            # Early stop
            if epochs_since_improvement >= cfg.early_stop_patience:
                log(f"[early-stop] no improvement for {cfg.early_stop_patience} epochs")
                early_stopped = True
                break

        # Final checkpoint
        final_dir = dirs["checkpoints"] / "final"
        save_checkpoint(
            final_dir, denoiser, ema=ema, optimizer=optimizer, scheduler=scheduler,
            meta={
                "version": __version__,
                "epoch": epoch,
                "global_step": global_step,
                "trigger": "final",
                "best_val_loss": best_val_loss,
                "early_stopped": early_stopped,
            },
        )
        log(f"[done] final checkpoint written")

    finally:
        csv_f.close()
        # Write run manifest
        try:
            write_manifest(dirs["base"], extra_meta={
                "tool": "train_phase1",
                "epochs_completed": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "early_stopped": early_stopped,
            })
            log("[done] manifest written")
        except Exception as e:
            log(f"[warn] manifest write failed: {e}")
        log_file.close()


def parse_args() -> Phase1Config:
    p = argparse.ArgumentParser(description="DPC Phase 1 — Natural-prior pretraining")
    p.add_argument("--coco-cache", type=str, required=True)
    p.add_argument("--apricot-val-cache", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--save-ckpt-every-steps", type=int, default=1000)
    p.add_argument("--max-steps-per-epoch", type=int, default=0)
    p.add_argument("--skip-data-gate", action="store_true")
    p.add_argument("--skip-loss-gate", action="store_true")
    p.add_argument("--data-gate-path", type=str, default="runs/sanity_check_data_phase1/report.json")
    p.add_argument("--loss-gate-path", type=str, default="runs/sanity_check_loss_phase1/report.json")
    args = p.parse_args()
    return Phase1Config(
        coco_cache=args.coco_cache,
        apricot_val_cache=args.apricot_val_cache,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        early_stop_patience=args.early_stop_patience,
        ema_decay=args.ema_decay,
        val_frac=args.val_frac,
        log_every_steps=args.log_every_steps,
        save_ckpt_every_steps=args.save_ckpt_every_steps,
        max_steps_per_epoch=args.max_steps_per_epoch,
        skip_data_gate=args.skip_data_gate,
        skip_loss_gate=args.skip_loss_gate,
        data_gate_path=args.data_gate_path,
        loss_gate_path=args.loss_gate_path,
    )


if __name__ == "__main__":
    main(parse_args())
