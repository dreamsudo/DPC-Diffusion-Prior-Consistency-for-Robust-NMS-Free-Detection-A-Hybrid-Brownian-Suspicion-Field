#!/usr/bin/env python3
"""10-step dry run to verify loss components are sane before committing to training.

For Phase 1, only the diffusion MSE is active. The check is simpler than
Phase 2's (which has multiple loss heads), but still useful: catches device
issues, NaN/Inf, gradient explosion, and trivially-broken setups.

Usage:
    python tools/sanity_check_loss.py \
        --coco-cache caches/coco_train2017_128.pt \
        --steps 10 \
        --batch-size 16 \
        --seed 42

Exit code:
    0  all checks passed
    1  at least one check failed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from dpc._version import __version__
from dpc.config import DPCConfig
from dpc.data import NormalImageDataset, collate_dpc_batch, make_coco_split
from dpc.data_cache import TensorCache
from dpc.denoiser import TinyUNetDenoiser
from dpc.diffusion import (
    add_noise,
    get_alpha_bars,
    make_beta_schedule,
    sample_timesteps,
)
from dpc.manifest import fingerprint_environment
from dpc.seeding import set_global_seed, make_generator


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def warn(m): print(f"  {C.Y}!{C.END} {m}")
def fail(m): print(f"  {C.R}✗{C.END} {m}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-cache", type=str, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="runs/sanity_check_loss_phase1")
    args = p.parse_args()

    print(f"{C.BOLD}DPC Loss Sanity Check (Phase 1) v{__version__}{C.END}")

    # Reproducibility
    set_global_seed(args.seed)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    info(f"device: {device}")

    checks_passed: list[str] = []
    checks_failed: list[str] = []

    # ── Load cache ────────────────────────────────────────────────────────
    stage("Load COCO cache")
    coco_path = Path(args.coco_cache).resolve()
    if not coco_path.is_file():
        fail(f"COCO cache not found: {coco_path}"); sys.exit(1)
    coco = TensorCache(coco_path)
    ok(f"loaded {len(coco)} images at {coco.resolution}")

    # Tiny dataset/loader — we just need a few batches
    train_idx, _ = make_coco_split(coco, val_frac=0.01, seed=args.seed)
    ds = NormalImageDataset(coco, indices=train_idx[:args.batch_size * args.steps * 2])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_dpc_batch, num_workers=0,
    )

    # ── Build model ───────────────────────────────────────────────────────
    stage("Build denoiser + optimizer")
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=1e-5)
    n_params = sum(p.numel() for p in denoiser.parameters())
    ok(f"denoiser params: {n_params:,}")

    # Diffusion buffers
    betas = make_beta_schedule(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end).to(device)
    alpha_bars = get_alpha_bars(betas)

    # ── Run dry steps ─────────────────────────────────────────────────────
    stage(f"Run {args.steps} dry steps")
    losses = []
    grad_norms = []
    step_times = []
    iter_loader = iter(loader)
    gen = make_generator(args.seed, device=device)

    for step in range(args.steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        t_start = time.perf_counter()
        x0 = batch["images"].to(device)
        b = x0.shape[0]

        t = sample_timesteps(b, cfg.timestep_min, cfg.timestep_max, device, gen)
        eps = torch.randn(x0.shape, device=device, generator=gen, dtype=x0.dtype)
        x_t = add_noise(x0, t, eps, alpha_bars)

        eps_hat = denoiser(x_t, t)
        loss = ((eps_hat - eps) ** 2).mean()

        if not torch.isfinite(loss):
            fail(f"step {step+1}: loss is non-finite ({loss.item()})")
            checks_failed.append("loss_finite")
            break

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            fail(f"step {step+1}: grad norm is non-finite ({grad_norm.item()})")
            checks_failed.append("grad_finite")
            break

        optimizer.step()

        step_ms = (time.perf_counter() - t_start) * 1000
        losses.append(float(loss.item()))
        grad_norms.append(float(grad_norm.item()))
        step_times.append(step_ms)
        print(f"  step {step+1}/{args.steps}: "
              f"loss={loss.item():.4f}  grad_norm={grad_norm.item():.3f}  "
              f"t={step_ms:.0f}ms")

    if "loss_finite" not in checks_failed and "grad_finite" not in checks_failed:
        ok("all losses and grad norms finite")
        checks_passed.append("loss_finite")
        checks_passed.append("grad_finite")

    # ── Quality checks ─────────────────────────────────────────────────────
    stage("Quality checks")

    losses_arr = np.array(losses)
    grad_arr = np.array(grad_norms)
    times_arr = np.array(step_times)

    # Loss should decrease
    if len(losses_arr) >= 4:
        first_half = losses_arr[: len(losses_arr) // 2].mean()
        second_half = losses_arr[len(losses_arr) // 2 :].mean()
        if second_half < first_half:
            ok(f"loss decreasing: first half {first_half:.4f} → second half {second_half:.4f}")
            checks_passed.append("loss_decreases")
        else:
            warn(f"loss NOT decreasing: first half {first_half:.4f} → "
                 f"second half {second_half:.4f}. With only {args.steps} steps "
                 f"this can happen by chance, but flagging.")
            # We don't fail this, just warn — 10 steps is short

    # Grad norm reasonable
    median_grad = float(np.median(grad_arr))
    if 0.001 < median_grad < 50.0:
        ok(f"grad norm sane: median {median_grad:.3f}")
        checks_passed.append("grad_norm_range")
    else:
        fail(f"grad norm extreme: median {median_grad:.3f}")
        checks_failed.append("grad_norm_range")

    # Step time reasonable (catches CPU-only running on CUDA system)
    median_ms = float(np.median(times_arr))
    info(f"median step time: {median_ms:.0f} ms")
    if median_ms < 5000:
        ok(f"step time < 5s — running on accelerator as expected")
        checks_passed.append("step_time_reasonable")
    else:
        warn(f"step time {median_ms:.0f} ms — possibly running on CPU only")

    # ── Report ─────────────────────────────────────────────────────────────
    stage("Write report")
    report = {
        "version": __version__,
        "tool": "sanity_check_loss",
        "phase": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "n_steps": int(len(losses)),
        "losses_per_step": losses,
        "grad_norms_per_step": grad_norms,
        "step_times_ms": step_times,
        "loss_first_half_mean": float(losses_arr[: len(losses_arr) // 2].mean()) if len(losses_arr) >= 2 else None,
        "loss_second_half_mean": float(losses_arr[len(losses_arr) // 2 :].mean()) if len(losses_arr) >= 2 else None,
        "grad_norm_median": median_grad,
        "step_time_ms_median": median_ms,
        "denoiser_params": int(n_params),
        "device": str(device),
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
    }
    out_path = output_dir / "report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    ok(f"report -> {out_path}")

    print()
    if checks_failed:
        print(f"  {C.R}{C.BOLD}FAIL:{C.END} {len(checks_failed)} checks failed")
        for c in checks_failed: print(f"    - {c}")
        sys.exit(1)
    else:
        print(f"  {C.G}{C.BOLD}PASS:{C.END} {len(checks_passed)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
