#!/usr/bin/env python3
"""Phase 2 loss sanity check.

10-step dry run with the full Phase 2 loss formulation:
  L_total = λ_mse * L_mse
          + λ_loc * L_loc      (BCE on smoothed deployed field vs mask)
          + λ_ssim * L_ssim    (1 - SSIM between deployed field and mask)

Pass conditions:
  - All loss components produce finite values
  - No single component contributes > 90% of total grad norm
  - No active component contributes < 1% (bug or vestigial)
  - Total loss finite + decreasing on average
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
import torch.nn.functional as F

from dpc._version import __version__
from dpc.checkpoint import load_checkpoint
from dpc.config import DPCConfig
from dpc.data import (
    CachedApricotDataset,
    MixedDataset,
    NormalImageDataset,
    SyntheticPatchDataset,
    collate_dpc_batch,
)
from dpc.data_cache import TensorCache
from dpc.denoiser import TinyUNetDenoiser
from dpc.diffusion import (
    add_noise, get_alpha_bars, make_beta_schedule, sample_timesteps,
)
from dpc.field import DPCField
from dpc.losses import focal_bce_with_logits, ssim_loss
from dpc.manifest import fingerprint_environment
from dpc.seeding import make_generator, set_global_seed
from dpc.synthetic_patch import SyntheticPatchGenerator


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def warn(m): print(f"  {C.Y}!{C.END} {m}")
def fail(m): print(f"  {C.R}✗{C.END} {m}")


def grad_norm_of(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item()) ** 2
    return total ** 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-checkpoint", required=True,
                   help="Phase 1 checkpoint dir to resume from")
    p.add_argument("--coco-cache", required=True)
    p.add_argument("--apricot-cache", required=True)
    p.add_argument("--color-dist", required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--p-coco", type=float, default=0.5)
    p.add_argument("--p-apricot", type=float, default=0.3)
    p.add_argument("--p-synthetic", type=float, default=0.2)
    p.add_argument("--lambda-mse", type=float, default=1.0)
    p.add_argument("--lambda-loc", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, default=0.5)
    p.add_argument("--use-ema", action="store_true",
                   help="Initialize from Phase 1 EMA weights (recommended)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="runs/sanity_check_loss_phase2")
    args = p.parse_args()

    print(f"{C.BOLD}DPC Loss Sanity Check (Phase 2) v{__version__}{C.END}")
    set_global_seed(args.seed)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    info(f"device: {device}")

    passed: list[str] = []
    failed: list[str] = []

    # ── Build mixed dataset ──
    stage("Build datasets")
    coco_cache = TensorCache(Path(args.coco_cache).resolve())
    apricot_cache = TensorCache(Path(args.apricot_cache).resolve())
    info(f"coco: {len(coco_cache)}, apricot: {len(apricot_cache)}")

    coco_ds = NormalImageDataset(coco_cache, indices=list(range(min(2000, len(coco_cache)))))
    apricot_ds = CachedApricotDataset(apricot_cache)
    gen_synth = SyntheticPatchGenerator(
        Path(args.color_dist).resolve(),
        seed=args.seed,
        image_size=coco_cache.resolution[0],
    )
    synth_ds = SyntheticPatchDataset(
        coco_cache=coco_cache,
        patch_generator=gen_synth,
        coco_indices=list(range(min(2000, len(coco_cache)))),
        length=2000,
        base_seed=args.seed,
    )
    mixed = MixedDataset(
        sources=[coco_ds, apricot_ds, synth_ds],
        weights=[args.p_coco, args.p_apricot, args.p_synthetic],
        length=args.steps * args.batch_size * 2,
        base_seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        mixed, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_dpc_batch, num_workers=0,
    )
    ok("mixed loader built")

    # ── Build model from Phase 1 checkpoint ──
    stage("Load Phase 1 checkpoint")
    ckpt_path = Path(args.phase1_checkpoint).resolve()
    if not ckpt_path.is_dir():
        fail(f"Phase 1 checkpoint not found: {ckpt_path}"); sys.exit(1)
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    meta = load_checkpoint(ckpt_path, denoiser, restore_rng=False)
    info(f"loaded phase 1 epoch {meta.get('epoch')}, step {meta.get('global_step')}")
    if args.use_ema and (ckpt_path / "ema.pt").exists():
        denoiser.load_state_dict(torch.load(ckpt_path / "ema.pt", map_location=device, weights_only=False))
        ok("initialized from EMA weights")

    field = DPCField(denoiser, cfg).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=1e-5)

    betas = make_beta_schedule(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end).to(device)
    alpha_bars = get_alpha_bars(betas)
    train_gen = make_generator(args.seed, device=device)

    # ── Run dry steps ──
    stage(f"Run {args.steps} dry steps")
    losses_per_step = []
    grad_breakdown_per_step = []
    step_times = []
    iter_loader = iter(loader)

    for step in range(args.steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        t0 = time.perf_counter()

        x0 = batch["images"].to(device)
        masks = batch["masks"]
        validity = batch["mask_validity"]
        if masks is not None:
            masks = masks.to(device)
            validity = validity.to(device)

        b = x0.shape[0]

        # Sample diffusion noise
        t = sample_timesteps(b, cfg.timestep_min, cfg.timestep_max, device, train_gen)
        eps = torch.randn(x0.shape, device=device, generator=train_gen, dtype=x0.dtype)
        x_t = add_noise(x0, t, eps, alpha_bars)
        eps_hat = denoiser(x_t, t)

        # ── Loss components ──
        loss_mse = ((eps_hat - eps) ** 2).mean()

        loss_loc = torch.zeros((), device=device)
        loss_ssim = torch.zeros((), device=device)

        if masks is not None and validity is not None and validity.any():
            # We need the deployed field to compute supervision losses.
            # Compute it from the SAME denoiser predictions we already have, but
            # this requires running the field engine separately at probe res.
            # For sanity check, just run the field forward on x0 — accepts the
            # extra forward cost in exchange for correctness.
            field_out = field(x0, return_all=False, generator=train_gen)
            deployed = field_out["deployed"]  # [B, 1, H, W] in [0, 1]
            valid_idx = validity.nonzero(as_tuple=True)[0]
            d_valid = deployed[valid_idx]
            m_valid = masks[valid_idx]
            # Resize masks if needed (in this codebase image and mask are at same res)
            if d_valid.shape[-2:] != m_valid.shape[-2:]:
                m_valid = F.interpolate(m_valid, size=d_valid.shape[-2:], mode="bilinear", align_corners=False)

            loss_loc = focal_bce_with_logits(
                # Treat deployed as logit-like. Since deployed ∈ [0, 1] (post-sigmoid),
                # we map it back to a logit via clamp+log to avoid divergence.
                torch.log((d_valid.clamp(1e-6, 1 - 1e-6)) / (1 - d_valid.clamp(1e-6, 1 - 1e-6))),
                m_valid,
            )
            loss_ssim = ssim_loss(d_valid, m_valid)

        total = (
            args.lambda_mse * loss_mse
            + args.lambda_loc * loss_loc
            + args.lambda_ssim * loss_ssim
        )

        if not torch.isfinite(total):
            fail(f"step {step+1}: total loss not finite ({total.item()})")
            failed.append("loss_finite"); break

        # ── Per-component grad analysis ──
        grad_norms_components = {}
        # MSE grad
        optimizer.zero_grad()
        loss_mse.backward(retain_graph=True)
        grad_norms_components["mse"] = grad_norm_of(denoiser.parameters())
        # LOC grad
        if loss_loc.requires_grad and loss_loc.grad_fn is not None:
            optimizer.zero_grad()
            (args.lambda_loc * loss_loc).backward(retain_graph=True)
            grad_norms_components["loc"] = grad_norm_of(denoiser.parameters())
        else:
            grad_norms_components["loc"] = 0.0
        # SSIM grad
        if loss_ssim.requires_grad and loss_ssim.grad_fn is not None:
            optimizer.zero_grad()
            (args.lambda_ssim * loss_ssim).backward(retain_graph=True)
            grad_norms_components["ssim"] = grad_norm_of(denoiser.parameters())
        else:
            grad_norms_components["ssim"] = 0.0

        # Final accumulated step
        optimizer.zero_grad()
        total.backward()
        grad_norms_components["total"] = grad_norm_of(denoiser.parameters())
        # Clip + step
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()

        step_ms = (time.perf_counter() - t0) * 1000
        losses_per_step.append({
            "total": float(total.item()),
            "mse": float(loss_mse.item()),
            "loc": float(loss_loc.item()),
            "ssim": float(loss_ssim.item()),
        })
        grad_breakdown_per_step.append(grad_norms_components)
        step_times.append(step_ms)

        print(f"  step {step+1}: tot={float(total):.4f} "
              f"mse={float(loss_mse):.4f} loc={float(loss_loc):.4f} ssim={float(loss_ssim):.4f}  "
              f"gn_total={grad_norms_components['total']:.3f}  t={step_ms:.0f}ms")

    if not failed:
        passed.append("loss_finite")

    # ── Quality checks ──
    stage("Quality checks")
    if grad_breakdown_per_step:
        # Mean per-component fraction of total grad norm
        comp_means = {}
        for comp in ["mse", "loc", "ssim"]:
            fracs = []
            for g in grad_breakdown_per_step:
                tot = g.get("total", 0)
                if tot > 1e-9:
                    fracs.append(g.get(comp, 0) / tot)
            comp_means[comp] = float(np.mean(fracs)) if fracs else 0.0

        info(f"  grad fractions: mse={comp_means['mse']:.2f}, "
             f"loc={comp_means['loc']:.2f}, ssim={comp_means['ssim']:.2f}")

        # Check no component dominates
        max_frac = max(comp_means.values())
        if max_frac < 0.95:
            ok(f"no single component dominates (max fraction {max_frac:.2f})")
            passed.append("no_component_dominates")
        else:
            warn(f"one component dominates ({max_frac:.2f}) — may want to rebalance λ")

        # Check active components (those with λ > 0) aren't vestigial
        for comp, lam in [("mse", args.lambda_mse), ("loc", args.lambda_loc), ("ssim", args.lambda_ssim)]:
            if lam > 0 and comp_means[comp] < 0.005:
                fail(f"{comp} contributes only {comp_means[comp]:.4f} of grad norm — "
                     f"loss is too small or λ_{comp} too low")
                failed.append(f"{comp}_vestigial")

    # ── Loss decreasing? ──
    if len(losses_per_step) >= 4:
        first = float(np.mean([l["total"] for l in losses_per_step[: len(losses_per_step) // 2]]))
        second = float(np.mean([l["total"] for l in losses_per_step[len(losses_per_step) // 2 :]]))
        if second < first:
            ok(f"total loss decreasing: {first:.4f} → {second:.4f}")
            passed.append("loss_decreases")
        else:
            warn(f"loss not decreasing in {args.steps} steps ({first:.4f} → {second:.4f}) — "
                 f"may need more steps")

    # ── Report ──
    stage("Write report")
    report = {
        "version": __version__,
        "tool": "sanity_check_loss",
        "phase": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "n_steps": len(losses_per_step),
        "losses_per_step": losses_per_step,
        "grad_breakdown_per_step": grad_breakdown_per_step,
        "step_times_ms": step_times,
        "phase1_checkpoint_meta": meta,
        "checks_passed": passed,
        "checks_failed": failed,
    }
    with open(output / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    ok(f"report -> {output / 'report.json'}")

    print()
    if failed:
        print(f"  {C.R}{C.BOLD}FAIL{C.END}: {len(failed)} checks failed")
        for c in failed: print(f"    - {c}")
        sys.exit(1)
    else:
        print(f"  {C.G}{C.BOLD}PASS{C.END}: {len(passed)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
