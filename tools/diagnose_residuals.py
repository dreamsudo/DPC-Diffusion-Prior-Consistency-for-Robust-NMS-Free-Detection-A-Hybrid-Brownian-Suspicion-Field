#!/usr/bin/env python3
"""Residual ratio diagnostic on the full 873-image APRICOT eval set.

This is THE Phase 1 success metric:
  - Median inside-over-outside residual ratio across 873 images
  - Distribution binned at 5x / 3x / 2x / 1.5x / 1x thresholds
  - Probe-randomness variance (run twice with different probe seeds)

Usage:
    python tools/diagnose_residuals.py \
        --checkpoint runs/phase1_seed42/checkpoints/best \
        --apricot-eval-cache caches/apricot_eval_640.pt \
        --use-ema \
        --seed 42 \
        --output-dir runs/phase1_seed42/diagnostic_873
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from dpc._version import __version__
from dpc.checkpoint import load_checkpoint
from dpc.config import DPCConfig
from dpc.data_cache import TensorCache
from dpc.denoiser import TinyUNetDenoiser
from dpc.field import DPCField
from dpc.manifest import fingerprint_environment
from dpc.metrics import (
    aggregate_residual_distribution,
    bootstrap_ci,
    probe_randomness_delta,
    residual_ratio_per_image,
)
from dpc.seeding import make_generator, set_global_seed


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def warn(m): print(f"  {C.Y}!{C.END} {m}")


def build_mask(bboxes_xyxy: list, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Rasterize a list of [x1, y1, x2, y2] bboxes into a binary [H, W] mask."""
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    for x1, y1, x2, y2 in bboxes_xyxy:
        ix1 = max(0, int(round(x1)))
        iy1 = max(0, int(round(y1)))
        ix2 = min(W, int(round(x2)))
        iy2 = min(H, int(round(y2)))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 1.0
    return mask


def run_diagnostic_pass(
    field: DPCField,
    cache: TensorCache,
    device: torch.device,
    probe_seed: int,
    n_images: int | None = None,
    progress_every: int = 100,
) -> list[dict]:
    """One pass of the diagnostic. Returns list of per-image dicts.

    The cache may be at a different resolution than the denoiser's probe_res;
    DPCField handles the down/upsample. We compute the inside-over-outside
    ratio at the cache's native resolution, where the bboxes live.
    """
    field.eval()
    n = n_images if n_images is not None else len(cache)
    n = min(n, len(cache))
    per_image: list[dict] = []
    gen = make_generator(probe_seed, device=device)

    with torch.no_grad():
        for i in range(n):
            item = cache[i]
            img = item["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
            bboxes = item["metadata"].get("bboxes_xyxy", [])
            H, W = img.shape[-2:]

            mask = build_mask(bboxes, H, W, device)
            if mask.sum() < 1 or mask.sum() >= H * W:
                # No valid mask — record a degenerate entry
                per_image.append({
                    "name": Path(item["path"]).name,
                    "is_degenerate": True,
                    "degenerate_reason": "no_valid_bbox",
                    "ratio": None,
                })
                continue

            raw = field.compute_raw_signal(img, generator=gen)
            hybrid = raw["hybrid"][0, 0]  # [H, W]

            stats = residual_ratio_per_image(hybrid, mask)
            stats["name"] = Path(item["path"]).name
            stats["raw_max"] = float(hybrid.max().item())
            stats["raw_mean"] = float(hybrid.mean().item())
            per_image.append(stats)

            if (i + 1) % progress_every == 0:
                info(f"  processed {i+1}/{n}")

    return per_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--apricot-eval-cache", required=True)
    p.add_argument("--use-ema", action="store_true",
                   help="Use EMA weights from checkpoint (recommended)")
    p.add_argument("--num-images", type=int, default=None,
                   help="Limit images for quick testing (default: all)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rerun-probe-seed", type=int, default=43,
                   help="Second probe seed for between-run variance")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap CI iterations")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    print(f"{C.BOLD}DPC Residual Diagnostic v{__version__}{C.END}")
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    info(f"device: {device}")

    # ─── Load checkpoint ────────────────────────────────────────────────────
    stage("Load denoiser")
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_dir():
        print(f"  {C.R}✗{C.END} checkpoint dir not found: {ckpt_path}")
        sys.exit(1)
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    meta = load_checkpoint(ckpt_path, denoiser, restore_rng=False)
    ok(f"loaded epoch {meta.get('epoch')}, step {meta.get('global_step')}")
    info(f"checkpoint version: {meta.get('version', 'unknown')}")

    if args.use_ema:
        ema_path = ckpt_path / "ema.pt"
        if ema_path.exists():
            ema_state = torch.load(ema_path, map_location=device, weights_only=False)
            denoiser.load_state_dict(ema_state)
            ok("using EMA weights")
        else:
            warn("EMA weights requested but ema.pt not in checkpoint")

    denoiser.eval()
    field = DPCField(denoiser, cfg).to(device)

    # ─── Load eval cache ────────────────────────────────────────────────────
    stage("Load APRICOT eval cache")
    cache = TensorCache(Path(args.apricot_eval_cache).resolve())
    n_total = len(cache)
    info(f"cache: {n_total} images at {cache.resolution}")

    # ─── Pass 1 ──────────────────────────────────────────────────────────────
    stage(f"Pass 1: probe seed {args.seed}")
    per_image_a = run_diagnostic_pass(
        field, cache, device, probe_seed=args.seed, n_images=args.num_images,
    )

    # ─── Pass 2 ──────────────────────────────────────────────────────────────
    stage(f"Pass 2: probe seed {args.rerun_probe_seed} (between-run variance)")
    per_image_b = run_diagnostic_pass(
        field, cache, device, probe_seed=args.rerun_probe_seed, n_images=args.num_images,
    )

    # ─── Aggregate ───────────────────────────────────────────────────────────
    stage("Aggregate")
    agg_a = aggregate_residual_distribution(per_image_a)
    delta = probe_randomness_delta(per_image_a, per_image_b)

    # Bootstrap CI on median ratio
    valid_ratios = np.array(
        [s["ratio"] for s in per_image_a if s.get("ratio") is not None],
        dtype=np.float64,
    )
    if len(valid_ratios) > 0:
        ci_med_lo, ci_med_hi = bootstrap_ci(
            valid_ratios, stat_fn=np.median, n_boot=args.n_boot, seed=args.seed,
        )
        ci_mean_lo, ci_mean_hi = bootstrap_ci(
            valid_ratios, stat_fn=np.mean, n_boot=args.n_boot, seed=args.seed,
        )
    else:
        ci_med_lo = ci_med_hi = ci_mean_lo = ci_mean_hi = float("nan")

    print()
    print(f"  {C.BOLD}n_total:        {agg_a['n_total']}{C.END}")
    print(f"  n_valid:        {agg_a['n_valid']}")
    print(f"  n_degenerate:   {agg_a['n_degenerate']}")
    print(f"  n_positive:     {agg_a['n_positive']}")
    print(f"  median_ratio:   {agg_a['median_ratio']:.3f}  (95% CI: [{ci_med_lo:.3f}, {ci_med_hi:.3f}])")
    print(f"  mean_ratio:     {agg_a['mean_ratio']:.3f}  (95% CI: [{ci_mean_lo:.3f}, {ci_mean_hi:.3f}])")
    print(f"  std_ratio:      {agg_a['std_ratio']:.3f}")
    print(f"  min/max:        {agg_a['min_ratio']:.3f} / {agg_a['max_ratio']:.3f}")
    print()
    print(f"  Bins:")
    _n_valid = max(1, agg_a.get("n_valid", 0))
    for bin_name, count in agg_a["bins"].items():
        pct = 100.0 * count / _n_valid
        print(f"    {bin_name:<22} {count:>4}  ({pct:.1f}%)")
    print()
    print(f"  Probe-randomness delta:")
    print(f"    median delta:      {delta['median_delta']}")
    print(f"    max abs delta:     {delta['max_abs_delta']}")
    print(f"    mean abs delta:    {delta['mean_abs_delta']}")

    # ─── Save ───────────────────────────────────────────────────────────────
    stage("Save outputs")

    # per_image.json
    per_image_path = output_dir / "per_image.json"
    with open(per_image_path, "w") as f:
        json.dump(per_image_a, f, indent=2)
    ok(f"per_image -> {per_image_path}")

    # summary.json
    summary = {
        "version": __version__,
        "tool": "diagnose_residuals",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "checkpoint_meta": meta,
        "n_probes": cfg.n_probes,
        "probe_res": cfg.probe_res,
        "input_res": list(cache.resolution),
        "timestep_min": cfg.timestep_min,
        "timestep_max": cfg.timestep_max,
        "use_ema": args.use_ema,
        "primary_seed": args.seed,
        "rerun_probe_seed": args.rerun_probe_seed,
        "aggregate": {
            **agg_a,
            "ci95_median": [ci_med_lo, ci_med_hi],
            "ci95_mean": [ci_mean_lo, ci_mean_hi],
        },
        "between_run_delta": delta,

    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    ok(f"summary -> {summary_path}")

    # Histogram plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(valid_ratios) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            bins_x = np.concatenate([np.linspace(0, 5, 21), np.linspace(5, 25, 11)[1:]])
            ax.hist(valid_ratios.clip(0, 25), bins=bins_x,
                    color="steelblue", edgecolor="black", alpha=0.85)
            ax.axvline(1.0, color="black", linestyle=":", label="ratio = 1.0")
            ax.axvline(agg_a["median_ratio"], color="green", linestyle="-",
                       linewidth=2, label=f"median = {agg_a['median_ratio']:.3f}")
            ax.axvline(1.2, color="red", linestyle="--", alpha=0.5,
                       label="S1 threshold 1.2")
            ax.set_xlabel("inside / outside residual ratio")
            ax.set_ylabel("count")
            ax.set_title(f"Residual ratio distribution (n={agg_a['n_valid']})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "histogram.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            ok(f"histogram -> {output_dir / 'histogram.png'}")
    except Exception as e:
        warn(f"histogram render failed: {e}")
    sys.exit(0)


if __name__ == "__main__":
    main()
