#!/usr/bin/env python3
"""S5 — Untrained-baseline residual diagnostic.

Per sub-claim S5: a randomly-initialized TinyUNet of
the same architecture should produce a residual ratio distribution materially
worse than trained Phase 1. If it doesn't, we cannot claim our learned
natural-prior structure is responsible for the patch-localization signal — it
might just be architectural bias.

Same protocol as diagnose_residuals.py, but skips checkpoint loading: builds
a fresh TinyUNet with the supplied seed and runs it on the 873-image cache.

Pass condition (from  / S5):
  median_ratio ≤ 1.05 AND strong_signal_fraction ≤ 0.5%

Usage:
    python tools/diagnose_untrained.py \
        --apricot-eval-cache caches/apricot_eval_640.pt \
        --seed 42 \
        --output-dir runs/untrained_baseline_seed42/diagnostic_873
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
from dpc.config import DPCConfig
from dpc.data_cache import TensorCache
from dpc.denoiser import TinyUNetDenoiser
from dpc.field import DPCField
from dpc.manifest import fingerprint_environment
from dpc.metrics import (
    aggregate_residual_distribution,
    bootstrap_ci,
    residual_ratio_per_image,
)
from dpc.seeding import make_generator, set_global_seed


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")


def build_mask(bboxes_xyxy, H, W, device):
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    for x1, y1, x2, y2 in bboxes_xyxy:
        ix1 = max(0, int(round(x1))); iy1 = max(0, int(round(y1)))
        ix2 = min(W, int(round(x2))); iy2 = min(H, int(round(y2)))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 1.0
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apricot-eval-cache", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    print(f"{C.BOLD}DPC Untrained-Baseline Diagnostic v{__version__}{C.END}")
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    info(f"device: {device}")

    # ── Build untrained model ──
    stage(f"Build randomly-initialized TinyUNet with seed={args.seed}")
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    denoiser.eval()
    n_params = sum(p.numel() for p in denoiser.parameters())
    info(f"params: {n_params:,}  (NO training, NO checkpoint loaded)")
    field = DPCField(denoiser, cfg).to(device)

    # ── Load cache ──
    stage("Load APRICOT eval cache")
    cache = TensorCache(Path(args.apricot_eval_cache).resolve())
    n_total = len(cache)
    info(f"cache: {n_total} images at {cache.resolution}")

    # ── Run diagnostic ──
    stage("Run diagnostic")
    n = args.num_images if args.num_images is not None else n_total
    n = min(n, n_total)
    per_image: list[dict] = []
    gen = make_generator(args.seed, device=device)

    with torch.no_grad():
        for i in range(n):
            item = cache[i]
            img = item["image"].unsqueeze(0).to(device)
            H, W = img.shape[-2:]
            bboxes = item["metadata"].get("bboxes_xyxy", [])
            mask = build_mask(bboxes, H, W, device)
            if mask.sum() < 1 or mask.sum() >= H * W:
                per_image.append({
                    "name": Path(item["path"]).name,
                    "is_degenerate": True,
                    "ratio": None,
                })
                continue
            raw = field.compute_raw_signal(img, generator=gen)
            hybrid = raw["hybrid"][0, 0]
            stats = residual_ratio_per_image(hybrid, mask)
            stats["name"] = Path(item["path"]).name
            per_image.append(stats)
            if (i + 1) % 100 == 0:
                info(f"  {i+1}/{n}")

    # ── Aggregate ──
    stage("Aggregate")
    agg = aggregate_residual_distribution(per_image)
    valid_ratios = np.array(
        [s["ratio"] for s in per_image if s.get("ratio") is not None],
        dtype=np.float64,
    )
    if len(valid_ratios) > 0:
        ci_lo, ci_hi = bootstrap_ci(
            valid_ratios, np.median, n_boot=args.n_boot, seed=args.seed,
        )
    else:
        ci_lo = ci_hi = float("nan")

    print()
    print(f"  median_ratio:    {agg['median_ratio']:.3f}  (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
    print(f"  mean_ratio:      {agg['mean_ratio']:.3f}")
    print(f"  n_positive:      {agg['n_positive']}/{agg['n_valid']}")
    print(f"  bin ≥5x:         {agg['bins'].get('greater_than_5x', 0)} ({agg['bin_pcts'].get('greater_than_5x', 0):.1f}%)")
    print(f"  bin 1-1.5x:      {agg['bins'].get('1_to_1.5x', 0)} ({agg['bin_pcts'].get('1_to_1.5x', 0):.1f}%)")
    print(f"  bin <1:          {agg['bins'].get('less_than_1x', 0)} ({agg['bin_pcts'].get('less_than_1x', 0):.1f}%)")

    strong_signal_pct = agg["bin_pcts"].get("greater_than_5x", 0)
    # ── Save ──
    stage("Save outputs")
    summary = {
        "version": __version__,
        "tool": "diagnose_untrained",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "denoiser_params": int(n_params),
        "n_probes": cfg.n_probes,
        "probe_res": cfg.probe_res,
        "input_res": list(cache.resolution),
        "timestep_min": cfg.timestep_min,
        "timestep_max": cfg.timestep_max,
        "seed": args.seed,
        "aggregate": {**agg, "ci95_median": [ci_lo, ci_hi]},
        "S5_pass": bool(s5_pass),
        "S5_threshold_median_max": 1.05,
        "S5_threshold_strong_signal_max_pct": 0.5,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(output_dir / "per_image.json", "w") as f:
        json.dump(per_image, f, indent=2)
    ok(f"summary -> {output_dir / 'summary.json'}")

    sys.exit(0 if s5_pass else 1)


if __name__ == "__main__":
    main()
