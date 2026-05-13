#!/usr/bin/env python3
"""Fit HSV joint distribution from real APRICOT patch regions.

ONE-TIME setup. Output is consumed by SyntheticPatchGenerator so synthetic
patches match the COLOR profile of real APRICOT patches without containing
any actual APRICOT pixels (mistake #17 prevented).

Process:
  1. Load APRICOT train cache (with bbox metadata).
  2. For each image, extract pixels INSIDE bboxes only.
  3. Convert RGB → HSV.
  4. Build a 3D joint histogram in [0,1]^3 with `--n-bins` bins per axis.
  5. Save to caches/color_distribution.pt.

Usage:
    python tools/fit_color_distribution.py \
        --apricot-cache caches/apricot_train_128.pt \
        --output caches/color_distribution.pt \
        --n-bins 16
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from dpc._version import __version__
from dpc.data_cache import TensorCache
from dpc.manifest import fingerprint_environment


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")


def rgb_to_hsv_batch(rgb: np.ndarray) -> np.ndarray:
    """rgb: [..., 3] in [0, 1]. Returns hsv [..., 3] in [0, 1]."""
    r = rgb[..., 0]; g = rgb[..., 1]; b = rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    nz = delta > 1e-9
    # max = r
    mask_r = nz & (cmax == r)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    # max = g
    mask_g = nz & (cmax == g) & (~mask_r)
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    # max = b
    mask_b = nz & (cmax == b) & (~mask_r) & (~mask_g)
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    h = h / 6.0  # to [0, 1]

    s = np.where(cmax > 1e-9, delta / np.maximum(cmax, 1e-9), 0.0)
    v = cmax
    return np.stack([h, s, v], axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apricot-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-bins", type=int, default=16)
    p.add_argument("--max-pixels-per-image", type=int, default=8192,
                   help="Subsample pixels inside each bbox to keep memory bounded")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"{C.BOLD}DPC Color Distribution Fitter v{__version__}{C.END}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    stage("Load APRICOT cache")
    cache = TensorCache(Path(args.apricot_cache).resolve())
    info(f"cache: {len(cache)} images at {cache.resolution}")

    stage(f"Extract patch pixels (n_bins={args.n_bins})")
    rs = np.random.RandomState(args.seed)
    nb = args.n_bins
    hist = np.zeros((nb, nb, nb), dtype=np.float64)
    n_pixels_total = 0
    n_images_with_bbox = 0

    for i in range(len(cache)):
        item = cache[i]
        bboxes = item["metadata"].get("bboxes_xyxy", [])
        if not bboxes:
            continue
        n_images_with_bbox += 1

        img = item["image"].numpy()  # [3, H, W] in [0, 1]
        H, W = img.shape[-2:]

        # Extract patch pixels
        patch_pixels = []
        for x1, y1, x2, y2 in bboxes:
            ix1 = max(0, int(round(x1))); iy1 = max(0, int(round(y1)))
            ix2 = min(W, int(round(x2))); iy2 = min(H, int(round(y2)))
            if ix2 > ix1 and iy2 > iy1:
                # img[c, y, x]; we want pixels [3, h*w] then transpose to [h*w, 3]
                crop = img[:, iy1:iy2, ix1:ix2]
                pix = crop.reshape(3, -1).T  # [h*w, 3]
                patch_pixels.append(pix)

        if not patch_pixels:
            continue
        all_pix = np.concatenate(patch_pixels, axis=0)
        if all_pix.shape[0] > args.max_pixels_per_image:
            keep = rs.choice(all_pix.shape[0], size=args.max_pixels_per_image, replace=False)
            all_pix = all_pix[keep]

        hsv = rgb_to_hsv_batch(all_pix)  # [N, 3] in [0, 1]
        # Bin
        idx_h = np.clip((hsv[:, 0] * nb).astype(int), 0, nb - 1)
        idx_s = np.clip((hsv[:, 1] * nb).astype(int), 0, nb - 1)
        idx_v = np.clip((hsv[:, 2] * nb).astype(int), 0, nb - 1)
        for hh, ss, vv in zip(idx_h, idx_s, idx_v):
            hist[hh, ss, vv] += 1.0
        n_pixels_total += all_pix.shape[0]

        if (i + 1) % 100 == 0:
            info(f"  done {i+1}/{len(cache)}, pixels so far: {n_pixels_total:,}")

    if n_images_with_bbox == 0:
        print(f"  {C.R}✗{C.END} no images with bboxes — cannot fit color distribution")
        sys.exit(1)

    info(f"total pixels sampled: {n_pixels_total:,} from {n_images_with_bbox} images")

    # Normalize
    total = hist.sum()
    if total > 0:
        hist_norm = hist / total
    else:
        hist_norm = hist
    h_marg = hist_norm.sum(axis=(1, 2))
    s_marg = hist_norm.sum(axis=(0, 2))
    v_marg = hist_norm.sum(axis=(0, 1))

    stage("Save")
    bin_edges = np.linspace(0, 1, nb + 1, dtype=np.float32)
    payload = {
        "version": __version__,
        "tool": "fit_color_distribution",
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "n_bins": nb,
        "h_bins": torch.from_numpy(bin_edges),
        "s_bins": torch.from_numpy(bin_edges),
        "v_bins": torch.from_numpy(bin_edges),
        "joint_hist": torch.from_numpy(hist_norm.astype(np.float32)),
        "marginal_h": torch.from_numpy(h_marg.astype(np.float32)),
        "marginal_s": torch.from_numpy(s_marg.astype(np.float32)),
        "marginal_v": torch.from_numpy(v_marg.astype(np.float32)),
        "n_patches_sampled": int(n_images_with_bbox),
        "n_pixels_sampled": int(n_pixels_total),
        "source_cache": str(Path(args.apricot_cache).resolve()),
    }
    torch.save(payload, output)
    ok(f"wrote {output} (~{output.stat().st_size / 1024:.1f} KB)")

    # Summary
    print()
    print(f"  {C.BOLD}Marginal hue distribution (top 5 bins):{C.END}")
    top_h = np.argsort(h_marg)[::-1][:5]
    for hi in top_h:
        bin_lo = bin_edges[hi]; bin_hi = bin_edges[hi + 1]
        print(f"    h ∈ [{bin_lo:.2f}, {bin_hi:.2f}]: {h_marg[hi]:.3f}")
    print(f"  {C.BOLD}Saturation:{C.END} median bin = {int(np.argmax(np.cumsum(s_marg) >= 0.5))} of {nb}")
    print(f"  {C.BOLD}Value:{C.END}      median bin = {int(np.argmax(np.cumsum(v_marg) >= 0.5))} of {nb}")


if __name__ == "__main__":
    main()
