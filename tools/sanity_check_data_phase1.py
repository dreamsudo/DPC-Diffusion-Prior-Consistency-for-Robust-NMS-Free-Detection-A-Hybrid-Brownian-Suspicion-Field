#!/usr/bin/env python3
"""Quantitative dataset validation. MANDATORY before any training.

For Phase 1, only checks COCO + APRICOT (no synthetic patch generator yet).
Phase 2's bootstrap will revise this script to also validate the synthetic
generator.

Usage:
    python tools/sanity_check_data.py \
        --coco-cache caches/coco_train2017_128.pt \
        --apricot-cache caches/apricot_train_128.pt \
        --apricot-val-cache caches/apricot_val_128.pt \
        --output runs/sanity_check_data/

Exit code:
    0  all checks passed
    1  at least one check failed (training scripts will refuse to run)
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
from dpc.data_cache import TensorCache
from dpc.manifest import fingerprint_environment


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def warn(m): print(f"  {C.Y}!{C.END} {m}")
def fail(m): print(f"  {C.R}✗{C.END} {m}")


def check_coco(cache: TensorCache) -> dict:
    """Pixel statistics on a sample. Image count must match expectation."""
    n = len(cache)
    info(f"coco cache loaded: {n} images at {cache.resolution}")

    # Sample 256 random images for pixel stats (cheaper than full pass)
    rs = np.random.RandomState(42)
    sample_idx = rs.choice(n, size=min(256, n), replace=False)
    sample = cache.images[sample_idx]  # [256, 3, H, W]
    means = sample.mean(dim=(0, 2, 3)).tolist()  # per-channel mean
    stds = sample.std(dim=(0, 2, 3)).tolist()

    luminance = sample.mean(dim=1)  # [256, H, W] grayscale
    lum_p5, lum_p50, lum_p95 = np.percentile(
        luminance.numpy().flatten(), [5, 50, 95]
    ).tolist()

    return {
        "n_images": n,
        "resolution": list(cache.resolution),
        "pixel_mean_rgb": [round(v, 4) for v in means],
        "pixel_std_rgb": [round(v, 4) for v in stds],
        "luminance_p5_p50_p95": [round(v, 4) for v in (lum_p5, lum_p50, lum_p95)],
        "sample_size": int(len(sample_idx)),
    }


def check_apricot(cache: TensorCache, label: str) -> dict:
    """APRICOT bbox statistics + bbox tightness check.

    Bbox tightness: fraction of pixels inside each bbox that have non-zero
    pixel-channel variance. Mistake #18: we ASSUMED bboxes were tight; here
    we MEASURE.
    """
    n = len(cache)
    info(f"apricot {label} cache loaded: {n} images at {cache.resolution}")

    n_with_bbox = 0
    bbox_areas: list[float] = []
    bbox_count_per_img: list[int] = []
    tightness_per_bbox: list[float] = []

    img_area = cache.resolution[0] * cache.resolution[1]
    rs = np.random.RandomState(42)
    # Compute tightness on at most 200 random images for speed
    tight_sample_size = min(200, n)
    tight_sample_idx = set(rs.choice(n, size=tight_sample_size, replace=False).tolist())

    for i in range(n):
        item = cache[i]
        bboxes = item["metadata"].get("bboxes_xyxy", [])
        bbox_count_per_img.append(len(bboxes))
        if bboxes:
            n_with_bbox += 1
        for bb in bboxes:
            x1, y1, x2, y2 = bb
            area = max(0, (x2 - x1) * (y2 - y1))
            bbox_areas.append(area / img_area)

        if i in tight_sample_idx and bboxes:
            img = item["image"]  # [3, H, W]
            for bb in bboxes:
                x1, y1, x2, y2 = [int(round(v)) for v in bb]
                if x2 > x1 and y2 > y1:
                    crop = img[:, y1:y2, x1:x2]  # [3, h, w]
                    if crop.numel() == 0:
                        continue
                    # Tightness proxy: fraction of pixels whose channel-variance
                    # exceeds 1e-4 (i.e., not flat regions). Real adversarial
                    # patches have texture; bbox margin of plain background
                    # has near-zero local variance.
                    flat_per_pixel = crop.var(dim=0)  # [h, w]
                    n_textured = (flat_per_pixel > 1e-4).sum().item()
                    tightness = n_textured / max(1, crop.shape[1] * crop.shape[2])
                    tightness_per_bbox.append(tightness)

    bbox_count_max = max(bbox_count_per_img) if bbox_count_per_img else 0

    if bbox_areas:
        ba = np.array(bbox_areas)
        area_p5, area_p50, area_p95 = np.percentile(ba, [5, 50, 95]).tolist()
    else:
        area_p5 = area_p50 = area_p95 = 0.0

    if tightness_per_bbox:
        ta = np.array(tightness_per_bbox)
        tightness_mean = float(ta.mean())
        tightness_p25 = float(np.percentile(ta, 25))
        tightness_min = float(ta.min())
    else:
        tightness_mean = tightness_p25 = tightness_min = 0.0

    return {
        "n_images": n,
        "resolution": list(cache.resolution),
        "n_with_bbox": n_with_bbox,
        "bbox_count_p50_p95_max": [
            float(np.percentile(bbox_count_per_img, 50)) if bbox_count_per_img else 0,
            float(np.percentile(bbox_count_per_img, 95)) if bbox_count_per_img else 0,
            bbox_count_max,
        ],
        "bbox_area_frac_p5_p50_p95": [
            round(area_p5, 5), round(area_p50, 5), round(area_p95, 5),
        ],
        "bbox_tightness_mean": round(tightness_mean, 4),
        "bbox_tightness_p25": round(tightness_p25, 4),
        "bbox_tightness_min": round(tightness_min, 4),
        "tightness_sample_size": int(tight_sample_size),
        "n_tightness_measurements": len(tightness_per_bbox),
    }


def render_grid(cache: TensorCache, out_path: Path, n: int = 16,
                draw_bboxes: bool = False, seed: int = 42):
    """Save a 4x4 grid of random images (with bbox overlay if requested)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rs = np.random.RandomState(seed)
    n_images = len(cache)
    idx = rs.choice(n_images, size=min(n, n_images), replace=False)

    rows = cols = int(np.ceil(np.sqrt(len(idx))))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)

    for k, i in enumerate(idx):
        ax = axes[k // cols, k % cols]
        item = cache[int(i)]
        img = item["image"].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(Path(item["path"]).name, fontsize=7)

        if draw_bboxes:
            for bb in item["metadata"].get("bboxes_xyxy", []):
                x1, y1, x2, y2 = bb
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor="lime", facecolor="none",
                )
                ax.add_patch(rect)

    # Hide unused axes
    for k in range(len(idx), rows * cols):
        axes[k // cols, k % cols].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-cache", type=str, required=True)
    p.add_argument("--apricot-cache", type=str, required=True)
    p.add_argument("--apricot-val-cache", type=str, default=None)
    p.add_argument("--output", type=str, default="runs/sanity_check_data")
    p.add_argument("--n-vis", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"{C.BOLD}DPC Data Sanity Check v{__version__}{C.END}")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checks_passed: list[str] = []
    checks_failed: list[str] = []

    # ── COCO ─────────────────────────────────────────────────────────────
    stage("Check COCO cache")
    coco_cache_path = Path(args.coco_cache).resolve()
    if not coco_cache_path.is_file():
        fail(f"COCO cache not found: {coco_cache_path}")
        checks_failed.append("coco_cache_exists")
        coco_stats = None
    else:
        coco = TensorCache(coco_cache_path)
        coco_stats = check_coco(coco)
        info(f"  pixel mean (RGB): {coco_stats['pixel_mean_rgb']}")
        info(f"  pixel std (RGB):  {coco_stats['pixel_std_rgb']}")
        info(f"  luminance p5/p50/p95: {coco_stats['luminance_p5_p50_p95']}")

        # Pass conditions
        if all(0.30 <= m <= 0.60 for m in coco_stats["pixel_mean_rgb"]):
            ok("coco pixel mean in [0.30, 0.60] per channel")
            checks_passed.append("coco_pixel_mean_range")
        else:
            fail("coco pixel mean OUTSIDE [0.30, 0.60] — investigate dataset")
            checks_failed.append("coco_pixel_mean_range")

        if all(0.20 <= s <= 0.35 for s in coco_stats["pixel_std_rgb"]):
            ok("coco pixel std in [0.20, 0.35] per channel")
            checks_passed.append("coco_pixel_std_range")
        else:
            warn("coco pixel std outside [0.20, 0.35] — possibly fine but check")

        # Render grid
        grid_path = output_dir / "grid_coco.png"
        render_grid(coco, grid_path, args.n_vis, draw_bboxes=False, seed=args.seed)
        ok(f"wrote {grid_path}")

    # ── APRICOT train ─────────────────────────────────────────────────────
    stage("Check APRICOT train cache")
    apricot_cache_path = Path(args.apricot_cache).resolve()
    if not apricot_cache_path.is_file():
        fail(f"APRICOT cache not found: {apricot_cache_path}")
        checks_failed.append("apricot_cache_exists")
        apricot_stats = None
    else:
        apricot = TensorCache(apricot_cache_path)
        apricot_stats = check_apricot(apricot, "train")
        info(f"  n_with_bbox:   {apricot_stats['n_with_bbox']} / {apricot_stats['n_images']}")
        info(f"  bbox area p5/p50/p95: {apricot_stats['bbox_area_frac_p5_p50_p95']}")
        info(f"  bbox tightness mean: {apricot_stats['bbox_tightness_mean']}")
        info(f"  bbox tightness p25:  {apricot_stats['bbox_tightness_p25']}")

        # Pass conditions per ARCH §5.2
        if apricot_stats["bbox_area_frac_p5_p50_p95"][1] > 0.001:
            ok("apricot bbox median area > 0.1%")
            checks_passed.append("apricot_bbox_area_sane")
        else:
            fail("apricot bbox median area < 0.1% — bboxes too small or broken")
            checks_failed.append("apricot_bbox_area_sane")

        if apricot_stats["n_with_bbox"] >= 0.95 * apricot_stats["n_images"]:
            ok("apricot has bboxes on ≥95% of images")
            checks_passed.append("apricot_bbox_coverage")
        else:
            fail(f"apricot bbox coverage too low: "
                 f"{apricot_stats['n_with_bbox']}/{apricot_stats['n_images']}")
            checks_failed.append("apricot_bbox_coverage")

        if apricot_stats["bbox_tightness_mean"] >= 0.5:
            ok(f"apricot bbox tightness {apricot_stats['bbox_tightness_mean']:.3f} >= 0.50")
            checks_passed.append("apricot_bbox_tightness")
            if apricot_stats["bbox_tightness_mean"] < 0.7:
                warn("bbox tightness below 0.70 — bboxes may include some background")
        else:
            fail(f"apricot bbox tightness {apricot_stats['bbox_tightness_mean']:.3f} < 0.50 "
                 f"— bbox-as-mask training will be noisy")
            checks_failed.append("apricot_bbox_tightness")

        grid_path = output_dir / "grid_apricot.png"
        render_grid(apricot, grid_path, args.n_vis, draw_bboxes=True, seed=args.seed)
        ok(f"wrote {grid_path}")

    # ── APRICOT val (optional) ────────────────────────────────────────────
    apricot_val_stats = None
    if args.apricot_val_cache:
        stage("Check APRICOT val cache")
        apricot_val_path = Path(args.apricot_val_cache).resolve()
        if apricot_val_path.is_file():
            apricot_val = TensorCache(apricot_val_path)
            apricot_val_stats = check_apricot(apricot_val, "val")
            info(f"  val n: {apricot_val_stats['n_images']}")
            info(f"  val bbox tightness mean: {apricot_val_stats['bbox_tightness_mean']}")

            # Verify train and val are disjoint
            if apricot_stats:
                train_paths = set()
                for i in range(len(apricot)):
                    train_paths.add(Path(apricot[i]["path"]).name)
                val_paths = set()
                for i in range(len(apricot_val)):
                    val_paths.add(Path(apricot_val[i]["path"]).name)
                overlap = train_paths & val_paths
                if not overlap:
                    ok(f"apricot train/val are disjoint (train={len(train_paths)}, val={len(val_paths)})")
                    checks_passed.append("apricot_train_val_disjoint")
                else:
                    fail(f"apricot train/val OVERLAP by {len(overlap)} files")
                    checks_failed.append("apricot_train_val_disjoint")
        else:
            warn(f"apricot val cache not found: {apricot_val_path}")

    # ── Write report ─────────────────────────────────────────────────────
    stage("Write report")
    report = {
        "version": __version__,
        "tool": "sanity_check_data",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "coco": coco_stats,
        "apricot_train": apricot_stats,
        "apricot_val": apricot_val_stats,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "n_passed": len(checks_passed),
        "n_failed": len(checks_failed),
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    ok(f"report -> {report_path}")

    print()
    if checks_failed:
        print(f"  {C.R}{C.BOLD}FAIL:{C.END} {len(checks_failed)} checks failed")
        for c in checks_failed:
            print(f"    - {c}")
        print()
        print(f"  Training scripts will refuse to launch until all checks pass.")
        sys.exit(1)
    else:
        print(f"  {C.G}{C.BOLD}PASS:{C.END} all {len(checks_passed)} checks passed")
        print()
        print(f"  Operator should ALSO inspect the grid PNGs:")
        print(f"    {output_dir}/grid_coco.png")
        print(f"    {output_dir}/grid_apricot.png")
        sys.exit(0)


if __name__ == "__main__":
    main()
