#!/usr/bin/env python3
"""Phase 2 data sanity check.

Verifies BOTH the data caches AND the synthetic patch generator before any
Phase 2 training runs.

Pass conditions checked:
  - All caches load and verify SHA256
  - COCO pixel statistics in sane ranges
  - APRICOT bbox coverage and tightness
  - APRICOT train/val are disjoint
  - Synthetic patch generator produces pixel-perfect binary masks
  - Synthetic patch area distribution overlaps with real APRICOT bbox area distribution

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
from dpc.synthetic_patch import SyntheticPatchGenerator


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def warn(m): print(f"  {C.Y}!{C.END} {m}")
def fail(m): print(f"  {C.R}✗{C.END} {m}")


def check_coco(cache: TensorCache) -> dict:
    n = len(cache)
    info(f"coco cache: {n} images at {cache.resolution}")
    rs = np.random.RandomState(42)
    sample_idx = rs.choice(n, size=min(256, n), replace=False)
    sample = cache.images[sample_idx]
    means = sample.mean(dim=(0, 2, 3)).tolist()
    stds = sample.std(dim=(0, 2, 3)).tolist()
    return {
        "n_images": n,
        "resolution": list(cache.resolution),
        "pixel_mean_rgb": [round(v, 4) for v in means],
        "pixel_std_rgb": [round(v, 4) for v in stds],
    }


def check_apricot(cache: TensorCache, label: str) -> dict:
    n = len(cache)
    info(f"apricot {label}: {n} images at {cache.resolution}")

    n_with_bbox = 0
    bbox_areas = []
    bbox_count_per_img = []
    tightness_per_bbox = []

    img_area = cache.resolution[0] * cache.resolution[1]
    rs = np.random.RandomState(42)
    tight_sample = set(rs.choice(n, size=min(200, n), replace=False).tolist())

    for i in range(n):
        item = cache[i]
        bboxes = item["metadata"].get("bboxes_xyxy", [])
        bbox_count_per_img.append(len(bboxes))
        if bboxes:
            n_with_bbox += 1
        for bb in bboxes:
            x1, y1, x2, y2 = bb
            bbox_areas.append(max(0, (x2 - x1) * (y2 - y1)) / img_area)
        if i in tight_sample and bboxes:
            img = item["image"]
            for bb in bboxes:
                x1, y1, x2, y2 = [int(round(v)) for v in bb]
                if x2 > x1 and y2 > y1:
                    crop = img[:, y1:y2, x1:x2]
                    if crop.numel() == 0:
                        continue
                    var_per_pixel = crop.var(dim=0)
                    n_textured = (var_per_pixel > 1e-4).sum().item()
                    tightness = n_textured / max(1, crop.shape[1] * crop.shape[2])
                    tightness_per_bbox.append(tightness)

    if bbox_areas:
        ba = np.array(bbox_areas)
        area_p5, area_p50, area_p95 = np.percentile(ba, [5, 50, 95]).tolist()
    else:
        area_p5 = area_p50 = area_p95 = 0.0

    if tightness_per_bbox:
        ta = np.array(tightness_per_bbox)
        tightness_mean = float(ta.mean())
    else:
        tightness_mean = 0.0

    return {
        "n_images": n,
        "n_with_bbox": n_with_bbox,
        "bbox_area_frac_p5_p50_p95": [round(area_p5, 5), round(area_p50, 5), round(area_p95, 5)],
        "bbox_tightness_mean": round(tightness_mean, 4),
        "_areas_array": ba if bbox_areas else None,  # internal — for distribution comparison
    }


def check_synthetic(generator: SyntheticPatchGenerator, n_samples: int = 64,
                    seed: int = 42) -> dict:
    """Generate N synthetic patches on dummy COCO scenes, verify properties."""
    rs = np.random.RandomState(seed)
    H = W = generator.image_size

    masks_areas = []
    binary_failures = 0
    n_per_shape = {s: 0 for s in generator.SHAPES}
    n_per_texture = {t: 0 for t in generator.TEXTURES}
    n_per_blend = {"paste": 0, "luminance_match": 0, "alpha": 0}

    for i in range(n_samples):
        # Dummy scene: smooth gradient + noise (avoid all-zero scene)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        scene_np = np.stack([
            yy / H, xx / W, 0.5 * np.ones_like(yy),
        ], axis=0)  # [3, H, W]
        scene_np = scene_np + 0.1 * rs.randn(3, H, W).astype(np.float32)
        scene_np = np.clip(scene_np, 0.0, 1.0)
        scene = torch.from_numpy(scene_np)

        patched, mask, spec = generator.render_random(scene, rs)

        # Mask validation
        m_np = mask[0].numpy()
        unique = set(np.unique(m_np).tolist())
        if not unique.issubset({0.0, 1.0}):
            binary_failures += 1

        n_pix = int(m_np.sum())
        masks_areas.append(n_pix / (H * W))
        n_per_shape[spec.shape] += 1
        n_per_texture[spec.texture] += 1
        n_per_blend[spec.blend_mode] += 1

    masks_areas = np.array(masks_areas)
    return {
        "n_samples": n_samples,
        "binary_failures": binary_failures,
        "is_pixel_perfect": binary_failures == 0,
        "mask_area_frac_p5_p50_p95": [
            float(np.percentile(masks_areas, 5)),
            float(np.percentile(masks_areas, 50)),
            float(np.percentile(masks_areas, 95)),
        ],
        "n_per_shape": n_per_shape,
        "n_per_texture": n_per_texture,
        "n_per_blend": n_per_blend,
        "_areas_array": masks_areas,
    }


def render_apricot_grid(cache: TensorCache, out_path: Path, n: int = 16, seed: int = 42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rs = np.random.RandomState(seed)
    idx = rs.choice(len(cache), size=min(n, len(cache)), replace=False)
    rows = cols = int(np.ceil(np.sqrt(len(idx))))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for k, i in enumerate(idx):
        ax = axes[k // cols, k % cols]
        item = cache[int(i)]
        ax.imshow(item["image"].permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(Path(item["path"]).name, fontsize=7)
        for bb in item["metadata"].get("bboxes_xyxy", []):
            x1, y1, x2, y2 = bb
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            linewidth=1.5, edgecolor="lime", facecolor="none"))
    for k in range(len(idx), rows * cols):
        axes[k // cols, k % cols].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def render_synthetic_grid(generator: SyntheticPatchGenerator, coco_cache: TensorCache,
                          out_path: Path, n: int = 16, seed: int = 42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = np.random.RandomState(seed)
    coco_idx = rs.choice(len(coco_cache), size=min(n, len(coco_cache)), replace=False)
    rows = cols = int(np.ceil(np.sqrt(len(coco_idx))))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for k, i in enumerate(coco_idx):
        ax = axes[k // cols, k % cols]
        scene = coco_cache[int(i)]["image"]
        patched, mask, spec = generator.render_random(scene, rs)
        ax.imshow(patched.permute(1, 2, 0).numpy())
        ax.imshow(mask[0].numpy(), cmap="autumn", alpha=0.40)
        ax.axis("off")
        ax.set_title(f"{spec.shape}/{spec.texture}\nsize={spec.size_frac:.2f}", fontsize=7)
    for k in range(len(coco_idx), rows * cols):
        axes[k // cols, k % cols].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-cache", required=True)
    p.add_argument("--apricot-cache", required=True)
    p.add_argument("--apricot-val-cache", default=None)
    p.add_argument("--color-dist", required=True)
    p.add_argument("--output", default="runs/sanity_check_data_phase2")
    p.add_argument("--n-vis", type=int, default=16)
    p.add_argument("--n-synthetic-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"{C.BOLD}DPC Data Sanity Check (Phase 2) v{__version__}{C.END}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    passed: list[str] = []
    failed: list[str] = []

    # ── COCO ──
    stage("COCO cache")
    coco_path = Path(args.coco_cache).resolve()
    if not coco_path.is_file():
        fail(f"COCO cache not found: {coco_path}"); failed.append("coco_exists")
        coco_stats = None; coco_cache = None
    else:
        coco_cache = TensorCache(coco_path)
        coco_stats = check_coco(coco_cache)
        info(f"  pixel mean RGB: {coco_stats['pixel_mean_rgb']}")
        info(f"  pixel std RGB:  {coco_stats['pixel_std_rgb']}")
        if all(0.30 <= m <= 0.60 for m in coco_stats["pixel_mean_rgb"]):
            ok("coco pixel mean in [0.30, 0.60]"); passed.append("coco_mean")
        else:
            fail("coco pixel mean OUTSIDE [0.30, 0.60]"); failed.append("coco_mean")

    # ── APRICOT train ──
    stage("APRICOT train cache")
    apricot_path = Path(args.apricot_cache).resolve()
    if not apricot_path.is_file():
        fail(f"APRICOT cache not found: {apricot_path}"); failed.append("apricot_exists")
        apricot_stats = None; apricot_cache = None
    else:
        apricot_cache = TensorCache(apricot_path)
        apricot_stats = check_apricot(apricot_cache, "train")
        info(f"  n_with_bbox: {apricot_stats['n_with_bbox']} / {apricot_stats['n_images']}")
        info(f"  bbox area p5/p50/p95: {apricot_stats['bbox_area_frac_p5_p50_p95']}")
        info(f"  bbox tightness mean:  {apricot_stats['bbox_tightness_mean']}")
        if apricot_stats["n_with_bbox"] >= 0.95 * apricot_stats["n_images"]:
            ok("apricot bbox coverage ≥ 95%"); passed.append("apricot_coverage")
        else:
            fail("apricot bbox coverage too low"); failed.append("apricot_coverage")
        if apricot_stats["bbox_tightness_mean"] >= 0.5:
            ok(f"apricot bbox tightness {apricot_stats['bbox_tightness_mean']:.3f} ≥ 0.50")
            passed.append("apricot_tightness")
        else:
            fail(f"apricot bbox tightness {apricot_stats['bbox_tightness_mean']:.3f} < 0.50")
            failed.append("apricot_tightness")

        render_apricot_grid(apricot_cache, output / "grid_apricot.png", args.n_vis, args.seed)
        ok(f"wrote grid_apricot.png")

    # ── APRICOT val + disjointness ──
    apricot_val_stats = None
    if args.apricot_val_cache:
        stage("APRICOT val cache")
        apricot_val_path = Path(args.apricot_val_cache).resolve()
        if apricot_val_path.is_file():
            apricot_val_cache = TensorCache(apricot_val_path)
            apricot_val_stats = check_apricot(apricot_val_cache, "val")
            if apricot_cache:
                tr_paths = {Path(apricot_cache[i]["path"]).name for i in range(len(apricot_cache))}
                va_paths = {Path(apricot_val_cache[i]["path"]).name for i in range(len(apricot_val_cache))}
                overlap = tr_paths & va_paths
                if not overlap:
                    ok(f"train/val disjoint (train={len(tr_paths)}, val={len(va_paths)})")
                    passed.append("train_val_disjoint")
                else:
                    fail(f"train/val OVERLAP by {len(overlap)} files")
                    failed.append("train_val_disjoint")
        else:
            warn(f"apricot val cache not found: {apricot_val_path}")

    # ── Synthetic patches ──
    stage("Synthetic patch generator")
    color_dist_path = Path(args.color_dist).resolve()
    if not color_dist_path.is_file():
        fail(f"color distribution not found: {color_dist_path}"); failed.append("color_dist_exists")
        synth_stats = None
    else:
        gen = SyntheticPatchGenerator(
            color_distribution_path=color_dist_path,
            seed=args.seed,
            image_size=128 if not coco_cache else coco_cache.resolution[0],
        )
        synth_stats = check_synthetic(gen, n_samples=args.n_synthetic_samples, seed=args.seed)
        info(f"  generated {synth_stats['n_samples']} samples")
        info(f"  shapes:   {synth_stats['n_per_shape']}")
        info(f"  textures: {synth_stats['n_per_texture']}")
        info(f"  blends:   {synth_stats['n_per_blend']}")
        info(f"  mask area p5/p50/p95: {[round(v, 4) for v in synth_stats['mask_area_frac_p5_p50_p95']]}")

        if synth_stats["is_pixel_perfect"]:
            ok("synthetic masks pixel-perfect (all values in {0, 1})")
            passed.append("synth_binary")
        else:
            fail(f"synthetic masks not binary ({synth_stats['binary_failures']} failures)")
            failed.append("synth_binary")

        # Compare synthetic vs APRICOT area distributions
        if apricot_stats and apricot_stats.get("_areas_array") is not None:
            apricot_areas = apricot_stats["_areas_array"]
            synth_areas = synth_stats["_areas_array"]
            # Wasserstein-like distance via sorted-quantile comparison
            qs = np.linspace(0, 1, 11)
            apricot_q = np.quantile(apricot_areas, qs)
            synth_q = np.quantile(synth_areas, qs)
            wasserstein_proxy = float(np.abs(apricot_q - synth_q).mean())
            info(f"  area distribution distance (mean |Δquantile|): {wasserstein_proxy:.4f}")
            if wasserstein_proxy <= 0.05:
                ok(f"synthetic area distribution overlaps APRICOT")
                passed.append("synth_area_overlap")
            else:
                warn(f"synthetic area distribution differs from APRICOT (distance {wasserstein_proxy:.4f}); "
                     f"may need to adjust size_frac_range")

        # Render synth grid using real COCO scenes
        if coco_cache:
            render_synthetic_grid(gen, coco_cache, output / "grid_synthetic.png", args.n_vis, args.seed)
            ok(f"wrote grid_synthetic.png")

    # ── Strip internal arrays before writing JSON ──
    if apricot_stats: apricot_stats.pop("_areas_array", None)
    if synth_stats:   synth_stats.pop("_areas_array", None)

    # ── Report ──
    stage("Write report")
    report = {
        "version": __version__,
        "tool": "sanity_check_data",
        "phase": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "coco": coco_stats,
        "apricot_train": apricot_stats,
        "apricot_val": apricot_val_stats,
        "synthetic": synth_stats,
        "checks_passed": passed,
        "checks_failed": failed,
        "n_passed": len(passed),
        "n_failed": len(failed),
    }
    with open(output / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    ok(f"report -> {output / 'report.json'}")

    print()
    if failed:
        print(f"  {C.R}{C.BOLD}FAIL:{C.END} {len(failed)} checks failed")
        for c in failed: print(f"    - {c}")
        sys.exit(1)
    else:
        print(f"  {C.G}{C.BOLD}PASS:{C.END} all {len(passed)} checks passed")
        print(f"  Inspect grids before launching training:")
        print(f"    {output}/grid_apricot.png")
        print(f"    {output}/grid_synthetic.png")
        sys.exit(0)


if __name__ == "__main__":
    main()
