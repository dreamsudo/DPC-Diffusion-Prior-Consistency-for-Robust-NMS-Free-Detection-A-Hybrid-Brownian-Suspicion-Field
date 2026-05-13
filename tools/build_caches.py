#!/usr/bin/env python3
"""Build TensorCache files for COCO and APRICOT.

ONE-TIME setup. Run once after cloning the repo and locating datasets.
After this, training and diagnostic scripts read from cache only — no JPEG
decode in any training loop (mistakes #9, #10 prevented).

Usage:
    python tools/build_caches.py \
        --coco-train ../datasets/coco/train2017 \
        --coco-val ../datasets/coco/val2017 \
        --apricot-base ../datasets/APRICOT/APRICOTv1.0 \
        --probe-res 128 \
        --eval-res 640 \
        --output caches/

Outputs (in caches/):
    coco_train2017_128.pt
    coco_val2017_128.pt
    apricot_train_128.pt        # 786 train images at probe res
    apricot_val_128.pt          # 87 val images at probe res
    apricot_eval_640.pt         # all 873 at YOLO inference res for Phase 3
    manifest.json               # SHA256 of every cache file

Wall time on M1 Max with 8 workers: ~10-15 min for COCO, <1 min for APRICOT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpc._version import __version__
from dpc.data_cache import (
    TensorCache,
    apricot_metadata_fn_factory,
    coco_metadata_fn_factory,
    _sha256_file,
)
from dpc.manifest import fingerprint_environment
from dpc.seeding import deterministic_split


# Color codes
class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def warn(m): print(f"  {C.Y}!{C.END} {m}")
def fail(m): print(f"  {C.R}✗{C.END} {m}"); sys.exit(1)


def find_apricot_split(apricot_base: Path):
    """Return (annotations_json, images_dir, split_name) for whichever split
    has both annotations and images present on disk.

    APRICOT v1.0 ships dev and test splits but local copies often only have
    one split with images. We use whichever has images.
    """
    for split in ["test", "dev"]:
        ann = apricot_base / "Annotations" / f"coco_apricot_annotations_{split}.json"
        img_dir = apricot_base / "Images" / split
        if ann.is_file() and img_dir.is_dir() and any(img_dir.glob("*.jpg")):
            return ann, img_dir, split
    raise RuntimeError(f"No usable APRICOT split in {apricot_base}")


def build_coco_cache(image_dir: Path, output_path: Path, resolution: tuple[int, int],
                    num_workers: int, seed: int, metadata_fn=None) -> dict:
    """Build a COCO cache. If metadata_fn is provided, per-image metadata
    (e.g. ground-truth bboxes and classes) is attached to each cache entry."""
    if output_path.exists():
        warn(f"{output_path.name} exists; overwriting")
    TensorCache.build_from_directory(
        image_dir=image_dir,
        output_path=output_path,
        resolution=resolution,
        num_workers=num_workers,
        metadata_fn=metadata_fn,
        seed=seed,
        progress=True,
    )
    sha = _sha256_file(output_path)
    return {
        "name": output_path.name,
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha,
        "resolution": list(resolution),
    }


def build_apricot_cache(
    annotation_json: Path,
    image_dir: Path,
    output_path: Path,
    resolution: tuple[int, int],
    num_workers: int,
    seed: int,
    indices: list[int] | None = None,
    label: str = "apricot",
) -> dict:
    """Build APRICOT cache with bbox metadata. If `indices` given, restricts to
    those indices in the sorted-by-image-id order.

    The metadata_fn callback expects to be called with full image paths in the
    image_dir. Indices are applied AFTER the cache is built by re-saving a
    filtered version. (Cleaner than threading subsetting through the worker pool.)
    """
    if output_path.exists():
        warn(f"{output_path.name} exists; overwriting")

    metadata_fn = apricot_metadata_fn_factory(
        annotation_json_path=annotation_json,
        image_root=image_dir,
        target_resolution=resolution,
    )
    TensorCache.build_from_directory(
        image_dir=image_dir,
        output_path=output_path,
        resolution=resolution,
        num_workers=num_workers,
        metadata_fn=metadata_fn,
        seed=seed,
        progress=True,
    )

    # If indices were specified, re-load and filter
    if indices is not None:
        info(f"filtering {label} cache to {len(indices)} indices")
        import torch
        data = torch.load(output_path, map_location="cpu", weights_only=False)
        keep = sorted(set(indices))
        data["images"] = data["images"][keep]
        data["paths"] = [data["paths"][i] for i in keep]
        data["metadata"] = [data["metadata"][i] for i in keep]
        data["n_images"] = len(keep)
        torch.save(data, output_path)

    sha = _sha256_file(output_path)
    return {
        "name": output_path.name,
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha,
        "resolution": list(resolution),
    }


def main():
    p = argparse.ArgumentParser(description="Build TensorCache files for DPC")
    p.add_argument("--coco-train", type=str, required=True,
                   help="Path to COCO train2017 directory")
    p.add_argument("--coco-annotations", type=str, default=None,
                   help="Path to instances_train2017.json; if set, GT boxes "
                        "and classes are attached to each image's metadata")
    p.add_argument("--coco-val", type=str, default=None,
                   help="Path to COCO val2017 directory (optional)")
    p.add_argument("--apricot-base", type=str, required=True,
                   help="Path to APRICOTv1.0 directory (with Annotations/ and Images/)")
    p.add_argument("--probe-res", type=int, default=128,
                   help="Resolution for training/diagnostic caches")
    p.add_argument("--eval-res", type=int, default=640,
                   help="Resolution for Phase 3 evaluation cache")
    p.add_argument("--apricot-train-frac", type=float, default=0.90,
                   help="Fraction of APRICOT to use as train (rest is val)")
    p.add_argument("--output", type=str, default="caches",
                   help="Output directory for caches")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="-1 = os.cpu_count() - 2; 0 = single-threaded")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for APRICOT train/val split")
    p.add_argument("--skip-coco", action="store_true",
                   help="Skip COCO cache build (e.g., for testing APRICOT only)")
    p.add_argument("--skip-coco-val", action="store_true",
                   help="Skip COCO val cache build")
    args = p.parse_args()

    print(f"{C.BOLD}DPC Cache Builder v{__version__}{C.END}")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info(f"output: {output_dir}")
    info(f"workers: {args.num_workers}")

    if args.num_workers < 0:
        actual_workers = max(1, (os.cpu_count() or 4) - 2)
        info(f"resolved workers: {actual_workers} (os.cpu_count={os.cpu_count()})")

    caches: dict[str, dict] = {}

    # ─── COCO train ────────────────────────────────────────────────────────
    if not args.skip_coco:
        stage("Build COCO train cache")
        coco_train_dir = Path(args.coco_train).resolve()
        if not coco_train_dir.is_dir():
            fail(f"COCO train dir not found: {coco_train_dir}")
        coco_train_path = output_dir / f"coco_train2017_{args.probe_res}.pt"
        coco_meta_fn = None
        if args.coco_annotations:
            ann_path = Path(args.coco_annotations).resolve()
            if not ann_path.is_file():
                fail(f"COCO annotations not found: {ann_path}")
            info(f"loading COCO annotations from {ann_path.name}")
            coco_meta_fn = coco_metadata_fn_factory(
                annotation_json_path=ann_path,
                image_root=coco_train_dir,
                target_resolution=(args.probe_res, args.probe_res),
            )
        meta = build_coco_cache(
            coco_train_dir, coco_train_path,
            (args.probe_res, args.probe_res), args.num_workers, args.seed,
            metadata_fn=coco_meta_fn,
        )
        ok(f"coco_train2017: {meta['size_bytes'] / 1e9:.2f} GB, sha={meta['sha256'][:12]}...")
        caches[meta["name"]] = meta

    # ─── COCO val (optional) ───────────────────────────────────────────────
    if args.coco_val and not args.skip_coco_val:
        stage("Build COCO val cache")
        coco_val_dir = Path(args.coco_val).resolve()
        if not coco_val_dir.is_dir():
            warn(f"COCO val dir not found: {coco_val_dir}; skipping")
        else:
            coco_val_path = output_dir / f"coco_val2017_{args.probe_res}.pt"
            meta = build_coco_cache(
                coco_val_dir, coco_val_path,
                (args.probe_res, args.probe_res), args.num_workers, args.seed,
            )
            ok(f"coco_val2017: {meta['size_bytes'] / 1e9:.2f} GB")
            caches[meta["name"]] = meta

    # ─── APRICOT split + caches ────────────────────────────────────────────
    stage("Build APRICOT caches")
    apricot_base = Path(args.apricot_base).resolve()
    if not apricot_base.is_dir():
        fail(f"APRICOT base dir not found: {apricot_base}")

    ann_json, img_dir, split_name = find_apricot_split(apricot_base)
    ok(f"using APRICOT split: '{split_name}'")

    # Count images that exist on disk + have annotations
    with open(ann_json) as f:
        coco = json.load(f)
    files_with_ann = set()
    file_lookup = {img["id"]: img["file_name"] for img in coco["images"]}
    for ann in coco["annotations"]:
        fn = file_lookup.get(ann["image_id"])
        if fn and (img_dir / fn).is_file():
            files_with_ann.add(fn)
    n_total = len(files_with_ann)
    info(f"APRICOT images with annotations + on disk: {n_total}")

    # First build the FULL eval cache at eval-res (used by Phase 3 unchanged)
    stage("Build APRICOT eval cache (full 873 at eval-res)")
    apricot_eval_path = output_dir / f"apricot_eval_{args.eval_res}.pt"
    meta = build_apricot_cache(
        annotation_json=ann_json,
        image_dir=img_dir,
        output_path=apricot_eval_path,
        resolution=(args.eval_res, args.eval_res),
        num_workers=args.num_workers,
        seed=args.seed,
        label="apricot_eval",
    )
    ok(f"apricot_eval_{args.eval_res}: {meta['size_bytes'] / 1e9:.2f} GB")
    caches[meta["name"]] = meta

    # Then build the train + val caches at probe-res. These are SUBSETS of the
    # full APRICOT, split deterministically.
    stage("Build APRICOT train + val caches (probe-res, deterministic split)")
    train_idx, val_idx = deterministic_split(n_total, args.apricot_train_frac, args.seed)
    info(f"split: {len(train_idx)} train / {len(val_idx)} val (seed={args.seed})")

    # Train at probe res
    apricot_train_path = output_dir / f"apricot_train_{args.probe_res}.pt"
    meta = build_apricot_cache(
        annotation_json=ann_json,
        image_dir=img_dir,
        output_path=apricot_train_path,
        resolution=(args.probe_res, args.probe_res),
        num_workers=args.num_workers,
        seed=args.seed,
        indices=train_idx,
        label="apricot_train",
    )
    ok(f"apricot_train_{args.probe_res}: {meta['size_bytes'] / 1e6:.1f} MB")
    caches[meta["name"]] = meta

    # Val at probe res. We need a fresh build then filter — simpler than threading
    # subsetting. The build_from_directory would otherwise decode all 873 again,
    # which is wasteful. Optimize by reading from train cache that was just built
    # at probe_res, then just filtering. But the train cache only has train_idx.
    # Cleanest: rebuild from disk for val. This is fast (~30s for 87 images).
    apricot_val_path = output_dir / f"apricot_val_{args.probe_res}.pt"
    meta = build_apricot_cache(
        annotation_json=ann_json,
        image_dir=img_dir,
        output_path=apricot_val_path,
        resolution=(args.probe_res, args.probe_res),
        num_workers=args.num_workers,
        seed=args.seed,
        indices=val_idx,
        label="apricot_val",
    )
    ok(f"apricot_val_{args.probe_res}: {meta['size_bytes'] / 1e6:.1f} MB")
    caches[meta["name"]] = meta

    # ─── Manifest ──────────────────────────────────────────────────────────
    stage("Write manifest")
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "version": __version__,
        "tool": "build_caches",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "apricot_split_used": split_name,
        "apricot_split_seed": args.seed,
        "apricot_train_frac": args.apricot_train_frac,
        "apricot_n_total_with_anns": n_total,
        "apricot_n_train": len(train_idx),
        "apricot_n_val": len(val_idx),
        "caches": caches,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    ok(f"manifest written: {manifest_path}")

    # ─── Summary ───────────────────────────────────────────────────────────
    stage("Summary")
    total_bytes = sum(c["size_bytes"] for c in caches.values())
    print(f"  Built {len(caches)} caches, total {total_bytes / 1e9:.2f} GB")
    for name, meta in caches.items():
        size_str = (
            f"{meta['size_bytes'] / 1e9:.2f} GB"
            if meta["size_bytes"] > 1e9
            else f"{meta['size_bytes'] / 1e6:.1f} MB"
        )
        print(f"    {name}: {size_str}")


if __name__ == "__main__":
    main()
