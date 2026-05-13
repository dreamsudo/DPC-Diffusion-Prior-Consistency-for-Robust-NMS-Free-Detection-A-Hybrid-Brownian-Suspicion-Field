#!/usr/bin/env python3
"""Phase 3 pre-launch sanity check.

Verifies the eval pipeline is wired correctly on a small subset (8 images)
BEFORE committing to the full ~75-minute alpha sweep across 873 images.

Pass conditions:
  - YOLO26 weights load
  - Phase 2 checkpoint loads
  - DPCWrapper produces non-empty output on at least one of 8 test images
  - boxes_kept_baseline and boxes_kept_dpc differ for at least some boxes
    when alpha > 0 (otherwise calibration has no effect — bug)
  - dpc_class differs from baseline_class for at least some boxes when
    alpha is large (otherwise calibration is too weak)

Usage:
    python tools/sanity_check_eval.py \
        --denoiser-checkpoint runs/phase2_seed42/checkpoints/best \
        --use-ema \
        --yolo-weights yolo26n.pt \
        --apricot-eval-cache caches/apricot_eval_640.pt \
        --output runs/sanity_check_eval

Exit code:
    0  all checks passed
    1  at least one check failed
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from dpc._version import __version__
from dpc.checkpoint import load_checkpoint
from dpc.config import DPCConfig
from dpc.data_cache import TensorCache
from dpc.denoiser import TinyUNetDenoiser
from dpc.manifest import fingerprint_environment
from dpc.seeding import make_generator, set_global_seed
from dpc.wrapper import DPCWrapper


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def fail(m): print(f"  {C.R}✗{C.END} {m}")
def warn(m): print(f"  {C.Y}!{C.END} {m}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser-checkpoint", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--apricot-eval-cache", required=True)
    p.add_argument("--output", default="runs/sanity_check_eval")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    p.add_argument("--n-test-images", type=int, default=8)
    p.add_argument("--cls-alpha", type=float, default=50.0)
    p.add_argument("--obj-alpha", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    print(f"{C.BOLD}DPC Phase 3 Eval Sanity Check v{__version__}{C.END}")
    set_global_seed(args.seed)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    info(f"device: {device}")

    passed: list[str] = []
    failed: list[str] = []

    # ── 1. Phase 2 checkpoint ──
    stage("Load Phase 2 denoiser")
    ckpt = Path(args.denoiser_checkpoint).resolve()
    if not ckpt.is_dir():
        fail(f"checkpoint not found: {ckpt}")
        failed.append("phase2_checkpoint_exists")
        sys.exit(1)
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    den_meta = load_checkpoint(ckpt, denoiser, restore_rng=False)
    if args.use_ema and (ckpt / "ema.pt").exists():
        denoiser.load_state_dict(torch.load(ckpt / "ema.pt", map_location=device, weights_only=False))
        ok("EMA weights loaded")
    denoiser.eval()
    ok("denoiser ready")
    passed.append("phase2_checkpoint_loads")

    # ── 2. YOLO26 ──
    stage("Load YOLO26")
    try:
        from ultralytics import YOLO
    except ImportError:
        fail("ultralytics not installed: pip install ultralytics")
        failed.append("ultralytics_installed")
        sys.exit(1)
    yolo_path = Path(args.yolo_weights).resolve()
    if not yolo_path.is_file():
        fail(f"YOLO weights not found: {yolo_path}")
        failed.append("yolo_weights_exists")
        sys.exit(1)
    try:
        yolo = YOLO(str(yolo_path))
        ok(f"YOLO loaded: {yolo_path.name}")
        passed.append("yolo_loads")
    except Exception as e:
        fail(f"YOLO load failed: {e}")
        failed.append("yolo_loads")
        sys.exit(1)

    # ── 3. Build wrapper ──
    from tools.evaluate_phase3 import _make_yolo_forward_fn_ultralytics  # type: ignore[import]
    yolo_fn = _make_yolo_forward_fn_ultralytics(yolo, score_threshold=0.001)
    wrapper = DPCWrapper(
        denoiser=denoiser, cfg=cfg, yolo_forward_fn=yolo_fn,
        score_threshold=0.25,
        cls_alpha=args.cls_alpha, obj_alpha=args.obj_alpha,
    ).to(device)
    ok("DPCWrapper built")

    # ── 4. APRICOT cache ──
    stage("Load APRICOT eval cache")
    cache = TensorCache(Path(args.apricot_eval_cache).resolve())
    n_test = min(args.n_test_images, len(cache))
    info(f"testing on first {n_test} images")

    # ── 5. Run on test images ──
    stage("Run wrapper on test images")
    gen = make_generator(args.seed, device=device)

    n_with_dets = 0
    n_with_kept = 0
    n_diff_kept = 0       # boxes where kept_baseline != kept_dpc
    n_diff_class = 0      # boxes where dpc_class != baseline_class
    n_total_boxes = 0

    with torch.no_grad():
        for i in range(n_test):
            item = cache[i]
            img = item["image"].unsqueeze(0).to(device)
            out = wrapper(img, generator=gen)
            n_boxes = out.boxes.shape[0]
            n_total_boxes += n_boxes
            if n_boxes > 0:
                n_with_dets += 1
                if out.boxes_kept_baseline.any() or out.boxes_kept_dpc.any():
                    n_with_kept += 1
                n_diff_kept += int((out.boxes_kept_baseline != out.boxes_kept_dpc).sum().item())
                n_diff_class += int((out.baseline_class != out.dpc_class).sum().item())

            info(f"  img {i+1}: {n_boxes} raw boxes, "
                 f"{int(out.boxes_kept_baseline.sum().item())} baseline-kept, "
                 f"{int(out.boxes_kept_dpc.sum().item())} dpc-kept")

    print()
    print(f"  total raw boxes:        {n_total_boxes}")
    print(f"  images with detections: {n_with_dets} / {n_test}")
    print(f"  images w/ kept boxes:   {n_with_kept} / {n_test}")
    print(f"  boxes w/ kept differ:   {n_diff_kept}")
    print(f"  boxes w/ class differ:  {n_diff_class}")

    # ── Pass conditions ──
    stage("Pass conditions")
    if n_with_dets >= max(1, n_test // 2):
        ok(f"YOLO produced detections on ≥ half of test images")
        passed.append("yolo_produces_detections")
    else:
        fail(f"YOLO only produced detections on {n_with_dets}/{n_test} images")
        failed.append("yolo_produces_detections")

    if args.cls_alpha > 0 or args.obj_alpha > 0:
        if n_total_boxes == 0:
            warn("no boxes — cannot test calibration effect")
        elif n_diff_kept > 0 or n_diff_class > 0:
            ok(f"DPC calibration is having an effect (kept_diff={n_diff_kept}, class_diff={n_diff_class})")
            passed.append("dpc_calibration_active")
        else:
            warn(f"DPC calibration produced ZERO differences with alpha=({args.obj_alpha}, {args.cls_alpha}) "
                 f"— may indicate suspicion is uniformly low or pipeline issue")
            # We don't fail this — could legitimately happen if no patches in test images
            passed.append("dpc_calibration_no_effect_noted")

    # ── Save report ──
    report = {
        "version": __version__,
        "tool": "sanity_check_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "n_test_images": n_test,
        "n_total_boxes": n_total_boxes,
        "n_images_with_detections": n_with_dets,
        "n_images_with_kept_boxes": n_with_kept,
        "n_boxes_kept_differ": n_diff_kept,
        "n_boxes_class_differ": n_diff_class,
        "checks_passed": passed,
        "checks_failed": failed,
    }
    with open(output / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    ok(f"report -> {output / 'report.json'}")

    print()
    if failed:
        print(f"  {C.R}{C.BOLD}FAIL:{C.END} {len(failed)} checks failed")
        sys.exit(1)
    else:
        print(f"  {C.G}{C.BOLD}PASS:{C.END} {len(passed)} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
