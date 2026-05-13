#!/usr/bin/env python3
"""End-to-end smoke test for Phase 3.

Run AFTER bootstrap.sh, AFTER caches are built, AFTER Phase 1+2 checkpoints
exist, BEFORE the multi-hour Phase 3 sweep.

Verifies:
  1. dpc imports (incl. wrapper, calibration, pooling, nms, coco_classes)
  2. ultralytics is installed and YOLO weights load
  3. Phase 2 checkpoint loads
  4. calibration.py — pure function correctness
  5. pooling.py — per-box suspicion + boxes_overlap_patch
  6. wrapper.py — full forward on a real image, all output keys present
  7. metrics.py — Phase 3 metrics on synthetic per_image data
  8. one full evaluate_phase3 micro-step on 2 images

Total runtime: ~1-2 minutes.

Exit code:
    0 — safe to launch full sweep
    1 — investigate before launching

Usage:
    python tools/smoke_test_phase3.py \
        --denoiser-checkpoint runs/phase2_seed42/checkpoints/best \
        --use-ema \
        --yolo-weights yolo26n.pt \
        --apricot-eval-cache caches/apricot_eval_640.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def phase(name: str):
    def deco(fn):
        def wrapped(*a, **kw):
            print(f"\n{C.BOLD}{C.B}── {name} ──{C.END}")
            t0 = time.perf_counter()
            try:
                fn(*a, **kw)
                dt = time.perf_counter() - t0
                print(f"  {C.G}✓ pass{C.END} ({dt:.1f}s)")
                PASSED.append(name)
            except Exception as e:
                import traceback
                print(f"  {C.R}✗ fail{C.END}: {e}")
                print(traceback.format_exc())
                FAILED.append((name, str(e)))
        return wrapped
    return deco


@phase("1. dpc imports (Phase 3 modules)")
def test_imports():
    from dpc import __version__
    from dpc.calibration import (
        calibrate_obj_and_cls, calibrate_logits, compute_size_factor,
    )
    from dpc.pooling import per_box_suspicion, box_areas_frac, boxes_overlap_patch
    from dpc.wrapper import DPCWrapper, DPCOutput, YoloOutputs
    from dpc.coco_classes import COCO_CLASSES, class_name
    from dpc.nms import nms, class_aware_nms, box_iou
    from dpc.metrics import (
        per_image_detection_metrics,
        aggregate_on_patch_suppression, aggregate_off_patch_retention, aggregate_per_image_margin,
        confusion_matrix, adversarial_class_table,
        discriminability_ratio,
    )
    print(f"  dpc version: {__version__}")
    assert len(COCO_CLASSES) == 80


@phase("2. ultralytics + YOLO weights")
def test_yolo_load(args):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("install ultralytics: pip install ultralytics") from e
    yolo_path = Path(args.yolo_weights)
    if not yolo_path.is_file():
        raise RuntimeError(f"YOLO weights not found: {yolo_path}")
    yolo = YOLO(str(yolo_path))
    print(f"  YOLO loaded: {yolo_path.name}")


@phase("3. Phase 2 checkpoint")
def test_phase2_ckpt(args):
    from dpc.checkpoint import load_checkpoint
    from dpc.config import DPCConfig
    from dpc.denoiser import TinyUNetDenoiser

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    ckpt = Path(args.denoiser_checkpoint).resolve()
    if not ckpt.is_dir():
        raise RuntimeError(f"checkpoint not found: {ckpt}")
    den = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    meta = load_checkpoint(ckpt, den, restore_rng=False)
    if (ckpt / "ema.pt").exists():
        den.load_state_dict(torch.load(ckpt / "ema.pt", map_location=device, weights_only=False))
    print(f"  meta: phase={meta.get('phase')}, epoch={meta.get('epoch')}")


@phase("4. calibration math")
def test_calibration():
    from dpc.calibration import calibrate_obj_and_cls, compute_size_factor

    raw_obj = torch.tensor([2.0, 1.5, 0.5])
    raw_cls = torch.tensor([
        [3.0, -1.0, 0.5],
        [0.5, 4.0, -2.0],
        [1.0, 1.5, 2.0],
    ])
    suspicion = torch.tensor([0.8, 0.1, 0.0])
    box_areas = torch.tensor([0.05, 0.01, 0.20])  # second is small target
    cal_obj, cal_cls = calibrate_obj_and_cls(
        raw_obj_logits=raw_obj, raw_cls_logits=raw_cls,
        suspicion=suspicion, box_areas_frac=box_areas,
        obj_alpha=10.0, cls_alpha=10.0,
        small_target_amplification=1.5,
        small_target_area_threshold=0.02,
    )
    # Box 0: not small → factor 1.0; cal_obj = 2.0 - 10*0.8*1.0 = -6.0
    assert torch.isclose(cal_obj[0], torch.tensor(-6.0), atol=1e-5), \
        f"expected -6.0, got {cal_obj[0]}"
    # Box 1: SMALL (0.01 ≤ 0.02) → factor 1.5; cal_obj = 1.5 - 10*0.1*1.5 = 0.0
    assert torch.isclose(cal_obj[1], torch.tensor(0.0), atol=1e-5), \
        f"expected 0.0, got {cal_obj[1]}"
    # Box 2: zero suspicion → cal_obj == raw_obj
    assert torch.isclose(cal_obj[2], torch.tensor(0.5), atol=1e-5)
    print(f"  calibration math correct (small-target amplification verified)")


@phase("5. pooling")
def test_pooling():
    from dpc.pooling import per_box_suspicion, box_areas_frac, boxes_overlap_patch

    # Field with high values in upper-left quadrant
    field = torch.zeros(1, 1, 64, 64)
    field[:, :, :32, :32] = 0.9
    field[:, :, 32:, :] = 0.1
    boxes = torch.tensor([
        [0, 0, 32, 32],     # entirely inside high region
        [32, 32, 64, 64],   # entirely inside low region
        [16, 16, 48, 48],   # mixed
    ], dtype=torch.float)
    img_idx = torch.tensor([0, 0, 0], dtype=torch.long)
    susp = per_box_suspicion(field, boxes, img_idx, pool_size=4)
    print(f"  suspicion: {susp.tolist()}")
    assert susp[0] > susp[2] > susp[1], "high > mixed > low region ordering broken"

    # boxes_overlap_patch
    patch_boxes = torch.tensor([[10, 10, 30, 30]], dtype=torch.float)
    boxes_to_test = torch.tensor([
        [10, 10, 30, 30],   # exact overlap
        [50, 50, 70, 70],   # no overlap
        [15, 15, 35, 35],   # partial overlap
    ], dtype=torch.float)
    overlap = boxes_overlap_patch(boxes_to_test, patch_boxes, iou_threshold=0.1)
    print(f"  overlap: {overlap.tolist()}")
    assert overlap[0] and not overlap[1] and overlap[2]


@phase("6. wrapper forward")
def test_wrapper_forward(args):
    from dpc.checkpoint import load_checkpoint
    from dpc.config import DPCConfig
    from dpc.data_cache import TensorCache
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.wrapper import DPCWrapper, YoloOutputs
    from dpc.seeding import make_generator

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    den = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    ckpt = Path(args.denoiser_checkpoint).resolve()
    load_checkpoint(ckpt, den, restore_rng=False)
    if (ckpt / "ema.pt").exists():
        den.load_state_dict(torch.load(ckpt / "ema.pt", map_location=device, weights_only=False))
    den.eval()

    # Stub YOLO function: produce 3 fixed boxes per image
    def stub_yolo(images: torch.Tensor) -> YoloOutputs:
        B = images.shape[0]
        H, W = images.shape[-2:]
        boxes_per_image = torch.tensor([
            [10, 10, 100, 100],
            [200, 200, 300, 300],
            [400, 100, 500, 200],
        ], dtype=torch.float, device=images.device)
        all_boxes = []
        all_idx = []
        all_obj = []
        all_cls = []
        for b in range(B):
            all_boxes.append(boxes_per_image)
            all_idx.append(torch.full((3,), b, dtype=torch.long, device=images.device))
            all_obj.append(torch.tensor([1.5, 2.0, 0.5], device=images.device))
            cls = torch.zeros((3, 80), device=images.device)
            cls[0, 9] = 2.5; cls[1, 12] = 3.0; cls[2, 0] = 1.0
            all_cls.append(cls)
        return YoloOutputs(
            boxes=torch.cat(all_boxes, dim=0),
            image_idx=torch.cat(all_idx, dim=0),
            obj_logits=torch.cat(all_obj, dim=0),
            cls_logits=torch.cat(all_cls, dim=0),
        )

    wrapper = DPCWrapper(
        denoiser=den, cfg=cfg, yolo_forward_fn=stub_yolo,
        score_threshold=0.25, cls_alpha=50.0, obj_alpha=50.0,
    ).to(device)

    cache = TensorCache(Path(args.apricot_eval_cache).resolve())
    img = cache[0]["image"].unsqueeze(0).to(device)
    gen = make_generator(42, device=device)
    out = wrapper(img, generator=gen)
    assert out.boxes.shape[0] == 3
    assert out.boxes_kept_baseline.shape == (3,)
    assert out.boxes_kept_dpc.shape == (3,)
    assert out.baseline_class.shape == (3,)
    assert out.dpc_class.shape == (3,)
    assert out.suspicion_field.shape[2:] == img.shape[2:]
    assert out.raw_residual.shape[2:] == img.shape[2:]
    print(f"  baseline_class: {out.baseline_class.tolist()}")
    print(f"  dpc_class:      {out.dpc_class.tolist()}")
    print(f"  per_box_susp:   {[round(s.item(), 3) for s in out.per_box_suspicion]}")


@phase("7. Phase 3 metrics on synthetic data")
def test_metrics():
    from dpc.metrics import (
        aggregate_on_patch_suppression, aggregate_off_patch_retention, aggregate_per_image_margin,
        confusion_matrix, adversarial_class_table,
        discriminability_ratio,
    )

    # Synthetic per_image: 3 images
    per_image = [
        {
            "name": "img_a.jpg",
            "patch_box_count": 1,
            "patch_areas_frac": [0.05],
            "baseline": {"n_dets": 10, "n_dets_on_patch": 6,
                         "on_patch_classes": [9, 9, 9, 12, 12, 12],
                         "off_patch_classes": [0, 0, 2, 2]},
            "dpc": {"n_dets": 7, "n_dets_on_patch": 2,
                    "on_patch_classes": [9, 12],
                    "off_patch_classes": [0, 0, 2, 2, 5]},
            "margin_pp": (100*4/6) - (100*-1/4),  # +66.67 - (-25) = +91.67
            "on_patch_suppression_pct": 100*4/6,
            "off_patch_suppression_pct": 100*-1/4,
            "per_box_suspicion_on_patch": [0.4, 0.5, 0.45, 0.6, 0.55, 0.5],
            "per_box_suspicion_off_patch": [0.1, 0.05, 0.15, 0.08],
            "elapsed_ms": 35.0,
        },
        {
            "name": "img_b.jpg",
            "patch_box_count": 1,
            "patch_areas_frac": [0.03],
            "baseline": {"n_dets": 5, "n_dets_on_patch": 3,
                         "on_patch_classes": [9, 9, 12], "off_patch_classes": [0, 2]},
            "dpc": {"n_dets": 5, "n_dets_on_patch": 3,
                    "on_patch_classes": [9, 9, 12], "off_patch_classes": [0, 2]},
            "margin_pp": 0.0,
            "on_patch_suppression_pct": 0.0,
            "off_patch_suppression_pct": 0.0,
            "per_box_suspicion_on_patch": [0.3, 0.35, 0.32],
            "per_box_suspicion_off_patch": [0.12, 0.10],
            "elapsed_ms": 38.0,
        },
        {
            "name": "img_c.jpg",
            "patch_box_count": 1,
            "patch_areas_frac": [0.10],
            "baseline": {"n_dets": 8, "n_dets_on_patch": 4,
                         "on_patch_classes": [11, 11, 9, 12], "off_patch_classes": [0, 2, 5, 7]},
            "dpc": {"n_dets": 6, "n_dets_on_patch": 1,
                    "on_patch_classes": [9], "off_patch_classes": [0, 2, 5, 7, 9]},
            "margin_pp": 75.0 - (-25.0),  # 100.0
            "on_patch_suppression_pct": 75.0,
            "off_patch_suppression_pct": -25.0,
            "per_box_suspicion_on_patch": [0.5, 0.55, 0.6, 0.45],
            "per_box_suspicion_off_patch": [0.08, 0.10, 0.05, 0.07],
            "elapsed_ms": 36.0,
        },
    ]

    c1 = aggregate_on_patch_suppression(per_image, seed=42)
    c2 = aggregate_off_patch_retention(per_image, seed=42)
    c3 = aggregate_per_image_margin(per_image)
    cm = confusion_matrix(per_image)
    act = adversarial_class_table(per_image)
    dr = discriminability_ratio(per_image)

    print(f"  C1 mean reduction: {c1['mean_reduction_pp']:.2f}pp")
    print(f"  C2 mean retention: {c2['mean_retention']:.3f}")
    print(f"  C3 median: {c3['median']:.2f}, mean: {c3['mean']:.2f}")
    print(f"  Discriminability ratio: {dr:.2f}")
    print(f"  Top adversarial class: {act[0]['class_name']} "
          f"(suppression {act[0]['suppression_pct']:.1f}%)")

    assert c1["n_applicable_images"] == 3
    assert c1["mean_reduction_pp"] > 0  # synthetic data designed to show reduction
    assert c2["mean_retention"] > 0  # at least some retention
    assert dr is not None and dr > 1.0  # patch suspicion > off-patch suspicion


@phase("8. Mini evaluate_phase3 on 2 real images")
def test_mini_evaluate(args):
    """Same path as evaluate_phase3 but only 2 images, no JSON writes."""
    from dpc.checkpoint import load_checkpoint
    from dpc.config import DPCConfig
    from dpc.data_cache import TensorCache
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.pooling import boxes_overlap_patch
    from dpc.seeding import make_generator
    from dpc.wrapper import DPCWrapper

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()

    den = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    ckpt = Path(args.denoiser_checkpoint).resolve()
    load_checkpoint(ckpt, den, restore_rng=False)
    if (ckpt / "ema.pt").exists():
        den.load_state_dict(torch.load(ckpt / "ema.pt", map_location=device, weights_only=False))
    den.eval()

    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    from tools.evaluate_phase3 import _make_yolo_forward_fn_ultralytics
    yolo_fn = _make_yolo_forward_fn_ultralytics(yolo, score_threshold=0.001)

    wrapper = DPCWrapper(
        denoiser=den, cfg=cfg, yolo_forward_fn=yolo_fn,
        score_threshold=0.25, cls_alpha=50.0, obj_alpha=50.0,
    ).to(device)

    cache = TensorCache(Path(args.apricot_eval_cache).resolve())
    gen = make_generator(42, device=device)

    n_b_kept = 0; n_d_kept = 0
    for i in range(min(2, len(cache))):
        item = cache[i]
        img = item["image"].unsqueeze(0).to(device)
        out = wrapper(img, generator=gen)
        n_b_kept += int(out.boxes_kept_baseline.sum().item())
        n_d_kept += int(out.boxes_kept_dpc.sum().item())
    print(f"  2-image baseline kept: {n_b_kept}, dpc kept: {n_d_kept}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser-checkpoint", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--apricot-eval-cache", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    print(f"{C.BOLD}DPC Phase 3 Smoke Test{C.END}")
    print(f"  denoiser: {args.denoiser_checkpoint}")
    print(f"  yolo:     {args.yolo_weights}")
    print(f"  apricot:  {args.apricot_eval_cache}")

    test_imports()
    test_yolo_load(args)
    test_phase2_ckpt(args)
    test_calibration()
    test_pooling()
    test_wrapper_forward(args)
    test_metrics()
    test_mini_evaluate(args)

    print()
    print(f"{C.BOLD}── Summary ──{C.END}")
    print(f"  Passed: {len(PASSED)}")
    print(f"  Failed: {len(FAILED)}")
    if FAILED:
        print()
        print(f"  {C.R}{C.BOLD}FAIL{C.END} — DO NOT launch sweep:")
        for name, err in FAILED:
            print(f"    - {name}: {err}")
        sys.exit(1)
    else:
        print()
        print(f"  {C.G}{C.BOLD}PASS{C.END} — Phase 3 pipeline is wired up correctly.")
        print(f"  Next: tools/sanity_check_eval.py + tools/sweep_alpha.py + tools/multi_seed_runner.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
