"""Phase 3 evaluator: baseline YOLO26 vs DPC-wrapped YOLO26 on APRICOT.

Reports the three primary metrics (renamed in v3.2.0, retained in v3.3.0):
  - on_patch_suppression: fraction of adversarial on-patch detections killed vs baseline
  - off_patch_retention: fraction of true off-patch detections preserved vs baseline
  - per_image_margin: per-image suppression minus retention loss

For v3.3.0, this evaluator expects a Phase 2 output directory that
contains both the denoiser checkpoint AND a fine-tuned YOLO26 head weights
file (yolo26_head_finetuned.pt). It loads YOLO26 with the pretrained
weights, then patches in the fine-tuned head state.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dpc.checkpoint import load_checkpoint
from dpc.config import DPCConfig
from dpc.data import CachedApricotDataset, collate_dpc_batch, make_apricot_indices
from dpc.data_cache import TensorCache
from dpc.denoiser import TinyUNetDenoiser
from dpc.ema import EMA
from dpc.manifest import write_manifest
from dpc.metrics import (
    aggregate_off_patch_retention,
    aggregate_on_patch_suppression,
    aggregate_per_image_margin,
    per_image_detection_metrics,
)
from dpc.pooling import boxes_overlap_patch
from dpc.seeding import make_generator, set_global_seed
from dpc.wrapper import DPCDetections, DPCWrapper
from dpc.yolo26_native import emit_final_detections, forward_yolo26_raw, load_yolo26, slice_raw

log = logging.getLogger("evaluate_phase3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPC-YOLO26 v3.3.0 Phase 3 evaluation")
    p.add_argument("--denoiser-checkpoint", type=str, required=True,
                   help="Phase 2 denoiser checkpoint directory")
    p.add_argument("--yolo-weights", type=str, required=True,
                   help="yolo26n.pt or any base YOLO26 weights")
    p.add_argument("--yolo-head-finetuned", type=str, default=None,
                   help="Path to fine-tuned head state (yolo26_head_finetuned.pt). "
                        "If omitted, uses pretrained head; will produce baseline behavior.")
    p.add_argument("--apricot-eval-cache", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    p.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--on-patch-iou", type=float, default=0.1)
    p.add_argument("--cls-alpha", type=float, default=None,
                   help="Override cfg.lambda_cls for this run")
    p.add_argument("--obj-alpha", type=float, default=None,
                   help="Override cfg.lambda_obj for this run")
    p.add_argument("--n-images", type=int, default=None,
                   help="Limit eval to N images (default: all)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-control", action="store_true",
                   help="Run on clean COCO instead of APRICOT (sanity check)")
    return p.parse_args()


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_global_seed(args.seed)

    cfg = DPCConfig()
    if args.device != "auto":
        cfg.device = args.device
    if args.cls_alpha is not None:
        cfg.lambda_cls = args.cls_alpha
    if args.obj_alpha is not None:
        cfg.lambda_obj = args.obj_alpha
    device = cfg.get_device()

    # Load denoiser
    log.info(f"Loading denoiser from {args.denoiser_checkpoint}")
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    ema = EMA(denoiser, decay=0.999)
    load_checkpoint(args.denoiser_checkpoint, denoiser, ema=ema, restore_rng=False)
    if args.use_ema:
        denoiser.load_state_dict(ema.state_dict())
    denoiser.eval()

    # Load YOLO26 base, then patch fine-tuned head if provided
    log.info(f"Loading YOLO26 from {args.yolo_weights}")
    yolo_inner = load_yolo26(args.yolo_weights, device)
    if args.yolo_head_finetuned:
        log.info(f"Patching fine-tuned head from {args.yolo_head_finetuned}")
        head_state = torch.load(args.yolo_head_finetuned, map_location=device)
        own = dict(yolo_inner.named_parameters())
        n_patched = 0
        for name, val in head_state.items():
            if name in own and own[name].shape == val.shape:
                own[name].data.copy_(val.to(device))
                n_patched += 1
        log.info(f"Patched {n_patched}/{len(head_state)} head params")
    else:
        log.warning("No --yolo-head-finetuned provided; using pretrained head only. "
                    "Eq. 22 modulation is not active without fine-tuned head.")
    yolo_inner.eval()

    # Wrapper
    wrapper = DPCWrapper(
        yolo_model=yolo_inner,
        denoiser=denoiser,
        cfg=cfg,
        score_threshold=args.score_threshold,
        top_k=300,
        n_classes=80,
    ).to(device)
    wrapper.eval()

    # Data
    cache = TensorCache(args.apricot_eval_cache)
    eval_idx = make_apricot_indices(cache)
    if args.n_images:
        eval_idx = eval_idx[: args.n_images]
    ds = CachedApricotDataset(cache, eval_idx)
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=collate_dpc_batch)

    gen = make_generator(args.seed, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = out_dir / "per_image.jsonl"
    per_image_records: list[dict] = []

    log.info(f"Evaluating on {len(eval_idx)} images")
    t_start = time.time()

    with per_image_path.open("w") as fh:
        for batch_idx, batch in enumerate(loader):
            images = batch["images"].to(device)
            patch_boxes = batch.get("patch_boxes", [torch.empty((0, 4))])[0].to(device)
            name = batch.get("names", [f"image_{batch_idx}"])[0]
            H, W = images.shape[2], images.shape[3]

            t0 = time.time()
            dpc_out_list: list[DPCDetections] = wrapper(images, generator=gen)
            dpc_det = dpc_out_list[0]
            baseline_list = wrapper.baseline_detections(images)
            base_det = baseline_list[0]
            t_elapsed = (time.time() - t0) * 1000.0

            # On-patch / off-patch classification using ground-truth patch boxes
            base_on = boxes_overlap_patch(base_det["boxes"], patch_boxes, iou_threshold=args.on_patch_iou)
            dpc_on = boxes_overlap_patch(dpc_det.boxes_xyxy, patch_boxes, iou_threshold=args.on_patch_iou)

            metrics = per_image_detection_metrics(
                name=name,
                patch_boxes=patch_boxes.cpu().tolist(),
                img_area=H * W,
                baseline_overlap_classes=base_det["classes"][base_on].cpu().tolist(),
                baseline_off_patch_classes=base_det["classes"][~base_on].cpu().tolist(),
                dpc_overlap_classes=dpc_det.classes[dpc_on].cpu().tolist(),
                dpc_off_patch_classes=dpc_det.classes[~dpc_on].cpu().tolist(),
                per_box_suspicion_overlap=dpc_det.suspicion[dpc_on].cpu().tolist(),
                per_box_suspicion_off_patch=dpc_det.suspicion[~dpc_on].cpu().tolist(),
                elapsed_ms=t_elapsed,
            )
            per_image_records.append(metrics)
            fh.write(json.dumps(metrics) + "\n")
            fh.flush()

            if (batch_idx + 1) % 20 == 0:
                log.info(f"  {batch_idx + 1}/{len(eval_idx)}")

    # Aggregate
    on_patch_agg = aggregate_on_patch_suppression(per_image_records, seed=args.seed)
    off_patch_agg = aggregate_off_patch_retention(per_image_records, seed=args.seed)
    margin_agg = aggregate_per_image_margin(per_image_records)
    aggregate = {
        "on_patch_suppression": on_patch_agg,
        "off_patch_retention": off_patch_agg,
        "per_image_margin": margin_agg,
        "n_images": len(per_image_records),
        "wall_clock_s": time.time() - t_start,
    }

    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    log.info(f"on_patch_suppression: {on_patch_agg}")
    log.info(f"off_patch_retention:  {off_patch_agg}")
    log.info(f"per_image_margin:     {margin_agg}")

    write_manifest(
        str(out_dir),
        extra_meta={"phase": "phase3", "args": vars(args), "cfg": cfg.asdict()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
