"""Strict YOLO26 native bridge.

Provides direct access to YOLO26's raw head output via the lower-level
Ultralytics API `model.model(x)`, bypassing the public `predict()`
postprocessing pipeline. Required for:

  1. Reading real per-class logits (v3.2.0 reconstructed them with a
     +10-margin trick from final confidence scores).
  2. Exposing the assignment cost matrix C_base during training so that
     §5.16 / Eq. (22) modulation can be applied.

This module never forks or monkey-patches Ultralytics; all access is
through the public Python attribute interface that Ultralytics 8.3+ exposes.

## YOLO26 head output (confirmed via Sapkota et al. arXiv:2509.25164)

Per-anchor: 4 box-regression scalars (no DFL), 1 objectness logit, K class
logits. Total tensor shape [B, 5 + K, A] where A ≈ 8400 for 640×640 input
(80×80 + 40×40 + 20×20 at strides 8, 16, 32).

## Decoding

YOLO26 uses anchor-free direct distance regression. For each pyramid
stride s, anchor at grid cell (cx, cy) maps to image point
(s·(cx + 0.5), s·(cy + 0.5)). The 4 box offsets are (left, top, right,
bottom) distances in stride units; xyxy decoding is straightforward.

## API compatibility

Tested against ultralytics==8.3.x. The `model.model` attribute exposes the
underlying nn.Module. If the API drifts in future Ultralytics versions,
this module is the single point of update.
"""

from __future__ import annotations
from typing import Optional

from dataclasses import dataclass

import torch
import torch.nn as nn


_TESTED_ULTRALYTICS_MAJOR_MINOR = (8, 3)


def _check_ultralytics_version() -> None:
    """Soft version check; warn on minor mismatch."""
    try:
        import ultralytics
        ver = ultralytics.__version__.split(".")
        major, minor = int(ver[0]), int(ver[1])
        if (major, minor) < _TESTED_ULTRALYTICS_MAJOR_MINOR:
            import warnings
            warnings.warn(
                f"ultralytics {ultralytics.__version__} is older than the "
                f"tested {_TESTED_ULTRALYTICS_MAJOR_MINOR[0]}."
                f"{_TESTED_ULTRALYTICS_MAJOR_MINOR[1]}.x; YOLO26 native "
                f"bridge may not work.",
                stacklevel=3,
            )
    except (ImportError, ValueError, IndexError):
        pass


@dataclass
class RawHeadOutput:
    """Decoded YOLO26 head output, per-anchor.

    All tensors share the batch dimension B and the anchor dimension A.

    NOTE: YOLO26 merges objectness into the per-class scores; there is no
    separate objectness logit. `obj_logits` is therefore Optional. When
    present (legacy YOLOv5/v8-style heads), it carries the pre-sigmoid
    objectness logit. When None, callers must derive "objectness" from
    the per-class scores (e.g., max-over-classes).
    """

    boxes_xyxy: torch.Tensor                      # [B, A, 4] image-pixel xyxy
    obj_logits: Optional[torch.Tensor]            # [B, A] pre-sigmoid, or None
    cls_logits: torch.Tensor                      # [B, A, K_cls] pre-sigmoid
    anchor_centers: torch.Tensor                  # [A, 2] image-pixel anchor centers
    anchor_strides: torch.Tensor                  # [A] integer strides (8, 16, or 32)


def load_yolo26(weights_path: str, device: torch.device) -> nn.Module:
    """Load YOLO26 and return its underlying nn.Module.

    Call the returned module directly with a [B, 3, H, W] tensor to get
    raw head output (use forward_yolo26_raw for the decoded form).
    """
    _check_ultralytics_version()
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Ultralytics not installed. Install with: pip install ultralytics"
        ) from e

    yolo = YOLO(weights_path)
    inner = yolo.model
    inner.to(device)
    inner.eval()
    return inner


def _build_anchor_grid(
    feature_shapes: list[tuple[int, int]],
    strides: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-anchor (center_xy, stride) tensors."""
    if len(feature_shapes) != len(strides):
        raise ValueError(
            f"feature_shapes ({len(feature_shapes)}) and strides "
            f"({len(strides)}) length mismatch"
        )
    centers_list: list[torch.Tensor] = []
    strides_list: list[torch.Tensor] = []
    for (h_l, w_l), s_l in zip(feature_shapes, strides):
        ys, xs = torch.meshgrid(
            torch.arange(h_l, device=device, dtype=torch.float32),
            torch.arange(w_l, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cx = (xs + 0.5) * s_l
        cy = (ys + 0.5) * s_l
        centers = torch.stack([cx.flatten(), cy.flatten()], dim=-1)
        centers_list.append(centers)
        strides_list.append(
            torch.full((h_l * w_l,), s_l, device=device, dtype=torch.long)
        )
    return torch.cat(centers_list, dim=0), torch.cat(strides_list, dim=0)


def _decode_boxes_from_offsets(
    box_offsets: torch.Tensor,
    anchor_centers: torch.Tensor,
    anchor_strides: torch.Tensor,
) -> torch.Tensor:
    """Decode (left, top, right, bottom) stride-unit distances to image xyxy."""
    cx = anchor_centers[:, 0].unsqueeze(0)
    cy = anchor_centers[:, 1].unsqueeze(0)
    s = anchor_strides.to(box_offsets.dtype).unsqueeze(0)

    l = box_offsets[..., 0]
    t = box_offsets[..., 1]
    r = box_offsets[..., 2]
    b = box_offsets[..., 3]

    x1 = cx - l * s
    y1 = cy - t * s
    x2 = cx + r * s
    y2 = cy + b * s
    return torch.stack([x1, y1, x2, y2], dim=-1)


def forward_yolo26_raw(
    model: nn.Module,
    images: torch.Tensor,
    n_classes: int = 80,
) -> RawHeadOutput:
    """Run YOLO26 natively and return decoded raw head output.

    YOLO26 (Ultralytics 8.4+) returns different structures in train vs eval mode:
      - train mode: dict{"one2many": {boxes, scores, feats}, "one2one": {...}}
      - eval mode:  tuple(decoded_dets[B, top_k, 6], dict)

    We use the **one2one** branch (the NMS-free training target per
    Sapkota et al. arXiv:2509.25164) because:
      (a) DPC's §5.16 modulated assignment fine-tunes the one-to-one branch
      (b) That's the branch YOLO26 uses at inference time

    Boxes from the head are in **stride-unit (l, t, r, b) distance form**
    relative to each anchor center; we decode to image-pixel xyxy here.

    Scores are pre-sigmoid logits, [B, K_cls, A], one logit per class per
    anchor. There is NO separate objectness logit; YOLO26 merges
    objectness into the class scores. RawHeadOutput.obj_logits is set to
    None and downstream code derives "objectness" from the per-class
    maximum.

    Args:
      model:     nn.Module from load_yolo26
      images:    [B, 3, H, W] in [0, 1]
      n_classes: K_cls expected (default 80 for COCO)
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [B, 3, H, W], got {images.shape}")

    # Force train mode so we get the raw dict; we'll restore later if needed
    prev_training = model.training
    if not prev_training:
        model.train()
    try:
        out = model(images)
    finally:
        if not prev_training:
            model.eval()

    # Both modes return a dict containing branches; train returns the dict
    # directly, eval returns a (decoded_tensor, dict) tuple.
    if isinstance(out, dict):
        branches = out
    elif isinstance(out, (tuple, list)) and len(out) >= 2 and isinstance(out[1], dict):
        branches = out[1]
    else:
        raise RuntimeError(
            f"unexpected YOLO26 head output type: {type(out)}. "
            f"Ultralytics API may have changed; update dpc/yolo26_native.py."
        )

    if "one2one" not in branches:
        raise RuntimeError(
            f"expected 'one2one' branch in head output; got keys {list(branches.keys())}"
        )
    branch = branches["one2one"]

    boxes_t = branch["boxes"]      # [B, 4, A] — (l, t, r, b) in stride units
    scores_t = branch["scores"]    # [B, K_cls, A] pre-sigmoid
    feats = branch["feats"]        # list of [B, C_l, h_l, w_l] per pyramid level

    if boxes_t.shape[1] != 4:
        raise RuntimeError(
            f"expected boxes channel dim 4, got {boxes_t.shape}"
        )
    if scores_t.shape[1] != n_classes:
        raise RuntimeError(
            f"expected {n_classes} class channels, got {scores_t.shape[1]} "
            f"(check n_classes argument)"
        )

    # Transpose to [B, A, C] convention used elsewhere
    box_offsets = boxes_t.transpose(1, 2).contiguous()    # [B, A, 4]
    cls_logits = scores_t.transpose(1, 2).contiguous()    # [B, A, K_cls]

    A = box_offsets.shape[1]

    # Build anchor grid from the feature map shapes the head exposed.
    # This is the authoritative source of (H_l, W_l) per pyramid level;
    # don't reconstruct from images.shape because the head may use a
    # different effective resolution internally.
    strides = [8, 16, 32]
    if len(feats) != len(strides):
        raise RuntimeError(
            f"expected {len(strides)} pyramid levels, got {len(feats)}"
        )
    feature_shapes = [(f.shape[-2], f.shape[-1]) for f in feats]
    expected_A = sum(h * w for h, w in feature_shapes)
    if A != expected_A:
        raise RuntimeError(
            f"anchor count mismatch: head emitted {A}, but pyramid "
            f"shapes {feature_shapes} would produce {expected_A}."
        )

    anchor_centers, anchor_strides = _build_anchor_grid(
        feature_shapes, strides, device=images.device
    )

    # Decode boxes: (l, t, r, b) in stride units → image-pixel xyxy.
    boxes_xyxy = _decode_boxes_from_offsets(
        box_offsets, anchor_centers, anchor_strides
    )

    return RawHeadOutput(
        boxes_xyxy=boxes_xyxy,
        obj_logits=None,           # YOLO26 has no separate objectness
        cls_logits=cls_logits,
        anchor_centers=anchor_centers,
        anchor_strides=anchor_strides,
    )


def _generalized_iou_pairwise(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """Pairwise GIoU [N, M] between [N, 4] and [M, 4] xyxy boxes."""
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=boxes_a.device, dtype=boxes_a.dtype)

    a = boxes_a.unsqueeze(1)
    b = boxes_b.unsqueeze(0)
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-9
    iou = inter / union

    enc_x1 = torch.minimum(a[..., 0], b[..., 0])
    enc_y1 = torch.minimum(a[..., 1], b[..., 1])
    enc_x2 = torch.maximum(a[..., 2], b[..., 2])
    enc_y2 = torch.maximum(a[..., 3], b[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-9

    return iou - (enc_area - union) / enc_area


def compute_base_cost(
    raw_single_image: RawHeadOutput,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> torch.Tensor:
    """Compute the base assignment cost matrix C_base for one image.

    Implements a DETR-style matching cost: weighted sum of classification,
    L1 box, and GIoU costs. This is the C_base that §5.16 / Eq. (22)
    modulates.

    Caller is responsible for slicing RawHeadOutput to a single image
    (boxes_xyxy[i], obj_logits[i], cls_logits[i]) before calling.

    Args:
      raw_single_image: single-image slice; boxes_xyxy is [A, 4]
      gt_boxes:         [M, 4] xyxy ground truth
      gt_classes:       [M] long class indices
    """
    if raw_single_image.boxes_xyxy.dim() != 2:
        raise ValueError(
            "compute_base_cost expects single-image input; "
            "got boxes_xyxy of shape "
            f"{tuple(raw_single_image.boxes_xyxy.shape)} (need [A, 4])"
        )

    A = raw_single_image.boxes_xyxy.shape[0]
    M = gt_boxes.shape[0]
    device = raw_single_image.boxes_xyxy.device
    dtype = raw_single_image.boxes_xyxy.dtype

    if M == 0:
        return torch.zeros((A, 0), device=device, dtype=dtype)

    cls_probs = raw_single_image.cls_logits.sigmoid()
    target_probs = cls_probs[:, gt_classes]
    cost_cls_simple = 1.0 - target_probs

    pred_boxes = raw_single_image.boxes_xyxy.unsqueeze(1)
    gt_boxes_e = gt_boxes.unsqueeze(0)
    l1 = (pred_boxes - gt_boxes_e).abs().sum(dim=-1)

    giou = _generalized_iou_pairwise(raw_single_image.boxes_xyxy, gt_boxes)
    cost_giou_term = 1.0 - giou

    return (
        cost_class * cost_cls_simple
        + cost_bbox * l1
        + cost_giou * cost_giou_term
    )


def emit_final_detections(
    raw_single_image: RawHeadOutput,
    score_threshold: float = 0.25,
    top_k: int = 300,
) -> dict:
    """Convert raw head output to final detections for one image.

    YOLO26 is NMS-free: top-k by combined obj·cls score, threshold-filtered.

    Args:
      raw_single_image: single-image slice
      score_threshold:  minimum score to retain
      top_k:            maximum detections to emit

    Returns:
      {"boxes": [N, 4], "scores": [N], "classes": [N]}
    """
    cls_score = raw_single_image.cls_logits.sigmoid()
    best_cls_score, best_cls = cls_score.max(dim=-1)
    if raw_single_image.obj_logits is not None:
        combined = raw_single_image.obj_logits.sigmoid() * best_cls_score
    else:
        combined = best_cls_score

    keep = combined >= score_threshold
    boxes = raw_single_image.boxes_xyxy[keep]
    scores = combined[keep]
    classes = best_cls[keep]

    if scores.shape[0] > top_k:
        top_scores, top_idx = scores.topk(top_k)
        boxes = boxes[top_idx]
        scores = top_scores
        classes = classes[top_idx]

    return {"boxes": boxes, "scores": scores, "classes": classes}


def slice_raw(raw: RawHeadOutput, batch_idx: int) -> RawHeadOutput:
    """Extract a single image's slice from a batched RawHeadOutput.

    Convenience for callers that want to iterate per-image (e.g., the
    assignment phase, which is per-image because the GT count varies).
    """
    return RawHeadOutput(
        boxes_xyxy=raw.boxes_xyxy[batch_idx],
        obj_logits=(raw.obj_logits[batch_idx] if raw.obj_logits is not None else None),
        cls_logits=raw.cls_logits[batch_idx],
        anchor_centers=raw.anchor_centers,
        anchor_strides=raw.anchor_strides,
    )
