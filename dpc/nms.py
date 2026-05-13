"""Class-aware non-maximum suppression.

YOLO26 in v3 is NMS-FREE (it uses dual assignment and produces non-overlapping
detections by design). However, the DPC pipeline can ALTER class identity via
calibration (mistake #39 fixed): a box that was 'traffic light' in baseline
might become 'car' under DPC. That can leave a redundant box in the same
location with a different class.

For evaluation purposes, we therefore optionally apply class-aware NMS over
DPC-classified boxes to avoid double-counting. The eval script can choose
whether to NMS or not. We also expose plain (class-agnostic) NMS as a fallback.
"""

from __future__ import annotations

import torch


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between [N, 4] and [M, 4] xyxy boxes. Returns [N, M]."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    a_area = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    b_area = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)

    ix1 = torch.maximum(ax1[:, None], bx1[None, :])
    iy1 = torch.maximum(ay1[:, None], by1[None, :])
    ix2 = torch.minimum(ax2[:, None], bx2[None, :])
    iy2 = torch.minimum(ay2[:, None], by2[None, :])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = a_area[:, None] + b_area[None, :] - inter + 1e-9
    return inter / union


def nms(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Standard greedy NMS. Returns [K] long tensor of indices to keep, sorted
    by descending score."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes_xyxy.device)
    order = scores.argsort(descending=True)
    keep = []
    suppressed = torch.zeros((boxes_xyxy.shape[0],), dtype=torch.bool, device=boxes_xyxy.device)
    for i in order.tolist():
        if suppressed[i]:
            continue
        keep.append(i)
        if not suppressed.any() or i == order[-1].item():
            pass
        # Compare box i to all remaining
        ious = box_iou(boxes_xyxy[i:i+1], boxes_xyxy)[0]  # [N]
        for j in range(boxes_xyxy.shape[0]):
            if j != i and not suppressed[j] and ious[j] >= iou_threshold:
                suppressed[j] = True
    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)


def class_aware_nms(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Class-aware NMS — runs NMS independently within each class.

    Returns [K] long indices into the input arrays.
    """
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes_xyxy.device)
    keep_indices = []
    unique_classes = torch.unique(classes)
    for c in unique_classes.tolist():
        mask = (classes == c)
        cls_idx = mask.nonzero(as_tuple=True)[0]
        if cls_idx.numel() == 0:
            continue
        kept_local = nms(boxes_xyxy[cls_idx], scores[cls_idx], iou_threshold)
        keep_indices.extend(cls_idx[kept_local].tolist())
    if not keep_indices:
        return torch.zeros((0,), dtype=torch.long, device=boxes_xyxy.device)
    out = torch.tensor(keep_indices, dtype=torch.long, device=boxes_xyxy.device)
    # Sort by descending score for downstream readability
    order = scores[out].argsort(descending=True)
    return out[order]
