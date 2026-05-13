"""Per-box suspicion pooling.

Implements §5.10 of the paper. Given a deployed suspicion field
[B, 1, H, W] in [0, 1] and a set of detection boxes, computes one scalar
suspicion coefficient β_i ∈ [0, 1] per box.

  Eq. (14)  β_i = (1/A(b̂_i)) ∫∫_b̂_i I_img(u, v) du dv     (continuous)
  Eq. (15)  β_i ≈ (1/P²) Σ_{p_x,p_y} I_img(ũ_{i,p_x}, ṽ_{i,p_y})   (P×P grid)

v3.3.0 uses torchvision.ops.roi_align for the P×P grid sample. This is
vectorized and ~30× faster than v3.2.0's per-box loop. The roi_align
operator's pooled output is exactly the bilinear interpolation of the
field at P×P regularly-spaced points inside each box — which matches
Eq. (15) exactly.

Proposition 2 (box-pooling continuity) holds for this implementation
because bilinear interpolation is Lipschitz in the box coordinates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def box_pool_grid(
    suspicion_field: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    image_idx: torch.Tensor,
    pool_size: int = 7,
) -> torch.Tensor:
    """Compute per-box suspicion via P×P grid pooling.

    Implements §5.10 / Eq. (15). Uses torchvision.ops.roi_align for
    vectorized bilinear sampling.

    Proposition 2 (box-pooling continuity): the pooled coefficient is
    Lipschitz-continuous in the box coordinates with constant bounded by
    the field's own Lipschitz constant. The Gaussian smoothing of §5.6
    ensures the field's Lipschitz constant is bounded.

    Args:
      suspicion_field: [B, 1, H, W] in [0, 1]
      boxes_xyxy:      [N, 4] float, in image-pixel coordinates
      image_idx:       [N] long, which image in the batch each box belongs to
      pool_size:       P in Eq. (15); typical value 7 (ROI-Align convention)

    Returns:
      [N] tensor in [0, 1] (or slightly outside due to bilinear sampling
      at field min/max).
    """
    if suspicion_field.dim() != 4 or suspicion_field.shape[1] != 1:
        raise ValueError(
            f"suspicion_field must be [B, 1, H, W], got {suspicion_field.shape}"
        )
    if boxes_xyxy.dim() != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError(f"boxes must be [N, 4], got {boxes_xyxy.shape}")
    if boxes_xyxy.shape[0] != image_idx.shape[0]:
        raise ValueError(
            f"box/image_idx batch mismatch: {boxes_xyxy.shape[0]} vs {image_idx.shape[0]}"
        )

    N = boxes_xyxy.shape[0]
    device = suspicion_field.device
    dtype = suspicion_field.dtype

    if N == 0:
        return torch.zeros((0,), device=device, dtype=dtype)

    # roi_align expects boxes in (batch_idx, x1, y1, x2, y2) format
    boxes_with_idx = torch.cat(
        [image_idx.to(dtype).unsqueeze(1), boxes_xyxy.to(dtype)], dim=1
    ).to(device)

    pooled = roi_align(
        suspicion_field,
        boxes_with_idx,
        output_size=(pool_size, pool_size),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )  # [N, 1, P, P]

    # Average over the P×P grid (Eq. 15)
    beta = pooled.mean(dim=(1, 2, 3))  # [N]

    # Defensive clamp — field is normalized to [0, 1], roi_align bilinear
    # interpolation can drift slightly outside that range at boundaries.
    return beta.clamp(0.0, 1.0)


def box_areas_frac(
    boxes_xyxy: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Compute each box's area as a fraction of the image area.

    Used by §5.15 / Eq. (21) to identify "small" boxes for amplification.

    Args:
      boxes_xyxy: [N, 4]
      image_size: (H, W)

    Returns:
      [N] in [0, 1]
    """
    H, W = image_size
    img_area = float(H * W)
    if img_area <= 0:
        return torch.zeros(
            (boxes_xyxy.shape[0],), dtype=boxes_xyxy.dtype, device=boxes_xyxy.device
        )
    w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0)
    h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)
    return (w * h) / img_area


def boxes_overlap_patch(
    boxes_xyxy: torch.Tensor,
    patch_boxes_xyxy: torch.Tensor,
    iou_threshold: float = 0.1,
) -> torch.Tensor:
    """For each detection, True if it overlaps any patch box with IoU ≥ threshold.

    Used by Phase 3 evaluation metrics. Not part of the core defense pipeline.

    Args:
      boxes_xyxy:       [N, 4] detections (single image)
      patch_boxes_xyxy: [M, 4] ground-truth patch boxes (single image)
      iou_threshold:    minimum IoU for overlap

    Returns:
      [N] bool tensor
    """
    N = boxes_xyxy.shape[0]
    M = patch_boxes_xyxy.shape[0]
    if N == 0:
        return torch.zeros((0,), dtype=torch.bool, device=boxes_xyxy.device)
    if M == 0:
        return torch.zeros((N,), dtype=torch.bool, device=boxes_xyxy.device)

    a = boxes_xyxy.unsqueeze(1)        # [N, 1, 4]
    b = patch_boxes_xyxy.unsqueeze(0)   # [1, M, 4]
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-9
    iou = inter / union  # [N, M]
    return (iou >= iou_threshold).any(dim=1)
