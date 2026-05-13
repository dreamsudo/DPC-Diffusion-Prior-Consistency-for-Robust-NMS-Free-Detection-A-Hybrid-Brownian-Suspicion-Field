"""DPC inference wrapper around YOLO26.

INFERENCE ONLY. Per §5.16 and the v3.3.0 design, the assignment-cost
modulation (Eq. 22) operates only during training. At inference, the
trained YOLO26 head — having been optimized under DPC-modulated
assignment — produces predictions that already reflect suspicion-aware
training. We then apply per-prediction calibration (Eqs. 16, 17, 21) as
the final filter.

Flow:
  1. YOLO26 native forward → raw head output
  2. DPC field forward → suspicion field at image resolution
  3. Box pooling per anchor → β_i (Eq. 15)
  4. Calibration → cal_obj, cal_cls (Eqs. 16, 17, 21)
  5. Top-k + threshold → final detection set
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .calibration import calibrate_predictions
from .config import DPCConfig
from .field import DPCField
from .pooling import box_areas_frac, box_pool_grid
from .yolo26_native import (
    emit_final_detections,
    forward_yolo26_raw,
    slice_raw,
)


@dataclass
class DPCDetections:
    """Per-image final detections under DPC."""

    boxes_xyxy: torch.Tensor       # [N, 4]
    scores: torch.Tensor           # [N]
    classes: torch.Tensor          # [N] long
    suspicion: torch.Tensor        # [N] β_i for diagnostics
    raw_obj_logits: object         # [N] pre-calibration objectness, or None for YOLO26
    raw_cls_logits: torch.Tensor   # [N, K_cls] pre-calibration class logits


class DPCWrapper(nn.Module):
    """Combine YOLO26 native forward, suspicion field, calibration."""

    def __init__(
        self,
        yolo_model: nn.Module,
        denoiser: nn.Module,
        cfg: DPCConfig,
        score_threshold: float = 0.25,
        top_k: int = 300,
        n_classes: int = 80,
    ):
        super().__init__()
        self.yolo = yolo_model
        self.cfg = cfg
        self.score_threshold = float(score_threshold)
        self.top_k = int(top_k)
        self.n_classes = int(n_classes)
        self.field = DPCField(denoiser, cfg)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> list[DPCDetections]:
        """Run end-to-end DPC-wrapped inference on a batch."""
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be [B, 3, H, W], got {images.shape}")
        B, _, H, W = images.shape
        device = images.device

        raw = forward_yolo26_raw(self.yolo, images, n_classes=self.n_classes)
        field_out = self.field(images, return_branches=False, generator=generator)
        deployed = field_out["deployed"]

        results: list[DPCDetections] = []
        for i in range(B):
            raw_i = slice_raw(raw, i)
            field_i = deployed[i : i + 1]
            A = raw_i.boxes_xyxy.shape[0]

            if A == 0:
                results.append(self._empty_detections(device, raw_i.cls_logits.dtype))
                continue

            image_idx = torch.zeros((A,), dtype=torch.long, device=device)
            beta = box_pool_grid(
                field_i, raw_i.boxes_xyxy, image_idx, pool_size=self.cfg.pool_size
            )
            areas_frac = box_areas_frac(raw_i.boxes_xyxy, (H, W))

            cal_obj, cal_cls, beta_small = calibrate_predictions(
                obj_logits=raw_i.obj_logits,
                cls_logits=raw_i.cls_logits,
                beta=beta,
                box_areas_frac=areas_frac,
                lambda_obj=self.cfg.lambda_obj,
                lambda_cls=self.cfg.lambda_cls,
                lambda_small=self.cfg.lambda_small,
                a_min=self.cfg.small_target_area_threshold,
                class_mode=self.cfg.class_calibration_mode,
            )

            # YOLO26: no separate objectness. The combined score is just
            # the max sigmoid-class score (Eq. 16 and Eq. 17 collapsed).
            cls_score = cal_cls.sigmoid()
            best_cls_score, best_cls = cls_score.max(dim=-1)
            if cal_obj is not None:
                combined = cal_obj.sigmoid() * best_cls_score
            else:
                combined = best_cls_score

            keep = combined >= self.score_threshold
            boxes = raw_i.boxes_xyxy[keep]
            scores = combined[keep]
            classes = best_cls[keep]
            kept_beta = beta_small[keep]
            kept_raw_obj = raw_i.obj_logits[keep] if raw_i.obj_logits is not None else None
            kept_raw_cls = raw_i.cls_logits[keep]

            if scores.shape[0] > self.top_k:
                top_scores, top_idx = scores.topk(self.top_k)
                boxes = boxes[top_idx]
                scores = top_scores
                classes = classes[top_idx]
                kept_beta = kept_beta[top_idx]
                if kept_raw_obj is not None:
                    kept_raw_obj = kept_raw_obj[top_idx]
                kept_raw_cls = kept_raw_cls[top_idx]

            results.append(DPCDetections(
                boxes_xyxy=boxes,
                scores=scores,
                classes=classes,
                suspicion=kept_beta,
                raw_obj_logits=kept_raw_obj,
                raw_cls_logits=kept_raw_cls,
            ))

        return results

    def _empty_detections(self, device, dtype) -> DPCDetections:
        return DPCDetections(
            boxes_xyxy=torch.zeros((0, 4), device=device, dtype=dtype),
            scores=torch.zeros((0,), device=device, dtype=dtype),
            classes=torch.zeros((0,), device=device, dtype=torch.long),
            suspicion=torch.zeros((0,), device=device, dtype=dtype),
            raw_obj_logits=None,
            raw_cls_logits=torch.zeros((0, self.n_classes), device=device, dtype=dtype),
        )

    @torch.no_grad()
    def baseline_detections(self, images: torch.Tensor) -> list[dict]:
        """Run YOLO26 alone (no DPC) for baseline comparison."""
        raw = forward_yolo26_raw(self.yolo, images, n_classes=self.n_classes)
        B = images.shape[0]
        return [
            emit_final_detections(
                slice_raw(raw, i),
                score_threshold=self.score_threshold,
                top_k=self.top_k,
            )
            for i in range(B)
        ]
