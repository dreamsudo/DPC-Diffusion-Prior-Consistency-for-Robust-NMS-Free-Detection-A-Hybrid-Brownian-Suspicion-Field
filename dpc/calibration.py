"""Per-prediction calibration: convert per-box suspicion to logit offsets.

Implements §5.11 (objectness calibration), §5.12 (class-logit suppression),
and §5.15 (small-target amplification).

The flow at inference time is:
  1. Box pooling produces β_i ∈ [0, 1] per detection (Eq. 14, see pooling.py)
  2. Small-target amplification produces β_i^(small) (Eq. 21)
  3. Objectness calibration: s'_i = s_i − λ_obj · β_i^(small)    (Eq. 16)
  4. Class calibration: z'_i = z_i − λ_cls · β_i^(small) · mask   (Eq. 17 or 17')

Where `mask` is either the all-ones vector (uniform shift) or a one-hot
on the argmax class (argmax-only variant).
"""

from __future__ import annotations

import torch


# ─── §5.15 Small-target amplification ────────────────────────────────────────


def amplify_small_targets(
    beta: torch.Tensor,
    box_areas_frac: torch.Tensor,
    lambda_small: float,
    a_min: float,
) -> torch.Tensor:
    """Amplify suspicion for small-target boxes.

    Implements §5.15 / Eq. (21):
        β_i^(small) = β_i · (1 + λ_small · 𝟙[A(b̂_i) < a_min])

    Args:
      beta:           [N] per-box suspicion in [0, 1]
      box_areas_frac: [N] each box's area as a fraction of image area
      lambda_small:   amplification factor λ_small (≥ 0)
      a_min:          area threshold below which a box is "small"

    Returns:
      [N] amplified suspicion β^(small). Note this can exceed 1.0 for small
      boxes — that's intended; the downstream λ values control overall scale.
    """
    is_small = (box_areas_frac < a_min).to(beta.dtype)
    return beta * (1.0 + lambda_small * is_small)


# ─── §5.11 Objectness calibration ────────────────────────────────────────────


def calibrate_objectness(
    obj_logits: torch.Tensor,
    beta_small: torch.Tensor,
    lambda_obj: float,
) -> torch.Tensor:
    """Subtract a suspicion penalty from each objectness logit.

    Implements §5.11 / Eq. (16): s'_i = s_i − λ_obj · β_i^(small).

    Args:
      obj_logits: [N] raw objectness logits
      beta_small: [N] per-box suspicion (after small-target amplification)
      lambda_obj: calibration constant (≥ 0)

    Returns:
      [N] calibrated objectness logits
    """
    if obj_logits is None:
        return None
    if obj_logits.shape != beta_small.shape:
        raise ValueError(
            f"obj_logits {obj_logits.shape} and beta_small {beta_small.shape} must match"
        )
    return obj_logits - lambda_obj * beta_small


# ─── §5.12 Class-logit calibration ───────────────────────────────────────────


def calibrate_class_uniform(
    cls_logits: torch.Tensor,
    beta_small: torch.Tensor,
    lambda_cls: float,
) -> torch.Tensor:
    """Uniform additive shift across all class channels.

    Implements §5.12 / Eq. (17):
        z'_i = z_i − λ_cls · β_i^(small) · 𝟙_{K_cls}

    A uniform additive shift produces no change in the softmax output
    (softmax is shift-invariant), but does preserve a decremented logit
    level that downstream score thresholds can read.

    Args:
      cls_logits: [N, K_cls] raw class logits
      beta_small: [N] per-box suspicion (after small-target amplification)
      lambda_cls: calibration constant (≥ 0)

    Returns:
      [N, K_cls] calibrated class logits
    """
    if cls_logits.dim() != 2:
        raise ValueError(f"cls_logits must be [N, K_cls], got {cls_logits.shape}")
    if cls_logits.shape[0] != beta_small.shape[0]:
        raise ValueError(
            f"batch mismatch: cls_logits {cls_logits.shape} vs beta_small {beta_small.shape}"
        )
    shift = (lambda_cls * beta_small).unsqueeze(-1)  # [N, 1]
    return cls_logits - shift


def calibrate_class_argmax(
    cls_logits: torch.Tensor,
    beta_small: torch.Tensor,
    lambda_cls: float,
) -> torch.Tensor:
    """Penalize only the top-predicted class (optional Eq. 17' variant).

    Implements §5.12 / Eq. (17'):
        z'_{i,c} = z_{i,c} − λ_cls · β_i^(small) · 𝟙[c = argmax_c' z_{i,c'}]

    Unlike Eq. (17), this is NOT softmax-invariant — it actively reduces
    the top-class probability mass under softmax.

    Args:
      cls_logits: [N, K_cls] raw class logits
      beta_small: [N] per-box suspicion (after small-target amplification)
      lambda_cls: calibration constant (≥ 0)

    Returns:
      [N, K_cls] calibrated class logits
    """
    if cls_logits.dim() != 2:
        raise ValueError(f"cls_logits must be [N, K_cls], got {cls_logits.shape}")
    if cls_logits.shape[0] != beta_small.shape[0]:
        raise ValueError(
            f"batch mismatch: cls_logits {cls_logits.shape} vs beta_small {beta_small.shape}"
        )
    out = cls_logits.clone()
    top_idx = cls_logits.argmax(dim=-1)  # [N]
    penalty = lambda_cls * beta_small  # [N]
    out[torch.arange(out.shape[0]), top_idx] = out[torch.arange(out.shape[0]), top_idx] - penalty
    return out


# ─── Convenience: full per-prediction calibration ────────────────────────────


def calibrate_predictions(
    obj_logits,
    cls_logits: torch.Tensor,
    beta: torch.Tensor,
    box_areas_frac: torch.Tensor,
    lambda_obj: float,
    lambda_cls: float,
    lambda_small: float,
    a_min: float,
    class_mode: str = "uniform",
) -> tuple:
    """Apply small-target amplification then objectness and class calibration.

    Composes §5.15 / Eq. (21), §5.11 / Eq. (16), and §5.12 / Eq. (17 or 17').

    For YOLO26 the head has no separate objectness logit (Sapkota
    arXiv:2509.25164) — `obj_logits` is None and Eqs. (16) and (17)
    collapse into a single class-logit shift with effective magnitude
    (λ_obj + λ_cls). Since both Eq. (16) and Eq. (17, uniform variant) are
    additive shifts on the same logit pre-image, this collapse preserves
    the paper's intended calibration semantics exactly. See
    docs/NOTES_LOG.md for the paper edit this requires.

    Args:
      obj_logits:     [N] raw objectness logits, OR None for YOLO26 heads
      cls_logits:     [N, K_cls] raw class logits
      beta:           [N] per-box suspicion in [0, 1] (from pooling)
      box_areas_frac: [N] each box's area as fraction of image area
      lambda_obj:     λ_obj from Eq. (16)
      lambda_cls:     λ_cls from Eq. (17)
      lambda_small:   λ_small from Eq. (21)
      a_min:          a_min from Eq. (21)
      class_mode:     "uniform" (Eq. 17) or "argmax" (Eq. 17')

    Returns:
      cal_obj:    [N] calibrated objectness logits, OR None if input was None
      cal_cls:    [N, K_cls] calibrated class logits
      beta_small: [N] amplified per-box suspicion (for diagnostics)
    """
    if class_mode not in {"uniform", "argmax"}:
        raise ValueError(f"class_mode must be 'uniform' or 'argmax'; got '{class_mode}'")

    beta_small = amplify_small_targets(beta, box_areas_frac, lambda_small, a_min)

    if obj_logits is not None:
        # Classic YOLOv5/v8-style head: Eq. (16) on objectness, Eq. (17) on class
        cal_obj = calibrate_objectness(obj_logits, beta_small, lambda_obj)
        effective_cls_lambda = lambda_cls
    else:
        # YOLO26 head: Eqs. (16) and (17) collapse — apply combined shift to cls
        cal_obj = None
        effective_cls_lambda = lambda_obj + lambda_cls

    if class_mode == "uniform":
        cal_cls = calibrate_class_uniform(cls_logits, beta_small, effective_cls_lambda)
    else:
        cal_cls = calibrate_class_argmax(cls_logits, beta_small, effective_cls_lambda)
    return cal_obj, cal_cls, beta_small
