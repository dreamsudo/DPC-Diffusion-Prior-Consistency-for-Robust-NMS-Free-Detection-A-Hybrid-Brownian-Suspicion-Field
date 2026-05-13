"""Training-time auxiliary losses.

Implements §5.13 (class-entropy regularizer) and §5.14 (Brownian
localization stability). Both are used only during Phase 2 training.

  Eq. (18)  per-prediction entropy H_i = −Σ_c p_{i,c} log p_{i,c}
  Eq. (19)  L_cls-prior = Σ_i β_i · (H_max − H_i)
  Eq. (20)  L_box-stab = Σ_j ‖b̂_j(t_a) − b̂_j(t_b)‖_1

These losses operate on YOLO26's outputs during training. They are added
to the unified detector objective (§5.17) with weights cfg.lambda_entropy
and cfg.lambda_locstab.
"""

from __future__ import annotations

import math

import torch


# ─── §5.13: class-entropy regularizer ───────────────────────────────────────


def per_prediction_entropy(
    cls_probs: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Shannon entropy of each per-prediction class distribution.

    Implements §5.13 / Eq. (18): H_i = −Σ_c p_{i,c} log p_{i,c}.

    Args:
      cls_probs: [N, K_cls] non-negative, rows sum to 1 (softmax output)
      eps:       numerical stability constant inside the log

    Returns:
      [N] entropy per prediction (in nats; H_max = log K_cls)
    """
    if cls_probs.dim() != 2:
        raise ValueError(f"cls_probs must be [N, K_cls], got {cls_probs.shape}")
    return -(cls_probs * (cls_probs + eps).log()).sum(dim=-1)


def class_entropy_regularizer(
    cls_logits: torch.Tensor,
    beta: torch.Tensor,
    K_cls: int,
) -> torch.Tensor:
    """Suspicion-weighted class-entropy regularizer.

    Implements §5.13 / Eq. (19): L_cls-prior = Σ_i β_i · (H_max − H_i).

    Small when either β_i is small (no penalty) or H_i is large (uncertain
    prediction, no penalty). Large only when β_i is large AND H_i is
    small — i.e., when a confident class prediction is being made in a
    suspect region.

    Note on cls_logits → probs: we apply softmax internally. If the head
    uses sigmoid-per-class scoring, the regularizer can still be computed
    by feeding sigmoid outputs and treating them as a quasi-distribution,
    but the H_max bound changes; in v3.3.0 we assume softmax-style
    multi-class logits for this regularizer.

    Args:
      cls_logits: [N, K_cls] raw class logits
      beta:       [N] per-box suspicion in [0, 1]
      K_cls:      number of classes (provides H_max = log K_cls)

    Returns:
      scalar tensor; the sum over all predictions
    """
    if cls_logits.shape[0] != beta.shape[0]:
        raise ValueError(
            f"cls_logits {cls_logits.shape} and beta {beta.shape} batch mismatch"
        )
    if K_cls <= 1:
        return torch.zeros((), device=cls_logits.device, dtype=cls_logits.dtype)

    probs = cls_logits.softmax(dim=-1)
    H = per_prediction_entropy(probs)  # [N]
    H_max = math.log(K_cls)
    return (beta * (H_max - H)).sum()


# ─── §5.14: Brownian localization stability ─────────────────────────────────


def box_stability_loss(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """L1 distance between two box predictions under different probe perturbations.

    Implements §5.14 / Eq. (20): L_box-stab = Σ_j ‖b̂_j(t_a) − b̂_j(t_b)‖_1.

    The mechanism: a box prediction supported by stable, semantic evidence
    should be approximately invariant under small stochastic perturbations.
    Two probe forward passes at different timesteps t_a, t_b produce two
    box predictions; their disagreement penalizes the model.

    Args:
      boxes_a: [N, 4] box predictions under one probe (e.g., at t_a)
      boxes_b: [N, 4] box predictions under another probe (e.g., at t_b)

    Returns:
      scalar L1 sum across all boxes
    """
    if boxes_a.shape != boxes_b.shape:
        raise ValueError(
            f"box shapes must match: {boxes_a.shape} vs {boxes_b.shape}"
        )
    if boxes_a.numel() == 0:
        return torch.zeros((), device=boxes_a.device, dtype=boxes_a.dtype)
    return (boxes_a - boxes_b).abs().sum()
