"""Assignment-cost modulation — the framework's central contribution.

Implements §5.16 of the paper. Modulates YOLO26's one-to-one assignment
cost matrix during training with a suspicion-dependent term:

  Eq. (22)  C_DPC(i, k) = C_base(i, k) + λ_match · β_i^(small)
  Eq. (23)  π*_DPC = arg min_π Σ_k C_DPC(π(k), k)

Theorem 3 establishes three properties of the modulated assignment:
  (a) Recovery under clean inputs: β = 0 ⇒ π*_DPC = π*_base
  (b) Suppression under concentrated suspicion: with λ_match large enough
      relative to the assignment gap, matches redirect away from suspect
      regions
  (c) Lipschitz stability: small perturbations of β do not flip the
      assignment

This module operates ONLY during training. At inference time, YOLO26 (in
its NMS-free configuration) does not have a ground-truth Hungarian step
to modulate. The trained head — having been optimized under modulated
assignment — produces predictions that already reflect the suspicion-aware
training signal. See docs/NOTES_LOG.md for the paper edits this
interpretation requires.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def modulate_cost_matrix(
    cost_base: torch.Tensor,
    beta_small: torch.Tensor,
    lambda_match: float,
) -> torch.Tensor:
    """Add the DPC suspicion term to YOLO26's base assignment cost.

    Implements §5.16 / Eq. (22):
        C_DPC(i, k) = C_base(i, k) + λ_match · β_i^(small)

    Theorem 3 property (a) Recovery: if β_small is all zeros, this
    function returns cost_base unchanged.

    Theorem 3 property (c) Lipschitz: the output is Lipschitz in β_small
    with constant λ_match. The Lipschitz bound on the assignment π* itself
    follows from the standard linear-assignment sensitivity result.

    Args:
      cost_base:    [N, M] base assignment cost matrix (N predictions, M ground-truths)
      beta_small:   [N] per-prediction suspicion after small-target amplification
      lambda_match: λ_match ≥ 0

    Returns:
      [N, M] modulated cost matrix
    """
    if cost_base.dim() != 2:
        raise ValueError(f"cost_base must be [N, M], got {cost_base.shape}")
    if cost_base.shape[0] != beta_small.shape[0]:
        raise ValueError(
            f"prediction batch mismatch: cost_base N={cost_base.shape[0]}, "
            f"beta_small N={beta_small.shape[0]}"
        )
    if lambda_match < 0:
        raise ValueError(f"lambda_match must be >= 0, got {lambda_match}")

    # The suspicion penalty is per-prediction (row-wise), broadcast across all M ground truths
    return cost_base + lambda_match * beta_small.unsqueeze(-1)


def hungarian_assign(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute optimal one-to-one assignment via the Hungarian algorithm.

    Implements §5.16 / Eq. (23):
        π* = arg min_π Σ_k C(π(k), k)
    subject to the injectivity constraint that π is one-to-one.

    Uses scipy.optimize.linear_sum_assignment (a CPU operation; YOLO26
    assignment matrices are small enough that this is milliseconds).

    Args:
      cost: [N, M] cost matrix (N predictions ≥ M ground-truths typical)

    Returns:
      pred_indices: [M] long, the prediction index matched to each ground-truth
      gt_indices:   [M] long, just torch.arange(M); kept for symmetry with
                    scipy's API
    """
    if cost.dim() != 2:
        raise ValueError(f"cost must be [N, M], got {cost.shape}")
    N, M = cost.shape
    if M == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=cost.device),
            torch.zeros((0,), dtype=torch.long, device=cost.device),
        )
    if N < M:
        raise ValueError(
            f"need N >= M for one-to-one assignment; got N={N}, M={M}"
        )

    # scipy expects numpy; move to CPU for the call, then put result back on device
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    # linear_sum_assignment with rectangular cost matrix:
    #   row_ind has shape [M] (selected predictions)
    #   col_ind has shape [M] (ground-truth indices, in order)
    pred = torch.as_tensor(row_ind, dtype=torch.long, device=cost.device)
    gt = torch.as_tensor(col_ind, dtype=torch.long, device=cost.device)
    return pred, gt


def assignment_gap(cost: torch.Tensor) -> torch.Tensor:
    """Compute the per-ground-truth assignment gap, a diagnostic.

    The "gap" for ground-truth k is the cost difference between the
    optimal prediction and the next-best prediction. A large gap means the
    Hungarian solver is confident about that match; a small gap means the
    match is fragile and could flip under modest perturbations.

    Theorem 3 property (b)'s suppression condition (λ_match · (τ − τ') > γ_k(R))
    is stated in terms of this gap. Reporting the gap distribution during
    training helps validate that λ_match is appropriately scaled.

    Args:
      cost: [N, M] cost matrix

    Returns:
      [M] per-ground-truth gap (difference between min and second-min cost)
    """
    if cost.dim() != 2:
        raise ValueError(f"cost must be [N, M], got {cost.shape}")
    N, M = cost.shape
    if M == 0:
        return torch.zeros((0,), device=cost.device, dtype=cost.dtype)
    if N < 2:
        return torch.zeros((M,), device=cost.device, dtype=cost.dtype)

    # For each column, take the two smallest entries
    top2 = cost.topk(k=2, dim=0, largest=False).values  # [2, M]
    return top2[1] - top2[0]
