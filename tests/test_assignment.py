"""Tests for dpc/assignment.py — verifies Theorem 3 properties (a), (b), (c).

These tests are the empirical correctness check for the central paper
contribution. If any of them fail, the implementation has diverged from
the math.
"""

from __future__ import annotations

import pytest
import torch

from dpc.assignment import (
    assignment_gap,
    hungarian_assign,
    modulate_cost_matrix,
)


def test_property_a_recovery_under_clean_inputs():
    """Theorem 3 (a): if β = 0 everywhere, π*_DPC = π*_base."""
    # 5 predictions × 3 ground-truths
    cost_base = torch.tensor([
        [1.0, 5.0, 9.0],
        [2.0, 4.0, 8.0],
        [3.0, 1.0, 7.0],
        [4.0, 2.0, 6.0],
        [5.0, 3.0, 1.0],
    ])
    beta_zero = torch.zeros(5)
    cost_dpc = modulate_cost_matrix(cost_base, beta_zero, lambda_match=10.0)

    # With β = 0, the modulated cost equals the base cost
    assert torch.allclose(cost_dpc, cost_base)

    pred_base, gt_base = hungarian_assign(cost_base)
    pred_dpc, gt_dpc = hungarian_assign(cost_dpc)
    assert torch.equal(pred_base, pred_dpc)
    assert torch.equal(gt_base, gt_dpc)


def test_property_b_suppression_under_concentrated_suspicion():
    """Theorem 3 (b): with sufficient lambda_match, assignment redirects.

    Single-ground-truth setup (M=1, N=3) to isolate the property without
    cross-assignment confounds. Preds 0, 1 are suspect; pred 2 is clean.
    Under base cost, gt0 prefers pred 0. With lambda_match*(tau-tau')
    larger than the assignment gap gamma_0(R), Theorem 3 (b) guarantees
    redirection.
    """
    cost_base = torch.tensor([
        [1.0],
        [1.5],
        [3.0],
    ])
    beta = torch.tensor([0.9, 0.9, 0.0])

    pred_base, _ = hungarian_assign(cost_base)
    assert pred_base.item() == 0

    # gamma_0(R) = 3.0 - 1.0 = 2.0; lambda_match * 0.9 = 9.0 > 2.0
    cost_dpc = modulate_cost_matrix(cost_base, beta, lambda_match=10.0)
    pred_dpc, _ = hungarian_assign(cost_dpc)
    assert pred_dpc.item() == 2

    # Sanity: with insufficient lambda_match, redirection does NOT occur
    cost_dpc_weak = modulate_cost_matrix(cost_base, beta, lambda_match=1.0)
    pred_dpc_weak, _ = hungarian_assign(cost_dpc_weak)
    assert pred_dpc_weak.item() == 0

def test_property_c_lipschitz_stability():
    """Theorem 3 (c): small β perturbations don't flip the assignment."""
    cost_base = torch.tensor([
        [1.0, 5.0, 9.0],
        [3.0, 1.0, 7.0],
        [5.0, 3.0, 1.0],
        [7.0, 4.0, 5.0],
    ])
    beta1 = torch.tensor([0.10, 0.12, 0.08, 0.11])
    beta2 = torch.tensor([0.10, 0.12, 0.08, 0.11]) + torch.tensor([0.001, -0.002, 0.001, 0.0])

    # λ_match small enough that ‖β − β'‖_∞ * λ_match * M = 0.002 * 0.1 * 3 < γ_min
    cost1 = modulate_cost_matrix(cost_base, beta1, lambda_match=0.1)
    cost2 = modulate_cost_matrix(cost_base, beta2, lambda_match=0.1)
    pred1, gt1 = hungarian_assign(cost1)
    pred2, gt2 = hungarian_assign(cost2)

    # The assignment as a mapping (gt -> pred) should be identical
    map1 = {int(g): int(p) for g, p in zip(gt1.tolist(), pred1.tolist())}
    map2 = {int(g): int(p) for g, p in zip(gt2.tolist(), pred2.tolist())}
    assert map1 == map2


def test_modulate_cost_matrix_shape_validation():
    """Misshapen inputs must raise."""
    cost = torch.zeros(5, 3)
    with pytest.raises(ValueError):
        modulate_cost_matrix(cost, torch.zeros(4), lambda_match=1.0)
    with pytest.raises(ValueError):
        modulate_cost_matrix(cost, torch.zeros(5), lambda_match=-0.1)
    with pytest.raises(ValueError):
        modulate_cost_matrix(torch.zeros(5), torch.zeros(5), lambda_match=1.0)


def test_hungarian_empty_gt():
    """Empty M (no ground truths) returns empty result without raising."""
    cost = torch.zeros(5, 0)
    pred, gt = hungarian_assign(cost)
    assert pred.numel() == 0
    assert gt.numel() == 0


def test_hungarian_requires_n_ge_m():
    """N < M raises (one-to-one not possible)."""
    cost = torch.zeros(2, 5)
    with pytest.raises(ValueError):
        hungarian_assign(cost)


def test_assignment_gap_diagnostic():
    """Verify the gap diagnostic computes column-wise differences correctly."""
    cost = torch.tensor([
        [1.0, 10.0],
        [3.0, 2.0],
        [5.0, 8.0],
    ])
    # gt 0: sorted costs are [1.0, 3.0, 5.0]; gap = 3.0 - 1.0 = 2.0
    # gt 1: sorted costs are [2.0, 8.0, 10.0]; gap = 8.0 - 2.0 = 6.0
    gaps = assignment_gap(cost)
    assert torch.allclose(gaps, torch.tensor([2.0, 6.0]))
