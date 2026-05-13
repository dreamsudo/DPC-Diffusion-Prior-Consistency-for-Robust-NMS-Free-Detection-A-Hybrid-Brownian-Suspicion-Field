"""Tests for dpc/auxiliary_losses.py — Eqs. (18), (19), (20)."""

from __future__ import annotations

import math
import pytest
import torch

from dpc.auxiliary_losses import (
    box_stability_loss,
    class_entropy_regularizer,
    per_prediction_entropy,
)


def test_per_prediction_entropy_uniform_is_log_K():
    """Uniform K-class distribution has entropy log K."""
    K = 8
    probs = torch.ones(1, K) / K
    H = per_prediction_entropy(probs)
    assert torch.allclose(H, torch.tensor([math.log(K)]), atol=1e-5)


def test_per_prediction_entropy_one_hot_is_zero():
    """One-hot has entropy 0 (up to log(eps) ≈ 0)."""
    K = 8
    probs = torch.zeros(1, K)
    probs[0, 3] = 1.0
    H = per_prediction_entropy(probs)
    assert H.item() < 1e-3


def test_class_entropy_regularizer_zero_when_beta_zero():
    """Eq. (19): L_cls-prior = 0 when β = 0."""
    K = 5
    logits = torch.randn(10, K)
    beta = torch.zeros(10)
    reg = class_entropy_regularizer(logits, beta, K)
    assert reg.item() == 0.0


def test_class_entropy_regularizer_zero_when_uniform_predictions():
    """Eq. (19): L_cls-prior = 0 when all predictions are uniform (H = H_max)."""
    K = 5
    # Logits = constant → softmax is uniform → H = log K = H_max
    logits = torch.zeros(10, K)
    beta = torch.ones(10) * 0.5
    reg = class_entropy_regularizer(logits, beta, K)
    assert abs(reg.item()) < 1e-5


def test_class_entropy_regularizer_large_when_confident_and_suspect():
    """Eq. (19): large only when β large AND H small."""
    K = 5
    # Very confident predictions (one-hot-ish)
    logits = torch.zeros(3, K)
    logits[:, 0] = 100.0
    beta = torch.ones(3)
    reg = class_entropy_regularizer(logits, beta, K)
    # Each row contributes ≈ H_max − 0 = log 5 ≈ 1.609
    # Sum over 3 rows ≈ 4.83
    expected = 3 * math.log(K)
    assert abs(reg.item() - expected) < 0.01


def test_box_stability_loss_zero_when_identical():
    """Eq. (20): L_box-stab = 0 when both probe forward passes produce the same boxes."""
    boxes = torch.randn(5, 4)
    loss = box_stability_loss(boxes, boxes.clone())
    assert loss.item() == 0.0


def test_box_stability_loss_linear_in_shift():
    """Eq. (20): output is the L1 sum of per-box coordinate differences."""
    boxes_a = torch.zeros(2, 4)
    boxes_b = torch.ones(2, 4) * 3.0  # each coord shifts by 3
    loss = box_stability_loss(boxes_a, boxes_b)
    # Expected: 2 boxes × 4 coords × |3| = 24
    assert loss.item() == pytest.approx(24.0)


def test_box_stability_loss_empty():
    """Empty inputs return zero, not raise."""
    boxes = torch.zeros(0, 4)
    loss = box_stability_loss(boxes, boxes.clone())
    assert loss.item() == 0.0


def test_box_stability_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        box_stability_loss(torch.zeros(3, 4), torch.zeros(4, 4))
