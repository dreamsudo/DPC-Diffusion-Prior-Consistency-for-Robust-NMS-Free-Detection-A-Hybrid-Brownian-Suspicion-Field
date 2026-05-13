"""Tests for dpc/calibration.py — Eqs. (16), (17), (17'), (21)."""

from __future__ import annotations

import pytest
import torch

from dpc.calibration import (
    amplify_small_targets,
    calibrate_class_argmax,
    calibrate_class_uniform,
    calibrate_objectness,
    calibrate_predictions,
)


def test_amplify_small_targets_eq21():
    """Eq. (21): small boxes get β × (1 + λ_small)."""
    beta = torch.tensor([0.5, 0.5, 0.5])
    areas = torch.tensor([0.001, 0.05, 0.5])  # small, medium, large
    out = amplify_small_targets(beta, areas, lambda_small=1.0, a_min=0.01)
    # Only the first box (area < 0.01) is amplified
    expected = torch.tensor([1.0, 0.5, 0.5])
    assert torch.allclose(out, expected)


def test_calibrate_objectness_eq16():
    """Eq. (16): s' = s − λ_obj · β_small."""
    s = torch.tensor([2.0, 1.0, 0.0])
    beta_small = torch.tensor([0.1, 0.5, 0.9])
    out = calibrate_objectness(s, beta_small, lambda_obj=10.0)
    expected = torch.tensor([2.0 - 1.0, 1.0 - 5.0, 0.0 - 9.0])
    assert torch.allclose(out, expected)


def test_calibrate_class_uniform_eq17():
    """Eq. (17): uniform shift across all class channels."""
    z = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    beta_small = torch.tensor([0.1, 0.5])
    out = calibrate_class_uniform(z, beta_small, lambda_cls=10.0)
    # Row 0 shifted by −1.0, row 1 shifted by −5.0
    expected = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]])
    assert torch.allclose(out, expected)


def test_calibrate_class_uniform_preserves_argmax():
    """Eq. (17) does not change which class is argmax (softmax shift-invariant)."""
    z = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    beta_small = torch.tensor([0.9, 0.9])
    out = calibrate_class_uniform(z, beta_small, lambda_cls=100.0)
    assert torch.equal(out.argmax(dim=-1), z.argmax(dim=-1))


def test_calibrate_class_argmax_eq17_prime():
    """Eq. (17'): penalty applied only to the top-predicted class."""
    z = torch.tensor([[1.0, 5.0, 3.0]])
    beta_small = torch.tensor([1.0])
    out = calibrate_class_argmax(z, beta_small, lambda_cls=10.0)
    # Top class is 1 (logit 5). It should be reduced to 5 - 10 = -5.
    # Others unchanged.
    expected = torch.tensor([[1.0, -5.0, 3.0]])
    assert torch.allclose(out, expected)


def test_calibrate_class_argmax_can_flip_argmax():
    """Eq. (17') CAN change argmax (unlike Eq. 17)."""
    z = torch.tensor([[1.0, 5.0, 3.0]])
    beta_small = torch.tensor([1.0])
    out = calibrate_class_argmax(z, beta_small, lambda_cls=10.0)
    # After penalty, class 2 (logit 3) should be the new argmax
    assert out.argmax(dim=-1).item() == 2


def test_calibrate_predictions_composition():
    """Verify the full inference-time composition: amplify → obj → cls."""
    obj = torch.tensor([2.0, 1.0])
    cls = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    beta = torch.tensor([0.5, 0.5])
    areas = torch.tensor([0.001, 0.1])  # first is small, second is not

    cal_obj, cal_cls, beta_small = calibrate_predictions(
        obj, cls, beta, areas,
        lambda_obj=10.0, lambda_cls=10.0,
        lambda_small=1.0, a_min=0.01,
        class_mode="uniform",
    )

    # First box: small → β_small = 1.0; obj − 10×1.0 = −8; cls − 10
    # Second box: not small → β_small = 0.5; obj − 10×0.5 = −4; cls − 5
    assert torch.allclose(beta_small, torch.tensor([1.0, 0.5]))
    assert torch.allclose(cal_obj, torch.tensor([-8.0, -4.0]))
    assert torch.allclose(cal_cls, torch.tensor([[-9.0, -8.0], [-2.0, -1.0]]))


def test_calibrate_predictions_invalid_mode():
    with pytest.raises(ValueError):
        calibrate_predictions(
            torch.zeros(1), torch.zeros(1, 5), torch.zeros(1), torch.zeros(1),
            10.0, 10.0, 0.5, 0.01,
            class_mode="bogus",
        )
