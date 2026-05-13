"""Tests for dpc/diffusion.py — §5.1 schedule and §5.4 fixed K-probe schedule."""

from __future__ import annotations

import math
import pytest
import torch

from dpc.diffusion import (
    add_noise,
    get_alpha_bars,
    log_sigma_schedule,
    make_beta_schedule,
    make_fixed_probe_timesteps,
)


def test_log_sigma_schedule_endpoints():
    """First σ equals σ_min, last equals σ_max."""
    sigmas = log_sigma_schedule(K=8, sigma_min=0.05, sigma_max=0.5)
    assert len(sigmas) == 8
    assert sigmas[0] == pytest.approx(0.05)
    assert sigmas[-1] == pytest.approx(0.5)


def test_log_sigma_schedule_log_spaced():
    """Adjacent ratios are constant for logarithmic spacing."""
    sigmas = log_sigma_schedule(K=8, sigma_min=0.05, sigma_max=0.5)
    ratios = [sigmas[k + 1] / sigmas[k] for k in range(len(sigmas) - 1)]
    for r in ratios[1:]:
        assert r == pytest.approx(ratios[0], rel=1e-6)


def test_log_sigma_schedule_validates_range():
    with pytest.raises(ValueError):
        log_sigma_schedule(K=8, sigma_min=0.5, sigma_max=0.05)  # min >= max
    with pytest.raises(ValueError):
        log_sigma_schedule(K=8, sigma_min=-0.1, sigma_max=0.5)


def test_fixed_probe_timesteps_distinct():
    """K=8 distinct integer timesteps in the standard schedule."""
    betas = make_beta_schedule(1000, 1e-4, 0.02)
    alpha_bars = get_alpha_bars(betas)
    t_indices = make_fixed_probe_timesteps(
        K=8, sigma_min=0.05, sigma_max=0.5, alpha_bars=alpha_bars,
    )
    assert len(t_indices) == 8
    assert len(set(t_indices)) == 8
    assert all(0 <= t < 1000 for t in t_indices)


def test_fixed_probe_timesteps_match_sigma_targets():
    """Inverted indices produce σ values close to the requested ones."""
    betas = make_beta_schedule(1000, 1e-4, 0.02)
    alpha_bars = get_alpha_bars(betas)
    targets = log_sigma_schedule(8, 0.05, 0.5)
    t_indices = make_fixed_probe_timesteps(
        K=8, sigma_min=0.05, sigma_max=0.5, alpha_bars=alpha_bars,
    )
    schedule_sigmas = (1.0 - alpha_bars).sqrt()
    for target, t in zip(targets, t_indices):
        recovered = schedule_sigmas[t].item()
        # Each recovered σ should be within one discrete step of the target
        assert recovered >= target  # by construction of searchsorted
        assert recovered - target < 0.02  # discretization tolerance


def test_add_noise_marginal_matches_eq_1():
    """Verify Eq. (1): x_t = √α_bar_t · x_0 + √(1 − α_bar_t) · ε."""
    betas = make_beta_schedule(1000, 1e-4, 0.02)
    alpha_bars = get_alpha_bars(betas)
    x0 = torch.ones(2, 3, 4, 4)
    eps = torch.zeros(2, 3, 4, 4)
    t = torch.tensor([100, 500])
    out = add_noise(x0, t, eps, alpha_bars)
    # eps = 0 so x_t = √α_bar_t · x_0
    expected_scale = alpha_bars[t].sqrt().view(2, 1, 1, 1)
    expected = expected_scale * x0
    assert torch.allclose(out, expected, atol=1e-6)
