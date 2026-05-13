"""Diffusion math.

Implements §5.1 (forward VP-SDE) and §5.4 (fixed K-probe schedule).

Pure functions, no state. All consumers (Phase 1 trainer, Phase 2 trainer,
field engine, diagnostics) import from this module so the schedule has a
single source of truth.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# ─── §5.1 Forward VP-SDE ─────────────────────────────────────────────────────


def make_beta_schedule(
    steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Linear β schedule for DDPM-style discretization of the VP-SDE.

    Implements the discrete analog of §5.1's continuous β(t) = β_min +
    (β_max − β_min)·t. Returns a [steps] float32 CPU tensor; move to device
    at use site.
    """
    if steps < 2:
        raise ValueError(f"steps must be >= 2, got {steps}")
    if not (0 < beta_start < beta_end < 1):
        raise ValueError(
            f"need 0 < beta_start ({beta_start}) < beta_end ({beta_end}) < 1"
        )
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)


def get_alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    """Cumulative product α_bar_t = Π_{s ≤ t} (1 − β_s).

    α_bar_t is the variance-preserving forward-process scale used in Eq. (1).
    """
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def get_noise_params(
    alpha_bars: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (√α_bar_t, √(1 − α_bar_t)) broadcast-ready as [B, 1, 1, 1].

    Used inside add_noise to satisfy Eq. (1): x_t = √α_bar_t · x_0 + √(1 − α_bar_t) · ε.
    """
    if t.dtype not in (torch.long, torch.int64, torch.int32):
        raise TypeError(f"t must be integer tensor, got {t.dtype}")
    sa = alpha_bars[t].sqrt().view(-1, 1, 1, 1)
    soma = (1.0 - alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
    return sa, soma


def add_noise(
    x0: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Forward diffusion: x_t = √α_bar_t · x_0 + √(1 − α_bar_t) · ε.

    Implements §5.1 / Eq. (1).

    Shapes:
      x0:         [B, C, H, W]
      t:          [B] long
      eps:        [B, C, H, W]
      alpha_bars: [T] (full schedule, indexed by t)
    """
    if x0.shape != eps.shape:
        raise ValueError(f"x0 shape {x0.shape} != eps shape {eps.shape}")
    if x0.shape[0] != t.shape[0]:
        raise ValueError(
            f"batch mismatch: x0 batch {x0.shape[0]} != t batch {t.shape[0]}"
        )
    sa, soma = get_noise_params(alpha_bars, t)
    return sa * x0 + soma * eps


def sample_timesteps(
    batch_size: int,
    t_min: int,
    t_max: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Uniform random timesteps in [t_min, t_max).

    Used by Phase 1 training (which samples random t per Eq. (2)). NOT used
    for the K-probe ensemble at inference time — those use the fixed
    schedule from log_sigma_schedule().
    """
    if t_min >= t_max:
        raise ValueError(f"need t_min ({t_min}) < t_max ({t_max})")
    return torch.randint(
        low=t_min,
        high=t_max,
        size=(batch_size,),
        device=device,
        generator=generator,
        dtype=torch.long,
    )


# ─── §5.4 Fixed K-probe schedule ─────────────────────────────────────────────


def log_sigma_schedule(
    K: int,
    sigma_min: float,
    sigma_max: float,
) -> list[float]:
    """K log-spaced σ values in [sigma_min, sigma_max].

    Implements §5.4: K fixed probes "such that σ(t_k) ranges over a
    logarithmic schedule from approximately 0.05 to 0.5." Returns
    σ_k = σ_min · (σ_max / σ_min)^((k-1)/(K-1)) for k = 1..K.

    The output is a plain Python list of floats; pass to
    make_fixed_probe_timesteps() to convert to integer step indices.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if not (0.0 < sigma_min < sigma_max < 1.0):
        raise ValueError(
            f"need 0 < sigma_min ({sigma_min}) < sigma_max ({sigma_max}) < 1"
        )
    if K == 1:
        return [math.sqrt(sigma_min * sigma_max)]  # geometric midpoint
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    return [
        math.exp(log_min + (log_max - log_min) * k / (K - 1))
        for k in range(K)
    ]


def make_fixed_probe_timesteps(
    K: int,
    sigma_min: float,
    sigma_max: float,
    alpha_bars: torch.Tensor,
) -> list[int]:
    """Convert log-σ schedule to integer step indices.

    Implements §5.4. For each σ_k in the log_sigma_schedule, finds the
    smallest step index t such that √(1 − α_bar_t) ≥ σ_k. Returns a list of
    K distinct integer indices.

    The forward process under Eq. (1) gives σ(t) = √(1 − α_bar_t), so this
    inversion is exact up to discretization.
    """
    sigmas = log_sigma_schedule(K, sigma_min, sigma_max)
    schedule_sigmas = (1.0 - alpha_bars).sqrt()  # [T]
    indices: list[int] = []
    for sigma_target in sigmas:
        # First index where schedule σ exceeds the target
        idx = int(torch.searchsorted(schedule_sigmas, torch.tensor(sigma_target)).item())
        idx = max(0, min(idx, alpha_bars.shape[0] - 1))
        indices.append(idx)
    # Deduplicate while preserving order (rare; only happens if K is very large)
    seen: set[int] = set()
    unique: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    if len(unique) != K:
        raise ValueError(
            f"could not produce K={K} distinct probe timesteps in "
            f"[σ={sigma_min:.4f}, {sigma_max:.4f}]; got {len(unique)} unique. "
            f"Increase diffusion_steps or widen the σ range."
        )
    return unique
