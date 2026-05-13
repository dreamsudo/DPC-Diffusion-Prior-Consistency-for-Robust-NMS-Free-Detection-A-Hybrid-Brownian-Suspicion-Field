"""DPC configuration — single source of all hyperparameters.

Every field is validated at construction time. Field naming follows the
paper's symbol conventions where reasonable (e.g., `lambda_match` for
λ_match, `lambda_obj` for λ_obj).

The config also derives the fixed K-probe timesteps from σ_min, σ_max, and
K once at validation time; this list is then stable for the lifetime of the
config object.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import torch

from .diffusion import get_alpha_bars, make_beta_schedule, make_fixed_probe_timesteps


@dataclass
class DPCConfig:
    """Runtime configuration. All hyperparameters live here."""

    # ─── §5.1 Diffusion schedule ──────────────────────────────────────────
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Phase 1 training samples t uniformly in [timestep_min, timestep_max).
    # Phase 2 training and field inference use the fixed schedule below.
    timestep_min: int = 0
    timestep_max: int = 1000

    # ─── §5.4 K-probe schedule ────────────────────────────────────────────
    n_probes: int = 8                  # K
    sigma_min: float = 0.05            # paper §5.4: "approximately 0.05"
    sigma_max: float = 0.5             # paper §5.4: "to 0.5"
    # probe_timesteps is derived from the above at validation time.
    probe_timesteps: list[int] = field(default_factory=list)

    # ─── §5.9 Low-resolution probe ────────────────────────────────────────
    probe_res: int = 128               # paper §5.9 default

    # ─── §5.6 Smoothing ───────────────────────────────────────────────────
    smoothing_sigma: float = 1.5       # σ_smooth in pixels at probe_res
    smoothing_kernel_size: int = 7

    # ─── §5.5, §5.8 Residual operators and fusion ─────────────────────────
    # Per Eq. (12), deployment_mode selects which branch's normalized field
    # is fed to box pooling.
    deployment_mode: str = "hybrid"    # "l1", "l2", or "hybrid"
    fusion_weight_l1: float = 0.5      # w_1 in Eq. (11); w_2 = 1 - w_1

    # ─── §5.11, §5.12 Per-prediction calibration ──────────────────────────
    lambda_obj: float = 50.0           # Eq. (16)
    lambda_cls: float = 50.0           # Eq. (17) / (17')
    class_calibration_mode: str = "uniform"   # "uniform" (Eq. 17) or "argmax" (Eq. 17')

    # ─── §5.15 Small-target amplification ─────────────────────────────────
    lambda_small: float = 0.5          # Eq. (21); multiplier
    small_target_area_threshold: float = 0.01   # a_min; fraction of image area

    # ─── §5.16 Assignment-cost modulation (training only) ─────────────────
    lambda_match: float = 10.0         # Eq. (22)

    # ─── §5.13, §5.14 Auxiliary training losses ───────────────────────────
    lambda_entropy: float = 0.1        # weight on L_cls-prior (Eq. 19)
    lambda_locstab: float = 0.1        # weight on L_box-stab (Eq. 20)

    # ─── Unified detector objective (§5.17) ───────────────────────────────
    # YOLO26 native losses (passed through to its training)
    lambda_loc: float = 1.0
    lambda_conf: float = 1.0
    # Phase 1 / Phase 2 diffusion MSE weight
    lambda_mse: float = 1.0
    # Weight-anchor regularizer to prevent COCO drift during Phase 2 fine-tune
    lambda_anchor: float = 1.0

    # ─── §5.10 Box pooling ────────────────────────────────────────────────
    pool_size: int = 7                 # P in Eq. (15)

    # ─── Architecture ─────────────────────────────────────────────────────
    use_attention: bool = True

    # ─── Device ───────────────────────────────────────────────────────────
    device: str = "auto"

    def __post_init__(self):
        self.validate_and_resolve()

    def validate_and_resolve(self) -> None:
        """Raise on misconfiguration; derive probe_timesteps if not supplied."""
        if not (0 <= self.timestep_min < self.timestep_max <= self.diffusion_steps):
            raise ValueError(
                f"need 0 <= timestep_min ({self.timestep_min}) < "
                f"timestep_max ({self.timestep_max}) <= "
                f"diffusion_steps ({self.diffusion_steps})"
            )
        if self.n_probes < 1:
            raise ValueError(f"n_probes must be >= 1, got {self.n_probes}")
        if self.probe_res < 32:
            raise ValueError(f"probe_res too small for U-Net: {self.probe_res}")
        if self.probe_res % 4 != 0:
            raise ValueError(
                f"probe_res must be divisible by 4 (encoder pools twice), "
                f"got {self.probe_res}"
            )
        if not (0.0 <= self.fusion_weight_l1 <= 1.0):
            raise ValueError(
                f"fusion_weight_l1 must be in [0, 1], got {self.fusion_weight_l1}"
            )
        if self.smoothing_kernel_size % 2 == 0:
            raise ValueError(
                f"smoothing_kernel_size must be odd, got {self.smoothing_kernel_size}"
            )
        if self.deployment_mode not in {"l1", "l2", "hybrid"}:
            raise ValueError(
                f"deployment_mode must be 'l1', 'l2', or 'hybrid'; got '{self.deployment_mode}'"
            )
        if self.class_calibration_mode not in {"uniform", "argmax"}:
            raise ValueError(
                f"class_calibration_mode must be 'uniform' or 'argmax'; "
                f"got '{self.class_calibration_mode}'"
            )

        # Derive the fixed K-probe timesteps from the σ-range.
        if not self.probe_timesteps:
            betas = make_beta_schedule(self.diffusion_steps, self.beta_start, self.beta_end)
            alpha_bars = get_alpha_bars(betas)
            self.probe_timesteps = make_fixed_probe_timesteps(
                K=self.n_probes,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                alpha_bars=alpha_bars,
            )

    def get_device(self) -> torch.device:
        """Resolve device='auto' to mps → cuda → cpu."""
        if self.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)

    def asdict(self) -> dict:
        return asdict(self)
