"""DPC suspicion field engine.

Implements §5.3 through §5.9 of the paper. The flow is:

  Eq. (4)  per-probe residual Δ_k = ε_θ(x_{t_k}, t_k) − ε_k
  Eq. (5)  K-probe ensemble of squared L2 norms  (Theorem 2)
  Eq. (6)  L2 residual r^(L2) = sqrt(Σ_c Δ_c²)
  Eq. (7)  L1 residual r^(L1) = (1/C) Σ_c |Δ_c|
  Eq. (8)  Gaussian smoothing per branch
  Eq. (9)  branch field R_m = (1/K) Σ smoothed-per-probe
  Eq. (10) per-image min-max normalization PER BRANCH
  Eq. (11) hybrid fusion R_hyb = w_1 · R̂_L1 + w_2 · R̂_L2
  Eq. (12) deployment selection (mode in {l1, l2, hybrid})
  Eq. (13) bilinear upsample to image resolution

The implementation order matters: the paper says smooth → average across
probes → normalize per branch → combine. Earlier versions of this code
fused first and normalized once; v3.3.0 fixes this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DPCConfig
from .diffusion import add_noise, get_alpha_bars, make_beta_schedule


class DPCField(nn.Module):
    """K-probe suspicion field computation, strict per §5.3–§5.9.

    The denoiser is trained at cfg.probe_res. Input images may arrive at a
    different resolution (e.g., 640 for YOLO26); we downsample to probe_res
    before the K-probe loop, then upsample the deployed field back to the
    input resolution for downstream box pooling.

    Note on score-function interpretation (§5.2 / Eq. 3): the denoiser
    output, under the optimal-denoiser assumption, satisfies
    ε*_θ(x_t, t) = −σ(t) · ∇ log p_t(x_t). This module does not invoke that
    identity directly; it computes the residual whose squared norm is, by
    Theorem 1, an unbiased estimator (up to additive constant) of the
    squared score magnitude.
    """

    def __init__(self, denoiser: nn.Module, cfg: DPCConfig):
        super().__init__()
        self.denoiser = denoiser
        self.cfg = cfg

        # Cached schedule
        betas = make_beta_schedule(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end)
        alpha_bars = get_alpha_bars(betas)
        self.register_buffer("alpha_bars", alpha_bars)

        # Fixed K-probe timesteps (§5.4). These are integer indices into alpha_bars.
        probe_t = torch.tensor(cfg.probe_timesteps, dtype=torch.long)
        if probe_t.numel() != cfg.n_probes:
            raise ValueError(
                f"cfg.probe_timesteps has {probe_t.numel()} entries; expected K={cfg.n_probes}"
            )
        self.register_buffer("probe_timesteps", probe_t)

        # 2D Gaussian smoothing kernel (Eq. 8)
        kernel = self._make_gaussian_kernel(
            cfg.smoothing_kernel_size, cfg.smoothing_sigma
        )
        self.register_buffer("smooth_kernel", kernel)

    # ─── Internals ────────────────────────────────────────────────────────

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """2D Gaussian kernel, separable construction, sum=1."""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def smooth_residuals(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian smoothing per Eq. (8).

        Implements §5.6 / Eq. (8). x is [B, 1, H, W]; applies per-channel
        convolution with the cached Gaussian kernel.
        """
        c = x.shape[1]
        kernel = self.smooth_kernel.expand(c, 1, -1, -1)
        pad = self.cfg.smoothing_kernel_size // 2
        return F.conv2d(x, kernel, padding=pad, groups=c)

    @staticmethod
    def normalize_per_image(x: torch.Tensor, delta: float = 1e-6) -> torch.Tensor:
        """Per-image min-max normalization to [0, 1].

        Implements §5.8 / Eq. (10): R̂_m = (R_m − min) / (max − min + δ).
        Operates independently on each image in the batch.
        """
        b = x.shape[0]
        flat = x.view(b, -1)
        mn = flat.min(dim=1).values.view(b, 1, 1, 1)
        mx = flat.max(dim=1).values.view(b, 1, 1, 1)
        return (x - mn) / (mx - mn + delta)

    # ─── §5.3, §5.4: per-probe residual and K-probe ensemble ──────────────

    def compute_per_probe_residual(
        self,
        x0_probe: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run K probes at the fixed schedule, return per-probe residuals.

        Implements §5.3 / Eq. (4): Δ_k(u, v) = ε_θ(x_{t_k}, t_k)(u, v) − ε_k(u, v).
        Connected to Theorem 1 (residual as score estimator): the expected
        squared norm of Δ_k is, up to scale σ(t_k)² and additive constant,
        equal to the squared score-function magnitude at (u, v).

        Args:
          x0_probe:  [B, C, h, w] at probe resolution
          generator: optional torch.Generator for reproducible probe noise

        Returns:
          residuals: [B, K, C, h, w] signed per-probe residuals Δ_k
          eps:       [B, K, C, h, w] noise that was actually injected (kept
                     for diagnostics; can be discarded by caller)
        """
        b, c, h, w = x0_probe.shape
        K = self.cfg.n_probes
        device = x0_probe.device

        # Replicate the batch K times along a new dim
        x0_rep = x0_probe.unsqueeze(1).expand(b, K, c, h, w).reshape(b * K, c, h, w)

        # Tile the fixed probe timesteps across the batch
        # probe_timesteps is [K]; we want [b*K] with each image getting the same K timesteps
        t = self.probe_timesteps.to(device).unsqueeze(0).expand(b, K).reshape(b * K)

        # Sample noise (one fresh draw per probe, per image)
        if generator is not None:
            eps = torch.randn(
                x0_rep.shape, device=device, generator=generator, dtype=x0_rep.dtype,
            )
        else:
            eps = torch.randn_like(x0_rep)

        # Forward diffusion: x_{t_k} = √α_bar_{t_k} · x_0 + √(1 − α_bar_{t_k}) · ε
        x_t = add_noise(x0_rep, t, eps, self.alpha_bars)

        # Denoiser prediction
        eps_hat = self.denoiser(x_t, t)

        # Residual (Eq. 4)
        delta = eps_hat - eps  # [b*K, C, h, w]

        # Reshape back to [B, K, C, h, w]
        delta = delta.view(b, K, c, h, w)
        eps = eps.view(b, K, c, h, w)
        return delta, eps

    def aggregate_probes(self, delta: torch.Tensor) -> torch.Tensor:
        """K-probe ensemble of squared L2 norms.

        Implements §5.4 / Eq. (5): r̄_K(u, v) = (1/K) Σ_k ‖Δ_k(u, v)‖₂².
        Connected to Theorem 2 (variance bound): variance is O(1/K), with
        independent noise samples ensured by the fresh draws in
        compute_per_probe_residual.

        Args:
          delta: [B, K, C, h, w] per-probe residuals

        Returns:
          [B, 1, h, w] K-probe ensemble of squared L2 norms
        """
        # ‖Δ_k‖₂² at each pixel: sum over channels of squares
        sq_norm_per_probe = (delta ** 2).sum(dim=2, keepdim=False)  # [B, K, h, w]
        # Average across K probes
        ensemble = sq_norm_per_probe.mean(dim=1, keepdim=True)  # [B, 1, h, w]
        return ensemble

    # ─── §5.5: L1 and L2 residual summaries ───────────────────────────────

    @staticmethod
    def compute_residual_summaries(
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 and L2 per-probe residual summaries.

        Implements §5.5 / Eqs. (6), (7).
          Eq. (6) L2: r_k^(L2)(u, v) = sqrt(Σ_c Δ_{k,c}(u, v)²)
          Eq. (7) L1: r_k^(L1)(u, v) = (1/C) Σ_c |Δ_{k,c}(u, v)|

        Connected to Proposition 1 (L1/L2 sensitivity asymmetry): with this
        precise formulation, channel-concentrated residuals are amplified by
        √C under L2 relative to L1.

        Args:
          delta: [B, K, C, h, w] per-probe residuals

        Returns:
          r_l1: [B, K, 1, h, w] L1 residuals
          r_l2: [B, K, 1, h, w] L2 residuals
        """
        # Eq. (7): channel-mean of absolute residuals
        r_l1 = delta.abs().mean(dim=2, keepdim=True)
        # Eq. (6): sqrt of channel-sum of squared residuals
        r_l2 = torch.sqrt((delta ** 2).sum(dim=2, keepdim=True) + 1e-12)
        return r_l1, r_l2

    # ─── §5.7: branch fields ──────────────────────────────────────────────

    def branch_field(self, r_per_probe: torch.Tensor) -> torch.Tensor:
        """Branch field: smooth each per-probe map, then average over K.

        Implements §5.7 / Eq. (9): R_m(u, v) = (1/K) Σ_k r̃_k^(m)(u, v),
        where r̃ is the Gaussian-smoothed per-probe map from Eq. (8).

        Args:
          r_per_probe: [B, K, 1, h, w] L1 or L2 per-probe residuals

        Returns:
          [B, 1, h, w] branch field
        """
        b, K, _, h, w = r_per_probe.shape
        # Smooth each probe independently; reshape to [B*K, 1, h, w] for conv2d
        flat = r_per_probe.reshape(b * K, 1, h, w)
        smoothed = self.smooth_residuals(flat)
        smoothed = smoothed.view(b, K, 1, h, w)
        # Average across K
        return smoothed.mean(dim=1, keepdim=False)

    # ─── §5.8: hybrid fusion ──────────────────────────────────────────────

    def fuse_branches(
        self,
        normed_l1: torch.Tensor,
        normed_l2: torch.Tensor,
    ) -> torch.Tensor:
        """Convex combination of per-branch normalized fields.

        Implements §5.8 / Eq. (11): R_hyb = w_1 · R̂_L1 + w_2 · R̂_L2.

        Both inputs are already normalized to [0, 1] per Eq. (10). The
        output may not strictly span [0, 1] (convex combination of [0, 1]
        values stays in [0, 1] only because w_1 + w_2 = 1), but for safety
        we clamp.
        """
        w1 = self.cfg.fusion_weight_l1
        w2 = 1.0 - w1
        return (w1 * normed_l1 + w2 * normed_l2).clamp(0.0, 1.0)

    # ─── §5.9: low-resolution probing and upsample ────────────────────────

    def upsample_to_image(
        self,
        field: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Bilinear upsample the suspicion field to image resolution.

        Implements §5.9 / Eq. (13): I_img = Upsample_{H × W}(R_deploy).
        """
        if field.shape[-2:] == (target_h, target_w):
            return field
        return F.interpolate(
            field, size=(target_h, target_w), mode="bilinear", align_corners=False,
        )


    @torch.no_grad()
    def compute_raw_signal(
        self,
        x0: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Return the unsmoothed, unnormalized per-probe residual summaries.

        Implements §5.5 / Eqs. (6), (7) directly: returns the raw L1, L2,
        and hybrid maps before §5.6 smoothing and §5.8 normalization are
        applied. Used by training-time diagnostics that need to compare
        on-patch vs off-patch residual magnitudes on an absolute scale.

        Args:
          x0:        [B, 3, H, W] in [0, 1]
          generator: optional torch.Generator for reproducible probe noise

        Returns:
          dict with keys "l1", "l2", "hybrid", each [B, 1, H, W] at INPUT
          resolution (upsampled bilinear from probe_res), and "residual"
          [B, 3, H, W] holding the signed per-channel residual averaged
          across the K probes.
        """
        b, c, h_in, w_in = x0.shape
        probe_res = self.cfg.probe_res

        if h_in != probe_res or w_in != probe_res:
            x0_probe = F.interpolate(
                x0, size=(probe_res, probe_res),
                mode="bilinear", align_corners=False,
            )
        else:
            x0_probe = x0

        # K-probe forward (§5.3 / Eq. 4)
        delta, _eps = self.compute_per_probe_residual(x0_probe, generator=generator)

        # Per-probe L1 / L2 summaries (§5.5 / Eqs. 6, 7), then average over K
        r_l1_per_probe, r_l2_per_probe = self.compute_residual_summaries(delta)
        l1_avg = r_l1_per_probe.mean(dim=1)  # [B, 1, probe_res, probe_res]
        l2_avg = r_l2_per_probe.mean(dim=1)

        # Hybrid: convex combination of L1 and L2 (raw, pre-normalization)
        w = self.cfg.fusion_weight_l1
        hybrid_avg = w * l1_avg + (1.0 - w) * l2_avg

        # Signed-residual diagnostic (mean over K probes), per channel
        residual_avg = delta.mean(dim=1)  # [B, C, probe_res, probe_res]

        # Upsample back to input resolution (§5.9 / Eq. 13)
        l1_full = self.upsample_to_image(l1_avg, h_in, w_in)
        l2_full = self.upsample_to_image(l2_avg, h_in, w_in)
        hybrid_full = self.upsample_to_image(hybrid_avg, h_in, w_in)
        residual_full = self.upsample_to_image(residual_avg, h_in, w_in)

        return {
            "l1": l1_full,
            "l2": l2_full,
            "hybrid": hybrid_full,
            "residual": residual_full,
        }

    # ─── End-to-end forward ───────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        x0: torch.Tensor,
        return_branches: bool = False,
        generator: torch.Generator | None = None,
    ) -> dict:
        """End-to-end suspicion field computation.

        Args:
          x0:              [B, 3, H, W] input image(s) in [0, 1]
          return_branches: if True, include per-branch normalized fields in output
          generator:       optional torch.Generator for reproducible probe noise

        Returns:
          {
            "deployed":         [B, 1, H, W] in [0, 1]   selected per cfg.deployment_mode
            "l1_field":         [B, 1, H, W] in [0, 1]   only if return_branches
            "l2_field":         [B, 1, H, W] in [0, 1]   only if return_branches
            "hybrid_field":     [B, 1, H, W] in [0, 1]   only if return_branches
            "ensemble_raw":     [B, 1, h, w]             only if return_branches (Eq. 5, pre-smooth, pre-normalize)
          }
        """
        b, c, h_in, w_in = x0.shape
        probe_res = self.cfg.probe_res

        # Downsample to probe resolution (§5.9)
        if h_in != probe_res or w_in != probe_res:
            x0_probe = F.interpolate(
                x0, size=(probe_res, probe_res),
                mode="bilinear", align_corners=False,
            )
        else:
            x0_probe = x0

        # §5.3 / Eq. (4): per-probe residuals
        delta, _eps = self.compute_per_probe_residual(x0_probe, generator=generator)

        # §5.5 / Eqs. (6), (7): L1 and L2 per-probe summaries
        r_l1_per_probe, r_l2_per_probe = self.compute_residual_summaries(delta)

        # §5.7 / Eq. (9): branch fields (smooth each probe, then average)
        l1_branch = self.branch_field(r_l1_per_probe)
        l2_branch = self.branch_field(r_l2_per_probe)

        # §5.8 / Eq. (10): per-image normalization PER BRANCH
        l1_norm = self.normalize_per_image(l1_branch)
        l2_norm = self.normalize_per_image(l2_branch)

        # §5.8 / Eq. (11): hybrid fusion
        hybrid = self.fuse_branches(l1_norm, l2_norm)

        # §5.8 / Eq. (12): deployment selection
        if self.cfg.deployment_mode == "l1":
            deployed_lo = l1_norm
        elif self.cfg.deployment_mode == "l2":
            deployed_lo = l2_norm
        else:  # "hybrid"
            deployed_lo = hybrid

        # §5.9 / Eq. (13): upsample
        deployed = self.upsample_to_image(deployed_lo, h_in, w_in)

        out: dict = {"deployed": deployed}
        if return_branches:
            out["l1_field"] = self.upsample_to_image(l1_norm, h_in, w_in)
            out["l2_field"] = self.upsample_to_image(l2_norm, h_in, w_in)
            out["hybrid_field"] = self.upsample_to_image(hybrid, h_in, w_in)
            out["ensemble_raw"] = self.aggregate_probes(delta)
        return out
