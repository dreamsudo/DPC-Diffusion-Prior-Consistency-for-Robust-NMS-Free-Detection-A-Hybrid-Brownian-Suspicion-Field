"""Loss functions for Stage 1 (denoiser) training.

CARRIED FORWARD FROM v2.x.

In Phase 1 only the diffusion MSE is active. focal_bce_with_logits and
ssim_loss are present but unused (they're activated in Phase 2's
field-supervision training). Keeping them in this bootstrap so we don't
have to ship a different losses.py for Phase 2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_mse_loss(
    eps_pred: torch.Tensor,
    eps_true: torch.Tensor,
) -> torch.Tensor:
    """Standard DDPM noise-prediction MSE."""
    return ((eps_pred - eps_true) ** 2).mean()


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal BCE: down-weights easy examples.

    Used in Phase 2 for field supervision. Not active in Phase 1.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t).pow(gamma)
    loss = focal_weight * bce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Approximate SSIM loss = 1 - SSIM. Used in Phase 2."""
    if pred.shape != target.shape:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    sigma = window_size / 6.0
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(pred.shape[1], 1, -1, -1)
    pad = window_size // 2

    mu_p = F.conv2d(pred, kernel, padding=pad, groups=pred.shape[1])
    mu_t = F.conv2d(target, kernel, padding=pad, groups=pred.shape[1])

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=pred.shape[1]) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, kernel, padding=pad, groups=pred.shape[1]) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, kernel, padding=pad, groups=pred.shape[1]) - mu_pt

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
    )
    return 1.0 - ssim_map.mean()
