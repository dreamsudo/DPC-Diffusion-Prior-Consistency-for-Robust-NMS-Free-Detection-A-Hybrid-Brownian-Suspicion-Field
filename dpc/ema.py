"""EMA (Exponential Moving Average) wrapper for denoiser weights.

CARRIED FORWARD FROM v2.x UNCHANGED.

Standard practice for diffusion training. Maintains a shadow copy of the
weights that's a smoothed average. The EMA weights are used for inference,
not the raw training weights.

Typical decay: 0.999 (means EMA "remembers" ~1000 steps of history).
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow weights from current model: theta_ema = decay * theta_ema + (1-decay) * theta."""
        msd = model.state_dict()
        for name, param in self.shadow.state_dict().items():
            if param.dtype.is_floating_point:
                param.mul_(self.decay).add_(msd[name].detach(), alpha=1.0 - self.decay)
            else:
                param.copy_(msd[name])

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state)

    def to(self, device):
        self.shadow.to(device)
        return self
