"""TinyUNet denoiser with optional bottleneck self-attention.

CARRIED FORWARD FROM v2.x UNCHANGED. The architecture is correct, the channel
arithmetic is documented, the file is one of the cleanest in the v2 codebase.
The only addition for v3 is a unit test that pins the channel arithmetic
(prevents mistake #6 from recurring).

Predicts the noise epsilon given (x_t, t) for the diffusion-prior consistency
suspicion field. Tiny by design: K=8 probes per inference call multiplies cost,
so per-forward latency must stay small.

CHANNEL ACCOUNTING
------------------
With base_channels=B (default 32), at probe_res=128:
  x1   has  64 channels at H   x W       (64 = B*2)
  x2   has 128 channels at H/2 x W/2     (128 = B*4)
  bottleneck has 128 channels at H/4 x W/4

PixelShuffle(2) divides channel count by 4 and doubles spatial dims, so:
  up_conv2 in: 128, out: 256 (B*2*4) → after PixelShuffle(2): 64, H/2 x W/2
  Then concat with x2 skip (128) → 192 → dec2 → 64

  up_conv1 in: 64, out: 128 (B*1*4) → after PixelShuffle(2): 32, H x W
  Then concat with x1 skip (64) → 96 → dec1 → 32

Self-attention is optional and only at the bottleneck (smallest spatial
resolution = cheapest), where it can correlate distant pixels — useful for
catching patches larger than the local conv receptive field.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding for diffusion timestep t."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeConditionedBlock(nn.Module):
    """ResBlock with FiLM-style additive time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class SelfAttention2D(nn.Module):
    """Multi-head self-attention over flattened 2D spatial features."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        qkv = self.qkv(h_norm)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)
        k = k.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-1, -2)) * scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(-1, -2).contiguous().view(b, c, h, w)
        return x + self.proj(out)


class TinyUNetDenoiser(nn.Module):
    """Tiny U-Net denoiser, ~1M params with attention enabled.

    Predicts epsilon given (x_t, t). Used K=8 times per DPC inference call.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        time_dim: int = 128,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.use_attention = use_attention

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = TimeConditionedBlock(base_channels, base_channels * 2, time_dim)
        self.pool1 = nn.AvgPool2d(2)
        self.down2 = TimeConditionedBlock(base_channels * 2, base_channels * 4, time_dim)
        self.pool2 = nn.AvgPool2d(2)

        self.bottleneck = TimeConditionedBlock(
            base_channels * 4, base_channels * 4, time_dim
        )
        self.attn = (
            SelfAttention2D(base_channels * 4, num_heads=4)
            if use_attention
            else nn.Identity()
        )

        # Decoder: see channel-arithmetic comment in module docstring
        self.up_conv2 = nn.Conv2d(
            base_channels * 4, base_channels * 2 * 4, kernel_size=1
        )
        self.up2 = nn.PixelShuffle(2)
        self.dec2 = TimeConditionedBlock(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
            time_dim,
        )

        self.up_conv1 = nn.Conv2d(
            base_channels * 2, base_channels * 1 * 4, kernel_size=1
        )
        self.up1 = nn.PixelShuffle(2)
        self.dec1 = TimeConditionedBlock(
            base_channels * 2 + base_channels,
            base_channels,
            time_dim,
        )

        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.in_conv(x_t)
        x1 = self.down1(x0, t_emb)
        x1p = self.pool1(x1)
        x2 = self.down2(x1p, t_emb)
        x2p = self.pool2(x2)

        h = self.bottleneck(x2p, t_emb)
        h = self.attn(h)

        u2 = self.up2(self.up_conv2(h))
        u2 = self.dec2(torch.cat([u2, x2], dim=1), t_emb)

        u1 = self.up1(self.up_conv1(u2))
        u1 = self.dec1(torch.cat([u1, x1], dim=1), t_emb)

        return self.out_conv(u1)
