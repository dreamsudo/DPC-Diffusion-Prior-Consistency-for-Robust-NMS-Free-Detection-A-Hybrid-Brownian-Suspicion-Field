"""Programmatic synthetic adversarial patch generator.

REPLACES the v2.x crop-and-paste approach (mistake #17). Real APRICOT patches
are NOT used to generate training data. Instead we synthesize patches with
shape, texture, color, blending, and rotation drawn from configurable
distributions, then composite them onto natural images.

Why this is the right approach:
  - Mask is pixel-perfect by construction (mistake #18 cannot happen)
  - Distribution is independent of the test set (mistake #20 prevented)
  - Generator parameters are hyperparameters that we control and report
  - Field supervision gradient is well-conditioned because the mask is exact

Color distribution is fitted ONCE from REAL APRICOT (in HSV space) by
tools/fit_color_distribution.py. The generator samples patch colors from
this distribution so synthetic patches are color-statistically similar to
real APRICOT — they match the COLOR profile without containing any actual
APRICOT pixels.

Public API:
  PatchSpec                      — dataclass describing a single patch
  SyntheticPatchGenerator        — render(scene, spec) and render_random(scene)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Public dataclass
# ============================================================================

@dataclass
class PatchSpec:
    """Description of a single synthetic adversarial patch."""

    shape: str          # "rect" | "ellipse" | "polygon" | "blob"
    size_frac: float    # patch area as fraction of image area, [0.005, 0.20]
    rotation_deg: float # [0, 360)
    aspect_ratio: float # h/w, in [0.4, 2.5]
    cx_frac: float      # patch center x, in [0, 1] of image width
    cy_frac: float      # patch center y, in [0, 1] of image height

    texture: str        # "uniform" | "gradient_linear" | "gradient_radial"
                        # | "perlin" | "voronoi" | "stripes"
    color_seed: tuple   # (h, s, v) anchor in [0, 1] for the texture
    contrast: float     # texture contrast/strength, [0, 1]

    blend_mode: str     # "paste" | "luminance_match" | "alpha"
    alpha: float        # for blend_mode="alpha", in [0.5, 1.0]

    # For polygon/blob shapes
    n_vertices: int = 6
    blob_seed: int = 0

    # For stripes
    stripe_period: float = 8.0
    stripe_angle_deg: float = 0.0


# ============================================================================
# Color distribution loading
# ============================================================================

class ColorDistribution:
    """Load + sample from the HSV joint distribution fitted from real APRICOT."""

    def __init__(self, path: Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"color distribution not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.n_bins = data["n_bins"]
        self.h_bins = data["h_bins"].numpy()
        self.s_bins = data["s_bins"].numpy()
        self.v_bins = data["v_bins"].numpy()
        # joint_hist is [Nb, Nb, Nb] normalized
        self.joint_hist = data["joint_hist"].numpy()
        # Flatten + cumsum for fast sampling
        flat = self.joint_hist.reshape(-1)
        s = flat.sum()
        if s > 0:
            self.cum = (flat / s).cumsum()
        else:
            self.cum = None  # degenerate — fall back to uniform

    def sample(self, rng: np.random.RandomState) -> tuple:
        """Sample a single (h, s, v) triple in [0, 1]^3."""
        if self.cum is None:
            return (rng.uniform(0, 1), rng.uniform(0.3, 1.0), rng.uniform(0.2, 0.9))
        u = rng.uniform()
        flat_idx = int(np.searchsorted(self.cum, u, side="right"))
        flat_idx = min(flat_idx, self.cum.size - 1)
        nb = self.n_bins
        # unflatten — joint_hist[hi, si, vi]
        hi = flat_idx // (nb * nb)
        rem = flat_idx % (nb * nb)
        si = rem // nb
        vi = rem % nb
        # Center of bin (assumes bins span [0, 1])
        bin_w = 1.0 / nb
        h = (hi + 0.5) * bin_w
        s = (si + 0.5) * bin_w
        v = (vi + 0.5) * bin_w
        # Add a little jitter
        h = (h + rng.uniform(-bin_w / 2, bin_w / 2)) % 1.0
        s = float(np.clip(s + rng.uniform(-bin_w / 2, bin_w / 2), 0.0, 1.0))
        v = float(np.clip(v + rng.uniform(-bin_w / 2, bin_w / 2), 0.0, 1.0))
        return (h, s, v)


# ============================================================================
# HSV → RGB
# ============================================================================

def hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    """Single (h, s, v) → (r, g, b), all in [0, 1]. Returns shape (3,)."""
    h = (h % 1.0) * 6.0
    i = int(math.floor(h))
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    rgb_table = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]
    r, g, b = rgb_table[i % 6]
    return np.array([r, g, b], dtype=np.float32)


# ============================================================================
# Mask rasterization
# ============================================================================

def _rasterize_rect(H: int, W: int, cx: int, cy: int,
                    size_frac: float, aspect: float, angle_deg: float
                    ) -> np.ndarray:
    """Render an axis-aligned-then-rotated rectangle into an [H, W] uint8 mask."""
    area = size_frac * H * W
    # solve for w, h with h/w = aspect, w*h = area
    w = math.sqrt(area / max(aspect, 1e-3))
    h = w * aspect
    half_w = w / 2
    half_h = h / 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    theta = math.radians(angle_deg)
    cos_t = math.cos(-theta); sin_t = math.sin(-theta)
    rx = cos_t * dx - sin_t * dy
    ry = sin_t * dx + cos_t * dy
    mask = ((np.abs(rx) <= half_w) & (np.abs(ry) <= half_h)).astype(np.uint8)
    return mask


def _rasterize_ellipse(H: int, W: int, cx: int, cy: int,
                       size_frac: float, aspect: float, angle_deg: float
                       ) -> np.ndarray:
    area = size_frac * H * W
    rx = math.sqrt(area / (math.pi * max(aspect, 1e-3)))
    ry = rx * aspect
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx; dy = yy - cy
    theta = math.radians(angle_deg)
    cos_t = math.cos(-theta); sin_t = math.sin(-theta)
    ex = cos_t * dx - sin_t * dy
    ey = sin_t * dx + cos_t * dy
    mask = (((ex / max(rx, 1e-3)) ** 2 + (ey / max(ry, 1e-3)) ** 2) <= 1.0).astype(np.uint8)
    return mask


def _rasterize_polygon(H: int, W: int, cx: int, cy: int,
                       size_frac: float, aspect: float,
                       angle_deg: float, n_vertices: int) -> np.ndarray:
    area = size_frac * H * W
    # Approximate as ellipse-bounded polygon
    radius = math.sqrt(area / math.pi)
    rx = radius / math.sqrt(max(aspect, 1e-3))
    ry = radius * math.sqrt(aspect)
    # Vertex angles
    base_angles = np.linspace(0, 2 * math.pi, n_vertices, endpoint=False)
    base_angles += math.radians(angle_deg)
    vx = cx + rx * np.cos(base_angles)
    vy = cy + ry * np.sin(base_angles)
    # Rasterize via even-odd rule
    mask = np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    px = xx.astype(np.float32); py = yy.astype(np.float32)
    inside = np.zeros((H, W), dtype=bool)
    j = n_vertices - 1
    for i in range(n_vertices):
        x_i, y_i = vx[i], vy[i]
        x_j, y_j = vx[j], vy[j]
        cond1 = (y_i > py) != (y_j > py)
        denom = (y_j - y_i) if (y_j - y_i) != 0 else 1e-6
        cross = (x_j - x_i) * (py - y_i) / denom + x_i
        cond2 = px < cross
        toggle = cond1 & cond2
        inside ^= toggle
        j = i
    mask[inside] = 1
    return mask


def _rasterize_blob(H: int, W: int, cx: int, cy: int,
                    size_frac: float, aspect: float,
                    angle_deg: float, blob_seed: int) -> np.ndarray:
    """Random blob: ellipse with radial perturbation."""
    rs = np.random.RandomState(blob_seed)
    area = size_frac * H * W
    base_r = math.sqrt(area / math.pi)
    rx = base_r / math.sqrt(max(aspect, 1e-3))
    ry = base_r * math.sqrt(aspect)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xx - cx; dy = yy - cy
    theta = math.radians(angle_deg)
    cos_t = math.cos(-theta); sin_t = math.sin(-theta)
    ex = cos_t * dx - sin_t * dy
    ey = sin_t * dx + cos_t * dy

    # Compute angle and normalized radius for each pixel
    angle = np.arctan2(ey, ex)
    norm_r = np.sqrt((ex / max(rx, 1e-3)) ** 2 + (ey / max(ry, 1e-3)) ** 2)

    # Radial bumps: sum of a few sinusoids in angle
    n_bumps = rs.randint(2, 6)
    perturb = np.zeros_like(angle)
    for _ in range(n_bumps):
        k = rs.randint(2, 6)
        phi = rs.uniform(0, 2 * math.pi)
        amp = rs.uniform(0.05, 0.20)
        perturb += amp * np.sin(k * angle + phi)
    threshold = 1.0 + perturb
    return (norm_r <= threshold).astype(np.uint8)


# ============================================================================
# Texture rendering
# ============================================================================

def _texture_uniform(mask_box, base_rgb, contrast, rng):
    H_b, W_b = mask_box.shape
    img = np.broadcast_to(base_rgb.reshape(3, 1, 1), (3, H_b, W_b)).copy()
    return img


def _texture_gradient_linear(mask_box, base_rgb, contrast, rng, angle_deg=None):
    H_b, W_b = mask_box.shape
    if angle_deg is None:
        angle_deg = rng.uniform(0, 180)
    yy, xx = np.mgrid[0:H_b, 0:W_b].astype(np.float32)
    cx_b, cy_b = W_b / 2, H_b / 2
    theta = math.radians(angle_deg)
    proj = (xx - cx_b) * math.cos(theta) + (yy - cy_b) * math.sin(theta)
    pmin, pmax = proj.min(), proj.max()
    if pmax - pmin < 1e-6:
        norm = np.zeros_like(proj)
    else:
        norm = (proj - pmin) / (pmax - pmin)  # in [0, 1]
    # Offset around base by ±contrast
    factor = 1.0 + (norm - 0.5) * 2 * contrast
    img = base_rgb.reshape(3, 1, 1) * factor[None, ...]
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _texture_gradient_radial(mask_box, base_rgb, contrast, rng):
    H_b, W_b = mask_box.shape
    yy, xx = np.mgrid[0:H_b, 0:W_b].astype(np.float32)
    cx_b, cy_b = W_b / 2, H_b / 2
    r = np.sqrt((xx - cx_b) ** 2 + (yy - cy_b) ** 2)
    rmax = r.max()
    if rmax < 1e-6:
        norm = np.zeros_like(r)
    else:
        norm = r / rmax
    factor = 1.0 + (norm - 0.5) * 2 * contrast
    img = base_rgb.reshape(3, 1, 1) * factor[None, ...]
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _texture_stripes(mask_box, base_rgb, contrast, rng, period, angle_deg):
    H_b, W_b = mask_box.shape
    yy, xx = np.mgrid[0:H_b, 0:W_b].astype(np.float32)
    theta = math.radians(angle_deg)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    stripe = 0.5 + 0.5 * np.sin(2 * math.pi * proj / max(period, 1.0))
    factor = 1.0 + (stripe - 0.5) * 2 * contrast
    img = base_rgb.reshape(3, 1, 1) * factor[None, ...]
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _texture_voronoi(mask_box, base_rgb, contrast, rng):
    H_b, W_b = mask_box.shape
    n_seeds = rng.randint(4, 16)
    sx = rng.uniform(0, W_b, size=n_seeds)
    sy = rng.uniform(0, H_b, size=n_seeds)
    seed_color_jitter = rng.uniform(-contrast, contrast, size=(n_seeds, 3))
    seed_colors = np.clip(
        base_rgb.reshape(1, 3) + seed_color_jitter, 0.0, 1.0,
    )
    yy, xx = np.mgrid[0:H_b, 0:W_b].astype(np.float32)
    # Distance to each seed
    d2 = ((xx[..., None] - sx[None, None, :]) ** 2
          + (yy[..., None] - sy[None, None, :]) ** 2)
    nearest = np.argmin(d2, axis=-1)
    img = seed_colors[nearest]  # [H, W, 3]
    return img.transpose(2, 0, 1).astype(np.float32)


def _texture_perlin(mask_box, base_rgb, contrast, rng, octaves=4):
    """Cheap pseudo-Perlin: sum of low-res random noises bilinearly upsampled."""
    H_b, W_b = mask_box.shape
    accum = np.zeros((H_b, W_b), dtype=np.float32)
    weight = 1.0
    for o in range(octaves):
        scale = 2 ** o
        gx = max(2, W_b // (16 // scale))
        gy = max(2, H_b // (16 // scale))
        low = rng.uniform(-1, 1, size=(gy, gx)).astype(np.float32)
        # Bilinear upsample with torch
        t = torch.from_numpy(low).unsqueeze(0).unsqueeze(0)
        up = F.interpolate(t, size=(H_b, W_b), mode="bilinear", align_corners=False)
        accum += weight * up.squeeze(0).squeeze(0).numpy()
        weight *= 0.5
    # Normalize to [-1, 1]
    a = accum
    span = max(1e-6, a.max() - a.min())
    norm = 2 * (a - a.min()) / span - 1.0
    factor = 1.0 + norm * contrast
    img = base_rgb.reshape(3, 1, 1) * factor[None, ...]
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ============================================================================
# Generator
# ============================================================================

class SyntheticPatchGenerator:
    """Generate (image_with_patch, mask, spec) triples on demand."""

    SHAPES = ["rect", "ellipse", "polygon", "blob"]
    TEXTURES = [
        "uniform", "gradient_linear", "gradient_radial",
        "perlin", "voronoi", "stripes",
    ]

    def __init__(
        self,
        color_distribution_path: Path,
        seed: int = 42,
        image_size: int = 128,
        size_frac_range: tuple[float, float] = (0.01, 0.18),
        aspect_ratio_range: tuple[float, float] = (0.4, 2.5),
        contrast_range: tuple[float, float] = (0.15, 0.55),
        # Shape mixing weights — sum to 1.0
        shape_weights: tuple[float, ...] = (0.30, 0.25, 0.20, 0.25),
        texture_weights: tuple[float, ...] = (0.10, 0.20, 0.15, 0.20, 0.20, 0.15),
        # Border mode — "shrink" keeps patch fully on canvas; "clip" allows truncation
        border_mode: str = "shrink",
    ):
        self.color_dist = ColorDistribution(color_distribution_path)
        self.seed = seed
        self.image_size = image_size
        self.size_frac_range = size_frac_range
        self.aspect_ratio_range = aspect_ratio_range
        self.contrast_range = contrast_range
        if len(shape_weights) != 4:
            raise ValueError(f"shape_weights must have length 4, got {len(shape_weights)}")
        if abs(sum(shape_weights) - 1.0) > 1e-3:
            raise ValueError(f"shape_weights must sum to 1.0, got {sum(shape_weights)}")
        if abs(sum(texture_weights) - 1.0) > 1e-3:
            raise ValueError(f"texture_weights must sum to 1.0, got {sum(texture_weights)}")
        self.shape_weights = list(shape_weights)
        self.texture_weights = list(texture_weights)
        self.border_mode = border_mode

    def sample_spec(self, rng: np.random.RandomState) -> PatchSpec:
        shape = rng.choice(self.SHAPES, p=self.shape_weights)
        texture = rng.choice(self.TEXTURES, p=self.texture_weights)
        size_frac = float(rng.uniform(*self.size_frac_range))
        aspect = float(rng.uniform(*self.aspect_ratio_range))
        rotation = float(rng.uniform(0, 360))
        contrast = float(rng.uniform(*self.contrast_range))
        h, s, v = self.color_dist.sample(rng)

        # Sample center; shrink-mode keeps patch fully inside canvas
        if self.border_mode == "shrink":
            # half-extents in fraction of image
            half = math.sqrt(size_frac) / 2 * 1.2
            half = min(half, 0.45)
            cx_frac = float(rng.uniform(half, 1 - half))
            cy_frac = float(rng.uniform(half, 1 - half))
        else:
            cx_frac = float(rng.uniform(0.1, 0.9))
            cy_frac = float(rng.uniform(0.1, 0.9))

        blend_mode = rng.choice(["paste", "luminance_match", "alpha"], p=[0.5, 0.3, 0.2])
        alpha = float(rng.uniform(0.65, 1.0)) if blend_mode == "alpha" else 1.0

        return PatchSpec(
            shape=shape,
            size_frac=size_frac,
            rotation_deg=rotation,
            aspect_ratio=aspect,
            cx_frac=cx_frac,
            cy_frac=cy_frac,
            texture=texture,
            color_seed=(h, s, v),
            contrast=contrast,
            blend_mode=blend_mode,
            alpha=alpha,
            n_vertices=int(rng.randint(5, 9)),
            blob_seed=int(rng.randint(0, 1_000_000)),
            stripe_period=float(rng.uniform(4.0, 16.0)),
            stripe_angle_deg=float(rng.uniform(0, 180)),
        )

    def render(
        self,
        scene: torch.Tensor,
        spec: PatchSpec,
        rng: Optional[np.random.RandomState] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Composite a patch with `spec` onto `scene`.

        scene: [3, H, W] in [0, 1]
        Returns:
          patched: [3, H, W] in [0, 1]
          mask:    [1, H, W] in {0, 1} pixel-perfect
        """
        if scene.dim() != 3 or scene.shape[0] != 3:
            raise ValueError(f"scene shape must be [3, H, W], got {scene.shape}")
        H, W = scene.shape[-2:]
        if rng is None:
            rng = np.random.RandomState(spec.blob_seed)

        cx = int(round(spec.cx_frac * W))
        cy = int(round(spec.cy_frac * H))

        # Shape mask
        if spec.shape == "rect":
            mask_np = _rasterize_rect(H, W, cx, cy, spec.size_frac,
                                      spec.aspect_ratio, spec.rotation_deg)
        elif spec.shape == "ellipse":
            mask_np = _rasterize_ellipse(H, W, cx, cy, spec.size_frac,
                                         spec.aspect_ratio, spec.rotation_deg)
        elif spec.shape == "polygon":
            mask_np = _rasterize_polygon(H, W, cx, cy, spec.size_frac,
                                         spec.aspect_ratio, spec.rotation_deg,
                                         spec.n_vertices)
        elif spec.shape == "blob":
            mask_np = _rasterize_blob(H, W, cx, cy, spec.size_frac,
                                      spec.aspect_ratio, spec.rotation_deg,
                                      spec.blob_seed)
        else:
            raise ValueError(f"unknown shape: {spec.shape}")

        if mask_np.sum() < 4:
            # Degenerate patch — return scene unchanged with empty mask
            return scene.clone(), torch.zeros((1, H, W), dtype=torch.float32)

        # Texture is rendered for the whole image then masked.
        base_rgb = hsv_to_rgb(*spec.color_seed)

        if spec.texture == "uniform":
            tex = _texture_uniform(mask_np, base_rgb, spec.contrast, rng)
        elif spec.texture == "gradient_linear":
            tex = _texture_gradient_linear(mask_np, base_rgb, spec.contrast, rng,
                                           angle_deg=spec.rotation_deg)
        elif spec.texture == "gradient_radial":
            tex = _texture_gradient_radial(mask_np, base_rgb, spec.contrast, rng)
        elif spec.texture == "perlin":
            tex = _texture_perlin(mask_np, base_rgb, spec.contrast, rng)
        elif spec.texture == "voronoi":
            tex = _texture_voronoi(mask_np, base_rgb, spec.contrast, rng)
        elif spec.texture == "stripes":
            tex = _texture_stripes(mask_np, base_rgb, spec.contrast, rng,
                                   period=spec.stripe_period,
                                   angle_deg=spec.stripe_angle_deg)
        else:
            raise ValueError(f"unknown texture: {spec.texture}")

        # Compose
        scene_np = scene.detach().cpu().numpy()  # [3, H, W]
        mask_3 = mask_np[None, ...].astype(np.float32)  # [1, H, W]

        if spec.blend_mode == "paste":
            patched = scene_np * (1 - mask_3) + tex * mask_3
        elif spec.blend_mode == "luminance_match":
            # Match luminance of patch region to the local scene luminance
            scene_y = (0.299 * scene_np[0] + 0.587 * scene_np[1] + 0.114 * scene_np[2])
            tex_y = (0.299 * tex[0] + 0.587 * tex[1] + 0.114 * tex[2])
            in_patch = mask_np > 0.5
            if in_patch.any():
                scene_lum_in = scene_y[in_patch].mean()
                tex_lum = tex_y[in_patch].mean()
                if tex_lum > 1e-3:
                    scale = scene_lum_in / tex_lum
                    tex_matched = np.clip(tex * scale, 0.0, 1.0)
                else:
                    tex_matched = tex
            else:
                tex_matched = tex
            patched = scene_np * (1 - mask_3) + tex_matched * mask_3
        elif spec.blend_mode == "alpha":
            a = spec.alpha
            patched = scene_np * (1 - a * mask_3) + tex * (a * mask_3)
        else:
            raise ValueError(f"unknown blend_mode: {spec.blend_mode}")

        patched_t = torch.from_numpy(patched.clip(0.0, 1.0)).float()
        mask_t = torch.from_numpy(mask_3).float()
        return patched_t, mask_t

    def render_random(
        self,
        scene: torch.Tensor,
        rng: np.random.RandomState,
    ) -> tuple[torch.Tensor, torch.Tensor, PatchSpec]:
        spec = self.sample_spec(rng)
        patched, mask = self.render(scene, spec, rng)
        return patched, mask, spec

    def validate_mask(self, mask: torch.Tensor) -> dict:
        """Diagnostic info about a generated mask."""
        if mask.dim() == 3:
            m = mask[0]
        else:
            m = mask
        m_np = m.detach().cpu().numpy()
        unique = np.unique(m_np)
        is_binary = bool(set(unique.tolist()).issubset({0.0, 1.0}))
        n_pix = int(m_np.sum())
        H, W = m_np.shape
        # Tight bbox
        ys, xs = np.where(m_np > 0.5)
        if len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bb_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            tight = n_pix / bb_area if bb_area > 0 else 0.0
        else:
            x1 = y1 = x2 = y2 = -1
            tight = 0.0
        return {
            "is_binary": is_binary,
            "n_pixels": n_pix,
            "area_frac": n_pix / max(1, H * W),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_tightness": tight,
            "unique_values_first5": unique[:5].tolist(),
        }
