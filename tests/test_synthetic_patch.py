"""Test the programmatic synthetic patch generator.

Pins the structural fixes for mistakes #17 and #18:
  - mask is pixel-perfect by construction (binary {0, 1})
  - render is reproducible from rng seed
  - area distribution is configurable
"""

import math
import os
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _make_dummy_color_dist(path: Path, n_bins: int = 16, seed: int = 42):
    """Create a synthetic ColorDistribution payload for tests."""
    rng = np.random.RandomState(seed)
    # Concentrated around mid-saturation, mid-value, hue mode at red
    hist = rng.dirichlet(np.ones(n_bins ** 3) * 0.5).reshape(n_bins, n_bins, n_bins)
    hist = hist.astype(np.float32)
    bin_edges = np.linspace(0, 1, n_bins + 1, dtype=np.float32)
    payload = {
        "version": "3.0.0",
        "n_bins": n_bins,
        "h_bins": torch.from_numpy(bin_edges),
        "s_bins": torch.from_numpy(bin_edges),
        "v_bins": torch.from_numpy(bin_edges),
        "joint_hist": torch.from_numpy(hist),
        "marginal_h": torch.from_numpy(hist.sum(axis=(1, 2))),
        "marginal_s": torch.from_numpy(hist.sum(axis=(0, 2))),
        "marginal_v": torch.from_numpy(hist.sum(axis=(0, 1))),
    }
    torch.save(payload, path)


@pytest.fixture(scope="module")
def color_dist_path():
    tmpdir = tempfile.mkdtemp(prefix="dpc_test_color_")
    p = Path(tmpdir) / "color_distribution.pt"
    _make_dummy_color_dist(p)
    yield p
    # tmpdir cleanup
    try:
        os.remove(p); os.rmdir(tmpdir)
    except Exception:
        pass


def test_color_distribution_loads(color_dist_path):
    from dpc.synthetic_patch import ColorDistribution
    cd = ColorDistribution(color_dist_path)
    assert cd.n_bins == 16
    rng = np.random.RandomState(42)
    h, s, v = cd.sample(rng)
    assert 0 <= h <= 1
    assert 0 <= s <= 1
    assert 0 <= v <= 1


def test_color_distribution_missing_file():
    from dpc.synthetic_patch import ColorDistribution
    with pytest.raises(FileNotFoundError):
        ColorDistribution("/nonexistent/color.pt")


def test_hsv_to_rgb_basic():
    from dpc.synthetic_patch import hsv_to_rgb
    # Pure red: h=0, s=1, v=1 → (1, 0, 0)
    rgb = hsv_to_rgb(0.0, 1.0, 1.0)
    assert np.allclose(rgb, [1.0, 0.0, 0.0], atol=1e-6)
    # Pure green: h=1/3
    rgb = hsv_to_rgb(1/3, 1.0, 1.0)
    assert np.allclose(rgb, [0.0, 1.0, 0.0], atol=1e-6)
    # Pure blue: h=2/3
    rgb = hsv_to_rgb(2/3, 1.0, 1.0)
    assert np.allclose(rgb, [0.0, 0.0, 1.0], atol=1e-6)


def test_synthetic_generator_constructs(color_dist_path):
    from dpc.synthetic_patch import SyntheticPatchGenerator
    gen = SyntheticPatchGenerator(color_dist_path, seed=42, image_size=128)
    assert gen.image_size == 128
    assert len(gen.SHAPES) == 4
    assert len(gen.TEXTURES) == 6


def test_synthetic_generator_invalid_weights(color_dist_path):
    from dpc.synthetic_patch import SyntheticPatchGenerator
    with pytest.raises(ValueError):
        SyntheticPatchGenerator(color_dist_path, shape_weights=(0.5, 0.5))  # only 2
    with pytest.raises(ValueError):
        SyntheticPatchGenerator(color_dist_path, shape_weights=(0.4, 0.4, 0.4, 0.4))  # sum != 1


@pytest.mark.parametrize("shape", ["rect", "ellipse", "polygon", "blob"])
def test_render_produces_pixel_perfect_mask(color_dist_path, shape):
    """Mistake #18 prevented: every shape produces a {0, 1} binary mask."""
    from dpc.synthetic_patch import SyntheticPatchGenerator, PatchSpec

    gen = SyntheticPatchGenerator(color_dist_path, seed=42, image_size=128)
    scene = torch.rand(3, 128, 128)
    spec = PatchSpec(
        shape=shape, size_frac=0.05, rotation_deg=30.0, aspect_ratio=1.2,
        cx_frac=0.5, cy_frac=0.5,
        texture="uniform", color_seed=(0.1, 0.7, 0.7),
        contrast=0.3, blend_mode="paste", alpha=1.0,
        n_vertices=6, blob_seed=1234,
    )
    rng = np.random.RandomState(42)
    patched, mask = gen.render(scene, spec, rng)
    assert patched.shape == (3, 128, 128)
    assert mask.shape == (1, 128, 128)
    unique = set(np.unique(mask.numpy()).tolist())
    assert unique.issubset({0.0, 1.0}), \
        f"mask not binary for shape={shape}, found values: {unique}"


def test_render_reproducible_with_same_rng(color_dist_path):
    from dpc.synthetic_patch import SyntheticPatchGenerator
    gen = SyntheticPatchGenerator(color_dist_path, seed=42, image_size=128)
    scene = torch.rand(3, 128, 128)

    rng1 = np.random.RandomState(123)
    p1, m1, _ = gen.render_random(scene, rng1)
    rng2 = np.random.RandomState(123)
    p2, m2, _ = gen.render_random(scene, rng2)

    assert torch.allclose(p1, p2)
    assert torch.equal(m1, m2)


def test_render_different_rng_gives_different_patch(color_dist_path):
    from dpc.synthetic_patch import SyntheticPatchGenerator
    gen = SyntheticPatchGenerator(color_dist_path, seed=42, image_size=128)
    scene = torch.rand(3, 128, 128)

    rng1 = np.random.RandomState(111)
    p1, m1, _ = gen.render_random(scene, rng1)
    rng2 = np.random.RandomState(222)
    p2, m2, _ = gen.render_random(scene, rng2)

    assert not torch.equal(m1, m2)


def test_validate_mask_returns_diagnostic(color_dist_path):
    from dpc.synthetic_patch import SyntheticPatchGenerator
    gen = SyntheticPatchGenerator(color_dist_path, seed=42, image_size=128)
    rng = np.random.RandomState(42)
    scene = torch.rand(3, 128, 128)
    _, mask, _ = gen.render_random(scene, rng)
    diag = gen.validate_mask(mask)
    assert diag["is_binary"] is True
    assert diag["n_pixels"] > 0
    assert 0.0 <= diag["bbox_tightness"] <= 1.0


@pytest.mark.parametrize("texture", [
    "uniform", "gradient_linear", "gradient_radial",
    "perlin", "voronoi", "stripes",
])
def test_all_textures_render(color_dist_path, texture):
    from dpc.synthetic_patch import SyntheticPatchGenerator, PatchSpec
    gen = SyntheticPatchGenerator(color_dist_path, seed=42, image_size=64)
    scene = torch.rand(3, 64, 64)
    spec = PatchSpec(
        shape="rect", size_frac=0.10, rotation_deg=0.0, aspect_ratio=1.0,
        cx_frac=0.5, cy_frac=0.5,
        texture=texture, color_seed=(0.3, 0.6, 0.7),
        contrast=0.4, blend_mode="paste", alpha=1.0,
    )
    rng = np.random.RandomState(7)
    patched, mask = gen.render(scene, spec, rng)
    assert patched.shape == scene.shape
    assert (mask > 0).any()


@pytest.mark.parametrize("blend_mode", ["paste", "luminance_match", "alpha"])
def test_all_blend_modes_render(color_dist_path, blend_mode):
    from dpc.synthetic_patch import SyntheticPatchGenerator, PatchSpec
    gen = SyntheticPatchGenerator(color_dist_path, seed=42, image_size=64)
    scene = torch.rand(3, 64, 64)
    spec = PatchSpec(
        shape="ellipse", size_frac=0.08, rotation_deg=15.0, aspect_ratio=1.5,
        cx_frac=0.5, cy_frac=0.5,
        texture="uniform", color_seed=(0.5, 0.7, 0.6),
        contrast=0.3, blend_mode=blend_mode, alpha=0.8,
    )
    rng = np.random.RandomState(7)
    patched, mask = gen.render(scene, spec, rng)
    assert (mask > 0).any()
    # patched must be in [0, 1]
    assert patched.min() >= 0 and patched.max() <= 1


def test_size_frac_respected(color_dist_path):
    """Patch area should be approximately size_frac of image area."""
    from dpc.synthetic_patch import SyntheticPatchGenerator, PatchSpec
    gen = SyntheticPatchGenerator(color_dist_path, seed=42, image_size=128)
    scene = torch.rand(3, 128, 128)
    spec = PatchSpec(
        shape="rect", size_frac=0.10, rotation_deg=0.0, aspect_ratio=1.0,
        cx_frac=0.5, cy_frac=0.5,
        texture="uniform", color_seed=(0.0, 0.0, 0.5),
        contrast=0.0, blend_mode="paste", alpha=1.0,
    )
    rng = np.random.RandomState(7)
    _, mask = gen.render(scene, spec, rng)
    actual_frac = mask.sum().item() / (128 * 128)
    # rect at 10% area → ~12,800 px; allow ±20% tolerance
    assert 0.07 < actual_frac < 0.13, f"size_frac=0.10 produced {actual_frac:.3f}"
