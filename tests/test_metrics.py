"""Test metrics module."""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def test_residual_ratio_basic():
    from dpc.metrics import residual_ratio_per_image

    # Patch region has high residual, outside has low
    res = torch.zeros(64, 64)
    res[:32, :32] = 4.0
    res[32:, 32:] = 1.0

    mask = torch.zeros(64, 64)
    mask[:32, :32] = 1.0

    stats = residual_ratio_per_image(res, mask)
    assert not stats["is_degenerate"]
    assert stats["inside_mean"] == pytest.approx(4.0)
    # outside has 2048 zeros + 1024 ones in 3072 cells -> mean 1024/3072
    assert stats["outside_mean"] == pytest.approx(1024 / 3072)
    assert stats["ratio"] == pytest.approx(4.0 / (1024 / 3072))




def test_residual_ratio_degenerate_no_inside():
    from dpc.metrics import residual_ratio_per_image

    res = torch.rand(8, 8)
    mask = torch.zeros(8, 8)  # no patch
    stats = residual_ratio_per_image(res, mask)
    assert stats["is_degenerate"]


def test_aggregate_distribution_excludes_degenerate():
    from dpc.metrics import aggregate_residual_distribution

    per_image = [
        {"ratio": 1.5, "is_degenerate": False},
        {"ratio": 2.0, "is_degenerate": False},
        {"ratio": None, "is_degenerate": True},
        {"ratio": 5.5, "is_degenerate": False},
    ]
    agg = aggregate_residual_distribution(per_image)
    assert agg["n_total"] == 4
    assert agg["n_valid"] == 3
    assert agg["n_degenerate"] == 1
    assert agg["median_ratio"] == 2.0
    assert agg["bins"]["greater_than_5x"] == 1


def test_bootstrap_ci_returns_interval():
    from dpc.metrics import bootstrap_ci

    rs = np.random.RandomState(42)
    samples = rs.normal(loc=5.0, scale=1.0, size=200)
    lo, hi = bootstrap_ci(samples, np.mean, n_boot=500, seed=42)
    assert lo < 5.0 < hi


def test_bootstrap_ci_empty():
    from dpc.metrics import bootstrap_ci
    lo, hi = bootstrap_ci(np.array([]), np.mean, n_boot=10, seed=42)
    assert np.isnan(lo) and np.isnan(hi)


