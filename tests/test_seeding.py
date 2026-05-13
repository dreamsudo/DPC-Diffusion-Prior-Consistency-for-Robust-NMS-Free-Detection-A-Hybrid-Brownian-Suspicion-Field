"""Test seeding determinism."""

import pytest

torch = pytest.importorskip("torch")


def test_set_global_seed_deterministic():
    from dpc.seeding import set_global_seed

    set_global_seed(42)
    a = torch.randn(10)
    set_global_seed(42)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_make_generator_reproducible():
    from dpc.seeding import make_generator

    g1 = make_generator(42)
    g2 = make_generator(42)
    a = torch.randn(8, generator=g1)
    b = torch.randn(8, generator=g2)
    assert torch.equal(a, b)


def test_make_generator_different_seeds_differ():
    from dpc.seeding import make_generator

    g1 = make_generator(42)
    g2 = make_generator(43)
    a = torch.randn(8, generator=g1)
    b = torch.randn(8, generator=g2)
    assert not torch.equal(a, b)


def test_deterministic_split_reproducible():
    from dpc.seeding import deterministic_split

    train_a, val_a = deterministic_split(100, 0.9, seed=42)
    train_b, val_b = deterministic_split(100, 0.9, seed=42)
    assert train_a == train_b
    assert val_a == val_b


def test_deterministic_split_disjoint():
    from dpc.seeding import deterministic_split

    train, val = deterministic_split(100, 0.9, seed=42)
    assert set(train).isdisjoint(set(val))
    assert len(set(train) | set(val)) == 100


def test_deterministic_split_invalid_args():
    from dpc.seeding import deterministic_split

    with pytest.raises(ValueError):
        deterministic_split(100, 0.0, seed=42)
    with pytest.raises(ValueError):
        deterministic_split(100, 1.0, seed=42)
    with pytest.raises(ValueError):
        deterministic_split(1, 0.5, seed=42)
