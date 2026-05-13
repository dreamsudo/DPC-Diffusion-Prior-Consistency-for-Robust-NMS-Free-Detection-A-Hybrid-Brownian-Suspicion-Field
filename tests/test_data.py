"""Test Phase 2 datasets: MixedDataset ratios, collate handles None masks."""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


class _StubDataset:
    """Minimal Dataset returning constant items, for testing MixedDataset."""

    def __init__(self, name, n=100, has_mask=True):
        self.name = name
        self.n = n
        self.has_mask = has_mask

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = torch.full((3, 32, 32), float(i % 7) / 7)
        if self.has_mask:
            mask = torch.zeros((1, 32, 32))
            mask[0, :8, :8] = 1.0
        else:
            mask = None
        return {
            "image": img,
            "mask": mask,
            "source": self.name,
            "path": f"{self.name}/{i}",
        }


def test_mixed_dataset_weight_ratios_approx():
    """With 5000 samples at weights 0.5/0.3/0.2, counts should be near nominal."""
    from dpc.data import MixedDataset

    a = _StubDataset("A", n=200, has_mask=False)
    b = _StubDataset("B", n=200, has_mask=True)
    c = _StubDataset("C", n=200, has_mask=True)

    mixed = MixedDataset([a, b, c], weights=[0.5, 0.3, 0.2], length=5000, base_seed=42)
    counts = {0: 0, 1: 0, 2: 0}
    for i in range(len(mixed)):
        item = mixed[i]
        counts[item["mix_source_idx"]] += 1
    total = sum(counts.values())
    pcts = {k: v / total for k, v in counts.items()}
    # 5000 samples → SE ≈ sqrt(0.5 * 0.5 / 5000) ≈ 0.007. Use 0.04 tolerance.
    assert abs(pcts[0] - 0.5) < 0.04, f"A ratio off: {pcts[0]:.3f}"
    assert abs(pcts[1] - 0.3) < 0.04, f"B ratio off: {pcts[1]:.3f}"
    assert abs(pcts[2] - 0.2) < 0.04, f"C ratio off: {pcts[2]:.3f}"


def test_mixed_dataset_invalid_weights():
    from dpc.data import MixedDataset
    a = _StubDataset("A")
    with pytest.raises(ValueError):
        MixedDataset([a, a], weights=[0.5, 0.4], length=10, base_seed=42)
    with pytest.raises(ValueError):
        MixedDataset([a, a], weights=[0.5], length=10, base_seed=42)
    with pytest.raises(ValueError):
        MixedDataset([], weights=[], length=10, base_seed=42)


def test_mixed_dataset_reproducible():
    from dpc.data import MixedDataset
    a = _StubDataset("A"); b = _StubDataset("B"); c = _StubDataset("C")
    m1 = MixedDataset([a, b, c], [0.5, 0.3, 0.2], length=200, base_seed=42)
    m2 = MixedDataset([a, b, c], [0.5, 0.3, 0.2], length=200, base_seed=42)
    src1 = [m1[i]["mix_source_idx"] for i in range(200)]
    src2 = [m2[i]["mix_source_idx"] for i in range(200)]
    assert src1 == src2


def test_collate_handles_none_masks():
    from dpc.data import collate_dpc_batch

    a = _StubDataset("A", has_mask=False)
    b = _StubDataset("B", has_mask=True)
    batch = [a[0], b[1], a[2], b[3]]
    batch = [dict(item, mix_source_idx=i % 2) for i, item in enumerate(batch)]
    out = collate_dpc_batch(batch)
    assert out["images"].shape == (4, 3, 32, 32)
    assert out["masks"] is not None
    assert out["mask_validity"] is not None
    # Items 0 and 2 are 'A' (no mask) → validity False
    assert out["mask_validity"].tolist() == [False, True, False, True]


def test_collate_all_none_masks():
    from dpc.data import collate_dpc_batch

    a = _StubDataset("A", has_mask=False)
    batch = [a[i] for i in range(3)]
    out = collate_dpc_batch(batch)
    assert out["masks"] is None
    assert out["mask_validity"] is None


def test_collate_preserves_paths_and_sources():
    from dpc.data import collate_dpc_batch

    a = _StubDataset("A", has_mask=False); b = _StubDataset("B", has_mask=True)
    batch = [a[0], b[1]]
    out = collate_dpc_batch(batch)
    assert out["sources"] == ["A", "B"]
    assert out["paths"] == ["A/0", "B/1"]
