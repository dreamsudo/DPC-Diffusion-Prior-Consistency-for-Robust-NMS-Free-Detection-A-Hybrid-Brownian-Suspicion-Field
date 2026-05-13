"""Test checkpoint atomicity + roundtrip."""

import shutil
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_save_load_roundtrip(tmp_path):
    from dpc.checkpoint import save_checkpoint, load_checkpoint

    model = torch.nn.Linear(10, 5)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, model, optimizer=optim, meta={"epoch": 3})

    for f in ["model.pt", "optimizer.pt", "rng.pt", "meta.json", "SHA256SUMS"]:
        assert (ckpt / f).is_file(), f"missing {f}"

    model2 = torch.nn.Linear(10, 5)
    optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    meta = load_checkpoint(ckpt, model2, optimizer=optim2, restore_rng=False)
    assert meta["epoch"] == 3

    for (n1, p1), (n2, p2) in zip(model.state_dict().items(),
                                   model2.state_dict().items()):
        assert torch.equal(p1, p2), f"weights diverged: {n1}"


def test_save_overwrites_existing(tmp_path):
    from dpc.checkpoint import save_checkpoint, load_checkpoint

    model = torch.nn.Linear(4, 2)
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, model, meta={"v": 1})
    save_checkpoint(ckpt, model, meta={"v": 2})
    meta = load_checkpoint(ckpt, model, restore_rng=False)
    assert meta["v"] == 2


def test_atomic_temp_dir_cleanup_on_failure(tmp_path, monkeypatch):
    """Force save_checkpoint to fail mid-save and verify no tmp dirs leak."""
    from dpc.checkpoint import save_checkpoint

    model = torch.nn.Linear(2, 2)
    ckpt = tmp_path / "ckpt"

    # Inject failure during meta.json write
    real_save = torch.save
    n_calls = {"x": 0}

    def boom_save(obj, path, *a, **kw):
        n_calls["x"] += 1
        if n_calls["x"] >= 2:  # fail on 2nd save (after model.pt)
            raise RuntimeError("simulated disk error")
        return real_save(obj, path, *a, **kw)

    monkeypatch.setattr(torch, "save", boom_save)
    with pytest.raises(RuntimeError, match="simulated"):
        save_checkpoint(ckpt, model)

    # Ensure no leftover .tmp dirs
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert not leftovers, f"tmp dirs leaked: {leftovers}"
