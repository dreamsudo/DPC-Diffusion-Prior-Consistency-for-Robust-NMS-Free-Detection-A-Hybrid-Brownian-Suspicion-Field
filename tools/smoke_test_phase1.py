#!/usr/bin/env python3
"""End-to-end smoke test for Phase 1.

Run this AFTER bootstrap.sh and BEFORE you commit hours of training.
Verifies that the entire pipeline works at small scale by running:

  1. dpc imports OK
  2. denoiser builds, forward pass works
  3. field engine forward pass works
  4. dataset loads from cache and yields batches
  5. one optimization step succeeds with finite loss
  6. residual diagnostic on 4 images succeeds
  7. checkpoint save + load round-trips successfully

Total runtime: ~1-2 minutes on M1 Max.

Exit code:
    0  all phases passed - safe to launch real training
    1  at least one phase failed - investigate before training

Usage:
    python tools/smoke_test_phase1.py \
        --coco-cache caches/coco_train2017_128.pt \
        --apricot-eval-cache caches/apricot_eval_640.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


PHASES_PASSED: list[str] = []
PHASES_FAILED: list[tuple[str, str]] = []


def phase(name: str):
    def deco(fn):
        def wrapped(*args, **kwargs):
            print(f"\n{C.BOLD}{C.B}── {name} ──{C.END}")
            t0 = time.perf_counter()
            try:
                fn(*args, **kwargs)
                dt = time.perf_counter() - t0
                print(f"  {C.G}✓ pass{C.END} ({dt:.1f}s)")
                PHASES_PASSED.append(name)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"  {C.R}✗ fail{C.END}: {e}")
                print(tb)
                PHASES_FAILED.append((name, str(e)))
        return wrapped
    return deco


@phase("1. dpc imports")
def test_imports():
    from dpc import __version__
    from dpc.config import DPCConfig
    from dpc.diffusion import (
        make_beta_schedule, get_alpha_bars, add_noise, sample_timesteps
    )
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.field import DPCField
    from dpc.ema import EMA
    from dpc.checkpoint import save_checkpoint, load_checkpoint
    from dpc.data_cache import TensorCache
    from dpc.data import (
        NormalImageDataset, TensorAugment, collate_dpc_batch, make_coco_split
    )
    from dpc.metrics import (
        residual_ratio_per_image,
        aggregate_residual_distribution,
        bootstrap_ci,
    )
    from dpc.seeding import set_global_seed, make_generator
    print(f"  dpc version: {__version__}")


@phase("2. denoiser construction + forward")
def test_denoiser(args):
    from dpc.config import DPCConfig
    from dpc.denoiser import TinyUNetDenoiser

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"  device: {device}, params: {n_params:,}")

    x = torch.randn(2, 3, cfg.probe_res, cfg.probe_res, device=device)
    t = torch.randint(0, cfg.diffusion_steps, (2,), device=device, dtype=torch.long)
    out = denoiser(x, t)
    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
    print(f"  forward shape: {tuple(out.shape)} ✓")


@phase("3. field engine forward")
def test_field(args):
    from dpc.config import DPCConfig
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.field import DPCField
    from dpc.seeding import make_generator

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    field = DPCField(denoiser, cfg).to(device)

    x = torch.rand(2, 3, cfg.probe_res, cfg.probe_res, device=device)
    gen = make_generator(42, device=device)
    out = field(x, return_all=True, generator=gen)
    assert "deployed" in out and "raw_hybrid" in out
    assert out["deployed"].shape == (2, 1, cfg.probe_res, cfg.probe_res)
    print(f"  deployed shape: {tuple(out['deployed'].shape)}")
    print(f"  raw_hybrid range: [{out['raw_hybrid'].min().item():.4f}, "
          f"{out['raw_hybrid'].max().item():.4f}]")
    print(f"  deployed range:   [{out['deployed'].min().item():.4f}, "
          f"{out['deployed'].max().item():.4f}]  (should be ~[0, 1])")

    # Test compute_raw_signal
    raw = field.compute_raw_signal(x, generator=gen)
    assert {"residual", "abs", "l1", "l2", "hybrid"} <= set(raw.keys())
    print(f"  compute_raw_signal keys: {sorted(raw.keys())}")


@phase("4. cache loading + dataset")
def test_data(args):
    from dpc.config import DPCConfig
    from dpc.data import (
        NormalImageDataset, TensorAugment, collate_dpc_batch, make_coco_split
    )
    from dpc.data_cache import TensorCache

    cache_path = Path(args.coco_cache)
    if not cache_path.is_file():
        raise RuntimeError(
            f"COCO cache not found at {cache_path}. "
            f"Run tools/build_caches.py first."
        )
    cache = TensorCache(cache_path)
    print(f"  cache: {len(cache)} images at {cache.resolution}")

    train_idx, val_idx = make_coco_split(cache, val_frac=0.01, seed=42)
    print(f"  split: {len(train_idx)} train / {len(val_idx)} val")

    aug = TensorAugment(seed=42)
    ds = NormalImageDataset(cache, indices=train_idx[:32], transform=aug)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, collate_fn=collate_dpc_batch, num_workers=0,
    )
    batch = next(iter(loader))
    assert "images" in batch
    print(f"  batch images shape: {tuple(batch['images'].shape)}")


@phase("5. one optimization step")
def test_one_step(args):
    from dpc.config import DPCConfig
    from dpc.data import NormalImageDataset, collate_dpc_batch, make_coco_split
    from dpc.data_cache import TensorCache
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.diffusion import (
        add_noise, get_alpha_bars, make_beta_schedule, sample_timesteps,
    )
    from dpc.seeding import make_generator, set_global_seed

    set_global_seed(42)
    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()

    cache = TensorCache(Path(args.coco_cache))
    train_idx, _ = make_coco_split(cache, val_frac=0.01, seed=42)
    ds = NormalImageDataset(cache, indices=train_idx[:8])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, collate_fn=collate_dpc_batch, num_workers=0,
    )

    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    optim_ = torch.optim.AdamW(denoiser.parameters(), lr=1e-4)
    betas = make_beta_schedule(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end).to(device)
    alpha_bars = get_alpha_bars(betas)
    gen = make_generator(42, device=device)

    batch = next(iter(loader))
    x0 = batch["images"].to(device)
    b = x0.shape[0]
    t = sample_timesteps(b, cfg.timestep_min, cfg.timestep_max, device, gen)
    eps = torch.randn(x0.shape, device=device, generator=gen, dtype=x0.dtype)
    x_t = add_noise(x0, t, eps, alpha_bars)

    eps_hat = denoiser(x_t, t)
    loss = ((eps_hat - eps) ** 2).mean()
    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
    optim_.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
    assert torch.isfinite(grad_norm), f"grad norm not finite: {grad_norm.item()}"
    optim_.step()
    print(f"  loss={loss.item():.4f}  grad_norm={grad_norm.item():.3f}")


@phase("6. residual diagnostic on 4 images")
def test_diagnostic(args):
    if not args.apricot_eval_cache:
        print("  (skipped — no --apricot-eval-cache)")
        return
    from dpc.config import DPCConfig
    from dpc.data_cache import TensorCache
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.field import DPCField
    from dpc.metrics import residual_ratio_per_image, aggregate_residual_distribution
    from dpc.seeding import make_generator

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()

    apricot_path = Path(args.apricot_eval_cache)
    if not apricot_path.is_file():
        raise RuntimeError(f"APRICOT eval cache not found: {apricot_path}")
    cache = TensorCache(apricot_path)

    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    denoiser.eval()
    field = DPCField(denoiser, cfg).to(device)
    gen = make_generator(42, device=device)

    n_test = min(4, len(cache))
    per_image = []
    with torch.no_grad():
        for i in range(n_test):
            item = cache[i]
            img = item["image"].unsqueeze(0).to(device)
            H, W = img.shape[-2:]
            bboxes = item["metadata"].get("bboxes_xyxy", [])
            if not bboxes:
                continue
            mask = torch.zeros((H, W), dtype=torch.float32, device=device)
            for x1, y1, x2, y2 in bboxes:
                ix1 = max(0, int(x1)); iy1 = max(0, int(y1))
                ix2 = min(W, int(x2)); iy2 = min(H, int(y2))
                if ix2 > ix1 and iy2 > iy1:
                    mask[iy1:iy2, ix1:ix2] = 1.0
            if mask.sum() < 1:
                continue
            raw = field.compute_raw_signal(img, generator=gen)
            stats = residual_ratio_per_image(raw["hybrid"][0, 0], mask)
            per_image.append(stats)

    if per_image:
        agg = aggregate_residual_distribution(per_image)
        print(f"  diagnostic on {len(per_image)} apricot images:")
        print(f"    median_ratio (untrained): {agg['median_ratio']}  "
              f"(expect ≈ 1.0 since untrained)")
    else:
        print(f"  no images with valid bboxes among first {n_test}")


@phase("7. checkpoint save + load roundtrip")
def test_checkpoint(args):
    from dpc.checkpoint import save_checkpoint, load_checkpoint
    from dpc.config import DPCConfig
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.ema import EMA

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()

    denoiser_a = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    ema_a = EMA(denoiser_a).to(device)
    optim_a = torch.optim.AdamW(denoiser_a.parameters(), lr=1e-4)

    tmpdir = Path(tempfile.mkdtemp(prefix="dpc_smoke_"))
    try:
        ckpt_dir = tmpdir / "ckpt_test"
        save_checkpoint(ckpt_dir, denoiser_a, ema=ema_a, optimizer=optim_a,
                        meta={"smoke": True, "epoch": 0})
        for required in ["model.pt", "ema.pt", "optimizer.pt", "rng.pt", "meta.json", "SHA256SUMS"]:
            assert (ckpt_dir / required).is_file(), f"missing {required}"

        denoiser_b = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
        ema_b = EMA(denoiser_b).to(device)
        optim_b = torch.optim.AdamW(denoiser_b.parameters(), lr=1e-4)
        meta = load_checkpoint(ckpt_dir, denoiser_b, ema=ema_b, optimizer=optim_b,
                               restore_rng=False)
        assert meta.get("smoke") is True

        # Verify weights match
        for (na, pa), (nb, pb) in zip(denoiser_a.state_dict().items(),
                                       denoiser_b.state_dict().items()):
            assert torch.allclose(pa, pb), f"weight mismatch on {na}"
        print(f"  saved + reloaded; weights match exactly")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-cache", type=str, required=True,
                   help="Path to COCO train cache (built via tools/build_caches.py)")
    p.add_argument("--apricot-eval-cache", type=str, default=None,
                   help="Path to APRICOT eval cache (optional, for diagnostic test)")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    print(f"{C.BOLD}DPC Phase 1 Smoke Test{C.END}")
    print(f"  coco-cache:         {args.coco_cache}")
    print(f"  apricot-eval-cache: {args.apricot_eval_cache or '(skipped)'}")
    print(f"  device:             {args.device}")

    test_imports()
    test_denoiser(args)
    test_field(args)
    test_data(args)
    test_one_step(args)
    test_diagnostic(args)
    test_checkpoint(args)

    print()
    print(f"{C.BOLD}── Summary ──{C.END}")
    print(f"  Passed: {len(PHASES_PASSED)}")
    print(f"  Failed: {len(PHASES_FAILED)}")
    if PHASES_FAILED:
        print()
        print(f"  {C.R}{C.BOLD}FAIL{C.END} — DO NOT launch training until these are fixed:")
        for name, err in PHASES_FAILED:
            print(f"    - {name}: {err}")
        sys.exit(1)
    else:
        print()
        print(f"  {C.G}{C.BOLD}PASS{C.END} — Phase 1 pipeline is wired up correctly.")
        print(f"  Next step: launch training with tools/train_phase1.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
