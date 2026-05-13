#!/usr/bin/env python3
"""End-to-end smoke test for Phase 2.

Run AFTER bootstrap.sh, AFTER caches are built, AFTER fit_color_distribution,
and BEFORE the multi-hour Phase 2 training run.

Verifies:
  1. dpc imports OK (including new modules synthetic_patch, extended data)
  2. Phase 1 checkpoint loads
  3. SyntheticPatchGenerator builds a pixel-perfect mask
  4. CachedApricotDataset returns image + bbox-derived mask
  5. MixedDataset yields the right ratios
  6. Phase 2 forward pass (denoiser + field) succeeds
  7. One Phase 2 optimization step succeeds with finite multi-component loss
  8. Field supervision gradient flows through

Total runtime: ~1-2 minutes.

Exit code:
    0 — all phases passed; safe to launch real training
    1 — at least one phase failed; investigate before training

Usage:
    python tools/smoke_test_phase2.py \
        --phase1-checkpoint runs/phase1_seed42/checkpoints/best \
        --coco-cache caches/coco_train2017_128.pt \
        --apricot-cache caches/apricot_train_128.pt \
        --color-dist caches/color_distribution.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def phase(name: str):
    def deco(fn):
        def wrapped(*a, **kw):
            print(f"\n{C.BOLD}{C.B}── {name} ──{C.END}")
            t0 = time.perf_counter()
            try:
                fn(*a, **kw)
                dt = time.perf_counter() - t0
                print(f"  {C.G}✓ pass{C.END} ({dt:.1f}s)")
                PASSED.append(name)
            except Exception as e:
                import traceback
                print(f"  {C.R}✗ fail{C.END}: {e}")
                print(traceback.format_exc())
                FAILED.append((name, str(e)))
        return wrapped
    return deco


@phase("1. dpc imports (Phase 2 modules)")
def test_imports():
    from dpc import __version__
    from dpc.synthetic_patch import (
        SyntheticPatchGenerator, ColorDistribution, PatchSpec,
    )
    from dpc.data import (
        CachedApricotDataset, SyntheticPatchDataset, MixedDataset,
        NormalImageDataset, collate_dpc_batch,
    )
    from dpc.losses import focal_bce_with_logits, ssim_loss
    print(f"  dpc version: {__version__}")


@phase("2. Phase 1 checkpoint loads")
def test_phase1_checkpoint(args):
    from dpc.checkpoint import load_checkpoint
    from dpc.config import DPCConfig
    from dpc.denoiser import TinyUNetDenoiser

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    ckpt = Path(args.phase1_checkpoint).resolve()
    if not ckpt.is_dir():
        raise RuntimeError(f"Phase 1 checkpoint dir not found: {ckpt}")
    den = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    meta = load_checkpoint(ckpt, den, restore_rng=False)
    if (ckpt / "ema.pt").exists():
        ema_state = torch.load(ckpt / "ema.pt", map_location=device, weights_only=False)
        den.load_state_dict(ema_state)
        print(f"  loaded EMA weights")
    print(f"  meta: epoch={meta.get('epoch')}  step={meta.get('global_step')}  "
          f"version={meta.get('version')}")


@phase("3. SyntheticPatchGenerator")
def test_synthetic_patch(args):
    from dpc.synthetic_patch import SyntheticPatchGenerator

    gen = SyntheticPatchGenerator(
        Path(args.color_dist).resolve(), seed=42, image_size=128,
    )
    rng = np.random.RandomState(42)
    scene = torch.rand(3, 128, 128)
    patched, mask, spec = gen.render_random(scene, rng)
    assert patched.shape == (3, 128, 128)
    assert mask.shape == (1, 128, 128)
    # mask must be binary
    unique = set(np.unique(mask.numpy()).tolist())
    assert unique.issubset({0.0, 1.0}), f"mask not binary: {unique}"
    n_pix = int(mask.sum().item())
    assert n_pix > 4, f"mask too small: {n_pix} pixels"
    print(f"  spec: {spec.shape}/{spec.texture}, area_frac={spec.size_frac:.3f}")
    print(f"  mask: binary, {n_pix} pixels ({100 * n_pix / (128*128):.1f}% of image)")


@phase("4. CachedApricotDataset")
def test_apricot_dataset(args):
    from dpc.data import CachedApricotDataset
    from dpc.data_cache import TensorCache

    cache = TensorCache(Path(args.apricot_cache).resolve())
    ds = CachedApricotDataset(cache)
    assert len(ds) == len(cache)

    # Pull a few items, verify mask is non-empty for items with bboxes
    n_with_mask = 0
    for i in range(min(8, len(ds))):
        item = ds[i]
        assert "image" in item and "mask" in item
        assert item["image"].shape == (3, *cache.resolution)
        if item["mask"] is not None and item["mask"].sum() > 0:
            n_with_mask += 1
    print(f"  sampled 8 items, {n_with_mask} have non-empty masks")


@phase("5. MixedDataset ratios")
def test_mixed_dataset(args):
    from dpc.data import (
        CachedApricotDataset, MixedDataset, NormalImageDataset,
        SyntheticPatchDataset, collate_dpc_batch,
    )
    from dpc.data_cache import TensorCache
    from dpc.synthetic_patch import SyntheticPatchGenerator

    coco = TensorCache(Path(args.coco_cache).resolve())
    apricot = TensorCache(Path(args.apricot_cache).resolve())
    gen = SyntheticPatchGenerator(
        Path(args.color_dist).resolve(), seed=42,
        image_size=coco.resolution[0],
    )
    coco_ds = NormalImageDataset(coco, indices=list(range(min(500, len(coco)))))
    apricot_ds = CachedApricotDataset(apricot)
    synth_ds = SyntheticPatchDataset(
        coco, gen, coco_indices=list(range(min(500, len(coco)))),
        length=2000, base_seed=42,
    )

    mixed = MixedDataset(
        sources=[coco_ds, apricot_ds, synth_ds],
        weights=[0.5, 0.3, 0.2],
        length=200, base_seed=42,
    )
    sources = []
    for i in range(len(mixed)):
        item = mixed[i]
        sources.append(item["mix_source_idx"])
    sources = np.array(sources)
    n_coco = int((sources == 0).sum())
    n_apricot = int((sources == 1).sum())
    n_synth = int((sources == 2).sum())
    total = n_coco + n_apricot + n_synth
    print(f"  mix over 200 samples: coco={n_coco} ({100*n_coco/total:.1f}%) "
          f"apricot={n_apricot} ({100*n_apricot/total:.1f}%) "
          f"synth={n_synth} ({100*n_synth/total:.1f}%)")
    # Loose assertion — finite-sample, just check no source is missing
    assert n_coco > 50 and n_apricot > 30 and n_synth > 15, "mixture too imbalanced"

    loader = torch.utils.data.DataLoader(
        mixed, batch_size=4, shuffle=False, collate_fn=collate_dpc_batch,
    )
    batch = next(iter(loader))
    assert "images" in batch and "masks" in batch and "mask_validity" in batch
    print(f"  collated batch: images {tuple(batch['images'].shape)}, "
          f"valid_masks={int(batch['mask_validity'].sum())}/4")


@phase("6. Phase 2 forward pass")
def test_phase2_forward(args):
    from dpc.checkpoint import load_checkpoint
    from dpc.config import DPCConfig
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.field import DPCField
    from dpc.seeding import make_generator

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    den = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    ckpt = Path(args.phase1_checkpoint).resolve()
    load_checkpoint(ckpt, den, restore_rng=False)
    if (ckpt / "ema.pt").exists():
        den.load_state_dict(torch.load(ckpt / "ema.pt", map_location=device, weights_only=False))
    field = DPCField(den, cfg).to(device)
    field.train()

    x = torch.rand(2, 3, cfg.probe_res, cfg.probe_res, device=device)
    gen = make_generator(42, device=device)
    out = field(x, return_all=False, generator=gen)
    assert out["deployed"].shape == (2, 1, cfg.probe_res, cfg.probe_res)
    print(f"  deployed range: [{out['deployed'].min().item():.4f}, "
          f"{out['deployed'].max().item():.4f}]  (should be ~[0, 1])")


@phase("7. One Phase 2 optimization step")
def test_one_step(args):
    from dpc.checkpoint import load_checkpoint
    from dpc.config import DPCConfig
    from dpc.data import (
        CachedApricotDataset, MixedDataset, NormalImageDataset,
        SyntheticPatchDataset, collate_dpc_batch,
    )
    from dpc.data_cache import TensorCache
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.diffusion import (
        add_noise, get_alpha_bars, make_beta_schedule, sample_timesteps,
    )
    from dpc.field import DPCField
    from dpc.losses import focal_bce_with_logits, ssim_loss
    from dpc.seeding import make_generator, set_global_seed
    from dpc.synthetic_patch import SyntheticPatchGenerator
    import torch.nn.functional as F

    set_global_seed(42)
    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    den = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    ckpt = Path(args.phase1_checkpoint).resolve()
    load_checkpoint(ckpt, den, restore_rng=False)
    if (ckpt / "ema.pt").exists():
        den.load_state_dict(torch.load(ckpt / "ema.pt", map_location=device, weights_only=False))

    field = DPCField(den, cfg).to(device)
    optim_ = torch.optim.AdamW(den.parameters(), lr=1e-5)

    coco = TensorCache(Path(args.coco_cache).resolve())
    apricot = TensorCache(Path(args.apricot_cache).resolve())
    gen_synth = SyntheticPatchGenerator(
        Path(args.color_dist).resolve(), seed=42, image_size=coco.resolution[0],
    )
    mixed = MixedDataset(
        sources=[
            NormalImageDataset(coco, indices=list(range(200))),
            CachedApricotDataset(apricot),
            SyntheticPatchDataset(coco, gen_synth, list(range(200)), length=200, base_seed=42),
        ],
        weights=[0.5, 0.3, 0.2], length=8, base_seed=42,
    )
    loader = torch.utils.data.DataLoader(mixed, batch_size=4, shuffle=False, collate_fn=collate_dpc_batch)

    betas = make_beta_schedule(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end).to(device)
    alpha_bars = get_alpha_bars(betas)
    train_gen = make_generator(42, device=device)

    batch = next(iter(loader))
    x0 = batch["images"].to(device)
    masks = batch["masks"]
    validity = batch["mask_validity"]
    if masks is not None:
        masks = masks.to(device); validity = validity.to(device)

    b = x0.shape[0]
    t = sample_timesteps(b, cfg.timestep_min, cfg.timestep_max, device, train_gen)
    eps = torch.randn(x0.shape, device=device, generator=train_gen, dtype=x0.dtype)
    x_t = add_noise(x0, t, eps, alpha_bars)
    eps_hat = den(x_t, t)
    loss_mse = ((eps_hat - eps) ** 2).mean()

    if masks is not None and validity.any():
        field_out = field(x0, return_all=False, generator=train_gen)
        deployed = field_out["deployed"]
        valid_idx = validity.nonzero(as_tuple=True)[0]
        d_valid = deployed[valid_idx]; m_valid = masks[valid_idx]
        if d_valid.shape[-2:] != m_valid.shape[-2:]:
            m_valid = F.interpolate(m_valid, size=d_valid.shape[-2:], mode="bilinear", align_corners=False)
        d_clamp = d_valid.clamp(1e-6, 1 - 1e-6)
        logit = torch.log(d_clamp / (1 - d_clamp))
        loss_loc = focal_bce_with_logits(logit, m_valid)
        loss_ssim = ssim_loss(d_valid, m_valid)
    else:
        loss_loc = torch.zeros((), device=device)
        loss_ssim = torch.zeros((), device=device)

    total = loss_mse + loss_loc + 0.5 * loss_ssim
    assert torch.isfinite(total), f"loss not finite: {total.item()}"
    optim_.zero_grad()
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
    assert torch.isfinite(grad_norm)
    optim_.step()
    print(f"  total={total.item():.4f}  mse={loss_mse.item():.4f}  "
          f"loc={loss_loc.item():.4f}  ssim={loss_ssim.item():.4f}  "
          f"gn={grad_norm.item():.3f}")


@phase("8. Field supervision gradient flow")
def test_field_grad_flow(args):
    """Loss computed from field output must propagate gradient back to denoiser."""
    from dpc.config import DPCConfig
    from dpc.denoiser import TinyUNetDenoiser
    from dpc.field import DPCField
    from dpc.losses import focal_bce_with_logits
    from dpc.seeding import make_generator

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    den = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    field = DPCField(den, cfg).to(device)

    x = torch.rand(2, 3, cfg.probe_res, cfg.probe_res, device=device)
    mask = torch.zeros(2, 1, cfg.probe_res, cfg.probe_res, device=device)
    mask[:, :, :32, :32] = 1.0  # corner patches

    gen = make_generator(42, device=device)
    out = field(x, return_all=False, generator=gen)
    deployed = out["deployed"]
    eps_clamp = 1e-6
    d_clamp = deployed.clamp(eps_clamp, 1 - eps_clamp)
    logit = torch.log(d_clamp / (1 - d_clamp))
    loss = focal_bce_with_logits(logit, mask)
    loss.backward()

    # Check that at least some denoiser params received gradient
    grad_total = 0.0
    n_params = 0
    for p in den.parameters():
        if p.grad is not None:
            grad_total += float(p.grad.data.norm(2).item()) ** 2
            n_params += 1
    grad_total = grad_total ** 0.5
    print(f"  field-supervision grad norm on denoiser: {grad_total:.4f} "
          f"({n_params} param tensors with gradient)")
    assert grad_total > 0, "no gradient flowed through field engine to denoiser"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-checkpoint", required=True)
    p.add_argument("--coco-cache", required=True)
    p.add_argument("--apricot-cache", required=True)
    p.add_argument("--color-dist", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    print(f"{C.BOLD}DPC Phase 2 Smoke Test{C.END}")
    print(f"  phase1-checkpoint: {args.phase1_checkpoint}")
    print(f"  coco-cache:        {args.coco_cache}")
    print(f"  apricot-cache:     {args.apricot_cache}")
    print(f"  color-dist:        {args.color_dist}")

    test_imports()
    test_phase1_checkpoint(args)
    test_synthetic_patch(args)
    test_apricot_dataset(args)
    test_mixed_dataset(args)
    test_phase2_forward(args)
    test_one_step(args)
    test_field_grad_flow(args)

    print()
    print(f"{C.BOLD}── Summary ──{C.END}")
    print(f"  Passed: {len(PASSED)}")
    print(f"  Failed: {len(FAILED)}")
    if FAILED:
        print()
        print(f"  {C.R}{C.BOLD}FAIL{C.END} — DO NOT launch Phase 2 training:")
        for name, err in FAILED:
            print(f"    - {name}: {err}")
        sys.exit(1)
    else:
        print()
        print(f"  {C.G}{C.BOLD}PASS{C.END} — Phase 2 pipeline is wired up correctly.")
        print(f"  Next: tools/sanity_check_data.py + tools/sanity_check_loss.py + tools/train_phase2.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
