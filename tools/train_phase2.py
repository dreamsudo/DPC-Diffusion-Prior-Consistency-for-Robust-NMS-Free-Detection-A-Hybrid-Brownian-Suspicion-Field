"""Phase 2 trainer: joint denoiser + YOLO26 head training.

Implements §5.16, §5.13, §5.14, §5.17 of the paper:

  - At each step, run the denoiser on perturbed inputs to produce a
    suspicion field over a batch of training images.
  - Run YOLO26 in native mode to get raw head output (anchors, obj logits,
    cls logits).
  - For each image with ground truth: compute base cost C_base, modulate
    with λ_match · β_i^(small) per Eq. (22), solve Hungarian per Eq. (23).
  - Compute the unified loss (§5.17):
      L_total = λ_mse·L_ε
              + λ_loc·L_loc_matched
              + λ_cls·L_cls_matched + λ_conf·L_conf_matched
              + λ_entropy·L_cls-prior (Eq. 19)
              + λ_locstab·L_box-stab (Eq. 20)
              + λ_anchor·||θ_head - θ_head^pretrained||²
  - Backprop into denoiser AND YOLO26 head (backbone frozen).

Datasets: mixture of COCO (clean, no patch), APRICOT (real patches), and
synthetic patches. Synthetic patches give the field-supervision signal
that lets the denoiser learn to localize off-manifold regions.

Args follow Phase 1's pattern; new flags handle YOLO26 weights, λ
hyperparameters, and the weight-anchor regularizer.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dpc.assignment import hungarian_assign, modulate_cost_matrix
from dpc.auxiliary_losses import box_stability_loss, class_entropy_regularizer
from dpc.calibration import amplify_small_targets
from dpc.checkpoint import load_checkpoint, prune_step_checkpoints, save_checkpoint, update_symlink
from dpc.config import DPCConfig
from dpc.data import (
    CachedApricotDataset,
    MixedDataset,
    NormalImageDataset,
    SyntheticPatchDataset,
    collate_dpc_batch,
    make_apricot_indices,
    make_coco_split,
)
from dpc.data_cache import TensorCache, apricot_metadata_fn_factory
from dpc.denoiser import TinyUNetDenoiser
from dpc.diffusion import add_noise, get_alpha_bars, make_beta_schedule, sample_timesteps
from dpc.ema import EMA
from dpc.field import DPCField
from dpc.manifest import write_manifest
from dpc.pooling import box_areas_frac, box_pool_grid
from dpc.seeding import make_generator, set_global_seed
from dpc.synthetic_patch import SyntheticPatchGenerator
from dpc.yolo26_native import (
    compute_base_cost,
    forward_yolo26_raw,
    load_yolo26,
    slice_raw,
)


log = logging.getLogger("train_phase2")


# ─── Argument parser ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPC-YOLO26 Phase 2 joint training")
    # Checkpoints and weights
    p.add_argument("--phase1-checkpoint", type=str, required=True,
                   help="Path to Phase 1 denoiser checkpoint directory")
    p.add_argument("--use-ema", dest="use_ema", action="store_true", default=True,
                   help="Use Phase 1 EMA weights (default)")
    p.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    p.add_argument("--yolo-weights", type=str, required=True,
                   help="Path to yolo26n.pt or any YOLO26 weights file")

    # Data caches
    p.add_argument("--coco-cache", type=str, required=True)
    p.add_argument("--apricot-cache", type=str, required=True)
    p.add_argument("--apricot-val-cache", type=str, default=None)
    p.add_argument("--color-dist", type=str, required=True,
                   help="Path to APRICOT color distribution JSON (for synthetic patches)")

    # Output and reproducibility
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    # Training schedule
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-denoiser", type=float, default=1e-4)
    p.add_argument("--lr-yolo-head", type=float, default=1e-5,
                   help="Lower than denoiser LR; YOLO head is fine-tuned, not retrained")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--ema-decay", type=float, default=0.999)

    # Mixture proportions
    p.add_argument("--p-coco", type=float, default=0.4)
    p.add_argument("--p-apricot", type=float, default=0.4)
    p.add_argument("--p-synthetic", type=float, default=0.2)

    # Loss weights (defaults from cfg; CLI can override)
    p.add_argument("--lambda-mse", type=float, default=None)
    p.add_argument("--lambda-match", type=float, default=None)
    p.add_argument("--lambda-entropy", type=float, default=None)
    p.add_argument("--lambda-locstab", type=float, default=None)
    p.add_argument("--lambda-anchor", type=float, default=None)
    p.add_argument("--lambda-loc", type=float, default=None)
    p.add_argument("--lambda-conf", type=float, default=None)

    # Logging
    p.add_argument("--log-every-steps", type=int, default=20)
    p.add_argument("--save-ckpt-every-steps", type=int, default=500)

    return p.parse_args()


# ─── Data setup ──────────────────────────────────────────────────────────────


def build_mixed_dataset(args: argparse.Namespace, cfg: DPCConfig):
    """Construct the COCO+APRICOT+synthetic mixed training dataset.

    Uses the real v3.3.0 data APIs:
      - TensorCache(path) (constructor, not .load())
      - SyntheticPatchGenerator(color_distribution_path, seed, image_size, ...)
      - SyntheticPatchDataset(coco_cache, patch_generator, coco_indices, length, base_seed)
      - MixedDataset(sources, weights, length, base_seed)
      - make_apricot_indices(cache) -> list[int]   (NOT a tuple)
    """
    coco_cache = TensorCache(args.coco_cache)
    apricot_cache = TensorCache(args.apricot_cache)

    coco_train_idx, _ = make_coco_split(coco_cache, val_frac=0.05, seed=args.seed)
    apricot_train_idx = make_apricot_indices(apricot_cache)

    coco_ds = NormalImageDataset(coco_cache, coco_train_idx)
    apricot_ds = CachedApricotDataset(apricot_cache, apricot_train_idx)

    synth_gen = SyntheticPatchGenerator(
        color_distribution_path=args.color_dist,
        seed=args.seed,
        image_size=cfg.probe_res,
    )
    synth_ds = SyntheticPatchDataset(
        coco_cache=coco_cache,
        patch_generator=synth_gen,
        coco_indices=coco_train_idx,
        length=len(coco_train_idx),
        base_seed=args.seed,
    )

    # MixedDataset draws `length` items uniformly across an epoch
    mixture_length = args.steps_per_epoch * args.batch_size
    return MixedDataset(
        sources=[coco_ds, apricot_ds, synth_ds],
        weights=[args.p_coco, args.p_apricot, args.p_synthetic],
        length=mixture_length,
        base_seed=args.seed,
    )


# ─── Joint training step ─────────────────────────────────────────────────────


def joint_training_step(
    images: torch.Tensor,
    gt_boxes: list[torch.Tensor],
    gt_classes: list[torch.Tensor],
    denoiser: torch.nn.Module,
    yolo_inner: torch.nn.Module,
    yolo_head_pretrained_state: dict,
    yolo_head_params: list[torch.nn.Parameter],
    field: DPCField,
    cfg: DPCConfig,
    alpha_bars: torch.Tensor,
    train_gen: torch.Generator,
    n_classes: int,
) -> dict:
    """Single joint training step.

    Returns a dict of scalar losses for logging:
      {loss_total, loss_mse, loss_match, loss_locstab, loss_entropy,
       loss_loc, loss_cls, loss_conf, loss_anchor, n_matched}
    """
    device = images.device
    B = images.shape[0]
    H, W = images.shape[2], images.shape[3]

    # ── Denoiser forward and L_ε (Eq. 2) ──────────────────────────────────
    t = sample_timesteps(B, cfg.timestep_min, cfg.timestep_max, device, train_gen)
    eps = torch.randn(images.shape, device=device, generator=train_gen, dtype=images.dtype)
    x_t = add_noise(images, t, eps, alpha_bars)
    eps_hat = denoiser(x_t, t)
    loss_mse = ((eps_hat - eps) ** 2).mean()

    # ── Suspicion field from K-probe ensemble (§5.3–§5.9) ─────────────────
    # Use the same generator for reproducibility
    with torch.enable_grad():
        # Field.forward is no_grad-decorated for inference; for training we
        # bypass and call the internal pipeline directly. The denoiser
        # weights need gradient flow.
        # NOTE: K-probe ensemble at training time uses the FIXED schedule
        # (§5.4) from cfg.probe_timesteps.
        x0_probe = images
        if H != cfg.probe_res or W != cfg.probe_res:
            x0_probe = F.interpolate(
                images, size=(cfg.probe_res, cfg.probe_res),
                mode="bilinear", align_corners=False,
            )

        # Run K probes with fresh noise and fixed timesteps
        delta, _ = field.compute_per_probe_residual(x0_probe, generator=train_gen)
        r_l1_per_probe, r_l2_per_probe = field.compute_residual_summaries(delta)
        l1_branch = field.branch_field(r_l1_per_probe)
        l2_branch = field.branch_field(r_l2_per_probe)
        l1_norm = field.normalize_per_image(l1_branch)
        l2_norm = field.normalize_per_image(l2_branch)
        hybrid = field.fuse_branches(l1_norm, l2_norm)
        if cfg.deployment_mode == "l1":
            deployed_lo = l1_norm
        elif cfg.deployment_mode == "l2":
            deployed_lo = l2_norm
        else:
            deployed_lo = hybrid
        deployed = field.upsample_to_image(deployed_lo, H, W)

    # ── YOLO26 native forward ─────────────────────────────────────────────
    raw = forward_yolo26_raw(yolo_inner, images, n_classes=n_classes)

    # ── Per-image assignment, modulation, matched losses ──────────────────
    loss_loc_per_image = []
    loss_cls_per_image = []
    loss_conf_per_image = []
    loss_entropy_per_image = []
    n_matched_total = 0

    for i in range(B):
        raw_i = slice_raw(raw, i)
        gt_b = gt_boxes[i].to(device) if gt_boxes[i].numel() > 0 else gt_boxes[i].new_empty((0, 4)).to(device)
        gt_c = gt_classes[i].to(device) if gt_classes[i].numel() > 0 else gt_classes[i].new_empty((0,)).to(device).long()
        M = gt_b.shape[0]

        if M == 0:
            # No GT in this image; only the "no-object" loss applies. For
            # simplicity in v3.3.0 we omit the negative-class loss here and
            # rely on the entropy regularizer to push uncertain predictions
            # in suspect regions.
            continue

        # Box pooling per anchor for this image
        A = raw_i.boxes_xyxy.shape[0]
        image_idx_zeros = torch.zeros((A,), dtype=torch.long, device=device)
        beta = box_pool_grid(
            deployed[i : i + 1], raw_i.boxes_xyxy, image_idx_zeros,
            pool_size=cfg.pool_size,
        )
        areas_frac = box_areas_frac(raw_i.boxes_xyxy, (H, W))
        beta_small = amplify_small_targets(
            beta, areas_frac, cfg.lambda_small, cfg.small_target_area_threshold,
        )

        # Base cost matrix [A, M]
        cost_base = compute_base_cost(raw_i, gt_b, gt_c)

        # Modulated cost matrix (Eq. 22)
        cost_dpc = modulate_cost_matrix(cost_base, beta_small, cfg.lambda_match)

        # Hungarian (Eq. 23)
        with torch.no_grad():
            # Hungarian solver is non-differentiable; we use it to pick
            # match indices, then compute differentiable losses on those.
            pred_idx, gt_idx = hungarian_assign(cost_dpc)

        n_matched_total += pred_idx.numel()

        # Matched losses (YOLO26: no objectness, only box + class)
        matched_pred_boxes = raw_i.boxes_xyxy[pred_idx]
        matched_pred_cls = raw_i.cls_logits[pred_idx]
        matched_gt_boxes = gt_b[gt_idx]
        matched_gt_cls = gt_c[gt_idx]

        # L1 box loss (paper notation: λ_loc · L_loc)
        loc_loss = (matched_pred_boxes - matched_gt_boxes).abs().mean()
        loss_loc_per_image.append(loc_loss)

        # Class loss: BCE on the matched class
        K_cls = matched_pred_cls.shape[-1]
        cls_target = F.one_hot(matched_gt_cls, num_classes=K_cls).to(matched_pred_cls.dtype)
        cls_loss = F.binary_cross_entropy_with_logits(matched_pred_cls, cls_target, reduction="mean")
        loss_cls_per_image.append(cls_loss)

        # YOLO26 has no separate objectness; the "matched anchor should be
        # confident" signal is already produced by loss_cls (BCE on the
        # one-hot class target). conf_loss is set to a zero tensor so the
        # accumulator and logging code below don't have to special-case
        # the YOLO26 path.
        conf_loss = torch.zeros((), device=device, dtype=matched_pred_cls.dtype)
        loss_conf_per_image.append(conf_loss)

        # Eq. (19) class-entropy regularizer (only on matched predictions
        # — applying to all 8400 anchors makes the regularizer dominate)
        entropy_reg = class_entropy_regularizer(matched_pred_cls, beta_small[pred_idx], K_cls)
        loss_entropy_per_image.append(entropy_reg / max(1, pred_idx.numel()))

    # Aggregate per-image losses
    def _safe_mean(items: list[torch.Tensor]) -> torch.Tensor:
        if not items:
            return torch.zeros((), device=device, dtype=images.dtype)
        return torch.stack(items).mean()

    loss_loc = _safe_mean(loss_loc_per_image)
    loss_cls = _safe_mean(loss_cls_per_image)
    loss_conf = _safe_mean(loss_conf_per_image)
    loss_entropy = _safe_mean(loss_entropy_per_image)

    # ── Eq. (20) Brownian localization stability ──────────────────────────
    # Run YOLO26 a second time on a different probe perturbation, compare boxes
    if cfg.lambda_locstab > 0:
        # Pick two random different probe timesteps
        K = cfg.n_probes
        ka, kb = 0, K - 1  # use extreme probes for biggest signal
        sigma_a_t = field.probe_timesteps[ka].item()
        sigma_b_t = field.probe_timesteps[kb].item()

        t_a = torch.full((B,), sigma_a_t, dtype=torch.long, device=device)
        t_b = torch.full((B,), sigma_b_t, dtype=torch.long, device=device)
        eps_a = torch.randn(images.shape, device=device, generator=train_gen, dtype=images.dtype)
        eps_b = torch.randn(images.shape, device=device, generator=train_gen, dtype=images.dtype)
        x_a = add_noise(images, t_a, eps_a, alpha_bars)
        x_b = add_noise(images, t_b, eps_b, alpha_bars)
        raw_a = forward_yolo26_raw(yolo_inner, x_a, n_classes=n_classes)
        raw_b = forward_yolo26_raw(yolo_inner, x_b, n_classes=n_classes)
        # Stability over all anchors — average L1 per box
        loss_locstab = box_stability_loss(raw_a.boxes_xyxy, raw_b.boxes_xyxy) / (
            raw_a.boxes_xyxy.numel()
        )
    else:
        loss_locstab = torch.zeros((), device=device, dtype=images.dtype)

    # ── Weight-anchor regularizer ─────────────────────────────────────────
    # ‖θ_head − θ_head^(pretrained)‖²
    if cfg.lambda_anchor > 0 and yolo_head_pretrained_state:
        anchor_terms = []
        for name, param in zip(
            (n for n, _ in yolo_inner.named_parameters() if "head" in n.lower()),
            yolo_head_params,
        ):
            pre = yolo_head_pretrained_state.get(name)
            if pre is not None and pre.shape == param.shape:
                anchor_terms.append(((param - pre.to(param.device)) ** 2).sum())
        loss_anchor = (
            torch.stack(anchor_terms).sum() if anchor_terms
            else torch.zeros((), device=device, dtype=images.dtype)
        )
    else:
        loss_anchor = torch.zeros((), device=device, dtype=images.dtype)

    # ── Total (Eq. §5.17) ─────────────────────────────────────────────────
    total = (
        cfg.lambda_mse * loss_mse
        + cfg.lambda_loc * loss_loc
        + cfg.lambda_cls * loss_cls
        + cfg.lambda_conf * loss_conf
        + cfg.lambda_entropy * loss_entropy
        + cfg.lambda_locstab * loss_locstab
        + cfg.lambda_anchor * loss_anchor
    )
    # NOTE: λ_match enters via the cost matrix in modulate_cost_matrix; it
    # is not a separate loss term. It affects the picked match indices and
    # therefore the matched-loss terms above.

    return {
        "loss_total": total,
        "loss_mse": loss_mse.detach(),
        "loss_loc": loss_loc.detach(),
        "loss_cls": loss_cls.detach(),
        "loss_conf": loss_conf.detach(),
        "loss_entropy": loss_entropy.detach(),
        "loss_locstab": loss_locstab.detach(),
        "loss_anchor": loss_anchor.detach(),
        "n_matched": n_matched_total,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    set_global_seed(args.seed)
    cfg = DPCConfig()
    if args.device != "auto":
        cfg.device = args.device
    device = cfg.get_device()

    # CLI overrides on loss weights
    for name in (
        "lambda_mse", "lambda_match", "lambda_entropy", "lambda_locstab",
        "lambda_anchor", "lambda_loc", "lambda_conf",
    ):
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)

    # Load Phase 1 denoiser
    log.info(f"Loading Phase 1 denoiser from {args.phase1_checkpoint}")
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    ema = EMA(denoiser, decay=args.ema_decay)
    load_checkpoint(args.phase1_checkpoint, denoiser, ema=ema, restore_rng=False)
    if args.use_ema:
        denoiser.load_state_dict(ema.state_dict())

    # Load YOLO26
    log.info(f"Loading YOLO26 weights from {args.yolo_weights}")
    yolo_inner = load_yolo26(args.yolo_weights, device)

    # Snapshot pretrained head weights for the anchor regularizer
    pretrained_head_state = {
        name: param.detach().clone().cpu()
        for name, param in yolo_inner.named_parameters()
        if name.startswith("model.23.")
    }
    log.info(f"Snapshotted {len(pretrained_head_state)} pretrained head params for anchor regularizer")

    # Freeze backbone; collect head params
    yolo_head_params: list[torch.nn.Parameter] = []
    for name, param in yolo_inner.named_parameters():
        if name.startswith("model.23."):
            param.requires_grad_(True)
            yolo_head_params.append(param)
        else:
            param.requires_grad_(False)
    log.info(f"Training {sum(p.numel() for p in yolo_head_params):,} YOLO26 head parameters")

    # Field engine (its denoiser ref is the same one we train)
    field = DPCField(denoiser, cfg).to(device)
    field.train()
    denoiser.train()
    yolo_inner.train()

    # Beta schedule for diffusion in the training step
    alpha_bars = field.alpha_bars.to(device)

    # Optimizers
    opt_denoiser = torch.optim.AdamW(
        denoiser.parameters(),
        lr=args.lr_denoiser, weight_decay=args.weight_decay,
    )
    opt_yolo = torch.optim.AdamW(
        yolo_head_params,
        lr=args.lr_yolo_head, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * args.steps_per_epoch
    sched_denoiser = torch.optim.lr_scheduler.LambdaLR(
        opt_denoiser,
        lambda s: min(1.0, s / max(1, args.warmup_steps)),
    )
    sched_yolo = torch.optim.lr_scheduler.LambdaLR(
        opt_yolo,
        lambda s: min(1.0, s / max(1, args.warmup_steps)),
    )

    # Data
    mixed_ds = build_mixed_dataset(args, cfg)
    train_gen = make_generator(args.seed, device)
    if args.num_workers < 0:
        args.num_workers = max(1, (os.cpu_count() or 4) - 2)
        log.info(f"resolved num_workers={args.num_workers}")
    loader = DataLoader(
        mixed_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_dpc_batch,
        shuffle=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Training loop
    global_step = 0
    best_loss = float("inf")
    stop = False

    def _sigint_handler(signum, frame):
        nonlocal stop
        log.warning("SIGINT received; will stop after this step")
        stop = True

    signal.signal(signal.SIGINT, _sigint_handler)

    for epoch in range(args.epochs):
        if stop:
            break
        epoch_start = time.time()
        steps_this_epoch = 0
        running = {"total": 0.0, "mse": 0.0, "match": 0.0, "loc": 0.0, "cls": 0.0,
                   "conf": 0.0, "ent": 0.0, "lstab": 0.0, "anc": 0.0}

        for batch in loader:
            if stop:
                break
            if steps_this_epoch >= args.steps_per_epoch:
                break

            images = batch["images"].to(device, non_blocking=True)
            B = images.shape[0]
            gt_boxes = batch.get("gt_boxes")
            gt_classes = batch.get("gt_classes")
            if gt_boxes is None or gt_classes is None:
                gt_boxes = [torch.empty((0, 4), dtype=torch.float32) for _ in range(B)]
                gt_classes = [torch.empty((0,), dtype=torch.long) for _ in range(B)]

            losses = joint_training_step(
                images=images,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                denoiser=denoiser,
                yolo_inner=yolo_inner,
                yolo_head_pretrained_state=pretrained_head_state,
                yolo_head_params=yolo_head_params,
                field=field,
                cfg=cfg,
                alpha_bars=alpha_bars,
                train_gen=train_gen,
                n_classes=80,
            )
            total = losses["loss_total"]
            if not torch.isfinite(total):
                log.error(f"step {global_step}: non-finite loss; skipping")
                opt_denoiser.zero_grad()
                opt_yolo.zero_grad()
                continue

            opt_denoiser.zero_grad()
            opt_yolo.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(yolo_head_params, args.grad_clip_norm)
            opt_denoiser.step()
            opt_yolo.step()
            sched_denoiser.step()
            sched_yolo.step()
            ema.update(denoiser)

            running["total"] += float(total)
            running["mse"] += float(losses["loss_mse"])
            running["loc"] += float(losses["loss_loc"])
            running["cls"] += float(losses["loss_cls"])
            running["conf"] += float(losses["loss_conf"])
            running["ent"] += float(losses["loss_entropy"])
            running["lstab"] += float(losses["loss_locstab"])
            running["anc"] += float(losses["loss_anchor"])

            global_step += 1
            steps_this_epoch += 1

            if global_step % args.log_every_steps == 0:
                n = args.log_every_steps
                log.info(
                    f"epoch {epoch+1}/{args.epochs} step {global_step} "
                    f"total={running['total']/n:.4f} mse={running['mse']/n:.4f} "
                    f"loc={running['loc']/n:.4f} cls={running['cls']/n:.4f} "
                    f"conf={running['conf']/n:.4f} ent={running['ent']/n:.4f} "
                    f"lstab={running['lstab']/n:.4f} anc={running['anc']/n:.4f} "
                    f"n_matched={losses['n_matched']}"
                )
                running = {k: 0.0 for k in running}

            if global_step % args.save_ckpt_every_steps == 0:
                step_ckpt = ckpt_dir / f"step_{global_step:08d}"
                save_checkpoint(
                    str(step_ckpt), denoiser, ema=ema,
                    optimizer=opt_denoiser, scheduler=sched_denoiser,
                    meta={"global_step": global_step, "epoch": epoch + 1, "phase": "2"},
                )
                update_symlink(str(step_ckpt), str(ckpt_dir / "latest"))
                prune_step_checkpoints(str(ckpt_dir), keep_last=3)

        epoch_dt = time.time() - epoch_start
        log.info(f"epoch {epoch+1} done in {epoch_dt:.1f}s ({steps_this_epoch} steps)")

    # Save final
    final_ckpt = ckpt_dir / "final"
    save_checkpoint(
        str(final_ckpt), denoiser, ema=ema,
        optimizer=opt_denoiser, scheduler=sched_denoiser,
        meta={"global_step": global_step, "phase": "2", "final": True},
    )
    update_symlink(str(final_ckpt), str(ckpt_dir / "latest"))

    # Save YOLO26 head weights too (the trained head is part of v3.3.0's
    # deliverable; Phase 3 must load these, not the pretrained ones)
    yolo_head_out = out_dir / "yolo26_head_finetuned.pt"
    yolo_head_state = {
        name: param.detach().cpu()
        for name, param in yolo_inner.named_parameters()
        if name.startswith("model.23.")
    }
    torch.save(yolo_head_state, yolo_head_out)
    log.info(f"saved fine-tuned head weights to {yolo_head_out}")

    write_manifest(
        str(out_dir),
        extra_meta={
            "phase": "phase2",
            "args": vars(args),
            "global_step": global_step,
            "cfg": cfg.asdict(),
        },
    )

    return 0 if not stop else 130


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
