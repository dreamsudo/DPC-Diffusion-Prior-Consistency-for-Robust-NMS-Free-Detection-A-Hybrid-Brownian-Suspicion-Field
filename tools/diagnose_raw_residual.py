#!/usr/bin/env python3
"""4-panel visualization: input | mask | residual GLOBAL scale | residual PER-IMAGE scale.

This is the diagnostic that produces visual evidence for sub-claim S3 from
per-image normalization. The two
right panels show the same residual map, but with different colormaps:
  - GLOBAL: same colormap range across all images (raw absolute scale)
  - PER-IMAGE: independently min-max normalized per image

If per-image normalization preserves signal, both panels should look
qualitatively similar (signal-where-patches-are). If global-scale shows
near-uniform brightness while per-image shows clear patch localization, the
field engine's normalization is destroying useful absolute-scale information.

Usage:
    python tools/diagnose_raw_residual.py \
        --checkpoint runs/phase1_seed42/checkpoints/best \
        --apricot-eval-cache caches/apricot_eval_640.pt \
        --use-ema \
        --n-vis 50 \
        --seed 42 \
        --output-dir runs/phase1_seed42/raw_residual_diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from dpc._version import __version__
from dpc.checkpoint import load_checkpoint
from dpc.config import DPCConfig
from dpc.data_cache import TensorCache
from dpc.denoiser import TinyUNetDenoiser
from dpc.field import DPCField
from dpc.manifest import fingerprint_environment
from dpc.metrics import residual_ratio_per_image
from dpc.seeding import make_generator, set_global_seed


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def warn(m): print(f"  {C.Y}!{C.END} {m}")


def build_mask(bboxes_xyxy, H, W, device):
    """Build a binary mask from xyxy bboxes. Returns zeros if no valid bbox."""
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    for x1, y1, x2, y2 in bboxes_xyxy:
        ix1 = max(0, int(round(x1))); iy1 = max(0, int(round(y1)))
        ix2 = min(W, int(round(x2))); iy2 = min(H, int(round(y2)))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 1.0
    return mask


def fmt(x, spec=".4f", default="N/A"):
    """Format-or-fallback. Lets us safely print possibly-None numerics."""
    if x is None:
        return default
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--apricot-eval-cache", required=True)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--n-vis", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    print(f"{C.BOLD}DPC 4-panel Residual Viz v{__version__}{C.END}")
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "grids").mkdir(parents=True, exist_ok=True)

    cfg = DPCConfig(device=args.device)
    device = cfg.get_device()
    info(f"device: {device}")

    # ── Load model ──
    stage("Load denoiser")
    ckpt_path = Path(args.checkpoint).resolve()
    denoiser = TinyUNetDenoiser(use_attention=cfg.use_attention).to(device)
    meta = load_checkpoint(ckpt_path, denoiser, restore_rng=False)
    if args.use_ema:
        ema_path = ckpt_path / "ema.pt"
        if ema_path.exists():
            denoiser.load_state_dict(torch.load(ema_path, map_location=device, weights_only=False))
            ok("using EMA weights")
    denoiser.eval()
    field = DPCField(denoiser, cfg).to(device)

    # ── Load eval cache ──
    stage("Load eval cache")
    cache = TensorCache(Path(args.apricot_eval_cache).resolve())
    info(f"cache: {len(cache)} images at {cache.resolution}")

    # ── Pick images: prefer images with bboxes, sample randomly ──
    rs = np.random.RandomState(args.seed)
    candidates = [i for i in range(len(cache))
                  if cache[i]["metadata"].get("bboxes_xyxy")]
    if not candidates:
        print(f"  {C.R}✗{C.END} no images with bboxes in cache")
        sys.exit(1)
    n_pick = min(args.n_vis, len(candidates))
    picked = rs.choice(candidates, size=n_pick, replace=False).tolist()
    info(f"picked {n_pick} images for visualization")

    # ── First pass: compute raw residuals AND get global min/max for the GLOBAL panel ──
    stage("Compute residuals on selected images")
    gen = make_generator(args.seed, device=device)

    raw_maps = []
    n_skipped_degenerate = 0
    global_max = -float("inf")
    global_min = float("inf")
    with torch.no_grad():
        for k, idx in enumerate(picked):
            item = cache[idx]
            img = item["image"].unsqueeze(0).to(device)
            H, W = img.shape[-2:]
            bboxes = item["metadata"]["bboxes_xyxy"]
            mask = build_mask(bboxes, H, W, device)

            # Defensive: if all bboxes were clipped to nothing, the mask is all zero.
            # We still render the panels; we just can't compute inside/outside stats.
            if mask.sum().item() == 0:
                n_skipped_degenerate += 1

            raw = field.compute_raw_signal(img, generator=gen)
            hybrid = raw["hybrid"][0, 0]  # [H, W]
            stats = residual_ratio_per_image(hybrid, mask)
            stats["name"] = Path(item["path"]).name
            raw_maps.append({
                "idx": int(idx),
                "name": stats["name"],
                "img": img[0].detach().cpu(),
                "mask": mask.detach().cpu(),
                "hybrid": hybrid.detach().cpu(),
                "stats": stats,
            })
            local_max = float(hybrid.max().item())
            local_min = float(hybrid.min().item())
            global_max = max(global_max, local_max)
            global_min = min(global_min, local_min)
            if (k + 1) % 10 == 0:
                info(f"  done {k+1}/{n_pick}")

    info(f"global residual range: [{global_min:.5f}, {global_max:.5f}]")
    if n_skipped_degenerate:
        warn(f"{n_skipped_degenerate}/{n_pick} images had degenerate (zero-area) "
             f"masks — they will be rendered without ratio stats")

    # ── Render 4-panel grids ──
    stage(f"Render 4-panel grids → {output_dir / 'grids'}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # S3 correlation: inside_mean (raw absolute) vs ratio (per-image normalized)
    inside_means = np.array([
        m["stats"]["inside_mean"] for m in raw_maps
        if m["stats"].get("inside_mean") is not None
        and m["stats"].get("ratio") is not None
    ])
    ratios = np.array([
        m["stats"]["ratio"] for m in raw_maps
        if m["stats"].get("inside_mean") is not None
        and m["stats"].get("ratio") is not None
    ])
    if len(inside_means) >= 2 and len(ratios) == len(inside_means):
        try:
            corr = float(np.corrcoef(inside_means, ratios)[0, 1])
        except Exception:
            corr = None
    else:
        corr = None

    n_rendered = 0
    n_render_errors = 0
    for k, m in enumerate(raw_maps):
        try:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

            # Panel 1: input
            axes[0].imshow(m["img"].permute(1, 2, 0).numpy())
            axes[0].set_title(f"{m['name']}\nINPUT", fontsize=9)
            axes[0].axis("off")

            # Panel 2: input + mask overlay
            axes[1].imshow(m["img"].permute(1, 2, 0).numpy())
            mask_area = int(m["mask"].sum().item())
            if mask_area > 0:
                axes[1].imshow(m["mask"].numpy(), cmap="autumn", alpha=0.45)
            axes[1].set_title(f"BBOX MASK\n(area={mask_area} px)", fontsize=9)
            axes[1].axis("off")

            # Panel 3: GLOBAL-scale residual (same vmin/vmax across all images)
            im_global = axes[2].imshow(
                m["hybrid"].numpy(), cmap="hot",
                vmin=global_min, vmax=global_max,
            )
            ax = axes[2]
            for x1, y1, x2, y2 in cache[m["idx"]]["metadata"].get("bboxes_xyxy", []):
                ax.add_patch(mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor="cyan", facecolor="none",
                ))
            s = m["stats"]
            ax.set_title(
                f"RESIDUAL (GLOBAL scale)\n"
                f"in={fmt(s.get('inside_mean'))} out={fmt(s.get('outside_mean'))}",
                fontsize=9,
            )
            ax.axis("off")
            plt.colorbar(im_global, ax=ax, fraction=0.046, pad=0.04)

            # Panel 4: PER-IMAGE-scale residual
            h = m["hybrid"].numpy()
            h_norm = (h - h.min()) / max(1e-12, h.max() - h.min())
            ax = axes[3]
            im_per = ax.imshow(h_norm, cmap="hot", vmin=0, vmax=1)
            for x1, y1, x2, y2 in cache[m["idx"]]["metadata"].get("bboxes_xyxy", []):
                ax.add_patch(mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor="cyan", facecolor="none",
                ))
            ax.set_title(
                f"RESIDUAL (PER-IMAGE scale)\nratio={fmt(s.get('ratio'), '.3f')}",
                fontsize=9,
            )
            ax.axis("off")
            plt.colorbar(im_per, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            out_path = output_dir / "grids" / f"{k:03d}_{m['name']}.png"
            plt.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            n_rendered += 1
        except Exception as e:
            n_render_errors += 1
            try:
                plt.close("all")
            except Exception:
                pass
            warn(f"render failed for {m.get('name', '?')}: {e!r}")
            # Continue rendering remaining images.

    ok(f"rendered {n_rendered}/{len(raw_maps)} grids" +
       (f" ({n_render_errors} errors)" if n_render_errors else ""))

    # ── Save JSON sidecar ──
    stage("Save stats JSON")

    def _safe(v):
        # Convert tensors / numpy / floats to plain Python or None for JSON.
        if v is None:
            return None
        if isinstance(v, (int, float, str, bool)):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    out_stats = {
        "version": __version__,
        "tool": "diagnose_raw_residual",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "checkpoint_meta": meta,
        "n_visualized": len(raw_maps),
        "n_rendered": n_rendered,
        "n_render_errors": n_render_errors,
        "n_skipped_degenerate_masks": n_skipped_degenerate,
        "global_residual_min": float(global_min),
        "global_residual_max": float(global_max),
        "S3_corr_inside_mean_vs_ratio": corr,
        "S3_pass": (corr is not None and corr >= 0.5),
        "S3_threshold": 0.5,
        "per_image": [
            {
                "name": m["name"],
                "idx": m["idx"],
                "stats": {k_: _safe(v) for k_, v in m["stats"].items() if k_ != "name"},
            }
            for m in raw_maps
        ],
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(out_stats, f, indent=2, default=str)
    ok(f"stats -> {output_dir / 'stats.json'}")

    print()
    print(f"  {C.BOLD}n images visualized:{C.END} {len(raw_maps)}")
    print(f"  {C.BOLD}n grids rendered:   {C.END} {n_rendered}")
    if corr is not None:
        print(f"  {C.BOLD}corr(inside_mean, ratio):{C.END} {corr:.3f}")
    else:
        print(f"  {C.BOLD}corr(inside_mean, ratio):{C.END} (not enough data)")


if __name__ == "__main__":
    main()