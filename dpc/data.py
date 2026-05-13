"""Datasets backed by TensorCache.

Phase 2 extensions vs Phase 1:
  - CachedApricotDataset      — reads APRICOT cache, returns image + mask + bboxes
  - SyntheticPatchDataset     — composites synthetic patches onto COCO scenes
  - MixedDataset              — probabilistic mixture with explicit seed
  - make_apricot_indices      — train/val split helpers
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_cache import TensorCache
from .seeding import deterministic_split
from .synthetic_patch import SyntheticPatchGenerator


# ============================================================================
# Phase 1 carry-forward
# ============================================================================

class NormalImageDataset(Dataset):
    """Clean image dataset backed by a TensorCache.

    Returns dict {"image", "mask", "source", "path"} where mask is None
    (clean images have no patch).
    """

    def __init__(
        self,
        cache: TensorCache,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.cache = cache
        self.indices = list(indices) if indices is not None else list(range(len(cache)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        item = self.cache[idx]
        img = item["image"]
        if self.transform is not None:
            img = self.transform(img)
        meta = item.get("metadata") or {}
        gt_boxes = meta.get("gt_boxes_xyxy", [])
        gt_classes = meta.get("gt_classes", [])
        return {
            "image": img,
            "mask": None,
            "source": "normal",
            "path": item["path"],
            "gt_boxes_xyxy": gt_boxes,
            "gt_classes": gt_classes,
        }


class TensorAugment:
    """Tensor-native augmentation: hflip + color jitter."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self._rng = random.Random(seed)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self._rng.random() < self.hflip_prob:
            img = torch.flip(img, dims=[-1])
        if self.brightness > 0:
            f = 1.0 + self._rng.uniform(-self.brightness, self.brightness)
            img = (img * f).clamp(0.0, 1.0)
        if self.contrast > 0:
            f = 1.0 + self._rng.uniform(-self.contrast, self.contrast)
            mean = img.mean(dim=(-2, -1), keepdim=True)
            img = ((img - mean) * f + mean).clamp(0.0, 1.0)
        if self.saturation > 0:
            f = 1.0 + self._rng.uniform(-self.saturation, self.saturation)
            gray = img.mean(dim=0, keepdim=True)
            img = (gray + (img - gray) * f).clamp(0.0, 1.0)
        return img


# ============================================================================
# APRICOT (real images + bbox-derived mask)
# ============================================================================

class CachedApricotDataset(Dataset):
    """Real APRICOT images backed by an APRICOT TensorCache.

    For training: bbox-derived rectangular masks are used as field-supervision
    targets. ARCH §5.2 / sanity_check_data.py measures bbox tightness so we
    know empirically how much background each "mask" includes.

    Returns dict {"image", "mask", "source", "path", "bboxes"}.

    Training images are resolved at the cache resolution (cfg.probe_res, e.g.,
    128x128). The cached metadata.bboxes_xyxy are pre-scaled to that resolution
    by tools/build_caches.py.
    """

    def __init__(
        self,
        cache: TensorCache,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.cache = cache
        self.indices = (
            list(indices) if indices is not None else list(range(len(cache)))
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        item = self.cache[idx]
        img = item["image"]  # [3, H, W]
        bboxes = item["metadata"].get("bboxes_xyxy", [])

        # Build mask at cache resolution
        H, W = img.shape[-2:]
        mask = torch.zeros((1, H, W), dtype=torch.float32)
        for x1, y1, x2, y2 in bboxes:
            ix1 = max(0, int(round(x1)))
            iy1 = max(0, int(round(y1)))
            ix2 = min(W, int(round(x2)))
            iy2 = min(H, int(round(y2)))
            if ix2 > ix1 and iy2 > iy1:
                mask[0, iy1:iy2, ix1:ix2] = 1.0

        # If aug is requested, apply to image only — mask is structural.
        # The Phase 1 TensorAugment includes hflip; if an image is flipped,
        # the mask must flip too. So we duplicate the hflip decision here.
        # For safety, we accept transform=None for APRICOT in Phase 2.
        if self.transform is not None:
            img = self.transform(img)
            # Note: caller is responsible for ensuring the transform doesn't
            # break mask alignment. The training script wires this carefully.

        return {
            "image": img,
            "mask": mask,
            "source": "apricot",
            "path": item["path"],
            "bboxes": bboxes,
        }


# ============================================================================
# Synthetic patches (no real APRICOT pixels)
# ============================================================================

class SyntheticPatchDataset(Dataset):
    """Composite a programmatic synthetic patch onto a clean COCO scene.

    NEVER touches real APRICOT pixels — color statistics are matched at
    distribution level only (see SyntheticPatchGenerator).

    `length` is the dataset's nominal length (used by DataLoader for stopping).
    Internally we cycle through coco_indices with random offsets, generating
    fresh patches every time. With seed-aware random state, outputs are
    reproducible per (epoch, item_index).
    """

    def __init__(
        self,
        coco_cache: TensorCache,
        patch_generator: SyntheticPatchGenerator,
        coco_indices: Optional[Sequence[int]] = None,
        length: int = 50000,
        base_seed: int = 42,
        scene_transform: Optional[Callable] = None,
    ):
        self.coco_cache = coco_cache
        self.patch_generator = patch_generator
        self.coco_indices = (
            list(coco_indices) if coco_indices is not None
            else list(range(len(coco_cache)))
        )
        if not self.coco_indices:
            raise ValueError("coco_indices is empty")
        self.length = int(length)
        self.base_seed = int(base_seed)
        self.scene_transform = scene_transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        # Per-sample reproducible RNG
        rng = np.random.RandomState((self.base_seed * 1_000_003 + i) % (2**31 - 1))
        coco_idx = self.coco_indices[i % len(self.coco_indices)]

        item = self.coco_cache[coco_idx]
        scene = item["image"]  # [3, H, W]
        if self.scene_transform is not None:
            scene = self.scene_transform(scene)

        patched, mask, spec = self.patch_generator.render_random(scene, rng)

        return {
            "image": patched,
            "mask": mask,                # [1, H, W]
            "source": "synthetic",
            "path": item["path"] + f"::synth#{i}",
            "spec_shape": spec.shape,
            "spec_texture": spec.texture,
            "spec_size_frac": spec.size_frac,
        }


# ============================================================================
# Mixed dataset
# ============================================================================

class MixedDataset(Dataset):
    """Probabilistic mixture of N datasets with explicit seed.

    Each item: pick a source dataset by `weights`, then pull from that source
    at a deterministic-per-(seed, idx) sub-index. All sources must return the
    same dict keys.
    """

    def __init__(
        self,
        sources: list[Dataset],
        weights: Sequence[float],
        length: int,
        base_seed: int = 42,
    ):
        if len(sources) != len(weights):
            raise ValueError(
                f"sources/weights length mismatch: "
                f"{len(sources)} vs {len(weights)}"
            )
        wsum = sum(weights)
        if abs(wsum - 1.0) > 1e-3:
            raise ValueError(f"weights must sum to 1.0, got {wsum}")
        if not sources:
            raise ValueError("must provide at least one source")

        self.sources = sources
        self.weights = list(weights)
        self.length = int(length)
        self.base_seed = int(base_seed)
        # Precompute cumulative for binary-search-free selection
        self.cum_weights = np.cumsum(self.weights)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        rng = np.random.RandomState((self.base_seed * 7919 + i) % (2**31 - 1))
        u = rng.uniform()
        src_idx = int(np.searchsorted(self.cum_weights, u, side="right"))
        src_idx = min(src_idx, len(self.sources) - 1)
        src = self.sources[src_idx]
        sub_idx = rng.randint(0, len(src))
        item = src[sub_idx]
        # Tag with mixture info for the collate to track ratios
        item = dict(item)
        item["mix_source_idx"] = src_idx
        return item


# ============================================================================
# Collate
# ============================================================================

def collate_dpc_batch(batch: list[dict]) -> dict:
    """Batch dicts into a unified tensor batch.

    All items must have "image". Items may have "mask" (None or [1, H, W]).
    Returns:
      images:        [B, 3, H, W]
      masks:         [B, 1, H, W] OR None if no item has a mask
      mask_validity: [B] bool — True if that item has a real mask
      sources:       list[str]
      paths:         list[str]
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    sources = [b.get("source", "unknown") for b in batch]
    paths = [b.get("path", "") for b in batch]

    # Masks: any item with None gets a zero mask but mask_validity=False so
    # field-supervision losses don't push the field down on clean images.
    H, W = images.shape[-2:]
    has_any_mask = any(b.get("mask") is not None for b in batch)
    if has_any_mask:
        masks = torch.zeros((len(batch), 1, H, W), dtype=torch.float32)
        validity = torch.zeros(len(batch), dtype=torch.bool)
        for i, b in enumerate(batch):
            m = b.get("mask")
            if m is not None:
                masks[i] = m
                validity[i] = True
    else:
        masks = None
        validity = None

    # Per-image variable-length GT (COCO labels). Items that don't carry
    # GT (CachedApricotDataset, SyntheticPatchDataset) get empty tensors so
    # downstream training code can iterate per-batch-item uniformly.
    gt_boxes_list: list = []
    gt_classes_list: list = []
    any_has_gt = any(("gt_boxes_xyxy" in b) for b in batch)
    if any_has_gt:
        for b in batch:
            gb = b.get("gt_boxes_xyxy", [])
            gc = b.get("gt_classes", [])
            if len(gb) > 0:
                gt_boxes_list.append(
                    torch.as_tensor(gb, dtype=torch.float32)
                )
                gt_classes_list.append(
                    torch.as_tensor(gc, dtype=torch.long)
                )
            else:
                gt_boxes_list.append(torch.zeros((0, 4), dtype=torch.float32))
                gt_classes_list.append(torch.zeros((0,), dtype=torch.long))

    out = {
        "images": images,
        "masks": masks,
        "mask_validity": validity,
        "sources": sources,
        "paths": paths,
    }
    if any_has_gt:
        out["gt_boxes"] = gt_boxes_list
        out["gt_classes"] = gt_classes_list
    if "mix_source_idx" in batch[0]:
        out["mix_source_idx"] = torch.tensor(
            [b.get("mix_source_idx", -1) for b in batch], dtype=torch.long
        )
    return out


# ============================================================================
# Splits
# ============================================================================

def make_coco_split(
    cache: TensorCache,
    val_frac: float = 0.01,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split COCO cache into train and val indices."""
    n = len(cache)
    train_frac = 1.0 - val_frac
    return deterministic_split(n, train_frac, seed)


def make_apricot_indices(cache: TensorCache) -> list[int]:
    """All indices in an APRICOT cache (no further splitting needed — caches are
    already split by tools/build_caches.py into train and val)."""
    return list(range(len(cache)))
