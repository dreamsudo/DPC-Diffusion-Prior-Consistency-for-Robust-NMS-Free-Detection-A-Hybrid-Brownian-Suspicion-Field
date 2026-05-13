"""In-memory tensor cache for COCO and APRICOT.

Format on disk: a single .pt file containing a dict with tensors and metadata.
For COCO: just images. For APRICOT: images + per-image bbox lists.

Memory budget at probe_res=128:
  COCO train2017: ~23 GB (118,287 images x 3 x 128 x 128 x float32)
  COCO val2017:   ~1 GB
  APRICOT train:  ~155 MB (786 images)
  APRICOT val:    ~17 MB (87 images)
  APRICOT eval @ 640: ~4.3 GB (873 images at 640x640)

All fit comfortably in 64 GB unified memory on M1 Max.
"""

from __future__ import annotations

import json
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from PIL import Image

from ._version import __version__


# ============================================================================
# Worker function for parallel JPEG decode
# ============================================================================

def _decode_one(args: tuple) -> tuple[int, np.ndarray, str]:
    """Worker: decode a single image to fixed resolution.

    Top-level (not a closure) so it pickles for multiprocessing.
    """
    idx, path, resolution = args
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        # Return a sentinel; caller handles
        return idx, None, str(e)
    img = img.resize(resolution, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    return idx, arr, str(path)


# ============================================================================
# Public API
# ============================================================================

class TensorCache:
    """In-memory cache of decoded image tensors at fixed resolution.

    Public API:
      cache = TensorCache(path)             # load from disk
      cache.images                          # [N, 3, H, W] tensor
      cache.paths                           # list[str]
      cache.metadata                        # list[dict] (per-image metadata)
      cache[i]                              # dict{"image", "path", "metadata"}
      len(cache)                            # N
      cache.verify()                        # SHA256 check
    """

    def __init__(
        self,
        cache_path: Path,
        verify_sha256: bool = True,
    ):
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        self.cache_path = cache_path

        if verify_sha256:
            ok = self._verify_sha256()
            if not ok:
                raise RuntimeError(
                    f"Cache SHA256 mismatch at {cache_path}. "
                    f"Rebuild caches via tools/build_caches.py."
                )

        # weights_only=False: cache contents include numpy arrays in metadata
        # (e.g. bboxes), and we trust our own caches.
        data = torch.load(cache_path, map_location="cpu", weights_only=False)

        self.version = data["version"]
        self.resolution = tuple(data["resolution"])
        self.images = data["images"]  # tensor [N, 3, H, W]
        self.paths = data["paths"]
        self.metadata = data["metadata"]  # list[dict]
        self.build_seed = data.get("build_seed")
        self.build_time_utc = data.get("build_time_utc")

        # Sanity checks at load
        if self.images.shape[0] != len(self.paths):
            raise RuntimeError(
                f"Cache integrity: {self.images.shape[0]} images vs "
                f"{len(self.paths)} paths"
            )
        if self.images.shape[2:] != self.resolution:
            raise RuntimeError(
                f"Cache integrity: tensor shape {self.images.shape[2:]} vs "
                f"declared resolution {self.resolution}"
            )
        if self.version != __version__:
            # Not fatal, but warn
            import sys
            print(
                f"  [warn] cache version {self.version} != code version {__version__}",
                file=sys.stderr,
            )

    def _verify_sha256(self) -> bool:
        """Check cache file SHA256 against manifest entry, if manifest exists."""
        manifest_path = self.cache_path.parent / "manifest.json"
        if not manifest_path.is_file():
            # No manifest yet — first build hasn't been finalized. Allow.
            return True
        with open(manifest_path) as f:
            manifest = json.load(f)
        entry = manifest.get("caches", {}).get(self.cache_path.name)
        if entry is None:
            return True
        expected = entry.get("sha256")
        if expected is None:
            return True
        actual = _sha256_file(self.cache_path)
        return actual == expected

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],         # [3, H, W] tensor
            "path": self.paths[idx],
            "metadata": self.metadata[idx] if self.metadata else {},
        }

    @classmethod
    def build_from_directory(
        cls,
        image_dir: Path,
        output_path: Path,
        resolution: tuple[int, int],
        num_workers: int = -1,
        metadata_fn: Optional[Callable[[Path], dict]] = None,
        seed: int = 42,
        progress: bool = True,
    ) -> Path:
        """Decode every image in directory to a single .pt cache file.

        Args:
          image_dir: directory containing .jpg/.png files (recursive)
          output_path: where to write the .pt cache
          resolution: (H, W) target resolution
          num_workers: -1 means os.cpu_count() - 2; 0 means single-threaded
          metadata_fn: optional callable(path) -> dict per-image metadata
          seed: for reproducibility of any random ordering (currently ordering
                is alphabetical, deterministic; seed is recorded for posterity)

        Returns the output path. Caller is responsible for writing manifest.json
        afterward.
        """
        image_dir = Path(image_dir).resolve()
        output_path = Path(output_path).resolve()
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")

        # Collect paths (sorted for determinism)
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
        if not paths:
            raise RuntimeError(f"No images found in {image_dir}")

        n = len(paths)
        if progress:
            print(f"  building cache: {n} images from {image_dir}")
            print(f"  resolution: {resolution}, workers: {num_workers}")

        # Allocate output tensor up front
        H, W = resolution
        images = torch.zeros((n, 3, H, W), dtype=torch.float32)
        paths_str = [str(p) for p in paths]
        per_image_meta: list[dict] = [{} for _ in range(n)]
        decode_errors: list[tuple[int, str]] = []

        # Decide worker count
        if num_workers < 0:
            num_workers = max(1, (os.cpu_count() or 4) - 2)

        # Parallel decode
        if num_workers > 0:
            args = [(i, p, resolution) for i, p in enumerate(paths)]
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                done = 0
                for idx, arr, path in pool.map(_decode_one, args, chunksize=64):
                    if arr is None:
                        decode_errors.append((idx, path))
                        # Use zero tensor for failures — caller can detect via paths/metadata
                    else:
                        images[idx] = torch.from_numpy(arr)
                    done += 1
                    if progress and done % 5000 == 0:
                        print(f"    decoded {done}/{n}")
        else:
            for i, p in enumerate(paths):
                idx, arr, path_str = _decode_one((i, p, resolution))
                if arr is None:
                    decode_errors.append((idx, path_str))
                else:
                    images[idx] = torch.from_numpy(arr)
                if progress and (i + 1) % 5000 == 0:
                    print(f"    decoded {i+1}/{n}")

        # Per-image metadata callback
        if metadata_fn is not None:
            if progress:
                print(f"  computing per-image metadata")
            for i, p in enumerate(paths):
                try:
                    per_image_meta[i] = metadata_fn(p)
                except Exception as e:
                    per_image_meta[i] = {"_metadata_error": str(e)}

        if decode_errors:
            print(f"  [warn] {len(decode_errors)} decode failures (first 5):")
            for idx, p in decode_errors[:5]:
                print(f"    {p}")

        payload = {
            "version": __version__,
            "resolution": list(resolution),
            "n_images": n,
            "n_decode_errors": len(decode_errors),
            "images": images,
            "paths": paths_str,
            "metadata": per_image_meta,
            "build_seed": seed,
            "build_time_utc": datetime.now(timezone.utc).isoformat(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        if progress:
            mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  wrote {output_path} ({mb:.1f} MB)")
        return output_path

    def sha256(self) -> str:
        """Compute SHA256 of the cache file."""
        return _sha256_file(self.cache_path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# APRICOT-specific helpers
# ============================================================================

def apricot_metadata_fn_factory(
    annotation_json_path: Path,
    image_root: Path,
    target_resolution: tuple[int, int],
):
    """Factory returning a metadata_fn(path) -> {bboxes, image_id, original_size}.

    bboxes are scaled from original-image coords to target_resolution coords
    so they're directly usable on the cached tensors.

    APRICOT's COCO-format JSON has:
      images[*]: {id, file_name, width, height}
      annotations[*]: {image_id, bbox: [x, y, w, h], category_id, ...}
    """
    annotation_json_path = Path(annotation_json_path)
    image_root = Path(image_root)

    with open(annotation_json_path) as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}
    file_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    H_t, W_t = target_resolution

    def fn(image_path: Path) -> dict:
        # Match on filename relative to image_root
        try:
            rel = image_path.relative_to(image_root)
        except ValueError:
            rel = image_path
        fname = rel.name
        img_id = file_to_id.get(fname)
        if img_id is None:
            return {
                "image_id": None,
                "original_size": None,
                "bboxes": [],
                "category_ids": [],
            }
        img_meta = img_lookup[img_id]
        W_o = img_meta["width"]
        H_o = img_meta["height"]
        sx = W_t / W_o
        sy = H_t / H_o

        bboxes_scaled = []
        cat_ids = []
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, x * sx)
            y1 = max(0.0, y * sy)
            x2 = min(float(W_t), (x + w) * sx)
            y2 = min(float(H_t), (y + h) * sy)
            if x2 > x1 and y2 > y1:
                bboxes_scaled.append([x1, y1, x2, y2])  # xyxy in target coords
                cat_ids.append(ann.get("category_id"))

        return {
            "image_id": img_id,
            "original_size": [W_o, H_o],
            "target_size": [W_t, H_t],
            "bboxes_xyxy": bboxes_scaled,
            "category_ids": cat_ids,
        }

    return fn


def coco_metadata_fn_factory(
    annotation_json_path: Path,
    image_root: Path,
    target_resolution: tuple[int, int],
):
    """Factory returning a metadata_fn(path) -> {gt_boxes_xyxy, gt_classes, ...}.

    Used by tools/build_caches.py to inject COCO ground-truth annotations into
    each image's metadata dict, rescaled from original-image coords to
    target_resolution coords.

    Skips:
      - crowd annotations (iscrowd=1) — not useful as training targets
      - annotations with category_id outside the 80-class YOLO26 subset
      - degenerate boxes (zero area)

    Stores dense class indices 0..79 to match YOLO26's output channel layout.

    Returned per-image dict has, in addition to image_id/original_size:
      gt_boxes_xyxy: list[list[float]]  (xyxy in target_resolution coords)
      gt_classes:    list[int]          (dense 0..79)
      n_objects:     int
      n_skipped_crowd:  int  (diagnostic)
      n_skipped_class:  int  (diagnostic)
      n_skipped_box:    int  (diagnostic)
    """
    from .coco_classes import COCO_CATEGORY_ID_TO_CLASS_ID

    annotation_json_path = Path(annotation_json_path)
    image_root = Path(image_root)
    with open(annotation_json_path) as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}
    file_to_id = {img["file_name"]: img["id"] for img in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    H_t, W_t = target_resolution

    def fn(image_path: Path) -> dict:
        try:
            rel = image_path.relative_to(image_root)
        except ValueError:
            rel = image_path
        fname = rel.name
        img_id = file_to_id.get(fname)
        if img_id is None:
            return {
                "image_id": None,
                "original_size": None,
                "target_size": [W_t, H_t],
                "gt_boxes_xyxy": [],
                "gt_classes": [],
                "n_objects": 0,
                "n_skipped_crowd": 0,
                "n_skipped_class": 0,
                "n_skipped_box": 0,
            }

        img_meta = img_lookup[img_id]
        W_o = img_meta["width"]
        H_o = img_meta["height"]
        sx = W_t / W_o
        sy = H_t / H_o

        gt_boxes_xyxy: list[list[float]] = []
        gt_classes: list[int] = []
        n_skipped_crowd = 0
        n_skipped_class = 0
        n_skipped_box = 0

        for ann in anns_by_image.get(img_id, []):
            if ann.get("iscrowd", 0):
                n_skipped_crowd += 1
                continue
            cat_id = ann.get("category_id")
            cls = COCO_CATEGORY_ID_TO_CLASS_ID.get(cat_id)
            if cls is None:
                n_skipped_class += 1
                continue
            x, y, w, h = ann["bbox"]
            x1 = max(0.0, x * sx)
            y1 = max(0.0, y * sy)
            x2 = min(float(W_t), (x + w) * sx)
            y2 = min(float(H_t), (y + h) * sy)
            if x2 <= x1 or y2 <= y1:
                n_skipped_box += 1
                continue
            gt_boxes_xyxy.append([x1, y1, x2, y2])
            gt_classes.append(int(cls))

        return {
            "image_id": img_id,
            "original_size": [W_o, H_o],
            "target_size": [W_t, H_t],
            "gt_boxes_xyxy": gt_boxes_xyxy,
            "gt_classes": gt_classes,
            "n_objects": len(gt_boxes_xyxy),
            "n_skipped_crowd": n_skipped_crowd,
            "n_skipped_class": n_skipped_class,
            "n_skipped_box": n_skipped_box,
        }
    return fn
