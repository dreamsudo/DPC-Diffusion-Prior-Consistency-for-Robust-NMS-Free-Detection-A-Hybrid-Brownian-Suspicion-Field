"""Metrics for the DPC pipeline.

Single source of truth for every quantitative measurement produced by
training, diagnostic, and evaluation tools. All functions return
JSON-serializable dicts.

Two metric families:

    residual_*   : Phase 1 / Phase 2 diagnostics. Compares the magnitude
                   of the denoiser residual inside vs. outside the patch
                   region (or any mask).

    detection_*  : Phase 3 evaluation. Compares baseline YOLO26 detections
                   to DPC-wrapped YOLO26 detections on the same images.

Detection metrics (formerly C1/C2/C3 in earlier releases):

    on_patch_suppression  - fraction of adversarial on-patch detections
                            killed by DPC
    off_patch_retention   - fraction of true off-patch detections preserved
                            by DPC
    per_image_margin      - per-image suppression minus retention loss
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ============================================================================
# Residual ratio (denoiser diagnostics)
# ============================================================================

def residual_ratio_per_image(
    residual_map: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Per-image residual ratio: mean(|residual| inside mask) / mean(|residual| outside mask).

    Returns a dict with keys:
        inside_mean, outside_mean, ratio (or None if degenerate),
        n_inside, n_outside, degenerate (bool).
    """
    if residual_map.dim() == 3:
        residual_map = residual_map.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    if residual_map.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: residual {residual_map.shape} vs mask {mask.shape}"
        )

    r = residual_map.detach().cpu()
    m = mask.detach().cpu().bool()

    n_inside = int(m.sum().item())
    n_outside = int((~m).sum().item())
    if n_inside == 0 or n_outside == 0:
        return {
            "inside_mean": None, "outside_mean": None, "ratio": None,
            "n_inside": n_inside, "n_outside": n_outside, "is_degenerate": True,
        }

    abs_r = r.abs()
    inside_mean = float(abs_r[m].mean().item())
    outside_mean = float(abs_r[~m].mean().item())
    ratio = None if outside_mean == 0.0 else inside_mean / outside_mean
    return {
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "ratio": ratio,
        "n_inside": n_inside,
        "n_outside": n_outside,
        "is_degenerate": False,
    }


RATIO_BINS = [
    ("greater_than_5x", lambda r: r >= 5.0),
    ("3_to_5x",         lambda r: 3.0 <= r < 5.0),
    ("2_to_3x",         lambda r: 2.0 <= r < 3.0),
    ("1.5_to_2x",       lambda r: 1.5 <= r < 2.0),
    ("1_to_1.5x",       lambda r: 1.0 <= r < 1.5),
    ("less_than_1x",    lambda r: r < 1.0),
]


def aggregate_residual_distribution(per_image_stats: list[dict]) -> dict:
    """Aggregate per-image residual ratios into summary statistics."""
    valid = [s for s in per_image_stats if not s.get("is_degenerate", False)
             and s.get("ratio") is not None]
    n_total = len(per_image_stats)
    n_valid = len(valid)
    n_degenerate = n_total - n_valid

    if n_valid == 0:
        return {
            "n_total": n_total, "n_valid": 0, "n_degenerate": n_degenerate,
            "n_positive": 0, "median_ratio": None, "mean_ratio": None,
            "std_ratio": None, "min_ratio": None, "max_ratio": None,
            "bins": {name: 0 for name, _ in RATIO_BINS},
        }

    ratios = np.array([s["ratio"] for s in valid], dtype=np.float64)
    n_positive = int((ratios > 1.0).sum())
    bins = {name: int(sum(1 for r in ratios if pred(r))) for name, pred in RATIO_BINS}
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_degenerate": n_degenerate,
        "n_positive": n_positive,
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(ratios.mean()),
        "std_ratio": float(ratios.std(ddof=1) if n_valid > 1 else 0.0),
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
        "bins": bins,
    }


def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic on a 1-D array."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[i] = stat_fn(values[idx])
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def probe_randomness_delta(per_image_run_a, per_image_run_b) -> dict:
    """Compare two diagnostic runs that differ only in probe seed.

    Returns the distribution of per-image ratio deltas.
    """
    if len(per_image_run_a) != len(per_image_run_b):
        raise ValueError("runs have different lengths")

    deltas = []
    for a, b in zip(per_image_run_a, per_image_run_b):
        if (a.get("is_degenerate", False) or b.get("is_degenerate", False)
                or a.get("ratio") is None or b.get("ratio") is None):
            continue
        deltas.append(a["ratio"] - b["ratio"])

    if not deltas:
        return {
            "n_pairs": 0,
            "median_delta": None,
            "max_abs_delta": None,
            "mean_abs_delta": None,
        }
    arr = np.array(deltas, dtype=np.float64)
    return {
        "n_pairs": len(deltas),
        "median_delta": float(np.median(arr)),
        "max_abs_delta": float(np.abs(arr).max()),
        "mean_abs_delta": float(np.abs(arr).mean()),
    }


# ============================================================================
# Detection metrics (eval)
# ============================================================================

MARGIN_BINS = [
    (">+20pp",      lambda m: m > 20.0),
    ("+5_to_+20pp", lambda m: 5.0 < m <= 20.0),
    ("-5_to_+5pp",  lambda m: -5.0 <= m <= 5.0),
    ("-20_to_-5pp", lambda m: -20.0 <= m < -5.0),
    ("<-20pp",      lambda m: m < -20.0),
]


def per_image_detection_metrics(
    name: str,
    patch_boxes: list,
    img_area: float,
    baseline_overlap_classes: list[int],
    baseline_off_patch_classes: list[int],
    dpc_overlap_classes: list[int],
    dpc_off_patch_classes: list[int],
    per_box_suspicion_overlap: list[float],
    per_box_suspicion_off_patch: list[float],
    elapsed_ms: float,
) -> dict:
    """Assemble a per-image record with all detection-side numbers."""
    n_b_on = len(baseline_overlap_classes)
    n_b_off = len(baseline_off_patch_classes)
    n_d_on = len(dpc_overlap_classes)
    n_d_off = len(dpc_off_patch_classes)

    on_patch_suppression_pct = (
        100.0 * (n_b_on - n_d_on) / n_b_on if n_b_on > 0 else None
    )
    off_patch_suppression_pct = (
        100.0 * (n_b_off - n_d_off) / n_b_off if n_b_off > 0 else None
    )
    margin_pp = None
    if on_patch_suppression_pct is not None and off_patch_suppression_pct is not None:
        margin_pp = on_patch_suppression_pct - off_patch_suppression_pct

    patch_areas_frac = []
    for box in patch_boxes:
        x1, y1, x2, y2 = box
        area = max(0.0, (x2 - x1) * (y2 - y1))
        patch_areas_frac.append(area / img_area if img_area > 0 else 0.0)

    return {
        "name": name,
        "patch_box_count": len(patch_boxes),
        "patch_areas_frac": patch_areas_frac,
        "baseline": {
            "n_dets": n_b_on + n_b_off,
            "n_dets_on_patch": n_b_on,
            "on_patch_classes": list(baseline_overlap_classes),
            "off_patch_classes": list(baseline_off_patch_classes),
        },
        "dpc": {
            "n_dets": n_d_on + n_d_off,
            "n_dets_on_patch": n_d_on,
            "on_patch_classes": list(dpc_overlap_classes),
            "off_patch_classes": list(dpc_off_patch_classes),
        },
        "margin_pp": margin_pp,
        "on_patch_suppression_pct": on_patch_suppression_pct,
        "off_patch_suppression_pct": off_patch_suppression_pct,
        "per_box_suspicion_on_patch": list(per_box_suspicion_overlap),
        "per_box_suspicion_off_patch": list(per_box_suspicion_off_patch),
        "elapsed_ms": float(elapsed_ms),
    }


def aggregate_on_patch_suppression(per_image: list[dict], seed: int = 42) -> dict:
    """How many adversarial on-patch detections does DPC suppress?

    Returns counts (baseline vs DPC) and a per-image suppression-percentage
    distribution with a bootstrap CI of the mean.
    """
    n_b_total = sum(p["baseline"]["n_dets_on_patch"] for p in per_image)
    n_d_total = sum(p["dpc"]["n_dets_on_patch"] for p in per_image)

    per_image_red = []
    for p in per_image:
        n_b = p["baseline"]["n_dets_on_patch"]
        n_d = p["dpc"]["n_dets_on_patch"]
        if n_b > 0:
            per_image_red.append(100.0 * (n_b - n_d) / n_b)

    n_applicable = len(per_image_red)
    if n_applicable == 0:
        return {
            "n_patch_overlapping_baseline": int(n_b_total),
            "n_patch_overlapping_dpc": int(n_d_total),
            "n_applicable_images": 0,
            "mean_reduction_pp": None,
            "ci95": [None, None],
        }
    arr = np.array(per_image_red, dtype=np.float64)
    ci_lo, ci_hi = bootstrap_ci(arr, np.mean, n_boot=1000, seed=seed)
    return {
        "n_patch_overlapping_baseline": int(n_b_total),
        "n_patch_overlapping_dpc": int(n_d_total),
        "n_applicable_images": n_applicable,
        "mean_reduction_pp": float(arr.mean()),
        "ci95": [float(ci_lo), float(ci_hi)],
    }


def aggregate_off_patch_retention(per_image: list[dict], seed: int = 42) -> dict:
    """How many off-patch (true) detections does DPC retain?

    Uses baseline off-patch detections as the scene-context reference: for
    each image, retention = min(1, dpc_off_count / baseline_off_count).
    """
    retentions = []
    for p in per_image:
        n_b_off = p["baseline"]["n_dets"] - p["baseline"]["n_dets_on_patch"]
        n_d_off = p["dpc"]["n_dets"] - p["dpc"]["n_dets_on_patch"]
        if n_b_off > 0:
            retentions.append(min(1.0, n_d_off / n_b_off))

    n_applicable = len(retentions)
    if n_applicable == 0:
        return {
            "n_applicable_images": 0,
            "mean_retention": None,
            "ci95": [None, None],
        }
    arr = np.array(retentions, dtype=np.float64)
    ci_lo, ci_hi = bootstrap_ci(arr, np.mean, n_boot=1000, seed=seed)
    return {
        "n_applicable_images": n_applicable,
        "mean_retention": float(arr.mean()),
        "ci95": [float(ci_lo), float(ci_hi)],
    }


def aggregate_per_image_margin(per_image: list[dict]) -> dict:
    """Per-image margin = on_patch_suppression_pct - off_patch_suppression_pct.

    Returns the distribution (median, mean, std, min/max, bucket counts).
    """
    margins = [p["margin_pp"] for p in per_image if p.get("margin_pp") is not None]
    n = len(margins)
    if n == 0:
        return {
            "n": 0, "median": None, "mean": None, "std": None,
            "bins": {name: 0 for name, _ in MARGIN_BINS},
        }
    arr = np.array(margins, dtype=np.float64)
    bins = {name: int(sum(1 for m in arr if pred(m))) for name, pred in MARGIN_BINS}
    return {
        "n": n,
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if n > 1 else 0.0),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "bins": bins,
    }


def confusion_matrix(per_image: list[dict], n_classes: int = 80) -> dict:
    """Coarse on-patch class confusion: counts of baseline class -> DPC class.

    The extra column (index n_classes) means the detection was suppressed.
    """
    matrix = np.zeros((n_classes, n_classes + 1), dtype=np.int64)
    diag = 0
    off_diag = 0
    suppressed = 0
    from collections import Counter
    for p in per_image:
        bc = Counter(p["baseline"]["on_patch_classes"])
        dc = Counter(p["dpc"]["on_patch_classes"])
        for cls, n_b in bc.items():
            n_d = dc.get(cls, 0)
            n_killed = max(0, n_b - n_d)
            matrix[cls, n_classes] += n_killed
            suppressed += n_killed
            n_kept = min(n_b, n_d)
            matrix[cls, cls] += n_kept
            diag += n_kept
        for cls, n_d in dc.items():
            if cls not in bc:
                off_diag += n_d
    total = diag + off_diag + suppressed
    return {
        "diagonal_total": int(diag),
        "off_diagonal_total": int(off_diag),
        "suppressed_total": int(suppressed),
        "grand_total": int(total),
    }


def adversarial_class_table(per_image: list[dict], n_classes: int = 80) -> list[dict]:
    """Per-class on-patch detection table (baseline count vs DPC count)."""
    from collections import Counter
    b_counter: Counter = Counter()
    d_counter: Counter = Counter()
    for p in per_image:
        b_counter.update(p["baseline"]["on_patch_classes"])
        d_counter.update(p["dpc"]["on_patch_classes"])
    out = []
    for cls in range(n_classes):
        if b_counter[cls] == 0 and d_counter[cls] == 0:
            continue
        out.append({
            "class_id": int(cls),
            "n_baseline_on_patch": int(b_counter[cls]),
            "n_dpc_on_patch": int(d_counter[cls]),
        })
    out.sort(key=lambda r: r["n_baseline_on_patch"], reverse=True)
    return out


def discriminability_ratio(per_image: list[dict]) -> Optional[float]:
    """Mean per-box suspicion on-patch / mean off-patch (across all images).

    Higher means DPC's suspicion score separates on-patch from off-patch
    boxes more cleanly.
    """
    susp_on: list[float] = []
    susp_off: list[float] = []
    for p in per_image:
        susp_on.extend(p.get("per_box_suspicion_on_patch", []))
        susp_off.extend(p.get("per_box_suspicion_off_patch", []))
    if not susp_on or not susp_off:
        return None
    mean_off = float(np.mean(susp_off))
    if mean_off == 0.0:
        return None
    return float(np.mean(susp_on)) / mean_off


def mean_with_ci(values: list[float], seed: int = 42) -> dict:
    """Convenience wrapper: mean of a 1-D list with a bootstrap 95% CI."""
    if not values:
        return {"mean": None, "ci95": [None, None], "n": 0}
    arr = np.array(values, dtype=np.float64)
    lo, hi = bootstrap_ci(arr, np.mean, n_boot=1000, seed=seed)
    return {
        "mean": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "n": int(len(values)),
    }
