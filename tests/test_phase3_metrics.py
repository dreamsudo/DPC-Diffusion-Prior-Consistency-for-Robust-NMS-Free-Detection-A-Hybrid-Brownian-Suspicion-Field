"""Test Phase 3 metric functions: aggregate_on_patch_suppression/aggregate_off_patch_retention/aggregate_per_image_margin, confusion, adversarial table."""

import math

import pytest

torch = pytest.importorskip("torch")


def _make_per_image_entry(
    name, n_b_total, n_b_on, n_d_total, n_d_on,
    b_on_classes, b_off_classes, d_on_classes, d_off_classes,
    susp_on=None, susp_off=None,
):
    if susp_on is None:
        susp_on = [0.5] * n_b_on
    if susp_off is None:
        susp_off = [0.1] * (n_b_total - n_b_on)
    on_pct = (100.0 * (n_b_on - n_d_on) / n_b_on) if n_b_on > 0 else None
    off_pct = (100.0 * ((n_b_total - n_b_on) - (n_d_total - n_d_on))
               / max(1, (n_b_total - n_b_on))) if (n_b_total - n_b_on) > 0 else None
    margin = (on_pct - off_pct) if on_pct is not None and off_pct is not None else None
    return {
        "name": name,
        "patch_box_count": 1,
        "patch_areas_frac": [0.05],
        "baseline": {
            "n_dets": n_b_total, "n_dets_on_patch": n_b_on,
            "on_patch_classes": b_on_classes, "off_patch_classes": b_off_classes,
        },
        "dpc": {
            "n_dets": n_d_total, "n_dets_on_patch": n_d_on,
            "on_patch_classes": d_on_classes, "off_patch_classes": d_off_classes,
        },
        "margin_pp": margin,
        "on_patch_suppression_pct": on_pct,
        "off_patch_suppression_pct": off_pct,
        "per_box_suspicion_on_patch": susp_on,
        "per_box_suspicion_off_patch": susp_off,
        "elapsed_ms": 35.0,
    }


def test_aggregate_on_patch_suppression_perfect_suppression():
    """Every image: all on-patch detections suppressed → mean reduction = 100%."""
    from dpc.metrics import aggregate_on_patch_suppression

    per_image = [
        _make_per_image_entry("a", 10, 4, 6, 0, [9, 9, 12, 12], [0, 1, 2, 3, 4, 5], [], [0, 1, 2, 3, 4, 5]),
        _make_per_image_entry("b", 8, 3, 5, 0, [9, 12, 9], [0, 1, 2, 3, 4], [], [0, 1, 2, 3, 4]),
    ]
    c1 = aggregate_on_patch_suppression(per_image, seed=42)
    assert c1["n_applicable_images"] == 2
    assert math.isclose(c1["mean_reduction_pp"], 100.0)
    assert c1["n_patch_overlapping_baseline"] == 7
    assert c1["n_patch_overlapping_dpc"] == 0


def test_aggregate_on_patch_suppression_no_suppression():
    """No on-patch reduction → mean 0%."""
    from dpc.metrics import aggregate_on_patch_suppression

    per_image = [
        _make_per_image_entry("a", 6, 3, 6, 3, [9, 9, 12], [0, 1, 2], [9, 9, 12], [0, 1, 2]),
    ]
    c1 = aggregate_on_patch_suppression(per_image, seed=42)
    assert c1["mean_reduction_pp"] == 0.0


def test_aggregate_on_patch_suppression_no_applicable_images():
    """If no image has on-patch baseline detections, returns None."""
    from dpc.metrics import aggregate_on_patch_suppression

    per_image = [
        _make_per_image_entry("a", 4, 0, 3, 0, [], [0, 1, 2, 3], [], [0, 1, 2]),
    ]
    c1 = aggregate_on_patch_suppression(per_image, seed=42)
    assert c1["n_applicable_images"] == 0
    assert c1["mean_reduction_pp"] is None


def test_aggregate_off_patch_retention_perfect_retention():
    """Off-patch baseline detections all kept by DPC → retention = 1.0."""
    from dpc.metrics import aggregate_off_patch_retention

    per_image = [
        _make_per_image_entry("a", 10, 4, 10, 4, [9]*4, [0]*6, [9]*4, [0]*6),
        _make_per_image_entry("b", 8, 2, 8, 2, [9]*2, [0]*6, [9]*2, [0]*6),
    ]
    c2 = aggregate_off_patch_retention(per_image, seed=42)
    assert c2["n_applicable_images"] == 2
    assert math.isclose(c2["mean_retention"], 1.0)


def test_aggregate_off_patch_retention_partial_retention():
    """Half of off-patch baseline retained → retention 0.5."""
    from dpc.metrics import aggregate_off_patch_retention

    per_image = [
        _make_per_image_entry("a", 10, 4, 7, 4, [9]*4, [0]*6, [9]*4, [0]*3),
    ]
    c2 = aggregate_off_patch_retention(per_image, seed=42)
    # baseline off = 6, dpc off = 3, retention = 3/6 = 0.5
    assert math.isclose(c2["mean_retention"], 0.5)


def test_aggregate_per_image_margin_distribution():
    from dpc.metrics import aggregate_per_image_margin, MARGIN_BINS

    per_image = [
        # margin > 20
        _make_per_image_entry("a", 10, 5, 5, 0, [9]*5, [0]*5, [], [0]*5),
        # margin between -5 and 5
        _make_per_image_entry("b", 8, 4, 8, 4, [9]*4, [0]*4, [9]*4, [0]*4),
        # margin < -20
        _make_per_image_entry("c", 10, 2, 4, 2, [9]*2, [0]*8, [9]*2, [0]*2),
    ]
    c3 = aggregate_per_image_margin(per_image)
    assert c3["n"] == 3
    assert c3["bins"][">+20pp"] == 1
    assert c3["bins"]["-5_to_+5pp"] == 1






def test_discriminability_ratio():
    from dpc.metrics import discriminability_ratio

    per_image = [
        {
            "per_box_suspicion_on_patch": [0.5, 0.6, 0.4],
            "per_box_suspicion_off_patch": [0.1, 0.05],
        },
        {
            "per_box_suspicion_on_patch": [0.55],
            "per_box_suspicion_off_patch": [0.08, 0.12],
        },
    ]
    dr = discriminability_ratio(per_image)
    # on mean = (0.5+0.6+0.4+0.55)/4 = 0.5125
    # off mean = (0.1+0.05+0.08+0.12)/4 = 0.0875
    expected = 0.5125 / 0.0875
    assert math.isclose(dr, expected, abs_tol=1e-3)


def test_discriminability_ratio_empty():
    from dpc.metrics import discriminability_ratio

    per_image = [
        {"per_box_suspicion_on_patch": [], "per_box_suspicion_off_patch": []},
    ]
    dr = discriminability_ratio(per_image)
    assert dr is None


def test_discriminability_ratio_zero_off():
    from dpc.metrics import discriminability_ratio

    per_image = [
        {"per_box_suspicion_on_patch": [0.5], "per_box_suspicion_off_patch": [0.0]},
    ]
    dr = discriminability_ratio(per_image)
    assert dr is None  # off mean is 0 → can't divide


def test_per_image_detection_metrics_margin_calc():
    from dpc.metrics import per_image_detection_metrics

    p = per_image_detection_metrics(
        name="test.jpg",
        patch_boxes=[[0, 0, 100, 100]],
        img_area=640.0 * 640.0,
        baseline_overlap_classes=[9, 9, 12, 12],     # 4 on-patch
        baseline_off_patch_classes=[0, 1, 2, 3, 4],  # 5 off-patch
        dpc_overlap_classes=[9],                      # 1 on-patch (suppressed 3/4 = 75%)
        dpc_off_patch_classes=[0, 1, 2, 3, 4],       # 5 off-patch (suppressed 0/5 = 0%)
        per_box_suspicion_overlap=[0.5, 0.55, 0.6, 0.45],
        per_box_suspicion_off_patch=[0.1, 0.08, 0.12, 0.05, 0.07],
        elapsed_ms=35.0,
    )
    # margin = 75% - 0% = 75pp
    assert math.isclose(p["margin_pp"], 75.0, abs_tol=1e-3)


def test_mean_with_ci():
    from dpc.metrics import mean_with_ci

    out = mean_with_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
    assert out["n"] == 5
    assert math.isclose(out["mean"], 3.0)
    assert out["ci95"][0] is not None and out["ci95"][1] is not None
    assert out["ci95"][0] <= 3.0 <= out["ci95"][1]


def test_mean_with_ci_empty():
    from dpc.metrics import mean_with_ci

    out = mean_with_ci([])
    assert out["n"] == 0
    assert out["mean"] is None
