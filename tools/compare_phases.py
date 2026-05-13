#!/usr/bin/env python3
"""Compare Phase 1 vs Phase 2 residual diagnostics on the same 873-image set.

Reads two diagnostic_873/summary.json files and produces:
  - JSON head-to-head table (deltas at each ratio bin, median, etc.)
  - PNG paired histogram
  - S2 verdict: does Phase 2 improve on Phase 1?
  Phase 2 strong-signal fraction (≥5x) ≥ 2× Phase 1 strong-signal fraction
  AND
  Phase 2 positive count ≥ Phase 1 positive count + 30
  AND
  KS test p < 0.05

Usage:
    python tools/compare_phases.py \
        --phase1-summary runs/phase1_seed42/diagnostic_873/summary.json \
        --phase2-summary runs/phase2_seed42/diagnostic_873/summary.json \
        --phase1-per-image runs/phase1_seed42/diagnostic_873/per_image.json \
        --phase2-per-image runs/phase2_seed42/diagnostic_873/per_image.json \
        --output runs/phase2_seed42/comparison_to_phase1.json
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

from dpc._version import __version__
from dpc.manifest import fingerprint_environment


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}── {m} ──{C.END}")


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Try scipy first; fall back to manual implementation."""
    try:
        from scipy import stats
        res = stats.ks_2samp(a, b)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # Manual two-sample KS — simple but adequate for binary verdict
        ab = np.concatenate([a, b])
        ab.sort()
        cdf_a = np.searchsorted(np.sort(a), ab, side="right") / len(a)
        cdf_b = np.searchsorted(np.sort(b), ab, side="right") / len(b)
        D = float(np.abs(cdf_a - cdf_b).max())
        # Asymptotic p-value
        n_eff = (len(a) * len(b)) / (len(a) + len(b))
        p = 2 * np.exp(-2 * n_eff * D ** 2)
        return D, float(min(1.0, max(0.0, p)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-summary", required=True)
    p.add_argument("--phase2-summary", required=True)
    p.add_argument("--phase1-per-image", default=None)
    p.add_argument("--phase2-per-image", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"{C.BOLD}Phase 1 vs Phase 2 Comparison v{__version__}{C.END}")

    with open(args.phase1_summary) as f:
        s1 = json.load(f)
    with open(args.phase2_summary) as f:
        s2 = json.load(f)

    a1 = s1["aggregate"]
    a2 = s2["aggregate"]

    print()
    print(f"  {C.BOLD}{'Metric':<28} {'Phase 1':>12} {'Phase 2':>12} {'Δ':>12}{C.END}")
    print(f"  {'-' * 64}")
    def row(label, k, fmt="{:>12.3f}"):
        v1 = a1.get(k); v2 = a2.get(k)
        if v1 is None or v2 is None:
            d = "—"
        else:
            d = fmt.format(v2 - v1)
        v1s = fmt.format(v1) if v1 is not None else "—"
        v2s = fmt.format(v2) if v2 is not None else "—"
        print(f"  {label:<28} {v1s:>12} {v2s:>12} {d:>12}")
    row("median_ratio", "median_ratio")
    row("mean_ratio", "mean_ratio")
    row("n_positive", "n_positive", fmt="{:>12d}")
    row("n_valid", "n_valid", fmt="{:>12d}")
    print()
    print(f"  {C.BOLD}Bin counts{C.END}")
    bins = list(a1.get("bins", {}).keys())
    for bn in bins:
        v1 = a1["bins"].get(bn, 0)
        v2 = a2["bins"].get(bn, 0)
        d = v2 - v1
        print(f"  {bn:<28} {v1:>12d} {v2:>12d} {d:>+12d}")

    # ── KS test if per-image data available ──
    ks_stat = ks_p = None
    if args.phase1_per_image and args.phase2_per_image:
        with open(args.phase1_per_image) as f:
            pi1 = json.load(f)
        with open(args.phase2_per_image) as f:
            pi2 = json.load(f)
        ratios1 = np.array(
            [s["ratio"] for s in pi1 if s.get("ratio") is not None],
            dtype=np.float64,
        )
        ratios2 = np.array(
            [s["ratio"] for s in pi2 if s.get("ratio") is not None],
            dtype=np.float64,
        )
        if len(ratios1) > 5 and len(ratios2) > 5:
            ks_stat, ks_p = ks_two_sample(ratios1, ratios2)
            print()
            print(f"  {C.BOLD}KS two-sample test{C.END}")
            print(f"    statistic: {ks_stat:.4f}")
            print(f"    p-value:   {ks_p:.4g}")
    p1_strong = a1["bins"].get("greater_than_5x", 0)
    p2_strong = a2["bins"].get("greater_than_5x", 0)
    p1_pos = a1.get("n_positive", 0)
    p2_pos = a2.get("n_positive", 0)

    cond_strong = (p2_strong >= 2 * max(1, p1_strong))
    cond_positive = (p2_pos >= p1_pos + 30)
    cond_ks = (ks_p is not None and ks_p < 0.05)

    print(f"  Strong-signal (≥5x):  P1={p1_strong}  P2={p2_strong}  "
          f"P2 ≥ 2*P1: {C.G if cond_strong else C.R}{cond_strong}{C.END}")
    print(f"  Positive count:        P1={p1_pos}  P2={p2_pos}  "
          f"P2 ≥ P1+30: {C.G if cond_positive else C.R}{cond_positive}{C.END}")
    if ks_p is not None:
        print(f"  KS test p < 0.05:     {C.G if cond_ks else C.R}{cond_ks}{C.END}  "
              f"(p={ks_p:.4g})")
    else:
        print(f"  KS test:               (skipped — no per-image data)")

    print()
    # ── Write JSON ──
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "version": __version__,
        "tool": "compare_phases",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "phase1_summary": s1.get("aggregate"),
        "phase2_summary": s2.get("aggregate"),
        "phase1_path": str(args.phase1_summary),
        "phase2_path": str(args.phase2_summary),
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,

        "S2_conditions": {
            "strong_signal_doubled": cond_strong,
            "positive_count_increased": cond_positive,
            "ks_significant": cond_ks if ks_p is not None else None,
        },
    }
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  comparison -> {out}")

    sys.exit(0)


if __name__ == "__main__":
    main()
