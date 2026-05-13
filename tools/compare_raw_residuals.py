#!/usr/bin/env python3
"""Compare Phase 1 vs Phase 2 raw residual diagnostics.

Runs `tools/diagnose_raw_residual.py` for both phases on all 873 APRICOT eval
images, then reads the resulting stats.json files and produces a side-by-side
comparison table — both for the aggregate S3 correlation and for the actual
per-image residual statistics that drive the S1 ratio metric.

Usage:
    python tools/compare_raw_residuals.py
    python tools/compare_raw_residuals.py --skip-run     # use existing outputs
    python tools/compare_raw_residuals.py --n-vis 100    # smaller sample
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, stdev


# ─────────────────────────── colors ───────────────────────────
class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    DIM = "\033[2m"; BOLD = "\033[1m"; END = "\033[0m"


def hdr(s):  print(f"\n{C.BOLD}{C.B}{'═' * 70}\n{s}\n{'═' * 70}{C.END}")
def step(s): print(f"\n{C.BOLD}── {s} ──{C.END}")
def info(s): print(f"  {C.B}•{C.END} {s}")
def ok(s):   print(f"  {C.G}✓{C.END} {s}")
def warn(s): print(f"  {C.Y}!{C.END} {s}")
def err(s):  print(f"  {C.R}✗{C.END} {s}")


# ─────────────────────────── runner ───────────────────────────
def run_diagnostic(checkpoint, eval_cache, output_dir, n_vis, seed, phase_label):
    """Run diagnose_raw_residual.py for one phase. Returns True on success."""
    step(f"Run diagnostic — {phase_label}")
    info(f"checkpoint: {checkpoint}")
    info(f"output:     {output_dir}")
    cmd = [
        sys.executable, "tools/diagnose_raw_residual.py",
        "--checkpoint", str(checkpoint),
        "--apricot-eval-cache", str(eval_cache),
        "--use-ema",
        "--n-vis", str(n_vis),
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    info("command: " + " ".join(cmd))
    print()
    rc = subprocess.call(cmd)
    print()
    if rc != 0:
        err(f"diagnostic failed for {phase_label} (exit {rc})")
        return False
    ok(f"diagnostic complete for {phase_label}")
    return True


# ─────────────────────────── analysis ───────────────────────────
def load_stats(output_dir):
    p = Path(output_dir) / "stats.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def per_image_arrays(stats):
    """Pull per-image arrays from a stats.json. Drops None entries."""
    inside, outside, ratio, names, bbox_areas = [], [], [], [], []
    for entry in stats.get("per_image", []):
        s = entry.get("stats", {})
        ins = s.get("inside_mean")
        out = s.get("outside_mean")
        rat = s.get("ratio")
        if ins is not None and out is not None and rat is not None:
            inside.append(float(ins))
            outside.append(float(out))
            ratio.append(float(rat))
            names.append(entry.get("name", "?"))
            bbox_areas.append(s.get("inside_n_pixels"))
    return inside, outside, ratio, names, bbox_areas


def summarize(values, name):
    if not values:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def fmt_row(label, p1, p2, fmt_spec=".4f", lo_is_good=False):
    """Pretty-print one row comparing Phase 1 and Phase 2 values."""
    if p1 is None or p2 is None:
        p1s = "N/A" if p1 is None else format(p1, fmt_spec)
        p2s = "N/A" if p2 is None else format(p2, fmt_spec)
        delta = ""
        color = ""
    else:
        p1s = format(p1, fmt_spec)
        p2s = format(p2, fmt_spec)
        d = p2 - p1
        delta = f"{d:+.4f}" if abs(d) >= 1e-4 else f"{d:+.6f}"
        better = (d < 0) if lo_is_good else (d > 0)
        color = C.G if better else (C.R if abs(d) > 1e-9 else C.DIM)
    print(f"  {C.BOLD}{label:32s}{C.END} "
          f"P1={p1s:>14s}   P2={p2s:>14s}   "
          f"{color}Δ={delta:>10s}{C.END}")


def section(title):
    print(f"\n  {C.BOLD}{C.Y}── {title} ──{C.END}")


def compare(p1_stats, p2_stats):
    hdr("PHASE 1 vs PHASE 2 — Side-by-side comparison")

    if p1_stats is None or p2_stats is None:
        err("missing stats.json for one or both phases")
        return

    # ── Aggregate-level metrics from stats.json itself ──
    section("Aggregate stats from stats.json")
    fmt_row("n_visualized",
            p1_stats.get("n_visualized"), p2_stats.get("n_visualized"), ".0f")
    fmt_row("global_residual_min",
            p1_stats.get("global_residual_min"),
            p2_stats.get("global_residual_min"),
            ".5f")
    fmt_row("global_residual_max",
            p1_stats.get("global_residual_max"),
            p2_stats.get("global_residual_max"),
            ".5f")
    fmt_row("S3_corr(inside_mean,ratio)",
            p1_stats.get("S3_corr_inside_mean_vs_ratio"),
            p2_stats.get("S3_corr_inside_mean_vs_ratio"),
            ".4f")

    # ── Per-image arrays ──
    p1_in, p1_out, p1_ratio, p1_names, p1_areas = per_image_arrays(p1_stats)
    p2_in, p2_out, p2_ratio, p2_names, p2_areas = per_image_arrays(p2_stats)

    if not p1_in or not p2_in:
        err("no per-image data found")
        return

    section(f"Per-image RATIO distribution  (n_p1={len(p1_ratio)}, n_p2={len(p2_ratio)})")
    s1 = summarize(p1_ratio, "p1")
    s2 = summarize(p2_ratio, "p2")
    fmt_row("ratio mean",   s1["mean"],   s2["mean"],   ".4f")
    fmt_row("ratio median", s1["median"], s2["median"], ".4f")
    fmt_row("ratio std",    s1["std"],    s2["std"],    ".4f")
    fmt_row("ratio min",    s1["min"],    s2["min"],    ".4f")
    fmt_row("ratio max",    s1["max"],    s2["max"],    ".4f")

    # S1 thresholds
    p1_pos = sum(1 for r in p1_ratio if r > 1.0)
    p2_pos = sum(1 for r in p2_ratio if r > 1.0)
    p1_5x  = sum(1 for r in p1_ratio if r >= 5.0)
    p2_5x  = sum(1 for r in p2_ratio if r >= 5.0)
    p1_3x  = sum(1 for r in p1_ratio if 3.0 <= r < 5.0)
    p2_3x  = sum(1 for r in p2_ratio if 3.0 <= r < 5.0)
    p1_2x  = sum(1 for r in p1_ratio if 2.0 <= r < 3.0)
    p2_2x  = sum(1 for r in p2_ratio if 2.0 <= r < 3.0)

    section("Bin counts (where the signal lives)")
    fmt_row("ratio >= 5x",      p1_5x,  p2_5x,  ".0f")
    fmt_row("ratio in [3,5)",   p1_3x,  p2_3x,  ".0f")
    fmt_row("ratio in [2,3)",   p1_2x,  p2_2x,  ".0f")
    fmt_row("ratio > 1 (positives)", p1_pos, p2_pos, ".0f")

    # ── Inside-mean (raw absolute signal magnitude inside bbox) ──
    section("inside_mean — raw absolute signal magnitude AT patches")
    si1 = summarize(p1_in, "p1")
    si2 = summarize(p2_in, "p2")
    fmt_row("inside_mean mean",   si1["mean"],   si2["mean"],   ".5f")
    fmt_row("inside_mean median", si1["median"], si2["median"], ".5f")
    fmt_row("inside_mean max",    si1["max"],    si2["max"],    ".5f")
    fmt_row("inside_mean std",    si1["std"],    si2["std"],    ".5f")

    section("outside_mean — background")
    so1 = summarize(p1_out, "p1")
    so2 = summarize(p2_out, "p2")
    fmt_row("outside_mean mean",   so1["mean"],   so2["mean"],   ".5f")
    fmt_row("outside_mean median", so1["median"], so2["median"], ".5f")
    fmt_row("outside_mean max",    so1["max"],    so2["max"],    ".5f")

    # Effective signal-to-background separation in raw units
    p1_sep = [a - b for a, b in zip(p1_in, p1_out)]
    p2_sep = [a - b for a, b in zip(p2_in, p2_out)]
    section("inside_mean - outside_mean (raw separation)")
    sp1 = summarize(p1_sep, "p1")
    sp2 = summarize(p2_sep, "p2")
    fmt_row("separation mean",   sp1["mean"],   sp2["mean"],   ".5f")
    fmt_row("separation median", sp1["median"], sp2["median"], ".5f")
    fmt_row("separation max",    sp1["max"],    sp2["max"],    ".5f")
    pos1 = sum(1 for v in p1_sep if v > 0)
    pos2 = sum(1 for v in p2_sep if v > 0)
    fmt_row("# images sep > 0",  pos1, pos2, ".0f")

    # ── Per-image overlap analysis ──
    common = set(p1_names) & set(p2_names)
    if common:
        idx1 = {n: i for i, n in enumerate(p1_names)}
        idx2 = {n: i for i, n in enumerate(p2_names)}
        improved = 0
        regressed = 0
        unchanged = 0
        diffs = []
        for nm in common:
            r1 = p1_ratio[idx1[nm]]
            r2 = p2_ratio[idx2[nm]]
            d = r2 - r1
            diffs.append(d)
            if d > 0.05:
                improved += 1
            elif d < -0.05:
                regressed += 1
            else:
                unchanged += 1
        section(f"Per-image change (P2 vs P1, on {len(common)} shared images)")
        info(f"images where ratio improved (Δ > +0.05): {C.G}{improved}{C.END}")
        info(f"images where ratio regressed (Δ < -0.05): {C.R}{regressed}{C.END}")
        info(f"images roughly unchanged: {unchanged}")
        if diffs:
            info(f"median Δratio: {median(diffs):+.4f}")
            info(f"mean Δratio:   {mean(diffs):+.4f}")
            info(f"max +Δ:        {max(diffs):+.4f}")
            info(f"max -Δ:        {min(diffs):+.4f}")

    hdr("VERDICT")
    p1_med = s1["median"]
    p2_med = s2["median"]

    print(f"  {C.BOLD}S1 (median ratio ≥ 1.20 AND positives ≥ 85%){C.END}")
    p1_pass = (p1_med >= 1.20) and (p1_pos / len(p1_ratio) >= 0.85)
    p2_pass = (p2_med >= 1.20) and (p2_pos / len(p2_ratio) >= 0.85)
    p1_color = C.G if p1_pass else C.R
    p2_color = C.G if p2_pass else C.R
    print(f"    P1: median={p1_med:.4f}  positives={p1_pos}/{len(p1_ratio)} "
          f"({100*p1_pos/len(p1_ratio):.1f}%)  →  {p1_color}{'PASS' if p1_pass else 'FAIL'}{C.END}")
    print(f"    P2: median={p2_med:.4f}  positives={p2_pos}/{len(p2_ratio)} "
          f"({100*p2_pos/len(p2_ratio):.1f}%)  →  {p2_color}{'PASS' if p2_pass else 'FAIL'}{C.END}")

    print()
    print(f"  {C.BOLD}S2 (≥5× bin shows non-trivial mass after Phase 2){C.END}")
    print(f"    P1: {p1_5x} images ≥5×")
    print(f"    P2: {p2_5x} images ≥5×")
    if p2_5x > p1_5x:
        print(f"    {C.G}Phase 2 produced more strong-signal images.{C.END}")
    elif p2_5x == p1_5x == 0:
        print(f"    {C.Y}Neither phase reached the strong-signal regime.{C.END}")
    else:
        print(f"    {C.R}Phase 2 lost strong-signal mass relative to Phase 1.{C.END}")

    print()
    print(f"  {C.BOLD}S3 (corr(inside_mean, ratio) ≥ 0.5){C.END}")
    c1 = p1_stats.get("S3_corr_inside_mean_vs_ratio")
    c2 = p2_stats.get("S3_corr_inside_mean_vs_ratio")
    print(f"    P1 correlation: {c1:.4f}" if c1 is not None else "    P1 correlation: N/A")
    print(f"    P2 correlation: {c2:.4f}" if c2 is not None else "    P2 correlation: N/A")
    if c1 is not None and c2 is not None and c2 > c1:
        print(f"    {C.G}Phase 2 has stronger absolute-vs-normalized correlation "
              f"(+{c2-c1:+.4f}).{C.END}")

    # ── Where to look ──
    p1_dir = Path(p1_stats["args"]["output_dir"])
    p2_dir = Path(p2_stats["args"]["output_dir"])
    print()
    info(f"P1 grids: {p1_dir / 'grids'}")
    info(f"P2 grids: {p2_dir / 'grids'}")
    info("Open them with:  open " + str(p1_dir / 'grids') + "  &&  open " + str(p2_dir / 'grids'))


# ─────────────────────────── main ───────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p1-checkpoint", default="runs/phase1_seed42/checkpoints/best")
    p.add_argument("--p2-checkpoint", default="runs/phase2_seed42/checkpoints/best")
    p.add_argument("--apricot-eval-cache", default="caches/apricot_eval_640.pt")
    p.add_argument("--p1-output", default="runs/phase1_seed42/raw_residual_visuals_full")
    p.add_argument("--p2-output", default="runs/phase2_seed42/raw_residual_visuals_full")
    p.add_argument("--n-vis", type=int, default=873)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-run", action="store_true",
                   help="Skip running diagnostics; just compare existing JSON outputs.")
    p.add_argument("--only", choices=["p1", "p2"], default=None,
                   help="Only run one phase.")
    args = p.parse_args()

    hdr("DPC Phase 1 vs Phase 2 — Raw Residual Comparison")
    info(f"n_vis: {args.n_vis}    seed: {args.seed}")

    if not args.skip_run:
        if args.only != "p2":
            ok_p1 = run_diagnostic(args.p1_checkpoint, args.apricot_eval_cache,
                                   args.p1_output, args.n_vis, args.seed, "PHASE 1")
            if not ok_p1:
                err("Phase 1 failed; aborting.")
                sys.exit(1)
        if args.only != "p1":
            ok_p2 = run_diagnostic(args.p2_checkpoint, args.apricot_eval_cache,
                                   args.p2_output, args.n_vis, args.seed, "PHASE 2")
            if not ok_p2:
                err("Phase 2 failed; aborting.")
                sys.exit(1)

    step("Load stats.json from both phases")
    p1_stats = load_stats(args.p1_output)
    p2_stats = load_stats(args.p2_output)
    if p1_stats is None:
        err(f"missing {args.p1_output}/stats.json")
    else:
        ok(f"loaded P1 stats ({p1_stats.get('n_visualized')} images)")
    if p2_stats is None:
        err(f"missing {args.p2_output}/stats.json")
    else:
        ok(f"loaded P2 stats ({p2_stats.get('n_visualized')} images)")
    if p1_stats is None or p2_stats is None:
        sys.exit(1)

    compare(p1_stats, p2_stats)


if __name__ == "__main__":
    main()
