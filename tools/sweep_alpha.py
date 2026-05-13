#!/usr/bin/env python3
"""Alpha sweep: invoke evaluate_phase3.py for each alpha in a list.

Usage:
    python tools/sweep_alpha.py \
        --denoiser-checkpoint runs/phase2_seed42/checkpoints/best \
        --yolo-weights yolo26n.pt \
        --apricot-eval-cache caches/apricot_eval_640.pt \
        --alphas 10,25,50,75,100 \
        --seed 42 \
        --root-output-dir runs/

Output:
  runs/phase3_seed42_alpha10/eval/{aggregate.json, per_image.json, ...}
  runs/phase3_seed42_alpha25/...
  ...
  runs/phase3_seed42_sweep/sweep_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpc._version import __version__
from dpc.manifest import fingerprint_environment


class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
    BOLD = "\033[1m"; END = "\033[0m"


def stage(m): print(f"\n{C.BOLD}{C.B}━━ {m} ━━{C.END}")
def ok(m): print(f"  {C.G}✓{C.END} {m}")
def info(m): print(f"  {C.B}•{C.END} {m}")
def fail(m): print(f"  {C.R}✗{C.END} {m}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser-checkpoint", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--apricot-eval-cache", required=True)
    p.add_argument("--alphas", type=str, default="10,25,50,75,100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--root-output-dir", default="runs")
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--on-patch-iou", type=float, default=0.1)
    p.add_argument("--n-images", type=int, default=None)
    p.add_argument("--no-use-ema", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true",
                   help="Continue sweep even if a single alpha fails")
    p.add_argument("--phase-prefix", default="phase3",
                   help="Run-dir prefix (default: phase3)")
    args = p.parse_args()

    print(f"{C.BOLD}DPC Phase 3 Alpha Sweep v{__version__}{C.END}")

    alphas = [float(a) for a in args.alphas.split(",")]
    info(f"alphas: {alphas}")
    info(f"seed:   {args.seed}")
    info(f"checkpoint: {args.denoiser_checkpoint}")

    root = Path(args.root_output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    sweep_dir = root / f"{args.phase_prefix}_seed{args.seed}_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    per_alpha_results: list[dict] = []
    failed_alphas: list[float] = []

    for alpha in alphas:
        out_dir = root / f"{args.phase_prefix}_seed{args.seed}_alpha{int(alpha)}"
        stage(f"alpha={alpha} → {out_dir.name}")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "evaluate_phase3.py"),
            "--denoiser-checkpoint", args.denoiser_checkpoint,
            "--yolo-weights", args.yolo_weights,
            "--apricot-eval-cache", args.apricot_eval_cache,
            "--output-dir", str(out_dir),
            "--score-threshold", str(args.score_threshold),
            "--on-patch-iou", str(args.on_patch_iou),
            "--cls-alpha", str(alpha),
            "--obj-alpha", str(alpha),
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        if args.n_images is not None:
            cmd.extend(["--n-images", str(args.n_images)])
        if args.no_use_ema:
            cmd.append("--no-use-ema")

        info(f"running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        # evaluate_phase3 exits 0 only if all 3 claims pass; we treat both 0 and 1
        # as "completed successfully" as long as the aggregate.json exists.
        agg_path = out_dir / "eval" / "aggregate.json"
        if not agg_path.is_file():
            fail(f"alpha={alpha} produced no aggregate.json")
            failed_alphas.append(alpha)
            if not args.continue_on_failure:
                fail("aborting sweep (use --continue-on-failure to skip)")
                sys.exit(1)
            continue

        with open(agg_path) as f:
            agg = json.load(f)
        per_alpha_results.append({
            "alpha": float(alpha),
            "output_dir": str(out_dir),

            "on_patch_suppression": agg.get("on_patch_suppression"),
            "off_patch_retention": agg.get("off_patch_retention"),
            "per_image_margin": agg.get("per_image_margin"),
            "discriminability_ratio": agg.get("discriminability_ratio"),
            "mean_inference_ms": agg.get("mean_inference_ms"),
        })
        ok(f"alpha={alpha} done")

    # ── Sweep summary ──
    stage("Sweep summary")
    print()
    print(f"  {C.BOLD}{'alpha':>6} {'C1 mean':>10} {'C1 ci_lo':>10} {'C2 mean':>10} {'C3 median':>11} {'pass':>6}{C.END}")
    print(f"  {'-' * 60}")
    for r in per_alpha_results:
        c1m = r["on_patch_suppression"]["mean_reduction_pp"] if r["on_patch_suppression"] else None
        c1l = r["on_patch_suppression"]["ci95"][0] if r["on_patch_suppression"] and r["on_patch_suppression"]["ci95"][0] is not None else None
        c2m = r["off_patch_retention"]["mean_retention"] if r["off_patch_retention"] else None
        c3m = r["per_image_margin"]["median"] if r["per_image_margin"] else None
        n_pass = sum(int(bool(r[f"{c}_pass"])) for c in ["on_patch_suppression", "off_patch_retention", "per_image_margin"])
        print(f"  {r['alpha']:>6.0f} "
              f"{(c1m if c1m is not None else float('nan')):>10.2f} "
              f"{(c1l if c1l is not None else float('nan')):>10.2f} "
              f"{(c2m if c2m is not None else float('nan')):>10.3f} "
              f"{(c3m if c3m is not None else float('nan')):>11.2f} "
              f"{n_pass:>4}/3")

    # Best alpha = max of (C1_mean) among passing; tie-break by best C2
    passing = [r for r in per_alpha_results
               if r["on_patch_suppression"] and r["off_patch_retention"] and r["per_image_margin"]]
    if passing:
        best = max(passing, key=lambda r: (r["on_patch_suppression"]["mean_reduction_pp"],
                                           r["off_patch_retention"]["mean_retention"]))
        info(f"best passing alpha: {best['alpha']}")
    else:
        # Soft choice: highest C1 mean even if not all pass
        candidates = [r for r in per_alpha_results
                      if r["on_patch_suppression"] and r["on_patch_suppression"]["mean_reduction_pp"] is not None]
        if candidates:
            best = max(candidates, key=lambda r: r["on_patch_suppression"]["mean_reduction_pp"])
            info(f"no alpha passes all 3 claims; best by C1: alpha={best['alpha']}")
        else:
            best = None
            fail("no alpha produced valid aggregate")

    # Save summary
    summary = {
        "version": __version__,
        "tool": "sweep_alpha",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": fingerprint_environment(),
        "args": vars(args),
        "alphas_tested": alphas,
        "failed_alphas": failed_alphas,
        "per_alpha": per_alpha_results,
        "best_alpha": best["alpha"] if best else None,
    }
    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    ok(f"summary -> {summary_path}")

    if failed_alphas:
        sys.exit(1)


if __name__ == "__main__":
    main()
