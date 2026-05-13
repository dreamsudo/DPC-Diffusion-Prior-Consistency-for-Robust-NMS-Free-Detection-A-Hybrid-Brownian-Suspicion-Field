"""aggregate phase — multi-seed mean/stdev across seeds.

Walks each seed's eval_p3/alpha_sweep.json and produces a
multi_seed_summary.json with per-alpha mean ± stdev of the on-patch
suppression, off-patch retention, and per-image margin numbers.

This replaces an earlier implementation that shelled out to
tools/aggregate_seeds.py with an incompatible CLI.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from dpcctl.phases._base import Phase, PhaseStatus, register


def _safe_mean_std(xs: list) -> tuple:
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, 0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var), n


@register("aggregate")
class AggregatePhase(Phase):
    is_shared = True
    depends_on = ("eval_p3",)

    def run(self) -> dict:
        cfg = self.ctx.cfg
        events = self.ctx.events
        events.set_global("aggregate", "running")
        self.write_status(PhaseStatus.RUNNING)

        runs_root = cfg.run_dir.parent
        seeds = cfg.seeds

        # alpha -> list of per-seed result dicts
        per_alpha: dict = {}
        seed_sweeps = {}
        for seed in seeds:
            sweep_path = (runs_root / cfg.name / f"seed_{seed}"
                          / "eval_p3" / "alpha_sweep.json")
            if not sweep_path.is_file():
                continue
            try:
                data = json.loads(sweep_path.read_text())
            except json.JSONDecodeError:
                continue
            seed_sweeps[seed] = data
            for r in data.get("results", []):
                a = r.get("alpha")
                per_alpha.setdefault(a, []).append(r)

        agg = {}
        for alpha, results in per_alpha.items():
            on = [r.get("on_patch_suppression", {}).get("mean_reduction_pp")
                  for r in results]
            off = [r.get("off_patch_retention", {}).get("mean_retention")
                   for r in results]
            margin = [r.get("per_image_margin", {}).get("median")
                      for r in results]

            m_on, s_on, n_on = _safe_mean_std(on)
            m_off, s_off, n_off = _safe_mean_std(off)
            m_marg, s_marg, n_marg = _safe_mean_std(margin)

            agg[str(alpha)] = {
                "alpha": alpha,
                "n_seeds": len(results),
                "on_patch_suppression_mean_reduction_pp": {
                    "mean": m_on, "stdev": s_on, "n": n_on,
                },
                "off_patch_retention_mean": {
                    "mean": m_off, "stdev": s_off, "n": n_off,
                },
                "per_image_margin_median": {
                    "mean": m_marg, "stdev": s_marg, "n": n_marg,
                },
            }

        out = {
            "run": cfg.name,
            "seeds": seeds,
            "n_seeds_with_data": len(seed_sweeps),
            "per_alpha": agg,
        }

        summary_path = runs_root / cfg.name / "multi_seed_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(out, indent=2, default=str))

        events.publish("aggregate", "aggregate", {
            "seeds": seeds,
            "n_alphas": len(agg),
            "n_seeds_with_data": len(seed_sweeps),
            "summary_path": str(summary_path),
        })
        events.set_global("aggregate", "complete")
        self.write_status(PhaseStatus.COMPLETE, n_alphas=len(agg))
        return {"n_alphas": len(agg)}
