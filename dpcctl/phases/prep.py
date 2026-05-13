"""prep phase — build tensor caches once across all seeds.

Wraps tools/build_caches.py and tools/fit_color_distribution.py.
Idempotent: skips work if all expected cache files already exist.

v3.3.0: flag names now match the actual CLIs of build_caches.py and
fit_color_distribution.py (the v3.2.0 prep.py used stale flag names that
were renamed in the tools layer but never updated here).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from dpcctl.phases._base import Phase, PhaseStatus, register


@register("prep")
class PrepPhase(Phase):
    is_shared = True
    depends_on = ()

    def run(self) -> dict:
        cfg = self.ctx.cfg
        events = self.ctx.events

        events.set_global("prep", "running")
        events.publish("prep", "progress", {"message": "starting prep"})
        self.write_status(PhaseStatus.RUNNING)

        cache_dir = cfg.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        probe_res = cfg.prep.probe_resolution
        eval_res = cfg.prep.eval_resolution

        coco_cache = cache_dir / f"coco_train2017_{probe_res}.pt"
        apricot_train_cache = cache_dir / f"apricot_train_{probe_res}.pt"
        apricot_val_cache = cache_dir / f"apricot_val_{probe_res}.pt"
        apricot_eval_cache = cache_dir / f"apricot_eval_{eval_res}.pt"
        color_dist = cache_dir / "color_distribution.json"

        t0 = time.time()
        staging = Path(__file__).resolve().parents[2]
        build_caches = staging / "tools" / "build_caches.py"
        fit_color = staging / "tools" / "fit_color_distribution.py"

        # build_caches: skip COCO if cache already exists (saves the long re-encode step)
        skip_coco_flag = ["--skip-coco"] if coco_cache.is_file() else []

        # Resolve COCO annotations path if configured. Resolves relative to
        # the config file's parent directory, matching how data.coco_dir is
        # resolved in dpcctl/config.py.
        coco_ann_flag: list = []
        if cfg.prep.coco_annotations:
            ann_p = (cfg.config_path.parent / cfg.prep.coco_annotations).resolve()
            if not ann_p.is_file():
                raise RuntimeError(
                    f"prep: coco_annotations file not found: {ann_p}"
                )
            coco_ann_flag = ["--coco-annotations", str(ann_p)]

        # APRICOT train fraction: derive from val fraction
        apricot_train_frac = 1.0 - cfg.data.apricot_val_fraction

        all_apricot_present = (
            apricot_train_cache.is_file()
            and apricot_val_cache.is_file()
            and apricot_eval_cache.is_file()
        )

        if not all_apricot_present or not coco_cache.is_file():
            cmd = [
                "python", str(build_caches),
                "--coco-train", str(cfg.coco_dir_path),
                "--apricot-base", str(cfg.apricot_dir_path),
                "--probe-res", str(probe_res),
                "--eval-res", str(eval_res),
                "--apricot-train-frac", f"{apricot_train_frac:.4f}",
                "--output", str(cache_dir),
                "--num-workers", str(cfg.data.num_workers),
                "--seed", str(cfg.seed),
            ] + skip_coco_flag + coco_ann_flag

            events.publish("prep", "progress", {
                "message": "building caches",
                "elapsed_sec": time.time() - t0,
            })
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                self.write_status(
                    PhaseStatus.FAILED,
                    error=f"build_caches exited {result.returncode}",
                )
                raise RuntimeError(
                    f"prep: build_caches failed (exit {result.returncode})"
                )
        else:
            events.publish("prep", "progress", {
                "message": "all caches present; skipping build_caches",
                "elapsed_sec": time.time() - t0,
            })

        if not color_dist.is_file():
            if not apricot_train_cache.is_file():
                raise RuntimeError(
                    f"cannot fit color distribution: {apricot_train_cache} missing"
                )
            cmd2 = [
                "python", str(fit_color),
                "--apricot-cache", str(apricot_train_cache),
                "--output", str(color_dist),
                "--seed", str(cfg.seed),
            ]
            events.publish("prep", "progress", {
                "message": "fitting color distribution",
                "elapsed_sec": time.time() - t0,
            })
            result = subprocess.run(cmd2, check=False)
            if result.returncode != 0:
                self.write_status(
                    PhaseStatus.FAILED,
                    error=f"fit_color_distribution exited {result.returncode}",
                )
                raise RuntimeError("prep: fit_color_distribution failed")
        else:
            events.publish("prep", "progress", {
                "message": "color_distribution.json present; skipping fit",
                "elapsed_sec": time.time() - t0,
            })

        elapsed_min = (time.time() - t0) / 60.0
        events.publish("prep", "diagnostic", {
            "message": "prep complete",
            "caches": {
                "coco_train": coco_cache.is_file(),
                "apricot_train": apricot_train_cache.is_file(),
                "apricot_val": apricot_val_cache.is_file(),
                "apricot_eval": apricot_eval_cache.is_file(),
                "color_dist": color_dist.is_file(),
            },
            "elapsed_min": round(elapsed_min, 2),
        })
        self.write_status(PhaseStatus.COMPLETE, elapsed_min=elapsed_min)
        return {"elapsed_min": elapsed_min}
