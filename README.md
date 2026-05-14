# DPC-YOLO26 v3.3.1

**Operational Deployment Guide** — Installation, Tools, Workflow, Tuning, and Deployment for the Reference Codebase

![version](https://img.shields.io/badge/version-3.3.1-blue)
![python](https://img.shields.io/badge/python-3.12-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.11-orange)
![backend](https://img.shields.io/badge/backend-MPS%20%7C%20CUDA-green)
![tests](https://img.shields.io/badge/tests-13%20modules%20%7C%20~107%20pass-brightgreen)
![smoke](https://img.shields.io/badge/smoke-13.3%20min-success)
![license](https://img.shields.io/badge/license-research-lightgrey)

> *Diffusion-Prior Consistency Defense against Adversarial Patch Attacks on NMS-Free Object Detection.*
> California State University San Marcos — May 2026

---

## About This Guide

This is the operational deployment guide for the DPC-YOLO26 v3.3.1 reference codebase. It documents every file, every tool, every configuration option, every phase, and the complete end-to-end workflow from a fresh machine to a production deployment. The guide is the practical companion to the research manuscript and the full research manual. It assumes readers are familiar with PyTorch and basic object detection; theoretical derivations are deferred to the paper.

**Reading order.** Sections 1–3 (overview, install, layout) before first use. Section 4 (commands) and 5 (phases) for the operational pattern. Section 6 (tools) and 7 (configuration) as reference. Section 8 (end-to-end walkthrough) as a check that everything works. Section 9 (tuning) when calibrating for a deployment. Section 10 (debugging) when something fails. Section 11 (deployment) for production operationalization. Section 12 (reproducibility & reference) for audits.

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Installation](#2-installation)
3. [Repository Layout: Every File](#3-repository-layout-every-file)
4. [Command Interface](#4-command-interface)
5. [The Eight Phases](#5-the-eight-phases)
6. [Tools (All 20 Scripts)](#6-tools-all-20-scripts)
7. [Configuration Reference](#7-configuration-reference)
8. [End-to-End Walkthrough](#8-end-to-end-walkthrough)
9. [Tuning](#9-tuning)
10. [Debugging](#10-debugging)
11. [Production Deployment](#11-production-deployment)
12. [Reproducibility and Reference](#12-reproducibility-and-reference)

---

## 1. System Overview

### 1.1 What This Defends Against

DPC-YOLO26 is a defense against localized adversarial patch attacks on the YOLO26 family of NMS-free object detectors. An adversary inserts a contiguous region of arbitrary pixel content (a "patch", area ρ ∈ [0.005, 0.05] of the image) into an input image, attempting to hide objects, mislabel them, insert spurious detections, or degrade box localization.

### 1.2 How the Defense Works

The framework wraps the YOLO26 detector with a **diffusion-based suspicion engine** that computes a per-pixel signal $R_{\text{deploy}} \in [0, 1]^{h \times w}$ indicating off-manifold image regions. The signal is injected directly into the detector's decision logic at four points:

- **Objectness logits** (Eq. 16 of the paper) — suppression in suspect regions.
- **Class logits** (Eq. 17) — uniform shift across classes.
- **Small-target amplification** (Eq. 21) — compensates for YOLO26's small-target label assignment.
- **Hungarian assignment cost matrix** (Eq. 22) — the framework's central novel intervention. No image-space defense can reach this point.

The defense adds approximately **2× latency** over the baseline detector (Theorem 4 of the paper) and preserves clean-input accuracy within **1% mAP** (Theorem 5).

### 1.3 Three Configurations

| Preset            | Wall-clock | Seeds              | Use for                                                  |
| ----------------- | ---------- | ------------------ | -------------------------------------------------------- |
| `quick.json`      | ~13 min    | [42]               | install verification, CI                                 |
| `default.json`    | ~1 hr      | [42]               | development experiments                                  |
| `production.json` | ~22 hr     | [42, 1337, 2718]   | paper-grade metrics with bootstrap 95% CIs              |

### 1.4 Validation Hardware and Software

| Component          | Value                                            |
| ------------------ | ------------------------------------------------ |
| Validation HW      | MacBook Pro M1 Max, 32 GB RAM, Apple MPS backend |
| Validation SW      | Python 3.12.13, PyTorch 2.11                     |
| Smoke wall-clock   | 13.3 minutes (8 phases)                          |
| Test suite         | 13 modules, ~107 tests pass                      |
| Reference run      | `runs_quick_v33/` committed in repo              |

### 1.5 Eight-Phase Pipeline

A run executes eight phases sequentially. The orchestrator handles all sequencing, state persistence, and resumability.

| Phase                     | Smoke time | Function                                                                                                          |
| ------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------- |
| `prep`                    | 3.7 min    | Verify cache SHA256s; load YOLO26 weights.                                                                        |
| `train_p1`                | 2.4 min    | Train TinyUNet denoiser on COCO.                                                                                  |
| `diagnose_p1`             | 1.4 min    | Measure residual ratio on validation patches.                                                                     |
| `train_p2`                | 7.0 min    | Joint fine-tune denoiser + YOLO26 head; modulated Hungarian assignment fires per batch.                          |
| `diagnose_p2`             | 1.4 min    | Re-measure residual ratio after joint training.                                                                   |
| `eval_p3`                 | 1.4 min    | APRICOT evaluation across α sweep.                                                                                |
| `eval_negative_control`   | 0.4 min    | Clean-image evaluation; Theorem 5 check.                                                                          |
| `aggregate`               | <0.1 min   | Cross-seed statistics with bootstrap 95% CIs.                                                                     |

> 💡 **Tip.** The single most consequential parameter for tuning is `calibration.lambda_match` (Equation 22's modulation strength). Start by running the alpha sweep, identify the elbow of the hide-rate-vs-FP-density curve, and set deployment to that alpha.

### 1.6 What This Guide Covers

This guide covers the complete operational lifecycle: installation (§2), repository layout and the role of every file (§3), the command interface (§4), each of the 8 phases in operational detail (§5), every `tools/` script (§6), every configuration knob (§7), the canonical end-to-end walkthrough (§8), tuning procedures (§9), debugging (§10), production deployment (§11), and reproducibility plus reference tables (§12).

Theoretical justification, methodology derivations, and module-level docs are in the companion paper and the full 93-page research manual.

---

## 2. Installation

### 2.1 Requirements

**Python.** 3.12.x. Validated against 3.12.13. Python 3.11 likely works; 3.13 untested; ≤ 3.10 unsupported.

**Memory.** Smoke: 4 GB device memory. Default: 8 GB. Production: 16 GB+. System RAM ≥ 2× device memory for cache loading and DataLoader workers.

**Backends.**

| Backend     | Status     | Notes                                  |
| ----------- | ---------- | -------------------------------------- |
| Apple MPS   | Validated  | Smoke 13.3 min on M1 Max               |
| NVIDIA CUDA | Supported  | Not yet validated end-to-end           |
| CPU         | Functional | ≥10× slower; unit tests only           |

### 2.2 Bootstrap (Recommended)

```bash
git clone <repo-url> dpcroot
cd dpcroot
./bootstrap.sh ~/dpc-v33-venv
source ~/dpc-v33-venv/bin/activate
python -c "import dpc; print(dpc.__version__)"   # expect: 3.3.1
```

The bootstrap script:

1. Validates Python 3.12 via `python3.12 --version`.
2. Creates a venv at the given path: `python3.12 -m venv $1`.
3. Upgrades `pip` and `wheel`.
4. Installs from `requirements.txt`.
5. Runs `pip install -e .` for editable install of `dpc` and `dpcctl`.
6. Verifies the package imports.

### 2.3 Manual Install

If the bootstrap fails or a more controlled install is preferred:

```bash
python3.12 -m venv ~/dpc-v33-venv
source ~/dpc-v33-venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install -e .
```

### 2.4 Verification

Three commands; all should succeed:

```bash
python -m pytest tests/ -q              # all 13 test modules pass
python -c "import torch; print(torch.backends.mps.is_available())"   # True on M-series
python -m dpcctl --help                 # orchestrator CLI loads
```

### 2.5 Environment Variables

| Variable                          | Purpose                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| `DPC_DEVICE`                      | Override default device (`mps`/`cuda`/`cpu`) without editing config.                          |
| `DPC_CACHE_DIR`                   | Override default cache directory location.                                                    |
| `DPC_RUNS_DIR`                    | Override default runs/output directory.                                                       |
| `PYTORCH_ENABLE_MPS_FALLBACK`     | Set `1` on Apple Silicon so unsupported MPS ops fall back to CPU rather than crash.           |
| `PYTHONHASHSEED`                  | Fixed value to make Python's hash-based randomness deterministic.                             |

### 2.6 Reproducibility Setup

After a successful install, freeze the dependency tree:

```bash
pip freeze > requirements.lock.txt
```

The lock file captures transitive dependencies that `requirements.txt` does not. Commit it alongside experimental results to enable bit-identical reconstruction.

### 2.7 Common Install Failures

<details>
<summary><strong>Wrong PyTorch wheel</strong></summary>

On Apple Silicon, pip may pull the x86_64 wheel under Rosetta. Verify:

```bash
file $(python -c "import torch; print(torch.__file__)")
```

If the binary is x86_64 on Apple Silicon, MPS will be unavailable. Re-install in an arm64-native shell.
</details>

<details>
<summary><strong>scipy import failure</strong></summary>

The Hungarian assignment depends on `scipy.optimize.linear_sum_assignment`. Re-install if shadowed:

```bash
pip install --force-reinstall --no-cache-dir scipy
```
</details>

<details>
<summary><strong>Missing YOLO26 weights</strong></summary>

The prep phase loads YOLO26 via Ultralytics, which downloads weights on first use. Pre-fetch to avoid mid-run network errors:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```
</details>

<details>
<summary><strong>MPS backend unavailable</strong></summary>

`torch.backends.mps.is_available()` returns `False` on Apple Silicon. Re-install with the official pip wheel:

```bash
pip install --upgrade --force-reinstall torch torchvision
```
</details>

---

## 3. Repository Layout: Every File

### 3.1 Top-Level Tree

```
dpcroot/
├── CHANGELOG.md            theorem-to-code mapping for v3.3.1
├── README.md               quickstart and library usage
├── bootstrap.sh            one-shot venv install
├── fix_merge.sh            git merge helper (release-only)
├── pyproject.toml          packaging metadata; pulls version from dpc/_version.py
├── requirements.txt        pinned runtime dependencies
├── caches/                 preprocessed datasets (committed)
├── configs/                three JSON configurations
├── docs/                   engineering log
├── dpc/                    math library — 22 modules
├── dpcctl/                 orchestrator — phase scheduler, CLI, state, dashboard
├── runs_quick_v33/         reference smoke output (committed)
├── tests/                  13 test modules (~107 tests)
└── tools/                  20 CLI scripts
```

### 3.2 `caches/` — Preprocessed Datasets

The cache directory holds tensorized versions of COCO and APRICOT for fast loading at training time. All six files are committed so smoke runs start immediately.

| File                          | Contents                                                                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `coco_train2017_128.pt`       | COCO 2017 train images at 128×128, `uint8`. Used by Phase 1 and Phase 2 training.                                                              |
| `apricot_train_128.pt`        | APRICOT train at 128×128. Used by Phase 2 mixture loader.                                                                                       |
| `apricot_val_128.pt`          | APRICOT val at 128×128. Used by the diagnose phases.                                                                                            |
| `apricot_eval_320.pt`         | APRICOT eval at 320×320. Used by Phase 3 evaluation (higher resolution because YOLO26 needs ≥320×320 input).                                    |
| `color_distribution.json`     | Per-channel color statistics fitted on COCO. Consumed by the synthetic patch generator.                                                         |
| `manifest.json`               | SHA256 hashes of every `.pt` file. Read by the prep phase to verify cache integrity.                                                            |

### 3.3 `configs/` — Three Presets

| File              | Purpose                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `quick.json`      | Smoke test. 2 P1 epochs / 1 P2 epoch / 30 eval images. Total ~13 min.                                |
| `default.json`    | Single-seed development. 20 P1 / 5 P2 epochs / 200 eval images. Total ~1 hr.                         |
| `production.json` | Three-seed publication grade. 50 P1 / 10 P2 / full APRICOT. Total ~22 hr.                            |

### 3.4 `dpc/` — Math Library (22 modules)

| Module                  | Responsibility                                                                                                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`           | Package exports; re-exports `DPCWrapper`, `DPCConfig`, `TinyUNetDenoiser`.                                              |
| `_version.py`           | `__version__ = "3.3.1"`.                                                                                                |
| `assignment.py`         | **Eq. 22 / Theorem 3.** Cost-matrix modulation. The framework's central novel mechanism.                                |
| `auxiliary_losses.py`   | Eq. 19 class-entropy regularizer, Eq. 20 box-stability loss.                                                            |
| `calibration.py`        | Eqs. 16, 17, 21, 23: objectness/class logit shifts, small-target amplifier, prediction composer.                        |
| `checkpoint.py`         | Atomic save/load. Bundles model, EMA, optimizer, scheduler, RNG state.                                                  |
| `coco_classes.py`       | COCO 80-class id-to-name map for visualization.                                                                         |
| `config.py`             | `DPCConfig` dataclass with paper-default parameters (frozen, immutable).                                                |
| `data.py`               | Datasets, collate, mixture loader (clean COCO + APRICOT + synthetic).                                                   |
| `data_cache.py`         | Cache load/save with SHA256 integrity check.                                                                            |
| `denoiser.py`           | `TinyUNetDenoiser`: encoder-decoder, 3 levels, ~1.05M params.                                                           |
| `diffusion.py`          | VP-SDE forward kernel (Eq. 5); K-probe schedule.                                                                        |
| `ema.py`                | Exponential moving average (d = 0.9999); evaluation uses EMA weights.                                                   |
| `field.py`              | **Algorithm 1.** K-probe suspicion engine: residual computation, smoothing, normalization, L1+L2 fusion.                |
| `losses.py`             | Detection loss harness; integrates the auxiliary losses.                                                                |
| `manifest.py`           | SHA256 file manifest with environment fingerprint for audits.                                                           |
| `metrics.py`            | Hide rate, FP density, mAP buckets, residual ratio, bootstrap CI.                                                       |
| `nms.py`                | Box utility math (IoU, area, format conversion). Naming retained; module supports NMS-*free* logic.                     |
| `pooling.py`            | Eq. 15: P×P ROI bilinear grid sampling (`box_pool_grid`).                                                               |
| `seeding.py`            | Deterministic seeding across torch/numpy/python; called at every phase entry.                                           |
| `synthetic_patch.py`    | Synthetic patch generator using fitted color distribution.                                                              |
| `wrapper.py`            | `DPCWrapper`: top-level inference module. Wraps YOLO26 + denoiser.                                                      |
| `yolo26_native.py`      | YOLO26 loading and fine-tuned head patching (132 head tensors, 309,656 scalars).                                        |

### 3.5 `dpcctl/` — Orchestrator

| Module                                | Responsibility                                          |
| ------------------------------------- | ------------------------------------------------------- |
| `__main__.py`                         | Module entry: enables `python -m dpcctl ...`.           |
| `cli.py`                              | Argparse-based command-line interface.                  |
| `config.py`                           | JSON config schema and resolution.                      |
| `orchestrator.py`                     | Phase scheduler with state and event bus.               |
| `events.py`                           | Event types and in-process pub/sub.                     |
| `dashboard.py`                        | Live terminal progress display.                         |
| `paths.py`                            | Run-directory layout conventions.                       |
| `phases/_base.py`                     | Abstract `Phase` class with lifecycle hooks.            |
| `phases/prep.py`                      | Cache integrity verification.                           |
| `phases/train_p1.py`                  | Denoiser training phase.                                |
| `phases/train_p2.py`                  | Joint training phase.                                   |
| `phases/diagnose.py`                  | Residual-ratio diagnostic.                              |
| `phases/eval_p3.py`                   | APRICOT evaluation with α sweep.                        |
| `phases/eval_negative_control.py`     | Clean-image control.                                    |
| `phases/aggregate.py`                 | Multi-seed aggregation with bootstrap CIs.              |

### 3.6 `tools/` — 20 CLI Scripts

Documented in detail in §6. Summary:

| Script                              | Function                                                  |
| ----------------------------------- | --------------------------------------------------------- |
| `build_caches.py`                   | Build the four `.pt` caches from raw datasets.            |
| `fit_color_distribution.py`         | Fit per-channel color statistics.                         |
| `train_phase1.py`                   | Phase 1 denoiser training entrypoint.                     |
| `train_phase2.py`                   | Phase 2 joint training entrypoint.                        |
| `evaluate_phase3.py`                | Phase 3 evaluation entrypoint.                            |
| `sweep_alpha.py`                    | Standalone alpha sweep over existing checkpoint.          |
| `compare_phases.py`                 | Side-by-side comparison of Mode A/B/C tables.             |
| `compare_raw_residuals.py`          | P1 vs P2 raw-residual comparison.                         |
| `diagnose_residuals.py`             | Residual-ratio measurement (called by diagnose phase).    |
| `diagnose_raw_residual.py`          | Raw per-pixel residual inspection.                        |
| `diagnose_untrained.py`             | Null-baseline (untrained denoiser) residual ratio.        |
| `render_panels.py`                  | Render side-by-side detection panels for paper figures.   |
| `sanity_check_data_phase1.py`       | Verify P1 data pipeline output.                           |
| `sanity_check_data_phase2.py`       | Verify P2 mixture loader.                                 |
| `sanity_check_loss_phase1.py`       | Verify P1 loss is finite, non-NaN.                        |
| `sanity_check_loss_phase2.py`       | Verify P2 `n_matched > 0`.                                |
| `sanity_check_eval.py`              | Verify single-image inference end-to-end.                 |
| `smoke_test_phase1.py`              | Run P1 inner loop on 50 steps.                            |
| `smoke_test_phase2.py`              | Run P2 inner loop on 50 steps.                            |
| `smoke_test_phase3.py`              | Run P3 evaluation on 10 images.                           |

### 3.7 `tests/` — 13 Test Modules

`conftest.py` + 13 `test_<module>.py` files. Total ~107 tests. Suite passes in ~30 s on M1 Max.

| Module                       | Coverage                                                                         |
| ---------------------------- | -------------------------------------------------------------------------------- |
| `test_assignment.py`         | Eq. 22 / Theorem 3 (recovery, suppression, Lipschitz).                           |
| `test_auxiliary_losses.py`   | Eqs. 19, 20 numerical correctness.                                               |
| `test_bootstrap.py`          | Bootstrap CI coverage probability.                                               |
| `test_calibration.py`        | Eqs. 16, 17, 21, 23 identity recovery (Theorem 5).                               |
| `test_checkpoint.py`         | Save/load round-trip; RNG state restoration.                                     |
| `test_data.py`               | Cache loading; deterministic split; mixture weights.                             |
| `test_denoiser.py`           | TinyUNet forward shapes and gradient flow.                                       |
| `test_diffusion.py`          | VP-SDE kernel; K-probe schedule.                                                 |
| `test_metrics.py`            | Hide rate, FP density, mAP on toy inputs.                                        |
| `test_nms.py`                | Box utility math.                                                                |
| `test_phase3_metrics.py`     | End-to-end metric pipeline on synthetic detections.                              |
| `test_seeding.py`            | Two runs with same seed produce identical output.                                |
| `test_synthetic_patch.py`    | Synthetic patch shape/area/color statistics.                                     |

### 3.8 `runs_quick_v33/`

The committed reference smoke run output. Layout (truncated):

```
runs_quick_v33/quick/
├── config_resolved.json           authoritative audit record for this run
├── multi_seed_summary.json
├── live/                          8 *_state.json + state.json
├── _shared/
│   ├── prep/prep_report.json
│   └── aggregate/aggregate_across_seeds.json
└── seed_42/
    ├── train_p1/checkpoints/latest/{model,ema,optim,sched,rng}.pt
    ├── train_p2/checkpoints/latest/{model,ema,head,optim,sched,rng}.pt
    ├── train_p2/yolo26_head_finetuned.pt
    ├── eval_p3/{results.json, alpha_sweep.json, per_image_metrics.jsonl}
    └── eval_negative_control/results.json
```

This is the reference output: anyone reproducing the smoke run should observe metrics in the same neighborhood and identical state-file structure.

---

## 4. Command Interface

The orchestrator is invoked as a Python module: `python -m dpcctl <command> [options]`. Five sub-commands.

### 4.1 Sub-commands

| Command    | Purpose                                                                                                   |
| ---------- | --------------------------------------------------------------------------------------------------------- |
| `validate` | Static-check a JSON configuration without running anything. Writes the resolved config to stdout.         |
| `run`      | Execute all (or a subset of) phases under a configuration.                                                |
| `resume`   | Re-attempt a failed or interrupted run. Skips completed phases.                                           |
| `status`   | Print live state of a running pipeline.                                                                   |
| `inspect`  | Dump phase output paths and metadata for a completed run.                                                 |

### 4.2 `validate`

```bash
python -m dpcctl validate -c configs/quick.json
```

Reads the JSON, applies schema validation, resolves all references (paths, env vars, defaults), and writes the resolved configuration to stdout (or to a file with `--out`). Does not execute any phase. Useful in CI to catch configuration regressions early.

### 4.3 `run`

The principal command.

```bash
python -m dpcctl run -c configs/quick.json -p all
```

**Options:**

| Flag                    | Meaning                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| `-c, --config PATH`     | Configuration JSON file (required).                                                                             |
| `-p, --phases SPEC`     | Phase selector. `all`, or comma-separated names: `prep,train_p1,train_p2,eval_p3,...`. Default `all`.           |
| `--seed N`              | Override seed list with a single seed.                                                                          |
| `--force`               | Re-run completed phases (ignores state files).                                                                  |
| `--no-dashboard`        | Disable live terminal dashboard.                                                                                |
| `--out-dir DIR`         | Override run directory.                                                                                         |
| `--device DEVICE`       | Override device (mps/cuda/cpu).                                                                                 |
| `-v, --verbose`         | Increase log verbosity.                                                                                         |

#### Phase selectors

```bash
# Run everything
python -m dpcctl run -c configs/quick.json -p all

# Only training phases (no eval)
python -m dpcctl run -c configs/default.json -p prep,train_p1,train_p2

# Only eval (assumes earlier phases completed)
python -m dpcctl run -c configs/production.json -p eval_p3,aggregate

# Re-run a specific phase, ignoring state
python -m dpcctl run -c configs/quick.json -p eval_p3 --force
```

### 4.4 `resume`

```bash
python -m dpcctl resume -c configs/production.json
```

Reads each phase's `runs_*/<name>/live/<phase>_state.json`. Skips any phase with `status: "completed"`. Restarts from the first incomplete phase. Within a phase, training reloads the latest checkpoint including the saved RNG state — the resumed run produces numerically identical output to an uninterrupted one.

### 4.5 `status` and `inspect`

```bash
# Live pipeline state during a running operation
python -m dpcctl status -c configs/quick.json

# Dump artifacts of a completed run
python -m dpcctl inspect -c configs/quick.json
```

`status` reads the state files and prints a tabular summary. `inspect` also enumerates the output paths (checkpoints, JSON reports, manifests) for each phase.

### 4.6 Orchestration Lifecycle

Per-phase lifecycle inside `run`:

1. **Resolve.** Look up the phase class; instantiate with the resolved configuration.
2. **Pre-flight.** `phase.preflight()`: verify inputs exist, check device availability, validate output directory writability.
3. **Skip-if-complete.** If state file says `completed` and `--force` not passed, skip and emit `phase_skipped` event.
4. **Mark started.** Write `state.json` with `status: "running"` and a wall-clock start timestamp.
5. **Execute.** `phase.run()`.
6. **Mark completed.** On success, update state with `status: "completed"` and end timestamp.
7. **On failure.** Catch the exception, write the stack trace to the state file under `error`, re-raise. Subsequent runs can `resume`.

### 4.7 Direct Tool Invocation

For development or debugging, individual scripts under `tools/` can be invoked directly without the orchestrator:

```bash
python tools/train_phase1.py \
  --cache caches/coco_train2017_128.pt \
  --val-cache caches/apricot_val_128.pt \
  --out-dir runs_dev/p1_test \
  --epochs 2 --batch-size 32 --lr 1e-4
```

The user manages sequencing. Outputs go to `--out-dir` rather than under the orchestrator's run-directory convention.

### 4.8 State and Event System

**State files.** Per-phase `live/<phase>_state.json` with status, start/end timestamps, metric snapshots, error stack traces. Schema example:

```json
{
  "phase": "train_p1",
  "status": "completed",
  "seed": 42,
  "started_at": "2026-05-12T09:23:11Z",
  "completed_at": "2026-05-12T09:25:36Z",
  "wall_clock_seconds": 145.2,
  "config_hash": "sha256:abc...",
  "metrics_snapshot": {
    "final_train_loss": 0.0234,
    "val_residual_ratio": 0.92
  }
}
```

**Event bus.** `dpcctl/events.py` defines an in-process pub/sub. Event types: `pipeline_started/completed/failed`, `phase_started/skipped/completed/failed`, `step_progressed`, `metric_recorded`, `checkpoint_written`. The dashboard subscribes; correctness does not depend on the bus.

---

## 5. The Eight Phases

Each phase is implemented as a subclass of `Phase` in `dpcctl/phases/`. Execution order:
`prep` → `train_p1` → `diagnose_p1` → `train_p2` → `diagnose_p2` → `eval_p3` → `eval_negative_control` → `aggregate`.

### 5.1 Phase: `prep`

**Module.** `dpcctl/phases/prep.py`.

**Objective.** Verify caches and weights are present and uncorrupted before training begins.

**Inputs.** `caches/manifest.json`, four `.pt` caches, `color_distribution.json`, YOLO26n weights.

**Flow.**

1. Iterate `caches/manifest.json`; SHA256-check every cache file.
2. Auto-regenerate `color_distribution.json` if missing via `tools/fit_color_distribution.py`.
3. Load YOLO26 to verify file integrity and head structure (132 tensors / 309,656 scalars).
4. Write `_shared/prep/prep_report.json`.

**Outputs.** `prep_report.json`; updated `live/prep_state.json`.

**Smoke wall-clock.** ~3.7 minutes (includes first-time YOLO26 download).

**Failure modes.** `CacheIntegrityError` (rebuild via `tools/build_caches.py`); `ModelStructureError` (Ultralytics version mismatch — confirm version against `requirements.txt`).

### 5.2 Phase: `train_p1`

**Module.** `dpcctl/phases/train_p1.py` → `tools/train_phase1.py`.

**Objective.** Train the TinyUNet denoiser on cached COCO with ε-prediction loss.

**Loss.**

$$\mathcal{L}_\varepsilon(\theta) = \mathbb{E}_{x_0, t, \varepsilon}\|\varepsilon_\theta(\sqrt{\alpha(t)} x_0 + \sigma(t) \varepsilon, t) - \varepsilon\|^2$$

Optionally combined with patch-mask field-supervision BCE when annotations are available.

**Per-batch flow.** Sample $t \sim \text{Uniform}(\varepsilon_{\min}, 1)$; perturb via `forward_perturb`; predict noise; MSE; backprop; AdamW step; EMA update on denoiser weights.

**Outputs.**

```
seed_N/train_p1/checkpoints/latest/{model,ema,optim,sched,rng}.pt
seed_N/train_p1/training_log.json     per-step loss values
seed_N/train_p1/manifest.json         SHA256 + env fingerprint
```

**Wall-clock.** 2.4 min smoke; 8–12 hr/seed production.

**Smoke evidence.** Validation residual ratio shifts from ~0.92 to ~0.59 over 2 epochs — denoiser is learning the data distribution.

### 5.3 Phases: `diagnose_p1` and `diagnose_p2`

**Module.** `dpcctl/phases/diagnose.py`. Runs twice: after `train_p1` and after `train_p2`.

**Objective.** Quantify the suspicion field's discriminative power between clean and patched pixels.

**Metric.** Residual ratio:

$$r = \frac{\text{median}\{R(u,v): (u,v) \in \text{patch region}\}}{\text{median}\{R(u,v): (u,v) \notin \text{patch region}\}}$$

$r > 1$ means the field is discriminating; $r \gg 1$ indicates strong discrimination.

**Flow.** Load EMA denoiser, iterate validation cache, compute $R_{\text{deploy}}$ per image, measure ratios over the patch mask.

**Outputs.** `seed_N/diagnose_p{1,2}/diagnose_report.json` with median, per-image distribution, and a CI over the validation set.

**Smoke evidence.** Median ratio 1.002 with CI [1.000, 1.004] from 873 validation images — small but consistent signal at smoke scale (full data needed for clean separation).

### 5.4 Phase: `train_p2`

**Module.** `dpcctl/phases/train_p2.py` → `tools/train_phase2.py`.

**Objective.** Joint fine-tuning of the denoiser and the YOLO26 head under the unified loss with modulated Hungarian assignment. This is the central training phase for the framework.

**Inputs.**

- Phase 1 denoiser checkpoint: `train_p1/checkpoints/latest/ema.pt`.
- All three training caches: COCO train, APRICOT train, color distribution.
- YOLO26n base weights.

**Per-batch flow.**

1. Sample from mixture loader (`dpc/data.py::MixtureDataset`): clean COCO + real APRICOT + synthetic patches according to `data.mixture_weights`.
2. Compute $R_{\text{deploy}}$ via `dpc/field.py::compute_field`.
3. Forward YOLO26 detector head; extract raw logits, predicted boxes, and base cost matrix $C_{\text{base}}$.
4. Box-pool $R$ against predicted boxes → per-prediction $\beta_i$ (`dpc/pooling.py`).
5. Apply small-target amplification (Eq. 21).
6. Augment cost matrix via Eq. 22: $C_{\text{DPC}} = C_{\text{base}} + \lambda_{\text{match}} \beta_i^{(\text{small})}$.
7. Run Hungarian assignment on $C_{\text{DPC}}$. **Record `n_matched`.**
8. Standard detection loss against modulated assignment.
9. Auxiliary losses: $\mathcal{L}_{\text{cls-prior}}$ (Eq. 19), $\mathcal{L}_{\text{box-stab}}$ (Eq. 20).
10. Backprop through the full stack; optimizer step; EMA update on both denoiser and head.

**Outputs.**

```
seed_N/train_p2/checkpoints/latest/{model,ema,head,optim,sched,rng}.pt
seed_N/train_p2/yolo26_head_finetuned.pt   packaged for deployment
seed_N/train_p2/training_log.json           includes per-batch n_matched
seed_N/train_p2/manifest.json
```

**Wall-clock.** ~7 minutes smoke; 10–14 hours per seed in production.

**Critical validation signal.** The instrumented `n_matched` value (the number of matched prediction–ground-truth pairs per batch) ranges over [3, 60] per batch in the validated smoke run. This is the v3.3.1 release's central correctness signal: it confirms Eq. 22 fires on real ground-truth labels and not on synthetic placeholders.

### 5.5 Phase: `eval_p3`

**Module.** `dpcctl/phases/eval_p3.py` → `tools/evaluate_phase3.py` and `tools/sweep_alpha.py`.

**Objective.** Quantify defense effectiveness on APRICOT across the alpha sweep.

**Flow.**

1. Load Phase 2 EMA denoiser and fine-tuned head.
2. Construct `DPCWrapper` at the configured device.
3. For each α in `alpha_sweep` (default {0.5, 1.0, 1.5, 2.0, 2.5}):
   - Set `cfg.lambda_match = alpha * base`.
   - Iterate `caches/apricot_eval_320.pt`.
   - Per image: compute Mode A (baseline YOLO26), Mode B (DPC-defended), Mode C (denoise-then-detect).
   - Record hide rate, false-positive density, mAP buckets, latency.
4. Write per-α results to `alpha_sweep.json`.

**Outputs.** `results.json`, `alpha_sweep.json`, `per_image_metrics.jsonl` (one JSON object per image for forensic inspection).

**Wall-clock.** 1.4 min smoke (30 images × 5 alphas); 30–60 min/seed production.

### 5.6 Phase: `eval_negative_control`

**Module.** `dpcctl/phases/eval_negative_control.py`.

**Objective.** Empirical validation of Theorem 5 (clean accuracy preserved). Same evaluation as `eval_p3` but on clean COCO images with no patches.

**Reports.**

- Mode A clean mAP — baseline detector performance.
- Mode B clean mAP — DPC-defended performance.
- Mode B / Mode A ratio — should be ≈ 1.0.

Substantial degradation here invalidates the defense for production use: any defense that breaks clean accuracy is not deployable.

**Wall-clock.** 0.4 min smoke.

### 5.7 Phase: `aggregate`

**Module.** `dpcctl/phases/aggregate.py`.

**Objective.** Combine per-seed results into cross-seed statistics with bootstrap 95% CIs.

**Flow.**

1. Glob all `seed_*/eval_p3/results.json` and `seed_*/eval_negative_control/results.json`.
2. For each metric: compute mean, median, std.
3. Bootstrap 95% percentile CI with `n_boot = 10000`.
4. Compute Cohen's d effect size for Mode B vs Mode A per metric.
5. Write `_shared/aggregate/aggregate_across_seeds.json` and a paper-table-ready `summary.md`.

**Wall-clock.** <0.1 min.

**Output is the paper-publishable record.** Headline excerpt from a successful production run:

```json
{
  "metrics": {
    "hide_rate": {
      "mode_a": {"mean": 0.42, "ci_95": [0.39, 0.45]},
      "mode_b": {"mean": 0.18, "ci_95": [0.16, 0.20]}
    },
    "false_positives_per_image": { "...": "..." },
    "map_50_95": { "...": "..." },
    "latency_ms": { "...": "..." }
  }
}
```

---

## 6. Tools (All 20 Scripts)

`tools/` contains twenty CLI scripts. Some are phase entrypoints invoked by the orchestrator; others are standalone diagnostics, sanity checks, smoke tests, and visualization aids. All are runnable directly.

### 6.1 Cache Generation

#### `tools/build_caches.py`

Builds the four `.pt` caches from raw COCO and APRICOT directories. Run once after dataset acquisition.

```bash
python tools/build_caches.py \
  --coco-root /path/to/coco/ \
  --apricot-root /path/to/apricot/ \
  --out-dir caches/ \
  --resolution 128 320 \
  --workers 8
```

**What it does:**

1. Reads COCO 2017 train and val. For each image: resize to requested resolution(s), convert to `uint8` tensor, attach COCO annotation.
2. Reads APRICOT images and annotations. Same conversion.
3. Saves each subset as `.pt` with keys `images`, `paths`, `annotations`.
4. Computes SHA256 of every output and writes `caches/manifest.json`.
5. Invokes `fit_color_distribution.py` to produce `color_distribution.json`.

#### `tools/fit_color_distribution.py`

Fits per-channel color statistics on cached COCO. Writes `caches/color_distribution.json`. Auto-invoked by the prep phase if the file is missing.

```bash
python tools/fit_color_distribution.py \
  --cache caches/coco_train2017_128.pt \
  --out caches/color_distribution.json
```

### 6.2 Training Entrypoints

#### `tools/train_phase1.py`

Phase 1 denoiser training. Invoked by `dpcctl/phases/train_p1.py` or directly:

```bash
python tools/train_phase1.py \
  --cache caches/coco_train2017_128.pt \
  --val-cache caches/apricot_val_128.pt \
  --out-dir runs_dev/p1_test \
  --epochs 2 --batch-size 32 --lr 1e-4
```

Trains TinyUNet with ε-prediction loss. Per-step EMA update. Atomic checkpoint per epoch.

#### `tools/train_phase2.py`

Phase 2 joint training. Same invocation pattern, but additionally requires a Phase 1 denoiser checkpoint:

```bash
python tools/train_phase2.py \
  --cache caches/coco_train2017_128.pt \
  --apricot caches/apricot_train_128.pt \
  --p1-checkpoint runs_dev/p1_test/checkpoints/latest \
  --out-dir runs_dev/p2_test \
  --epochs 1 --batch-size 16 --lr 5e-5
```

Performs the modulated Hungarian assignment in the inner loop. Logs `n_matched` per batch.

### 6.3 Evaluation

#### `tools/evaluate_phase3.py`

Phase 3 evaluation with the full alpha sweep. Produces `results.json`, `alpha_sweep.json`, `per_image_metrics.jsonl`.

```bash
python tools/evaluate_phase3.py \
  --checkpoint runs_dev/p2_test/checkpoints/latest \
  --eval-cache caches/apricot_eval_320.pt \
  --out-dir runs_dev/p3_test \
  --alphas 0.5,1.0,1.5,2.0,2.5
```

#### `tools/sweep_alpha.py`

Standalone alpha sweep over an existing trained model. Useful when you want to re-run evaluation with a different sweep grid without re-training.

```bash
python tools/sweep_alpha.py \
  --checkpoint runs_dev/p2_test/checkpoints/latest \
  --eval-cache caches/apricot_eval_320.pt \
  --alphas 0,0.25,0.5,1.0,1.5,2.0,3.0,5.0 \
  --out-dir runs_dev/p3_dense_sweep
```

#### `tools/compare_phases.py`

Cross-phase comparison. Reads each phase's `results.json` and emits side-by-side metric tables (Mode A baseline vs Mode B DPC-defended vs Mode C denoise-then-detect) in markdown.

```bash
python tools/compare_phases.py \
  --p1-results runs_a_v33/.../aggregate.json \
  --p2-results runs_b_v33/.../aggregate.json \
  --out comparison.md
```

### 6.4 Diagnostics

#### `tools/diagnose_residuals.py`

Measures residual ratio on the validation cache. Invoked by the diagnose phase but runnable standalone for ad-hoc inspection.

#### `tools/diagnose_raw_residual.py`

Operates on raw (un-normalized, un-smoothed) per-pixel residuals. Use this to sanity-check the underlying signal before aggregation when the smoothed residual ratio is anomalous.

#### `tools/diagnose_untrained.py`

Computes residual ratio with an *untrained* (randomly initialized) denoiser. Establishes the null baseline against which trained-denoiser improvements are measured.

#### `tools/compare_raw_residuals.py`

Side-by-side comparison of raw residuals between Phase 1 and Phase 2 checkpoints. Quantifies how much joint training shifted the denoiser's residual behavior.

### 6.5 Sanity Checks

These are quick (<30s) verifications to run before launching long training operations.

#### `tools/sanity_check_data_phase1.py`

Loads a single batch from the Phase 1 data pipeline. Verifies tensor shapes, dtype, value ranges, label integrity. Run after any cache modification.

#### `tools/sanity_check_data_phase2.py`

Same for Phase 2 (the mixture loader). Verifies the patch-injection logic produces correctly-mixed batches with the configured weights.

#### `tools/sanity_check_loss_phase1.py`

Forwards a single batch through Phase 1 training, computes the loss, asserts it is finite and within an expected magnitude range. Run after any configuration edit.

#### `tools/sanity_check_loss_phase2.py`

Same for Phase 2. **Additionally verifies `n_matched > 0`** (the Hungarian assignment is firing) and that `n_matched` is within [1, 200]. This is the single most useful diagnostic when the defense appears inactive.

#### `tools/sanity_check_eval.py`

Forwards a single image through the full `DPCWrapper` pipeline, verifies outputs (boxes, scores, classes, beta), prints shapes and ranges. Run before launching evaluation.

### 6.6 Smoke Tests

#### `tools/smoke_test_phase1.py`

Runs Phase 1 inner loop for 50 steps on a small subset. Verifies loss decreases. <1 minute.

#### `tools/smoke_test_phase2.py`

Runs Phase 2 inner loop for 50 steps. Additionally asserts `n_matched > 0` for ≥80% of batches.

#### `tools/smoke_test_phase3.py`

End-to-end smoke test of Phase 3 on 10 images. Verifies all metrics are computed without errors.

### 6.7 Visualization

#### `tools/render_panels.py`

Renders side-by-side debugging panels for each image: original → suspicion field heatmap → Mode A detections → Mode B detections → ground truth.

```bash
python tools/render_panels.py \
  --checkpoint runs_production_v33/production/seed_42/train_p2/checkpoints/latest \
  --image-indices 0,5,10,15,20 \
  --out-dir panels/
```

Outputs JPEG composites suitable for paper figures and presentations.

### 6.8 Quick Reference: Tool by Use Case

| Use case                                       | Tool                                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| Set up caches the first time                   | `build_caches.py`                                                           |
| Train P1 standalone                            | `train_phase1.py`                                                           |
| Train P2 standalone                            | `train_phase2.py`                                                           |
| Evaluate with full sweep                       | `evaluate_phase3.py`                                                        |
| Re-evaluate with a different grid              | `sweep_alpha.py`                                                            |
| Compare runs                                   | `compare_phases.py`                                                         |
| Before launching long training                 | `sanity_check_loss_phase{1,2}.py`                                           |
| After changing the data pipeline               | `sanity_check_data_phase{1,2}.py`                                           |
| Before launching evaluation                    | `sanity_check_eval.py`                                                      |
| Quick "does it still work" check               | `smoke_test_phase{1,2,3}.py`                                                |
| Inspect why the defense seems inactive         | `diagnose_residuals.py`, `diagnose_raw_residual.py`                         |
| Compare P1 vs P2 denoiser behavior             | `compare_raw_residuals.py`                                                  |
| Establish null baseline                        | `diagnose_untrained.py`                                                     |
| Produce paper figures                          | `render_panels.py`                                                          |

---

## 7. Configuration Reference

The orchestrator reads a single JSON configuration file per run. This section documents every section and every parameter.

### 7.1 Top-Level Structure

```json
{
  "name": "<run-name>",
  "seeds": [42],
  "device": "mps",
  "data":        { },
  "model":       { },
  "training":    { },
  "diffusion":   { },
  "calibration": { },
  "evaluation":  { },
  "alpha_sweep": [0.5, 1.0, 1.5, 2.0, 2.5],
  "logging":     { }
}
```

| Key            | Type | Purpose                                              |
| -------------- | ---- | ---------------------------------------------------- |
| `name`         | str  | Run identifier; used as the run-directory name.      |
| `seeds`        | list | Seeds to execute. Production: [42, 1337, 2718].      |
| `device`       | str  | `mps`, `cuda`, or `cpu`.                             |
| `alpha_sweep`  | list | λ_match multipliers for Phase 3.                     |

### 7.2 `data`

```json
"data": {
  "coco_cache":          "caches/coco_train2017_128.pt",
  "apricot_train_cache": "caches/apricot_train_128.pt",
  "apricot_val_cache":   "caches/apricot_val_128.pt",
  "apricot_eval_cache":  "caches/apricot_eval_320.pt",
  "color_distribution":  "caches/color_distribution.json",
  "mixture_weights": {
    "coco_clean":   0.5,
    "apricot_real": 0.3,
    "synthetic":    0.2
  }
}
```

**`mixture_weights`** must sum to 1.0 (validated at config resolution). `coco_clean` biases toward Theorem 5's clean preservation; `apricot_real` provides real adversarial signal; `synthetic` provides distributional diversity.

### 7.3 `model`

```json
"model": {
  "denoiser_channels": [32, 64, 128],
  "denoiser_time_emb_dim": 128,
  "yolo26_weights": "yolo26n.pt",
  "yolo26_finetune_head": true
}
```

**`denoiser_channels`.** Channel widths per U-Net level. `quick`: [16,32,64]. Others: [32,64,128]. Larger widths yield better denoising at the cost of latency.

**`yolo26_finetune_head`.** `true`: Phase 2 fine-tunes the head jointly (Stage 3 deployment). `false`: Phase 2 trains only the denoiser (Stage 2 inference-only deployment).

### 7.4 `training`

```json
"training": {
  "p1": {
    "epochs": 2, "batch_size": 32, "lr": 1e-4,
    "optimizer": "adamw", "ema_decay": 0.9999
  },
  "p2": {
    "epochs": 1, "batch_size": 16, "lr": 5e-5,
    "head_lr_multiplier": 0.1,
    "optimizer": "adamw", "ema_decay": 0.9999,
    "musgd_compat": true
  }
}
```

**`head_lr_multiplier`.** Fine-tunes the YOLO26 head at 1/10 the denoiser's LR. Prevents destabilization of the pre-trained head. Lower further (e.g., 0.05) if head loss oscillates.

**`musgd_compat`.** When `true`, head updates are routed through a MuSGD-compatible wrapper consistent with the YOLO26 training recipe. Default is AdamW for both.

### 7.5 `diffusion`

```json
"diffusion": {
  "n_probes": 8,
  "probe_resolution": [128, 128],
  "t_min": 0.05, "t_max": 0.50,
  "beta_min": 1e-4, "beta_max": 0.02,
  "sigma_smooth": 1.5,
  "smooth_kernel": 7,
  "fusion_mode": "hybrid",
  "fusion_w1": 0.5,
  "fusion_w2": 0.5
}
```

**`n_probes`.** $K$ in the paper. Variance of the suspicion field reduces as $O(1/K)$ per Theorem 2. Default 8.

**`probe_resolution`.** Denoiser operates at this resolution. Cost scales as $h \times w$. Keep at 128×128 unless finer localization is required.

**`fusion_mode`.** `"l1"`, `"l2"`, or `"hybrid"`. Hybrid is recommended; pure L1 or L2 exposes Proposition 1's sensitivity asymmetry.

### 7.6 `calibration`

```json
"calibration": {
  "lambda_obj":       1.0,
  "lambda_cls":       1.0,
  "lambda_match":     1.5,
  "lambda_cls_prior": 0.05,
  "lambda_box_stab":  0.05,
  "lambda_small":     1.0,
  "a_min":            0.01,
  "pool_grid":        7
}
```

**`lambda_match`.** The single most consequential parameter. Sets the baseline match-modulation strength; the Phase 3 alpha sweep evaluates a range around this value. Theorem 3(b)'s suppression guarantee requires $\lambda_{\text{match}}(\tau - \tau') > \gamma_k(R)$.

**`lambda_obj`, `lambda_cls`.** Per-prediction logit shifts (Eqs. 16, 17). Raise to make the defense more aggressive at suppressing suspect detections; lower if the negative control shows clean-accuracy degradation > 1% mAP.

**`lambda_small`.** Amplifies the suspicion coefficient for small predictions (area below `a_min` of the image). YOLO26's small-target-aware label assignment makes small predictions more soak-vulnerable; this compensates.

**`a_min`.** Fraction of image area below which a prediction is treated as "small". Default 0.01 = 1% of image area.

**`pool_grid`.** $P$ in Eq. 15. ROI sampler uses a $P \times P$ grid. Larger values increase precision at quadratic compute cost. $P = 7$ is the empirical sweet spot.

### 7.7 `evaluation`

```json
"evaluation": {
  "iou_thresholds": [0.5, 0.75, 0.9],
  "small_threshold": 0.01,
  "max_dets": 300,
  "bootstrap_n": 10000,
  "bootstrap_alpha": 0.05
}
```

### 7.8 `alpha_sweep`

```json
"alpha_sweep": [0.5, 1.0, 1.5, 2.0, 2.5]
```

Multipliers on `lambda_match` for Phase 3. Produces the canonical hide-rate-vs-FP-density trade-off curve. For paper figures, use a denser grid:

```json
"alpha_sweep": [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
```

The α = 0 value verifies Theorem 3(a) (recovery property) empirically.

### 7.9 Preset Comparison

|                       | **quick**       | **default**     | **production**     | **Notes**                |
| --------------------- | --------------- | --------------- | ------------------ | ------------------------ |
| Seeds                 | [42]            | [42]            | [42, 1337, 2718]   |                          |
| P1 epochs             | 2               | 20              | 50                 |                          |
| P1 batch size         | 32              | 64              | 96                 | Dominant P1 memory       |
| P2 epochs             | 1               | 5               | 10                 |                          |
| P2 batch size         | 16              | 24              | 32                 | Dominant overall memory  |
| Eval images           | 30              | 200             | full APRICOT       |                          |
| Denoiser channels     | [16,32,64]      | [32,64,128]     | [32,64,128]        |                          |
| K probes              | 8               | 8               | 8                  |                          |
| Probe resolution      | 128             | 128             | 128                |                          |
| Wall-clock (M1 Max)   | ~13 min         | ~1 hr           | ~22 hr             | Total across all seeds   |

### 7.10 Configuration Resolution

The orchestrator processes the JSON once at startup:

1. Read raw JSON.
2. Apply dataclass defaults from `DPCConfig`.
3. Expand environment variables (`$DPC_CACHE_DIR`, etc.).
4. Resolve relative paths against the config file's directory.
5. Validate types and ranges (`n_probes >= 1`, etc.).
6. Compute SHA256 of the canonical JSON serialization for the manifest.
7. Write `runs_*/<name>/config_resolved.json` — the authoritative audit record.

> 📝 **Note.** Editing the source config mid-run does not affect a running pipeline. The orchestrator reads the source config once; all subsequent decisions consult the resolved config in memory.

---

## 8. End-to-End Walkthrough

This section is the canonical sequence from a fresh machine to a publishable evaluation result. Every command has been verified on the reference hardware (M1 Max).

### Step 1 — Environment

```bash
git clone <repo-url> dpcroot
cd dpcroot
python3.12 --version             # expect Python 3.12.13
./bootstrap.sh ~/dpc-v33-venv
source ~/dpc-v33-venv/bin/activate
python -c "import dpc; print(dpc.__version__)"
# expect: 3.3.1
python -m pytest tests/ -q
# expect: all 13 test modules pass
```

### Step 2 — Verify Caches

The shipped `caches/` directory contains pre-built tensors for smoke. For full-scale runs on your own data, rebuild:

```bash
python tools/build_caches.py \
  --coco-root /data/coco/ \
  --apricot-root /data/apricot/ \
  --out-dir caches/ \
  --resolution 128 320
```

This populates the four `.pt` caches and `manifest.json`, plus `color_distribution.json` via the auto-invocation of `fit_color_distribution.py`.

### Step 3 — Validate the Configuration

```bash
python -m dpcctl validate -c configs/quick.json
```

Prints the resolved configuration. No errors expected. If errors appear, fix them before proceeding — running a misconfigured pipeline is the easiest way to waste hours.

### Step 4 — Smoke Run

```bash
python -m dpcctl run -c configs/quick.json -p all
```

Expected output (truncated):

```
[phase prep]                 PASS   3.7 min
[phase train_p1]             PASS   2.4 min
  val_ratio 0.92 -> 0.59 over 2 epochs
[phase diagnose_p1]          PASS   1.4 min
  median_ratio 1.002 (CI [1.000, 1.004])
[phase train_p2]             PASS   7.0 min
  n_matched range [3, 60] per batch -- Eq. 22 firing
[phase diagnose_p2]          PASS   1.4 min
[phase eval_p3]              PASS   1.4 min
[phase eval_negative_control] PASS   0.4 min
[phase aggregate]            PASS   <0.1 min
Total: 13.3 min wall-clock
```

### Step 5 — Inspect Smoke Outputs

```bash
# Top-level state
cat runs_quick_v33/quick/live/state.json

# Phase 2 n_matched distribution
python -c "
import json
with open('runs_quick_v33/quick/seed_42/train_p2/training_log.json') as f:
    log = json.load(f)
n = [e['n_matched'] for e in log['entries']]
print(f'min={min(n)} max={max(n)} median={sorted(n)[len(n)//2]}')
"
# expect: min=3 max=60 median ~25-35

# Aggregate report
cat runs_quick_v33/quick/_shared/aggregate/aggregate_across_seeds.json
```

> 💡 **Tip.** `n_matched > 0` on ≥80% of batches is the v3.3.1 release's central correctness signal. If you see `n_matched = 0` everywhere, jump straight to §10's first entry.

### Step 6 — Single-Seed Development Run

```bash
python -m dpcctl run -c configs/default.json -p all
```

~1 hour wall-clock. Uses one seed (42), full caches, moderate epoch count. Suitable for iterating on calibration constants.

### Step 7 — Production Multi-Seed Run

```bash
python -m dpcctl run -c configs/production.json -p all
```

~22 hours on M1 Max. Three seeds. After completion, `_shared/aggregate/aggregate_across_seeds.json` contains bootstrap 95% CIs ready for paper tables.

> 💡 **Tip.** Before launching the 22-hour run, run the 13-minute smoke first. If the smoke fails, the production run will fail for the same reason — and you'll lose far less wall-clock catching it early.

### Step 8 — Visualize Detections

```bash
python tools/render_panels.py \
  --checkpoint runs_production_v33/production/seed_42/train_p2/checkpoints/latest \
  --image-indices 0,5,10,15,20,25,30 \
  --out-dir panels_production/
```

Produces composite JPEG panels showing: original image → suspicion field heatmap → Mode A detections → Mode B detections → ground truth.

### Step 9 — Interpret the Aggregate Report

A successful production run produces a table of this form:

| Metric                       | Mode A (baseline)   | Mode B (DPC-defended) |
| ---------------------------- | ------------------- | --------------------- |
| hide_rate                    | 0.42 [0.39, 0.45]   | 0.18 [0.16, 0.20]     |
| false_positives_per_image    | 1.8 [1.5, 2.1]      | 0.6 [0.5, 0.8]        |
| mAP_50:95                    | 0.31 [0.29, 0.33]   | 0.45 [0.43, 0.47]     |
| latency_ms (median)          | 39.0                | 80.2 (2.06×)          |

**Healthy ranges:**

- Hide-rate Mode B / Mode A: [0.30, 0.50].
- FP-density ratio: [0.20, 0.50].
- mAP gain: [+0.05, +0.15] units.
- Clean mAP degradation (from `eval_negative_control`): ≤ 1%.
- Latency multiplier: [1.8×, 2.2×].

Values substantially outside these ranges suggest a configuration error or a methodological issue. Use §10 to diagnose.

### Step 10 — Library-Mode Deployment

For inference deployment without the orchestrator:

```python
import torch
from dpc.wrapper import DPCWrapper
from dpc.denoiser import TinyUNetDenoiser
from dpc.config import DPCConfig
from dpc.yolo26_native import load_yolo26, patch_finetuned_head

cfg = DPCConfig()
device = torch.device("mps")

denoiser = TinyUNetDenoiser().to(device)
denoiser.load_state_dict(torch.load(
    "runs_production_v33/production/seed_42/train_p2/"
    "checkpoints/latest/ema.pt"))

yolo26 = load_yolo26("yolo26n.pt", device)
patch_finetuned_head(yolo26,
    "runs_production_v33/production/seed_42/train_p2/"
    "yolo26_head_finetuned.pt")

defense = DPCWrapper(yolo26, denoiser, cfg).to(device).eval()

with torch.no_grad():
    detections = defense(image)
# detections: boxes_xyxy, scores, classes, beta
```

### Pre-Long-Run Sanity Sequence

Before launching the 22-hour production run, run these six commands. If all pass, the production run will not fail for configuration, environment, or pipeline reasons:

```bash
python -m dpcctl validate -c configs/production.json
python -m pytest tests/ -q
python tools/smoke_test_phase1.py
python tools/smoke_test_phase2.py
python tools/sanity_check_loss_phase2.py
python -m dpcctl run -c configs/quick.json -p all
```

The first four are quick (<2 min each). The last is the 13-minute smoke. Hardware-induced failures (OOM, MPS edge cases) remain possible; the resume mechanism handles them.

---

## 9. Tuning

All knobs live in the JSON configuration. Defaults match the paper's specifications and are appropriate for most deployments. Tune one parameter at a time and re-run the smoke test after each change.

### 9.1 Recommended Tuning Order

1. `lambda_match` (alpha sweep, then deployment choice).
2. `lambda_obj` and `lambda_cls` (calibration aggressiveness).
3. `lambda_small` + `a_min` (small-target compensation).
4. Mixture weights (clean-vs-adversarial training balance).
5. K and probe resolution (if latency budget allows).
6. Training intensity (epochs/LR; only if loss curves indicate undertraining).

### 9.2 `lambda_match` (Most Consequential)

Controls Equation 22's cost-matrix modulation strength. The single most consequential parameter.

#### Diagnostic decision tree

| Symptom                                          | Diagnosis and action                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Mode B hide rate ≈ Mode A                        | `lambda_match` too low. Theorem 3(b) condition not met. Raise.                                   |
| Mode B mAP < Mode A on *clean* input             | `lambda_match` too high. Legitimate predictions redirected. Lower.                                |
| Mode B unstable across seeds                     | `lambda_match` at gap boundary. Move away from the gap; recheck stability.                       |
| Mode B hide rate falls but FP density rises      | Defense over-redirects to other regions. Raise `lambda_obj`/`lambda_cls` instead.                |

#### Search procedure

1. Start at default 1.5.
2. Run alpha sweep {0.5, 1.0, 1.5, 2.0, 2.5}.
3. Plot hide-rate vs FP-density curve.
4. Choose the elbow of the curve.
5. Refine around the elbow with a finer grid if needed.

#### Deployment choice

Set `cfg.lambda_match` at deployment to the alpha that produced the best aggregate trade-off in the Phase 3 sweep — *not* necessarily the training-time default.

### 9.3 `lambda_obj` and `lambda_cls`

Per-prediction logit shifts (Eqs. 16, 17). Adjust together; they multiply the same $\beta_i^{(\text{small})}$.

| Symptom                                       | Action                              |
| --------------------------------------------- | ----------------------------------- |
| Clean mAP drop > 1%                           | Lower by 0.25.                      |
| FP density same as baseline                   | Raise by 0.25.                      |
| Hide rate *increases* under DPC               | Lower (over-suppression).           |
| β histogram bimodal but defense weak          | Raise.                              |

After each adjustment, run `eval_negative_control` to verify Theorem 5's empirical preservation holds.

### 9.4 `lambda_small` + `a_min`

Compensate for YOLO26's small-target label assignment.

**Defaults.** $\lambda_{\text{small}} = 1.0$, $a_{\min} = 0.01$.

**When to raise `lambda_small`.** If FP density on small objects substantially exceeds FP density on large objects, the soaking attack is winning on small targets specifically. Raising `lambda_small` to 1.5 or 2.0 amplifies suspicion for small predictions.

**When to lower `a_min`.** If the training data has many micro-objects (e.g., distant objects in surveillance footage), set `a_min = 0.005` so the amplification kicks in at smaller areas.

### 9.5 Data Mixture Weights

Phase 2 training distribution. Must sum to 1.0.

**Default.** `coco_clean: 0.5, apricot_real: 0.3, synthetic: 0.2`.

| Symptom                                       | Action                                                                  |
| --------------------------------------------- | ----------------------------------------------------------------------- |
| Clean mAP degraded                            | Raise `coco_clean` (≥ 0.4 enforces Theorem 5 in practice).              |
| Defense weak on real patches                  | Raise `apricot_real`.                                                   |
| Defense brittle to novel patches              | Raise `synthetic`.                                                      |
| Mode B unstable across seeds                  | Add 0.05 to `coco_clean`; subtract from the noisier component.          |

### 9.6 K (Probe Count)

Theorem 2: suspicion-field variance scales as $O(1/K)$. Latency is linear in $K$.

| K  | Trade-off                                                                                                                |
| -- | ------------------------------------------------------------------------------------------------------------------------ |
| 4  | Latency-critical; small accuracy cost.                                                                                   |
| 8  | **Default, validated.**                                                                                                  |
| 16 | Smoother fields; 2× latency. Worth it when adaptive evaluation produces noisy results at K = 8.                          |

### 9.7 Probe Resolution

Default 128×128. Cost scales as $h \times w$.

- Higher (192×192, 256×256): finer patch localization. Useful when patches are small (ρ < 0.01). Roughly quadratic latency cost.
- Lower (96×96, 64×64): latency-constrained edge deployment. At 64×64, patch localization degrades significantly.

### 9.8 Training Intensity

**Phase 1.** If `p1` val residual ratio < 1.0 after the default 2 epochs: undertrained. Increase epochs to 5–10. If still no discrimination at 20 epochs, inspect the val cache for patch annotations.

**Phase 2.** If `n_matched` drifts toward 0 over training: overfitting to placeholders. Reduce `head_lr_multiplier` to 0.05. If loss oscillates, lower base `lr` from 5e-5 to 2e-5.

**Batch size and OOM.** Phase 2 has the largest memory footprint (denoiser + head + K-probe activations + both backward graphs). OOM recovery:

1. Halve `training.p2.batch_size`.
2. `python -m dpcctl resume -c <config>`. Phase 1 is preserved.
3. If still OOM, halve `p1.batch_size` too.

### 9.9 Alpha Sweep Grid

For paper-grade results, use a denser grid that includes the recovery test:

```json
"alpha_sweep": [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
```

The α = 0 case empirically verifies Theorem 3(a) (recovery): Mode B with α = 0 should equal Mode A.

### 9.10 Ablation Templates

Each is a one-line config edit + re-run. Outputs go under different `name` so multiple ablations coexist.

**Disable cost-matrix modulation.** `calibration.lambda_match: 0`. Isolates calibration vs assignment-modulation contributions.

**Inference-only deployment.** `model.yolo26_finetune_head: false`. Compares Stage 2 to Stage 3.

**Vary K.** Three configs with K ∈ {2, 8, 16}. Measures Theorem 2 operationally.

**Fusion ablation.** Three configs with `fusion_mode` ∈ {l1, l2, hybrid}. Measures Proposition 1's sensitivity asymmetry.

> 💡 **Tip.** Use `tools/compare_phases.py` to produce side-by-side tables across ablations.

### 9.11 When to Stop Tuning

The defense is deployment-ready when all five conditions hold simultaneously on the production aggregate:

1. Hide-rate Mode B / Mode A within [0.30, 0.50].
2. FP-density ratio within [0.20, 0.50].
3. mAP gain ≥ +0.05.
4. Clean mAP degradation ≤ 1%.
5. Latency multiplier ≤ 2.5×.

Further tuning beyond this point typically yields marginal gains and risks overfitting to the eval set.

---

## 10. Debugging

This section catalogs failure modes in roughly the order they appear during deployment. Each entry: symptom, cause, fix.

### 10.1 `n_matched = 0` Throughout Phase 2

**Symptom.** Training log shows zero matched (prediction, ground-truth) pairs at every step. The defense is structurally inactive.

**Causes.** (a) `lambda_match = 0` in config. (b) `model.yolo26_finetune_head: false`. (c) Ground-truth labels empty for the batch.

**Fix.**

```bash
python tools/sanity_check_loss_phase2.py
```

Asserts `n_matched > 0` and prints offending batch indices. Confirm configuration values. Rebuild caches if labels are truly empty.

### 10.2 Mode B mAP Worse Than Mode A on Clean Inputs

**Symptom.** Negative-control phase reports clean degradation > 1%.

**Cause.** Calibration over-suppresses legitimate detections.

**Fix.** Lower `lambda_obj` and `lambda_cls` in steps of 0.25. Re-run `eval_negative_control` after each step. Theorem 5 guarantees clean preservation in the limit; deviations are calibration choices, not framework bugs.

### 10.3 Mode B Hide Rate Identical to Mode A

**Symptom.** Defense not active despite training completion.

**Causes.** The suspicion field is constant (denoiser undertrained), or box-pooling reads the wrong resolution.

**Fix.**

1. Inspect training log for non-zero suspicion-field values.
2. Confirm box-pooling reads the 640×640 upsampled field, not the 128×128 probe-resolution tensor.
3. Re-run Phase 1 for more epochs if the denoiser is undertrained.

### 10.4 Validation Residual Ratio ≤ 1.0 After Phase 1

**Symptom.** Diagnose-P1 reports the denoiser is not discriminating between clean and patched pixels.

**Causes.** Insufficient training epochs, sparse patches in the validation cache, or normalization mismatch.

**Fix.**

- Increase `p1.epochs`.
- Verify val cache contains patched images: `python tools/diagnose_raw_residual.py --cache caches/apricot_val_128.pt`.
- Check that normalization constants match between training and validation.

### 10.5 Cache SHA256 Mismatch in Prep

**Symptom.** Prep aborts with `CacheIntegrityError`.

**Causes.** A cache file was modified, or `caches/manifest.json` is stale relative to the cache files.

**Fix.** Regenerate everything:

```bash
python tools/build_caches.py \
  --coco-root <p> --apricot-root <p> \
  --out-dir caches/
```

### 10.6 OOM During Phase 2

**Symptom.** `torch.cuda.OutOfMemoryError` (CUDA) or MPS memory pressure.

**Fix.**

1. Halve `training.p2.batch_size`.
2. `python -m dpcctl resume -c <config>`. Phase 1 progress is preserved.
3. If still OOM, halve `p1.batch_size` too.
4. On Apple Silicon: `export PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported MPS ops fall back to CPU rather than crash.

### 10.7 NaN in Suspicion Field

**Cause.** Per-image min-max normalization divides by zero when the field is constant across the image.

**Fix.** The code guards this with `delta=1e-6` in `dpc/field.py::normalize`. If you have modified the file, re-add the stabilizer.

### 10.8 NaN in Denoiser Output

**Cause.** Gradient explosion.

**Fix.** Add gradient clipping to the training inner loop:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0)
```

### 10.9 NaN in L2 Residual

**Cause.** Square root of zero produces NaN gradient.

**Fix.** The codebase adds `eps_num = 1e-8` inside the sqrt in `dpc/field.py`. Re-add if modified.

### 10.10 Checkpoint Corruption

**Symptom.** A checkpoint fails to deserialize.

**Cause.** Process killed mid-save before atomic rename. Rare due to the atomic-save mechanism in `dpc/checkpoint.py`.

**Fix.**

1. Check `checkpoints/epoch_<N-1>/` for a valid earlier checkpoint.
2. Copy to `checkpoints/latest/`.
3. `python -m dpcctl resume -c <config>`.

### 10.11 Shape Mismatches

**In `field.py`.** Probe resolution doesn't match denoiser input. Confirm `cfg.probe_resolution` matches the denoiser's trained resolution.

**In `pooling.py`.** Boxes in wrong format. `box_pool_grid` expects (B, N, 4) `xyxy`. YOLO26 emits `xywh` natively. Convert via `dpc.nms.xywh_to_xyxy` before pooling.

### 10.12 MPS-Specific Issues

**NotImplementedError on MPS.** Some ops not yet accelerated:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**Memory pressure.** The MPS allocator is more conservative than CUDA. Lower batch size by 50% when porting from CUDA configs.

### 10.13 Performance Bottlenecks

**Slow Phase 2 step.** Profile with `torch.profiler`:

- Verify K probes are batched (single denoiser call), not sequential.
- Lower probe resolution if appropriate.
- Increase `dataloader_workers`.

**DataLoader wait.**

- Set `dataloader_workers` to (cores - 1).
- Move caches to fast local SSD; network volumes starve loaders.
- Confirm `torch.load(..., mmap=True)` is used.

### 10.14 Diagnostic Toolchain Reference

| Use this tool                            | When                                                                |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `sanity_check_data_phase1.py`            | After cache rebuild.                                                |
| `sanity_check_data_phase2.py`            | After changing mixture weights.                                     |
| `sanity_check_loss_phase1.py`            | Before launching P1 training.                                       |
| `sanity_check_loss_phase2.py`            | Before launching P2. Asserts `n_matched > 0`.                       |
| `sanity_check_eval.py`                   | Before launching evaluation.                                        |
| `smoke_test_phase{1,2,3}.py`             | Quick verification inner loops run.                                 |
| `diagnose_raw_residual.py`               | Inspect raw residuals before smoothing/normalization.               |
| `diagnose_untrained.py`                  | Establish null baseline.                                            |
| `compare_raw_residuals.py`               | Compare P1 vs P2 residuals.                                         |
| `render_panels.py`                       | Visual inspection of detections.                                    |

### 10.15 When All Else Fails

1. Reproduce the committed `runs_quick_v33/` smoke output. If *that* fails, the environment is the problem (Python version, PyTorch wheel, MPS support).
2. Compare `config_resolved.json` of the failing run against the committed reference.
3. Compare `manifest.json` environment fingerprints between failing and reference runs.
4. Check `docs/NOTES_LOG.md` for known issues during development.

---

## 11. Production Deployment

### 11.1 Pre-Deployment Validation Checklist

Before routing production traffic to a defended endpoint, all of the following must hold:

- [ ] Production configuration completed all 8 phases with `status: "completed"` on at least one seed (preferably three).
- [ ] Aggregate report shows hide-rate reduction ≥ 30% vs Mode A baseline.
- [ ] Negative-control phase shows clean mAP degradation ≤ 1% on the same image distribution as production traffic.
- [ ] Latency multiplier in `eval_p3/results.json` is ≤ 2.5× baseline at the target batch size and resolution.
- [ ] Manifest SHA256 hashes match across two independent runs of the same configuration (confirms determinism).
- [ ] Unit-test suite (`pytest tests/`) passes on the deployment hardware.

### 11.2 Inference-Time Configuration

- Load the **EMA** denoiser weights, not the raw training weights. Path: `train_p2/checkpoints/latest/ema.pt`.
- Apply the fine-tuned head via `patch_finetuned_head` from `train_p2/yolo26_head_finetuned.pt`.
- Set `DPCConfig.lambda_match` to the alpha that produced the best aggregate trade-off in the Phase 3 sweep — not necessarily the training-time default.
- Call `wrapper.eval()` and wrap inference in `torch.no_grad()`.
- Verify the production input resolution matches the wrapper's expected size (typically 640×640 for YOLO26n).

### 11.3 Library Usage Pattern

```python
import torch
from dpc.wrapper import DPCWrapper
from dpc.denoiser import TinyUNetDenoiser
from dpc.config import DPCConfig
from dpc.yolo26_native import (
    load_yolo26, patch_finetuned_head)

# Configuration with deployment-tuned lambda_match
cfg = DPCConfig()
cfg.lambda_match = 1.75   # from Phase 3 alpha-sweep elbow

device = torch.device("mps")  # or "cuda"

# Load EMA denoiser
denoiser = TinyUNetDenoiser().to(device)
denoiser.load_state_dict(torch.load(
    "checkpoints/ema.pt", map_location=device))

# Load YOLO26 + fine-tuned head
yolo26 = load_yolo26("yolo26n.pt", device)
patch_finetuned_head(yolo26,
    "checkpoints/yolo26_head_finetuned.pt")

# Wrap and configure for inference
defense = DPCWrapper(yolo26, denoiser, cfg)
defense = defense.to(device).eval()

# Inference
with torch.no_grad():
    out = defense(image_tensor)  # (B, 3, 640, 640)

# out: boxes_xyxy, scores, classes, beta
```

The wrapper is fully differentiable end-to-end, so adaptive-evaluation attacks can backpropagate through it without modification.

### 11.4 Production Monitoring

**β_i histogram drift.** Log the per-prediction β_i histogram per batch. Sudden shifts indicate either a distribution shift in production traffic or denoiser drift.

**Latency variance.** The deterministic batch-size design implies low latency variance. Step changes usually indicate thermal throttling or compute contention rather than a defense issue.

**Suppression rate.** Track Mode B detection counts vs an undefended sentinel branch on a sampled fraction of traffic. Divergence > 50% suppression rate indicates calibration drift or upstream input corruption.

**Field statistics.** Per-image mean and max R_deploy. Anomalously low means the denoiser may be drifting; anomalously high may indicate adversarial pressure on the input distribution.

**Per-class detection counts.** Track Mode B detection counts per class against a moving baseline. A sudden drop in a specific class suggests a class-targeted attack pattern.

### 11.5 Update and Retraining Triggers

Retrain when any of:

- The production input distribution shifts substantially (new lighting conditions, new camera hardware, new scene types). Theorem 5's clean preservation depends on training-distribution coverage.
- New patch attack patterns appear in the wild that the deployed defense does not block. Add them to the APRICOT training partition, rebuild caches, retrain.
- The diagnostic residual ratio on a held-out production sample falls below 1.2 (suggests denoiser drift).

**Retraining workflow.**

1. Keep the same seeds for reproducibility comparison.
2. Rebuild caches if dataset changed.
3. Use the orchestrator's resumability to re-run only affected phases.
4. Validate the new artifact against the same pre-deployment checklist (§11.1).
5. Compare the new aggregate against the previous to confirm no regression on existing patches.

### 11.6 Edge Deployment Considerations

For latency-constrained edge deployment (mobile NPU, embedded GPU):

- Reduce `n_probes` from 8 to 4 (O(1/K) variance penalty, 2× latency saving).
- Reduce `probe_resolution` from 128 to 96. Quadratic latency saving with manageable localization degradation.
- Reduce `pool_grid` from 7 to 5. Quadratic compute saving.
- Consider mixed-precision (fp16) inference on CUDA. Not yet stable on MPS.

Validate each reduction against `eval_p3` to confirm the deployment metric envelope is still acceptable. The `tools/sweep_alpha.py` script can re-evaluate without re-training.

### 11.7 Dependency Pinning Before Deployment

```bash
pip freeze > requirements.lock.txt
```

Commit `requirements.lock.txt` alongside the deployment artifact. Floating dependency versions are the most common source of numerical drift between training and inference. The lock file captures transitive dependencies that `requirements.txt` does not.

### 11.8 Production Inference Failure Modes

**Sudden Mode B / Mode A divergence.** Most likely cause: input preprocessing pipeline drift. Verify normalization constants match between training and inference. Confirm input resolution matches the wrapper's expected size.

**Per-prediction β_i saturating at 1.0.** Suggests the denoiser is failing on the production distribution. Retrain on production-similar data or revert to a known-good checkpoint.

**Latency exceeds Theorem 4 bound.** Verify K probes are batched in a single denoiser forward pass. Verify the host system isn't thermal-throttling. Check that `torch.no_grad()` wraps inference.

**Memory growth during long-running inference.** The wrapper itself has no internal state that grows. If memory grows, the leak is elsewhere — typically in the surrounding deployment harness (Python references, MPS caching). Periodic `torch.cuda.empty_cache()` or `torch.mps.empty_cache()` usually resolves it.

### 11.9 Recommended Production Logging

For each inference call, persist at minimum:

- Input image hash (for forensic replay).
- Output detections (boxes, classes, scores).
- Per-prediction β_i values.
- Median wall-clock latency.
- Suspicion-field max and mean.

These are the minimum forensic signals for post-incident analysis. The volume is modest (few hundred bytes per inference).

---

## 12. Reproducibility and Reference

### 12.1 Reproducibility Guarantees

The v3.3.1 release provides four guarantees, each backed by a test under `tests/`:

**(1) Within-seed bit-identical determinism.** Two runs on the same hardware with the same seed produce bit-identical output. Tested in `tests/test_seeding.py`.

**(2) Cross-hardware statistical reproducibility.** Two runs on different hardware with the same seed produce statistically identical output (every metric within numerical tolerance), even though intermediate tensors may differ at floating-point precision.

**(3) Resume reproducibility.** A run interrupted and resumed from a checkpoint produces output identical to an uninterrupted one. Tested in `tests/test_checkpoint.py`.

**(4) Manifest auditability.** Every output file carries a SHA256 hash plus environment fingerprint, enabling external reproducibility audits without access to the originating compute environment.

### 12.2 Seeding Mechanism

`dpc/seeding.py::set_seed` is called at the start of every phase:

```python
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Additional per-source seeding:

- DataLoader workers: `worker_init_fn = seed * 7919 + worker_id`.
- K-probe RNG per step: `seed * 7919 + step_idx`.
- Synthetic patch generator: `(base_seed * 7919 + i) mod 2^31`.
- COCO train/val splits: deterministic by SHA-based index hash with release-specific salt.

### 12.3 Checkpoint and Manifest Formats

Each checkpoint directory contains five files saved atomically:

| File        | Contents                                                                       |
| ----------- | ------------------------------------------------------------------------------ |
| `model.pt`  | `state_dict()` of denoiser (P1) or denoiser + head (P2).                       |
| `ema.pt`    | EMA weights. Use these for inference.                                          |
| `head.pt`   | (P2 only) Fine-tuned YOLO26 head, separable for deployment.                    |
| `optim.pt`  | Optimizer state. Needed for resume.                                            |
| `sched.pt`  | LR scheduler state.                                                            |
| `rng.pt`    | Captured RNG state (torch, numpy, python, cuda).                               |

Each phase output directory has a `manifest.json` with SHA256 of every output plus the environment fingerprint (Python version, PyTorch version, platform, device, resolved-config hash).

### 12.4 Reproduction Procedure

To reproduce a published run:

1. Clone the repository at the relevant git tag.
2. Create a venv; install from `requirements.lock.txt`.
3. Place the published `config_resolved.json` at `configs/repro.json`.
4. Place the original caches (or rebuild from the same source data with the same salt).
5. `python -m dpcctl run -c configs/repro.json -p all`.
6. Compare output manifests against the published ones via SHA256.

Bit-identical reproduction on the same hardware. Statistically identical on different hardware.

### 12.5 Equation → Code Map

| Eq.    | Implementation                                                |
| ------ | ------------------------------------------------------------- |
| Eq. 5  | `dpc/diffusion.py::forward_perturb`                           |
| Eq. 10 | Implicit (Theorem 1 derivation)                               |
| Eq. 11 | `dpc/field.py::compute_field` (per-probe residual)            |
| Eq. 15 | `dpc/pooling.py::box_pool_grid`                               |
| Eq. 16 | `dpc/calibration.py::calibrate_objectness`                    |
| Eq. 17 | `dpc/calibration.py::calibrate_class_uniform`                 |
| Eq. 19 | `dpc/auxiliary_losses.py::class_entropy_regularizer`          |
| Eq. 20 | `dpc/auxiliary_losses.py::box_stability_loss`                 |
| Eq. 21 | `dpc/calibration.py::amplify_small_targets`                   |
| Eq. 22 | `dpc/assignment.py::modulate_cost_matrix`                     |
| Eq. 23 | `dpc/calibration.py::calibrate_predictions`                   |

### 12.6 Theorem → Test Map

| Theorem | Test                                                                                       |
| ------- | ------------------------------------------------------------------------------------------ |
| T1      | Empirical via residual-ratio metric; synthetic check in `test_diffusion.py`.               |
| T2      | `test_diffusion.py::test_variance_decay_with_K`.                                           |
| T3(a)   | `test_assignment.py::test_recovery_under_clean_inputs`.                                    |
| T3(b)   | `test_assignment.py::test_suppression_under_concentrated_suspicion`.                       |
| T3(c)   | `test_assignment.py::test_lipschitz_stability_in_beta`.                                    |
| T4      | `record_timings` instrumentation in `field.py`.                                            |
| T5      | `test_calibration.py` plus `eval_negative_control` phase.                                  |

### 12.7 Tensor Shape Reference

| Symbol                          | Shape              | Where                  |
| ------------------------------- | ------------------ | ---------------------- |
| Input $x_0$                     | (B, 3, H, W)       | wrapper input          |
| Probe $x_{t_k}$                 | (K, B, 3, h, w)    | `field.py`             |
| Denoiser pred $\hat\varepsilon$ | (K, B, 3, h, w)    | `denoiser.py`          |
| Per-probe residual $\Delta_k$   | (K, B, h, w)       | post channel reduction |
| Suspicion field $R$             | (B, h, w)          | `field.py` output      |
| Upsampled $I_{\text{img}}$      | (B, H, W)          | wrapper forward        |
| Box-pooled $\beta_i$            | (B, N)             | `pooling.py`           |
| Base cost $C_{\text{base}}$     | (B, N, M)          | YOLO26 matcher         |
| Modulated $C_{\text{DPC}}$      | (B, N, M)          | `assignment.py`        |
| Assignment $\pi^*$              | (B, M) indices     | Hungarian output       |

### 12.8 DPCConfig Defaults

| Field                | Default     | Range / notes                                                    |
| -------------------- | ----------- | ---------------------------------------------------------------- |
| `n_probes`           | 8           | ≥ 1. Variance O(1/K).                                            |
| `probe_resolution`   | (128, 128)  | Powers of 2 recommended.                                         |
| `t_min`              | 0.05        | > 0. Lower bound of probe schedule.                              |
| `t_max`              | 0.50        | < 1. Upper bound of schedule.                                    |
| `fusion_mode`        | "hybrid"    | "l1", "l2", or "hybrid".                                         |
| `fusion_w1, w2`      | 0.5, 0.5    | Sum to 1.0.                                                      |
| `sigma_smooth`       | 1.5         | > 0 pixels.                                                      |
| `smooth_kernel`      | 7           | Odd, ≥ 3.                                                        |
| `lambda_obj`         | 1.0         | ≥ 0. Eq. 16.                                                     |
| `lambda_cls`         | 1.0         | ≥ 0. Eq. 17.                                                     |
| `lambda_match`       | 1.5         | ≥ 0. Eq. 22. **Most consequential**.                             |
| `lambda_cls_prior`   | 0.05        | ≥ 0. Eq. 19.                                                     |
| `lambda_box_stab`    | 0.05        | ≥ 0. Eq. 20.                                                     |
| `lambda_small`       | 1.0         | ≥ 0. Eq. 21.                                                     |
| `a_min`              | 0.01        | 0–1. Image-area fraction.                                        |
| `pool_grid`          | 7           | Odd, ≥ 3. Quadratic compute.                                     |

### 12.9 What Reproducibility Cannot Guarantee

- **Floating-point determinism across architectures.** Runs on M1 Max vs CUDA H100 produce different intermediate tensors. Metrics agree within tolerance; weights differ in lowest bits.
- **Determinism under non-deterministic CUDA ops.** A small set of CUDA ops have no deterministic kernel. The codebase avoids them where possible; some are unavoidable.
- **Cross-PyTorch-version reproducibility.** A PyTorch version change can shift numerical results. Pin via `requirements.lock.txt`.
- **Cross-driver reproducibility.** CUDA/MPS driver versions can affect numerics. Recorded in the environment fingerprint for audit.

### 12.10 Acronyms

| Acronym  | Expansion                                                             |
| -------- | --------------------------------------------------------------------- |
| AP       | Average Precision                                                     |
| BCE      | Binary Cross-Entropy                                                  |
| CI       | Confidence Interval                                                   |
| COCO     | Common Objects in Context (dataset)                                   |
| DAPRICOT | Dataset of Adv. Physical Patches w/ Realistic Imaging Conditions      |
| DDP      | Distributed Data Parallel                                             |
| DPC      | Diffusion-Prior Consistency                                           |
| EMA      | Exponential Moving Average                                            |
| EOT      | Expectation over Transformation                                       |
| FP       | False Positive                                                        |
| GIoU     | Generalized Intersection over Union                                   |
| GT       | Ground Truth                                                          |
| IoU      | Intersection over Union                                               |
| mAP      | mean Average Precision                                                |
| MPS      | Metal Performance Shaders (Apple)                                     |
| MSE      | Mean Squared Error                                                    |
| NMS      | Non-Maximum Suppression                                               |
| OBB      | Oriented Bounding Box                                                 |
| ROI      | Region of Interest                                                    |
| SDE      | Stochastic Differential Equation                                      |
| SHA      | Secure Hash Algorithm                                                 |
| VP-SDE   | Variance-Preserving SDE                                               |
| YOLO     | You Only Look Once                                                    |

### 12.11 Where to Go Next

- Full 93-page research manual: module-by-module documentation, methodology derivations, complete glossary, experiment templates.
- Companion paper: five theorems, three propositions, four algorithms, adaptive evaluation protocol, threat-model boundary cases.
- `CHANGELOG.md`: canonical theorem-to-code mapping for v3.3.1.
- `runs_quick_v33/`: committed reference smoke output for comparison.
- `docs/NOTES_LOG.md`: engineering log with known issues during development.

---

<div align="center">

**DPC-YOLO26 v3.3.1 — Operational Deployment Guide**
California State University San Marcos — May 2026

</div>
