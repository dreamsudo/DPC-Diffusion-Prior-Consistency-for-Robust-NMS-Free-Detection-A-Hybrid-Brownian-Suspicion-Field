# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.3.1] — 2026-05-13

Initial public release of DPC-YOLO26 — the reference implementation of *Diffusion-Prior Consistency for Robust NMS-Free Detection*.

### Theorems implemented

The following theoretical results from the paper are operationalized in code. Each function carries an inline docstring citing the equation or theorem property it implements.

- **Theorem 1 — Residual as Score Estimator.** The per-probe residual `Δ_k = ε_θ(x_t, t) − ε_k` is implemented as an asymptotically unbiased estimator of the natural-image score function magnitude. Code: `dpc/field.py::compute_field`.

- **Theorem 2 — Multi-Probe Variance Bound.** K-probe ensemble residual achieves `Var[r̄_K] ≤ V_max / K` variance reduction. `K = 8` is the default in `dpc/config.py`. Independence enforced by per-probe noise sampling in `dpc/field.py`.

- **Theorem 3 — Assignment-Cost Modulation.** The Hungarian cost matrix is augmented by the suspicion signal (Eq. 22): `C_DPC(i, k) = C_base(i, k) + λ_match · β_i`. The three properties of the theorem — recovery under clean inputs, suppression under concentrated suspicion, and Lipschitz stability — are directly unit-tested in `tests/test_assignment.py`. Code: `dpc/assignment.py::modulate_cost_matrix` wrapping `scipy.optimize.linear_sum_assignment`.

- **Theorem 4 — Latency Bound.** End-to-end per-image latency bounded by `~2 ×` baseline detector cost under the standard configuration (`K = 8`, `h = w = 128`, hybrid fusion). Verified at production scale.

- **Theorem 5 — No-False-Suppression Under Clean Inputs.** Operational consequence of Theorem 3 property (a). `dpc/calibration.py::calibrate_predictions` reduces to the identity map when the suspicion coefficient `β = 0`.

- **Proposition 1 — L1/L2 Sensitivity Asymmetry.** Both residual operators are computed; the hybrid fusion `R_hyb = w₁·R̂_L1 + w₂·R̂_L2` is implemented in `dpc/field.py` with paper defaults `w₁ = w₂ = 0.5`.

- **Proposition 2 — Box-Pooling Continuity.** Bilinear `P × P` grid sampling (`P = 7` default) over the upsampled suspicion field. Code: `dpc/pooling.py::box_pool_grid`.

- **Proposition 3 — Multi-Task Propagation Boundedness.** Detection injection path implemented (`dpc/calibration.py` + `dpc/wrapper.py`). The four remaining task-specific injection mechanisms (segmentation, pose, OBB, classification) are specified in §6.2 of the paper and pending in a future point release.

### Equations implemented

- **Eq. 16 — Objectness calibration.** `dpc/calibration.py::calibrate_objectness`.
- **Eq. 17 — Class-logit suppression.** `dpc/calibration.py::calibrate_class_uniform`.
- **Eq. 19 — Suspicion-weighted class-entropy regularizer.** `dpc/auxiliary_losses.py::class_entropy_regularizer`.
- **Eq. 20 — Brownian box-stability loss.** `dpc/auxiliary_losses.py::box_stability_loss`.
- **Eq. 21 — Small-target amplification.** `dpc/calibration.py::amplify_small_targets`.
- **Eq. 22 — Modulated cost matrix.** `dpc/assignment.py::modulate_cost_matrix`.

### Components

- **Brownian suspicion engine** (`dpc/field.py`) — K-probe forward VP-SDE with batched denoiser pass, L1 and L2 residual operators, Gaussian smoothing, per-image min-max normalization, hybrid fusion.
- **TinyUNet denoiser** (`dpc/denoiser.py`) — compact ε-prediction U-Net trained at `128 × 128` probe resolution.
- **Native YOLO26 bridge** (`dpc/yolo26_native.py`) — loads the YOLO26 detector and applies Phase 2 fine-tuned head weights without monkey-patching.
- **DPCWrapper** (`dpc/wrapper.py`) — single `nn.Module` exposing the full inference pipeline; differentiable end-to-end for adaptive evaluation.
- **Orchestrator** (`dpcctl/`) — eight-phase pipeline: `prep`, `train_p1`, `diagnose_p1`, `train_p2`, `diagnose_p2`, `eval_p3`, `eval_negative_control`, `aggregate`. Resumable from per-phase state.
- **Multi-seed runner** — production config runs seeds `42, 1337, 2718` sequentially; aggregate phase produces bootstrap 95% confidence intervals.

### Validation

- 13 test modules in `tests/`, full suite passes.
- End-to-end smoke validated on macOS 14, M1 Max, Python 3.12.13, PyTorch 2.11, Apple MPS backend. Reference output committed under `runs_quick_v33/`.
- Modulated Hungarian assignment confirmed firing on real COCO ground truth during Phase 2: `n_matched` range `[3, 60]` per batch.

### Configurations

- `configs/quick.json` — 13-minute smoke pipeline.
- `configs/default.json` — single-seed development run (~1 hour).
- `configs/production.json` — 3-seed full evaluation (~20–24 hours).

### Reproducibility

All RNG sources seeded deterministically (`torch`, `numpy`, Python `random`, DataLoader workers, K-probe RNG, synthetic patch generator). EMA, optimizer state, scheduler state, and RNG state checkpointed atomically. Each run writes a manifest with SHA256 hashes of every output and an environment fingerprint.

---

## Versioning

- **3.3.1** — first public release.

Earlier development versions (3.x.x) were internal and are not documented here.
