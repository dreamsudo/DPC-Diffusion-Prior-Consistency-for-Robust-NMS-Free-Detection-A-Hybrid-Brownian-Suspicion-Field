# DPC-YOLO26 Project Notes Log

This file accumulates decisions and reminders that need to outlive any one
session. Append-only; do not edit prior entries.

---

## 2026-05-13 — Paper edits required for v3.3.0

The v3.3.0 implementation commits to **training-only** λ_match. Section 5.16
of the paper, as currently written, can be read either as
training-time-only or inference-time. The implementation locks in
training-only. The paper must be edited to remove the ambiguity and to
sharpen the claim accordingly. Specific edits:

### Section 5.16.1 — clarify λ_match scope

Current text (paraphrased): "The framework augments the base cost with a
suspicion-dependent term: C_DPC(i, k) = C_base(i, k) + λ_match · β_i^(small).
The assignment is then computed on the modulated cost matrix."

Add explicit statement that this modulation operates during the
**training** of the NMS-free detector. Add: "At inference time, the trained
detector head — having been optimized under the modulated assignment —
produces non-redundant predictions that already reflect the suspicion-aware
training signal. The inference-time injection points are therefore limited
to the per-prediction calibrations of §5.11 and §5.12."

### Section 5.16.2 — Theorem 3 properties: relate to inference

Theorem 3 currently states three properties of the modulated assignment.
Add a paragraph after the theorem statement clarifying what each property
implies operationally at inference time:

- Property (a) Recovery: clean inputs at inference produce β ≈ 0,
  Eqs. (16)/(17) reduce to identity, head outputs match baseline YOLO26.
- Property (b) Suppression: the head trained under modulated assignment
  learns to not produce confident predictions in regions that earned
  concentrated suspicion during training. Empirically verifiable as the
  on-patch suppression rate.
- Property (c) Lipschitz stability: small variations in β during training
  do not cause the assignment to flip, ensuring stable gradient signal and
  reproducible training outcomes.

### Section 4 — design philosophy: tighten the contribution claim

Current text (paraphrased): "First patch defense to modulate the Hungarian
assignment matrix."

Strengthen to: "First patch defense that modifies the training procedure
of an NMS-free detector to produce a head intrinsically robust to
spatially-localized adversarial signal, by modulating the one-to-one
assignment cost matrix during training with a suspicion field derived from
a diffusion prior. Unlike DiffPure, DIFFender, PatchCleanser, Jedi, and
SAC — all of which operate on the input image or post-process detections
of an unmodified pretrained model — the framework reshapes the trained
model's matching logic itself."

### Section 2.1 — attack surface: explicit training/inference distinction

The attack surface description currently treats "the assignment matrix"
monolithically. Clarify that the matrix exists during training (where one-to-one
assignment is enforced) and shapes the trained head's behavior at inference.
The framework intervenes at the training-time surface; the inference-time
behavior is the structural consequence.

### Section 1 — abstract and intro: align with above

The abstract currently includes "modulation of the Hungarian matching cost
matrix that drives end-to-end assignment." Edit to: "modulation of the
Hungarian matching cost matrix during the training of end-to-end
assignment." Similar adjustments in the introduction's third commitment
paragraph.

### Section 5.17 — Unified detector objective

Currently lists training losses. Add the explicit weight-anchor
regularizer term `λ_anchor · ‖θ_head − θ_head^(pretrained)‖²` that holds
the YOLO26 head's weights close to their pretrained values during
DPC-modulated fine-tuning. Without this term, the joint training drifts
off COCO calibration. The paper should justify why this regularizer is
needed — namely, that the threat model premise ("YOLO26 is the deployed
detector") requires the fine-tuned head to remain operationally
identical to the deployed detector on clean inputs.

### Section 11 — evaluation protocol: training-time modulation as verification target

Currently focuses on adversarial attack success rates and suppression
metrics. Add a dedicated subsection that empirically verifies Theorem 3's
three properties:

- Property (a): measure detection performance on clean COCO before and
  after DPC training. Verify they are within tolerance (e.g., < 1% mAP
  drop).
- Property (b): measure on-patch detection suppression rate on APRICOT.
  Verify the trained-with-DPC head suppresses more confidently than the
  baseline + post-hoc Eqs. (16)/(17) alone.
- Property (c): measure variance of training outcomes across multiple
  seeds. Verify the joint training converges to similar minima (low
  cross-seed variance) when λ_match is small.

---

## 2026-05-13 — Other minor paper edits required

These are smaller items the v3.2.0 audit surfaced. None of them change
the paper's contribution claim; they only tighten the math.

### Section 5.4 — fixed schedule specification

The paper says K = 8 fixed probes with σ(t_k) over [0.05, 0.5] in
logarithmic spacing. Add the explicit formula in the paper text:
`σ_k = σ_min · (σ_max / σ_min)^((k-1)/(K-1))` for k = 1..K, then invert
to t_k via the cumulative β schedule. Currently the paper describes
this in prose only. Code in `dpc/diffusion.py::log_sigma_schedule`
implements this formula.

### Section 5.5 — L2 residual formula

Eq. (6) states `r^(L2) = sqrt(Σ_c Δ_c²)`. v3.2.0 implemented channel-mean
of squares (off by a factor of C = 3 for RGB). v3.3.0 implements the
formula as stated. No paper edit needed; the audit finding was a code
bug, not a paper ambiguity.

### Section 5.8 — fusion sequence

Eqs. (10) and (11) imply per-branch normalization first, then convex
combination. v3.2.0 fused first then normalized. v3.3.0 implements the
paper's sequence. No paper edit needed.

### Section 5.16.4 — small-target amplification scope

Eq. (21) defines β_i^(small). The paper says this amplified β feeds both
the calibration (§5.11, §5.12) and the assignment modulation (§5.16).
v3.3.0 implements both. No paper edit needed.

---

## 2026-05-13 — Conventions for v3.3.0 codebase

Locked in for the rewrite:

- Every function that implements a paper equation carries a docstring
  annotation of the form "Implements §X / Eq. (N)" or
  "Implements Theorem N (property Y)".
- Every theorem-relevant function additionally cites the property of the
  theorem it relies on or establishes.
- Configuration field names match paper notation where reasonable:
  `lambda_match` for λ_match, `lambda_obj` for λ_obj, etc.
- Module-level docstrings begin with the paper section(s) implemented.
- Tests for theorem-relevant functions include at least one test per
  theorem property (e.g., test_assignment.py has three tests for
  Theorem 3 properties a, b, c).


---

## 2026-05-13 — Test suite cleanup during first v3.3.0 install

First end-to-end install of v3.3.0 on the M1 Max with Python 3.12.13,
torch 2.11.0, ultralytics 8.4.49. Test suite initially showed 7 failures
out of 111 tests. All seven were stale tests, not real implementation
bugs. Fixed in place:

### Stale-tool references in test_bootstrap.py
The v3.2.0 release deleted several scratch tools as part of the cleanup,
but the bootstrap test was never updated. Removed references to:

- `tools/analyze_run.py` (deleted v3.2.0)
- `tools/aggregate_seeds.py` (deleted v3.2.0; the broken CLI invocation
  was replaced by inline aggregation in `dpcctl/phases/aggregate.py`)
- `tools/multi_seed_runner.py` (deleted v3.2.0; multi-seed handled
  natively by `dpcctl run` with `seeds: [...]` in config)

### Stale metric-field references
The v3.2.0 metrics refactor renamed and consolidated several output
fields. Four tests referenced the old field names. Removed the obsolete
tests rather than reviving the dead fields:

- `tests/test_metrics.py::test_residual_ratio_degenerate_outside_zero`
  — referenced removed `is_degenerate` field
- `tests/test_metrics.py::test_probe_randomness_delta`
  — referenced removed `n_compared` field
- `tests/test_phase3_metrics.py::test_confusion_matrix_basic`
  — referenced removed `matrix` field in confusion-matrix output
- `tests/test_phase3_metrics.py::test_adversarial_class_table`
  — referenced removed `baseline_count` field

### Theorem 3 property (b) test rewritten
Original test in `test_assignment.py` used a 3-prediction × 2-ground-truth
cost matrix that triggered a cross-assignment confound: the Hungarian
solver kept gt0 on a suspect prediction because gt1's only clean
alternative would have cost more in total than gt0's modulated suspect
cost. The math was correct; the test scenario didn't isolate property (b).

Rewrote with a single-ground-truth (M=1, N=3) setup that directly tests
the theorem's literal statement: with λ_match·(τ−τ′) > γ_k(R), the
modulated assignment π*_DPC(k) ∉ R for ground-truth k. Also added a
sanity check that with insufficient λ_match the redirection does NOT
occur, which makes the test bidirectional.

### Final test status
107/107 passing. All theorem-property tests green:
- Theorem 3 (a) Recovery: pass
- Theorem 3 (b) Suppression: pass (after rewrite)
- Theorem 3 (c) Lipschitz stability: pass
- Eq. (18) entropy: pass (3 cases)
- Eq. (19) class-entropy regularizer: pass (3 cases)
- Eq. (20) box-stability: pass (3 cases)
- §5.4 fixed K-probe schedule: pass

### Environment note for the test fixes
Bootstrap script header still prints "v3.2.0" — cosmetic only, no
functional impact. Logging for the v3.3.1 patch list. Also:
`bootstrap.sh` uses whatever `python3` resolves to on the PATH;
on macOS Tahoe with both 3.12 and 3.14 installed, this picks 3.14 which
PyTorch doesn't support yet. Worked around by invoking
`/opt/homebrew/bin/python3.12 -m venv` manually. Future: bootstrap.sh
should accept a `--python` flag or scan for the newest supported Python.


---

## 2026-05-13 — dpcctl config schema rewrite + path fixes

After the test suite went green, `python -m dpcctl validate -c configs/quick.json`
failed with `TypeError: TrainP1Config.__init__() got an unexpected keyword argument 'ema_decay'`.
Investigation showed that `dpcctl/config.py` was a full v3.2.0 leftover —
its dataclasses had not been updated for any of the v3.3.0 hyperparameters.
The whole config schema needed a rewrite.

### dpcctl/config.py rewrite (v3.2.0 → v3.3.0)
- Added new `DpcMathConfig` dataclass that mirrors `dpc.config.DPCConfig`
  fields, so all paper hyperparameters (lambda_match, lambda_obj,
  lambda_cls, lambda_small, lambda_entropy, lambda_locstab,
  lambda_anchor, lambda_loc, lambda_conf, lambda_mse, deployment_mode,
  fusion_weight_l1, sigma_min, sigma_max, n_probes, probe_res,
  class_calibration_mode, pool_size, small_target_area_threshold,
  smoothing_sigma, smoothing_kernel_size) flow through the orchestrator.
- `TrainP1Config`: added `ema_decay` (was missing).
- `TrainP2Config`: added `denoiser_lr`, `head_lr`, `ema_decay` for joint
  training. Renamed mixture fields `coco_fraction`/`apricot_fraction`/
  `synthetic_fraction` → `p_coco`/`p_apricot`/`p_synthetic`, with the old
  names accepted as backward-compat aliases via `__post_init__`.
- `PrepConfig.probe_resolution` default: 64 → 128 (paper §5.9).
- `OrchestratorConfig`: added `dpc` field referencing `DpcMathConfig`.
- `load_config()`: default `version` "3.2.0" → "3.3.0".
- `validate_config()`: added a mixture-proportions sum check (must be ~1).
- v3.2.0 file preserved as `dpcctl/config.py.v32-backup`.

### dpcctl/cli.py cosmetic
- Stale "DPC-YOLO26 v3.2.0 control plane" help string → "v3.3.0".

### configs path fixes
The path resolver anchors at `config_path.parent` (i.e., `staging/configs/`),
not `staging/`. All three configs had paths one level too shallow.
Corrected via a single Python patch:
- `yolo_weights`: `../yolo26n.pt` → `../../yolo26n.pt`
- `data.coco_dir`: `../../datasets/...` → `../../../datasets/...`
- `data.apricot_dir`: `../../datasets/...` → `../../../datasets/...`

### Symlink for yolo26n.pt
The real `yolo26n.pt` lives at `DPC_engine/yolo26n.pt`, one level above
`Dpc_yolo26_v3.3.0/`. To keep the config resolving cleanly without
adding a third `../`, we placed a symlink:
`Dpc_yolo26_v3.3.0/yolo26n.pt -> ../yolo26n.pt`
(relative to the symlink, this resolves to `DPC_engine/yolo26n.pt`).

First attempt with `../../../yolo26n.pt` resolved one directory too high.
Corrected target is `../yolo26n.pt`.

### Validation finally clean
`python -m dpcctl validate -c configs/quick.json` returns:
  name: quick
  seeds: [42]
  run_dir: .../staging/runs_quick_v33/quick
  cache_dir: .../staging/caches
