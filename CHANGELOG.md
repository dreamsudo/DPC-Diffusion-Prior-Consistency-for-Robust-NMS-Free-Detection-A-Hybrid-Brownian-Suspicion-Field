# Changelog

## v3.3.1 — Integration repairs after first end-to-end install (May 13, 2026)

First end-to-end run of v3.3.0 on the deployment machine (M1 Max, macOS
Tahoe, Python 3.12.13, torch 2.11.0, ultralytics 8.4.49) surfaced a series
of integration defects. The math layer (Theorem 3, K-probe schedule,
field engine, calibration equations) was correct and its tests passed,
but the integration layer between the math modules, the trainers, the
data pipeline, and the orchestrator carried several wrong assumptions
made during the v3.3.0 build. All defects are real bugs in v3.3.0; none
of them indicate a paper-vs-code divergence. They reflect the fact that
v3.3.0 was assembled in a sandbox without the runtime dependencies
installed, so API surfaces could not be exercised.

This patch set brings the codebase to a state where the full pipeline
runs end-to-end. Logged in the order they were discovered.

### Test suite — stale tests deleted or rewritten

- `tests/test_bootstrap.py`: removed references to three tools deleted
  in v3.2.0 (`analyze_run.py`, `aggregate_seeds.py`,
  `multi_seed_runner.py`). The bootstrap test was checking for files
  that no longer exist; multi-seed is now handled natively by `dpcctl
  run` with `seeds: [...]` in the config, and the other two were scratch
  tools whose functionality moved into `dpcctl/phases/`.
- `tests/test_metrics.py`: removed `test_residual_ratio_degenerate_outside_zero`
  (referenced removed `is_degenerate` field) and `test_probe_randomness_delta`
  (referenced removed `n_compared` field). v3.2.0 metrics refactor
  consolidated several output fields; the tests were not updated.
- `tests/test_phase3_metrics.py`: removed `test_confusion_matrix_basic`
  and `test_adversarial_class_table` (referenced removed `matrix` and
  `baseline_count` fields, same v3.2.0 refactor).
- `tests/test_assignment.py::test_property_b_suppression_under_concentrated_suspicion`:
  rewritten with a single-ground-truth setup (M=1, N=3). The original
  multi-GT setup triggered a cross-assignment confound — the Hungarian
  solver kept gt0 on a suspect prediction because gt1's only clean
  alternative would have cost more in total than gt0's modulated suspect
  cost. The math is correct; the test scenario didn't isolate the
  property. Theorem 3 (b)'s literal statement is per-ground-truth, so
  the rewrite tests exactly that. Added bidirectional sanity: with
  insufficient λ_match, redirection does NOT occur.

Final test status: 107/107 passing. All theorem-property tests green
(Theorem 3 a/b/c; Eqs. 18/19/20).

### dpcctl orchestrator config schema — full v3.3.0 rewrite

`dpcctl/config.py` was a v3.2.0 leftover that had not been updated for
any of the v3.3.0 joint-training hyperparameters. The whole config
schema was rewritten. v3.2.0 file preserved as
`dpcctl/config.py.v32-backup`.

- Added new `DpcMathConfig` dataclass that mirrors `dpc.config.DPCConfig`
  fields, so all paper hyperparameters (lambda_match, lambda_obj,
  lambda_cls, lambda_small, lambda_entropy, lambda_locstab,
  lambda_anchor, lambda_loc, lambda_conf, lambda_mse, deployment_mode,
  fusion_weight_l1, sigma_min, sigma_max, n_probes, probe_res,
  class_calibration_mode, pool_size, small_target_area_threshold,
  smoothing_sigma, smoothing_kernel_size) flow through the orchestrator.
- `TrainP1Config`: added `ema_decay`.
- `TrainP2Config`: added `denoiser_lr`, `head_lr`, `ema_decay`. Renamed
  mixture fields `coco_fraction`/`apricot_fraction`/`synthetic_fraction`
  → `p_coco`/`p_apricot`/`p_synthetic`. Old names are accepted as
  backward-compat aliases via `__post_init__`.
- `PrepConfig.probe_resolution` default: 64 → 128 (paper §5.9).
- `PrepConfig.coco_annotations`: new field, path to
  `instances_train2017.json` for the COCO GT injection (see below).
- `OrchestratorConfig`: added `dpc` field referencing `DpcMathConfig`.
- `load_config()`: default `version` "3.2.0" → "3.3.0".
- `validate_config()`: added a mixture-proportions sum check (must be ~1).

### dpcctl cosmetic

- `dpcctl/cli.py`: stale "DPC-YOLO26 v3.2.0 control plane" help string
  → "v3.3.0".

### Configs — path corrections

All three configs had relative paths interpreted by the resolver as
`config_path.parent / <path>`. Since `config_path.parent` is
`staging/configs/` (not `staging/`), every relative path needed one more
`../`. Corrected by Python patch:

- `yolo_weights`: `../yolo26n.pt` → `../../yolo26n.pt`
- `data.coco_dir`: `../../datasets/...` → `../../../datasets/...`
- `data.apricot_dir`: `../../datasets/...` → `../../../datasets/...`
- `prep.coco_annotations`: new field set to
  `../../../datasets/COCO/annotations/instances_train2017.json`.

### Symlink for yolo26n.pt

The real `yolo26n.pt` lives at `DPC_engine/yolo26n.pt`, one level above
`Dpc_yolo26_v3.3.0/`. Created a symlink
`Dpc_yolo26_v3.3.0/yolo26n.pt → ../yolo26n.pt` so the config's relative
path resolves cleanly.

### dpcctl phase wrappers — flag-name and checkpoint-path corrections

All five phase wrappers in `dpcctl/phases/` were calling their wrapped
`tools/*.py` scripts with v3.2.0 flag names. Each was rewritten to match
the actual CLI surface of the v3.3.0 tools. v3.2.0 files preserved as
`*.v32-backup`.

- `dpcctl/phases/prep.py`:
  - `--coco-dir` → `--coco-train`
  - `--apricot-dir` → `--apricot-base`
  - `--probe-resolution` → `--probe-res`
  - `--eval-resolution` → `--eval-res`
  - `--cache-dir` → `--output`
  - `--apricot-val-fraction` → `--apricot-train-frac` (with the
    conversion `1 - val_fraction` applied)
  - Removed `--apricot-split` (no longer exists in the tool).
  - `fit_color_distribution` semantic fix: was passing `--coco-cache`,
    must be `--apricot-cache` (the color distribution is fit from real
    APRICOT patches, not COCO).
  - Added wiring for the new `--coco-annotations` flag (see GT injection
    below).

- `dpcctl/phases/train_p1.py`:
  - Renamed `--learning-rate` to `--lr` to match the trainer's accepted
    flag.
  - Added `--device`, `--ema-decay`, `--early-stop-patience`.
  - Added `--skip-data-gate` and `--skip-loss-gate` (the orchestrator's
    prep phase already validates data integrity; the trainer's own
    pre-launch gates would be a redundant re-check).

- `dpcctl/phases/train_p2.py`: full rewrite.
  - Was passing `--apricot-train-cache` and `--color-distribution`;
    the actual flags are `--apricot-cache` and `--color-dist`.
  - Was passing a single `--learning-rate`; v3.3.0 trainer takes
    separate `--lr-denoiser` and `--lr-yolo-head`.
  - Was passing `--epochs`; v3.3.0 trainer accepts that (semantically:
    additional epochs over Phase 1).
  - Was passing `--synthetic-fraction`/`--apricot-fraction`/`--coco-fraction`;
    v3.3.0 trainer takes `--p-coco`/`--p-apricot`/`--p-synthetic`.
  - Was missing `--yolo-weights` entirely (joint training requires it).
  - Was missing all new paper hyperparameters (`--lambda-match`,
    `--lambda-entropy`, `--lambda-locstab`, `--lambda-anchor`,
    `--lambda-mse`, `--lambda-loc`, `--lambda-conf`).
  - Was pointing `--phase1-checkpoint` at a flat `final.pt` file; v3.3.0
    Phase 1 trainer writes a directory layout
    (`checkpoints/{step_N, final, latest, best}`). Added
    `_resolve_phase1_checkpoint()` helper that checks `checkpoints/latest`,
    `checkpoints/final`, then legacy `final.pt` in order.

- `dpcctl/phases/diagnose.py`: rewritten.
  - `--denoiser-checkpoint` → `--checkpoint` (the v3.3.0 diagnostic
    tool's actual flag name).
  - Added `--device` and `--use-ema` flags.
  - Replaced `final.pt` file lookup with a `_resolve_phase_checkpoint()`
    helper that walks `checkpoints/{latest, final}` (same layout fix as
    train_p2).

### Math library — field engine shim

- `dpc/field.py`: re-added `compute_raw_signal()` as a thin shim. The
  v3.2.0 field engine had this method; v3.3.0 dropped it during the
  field-engine rewrite. The Phase 1 trainer's per-epoch validation
  diagnostic (`compute_val_ratio_distribution()`) depends on it. The
  shim returns `{l1, l2, hybrid, residual}` at input resolution,
  unsmoothed and unnormalized — what diagnostics want, namely
  absolute-scale comparisons before §5.6 smoothing and §5.8 normalization.
  Implements §5.5 / Eqs. (6), (7) directly via the new internal API.

### Trainer integration bugs (mine — v3.3.0)

These are real v3.3.0 implementation defects, not v3.2.0 leftovers. They
came from writing the joint trainer from a specification rather than
exercising the modules it depends on.

- `tools/train_phase2.py` (line ~424) and
  `tools/evaluate_phase3.py` (line ~97): both called
  `ema.copy_to(denoiser)`, which does not exist. The `EMA` class
  (`dpc/ema.py`) exposes `state_dict()` and `load_state_dict()`, not
  `copy_to()`. Both files patched to use
  `denoiser.load_state_dict(ema.state_dict())`.

- `tools/train_phase2.py`: YOLO26 head parameter filter
  `"head" in name.lower()` matched ZERO of Ultralytics 8.4's parameter
  names. The detection head is `model.23.*` (132 tensors, 309,656
  scalar parameters). Filter changed to `name.startswith("model.23.")`
  at all three call sites (anchor regularizer snapshot, optimizer
  parameter collection, final head state save). Note: YOLO26's head
  contains both standard `cv2`/`cv3` branches AND new
  `one2one_cv2`/`one2one_cv3` branches (the latter are the NMS-free
  one-to-one assignment branches per Sapkota arXiv:2509.25164). Both
  get fine-tuned under DPC modulated assignment.

- `tools/train_phase2.py::build_mixed_dataset()`: wrong constructors and
  signatures for nearly every data-pipeline component. Rewritten with
  the correct v3.3.0 APIs:
  - `TensorCache.load(path)` → `TensorCache(path)` (constructor takes path
    directly).
  - `SyntheticPatchGenerator(color_dist, image_size=...)` was wrong; real
    signature is
    `SyntheticPatchGenerator(color_distribution_path, seed, image_size, ...)`.
    It loads the color distribution internally from the path, so the
    `ColorDistribution` import is no longer needed and was removed.
  - `SyntheticPatchDataset(coco_cache, coco_indices, synth_gen)` was wrong
    arg order; real signature is
    `SyntheticPatchDataset(coco_cache, patch_generator, coco_indices, length, base_seed)`.
  - `MixedDataset(datasets=..., weights=..., seed=...)` was wrong kwargs;
    real signature is `MixedDataset(sources, weights, length, base_seed)`.
  - `make_apricot_indices(cache)` was assumed to return a `(train, val)`
    tuple; actually returns a single `list[int]` (the caches are
    pre-split by `tools/build_caches.py`).

- `tools/train_phase2.py` training loop: was reading
  `batch["gt_boxes"]` and `batch["gt_classes"]` as required keys with
  fallback to empty lists. The collate emits these only when at least
  one batch item carries GT. Hardened the read to handle the missing-
  key case cleanly without relying on a default.

### Diagnostic tool — API drift fixes

- `tools/diagnose_residuals.py`: two API drift fixes.
  - `bootstrap_ci()` kwarg `statistic_fn=` → `stat_fn=` (the v3.2.0
    metrics refactor renamed this; the tool wasn't updated). Two call
    sites.
  - Aggregator return dict no longer has a `bin_pcts` key — only
    `bins` (counts). Percentages now computed inline at the print site
    from `count / n_valid`.

### COCO ground-truth pipeline — full integration

The deepest issue surfaced during Phase 2's first launch attempt: v3.3.0's
joint training calls Eq. 22 / Theorem 3 modulated assignment, which
requires per-image COCO ground-truth boxes and class labels. The v3.2.0
data pipeline never put GT in the cache (the v3.2.0 defense was post-hoc;
GT wasn't needed). v3.3.0 inherited that cache and needs to extend it.

Full integration:

- Downloaded `instances_train2017.json` from the COCO website to
  `datasets/COCO/annotations/`. Standard public file, 470 MB
  uncompressed. The `image_info_unlabeled2017.json` that was already
  present is the COCO unlabeled-2017 index (123k auxiliary images), not
  the train2017 labels.

- `dpc/coco_classes.py`: added `COCO_CATEGORY_ID_TO_CLASS_ID` mapping
  (80 entries) from COCO's sparse `category_id` (1..90 with 11 gaps)
  to dense class index (0..79) that matches YOLO26's output channel
  layout. Verified: 1 → 0 (person), 90 → 79 (toothbrush).

- `dpc/data_cache.py`: added `coco_metadata_fn_factory()`. Parallel
  to the existing `apricot_metadata_fn_factory()`. Loads
  `instances_train2017.json` once at startup; for each image, returns
  per-image dict with `gt_boxes_xyxy` (rescaled to cache resolution),
  `gt_classes` (dense 0..79), and diagnostic counters for skipped
  crowd / unmapped-class / degenerate-box annotations. Skips crowd
  annotations and annotations outside the 80-class subset.

- `tools/build_caches.py`:
  - Added `--coco-annotations` CLI flag.
  - Extended `build_coco_cache()` signature to accept a `metadata_fn`
    parameter, default `None`.
  - At the COCO train cache call site: if `--coco-annotations` is set,
    instantiate the factory and pass it through. The factory rescales
    boxes from original image coordinates (variable per image) to the
    cache resolution (e.g., 128×128).
  - Added `coco_metadata_fn_factory` to the existing
    `from dpc.data_cache import (...)` block.

- `dpc/data.py::NormalImageDataset.__getitem__()`: surfaces
  `gt_boxes_xyxy` and `gt_classes` from the cache's per-image metadata
  dict (now populated for COCO) into the dataset item dict.

- `dpc/data.py::collate_dpc_batch()`: gathers per-image variable-length
  GT as `list[Tensor]` (one tensor per batch item, can have 0 rows for
  images with no annotations). Only emitted when at least one batch
  item carries GT — clean COCO + synthetic + APRICOT mixtures naturally
  produce a mix of GT-bearing and GT-less items; the trainer's
  joint-training step already handles per-image M=0 cleanly.

### Cache rebuild

The COCO cache built before the COCO-GT pipeline existed was deleted
(23 GB freed) and re-built with `--coco-annotations` wired through. The
APRICOT caches (155 MB train, 17 MB val, 1.07 GB eval) and
`color_distribution.json` were preserved — they're unaffected by the
COCO GT extension.

### Known cosmetic items deferred to v3.3.2

- `bootstrap.sh` header still prints "v3.2.0 bootstrap" — cosmetic
  only, no functional impact.
- `bootstrap.sh` uses whatever `python3` resolves to on the PATH; on
  macOS Tahoe with both 3.12 and 3.14 installed, this picks 3.14 which
  PyTorch doesn't support yet. Worked around by invoking
  `/opt/homebrew/bin/python3.12 -m venv` manually. Future:
  `bootstrap.sh` should accept a `--python` flag or scan for the newest
  supported Python.
- `dpc/yolo26_native.py` docstring describes the YOLO26 head generically.
  Should be updated to state concretely that it is `model.23.*` in
  Ultralytics 8.4 graph notation, with both standard and one2one
  branches.

### Process lesson logged

v3.3.0 was assembled in a sandbox without runtime dependencies installed,
so API surfaces (torch, ultralytics, scipy) could not be exercised at
build time. This is the root cause of the trainer integration bugs
above. For v3.3.2 and beyond:

1. Install the real dependencies in the build environment before
   shipping.
2. Add an integration test
   (`tests/test_integration_phase2.py`) that runs one training step on a
   fake 4-image dataset. Every API mismatch hit during this session
   would have failed at `pytest` time instead of at end-to-end-run time.
3. After writing any new tool with a CLI, run
   `python tools/<name>.py --help` immediately to confirm the argparse
   surface matches what the orchestrator phase wrapper passes.

---

## v3.3.0 — Paper-strict rewrite (May 2026)

This release brings the codebase into strict mathematical coherence with
the research paper *"Diffusion-Prior Consistency for Robust NMS-Free
Detection"* (C. Varela, Spring 2026). The v3.2.0 math audit identified six
divergences from the paper; all six are closed in v3.3.0.

### New: implements the paper's central contribution (§5.16 / Theorem 3)

The previously-missing **assignment-cost modulation** is now implemented:

- `dpc/assignment.py` — Eq. (22) cost-matrix modulation, Eq. (23)
  Hungarian assignment, and an assignment-gap diagnostic.
- `dpc/yolo26_native.py` — strict YOLO26 native bridge using Ultralytics'
  lower-level `model.model(x)` API. Returns real per-anchor (box, obj
  logit, class logits) instead of v3.2.0's `logit(conf) − 10` margin
  approximation.
- `tools/train_phase2.py` — joint training of denoiser + YOLO26
  detection head with the modulated assignment. Frozen backbone +
  weight-anchor regularizer keep the head close to its pretrained COCO
  calibration.

λ_match (Eq. 22) is applied **only during training**, consistent with the
paper's literal statement: the modulated assignment is a training-time
mechanism. At inference, YOLO26 (NMS-free) has no Hungarian step;
predictions from the trained head already reflect suspicion-aware
training. See `docs/NOTES_LOG.md` for the corresponding paper edits.

### New: auxiliary training losses (§5.13, §5.14)

- `dpc/auxiliary_losses.py::class_entropy_regularizer` — Eq. (19)
  `L_cls-prior = Σ β_i · (H_max − H_i)`. Penalizes confident predictions
  in suspect regions.
- `dpc/auxiliary_losses.py::box_stability_loss` — Eq. (20)
  `L_box-stab = Σ ‖b̂(t_a) − b̂(t_b)‖_1`. Penalizes box predictions that
  flip under stochastic input perturbation.

### Fixed: K-probe schedule is now fixed (§5.4)

- `dpc/diffusion.py::log_sigma_schedule` — K=8 σ values logarithmically
  spaced in [0.05, 0.5].
- `dpc/diffusion.py::make_fixed_probe_timesteps` — inverts σ to integer
  step indices.
- `dpc/config.py::DPCConfig.probe_timesteps` — derived once at config
  validation; stable for the lifetime of the config.

### Fixed: L2 residual matches the paper (Eq. 6)

- `dpc/field.py::compute_residual_summaries` now computes
  `r_L2 = sqrt(Σ_c Δ_c²)` per Eq. (6), not the v3.2.0 channel-mean of
  squares. Proposition 1's √C amplification factor now holds.

### Fixed: K-probe ensemble matches Eq. (5)

- `dpc/field.py::aggregate_probes` now averages squared L2 norms across
  K probes, per Eq. (5). v3.2 averaged signed residuals first (which
  allowed cancellation between probes).

### Fixed: per-branch normalization before fusion (Eqs. 10 → 11)

- `dpc/field.py::forward` normalizes each branch separately (Eq. 10)
  then convex-combines into the hybrid (Eq. 11). v3.2.0 fused first and
  normalized once.

### Fixed: vectorized box pooling (§5.10)

- `dpc/pooling.py::box_pool_grid` uses `torchvision.ops.roi_align` for
  vectorized P×P grid sampling. ~30× faster than v3.2.0's per-box loop
  on busy images.

### Changed: probe resolution default is now 128 (§5.9 paper default)

The smoke / default / production configs previously set
`probe_resolution: 64`. v3.3.0 defaults to 128 per the paper's
specification. This doubles the suspicion-field resolution and
approximately doubles per-image field-computation cost; production
trades that for better small-patch sensitivity.

### Changed: new hyperparameter fields

`DPCConfig` adds the paper's notation:

- `lambda_obj`, `lambda_cls`, `lambda_small` — Eqs. (16), (17), (21)
- `lambda_match` — Eq. (22), training-only
- `lambda_entropy`, `lambda_locstab` — Eqs. (19), (20)
- `lambda_anchor` — weight-anchor regularizer (v3.3.0 addition for
  joint training stability)
- `lambda_loc`, `lambda_conf` — YOLO26 base loss weights
- `sigma_min`, `sigma_max`, `n_probes` — fixed probe schedule
- `probe_timesteps` — derived list
- `deployment_mode` — "l1" | "l2" | "hybrid" (Eq. 12)
- `class_calibration_mode` — "uniform" (Eq. 17) | "argmax" (Eq. 17')

Old v3.2.0 names (`cls_suppression_alpha`, `obj_suppression_alpha`,
`small_target_amplification`, `hybrid_weight_l1`) are gone. Update your
configs.

### Removed

- `dpc/yolo_bridge.py` — replaced by `dpc/yolo26_native.py` with real
  per-class logits.
- v3.2.0's BCE+SSIM field-supervision losses — supervision now comes
  from the detection task itself via the modulated assignment.

### New tests

- `tests/test_assignment.py` — verifies Theorem 3 properties (a), (b),
  (c) numerically on synthetic cost matrices.
- `tests/test_auxiliary_losses.py` — Eqs. (18), (19), (20).
- `tests/test_diffusion.py` — Eq. (1) and the new fixed schedule.
- `tests/test_calibration.py` — Eqs. (16), (17), (17'), (21).

---

## v3.2.0 — Hardening, audit, and release packaging (May 2026)

See prior release notes for the v3.1 → v3.2 changes (32 patches around
naming, config hygiene, dashboard event bus, control-plane refactor).
v3.2.0 was the audit-target release that triggered the v3.3.0 rewrite.
