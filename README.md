# DPC-YOLO26 v3.3.1

Diffusion-prior consistency defense against APRICOT adversarial patches on YOLO26, implementing the framework described in *Diffusion-Prior Consistency for Robust NMS-Free Detection: A Hybrid Brownian Suspicion Field for YOLO26 Under Adversarial Patch Attacks* (Spring 2026).

---

## What this is

A defense pipeline that wraps a YOLO26 object detector and suppresses adversarial patch attacks by:

1. Training a small TinyUNet denoiser on COCO so its residual on off-manifold inputs (i.e. adversarial patches) is large — **Phase 1**.
2. Jointly fine-tuning the denoiser and YOLO26's detection head on a mix of clean COCO, real APRICOT patches, and synthetic patches, with the one-to-one Hungarian assignment cost matrix **modulated by suspicion** (Equation 22 / Theorem 3, the paper's central novel contribution) — **Phase 2**.
3. Evaluating DPC-wrapped YOLO26 against the undefended baseline on APRICOT, sweeping the calibration constants — **Phase 3**.

The framework rests on five theorems. The mapping from each paper result to its concrete implementation lives in `docs/NOTES_LOG.md`, and every theorem-relevant function carries an inline docstring citing the equation or theorem property it implements.

---

## Quickstart

The data caches are committed under `caches/`, so the smoke run starts immediately without rebuilding them.

```bash
# 1. Bootstrap a fresh virtual environment
./bootstrap.sh ~/dpc-v33-venv
source ~/dpc-v33-venv/bin/activate

# 2. Run the test suite (~30 seconds)
python -m pytest tests/ -q

# 3. Validate the smoke configuration
python -m dpcctl validate -c configs/quick.json

# 4. Run the end-to-end smoke pipeline (~13 minutes on M1 Max MPS)
python -m dpcctl run -c configs/quick.json -p all
```

A reference smoke run is committed under `runs_quick_v33/quick/seed_42/` for comparison.

Full-scale runs:

```bash
# Single-seed development run (~1 hour)
python -m dpcctl run -c configs/default.json -p all

# Production 3-seed evaluation (~20–24 hours, seeds 42, 1337, 2718)
python -m dpcctl run -c configs/production.json -p all
```

Aggregate cross-seed results with bootstrap 95% confidence intervals land in `runs_production_v33/production/aggregate/aggregate_across_seeds.json`.

---

## Library usage

```python
import torch
from dpc.wrapper import DPCWrapper
from dpc.denoiser import TinyUNetDenoiser
from dpc.config import DPCConfig
from dpc.yolo26_native import load_yolo26, patch_finetuned_head

cfg = DPCConfig()              # paper defaults: K=8, h=w=128, hybrid, sigma_smooth=1.5
device = torch.device("mps")   # or "cuda", or "cpu"

denoiser = TinyUNetDenoiser().to(device)
denoiser.load_state_dict(torch.load(
    "runs_production_v33/production/seed_42/train_p2/checkpoints/latest/ema.pt"
))

yolo26 = load_yolo26("yolo26n.pt", device)
patch_finetuned_head(
    yolo26,
    "runs_production_v33/production/seed_42/train_p2/yolo26_head_finetuned.pt",
)

defense = DPCWrapper(yolo26, denoiser, cfg).to(device).eval()
image = torch.rand(1, 3, 320, 320, device=device)
detections = defense(image)
# detections.boxes_xyxy, .scores, .classes, .beta
```

All operations in `DPCWrapper` are differentiable end-to-end.

---

## Status

- 13 test modules in `tests/`, full suite passes.
- End-to-end smoke validated: 8-phase pipeline in 13.3 minutes on M1 Max, Python 3.12.13, PyTorch 2.11, Apple MPS.
- Modulated Hungarian assignment confirmed firing on real COCO ground truth during Phase 2 (`n_matched` range [3, 60] per batch) — the central empirical signal that Equation 22 is correctly coupled to the detector.
- Production multi-seed runs scheduled for full empirical evaluation against APRICOT.

---

## Repository layout

```
dpcroot/
├── bootstrap.sh                # venv bootstrap
├── requirements.txt
├── pyproject.toml
├── README.md
├── CHANGELOG.md
│
├── configs/                    # JSON orchestrator configs
│   ├── quick.json              # 13-min smoke
│   ├── default.json            # single-seed full
│   └── production.json         # 3-seed full
│
├── caches/                     # Prebuilt data caches (committed)
│   ├── apricot_eval_320.pt
│   ├── apricot_train_128.pt
│   ├── apricot_val_128.pt
│   ├── coco_train2017_128.pt
│   ├── color_distribution.json
│   └── manifest.json
│
├── dpc/                        # Library — pure-function math modules
│   ├── _version.py
│   ├── assignment.py           # §5.16 / Eq. 22 / Theorem 3
│   ├── auxiliary_losses.py     # §5.13 (Eq. 19), §5.14 (Eq. 20)
│   ├── calibration.py          # §5.11, §5.12, §5.15
│   ├── checkpoint.py
│   ├── coco_classes.py
│   ├── config.py
│   ├── data.py
│   ├── data_cache.py
│   ├── denoiser.py             # TinyUNet
│   ├── diffusion.py            # §5.1, §5.4
│   ├── ema.py
│   ├── field.py                # §5.3–§5.9 (K-probe engine)
│   ├── losses.py
│   ├── manifest.py
│   ├── metrics.py
│   ├── nms.py
│   ├── pooling.py              # §5.10 (Eq. 15)
│   ├── seeding.py
│   ├── synthetic_patch.py
│   ├── wrapper.py              # DPCWrapper
│   └── yolo26_native.py
│
├── dpcctl/                     # Orchestrator / control plane
│   ├── cli.py
│   ├── config.py
│   ├── orchestrator.py
│   ├── events.py
│   ├── dashboard.py
│   ├── paths.py
│   └── phases/
│       ├── _base.py
│       ├── prep.py
│       ├── train_p1.py
│       ├── train_p2.py
│       ├── diagnose.py
│       ├── eval_p3.py
│       ├── eval_negative_control.py
│       └── aggregate.py
│
├── docs/
│   └── NOTES_LOG.md            # decisions, paper-edit reminders
│
├── runs_quick_v33/             # Reference smoke run output
│
├── tests/                      # 13 test modules
│
└── tools/                      # 20 CLI scripts
    ├── build_caches.py
    ├── train_phase1.py
    ├── train_phase2.py
    ├── evaluate_phase3.py
    ├── fit_color_distribution.py
    ├── compare_phases.py
    ├── compare_raw_residuals.py
    ├── diagnose_residuals.py
    ├── diagnose_raw_residual.py
    ├── diagnose_untrained.py
    ├── render_panels.py
    ├── sweep_alpha.py
    ├── sanity_check_data_phase1.py
    ├── sanity_check_data_phase2.py
    ├── sanity_check_eval.py
    ├── sanity_check_loss_phase1.py
    ├── sanity_check_loss_phase2.py
    ├── smoke_test_phase1.py
    ├── smoke_test_phase2.py
    └── smoke_test_phase3.py
```

---

## Tested environment

- macOS 14, MacBook Pro M1 Max, 64 GB RAM, Apple MPS backend.
- Python 3.12.13, PyTorch 2.11.

Linux + CUDA should work but is not the validated path.

---

## Reproducibility

Within one seed, every random decision is deterministic. EMA, optimizer state, scheduler state, and RNG state are checkpointed atomically; resuming produces numerically identical output to an uninterrupted run. Each run writes a manifest with SHA256 hashes and an environment fingerprint under `runs_*/seed_*/<phase>/manifest.json`.

---

## License and citation

Research software distributed without warranty. See the paper for the academic reference. Version history is in `CHANGELOG.md`.
