# DPC-YOLO26 v3.3.0

Diffusion-prior consistency defense against APRICOT adversarial patches
on YOLO26, with paper-strict implementation of the framework described in
*"Diffusion-Prior Consistency for Robust NMS-Free Detection"*
(C. Varela, PsypherLabs / CSU San Marcos, Spring 2026).

## What this is

A defense pipeline that wraps a YOLO26 object detector and suppresses
adversarial patch attacks (specifically the APRICOT benchmark) by:

1. Training a small TinyUNet denoiser on COCO so its residual on
   off-manifold inputs is large (Phase 1).
2. Jointly fine-tuning the denoiser and YOLO26's detection head on a mix
   of clean COCO, real APRICOT patches, and synthetic patches, with the
   one-to-one assignment cost matrix modulated by suspicion (Phase 2).
3. Evaluating DPC-wrapped YOLO26 against baseline YOLO26 on APRICOT,
   sweeping the calibration constants (Phase 3).

The mathematical framework rests on five theorems. The code's mapping
from paper section to function is documented in `docs/DESIGN_v3.3.0.md`.
Every theorem-relevant function carries a docstring annotation citing the
equation or theorem property it implements.

See the comprehensive engineering manual at
`docs/USER_MANUAL_v3.3.0.md` for everything else.

## Quickstart

```bash
# 1. Place the release zip next to your dataset tree
cd /path/to/your/DPC_engine
ls
# expected: datasets/  yolo26n.pt  dpc-yolo26-v3.3.0-complete.zip

# 2. Extract
mkdir staging-v3.3 && cd staging-v3.3
unzip ../dpc-yolo26-v3.3.0-complete.zip
mv staging/* . && rmdir staging

# 3. Bootstrap a fresh venv
./bootstrap.sh ~/dpc-v33-venv
source ~/dpc-v33-venv/bin/activate

# 4. Run the test suite first
python -m pytest tests/ -q

# 5. Validate the smoke config
python -m dpcctl validate -c configs/quick.json

# 6. Smoke pipeline end-to-end (~15 minutes on M1 Max)
python -m dpcctl run -c configs/quick.json -p all
```

## v3.3.0 vs v3.2.0

v3.3.0 closes the math audit findings from v3.2.0. The headline change is
the implementation of §5.16 (assignment-cost modulation, Theorem 3) —
the paper's central novel contribution, which v3.2.0 did not implement.
Other gaps closed: fixed K-probe schedule (§5.4), real per-class YOLO
logits (no more +10 margin trick), Eqs. (19) and (20) auxiliary losses,
strict L2 residual (Eq. 6), strict fusion sequence (Eqs. 10 → 11), and
vectorized box pooling.

See `CHANGELOG.md` for the full list.

## Repository layout

```
staging-v3.3/
├── bootstrap.sh
├── requirements.txt
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── configs/                            # JSON orchestrator configs
│   ├── quick.json                      # smoke
│   ├── default.json                    # single-seed full
│   └── production.json                 # 3-seed full
├── dpc/                                # Library
│   ├── _version.py
│   ├── __init__.py
│   ├── assignment.py                   # §5.16 / Theorem 3 (NEW)
│   ├── auxiliary_losses.py             # §5.13, §5.14 (NEW)
│   ├── calibration.py                  # §5.11, §5.12, §5.15
│   ├── checkpoint.py
│   ├── coco_classes.py
│   ├── config.py
│   ├── data.py
│   ├── data_cache.py
│   ├── denoiser.py
│   ├── diffusion.py                    # §5.1, §5.4
│   ├── ema.py
│   ├── field.py                        # §5.3 – §5.9
│   ├── losses.py
│   ├── manifest.py
│   ├── metrics.py
│   ├── nms.py
│   ├── pooling.py                      # §5.10
│   ├── seeding.py
│   ├── synthetic_patch.py
│   ├── wrapper.py                      # inference-only
│   └── yolo26_native.py                # native YOLO26 bridge (NEW)
├── dpcctl/                             # Orchestrator / dashboard
├── docs/
│   ├── DESIGN_v3.3.0.md                # paper-to-code map
│   ├── NOTES_LOG.md                    # decisions and paper-edit reminders
│   └── USER_MANUAL_v3.3.0.md           # comprehensive manual
├── tests/
└── tools/
    ├── train_phase1.py
    ├── train_phase2.py                 # full rewrite for joint training
    ├── evaluate_phase3.py              # uses native bridge
    └── ...
```

## License and citation

See the paper for the academic reference. This code is research software
distributed without warranty.
