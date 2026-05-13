"""Smoke tests that don't need torch — for the very first sanity check.

This is the COMPLETE merged repo: all three phases of training + evaluation
in one directory. Tests verify every tool from every phase is present.
"""

import re
import sys
from pathlib import Path

import pytest


def test_python_version():
    assert sys.version_info >= (3, 12), \
        f"Python 3.12+ required, got {sys.version_info.major}.{sys.version_info.minor}"


def test_dpc_version_string():
    """Read version directly without importing dpc/__init__.py (which pulls torch)."""
    version_file = Path(__file__).resolve().parent.parent / "dpc" / "_version.py"
    assert version_file.is_file()
    src = version_file.read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', src)
    assert m, "could not find __version__ in dpc/_version.py"
    version = m.group(1)
    assert re.match(r"^\d+\.\d+\.\d+$", version), f"bad semver: {version!r}"
    assert version.startswith("3."), f"expected v3.x.x, got {version}"


def test_all_dpc_modules_present():
    repo = Path(__file__).resolve().parent.parent
    for f in [
        "dpc/__init__.py", "dpc/_version.py",
        # core (all phases)
        "dpc/seeding.py", "dpc/manifest.py", "dpc/config.py",
        "dpc/diffusion.py", "dpc/denoiser.py", "dpc/ema.py",
        "dpc/checkpoint.py", "dpc/losses.py", "dpc/data_cache.py",
        "dpc/data.py", "dpc/metrics.py", "dpc/field.py",
        # Phase 2
        "dpc/synthetic_patch.py",
        # Phase 3
        "dpc/calibration.py", "dpc/pooling.py", "dpc/wrapper.py",
        "dpc/coco_classes.py", "dpc/nms.py",
    ]:
        assert (repo / f).is_file(), f"missing {f}"


def test_all_phase1_tools_present():
    repo = Path(__file__).resolve().parent.parent
    for f in [
        "tools/build_caches.py",
        "tools/sanity_check_data_phase1.py",
        "tools/sanity_check_loss_phase1.py",
        "tools/train_phase1.py",
        "tools/smoke_test_phase1.py",
        "tools/diagnose_residuals.py",
        "tools/diagnose_raw_residual.py",
        "tools/diagnose_untrained.py",
            ]:
        assert (repo / f).is_file(), f"missing {f}"


def test_all_phase2_tools_present():
    repo = Path(__file__).resolve().parent.parent
    for f in [
        "tools/fit_color_distribution.py",
        "tools/sanity_check_data_phase2.py",
        "tools/sanity_check_loss_phase2.py",
        "tools/train_phase2.py",
        "tools/smoke_test_phase2.py",
        "tools/compare_phases.py",
    ]:
        assert (repo / f).is_file(), f"missing {f}"


def test_all_phase3_tools_present():
    repo = Path(__file__).resolve().parent.parent
    for f in [
        "tools/evaluate_phase3.py",
        "tools/sweep_alpha.py",
                        "tools/sanity_check_eval.py",
        "tools/smoke_test_phase3.py",
    ]:
        assert (repo / f).is_file(), f"missing {f}"



def test_requirements_includes_ultralytics():
    repo = Path(__file__).resolve().parent.parent
    reqs = (repo / "requirements.txt").read_text()
    assert "ultralytics" in reqs.lower(), "merged repo requires ultralytics"
