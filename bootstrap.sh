#!/usr/bin/env bash
#
# DPC-YOLO26 v3.2.0 bootstrap.
#
# Creates a fresh virtual environment, applies the macOS Tahoe libexpat
# shim BEFORE running pip (so pip can import xmlrpc), installs all
# dependencies, and verifies the install.
#
# Usage:
#     ./bootstrap.sh                          # default venv at ~/dpc-venv
#     ./bootstrap.sh /custom/venv/path        # custom venv path
#
# Idempotent — safe to re-run.

set -euo pipefail

VENV_PATH="${1:-$HOME/dpc-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "DPC-YOLO26 v3.2.0 bootstrap"
echo "  venv path: $VENV_PATH"
echo "  python:    $($PYTHON_BIN --version)"
echo "  staging:   $HERE"
echo

# ---- 1. Create venv if missing ---------------------------------------------
if [[ ! -d "$VENV_PATH" ]]; then
    echo "[1/5] Creating venv at $VENV_PATH"
    "$PYTHON_BIN" -m venv "$VENV_PATH"
else
    echo "[1/5] venv exists at $VENV_PATH"
fi

# ---- 2. Apply macOS Tahoe libexpat shim (before pip install) ---------------
# Homebrew Python 3.12 on macOS Tahoe links against newer libexpat symbols
# (_XML_SetAllocTrackerActivationThreshold) than /usr/lib provides. pip
# itself imports xml.parsers.expat at startup, so the shim must be in the
# venv activate script BEFORE the first pip install.
if [[ "$(uname -s)" == "Darwin" ]]; then
    EXPAT_LIB="/opt/homebrew/Cellar/expat"
    if [[ -d "$EXPAT_LIB" ]]; then
        # Find the latest installed expat version
        EXPAT_VERSION=$(ls "$EXPAT_LIB" | sort -V | tail -1)
        EXPAT_FULL_PATH="$EXPAT_LIB/$EXPAT_VERSION/lib"
        SHIM_LINE="export DYLD_LIBRARY_PATH=\"$EXPAT_FULL_PATH:\${DYLD_LIBRARY_PATH:-}\""
        ACTIVATE="$VENV_PATH/bin/activate"

        if ! grep -qF "$EXPAT_FULL_PATH" "$ACTIVATE"; then
            echo "[2/5] Installing libexpat shim into venv activate"
            echo "" >> "$ACTIVATE"
            echo "# macOS libexpat compatibility shim for Homebrew Python" >> "$ACTIVATE"
            echo "$SHIM_LINE" >> "$ACTIVATE"
        else
            echo "[2/5] libexpat shim already present in venv activate"
        fi
    else
        echo "[2/5] Homebrew expat not found at $EXPAT_LIB; skipping shim"
        echo "      If pip fails with 'Symbol not found: _XML_SetAllocTracker...'"
        echo "      install Homebrew expat: brew install expat"
    fi
else
    echo "[2/5] Non-macOS host; libexpat shim not needed"
fi

# ---- 3. Activate venv -------------------------------------------------------
echo "[3/5] Activating venv"
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

python -c "import xml.parsers.expat; print('     expat OK')"

# ---- 4. Upgrade pip + install dependencies ---------------------------------
echo "[4/5] Installing dependencies"
pip install --upgrade pip
pip install -r "$HERE/requirements.txt"

# ---- 5. Sanity check --------------------------------------------------------
echo "[5/5] Sanity check"
python -c "
import dpc, torch, numpy, ultralytics
print(f'     dpc version:         {dpc.__version__}')
print(f'     torch:               {torch.__version__}')
print(f'     numpy:               {numpy.__version__}')
print(f'     ultralytics:         {ultralytics.__version__}')
"

echo
echo "Bootstrap complete. To enter the environment:"
echo "  source $VENV_PATH/bin/activate"
echo "  cd $HERE"
echo "  python -m dpcctl validate -c configs/quick.json"
