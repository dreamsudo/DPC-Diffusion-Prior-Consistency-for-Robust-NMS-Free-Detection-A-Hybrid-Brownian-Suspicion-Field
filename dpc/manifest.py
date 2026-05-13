"""Manifest writing and verification.

Every run directory and every cache file ships with a manifest.json containing:
  - SHA256 of every artifact
  - git commit hash
  - Python and PyTorch versions
  - hardware fingerprint
  - timestamp

Tools that consume artifacts call verify_manifest() at startup. If anything
has changed since the manifest was written, they refuse to run with a clear
error. This is mistake #50 fixed: manifest checksums are RUNTIME-CHECKED, not
just documented.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ._version import __version__


def sha256_file(path: Path, chunk_size: int = 1 << 16) -> str:
    """Compute SHA-256 of a file, streaming in chunks (memory-safe for big files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_environment() -> dict:
    """Capture the environment in enough detail to debug reproducibility failures.

    Returns a dict with:
      - dpc_version: from _version.py
      - git_commit: short SHA, or "unknown" if not in a git repo
      - python: full version string
      - torch: version string and device count
      - hostname, platform, cpu_count: hardware identity
      - utc_timestamp: ISO-8601
    """
    try:
        import torch
        torch_version = torch.__version__
        torch_cuda = torch.cuda.is_available()
        torch_mps = torch.backends.mps.is_available()
        torch_cuda_devices = torch.cuda.device_count() if torch_cuda else 0
    except ImportError:
        torch_version = "not installed"
        torch_cuda = False
        torch_mps = False
        torch_cuda_devices = 0

    # git commit (short SHA, or "unknown")
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"

    return {
        "dpc_version": __version__,
        "git_commit": git_commit,
        "python": platform.python_version(),
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "torch_cuda_device_count": torch_cuda_devices,
        "torch_mps": torch_mps,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(
    run_dir: Path,
    extra_meta: Optional[dict] = None,
    skip_files: Optional[list[str]] = None,
) -> Path:
    """Walk run_dir, compute SHA256 of every file, write manifest.json.

    Skips:
      - manifest.json itself (would create a circular hash)
      - any file in skip_files
      - hidden files (.tmp, .lock, etc.)
      - __pycache__ contents

    Returns the path of the written manifest.
    """
    run_dir = Path(run_dir).resolve()
    skip = set(skip_files or [])
    skip.add("manifest.json")

    files = []
    total_size = 0
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name in skip:
            continue
        if any(part.startswith(".") for part in p.relative_to(run_dir).parts):
            continue
        if "__pycache__" in p.parts:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue

        # Hash everything under 100 MB; for larger files, hash anyway but allow
        # the caller to skip via skip_files.
        files.append({
            "path": str(p.relative_to(run_dir)),
            "size_bytes": size,
            "sha256": sha256_file(p),
        })
        total_size += size

    manifest = {
        "version": __version__,
        "run_dir": str(run_dir),
        "n_files": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "environment": fingerprint_environment(),
        "files": files,
    }
    if extra_meta:
        manifest["extra"] = extra_meta

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    return manifest_path


def verify_manifest(run_dir: Path) -> dict:
    """Verify that every file recorded in manifest.json still has its expected hash.

    Returns a dict:
      {
        "ok": bool,                      # True if all files verified
        "n_checked": int,
        "n_missing": int,
        "n_mismatched": int,
        "missing": list[str],            # paths
        "mismatched": list[str],         # paths
      }

    Does NOT raise — caller decides what to do with mismatches.
    """
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {
            "ok": False,
            "n_checked": 0,
            "n_missing": 0,
            "n_mismatched": 0,
            "missing": [],
            "mismatched": [],
            "error": f"manifest.json not found in {run_dir}",
        }

    with open(manifest_path) as f:
        manifest = json.load(f)

    missing = []
    mismatched = []
    n_checked = 0
    for entry in manifest.get("files", []):
        rel = entry["path"]
        expected = entry["sha256"]
        full = run_dir / rel
        if not full.is_file():
            missing.append(rel)
            continue
        actual = sha256_file(full)
        n_checked += 1
        if actual != expected:
            mismatched.append(rel)

    return {
        "ok": len(missing) == 0 and len(mismatched) == 0,
        "n_checked": n_checked,
        "n_missing": len(missing),
        "n_mismatched": len(mismatched),
        "missing": missing,
        "mismatched": mismatched,
    }
