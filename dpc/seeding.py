"""Reproducibility primitives.

EVERY stochastic operation in the codebase must trace to a seed via this module.
There are no implicit RNG calls — torch.randn() without a generator is banned;
np.random.* without a RandomState is banned. The grep tests in tests/ enforce this.

Why so strict: mistake #46/#47 from the catalog (single-seed everything,
no variance characterization). v3 reports mean ± std across 3 seeds for
every metric. That's only possible if every random draw is reproducible from
its seed.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> dict:
    """Seed all global RNGs.

    This sets:
      - Python's `random` module
      - numpy's global RandomState (via np.random.seed)
      - torch CPU RNG
      - torch CUDA RNG (if available)
      - torch MPS RNG (via torch.manual_seed which covers it)

    Use this ONCE at the start of every script. For per-operation reproducibility
    inside a script, use `make_generator()` instead of relying on global state.

    Returns a dict snapshot of the post-seeding state, useful for logging.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS is covered by torch.manual_seed.

    return {
        "seed": seed,
        "torch_initial_seed": torch.initial_seed(),
        "numpy_state_sample": int(np.random.get_state()[1][0]),
        "python_state_sample": random.getstate()[1][0],
    }


def make_generator(seed: int, device: Optional[torch.device] = None) -> torch.Generator:
    """Create a torch.Generator pinned to a specific seed and device.

    Use this for ALL stochastic torch operations that should be reproducible:
      - torch.randn(..., generator=g)
      - torch.randint(..., generator=g)
      - tensor.normal_(generator=g)
      - DataLoader(..., generator=g)

    A Generator on the wrong device crashes torch — pass device=None for CPU
    sampling, the actual device for device-side sampling.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    if device is None:
        gen = torch.Generator()
    else:
        gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def deterministic_split(
    n: int,
    train_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Reproducible permutation split.

    Returns (train_indices, val_indices). Uses numpy.random.RandomState directly
    so that successive calls to this function don't perturb the global numpy
    state — important when this is called from inside a training script that
    has already seeded its own RNG.

    train_frac is the FRACTION going to TRAIN. Val gets 1 - train_frac.
    Both lists are sorted ascending so they're stable under set comparisons.
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if n < 2:
        raise ValueError(f"need at least 2 items to split, got {n}")

    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)

    n_train = max(1, int(round(n * train_frac)))
    if n_train == n:
        n_train = n - 1  # ensure val gets at least 1
    if n_train < 1:
        n_train = 1

    train_idx = sorted(perm[:n_train].tolist())
    val_idx = sorted(perm[n_train:].tolist())
    return train_idx, val_idx
