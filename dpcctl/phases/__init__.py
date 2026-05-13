"""Phase registry.

Importing this module imports every phase submodule, which causes each
to register itself via @register(name) in _base. The orchestrator
relies on the registry being populated before it starts.
"""

# Importing each module triggers the @register(...) decorator. Order
# does not matter; depends_on is resolved at runtime.
from dpcctl.phases import prep            # noqa: F401
from dpcctl.phases import train_p1        # noqa: F401
from dpcctl.phases import diagnose        # noqa: F401
from dpcctl.phases import train_p2        # noqa: F401
from dpcctl.phases import eval_p3         # noqa: F401
from dpcctl.phases import eval_negative_control  # noqa: F401
from dpcctl.phases import aggregate       # noqa: F401

from dpcctl.phases._base import (
    Phase,
    PhaseContext,
    PhaseStatus,
    get_phase_class,
    all_phase_names,
    make_context,
)

__all__ = [
    "Phase",
    "PhaseContext",
    "PhaseStatus",
    "get_phase_class",
    "all_phase_names",
    "make_context",
]
