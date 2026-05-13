"""DPC-YOLO26 v3.2.0 control plane.

Lives next to the dpc/ library. Provides:

  - dpcctl.cli                Command-line entry point (python -m dpcctl ...)
  - dpcctl.config             Pydantic-ish config schema (dataclasses)
  - dpcctl.orchestrator       Phase dependency graph + execution loop
  - dpcctl.dashboard          HTTP+SSE live dashboard with generic event bus
  - dpcctl.phases.*           One module per phase (registered into a registry)

To run the full pipeline:
    python -m dpcctl run -c configs/quick.json -p all
To serve the dashboard:
    python -m dpcctl serve -c configs/quick.json --port 8080
"""

from dpc import __version__

__all__ = ["__version__"]
