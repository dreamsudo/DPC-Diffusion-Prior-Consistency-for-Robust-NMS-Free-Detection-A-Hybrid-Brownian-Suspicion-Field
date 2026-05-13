"""Command-line entry point for the DPC-YOLO26 control plane.

Subcommands:
    validate  - load and validate a config file
    run       - run the orchestrator
    serve     - start the live dashboard HTTP server
    list-phases - list the registered phases
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_validate(args) -> int:
    from dpcctl.config import load_config, validate_config

    cfg = load_config(args.config)
    issues = validate_config(cfg)
    if issues:
        print(f"VALIDATION FAILED ({len(issues)} issues):")
        for i in issues:
            print(f"  - {i}")
        return 1
    print(f"VALIDATION OK: {args.config}")
    print(f"  name:       {cfg.name}")
    print(f"  seeds:      {cfg.seeds}")
    print(f"  run_dir:    {cfg.run_dir}")
    print(f"  cache_dir:  {cfg.cache_dir}")
    return 0


def cmd_run(args) -> int:
    from dpcctl.config import load_config
    from dpcctl.orchestrator import run_orchestrator

    cfg = load_config(args.config)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    ok = run_orchestrator(cfg, phases, force=args.force)
    return 0 if ok else 1


def cmd_serve(args) -> int:
    from dpcctl.config import load_config
    from dpcctl.dashboard import serve_dashboard

    cfg = load_config(args.config)
    port = args.port or cfg.viz.dashboard_port
    serve_dashboard(cfg, port=port)
    return 0


def cmd_list_phases(args) -> int:
    from dpcctl.phases import all_phase_names, get_phase_class

    for name in all_phase_names():
        cls = get_phase_class(name)
        scope = "shared" if cls.is_shared else "per-seed"
        deps = ", ".join(cls.depends_on) or "(none)"
        print(f"  {name:30s}  scope={scope:8s}  depends_on={deps}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="python -m dpcctl",
        description="DPC-YOLO26 v3.3.0 control plane",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="validate a config file")
    p_val.add_argument("-c", "--config", required=True)
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="run the orchestrator")
    p_run.add_argument("-c", "--config", required=True)
    p_run.add_argument(
        "-p", "--phases", default="all",
        help="comma-separated list of phases, or 'all'",
    )
    p_run.add_argument("--force", action="store_true",
                       help="force re-run of phases marked complete")
    p_run.set_defaults(func=cmd_run)

    p_serve = sub.add_parser("serve", help="start the dashboard")
    p_serve.add_argument("-c", "--config", required=True)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_list = sub.add_parser("list-phases", help="list registered phases")
    p_list.set_defaults(func=cmd_list_phases)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
