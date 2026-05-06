from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_experiment_config
from .runner import run_experiment, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent memory experiments on LongMemEval.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate an experiment config.")
    validate_parser.add_argument("config", help="Path to experiment YAML.")

    run_parser = subparsers.add_parser("run", help="Run an experiment.")
    run_parser.add_argument("config", help="Path to experiment YAML.")
    run_parser.add_argument("--limit", type=int, default=None, help="Limit samples for smoke tests.")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate without calling LLM/backend.")

    args = parser.parse_args()

    if args.command == "validate":
        config = load_experiment_config(args.config)
        errors = validate_config(config)
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Config OK: {Path(args.config)}")
        return

    if args.command == "run":
        run_dir = run_experiment(args.config, limit=args.limit, dry_run=args.dry_run)
        if args.dry_run:
            print(f"Dry run OK: {run_dir}")
        else:
            print(f"Run complete: {run_dir}")

