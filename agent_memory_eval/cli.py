from __future__ import annotations

import argparse
from pathlib import Path

from .suite_runner import run_suite, validate_suite_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent memory benchmark suites.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a suite YAML.")
    validate_parser.add_argument("config", help="Path to suite YAML.")
    validate_parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend name to validate. Can be provided multiple times.",
    )
    validate_parser.add_argument("--limit", type=int, default=None, help="Override suite sample limit.")
    validate_parser.add_argument("--no-eval", action="store_true", help="Validate with evaluation disabled.")

    run_parser = subparsers.add_parser("run", help="Run a suite YAML.")
    run_parser.add_argument("config", help="Path to suite YAML.")
    run_parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend name to run. Can be provided multiple times.",
    )
    run_parser.add_argument("--limit", type=int, default=None, help="Override suite sample limit.")
    run_parser.add_argument("--no-eval", action="store_true", help="Disable benchmark evaluation.")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate without calling LLM/backend.")
    run_parser.add_argument("--quiet", action="store_true", help="Disable progress logs.")

    args = parser.parse_args()

    if args.command == "validate":
        errors = validate_suite_config(
            args.config,
            backend_filter=args.backends,
            limit=args.limit,
            no_eval=args.no_eval,
        )
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Config OK: {Path(args.config)}")
        return

    if args.command == "run":
        run_dirs = run_suite(
            args.config,
            backend_filter=args.backends,
            limit=args.limit,
            no_eval=args.no_eval,
            dry_run=args.dry_run,
            progress=not args.quiet,
        )
        if args.dry_run:
            print(f"Dry run OK: {len(run_dirs)} experiment(s)")
        else:
            print(f"Run complete: {len(run_dirs)} experiment(s)")
