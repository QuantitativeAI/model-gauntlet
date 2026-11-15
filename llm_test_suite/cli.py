"""Command line interface: python -m llm_test_suite {validate,run}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_TESTS_DIR,
    load_config,
    load_env,
    resolve_output_dir,
)
from .runner import run_tests
from .validation import validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_test_suite",
        description="model-gauntlet: automated LLM test runner",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to config.json (default: {DEFAULT_CONFIG_PATH})",
    )

    sub.add_parser(
        "validate",
        parents=[common],
        help="Check config, models and test suites without running anything",
    )

    p_run = sub.add_parser("run", parents=[common], help="Run the configured test suites")
    p_run.add_argument("--model", action="append", default=None, metavar="NAME", help="Only run this model (repeatable)")
    p_run.add_argument("--output-dir", default=None, help="Override the results directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    report = validate(config, DEFAULT_TESTS_DIR)

    if args.command == "validate":
        for issue in report.issues:
            print(f"  [{issue.severity.upper():7s}] {issue.subject}: {issue.message}")
        if not report.issues:
            print("  All checks passed.")
        print(
            f"Validation {'passed' if report.ok else 'FAILED'}: "
            f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
        )
        return 0 if report.ok else 1

    # command == "run": fail fast on validation errors
    for issue in report.errors:
        print(f"error: {issue.subject}: {issue.message}", file=sys.stderr)
    if not report.ok:
        print("Fix the errors above (see 'validate' for details).", file=sys.stderr)
        return 1
    for issue in report.warnings:
        print(f"warning: {issue.subject}: {issue.message}")

    output_dir = resolve_output_dir(config, args.output_dir)
    try:
        run_tests(config, DEFAULT_TESTS_DIR, output_dir, model_names=args.model)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
