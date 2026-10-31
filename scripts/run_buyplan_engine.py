#!/usr/bin/env python3
"""Explicit BuyPlan engine switch. Next is the official default; legacy is retained for audits."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command_for(args) -> list[str]:
    if args.engine == "legacy":
        if getattr(args, "check_only", False):
            raise ValueError("--check-only is supported only for next")
        command = [sys.executable, str(ROOT / "buyplan.py"), "--in", str(args.input_csv), "--outdir", str(args.outdir)]
        if args.toto_order_csv:
            command += ["--toto-order-csv", str(args.toto_order_csv)]
        if args.context_csv:
            command += ["--context-csv", str(args.context_csv)]
        return command
    if not args.round or args.logical_season is None:
        raise ValueError("--engine next requires --round and --logical-season")
    command = [
        sys.executable, str(ROOT / "scripts/next_model/run_next_buyplan.py"),
        "--in", str(args.input_csv), "--outdir", str(args.outdir),
        "--round", args.round, "--logical-season", str(args.logical_season),
        "--replace",
    ]
    if args.toto_order_csv:
        command += ["--toto-order-csv", str(args.toto_order_csv)]
    if getattr(args, "check_only", False):
        command += ["--check-only"]
    command += ["--max-age-days", str(getattr(args, "max_age_days", 7))]
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["legacy", "next"], default="next")
    parser.add_argument("--in", dest="input_csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--round", default="")
    parser.add_argument("--logical-season", type=int)
    parser.add_argument("--toto-order-csv", type=Path)
    parser.add_argument("--context-csv", type=Path)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-age-days", type=int, default=7)
    args = parser.parse_args()
    command = command_for(args)
    if args.print_command:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
