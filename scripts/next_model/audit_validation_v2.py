#!/usr/bin/env python3
"""Audit corrected Football LAB chance inputs and candidate model metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ROOT


DEFAULT_INPUT = ROOT / "data/eval/next_model/validation_v2/matches.csv"
DEFAULT_OUT = ROOT / "data/eval/next_model/validation_v2/input_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    fields = [column for column in data if column.startswith("flab_chance_")]
    numeric_fields = [column for column in fields if column != "flab_chance_snapshot_date"]
    numeric = data[numeric_fields].apply(pd.to_numeric, errors="coerce")
    summaries = {
        column: {
            "non_null": int(numeric[column].notna().sum()),
            "min": float(numeric[column].min()),
            "mean": float(numeric[column].mean()),
            "max": float(numeric[column].max()),
        }
        for column in numeric_fields
    }
    correlations = {}
    pairs = (
        ("expected_for_xg", "flab_expected_for_xg", "flab_chance_shots"),
        ("expected_for_xg_vs_build_rate", "flab_expected_for_xg", "flab_chance_build_rate"),
        ("expected_against_xg", "flab_expected_against_xg", "flab_chance_allowed_shots"),
        ("expected_against_xg_vs_allowed_build_rate", "flab_expected_against_xg", "flab_chance_allowed_build_rate"),
    )
    for side in ("home", "away"):
        for label, left, right in pairs:
            correlations[f"{label}_{side}"] = float(
                data[f"{left}_{side}"].corr(data[f"{right}_{side}"])
            )
    report = {
        "rows": len(data),
        "numeric_chance_columns": len(numeric_fields),
        "invalid_or_missing_numeric_cells": int(numeric.isna().sum().sum()),
        "correlations": correlations,
        "field_summaries": summaries,
        "interpretation": (
            "shots/build rates are highly correlated with xG; do not count them as independent additive evidence"
        ),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
