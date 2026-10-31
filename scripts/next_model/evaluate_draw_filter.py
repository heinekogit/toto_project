#!/usr/bin/env python3
"""Evaluate the metadata-only score-structure D filter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ROOT
from draw_filter import (
    FILTER_VERSION_V1, FILTER_VERSION_V2, classify_draw_v1, classify_draw_v2,
)


DEFAULT_INPUT = ROOT / "data/eval/next_model/validation_v2/lab_xg_v0/predictions.csv"
DEFAULT_OUT = ROOT / "data/eval/next_model/validation_v2/draw_filter_v1"


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(frame: pd.DataFrame, class_column: str, fallback_column: str) -> dict:
    actionable = frame[class_column].isin(["D-core", "D-cover"])
    actual_draw = frame["actual_symbol"].eq("D")
    fallback_correct = frame[fallback_column].eq(frame["actual_symbol"])
    decided = ~actual_draw
    raw_hit = frame["raw_prediction"].eq(frame["actual_symbol"])
    core = frame[class_column].eq("D-core")
    core_pick = frame["raw_prediction"].where(~core, "D")
    core_hit = core_pick.eq(frame["actual_symbol"])
    actionable_pick = frame["raw_prediction"].where(~actionable, "D")
    actionable_hit = actionable_pick.eq(frame["actual_symbol"])
    classes = {}
    for draw_class, group in frame.groupby(class_column):
        actuals = group["actual_symbol"].value_counts()
        classes[str(draw_class)] = {
            "rows": len(group),
            "actual_H": int(actuals.get("H", 0)),
            "actual_D": int(actuals.get("D", 0)),
            "actual_A": int(actuals.get("A", 0)),
            "draw_rate": float(group["actual_symbol"].eq("D").mean()),
        }
    return {
        "rows": len(frame),
        "actual_draws": int(actual_draw.sum()),
        "actionable_draw_candidates": int(actionable.sum()),
        "actionable_true_draws": int((actionable & actual_draw).sum()),
        "actionable_precision": ratio(int((actionable & actual_draw).sum()), int(actionable.sum())),
        "actionable_recall": ratio(int((actionable & actual_draw).sum()), int(actual_draw.sum())),
        "fallback_side_accuracy_on_decided_matches": ratio(
            int((fallback_correct & decided).sum()), int(decided.sum())
        ),
        "counterfactual_core_as_primary": {
            "changed_matches": int(core.sum()),
            "rescue": int((core & ~raw_hit & core_hit).sum()),
            "harm": int((core & raw_hit & ~core_hit).sum()),
            "net_rescue": int(core_hit.sum() - raw_hit.sum()),
            "raw_accuracy": float(raw_hit.mean()),
            "filtered_accuracy": float(core_hit.mean()),
        },
        "counterfactual_core_and_cover_as_primary": {
            "note": "diagnostic only; D-cover is intended as portfolio coverage, not primary replacement",
            "changed_matches": int(actionable.sum()),
            "rescue": int((actionable & ~raw_hit & actionable_hit).sum()),
            "harm": int((actionable & raw_hit & ~actionable_hit).sum()),
            "net_rescue": int(actionable_hit.sum() - raw_hit.sum()),
            "raw_accuracy": float(raw_hit.mean()),
            "filtered_accuracy": float(actionable_hit.mean()),
        },
        "classes": classes,
    }


def legacy_strata(frame: pd.DataFrame) -> dict:
    """Report old labels for diagnosis only; they are not classifier inputs."""
    output = {}
    for column in ("match_purchase_type", "draw_purchase_tier", "lab_basis_hint"):
        groups = {}
        for value, group in frame.groupby(column, dropna=False):
            groups[str(value)] = {
                "rows": len(group),
                "actual_draws": int(group["actual_symbol"].eq("D").sum()),
                "actual_draw_rate": float(group["actual_symbol"].eq("D").mean()),
                "mean_new_draw_probability": float(group["next_prob_draw"].mean()),
            }
        output[column] = groups
    return output


def probability_calibration(frame: pd.DataFrame) -> list[dict]:
    bins = pd.cut(
        frame["next_prob_draw"],
        bins=[0.0, 0.22, 0.24, 0.26, 0.28, 1.0],
        include_lowest=True,
    )
    rows = []
    for interval, group in frame.groupby(bins, observed=True):
        rows.append({
            "bin": str(interval),
            "rows": len(group),
            "mean_predicted_draw_probability": float(group["next_prob_draw"].mean()),
            "actual_draws": int(group["actual_symbol"].eq("D").sum()),
            "actual_draw_rate": float(group["actual_symbol"].eq("D").mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--filter", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    classifier = classify_draw_v2 if args.filter == "v2" else classify_draw_v1
    version = FILTER_VERSION_V2 if args.filter == "v2" else FILTER_VERSION_V1
    suffix = args.filter
    assessments = [classifier(row) for _, row in data.iterrows()]
    detail = data.copy()
    class_column = f"draw_filter_class_{suffix}"
    fallback_column = f"draw_filter_fallback_side_{suffix}"
    detail[class_column] = [item.draw_class for item in assessments]
    detail[f"draw_filter_confidence_{suffix}"] = [item.confidence for item in assessments]
    detail[fallback_column] = [item.fallback_side for item in assessments]
    detail[f"draw_filter_reason_{suffix}"] = [item.reason for item in assessments]
    prematch = detail[detail["prematch_snapshot_proven"].fillna(False).astype(bool)]
    report = {
        "filter_version": version,
        "status": "diagnostic; probabilities and raw prediction unchanged",
        "uses_legacy_d_class_as_input": args.filter == "v2",
        "legacy_input_scope": (
            "lab_basis_hint only: trap rejection and split_side cover; never D-core"
            if args.filter == "v2" else "none"
        ),
        "changes_probabilities": False,
        "changes_raw_prediction": False,
        "all_rows": summarize(detail, class_column, fallback_column),
        "prematch_proven_rows": summarize(prematch, class_column, fallback_column),
        "prematch_draw_probability_calibration": probability_calibration(prematch),
        "legacy_strata_diagnostic_only": legacy_strata(prematch),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out / "classifications.csv", index=False, encoding="utf-8-sig")
    (args.out / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.out / "metrics.json")


if __name__ == "__main__":
    main()
