#!/usr/bin/env python3
"""Evaluate the first Lab-free lambda baseline against frozen current outputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, result_symbol
from draw_filter import FILTER_VERSION, classify_draw
from expected_goals import MODEL_VERSION, MODEL_VERSION_V1, base_lambdas, base_lambdas_v1, score_distribution
from lab_matchup import (
    MODEL_VERSION as LAB_MODEL_VERSION,
    MODEL_VERSION_V1 as LAB_MODEL_VERSION_V1,
    apply_lab_matchup,
    apply_lab_matchup_v1,
)


DEFAULT_INPUT = ROOT / "data/eval/next_model/validation_v1/matches.csv"
DEFAULT_OUT = ROOT / "data/eval/next_model/validation_v1/base_lambda_v0"
SYMBOLS = ("H", "D", "A")


def _probability_columns(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)


def metrics(frame: pd.DataFrame, prefix: str) -> dict:
    probs = _probability_columns(frame, prefix)
    actual_symbols = frame["actual_symbol"].tolist()
    actual_idx = np.array([SYMBOLS.index(symbol) for symbol in actual_symbols])
    one_hot = np.eye(3)[actual_idx]
    clipped = np.clip(probs, 1e-12, 1.0)
    predicted_idx = probs.argmax(axis=1)
    draw_predictions = predicted_idx == 1
    actual_draws = actual_idx == 1
    return {
        "rows": len(frame),
        "accuracy": float((predicted_idx == actual_idx).mean()),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(frame)), actual_idx]))),
        "predicted_draws": int(draw_predictions.sum()),
        "actual_draws": int(actual_draws.sum()),
        "draw_precision": float((draw_predictions & actual_draws).sum() / draw_predictions.sum())
        if draw_predictions.any() else math.nan,
        "draw_recall": float((draw_predictions & actual_draws).sum() / actual_draws.sum())
        if actual_draws.any() else math.nan,
    }


def evaluate(source: pd.DataFrame, model: str = "v0") -> tuple[pd.DataFrame, dict]:
    lambda_builder = base_lambdas if model == "v0" else base_lambdas_v1
    model_version = {
        "v0": MODEL_VERSION, "v1": MODEL_VERSION_V1,
        "v2": LAB_MODEL_VERSION, "v3": LAB_MODEL_VERSION_V1,
    }[model]
    rows = []
    for _, row in source.iterrows():
        home_lambda, away_lambda, reason = lambda_builder(row)
        lab_fields = {}
        final_home, final_away = home_lambda, away_lambda
        if model in {"v2", "v3"}:
            lab_fields = (
                apply_lab_matchup_v1(row, home_lambda, away_lambda)
                if model == "v3" else apply_lab_matchup(row, home_lambda, away_lambda)
            )
            final_home = lab_fields["final_lambda_home"]
            final_away = lab_fields["final_lambda_away"]
        distribution = score_distribution(final_home, final_away)
        record = row.to_dict()
        record.update({
            "actual_symbol": result_symbol(row["actual_result"]),
            "base_lambda_home": home_lambda,
            "base_lambda_away": away_lambda,
            "base_lambda_total": home_lambda + away_lambda,
            "base_lambda_diff": home_lambda - away_lambda,
            "base_lambda_reason": reason,
            "final_lambda_home": final_home,
            "final_lambda_away": final_away,
            "final_lambda_total": final_home + final_away,
            "final_lambda_diff": final_home - final_away,
            "next_prob_home": distribution["prob_home"],
            "next_prob_draw": distribution["prob_draw"],
            "next_prob_away": distribution["prob_away"],
            "next_mode_score": f"{distribution['mode_home_goals']}-{distribution['mode_away_goals']}",
            "next_prob_0_0": distribution["prob_0_0"],
            "next_prob_1_1": distribution["prob_1_1"],
            "next_prob_2_2": distribution["prob_2_2"],
            "poisson_grid_mass": distribution["grid_mass_before_normalization"],
        })
        record.update(lab_fields)
        assessment = classify_draw(record)
        record.update({
            "raw_prediction": max(
                ("H", "D", "A"),
                key=lambda symbol: record[{"H": "next_prob_home", "D": "next_prob_draw", "A": "next_prob_away"}[symbol]],
            ),
            "draw_filter_class": assessment.draw_class,
            "draw_filter_confidence": assessment.confidence,
            "draw_filter_fallback_side": assessment.fallback_side,
            "draw_filter_reason": assessment.reason,
        })
        rows.append(record)
    detail = pd.DataFrame(rows)
    current_available = detail[["prob_final_home", "prob_final_draw", "prob_final_away"]].notna().all(axis=1)
    metric_label = "next_lab_matchup" if model in {"v2", "v3"} else "next_lab_free"
    comparisons = {metric_label: metrics(detail, "next_prob")}
    if current_available.all():
        comparisons["current_final"] = metrics(detail, "prob_final")
    prematch = detail[detail["prematch_snapshot_proven"].fillna(False).astype(bool)]
    prematch_metrics = {metric_label: metrics(prematch, "next_prob")}
    if not prematch.empty and prematch[["prob_final_home", "prob_final_draw", "prob_final_away"]].notna().all(axis=1).all():
        prematch_metrics["current_final"] = metrics(prematch, "prob_final")
    summary = {
        "model_version": model_version,
        "draw_filter_version": FILTER_VERSION,
        "status": "diagnostic baseline; not production-ready",
        "draw_filter_changes_probabilities": False,
        "draw_filter_changes_raw_prediction": False,
        "lab_features_used": model in {"v2", "v3"},
        "lab_features": (
            ["expected_for_xg", "expected_against_xg", "shots", "shot_conversion",
             "allowed_shots", "allowed_shot_conversion"] if model == "v3"
            else (["expected_for_xg", "expected_against_xg"] if model == "v2" else [])
        ),
        "lab_adjustment_weight": 0.25 if model in {"v2", "v3"} else None,
        "lab_weight_selection": "a-priori diagnostic; not outcome-fitted" if model in {"v2", "v3"} else None,
        "xg_features_used": model in {"v1", "v2", "v3"},
        "xg_unit_treatment": (
            "cumulative attack xG divided by games reconstructed from cumulative/per-game xGA"
            if model in {"v1", "v2", "v3"} else "excluded pending unit validation"
        ),
        "all_rows": comparisons,
        "prematch_proven_rows": prematch_metrics,
        "draw_class_counts": {
            str(key): int(value) for key, value in detail["draw_filter_class"].value_counts().items()
        },
        "draw_class_actuals": {
            str(draw_class): {
                str(symbol): int(count)
                for symbol, count in group["actual_symbol"].value_counts().items()
            }
            for draw_class, group in detail.groupby("draw_filter_class")
        },
    }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", choices=["v0", "v1", "v2", "v3"], default="v0")
    args = parser.parse_args()
    source = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    detail, summary = evaluate(source, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out / "predictions.csv", index=False, encoding="utf-8-sig")
    (args.out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.out / "metrics.json")


if __name__ == "__main__":
    main()
