#!/usr/bin/env python3
"""Build a reproducible prematch history for the existing next corrections.

This exports observations for later calibration; it does not fit or change
production coefficients. Actual outcomes are attached only after forecasting.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import ROOT, match_key, sha256
from run_next_buyplan import build_next_probabilities


ROUNDS = (1649, 1650, 1653, 1654, 1660, 1663, 1664)
VALIDATION_START = 1660
OUT = ROOT / "data/eval/next_model/correction_history_v1"
SIDES = ("home", "draw", "away")
INPUT_COLUMNS = (
    "home_rest_fatigue", "away_rest_fatigue",
    "home_recent_load_carry", "away_recent_load_carry",
    "home_travel_fatigue", "away_travel_fatigue",
    "away_travel_distance_km", "home_total_fatigue_score", "away_total_fatigue_score",
    "is_rain", "is_heavy_rain", "is_strong_wind", "weather_missing",
    "absence_missing_home", "absence_missing_away",
    "absence_effective_attack_home", "absence_effective_attack_away",
    "absence_effective_defense_home", "absence_effective_defense_away",
)
REQUIRED_INPUT_COLUMNS = (
    "home_total_fatigue_score", "away_total_fatigue_score",
    "is_rain", "is_heavy_rain", "is_strong_wind", "weather_missing",
    "absence_missing_home", "absence_missing_away",
    "absence_effective_attack_home", "absence_effective_attack_away",
    "absence_effective_defense_home", "absence_effective_defense_away",
)
AUDIT_COLUMNS = (
    "lab_final_lambda_home", "lab_final_lambda_away",
    "absence_input_missing_home", "absence_input_missing_away",
    "absence_attack_used_home", "absence_attack_used_away",
    "absence_defense_used_home", "absence_defense_used_away",
    "absence_lambda_delta_home", "absence_lambda_delta_away",
    "final_lambda_home", "final_lambda_away",
    "next_uncertainty_fatigue_signal", "next_uncertainty_weather_signal",
    "next_uncertainty_weather_known", "next_uncertainty_calibration_mix",
)


def build(root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    frames = []
    sources = {}
    for number in ROUNDS:
        round_id = f"toto{number}"
        directory = root / "data/eval/toto_rounds" / round_id
        prediction_path = directory / "snapshot/predictions.csv"
        actual_path = directory / "actual_results.csv"
        snapshot_manifest = directory / "snapshot/manifest.json"
        predictions = pd.read_csv(prediction_path, encoding="utf-8-sig", low_memory=False)
        actual = pd.read_csv(actual_path, encoding="utf-8-sig").sort_values("match_no")
        manifest = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        if len(actual) != 13 or actual["match_no"].tolist() != list(range(1, 14)):
            raise ValueError(f"{round_id}: results must contain match_no 1..13")
        prediction_keys = [match_key(h, a) for h, a in zip(predictions.home_team, predictions.away_team)]
        if len(set(prediction_keys)) != len(prediction_keys):
            raise ValueError(f"{round_id}: duplicate prediction match")
        indexed = predictions.copy()
        indexed["__key"] = prediction_keys
        indexed = indexed.set_index("__key")
        selected = []
        matched_actual = []
        for result in actual.itertuples(index=False):
            key = match_key(result.home_team, result.away_team)
            if key not in indexed.index:
                if number != 1649:
                    raise ValueError(f"{round_id}: missing frozen prediction for {key}")
                continue
            row = indexed.loc[key].copy()
            row["match_no"] = int(result.match_no)
            selected.append(row)
            matched_actual.append(result)
        if not selected:
            raise ValueError(f"{round_id}: no matched predictions")
        selected_frame = pd.DataFrame(selected).reset_index(drop=True)
        missing_inputs = [column for column in REQUIRED_INPUT_COLUMNS if column not in selected_frame]
        if missing_inputs:
            raise ValueError(f"{round_id}: missing correction inputs {missing_inputs}")
        forecast = build_next_probabilities(selected_frame)
        for column in INPUT_COLUMNS:
            if column not in forecast:
                forecast[column] = pd.NA
        columns = ["match_no", "match_id", "datetime", "home_team", "away_team"]
        columns += list(INPUT_COLUMNS) + list(AUDIT_COLUMNS)
        for prefix in ("next_prob_pre_absence", "next_prob_pre_uncertainty", "next_prob"):
            columns += [f"{prefix}_{side}" for side in SIDES]
        missing_outputs = [column for column in columns if column not in forecast]
        if missing_outputs:
            raise ValueError(f"{round_id}: missing forecast outputs {missing_outputs}")
        history = forecast[columns].copy()
        history.insert(0, "round_id", round_id)
        history.insert(1, "split", "validation" if number >= VALIDATION_START else "fit")
        history["actual_result"] = [int(row.result) for row in matched_actual]
        history["actual_home_score"] = [row.home_score for row in matched_actual]
        history["actual_away_score"] = [row.away_score for row in matched_actual]
        frames.append(history)
        sources[round_id] = {
            "prematch_snapshot_created_at": manifest.get("created_at"),
            "predictions": str(prediction_path.relative_to(root)),
            "predictions_sha256": sha256(prediction_path),
            "actual_results": str(actual_path.relative_to(root)),
            "actual_results_sha256": sha256(actual_path),
            "matched_matches": len(history),
            "excluded_unmatched_results": 13 - len(history),
        }
    combined = pd.concat(frames, ignore_index=True)
    report = {
        "version": "next_correction_history_v1",
        "purpose": "Fit and validate existing absence and fatigue/weather corrections; no production weights changed",
        "forecast_source": "recomputed with current build_next_probabilities from frozen prematch predictions",
        "split": {"fit_rounds": [f"toto{n}" for n in ROUNDS if n < VALIDATION_START],
                  "validation_rounds": [f"toto{n}" for n in ROUNDS if n >= VALIDATION_START]},
        "rows": len(combined),
        "fit_rows": int(combined.split.eq("fit").sum()),
        "validation_rows": int(combined.split.eq("validation").sum()),
        "weather_known_rows": int(combined.next_uncertainty_weather_known.sum()),
        "absence_both_known_rows": int((~combined.absence_input_missing_home & ~combined.absence_input_missing_away).sum()),
        "sources": sources,
    }
    return combined, report


def main() -> None:
    frame, manifest = build()
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "matches.csv", index=False, encoding="utf-8-sig")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("rows", "fit_rows", "validation_rows", "weather_known_rows", "absence_both_known_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
