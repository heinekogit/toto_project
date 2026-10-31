#!/usr/bin/env python3
"""Fit existing next-model correction coefficients on early rounds only.

The later rounds are used once for evaluation. No production config is edited.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, sha256
from expected_goals import LAMBDA_MAX, LAMBDA_MIN, score_distribution

sys.path.insert(0, str(ROOT / "scripts"))
from uncertainty_calibration import adjust


SOURCE = ROOT / "data/eval/next_model/correction_history_v1/matches.csv"
OUT = ROOT / "data/reports/next_correction_fit_v1"
SIDES = ("home", "draw", "away")
RESULT_INDEX = {1: 0, 0: 1, 2: 2}
IMPACT_CAP = 0.5
MAX_MIX = 0.25
GRID = np.linspace(0, 0.5, 51)


def probabilities_for_absence(frame: pd.DataFrame, coefficient: float) -> np.ndarray:
    home = frame.lab_final_lambda_home.to_numpy(dtype=float)
    away = frame.lab_final_lambda_away.to_numpy(dtype=float)
    ha = frame.absence_attack_used_home.to_numpy(dtype=float)
    hd = frame.absence_defense_used_home.to_numpy(dtype=float)
    aa = frame.absence_attack_used_away.to_numpy(dtype=float)
    ad = frame.absence_defense_used_away.to_numpy(dtype=float)
    # The historical audit columns already contain the production missing-data
    # handling and impact cap. Unknown sides have zero used impact.
    h = np.clip(home + coefficient * (ad - ha), LAMBDA_MIN, LAMBDA_MAX)
    a = np.clip(away + coefficient * (hd - aa), LAMBDA_MIN, LAMBDA_MAX)
    return np.array([
        [distribution[f"prob_{side}"] for side in SIDES]
        for distribution in (score_distribution(x, y) for x, y in zip(h, a))
    ])


def logloss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(-np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes], 1e-12, 1)).mean())


def metrics(probabilities: np.ndarray, outcomes: np.ndarray) -> dict:
    one_hot = np.eye(3)[outcomes]
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == outcomes).astype(float)
    return {
        "matches": len(outcomes),
        "logloss": logloss(probabilities, outcomes),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "mean_top_confidence": float(confidence.mean()),
        "top_accuracy": float(correct.mean()),
        "top_overconfidence": float(confidence.mean() - correct.mean()),
    }


def fit(frame: pd.DataFrame) -> dict:
    early = frame.split.eq("fit").to_numpy()
    later = frame.split.eq("validation").to_numpy()
    if not early.any() or not later.any() or np.flatnonzero(early).max() >= np.flatnonzero(later).min():
        raise ValueError("history must have earlier fit rows and later validation rows")
    outcomes = frame.actual_result.map(RESULT_INDEX).to_numpy(dtype=int)
    if np.any(outcomes < 0) or np.any(outcomes > 2):
        raise ValueError("invalid actual_result")
    known = (~frame.absence_input_missing_home.astype(bool) &
             ~frame.absence_input_missing_away.astype(bool)).to_numpy()
    absence_fit = early & known
    if not absence_fit.any():
        raise ValueError("no known absence rows for fitting")

    absence_candidates = [probabilities_for_absence(frame, float(value)) for value in GRID]
    absence_losses = np.array([logloss(p[absence_fit], outcomes[absence_fit]) for p in absence_candidates])
    best_absence_index = int(absence_losses.argmin())
    absence_coefficient = float(GRID[best_absence_index])
    base = absence_candidates[0]
    fitted_absence = absence_candidates[best_absence_index]
    current_absence = probabilities_for_absence(frame, 0.25)

    fatigue = frame.next_uncertainty_fatigue_signal.to_numpy(dtype=float)
    weather = frame.next_uncertainty_weather_signal.to_numpy(dtype=float)
    # Candidate choice uses early outcomes only. Validation is not used here.
    best = (float("inf"), 0.0, 0.0)
    for fatigue_weight in GRID:
        for weather_weight in GRID:
            candidate, _ = adjust(fitted_absence[early], fatigue[early], weather[early],
                                  float(fatigue_weight), float(weather_weight), MAX_MIX)
            loss = logloss(candidate, outcomes[early])
            if loss < best[0] - 1e-12:
                best = (loss, float(fatigue_weight), float(weather_weight))
    fitted, fitted_mix = adjust(fitted_absence, fatigue, weather, best[1], best[2], MAX_MIX)
    current, current_mix = adjust(current_absence, fatigue, weather, 0.25, 0.25, MAX_MIX)

    versions = {"no_corrections": base, "fitted_absence_only": fitted_absence,
                "fitted_full": fitted, "current_production": current}
    results = {}
    round_rows = []
    for label, mask in (("fit", early), ("validation", later)):
        results[label] = {name: metrics(p[mask], outcomes[mask]) for name, p in versions.items()}
    for round_id, index in frame.groupby("round_id", sort=False).indices.items():
        take = np.asarray(index)
        for name, probabilities in versions.items():
            round_rows.append({"round_id": round_id, "version": name,
                               **metrics(probabilities[take], outcomes[take])})
    report = {
        "version": "next_correction_fit_v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "method": "sequential fit: absence coefficient on both-known early rows, then fatigue/weather uniform-mix weights on all early rows; 0.01 grid, nonnegative [0, 0.5], max mix 0.25; later rows held out",
        "absence_fit_rows": int(absence_fit.sum()),
        "absence_coefficient_goals_per_impact": absence_coefficient,
        "fatigue_weight": best[1], "weather_weight": best[2], "max_mix": MAX_MIX,
        "fit_mean_mix": float(fitted_mix[early].mean()),
        "validation_mean_mix": float(fitted_mix[later].mean()),
        "current_validation_mean_mix": float(current_mix[later].mean()),
        "metrics": results,
        "production_config_changed": False,
    }
    return {"report": report, "rounds": pd.DataFrame(round_rows)}


def main() -> None:
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig")
    result = fit(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["rounds"].to_csv(OUT / "rounds.csv", index=False, encoding="utf-8-sig")
    report = result["report"]
    print(json.dumps({key: report[key] for key in ("absence_fit_rows", "absence_coefficient_goals_per_impact", "fatigue_weight", "weather_weight", "metrics")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
