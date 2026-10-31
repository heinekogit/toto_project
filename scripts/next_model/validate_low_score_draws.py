#!/usr/bin/env python3
"""Time-ordered audit of scoreline and draw calibration for next probabilities.

The diagnostic Dixon-Coles low-score parameter is fitted on early frozen
matches only. This script does not alter production scores or probabilities.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import poisson

from common import ROOT, sha256


SOURCE = ROOT / "data/eval/next_model/correction_history_v1/matches.csv"
OUT = ROOT / "data/reports/next_low_score_draw_validation"
SIDES = ("home", "draw", "away")
SCORE_MAX = 10
RHO_GRID = np.linspace(-0.2, 0.2, 401)
RESULT_INDEX = {1: 0, 0: 1, 2: 2}
LOW_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))


def score_matrices(frame: pd.DataFrame, rho: float) -> np.ndarray:
    home_lambda = frame.final_lambda_home.to_numpy(dtype=float)
    away_lambda = frame.final_lambda_away.to_numpy(dtype=float)
    goals = np.arange(SCORE_MAX + 1)
    home = poisson.pmf(goals[None, :], home_lambda[:, None])
    away = poisson.pmf(goals[None, :], away_lambda[:, None])
    matrix = home[:, :, None] * away[:, None, :]
    # Dixon-Coles low-score adjustment. rho=0 is the production independent
    # Poisson score model. Use one shared parameter, never a fitted score cell.
    matrix[:, 0, 0] *= 1 - home_lambda * away_lambda * rho
    matrix[:, 0, 1] *= 1 + home_lambda * rho
    matrix[:, 1, 0] *= 1 + away_lambda * rho
    matrix[:, 1, 1] *= 1 - rho
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f"invalid low-score probabilities for rho={rho}")
    return matrix / matrix.sum(axis=(1, 2), keepdims=True)


def hda_probabilities(matrix: np.ndarray) -> np.ndarray:
    home = np.tril(matrix, -1).sum(axis=(1, 2))
    draw = np.diagonal(matrix, axis1=1, axis2=2).sum(axis=1)
    away = np.triu(matrix, 1).sum(axis=(1, 2))
    return np.stack((home, draw, away), axis=1)


def metrics(frame: pd.DataFrame, matrix: np.ndarray) -> dict:
    home_score = frame.actual_home_score.to_numpy(dtype=int)
    away_score = frame.actual_away_score.to_numpy(dtype=int)
    if (home_score > SCORE_MAX).any() or (away_score > SCORE_MAX).any():
        raise ValueError("observed score exceeds SCORE_MAX")
    outcome = frame.actual_result.map(RESULT_INDEX).to_numpy(dtype=int)
    row = np.arange(len(frame))
    score_actual = matrix[row, home_score, away_score]
    hda = hda_probabilities(matrix)
    mix = frame.next_uncertainty_calibration_mix.to_numpy(dtype=float)
    final_hda = (1 - mix[:, None]) * hda + mix[:, None] / 3
    one_hot = np.eye(3)[outcome]
    return {
        "matches": len(frame),
        "scoreline_logloss": float(-np.log(np.clip(score_actual, 1e-12, 1)).mean()),
        "raw_hda_logloss": float(-np.log(np.clip(hda[row, outcome], 1e-12, 1)).mean()),
        "final_hda_logloss": float(-np.log(np.clip(final_hda[row, outcome], 1e-12, 1)).mean()),
        "final_hda_brier": float(np.mean(np.sum((final_hda - one_hot) ** 2, axis=1))),
        "actual_draws": int((outcome == 1).sum()),
        "expected_draws_raw": float(hda[:, 1].sum()),
        "expected_draws_final": float(final_hda[:, 1].sum()),
        "actual_0_0": int(((home_score == 0) & (away_score == 0)).sum()),
        "expected_0_0": float(matrix[:, 0, 0].sum()),
        "actual_1_1": int(((home_score == 1) & (away_score == 1)).sum()),
        "expected_1_1": float(matrix[:, 1, 1].sum()),
        "actual_low_four_cells": int(sum(((home_score == h) & (away_score == a)).sum()
                                         for h, a in LOW_CELLS)),
        "expected_low_four_cells": float(sum(matrix[:, h, a].sum() for h, a in LOW_CELLS)),
    }


def main() -> None:
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig")
    fit = frame.loc[frame.split.eq("fit")].copy()
    validation = frame.loc[frame.split.eq("validation")].copy()
    if len(fit) != 49 or len(validation) != 39:
        raise ValueError("expected 49 earlier fit and 39 later validation matches")
    fit_score = np.array([metrics(fit, score_matrices(fit, float(rho)))["scoreline_logloss"]
                          for rho in RHO_GRID])
    best_index = int(fit_score.argmin())
    fitted_rho = float(RHO_GRID[best_index])
    report = {
        "version": "next_low_score_draw_validation_v1",
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "method": "independent Poisson rho=0 versus one Dixon-Coles low-score rho fitted by exact-score logloss on earlier 49 matches only; later 39 held out; current fatigue/weather uniform mix held fixed for final HDA comparison",
        "rho_grid": [float(RHO_GRID[0]), float(RHO_GRID[-1]), float(RHO_GRID[1] - RHO_GRID[0])],
        "fitted_rho": fitted_rho,
        "segments": {},
        "production_changed": False,
    }
    segments = {"fit": fit, "validation": validation}
    for round_id, group in frame.groupby("round_id", sort=False):
        segments[round_id] = group
    # The thresholds are fixed from the early period before checking heldout.
    total_threshold = float((fit.final_lambda_home + fit.final_lambda_away).median())
    gap_threshold = float((fit.final_lambda_home - fit.final_lambda_away).abs().median())
    report["strata_thresholds_from_fit"] = {"total_lambda_median": total_threshold,
                                             "lambda_gap_median": gap_threshold}
    for label, subset in (("validation_low_total", validation.loc[(validation.final_lambda_home + validation.final_lambda_away) <= total_threshold]),
                          ("validation_high_total", validation.loc[(validation.final_lambda_home + validation.final_lambda_away) > total_threshold]),
                          ("validation_close_lambda", validation.loc[(validation.final_lambda_home - validation.final_lambda_away).abs() <= gap_threshold]),
                          ("validation_wide_lambda", validation.loc[(validation.final_lambda_home - validation.final_lambda_away).abs() > gap_threshold])):
        if not subset.empty:
            segments[label] = subset
    for label, subset in segments.items():
        report["segments"][label] = {
            "independent_poisson": metrics(subset, score_matrices(subset, 0.0)),
            "fitted_low_score": metrics(subset, score_matrices(subset, fitted_rho)),
        }
    # Replayed production values should equal the stored pre/post uncertainty
    # probabilities, proving the comparator follows the existing path.
    raw = hda_probabilities(score_matrices(frame, 0.0))
    stored_raw = frame[[f"next_prob_pre_uncertainty_{side}" for side in SIDES]].to_numpy(dtype=float)
    stored_final = frame[[f"next_prob_{side}" for side in SIDES]].to_numpy(dtype=float)
    mix = frame.next_uncertainty_calibration_mix.to_numpy(dtype=float)
    if not np.allclose(raw, stored_raw, atol=1e-10) or not np.allclose(
            (1 - mix[:, None]) * raw + mix[:, None] / 3, stored_final, atol=1e-10):
        raise AssertionError("independent Poisson comparator does not reproduce saved next probabilities")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fitted_rho": fitted_rho,
                      "fit": report["segments"]["fit"],
                      "validation": report["segments"]["validation"],
                      "validation_low_total": report["segments"]["validation_low_total"],
                      "validation_high_total": report["segments"]["validation_high_total"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
