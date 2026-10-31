"""Shared pre-purchase forecast: model, absences, fatigue and weather.

All ticket generators consume the resulting next_prob_* columns. This module
has no knowledge of ticket counts or placement strategies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from absence_adjustment import apply_absence_to_lambdas
from common import ROOT
from expected_goals import base_lambdas_v1, score_distribution
from lab_matchup import apply_lab_matchup

sys.path.insert(0, str(ROOT / "scripts"))
from uncertainty_calibration import apply_to_frame as apply_uncertainty_calibration


def build_next_probabilities(selected: pd.DataFrame, *, config_path: Path | None = None) -> pd.DataFrame:
    config_path = config_path or ROOT / "data/models/uncertainty_calibration_v1.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("version") != "fatigue_weather_uniform_v1":
        raise ValueError("unsupported uncertainty calibration version")
    absence_weight = float(config["absence_goals_per_impact"])
    rows = []
    for _, source in selected.iterrows():
        base_home, base_away, base_reason = base_lambdas_v1(source)
        lab = apply_lab_matchup(source, base_home, base_away)
        without_absence = score_distribution(lab["final_lambda_home"], lab["final_lambda_away"])
        absence = apply_absence_to_lambdas(source, lab["final_lambda_home"], lab["final_lambda_away"], goals_per_impact=absence_weight)
        distribution = score_distribution(absence["final_lambda_home"], absence["final_lambda_away"])
        row = source.to_dict()
        row.update({
            "base_lambda_home": base_home, "base_lambda_away": base_away,
            "base_lambda_reason": base_reason,
            **lab,
            "lab_final_lambda_home": lab["final_lambda_home"],
            "lab_final_lambda_away": lab["final_lambda_away"],
            **absence,
            "final_lambda_total": absence["final_lambda_home"] + absence["final_lambda_away"],
            **{f"next_prob_pre_absence_{side}": without_absence[f"prob_{side}"]
               for side in ("home", "draw", "away")},
            "next_prob_home": distribution["prob_home"],
            "next_prob_draw": distribution["prob_draw"],
            "next_prob_away": distribution["prob_away"],
            "buyplan_probability_source": "next_score_distribution_HDA + absence_attack_defense_lambda_v1",
            "next_prob_0_0": distribution["prob_0_0"],
            "next_prob_1_1": distribution["prob_1_1"],
            "next_prob_2_2": distribution["prob_2_2"],
        })
        rows.append(row)
    output = pd.DataFrame(rows)
    probability_sum = output[["next_prob_home", "next_prob_draw", "next_prob_away"]].sum(axis=1)
    if not probability_sum.between(0.999999, 1.000001).all():
        raise ValueError("next H/D/A probabilities do not sum to one")
    output = apply_uncertainty_calibration(
        output,
        fatigue_weight=float(config["fatigue_weight"]),
        weather_weight=float(config["weather_weight"]),
        max_mix=float(config["max_mix"]),
        probability_prefix="next_prob",
        audit_prefix="next_uncertainty",
        rain_signal=float(config.get("rain_signal", 0.35)),
        heavy_rain_signal=float(config.get("heavy_rain_signal", 1.0)),
        strong_wind_signal=float(config.get("strong_wind_signal", 0.65)),
    )
    output["condition_absence_goals_per_impact"] = absence_weight
    output["condition_fatigue_weight"] = float(config["fatigue_weight"])
    output["condition_weather_weight"] = float(config["weather_weight"])
    output["condition_max_mix"] = float(config["max_mix"])
    output["condition_rain_signal"] = float(config.get("rain_signal", 0.35))
    output["condition_heavy_rain_signal"] = float(config.get("heavy_rain_signal", 1.0))
    output["condition_strong_wind_signal"] = float(config.get("strong_wind_signal", 0.65))
    output["buyplan_probability_source"] = "next_score_distribution_HDA + absence_attack_defense_lambda_v1 + fatigue_weather_uniform_v1"
    return output

