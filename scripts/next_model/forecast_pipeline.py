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
from goal_rate_adjustment import adjust_frame

sys.path.insert(0, str(ROOT / "scripts"))
from condition_policy import prepare_conditions, team_lambda, final_policy


def build_next_probabilities(selected: pd.DataFrame, *, config_path: Path | None = None,
                             goal_rate_adjustment: bool = True) -> pd.DataFrame:
    config_path = config_path or ROOT / "data/models/next_conditions_v2.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("version") not in {"fatigue_weather_uniform_v1", "next_team_conditions_v2"}:
        raise ValueError("unsupported uncertainty calibration version")
    absence_weight = float(config["absence_goals_per_impact"])
    original = selected
    if goal_rate_adjustment:
        selected = adjust_frame(selected)
    rows = []
    for position, (_, source) in enumerate(selected.iterrows()):
        baseline_final = baseline_base = None
        if goal_rate_adjustment:
            old_source = original.iloc[position]
            old_home, old_away, _ = base_lambdas_v1(old_source)
            baseline_base = (old_home, old_away)
            old_lab = apply_lab_matchup(old_source, old_home, old_away)
            old_absence = apply_absence_to_lambdas(
                old_source, old_lab["final_lambda_home"], old_lab["final_lambda_away"],
                goals_per_impact=absence_weight)
            baseline_final = (old_absence["final_lambda_home"], old_absence["final_lambda_away"])
        source = prepare_conditions(source)
        base_home, base_away, base_reason = base_lambdas_v1(source)
        lab = apply_lab_matchup(source, base_home, base_away)
        without_absence = score_distribution(lab["final_lambda_home"], lab["final_lambda_away"])
        absence = apply_absence_to_lambdas(source, lab["final_lambda_home"], lab["final_lambda_away"], goals_per_impact=absence_weight)
        condition_home, condition_away, condition_audit = team_lambda(
            source, absence["final_lambda_home"], absence["final_lambda_away"], config)
        distribution = score_distribution(condition_home, condition_away)
        row = source.to_dict()
        row.update({
            "base_lambda_home": base_home, "base_lambda_away": base_away,
            "base_lambda_reason": base_reason,
            **lab,
            "lab_final_lambda_home": lab["final_lambda_home"],
            "lab_final_lambda_away": lab["final_lambda_away"],
            **absence,
            **condition_audit,
            "final_lambda_home": condition_home, "final_lambda_away": condition_away,
            "final_lambda_total": condition_home + condition_away,
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
        if baseline_base is not None and baseline_final is not None:
            row["goal_rate_base_lambda_delta_home"] = base_home - baseline_base[0]
            row["goal_rate_base_lambda_delta_away"] = base_away - baseline_base[1]
            row["goal_rate_final_lambda_delta_home"] = absence["final_lambda_home"] - baseline_final[0]
            row["goal_rate_final_lambda_delta_away"] = absence["final_lambda_away"] - baseline_final[1]
        rows.append(row)
    output = pd.DataFrame(rows)
    probability_sum = output[["next_prob_home", "next_prob_draw", "next_prob_away"]].sum(axis=1)
    if not probability_sum.between(0.999999, 1.000001).all():
        raise ValueError("next H/D/A probabilities do not sum to one")
    output = final_policy(output, config)
    output["condition_absence_goals_per_impact"] = absence_weight
    output["condition_fatigue_weight"] = float(config["fatigue_weight"])
    output["condition_weather_weight"] = float(config["weather_weight"])
    output["condition_max_mix"] = float(config["max_mix"])
    output["condition_rain_signal"] = float(config.get("rain_signal", 0.35))
    output["condition_heavy_rain_signal"] = float(config.get("heavy_rain_signal", 1.0))
    output["condition_strong_wind_signal"] = float(config.get("strong_wind_signal", 0.65))
    output["buyplan_probability_source"] = "next_score_distribution_HDA + absence_attack_defense_lambda_v1 + " + config["version"]
    if goal_rate_adjustment:
        output["buyplan_probability_source"] += " + goal_rate_surprise_weight_v1"
    return output

