"""Football Lab matchup correction applied after the base lambda model.

This layer uses only the four complete, per-match xG-rate fields.  It does not
consume legacy H/D/A probabilities or any draw label.  The fixed blend weight
is an a-priori diagnostic setting, not a value fitted to match outcomes.
"""
from __future__ import annotations

import math

import numpy as np

from common import number
from expected_goals import LAMBDA_MAX, LAMBDA_MIN


MODEL_VERSION = "lab_xg_matchup_logblend_v0"
MODEL_VERSION_V1 = "lab_xg_shot_matchup_logblend_v1"
DEFAULT_WEIGHT = 0.25


def _rate(row, name: str) -> float:
    value = number(row.get(name))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"missing_or_invalid:{name}")
    return value


def apply_lab_matchup(
    row, base_home: float, base_away: float, weight: float = DEFAULT_WEIGHT
) -> dict:
    """Return an auditable log-space blend of base and Lab matchup lambdas."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be within [0, 1]")
    home_for = _rate(row, "flab_expected_for_xg_home")
    away_for = _rate(row, "flab_expected_for_xg_away")
    home_against = _rate(row, "flab_expected_against_xg_home")
    away_against = _rate(row, "flab_expected_against_xg_away")
    lab_home = math.sqrt(home_for * away_against)
    lab_away = math.sqrt(away_for * home_against)
    home_delta = math.log(lab_home / base_home)
    away_delta = math.log(lab_away / base_away)
    home_multiplier = math.exp(weight * home_delta)
    away_multiplier = math.exp(weight * away_delta)
    final_home = float(np.clip(base_home * home_multiplier, LAMBDA_MIN, LAMBDA_MAX))
    final_away = float(np.clip(base_away * away_multiplier, LAMBDA_MIN, LAMBDA_MAX))
    return {
        "lab_matchup_lambda_home": lab_home,
        "lab_matchup_lambda_away": lab_away,
        "lab_log_delta_home": home_delta,
        "lab_log_delta_away": away_delta,
        "lab_adjustment_weight": weight,
        "lab_adjustment_multiplier_home": home_multiplier,
        "lab_adjustment_multiplier_away": away_multiplier,
        "final_lambda_home": final_home,
        "final_lambda_away": final_away,
        "lab_adjustment_reason": (
            f"H=sqrt(Lab_for_H={home_for:.3f}*Lab_against_A={away_against:.3f});"
            f"A=sqrt(Lab_for_A={away_for:.3f}*Lab_against_H={home_against:.3f});"
            f"log_blend_weight={weight:.2f}"
        ),
    }


def apply_lab_matchup_v1(
    row, base_home: float, base_away: float, weight: float = DEFAULT_WEIGHT
) -> dict:
    """Combine independent xG and shot-conversion matchup estimates.

    Chance build rate is deliberately excluded because it is algebraically
    derived from shots / attacks and is strongly redundant with shot volume.
    """
    xg = apply_lab_matchup(row, base_home, base_away, weight=0.0)

    def goal_rate(prefix: str, side: str) -> float:
        shots = _rate(row, f"flab_chance_{prefix}shots_{side}")
        conversion = number(row.get(f"flab_chance_{prefix}shot_conversion_{side}"))
        # A zero conversion is a valid observed value but cannot enter a
        # geometric mean; retain a small explicit floor for diagnostic use.
        conversion = max(conversion, 0.5) / 100.0
        return max(shots * conversion, LAMBDA_MIN)

    home_for = goal_rate("", "home")
    away_for = goal_rate("", "away")
    home_against = goal_rate("allowed_", "home")
    away_against = goal_rate("allowed_", "away")
    shot_home = math.sqrt(home_for * away_against)
    shot_away = math.sqrt(away_for * home_against)
    combined_home = math.sqrt(xg["lab_matchup_lambda_home"] * shot_home)
    combined_away = math.sqrt(xg["lab_matchup_lambda_away"] * shot_away)
    home_delta = math.log(combined_home / base_home)
    away_delta = math.log(combined_away / base_away)
    home_multiplier = math.exp(weight * home_delta)
    away_multiplier = math.exp(weight * away_delta)
    return {
        "lab_xg_matchup_lambda_home": xg["lab_matchup_lambda_home"],
        "lab_xg_matchup_lambda_away": xg["lab_matchup_lambda_away"],
        "lab_shot_matchup_lambda_home": shot_home,
        "lab_shot_matchup_lambda_away": shot_away,
        "lab_matchup_lambda_home": combined_home,
        "lab_matchup_lambda_away": combined_away,
        "lab_log_delta_home": home_delta,
        "lab_log_delta_away": away_delta,
        "lab_adjustment_weight": weight,
        "lab_adjustment_multiplier_home": home_multiplier,
        "lab_adjustment_multiplier_away": away_multiplier,
        "final_lambda_home": float(np.clip(base_home * home_multiplier, LAMBDA_MIN, LAMBDA_MAX)),
        "final_lambda_away": float(np.clip(base_away * away_multiplier, LAMBDA_MIN, LAMBDA_MAX)),
        "lab_adjustment_reason": (
            f"geomean(xG_match_H={xg['lab_matchup_lambda_home']:.3f},shot_match_H={shot_home:.3f});"
            f"geomean(xG_match_A={xg['lab_matchup_lambda_away']:.3f},shot_match_A={shot_away:.3f});"
            f"log_blend_weight={weight:.2f}"
        ),
    }
