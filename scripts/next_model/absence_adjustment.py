"""Bounded attack/defense absence correction after the LAB goal model.

The 0.25 goals per impact unit matches the existing prediction-side setting.
Unknown absences contribute no directional correction; they are flagged.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from expected_goals import LAMBDA_MAX, LAMBDA_MIN


MODEL_VERSION = "absence_attack_defense_lambda_v1"
IMPACT_CAP = 0.5
GOALS_PER_IMPACT = 0.25


def _side_impacts(row, side: str) -> tuple[float, float, bool]:
    flag = str(row.get(f"absence_missing_{side}")).strip().lower()
    if flag not in {"true", "false", "1", "0"}:
        raise ValueError(f"absence_missing_{side} is unknown")
    missing = flag in {"true", "1"}
    if missing:
        return 0.0, 0.0, True
    impacts = []
    for role in ("attack", "defense"):
        column = f"absence_effective_{role}_{side}"
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.isna(value) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{column} is invalid")
        impacts.append(min(float(value), IMPACT_CAP))
    return impacts[0], impacts[1], False


def apply_absence_to_lambdas(row, home_lambda: float, away_lambda: float,
                             goals_per_impact: float = GOALS_PER_IMPACT) -> dict:
    if not math.isfinite(goals_per_impact) or goals_per_impact < 0:
        raise ValueError("goals_per_impact must be finite and nonnegative")
    home_attack, home_defense, home_missing = _side_impacts(row, "home")
    away_attack, away_defense, away_missing = _side_impacts(row, "away")
    home_delta = goals_per_impact * (away_defense - home_attack)
    away_delta = goals_per_impact * (home_defense - away_attack)
    return {
        "absence_adjustment_version": MODEL_VERSION,
        "absence_input_missing_home": home_missing,
        "absence_input_missing_away": away_missing,
        "absence_attack_used_home": home_attack,
        "absence_defense_used_home": home_defense,
        "absence_attack_used_away": away_attack,
        "absence_defense_used_away": away_defense,
        "absence_lambda_delta_home": home_delta,
        "absence_lambda_delta_away": away_delta,
        "final_lambda_home": float(np.clip(home_lambda + home_delta, LAMBDA_MIN, LAMBDA_MAX)),
        "final_lambda_away": float(np.clip(away_lambda + away_delta, LAMBDA_MIN, LAMBDA_MAX)),
    }
