"""Transparent Lab-free expected-goals baseline for acceptance testing.

Version v0 intentionally uses only clearly per-match-scaled production fields:
team scoring rate and opponent conceding rate. Historical xG columns are kept in
the frozen dataset but are not consumed until their units/windows are verified.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import poisson

from common import number


MODEL_VERSION = "base_lambda_goals_xga_geomean_v0"
MODEL_VERSION_V1 = "base_lambda_xg_rate_matchup_v1"
LAMBDA_MIN = 0.15
LAMBDA_MAX = 4.50
DEFAULT_GOALS = 1.35


def _positive(value: object, default: float = DEFAULT_GOALS) -> float:
    parsed = number(value)
    return default if not math.isfinite(parsed) or parsed <= 0 else parsed


def base_lambdas(row) -> tuple[float, float, str]:
    home_attack = _positive(row.get("stats_1試合平均得点数_home"))
    away_defence = _positive(row.get("stats_1試合平均失点数_away"))
    away_attack = _positive(row.get("stats_1試合平均得点数_away"))
    home_defence = _positive(row.get("stats_1試合平均失点数_home"))
    home_lambda = float(np.clip(math.sqrt(home_attack * away_defence), LAMBDA_MIN, LAMBDA_MAX))
    away_lambda = float(np.clip(math.sqrt(away_attack * home_defence), LAMBDA_MIN, LAMBDA_MAX))
    reason = (
        f"sqrt(home_gpg={home_attack:.3f}*away_gapg={away_defence:.3f});"
        f"sqrt(away_gpg={away_attack:.3f}*home_gapg={home_defence:.3f})"
    )
    return home_lambda, away_lambda, reason


def _games_from_xga(row, side: str) -> float:
    total = number(row.get(f"stats_被ゴール期待値_{side}"))
    per_game = number(row.get(f"stats_1試合平均被ゴール期待値_{side}"))
    if not math.isfinite(total) or not math.isfinite(per_game) or total <= 0 or per_game <= 0:
        return math.nan
    games = float(round(total / per_game))
    return games if games >= 1 else math.nan


def _attack_rate_v1(row, side: str) -> tuple[float, float, float]:
    goals_per_game = _positive(row.get(f"stats_1試合平均得点数_{side}"))
    games = _games_from_xga(row, side)
    cumulative_xg = number(row.get(f"stats_ゴール期待値_{side}"))
    xg_per_game = cumulative_xg / games if math.isfinite(cumulative_xg) and math.isfinite(games) else math.nan
    if not math.isfinite(xg_per_game) or xg_per_game <= 0:
        xg_per_game = goals_per_game
    attack = 0.5 * goals_per_game + 0.5 * xg_per_game
    return attack, xg_per_game, games


def _defence_rate_v1(row, side: str) -> tuple[float, float]:
    goals_against = _positive(row.get(f"stats_1試合平均失点数_{side}"))
    xga_per_game = _positive(row.get(f"stats_1試合平均被ゴール期待値_{side}"), goals_against)
    return 0.5 * goals_against + 0.5 * xga_per_game, xga_per_game


def base_lambdas_v1(row) -> tuple[float, float, str]:
    """Build lambdas from per-game scoring/xG and opponent conceding/xGA.

    J.League's attack xG field is cumulative. Games played is reconstructed
    from cumulative xGA / explicitly per-game xGA in the same frozen row.
    No Football Lab source is consumed here.
    """
    home_attack, home_xgpg, home_games = _attack_rate_v1(row, "home")
    away_attack, away_xgpg, away_games = _attack_rate_v1(row, "away")
    home_defence, home_xgapg = _defence_rate_v1(row, "home")
    away_defence, away_xgapg = _defence_rate_v1(row, "away")
    home_lambda = float(np.clip(math.sqrt(home_attack * away_defence), LAMBDA_MIN, LAMBDA_MAX))
    away_lambda = float(np.clip(math.sqrt(away_attack * home_defence), LAMBDA_MIN, LAMBDA_MAX))
    reason = (
        f"H=sqrt(attack={home_attack:.3f}[xGpg={home_xgpg:.3f},games={home_games:.0f}]"
        f"*opp_def={away_defence:.3f}[xGApg={away_xgapg:.3f}]);"
        f"A=sqrt(attack={away_attack:.3f}[xGpg={away_xgpg:.3f},games={away_games:.0f}]"
        f"*opp_def={home_defence:.3f}[xGApg={home_xgapg:.3f}])"
    )
    return home_lambda, away_lambda, reason


def score_distribution(home_lambda: float, away_lambda: float, max_goals: int = 10) -> dict:
    grid = np.arange(max_goals + 1)
    home = poisson.pmf(grid, home_lambda)
    away = poisson.pmf(grid, away_lambda)
    matrix = np.outer(home, away)
    total = float(matrix.sum())
    matrix = matrix / total
    home_prob = float(np.tril(matrix, -1).sum())
    draw_prob = float(np.trace(matrix))
    away_prob = float(np.triu(matrix, 1).sum())
    flat = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return {
        "prob_home": home_prob,
        "prob_draw": draw_prob,
        "prob_away": away_prob,
        "mode_home_goals": int(flat[0]),
        "mode_away_goals": int(flat[1]),
        "prob_0_0": float(matrix[0, 0]),
        "prob_1_1": float(matrix[1, 1]),
        "prob_2_2": float(matrix[2, 2]),
        "grid_mass_before_normalization": total,
    }
