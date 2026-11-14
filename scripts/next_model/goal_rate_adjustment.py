"""Prematch, symmetric scoring/conceding-rate adjustment.

Only completed matches before the target kickoff can affect a rate. Historical
red cards are not inferred from future suspension/absence fields.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from common import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from jleague_team_names import canonical_team_name

VERSION = "goal_rate_surprise_weight_v1"
SURPRISE_SCALE = 1.5
MIN_WEIGHT = 0.5
MIN_PRIOR_GAMES = 5
MIN_TARGET_GAMES = 8


def load_history(league: str, season: str, data_dir: Path = ROOT / "data") -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"{league}_{season}_latest_results.csv", encoding="utf-8-sig")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for column in ("home_score", "away_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["datetime", "home_score", "away_score"]).copy()
    frame["home_key"] = frame.home_team.map(canonical_team_name)
    frame["away_key"] = frame.away_team.map(canonical_team_name)
    return frame.sort_values("datetime")


def team_scores(history: pd.DataFrame, team: str, cutoff: pd.Timestamp, kind: str) -> np.ndarray:
    games = history.loc[history.datetime.lt(cutoff) &
                        ((history.home_key == team) | (history.away_key == team))]
    home = games.home_key.eq(team).to_numpy()
    if kind == "for":
        return np.where(home, games.home_score, games.away_score).astype(float)
    return np.where(home, games.away_score, games.home_score).astype(float)


def rates_for_match(history: pd.DataFrame, team: str, cutoff: pd.Timestamp,
                    kind: str, *, scale: float = SURPRISE_SCALE) -> dict:
    games = history.loc[history.datetime.lt(cutoff) &
                        ((history.home_key == team) | (history.away_key == team))]
    home = games.home_key.eq(team).to_numpy()
    values = np.where(home, games.home_score, games.away_score).astype(float) if kind == "for" else \
             np.where(home, games.away_score, games.home_score).astype(float)
    if len(values) == 0:
        return {"n": 0, "ordinary": math.nan, "recent5": math.nan,
                "weighted": math.nan, "downweighted": 0}
    weights = np.ones(len(values))
    for j, (_, game) in enumerate(games.iterrows()):
        earlier = game.datetime
        own_prior = team_scores(history, team, earlier, kind)
        opponent = game.away_key if home[j] else game.home_key
        opp_prior = team_scores(history, opponent, earlier,
                                "against" if kind == "for" else "for")
        if len(own_prior) < MIN_PRIOR_GAMES or len(opp_prior) < MIN_PRIOR_GAMES:
            continue
        expected = math.sqrt(max(float(own_prior.mean()), 0.1) *
                             max(float(opp_prior.mean()), 0.1))
        if values[j] <= expected:
            continue
        tail = max(float(poisson.sf(values[j] - 1, expected)), 1e-12)
        weights[j] = max(MIN_WEIGHT, min(1.0, scale / -math.log(tail)))
    return {"n": len(values), "ordinary": float(values.mean()),
            "recent5": float(values[-5:].mean()),
            "weighted": float(np.average(values, weights=weights)),
            "downweighted": int((weights < 1).sum())}


def adjust_frame(selected: pd.DataFrame, *, data_dir: Path = ROOT / "data",
                 scale: float = SURPRISE_SCALE) -> pd.DataFrame:
    """Adjust only observed-goals inputs; retain xG and every audit value."""
    histories: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    for _, source in selected.iterrows():
        row = source.to_dict()
        league = str(source.get("league", ""))
        parts = str(source.get("match_id", "")).split("_", 2)
        season = parts[1] if len(parts) >= 2 and parts[0] == league and parts[1].isdigit() else ""
        kickoff = pd.to_datetime(source.get("datetime"), errors="coerce")
        row["goal_rate_adjustment_version"] = VERSION
        row["goal_rate_historical_red_card_status"] = "not_available"
        if not season or pd.isna(kickoff):
            row["goal_rate_adjustment_status"] = "missing_identity_or_kickoff"
            rows.append(row)
            continue
        key = (league, season)
        if key not in histories:
            path = data_dir / f"{league}_{season}_latest_results.csv"
            histories[key] = load_history(league, season, data_dir) if path.exists() else pd.DataFrame()
        history = histories[key]
        if history.empty:
            row["goal_rate_adjustment_status"] = "missing_history"
            rows.append(row)
            continue
        row["goal_rate_adjustment_status"] = "applied"
        for side in ("home", "away"):
            team = canonical_team_name(source[f"{side}_team"])
            for kind, japanese in (("for", "得点数"), ("against", "失点数")):
                rates = rates_for_match(history, team, kickoff, kind, scale=scale)
                prefix = f"goal_rate_{kind}_{side}"
                row[f"{prefix}_games"] = rates["n"]
                row[f"{prefix}_ordinary"] = rates["ordinary"]
                row[f"{prefix}_weighted"] = rates["weighted"]
                row[f"{prefix}_downweighted_matches"] = rates["downweighted"]
                column = f"stats_1試合平均{japanese}_{side}"
                published = pd.to_numeric(source.get(column), errors="coerce")
                row[f"{prefix}_published"] = published
                if rates["n"] >= MIN_TARGET_GAMES and math.isfinite(rates["ordinary"]) and pd.notna(published):
                    row[column] = max(0.05, float(published) + rates["weighted"] - rates["ordinary"])
                row[f"{prefix}_used"] = row.get(column)
                row[f"{prefix}_delta"] = (float(row[column]) - float(published)) if pd.notna(published) else math.nan
        rows.append(row)
    return pd.DataFrame(rows, index=selected.index)
