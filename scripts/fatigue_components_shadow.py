"""Cause-preserving fatigue calculations for shadow evaluation only."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


def finite_carry(history, weights=(0.35, 0.20, 0.10)):
    """Sum the last three event causes once; never carry a prior aggregate."""
    totals = defaultdict(float)
    for weight, event in zip(weights, reversed(history[-len(weights):])):
        for cause in ("recovery", "travel", "external"):
            totals[cause] += float(event.get(cause, 0.0)) * float(weight)
    return dict(totals)


def build_nonrecursive_components(matches, matrix, external_events, fatigue_module):
    """Build a legacy-scale, nonrecursive shadow while retaining every cause."""
    matrix = fatigue_module.normalize_distance_matrix(matrix)
    events = [("league", row.datetime, row) for row in matches.itertuples(index=False)]
    if external_events is not None and not external_events.empty:
        events += [("external", row.datetime, row) for row in external_events.itertuples(index=False)]
    events.sort(key=lambda item: (item[1], 0 if item[0] == "external" else 1))

    last_event = {}
    history = defaultdict(list)
    league_team_days = set()
    for row in matches.itertuples(index=False):
        day = pd.Timestamp(row.datetime).normalize()
        league_team_days.add((fatigue_module.canonical_team_name(row.home_team), day))
        league_team_days.add((fatigue_module.canonical_team_name(row.away_team), day))
    rows = []
    for kind, when, row in events:
        if kind == "external":
            team = fatigue_module.canonical_team_name(row.team)
            if (team, pd.Timestamp(when).normalize()) in league_team_days:
                continue
            load = pd.to_numeric(getattr(row, "event_load", np.nan), errors="coerce")
            history[team].append({"at": when, "recovery": 0.0, "travel": 0.0,
                                  "external": 0.0 if pd.isna(load) else float(load),
                                  "external_unknown": bool(pd.isna(load))})
            last_event[team] = when
            continue

        home = fatigue_module.canonical_team_name(row.home_team)
        away = fatigue_module.canonical_team_name(row.away_team)
        record = {"match_id": row.match_id, "datetime": when,
                  "home_team": row.home_team, "away_team": row.away_team}
        for side, team, opponent in (("home", home, away), ("away", away, home)):
            prior = last_event.get(team)
            rest_hours = (when - prior).total_seconds() / 3600 if prior is not None else np.nan
            recovery = fatigue_module.calc_rest_fatigue(rest_hours / 24) if math.isfinite(rest_hours) else 0.0
            travel_km = 0.0
            lookup = "not_applicable_home"
            if side == "away":
                lookup = "team_not_found"
                if team in matrix.index and opponent in matrix.columns:
                    value = pd.to_numeric(matrix.loc[team, opponent], errors="coerce")
                    if pd.notna(value):
                        travel_km = float(value)
                        lookup = "hit"
            travel = travel_km * float(fatigue_module.TRAVEL_DISTANCE_WEIGHT)
            recent = [event for event in history[team] if pd.Timedelta(0) < when - event["at"] <= pd.Timedelta(days=14)]
            carry = finite_carry(recent, fatigue_module.RECENT_MATCH_CARRY_WEIGHTS)
            carry_recovery = carry.get("recovery", 0.0)
            carry_travel = carry.get("travel", 0.0)
            carry_external = carry.get("external", 0.0)
            total = recovery + travel + carry_recovery + carry_travel + carry_external
            record.update({
                f"{side}_rest_hours": rest_hours,
                f"{side}_recovery_load": recovery,
                f"{side}_travel_km": travel_km,
                f"{side}_travel_lookup_status": lookup,
                f"{side}_travel_load": travel,
                f"{side}_carry_recovery_load": carry_recovery,
                f"{side}_carry_travel_load": carry_travel,
                f"{side}_carry_external_load": carry_external,
                f"{side}_matches_last7d": sum(when - event["at"] <= pd.Timedelta(days=7) for event in recent),
                f"{side}_matches_last14d": len(recent),
                f"{side}_external_load_unknown_count": sum(
                    bool(event.get("external_unknown", False)) for event in recent
                ),
                f"{side}_component_total_legacy_scale": total,
            })
        rows.append(record)
        for team, recovery, travel in (
            (home, record["home_recovery_load"], record["home_travel_load"]),
            (away, record["away_recovery_load"], record["away_travel_load"]),
        ):
            history[team].append({"at": when, "recovery": recovery, "travel": travel, "external": 0.0})
            last_event[team] = when
    return pd.DataFrame(rows)
