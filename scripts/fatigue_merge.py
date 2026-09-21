"""Side-effect-free helpers for matching fatigue rows to fixtures."""

from __future__ import annotations

import pandas as pd


def validate_prediction_fatigue(frame):
    """Reject old or incomplete measurement inputs before prediction is generated."""
    if frame.empty:
        return
    required = {'fatigue_measurement_version', 'travel_lookup_status', 'away_travel_distance_km',
                'home_rest_hours', 'away_rest_hours', 'home_external_events_json', 'away_external_events_json'}
    if not required.issubset(frame):
        raise ValueError(f'FATIGUE_INPUT: missing measurement columns {sorted(required - set(frame))}; rerun fatigue calculation')
    bad = ~frame.fatigue_measurement_version.eq('conditions_v2') | ~frame.travel_lookup_status.eq('hit')
    distance = pd.to_numeric(frame.away_travel_distance_km, errors='coerce')
    bad |= distance.isna() | distance.lt(0) | distance.isin([float('inf'), float('-inf')])
    if bad.any():
        raise ValueError(f'FATIGUE_INPUT: {int(bad.sum())}/{len(frame)} missing/invalid measurements; prediction stopped')


def add_fatigue_time_aliases(left_df, right_df, merge_keys, tolerance_minutes=10):
    """Alias a unique nearest same-matchup row when kickoff times drift slightly."""
    if left_df.empty or right_df.empty:
        return right_df, 0
    exact = left_df[merge_keys].merge(
        right_df[merge_keys].drop_duplicates(), on=merge_keys, how="left", indicator=True
    )
    missing = exact[exact["_merge"].eq("left_only")][merge_keys].drop_duplicates()
    aliases = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    for row in missing.itertuples(index=False):
        candidates = right_df[
            right_df["home_team"].eq(row.home_team)
            & right_df["away_team"].eq(row.away_team)
            & right_df["datetime"].notna()
        ].copy()
        if candidates.empty or pd.isna(row.datetime):
            continue
        candidates["__time_delta"] = (candidates["datetime"] - row.datetime).abs()
        candidates = candidates[candidates["__time_delta"].le(tolerance)].sort_values("__time_delta")
        if candidates.empty:
            continue
        nearest = candidates.iloc[0]
        if len(candidates) > 1 and candidates.iloc[1]["__time_delta"] == nearest["__time_delta"]:
            continue
        alias = nearest.drop(labels=["__time_delta"]).copy()
        alias["datetime"] = row.datetime
        aliases.append(alias)
    if not aliases:
        return right_df, 0
    augmented = pd.concat([right_df, pd.DataFrame(aliases)], ignore_index=True, sort=False)
    augmented = augmented.drop_duplicates(subset=merge_keys, keep="last")
    return augmented, len(aliases)
