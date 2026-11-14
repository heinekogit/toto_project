"""Conservative fixture matching for Football LAB comparison snapshots."""
import re
from datetime import datetime

import pandas as pd

from jleague_team_names import canonical_team_name


def _identity(value):
    match = re.fullmatch(r"(j[123])_(\d{4})_(\d{8})_([^_]+)_([^_]+)", str(value))
    if not match:
        return None
    league, season, stamp, home, away = match.groups()
    try:
        # The ID encodes season, month/day and kickoff minute, including Jan
        # fixtures of a season spanning two calendar years.
        when = datetime.strptime(season + stamp, "%Y%m%d%H%M")
    except ValueError:
        return None
    return (league, season, stamp[:4], canonical_team_name(home), canonical_team_name(away)), when


def merge_football_lab_fixtures(fixtures, compare, tolerance_minutes=60):
    """Exact ID first; otherwise choose the uniquely nearest nearby fixture.

    Never synthesize metrics or borrow from a different date/season. Equally near
    candidates and source reuse fail explicitly. One-hour tolerance covers
    documented schedule changes; same date and ordered teams are still required. Original fixture IDs survive.
    """
    for label, frame in [("fixtures", fixtures), ("compare", compare)]:
        if frame.match_id.isna().any() or frame.match_id.astype(str).str.strip().eq("").any() or frame.match_id.duplicated().any():
            raise ValueError(f"[FOOTBALL_LAB][MATCH] {label}: missing/duplicate match_id")
    source = compare.set_index("match_id", drop=False)
    groups = {}
    for match_id in source.index:
        parsed = _identity(match_id)
        if parsed:
            key, when = parsed
            groups.setdefault(key, []).append((match_id, when))
    chosen, methods, deltas, used = [], [], [], set()
    exact_ids = set(fixtures.match_id) & set(source.index)
    for match_id in fixtures.match_id:
        selected, method, delta = None, "unmatched", None
        if match_id in source.index:
            selected, method, delta = match_id, "exact", 0.0
        else:
            parsed = _identity(match_id)
            if parsed:
                key, when = parsed
                candidates = [(sid, (when - dt).total_seconds() / 60)
                              for sid, dt in groups.get(key, [])
                              if abs((when - dt).total_seconds()) <= tolerance_minutes * 60]
                if candidates:
                    nearest_distance = min(abs(delta) for _, delta in candidates)
                    nearest = [(sid, delta) for sid, delta in candidates
                               if abs(delta) == nearest_distance]
                    if len(nearest) > 1:
                        raise ValueError(f"[FOOTBALL_LAB][MATCH] ambiguous kickoff alias: {match_id} candidates={nearest}")
                    selected, delta = nearest[0]
                    method = "kickoff_alias"
                    if selected in exact_ids:
                        raise ValueError(f"[FOOTBALL_LAB][MATCH] source already assigned by exact ID: {selected}")
        if selected is not None:
            if selected in used:
                raise ValueError(f"[FOOTBALL_LAB][MATCH] source reused: {selected}")
            used.add(selected)
        chosen.append(selected)
        methods.append(method)
        deltas.append(delta)
    flab_cols = [c for c in compare if c.startswith("flab_")]
    out = fixtures.drop(columns=[c for c in fixtures if c in flab_cols]).copy().reset_index(drop=True)
    metrics = source.reindex(chosen)[flab_cols].reset_index(drop=True)
    out = pd.concat([out, metrics], axis=1)
    out["flab_match_source_id"] = chosen
    out["flab_match_method"] = methods
    out["flab_kickoff_delta_minutes"] = deltas
    return out
