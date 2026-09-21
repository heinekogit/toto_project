"""Normalize the editable ACL fixture list into prediction fatigue events."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from jleague_team_names import canonical_team_name


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STADIUMS_CSV = ROOT / "data/manual/stadiums.csv"
DEFAULT_STADIUMS_SUPPLEMENTAL_CSV = ROOT / "data/manual/stadiums_supplemental.csv"
DEFAULT_VENUES_CSV = ROOT / "data/manual/acl_venue_locations.csv"

TRAVEL_OVERRIDE_GRADES = {
    "none": 2.0,
    "short": 3.0,
    "medium": 4.0,
    "long": 5.0,
    "long_haul": 7.0,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _text(value: object) -> str:
    return str(value or "").strip()


def _location_key(city: object, country: object) -> tuple[str, str]:
    return (_text(city), _text(country))


def load_jleague_home_coordinates() -> dict[str, tuple[float, float]]:
    frames = []
    for path in (DEFAULT_STADIUMS_CSV, DEFAULT_STADIUMS_SUPPLEMENTAL_CSV):
        if path.exists():
            frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    if not frames:
        raise FileNotFoundError("J.League stadium coordinate master not found")
    stadiums = pd.concat(frames, ignore_index=True)
    stadiums["team"] = stadiums["team"].map(canonical_team_name)
    stadiums["lat"] = pd.to_numeric(stadiums["lat"], errors="coerce")
    stadiums["lon"] = pd.to_numeric(stadiums["lon"], errors="coerce")
    stadiums = stadiums.dropna(subset=["team", "lat", "lon"])
    return {
        str(row.team): (float(row.lat), float(row.lon))
        for row in stadiums.drop_duplicates("team", keep="first").itertuples()
    }


def load_acl_venue_coordinates() -> dict[tuple[str, str], tuple[float, float]]:
    if not DEFAULT_VENUES_CSV.exists():
        return {}
    venues = pd.read_csv(DEFAULT_VENUES_CSV, encoding="utf-8-sig")
    required = {"city", "country", "lat", "lon"}
    if not required.issubset(venues.columns):
        raise ValueError(f"ACL venue master missing columns: {sorted(required - set(venues.columns))}")
    venues["lat"] = pd.to_numeric(venues["lat"], errors="coerce")
    venues["lon"] = pd.to_numeric(venues["lon"], errors="coerce")
    venues = venues.dropna(subset=["city", "country", "lat", "lon"])
    return {
        _location_key(row.city, row.country): (float(row.lat), float(row.lon))
        for row in venues.itertuples()
    }


def classify_distance(distance_km: float) -> tuple[str, float]:
    if distance_km < 1000:
        return "short", 3.0
    if distance_km < 2500:
        return "medium", 4.0
    if distance_km < 5000:
        return "long", 5.0
    return "long_haul", 7.0


def _normalize_legacy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().dropna(how="all")
    out["team"] = out["team"].map(canonical_team_name)
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["fatigue_grade"] = pd.to_numeric(out["fatigue_grade"], errors="coerce")
    if "travel_type" not in out:
        out["travel_type"] = ""
    return out.dropna(subset=["match_date", "team", "fatigue_grade"])


def normalize_acl_schedule(path: str | Path) -> pd.DataFrame:
    """Read either the legacy event schema or the editable home/away schema.

    Historical rows can retain a travel type in ``memo``; that value is treated
    as an explicit compatibility override. New rows are calculated from venue
    and J.League home coordinates.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if {"match_date", "team", "fatigue_grade"}.issubset(df.columns):
        return _normalize_legacy(df)

    required = {"match_date", "home_team", "away_team", "city", "country"}
    if not required.issubset(df.columns):
        raise ValueError(f"ACL schedule missing columns: {sorted(required - set(df.columns))}")

    home_coords = load_jleague_home_coordinates()
    venue_coords = load_acl_venue_coordinates()
    rows = []
    errors = []
    for source_row, row in df.dropna(how="all").iterrows():
        match_date = pd.to_datetime(row.get("match_date"), errors="coerce")
        home = canonical_team_name(row.get("home_team"))
        away = canonical_team_name(row.get("away_team"))
        candidates = [team for team in (home, away) if team in home_coords]
        if pd.isna(match_date):
            errors.append(f"row {source_row + 2}: invalid match_date")
            continue
        if len(candidates) != 1:
            errors.append(f"row {source_row + 2}: expected one J.League club, found {candidates}")
            continue

        team = candidates[0]
        city = _text(row.get("city"))
        country = _text(row.get("country"))
        venue_type = "home" if team == home else "away"
        override = _text(row.get("memo")).lower()
        distance_km = 0.0
        if override in TRAVEL_OVERRIDE_GRADES:
            travel_type = override
            fatigue_grade = TRAVEL_OVERRIDE_GRADES[override]
            if override != "none":
                destination = venue_coords.get(_location_key(city, country))
                if destination:
                    origin = home_coords[team]
                    distance_km = round(haversine_km(*origin, *destination), 1)
        elif venue_type == "home" and country == "日本":
            travel_type, fatigue_grade = "none", 2.0
        else:
            destination = venue_coords.get(_location_key(city, country))
            if destination is None:
                errors.append(f"row {source_row + 2}: venue coordinates missing for {city}, {country}")
                continue
            origin = home_coords[team]
            distance_km = round(haversine_km(*origin, *destination), 1)
            travel_type, fatigue_grade = classify_distance(distance_km)

        record = row.to_dict()
        record.update({
            "match_date": match_date,
            "team": team,
            "opponent": away if team == home else home,
            "venue_type": venue_type,
            "distance_km": distance_km,
            "travel_type": travel_type,
            "fatigue_grade": fatigue_grade,
        })
        rows.append(record)

    if errors:
        raise ValueError("ACL schedule normalization failed: " + "; ".join(errors))
    return pd.DataFrame(rows).sort_values(["match_date", "team"], kind="mergesort").reset_index(drop=True)
