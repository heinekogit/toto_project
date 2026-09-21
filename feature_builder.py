import pandas as pd

from weather_rules import STRONG_WIND_THRESHOLD_KMH, WIND_SPEED_UNIT, strong_wind_from_kmh


def hourly_json_to_df(json_data):
    hourly = json_data.get("hourly", {})
    times = hourly.get("time", [])
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(times, errors="coerce"),
            "temperature_2m": hourly.get("temperature_2m", [None] * len(times)),
            "precipitation": hourly.get("precipitation", [None] * len(times)),
            "wind_speed_10m": hourly.get("wind_speed_10m", [None] * len(times)),
            "weather_code": hourly.get("weather_code", [None] * len(times)),
        }
    )
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    unit = json_data.get("hourly_units", {}).get("wind_speed_10m")
    factor = {"km/h": 1, "m/s": 3.6, "mph": 1.609344, "kn": 1.852}.get(unit)
    if factor is None:
        raise ValueError(f"unknown wind speed unit: {unit!r}")
    for col in ["temperature_2m", "precipitation", "wind_speed_10m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wind_speed_10m"] *= factor
    meta = json_data.get("_acquisition", {})
    df["weather_fetched_at"] = meta.get("fetched_at")
    df["weather_data_kind"] = meta.get("data_kind", "unknown")
    return df


def build_features_for_match(match_row, hourly_df):
    kickoff = match_row["kickoff_jst"]
    if pd.isna(kickoff) or hourly_df.empty:
        return {}

    hourly_df = hourly_df.copy()
    hourly_df["diff"] = (hourly_df["datetime"] - kickoff).abs()
    nearest = hourly_df.sort_values("diff").iloc[0]
    if nearest["diff"] > pd.Timedelta(minutes=60):
        return {"weather_quality_status": "outside_match_window"}

    window_start = kickoff - pd.Timedelta(hours=1)
    window_end = kickoff + pd.Timedelta(hours=1)
    window = hourly_df[(hourly_df["datetime"] >= window_start) & (hourly_df["datetime"] <= window_end)]

    features = {
        "temp_kickoff": float(nearest["temperature_2m"]),
        "precip_kickoff": float(nearest["precipitation"]),
        "wind_kickoff": float(nearest["wind_speed_10m"]),
        "code_kickoff": int(nearest["weather_code"]) if pd.notna(nearest["weather_code"]) else None,
        "temp_avg_pm1h": float(window["temperature_2m"].mean()) if not window.empty else None,
        "precip_sum_pm1h": float(window["precipitation"].sum()) if not window.empty else None,
        "wind_max_pm1h": float(window["wind_speed_10m"].max()) if not window.empty else None,
        "wind_speed_unit": WIND_SPEED_UNIT,
        "strong_wind_threshold_kmh": STRONG_WIND_THRESHOLD_KMH,
    }
    # Hourly buckets intersecting kickoff through +2h (includes both boundaries).
    match_window = hourly_df[hourly_df.datetime.between(kickoff.floor("h"), (kickoff + pd.Timedelta(hours=2)).ceil("h"))]
    match_window = match_window.drop_duplicates("datetime").sort_values("datetime")
    expected = pd.date_range(kickoff.floor("h"), (kickoff + pd.Timedelta(hours=2)).ceil("h"), freq="h")
    complete = set(expected).issubset(set(match_window.datetime)) and not match_window[["precipitation", "wind_speed_10m"]].isna().any().any()
    features["weather_quality_status"] = "complete" if complete else "partial_match_window"
    features["weather_window_hours"] = len(match_window)
    features["precip_max_match"] = match_window.precipitation.max() if not match_window.empty else None
    features["precip_sum_match"] = match_window.precipitation.sum(min_count=1) if not match_window.empty else None
    features["wind_max_match"] = match_window.wind_speed_10m.max() if not match_window.empty else None
    features["last_updated_at"] = nearest.get("weather_fetched_at")
    features["weather_data_kind"] = nearest.get("weather_data_kind", "unknown")
    # Keep kickoff classification for compatibility; expose in-match risk separately.
    features["is_rain"] = int(features["precip_kickoff"] > 0) if pd.notna(features["precip_kickoff"]) else None
    features["is_heavy_rain"] = int(features["precip_kickoff"] >= 5) if pd.notna(features["precip_kickoff"]) else None
    features["is_strong_wind"] = int(strong_wind_from_kmh(features["wind_kickoff"])) if pd.notna(features["wind_kickoff"]) else None
    features["adverse_weather_during_match"] = bool(features["precip_max_match"] >= 5 or features["wind_max_match"] >= STRONG_WIND_THRESHOLD_KMH) if complete else None

    return features
