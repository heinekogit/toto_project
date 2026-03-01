import pandas as pd


def hourly_json_to_df(json_data):
    hourly = json_data.get("hourly", {})
    times = hourly.get("time", [])
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(times, errors="coerce"),
            "temperature_2m": hourly.get("temperature_2m", []),
            "precipitation": hourly.get("precipitation", []),
            "wind_speed_10m": hourly.get("wind_speed_10m", []),
            "weather_code": hourly.get("weather_code", []),
        }
    )
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    return df


def build_features_for_match(match_row, hourly_df):
    kickoff = match_row["kickoff_jst"]
    if pd.isna(kickoff) or hourly_df.empty:
        return {}

    hourly_df = hourly_df.copy()
    hourly_df["diff"] = (hourly_df["datetime"] - kickoff).abs()
    nearest = hourly_df.sort_values("diff").iloc[0]

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
    }

    features["is_rain"] = 1 if features["precip_kickoff"] is not None and features["precip_kickoff"] > 0 else 0
    features["is_heavy_rain"] = 1 if features["precip_kickoff"] is not None and features["precip_kickoff"] >= 5 else 0
    features["is_strong_wind"] = 1 if features["wind_kickoff"] is not None and features["wind_kickoff"] >= 8 else 0

    return features
