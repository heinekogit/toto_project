import argparse
import os
from datetime import datetime

import pandas as pd

from feature_builder import build_features_for_match, hourly_json_to_df
from open_meteo_client import OpenMeteoClient


def parse_args():
    parser = argparse.ArgumentParser(description="Update weather_cache.csv from Open-Meteo (CSV-only).")
    parser.add_argument("--matches-csv", required=True, help="upcoming matches CSV (match_id, datetime, stadium)")
    parser.add_argument("--stadiums-csv", required=True, help="stadium master CSV (stadium, lat, lon)")
    parser.add_argument("--out", default="data/manual/weather_cache.csv", help="weather cache CSV path")
    parser.add_argument("--cache-dir", default="weather_raw", help="raw weather cache directory")
    parser.add_argument("--full-refresh", action="store_true", help="ignore existing cache rows")
    return parser.parse_args()


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_matches(path):
    try:
        df = pd.read_csv(path, parse_dates=["datetime"])
    except Exception:
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    match_id_col = _pick_col(df, ["match_id"])
    datetime_col = _pick_col(df, ["datetime", "kickoff_jst"])
    stadium_col = _pick_col(df, ["stadium", "stadium_name"])
    if not (match_id_col and datetime_col and stadium_col):
        raise ValueError("matches-csv must include match_id, datetime, stadium")

    out = pd.DataFrame(
        {
            "match_id": df[match_id_col],
            "datetime": pd.to_datetime(df[datetime_col], errors="coerce"),
            "stadium": df[stadium_col].astype(str).str.strip(),
        }
    )
    out = out.dropna(subset=["match_id"]).copy()
    out["match_id"] = out["match_id"].astype(str).str.strip()
    return out


def load_stadiums(path):
    df = pd.read_csv(path)
    stadium_col = _pick_col(df, ["stadium", "stadium_name"])
    lat_col = _pick_col(df, ["lat", "latitude"])
    lon_col = _pick_col(df, ["lon", "longitude"])
    if not (stadium_col and lat_col and lon_col):
        raise ValueError("stadiums-csv must include stadium, lat, lon")
    out = pd.DataFrame(
        {
            "stadium": df[stadium_col].astype(str).str.strip(),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )
    return out.drop_duplicates(subset=["stadium"], keep="last")


def load_existing_cache(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "match_id" not in df.columns:
        return pd.DataFrame()
    df["match_id"] = df["match_id"].astype(str).str.strip()
    return df.drop_duplicates(subset=["match_id"], keep="last")


def build_weather_row(client, row):
    base = {
        "match_id": row["match_id"],
        "datetime": row["datetime"],
        "stadium": row["stadium"],
        "is_rain": pd.NA,
        "is_heavy_rain": pd.NA,
        "is_strong_wind": pd.NA,
        "temperature": pd.NA,
        "wind_speed": pd.NA,
        "last_updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    kickoff = row["datetime"]
    lat = row["lat"]
    lon = row["lon"]
    if pd.isna(kickoff) or pd.isna(lat) or pd.isna(lon):
        return base

    datasets = client.fetch_hourly_range(kickoff, float(lat), float(lon))
    hourly_df = pd.concat([hourly_json_to_df(d) for d in datasets], ignore_index=True)
    f = build_features_for_match({"kickoff_jst": kickoff}, hourly_df)

    base["is_rain"] = bool(f["is_rain"]) if "is_rain" in f else pd.NA
    base["is_heavy_rain"] = bool(f["is_heavy_rain"]) if "is_heavy_rain" in f else pd.NA
    base["is_strong_wind"] = bool(f["is_strong_wind"]) if "is_strong_wind" in f else pd.NA
    base["temperature"] = f.get("temp_kickoff", pd.NA)
    base["wind_speed"] = f.get("wind_kickoff", pd.NA)
    return base


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    matches = load_matches(args.matches_csv)
    stadiums = load_stadiums(args.stadiums_csv)
    merged = matches.merge(stadiums, how="left", on="stadium", validate="many_to_one")
    merged = merged.drop_duplicates(subset=["match_id"], keep="last")

    existing = pd.DataFrame() if args.full_refresh else load_existing_cache(args.out)
    existing_ids = set(existing["match_id"].tolist()) if not existing.empty else set()
    target = merged[~merged["match_id"].isin(existing_ids)].copy()

    print(f"[weather_cache] matches={len(matches)} stadium_joined={len(merged)} existing={len(existing_ids)} to_update={len(target)}")

    cache_dir = os.path.abspath(args.cache_dir)
    client = OpenMeteoClient(cache_dir=cache_dir)
    rows = []
    success = 0
    failed = 0
    for _, r in target.iterrows():
        match_id = r["match_id"]
        try:
            out = build_weather_row(client, r)
            if pd.notna(out.get("temperature")) or pd.notna(out.get("wind_speed")):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[weather_cache] ERROR processing match_id={match_id}: {repr(e)}")
            raise
        rows.append(out)

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    if combined.empty:
        combined = merged[["match_id", "datetime", "stadium"]].copy()
        combined["is_rain"] = pd.NA
        combined["is_heavy_rain"] = pd.NA
        combined["is_strong_wind"] = pd.NA
        combined["temperature"] = pd.NA
        combined["wind_speed"] = pd.NA
        combined["last_updated_at"] = datetime.now().isoformat(timespec="seconds")
    combined = combined.drop_duplicates(subset=["match_id"], keep="last")
    combined = combined.sort_values("datetime", na_position="last")
    combined.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[weather_cache] wrote={args.out} rows={len(combined)} success={success} failed_or_missing={failed}")


if __name__ == "__main__":
    main()
