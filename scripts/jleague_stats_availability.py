"""Availability and fallback checks for J.LEAGUE team-stat updates."""

import glob
import os

import pandas as pd
from bs4 import BeautifulSoup


def is_unavailable_page(html):
    raw = html or ""
    if 'id="__next_error__"' in raw or "NEXT_HTTP_ERROR_FALLBACK;404" in raw:
        return True
    soup = BeautifulSoup(raw, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body_text = soup.get_text(" ", strip=True)
    markers = (
        "お探しのページは見つかりませんでした",
        "ページが見つかりません",
        "Page Not Found",
        "当該シーズンのスタッツデータはございません",
    )
    return any(marker in title or marker in body_text for marker in markers)


def validate_stats_fallback(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "team_name" not in df.columns or df["team_name"].dropna().empty:
        return None
    metadata = {"team_name", "team_id", "league", "season", "round", "fetched_date", "stats_missing_ratio"}
    value_cols = [col for col in df.columns if col not in metadata]
    if not value_cols:
        return None
    numeric_values = df[value_cols].apply(pd.to_numeric, errors="coerce")
    filled = int(numeric_values.notna().sum().sum())
    if filled == 0:
        return None
    return {"path": path, "rows": len(df), "filled_values": filled}


def find_latest_stats_fallback(base_dir, league, season):
    current = os.path.join(base_dir, "data", f"team_master_stats_{league}_{season}.csv")
    snapshots = sorted(
        glob.glob(
            os.path.join(
                base_dir,
                "data",
                "stats_snapshots",
                f"team_master_stats_{league}_{season}_asof_*.csv",
            )
        ),
        reverse=True,
    )
    for path in [current] + snapshots:
        result = validate_stats_fallback(path)
        if result:
            return result
    return None
