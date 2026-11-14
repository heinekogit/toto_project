#!/usr/bin/env python3
"""Match a toto round's dated cards against the J.League official schedule.

The toto list owns the target date; the official schedule supplies kickoff and venue.
No league round number or same-card match on another date is used as a fallback.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from jleague_team_names import canonical_team_name


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_URL = "https://data.j-league.or.jp/SFMS01/search"


def target_date(value: str, calendar_year: int) -> tuple[pd.Timestamp, str | None]:
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2})月(\d{1,2})日(?:\s+(\d{1,2}:\d{2}))?", text)
    if not match:
        raise ValueError(f"toto節リストのmatch_date形式が不明: {text!r}")
    return pd.Timestamp(year=calendar_year, month=int(match[1]), day=int(match[2])), match[3]


def fetch_official_frame(year: int, frame: int, month: int) -> pd.DataFrame:
    params = {"competition_years": year, "competition_frame_ids": frame,
              "section_months": month, "tv_relay_station_name": ""}
    response = requests.get(OFFICIAL_URL, params=params, timeout=(5, 25))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.select_one("table.table-base00.search-table, table.search-table")
    if table is None or table.find("tbody") is None:
        raise RuntimeError(f"Jリーグ公式の日程表がありません: {response.url}")
    head = table.find("thead")
    headers = [cell.get_text(" ", strip=True) for cell in head.find_all("th")] if head else []
    rows = [[cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            for tr in table.find("tbody").find_all("tr")]
    if not headers or any(len(row) != len(headers) for row in rows):
        raise RuntimeError(f"Jリーグ公式の日程表の列構造が想定外: {response.url}")
    frame_df = pd.DataFrame(rows, columns=headers)
    needed = {"試合日", "K/O時刻", "ホーム", "アウェイ", "スタジアム"}
    if not needed.issubset(frame_df.columns):
        raise RuntimeError(f"Jリーグ公式の日程表の列不足: {sorted(needed - set(frame_df.columns))}")
    frame_df["official_source_url"] = response.url
    frame_df["competition_frame_id"] = frame
    return frame_df


def normalize_official(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    dates = frame["試合日"].astype(str).str.replace(r"\s*\([^)]*\)", "", regex=True).str.strip()
    times = frame["K/O時刻"].astype(str).str.extract(r"(\d{1,2}:\d{2})")[0]
    frame["official_date"] = pd.to_datetime(dates, format="%y/%m/%d", errors="coerce")
    frame["official_kickoff"] = pd.to_datetime(dates + " " + times.fillna(""),
                                                 format="%y/%m/%d %H:%M", errors="coerce")
    frame["home_key"] = frame["ホーム"].map(canonical_team_name)
    frame["away_key"] = frame["アウェイ"].map(canonical_team_name)
    return frame


def resolve(order: pd.DataFrame, official: pd.DataFrame, round_id: str,
            logical_season: int, calendar_year: int) -> pd.DataFrame:
    digits = str(round_id).lower().removeprefix("toto")
    target = order[(order["season"].astype(str) == str(logical_season)) &
                   (order["toto_round"].astype(str) == digits)].copy()
    if len(target) != 13 or target["match_no"].nunique() != 13:
        raise RuntimeError(f"toto{digits}: toto節リストは13試合・番号一意が必要 (rows={len(target)})")
    official = normalize_official(official)
    resolved = []
    errors = []
    for _, row in target.iterrows():
        day, listed_time = target_date(row["match_date"], calendar_year)
        home = canonical_team_name(row["home_team"])
        away = canonical_team_name(row["away_team"])
        hit = official[(official["official_date"] == day) &
                       (official["home_key"] == home) & (official["away_key"] == away)]
        if listed_time:
            hit = hit[hit["official_kickoff"].dt.strftime("%H:%M") == listed_time]
        hit = hit.drop_duplicates(subset=["official_date", "official_kickoff", "home_key", "away_key"])
        label = f"No.{row['match_no']} {day.date()} {home}-{away}"
        if len(hit) != 1:
            errors.append(f"{label}: 公式日程との一致が{len(hit)}件（日時・対戦カードを確認）")
            continue
        match = hit.iloc[0]
        if pd.isna(match["official_kickoff"]) or not str(match["スタジアム"]).strip():
            errors.append(f"{label}: 公式の開始時刻または会場が未確定")
            continue
        frame_id = int(match["competition_frame_id"])
        league = "j2" if frame_id == 2 else "j1"
        match_id = f"{league}_{calendar_year}_{match['official_kickoff']:%m%d%H%M}_{home}_{away}"
        resolved.append({"season": logical_season, "toto_round": int(digits),
                         "match_no": int(row["match_no"]), "league": league,
                         "competition_frame_id": int(match["competition_frame_id"]),
                         "match_date_from_toto": str(day.date()),
                         "match_id": match_id, "datetime": match["official_kickoff"],
                         "stadium": match["スタジアム"], "home_team": match["ホーム"],
                         "away_team": match["アウェイ"],
                         "official_source_url": match["official_source_url"]})
    if errors:
        raise RuntimeError("totoとJリーグ公式の照合失敗:\n" + "\n".join(errors))
    return pd.DataFrame(resolved).sort_values("match_no")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, help="例: toto1672")
    parser.add_argument("--logical-season", type=int, required=True)
    parser.add_argument("--calendar-year", type=int, required=True)
    parser.add_argument("--order", default=str(ROOT / "data/manual/toto節リスト.csv"))
    parser.add_argument("--official-csv", help="公式日程の保存CSV。未指定時は公式サイトから取得")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    order = pd.read_csv(args.order, dtype=str, encoding="utf-8-sig")
    if args.official_csv:
        official = pd.read_csv(args.official_csv, dtype=str, encoding="utf-8-sig")
    else:
        months = {target_date(value, args.calendar_year)[0].month for value in order.loc[
            (order["season"] == str(args.logical_season)) &
            (order["toto_round"] == args.round.lower().removeprefix("toto")), "match_date"]}
        official = pd.concat([fetch_official_frame(args.calendar_year, frame, month)
                              for month in sorted(months) for frame in (1, 2, 11)], ignore_index=True)
    result = resolve(order, official, args.round, args.logical_season, args.calendar_year)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[TOTO_OFFICIAL] {args.round}: {len(result)} fixtures matched; saved={path}")


if __name__ == "__main__":
    main()
