#!/usr/bin/env python3
"""Build a non-production toto match manifest and recent-history audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jleague_team_names import canonical_team_name

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _norm(series: pd.Series) -> pd.Series:
    return series.map(canonical_team_name)


def build(season: int, logical_season: int, toto_round: int, output_dir: Path) -> tuple[Path, Path]:
    order = _read(ROOT / "data/manual/toto節リスト.csv")
    order = order[(order["season"] == logical_season) & (order["toto_round"] == toto_round)].copy()
    if len(order) != 13:
        raise ValueError(f"toto{toto_round}: toto order must have 13 rows (found {len(order)})")
    order["home_key"] = _norm(order["home_team"])
    order["away_key"] = _norm(order["away_team"])

    scheduled = []
    for league in ("j1", "j2", "j3"):
        path = ROOT / f"data/{league}_{season}_upcoming.csv"
        if not path.exists():
            continue
        frame = _read(path)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["home_key"] = _norm(frame["home_team"])
        frame["away_key"] = _norm(frame["away_team"])
        frame["source_league"] = league
        scheduled.append(frame)
    schedule = pd.concat(scheduled, ignore_index=True)

    special = _read(ROOT / "data/manual/toto_special_matches.csv")
    special = special[(special["logical_season"] == logical_season) & (special["toto_round"] == toto_round)].copy()
    special["datetime"] = pd.to_datetime(special["datetime"], errors="coerce")
    special["home_key"] = _norm(special["home_team"])
    special["away_key"] = _norm(special["away_team"])

    rows = []
    for item in order.sort_values("match_no").itertuples():
        hit = schedule[(schedule.home_key == item.home_key) & (schedule.away_key == item.away_key)]
        hit = hit[hit.datetime.dt.month.eq(9) & hit.datetime.dt.day.eq(2)]
        if not hit.empty:
            m = hit.sort_values("datetime").iloc[0]
            rows.append(dict(match_no=item.match_no, datetime=m.datetime, competition="league",
                             source_league=m.source_league, home_team=item.home_key, away_team=item.away_key,
                             stadium=m.stadium, source="official_league_schedule"))
            continue
        hit = special[special.match_no == item.match_no]
        if len(hit) != 1:
            raise ValueError(f"match {item.match_no} {item.home_key}-{item.away_key}: schedule not found")
        m = hit.iloc[0]
        if m.home_key != item.home_key or m.away_key != item.away_key:
            raise ValueError(f"match {item.match_no}: supplemental teams do not match toto order")
        rows.append(dict(match_no=item.match_no, datetime=m.datetime, competition=m.competition,
                         source_league="mixed", home_team=item.home_key, away_team=item.away_key,
                         stadium=m.stadium, source="manual_special_match"))

    manifest = pd.DataFrame(rows).sort_values("match_no")
    manifest.insert(0, "toto_round", toto_round)
    manifest.insert(0, "logical_season", logical_season)
    manifest.insert(0, "season", season)
    if manifest[["datetime", "home_team", "away_team"]].isna().any().any() or len(manifest) != 13:
        raise ValueError("manifest completeness check failed")

    result_frames = []
    for league in ("j1", "j2", "j3"):
        path = ROOT / f"data/{league}_{season}_latest_results.csv"
        if not path.exists():
            continue
        frame = _read(path)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["home_team"] = _norm(frame["home_team"])
        frame["away_team"] = _norm(frame["away_team"])
        frame["source_league"] = league
        result_frames.append(frame)
    results = pd.concat(result_frames, ignore_index=True)
    results = results[results.home_score.notna() & results.away_score.notna()].copy()

    history_rows = []
    for match in manifest.itertuples():
        for side, team in (("home", match.home_team), ("away", match.away_team)):
            past = results[((results.home_team == team) | (results.away_team == team)) &
                           (results.datetime < match.datetime)].sort_values("datetime").tail(3)
            if past.empty:
                history_rows.append(dict(match_no=match.match_no, side=side, team=team, history_no=0,
                                         source_league="", previous_datetime=pd.NaT, venue_side="",
                                         opponent="", goals_for=pd.NA, goals_against=pd.NA,
                                         days_before_target=pd.NA, status="missing"))
                continue
            for history_no, (_, p) in enumerate(past.iloc[::-1].iterrows(), start=1):
                was_home = p.home_team == team
                history_rows.append(dict(match_no=match.match_no, side=side, team=team,
                                         history_no=history_no, source_league=p.source_league,
                                         previous_datetime=p.datetime, venue_side="home" if was_home else "away",
                                         opponent=p.away_team if was_home else p.home_team,
                                         goals_for=p.home_score if was_home else p.away_score,
                                         goals_against=p.away_score if was_home else p.home_score,
                                         days_before_target=round((match.datetime-p.datetime).total_seconds()/86400, 2),
                                         status="ok"))
    history = pd.DataFrame(history_rows).sort_values(["match_no", "side", "history_no"])

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"toto{toto_round}_matches.csv"
    history_path = output_dir / f"toto{toto_round}_recent_history.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    return manifest_path, history_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--logical-season", type=int, required=True)
    p.add_argument("--toto-round", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/toto_round_inputs")
    args = p.parse_args()
    manifest, history = build(args.season, args.logical_season, args.toto_round, args.output_dir)
    print(f"OK: 13試合入力: {manifest.resolve()}")
    print(f"OK: 直近3試合・休養日数: {history.resolve()}")
    print("NOTE: 正式予測・疲労・buyplanには未接続です。")


if __name__ == "__main__":
    main()
