#!/usr/bin/env python3
"""Prepare a separate J1 prediction input with official toto cup fixtures."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_football_lab_snapshot import build_match_comparison
from jleague_team_names import canonical_team_name


def prepare(league_csv: Path, toto_csv: Path, team_metrics_csv: Path,
            upcoming_out: Path, compare_out: Path) -> int:
    league = pd.read_csv(league_csv, encoding="utf-8-sig")
    toto = pd.read_csv(toto_csv, encoding="utf-8-sig")
    cup = toto[toto["competition_frame_id"].eq(11)].copy()
    if cup.empty:
        return 0
    if not cup["league"].eq("j1").all():
        raise RuntimeError("ルヴァン杯のリーグ区分がJ1でありません")
    required = ["節", "match_id", "datetime", "stadium", "home_team", "away_team", "home_score", "away_score"]
    if not set(required).issubset(league):
        raise RuntimeError("J1日程CSVの列不足")
    cup["節"] = "ルヴァン準々決勝"
    cup["home_score"] = pd.NA
    cup["away_score"] = pd.NA
    cup = cup[required]
    if cup["match_id"].duplicated().any():
        raise RuntimeError("ルヴァン杯の日程に重複match_idがあります")
    combined = pd.concat([league, cup], ignore_index=True)
    key = pd.to_datetime(combined["datetime"]).dt.strftime("%Y-%m-%d") + "|" + \
        combined["home_team"].map(canonical_team_name) + "|" + \
        combined["away_team"].map(canonical_team_name)
    if key.duplicated().any() or combined["match_id"].duplicated().any():
        raise RuntimeError("J1リーグ日程とルヴァン杯で試合が重複しています")
    metrics = pd.read_csv(team_metrics_csv, encoding="utf-8-sig").to_dict("records")
    if not metrics or any(not row.get("team_key") for row in metrics):
        raise RuntimeError("Football LABチーム指標が空または不正です")
    comparison = pd.DataFrame(build_match_comparison(cup.to_dict("records"), metrics))
    snapshot_dates = {str(row.get("snapshot_date", "")).strip() for row in metrics}
    if len(snapshot_dates) != 1 or not next(iter(snapshot_dates)):
        raise RuntimeError("Football LABチーム指標の取得日が一意でありません")
    comparison["flab_snapshot_date"] = next(iter(snapshot_dates))
    needed = [f"flab_chance_shots_{side}" for side in ("home", "away")]
    if comparison.empty or comparison[needed].isna().any().any():
        raise RuntimeError("ルヴァン杯のFootball LABチーム指標が不足しています")
    upcoming_out.parent.mkdir(parents=True, exist_ok=True)
    compare_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(upcoming_out, index=False, encoding="utf-8-sig")
    comparison.to_csv(compare_out, index=False, encoding="utf-8-sig")
    return len(cup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-upcoming", type=Path, required=True)
    parser.add_argument("--toto-fixtures", type=Path, required=True)
    parser.add_argument("--team-metrics", type=Path, required=True)
    parser.add_argument("--out-upcoming", type=Path, required=True)
    parser.add_argument("--out-compare", type=Path, required=True)
    args = parser.parse_args()
    count = prepare(args.league_upcoming, args.toto_fixtures, args.team_metrics,
                    args.out_upcoming, args.out_compare)
    print(f"[TOTO_CUP_PREDICTION] added={count} upcoming={args.out_upcoming} compare={args.out_compare}")


if __name__ == "__main__":
    main()
