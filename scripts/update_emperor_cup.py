"""Fetch JFA's official Emperor's Cup results as confirmed fatigue events.

Future/undecided fixtures remain in the raw snapshot, never counted as played.
Match duration is not individual player minutes or a measured workload.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from http_retry import get_with_retry
from jleague_team_names import canonical_team_name

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["datetime", "team", "competition", "source", "status", "match_id",
           "opponent", "stadium", "round", "home_score", "away_score",
           "extra_time_minutes", "match_duration_minutes"]


def parse_results(payload, season):
    container = payload.get("matchScheduleList", {})
    matches = container.get("matchSchedule")
    if "天皇杯" not in container.get("competitionName", "") or not isinstance(matches, list) or not matches:
        raise ValueError("JFA Emperor's Cup schedule schema missing or empty")
    events, ids = [], set()
    for row in matches:
        number = str(row["matchNumber"])
        if number in ids:
            raise ValueError(f"Duplicate JFA match number: {number}")
        ids.add(number)
        if row.get("matchStatus") != "試合終了":
            continue
        when = pd.to_datetime(f"{row.get('matchDateJpn', '')} {row.get('matchTimeJpn', '')}", errors="coerce")
        score = row.get("score", {})
        if pd.isna(when) or when.year not in {int(season), int(season) + 1}:
            raise ValueError(f"Invalid completed match timestamp: {number}")
        scores = [pd.to_numeric(score.get(k), errors="coerce") for k in ("homeScore", "awayScore")]
        if any(pd.isna(v) or v < 0 or v != int(v) for v in scores):
            raise ValueError(f"Invalid completed match score: {number}")
        teams = [canonical_team_name(row.get(k)) for k in ("homeTeamName", "awayTeamName")]
        if any(not t or "勝者" in t or "未定" in t for t in teams) or teams[0] == teams[1]:
            raise ValueError(f"Invalid completed match teams: {number}")
        extra = 30 if score.get("exMatch") is True else 0 if score.get("exMatch") is False else None
        for index, team in enumerate(teams):
            events.append(dict(zip(COLUMNS, [
                when, team, "emperor_cup",
                f"https://www.jfa.jp/match/emperorscup_{season}/match_page/m{number}.html",
                "confirmed", f"emperor_cup_{season}_{number}", teams[1-index],
                row.get("venueFullName", ""), row.get("matchTypeName", ""), *scores,
                extra, 90 + extra if extra is not None else None,
            ])))
    frame = pd.DataFrame(events, columns=COLUMNS)
    if frame.duplicated(["team", "datetime"]).any():
        raise ValueError("Duplicate Emperor's Cup team appearance")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=os.environ.get("SEASON_YEAR", "2026"), type=int)
    parser.add_argument("--input-json", type=Path, help="Use a downloaded official snapshot (offline validation)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    url = f"https://www.jfa.jp/match/emperorscup_{args.season}/match/schedule.json"
    if args.input_json:
        payload = json.loads(args.input_json.read_text(encoding="utf-8-sig"))
    else:
        response = get_with_retry(url, timeout=(5, 30), max_retries=3)
        response.raise_for_status()
        payload = response.json()
    frame = parse_results(payload, args.season)
    output = args.data_dir / f"emperor_cup_events_{args.season}.csv"
    # A parse/network failure never overwrites the last valid CSV.
    if output.exists():
        previous = pd.read_csv(output)
        if not set(previous.match_id).issubset(set(frame.match_id)):
            raise ValueError("Previously completed cup matches disappeared; preserving existing CSV")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.data_dir / "raw" / f"emperor_cup_{args.season}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary = output.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output)
    print(f"[EMPEROR_CUP] confirmed_matches={frame.match_id.nunique()} team_events={len(frame)} output={output}")


if __name__ == "__main__":
    main()
