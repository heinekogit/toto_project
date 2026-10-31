#!/usr/bin/env python3
"""Build validation v2 with correctly parsed historical Football LAB chance data."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))

from build_football_lab_snapshot import (  # noqa: E402
    build_wide_metrics,
    extract_chart_rows,
    extract_team_tables,
    filter_team_metrics,
    normalize_team_key,
    parse_asof_date,
    parse_title_info,
    strip_tags,
    validate_chance_metrics,
)
from common import ROOT, sha256  # noqa: E402


JST = ZoneInfo("Asia/Tokyo")
DEFAULT_INPUT = ROOT / "data/eval/next_model/validation_v1/matches.csv"
DEFAULT_OUT = ROOT / "data/eval/next_model/validation_v2"
ROUND_SNAPSHOT = {
    "toto1644": "20260530",
    "toto1645": "20260813",
    "toto1647": "20260821",
    "toto1648": "20260827",
    "toto1649": "20260901",
    "toto1650": "20260905",
    "toto1653": "20260912",
    "toto1654": "20260919",
}
CHANCE_BASES = (
    "flab_chance_attack_count", "flab_chance_shots", "flab_chance_build_rate",
    "flab_chance_goals", "flab_chance_shot_conversion",
    "flab_chance_allowed_attack_count", "flab_chance_allowed_shots",
    "flab_chance_allowed_build_rate", "flab_chance_allowed_goals",
    "flab_chance_allowed_shot_conversion",
)


def load_chance_metrics(snapshot_date: str, league: str) -> tuple[dict, dict]:
    path = ROOT / "external_contents" / snapshot_date / league / "チャンス構築率.html"
    if not path.exists():
        raise FileNotFoundError(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    title_match = re.search(r"<title>(.*?)</title>", html, re.S)
    title = strip_tags(title_match.group(1)) if title_match else path.stem
    season, parsed_league, page_label = parse_title_info(title, league)
    meta = {
        "snapshot_date": snapshot_date,
        "asof_date": parse_asof_date(html, snapshot_date),
        "league": parsed_league,
        "season": season,
        "source_file": path.name,
        "page_label": page_label,
        "title": title,
    }
    rows = build_wide_metrics(
        extract_team_tables(html, meta), extract_chart_rows(html, meta)
    )
    rows = filter_team_metrics(rows, parsed_league, season)
    validate_chance_metrics(rows)
    by_team = {row["team_key"]: row for row in rows if "flab_chance_shots" in row}
    return by_team, {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "page_asof": meta["asof_date"],
        "teams": len(by_team),
    }


def build(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    output = source.copy()
    # V1 contains rank text in two historically misparsed shot columns.
    # Recreate every chance value/rank column as numeric before replacement.
    for base in CHANCE_BASES:
        for side in ("home", "away"):
            output[f"{base}_{side}"] = float("nan")
            output[f"{base}_rank_{side}"] = float("nan")
    cache = {}
    sources = {}
    added = set()
    for idx, row in output.iterrows():
        round_id = str(row["round_id"])
        league = str(row["league"]).lower()
        snapshot_date = ROUND_SNAPSHOT[round_id]
        cache_key = (snapshot_date, league)
        if cache_key not in cache:
            cache[cache_key], source_info = load_chance_metrics(*cache_key)
            sources[f"{snapshot_date}_{league}"] = source_info
        metrics = cache[cache_key]
        for side in ("home", "away"):
            team = row[f"{side}_team"]
            team_key = normalize_team_key(team)
            if team_key not in metrics:
                raise ValueError(f"{round_id}:{side}:{team}: Football LAB team not found")
            team_metrics = metrics[team_key]
            for base in CHANCE_BASES:
                column = f"{base}_{side}"
                output.at[idx, column] = team_metrics[base]
                added.add(column)
                rank = f"{base}_rank"
                if rank in team_metrics:
                    rank_column = f"{rank}_{side}"
                    output.at[idx, rank_column] = team_metrics[rank]
                    added.add(rank_column)
        output.at[idx, "flab_chance_snapshot_date"] = snapshot_date
        added.add("flab_chance_snapshot_date")
    numeric = [column for column in added if column != "flab_chance_snapshot_date"]
    bad = output[numeric].apply(pd.to_numeric, errors="coerce").isna()
    if bad.any(axis=None):
        raise ValueError(f"validation v2 contains invalid chance values: {int(bad.sum().sum())}")
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
        "parent": str(DEFAULT_INPUT.relative_to(ROOT)),
        "rows": len(output),
        "purpose": "validation v1 plus correctly parsed historical Football LAB chance metrics",
        "round_snapshot_mapping": ROUND_SNAPSHOT,
        "columns_added_or_replaced": sorted(added),
        "sources": sources,
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"validation output already exists: {args.out}")
    source = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    dataset, manifest = build(source)
    args.out.mkdir(parents=True)
    dataset.to_csv(args.out / "matches.csv", index=False, encoding="utf-8-sig")
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.out / "matches.csv")


if __name__ == "__main__":
    main()
