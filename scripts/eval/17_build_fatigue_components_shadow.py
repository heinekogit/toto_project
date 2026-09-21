#!/usr/bin/env python3
"""Build priority 1-3 fatigue shadows without changing production inputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from fatigue_components_shadow import build_nonrecursive_components


def load_fatigue_module():
    path = SCRIPTS / "06_calculate_fatigue.py"
    spec = importlib.util.spec_from_file_location("production_fatigue_readonly", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(league, season, module):
    frames = []
    for kind, priority in (("upcoming", 0), ("latest_results", 1)):
        path = ROOT / "data" / f"{league}_{season}_{kind}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["__source_priority"] = priority
        frames.append(frame)
    matches = module.dedupe_matches(pd.concat(frames, ignore_index=True)).drop(
        columns="__source_priority", errors="ignore"
    )
    matrix = module.load_travel_distances()
    acl = module.load_acl_team_events(str(ROOT / "data/manual/acl_schedule.csv"))
    cup = module.load_confirmed_cup_events(str(ROOT / "data"))
    manual = module.load_external_team_events(str(ROOT / "data/manual/external_match_events.csv"))
    extra = [frame for frame in (acl, cup, manual) if frame is not None and not frame.empty]
    external = pd.concat(extra, ignore_index=True) if extra else pd.DataFrame()
    if not external.empty:
        external["team"] = external.team.map(module.canonical_team_name)
        external["datetime"] = pd.to_datetime(external.datetime, errors="coerce")
        external = external.sort_values("datetime").drop_duplicates(["team", "datetime"], keep="last")
    return matches, matrix, external


def build_league(league, season, outdir, module):
    matches, matrix, external = load_inputs(league, season, module)
    baseline = module.calculate_fatigue(matches, matrix, external)
    original_away = module.AWAY_GAME_PENALTY
    try:
        module.AWAY_GAME_PENALTY = 0.0
        no_fixed = module.calculate_fatigue(matches, matrix, external)
    finally:
        module.AWAY_GAME_PENALTY = original_away
    components = build_nonrecursive_components(matches, matrix, external, module)

    keys = ["match_id", "datetime", "home_team", "away_team"]
    base_cols = keys + ["home_fatigue_score", "away_fatigue_score"]
    out = baseline[base_cols].merge(no_fixed[base_cols], on=keys, suffixes=("_baseline", "_no_fixed_away"))
    out = out.merge(components, on=keys, validate="one_to_one")
    for side in ("home", "away"):
        out[f"{side}_fixed_away_removal_delta"] = (
            out[f"{side}_fatigue_score_no_fixed_away"] - out[f"{side}_fatigue_score_baseline"]
        )
        out[f"{side}_recursion_removal_delta"] = (
            out[f"{side}_component_total_legacy_scale"] - out[f"{side}_fatigue_score_no_fixed_away"]
        )
    if not out.away_travel_lookup_status.eq("hit").all():
        raise ValueError(f"{league}: incomplete travel lookup")

    league_dir = outdir / league
    league_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(league_dir / "fatigue_components_shadow.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    summary = {
        "league": league.upper(), "season": season, "matches": len(out),
        "baseline_home_mean": out.home_fatigue_score_baseline.mean(),
        "baseline_away_mean": out.away_fatigue_score_baseline.mean(),
        "no_fixed_home_mean": out.home_fatigue_score_no_fixed_away.mean(),
        "no_fixed_away_mean": out.away_fatigue_score_no_fixed_away.mean(),
        "nonrecursive_home_mean": out.home_component_total_legacy_scale.mean(),
        "nonrecursive_away_mean": out.away_component_total_legacy_scale.mean(),
        "travel_lookup_hits": int(out.away_travel_lookup_status.eq("hit").sum()),
        "external_unknown_rows": int((out.home_external_load_unknown_count.gt(0) | out.away_external_load_unknown_count.gt(0)).sum()),
    }
    pd.DataFrame([summary]).to_csv(league_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return out, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--leagues", nargs="+", default=["j1", "j2"], choices=["j1", "j2"])
    parser.add_argument("--out", type=Path, default=ROOT / "data/eval/fatigue_components_shadow/2026")
    args = parser.parse_args()
    module = load_fatigue_module()
    summaries = []
    for league in args.leagues:
        _, summary = build_league(league, args.season, args.out, module)
        summaries.append(summary)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out / "summary.csv", index=False, encoding="utf-8-sig")
    table = summary.to_html(index=False, border=1, float_format=lambda value: f"{value:.3f}")
    (args.out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>疲労原因別シャドー</title>"
        "<h1>疲労原因別シャドー：優先順位1〜3</h1>"
        "<p><b>標準版・予測確率・BuyPlanは変更していません。</b> "
        "baseline、固定アウェー4点除去、さらに再帰持越しを原因別有限窓へ変更した値を比較します。"
        "fatigue_gap/totalとuncertaintyは未実装です。</p>" + table
        + "<p><a href='j1/fatigue_components_shadow.csv'>J1詳細CSV</a> / "
          "<a href='j2/fatigue_components_shadow.csv'>J2詳細CSV</a></p>",
        encoding="utf-8",
    )
    (args.out / "manifest.json").write_text(json.dumps({
        "shadow_only": True, "production_changed": False, "season": args.season,
        "leagues_evaluated_separately": args.leagues,
        "implemented": ["remove_fixed_away_4", "remove_recursive_carry", "cause_preserving_components"],
        "not_implemented": ["fatigue_gap", "fatigue_total", "uncertainty", "prediction_adjustment", "buyplan_adjustment"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out / "index.html")


if __name__ == "__main__":
    main()

