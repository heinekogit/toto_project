#!/usr/bin/env python3
"""Compare current fatigue with an opt-in normalized travel-distance shadow."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from jleague_team_names import canonical_team_name


def _load_fatigue_module():
    path = SCRIPTS / "06_calculate_fatigue.py"
    spec = importlib.util.spec_from_file_location("fatigue_calculator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def _all_matches(season: int, manifest: pd.DataFrame, fatigue_module) -> pd.DataFrame:
    frames = []
    for league in ("j1", "j2", "j3"):
        for kind, priority in (("upcoming", 0), ("latest_results", 1)):
            path = ROOT / f"data/{league}_{season}_{kind}.csv"
            if not path.exists():
                continue
            frame = _read(path)
            frame["__source_priority"] = priority
            frames.append(frame)
    matches = pd.concat(frames, ignore_index=True)
    matches["home_team"] = matches.home_team.map(canonical_team_name)
    matches["away_team"] = matches.away_team.map(canonical_team_name)
    matches = fatigue_module.dedupe_matches(matches)

    cup = manifest[manifest.competition != "league"].copy()
    cup["match_id"] = cup.apply(
        lambda r: f"shadow_toto{int(r.toto_round)}_{int(r.match_no):02d}_{r.home_team}_{r.away_team}", axis=1
    )
    cup["節"] = "カップ戦"
    cup["home_score"] = pd.NA
    cup["away_score"] = pd.NA
    keep = ["節", "match_id", "datetime", "stadium", "home_team", "away_team", "home_score", "away_score"]
    matches = pd.concat([matches[keep], cup[keep]], ignore_index=True)
    matches["datetime"] = pd.to_datetime(matches.datetime, errors="coerce")
    return matches.dropna(subset=["datetime"]).sort_values("datetime", kind="mergesort")


def _external_events(fatigue_module) -> pd.DataFrame:
    events = fatigue_module.load_acl_team_events(str(ROOT / "data/manual/acl_schedule.csv"))
    if not events.empty:
        events["team"] = events.team.map(canonical_team_name)
    return events


def _impact_level(home_xg_delta: float, away_xg_delta: float, swing: float) -> str:
    peak = max(abs(home_xg_delta), abs(away_xg_delta), abs(swing))
    if peak >= 0.08:
        return "high"
    if peak >= 0.04:
        return "medium"
    return "low"


def build(toto_round: int, output_dir: Path) -> tuple[Path, Path]:
    manifest_path = ROOT / f"data/toto_round_inputs/toto{toto_round}_matches.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run prepare_toto_round_inputs.py first: {manifest_path}")
    manifest = _read(manifest_path)
    manifest["datetime"] = pd.to_datetime(manifest.datetime, errors="coerce")
    season = int(manifest.season.iloc[0])

    fatigue = _load_fatigue_module()
    matches = _all_matches(season, manifest, fatigue)
    events = _external_events(fatigue)
    matrix_path = ROOT / "data/manual/team_travel_distances_normalized_all.csv"
    matrix = _read(matrix_path, sep="\t", index_col=0)
    matrix.index = matrix.index.map(canonical_team_name)
    matrix.columns = [canonical_team_name(c) for c in matrix.columns]

    current = fatigue.calculate_fatigue(matches, pd.DataFrame(), events)
    shadow = fatigue.calculate_fatigue(matches, matrix, events)
    keys = ["match_id", "datetime", "home_team", "away_team"]
    compared = current.merge(shadow, on=keys, suffixes=("_current", "_travel_shadow"))

    target_ids = {}
    for row in manifest.itertuples():
        if row.competition == "league":
            candidates = matches[(matches.datetime == row.datetime) & (matches.home_team == row.home_team) &
                                 (matches.away_team == row.away_team)]
            if len(candidates) != 1:
                raise ValueError(f"cannot uniquely identify match {row.match_no}: {row.home_team}-{row.away_team}")
            target_ids[candidates.iloc[0].match_id] = int(row.match_no)
        else:
            target_ids[f"shadow_toto{toto_round}_{int(row.match_no):02d}_{row.home_team}_{row.away_team}"] = int(row.match_no)
    out = compared[compared.match_id.isin(target_ids)].copy()
    out["match_no"] = out.match_id.map(target_ids)
    out = out.sort_values("match_no")

    out["away_travel_km"] = [float(matrix.loc[a, h]) for a, h in zip(out.away_team, out.home_team)]
    out["direct_away_travel_fatigue"] = out.away_travel_km * float(fatigue.TRAVEL_DISTANCE_WEIGHT)
    for side in ("home", "away"):
        out[f"{side}_fatigue_delta"] = out[f"{side}_fatigue_score_travel_shadow"] - out[f"{side}_fatigue_score_current"]
        out[f"{side}_xg_delta"] = -out[f"{side}_fatigue_delta"] * 0.01
    out["home_advantage_xg_swing"] = out.away_fatigue_delta.mul(0.01) - out.home_fatigue_delta.mul(0.01)
    out["inherited_away_fatigue_delta"] = out.away_fatigue_delta - out.direct_away_travel_fatigue
    out["impact"] = [_impact_level(h, a, s) for h, a, s in
                     zip(out.home_xg_delta, out.away_xg_delta, out.home_advantage_xg_swing)]
    out["interpretation"] = out.apply(
        lambda r: ("ホーム側に相対追い風" if r.home_advantage_xg_swing > 0.015 else
                   "アウェイ側に相対追い風" if r.home_advantage_xg_swing < -0.015 else "相対差は小さい"), axis=1
    )

    columns = ["match_no", "datetime", "home_team", "away_team", "away_travel_km",
               "home_fatigue_score_current", "away_fatigue_score_current",
               "home_fatigue_score_travel_shadow", "away_fatigue_score_travel_shadow",
               "home_rest_fatigue_current", "away_rest_fatigue_current",
               "home_recent_load_carry_current", "away_recent_load_carry_current",
               "home_away_condition_penalty_current", "away_away_condition_penalty_current",
               "away_travel_fatigue_travel_shadow", "travel_lookup_status_travel_shadow",
               "home_fatigue_delta", "away_fatigue_delta", "direct_away_travel_fatigue",
               "inherited_away_fatigue_delta", "home_xg_delta", "away_xg_delta",
               "home_advantage_xg_swing", "impact", "interpretation"]
    out = out[columns]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"toto{toto_round}_travel_fatigue_shadow.csv"
    html_path = output_dir / f"toto{toto_round}_travel_fatigue_shadow.html"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    display = out.copy()
    display["対戦"] = display.home_team + "－" + display.away_team
    display["移動km"] = display.away_travel_km.map(lambda x: f"{x:,.0f}")
    display["現行疲労 H/A"] = display.apply(lambda r: f"{r.home_fatigue_score_current:.2f} / {r.away_fatigue_score_current:.2f}", axis=1)
    display["距離あり H/A"] = display.apply(lambda r: f"{r.home_fatigue_score_travel_shadow:.2f} / {r.away_fatigue_score_travel_shadow:.2f}", axis=1)
    display["疲労増 H/A"] = display.apply(lambda r: f"+{r.home_fatigue_delta:.2f} / +{r.away_fatigue_delta:.2f}", axis=1)
    display["現行内訳 H"] = display.apply(
        lambda r: f"間隔 {r.home_rest_fatigue_current:.2f} / 持越 {r.home_recent_load_carry_current:.2f}", axis=1
    )
    display["現行内訳 A"] = display.apply(
        lambda r: (f"間隔 {r.away_rest_fatigue_current:.2f} / 持越 {r.away_recent_load_carry_current:.2f} / "
                   f"AW {r.away_away_condition_penalty_current:.2f}"), axis=1
    )
    display["xG変化 H/A"] = display.apply(lambda r: f"{r.home_xg_delta:+.3f} / {r.away_xg_delta:+.3f}", axis=1)
    display["ホーム相対xG"] = display.home_advantage_xg_swing.map(lambda x: f"{x:+.3f}")
    table = display[["match_no", "対戦", "移動km", "現行疲労 H/A", "現行内訳 H", "現行内訳 A",
                     "距離あり H/A", "疲労増 H/A",
                     "xG変化 H/A", "ホーム相対xG", "impact", "interpretation"]].rename(
                         columns={"match_no": "No.", "impact": "影響度", "interpretation": "見方"}).to_html(
                             index=False, escape=True, classes="shadow")
    high = int((out.impact == "high").sum())
    medium = int((out.impact == "medium").sum())
    html = f"""<!doctype html><html lang=\"ja\"><meta charset=\"utf-8\"><title>toto{toto_round} 移動疲労シャドー</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Hiragino Sans',sans-serif;margin:28px;color:#172033}}h1{{font-size:24px}}.note{{background:#f2f6fb;padding:14px;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d7deea;padding:7px;text-align:right}}th:nth-child(2),td:nth-child(2),th:last-child,td:last-child{{text-align:left}}th{{background:#eaf0f8;position:sticky;top:0}}.summary{{font-size:18px}}code{{background:#eef1f5;padding:2px 4px}}</style>
<h1>toto{toto_round} 移動距離・疲労シャドー比較</h1>
<p class=\"summary\">影響度 high: {high}試合 / medium: {medium}試合 / 全13試合</p>
<div class=\"note\"><b>正式予測には未反映。</b> 現行の疲労計算と、正規化44クラブ距離行列を接続した場合を比較しています。xG変化は現行係数 <code>疲労1点 = xG -0.01</code> の一次影響です。確率・本命記号・buyplanは再計算していません。ホーム疲労増は過去のアウェイ移動が直近3試合から持ち越された分です。</div>
{table}
<p>生成元: {escape(str(csv_path.resolve()))}</p></html>"""
    html_path.write_text(html, encoding="utf-8")
    return csv_path, html_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", required=True, help="例: toto1649 または 1649")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/eval/travel_fatigue_shadow")
    args = p.parse_args()
    toto_round = int(str(args.round).lower().removeprefix("toto"))
    csv_path, html_path = build(toto_round, args.output_dir)
    print(f"OK: 比較CSV: {csv_path.resolve()}")
    print(f"OK: 参照HTML: file://{html_path.resolve()}")
    print("NOTE: 正式予測・buyplanは変更していません。")


if __name__ == "__main__":
    main()
