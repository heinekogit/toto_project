#!/usr/bin/env python3
"""Freeze pre-match cup rotation and post-cup fatigue hypotheses.

This is deliberately separate from production predictions, fatigue CSVs, and buyplan.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JST = ZoneInfo("Asia/Tokyo")
CATEGORY_RANK = {"j1": 1, "j2": 2, "j3": 3}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def expected_core_usage(team_category: str, opponent_category: str) -> float:
    """Return the frozen 1.0/0.7/0.4 category-gap hypothesis."""
    team_rank = CATEGORY_RANK[team_category]
    opponent_rank = CATEGORY_RANK[opponent_category]
    lower_division_gap = opponent_rank - team_rank
    if lower_division_gap >= 2:
        return 0.4
    if lower_division_gap == 1:
        return 0.7
    return 1.0


def _risk_label(usage: float) -> str:
    if usage <= 0.4:
        return "high_rotation"
    if usage <= 0.7:
        return "medium_rotation"
    return "standard_selection"


def _team_categories(history: pd.DataFrame) -> dict[str, str]:
    ok = history[(history.status == "ok") & history.source_league.isin(CATEGORY_RANK)].copy()
    counts = ok.groupby(["team", "source_league"]).size().reset_index(name="count")
    counts = counts.sort_values(["team", "count"], ascending=[True, False]).drop_duplicates("team")
    return dict(zip(counts.team, counts.source_league))


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(toto_round: int, output_root: Path) -> tuple[Path, Path, Path]:
    manifest_path = ROOT / f"data/toto_round_inputs/toto{toto_round}_matches.csv"
    history_path = ROOT / f"data/toto_round_inputs/toto{toto_round}_recent_history.csv"
    travel_path = ROOT / f"data/eval/travel_fatigue_shadow/toto{toto_round}_travel_fatigue_shadow.csv"
    matrix_path = ROOT / "data/manual/team_travel_distances_normalized_all.csv"
    for path in (manifest_path, history_path, travel_path, matrix_path):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = _read(manifest_path)
    cup = manifest[manifest.competition.eq("league_cup")].copy()
    if cup.empty:
        raise ValueError(f"toto{toto_round}: league_cup rows not found")
    history = _read(history_path)
    categories = _team_categories(history)
    travel_shadow = _read(travel_path).set_index("match_no")
    matrix = _read(matrix_path, sep="\t", index_col=0)
    snapshot_at = datetime.now(JST).isoformat(timespec="seconds")

    rows = []
    for match in cup.sort_values("match_no").itertuples():
        home_category = categories.get(match.home_team)
        away_category = categories.get(match.away_team)
        if home_category not in CATEGORY_RANK or away_category not in CATEGORY_RANK:
            raise ValueError(f"category missing: {match.home_team}={home_category}, {match.away_team}={away_category}")
        fatigue_row = travel_shadow.loc[int(match.match_no)]
        distance = float(matrix.loc[match.away_team, match.home_team])
        for side, team, opponent, category, opponent_category in (
            ("home", match.home_team, match.away_team, home_category, away_category),
            ("away", match.away_team, match.home_team, away_category, home_category),
        ):
            usage = expected_core_usage(category, opponent_category)
            pre_match_fatigue = float(fatigue_row[f"{side}_fatigue_score_current"])
            travel_km = distance if side == "away" else 0.0
            travel_fatigue = travel_km * 0.005
            scaled_match_load = pre_match_fatigue * usage
            rows.append({
                "snapshot_at_jst": snapshot_at,
                "toto_round": toto_round,
                "match_no": int(match.match_no),
                "datetime": match.datetime,
                "side": side,
                "team": team,
                "opponent": opponent,
                "team_category": category,
                "opponent_category": opponent_category,
                "category_gap": CATEGORY_RANK[opponent_category] - CATEGORY_RANK[category],
                "assumed_core_usage": usage,
                "rotation_risk": _risk_label(usage),
                "matchday_strength_index": usage,
                "matchday_xg_adjustment": pd.NA,
                "pre_match_fatigue_current": pre_match_fatigue,
                "scaled_match_load": scaled_match_load,
                "travel_km": travel_km,
                "travel_fatigue_unscaled": travel_fatigue,
                "assumed_post_cup_event_load": scaled_match_load + travel_fatigue,
                "extra_time_load": 0.0,
                "hypothesis_status": "pre_match_frozen",
            })
    assumptions = pd.DataFrame(rows)

    output_dir = output_root / f"toto{toto_round}"
    output_dir.mkdir(parents=True, exist_ok=True)
    assumptions_path = output_dir / "cup_rotation_assumptions.csv"
    scoring_path = output_dir / "cup_rotation_actuals_template.csv"
    html_path = output_dir / "cup_rotation_shadow.html"
    assumptions.to_csv(assumptions_path, index=False, encoding="utf-8-sig", float_format="%.4f")

    actuals = assumptions[["toto_round", "match_no", "side", "team", "opponent", "assumed_core_usage"]].copy()
    actuals["actual_core_starters"] = pd.NA
    actuals["actual_core_minutes"] = pd.NA
    actuals["actual_total_available_core_minutes"] = pd.NA
    actuals["actual_core_usage"] = pd.NA
    actuals["went_to_extra_time"] = pd.NA
    actuals["actual_extra_minutes"] = pd.NA
    actuals["actual_result"] = pd.NA
    actuals["notes"] = pd.NA
    actuals.to_csv(scoring_path, index=False, encoding="utf-8-sig")

    view = assumptions.copy()
    view["対戦"] = view.apply(lambda r: f"{r.team}（{r.team_category.upper()}）→ {r.opponent}（{r.opponent_category.upper()}）", axis=1)
    view["想定主力投入"] = view.assumed_core_usage.map(lambda x: f"{x:.1f}")
    view["ローテーション"] = view.rotation_risk.map({
        "standard_selection": "標準想定", "medium_rotation": "中程度", "high_rotation": "大きい"
    })
    view["直前疲労"] = view.pre_match_fatigue_current.map(lambda x: f"{x:.2f}")
    view["移動負荷"] = view.travel_fatigue_unscaled.map(lambda x: f"{x:.2f}")
    view["持越し負荷（仮）"] = view.assumed_post_cup_event_load.map(lambda x: f"{x:.2f}")
    table = view[["match_no", "side", "対戦", "想定主力投入", "ローテーション", "直前疲労", "travel_km", "移動負荷", "持越し負荷（仮）"]].rename(
        columns={"match_no": "No.", "side": "側", "travel_km": "移動km"}
    ).to_html(index=False, escape=True)
    checksum = _checksum(assumptions_path)
    html = f"""<!doctype html><html lang=\"ja\"><meta charset=\"utf-8\"><title>toto{toto_round} カップ戦ローテーション仮説</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Hiragino Sans',sans-serif;margin:28px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d7deea;padding:7px;text-align:right}}th:nth-child(3),td:nth-child(3){{text-align:left}}th{{background:#eaf0f8}}.note{{background:#f2f6fb;padding:14px;border-radius:8px;margin:14px 0}}code{{background:#eef1f5;padding:2px 4px}}</style>
<h1>toto{toto_round} カップ戦ローテーション仮説</h1>
<p>試合前固定日時: {escape(snapshot_at)}</p>
<div class=\"note\"><b>正式版には未接続。</b> カテゴリ差により、同格・格上=1.0、1カテゴリ下=0.7、2カテゴリ下=0.4と仮定。<br>
当日戦力は投入指数のみ固定し、未検証のxG換算は行いません。次節への暫定持越しは <code>直前疲労×投入係数＋移動km×0.005</code>。移動負荷には投入係数を掛けません。延長負荷は試合後に別加算します。</div>
{table}
<p>assumptions SHA-256: <code>{checksum}</code></p>
<p>実績記入用: {escape(str(scoring_path.resolve()))}</p></html>"""
    html_path.write_text(html, encoding="utf-8")
    return assumptions_path, scoring_path, html_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", required=True, help="例: toto1649 または 1649")
    p.add_argument("--output-root", type=Path, default=ROOT / "data/eval/cup_rotation_shadow")
    args = p.parse_args()
    toto_round = int(str(args.round).lower().removeprefix("toto"))
    assumptions, scoring, html = build(toto_round, args.output_root)
    print(f"OK: 仮説固定CSV: {assumptions.resolve()}")
    print(f"OK: 試合後採点テンプレート: {scoring.resolve()}")
    print(f"OK: 参照HTML: file://{html.resolve()}")
    print("NOTE: 正式予測・疲労CSV・buyplanは変更していません。")


if __name__ == "__main__":
    main()
