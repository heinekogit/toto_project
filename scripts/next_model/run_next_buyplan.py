#!/usr/bin/env python3
"""Production CLI for the next prediction and BuyPlan pipeline."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from build_next_buyplan import build_round, build_shadow_round
from common import ROOT, match_key, sha256
from preflight import check_inputs
from forecast_pipeline import build_next_probabilities
from shadow_free_coverage import build_shadow as build_free_coverage_shadow
from random_remaining_v24 import prediction_fingerprint, seed_from_fingerprint
JST = ZoneInfo("Asia/Tokyo")


def load_order(path: Path, round_id: str, logical_season: int) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    number = int(round_id.lower().removeprefix("toto"))
    selected = table[
        pd.to_numeric(table["toto_round"], errors="coerce").eq(number)
        & pd.to_numeric(table["season"], errors="coerce").eq(logical_season)
    ].sort_values("match_no")
    if len(selected) != 13 or set(selected["match_no"].astype(int)) != set(range(1, 14)):
        raise ValueError(f"{round_id}: toto order must contain exactly match_no 1..13")
    return selected


def select_round(predictions: pd.DataFrame, order: pd.DataFrame, round_id: str) -> pd.DataFrame:
    work = predictions.copy()
    work["__key"] = [match_key(h, a) for h, a in zip(work["home_team"], work["away_team"])]
    if work["__key"].duplicated().any():
        raise ValueError("prediction input contains duplicate home/away pairs")
    by_key = work.set_index("__key", drop=False)
    rows = []
    for match in order.itertuples(index=False):
        key = match_key(match.home_team, match.away_team)
        if key not in by_key.index:
            raise ValueError(
                f"{round_id}: prediction input belongs to another/incomplete round; "
                f"missing match_no={match.match_no} {match.home_team}-{match.away_team}"
            )
        row = by_key.loc[key].copy()
        row["match_no"] = int(match.match_no)
        row["home_team"] = match.home_team
        row["away_team"] = match.away_team
        rows.append(row)
    selected = pd.DataFrame(rows).sort_values("match_no").reset_index(drop=True)
    if len(selected) != 13:
        raise ValueError(f"{round_id}: selected prediction rows != 13")
    return selected.drop(columns=["__key"], errors="ignore")


def write_html(plan: pd.DataFrame, round_id: str, path: Path, placement_audit: dict | None = None,
               generator_audit: dict | None = None) -> None:
    columns = [f"ticket{i:02d}" for i in range(1, 11)]
    labels = {1: "H", 0: "D", 2: "A"}
    view = plan[["match_no", "home_team", "away_team", "draw_filter_class"] + columns].copy()
    view.insert(1, "match", view["home_team"].astype(str) + "－" + view["away_team"].astype(str))
    view = view.drop(columns=["home_team", "away_team"])
    for column in columns:
        view[column] = view[column].map(labels)
    view = view.rename(columns={"match_no": "No.", "match": "対戦", "draw_filter_class": "D分類"})
    probability_columns = {1: "next_prob_home", 0: "next_prob_draw", 2: "next_prob_away"}
    joint_probabilities = [
        math.prod(float(row[probability_columns[int(row[column])]]) for _, row in plan.iterrows())
        for column in columns
    ]
    probability_view = pd.DataFrame({
        "候補": [f"候補{i:02d}" for i in range(1, 11)],
        "13的中のモデル確率": [f"{value * 100:.8f}%" for value in joint_probabilities],
        "D本数": [int((plan[column] == 0).sum()) for column in columns],
    })
    score_html = ""
    if placement_audit is not None:
        scores = placement_audit["placement_scores_HDA"]
        if len(scores) != len(plan):
            raise ValueError("Placement scores do not match the 13 BuyPlan matches")
        score_view = pd.DataFrame({
            "No.": plan["match_no"].astype(int),
            "対戦": plan["home_team"].astype(str) + "－" + plan["away_team"].astype(str),
            "H配券スコア": [f"{row[0]:.3f}" for row in scores],
            "D配券スコア": [f"{row[1]:.3f}" for row in scores],
            "A配券スコア": [f"{row[2]:.3f}" for row in scores],
        })
        score_html = '<h2>試合別の配券スコア</h2><p>券への配置だけに使う評価値です。予測確率やH/D/Aの配置枚数は変更していません。</p>' + score_view.to_html(index=False, escape=True)
    v24 = generator_audit is not None and generator_audit.get("generator") == "random_remaining_v24"
    title = "v2.4" if v24 else "v2.3"
    if v24:
        description = ("各試合の予測確率を10枚へ最大剰余法で配分し、残数に比例する抽選で候補01～10を生成します。"
                       "候補番号は抽選順です。13的中のモデル確率は元の予測確率から計算しています。")
        pairs = generator_audit["duplicate_pairs"]
        duplicate_note = (f"<p class='note'>警告：完全一致する購入券：{escape(str(pairs))}。重複券は補充していません。</p>"
                          if pairs else "<p class='note'>完全一致する購入券はありません。</p>")
    else:
        description = ("次期得点モデルのH/D/A予測確率で10口の配置枚数を決め、Dと最有力H/Aの相対関係から作った配券スコアで段階型に配置します。"
                       "配置は乱数を使わず決定的に探索します。10的中以上を目標とする旧最適化は使いません。各試合の枚数と予測確率は変えません。"
                       "候補01～10は配券スコアによる組み合わせの強い順です。下表の13的中モデル確率も次期得点モデルで計算しています。")
        duplicate_note = ""
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>{escape(round_id)} 次期BuyPlan</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#172033}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd5df;padding:7px;text-align:center}}th{{background:#edf2f7}}
td:nth-child(2){{text-align:left;white-space:nowrap}}.note{{background:#eef7ff;border-left:5px solid #2878b5;padding:12px}}
</style><h1>{escape(round_id)} BuyPlan {title}（正式版）</h1>
<p class="note">{escape(description)}</p>{duplicate_note}
<h2>候補ごとのモデル確率</h2>{probability_view.to_html(index=False, escape=True)}
{score_html}<h2>配券</h2>{view.to_html(index=False, escape=True)}</html>"""
    path.write_text(html, encoding="utf-8")


def write_shadow_html(plan: pd.DataFrame, round_id: str, path: Path, label: str, version: str, first_ticket: int) -> None:
    columns = [f"ticket{i:02d}" for i in range(first_ticket, first_ticket + 10)]
    labels = {1: "H", 0: "D", 2: "A"}
    optional = ["draw_filter_class"] if "draw_filter_class" in plan else []
    view = plan[["match_no", "home_team", "away_team"] + optional + columns].copy()
    view.insert(1, "match", view["home_team"].astype(str) + "－" + view["away_team"].astype(str))
    view = view.drop(columns=["home_team", "away_team"])
    for column in columns:
        view[column] = view[column].map(labels)
    view = view.rename(columns={"match_no": "No.", "match": "対戦", "draw_filter_class": "D分類"})
    probability_note = "正式版と同じ次期得点モデルの予測確率を使用しています。"
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>{escape(round_id)} BuyPlanシャドー検証</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#172033}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd5df;padding:7px;text-align:center}}th{{background:#fff1cf}}
td:nth-child(2){{text-align:left;white-space:nowrap}}.warn{{background:#fff6df;border-left:5px solid #d99000;padding:12px}}
</style><h1>{escape(round_id)} BuyPlan シャドー{escape(label)}：{escape(version)}（10口）</h1>
<p class="warn"><b>購入対象外です。</b> {escape(probability_note)}シャドー{escape(label)}の候補{first_ticket:02d}～{first_ticket+9:02d}は比較用の番号で、正式版や他のシャドーとの重複を許します。</p>
{view.to_html(index=False, escape=True)}</html>"""
    path.write_text(html, encoding="utf-8")


def build_comparison_shadow(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Shadow A: retain the previous official v2.2 portfolio."""
    plan, audit = build_round(frame, strategy="probability_allocation_ge10_v1")
    plan = plan.rename(columns={f"ticket{i:02d}":f"ticket{i+10:02d}" for i in range(1,11)})
    audit.update(status="non-purchase shadow A", version="v2.2",
                 shadow_ticket_count=10, official_ticket_count=10,
                 official_tickets_unchanged=True, cross_version_duplicates_allowed=True,
                 comparison_mode="same_input_equal_ticket_count",
                 ticket_number_mapping="v2.2 ticket01..10 displayed as ticket11..20")
    return plan, audit


def build_anchor_shadow(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Shadow B: the Anchor graded arrangement without D-score adjustment."""
    plan, audit = build_round(frame, strategy="anchor_graded_shadow")
    plan = plan.rename(columns={f"ticket{i:02d}":f"ticket{i+20:02d}" for i in range(1,11)})
    audit.update(status="non-purchase shadow B", version="Anchor段階型",
                 shadow_ticket_count=10, official_ticket_count=10,
                 official_tickets_unchanged=True, cross_version_duplicates_allowed=True,
                 comparison_mode="same_allocations_without_D_score_adjustment",
                 ticket_number_mapping="Anchor ticket01..10 displayed as ticket21..30")
    return plan, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_csv", type=Path, required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--logical-season", type=int, required=True)
    parser.add_argument("--toto-order-csv", type=Path, default=ROOT / "data/manual/toto節リスト.csv")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", help="replace only next BuyPlan generated files in an existing directory")
    parser.add_argument("--check-only", action="store_true", help="validate inputs without writing outputs")
    parser.add_argument("--max-age-days", type=int, default=7, help="maximum stats/LAB snapshot age in days")
    args = parser.parse_args()
    round_id = "toto" + str(args.round).lower().removeprefix("toto")
    if args.outdir.exists() and not args.replace and not args.check_only:
        raise FileExistsError(f"output already exists: {args.outdir}")
    order = load_order(args.toto_order_csv, round_id, args.logical_season)
    predictions = pd.read_csv(args.input_csv, encoding="utf-8-sig", low_memory=False)
    selected = select_round(predictions, order, round_id)
    preflight = check_inputs(selected, order, round_id, max_age_days=args.max_age_days)
    if args.check_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    next_predictions = build_next_probabilities(selected)
    fingerprint = prediction_fingerprint(next_predictions, round_id)
    seed = seed_from_fingerprint(fingerprint)
    plan, audit = build_round(next_predictions, strategy="random_remaining_v24", seed=seed)
    audit["prediction_fingerprint_sha256"] = fingerprint
    audit["seed_derivation"] = "integer value of prediction_fingerprint_sha256"
    shadow_plan, shadow_audit = build_comparison_shadow(next_predictions)
    shadow_b_plan, shadow_b_audit = build_anchor_shadow(next_predictions)
    shadow_c_plan, shadow_c_audit = build_free_coverage_shadow(next_predictions)
    args.outdir.mkdir(parents=True, exist_ok=args.replace)
    next_predictions.to_csv(args.outdir / "next_predictions.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(args.outdir / "buyplan.csv", index=False, encoding="utf-8-sig")
    write_html(plan, round_id, args.outdir / "buyplan.html", generator_audit=audit["allocation"])
    shadow_plan.to_csv(args.outdir / "shadow_buyplan.csv", index=False, encoding="utf-8-sig")
    write_shadow_html(shadow_plan, round_id, args.outdir / "shadow_buyplan.html", "A", "v2.2", 11)
    shadow_b_plan.to_csv(args.outdir / "shadow_buyplan_b.csv", index=False, encoding="utf-8-sig")
    write_shadow_html(shadow_b_plan, round_id, args.outdir / "shadow_buyplan_b.html", "B", "Anchor段階型", 21)
    shadow_c_plan.to_csv(args.outdir / "shadow_buyplan_c.csv", index=False, encoding="utf-8-sig")
    write_shadow_html(shadow_c_plan, round_id, args.outdir / "shadow_buyplan_c.html", "C", "採用数自由・Dカバー型", 31)
    manifest = {
        "status": "BuyPlan v2.4 official production output with historical comparison shadows",
        "official_version": "v2.4",
        "prediction_model": "base_lambda_v1 + lab_xg_v0 + absence_attack_defense_lambda_v1 + score_distribution + fatigue_weather_uniform_v1",
        "buyplan_probability_source": "next_score_distribution_HDA + absence_attack_defense_lambda_v1 + fatigue_weather_uniform_v1",
        "round_id": round_id, "logical_season": args.logical_season,
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
        "input": str(args.input_csv.resolve()), "input_sha256": sha256(args.input_csv),
        "toto_order": str(args.toto_order_csv.resolve()),
        "generator": audit,
        "shadow_observation": shadow_audit,
        "shadow_b_observation": shadow_b_audit,
        "shadow_c_observation": shadow_c_audit,
        "production_switched": True,
        "preflight": preflight,
        "output_sha256": {name: sha256(args.outdir / name) for name in (
            "next_predictions.csv", "buyplan.csv", "buyplan.html",
            "shadow_buyplan.csv", "shadow_buyplan.html",
            "shadow_buyplan_b.csv", "shadow_buyplan_b.html",
            "shadow_buyplan_c.csv", "shadow_buyplan_c.html")},
        "toto_order_sha256": sha256(args.toto_order_csv),
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if audit["allocation"]["duplicate_pairs"]:
        print(f"警告: 完全一致する購入券があります: {audit['allocation']['duplicate_pairs']}")
    print(args.outdir / "buyplan.html")


if __name__ == "__main__":
    main()
