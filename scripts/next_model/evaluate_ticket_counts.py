#!/usr/bin/env python3
"""Evaluate BuyPlan ticket-count marginal utility on frozen historical rounds."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from build_next_buyplan import (
    extend_tickets_coverage, ranked_outcomes, select_tickets_coverage, top_joint_candidates,
)
from common import ROOT


DEFAULT_INPUT = ROOT / "data/eval/next_model/validation_v2/lab_xg_v0/predictions.csv"
DEFAULT_OUT = ROOT / "data/eval/next_model/validation_v2/ticket_count_simulation_v1"
COUNTS = (5, 10, 15, 20, 30)
FLOORS = (0.50, 0.35, 0.25)
JST = ZoneInfo("Asia/Tokyo")
RESULT_TO_SYMBOL = {1: "H", 0: "D", 2: "A"}


def evaluate_plan(frame: pd.DataFrame, tickets: list[dict]) -> dict:
    actual = tuple(RESULT_TO_SYMBOL[int(value)] for value in frame["actual_result"])
    hits = [sum(pick == result for pick, result in zip(ticket["symbols"], actual)) for ticket in tickets]
    covered = sum(any(ticket["symbols"][i] == actual[i] for ticket in tickets) for i in range(13))
    probabilities = [math.exp(ticket["log_probability"]) for ticket in tickets]
    distances = [
        sum(a != b for a, b in zip(tickets[left]["symbols"], tickets[right]["symbols"]))
        for left in range(len(tickets)) for right in range(left + 1, len(tickets))
    ]
    return {
        "ticket01_hits": hits[0],
        "average_hits": sum(hits) / len(hits),
        "best_hits": max(hits),
        "portfolio_cover": covered,
        "tickets_8plus": sum(value >= 8 for value in hits),
        "tickets_9plus": sum(value >= 9 for value in hits),
        "tickets_10plus": sum(value >= 10 for value in hits),
        "ticket_probability_mass": sum(probabilities),
        "minimum_probability_ratio": min(probabilities) / probabilities[0],
        "mean_pairwise_hamming": sum(distances) / len(distances) if distances else 0.0,
    }


def extend_tickets(
    candidates: list[dict], ranked: list[list[tuple[str, float]]], count: int,
    probability_floor_ratio: float,
) -> list[dict]:
    """Keep the official first 10 tickets and select only additional tickets."""
    official = select_tickets_coverage(
        candidates, ranked, count=10, probability_floor_ratio=0.50, max_same_picks=11
    )
    if count <= 10:
        return official[:count]
    return extend_tickets_coverage(
        candidates, ranked, official, total_count=count,
        probability_floor_ratio=probability_floor_ratio, max_same_picks=11,
    )


def simulate(source: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    exclusions = []
    for round_id, group in source.groupby("round_id", sort=True):
        group = group.sort_values("match_no").reset_index(drop=True)
        if len(group) != 13 or set(group["match_no"].astype(int)) != set(range(1, 14)):
            exclusions.append({"round_id": round_id, "rows": len(group), "reason": "not a complete 13-match round"})
            continue
        candidates = top_joint_candidates(group)
        ranked = [ranked_outcomes(row) for _, row in group.iterrows()]
        for floor in FLOORS:
            for count in COUNTS:
                base = {
                    "round_id": round_id,
                    "prematch_snapshot_proven": bool(group["prematch_snapshot_proven"].fillna(False).all()),
                    "ticket_count": count,
                    "purchase_yen": count * 100,
                    "probability_floor_ratio": floor,
                }
                try:
                    tickets = extend_tickets(candidates, ranked, count, floor)
                    rows.append({**base, "status": "ok", **evaluate_plan(group, tickets)})
                except ValueError as exc:
                    rows.append({**base, "status": "unavailable", "reason": str(exc)})
    detail = pd.DataFrame(rows)
    ok = detail[detail["status"].eq("ok")].copy()
    ok = ok.sort_values(["round_id", "probability_floor_ratio", "ticket_count"])
    ok["incremental_best_hits"] = ok.groupby(
        ["round_id", "probability_floor_ratio"]
    )["best_hits"].diff()
    ok["incremental_cover"] = ok.groupby(
        ["round_id", "probability_floor_ratio"]
    )["portfolio_cover"].diff()
    detail = detail.drop(columns=[c for c in ("incremental_best_hits", "incremental_cover") if c in detail])
    detail = detail.merge(
        ok[["round_id", "probability_floor_ratio", "ticket_count", "incremental_best_hits", "incremental_cover"]],
        on=["round_id", "probability_floor_ratio", "ticket_count"], how="left",
    )
    return detail, exclusions


def aggregate(detail: pd.DataFrame, prematch_only: bool) -> pd.DataFrame:
    work = detail[detail["status"].eq("ok")].copy()
    if prematch_only:
        work = work[work["prematch_snapshot_proven"]]
    metrics = [
        "ticket01_hits", "average_hits", "best_hits", "portfolio_cover",
        "tickets_8plus", "tickets_9plus", "tickets_10plus",
        "ticket_probability_mass", "minimum_probability_ratio", "mean_pairwise_hamming",
        "incremental_best_hits", "incremental_cover",
    ]
    grouped = work.groupby(["probability_floor_ratio", "ticket_count"])
    summary = grouped[metrics].mean().reset_index()
    summary.insert(2, "rounds_available", grouped["round_id"].nunique().to_numpy())
    return summary


def display_table(summary: pd.DataFrame) -> pd.DataFrame:
    shown = summary.copy()
    shown["probability_floor_ratio"] = shown["probability_floor_ratio"].map(lambda x: f"{x:.0%}")
    columns = {
        "probability_floor_ratio": "確率下限",
        "ticket_count": "口数",
        "rounds_available": "集計回数",
        "ticket01_hits": "候補01",
        "average_hits": "候補平均",
        "best_hits": "最高的中",
        "portfolio_cover": "cover",
        "tickets_8plus": "8点以上口数",
        "tickets_9plus": "9点以上口数",
        "tickets_10plus": "10点以上口数",
        "incremental_best_hits": "追加口の最高改善",
        "incremental_cover": "追加口のcover改善",
        "minimum_probability_ratio": "最低候補/候補01",
        "mean_pairwise_hamming": "平均券間距離",
    }
    shown = shown[list(columns)].rename(columns=columns)
    for column in shown.select_dtypes(include="number").columns:
        if column != "口数":
            shown[column] = shown[column].round(2)
    shown["最低候補/候補01"] = shown["最低候補/候補01"].map(lambda x: f"{x:.1%}")
    return shown


def write_html(path: Path, prematch: pd.DataFrame, all_rounds: pd.DataFrame, detail: pd.DataFrame, exclusions: list[dict]) -> None:
    style = """
body{font-family:system-ui,sans-serif;margin:24px;color:#172033;background:#f4f7fa}main{max-width:1400px;margin:auto}
section{background:#fff;border:1px solid #d6dee8;border-radius:10px;padding:18px;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid #d6dee8;padding:7px;text-align:right}th{background:#eaf1f7;position:sticky;top:0}th:first-child,td:first-child{text-align:left}
.note{border-left:5px solid #2878b5;background:#eef7ff;padding:12px}.warn{border-left:5px solid #d99000;background:#fff8e8;padding:12px}
"""
    prematch_view = display_table(prematch)
    all_view = display_table(all_rounds)
    unavailable = detail[~detail["status"].eq("ok")]
    html = ["<!doctype html><html lang='ja'><meta charset='utf-8'>", f"<title>BuyPlan口数シミュレーション</title><style>{style}</style><main>"]
    html += ["<h1>BuyPlan口数シミュレーション</h1>",
             "<p class='note'>v2確率を固定し、5・10・15・20・30口を生成。候補01比50％・35％・25％の確率下限を比較しています。</p>",
             "<p class='warn'>主評価は結果判明前の保存を確認できた完全3開催回です。全7回は参考値です。最高的中は口数増加で機械的に上がりやすいため、cover・高得点券数・確率低下も併せて判断します。</p>"]
    html += ["<p class='note'><b>主な結果:</b> 現行10口を固定した追加検証です。試合前保存保証3回で35％下限を使うと、20口まで最高的中平均は7.67のままで、coverは10.33から12.00へ上がりました。30口で最高的中が8.00へ上がる一方、購入額は3倍です。固定増口の根拠はまだ弱く、15〜20口の追加枠を前向き検証するのが妥当です。</p>"]
    html += ["<section><h2>主評価：試合前保存保証3開催回の平均</h2>", prematch_view.to_html(index=False, escape=True), "</section>"]
    html += ["<section><h2>参考：完全7開催回の平均</h2>", all_view.to_html(index=False, escape=True), "</section>"]
    html += ["<section><h2>開催回別詳細</h2>", detail.to_html(index=False, escape=True), "</section>"]
    html += [f"<section><h2>除外</h2><pre>{escape(json.dumps(exclusions, ensure_ascii=False, indent=2))}</pre></section>"]
    if not unavailable.empty:
        html += ["<section><h2>生成不能条件</h2>", unavailable.to_html(index=False, escape=True), "</section>"]
    html += ["</main></html>"]
    path.write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.replace:
        raise FileExistsError(f"output already exists: {args.out}")
    source = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    detail, exclusions = simulate(source)
    prematch = aggregate(detail, prematch_only=True)
    all_rounds = aggregate(detail, prematch_only=False)
    args.out.mkdir(parents=True, exist_ok=args.replace)
    detail.to_csv(args.out / "round_detail.csv", index=False, encoding="utf-8-sig")
    prematch.to_csv(args.out / "summary_prematch_proven.csv", index=False, encoding="utf-8-sig")
    all_rounds.to_csv(args.out / "summary_all_complete.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
        "input": str(args.input.resolve()), "ticket_counts": COUNTS, "probability_floors": FLOORS,
        "complete_rounds": int(detail["round_id"].nunique()),
        "prematch_proven_rounds": int(detail.loc[detail["prematch_snapshot_proven"], "round_id"].nunique()),
        "exclusions": exclusions,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html(args.out / "index.html", prematch, all_rounds, detail, exclusions)
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
