#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import List, Tuple

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import buyplan

PURCHASE_DIR = os.path.join(ROOT_DIR, "data", "purchase_reference")
BACKTEST_ROOT = os.path.join(PURCHASE_DIR, "backtest")
ROUNDS_ROOT = os.path.join(ROOT_DIR, "data", "eval", "rounds")


def _discover_rounds(backtest_root: str) -> List[str]:
    rounds: List[str] = []
    if not os.path.isdir(backtest_root):
        return rounds
    for name in sorted(os.listdir(backtest_root)):
        if (
            re.fullmatch(r"round\d{2}", name)
            and os.path.exists(os.path.join(backtest_root, name, "buyplan_scored_summary.csv"))
            and _round_is_backtestable(name)
        ):
            rounds.append(name)
    return rounds


def _round_is_backtestable(round_name: str) -> bool:
    pred_csv = os.path.join(ROUNDS_ROOT, round_name, "snapshot", "predictions.csv")
    actual_csv = os.path.join(ROUNDS_ROOT, round_name, "actual_results.csv")
    if not (os.path.exists(pred_csv) and os.path.exists(actual_csv)):
        return False
    pred = pd.read_csv(pred_csv, usecols=["home_team", "away_team"])
    pred_keys = {
        (buyplan._norm_team_key(h), buyplan._norm_team_key(a))
        for h, a in pred.drop_duplicates().itertuples(index=False, name=None)
    }
    actual = pd.read_csv(actual_csv, usecols=["home_team", "away_team"])
    for h, a in actual.itertuples(index=False, name=None):
        if (buyplan._norm_team_key(h), buyplan._norm_team_key(a)) not in pred_keys:
            return False
    return True


def _best_summary(summary_csv: str) -> Tuple[str, int, int, float]:
    df = pd.read_csv(summary_csv)
    best = df.sort_values(["hits", "ticket"], ascending=[False, True]).iloc[0]
    return str(best["ticket"]), int(best["hits"]), int(best["total"]), float(df["hit_rate"].mean())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build buyplan simulation index page")
    p.add_argument("--purchase-dir", default=PURCHASE_DIR)
    p.add_argument("--backtest-root", default=BACKTEST_ROOT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    purchase_dir = os.path.abspath(args.purchase_dir)
    backtest_root = os.path.abspath(args.backtest_root)
    out_html = os.path.join(purchase_dir, "buyplan_simulation.html")

    rows = []
    current_summary = os.path.join(purchase_dir, "buyplan_scored_summary.csv")
    current_html = "buyplan_simulation_current.html"
    if os.path.exists(current_summary):
        ticket, hits, total, avg = _best_summary(current_summary)
        rows.append(("現在", current_html, ticket, hits, total, avg))

    rounds = _discover_rounds(backtest_root)
    for round_name in rounds:
        summary_csv = os.path.join(backtest_root, round_name, "buyplan_scored_summary.csv")
        ticket, hits, total, avg = _best_summary(summary_csv)
        round_no = int(round_name.replace("round", ""))
        rows.append((f"第{round_no:02d}節", os.path.join("backtest", round_name, "buyplan_simulation.html"), ticket, hits, total, avg))

    html: List[str] = []
    html.append("<!doctype html>")
    html.append("<html lang='ja'><head><meta charset='utf-8'>")
    html.append("<title>buyplanシミュレーション一覧</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui,-apple-system,sans-serif;margin:20px;}")
    html.append("table{border-collapse:collapse;width:100%;font-size:13px;margin-top:12px;}")
    html.append("th,td{border:1px solid #ddd;padding:8px;text-align:center;}")
    html.append("th{background:#f5f5f5;}")
    html.append(".nav{margin:10px 0 14px;padding:10px 12px;border:1px solid #ddd;background:#fafafa;display:flex;gap:10px;align-items:center;flex-wrap:wrap;}")
    html.append(".nav label{font-weight:700;}")
    html.append(".nav select{font:inherit;padding:4px 8px;min-width:220px;}")
    html.append(".left{text-align:left;}")
    html.append("</style></head><body>")
    html.append("<h2>buyplanシミュレーション一覧</h2>")
    html.append("<p>現行の buyplan ロジックを、現在データと過去節 snapshot に再適用したバックテスト入口です。</p>")
    html.append("<div class='nav'>")
    html.append("<label for='round-select'>表示を選択</label>")
    html.append("<select id='round-select' onchange=\"if(this.value){window.location.href=this.value;}\">")
    for label, rel, _, _, _, _ in rows:
        html.append(f"<option value='{rel}'>{label}</option>")
    html.append("</select>")
    html.append("</div>")
    html.append("<table><thead><tr><th>対象</th><th>リンク</th><th>best</th><th>best_hits</th><th>avg_hit</th></tr></thead><tbody>")
    for label, rel, ticket, hits, total, avg in rows:
        html.append(
            f"<tr><td>{label}</td><td class='left'><a href='{rel}'>{rel}</a></td>"
            f"<td>{ticket}</td><td>{hits}/{total}</td><td>{avg:.1%}</td></tr>"
        )
    html.append("</tbody></table>")
    html.append("</body></html>")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[OK] {out_html}")


if __name__ == "__main__":
    main()
