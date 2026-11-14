"""Render research portfolios in the familiar 13-match x candidate01-10 layout."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


def render(source: Path, research_dir: Path):
    forecast = pd.read_csv(source, encoding="utf-8-sig").sort_values("match_no").reset_index(drop=True)
    tickets = pd.read_csv(research_dir / "one_multi_D_ticket_portfolios.csv", encoding="utf-8-sig")
    frontier = pd.read_csv(research_dir / "multi_D_frontier.csv", encoding="utf-8-sig")
    links = []
    for minimum in (2, 3, 4):
        chosen = tickets[tickets.minimum_D_per_eligible_ticket.eq(minimum)].sort_values("candidate")
        if len(chosen) != 10 or chosen.symbols.nunique() != 10:
            raise ValueError("expected ten unique tickets")
        summary = frontier[(frontier.minimum_D_per_eligible_ticket.eq(minimum)) & (frontier.eligible_ticket_count.eq(1))].iloc[0]
        probability_rows = []
        for _, ticket in chosen.iterrows():
            probability_rows.append({"候補": f"候補{int(ticket.candidate):02d}",
                                     "13的中のモデル確率": f"{100*float(ticket.probability):.8f}%",
                                     "D本数": int(ticket.D_count)})
        matrix_rows = []
        for i, match in forecast.iterrows():
            row = {"No.": i + 1, "対戦": f"{match.home_team}－{match.away_team}"}
            for _, ticket in chosen.iterrows():
                row[f"候補{int(ticket.candidate):02d}"] = ticket[f"match_{i+1:02d}"]
            matrix_rows.append(row)
        title = f"toto1672 研究用BuyPlan：D{minimum}個以上の券を1口"
        document = ("<!doctype html><html lang='ja'><meta charset='utf-8'>"
                    f"<title>{html.escape(title)}</title>"
                    "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#172033}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd5df;padding:7px;text-align:center}th{background:#edf2f7}td:nth-child(2){text-align:left;white-space:nowrap}.note{background:#eef7ff;border-left:5px solid #2878b5;padding:12px}.scroll{overflow:auto}.draw{background:#ffe5a8;font-weight:700}</style>"
                    f"<h1>{html.escape(title)}</h1>"
                    f"<p class='note'>保存済みH/D/A確率を変更せず、Dが{minimum}個以上の券を最低1口含む条件で13的中モデル確率が最大の10口を厳密に選択。研究用の比較であり、正式購入券やd-scoop補正後の予測ではありません。</p>"
                    f"<p><strong>10口合計の13的中モデル確率：{float(summary.P13_percent):.9f}%</strong>（制約なし最適10口の{100*float(summary.ratio_to_unrestricted):.3f}%）</p>"
                    "<h2>候補ごとのモデル確率</h2>"+pd.DataFrame(probability_rows).to_html(index=False, escape=True)
                    +"<h2>配券</h2><div class='scroll'>"+pd.DataFrame(matrix_rows).to_html(index=False, escape=True).replace("<td>D</td>", "<td class='draw'>D</td>")+"</div>"
                    +"<p><a href='buyplan_index.html'>3案の一覧</a>／<a href='multi_D_frontier.csv'>全条件の比較CSV</a></p></html>")
        path = research_dir / f"buyplan_D{minimum}_one_ticket.html"
        path.write_text(document, encoding="utf-8")
        links.append(f"<li><a href='{path.name}'>{html.escape(title)}</a>：{float(summary.P13_percent):.9f}%</li>")
    landing = ("<!doctype html><html lang='ja'><meta charset='utf-8'><title>toto1672 複数D研究用配券</title>"
               "<style>body{font-family:system-ui;margin:24px;color:#172033}li{margin:12px 0}</style>"
               "<h1>toto1672 複数D研究用配券</h1><p>いつもの13試合×候補01〜10形式で3案を表示します。いずれも仮の条件を置いた研究用配券です。</p><ul>"
               + "".join(links) + "</ul><p><a href='index.html'>機会損失の詳細</a></p></html>")
    (research_dir / "buyplan_index.html").write_text(landing, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--research-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.forecast, args.research_dir)
