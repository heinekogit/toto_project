"""Render a retrospective purchase-style page from saved v4.0 research tickets."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/reports/buyplan_v40_prototype"
ROUND = "toto1670"
METHOD = "試作・最悪状態比率"
OUT = DATA / "toto1670_purchase_preview.html"


def main() -> None:
    tickets = pd.read_csv(DATA / "tickets.csv", encoding="utf-8-sig")
    matrix = pd.read_csv(DATA / "allocation_matrices.csv", encoding="utf-8-sig")
    summary = pd.read_csv(DATA / "portfolio_summary.csv", encoding="utf-8-sig")
    t = tickets[(tickets["round"] == ROUND) & (tickets.method == METHOD)].sort_values("ticket_no")
    m = matrix[(matrix["round"] == ROUND) & (matrix.method == METHOD)].sort_values("match_no")
    s = summary[summary["round"] == ROUND]
    if len(t) != 10 or len(m) != 13 or len(set(t.symbols)) != 10:
        raise ValueError("第1670回の相異なる10口または13試合が揃っていません")
    for row in m.itertuples():
        for n in range(1, 11):
            if getattr(row, f"ticket_{n:02}") != t.iloc[n - 1][f"match_{row.match_no:02}"]:
                raise ValueError("券一覧と配置表が一致しません")

    esc = html.escape
    cards = "".join(
        f'<article class="card"><div class="cardtop"><strong>候補{int(r.ticket_no):02}</strong><span>{int(r.D_count)} D</span></div>'
        f'<div class="symbols">{esc(r.symbols)}</div>'
        f'<div class="prob">単独13的中モデル確率 <b>{100*r.baseline_probability:.6f}%</b></div>'
        f'<div class="sub">17状態の平均 {100*r.mean_state_probability:.6f}% ・ 実結果 {int(r.hits)}/13</div></article>'
        for r in t.itertuples()
    )
    header = "".join(f"<th>{i:02}</th>" for i in range(1, 11))
    rows = []
    for r in m.itertuples():
        cells = "".join(
            f'<td class="{esc(getattr(r, f"ticket_{i:02}").lower())}">{esc(getattr(r, f"ticket_{i:02}"))}</td>'
            for i in range(1, 11)
        )
        rows.append(f'<tr><th>{int(r.match_no):02}</th><td class="fixture">{esc(str(r.home_team))} − {esc(str(r.away_team))}</td>{cells}</tr>')
    selected = s[s.method == METHOD].iloc[0]
    comparisons = "".join(
        f'<tr><td>{esc(str(r.method))}</td><td>{100*r.baseline_probability:.6f}%</td>'
        f'<td>{100*r.global_mean_probability:.6f}%</td><td>{int(r.D_tickets)}</td><td>{int(r.best_hits)}/13</td></tr>'
        for r in s.itertuples()
    )
    document = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第1670回 BuyPlan v4.0 初期試作購入画面</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#203047;background:#f3f6fb}}body{{max-width:1280px;margin:auto;padding:24px}}h1{{font-size:2rem;margin-bottom:8px}}h2{{margin-top:36px}}p{{line-height:1.6}}.notice{{background:#fff4d6;border-left:5px solid #dc9b1f;padding:14px 18px;border-radius:8px}}.stats{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.stat,.card{{background:white;border:1px solid #d9e2ed;border-radius:12px;padding:16px;box-shadow:0 2px 8px #182d4910}}.stat{{min-width:170px}}.stat b{{display:block;font-size:1.3rem;margin-top:5px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:12px}}.cardtop{{display:flex;justify-content:space-between}}.symbols{{font:700 1.35rem ui-monospace,monospace;letter-spacing:.13em;margin:20px 0 12px;overflow-wrap:anywhere}}.prob{{font-size:.9rem}}.sub{{font-size:.78rem;color:#596b7e;margin-top:7px}}.scroll{{overflow:auto;background:white;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:.85rem}}th,td{{border-bottom:1px solid #dde5ef;padding:10px;text-align:center;white-space:nowrap}}thead th{{background:#deebf8;position:sticky;top:0}}.fixture{{text-align:left;min-width:155px}}td.h{{background:#e6f4ec}}td.d{{background:#fff0d2;font-weight:700}}td.a{{background:#e9edfb}}.legend{{font-size:.88rem}}footer{{font-size:.82rem;color:#56687c;margin:34px 0}}a{{color:#175b9d}}
</style><main><h1>第1670回 BuyPlan v4.0 初期試作</h1>
<p class="notice"><strong>過去開催の回顧用シミュレーションです。Phase 3.6〜3.9と直近の条件別修正は反映していません。</strong> 実際の購入画面や正式BuyPlanには接続していません。±0.1/0.2のλ変動と17状態は研究上の仮定です。</p>
<div class="stats"><div class="stat">購入構成<b>10口 / 1,000円</b></div><div class="stat">方式<b>{esc(METHOD)}</b></div><div class="stat">10口の13的中モデル確率<b>{100*selected.baseline_probability:.6f}%</b></div><div class="stat">D入り券<b>{int(selected.D_tickets)}口</b></div></div>
<h2>購入候補10口</h2><div class="grid">{cards}</div>
<h2>試合別の指定</h2><p class="legend">H＝ホーム勝ち、D＝引き分け、A＝アウェー勝ち。各列が1口です。</p><div class="scroll"><table><thead><tr><th>No.</th><th>対戦</th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>保存方式との比較</h2><div class="scroll"><table><thead><tr><th>方式</th><th>元のfinal確率での13的中</th><th>仮定17状態の平均</th><th>D入り券</th><th>実結果の最高的中</th></tr></thead><tbody>{comparisons}</tbody></table></div>
<footer>保存済み試合前予測とv4.0研究用配券を表示。実結果は事後採点のみ。<a href="index.html">研究レポート</a> / <a href="tickets.csv">配券CSV</a> / <a href="portfolio_summary.csv">比較CSV</a></footer></main></html>'''
    OUT.write_text(document, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
