"""Show the latest saved Phase 3.7 round as a retrospective purchase preview."""
from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/reports/score_structure_phase37"
OUT = BASE / "toto1670_purchase_preview.html"
ROUND = "toto1670"
CONFIGS = [
    ("補正なし", "E0625", "全H/D/A許容・各状態上位30組。固定final確率での基準。"),
    ("中間の感度試験", "E0626", "Dオッズ1.5倍。未校正の仮定。"),
    ("D分岐の感度試験", "E0622", "Dオッズ2倍・全許容空間を監査。未校正の仮定。"),
]


def main() -> None:
    evaluations = pd.read_csv(BASE / "evaluation.csv")
    tickets = pd.read_csv(BASE / "tickets.csv")
    matches = pd.read_csv(BASE / "match_stage_audit.csv")
    blocks = []
    compare = []
    for label, config_id, note in CONFIGS:
        e = evaluations[evaluations.id.eq(config_id)].iloc[0]
        t = tickets[tickets.id.eq(config_id)].sort_values("ticket_no")
        m = matches[matches.id.eq(config_id)].sort_values("match_no")
        if e["round"] != ROUND or len(t) != 10 or len(m) != 13 or t.symbols.nunique() != 10:
            raise ValueError(f"{config_id}: 13試合・相異なる10口の保存結果が揃っていません")
        for _, row in m.iterrows():
            no = int(row.match_no)
            if "".join(t[f"match_{no:02}"].astype(str)) != "".join(t.symbols.str[no - 1]):
                raise ValueError(f"{config_id}: 配置が一致しません No.{no}")
        cards = "".join(
            f'<article class="card"><strong>候補{int(r.ticket_no):02}</strong><span class="symbols">{escape(r.symbols)}</span>'
            f'<small>D {r.symbols.count("D")}試合 ・ 単独13的中モデル確率 {100*r.final_probability:.6f}% ・ 実結果 {int(r.hits)}/13</small></article>'
            for r in t.itertuples()
        )
        rows = "".join(
            f'<tr><th>{int(r.match_no):02}</th><td class="fixture">{escape(str(r.card))}</td>'
            + "".join(f'<td class="{str(v).lower()}">{escape(str(v))}</td>' for v in t[f"match_{int(r.match_no):02}"])
            + "</tr>" for r in m.itertuples()
        )
        head = "".join(f"<th>{n:02}</th>" for n in range(1, 11))
        blocks.append(f'<section id="{config_id}"><h2>{escape(label)}</h2><p>{escape(note)} 対象：条件S優先・5候補・候補外D禁止なし。探索設定：{escape(str(e.topN))}。</p>'
                      f'<div class="cards">{cards}</div><div class="scroll"><table><thead><tr><th>No.</th><th>対戦</th>{head}</tr></thead><tbody>{rows}</tbody></table></div></section>')
        compare.append(f'<tr><td><a href="#{config_id}">{escape(label)}</a></td><td>{100*e.final_prob13:.6f}%</td><td>{100*e.baseline_retention:.2f}%</td>'
                       f'<td>{int(e.D_cells)}</td><td>{int(e.best_hits)}/13</td><td>{escape(str(e.topN))}</td></tr>')
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第1670回 Phase 3.7 最新配券検証</title><style>
:root{{font-family:system-ui,sans-serif;color:#1e3045;background:#f2f5f9}}body{{max-width:1280px;margin:auto;padding:24px}}h1{{margin-bottom:8px}}h2{{margin-top:38px}}p{{line-height:1.6}}.notice{{padding:15px 18px;background:#fff0d4;border-left:5px solid #d28c12;border-radius:8px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:10px;margin:18px 0}}.card{{background:white;border:1px solid #d6dfeb;border-radius:10px;padding:13px;display:flex;flex-direction:column;gap:8px}}.symbols{{font:700 1.3rem ui-monospace,monospace;letter-spacing:.13em;overflow-wrap:anywhere}}small{{color:#52677e;line-height:1.5}}.scroll{{overflow:auto;background:white;border:1px solid #d6dfeb;border-radius:10px;margin-bottom:20px}}table{{border-collapse:collapse;width:100%;font-size:.84rem}}th,td{{padding:10px;text-align:center;border-bottom:1px solid #e2e8f0;white-space:nowrap}}thead th{{background:#dce9f7}}.fixture{{text-align:left;min-width:140px}}td.h{{background:#e7f5ec}}td.d{{background:#fff0d0;font-weight:700}}td.a{{background:#e8edfc}}a{{color:#165b9b}}footer{{color:#586d83;font-size:.85rem;margin:34px 0}}
</style><main><h1>第1670回・最新の配券検証</h1>
<p class="notice"><strong>Phase 3.7までの保存済み10口を表示する回顧用画面です。</strong> Phase 3.8・3.9は診断で、配券の採用方式は決めていません。以下のDオッズ1.5倍・2倍は未校正の感度試験で、正式なv4.0購入券ではありません。3方式とも実結果を使って選択していません。</p>
<p>条件SはD候補の優先評価に使い、候補外のDは禁じません。各方式とも13試合・相異なる10口・1,000円相当。H＝ホーム勝ち、D＝引き分け、A＝アウェー勝ち。</p>
<h2>10口の比較</h2><div class="scroll"><table><thead><tr><th>条件</th><th>元のfinal確率での13的中</th><th>補正なし比</th><th>D配置総数</th><th>実結果の最高的中</th><th>候補探索</th></tr></thead><tbody>{''.join(compare)}</tbody></table></div>
{''.join(blocks)}<footer>出典：Phase 3.7の<a href="evaluation.csv">評価CSV</a>・<a href="tickets.csv">配券CSV</a>・<a href="match_stage_audit.csv">試合監査CSV</a>。<a href="index.html">詳細レポート</a>。正式Next ModelとBuyPlanは変更していません。</footer></main></html>'''
    OUT.write_text(page, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
