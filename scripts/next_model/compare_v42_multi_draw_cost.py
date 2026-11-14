"""Exact cost of tickets containing multiple draws under saved final probabilities."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure_phase4b_stage1 import N, POWERS, SYMBOLS, decode, top_ten, universe
from analyze_score_structure_phase4b_stage3 import draw_distribution, top_category
from common import sha256


def analyse(source: Path, out: Path, round_id: str):
    f = pd.read_csv(source, encoding="utf-8-sig").sort_values("match_no").reset_index(drop=True)
    if len(f) != 13 or f.match_no.tolist() != list(range(1, 14)):
        raise ValueError("13 ordered matches required")
    p = f[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-9):
        raise ValueError("invalid saved probabilities")
    codes = np.arange(N, dtype=np.int64)
    d_count = np.zeros(N, dtype=np.uint8)
    for power in POWERS:
        d_count += ((codes // power) % 3 == 1).astype(np.uint8)
    mass = universe(p)
    baseline = top_ten(mass)
    baseline_mass = float(mass[baseline].sum())
    dist = draw_distribution(p)
    p_atleast2 = float(dist[2:].sum())
    out.mkdir(parents=True, exist_ok=True)

    def symbol(code):
        return "".join(SYMBOLS[decode([code])[0]])

    def rank(code):
        return int(np.count_nonzero(mass > mass[code]) + np.count_nonzero((mass == mass[code]) & (codes < code)) + 1)

    best_rows = []
    for d in range(5):
        eligible = np.flatnonzero(d_count == d)
        chosen = eligible[np.lexsort((eligible, -mass[eligible]))[0]]
        best_rows.append(dict(D_count=d, rank_all=rank(chosen), symbols=symbol(chosen),
                              ticket_probability=float(mass[chosen]), ticket_probability_percent=float(100 * mass[chosen]),
                              D_match_numbers=";".join(str(i + 1) for i, s in enumerate(symbol(chosen)) if s == "D"),
                              **{f"match_{i+1:02d}": s for i, s in enumerate(symbol(chosen))}))

    # At least one ticket with >= r draws: exact unconstrained optimum in each category.
    threshold_rows = []
    ticket_rows = []
    for r in range(2, 5):
        low = top_category(mass, d_count < r)
        high = top_category(mass, d_count >= r)
        for k in range(1, 11):
            selected = np.concatenate((low[:10-k], high[:k]))
            selected = selected[np.lexsort((selected, -mass[selected]))]
            total = float(mass[selected].sum())
            high_mass = float(mass[selected[d_count[selected] >= r]].sum())
            multi_mass = float(mass[selected[d_count[selected] >= 2]].sum())
            threshold_rows.append(dict(minimum_D_per_eligible_ticket=r, eligible_ticket_count=k,
                P13=total, P13_percent=100*total, ratio_to_unrestricted=total/baseline_mass,
                opportunity_loss=baseline_mass-total, opportunity_loss_percent=100*(baseline_mass-total),
                P13_mass_D_atleast2=multi_mass, conditional_P13_given_D_atleast2=multi_mass/p_atleast2,
                P13_mass_eligible=high_mass, conditional_P13_given_D_atleast_r=high_mass/float(dist[r:].sum()),
                actual_D_ticket_counts=";".join(map(str, d_count[selected])),
                added_tickets=";".join(symbol(c) for c in selected if c not in baseline),
                removed_tickets=";".join(symbol(c) for c in baseline if c not in selected)))
            if k == 1:
                for number, code in enumerate(selected, 1):
                    ticket_rows.append(dict(minimum_D_per_eligible_ticket=r, candidate=number,
                        symbols=symbol(code), D_count=int(d_count[code]), probability=float(mass[code]),
                        **{f"match_{i+1:02d}": s for i, s in enumerate(symbol(code))}))

    # Independently check the one-ticket threshold: best high ticket replaces the
    # least valuable top-ten baseline ticket when all baseline tickets are D-free.
    if np.any(d_count[baseline]):
        raise ValueError("current audit assumes the saved unrestricted top ten are D-free")
    for r in range(2, 5):
        row = next(x for x in threshold_rows if x["minimum_D_per_eligible_ticket"] == r and x["eligible_ticket_count"] == 1)
        high = top_category(mass, d_count >= r)
        expected = baseline_mass - mass[baseline[-1]] + mass[high[0]]
        if not np.isclose(row["P13"], expected, rtol=0, atol=1e-15):
            raise AssertionError("one-ticket optimum mismatch")
    if not all(row["P13"] <= baseline_mass + 1e-15 for row in threshold_rows):
        raise AssertionError("constraint beat unrestricted optimum")
    for r in range(2, 5):
        values = [x["P13"] for x in threshold_rows if x["minimum_D_per_eligible_ticket"] == r]
        if np.any(np.diff(values) > 1e-15):
            raise AssertionError("threshold frontier is not monotone")

    best = pd.DataFrame(best_rows)
    frontier = pd.DataFrame(threshold_rows)
    tickets = pd.DataFrame(ticket_rows)
    best.to_csv(out / "best_ticket_by_D_count.csv", index=False, encoding="utf-8-sig")
    frontier.to_csv(out / "multi_D_frontier.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(out / "one_multi_D_ticket_portfolios.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"D_count": range(14), "probability": dist}).to_csv(out / "D_count_distribution.csv", index=False, encoding="utf-8-sig")
    audit = dict(round=round_id, forecast_sha256=sha256(source), outcomes=N, baseline_P13=baseline_mass,
                 P_D_atleast2=p_atleast2, fixed_saved_final_probabilities=True, actual_results_used=False,
                 all_11x11_score_structure_unused_for_probability_adjustment=True,
                 optimization="exact category top-k for each threshold and count; no D count or placement constraints within categories")
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    selected = frontier[frontier.eligible_ticket_count.eq(1)].copy()
    display = selected[["minimum_D_per_eligible_ticket", "P13_percent", "ratio_to_unrestricted", "opportunity_loss_percent", "conditional_P13_given_D_atleast2", "added_tickets", "removed_tickets"]].copy()
    display["ratio_to_unrestricted"] *= 100
    display["conditional_P13_given_D_atleast2"] *= 100
    for col in ("P13_percent", "opportunity_loss_percent", "conditional_P13_given_D_atleast2"):
        display[col] = display[col].map(lambda x: f"{x:.9f}%")
    display["ratio_to_unrestricted"] = display.ratio_to_unrestricted.map(lambda x: f"{x:.3f}%")
    tables = []
    for r in range(2, 5):
        subset = tickets[tickets.minimum_D_per_eligible_ticket.eq(r)]
        head = "<tr><th>No.</th><th>対戦</th>" + "".join(f"<th>{j:02d}</th>" for j in range(1, 11)) + "</tr>"
        body = []
        for i, row in f.iterrows():
            cells = [str(i+1), f"{row.home_team}－{row.away_team}"] + [str(t[f"match_{i+1:02d}"]) for _, t in subset.iterrows()]
            body.append("<tr>"+"".join(f"<td>{html.escape(c)}</td>" for c in cells)+"</tr>")
        tables.append(f"<details><summary>D{r}個以上の券を1口：13×10配券</summary><div class='scroll'><table>{head}{''.join(body)}</table></div></details>")
    doc = ("<!doctype html><html lang='ja'><meta charset='utf-8'><title>複数D券の機会損失</title>"
        "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}details{padding:8px;border-bottom:1px solid #ddd}.note{background:#fff0d3;padding:12px}</style>"
        f"<h1>{html.escape(round_id)} 複数D券の順位と機会損失</h1>"
        f"<p>保存P(D)でDが2試合以上発生する確率：{100*p_atleast2:.3f}%。固定確率下の無制約最適10口：{100*baseline_mass:.9f}%。</p>"
        "<p class='note'>D2個以上、3個以上、4個以上の券を仮に含める場合の代償。券の購入推奨ではなく、得点構造によるD予測改善も表さない。結果は選択に使っていない。</p>"
        "<h2>D個数別の最高確率券</h2><div class='scroll'>"+best.to_html(index=False)+"</div>"
        "<h2>複数D券を1口入れた場合</h2><div class='scroll'>"+display.to_html(index=False)+"</div>"
        "<h2>対象券を1～10口に増やした場合</h2><div class='scroll'>"+frontier.to_html(index=False)+"</div>"
        +"".join(tables)+"<p><a href='best_ticket_by_D_count.csv'>D個数別最有力券</a>／<a href='multi_D_frontier.csv'>全条件CSV</a>／<a href='one_multi_D_ticket_portfolios.csv'>配券CSV</a>／<a href='report.md'>解説</a></p></html>")
    (out / "index.html").write_text(doc, encoding="utf-8")
    report = (f"# {round_id} 複数D券の機会損失\n\n"
        f"保存final確率によるD2試合以上発生確率は {100*p_atleast2:.6f}% 。"
        "D発生数の高い事象確率と、そのうち特定の13試合結果を10口で買う確率は別である。\n\n"
        "各D個数の最有力券と全順位は best_ticket_by_D_count.csv。"
        "複数D券を含む厳密な10口最適解は multi_D_frontier.csv。"
        "条件ごとの13×10配券はHTMLとCSVに保存。"
        "D枚数・試合別D採用数・券別D数は固定していない。"
        "今回は配券側の代償を測定したもので、d-scoopの予測側の識別力は検証していない。\n\n"
        "## 1口だけ含める場合\n\n```csv\n"+selected.to_csv(index=False)+"```\n")
    (out / "report.md").write_text(report, encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    print(json.dumps(analyse(args.forecast, args.outdir, args.round), ensure_ascii=False))
