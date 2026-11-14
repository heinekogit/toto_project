"""Research-only score-neighbor purchase scenarios; no calibrated transition claim."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure import score_grid
from analyze_score_structure_phase4b_stage1 import decode, top_ten, universe

FILES = ("shadow_v41_score_neighbors.html", "shadow_v41_score_neighbors_tickets.csv",
         "shadow_v41_score_neighbors_matches.csv", "shadow_v41_score_neighbors_audit.json",
         "shadow_v41_score_neighbors_edges.csv")
STRESS_FRACTION = 0.25  # Purely hypothetical, not a fitted goal-change rate.
STATE_WEIGHTS = (0.0, 0.5, 0.75, 1.0)
SYMBOLS = np.array(["H", "D", "A"])


def neighbor_state(forecast: pd.DataFrame, fraction: float = STRESS_FRACTION):
    """Move hypothetical one-goal score mass toward adjacent draws, preserving H/D/A sum."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction outside [0,1]")
    f = forecast.sort_values("match_no").reset_index(drop=True)
    if len(f) != 13 or f.match_no.tolist() != list(range(1, 14)):
        raise ValueError("13 ordered matches required")
    p = f[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-9):
        raise ValueError("invalid final probabilities")
    q = p.copy()
    rows = []
    for n, match in f.iterrows():
        matrix, _ = score_grid(float(match.final_lambda_home), float(match.final_lambda_away))
        h1 = a1 = transfer_h = transfer_a = 0.0
        for h in range(11):
            for a in range(11):
                if abs(h-a) != 1:
                    continue
                edge = float(matrix[h,a])
                low, high = min(h,a), max(h,a)
                lower_draw = float(matrix[low,low])
                upper_draw = float(matrix[high,high])
                adjacent = lower_draw + upper_draw
                # This is a structural stress rule, not P(a real score changes).
                moved = fraction * edge * adjacent / (edge + adjacent)
                to_lower = moved * lower_draw / adjacent
                to_upper = moved * upper_draw / adjacent
                if h > a:
                    h1 += edge
                    transfer_h += moved
                else:
                    a1 += edge
                    transfer_a += moved
                rows.append(dict(match_no=n+1, home_team=match.home_team, away_team=match.away_team,
                                 score=f"{h}-{a}", side="H" if h>a else "A", score_mass=edge,
                                 lower_draw=f"{low}-{low}", lower_draw_mass=lower_draw,
                                 upper_draw=f"{high}-{high}", upper_draw_mass=upper_draw,
                                 hypothetical_to_lower=to_lower, hypothetical_to_upper=to_upper,
                                 adjacent_draw_mass=adjacent, hypothetical_transfer=moved))
        q[n,0] -= transfer_h
        q[n,2] -= transfer_a
        q[n,1] += transfer_h + transfer_a
        if (q[n] < 0).any() or not np.isclose(q[n].sum(), 1, atol=1e-12):
            raise AssertionError("invalid neighbor scenario")
    return p, q, pd.DataFrame(rows)


def build(forecast: pd.DataFrame):
    f = forecast.sort_values("match_no").reset_index(drop=True)
    p, q, edges = neighbor_state(f)
    base = universe(p)
    stress = universe(q)
    reference = top_ten(base)
    reference_set = set(map(int, reference))
    tickets, summary = [], []
    for weight in STATE_WEIGHTS:
        mixture = (1-weight)*base + weight*stress
        codes = top_ten(mixture)
        symbols = ["".join(SYMBOLS[row]) for row in decode(codes)]
        summary.append(dict(stress_fraction=STRESS_FRACTION, state_weight=weight,
                            P_prob13=float(base[codes].sum()), Q_prob13=float(stress[codes].sum()),
                            mixture_prob13=float(mixture[codes].sum()),
                            P_loss_vs_top10=float(base[reference].sum()-base[codes].sum()),
                            changed_tickets_vs_P=10-len(reference_set & set(map(int,codes))),
                            D_tickets=sum("D" in value for value in symbols)))
        for number, (code, symbol) in enumerate(zip(codes, symbols), 1):
            tickets.append(dict(state_weight=weight, candidate=number, symbols=symbol,
                                P_ticket_probability=float(base[code]), Q_ticket_probability=float(stress[code]),
                                mixture_ticket_probability=float(mixture[code]), D_count=symbol.count("D"),
                                **{f"match_{i+1:02d}":symbol[i] for i in range(13)}))
    matches = f[["match_no", "home_team", "away_team"]].copy()
    for i, label in enumerate("HDA"):
        matches[f"P_{label}"] = p[:,i]
        matches[f"Q_{label}"] = q[:,i]
    # Keep directional one-goal masses and transfers explicit.
    for n in range(1,14):
        part=edges[edges.match_no.eq(n)]
        for side in ("H","A"):
            selected=part[part.side.eq(side)]
            matches.loc[n-1,f"{side}_one_goal_mass"] = selected.score_mass.sum()
            matches.loc[n-1,f"{side}_neighbor_transfer"] = selected.hypothetical_transfer.sum()
    return matches, edges, pd.DataFrame(tickets), pd.DataFrame(summary)


def write(forecast: pd.DataFrame, round_id: str, outdir: Path) -> dict:
    outdir=Path(outdir)
    matches, edges, tickets, summary=build(forecast)
    tickets.to_csv(outdir/FILES[1],index=False,encoding="utf-8-sig")
    matches.to_csv(outdir/FILES[2],index=False,encoding="utf-8-sig")
    edges.to_csv(outdir/FILES[4],index=False,encoding="utf-8-sig")
    audit=dict(mode="v4.1 score-neighbor stress research; not calibrated or adopted",
               fraction=STRESS_FRACTION, weights=STATE_WEIGHTS, score_range="0..10 each side",
               score_rule="one-goal edge mass * adjacent draw mass / (edge + adjacent) * fraction",
               two_plus_direct_transfer=0, formal_prediction_changed=False, formal_buyplan_changed=False,
               scenarios=summary.to_dict(orient="records"))
    (outdir/FILES[3]).write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tables=[]
    for weight in STATE_WEIGHTS:
        group=tickets[tickets.state_weight.eq(weight)]
        head="<tr><th>No.</th><th>対戦</th>"+"".join(f"<th>{i:02d}</th>" for i in range(1,11))+"</tr>"
        body=[]
        for _,match in matches.iterrows():
            cells=[str(int(match.match_no)),f"{match.home_team}－{match.away_team}"]
            cells += [str(ticket[f"match_{int(match.match_no):02d}"]) for _,ticket in group.iterrows()]
            body.append("<tr>"+"".join(f"<td>{html.escape(c)}</td>" for c in cells)+"</tr>")
        tables.append(f"<h2>仮説状態の重み {weight:.0%}：{int(summary[summary.state_weight.eq(weight)].D_tickets.iloc[0])}口がD入り</h2><div class='scroll'><table>{head}{''.join(body)}</table></div>")
    summary_view=summary.copy()
    for col in ("P_prob13","Q_prob13","mixture_prob13","P_loss_vs_top10"):
        summary_view[col]=(summary_view[col]*100).map(lambda x:f"{x:.9f}%")
    match_view=matches.copy()
    for col in match_view.columns:
        if col.startswith(("P_","Q_")) or col.endswith(("mass","transfer")):
            match_view[col]=(match_view[col]*100).round(3)
    document=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>v4.1 得点差・隣接同点</title>"
              "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.note{background:#fff0d3;padding:12px}</style>"
              f"<h1>{html.escape(round_id)} v4.1 得点差・隣接同点の研究用配券</h1>"
              "<p class='note'>1-0、2-1、3-2および逆方向の全1点差スコアを隣接同点セルと対応付けます。2点差以上は1得点の変化で同点にならないため直接移しません。移動率25%と状態重みは未校正のストレス条件で、現実の得点変化率・予測確率ではありません。単一の購入推奨10口は決めません。</p>"
              "<p>H側とA側の1点差セルを別々に積算し、各セルの確率と隣接同点セルの確率の関係から仮説状態Qを作成。1-0なら0-0側と1-1側への仮説移動を別々に記録しています。保存final P、仮説Q、混合状態の10口全体13的中確率を並べます。条件SでD採用を強制しません。</p>"
              "<h2>各試合のH/D/A関係</h2><div class='scroll'>"+match_view.to_html(index=False,escape=True)+"</div>"
              "<h2>10口の評価とPでの機会損失</h2>"+summary_view.to_html(index=False,escape=True)+"".join(tables)
              +f"<p><a href='{FILES[1]}'>券CSV</a>／<a href='{FILES[2]}'>試合別CSV</a>／<a href='{FILES[4]}'>1点差から同点への行き先別CSV</a>／<a href='{FILES[3]}'>監査JSON</a></p></html>")
    (outdir/FILES[0]).write_text(document,encoding="utf-8")
    return audit
