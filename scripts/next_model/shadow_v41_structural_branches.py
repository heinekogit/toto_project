"""Research portfolios from one-goal score adjacency, without changing H/D/A probabilities."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure import score_grid
from analyze_score_structure_phase4b_stage1 import N, POWERS, decode, top_ten, universe

FILES = ("shadow_v41_structural_branches.html", "shadow_v41_structural_branches.csv",
         "shadow_v41_structural_matches.csv", "shadow_v41_structural_audit.json")
SYMBOLS = np.array(list("HDA"))


def evaluate(forecast: pd.DataFrame):
    f=forecast.sort_values("match_no").reset_index(drop=True)
    if len(f)!=13 or f.match_no.tolist()!=list(range(1,14)):
        raise ValueError("13 ordered matches required")
    p=f[["next_prob_home","next_prob_draw","next_prob_away"]].to_numpy(float)
    if not np.isfinite(p).all() or (p<=0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-9):
        raise ValueError("invalid final probabilities")
    mass=universe(p)
    baseline=top_ten(mass)
    codes=np.arange(N,dtype=np.int64)
    matches=[]
    for i,r in f.iterrows():
        grid,_=score_grid(float(r.final_lambda_home),float(r.final_lambda_away))
        # For every 1-goal score, compare its mass to both adjacent equal-score cells.
        # Harmonic overlap is a structural ranking measure, not a probability correction.
        side={"H":0.0,"A":0.0};one_goal={"H":0.0,"A":0.0}
        for h in range(11):
            for a in range(11):
                if abs(h-a)!=1:continue
                edge=float(grid[h,a]);lower=min(h,a);upper=max(h,a)
                adjacent=float(grid[lower,lower]+grid[upper,upper])
                label="H" if h>a else "A"
                side[label]+=2*edge*adjacent/(edge+adjacent)
                one_goal[label]+=edge
        eligible=np.flatnonzero((codes//POWERS[i])%3==1)
        # Best original-P ticket with D in this match; no fixed other-match symbols.
        highest=float(mass[eligible].max())
        best=int(eligible[np.flatnonzero(mass[eligible]>=highest-1e-18)[0]])
        matches.append(dict(match_no=i+1,home_team=r.home_team,away_team=r.away_team,
                            final_P_H=p[i,0],final_P_D=p[i,1],final_P_A=p[i,2],
                            H_one_goal_mass=one_goal["H"],A_one_goal_mass=one_goal["A"],
                            H_adjacent_overlap=side["H"],A_adjacent_overlap=side["A"],
                            structural_priority=side["H"]+side["A"],
                            strongest_D_ticket="".join(SYMBOLS[decode([best])[0]]),
                            strongest_D_ticket_P=highest,code=best))
    matches=pd.DataFrame(matches).sort_values(["structural_priority","match_no"],ascending=[False,True]).reset_index(drop=True)
    matches["structural_rank"]=np.arange(1,14)
    portfolios=[];summaries=[]
    # No single D count is selected: show the full cumulative structural frontier.
    for k in range(11):
        required=[int(x) for x in matches.head(k).code]
        selected=[]
        for code in required:
            if code not in selected:selected.append(code)
        for code in np.argsort(-mass,kind="stable")[:100]:
            if len(selected)==10:break
            if int(code) not in selected:selected.append(int(code))
        if len(selected)!=10:raise AssertionError("ten tickets not formed")
        selected=sorted(selected,key=lambda code:(-mass[code],code))
        symbols=["".join(SYMBOLS[row]) for row in decode(selected)]
        covered=[int(no) for no in matches.head(k).match_no if any(s[int(no)-1]=="D" for s in symbols)]
        if len(covered)!=k:raise AssertionError("required D branch missing")
        total=float(mass[selected].sum())
        summaries.append(dict(structural_top_k=k, P_prob13=total, P_loss_vs_unrestricted=float(mass[baseline].sum()-total),
                              D_tickets=sum("D" in s for s in symbols),required_D_matches=";".join(map(str,covered)),
                              distinct_tickets=len(set(symbols))))
        for n,(code,symbol) in enumerate(zip(selected,symbols),1):
            portfolios.append(dict(structural_top_k=k,candidate=n,symbols=symbol,P_ticket_probability=float(mass[code]),
                                   **{f"match_{j+1:02d}":symbol[j] for j in range(13)}))
    return matches,pd.DataFrame(portfolios),pd.DataFrame(summaries)


def write(forecast: pd.DataFrame,round_id: str,outdir: Path) -> dict:
    outdir=Path(outdir)
    matches,tickets,summary=evaluate(forecast)
    tickets.to_csv(outdir/FILES[1],index=False,encoding="utf-8-sig")
    matches.drop(columns=["code"]).to_csv(outdir/FILES[2],index=False,encoding="utf-8-sig")
    audit=dict(mode="v4.1 structural branch frontier; research only",forecast_probabilities_changed=False,
               actual_result_used=False,priority="sum of H/A one-goal edge to adjacent-draw harmonic overlap",
               branch_ticket="highest original-P ticket containing D in each ranked match",
               portfolio="mandatory branch candidates for top k, then highest original-P remaining tickets",
               k_values=list(range(11)),global_optimum_under_constraint=False,
               interpretation="diagnostic frontier, not a chosen purchase portfolio",
               scenarios=summary.to_dict(orient="records"))
    (outdir/FILES[3]).write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    view=matches.drop(columns=["code"]).copy()
    for c in view.columns:
        if c.startswith(("final_P_","H_one","A_one","H_adjacent","A_adjacent","structural_priority","strongest_D_ticket_P")):
            view[c]=(view[c]*100).round(3)
    summary_view=summary.copy()
    for c in ("P_prob13","P_loss_vs_unrestricted"):
        summary_view[c]=(summary_view[c]*100).map(lambda x:f"{x:.9f}%")
    tables=[]
    f=forecast.sort_values("match_no").reset_index(drop=True)
    for k in range(11):
        group=tickets[tickets.structural_top_k.eq(k)]
        head="<tr><th>No.</th><th>対戦</th>"+"".join(f"<th>{n:02d}</th>" for n in range(1,11))+"</tr>"
        body=[]
        for i,row in f.iterrows():
            cells=[str(i+1),f"{row.home_team}－{row.away_team}"]
            cells += [str(t[f"match_{i+1:02d}"]) for _,t in group.iterrows()]
            body.append("<tr>"+"".join(f"<td>{html.escape(c)}</td>" for c in cells)+"</tr>")
        tables.append(f"<details><summary>構造順位の上位{k}試合にD分岐を残す10口</summary><div class='scroll'><table>{head}{''.join(body)}</table></div></details>")
    document=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>v4.1得点構造分岐</title>"
              "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.note{background:#fff0d3;padding:12px}details{padding:8px;border-bottom:1px solid #ddd}</style>"
              f"<h1>{html.escape(round_id)} 1点差と隣接同点から作る配券分岐</h1>"
              "<p class='note'>予測H/D/Aを変更せず、25%などの移動率も使いません。1-0→0-0／1-1、2-1→1-1／2-2、3-2→2-2／3-3などの構造からD分岐の優先順位を作り、優先順位の上位k試合を券に含めた場合の10口をすべて示します。これはD予測誤差の実証でも購入推奨でもありません。</p>"
              "<p>構造指標は各1点差セルと隣接同点セルの調和的な重なりをH側・A側別に集計した値です。Dを含む券は元のPで最も強いものを各試合で選び、残りは元のP上位から補います。k=0〜10を比較し、一つを正式値として選びません。k別の結果は探索的な構成で、制約付きの大域最適を保証しません。</p>"
              "<h2>試合別の関係とD分岐券</h2><div class='scroll'>"+view.to_html(index=False,escape=True)+"</div>"
              "<h2>10口の結果と元Pの機会損失</h2><div class='scroll'>"+summary_view.to_html(index=False,escape=True)+"</div>"
              +"".join(tables)+f"<p><a href='{FILES[1]}'>全配券CSV</a>／<a href='{FILES[2]}'>試合別構造CSV</a>／<a href='{FILES[3]}'>監査JSON</a></p></html>")
    (outdir/FILES[0]).write_text(document,encoding="utf-8")
    return audit
