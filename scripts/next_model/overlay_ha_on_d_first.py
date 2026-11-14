"""Keep D masks, overlay the ten strongest H/A-only patterns, then pair them."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from common import sha256


def ha_top_ten(p):
    codes=np.arange(2**13,dtype=np.int64)
    bits=((codes[:,None]>>np.arange(12,-1,-1))&1).astype(np.uint8)
    tickets=bits*2
    conditional=p[:,[0,2]]/p[:,[0,2]].sum(axis=1,keepdims=True)
    mass=np.prod(conditional[np.arange(13)[None,:],bits],axis=1)
    order=np.lexsort((codes,-mass))[:10]
    return tickets[order],mass[order]


def overlay_and_pair(forecast, existing):
    f=forecast.sort_values("match_no").reset_index(drop=True)
    if len(f)!=13 or f.match_no.tolist()!=list(range(1,14)):
        raise ValueError("expected 13 ordered matches")
    p=f[["next_prob_home","next_prob_draw","next_prob_away"]].to_numpy(float)
    old=np.stack([existing.sort_values("candidate")[f"match_{i+1:02d}"].map({"H":0,"D":1,"A":2}).to_numpy(np.uint8) for i in range(13)],axis=1)
    if old.shape!=(10,13) or len({tuple(row) for row in old})!=10:
        raise ValueError("existing portfolio invalid")
    masks=old==1
    if len({tuple(row) for row in masks})!=10:
        raise ValueError("this pairing procedure expects 10 distinct D masks")
    patterns,conditional_mass=ha_top_ten(p)
    leader=np.where(p[:,0]>=p[:,2],0,2)
    pair_ticket=np.empty((10,10,13),dtype=np.uint8)
    flip=np.zeros((10,10,13),dtype=np.int8)
    prob=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            t=patterns[j].copy();t[masks[i]]=1
            pair_ticket[i,j]=t
            flip[i,j]=((t!=1)&(t!=leader)).astype(np.int8)
            prob[i,j]=np.prod(p[np.arange(13),t])
    # Binary assignment x[i,j], and y[m] for whether at least one ticket
    # retains the opposite H/A symbol at match m.
    n=113
    assignment=np.zeros((20,n))
    for i in range(10):assignment[i,i*10:(i+1)*10]=1
    for j in range(10):assignment[10+j,j:100:10]=1
    cover=np.zeros((13,n))
    for m in range(13):
        cover[m,:100]=-flip[:,:,m].reshape(-1)
        cover[m,100+m]=1
    bounds=Bounds(np.zeros(n),np.ones(n))
    integrality=np.ones(n)
    basic=[LinearConstraint(assignment,np.ones(20),np.ones(20)),
           LinearConstraint(cover,np.full(13,-np.inf),np.zeros(13))]
    flip_count=flip.sum(axis=2).reshape(-1)
    first=np.zeros(n);first[:100]=-flip_count;first[100:]=-1000
    first_result=milp(first,integrality=integrality,bounds=bounds,constraints=basic,
                      options={"time_limit":30})
    if first_result.x is None or first_result.status!=0:
        raise RuntimeError("pairing coverage optimization failed")
    max_coverage=int(round(first_result.x[100:].sum()))
    max_flips=int(round(flip_count@first_result.x[:100]))
    coverage_constraint=np.zeros(n);coverage_constraint[100:]=1
    flips_constraint=np.zeros(n);flips_constraint[:100]=flip_count
    second=milp(np.r_[-prob.reshape(-1),np.zeros(13)],integrality=integrality,bounds=bounds,
                constraints=basic+[LinearConstraint(coverage_constraint,max_coverage,max_coverage),
                                   LinearConstraint(flips_constraint,max_flips,max_flips)],
                options={"time_limit":30})
    if second.x is None or second.status!=0:
        raise RuntimeError("pairing probability tie-break failed")
    chosen=np.argmax(second.x[:100].reshape(10,10),axis=1)
    overlay=pair_ticket[np.arange(10),chosen]
    if not np.array_equal(overlay==1,masks) or len({tuple(row) for row in overlay})!=10:
        raise AssertionError("D changed or duplicate ticket")
    return p,old,patterns,conditional_mass,chosen,overlay,max_coverage,max_flips


def write(forecast_path,existing_path,outdir):
    f=pd.read_csv(forecast_path,encoding="utf-8-sig")
    existing=pd.read_csv(existing_path,encoding="utf-8-sig")
    p,old,patterns,pattern_mass,pairing,overlay,coverage,flip_total=overlay_and_pair(f,existing)
    outdir.mkdir(parents=True,exist_ok=True)
    leader=np.where(p[:,0]>=p[:,2],0,2)
    symbol=np.array(["H","D","A"])
    rows=[]
    for portfolio_name,matrix in (("previous",old),("HA_overlay",overlay)):
        for j,ticket in enumerate(matrix,1):
            rows.append(dict(portfolio=portfolio_name,candidate=j,
                symbols="".join(symbol[ticket]),D_count=int((ticket==1).sum()),
                HA_reverse_count=int(((ticket!=1)&(ticket!=leader)).sum()),
                ticket_probability=float(np.prod(p[np.arange(13),ticket])),
                **{f"match_{i+1:02d}":symbol[ticket[i]] for i in range(13)}))
    pd.DataFrame(rows).to_csv(outdir/"portfolio_comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([dict(HA_pattern_rank=j+1,pattern="".join(symbol[t]),
                      conditional_HA_probability=float(pattern_mass[j]),
                      assigned_D_mask_ticket=int(np.flatnonzero(pairing==j)[0])+1)
                  for j,t in enumerate(patterns)]).to_csv(outdir/"HA_top10_assignment.csv",index=False,encoding="utf-8-sig")
    diag=[]
    for i,match in f.sort_values("match_no").reset_index(drop=True).iterrows():
        diag.append(dict(match_no=i+1,home_team=match.home_team,away_team=match.away_team,
            candidate00=symbol[leader[i]],D_cells=int((overlay[:,i]==1).sum()),
            HA_reverse_cells_previous=int(((old[:,i]!=1)&(old[:,i]!=leader[i])).sum()),
            HA_reverse_cells_overlay=int(((overlay[:,i]!=1)&(overlay[:,i]!=leader[i])).sum())))
    pd.DataFrame(diag).to_csv(outdir/"HA_coverage.csv",index=False,encoding="utf-8-sig")
    previous_p13=float(np.prod(p[np.arange(13)[None,:],old],axis=1).sum())
    overlay_p13=float(np.prod(p[np.arange(13)[None,:],overlay],axis=1).sum())
    audit=dict(forecast_sha256=sha256(forecast_path),prior_portfolio_sha256=sha256(existing_path),
        buyplan_research_version="v4.2",status="provisional_comparison_baseline_not_formal",
        D_masks_unchanged=True,D_total=int((overlay==1).sum()),HA_top10_count=10,
        HA_reverse_matches_covered=coverage,HA_reverse_cells=flip_total,
        P13_previous=previous_p13,P13_overlay=overlay_p13,
        pairing="exact binary assignment: maximize distinct HA-reverse matches, then reverse cells, then P13",
        actual_results_used=False,production_changed=False,research_only=True)
    (outdir/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    plans=[]
    for label,matrix in (("従来のD先行試作",old),("H/A上位10通りを重ねた案",overlay)):
        header="<tr><th>No.</th><th>対戦</th><th>候補00<br>購入外</th>"+"".join(f"<th>候補{j:02d}</th>" for j in range(1,11))+"</tr>"
        body=[]
        for i,match in f.sort_values("match_no").reset_index(drop=True).iterrows():
            values=[str(i+1),f"{match.home_team}－{match.away_team}",symbol[leader[i]]]+[symbol[matrix[j,i]] for j in range(10)]
            body.append("<tr>"+"".join(f"<td class='{('draw' if v=='D' else 'reverse' if j>=3 and v!=symbol[leader[i]] else '')}'>{html.escape(v)}</td>" for j,v in enumerate(values))+"</tr>")
        plans.append(f"<h2>{label}</h2><div class='scroll'><table>{header}{''.join(body)}</table></div>")
    note=(f"D配置は完全一致。H/A逆側が残る試合は0→{coverage}試合、逆側セルは0→{flip_total}セル。"
          f"10口P13は{100*previous_p13:.8f}%→{100*overlay_p13:.8f}%（保存モデル上）。"
          "H/A上位10通りは、Dを除いた条件付き確率で選んだ。"
          "このH/A上位10通り自体が逆側を含むのは7/13試合だけで、残る6試合の逆側は未対応。"
          "D強度・P(D)のD配置はそのまま保持。実的中の優位は未検証。")
    (outdir/"report.md").write_text("# BuyPlan v4.2 暫定比較基準\n\n正式版・購入推奨ではない。今後の研究案を同じ入力で比較するための到達点。\n\n"+note+"\n",encoding="utf-8")
    page=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>BuyPlan v4.2 暫定比較基準</title>"
          "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.draw{background:#ffe299;font-weight:700}.reverse{background:#c6e8ff;font-weight:700}.note{background:#fff0d3;padding:12px}</style>"
          "<h1>第1672回 BuyPlan v4.2 暫定比較基準</h1>"
          f"<p class='note'>正式版・購入推奨ではなく、今後の比較に使う到達点です。{html.escape(note)} 正式購入券は変更していません。</p>"
          +"".join(plans)+"<h2>試合別のH/A逆側</h2>"+pd.DataFrame(diag).to_html(index=False)
          +"<h2>H/A上位10通りの対応</h2>"+pd.read_csv(outdir/"HA_top10_assignment.csv").to_html(index=False)
          +"<p><a href='portfolio_comparison.csv'>両配券CSV</a>／<a href='audit.json'>監査</a></p></html>")
    (outdir/"index.html").write_text(page,encoding="utf-8")
    return audit


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--forecast",type=Path,required=True)
    parser.add_argument("--existing",type=Path,required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(write(args.forecast,args.existing,args.outdir),ensure_ascii=False))
