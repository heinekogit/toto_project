"""Three concrete v4.2 ticket portfolios under unchanged saved probabilities."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure_phase4b_stage1 import decode, top_ten, universe
from common import sha256

SYMBOLS=np.array(list('HDA'))
MODES=('P上位10口','D比例枚数を配置','D枚数を緩和')
SEED=420172


def value(matrix,p):
    if len({tuple(row) for row in matrix})!=10:return float('-inf')
    return float(np.prod(p[np.arange(13)[None,:],matrix],axis=1).sum())


def initial(p,counts,rng,base=None,required_counts=()):
    m=np.tile(np.where(p[:,0]>=p[:,2],0,2),(10,1)).astype(np.uint8) if base is None else base.copy()
    if required_counts:
        remaining=np.array(counts,int).copy()
        # Reserve one row for each supported D-count value. Prefer uncovered matches.
        covered=np.zeros(13,dtype=bool)
        for row,k in enumerate(sorted(required_counts,reverse=True)):
            for _ in range(k):
                eligible=np.flatnonzero((remaining>0)&(m[row]!=1))
                if not len(eligible):raise ValueError('D count support infeasible')
                priorities=remaining[eligible].astype(float)+2*(~covered[eligible])+.001*rng.random(len(eligible))
                i=int(eligible[np.argmax(priorities)])
                m[row,i]=1;remaining[i]-=1;covered[i]=True
        for i,count in enumerate(remaining):
            if count:
                chosen=rng.choice(np.arange(len(required_counts),10),int(count),replace=False)
                m[chosen,i]=1
    else:
        for i,count in enumerate(counts):
            selected=rng.choice(10,int(count),replace=False)
            m[:,i]=np.where(m[:,i]==1,np.where(p[i,0]>=p[i,2],0,2),m[:,i])
            m[selected,i]=1
    for _ in range(100):
        keys=[tuple(row) for row in m]
        if len(set(keys))==10:break
        duplicate=next(j for j in range(10) if keys[:j].count(keys[j]))
        options=[]
        others=set(keys[:duplicate]+keys[duplicate+1:])
        for i in range(13):
            if m[duplicate,i]==1:continue
            candidate=m[duplicate].copy()
            candidate[i]=2 if candidate[i]==0 else 0
            if tuple(candidate) not in others:
                options.append((abs(p[i,0]-p[i,2]),i,candidate))
        if not options:raise ValueError('cannot resolve duplicate ticket')
        _,_,chosen=min(options,key=lambda item:(item[0],item[1]))
        m[duplicate]=chosen
    if not np.isfinite(value(m,p)):raise ValueError('initial portfolio not distinct')
    return m


def optimize(p,lower,upper,starts=24,steps=16000,seed=SEED,required_counts=(),initial_counts=None,max_ticket_D=13):
    rng=np.random.default_rng(seed)
    best=None;best_score=-1.
    for start in range(starts):
        counts=(initial_counts if initial_counts is not None else lower) if start==0 else np.array([rng.integers(lo,hi+1) for lo,hi in zip(lower,upper)])
        if required_counts and sum(counts)<sum(required_counts):counts=np.array(initial_counts,int)
        m=initial(p,counts,rng,required_counts=required_counts)
        if np.any((m==1).sum(axis=1)>max_ticket_D):continue
        score=value(m,p)
        if score>best_score:best,best_score=m.copy(),score
        for step in range(steps):
            trial=m.copy();match=int(rng.integers(13));one=int(rng.integers(10))
            if rng.random()<.65:
                other=int(rng.integers(10))
                if one==other:continue
                trial[one,match],trial[other,match]=trial[other,match],trial[one,match]
            else:
                old=int(trial[one,match])
                options=[v for v in (0,1,2) if v!=old]
                new=int(rng.choice(options))
                d_count=int((trial[:,match]==1).sum())+(new==1)-(old==1)
                if d_count<lower[match] or d_count>upper[match]:continue
                trial[one,match]=new
            candidate=value(trial,p)
            if not np.isfinite(candidate):continue
            if required_counts and not set(required_counts).issubset(set((trial==1).sum(axis=1))):continue
            if np.any((trial==1).sum(axis=1)>max_ticket_D):continue
            temperature=1e-7*(1-step/steps)+1e-10
            if candidate>score or rng.random()<np.exp(min(0.,(candidate-score)/temperature)):
                m,score=trial,candidate
            if score>best_score+1e-18:
                best,best_score=m.copy(),score
    if best is None:raise ValueError('no feasible portfolio found')
    counts=(best==1).sum(axis=0)
    if np.any(counts<lower) or np.any(counts>upper):raise AssertionError('D constraint failed')
    if required_counts and not set(required_counts).issubset(set((best==1).sum(axis=1))):
        raise AssertionError('D-count support failed')
    if np.any((best==1).sum(axis=1)>max_ticket_D):raise AssertionError('D-count cap failed')
    return best,best_score


def compare(forecast):
    f=forecast.sort_values('match_no').reset_index(drop=True)
    if len(f)!=13 or f.match_no.tolist()!=list(range(1,14)):raise ValueError('13 ordered matches required')
    p=f[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-9):raise ValueError('invalid probabilities')
    universe_mass=universe(p)
    baseline=decode(top_ten(universe_mass))
    target=np.rint(10*p[:,1]).astype(int)
    distribution=np.array([1.])
    for draw in p[:,1]:distribution=np.convolve(distribution,[1-draw,draw])
    required_counts=tuple(range(1,7))
    max_ticket_D=int(np.searchsorted(np.cumsum(distribution),.99))
    exact,exact_value=optimize(p,target,target,required_counts=required_counts,initial_counts=target,max_ticket_D=max_ticket_D)
    relaxed,relaxed_value=optimize(p,np.ones(13,dtype=int),target,seed=SEED+1,
                                   required_counts=required_counts,initial_counts=target,max_ticket_D=max_ticket_D)
    portfolios=[baseline,exact,relaxed]
    reference={tuple(row) for row in baseline}
    pd0=float(np.prod(1-p[:,1]))
    summary=[];rows=[];per_match=[]
    for mode,m in zip(MODES,portfolios):
        score=value(m,p)
        symbols=[''.join(SYMBOLS[row]) for row in m]
        d_tickets=sum('D' in x for x in symbols)
        d_mass=sum(float(np.prod(p[np.arange(13),row])) for row in m if 1 in row)
        summary.append(dict(mode=mode,P_prob13=score,baseline_ratio=score/value(baseline,p),
                            P_mass_D_result=d_mass,P_mass_D0_result=score-d_mass,
                            conditional_P13_given_D=d_mass/(1-pd0),
                            D_tickets=d_tickets,total_D_cells=int((m==1).sum()),
                            D_count_per_ticket=';'.join(str(x.count('D')) for x in symbols),
                            changed_tickets=10-len(reference&{tuple(row) for row in m}),
                            all_unique=len(set(symbols))==10))
        for j,(row,s) in enumerate(zip(m,symbols),1):
            rows.append(dict(mode=mode,candidate=j,symbols=s,D_count=s.count('D'),
                             P_ticket_probability=float(np.prod(p[np.arange(13),row])),
                             **{f'match_{i+1:02d}':s[i] for i in range(13)}))
        for i,r in f.iterrows():
            per_match.append(dict(mode=mode,match_no=i+1,home_team=r.home_team,away_team=r.away_team,
                                  P_D=float(p[i,1]),target_D_cells=int(target[i]),
                                  actual_D_cells=int((m[:,i]==1).sum()),
                                  actual_H_cells=int((m[:,i]==0).sum()),actual_A_cells=int((m[:,i]==2).sum())))
    return pd.DataFrame(summary),pd.DataFrame(rows),pd.DataFrame(per_match),pd0,float(distribution[1:7].sum()),max_ticket_D


def write(forecast_path,outdir,round_id):
    f=pd.read_csv(forecast_path,encoding='utf-8-sig',low_memory=False)
    summary,tickets,matches,pd0,covered_count_mass,max_ticket_D=compare(f)
    outdir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(outdir/'summary.csv',index=False,encoding='utf-8-sig')
    tickets.to_csv(outdir/'tickets.csv',index=False,encoding='utf-8-sig')
    matches.to_csv(outdir/'match_allocations.csv',index=False,encoding='utf-8-sig')
    audit=dict(round=round_id,source_sha256=sha256(forecast_path),seed=SEED,
               fixed_D_rule='round(10*saved P(D)) per match, H/A counts free',
               relaxed_D_rule='per match D count 1..round(10*saved P(D)); H/A counts free',
               optimization='24 seeded simulated-annealing starts x 16000 iterations; approximate, not global optimum',
               stability='48 starts x 16000 iterations returned the same P13 value for both constrained modes on toto1672',
               D_count_support='at least one ticket each with 1,2,3,4,5,6 D (fixed/relaxed modes only)',
               D_count_1_to_6_model_mass=covered_count_mass,
               maximum_ticket_D=max_ticket_D,maximum_ticket_D_rule='99th percentile of model D-count distribution',
               probabilities_changed=False,actual_results_used=False,
               prediction_of_any_D=1-pd0,source='saved final Next Model probabilities')
    (outdir/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    f=f.sort_values('match_no').reset_index(drop=True)
    tables=[]
    for mode in MODES:
        group=tickets[tickets['mode'].eq(mode)]
        head='<tr><th>No.</th><th>対戦</th>'+''.join(f'<th>{n:02d}</th>' for n in range(1,11))+'</tr>'
        body=[]
        for i,r in f.iterrows():
            cells=[str(i+1),f'{r.home_team}－{r.away_team}']+[str(t[f'match_{i+1:02d}']) for _,t in group.iterrows()]
            body.append('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in cells)+'</tr>')
        tables.append(f'<h2>{html.escape(mode)}</h2><div class="scroll"><table>{head}{"".join(body)}</table></div>')
    display=summary.copy()
    for column in ('P_prob13','P_mass_D_result','P_mass_D0_result','conditional_P13_given_D'):
        display[column]=(100*display[column]).map(lambda x:f'{x:.9f}%')
    display['baseline_ratio']=(100*display.baseline_ratio).round(2)
    document=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>v4.2 三方式配券比較</title>"
              "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0 26px}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.note{background:#fff0d3;padding:12px}</style>"
              f'<h1>{html.escape(round_id)} v4.2 D配置を実行した10口比較</h1>'
              '<p class="note">保存H/D/A確率は変更しません。比例枚数方式と緩和方式はDを券に残す場合の比較実験で、予測精度の改善や購入推奨ではありません。探索は近似で大域最適を保証しません。</p>'
              f'<p>モデルが予測する「少なくとも1試合D」は{100*(1-pd0):.3f}%。D発生数1〜6の合計は{100*covered_count_mass:.3f}%。比例枚数方式と緩和方式には、この各D本数の券を少なくとも1口ずつ残し、1券のD上限をモデルの99%分位である{max_ticket_D}枚とする条件を課します。これは予測分布から決めた比較条件で、正式な配券仕様ではありません。Dが出た場合に13的中できる確率を別列で表示します。</p>'
              '<h2>10口全体の比較</h2><div class="scroll">'+display.to_html(index=False,escape=True)+'</div>'
              '<h2>各試合のD・H・A採用枚数</h2><div class="scroll">'+matches.to_html(index=False,escape=True)+'</div>'
              +''.join(tables)+'<p><a href="summary.csv">集計CSV</a>／<a href="tickets.csv">全券CSV</a>／<a href="match_allocations.csv">試合別枚数CSV</a>／<a href="audit.json">監査</a></p></html>')
    (outdir/'index.html').write_text(document,encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--forecast',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--round',required=True)
    args=parser.parse_args()
    print(write(args.forecast,args.outdir,args.round).to_string(index=False))


if __name__=='__main__':main()
