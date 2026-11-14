"""Exact opportunity cost of D-containing tickets under archived prematch forecasts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,sha256
from forecast_registry import selected_matches,DEFAULT_ROOT
from analyze_score_structure_phase4b_stage1 import ROUNDS,N,POWERS,SYMBOLS,universe,top_ten,decode,encode
from analyze_score_structure_phase4b_stage2 import portfolio_probability

OUT=ROOT/'data/reports/score_structure_phase4b_stage3'
STAGE1=ROOT/'data/reports/score_structure_phase4b_stage1'
STAGE2=ROOT/'data/reports/score_structure_phase4b_stage2'


def top_category(mass,eligible):
    codes=np.flatnonzero(eligible)
    if len(codes)<10:raise ValueError('fewer than ten eligible combinations')
    cutoff=np.partition(mass[codes],-10)[-10]
    tied=codes[mass[codes]>=cutoff]
    return tied[np.lexsort((tied,-mass[tied]))[:10]]


def constrained(mass,has_draw,k):
    """Exact best ten with at least k D-containing tickets; ties by base-3 code."""
    no=top_category(mass,~has_draw);yes=top_category(mass,has_draw)
    candidates=[]
    for n in range(k,11):
        codes=np.concatenate((no[:10-n],yes[:n]))
        candidates.append((float(mass[codes].sum()),n,codes))
    best=max(candidates,key=lambda x:(x[0],-x[1]))
    codes=best[2]
    return codes[np.lexsort((codes,-mass[codes]))],best[1]


def draw_distribution(prob):
    dist=np.array([1.])
    for p in prob[:,1]:dist=np.convolve(dist,[1-p,p])
    if len(dist)!=14 or not np.isclose(dist.sum(),1):raise AssertionError('D count distribution invalid')
    return dist


def event_mass(matrix,p):
    values=np.prod(p[np.arange(13)[None,:],matrix],axis=1)
    zero=(matrix==1).sum(axis=1)==0
    return float(values[zero].sum()),float(values[~zero].sum())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'BUYPLAN_V4_0.md',STAGE1/'report.md',STAGE1/'ticket_probabilities.csv',STAGE1/'audit.json',
        STAGE2/'report.md',STAGE2/'scenarios.csv',DEFAULT_ROOT/'forecasts.jsonl',DEFAULT_ROOT/'results.jsonl']
    hashes={str(p):sha256(p) for p in sources}
    reg=selected_matches(DEFAULT_ROOT)
    if len(reg)!=78 or not reg.provenance_class.eq('A').all() or reg.fixture_id.duplicated().any():
        raise ValueError('only original prematch A rows accepted')
    original=pd.read_csv(STAGE1/'ticket_probabilities.csv')
    scenario_data=pd.read_csv(STAGE2/'scenarios.csv')
    first=[];curve=[];ticket_rows=[];distribution=[];mass_rows=[];scenario_rows=[]
    for rid in ROUNDS:
        g=reg[reg.round_id.eq(rid)].sort_values('match_no')
        p=g[['final_prob_home','final_prob_draw','final_prob_away']].to_numpy(float)
        if len(g)!=13 or not np.allclose(p.sum(axis=1),1,atol=1e-10):raise ValueError(rid)
        mass=universe(p)
        codes=np.arange(N,dtype=np.int64)
        has_draw=np.zeros(N,dtype=bool)
        for power in POWERS:has_draw|=((codes//power)%3)==1
        first_d=int(top_category(mass,has_draw)[0]);top=top_ten(mass);tenth=int(top[9])
        rank=int(np.sum(mass>mass[first_d])+np.sum((mass==mass[first_d])&(codes<first_d))+1)
        chosen=decode(np.array([first_d]))[0]
        first.append(dict(round=rid,rank=rank,first_D_ticket=''.join(SYMBOLS[chosen]),
            D_match_numbers=','.join(str(i+1) for i,v in enumerate(chosen) if v==1),
            first_D_probability=float(mass[first_d]),tenth_ticket=''.join(SYMBOLS[decode(np.array([tenth]))[0]]),
            tenth_probability=float(mass[tenth]),probability_gap=float(mass[tenth]-mass[first_d]),
            first_D_to_tenth_ratio=float(mass[first_d]/mass[tenth])))
        dist=draw_distribution(p)
        for k,prob in enumerate(dist):
            distribution.append(dict(round=rid,D_count=k,probability=float(prob),
                grouped='0' if k==0 else str(k) if k<=4 else '5以上',
                actual_D=int(np.sum(g.actual_home_score.to_numpy()==g.actual_away_score.to_numpy()))))
        plans={}
        optimum=float(mass[top].sum())
        for k in range(11):
            selected,n=constrained(mass,has_draw,k)
            selected_mass=float(mass[selected].sum())
            if selected_mass>optimum+1e-15:raise AssertionError('constraint beats unconstrained optimum')
            matrix=decode(selected)
            plans[f'最低{k}口D入り']=matrix
            no_d,d_yes=event_mass(matrix,p)
            curve.append(dict(round=rid,min_D_tickets=k,actual_D_tickets=n,
                probability=selected_mass,probability_percent=100*selected_mass,
                optimum_probability=optimum,attainment_percent=100*selected_mass/optimum,
                loss_probability=optimum-selected_mass,loss_percentage_points=100*(optimum-selected_mass),
                displaced_undrawn_tickets=','.join(''.join(SYMBOLS[x]) for x in decode(top) if int(encode(x[None,:])[0]) not in set(selected)),
                added_D_tickets=','.join(''.join(SYMBOLS[x]) for x in matrix if np.any(x==1))))
            for ix,row in enumerate(matrix):
                prob=float(mass[selected[ix]])
                ticket_rows.append(dict(round=rid,min_D_tickets=k,ticket_no=ix+1,symbols=''.join(SYMBOLS[row]),
                    D_count=int(np.sum(row==1)),D_match_numbers=','.join(str(i+1) for i,v in enumerate(row) if v==1),
                    probability=prob))
            mass_rows.append(dict(round=rid,method=f'最低{k}口D入り',no_D_event_mass=no_d,D_event_mass=d_yes,
                total_probability=no_d+d_yes,P_no_D_event=float(dist[0]),P_D_event=float(1-dist[0]),
                conditional_exact13_given_no_D=no_d/dist[0],
                conditional_exact13_given_D=d_yes/(1-dist[0])))
        shadow=original[(original['round']==rid)&(original.method=='保存Shadow C')&(original.evaluation=='final')]
        if not shadow.empty:
            shadow_matrix=np.array([[{'H':0,'D':1,'A':2}[v] for v in seq] for seq in shadow.symbols],dtype=np.uint8)
            if shadow_matrix.shape!=(10,13):raise ValueError('shadow ticket shape')
            plans['保存Shadow C']=shadow_matrix
            no_d,d_yes=event_mass(shadow_matrix,p)
            mass_rows.append(dict(round=rid,method='保存Shadow C',no_D_event_mass=no_d,D_event_mass=d_yes,
                total_probability=no_d+d_yes,P_no_D_event=float(dist[0]),P_D_event=float(1-dist[0]),
                conditional_exact13_given_no_D=no_d/dist[0],
                conditional_exact13_given_D=d_yes/(1-dist[0])))
        ss=scenario_data[scenario_data['round']==rid]
        for s in ss.itertuples():
            q=np.array(json.loads(s.probabilities_json),dtype=float)
            if q.shape!=(13,3) or np.any(q<0) or not np.allclose(q.sum(axis=1),1,atol=1e-10):raise ValueError('bad stored scenario')
            baseline=portfolio_probability(q,plans['最低0口D入り'])
            for name,arr in plans.items():
                value=portfolio_probability(q,arr)
                scenario_rows.append(dict(scenario_id=s.scenario_id,round=rid,kind=s.kind,delta_pp=s.delta_pp,trial=s.trial,
                    method=name,P_probability=portfolio_probability(p,arr),Q_probability=value,
                    Q_percent=100*value,Q_minus_fixed_optimum=value-baseline,
                    beats_fixed_optimum=bool(value>baseline+1e-18)))
    results={'first_D_ticket.csv':first,'constrained_portfolios.csv':curve,'constrained_tickets.csv':ticket_rows,
        'draw_count_distribution.csv':distribution,'event_mass_decomposition.csv':mass_rows,'scenario_comparison.csv':scenario_rows}
    for filename,rows in results.items():pd.DataFrame(rows).to_csv(OUT/filename,index=False,encoding='utf-8-sig')
    if any(sha256(Path(path))!=value for path,value in hashes.items()):raise AssertionError('input changed')
    audit=dict(input_hashes=hashes,rounds=list(ROUNDS),matches=78,combinations=N,
        constraint='at least k D-containing distinct tickets, k=0..10',
        proof='best m D-containing and best 10-m D-free, maximize over m>=k',
        tie_rule='probability descending, then H/D/A lexicographic',
        all_constrained_below_unconstrained=True,archived_scenarios_reused=True,
        scenario_portfolios_fixed=True,result_used_for_selection=False,
        discarded_v30_used=False,fixed_HDA_counts_used=False,production_changed=False,originals_unchanged=True)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    report(*(pd.DataFrame(results[k]) for k in results),audit)
    print(json.dumps(dict(rounds=len(ROUNDS),scenario_comparisons=len(scenario_rows),first_D_ranks=[r['rank'] for r in first]),ensure_ascii=False))


def report(first,curve,tickets,distribution,mass,scenario,audit):
    p0=distribution[distribution.D_count.eq(0)][['round','probability','actual_D']]
    key=curve[curve.min_D_tickets.isin([0,1,2])][['round','min_D_tickets','actual_D_tickets','probability_percent','attainment_percent','loss_percentage_points']]
    reversals=scenario[(scenario.method.isin(['最低1口D入り','最低2口D入り','保存Shadow C']))&scenario.kind.ne('基準')]
    rev_summary=reversals.groupby(['method','kind','delta_pp']).agg(cases=('beats_fixed_optimum','size'),reversals=('beats_fixed_optimum','sum'),
        max_advantage=('Q_minus_fixed_optimum','max')).reset_index()
    rev_cases=reversals[reversals.beats_fixed_optimum]
    table=lambda d:d.to_html(index=False,escape=True,float_format=lambda x:f'{x:.7g}')
    draw_summary=distribution.assign(bucket=lambda x:np.where(x.D_count>=5,'5以上',x.D_count.astype(str))).groupby(['round','bucket'],as_index=False).probability.sum()
    order=['0','1','2','3','4','5以上'];draw_summary['bucket']=pd.Categorical(draw_summary.bucket,order,ordered=True)
    draw_summary=draw_summary.sort_values(['round','bucket'])
    def line_svg(g,rid):
        x=g.min_D_tickets.to_numpy(float);y=g.attainment_percent.to_numpy(float)
        pts=' '.join(f'{55+v*36:.1f},{190-(w-0)*1.5:.1f}' for v,w in zip(x,y))
        return f'<svg viewBox="0 0 460 230" width="460" height="230"><rect width="460" height="230" fill="white"/><text x="55" y="20">{rid} D入り最低口数と最適値達成率</text><path d="M55 40V190H420" fill="none" stroke="#777"/><polyline points="{pts}" fill="none" stroke="#236aa0" stroke-width="2"/><text x="55" y="210">0</text><text x="392" y="210">10口</text><text x="5" y="45">100%</text></svg>'
    figs=''.join(line_svg(g,rid) for rid,g in curve.groupby('round'))
    compact=key[key.min_D_tickets.isin([1,2])].pivot(index='round',columns='min_D_tickets',values='attainment_percent').reset_index().rename(columns={1:'最低1口達成率%',2:'最低2口達成率%'})
    text=f'''# Phase 4-B第3段階：D入り券の確率的代償と選択余地\n\n参照：`BUYPLAN_V4_0.md` と第1・第2段階。旧v3.0の上位4選出・固定ダブル、v2.4のH/D/A枚数固定は使用していない。試合前原本Aのfinal確率を使い、結果は事後説明のみに使用。\n\n## 要点\n\nD入り最上位の全組み合わせ順位は開催順に {', '.join(str(int(v)) for v in first['rank'])} 位。Dなし開催の予測確率は {100*p0.probability.min():.2f}〜{100*p0.probability.max():.2f}%。最低1口D入りの最適値達成率は {compact['最低1口達成率%'].min():.3f}〜{compact['最低1口達成率%'].max():.3f}%、最低2口は {compact['最低2口達成率%'].min():.3f}〜{compact['最低2口達成率%'].max():.3f}%。既存の1・3・5ポイント仮定シナリオで、最低1口D入り券が固定無制約最適券を上回るケースは {int(reversals[reversals.method.eq('最低1口D入り')].beats_fixed_optimum.sum())} 件、最低2口は {int(reversals[reversals.method.eq('最低2口D入り')].beats_fixed_optimum.sum())} 件。これらは試験条件の件数であり、現実の発生率ではない。\n\n## 最も確率の高いD入り券\n\n{first.to_string(index=False)}\n\n順位は全3^13={N:,}通りでの厳密順位。1口のD入りとは試合数を指定したものではなく、13試合の組み合わせにDが1つ以上あることを指す。\n\n## D入り最低口数という仮想制約の機会損失\n\n{key.to_string(index=False)}\n\n最低k口という条件下で、D入り上位m口とDなし上位(10-m)口をm>=kで比較した厳密解。全k=0..10を `constrained_portfolios.csv` と `constrained_tickets.csv` に保存。kを正式配券ルールにする提案ではない。損失は同一final分布での全的中確率差。\n\n## 開催全体でDが発生する確率\n\n{draw_summary.to_string(index=False)}\n\nDなし開催の予測確率と、Dなしの個別券が上位に来ることは別問題。過去にDなし開催が約50回に1回という集計は参考にとどめ、選出条件には使っていない。各券群のDなし結果・Dあり結果への確率質量と条件付き全的中確率は `event_mass_decomposition.csv`。\n\n## 既存の仮定シナリオ下での固定券比較\n\n{rev_summary.to_string(index=False)}\n\n逆転した個別ケースは{len(rev_cases)}件。各ケースでは10口を保存final確率Pで選んで固定し、第2段階の同一変動後Qで比較。シナリオは1・3・5ポイントの仮定であり実測誤差の発生頻度を示さない。第2段階のShadow C逆転は、最大20ポイントまで別途伸ばした同方向の格子探索で起きたもので、今回の小変動シナリオとは範囲が異なる。\n\n## 判断\n\n固定Pの下では無制約上位10口が最適。D入り券を採用するには計算上の機会損失があり、その大きさは開催ごとに異なる。予測分布Qが変わればその順位も変化し得るが、仮定したQの好成績から現実の優位は言えない。実際の予測誤差の幅・方向・試合間依存を、同一モデルの試合前原本を蓄積して後続期間で検証する必要がある。D入り口数は未決定で、v4.0の正式方式は変更しない。\n\n[最上位D券](first_D_ticket.csv) ／ [最低口数別](constrained_portfolios.csv) ／ [10口の全構成](constrained_tickets.csv) ／ [D数分布](draw_count_distribution.csv) ／ [確率質量分解](event_mass_decomposition.csv) ／ [変動後比較](scenario_comparison.csv) ／ [監査](audit.json)\n'''
    (OUT/'report.md').write_text(text)
    html=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 4-B第3段階</title><style>body{{font:16px/1.6 system-ui;margin:28px;color:#203047}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{border:1px solid #ccd4df;padding:5px}}th{{background:#edf3f8}}.scroll{{overflow:auto}}svg{{margin:8px;border:1px solid #dae1e8}}</style><h1>v4.0 Phase 4-B第3段階：D入り券の確率的代償</h1><p>試合前原本Aのfinal確率、試合間独立、1,594,323組の厳密比較。D入り口数を正式に決める分析ではない。</p><h2>最も確率の高いD入り券</h2><div class="scroll">{table(first)}</div><h2>D入りを最低1口・2口とした場合</h2><div class="scroll">{table(key)}</div><h2>最低D入り口数と最適値達成率</h2>{figs}<h2>開催全体のD発生数分布</h2><div class="scroll">{table(draw_summary)}</div><h2>Dなし開催の確率</h2><div class="scroll">{table(p0)}</div><h2>仮定した確率変動後の逆転</h2><div class="scroll">{table(rev_summary)}</div><p>逆転ケース{len(rev_cases)}件。1・3・5ポイントの仮定変動であり、実測された誤差ではない。Shadow Cが存在する開催だけを比較。</p><p><a href="report.md">詳細</a> ／ <a href="first_D_ticket.csv">D券順位</a> ／ <a href="constrained_portfolios.csv">口数別</a> ／ <a href="constrained_tickets.csv">全券</a> ／ <a href="draw_count_distribution.csv">D数分布</a> ／ <a href="event_mass_decomposition.csv">条件付き確率</a> ／ <a href="scenario_comparison.csv">変動比較</a> ／ <a href="audit.json">監査</a></p></html>'''
    (OUT/'index.html').write_text(html)

if __name__=='__main__':main()
