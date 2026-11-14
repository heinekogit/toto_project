"""Research-only v4.0 scenario portfolio prototype; never feeds production BuyPlan."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,sha256
from forecast_registry import selected_matches,DEFAULT_ROOT
from analyze_score_structure_phase4b_stage1 import N,POWERS,SYMBOLS,ROUNDS,universe,top_ten,decode,encode

OUT=ROOT/'data/reports/buyplan_v40_prototype'
PHASE3=ROOT/'data/reports/score_structure_phase3/scenario_structure.csv'
STAGE1=ROOT/'data/reports/score_structure_phase4b_stage1/ticket_probabilities.csv'
SCENARIOS=[('基準',0.)]+[(name,delta) for delta in (.1,.2) for name in
    ('H低下','H上昇','A低下','A上昇','双方低下','双方上昇','H低下・A上昇','H上昇・A低下')]
POOL_PER_STATE=100
METHODS=('保存正式版','数学上位10口','保存Shadow C','試作・共通状態平均','試作・独立状態平均','試作・最悪状態比率')


def final_state(g,scenario,delta):
    s=g[(g.scenario==scenario)&(g.delta==delta)].sort_values('match_no')
    if len(s)!=13 or s.match_no.tolist()!=list(range(1,14)):raise ValueError(f'missing phase3 {scenario} {delta}')
    raw=s[['raw_P_H','P_draw','raw_P_A']].to_numpy(float)
    mix=s.uncertainty_mix_reference.to_numpy(float)[:,None]
    final=(1-mix)*raw+mix/3
    if not np.allclose(final.sum(axis=1),1,atol=1e-10):raise AssertionError('state not normalized')
    return s,final


def ticket_probabilities(prob,tickets):
    return np.prod(prob[np.arange(13)[None,:],tickets],axis=1)


def portfolio(prob,tickets):
    return float(ticket_probabilities(prob,tickets).sum())


def greedy_relative_robust(pool,state_values,state_opt,baseline_codes):
    """Deterministic greedy+swap heuristic for max-min state-relative 13-hit mass."""
    relative=state_values/state_opt[:,None]
    selected=[];remaining=list(range(len(pool)))
    totals=np.zeros(relative.shape[0])
    for _ in range(10):
        options=totals[:,None]+relative[:,remaining]
        worst=options.min(axis=0);mean=options.mean(axis=0)
        candidate=min(range(len(remaining)),key=lambda ix:(-worst[ix],-mean[ix],pool[remaining[ix]]))
        choice=remaining.pop(candidate);selected.append(choice);totals+=relative[:,choice]
    baseline_indices=[int(np.where(pool==code)[0][0]) for code in baseline_codes]
    baseline_totals=relative[:,baseline_indices].sum(axis=1)
    if (float(baseline_totals.min()),float(baseline_totals.mean()))>(float(totals.min()),float(totals.mean())):
        selected=baseline_indices;totals=baseline_totals
    for _ in range(12):
        best=None;current=(totals.min(),totals.mean())
        chosen=set(selected)
        for pos,old in enumerate(selected):
            options=np.flatnonzero(~np.isin(np.arange(len(pool)),list(chosen)))
            candidates=totals[:,None]-relative[:,old,None]+relative[:,options]
            worst=candidates.min(axis=0);mean=candidates.mean(axis=0)
            for k,new in enumerate(options):
                objective=(float(worst[k]),float(mean[k]))
                if objective>current and (best is None or objective>best[0] or (objective==best[0] and pool[new]<pool[best[2]])):
                    best=(objective,pos,int(new),candidates[:,k])
        if best is None:break
        _,pos,new,totals=best;selected[pos]=new
    codes=pool[selected]
    return codes[np.lexsort((codes,-state_values[0,selected]))]


def archived_tickets(frame,rid):
    f=frame[(frame['round']==rid)&(frame.evaluation=='final')]
    plans={}
    for method,d in f.groupby('method',sort=False):
        if method.startswith('保存正式版'):label='保存正式版'
        elif method=='保存Shadow C':label=method
        else:continue
        tickets=np.array([[{'H':0,'D':1,'A':2}[c] for c in item] for item in d.symbols],dtype=np.uint8)
        if tickets.shape!=(10,13) or len(set(encode(tickets)))!=10:raise ValueError(f'{rid}: invalid archived {method}')
        plans[label]=tickets
    return plans


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'BUYPLAN_V4_0.md',PHASE3,STAGE1,DEFAULT_ROOT/'forecasts.jsonl',DEFAULT_ROOT/'results.jsonl']
    source_hashes={str(p):sha256(p) for p in sources}
    forecast=selected_matches(DEFAULT_ROOT)
    if len(forecast)!=78 or not forecast.provenance_class.eq('A').all() or forecast.fixture_id.duplicated().any():
        raise ValueError('only 78 original prematch A matches')
    phases=pd.read_csv(PHASE3,encoding='utf-8-sig')
    history=pd.read_csv(STAGE1,encoding='utf-8-sig')
    state_rows=[];feature_rows=[];ticket_rows=[];grid_rows=[];comparison_rows=[];change_rows=[];matrix_rows=[];audit_rounds=[]
    for rid in ROUNDS:
        g=forecast[forecast.round_id.eq(rid)].sort_values('match_no')
        if len(g)!=13:raise ValueError(rid)
        pg=phases[phases['round']==rid]
        states=[];distributions=[];state_pools=[];state_opt=[]
        for name,delta in SCENARIOS:
            detail,p=final_state(pg,name,delta)
            states.append((name,delta,p,detail))
            mass=universe(p);distributions.append(mass)
            top=top_ten(mass);state_opt.append(float(mass[top].sum()))
            cutoff=np.partition(mass,-POOL_PER_STATE)[-POOL_PER_STATE]
            eligible=np.flatnonzero(mass>=cutoff)
            top100=eligible[np.lexsort((eligible,-mass[eligible]))[:POOL_PER_STATE]]
            state_pools.extend(top100.tolist())
            state_rows.append(dict(round=rid,state=f'{name}_{delta:.1f}',scenario=name,delta=delta,
                assumption='等重みは計算仮定、誤差の実測頻度ではない',state_optimal_13_probability=state_opt[-1],
                mix_min=float(detail.uncertainty_mix_reference.min()),mix_max=float(detail.uncertainty_mix_reference.max())))
        baseline=g[['final_prob_home','final_prob_draw','final_prob_away']].to_numpy(float)
        if not np.allclose(states[0][2],baseline,atol=1e-10):raise AssertionError(f'{rid}: phase3 baseline/final mismatch')
        global_mean=np.mean(distributions,axis=0)
        independent_mean=np.mean(np.stack([x[2] for x in states]),axis=0)
        independent_mass=universe(independent_mean)
        base_top=top_ten(distributions[0]);global_top=top_ten(global_mean);independent_top=top_ten(independent_mass)
        state_pools.extend(base_top.tolist()+global_top.tolist()+independent_top.tolist())
        pool=np.array(sorted(set(state_pools)),dtype=np.int64)
        state_values=np.stack([mass[pool] for mass in distributions])
        robust=greedy_relative_robust(pool,state_values,np.array(state_opt),base_top)
        plans=archived_tickets(history,rid)
        plans.update({'数学上位10口':decode(base_top),'試作・共通状態平均':decode(global_top),
                      '試作・独立状態平均':decode(independent_top),'試作・最悪状態比率':decode(robust)})
        truth=np.where(g.actual_home_score.to_numpy(int)>g.actual_away_score.to_numpy(int),0,
            np.where(g.actual_home_score.to_numpy(int)==g.actual_away_score.to_numpy(int),1,2)).astype(np.uint8)
        for i,row in enumerate(g.itertuples()):
            per=np.stack([p[i] for _,_,p,_ in states])
            rawbase=states[0][3].iloc[i]
            changed=int(np.sum(np.argmax(per,axis=1)!=np.argmax(per[0])))
            feature_rows.append(dict(round=rid,match_no=i+1,home_team=row.home_team,away_team=row.away_team,
                lambda_home=row.lambda_home,lambda_away=row.lambda_away,uncertainty_mix=row.uncertainty_mix,
                baseline_P_H=baseline[i,0],baseline_P_D=baseline[i,1],baseline_P_A=baseline[i,2],
                P_H_min=float(per[:,0].min()),P_H_max=float(per[:,0].max()),
                P_D_min=float(per[:,1].min()),P_D_max=float(per[:,1].max()),
                P_A_min=float(per[:,2].min()),P_A_max=float(per[:,2].max()),
                top_symbol_changes=changed,baseline_P00=rawbase.P_0_0,baseline_P11=rawbase.P_1_1,
                baseline_P22plus=rawbase.P_draw_2plus,baseline_P_margin1=rawbase.P_margin_1,
                draw_structure_range=float(max(x[3].iloc[i].P_draw_2plus for x in states)-min(x[3].iloc[i].P_draw_2plus for x in states)),
                actual_symbol=SYMBOLS[truth[i]]))
        for method,tickets in plans.items():
            for match_ix,match in enumerate(g.itertuples()):
                matrix_rows.append(dict(round=rid,method=method,match_no=match_ix+1,home_team=match.home_team,away_team=match.away_team,
                    **{f'ticket_{j+1:02}':SYMBOLS[v] for j,v in enumerate(tickets[:,match_ix])}))
            codes=encode(tickets)
            if tickets.shape!=(10,13) or len(set(codes))!=10:raise AssertionError('not ten distinct')
            state_prob=np.array([float(mass[codes].sum()) for mass in distributions])
            p0=float(state_prob[0]);hits=np.sum(tickets==truth[None,:],axis=1)
            grid_rows.append(dict(round=rid,method=method,model_signature=g.model_signature.iloc[0],
                baseline_probability=p0,global_mean_probability=float(state_prob.mean()),
                independent_mean_probability=portfolio(independent_mean,tickets),
                worst_state_probability=float(state_prob.min()),
                worst_state_relative_to_state_optimum=float(np.min(state_prob/np.array(state_opt))),
                best_hits=int(hits.max()),exact13=bool(np.any(hits==13)),
                overlap_with_baseline_optimum=len(set(codes)&set(base_top)),
                D_tickets=int(np.sum(np.any(tickets==1,axis=1))),D_cells=int(np.sum(tickets==1))))
            for j,item in enumerate(tickets):
                individual=ticket_probabilities(baseline,item[None,:])[0]
                per_state=np.array([ticket_probabilities(p,item[None,:])[0] for _,_,p,_ in states])
                ticket_rows.append(dict(round=rid,method=method,ticket_no=j+1,symbols=''.join(SYMBOLS[item]),
                    baseline_probability=float(individual),mean_state_probability=float(per_state.mean()),
                    min_state_probability=float(per_state.min()),max_state_probability=float(per_state.max()),
                    best_state=f'{SCENARIOS[int(np.argmax(per_state))][0]}_{SCENARIOS[int(np.argmax(per_state))][1]:.1f}',worst_state=f'{SCENARIOS[int(np.argmin(per_state))][0]}_{SCENARIOS[int(np.argmin(per_state))][1]:.1f}',
                    D_count=int(np.sum(item==1)),hits=int(hits[j]),
                    **{f'match_{i+1:02}':SYMBOLS[x] for i,x in enumerate(item)}))
            for k,(name,delta,p,_) in enumerate(states):
                comparison_rows.append(dict(round=rid,method=method,state=f'{name}_{delta:.1f}',scenario=name,delta=delta,
                    total_13_probability=float(state_prob[k]),state_optimum=state_opt[k],
                    ratio_to_state_optimum=float(state_prob[k]/state_opt[k]),
                    baseline_13_probability=p0))
            if method.startswith('試作'):
                for j,item in enumerate(tickets):
                    if int(codes[j]) in set(base_top):continue
                    differing=np.flatnonzero(item!=decode(base_top)[0])
                    feature=g.iloc[differing]
                    feature_labels=[]
                    for ix in differing:
                        fr=feature_rows[-13+int(ix)]
                        feature_labels.append(f'{ix+1}:{SYMBOLS[item[ix]]}(順位反転状態{fr["top_symbol_changes"]},D幅{fr["P_D_max"]-fr["P_D_min"]:.3f})')
                    change_rows.append(dict(round=rid,method=method,ticket_no=j+1,symbols=''.join(SYMBOLS[item]),
                        baseline_probability=float(distributions[0][codes[j]]),
                        mean_state_probability=float(global_mean[codes[j]]),
                        anchor_differences='; '.join(feature_labels),
                        note='基準最強券との違い。個別特徴の因果的寄与ではない'))
        audit_rounds.append(dict(round=rid,states=len(states),candidate_pool=len(pool),mix_applied=bool(np.any(g.uncertainty_mix.to_numpy(float)>0)),
            baseline_optimal_total=float(distributions[0][base_top].sum()),
            global_mean_optimal_total=float(global_mean[global_top].sum()),
            independent_mean_optimal_total=float(independent_mass[independent_top].sum())))
    frames={'states.csv':state_rows,'match_stability.csv':feature_rows,'tickets.csv':ticket_rows,
            'portfolio_summary.csv':grid_rows,'state_evaluation.csv':comparison_rows,'changed_tickets.csv':change_rows,
            'round_audit.csv':audit_rounds,'allocation_matrices.csv':matrix_rows}
    for path,rows in frames.items():pd.DataFrame(rows).to_csv(OUT/path,index=False,encoding='utf-8-sig')
    if any(sha256(Path(path))!=digest for path,digest in source_hashes.items()):raise AssertionError('source changed')
    audit=dict(source_hashes=source_hashes,rounds=list(ROUNDS),matches=78,states_per_round=17,
        scenario_deltas=[0,.1,.2],scenario_weights='equal within compared scenario set; hypothetical, not empirical frequency',
        independent_aggregation='per-match arithmetic mean HDA, then product across matches',
        common_aggregation='mean of 17 joint 13-match probability distributions',
        robust_algorithm='deterministic greedy, compare baseline top10 as second start, plus 12 swap sweeps on union of each state top100; local heuristic, no global optimum claim',
        discarded_v30_used=False,fixed_HDA_counts_used=False,actual_results_used_for_selection=False,
        originals_unchanged=True,official_buyplan_unchanged=True)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    make_report(*(pd.DataFrame(frames[x]) for x in ('portfolio_summary.csv','tickets.csv','match_stability.csv','state_evaluation.csv','changed_tickets.csv','round_audit.csv','allocation_matrices.csv')),audit)
    print(json.dumps(dict(rounds=len(ROUNDS),methods=sorted(set(x['method'] for x in grid_rows)),changed_tickets=len(change_rows)),ensure_ascii=False))


def summarize_findings(summary):
    p=summary.pivot(index='round',columns='method',values='overlap_with_baseline_optimum')
    common=[int(x) for x in p['試作・共通状態平均']]
    robust=[int(x) for x in p['試作・最悪状態比率']]
    d=summary[summary.method.eq('試作・最悪状態比率')]
    return (f'数学上位10口との共通券数は、共通状態平均が{common}口、最悪状態比率が{robust}口。'
        f'独立状態平均は全6開催で10口すべて一致した。最悪状態比率のD入り券数は{d.D_tickets.tolist()}口。'
        '試作・保存正式版・保存Shadow Cのいずれも、6開催の実結果で13的中はなかった。'
        '最悪状態比率方式は定義した17状態内の最低達成率を基準券以上にしたが、現実の予測誤差への優位を示すものではない。')

def robustness_svg(summary):
    methods=['数学上位10口','試作・共通状態平均','試作・最悪状態比率']
    colors=['#708090','#d08735','#2876a8']
    rects=[];labels=[]
    for i,(rid,g) in enumerate(summary.groupby('round',sort=False)):
        x=65+i*120
        labels.append(f'<text x="{x+27}" y="260" font-size="11" text-anchor="middle">{rid[-4:]}</text>')
        for j,method in enumerate(methods):
            val=float(g[g.method==method].worst_state_relative_to_state_optimum.iloc[0])*100
            height=val*3.5
            rects.append(f'<rect x="{x+j*22}" y="{235-height:.1f}" width="18" height="{height:.1f}" fill="{colors[j]}"/><title>{rid} {method}: {val:.1f}%</title>')
    legend=' '.join(f'<text x="{75+i*220}" y="20" fill="{colors[i]}" font-size="13">■ {name}</text>' for i,name in enumerate(methods))
    return f'<svg viewBox="0 0 850 280" width="850" role="img" aria-label="最悪状態の達成率比較"><rect width="850" height="280" fill="white"/>{legend}<path d="M50 35V235H800" stroke="#888" fill="none"/>{"".join(rects)}{"".join(labels)}</svg>'


def make_report(summary,tickets,stability,evaluation,changes,rounds,matrices,audit):
    compact=summary[['round','method','baseline_probability','global_mean_probability','independent_mean_probability',
                     'worst_state_relative_to_state_optimum','best_hits','exact13','overlap_with_baseline_optimum','D_tickets']]
    state_pick=evaluation[evaluation.scenario.isin(['基準','H低下・A上昇','H上昇・A低下'])][['round','method','state','total_13_probability','ratio_to_state_optimum']]
    def tab(d):return d.to_html(index=False,escape=True,float_format=lambda x:f'{x:.7g}')
    table=tickets[tickets.method.str.startswith('試作')][['round','method','ticket_no','symbols','baseline_probability','mean_state_probability','D_count','hits']]
    matrix_html=''.join(f'<h3>{rid} {method}</h3><div class="scroll">{tab(group.drop(columns=["round","method"]))}</div>' for (rid,method),group in matrices.groupby(['round','method'],sort=False))
    outcome=summarize_findings(summary)
    text=f'''# BuyPlan v4.0 独立試作：λ変動・分岐・10口一括選択\n\n**研究用シャドーであり、正式版・購入フローには未接続。** `BUYPLAN_V4_0.md` の13的中目的を守り、旧v2枚数固定・廃棄済みv3.0の4ダブルを使わない。6開催の試合前原本Aから、実結果を読まずに候補・券を決定した。結果は最後の採点だけに使用。\n\n## 計算構造\n\nPhase 1・2の11×11独立Poisson得点行列と同点/1点差/2点差以上の構造、Phase 3の保存λに対する8方向×0.1/0.2点の行列を再利用した。各シナリオのraw H/D/Aを集計し、保存された試合別mixで `(1-mix)*raw+mix/3` としてfinal相当を得た。基準状態が保存finalと全78試合で一致することを確認。Phase 1・2構造は `match_stability.csv` の同点帯・1点差とD変動幅の診断に保持し、P(D)への別加点には使用しない。\n\n共通状態平均は17個の13試合同方向シナリオの組み合わせ確率を等重み平均し、全3^13組から上位10を取る。独立状態平均は各試合で17状態のfinal H/D/Aを等重み平均し、その積の上位10を取る。最悪状態比率は各状態の理論最適10口に対する比率の最小値を高める貪欲探索＋交換探索。基準の数学上位10口も初期解として比較し、悪い初期解を避ける。候補プールは各状態と両平均の確率上位100組の和集合。最悪状態比率方式は局所解であり、大域最適を保証しない。シナリオの等重みと変動幅は**仮定**で、実際のλ誤差の頻度を示さない。\n\n## 6開催の10口全体\n\n{outcome}\n\n{compact.to_string(index=False)}\n\n実結果との的中数は補助診断。各試作の券は `tickets.csv` の13試合列に全口記録し、`allocation_matrices.csv` に試合行×10口の配置表を保存。数学上位10口と異なる券は `changed_tickets.csv` に列挙した。個別変更理由欄は基準券との差と、順位が変わるシナリオ数・D確率幅を示す診断であり、個々の特徴の因果的寄与ではない。\n\n## シナリオ別の比較例\n\n{state_pick.to_string(index=False)}\n\n全17状態×全方式の10口13的中確率は `state_evaluation.csv`。どの試作も1,000円・異なる10口。保存Shadow Cは存在する開催のみ比較。\n\n## 判断\n\n固定final確率が真なら数学上位10口が最適。試作で異なる券を選ぶのは、仮定したλ変動状態での期待値または最悪状態比率を評価したため。過去6開催の実結果でパラメータを選んでおらず、13的中の改善を証明していない。共通変動は複数試合の誤差相関を仮定するが、実証されていない。独立状態も実際の試合別誤差分布を推定したものではない。正式化には同一モデル版の試合前原本蓄積、λ誤差・相関の推定、後続開催での予測分布・13的中性能の評価が必要。v4.0の正式アルゴリズムは未決定。\n\n[配券全口](tickets.csv) ／ [13試合×10口](allocation_matrices.csv) ／ [10口比較](portfolio_summary.csv) ／ [試合別安定性](match_stability.csv) ／ [変更券](changed_tickets.csv) ／ [全状態評価](state_evaluation.csv) ／ [状態監査](states.csv) ／ [監査](audit.json)\n'''
    (OUT/'report.md').write_text(text)
    chart=robustness_svg(summary)
    html=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>BuyPlan v4.0 試作</title><style>body{{font:16px/1.65 system-ui;margin:28px;color:#203047}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{border:1px solid #ccd4df;padding:5px}}th{{background:#edf3f8}}.scroll{{overflow:auto}}.cards{{display:flex;flex-wrap:wrap;gap:8px}}.card{{border:1px solid #cbd6e0;padding:10px}}</style><h1>BuyPlan v4.0：λ変動を使う独立試作</h1><p>試合前原本A・6開催。正式券や本番フローを変更しない。±0.1/0.2点と17状態の等重みは仮定であり、予測誤差の実測分布ではない。</p><h2>各方式の10口評価</h2><div class="scroll">{tab(compact)}</div><h2>最悪状態での達成率（各状態の理論最適に対する比）</h2>{chart}<h2>試作版の全10口</h2><div class="scroll">{tab(table)}</div><h2>13試合×10口の配券表（全方式）</h2>{matrix_html}<h2>数学上位10口から変更された券</h2><div class="scroll">{tab(changes[['round','method','ticket_no','symbols','baseline_probability','mean_state_probability','anchor_differences']])}</div><h2>シナリオ別の10口13的中確率</h2><div class="scroll">{tab(state_pick)}</div><p>全状態は <a href="state_evaluation.csv">CSV</a>。結果の的中数は事後採点で、最適化には使っていない。</p><p><a href="report.md">詳細レポート</a> ／ <a href="tickets.csv">全券</a> ／ <a href="allocation_matrices.csv">配置表</a> ／ <a href="portfolio_summary.csv">方式比較</a> ／ <a href="match_stability.csv">得点構造と安定性</a> ／ <a href="changed_tickets.csv">変更理由</a> ／ <a href="states.csv">状態</a> ／ <a href="audit.json">監査</a></p></html>'''
    (OUT/'index.html').write_text(html)

if __name__=='__main__':run()
