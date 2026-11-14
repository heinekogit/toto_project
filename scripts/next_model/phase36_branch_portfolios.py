"""Independent Phase 3.6: frozen selectors and hypothetical odds-stress portfolios."""
from pathlib import Path
import hashlib,json,heapq,time,html
import numpy as np
import pandas as pd
from analyze_phase35_high_draw_structure import conditions
from analyze_score_structure_phase4b_stage1 import encode,decode,SYMBOLS
from prototype_buyplan_v40 import greedy_relative_robust,ticket_probabilities

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/reports/score_structure_phase36'
SOURCE=ROOT/'data/reports/score_structure_phase35/diagnostics_78.csv'
FEATURES=['round','match_no','home_team','away_team','lambda_home','lambda_away','P_0_0','P_1_1','log_R','baseline_P_H','baseline_P_D','baseline_P_A']
SPEC=dict(version='phase36-research-1',candidate_counts=[3,4,5],selectors={
 'A':'final PD descending; match_no ascending ties',
 'B':'S first, then final PD descending, match_no ascending',
 'C':'S first; S sorted abs(logR), then final PD descending, match_no ascending; outside S sorted final PD'},
 condition_S='P00<=within-round median AND P11>=within-round median, unrounded values',
 stress_odds_factor=2.0,stress_formula='pD*=2 on active candidates; normalize all 3 probabilities; H/A ratio preserved',
 stress_interpretation='Uncalibrated research hypothesis, not lambda perturbation or observed error frequency; no 75% replacement',
 states='All subsets of selected D candidates active; baseline included; no state occurrence frequency asserted',
 generator='H/A allowed all 13 matches; D allowed only selected candidates; exact top30 per state + baseline top10; union pool',
 selection='max min_s(sum_ticket q_s / best_allowed10_s); deterministic greedy+12 swap sweeps, baseline seed; approximate',
 tie_break='mean relative value then ternary code H,D,A; candidate labels irrelevant after membership',
 pool_per_state=30,number_tickets=10,common_evaluation='base final and union of all selector/k stress masks per round',
 no_result_input=True,new_parameter_search=False,official_flow_modified=False)

def select_candidates(g,method,k):
    g=g.copy();g['S']=conditions(g)['inclusive'];g['abs_logR']=g.log_R.abs()
    if method=='A':s=g.sort_values(['baseline_P_D','match_no'],ascending=[False,True])
    elif method=='B':s=g.sort_values(['S','baseline_P_D','match_no'],ascending=[False,False,True])
    elif method=='C':
        s=pd.concat([g[g.S].sort_values(['abs_logR','baseline_P_D','match_no'],ascending=[True,False,True]),g[~g.S].sort_values(['baseline_P_D','match_no'],ascending=[False,True])])
    else:raise ValueError(method)
    return s.match_no.iloc[:k].to_numpy(int)-1


def top_codes(p,allowed,count=30):
    """Exact best Cartesian products by heap; non-permitted symbols never renormalized."""
    choices=[sorted(a,key=lambda s:(-p[i,s],s)) for i,a in enumerate(allowed)]
    def entry(ranks):
        ticket=np.array([choices[i][r] for i,r in enumerate(ranks)],dtype=np.uint8)
        return (-float(np.prod(p[np.arange(len(p)),ticket])),int(encode(ticket[None,:])[0]),ranks)
    start=(0,)*13;heap=[entry(start)];seen={start};out=[]
    while heap and len(out)<count:
        _,code,ranks=heapq.heappop(heap);out.append(code)
        for i in range(13):
            if ranks[i]+1<len(choices[i]):
                nxt=list(ranks);nxt[i]+=1;nxt=tuple(nxt)
                if nxt not in seen:seen.add(nxt);heapq.heappush(heap,entry(nxt))
    return np.array(out,dtype=np.int64)


def masks_for(selected):
    selected=sorted(map(int,selected))
    return sorted(sum(1<<selected[j] for j in range(len(selected)) if b>>j&1) for b in range(2**len(selected)))

def stress(p,mask):
    q=p.copy()
    for i in range(13):
        if mask>>i&1:q[i,1]*=SPEC['stress_odds_factor'];q[i]/=q[i].sum()
    assert np.allclose(q.sum(axis=1),1)
    return q


def generate(p,selected):
    allowed=[[0,1,2] if i in selected else [0,2] for i in range(13)]
    masks=masks_for(selected);states=[stress(p,m) for m in masks]
    tops=[top_codes(q,allowed,SPEC['pool_per_state']) for q in states]
    base=top_codes(p,allowed,10)
    pool=np.array(sorted(set(np.concatenate(tops+[base]))),dtype=np.int64)
    values=np.array([ticket_probabilities(q,decode(pool)) for q in states])
    opts=np.array([ticket_probabilities(q,decode(c[:10])).sum() for q,c in zip(states,tops)])
    codes=greedy_relative_robust(pool,values,opts,base)
    assert len(set(codes))==10
    return codes,dict(pool_size=len(pool),state_count=len(masks),own_worst_ratio=float((np.array([ticket_probabilities(q,decode(codes)).sum() for q in states])/opts).min()),
                      pool_hash=hashlib.sha256(pool.tobytes()).hexdigest()),pool,masks


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    # Written before generation and before loading outcomes.
    (OUT/'fixed_spec.json').write_text(json.dumps(SPEC,ensure_ascii=False,indent=2))
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    forecasts=pd.read_csv(SOURCE,usecols=FEATURES)
    events=[json.loads(x) for x in (ROOT/'data/forecast_registry/forecasts.jsonl').read_text().splitlines()]
    for e in events:
        assert e['provenance_class']=='A'
        assert hashlib.sha256(Path(e['source_file']).read_bytes()).hexdigest()==e['output_sha256']
        row=forecasts[(forecasts['round']==e['round_id'])&(forecasts.match_no==e['match_no'])].iloc[0]
        for a,b in [('lambda_home','lambda_home'),('lambda_away','lambda_away'),('baseline_P_H','final_prob_home'),('baseline_P_D','final_prob_draw'),('baseline_P_A','final_prob_away')]:assert abs(row[a]-e[b])<1e-12
    plans={};membership={};meta={};pools={};allmasks={};selection_rows=[];state_rows=[]
    for rnd,g in forecasts.groupby('round',sort=True):
        g=g.sort_values('match_no');assert g.match_no.tolist()==list(range(1,14))
        p=g[['baseline_P_H','baseline_P_D','baseline_P_A']].to_numpy();base=top_codes(p,[[0,1,2]]*13,10)
        plans[(rnd,'基準',0)]=base;meta[(rnd,'基準',0)]={'pool_size':3**13,'state_count':1,'own_worst_ratio':1,'seconds':0}
        cache={};union={0};s=conditions(g)['inclusive'].to_numpy();ranks=g.baseline_P_D.rank(ascending=False,method='min').to_numpy()
        for k in SPEC['candidate_counts']:
            for method in ['A','B','C']:
                sel=select_candidates(g,method,k);membership[(rnd,method,k)]=set(map(int,sel));canonical=tuple(sorted(sel))
                t=time.perf_counter()
                if canonical not in cache:cache[canonical]=generate(p,canonical)
                codes,info,pool,masks=cache[canonical];union.update(masks)
                key=(rnd,method,k);plans[key]=codes;meta[key]={**info,'seconds':time.perf_counter()-t,'cache_reused':canonical in [tuple(sorted(v)) for kk,v in membership.items() if kk!=key and kk[0]==rnd]};pools[key]=pool
                for priority,i in enumerate(sel,1):
                    r=g.iloc[i];selection_rows.append(dict(round=rnd,method=method,k=k,priority=priority,match_no=i+1,home_team=r.home_team,away_team=r.away_team,
                       PD=r.baseline_P_D,PD_rank=ranks[i],S=bool(s[i]),abs_logR=abs(r.log_R),HA_gap=abs(r.baseline_P_H-r.baseline_P_A)))
        allmasks[rnd]=sorted(union)
        for m in sorted(union):
            q=stress(p,m)
            for i,r in enumerate(g.itertuples()):state_rows.append(dict(round=rnd,mask=m,active_matches=';'.join(str(j+1) for j in range(13) if m>>j&1),match_no=i+1,H=q[i,0],D=q[i,1],A=q[i,2]))
        print(f'{rnd}: generated baseline + 9 configurations; unique selector sets={len(cache)}; common states={len(union)}',flush=True)
    # Outcomes are loaded only after all candidate and portfolio selection is complete.
    outcomes=pd.read_csv(SOURCE,usecols=['round','match_no','actual_symbol'])
    sel=pd.DataFrame(selection_rows).merge(outcomes,on=['round','match_no'],validate='many_to_one')
    selection_summary=[];changes=[];ticketrows=[];reasonrows=[];metrics=[];alloc=[];evaluations=[];matchdiag=[]
    for rnd,g in forecasts.groupby('round',sort=True):
        g=g.sort_values('match_no');p=g[['baseline_P_H','baseline_P_D','baseline_P_A']].to_numpy()
        truth=outcomes[outcomes['round'].eq(rnd)].sort_values('match_no').actual_symbol.map({'H':0,'D':1,'A':2}).to_numpy()
        base=plans[(rnd,'基準',0)];bt=decode(base);baseprob=ticket_probabilities(p,bt).sum();basecover=(bt==truth).any(axis=0)
        base_cells=(bt==truth).sum(axis=0)
        for key,codes in [(key,value) for key,value in plans.items() if key[0]==rnd]:
            _,method,k=key;t=decode(codes);cover=(t==truth).any(axis=0);correct=(t==truth).sum(axis=0);hits=(t==truth).sum(axis=1)
            rawprob=ticket_probabilities(p,t);new=set(codes)-set(base);removed=set(base)-set(codes)
            cand=membership.get(key,set());actuald=set(np.flatnonzero(truth==1));capt=cand&actuald
            if method!='基準':
                a=membership[(rnd,'A',k)];added=cand-a;dropped=a-cand
                selection_summary.append(dict(round=rnd,method=method,k=k,selected=';'.join(str(i+1) for i in sorted(cand)),D_captured=len(capt),nonD_selected=k-len(capt),D_missed=len(actuald-cand),
                    overlap_A=len(cand&a),added=';'.join(str(i+1) for i in sorted(added)),removed=';'.join(str(i+1) for i in sorted(dropped)),gained_D=len(added&actuald),lost_D=len(dropped&actuald),net_D=len(capt)-len(a&actuald)))
                for kind,ids in [('added',added),('removed',dropped)]:
                    for i in sorted(ids):changes.append(dict(round=rnd,method=method,k=k,change=kind,match_no=i+1,card=g.iloc[i].home_team+'－'+g.iloc[i].away_team,actual=SYMBOLS[truth[i]],PD=p[i,1]))
            scenario_values=[]
            for mask in allmasks[rnd]:
                q=stress(p,mask);v=ticket_probabilities(q,t).sum();vb=ticket_probabilities(q,bt).sum();scenario_values.append(v)
                evaluations.append(dict(round=rnd,method=method,k=k,mask=mask,prob13=v,baseline_prob13=vb,difference=v-vb))
            allowedtruth=all(truth[i]!=1 or i in cand for i in range(13)) if method!='基準' else True
            inpool=int(encode(truth[None,:])[0]) in set(pools[key]) if method!='基準' else True
            metrics.append(dict(round=rnd,method=method,k=k,final_prob13=float(rawprob.sum()),baseline_retention=float(rawprob.sum()/baseprob),
                common_state_min_prob13=min(scenario_values),common_state_max_prob13=max(scenario_values),best_hits=int(hits.max()),exact13=bool(max(hits)==13),
                selected_D_capture=len(capt) if method!='基準' else 0,purchased_D_capture=int((cover&(truth==1)).sum()),D_cells=int((t==1).sum()),D_tickets=int((t==1).any(axis=1).sum()),
                HA_match_coverage_lost=int((basecover&~cover&(truth!=1)).sum()),HA_correct_cells_change=int((correct-base_cells)[truth!=1].sum()),
                correct_symbols_available=int(cover.sum()),combination_gap=int(cover.sum()-hits.max()),allowed_truth=allowedtruth,truth_in_search_pool=inpool,
                all_symbols_but_no13=bool(cover.all() and max(hits)<13),overlap_baseline=len(set(codes)&set(base)),
                added_final_mass=sum(ticket_probabilities(p,decode(list(new)))) if new else 0,removed_final_mass=sum(ticket_probabilities(p,decode(list(removed)))) if removed else 0,**meta[key]))
            for j,code in enumerate(codes):
                item=t[j];distance=(bt!=item).sum(axis=1);anchor=int(np.argmin(distance));diff=np.flatnonzero(bt[anchor]!=item)
                active=';'.join(str(i+1) for i in np.flatnonzero(item==1)); probs=np.array([ticket_probabilities(stress(p,m),item[None,:])[0] for m in allmasks[rnd]])
                ticketrows.append(dict(round=rnd,method=method,k=k,ticket_no=j+1,code=int(code),symbols=''.join(SYMBOLS[item]),D_count=int((item==1).sum()),final_probability=rawprob[j],hits=int(hits[j]),
                    max_probability_state=allmasks[rnd][int(probs.argmax())],max_state_probability=probs.max(),**{f'match_{i+1:02}':SYMBOLS[x] for i,x in enumerate(item)}))
                reasonrows.append(dict(round=rnd,method=method,k=k,ticket_no=j+1,code=int(code),status='retained' if code in set(base) else 'added',
                    nearest_baseline_ticket=anchor+1,changes=';'.join(f'{i+1}:{SYMBOLS[bt[anchor,i]]}>{SYMBOLS[item[i]]}' for i in diff),
                    D_candidate_branches=active,reason='候補選出→Dオッズ2倍の全部分集合状態→最悪状態の最適値比を評価。近い基準券は説明用対応で置換命令ではない。' if len(diff) else '基準券と一致'))
            for code in sorted(removed):reasonrows.append(dict(round=rnd,method=method,k=k,code=int(code),status='removed',reason='10口の集合差分。基準分布ではこの券の確率質量を失う。',changes=''.join(SYMBOLS[decode([code])[0]])))
            for i,r in enumerate(g.itertuples()):
                alloc.append(dict(round=rnd,method=method,k=k,match_no=i+1,card=r.home_team+'－'+r.away_team,actual=SYMBOLS[truth[i]],**{f'ticket_{j+1:02}':SYMBOLS[t[j,i]] for j in range(10)}))
                matchdiag.append(dict(round=rnd,method=method,k=k,match_no=i+1,card=r.home_team+'－'+r.away_team,actual=SYMBOLS[truth[i]],D_candidate=i in cand,H_count=int((t[:,i]==0).sum()),D_count=int((t[:,i]==1).sum()),A_count=int((t[:,i]==2).sum()),correct_count=int(correct[i]),baseline_correct_count=int(base_cells[i]),
                    uncovered=not bool(cover[i]),correct_count_change=int(correct[i]-base_cells[i])))
    frames={'D_candidate_selection':sel,'selection_summary':pd.DataFrame(selection_summary),'selection_changes':pd.DataFrame(changes),
            'shadow_tickets':pd.DataFrame(ticketrows),'ticket_changes':pd.DataFrame(reasonrows),'evaluation_summary':pd.DataFrame(metrics),
            'allocation_matrix':pd.DataFrame(alloc),'match_coverage':pd.DataFrame(matchdiag),'scenario_evaluation':pd.DataFrame(evaluations),'scenario_probabilities':pd.DataFrame(state_rows)}
    for name,frame in frames.items():frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    assert before==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    audit=dict(source_sha256=before,forecast_A_count=78,source_hashes_verified=True,results_loaded_after_all_plans=True,
        methods_and_parameters_frozen=True,all_tickets_distinct=all(len(set(x))==10 for x in plans.values()),configuration_count=len(plans),
        B_C_equal=all(np.array_equal(plans[(r,'B',k)],plans[(r,'C',k)]) for r in forecasts['round'].unique() for k in [3,4,5]),
        original_data_unchanged=True,solver='reused deterministic greedy+swap; restricted pool, not global optimal',
        spec_sha256=hashlib.sha256((OUT/'fixed_spec.json').read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    return frames


def render(f):
    e=f['evaluation_summary'];ss=f['selection_summary'];a=f['allocation_matrix'];sel=f['D_candidate_selection'];change=f['selection_changes']
    def table(x):return '<div class="scroll">'+x.to_html(index=False,border=0,float_format=lambda v:f'{v:.7f}')+'</div>'
    summary=ss.groupby(['method','k']).agg(selected=('k','sum'),D_captured=('D_captured','sum'),gained_D=('gained_D','sum'),lost_D=('lost_D','sum'),net_D=('net_D','sum')).reset_index()
    es=e.groupby(['method','k']).agg(exact13_rounds=('exact13','sum'),mean_best_hits=('best_hits','mean'),purchased_D_capture=('purchased_D_capture','sum'),D_cells=('D_cells','sum'),mean_baseline_retention=('baseline_retention','mean'),HA_match_coverage_lost=('HA_match_coverage_lost','sum')).reset_index()
    summary.to_csv(OUT/'selection_totals.csv',index=False,encoding='utf-8-sig');es.to_csv(OUT/'evaluation_totals.csv',index=False,encoding='utf-8-sig')
    sections=[]
    def add(h,text,frame=None):sections.append('<section><h2>'+h+'</h2><p>'+text+'</p>'+('' if frame is None else table(frame))+'</section>')
    add('結果の読み方','条件Sは今回と同じ78試合から発見済み。これは開発データの再現であり将来性能ではない。元のfinal確率は変更せず、研究用状態を別に保存した。通常予算1,000円・相異なる10口、正式処理への接続なし。')
    add('固定した選出規則','A=P(D)順、B=S優先＋P(D)順、C=S優先＋S内|logR|順。S外はP(D)順、同順位は試合番号。Sは各開催1〜2試合で候補数3〜5を下回るため、BとCの集合・10口は全条件で同一。優先順を配券スコアへ密かに流用せず、追加効果なしと報告する。',summary)
    add('開催ごとの選出差','追加Dと失ったDを別々に数える。候補数の違いと方式の違いを分離する。',ss)
    add('追加・除外された試合','方式Aとの比較。正解を参照して選出や係数を変えていない。',change)
    add('10口の評価式と仮定','候補試合iのDオッズを研究用に2倍：qD=2pD/(1+pD)、qH=pH/(1+pD)、qA=pA/(1+pD)。各候補について基準/変動の全2^k状態を作る。これはλ変動から推定した確率でも、実績75%の転用でもない。強度2倍は一つの未校正の試験条件として計算前に固定し、強度探索は行わない。元のmixはfinalに含まれ、再適用しない。')
    add('候補生成と10口選択','13試合すべてでH/A双方を許し、選出試合のみDを許す。複数DやH/A逆転も可能。各状態の厳密上位30組と基準10組を候補集合にする。目的は max_P min_s [Σ(t∈P) q_s(t) / O_s]、O_sは同じ許容結果空間の最適10組確率和。既存の決定的greedy＋最大12回交換探索を再利用。D本数やD券数、基準券数は固定せず、D首位も候補条件にしない。有限プール・局所探索なので大域最適ではない。平均比は同値時の順位付けにのみ使用し、現実の状態頻度を意味しない。')
    add('同予算の10口比較','基準は条件選出を使わないfinal確率上位10組。元分布における13的中確率は相異なる券の確率和で厳密。共通シナリオ評価はA/B/C・全kの状態を合併して同じ分布で比較する。状態の最小/最大値は現実の信頼区間ではない。',es)
    add('開催別評価','final_prob13は0〜1、百分率表示するには100倍。baseline_retentionは基準確率の維持率。HA損失は全10口で正解H/Aが消えた試合数と、正解セル数の増減を区別する。',e[['round','method','k','final_prob13','baseline_retention','best_hits','exact13','D_cells','purchased_D_capture','HA_match_coverage_lost','HA_correct_cells_change','correct_symbols_available','combination_gap','allowed_truth','truth_in_search_pool','overlap_baseline']])
    best=e[e.method.isin(['基準','A','B'])].pivot(index='round',columns=['method','k'],values='best_hits').reset_index()
    best.columns=['開催']+[f'{m}/{k}' for m,k in best.columns[1:]]
    add('具体的に何が起きたか','4候補では条件Sにより千葉－長崎、今治－八戸、名古屋－C大阪、神戸－町田のDを新しく候補に加えた。一方、清水－G大阪とC大阪－千葉のDを候補から外した。4候補の平均最高的中はA7.50→B7.17で、D捕捉6→8だけでは配券の改善にならなかった。5候補Bは1665で10的中に到達したが、他開催で基準より低い結果も残る。',best)
    add('Dを全て候補にしたのに届かなかった例','1665・B/C・5候補では実際のDを全て候補に含み、H/A双方も許可しているため、全実結果は許容組合せ空間に存在する。しかし各状態上位30組の探索プールには実結果が入らず、購入10口も最高10的中だった。候補選出→プール絞り込み→10口選択の別問題として記録した。全13試合の正解が購入10口のどこかに存在するのに13的中だけ逃したケースは、今回0件。1665・B・4候補では正解が存在する10試合に対し最高8的中で、部分的な組合せ損失もある。')
    add('券が変わった具体例','1665・B・5候補の候補10はHHDDHDDDHHHHH。基準候補01からNo.3 A→D、No.4 H→D、No.6 A→D、No.7 H→D、No.8 A→Dに変わり、10的中。これはD5個を指定した結果ではなく、状態比較による10口選択の結果である。同じ10口にはD0〜5個の券が残る。ただし鹿島－柏のA、東京V－福岡のH、富山－札幌のAは10口全体から欠落し、13的中を妨げた。')
    add('仮定した状態での比較','共通280状態（開催ごとの状態数は異なる）で基準を上回った評価点は、Bの3/4/5候補で172/160/163点。これは現実の発生頻度でも勝率でもない。元の予測分布でのBの13的中確率維持率は開催平均73.09%/63.71%/58.53%であり、D分岐を持つ代償は明確。2倍という未校正仮定が配券変更の直接原因で、条件Sだけからこの代償を正当化したとは扱わない。')
    add('13的中を逃す段階','allowed_truth=falseならD候補選出段階で実結果の全組合せを許せていない。trueでもtruth_in_search_pool=falseなら上位30プールへの絞り込みで失った。正解記号が10口のどこかにある試合数と最高的中との差は組合せ上の損失を示すが、全13記号が揃わない場合は組合せだけを直しても13的中にはならない。個別の欠落試合はmatch_coverage.csvに保存。')
    add('券の追加・除外と機会損失','追加した券は新しく的中可能になった13試合結果そのもの、除外券は拾えなくなった結果そのもの。ticket_changes.csvに全件保存。近い基準券との記号差は説明用で、候補番号同士を実際に置換したという意味ではない。',e[['round','method','k','added_final_mass','removed_final_mass','common_state_min_prob13','common_state_max_prob13','own_worst_ratio','pool_size','seconds']])
    # Simple bar chart from generated summaries.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 390"><rect width="850" height="390" fill="white"/><g font-family="sans-serif" font-size="15"><text x="25" y="25">同数候補のD捕捉（6開催合計・全D18）</text>']
    for j,r in summary.iterrows():
        x=45+j*86;h=float(r.D_captured)*15
        svg.append(f'<rect x="{x}" y="{330-h}" width="55" height="{h}" fill="{ {"A":"#71899d","B":"#166c93","C":"#379684"}[r.method]}"/><text x="{x}" y="{320-h}">{r.D_captured}</text><text x="{x}" y="355">{r.method}/{r.k}</text>')
    svg.append('</g></svg>');(OUT/'candidate_capture.svg').write_text(''.join(svg));sections.insert(2,'<img src="candidate_capture.svg" alt="候補数別D捕捉">')
    for (rnd,method,k),g in a.groupby(['round','method','k'],sort=False):
        z=e[(e['round']==rnd)&(e.method==method)&(e.k==k)].iloc[0]
        sections.append(f'<details><summary>{rnd} {method} 候補{k}：最高{z.best_hits}的中／D {z.D_cells}枠／元分布13的中 {100*z.final_prob13:.7f}%</summary>'+table(g.drop(columns=['round','method','k']))+table(f['shadow_tickets'].query('round == @rnd and method == @method and k == @k')[['ticket_no','symbols','D_count','hits','final_probability']])+'</details>')
    add('未解決と次段階','Sによる候補集合の変更と、未校正のDオッズ変動による配券の変更を分離する必要がある。Bの成績だけでSや2倍仮定の妥当性を証明しない。Cは今回Bと同じため得点力比の追加効果を検証できていない。候補集合・プール絞り込み・10口選択の各段階を記録する構成は後続検証へ引き継げる。正式採用、D最低口数、固定ダブルは導入しない。')
    links=' '.join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(OUT.glob('*.csv')))
    (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 3.6 得点構造と10口分岐配券</title><style>body{font:16px/1.8 system-ui,sans-serif;color:#24384b;max-width:1250px;margin:30px auto;padding:0 20px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:5px 9px;border-bottom:1px solid #d9e1e8;text-align:right}th{background:#edf2f7}section{margin:35px 0}details{margin:15px 0;border:1px solid #d9e1e8;padding:12px}summary{cursor:pointer;font-weight:bold}img{width:100%;max-width:850px}a{margin:8px}</style><h1>Phase 3.6：D候補選出と10口シャドー配券</h1>'+''.join(sections)+'<h2>データ</h2>'+links+'<p><a href="fixed_spec.json">固定仕様</a><a href="audit.json">監査</a></p></html>')
    print(summary.to_string(index=False));print(es.to_string(index=False))

if __name__=='__main__':render(run())
