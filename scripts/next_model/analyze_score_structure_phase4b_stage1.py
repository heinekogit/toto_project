"""Phase 4-B stage 1: exact top-ten benchmark, read-only archived-plan comparison."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,sha256
from forecast_registry import selected_matches,DEFAULT_ROOT,read_log

OUT=ROOT/'data/reports/score_structure_phase4b_stage1'
ROUNDS=('toto1660','toto1663','toto1664','toto1665','toto1668','toto1670')
SYMBOLS=np.array(['H','D','A'])
N=3**13
POWERS=(3**np.arange(12,-1,-1,dtype=np.int64))


def universe(prob):
    """Enumerate 3^13 combinations in lexicographic H,D,A order."""
    codes=np.arange(N,dtype=np.int64)
    mass=np.ones(N,dtype=np.float64)
    for i,power in enumerate(POWERS):
        mass*=prob[i,(codes//power)%3]
    return mass


def decode(codes):
    arr=np.asarray(codes,dtype=np.int64)
    return ((arr[:,None]//POWERS)%3).astype(np.uint8)


def encode(tickets):
    return np.asarray(tickets,dtype=np.int64)@POWERS


def top_ten(mass):
    cutoff=float(np.partition(mass,-10)[-10])
    eligible=np.flatnonzero(mass>=cutoff)
    order=np.lexsort((eligible,-mass[eligible]))
    return eligible[order[:10]]


def score_tickets(matrix,prob):
    return np.prod(prob[np.arange(13)[None,:],matrix],axis=1)


def saved_plan(path,manifest,source_hash,expected_created):
    if manifest['created_at']!=expected_created:
        raise ValueError(f'{path}: manifest timestamp differs from registered forecast')
    name=path.name
    if manifest.get('output_sha256',{}).get(name)!=sha256(path):
        raise ValueError(f'{path}: plan file hash differs from prematch manifest')
    if manifest['output_sha256'].get('next_predictions.csv')!=source_hash:
        raise ValueError(f'{path}: plan/forecast snapshot mismatch')
    d=pd.read_csv(path,encoding='utf-8-sig',low_memory=False).sort_values('match_no')
    if len(d)!=13 or d.match_no.tolist()!=list(range(1,14)):
        raise ValueError(f'{path}: not 13 ordered matches')
    columns=sorted((c for c in d if c.startswith('ticket') and c[6:].isdigit()),key=lambda x:int(x[6:]))
    if len(columns)!=10:raise ValueError(f'{path}: expected ten tickets')
    values=d[columns].to_numpy()
    if not np.isin(values,[0,1,2]).all():raise ValueError(f'{path}: invalid symbol')
    # Archived conventions: 1=H, 0=D, 2=A.
    return np.where(values==1,0,np.where(values==0,1,2)).T.astype(np.uint8),columns


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    policy=ROOT/'BUYPLAN_V4_0.md'
    if '13試合全的中確率を最適化対象' not in policy.read_text():raise ValueError('v4.0 policy missing')
    reg=selected_matches(DEFAULT_ROOT)
    if len(reg)!=78 or not reg.provenance_class.eq('A').all() or reg.fixture_id.duplicated().any():
        raise ValueError('Only 78 unique A originals accepted')
    events=read_log(DEFAULT_ROOT/'forecasts.jsonl')
    links=read_log(DEFAULT_ROOT/'portfolio_links.jsonl')
    source_paths=[DEFAULT_ROOT/'forecasts.jsonl',DEFAULT_ROOT/'results.jsonl',DEFAULT_ROOT/'portfolio_links.jsonl',policy]
    all_tickets=[];summary=[];content=[];actuals=[];marginals=[];source_manifest=[]
    for rid in ROUNDS:
        g=reg[reg.round_id.eq(rid)].sort_values('match_no')
        if len(g)!=13 or g.match_no.tolist()!=list(range(1,14)):raise ValueError(rid)
        ev=[e for e in events if e['round_id']==rid]
        hashes={e['output_sha256'] for e in ev};times={e['forecast_created_at'] for e in ev}
        if len(hashes)!=1 or len(times)!=1:raise ValueError(f'{rid}: nonunique forecast snapshots')
        source_hash=next(iter(hashes));created=next(iter(times))
        link=[e for e in links if e['round_id']==rid and e['forecast_snapshot_sha256']==source_hash]
        if len(link)!=1:raise ValueError(f'{rid}: missing/ambiguous portfolio registry link')
        directory=ROOT/'data/eval/toto_rounds'/rid/'snapshot'
        mp=directory/'buyplan_engine_manifest.json';manifest=json.loads(mp.read_text())
        if sha256(mp)!=link[0]['manifest_sha256']:raise ValueError(f'{rid}: link manifest hash mismatch')
        source_paths.append(mp)
        sources=[('保存正式版 '+manifest['official_version'],directory/'buyplan.csv')]
        for label,name in [('保存Shadow C','shadow_buyplan_c.csv'),('保存v2.5 shadow','shadow_buyplan_v25.csv')]:
            if (directory/name).exists() and name in manifest.get('output_sha256',{}):sources.append((label,directory/name))
        archive=[]
        for label,path in sources:
            matrix,cols=saved_plan(path,manifest,source_hash,created)
            source_paths.append(path);archive.append((label,path,matrix,cols))
        raw=g[['raw_prob_home','raw_prob_draw','raw_prob_away']].to_numpy(float)
        final=g[['final_prob_home','final_prob_draw','final_prob_away']].to_numpy(float)
        for stage,p in [('raw',raw),('final',final)]:
            if not np.allclose(p.sum(axis=1),1,atol=1e-10):raise ValueError(f'{rid}: {stage} non-normalized')
        outcome=np.where(g.actual_home_score.to_numpy(int)>g.actual_away_score.to_numpy(int),0,
                         np.where(g.actual_home_score.to_numpy(int)==g.actual_away_score.to_numpy(int),1,2)).astype(np.uint8)
        actual_code=int(encode(outcome[None,:])[0]);actual_symbols=''.join(SYMBOLS[outcome])
        plans={name:(matrix,cols,str(path)) for name,path,matrix,cols in archive}
        opt={};masses={}
        for stage,p in [('raw',raw),('final',final)]:
            mass=universe(p);masses[stage]=mass
            indices=top_ten(mass);matrix=decode(indices);opt[stage]=matrix
            if len(np.unique(indices))!=10:raise AssertionError('duplicate exact top 10')
            if np.any(np.diff(mass[indices])>1e-18):raise AssertionError('top ten not sorted')
            if np.any(mass[indices]<np.partition(mass,-10)[-10]):raise AssertionError('not exact top ten')
            name=f'数学最適10口_{stage}'
            plans[name]=(matrix,[f'opt{stage}{i:02}' for i in range(1,11)],'calculated')
            pd.DataFrame([dict(round=rid,ticket_no=i+1,symbols=''.join(SYMBOLS[row]),
                               joint_probability=float(mass[indices[i]]),probability_percent=float(mass[indices[i]]*100),
                               **{f'match_{j+1:02}':SYMBOLS[v] for j,v in enumerate(row)}) for i,row in enumerate(matrix)]).to_csv(
                                   OUT/f'{rid}_optimal_{stage}.csv',index=False,encoding='utf-8-sig')
            source_manifest.append(dict(round=rid,stage=stage,forecast_hash=source_hash,forecast_created_at=created,
                                        official_version=manifest['official_version'],top10_total=float(mass[indices].sum()),
                                        cutoff_10=float(mass[indices[-1]]),actual_rank=int(np.sum(mass>mass[actual_code])+
                                          np.sum((mass==mass[actual_code])&(np.arange(N)<actual_code))+1)))
        for name,(matrix,cols,path) in plans.items():
            codes=encode(matrix);unique=np.unique(codes);actual_hit=(matrix==outcome).sum(axis=1)
            overlap=int(len(set(codes)&set(encode(opt['final']))))
            for stage,p in [('raw',raw),('final',final)]:
                mass=masses[stage];individual=mass[codes]
                total=float(mass[unique].sum()) # duplicate tickets cover one result once
                optimum=float(mass[encode(opt[stage])].sum())
                summary.append(dict(round=rid,model_signature=g.model_signature.iloc[0],official_version=manifest['official_version'],
                    method=name,evaluation=stage,total_probability=total,total_percent=100*total,
                    optimum_probability=optimum,optimum_percent=100*optimum,attainment_percent=100*total/optimum,
                    gap_probability=optimum-total,gap_percentage_points=100*(optimum-total),
                    improvement_over_method_percent=100*(optimum/total-1) if total else np.nan,
                    unique_tickets=len(unique),duplicate_tickets=10-len(unique),overlap_with_final_optimum=overlap,
                    exact13=int(np.any(codes==actual_code)),best_hits=int(actual_hit.max()),
                    best_ticket=cols[int(np.argmax(actual_hit))],actual_probability=float(mass[actual_code]),
                    actual_rank=next(x['actual_rank'] for x in source_manifest if x['round']==rid and x['stage']==stage),
                    source=path))
                for i,row in enumerate(matrix):
                    all_tickets.append(dict(round=rid,model_signature=g.model_signature.iloc[0],method=name,evaluation=stage,
                        ticket=cols[i],symbols=''.join(SYMBOLS[row]),probability=float(individual[i]),
                        hits=int(actual_hit[i]),D_count=int(np.sum(row==1)),source=path))
            for j,(_,match) in enumerate(g.iterrows()):
                counts=np.bincount(matrix[:,j],minlength=3)
                marginals.append(dict(round=rid,method=name,match_no=j+1,home_team=match.home_team,away_team=match.away_team,
                    H=int(counts[0]),D=int(counts[1]),A=int(counts[2]),
                    final_opt_H=int(np.sum(opt['final'][:,j]==0)),final_opt_D=int(np.sum(opt['final'][:,j]==1)),
                    final_opt_A=int(np.sum(opt['final'][:,j]==2)),
                    adopted_not_in_final_opt=int(np.sum(~np.isin(codes,encode(opt['final']))))))
            content.append(dict(round=rid,method=name,tickets=' / '.join(''.join(SYMBOLS[x]) for x in matrix),
                overlap_with_final_optimum=overlap,only_method=' / '.join(''.join(SYMBOLS[x]) for x in matrix if int(encode(x[None,:])[0]) not in set(encode(opt['final']))),
                only_final_optimum=' / '.join(''.join(SYMBOLS[x]) for x in opt['final'] if int(encode(x[None,:])[0]) not in set(codes))))
        actuals.append(dict(round=rid,model_signature=g.model_signature.iloc[0],actual_symbols=actual_symbols,
            actual_home_goals='-'.join(map(str,g.actual_home_score.to_numpy(int))),
            actual_away_goals='-'.join(map(str,g.actual_away_score.to_numpy(int))),
            raw_probability=float(masses['raw'][actual_code]),final_probability=float(masses['final'][actual_code]),
            raw_rank=next(x['actual_rank'] for x in source_manifest if x['round']==rid and x['stage']=='raw'),
            final_rank=next(x['actual_rank'] for x in source_manifest if x['round']==rid and x['stage']=='final'),
            in_raw_opt=int(actual_code in set(encode(opt['raw']))),in_final_opt=int(actual_code in set(encode(opt['final'])))))
    sources={str(p):sha256(p) for p in source_paths}
    # Full brute-force verification independently sorts all 1,594,323 probabilities for one round.
    g=reg[reg.round_id.eq('toto1670')].sort_values('match_no')
    p=g[['final_prob_home','final_prob_draw','final_prob_away']].to_numpy(float)
    full=universe(p);brute=np.lexsort((np.arange(N),-full))[:10]
    observed=pd.read_csv(OUT/'toto1670_optimal_final.csv')
    if list(observed.symbols)!=[''.join(SYMBOLS[x]) for x in decode(brute)]:raise AssertionError('full enumeration mismatch')
    pd.DataFrame(all_tickets).to_csv(OUT/'ticket_probabilities.csv',index=False,encoding='utf-8-sig')
    compare=pd.DataFrame(summary);compare.to_csv(OUT/'portfolio_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(content).to_csv(OUT/'ticket_content_differences.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(marginals).to_csv(OUT/'match_symbol_counts.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(actuals).to_csv(OUT/'actual_outcomes.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(source_manifest).to_csv(OUT/'optimal_audit.csv',index=False,encoding='utf-8-sig')
    if any(sha256(Path(path))!=hash for path,hash in sources.items()):raise AssertionError('source changed')
    audit=dict(policy=str(policy),policy_sha256=sha256(policy),rounds=list(ROUNDS),matches=78,combinations=N,
        full_enumeration_verified_round='toto1670',tie_rule='joint probability descending, then H/D/A lexicographic',
        input_class='prematch A only',raw_final_separate=True,old_v30_used=False,fixed_allocation_constraint=False,
        source_hashes=sources,official_versions=compare.drop_duplicates('round').set_index('round').official_version.to_dict(),
        existing_files_unchanged=True,new_algorithm_connected_to_production=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    make_report(compare,pd.DataFrame(actuals),pd.DataFrame(marginals),pd.DataFrame(content),audit)
    print(json.dumps(compare[(compare.evaluation=='final')&(compare.method.str.startswith('保存正式版'))][['round','official_version','attainment_percent','total_percent','optimum_percent']].to_dict('records'),ensure_ascii=False))


def make_report(c,a,m,content,audit):
    main=c[(c.evaluation=='final')&(c.method.str.startswith(('保存正式版','数学最適')))].copy()
    raw=c[(c.evaluation=='raw')&(c.method.str.startswith(('保存正式版','数学最適')))].copy()
    history=c[(c.evaluation=='final')&(~c.method.str.startswith(('保存正式版','数学最適')))].copy()
    cols=['round','official_version','method','total_percent','attainment_percent','gap_percentage_points','unique_tickets','overlap_with_final_optimum','best_hits','actual_rank']
    table=lambda d,cc:d[cc].to_html(index=False,escape=True,float_format=lambda x:f'{x:.6f}')
    vals=main[main.method.str.startswith('保存正式版')]
    by_model=vals.groupby('model_signature',as_index=False).agg(rounds=('round','size'),mean_attainment_percent=('attainment_percent','mean'),min_attainment_percent=('attainment_percent','min'),max_attainment_percent=('attainment_percent','max'))
    text=f'''# Phase 4-B 第1段階：v4.0固定確率の数学的基準\n\n参照方針：`BUYPLAN_V4_0.md`。旧v3.0のP(D)上位4・ダブル展開も、v2.4の固定配分も最適化条件に使用していない。本番配券は変更していない。\n\n## 結論\n\n試合前原本Aの6開催78試合。3^13={N:,}組を対象に、試合間独立・保存確率が正しいという条件で上位の異なる10組を厳密算出した。保存正式版は開催順にv2.1、v2.2、v2.3、v2.3、v2.4、v2.4であり、全6開催をv2.4とは呼ばない。達成率は最適値に対する同一final分布下の比率。これは真の的中確率の推定ではない。\n\n{main[cols].to_string(index=False)}\n\n## 予測モデル系統別（保存正式版・final）\n\n{by_model.to_string(index=False)}\n\n3系統は開催数が異なり、モデル系統とBuyPlan版が交絡する。系統間の因果的な優劣は判断しない。\n\n## raw分布での比較\n\n{raw[cols].to_string(index=False)}\n\nrawとfinalの最適値は異なる確率分布に条件付けられる。数値差からどちらの予測精度が高いとは判断しない。各券は両分布で評価しており、同一評価分布での交差比較は `portfolio_comparison.csv` を参照。\n\n## 保存済みのシャドー（存在する開催のみ）\n\n{history[cols].to_string(index=False) if len(history) else '該当なし'}\n\n保存されていない方式は補完生成していない。旧v3.0は比較対象外。\n\n## 実際の結果\n\n{a[['round','actual_symbols','raw_probability','final_probability','raw_rank','final_rank','in_raw_opt','in_final_opt']].to_string(index=False)}\n\n実結果は配券・確率の計算に使用していない。最高的中数は補助診断であり、13的中確率の代用ではない。\n\n## 配券内容と制約の影響\n\n完全一致券、各試合のH/D/A採用枚数、最適10組に含まれない既存券をCSVに示した。既存正式版と最適値の差は、同一確率を固定したときの配券方式全体の機会差である。固定枚数、乱数、配置目的など個別要因への因果分解は行っていない。数学的最適券の集中は固定分布では合理的であり、予測誤差への脆弱性は別途検証する。\n\n## 次段階へ渡す課題\n\nH/D/Aの校正誤差、λ予測誤差、得点分布の追加分散、試合ごとの不確実性、試合間依存、確率シナリオ変動時の13的中確率低下を、試合前原本を継続蓄積して時系列の未使用期間で評価する。78試合を使った係数の後付け調整はしない。固定確率の上位10組を正式方式にする判断もまだ行わない。\n\n## 出力と検証\n\n全組み合わせの確率を厳密計算し、1670回finalでは1,594,323組の全ソート結果と上位10券の一致を確認。各試合H/D/A和1、相異なる10券、保存配券のハッシュ・時刻・レジストリ連結を確認。比較対象の重複券は確率質量を一度だけ算入。計算前後の原本ハッシュは一致。\n\n[比較](portfolio_comparison.csv) ／ [券確率](ticket_probabilities.csv) ／ [券差分](ticket_content_differences.csv) ／ [試合別記号枚数](match_symbol_counts.csv) ／ [実結果](actual_outcomes.csv) ／ [監査](audit.json)\n'''
    (OUT/'report.md').write_text(text)
    bars=''.join(f'<p>{r["round"]} 保存{r["official_version"]}：{r["attainment_percent"]:.2f}%<br><span style="display:inline-block;background:#327bb0;width:{max(1,r["attainment_percent"]*4):.0f}px;height:13px"></span></p>' for _,r in vals.iterrows())
    html=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>v4.0 Phase 4-B 第1段階</title><style>body{{font:16px/1.7 system-ui;margin:26px;color:#203047}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{border:1px solid #ccd4df;padding:5px}}th{{background:#edf3f8}}.scroll{{overflow:auto}}a{{color:#245b92}}</style><h1>v4.0 Phase 4-B 第1段階</h1><p>試合前原本A・6開催78試合。固定H/D/A確率と試合間独立の下で、全{N:,}組の上位10券が13的中モデル確率を最大にする。正式方式の採用判断ではない。</p><h2>保存正式版の最適値達成率（final）</h2>{bars}<div class="scroll">{table(main,cols)}</div><p>1660〜1665は保存当時のv2.1〜v2.3。全開催をv2.4扱いしない。達成率は同一final確率での比。</p><h2>予測モデル系統別（保存正式版・final）</h2><div class="scroll">{table(by_model,list(by_model.columns))}</div><p>開催数が少なくモデル系統と配券版が交絡するため、因果比較ではない。</p><h2>raw確率での比較</h2><div class="scroll">{table(raw,cols)}</div><h2>保存シャドー</h2><div class="scroll">{table(history,cols) if len(history) else '該当なし'}</div><h2>実結果</h2><div class="scroll">{table(a,['round','actual_symbols','raw_probability','final_probability','raw_rank','final_rank','in_raw_opt','in_final_opt'])}</div><h2>読み方と引き継ぎ</h2><p>旧v3.0の4ダブルやv2.4の固定枚数は最適化条件に使用していない。raw/finalの確率値を直接比べて予測精度の優劣とはしない。実際の最高的中数は補助診断。既存方式との差は配券方式全体の差であり、個別制約の因果効果ではない。次は試合前データを蓄積し、予測確率の校正・誤差・依存性を後続期間で検証する。</p><p><a href="report.md">詳細レポート</a> ／ <a href="portfolio_comparison.csv">比較CSV</a> ／ <a href="ticket_probabilities.csv">各券</a> ／ <a href="ticket_content_differences.csv">券差分</a> ／ <a href="match_symbol_counts.csv">試合別採用</a> ／ <a href="actual_outcomes.csv">結果</a> ／ <a href="optimal_audit.csv">最適券監査</a> ／ <a href="audit.json">原本監査</a></p></html>'''
    (OUT/'index.html').write_text(html)

if __name__=='__main__':main()
