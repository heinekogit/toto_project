"""Compare preserved operational A, direct exact13 B, constrained coverage C."""
import json
import time
from itertools import combinations
import numpy as np
import pandas as pd
from compare_v25_match_multiplier import round_inputs, internal_tickets
from shadow_v25_legacy import build, REFERENCE
from audit_draw_portfolio import ExactPortfolio
from fixed_exact13_shadow import optimize_b,optimize_c,validate,mass,coverage
from simulate_free_structural_draw import ROOT,PROB_COLUMNS,sha,page,table

OUT=ROOT/'data/reports/toto1670_fixed_exact13'


def main():
    OUT.mkdir(exist_ok=True)
    base=ROOT/'data/purchase_reference'
    protected=[p for p in base.glob('*') if p.is_file()]
    protected += [ROOT/'scripts/next_model'/f for f in ['shadow_v25_operational.py','score_placement_v25_shadow.py']]
    hashes={str(p):sha(p) for p in protected}
    frame,official,actual=round_inputs('toto1670');p=frame[PROB_COLUMNS].to_numpy()
    manifest=json.loads((base/'manifest.json').read_text());assert manifest['output_sha256']['next_predictions.csv']==sha(base/'next_predictions.csv')
    saved_path=base/'shadow_buyplan_v25.csv'
    if json.loads((base/'shadow_v25_audit.json').read_text()).get('mode')=='C95':
        saved_path=OUT/'A_saved.csv'  # Preserve the original legacy A comparison after the C95 switch.
    saved=pd.read_csv(saved_path).sort_values('match_no').reset_index(drop=True)
    for c in ['match_no','home_team','away_team']:assert saved[c].tolist()==frame[c].tolist()
    a=internal_tickets(saved);counts=np.stack([(a==s).sum(0) for s in range(3)],1)
    start=time.perf_counter();reproduced=build(frame,official,REFERENCE);aseconds=time.perf_counter()-start
    np.testing.assert_array_equal(a,internal_tickets(reproduced[0]));validate(a,counts)
    for sym in range(3):np.testing.assert_array_equal(counts[:,sym],(internal_tickets(official)==sym).sum(0))
    frame.to_csv(OUT/'forecast_input.csv',index=False)
    saved.to_csv(OUT/'A_saved.csv',index=False)
    modes={'A_current':a};audits={'A_current':dict(seconds=aseconds,reproduced_saved_exactly=True,audit=reproduced[3])}
    print('A reproduced',flush=True)
    b,ba=optimize_b(p,a);modes['B_exact13']=b;audits['B_exact13']=ba
    print('B complete',mass(p,a).sum(),mass(p,b).sum(),ba['seconds'],flush=True)
    evaluator=ExactPortfolio(p)
    for k in [12,11]:
        for retain in [0.99,0.95]:
            name=f'C_{k}_retain{int(retain*100)}'
            t,audit=optimize_c(p,b,evaluator.mass,k,retain)
            modes[name]=t;audits[name]=audit
            print(name,'complete',audit['seconds'],len(audit['trace'])-1,flush=True)
    summaries=[];details=[];audits_counts=[];sections=[];draw_rows=[]
    draw_dist=np.array([1.0])
    for pd_ in p[:,1]:draw_dist=np.convolve(draw_dist,[1-pd_,pd_])
    for name,t in modes.items():
        validate(t,counts);started=time.perf_counter();metrics=evaluator.metrics(t);evaluation_seconds=time.perf_counter()-started
        # Independent exact neighborhood-union implementation versus full 3^13 enumeration.
        for k in [11,12]:np.testing.assert_allclose(coverage(evaluator.mass,t,13-k),metrics[f'p{k}'],rtol=1e-12)
        np.testing.assert_allclose(mass(p,t).sum(),metrics['p13'],rtol=1e-12)
        row=dict(mode=name,p13=metrics['p13'],p12=metrics['p12'],p11=metrics['p11'],p13_relative_B=metrics['p13']/mass(p,b).sum(),D_total=int((t==1).sum()),D_counts='/'.join(map(str,(t==1).sum(1))),duplicates=0,mean_shared_symbols=13-metrics['mean_distance'],max_shared_symbols=13-metrics['min_distance'],counts_match=True,optimization_seconds=audits[name]['seconds'],evaluation_seconds=evaluation_seconds,actual_best=None if actual is None else int((t==actual).sum(1).max()),actual_mean=None if actual is None else float((t==actual).sum(1).mean()))
        summaries.append(row)
        for k in range(14):
            joint=float(mass(p,t)[(t==1).sum(1)==k].sum())
            draw_rows.append(dict(mode=name,actual_D_count=k,model_probability_D_count=float(draw_dist[k]),ticket_count=int(((t==1).sum(1)==k).sum()),p13_joint=joint,p13_given_D_count=joint/draw_dist[k]))
        grid=frame[['match_no','home_team','away_team']].copy();grid['reference00']=['HDA'[v] for v in p.argmax(1)]
        # Sort only for presentation; row labels do not enter B/C objectives.
        order=np.argsort(-mass(p,t),kind='stable');view=t[order]
        if name=='A_current':view=t
        row['D_counts']='/'.join(map(str,(view==1).sum(1)))
        pair_shared=np.array([int((u==v).sum()) for u,v in combinations(t,2)])
        row['pairs_shared_11plus']=int((pair_shared>=11).sum())
        row['pairs_shared_0']=int((pair_shared==0).sum())
        for j,ticket in enumerate(view,1):
            grid[f'{j:02}']=['HDA'[v] for v in ticket]
            details.append(dict(mode=name,candidate=f'{j:02}',symbols=''.join('HDA'[v] for v in ticket),D_count=int((ticket==1).sum()),exchanges=int((ticket!=p.argmax(1)).sum()),p13=float(mass(p,ticket[None])[0])))
        grid.to_csv(OUT/f'{name}_tickets.csv',index=False)
        for m in range(13):
            for sym,label in enumerate('HDA'):audits_counts.append(dict(mode=name,match_no=m+1,symbol=label,expected=int(counts[m,sym]),actual=int((t[:,m]==sym).sum())))
        sections.append('<h2>'+name+'</h2>'+table(grid))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(draw_rows).to_csv(OUT/'draw_count_coverage.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'candidate_diagnostics.csv',index=False)
    pd.DataFrame(audits_counts).to_csv(OUT/'counts_audit.csv',index=False)
    display=summary[['mode','p13','p12','p11','p13_relative_B','D_total','mean_shared_symbols','optimization_seconds']].copy()
    for c in ['p13','p12','p11']:display[c]=display[c].map(lambda v:f'{100*v:.6f}%')
    body='<p>第1670回のみの比較試作。Aは保存済み現行v2.5を再現。全方式で元H/D/A予測と試合別枚数を固定し、完全重複を禁止。00は参照用で枚数を消費しません。予測・購入記録・現行実装は変更していません。</p>'
    body+='<h2>比較結果</h2>'+table(display)
    body+='<p>p13/p12/p11は、10口の少なくとも1口がそれぞれ13/12以上/11以上的中するモデル確率。13試合独立を仮定し、全3^13=1,594,323結果を列挙して重複なく集計。別実装の近傍集合和でも11・12以上を照合しました。券ごとの確率を単純に足すのは、完全一致で事象が排他的な13的中だけです。</p>'
    body+='<h2>改善点と残る偏り</h2><p>Bでは13的中確率がAの約2.07倍となり、11・12以上も改善しました。しかしD38枠は一部の券へ強く集中し、全13試合Dの券も出ています。Cもこの集中を大きく解消していません。これはD係数の作用ではなく、固定枚数の下で強い記号を一部の券へ集めることで確率の合計を上げる探索結果です。BにはD2〜10本の券が1口もなく、実際のDが2〜10試合なら13的中は不可能です。D総数が38枠のままでもこの偏りが発生します。モデル確率の改善だけで、購入構成として妥当だとは判断しません。</p>'
    body+=table(summary[['mode','D_counts','pairs_shared_11plus','pairs_shared_0']])
    body+='<p>45ペアの平均共通記号数は固定枚数から決まり、どの配置でも同じです。平均類似度を改善指標にはできません。11記号以上一致するペア数、全く一致しないペア数、各券D数などの分布で違いを確認します。</p>'
    body+='<h2>目的と探索</h2><p>Aは交換数段階などの現行ペナルティ。Bは相異なる10券の元予測確率の積を合計した値を直接最大化。1試合交換と任意2試合同時交換を全列挙する改善探索を、24初期値で実施（現行A、固定シードのランダム列配置、暫定最良配置の6回交換による摂動）。1初期値160反復上限。乱数シード・各探索履歴はaudit.jsonに記録。大域最適の保証はありません。</p>'
    body+='<p>CはBを出発点に、Bの13的中確率の99%または95%を下限として、12以上または11以上の集合和確率を最大化。恣意的な加重和を使わず、犠牲にできる13的中確率を明示した4条件です。同じ1・2試合同時交換を全探索し、16反復上限。対象確率が同率なら13的中確率が高い方を選択します。新たなD係数・Q・段階構造・類似度罰則は使いません。</p>'
    body+='<p>完全結果空間を候補券とする整数計画は最大159万の二値変数になるため、今回の初回比較では採用せず、多初期値局所探索と計算時間を検証しました。したがってBは最大化目的の探索結果であり、数学的に証明された最大値ではありません。</p>'
    body+='<h2>実結果と採用判断</h2><p>第1670回の確定実結果ファイルがないため、最高・平均的中数は未評価です。固定枚数が同じなら、実結果に対する平均的中数は配置に依存せず同じになります。今回のモデル確率の改善を実際の的中率改善とは呼びません。まず本回の配置を確認し、その後に過去開催比較を実施する段階で、標準設定への反映は行いません。</p>'
    body+='<p>類似度は45組の券ペアの共通記号数で表示（高いほど似ている）。B/Cの番号は表示用の券確率降順で、最適化条件ではありません。似た券が増えたことだけで悪化とは判断せず、11・12・13の集合和確率を併記しています。</p>'
    body+=''.join(sections)
    for path in sorted(OUT.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    (OUT/'index.html').write_text(page('第1670回：固定枚数A・B・Cの13的中最適化比較',body))
    assert all(sha(path)==h for path,h in hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(protected_sha256=hashes,modes=audits,actual_available=actual is not None,global_optimum_guaranteed=False,production_changed=False),ensure_ascii=False,indent=2))
    print(summary.to_string(index=False),flush=True);print(OUT/'index.html',flush=True)

if __name__=='__main__':main()
