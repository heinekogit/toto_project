"""Exact unrestricted top ten versus saved v2.4 and operational C95, toto1670."""
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from audit_draw_portfolio import ExactPortfolio
from simulate_free_portfolio_1664 import top_joint_candidates
from draw_count_cover_shadow import draw_distribution
from simulate_free_structural_draw import ROOT,PROB_COLUMNS,page,table

OUT=ROOT/'data/reports/toto1670_unrestricted_top10'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_tickets(plan,frame):
    plan=plan.sort_values('match_no').reset_index(drop=True)
    for c in ['match_no','home_team','away_team']:assert plan[c].tolist()==frame[c].tolist(),c
    for c in PROB_COLUMNS:
        if c in plan:np.testing.assert_allclose(plan[c],frame[c],rtol=1e-13,atol=1e-15)
    raw=plan[[f'ticket{j:02}' for j in range(1,11)]].to_numpy(dtype=int).T
    if not np.isin(raw,[0,1,2]).all():raise ValueError('Unknown ticket symbol')
    return np.where(raw==1,0,np.where(raw==0,1,2))


def main():
    OUT.mkdir(parents=True,exist_ok=True);base=ROOT/'data/purchase_reference'
    frozen=ROOT/'data/reports/toto1670_fixed_exact13/forecast_input.csv'
    protected=[frozen]+[path for path in base.glob('*') if path.is_file()];hashes={str(path):sha(path) for path in protected}
    manifest=json.loads((base/'manifest.json').read_text());assert manifest['round_id']=='toto1670' and manifest['official_version']=='v2.4'
    for name in ['next_predictions.csv','buyplan.csv','shadow_buyplan_v25.csv','shadow_v25_audit.json']:
        assert manifest['output_sha256'][name]==sha(base/name),name
    sa=json.loads((base/'shadow_v25_audit.json').read_text());assert sa['mode']=='C95'
    frame=pd.read_csv(frozen).sort_values('match_no').reset_index(drop=True)
    current=pd.read_csv(base/'next_predictions.csv').sort_values('match_no').reset_index(drop=True)
    for c in ['match_no','home_team','away_team']:assert frame[c].tolist()==current[c].tolist()
    p=frame[PROB_COLUMNS].to_numpy();np.testing.assert_array_equal(p,current[PROB_COLUMNS].to_numpy())
    assert p.shape==(13,3) and np.isfinite(p).all() and (p>0).all();np.testing.assert_allclose(p.sum(1),1)
    started=time.perf_counter();top,pm=top_joint_candidates(p,10);selection_seconds=time.perf_counter()-started
    ev=ExactPortfolio(p)
    # Independent complete-outcome verification, including deterministic tie order.
    best_ids=np.lexsort((np.arange(len(ev.mass)),-ev.mass))[:10]
    np.testing.assert_array_equal(top,ev.outcomes[best_ids]);np.testing.assert_allclose(pm,ev.mass[best_ids],rtol=1e-14)
    modes={'v2.4':saved_tickets(pd.read_csv(base/'buyplan.csv'),frame),'v2.5_C95':saved_tickets(pd.read_csv(base/'shadow_buyplan_v25.csv'),frame),'unrestricted_top10':top}
    dist=draw_distribution(p);summary=[];candidates=[];allocations=[];strata=[];sections=[]
    labels={'v2.4':'正式v2.4 保存券','v2.5_C95':'現行v2.5 C95 保存券','unrestricted_top10':'制約なし・13的中確率上位10口'}
    for name,t in modes.items():
        unique=np.unique(t,axis=0);individual=np.prod(p[np.arange(13),t],axis=1);metrics=ev.metrics(t)
        unique_mass=np.prod(p[np.arange(13),unique],axis=1).sum();np.testing.assert_allclose(metrics['p13'],unique_mass,rtol=1e-12)
        dc=(t==1).sum(1);ks=sorted(set(map(int,dc)))
        summary.append(dict(mode=name,label=labels[name],p13=metrics['p13'],relative_unrestricted_percent=100*metrics['p13']/pm.sum(),p12=metrics['p12'],p11=metrics['p11'],D_total=int(dc.sum()),D_counts='/'.join(map(str,dc)),covered_D_counts='/'.join(map(str,ks)),D_count_coverage=float(dist[ks].sum()),duplicate_lines=10-len(unique),minimum_ticket_p13=float(individual.min()),maximum_ticket_p13=float(individual.max()),expected_average_hits=float(p[np.arange(13),t].sum()/10)))
        grid=frame[['match_no','home_team','away_team']].copy();grid['reference00']=['HDA'[s] for s in p.argmax(1)]
        for j,row in enumerate(t,1):
            grid[f'{j:02}']=['HDA'[s] for s in row]
            candidates.append(dict(mode=name,candidate=f'{j:02}',symbols=''.join('HDA'[s] for s in row),D_count=int(dc[j-1]),exchanges=int((row!=p.argmax(1)).sum()),p13=float(individual[j-1]),contribution=float(individual[j-1]/metrics['p13'])))
        grid.to_csv(OUT/f'{name}_tickets.csv',index=False)
        sections.append('<h2>'+labels[name]+'</h2>'+table(grid))
        for m in range(13):allocations.append(dict(mode=name,match_no=m+1,home_team=frame.loc[m,'home_team'],away_team=frame.loc[m,'away_team'],P_H=p[m,0],P_D=p[m,1],P_A=p[m,2],count_H=int((t[:,m]==0).sum()),count_D=int((t[:,m]==1).sum()),count_A=int((t[:,m]==2).sum())))
        for k in range(14):
            joint=float(np.prod(p[np.arange(13),unique],axis=1)[(unique==1).sum(1)==k].sum())
            strata.append(dict(mode=name,D_count=k,model_probability=float(dist[k]),tickets=int((dc==k).sum()),p13_joint=joint,p13_given_D_count=joint/dist[k]))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/'candidate_diagnostics.csv',index=False)
    pd.DataFrame(allocations).to_csv(OUT/'match_allocations.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'draw_count_coverage.csv',index=False)
    pd.DataFrame(p,columns=['P_H','P_D','P_A']).assign(match_no=frame.match_no).to_csv(OUT/'forecast_probabilities.csv',index=False)
    display=summary[['label','p13','p12','p11','D_total','D_counts','D_count_coverage','relative_unrestricted_percent']].copy()
    for c in ['p13','p12','p11','D_count_coverage']:display[c]=display[c].map(lambda v:f'{100*v:.6f}%')
    body='<p>第1670回の同じ保存予測で比較。正式v2.4・現行C95は保存券をそのまま評価。制約なし方式は試合別枚数、D係数、D本数カバー、マルチ構造、候補番号の優先度を使わず、相異なる13試合の結果組み合わせを確率上位から10口選択します。00は参照用で購入対象外。01が00と一致する場合も、購入分は01の1口のみです。</p>'
    body+='<h2>比較結果</h2>'+table(display)
    body+='<p>13的中は10口の少なくとも1口が13試合すべて一致するモデル確率。相異なる券の事象は排他的なので確率を合計できます。11・12以上は重なる結果を二重計上せず、全3^13=1,594,323結果上で集計しました。D本数カバー率は本数一致の範囲であり、的中確率ではありません。</p>'
    body+='<h2>厳密最適である範囲</h2><p>各試合独立と保存確率を前提にすると、相異なる10通りの確率合計の最大値は上位10通りの和です。段階的な上位候補計算で生成し、全159万通りを別に列挙・並べ替えて、10口そのものが一致することを検査しました。今回の制約なし方式は局所探索ではなく、このモデル上の厳密最適です。現実の的中率や予測確率の正しさを保証する意味ではありません。</p>'
    body+='<h2>Dの扱いと比較条件</h2><p>制約なし上位10口のDは全て0枚でした。Dを禁止したのではなく、Dを含む組み合わせが上位10位以内に入らなかった結果です。モデルではDが0試合の確率は約1.4245%、1試合以上は約98.5755%。制約なし券で13的中できるのはD0試合の結果だけです。</p><p>この上位10口と、過去比較の「B」は別です。過去Bは各試合のH/D/A枚数固定でした。本比較の制約なし方式は枚数固定を解除しています。v2.4とC95の枚数は一致しますが、制約なし方式の枚数は一致させていません。</p>'
    body+='<p>実結果は生成・評価に使用していません。現行の予測・購入券・C95シャドーは更新せず、比較用ファイルだけを出力しています。</p>'
    body+=''.join(sections)
    diag=pd.DataFrame(candidates);diag['p13']=diag.p13.map(lambda v:f'{100*v:.8f}%');diag['contribution']=diag.contribution.map(lambda v:f'{100*v:.4f}%')
    body+='<h2>各券の13的中確率</h2>'+table(diag)
    for path in sorted(OUT.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    (OUT/'index.html').write_text(page('第1670回：制約なし上位10口・v2.4・C95比較',body))
    assert all(sha(Path(path))==digest for path,digest in hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(protected_sha256=hashes,round='toto1670',forecast_identical_to_current=True,complete_enumeration_verified=True,outcomes=3**13,selection_seconds=selection_seconds,model_global_optimum=True,production_changed=False),ensure_ascii=False,indent=2))
    print(display.to_string(index=False));print(pd.DataFrame(candidates)[pd.DataFrame(candidates)['mode']=='unrestricted_top10'][['candidate','symbols','D_count']].to_string(index=False));print(OUT/'index.html')

if __name__=='__main__':main()
