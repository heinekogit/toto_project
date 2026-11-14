"""Strict no-D-only purchase specifications, same frozen probabilities and 10 lines."""
import json,time,hashlib
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from strict_multi_search import TYPES,COST,POWERS,enumerate_blocks,all_patterns,solve_mixed,validate_purchase,expand
from audit_draw_portfolio import ExactPortfolio
from draw_count_cover_shadow import draw_distribution
from simulate_free_structural_draw import ROOT,PROB_COLUMNS,page,table,structure

OUT=ROOT/'data/reports/toto1670_strict_multi'
TYPE_LABEL={'D1':'1ダブル','D2':'2ダブル','D3':'3ダブル','T1':'1トリプル','TD':'1トリプル＋1ダブル','T2':'2トリプル'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pattern_name(counts,singles):return '_'.join([f'{k}x{v}' for k,v in counts.items() if v]+[f'S{singles}'])
def pattern_label(counts,singles):
    return ' ＋ '.join([f'{TYPE_LABEL[k]}のマルチ{v}組（{COST[k]*v}口）' for k,v in counts.items() if v]+([f'シングル{singles}口'] if singles else []))
def read_grid(path):
    g=pd.read_csv(path).sort_values('match_no').reset_index(drop=True)
    return g,np.array([['HDA'.index(s) for s in g[f'{j:02}']] for j in range(1,11)])
def masks_text(masks):return ['/'.join('HDA'[s] for s in range(3) if int(mask)&(1<<s)) for mask in masks]


def main():
    OUT.mkdir(exist_ok=True);base=ROOT/'data/purchase_reference';old=ROOT/'data/reports/toto1670_multi_draw_shadow'
    forecast=ROOT/'data/reports/toto1670_fixed_exact13/forecast_input.csv'
    protected=[forecast]+[path for d in [base,old] for path in d.glob('*') if path.is_file()]
    hashes={str(path):sha(path) for path in protected}
    f=pd.read_csv(forecast).sort_values('match_no').reset_index(drop=True);p=f[PROB_COLUMNS].to_numpy()
    manifest=json.loads((base/'manifest.json').read_text());assert manifest['round_id']=='toto1670'
    np.testing.assert_array_equal(p,pd.read_csv(base/'next_predictions.csv').sort_values('match_no')[PROB_COLUMNS].to_numpy())
    assert manifest['output_sha256']['shadow_buyplan_v25.csv']==sha(base/'shadow_buyplan_v25.csv')
    ha,hp,ha_order,pools,exact,searchstats=enumerate_blocks(p,pool_size=256)
    solutions={};labels={};records=[]
    patterns=all_patterns()
    for counts,singles in sorted(patterns,key=lambda cs:(sum(cs[0].values()),cs[1],pattern_name(*cs))):
        name=pattern_name(counts,singles);labels[name]=pattern_label(counts,singles)
        if sum(counts.values())==1:
            kind=next(k for k,v in counts.items() if v);solution=exact[kind]
        else:solution=solve_mixed(counts,singles,ha,hp,ha_order,pools,searchstats,time_limit=3.)
        solutions[name]=solution
        records.append(dict(mode=name,structure=labels[name],blocks=counts,singles=singles,**{k:solution.get(k) for k in ['p13','status','strict_optimal','seconds','omitted_upper_bound','pool_upper_bound','full_upper_bound','pool_gap']}))
        print(name,solution['status'],solution.get('p13'),flush=True)
    pd.DataFrame(records).to_csv(OUT/'search_results.csv',index=False)
    if any('p13' not in s for s in solutions.values()):print('Some mixed pools did not yield a feasible solution; see search audit.',flush=True)
    modes={};purchase={};labels['C95']='現行C95（今回条件外の参考）'
    cg=pd.read_csv(base/'shadow_buyplan_v25.csv').sort_values('match_no').reset_index(drop=True)
    for c in ['match_no','home_team','away_team']:assert cg[c].tolist()==f[c].tolist()
    raw=cg[[f'ticket{j:02}' for j in range(1,11)]].to_numpy(dtype=int).T
    c95=np.where(raw==1,0,np.where(raw==0,1,2));modes['C95']=c95
    purchase['C95']=[dict(name=f'単独{j+1:02}',cost=1,masks=(1<<row).tolist()) for j,row in enumerate(c95)]
    old_details=json.loads((old/'audit.json').read_text())['details']
    for key,label in [('3Doubles_plus_2Singles','前回3ダブル＋2シングル'),('1Triple_1Double_plus_4Singles','前回トリプル＋ダブル＋4シングル')]:
        g,t=read_grid(old/f'{key}_tickets.csv')
        for c in ['match_no','home_team','away_team']:assert g[c].tolist()==f[c].tolist()
        name='old_'+key;modes[name]=t;labels[name]=label+'（D単独を含む参考）';detail=old_details[key]
        anchor=c95[detail['base_candidate']-1];masks=(1<<anchor).astype(int)
        ds=detail.get('double_matches',[detail.get('double_match')]);ts=[] if 'triple_match' not in detail else [detail['triple_match']]
        for m in ds:masks[m-1]|=2
        for m in ts:masks[m-1]=7
        block=expand(masks)
        # Old script used explicit option order for triples, so compare sets.
        assert set(map(tuple,block))==set(map(tuple,t[:len(block)]))
        purchase[name]=[dict(name='マルチ1',cost=len(block),masks=masks.tolist())]+[dict(name=f'単独{j+1}',cost=1,masks=(1<<row).tolist()) for j,row in enumerate(t[len(block):])]
    for name,s in solutions.items():
        if 'p13' not in s:continue
        t=validate_purchase(s,ha,p);modes[name]=t
        purchase[name]=[dict(name=f'マルチ{j+1}',cost=b['cost'],masks=b['masks']) for j,b in enumerate(s['blocks'])]+[dict(name=f'単独{j+1}',cost=1,masks=(1<<ha[bid]).tolist()) for j,bid in enumerate(s['singles'])]
    evaluator=ExactPortfolio(p);dist=draw_distribution(p);baseline=float(np.prod(p[np.arange(13),c95],axis=1).sum())
    summaries=[];spec_rows=[];line_rows=[];concentrations=[];block_failures=[];sections=[]
    feat=structure(f,'corrected');context=pd.concat([f[['match_no','home_team','away_team']],feat[['P_H','P_D','P_A','lambda_home','lambda_away','P_00','P_11','P_margin1']]],axis=1);context.to_csv(OUT/'match_forecast_context.csv',index=False)
    for name,t in modes.items():
        assert t.shape==(10,13) and len(np.unique(t,axis=0))==10
        metrics=evaluator.metrics(t);pm=np.prod(p[np.arange(13),t],axis=1)
        np.testing.assert_allclose(metrics['p13'],pm.sum(),rtol=1e-12)
        specs=purchase[name];expanded=np.vstack([expand(b['masks']) for b in specs]);assert set(map(tuple,expanded))==set(map(tuple,t));assert sum(b['cost'] for b in specs)==10
        compliant=all(mask in [1,4,3,6,7] for b in specs for mask in b['masks'])
        if name in solutions:assert compliant
        dc=(t==1).sum(1);counts=np.stack([(t==s).sum(0) for s in range(3)],1);ks=sorted(set(map(int,dc)))
        summaries.append(dict(mode=name,label=labels[name],new_D_rule_met=compliant,p13=metrics['p13'],C95_improvement_percent=(metrics['p13']/baseline-1)*100,p12=metrics['p12'],p11=metrics['p11'],price_yen=1000,D_total=int(dc.sum()),D_counts='/'.join(map(str,dc)),D_count_coverage=float(dist[ks].sum()),all10_fixed_matches=int((counts.max(1)==10).sum()),mean_pair_shared_symbols=13-metrics['mean_distance'],max_same_prediction_count=int(counts.max()),strict_optimal=solutions[name]['strict_optimal'] if name in solutions else None,selection_seconds=solutions[name]['seconds'] if name in solutions else None))
        spec_grid=f[['match_no','home_team','away_team']].copy();grid=spec_grid.copy()
        for j,b in enumerate(specs,1):
            spec_grid[f"{b['name']}（{b['cost']}口）"]=masks_text(b['masks'])
            for m,mask in enumerate(b['masks']):
                spec_rows.append(dict(mode=name,specification=j,kind=b['name'],expanded_lines=b['cost'],match_no=m+1,selection=masks_text([mask])[0],D_singleton=mask==2))
                for s,label in enumerate('HDA'):
                    if not mask&(1<<s):block_failures.append(dict(mode=name,specification=j,match_no=m+1,actual_result=label,all_block_lines_lost=b['cost']))
        for j,row in enumerate(t,1):
            grid[f'{j:02}']=['HDA'[s] for s in row];line_rows.append(dict(mode=name,line=j,symbols=''.join('HDA'[s] for s in row),D_count=int(dc[j-1]),p13=float(pm[j-1])))
        spec_grid.to_csv(OUT/f'{name}_purchase.csv',index=False);grid.to_csv(OUT/f'{name}_expanded.csv',index=False)
        cr=[]
        for m in range(13):
            r=dict(mode=name,match_no=m+1,home_team=f.loc[m,'home_team'],away_team=f.loc[m,'away_team'],H_count=int(counts[m,0]),D_count=int(counts[m,1]),A_count=int(counts[m,2]),if_H_lost=10-int(counts[m,0]),if_D_lost=10-int(counts[m,1]),if_A_lost=10-int(counts[m,2]),all10_fixed=bool(counts[m].max()==10));concentrations.append(r);cr.append(r)
        sections.append('<h2 id="'+name+'">'+labels[name]+'</h2><h3>購入指定</h3>'+table(spec_grid)+'<h3>展開後10口</h3>'+table(grid)+'<h3>各試合で13的中の可能性を失う口数</h3><p>if_H/D/A_lostは、その結果になったとき13的中を失う口数。11・12的中も失うという意味ではありません。</p>'+table(pd.DataFrame(cr).drop(columns='mode')))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(spec_rows).to_csv(OUT/'purchase_specifications.csv',index=False);pd.DataFrame(line_rows).to_csv(OUT/'expanded_lines.csv',index=False)
    pd.DataFrame(concentrations).to_csv(OUT/'match_failure_exposure.csv',index=False);pd.DataFrame(block_failures).to_csv(OUT/'block_failure_exposure.csv',index=False)
    required=['D3x1_S2','TDx1_S4','D2x1_S6','D1x1_S8']
    shown=summary.copy()
    for c in ['p13','p12','p11','D_count_coverage']:shown[c]=shown[c].map(lambda v:f'{100*v:.6f}%')
    compact=shown[['label','new_D_rule_met','p13','C95_improvement_percent','p12','p11','D_counts','all10_fixed_matches','strict_optimal']]
    main_names=['C95','old_3Doubles_plus_2Singles','old_1Triple_1Double_plus_4Singles']+required
    new=summary[summary.new_D_rule_met];winner=new.sort_values('p13',ascending=False).iloc[0]
    upper=max([s.get('full_upper_bound',s.get('p13',float('inf'))) for s in solutions.values()])
    all_certified=all('p13' in s for s in solutions.values()) and upper<=winner.p13+1e-14
    body='<p>第1670回の購入前保存予測で比較。今回の新案は、Dを含む購入指定をH/D・D/A・H/D/Aに限定し、Dだけの指定を禁止。マルチ外の単独券は全てH/Aのみ。全組み合わせへ展開して重複なし10口・1,000円です。現行C95・正式BuyPlan・保存予測を変更せず、実結果は使用していません。</p>'
    body+='<h2>指定4構成と従来比較</h2>'+table(compact[shown['mode'].isin(main_names)])
    body+='<p>前回の2案は追加シングルにD単独を含むため、今回の条件を満たしません。C95も参考比較です。今回の目的は13的中確率で、D本数の均等化や最低枚数の条件は追加していません。指定された各構成にはDを含むマルチが存在します。</p>'
    body+=f'<h2>その他の構成も含めた探索</h2><p>最大10口に入るD入りマルチは、1/2/3ダブル、1/2トリプル、1トリプル＋1ダブルの6種類。これらを1組以上使う全30構成を比較し、残りをH/A単独で10口まで埋めました。最良案は「{winner.label}」。候補プール外の上界も含む全30構成の最良性確認：{all_certified}。</p>'+table(compact[shown.new_D_rule_met])
    body+='<h2>厳密最適と近似探索</h2><p>マルチ1組＋シングルの6構成は、マルチ対象試合・H/DまたはD/A・HDA・残り全試合H/Aの全指定を列挙しました。追加単独券も、マルチと重複しないH/A券の確率上位から選択。土台を候補00やC95の数券だけに制限していません。この6構成は保存確率・独立仮定の下で厳密最適です。</p><p>マルチ複数組の24構成は、種類別に確率上位256ブロックの候補集合を作り、展開後の重複を禁止した整数計画で探索（各3秒上限）。プール外のブロックを1つ以上使う解の確率上界も計算。候補内最適と上界で証明できた構成のみstrict_optimal=Trueと表示し、それ以外は近似探索です。全条件と上界をsearch_results.csvに記録しました。数値判定許容差は1e-14です。</p>'
    body+='<h2>確率・得点構造と購入の集中</h2><p>選択の評価は保存済みH/D/A確率の積です。得点構造はその元となるモデル値を下表に併記し、Qによる未検証の加点は使っていません。マルチ入力そのものに確率的優位性はなく、同じ展開10口を単独購入すれば確率は同じです。</p>'+table(context)
    body+='<p>13的中確率は10個の相異なる結果事象の確率合計。11・12以上は全159万通りを列挙して重複なく計算。all10_fixed_matchesは全10口が同じ1記号に固定された試合数です。各試合でH/D/Aが出たとき同時に何口が13的中を失うかを、方式ごとの表とCSVに記録しました。D入りのマルチを使っても、その外の固定試合を外せばブロック全体が13的中を失います。</p>'
    body+=''.join(sections)
    for path in sorted(OUT.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    (OUT/'index.html').write_text(page('第1670回：D単独指定なし・マルチ購入構造の比較',body))
    assert all(sha(path)==digest for path,digest in hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(searchstats=searchstats,solutions=solutions,protected_sha256=hashes,winner=winner['mode'],overall_optimal_certified=bool(all_certified),overall_upper_bound=upper,production_changed=False),ensure_ascii=False,indent=2))
    print(compact[shown['mode'].isin(main_names)].to_string(index=False),flush=True)
    print('BEST',winner['mode'],winner.p13,'overall_certified',all_certified,flush=True)
    print(OUT/'index.html',flush=True)

if __name__=='__main__':main()
