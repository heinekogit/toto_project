"""Offline B vs B-D1 vs B-D2 using the frozen preceding comparison input."""
import json
import time
from itertools import combinations
import numpy as np
import pandas as pd
from fixed_draw_bounds_shadow import optimize,proposed_bounds
from fixed_exact13_shadow import mass,validate,coverage
from audit_draw_portfolio import ExactPortfolio
from simulate_free_structural_draw import ROOT,PROB_COLUMNS,sha,page,table

OUT=ROOT/'data/reports/toto1670_draw_bounds'


def main():
    OUT.mkdir(exist_ok=True);base=ROOT/'data/reports/toto1670_fixed_exact13'
    protected=[p for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()]
    protected += [p for p in base.glob('*') if p.is_file()]
    protected += [ROOT/'scripts/next_model'/name for name in ['score_placement_v25_shadow.py','shadow_v25_operational.py','fixed_exact13_shadow.py']]
    hashes={str(p):sha(p) for p in protected}
    frame=pd.read_csv(base/'forecast_input.csv').sort_values('match_no').reset_index(drop=True)
    p=frame[PROB_COLUMNS].to_numpy();grid=pd.read_csv(base/'B_exact13_tickets.csv')
    for c in ['match_no','home_team','away_team']:assert frame[c].tolist()==grid[c].tolist()
    b=grid[[f'{i:02}' for i in range(1,11)]].replace({'H':0,'D':1,'A':2}).to_numpy(dtype=int).T
    counts=np.stack([(b==s).sum(0) for s in range(3)],1);validate(b,counts);assert counts[:,1].sum()==38
    old_summary=pd.read_csv(base/'comparison.csv');old_b=old_summary[old_summary['mode'].eq('B_exact13')].iloc[0]
    np.testing.assert_allclose(mass(p,b).sum(),old_b.p13,rtol=1e-12)
    modes={'B':b};audits={'B':dict(seconds=float(old_b.optimization_seconds),source='preserved preceding B; not regenerated')};start_rows=[]
    for name,lo,hi in [('B-D1',2,5),('B-D2',1,6)]:
        seeds=[251670,351670,451670];all_runs=[];best=None;bestval=-1
        for seed in seeds:
            t,audit=optimize(p,counts,lo,hi,starts=64,cap=240,seed=seed)
            value=mass(p,t).sum();all_runs.append(audit)
            for run in audit['runs']:start_rows.append(dict(mode=name,seed=seed,start=run['start'],p13=run['final'],moves=run['moves'],three_ticket_cycles=run['three_ticket_cycles'],local_stop=run['local_stop']))
            if value>bestval:best=t.copy();bestval=value
            print(name,seed,'p13',value,'seconds',audit['seconds'],flush=True)
        modes[name]=best;audits[name]=dict(seconds=sum(a['seconds'] for a in all_runs),lower=lo,upper=hi,seed_runs=all_runs,global_optimum_guaranteed=False)
    evaluator=ExactPortfolio(p);summary=[];cand=[];count_rows=[];sections=[];strata=[]
    drawdist=np.array([1.])
    for pd_ in p[:,1]:drawdist=np.convolve(drawdist,[1-pd_,pd_])
    for name,t in modes.items():
        validate(t,counts);metrics=evaluator.metrics(t)
        for k in [11,12]:np.testing.assert_allclose(metrics[f'p{k}'],coverage(evaluator.mass,t,13-k),rtol=1e-12)
        np.testing.assert_allclose(metrics['p13'],mass(p,t).sum(),rtol=1e-12)
        order=np.argsort(-mass(p,t),kind='stable');t=t[order];d=(t==1).sum(1)
        if name!='B':assert (d>=audits[name]['lower']).all() and (d<=audits[name]['upper']).all()
        row=dict(mode=name,p13=metrics['p13'],p12=metrics['p12'],p11=metrics['p11'],retention_percent=100*metrics['p13']/old_b.p13,D_total=int(d.sum()),D_counts='/'.join(map(str,d)),duplicates=0,counts_match=True,optimization_seconds=audits[name]['seconds'],actual_best=None,actual_mean=None)
        summary.append(row)
        v=frame[['match_no','home_team','away_team']].copy();v['reference00']=['HDA'[s] for s in p.argmax(1)]
        diag=[]
        for j,ticket in enumerate(t,1):
            v[f'{j:02}']=['HDA'[s] for s in ticket]
            r=dict(mode=name,candidate=f'{j:02}',symbols=''.join('HDA'[s] for s in ticket),D_count=int((ticket==1).sum()),p13=float(mass(p,ticket[None])[0]),exchanges=int((ticket!=p.argmax(1)).sum()))
            diag.append(r);cand.append(r)
        v.to_csv(OUT/f'{name}_tickets.csv',index=False)
        for m in range(13):
            for symbol,label in enumerate('HDA'):count_rows.append(dict(mode=name,match_no=m+1,symbol=label,expected=int(counts[m,symbol]),actual=int((t[:,m]==symbol).sum())))
        for k in range(14):
            joint=float(mass(p,t)[d==k].sum())
            strata.append(dict(mode=name,actual_D_count=k,model_D_count_probability=float(drawdist[k]),tickets=int((d==k).sum()),p13_joint=joint,p13_given_D_count=joint/drawdist[k]))
        sections.append('<h2>'+name+'：10口</h2>'+table(v)+table(pd.DataFrame(diag)))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(cand).to_csv(OUT/'candidate_diagnostics.csv',index=False)
    pd.DataFrame(count_rows).to_csv(OUT/'counts_audit.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'draw_count_coverage.csv',index=False)
    pd.DataFrame(start_rows).to_csv(OUT/'search_starts.csv',index=False)
    general=pd.DataFrame([dict(total_D=n,mean_D=n/10,D1_lower=proposed_bounds(n,1)[0],D1_upper=proposed_bounds(n,1)[1],D2_lower=proposed_bounds(n,2)[0],D2_upper=proposed_bounds(n,2)[1]) for n in [0,5,10,20,30,38,50,80,120,130]])
    general.to_csv(OUT/'proposed_general_bounds.csv',index=False)
    shown=summary[['mode','p13','p12','p11','retention_percent','D_counts','optimization_seconds']].copy()
    for c in ['p13','p12','p11']:shown[c]=shown[c].map(lambda v:f'{v*100:.6f}%')
    body='<p>第1670回の前回Bと同じ保存予測・固定H/D/A枚数で比較。元確率・現行v2.4/v2.5・保存券は変更していません。Bは前回の保存結果を使用し、新方式で都合よく再探索した基準ではありません。</p>'
    body+='<h2>比較</h2>'+table(shown)
    body+='<p>13/12/11以上は10口の少なくとも1口が達成する確率。試合間独立の下で全3^13結果を列挙し、重複を除いて集計。11/12以上は別の近傍集合和実装とも照合。維持率は前回Bの13的中確率に対する比率です。確定実結果は使用せず、実績改善とは評価しません。</p>'
    body+='<h2>今回の結果の解釈</h2><p>B-D1はBの13的中確率を61.97%維持（38.03%低下）、B-D2は79.27%維持（20.73%低下）。11・12以上もBより低下しました。3つの固定シードはいずれも同じ最良確率に到達しましたが、これは大域最適の証明ではありません。</p><p>B-D1はD2枚が4口・5枚が6口。B-D2はD1枚が4口・5枚が2口・6枚が4口です。どちらもD3枚・4枚の券がなく、実際のDが3〜4試合なら13的中できません。制約違反ではなく、指定された範囲内で確率の高い券へ集中した結果です。</p><p>今回の目的を「極端な11〜13枚集中をなくしつつBの確率をできるだけ残す」と限定すれば、B-D2が比較継続の候補です。ただし20.73%のモデル確率低下があり、D本数の空白も残るため、そのまま採用を勧める結果ではありません。D3〜4も含めた構成を求めるなら、別の制約の検討が必要です。今回は指定外の本数割当は追加していません。</p>'
    body+='<h2>制約と探索</h2><p>B-D1は各券D2〜5枚、B-D2は1〜6枚。候補別の具体的枚数・番号順は指定せず、H/Aへの均等化もなし。いずれも相異なる10券の13的中確率の和だけを最大化します。D総数38、試合別枚数、10口の一意性を監査。00は購入対象外です。</p><p>初期実現可能性を390二値変数の整数計画で確認。重複券が出た場合は該当ペアの重複を禁止する制約を追加して再求解。探索は3固定シード×64初期配置（計192）、各240反復上限。ランダム配置をD範囲へ修復した初期解と、暫定最良解への10回の合法交換による摂動を併用します。</p><p>各反復では1試合交換・任意2試合同時交換・同一試合で3券間の循環交換を全列挙し、合法かつ最も13的中確率が増える操作を選択。全探索履歴を保存。局所最適の可能性は残り、大域最適を断定しません。計算時間には実現可能性確認を含みます。Bの時間は前回24初期値の探索時間なので、探索予算の違いがあります。</p>'
    body+='<h2>他開催への暫定提案</h2><p>μ=D総数/10として、下限=max(0,floor(μ−w))、上限=min(13,ceil(μ+w))。w=1をD1相当、w=2をD2相当として比較する案です。38枚ならそれぞれ2〜5、1〜6になります。幅・丸め方に予測上の裏付けはなく正式値ではありません。各開催で整数計画により固定枚数と一意性を含む実現可能性を確認し、不可能なら枚数を勝手に変更せず報告します。</p>'+table(general)
    body+='<h2>判断上の注意</h2><p>範囲制約は極端なD集中を防ぎますが、範囲内の全D本数をカバーする保証はありません。たとえばD2〜5という制約でも、3本・4本の券が生成されるとは限りません。D本数別カバレッジCSVも確認してください。幅を狭めた分のモデル確率低下と、避けたい集中の程度を比べる試験であり、制約適合自体を予測改善とは呼びません。</p>'
    body+=''.join(sections)
    for path in sorted(OUT.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    (OUT/'index.html').write_text(page('第1670回：D枚数制約付き13的中最適化',body))
    assert all(sha(path)==h for path,h in hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(modes=audits,protected_sha256=hashes,production_changed=False,general_bounds_adopted=False),ensure_ascii=False,indent=2))
    print(shown.to_string(index=False),flush=True);print(OUT/'index.html',flush=True)

if __name__=='__main__':main()
