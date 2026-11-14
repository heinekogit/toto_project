"""C95/C90/C80 D-count coverage and shared probability perturbations."""
import json
import numpy as np
import pandas as pd
from draw_count_cover_shadow import optimize,draw_distribution,lookup_table,signature,better
from fixed_exact13_shadow import mass,validate,coverage
from audit_draw_portfolio import ExactPortfolio
from simulate_free_structural_draw import ROOT,PROB_COLUMNS,sha,page,table

OUT=ROOT/'data/reports/toto1670_draw_count_cover'


def chart(dist,modes):
    width=1000;height=570;left=75;barw=54;colors=['#555555','#277da1','#f8961e','#43aa8b']
    chunks=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>','<g font-family="sans-serif" font-size="15" fill="#203147">','<text x="40" y="28">toto1670: draw-count probability and covered counts (not winning probability)</text>']
    for k,v in enumerate(dist):
        x=left+k*barw;h=v*1000
        chunks.extend([f'<rect x="{x}" y="{300-h}" width="40" height="{h}" fill="#b9d5e8"/>',f'<text x="{x+20}" y="320" text-anchor="middle">{k}</text>',f'<text x="{x+20}" y="{292-h}" text-anchor="middle" font-size="11">{100*v:.2f}%</text>'])
    chunks.append('<text x="830" y="320">D count</text>')
    for j,(name,t) in enumerate(modes.items()):
        y=365+j*43;counts=set((t==1).sum(1));cov=sum(dist[k] for k in counts)
        chunks.append(f'<text x="8" y="{y+19}">{name}</text>')
        for k in range(14):chunks.append(f'<rect x="{left+k*barw}" y="{y}" width="40" height="27" fill="{colors[j] if k in counts else "#eeeeee"}"/>')
        chunks.append(f'<text x="850" y="{y+19}">{100*cov:.2f}%</text>')
    return ''.join(chunks)+ '</g></svg>'


def main():
    OUT.mkdir(exist_ok=True);base=ROOT/'data/reports/toto1670_fixed_exact13'
    protected=[p for d in [base,ROOT/'data/purchase_reference'] for p in d.glob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected}
    f=pd.read_csv(base/'forecast_input.csv').sort_values('match_no').reset_index(drop=True);p=f[PROB_COLUMNS].to_numpy()
    g=pd.read_csv(base/'B_exact13_tickets.csv')
    for c in ['match_no','home_team','away_team']:assert f[c].tolist()==g[c].tolist()
    b=np.array([['HDA'.index(s) for s in g[f'{i:02}']] for i in range(1,11)])
    counts=np.stack([(b==s).sum(0) for s in range(3)],1);dist=draw_distribution(p);lookup=lookup_table(dist);bp=float(mass(p,b).sum())
    old=pd.read_csv(base/'comparison.csv');np.testing.assert_allclose(bp,old.loc[old['mode'].eq('B_exact13'),'p13'].iloc[0],rtol=1e-12)
    modes={'B':b};audits={'B':dict(seconds=float(old.loc[old['mode'].eq('B_exact13'),'optimization_seconds'].iloc[0]))};warm=None;stability=[]
    for name,retain in [('C95',.95),('C90',.90),('C80',.80)]:
        best=b.copy();bc=lookup[signature(b)];bestp=bp;seed_runs=[]
        for seed in [951670,1951670,2951670]:
            t,audit,pool=optimize(p,b,retain,seed=seed,warm=warm)
            c=lookup[signature(t)];v=mass(p,t).sum();seed_runs.append(audit)
            stability.append(dict(mode=name,seed=seed,coverage=c,p13=v,distinct_initial_D_histograms=len(set(tuple(sorted(r['initial_D_counts'])) for r in audit['runs']))))
            if better(c,v,bc,bestp):best=t.copy();bc=c;bestp=v
            print(name,seed,'coverage',c,'retention',v/bp,'seconds',audit['seconds'],flush=True)
        modes[name]=best;warm=best.copy();audits[name]=dict(seconds=sum(r['seconds'] for r in seed_runs),p13_floor=bp*retain,seed_runs=seed_runs)
    evaluator=ExactPortfolio(p);summaries=[];details=[];checks=[];sections=[]
    dv=pd.DataFrame({'D_count':range(14),'probability':dist})
    for name,t in modes.items():
        validate(t,counts);metrics=evaluator.metrics(t)
        for k in [11,12]:np.testing.assert_allclose(metrics[f'p{k}'],coverage(evaluator.mass,t,13-k),rtol=1e-12)
        np.testing.assert_allclose(metrics['p13'],mass(p,t).sum(),rtol=1e-12)
        if name!='B':assert metrics['p13']>=audits[name]['p13_floor']-1e-16
        t=t[np.argsort(-mass(p,t),kind='stable')];modes[name]=t;ds=(t==1).sum(1);ks=sorted(set(map(int,ds)))
        dv[name+'_tickets']=[int((ds==k).sum()) for k in range(14)];dv[name+'_covered']=dv[name+'_tickets']>0
        summaries.append(dict(mode=name,p13=metrics['p13'],retention_percent=100*metrics['p13']/bp,p12=metrics['p12'],p11=metrics['p11'],D_count_coverage=float(dist[ks].sum()),covered_counts='/'.join(map(str,ks)),D_counts='/'.join(map(str,ds)),D_total=int(ds.sum()),counts_match=True,duplicates=0,optimization_seconds=audits[name]['seconds']))
        v=f[['match_no','home_team','away_team']].copy();v['reference00']=['HDA'[s] for s in p.argmax(1)]
        for j,ticket in enumerate(t,1):
            v[f'{j:02}']=['HDA'[s] for s in ticket];details.append(dict(mode=name,candidate=f'{j:02}',symbols=''.join('HDA'[s] for s in ticket),D_count=int((ticket==1).sum()),p13=float(mass(p,ticket[None])[0])))
        v.to_csv(OUT/f'{name}_tickets.csv',index=False);sections.append('<h2>'+name+'</h2>'+table(v))
        for m in range(13):
            for s,label in enumerate('HDA'):checks.append(dict(mode=name,match_no=m+1,symbol=label,expected=int(counts[m,s]),actual=int((t[:,m]==s).sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'comparison.csv',index=False);dv.to_csv(OUT/'draw_distribution.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'candidate_diagnostics.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'counts_audit.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'search_stability.csv',index=False)
    # Shared Dirichlet perturbations, E[p_perturbed]=p exactly; no selection/refit.
    rng=np.random.default_rng(81670);perturb=[];draws=20000
    for concentration in [1000,250,100]:
        pp=np.stack([rng.dirichlet(concentration*row,size=draws) for row in p],axis=1)
        np.testing.assert_allclose(pp.sum(2),1,atol=1e-14)
        probabilities={}
        for name,t in modes.items():
            values=sum(np.prod(pp[:,np.arange(13),ticket],axis=1) for ticket in t);probabilities[name]=values
        for name,values in probabilities.items():
            ratios=values/probabilities['B']
            perturb.append(dict(concentration=concentration,scenarios=draws,mode=name,mean_p13=float(values.mean()),median_p13=float(np.median(values)),p05_p13=float(np.quantile(values,.05)),mean_ratio_B=float(ratios.mean()),median_ratio_B=float(np.median(ratios)),p05_ratio_B=float(np.quantile(ratios,.05)),fraction_beating_B=float((values>probabilities['B']+1e-18).mean()),min_component_sd=float(np.sqrt(p*(1-p)/(concentration+1)).min()),max_component_sd=float(np.sqrt(p*(1-p)/(concentration+1)).max())))
    perturb=pd.DataFrame(perturb);perturb.to_csv(OUT/'perturbation.csv',index=False)
    svg=chart(dist,modes);(OUT/'draw_coverage.svg').write_text(svg)
    # Shortest contiguous interval with >=90% mass, tie-break by mass then lower endpoint.
    intervals=[(hi-lo,-float(dist[lo:hi+1].sum()),lo,hi) for lo in range(14) for hi in range(lo,14) if dist[lo:hi+1].sum()>=.9]
    _,neg,lo,hi=min(intervals)
    facts=dict(expected_D=float(p[:,1].sum()),modal_D=int(dist.argmax()),central90=[int(np.searchsorted(dist.cumsum(),.05)),int(np.searchsorted(dist.cumsum(),.95))],shortest90=[lo,hi],shortest90_mass=-neg)
    shown=summary.copy()
    for c in ['p13','p12','p11','D_count_coverage']:shown[c]=shown[c].map(lambda z:f'{100*z:.6f}%')
    body='<p>第1670回・元予測と試合別H/D/A枚数を固定。D総数38、完全重複なし。現行予測・購入券は変更していません。ここでのC95/C90/C80はD発生数カバー方式であり、以前の11/12的中最適化Cとは別方式です。</p>'
    body+='<h2>比較</h2>'+table(shown)
    body+=f'<h2>D発生数分布</h2><p>期待値{facts["expected_D"]:.6f}、最頻値{facts["modal_D"]}。確率90%以上を含む最短の連続範囲は{lo}〜{hi}試合（{-neg*100:.4f}%）。各試合のP(D)からポアソン二項分布を畳み込みで厳密計算。試合間独立を仮定します。</p>'
    body+='<p><b>色付きのkは「そのD本数を持つ券が存在する」だけです。Dとなる試合の位置やH/Aまで一致する確率ではありません。</b></p>'+svg+table(dv)
    body+='<h2>今回の判断</h2><p>C95は13的中確率を95.19%維持してD本数カバー率87.54%、C90は90.58%維持して91.69%、C80は81.29%維持して99.05%となりました。C95の13的中確率低下は約4.81%、12以上は約3.19%、11以上は約1.37%です。いずれもD3枚・4枚の候補が生成されました。ただし、該当D数の全結果を的中できる意味ではありません。</p><p>全3段階の確率摂動で、C方式の平均・中央値・下位5%の13的中確率はBを上回りませんでした。C95がBを上回るシナリオの割合はκ=1000/250/100の順に1.43%/13.07%/23.60%。一部の変動で有利になる可能性はありますが、今回の仮定で全般的な耐性改善を確認したとは言えません。</p><p>比較を継続するなら、モデル確率の犠牲が最も小さいC95が候補です。カバー率増加だけを採用理由にはせず、正式採用は保留します。C90/C80は追加のモデル確率低下に見合う利点が今回の摂動で確認できていません。実績評価や系統的な予測誤差を含む検証は別途必要です。</p>'
    body+='<h2>探索条件</h2><p>前回Bの丸め前の13的中確率を基準に、95/90/80%下限をハード制約としています。カバー率を主目的、同率なら13的中確率を副目的とする辞書式比較。Bを含む3シード×32初期配置、各240反復上限。初期値はBまたは暫定最良解から、下限を満たす交換だけを用いた15〜30回のランダムウォークで生成。後続の緩い条件には先の厳しい条件の解も渡します。乱数は探索初期値用で、単にランダム券を採用するものではありません。</p><p>1試合交換、2試合同時交換、3候補間の同一試合循環交換を全列挙。全ての採用解で固定枚数・一意性・13的中下限を検査。D本数の上下限やグループ制約はなし。探索予算内の最良解で、大域最適の保証はありません。B自身が各条件の実現可能解なので、「条件を満たす解がない」とはなりません。改善できる最大カバー率の不可能性も証明していません。</p>'+table(pd.DataFrame(stability))
    body+='<h2>確率摂動への感度</h2><p>各試合独立のDirichlet(κ×元HDA)から20,000シナリオを生成。κ=1000/250/100、固定seed=81670。同じシナリオを全方式に適用し、券は固定、再最適化なし。期待確率は元の値と一致し、各試合の合計は必ず1。標準偏差はsqrt(p(1−p)/(κ+1))でCSVに記録。平均・中央値・下位5%・同じシナリオでBを上回る割合を集計しました。</p><p>この独立・平均一致の摂動では、各券の確率の積の期待値も元と一致します。従って平均13的中確率の逆転を期待する設定ではなく、変動と下位側の感度を見る検査です。共通のD過小評価などの系統誤差は再現せず、実際の予測精度改善を示すものではありません。</p>'+table(perturb)
    body+=''.join(sections)
    for path in sorted(OUT.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    body+='<p><a href="draw_coverage.svg">分布・カバー範囲グラフSVG</a></p>'
    (OUT/'index.html').write_text(page('第1670回：D発生数カバー C95・C90・C80',body))
    assert all(sha(path)==h for path,h in hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(modes=audits,protected_sha256=hashes,distribution=facts,perturbation_seed=81670,production_changed=False),ensure_ascii=False,indent=2))
    print(shown[['mode','p13','retention_percent','p12','p11','D_count_coverage','covered_counts','D_counts']].to_string(index=False),flush=True);print(facts,flush=True);print(OUT/'index.html',flush=True)

if __name__=='__main__':main()
