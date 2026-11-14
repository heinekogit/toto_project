"""Offline all-ticket quality analysis and Q1/Q2/Q3 tradeoffs for toto1670."""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ticket_quality_shadow import optimize,quality,objectives,lex_better
from draw_count_cover_shadow import draw_distribution,lookup_table,signature
from fixed_exact13_shadow import mass,validate,coverage
from audit_draw_portfolio import ExactPortfolio
from simulate_free_structural_draw import ROOT,PROB_COLUMNS,sha,page,table

OUT=ROOT/'data/reports/toto1670_ticket_quality'


def graph(modes,p,ranked):
    names=list(modes);height=65+155*len(names);mx=max(mass(p,t).max() for t in modes.values())
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">','<rect width="100%" height="100%" fill="white"/>','<g font-family="sans-serif" fill="#203147" font-size="13">',f'<text x="25" y="25">Individual P(13 hits), common linear scale; {"probability rank" if ranked else "candidate number"}</text>']
    for j,name in enumerate(names):
        y=65+j*155;v=mass(p,modes[name]);v=np.sort(v)[::-1] if ranked else v
        svg.append(f'<text x="10" y="{y+35}">{name}</text>')
        for i,z in enumerate(v):
            h=float(z/mx*100);x=200+i*75
            svg.extend([f'<rect x="{x}" y="{y+105-h}" width="48" height="{h}" fill="#277da1"/>',f'<text x="{x+24}" y="{y+122}" text-anchor="middle">{i+1:02}</text>',f'<text x="{x+24}" y="{y+98-h}" text-anchor="middle" font-size="9">{z*100:.6f}%</text>'])
    return ''.join(svg)+'</g></svg>'


def main():
    OUT.mkdir(exist_ok=True);base=ROOT/'data/reports/toto1670_draw_count_cover';source=ROOT/'data/reports/toto1670_fixed_exact13/forecast_input.csv'
    protected=[source]+[p for d in [base,ROOT/'data/purchase_reference'] for p in d.glob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected};f=pd.read_csv(source).sort_values('match_no').reset_index(drop=True);p=f[PROB_COLUMNS].to_numpy()
    modes={};audits={};lookup=lookup_table(draw_distribution(p))
    for name in ['B','C95','C90','C80']:
        grid=pd.read_csv(base/f'{name}_tickets.csv')
        for c in ['match_no','home_team','away_team']:assert grid[c].tolist()==f[c].tolist()
        modes[name]=np.array([['HDA'.index(s) for s in grid[f'{j:02}']] for j in range(1,11)])
        audits[name]=dict(source='preserved previous results',seconds=0)
    counts=np.stack([(modes['B']==s).sum(0) for s in range(3)],1);bp=mass(p,modes['B']).sum();coverbase=lookup[signature(modes['C95'])]
    specs=[(f'{kind}_{pct}',kind,pct/100,0) for kind in ['Q1','Q2'] for pct in [95,90,80]]
    specs += [(f'Q3_{pct}_cover{cv}', 'Q3',pct/100,coverbase*cv/100) for pct in [95,90,80] for cv in [100,90]]
    stability=[]
    for name,kind,retain,cfloor in specs:
        initial=list(modes.values());best=None;bo=None;seed_audits=[]
        for seed in [171670,271670,371670]:
            t,a=optimize(p,counts,initial,bp*retain,cfloor,kind,seed=seed)
            obj=objectives(mass(p,t),kind);seed_audits.append(a)
            stability.append(dict(mode=name,seed=seed,p13=float(mass(p,t).sum()),minimum=float(mass(p,t).min()),bottom5=float(np.sort(mass(p,t))[:5].sum()),coverage=float(lookup[signature(t)]),local_stops=sum(r['local_stop'] for r in a['runs'])))
            if bo is None or lex_better(obj,bo):best=t.copy();bo=obj
            print(name,seed,'retention',mass(p,t).sum()/bp,'min',mass(p,t).min(),'bottom5',np.sort(mass(p,t))[:5].sum(),flush=True)
        modes[name]=best;audits[name]=dict(seconds=sum(a['seconds'] for a in seed_audits),total_floor=float(bp*retain),coverage_floor=float(cfloor),seed_runs=seed_audits)
    evaluator=ExactPortfolio(p);summaries=[];details=[];checks=[];sections=[];logproduct=None
    for name,t in modes.items():
        validate(t,counts);pm=mass(p,t);q=quality(pm);cov=float(lookup[signature(t)])
        if logproduct is None:logproduct=float(np.log(pm).sum())
        np.testing.assert_allclose(np.log(pm).sum(),logproduct,atol=1e-12)
        metrics=evaluator.metrics(t)
        for k in [11,12]:np.testing.assert_allclose(metrics[f'p{k}'],coverage(evaluator.mass,t,13-k),rtol=1e-12)
        np.testing.assert_allclose(metrics['p13'],pm.sum(),rtol=1e-12)
        summaries.append(dict(mode=name,p13=float(pm.sum()),retention_percent=float(pm.sum()/bp*100),p12=metrics['p12'],p11=metrics['p11'],D_count_coverage=cov,D_counts='/'.join(map(str,(t==1).sum(1))),D_total=int((t==1).sum()),**q,optimization_seconds=audits[name]['seconds'],duplicates=0,counts_match=True))
        ranks=rankdata(-pm,method='min').astype(int)
        view=f[['match_no','home_team','away_team']].copy();view['reference00']=['HDA'[s] for s in p.argmax(1)]
        diag=[]
        for j,row in enumerate(t,1):
            view[f'{j:02}']=['HDA'[v] for v in row]
            r=dict(mode=name,candidate=f'{j:02}',symbols=''.join('HDA'[v] for v in row),D_count=int((row==1).sum()),p13=float(pm[j-1]),contribution=float(pm[j-1]/pm.sum()),probability_rank=int(ranks[j-1]));diag.append(r);details.append(r)
        view.to_csv(OUT/f'{name}_tickets.csv',index=False)
        diag_view=pd.DataFrame(diag)
        diag_view['p13']=diag_view.p13.map(lambda v:f'{v*100:.8f}%')
        diag_view['contribution']=diag_view.contribution.map(lambda v:f'{v*100:.4f}%')
        sections.append('<h2>'+name+'</h2>'+table(view)+table(diag_view))
        for m in range(13):
            for s,label in enumerate('HDA'):checks.append(dict(mode=name,match_no=m+1,symbol=label,expected=int(counts[m,s]),actual=int((t[:,m]==s).sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'candidate_diagnostics.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'counts_audit.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'search_stability.csv',index=False)
    for ranked,label in [(False,'candidate_order'),(True,'probability_order')]: (OUT/f'{label}.svg').write_text(graph(modes,p,ranked))
    shown=summary[['mode','p13','retention_percent','p12','p11','D_count_coverage','minimum','median','bottom5_sum','bottom5_share','D_counts']].copy()
    for c in ['p13','p12','p11','D_count_coverage','minimum','median','bottom5_sum','bottom5_share']:
        precision=8 if c in ['minimum','median','bottom5_sum'] else 6
        shown[c]=shown[c].map(lambda v:f'{100*v:.{precision}f}%')
    body='<p>第1670回のみ。元予測・試合別枚数・D総数38・相異なる10券を維持。00は購入対象外。D本数上下限・均等化・グループ制約は使用せず、正式版や購入記録は変更していません。</p>'
    body+='<h2>全条件の結果</h2>'+table(shown)
    baseline_display=summary[summary['mode'].isin(['B','C95','C90','C80'])][['mode','maximum','minimum','median','top3_sum','top3_share','bottom3_sum','bottom3_share','bottom5_sum','bottom5_share','max_min_ratio']].copy()
    for col in baseline_display.columns:
        if col not in ['mode','max_min_ratio']:baseline_display[col]=baseline_display[col].map(lambda v:f'{v*100:.8f}%')
    body+='<h2>基準4方式の個別確率集計</h2>'+table(baseline_display)
    body+='<h2>C95のどの券が低いのか</h2><p>D0枚の3口は確率上位3口で、合計寄与率74.70%です。低確率ではありません。D10枚・13枚は下位2口で、合計寄与率約0.505%。最高券と最低券の倍率は約102.91倍。下位5口の寄与率は3.7356%、下位3口は1.1094%でした。個別値は方式ごとの診断表とCSVに全件記載しています。</p>'
    body+='<h2>95%維持条件の判断</h2><p>Q1はC95比で最低券を約2.78倍にしましたが、下位5口合計は約8.91%低下し、Dカバー率も28.64%に低下。最低を上げることと下位全体を上げることは一致しません。Q2は下位5口合計を約15.26%増やしましたが、最低券は約4.05%低下し、Dカバー率は32.22%に低下しました。</p><p>Q3_95_cover100はC95のDカバー率87.54%を維持し、下位5口合計を約4.55%改善。全体13的中確率はC95から約0.059%の相対低下に留まりました。ただし最低券は改善せず、D10枚・13枚が残ります。複数目的の比較継続候補としてはこの条件ですが、購入対象全体の格差を解消した方式とは評価しません。</p><p>90%・80%条件も全て併記しています。より合計確率を犠牲にすれば改善できる指標はありますが、合計確率・最低券・下位5券・Dカバーを同時に大幅改善する解は今回見つかっていません。一部条件ではシード間に差が残り、実現不可能と証明したわけではありません。正式採用は保留し、個別券の偏りが残る点を重視します。</p>'
    body+='<h2>比較条件</h2><p>Q1は最低券確率を最大化し、同率なら下位5券合計、さらに同率なら全10券合計を優先。Q2は下位5券合計を最大化し、同率なら最低券、さらに全体合計を優先。いずれもB比95/90/80%を合計13的中確率のハード下限に設定。</p><p>Q3はQ2と同じ目的に、C95の丸め前のD発生数カバー率の100%または90%維持というハード下限を追加。カバー率の下限は約87.54%または78.79%です。合計確率の3条件と組み合わせて6条件を比較。90%という低下許容は今回の暫定比較条件で正式値ではありません。重み付き和は使用していません。</p><p>各条件3シード×24初期配置、各180反復上限。B/C95/C90/C80と先行試験解から条件を満たす解を選び、合法交換による12〜23回のランダムウォークでも初期値を多様化。1試合交換・2試合同時交換・3券循環交換を全列挙して辞書式に改善。各シードの結果と履歴を保存。局所最適の可能性があり、最大値の証明ではありません。</p>'
    body+='<p>11/12以上は全3^13結果を列挙して10券のいずれかが達成する事象を重複なく評価し、近傍集合和でも照合。13的中は相異なる券の元確率積の合計です。全てモデル確率で、実績改善を示すものではありません。</p>'
    body+='<h2>固定枚数によるトレードオフ</h2><p>試合別枚数が固定なら、10券の個別確率の積（対数確率の合計）は配置によらず一定です。底上げと上位券の確率維持には構造的なトレードオフがあります。ただし、この事実だけで合計確率が必ず低下するとは言えず、比較表で実際の変化を確認します。</p>'
    body+='<h2>個別券確率の棒グラフ（共通の線形尺度）</h2><p>候補番号順では保存券の番号を維持し、順位順では確率の高い順に並べます。新方式にも候補番号優先はありません。</p><p><a href="candidate_order.svg">候補番号順</a> ／ <a href="probability_order.svg">個別確率順位順</a></p><img src="probability_order.svg" style="max-width:100%;height:auto" alt="Individual ticket probability by rank"/>'
    body+='<h2>探索安定性</h2>'+table(pd.DataFrame(stability))
    body+=''.join(sections)
    for path in sorted(OUT.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    (OUT/'index.html').write_text(page('第1670回：個別券確率とQ1・Q2・Q3の比較',body))
    assert all(sha(path)==h for path,h in hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(modes=audits,protected_sha256=hashes,C95_coverage_baseline=float(coverbase),production_changed=False),ensure_ascii=False,indent=2))
    print(shown.to_string(index=False),flush=True);print(OUT/'index.html',flush=True)

if __name__=='__main__':main()
