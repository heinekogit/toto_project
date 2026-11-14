"""Reproduce the existing structural placement-score experiment, not D probability probes."""
import json
import numpy as np
import pandas as pd
from simulate_free_structural_draw import ROOT, PROB_COLUMNS, structure, placement_scores, metrics, candidates, page, table, sha
from simulate_free_portfolio_1664 import top_joint_candidates
from simulate_gated_structural_draw import Gate, eligibility


def main():
    out=ROOT/'data/reports/toto1670_placement_score_tickets';out.mkdir(exist_ok=True)
    source=ROOT/'data/reports/draw_proximity_fix/corrected/toto1670/forecast_input.csv'
    protected=[source]+[p for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected}
    f=pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
    assert f.match_no.tolist()==list(range(1,14))
    p=f[PROB_COLUMNS].to_numpy(); original=p.copy(); feat=structure(f,'corrected')
    gate,reasons=eligibility(feat,Gate())
    features=pd.concat([f[['match_no','home_team','away_team']],feat],axis=1)
    features['gate_near60_eligible']=gate;features['gate_reason']=reasons
    features.to_csv(out/'features.csv',index=False)
    summaries=[]; sections=[]
    for name,mask in [('ungated',np.ones(13,dtype=bool)),('gate_near60',gate)]:
        for alpha in (1.,1.5,2.):
            scores=placement_scores(p,feat.Q_corrected.to_numpy()*mask,alpha)
            np.testing.assert_array_equal(scores[:,[0,2]],p[:,[0,2]])
            np.testing.assert_array_equal(scores[~mask,1],p[~mask,1])
            tickets,_=top_joint_candidates(scores,10)
            diag=candidates(f,p,tickets,scores,None)
            old=pd.read_csv(ROOT/f'data/reports/gated_structural_draw/toto1670/{name}/alpha_{alpha:.1f}/candidates.csv')
            assert diag.symbols.tolist()==old.symbols.tolist()
            result,strata=metrics(p,tickets,None)
            summaries.append(dict(mode=name,alpha=alpha,**result))
            key=f'{name}_alpha_{alpha:.1f}'
            grid=pd.DataFrame({'No':f.match_no,'試合':f.home_team+'－'+f.away_team,'参照00':['HDA'[v] for v in p.argmax(1)]})
            for i,t in enumerate(tickets,1):grid[f'{i:02}']=['HDA'[v] for v in t]
            grid.to_csv(out/f'{key}_tickets.csv',index=False)
            diag.to_csv(out/f'{key}_diagnostics.csv',index=False)
            per_match=f[['match_no','home_team','away_team']].copy()
            for k,symbol in enumerate('HDA'):
                per_match['P_'+symbol]=p[:,k];per_match['score_'+symbol]=scores[:,k];per_match['count_'+symbol]=(tickets==k).sum(0)
            per_match.to_csv(out/f'{key}_scores.csv',index=False)
            pd.DataFrame(strata).to_csv(out/f'{key}_draw_strata.csv',index=False)
            title=('全試合へ試合別Qを適用' if name=='ungated' else '既存の対象選別あり')+f'／α={alpha}'
            sections.append(f'<h2 id="{key}">{title}</h2>'+table(grid)+table(diag[['candidate','reference_only','D_count','exchanges','rank2','rank3','original_P13','placement_product_not_probability']]))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'comparison.csv',index=False)
    display=summary[['mode','alpha','D_total','D_counts','P_any13']].copy();display.P_any13=display.P_any13.map(lambda v:f'{100*v:.6f}%')
    body='<p>元のH/D/A予測確率を変更せず、検討済みのD配券スコアで作成した非固定式10口。00は購入対象外の参照券です。前回の学習D確率による10口とは異なります。</p>'
    body+='<p>配券評価：H=P(H)、A=P(A)、D=P(D)+α×Q×max(max(P(H),P(A))−P(D),0)。Q=L×[B+(1−B)N]。Lは総得点2以下（0–0を含む）、Bは得点分布全体のH/A拮抗度、Nは優勢側が勝つ場合の1点差割合です。0–0はLに含まれますが独立の加点項ではありません。1–1・2–2は診断値で、別加点にはしていません。</p>'
    body+='<p>配券評価の積が大きい10組を厳密に選択します。D確率の補正学習は使いません。ただし「評価の積の上位10組」という選択の集中性は未解決です。新たな分散最適化を実装したものではありません。αも選別条件も従来値のままです。</p>'
    body+='<h2>比較</h2>'+table(display)
    body+='<p>α=1.0/1.5ではDが2/3枠に留まり、α=2.0では69枠に増えます。配券スコアの反映は確認できますが、極端な配置になる問題は残っています。対象選別ありも既存条件の再現であり、過去1664回の実際のDを全て対象外にした問題があるため推奨ではありません。</p>'
    body+='<p>13的中モデル確率は、全設定共通の元予測確率を使用し、試合間独立の下で重複のない10券の確率を合計。配券評価は確率ではなく、13的中確率の計算に使用していません。第1670回の実結果は選択にも評価にも使用していません。</p>'
    body+=''.join(sections)
    for path in sorted(out.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    (out/'index.html').write_text(page('第1670回：検討済みD配券スコアによる10口',body))
    np.testing.assert_array_equal(p,original)
    assert all(sha(path)==h for path,h in hashes.items())
    (out/'audit.json').write_text(json.dumps({'protected_sha256':hashes,'all_six_match_previous_simulations':True,'probabilities_changed':False,'production_changed':False,'selection':'exact top10 product of placement scores'},ensure_ascii=False,indent=2))
    print(display.to_string(index=False))
    print(pd.read_csv(out/'ungated_alpha_1.0_tickets.csv').to_string(index=False))
    print(out/'index.html')

if __name__=='__main__':main()
