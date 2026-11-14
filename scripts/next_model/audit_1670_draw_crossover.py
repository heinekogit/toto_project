"""Analytic crossover audit of the existing placement formula; no tuning."""
import json
import numpy as np
import pandas as pd
from simulate_free_structural_draw import ROOT, sha, page, table, placement_scores, structure, PROB_COLUMNS
from simulate_free_portfolio_1664 import top_joint_candidates


def main():
    out=ROOT/'data/reports/toto1670_draw_crossover';out.mkdir(exist_ok=True)
    source=ROOT/'data/reports/draw_proximity_fix/corrected/toto1670/forecast_input.csv'
    protected=[source,ROOT/'scripts/next_model/simulate_free_structural_draw.py']+[p for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected}
    f=pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
    p=f[PROB_COLUMNS].to_numpy();s=structure(f,'corrected');q=s.Q_corrected.to_numpy()
    best=p[:,[0,2]].max(1); gap=np.maximum(best-p[:,1],0);slope=q*gap
    # This audit input has D strictly below the strongest H/A in every match.
    assert (gap>0).all() and (q>0).all()
    alpha=gap/slope
    np.testing.assert_allclose(alpha,1/q,rtol=1e-14)
    x=f[['match_no','home_team','away_team']].copy()
    x['P_H']=p[:,0];x['P_D']=p[:,1];x['P_A']=p[:,2];x['best_HA']=np.where(p[:,0]>=p[:,2],'H','A')
    x['best_HA_P']=best;x['gap_percentage_points']=100*gap
    x['L']=s.low_total_le2;x['B']=s.full_win_balance;x['N']=s.favoured_one_goal_share
    x['C']=s.balance_or_one_goal_rating;x['Q']=q;x['D_score_increment_per_alpha']=slope
    x['alpha_tie_H']=(p[:,0]-p[:,1])/slope;x['alpha_tie_A']=(p[:,2]-p[:,1])/slope
    x['alpha_tie_best_HA']=alpha;x['D_score_at_tie']=best
    x['relative_D_score_increase_at_tie']=gap/p[:,1]
    x['D_leads_at_alpha2']=alpha<2
    for i,a in enumerate(alpha):
        np.testing.assert_allclose(placement_scores(p,q,float(a))[i,1],best[i],atol=1e-14)
        assert placement_scores(p,q,float(a-1e-7))[i,1]<best[i]
        assert placement_scores(p,q,float(a+1e-7))[i,1]>best[i]
    x.to_csv(out/'thresholds.csv',index=False)
    rows=[]
    # Evaluate only infinitesimally around analytically derived events, not an alpha grid search.
    for i,a in enumerate(alpha):
        record={'crossing_match_no':i+1,'alpha_tie':a}
        for label,eps in [('below',-1e-7),('above',1e-7)]:
            tickets,_=top_joint_candidates(placement_scores(p,q,float(a+eps)),10)
            record[label+'_D_total']=int((tickets==1).sum())
            record[label+'_D_counts']='/'.join(map(str,(tickets==1).sum(1)))
            record[label+'_crossing_match_D_copies']=int((tickets[:,i]==1).sum())
        rows.append(record)
    events=pd.DataFrame(rows).sort_values('alpha_tie');events.to_csv(out/'threshold_events.csv',index=False)
    display=x[['match_no','home_team','away_team','P_H','P_D','P_A','gap_percentage_points','Q','alpha_tie_best_HA']].copy()
    for c in ['P_H','P_D','P_A']:display[c]=display[c].map(lambda z:f'{z*100:.2f}%')
    display['gap_percentage_points']=display.gap_percentage_points.map(lambda z:f'{z:.2f}')
    for c in ['Q','alpha_tie_best_HA']:display[c]=display[c].map(lambda z:f'{z:.4f}')
    body='<p>既存の第1670回保存予測・試合別Q・補正式をそのまま検査。αの採用値を探索せず、購入券・正式予測は変更していません。対象選別なしの方式です。</p>'
    body+='<h2>原因：元確率差が閾値の式から消える</h2><p>M=max(P(H),P(A))、Δ=M−P(D)>0 と置くと、D評価=P(D)+αQΔ。DとMが同じ評価になる条件はαQΔ=Δ、したがってα*=1/Qです。α=α*で同率、これを上回るとDが単独首位です。対象の13試合すべてがこの条件に該当します。</p><p>元のD確率がH/Aから大きく離れているほど、1単位αあたりの加点も大きくなるため、差が相殺されます。元確率差を考慮しているように見えて、首位へ逆転させる難しさには直接反映されていません。Q自体は得点分布に依存するので、元予測と無関係という意味ではありません。</p>'
    body+='<h2>試合別閾値</h2>'+table(display)
    body+='<p>gap_percentage_pointsは元P(D)と強い方のH/Aとの差（パーセントポイント）。alpha_tie_best_HAは同率になるαで、厳密な逆転にはこれを超える必要があります。個別のH・Aとの同率閾値もCSVに保存しました。</p>'
    leading=x[x.D_leads_at_alpha2]
    body+=f'<h2>密集の数値</h2><p>全13試合のQは{q.min():.6f}〜{q.max():.6f}、逆転境界は{alpha.min():.6f}〜{alpha.max():.6f}。α=1.5では首位Dが0試合、α=2では{len(leading)}試合となります。この{len(leading)}試合の境界は{leading.alpha_tie_best_HA.min():.6f}〜{leading.alpha_tie_best_HA.max():.6f}に集中しています。</p>'
    body+=f'<p>Q=L×C、C=B+(1−B)N。Lの範囲は{x.L.min():.6f}〜{x.L.max():.6f}、Cは{x.C.min():.6f}〜{x.C.max():.6f}です。似た総得点構造と拮抗度からQが近い値を取り、それを1/Qへ変換する補正式によって逆転境界が密集します。</p>'+table(x[['match_no','L','B','N','C','Q']])
    body+='<h2>上位10組選択との関係</h2><p>積を最大にする1組は各試合の評価首位を選んだ券です。Dが複数試合で首位に変わると、最強券を中心とする上位10組も一斉に変わりやすくなります。ただしDが首位でなくても上位10組には入り得ます。首位逆転数とD総枠数は同じではありません。閾値直前直後で必ず大幅な枠数変化が起きるとも限りません。</p>'
    body+='<p>下表は計算上の各境界±0.0000001のみで再評価した診断です。適度なD数を得るためのα調整ではありません。</p>'+table(events)
    body+='<p>今回の13境界すべてで、直前・直後のD総枠数は同じでした。首位の切替そのものが一瞬で大量のDを追加したのではなく、近接する境界の周辺で上位10組が入れ替わり、狭いα範囲で総枠数が増えることを区別してください。</p>'
    body+='<h2>結論</h2><p>急増は「近いQの分布」「元確率差を相殺する加点式」「積の上位10組選択」の組み合わせで説明できます。元確率差が閾値に残る補正式への変更と、10口全体の選択設計は分けて検討すべきです。この診断はQの予測有効性を証明せず、係数採用・正式反映も行いません。</p>'
    for name in ['thresholds.csv','threshold_events.csv']:body+=f'<p><a href="{name}">{name}</a></p>'
    (out/'index.html').write_text(page('第1670回：D配券評価の逆転閾値',body))
    assert all(sha(path)==h for path,h in hashes.items())
    (out/'audit.json').write_text(json.dumps({'protected_sha256':hashes,'alpha_tie_formula':'1/Q when max(PH,PA)>PD and Q>0','production_changed':False,'threshold_validation':'13 ties and both sides verified against existing placement_scores','event_epsilon':1e-7},ensure_ascii=False,indent=2))
    print(display.to_string(index=False));print(events.to_string(index=False));print(out/'index.html')

if __name__=='__main__':main()
