"""Read-only retrospective diagnosis; no parameter fitting or ticket changes."""
import json
import numpy as np
import pandas as pd
from simulate_free_structural_draw import ROOT, page, table, sha


def main():
    source=ROOT/'data/reports/gated_structural_draw'
    out=ROOT/'data/reports/draw_gate_1664_audit'
    out.mkdir(exist_ok=True)
    paths=[source/'eligibility.csv',source/'all_allocations.csv',ROOT/'data/eval/toto_rounds/toto1664/actual_results.csv']
    hashes={str(p):sha(p) for p in paths}
    x=pd.read_csv(paths[0]); x=x[(x['round']=='toto1664')&(x.gate=='gate_near60')].copy().reset_index(drop=True)
    actual=pd.read_csv(paths[2]).sort_values('match_no')
    assert x.match_no.tolist()==actual.match_no.tolist()
    for c in ['home_team','away_team']: assert x[c].tolist()==actual[c].tolist()
    x['match']=x.home_team+'－'+x.away_team
    x['actual_score']=[f'{int(h)}–{int(a)}' for h,a in zip(actual.home_score,actual.away_score)]
    # Exact 2–2 mass under the same independent Poisson score model; not 2–2-plus.
    x['P_22']=x.P_00*(x.lambda_home*x.lambda_away)**2/4
    assert (x.P_22<=x.P_22plus+1e-12).all()
    x['Q_rank']=x.Q_corrected.rank(ascending=False,method='min').astype(int)
    x.to_csv(out/'match_features.csv',index=False)
    cols=['match_no','match','actual','actual_score','P_H','P_D','P_A','full_win_balance','low_total_le2','P_00','P_11','P_22','favoured_one_goal_share','Q_corrected','Q_rank','eligible','reason']
    display=x[cols].copy()
    pct=['P_H','P_D','P_A','full_win_balance','low_total_le2','P_00','P_11','P_22','favoured_one_goal_share']
    for c in pct: display[c]=display[c].map(lambda z:f'{100*z:.2f}%')
    display=display.rename(columns={'match':'試合','actual_score':'実スコア','full_win_balance':'拮抗度B','low_total_le2':'総得点≤2','favoured_one_goal_share':'優勢側勝利中の1点差割合','Q_corrected':'Q','Q_rank':'Q順位','reason':'判定理由'})
    groups=x.assign(group=np.where(x.actual.eq('D'),'実際D（4試合）','実際非D（9試合）')).groupby('group')[['P_D','full_win_balance','low_total_le2','favoured_one_goal_share','Q_corrected']].mean().reset_index()
    groups.to_csv(out/'group_means.csv',index=False)
    allocations=pd.read_csv(paths[1]); allocations=allocations[allocations['round'].eq('toto1664')]
    rows=[]
    for (gate,alpha),a in allocations.groupby(['gate','strength']):
        a=a.sort_values('match_no'); scores=a[['score_H','score_D','score_A']].to_numpy()
        best=np.prod(scores.max(axis=1))
        tickets=pd.read_csv(source/f'toto1664/{gate}/alpha_{alpha:.1f}/candidates.csv')
        cutoff=tickets.loc[~tickets.reference_only,'placement_product_not_probability'].min()
        for i in range(13):
            forced=best*scores[i,1]/scores[i].max()
            rows.append(dict(gate=gate,alpha=alpha,match_no=i+1,match=x.loc[i,'match'],actual=x.loc[i,'actual'],eligible=bool(a.iloc[i].eligible),D_count=int(a.iloc[i].count_D),D_score=scores[i,1],best_HA_score=max(scores[i,0],scores[i,2]),best_ticket_with_this_D=forced,top10_cutoff=cutoff,relative_to_cutoff=forced/cutoff,below_cutoff=bool(forced<cutoff*(1-1e-12))))
            if forced<cutoff*(1-1e-12): assert a.iloc[i].count_D==0
    selection=pd.DataFrame(rows); selection.to_csv(out/'selection_diagnosis.csv',index=False)
    focus=selection[(selection.gate=='gate_near60')&(selection.alpha==2)&selection.actual.eq('D')]
    body='<p>既存の予測・係数・選別条件・購入券は変更していません。第1664回を既知の結果に照らして調べる探索的診断です。</p>'
    body+='<h2>実際にDだった4試合</h2>'+table(display[x.actual.eq('D')])
    body+='<p>横浜FM－東京Vは低得点条件を満たしますが、B≥0.90・優勢側1点差割合≥0.60の両方に未達。他の3試合は低得点条件L≥0.50に未達です。全4試合で両λ≤1の0–0最頻条件も満たしていません。</p>'
    body+='<h2>Dではなかった9試合</h2>'+table(display[~x.actual.eq('D')])
    body+='<h2>結果別の平均（0〜1尺度）</h2>'+table(groups)
    body+='<p>この開催では高いQの試合が非Dに偏りました。ただし13試合だけでQの一般的な有効性を否定できません。実際の2–2は低得点条件外でも生じ、FC東京－柏はモデルの低得点確率が45.36%でも実際には0–0でした。予測上の傾向と実現した1試合の結果は区別する必要があります。</p>'
    body+='<h2>上位10口に入らない理由（厳しい条件・α=2）</h2>'+table(focus)
    body+='<p>各試合のDを固定し、他の12試合を配券評価最大にした「そのDを含む最強券」を計算しました。4試合ともその評価が10位券未満です。Dは禁止されていなくても、積の上位10組だけを選ぶ方式では1口も残りません。全3条件×全5強度×13試合の診断をCSVに保存しています。</p>'
    body+='<h2>評価式の問題と結論</h2><p>Q=L×[B+(1−B)N]。しかし選別は0–0最頻、またはL≥0.50かつ(B≥0.90またはN≥0.60)という別の判定です。Qが連続的に評価した根拠が、選別境界をわずかに下回ると補正ゼロになります。0–0最頻条件も、0–0の可能性そのものを判定する条件ではありません。</p><p>今回の結果は選別条件の限界と上位10組選択の集中性を示します。非固定方式全般の否定ではありません。一方、この開催の結果だけに合わせた閾値変更も行いません。拮抗・低得点などが元のD確率を超える識別力を持つかは、別の開催でも検証が必要です。</p>'
    body+='<h2>指標の定義</h2><p>B=2min(H決着確率,A決着確率)/(H決着確率+A決着確率)。得点分布からの値を使用し、100%が完全拮抗です。Nは優勢側が勝つ場合の1点差割合で、試合途中からDへ移る確率ではありません。0–0/1–1/2–2は保存λから再構成した得点分布の確率です。元H/D/Aには不確実性補正があるため、同点スコア確率の和と元P(D)は必ずしも一致しません。</p>'
    for f in ['match_features.csv','group_means.csv','selection_diagnosis.csv']: body+=f'<p><a href="{f}">{f}</a></p>'
    (out/'index.html').write_text(page('第1664回：Dの評価条件と上位10口選択の診断',body))
    assert all(sha(p)==h for p,h in hashes.items())
    (out/'audit.json').write_text(json.dumps({'unchanged_inputs':hashes,'parameters_changed':False},indent=2))
    print(groups.to_string(index=False)); print(focus[['match','relative_to_cutoff','D_count']].to_string(index=False)); print(out/'index.html')

if __name__=='__main__': main()
