"""Phase 3: mathematical lambda sensitivity of archived prematch score grids."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure import score_grid, structure, adjacency
from compare_d_scoop_uncertainty import ROOT, sha

SOURCE = ROOT / 'data/reports/score_structure_phase12/match_structure_78.csv'
OUT = ROOT / 'data/reports/score_structure_phase3'
SCENARIOS = {
    'H低下': (-1, 0), 'H上昇': (1, 0), 'A低下': (0, -1), 'A上昇': (0, 1),
    '双方低下': (-1, -1), '双方上昇': (1, 1),
    'H低下・A上昇': (-1, 1), 'H上昇・A低下': (1, -1),
}
MEASURES = ('P_0_0', 'P_1_1', 'P_2_2', 'P_3_3', 'P_draw_4plus', 'P_draw',
            'P_home_1', 'P_away_1', 'P_home_2', 'P_away_2',
            'P_home_3plus', 'P_away_3plus', 'P_margin_1', 'P_margin_2plus',
            'P_draw_2plus')


def compute(base: pd.DataFrame, *, theoretical: bool = False):
    rows=[];edges=[];matrices={};skipped=[]
    for _,r in base.iterrows():
        identity=dict(round=str(r['round']),match_no=int(r.match_no),
                      home_team=str(r.home_team),away_team=str(r.away_team))
        h0,a0=float(r.lambda_home),float(r.lambda_away)
        baseline,retained=score_grid(h0,a0)
        original=structure(baseline)
        for name,dh,da,delta in [('基準',0,0,0.0)]+[
            (name,v[0],v[1],delta) for delta in (.1,.2) for name,v in SCENARIOS.items()]:
            h,a=h0+dh*delta,a0+da*delta
            if h<=0 or a<=0:
                skipped.append(dict(**identity,scenario=name,delta=delta,lambda_home=h,lambda_away=a,
                                    reason='nonpositive_lambda'))
                continue
            matrix,ret=score_grid(h,a)
            values=structure(matrix)
            key=(identity['round'],identity['match_no'],name,delta)
            matrices[key]=matrix
            record=dict(**identity,scenario=name,delta=delta,
                        lambda_home_baseline=h0,lambda_away_baseline=a0,
                        lambda_home=h,lambda_away=a,
                        lambda_total=h+a,lambda_abs_diff=abs(h-a),
                        raw_P_H=float(np.tril(matrix,-1).sum()),
                        raw_P_A=float(np.triu(matrix,1).sum()),
                        retained_grid_mass=ret,
                        final_P_D_reference=float(r.get('final_P_D',np.nan)),
                        uncertainty_mix_reference=float(r.get('uncertainty_mix',np.nan)),**values)
            for k in MEASURES:
                record[f'baseline_{k}']=original[k]
                record[f'delta_{k}']=values[k]-original[k]
                record[f'sensitivity_{k}_per_lambda_point']=(values[k]-original[k])/delta if delta else 0.0
            rows.append(record)
            for er in adjacency(matrix,identity['round'],identity['match_no']).to_dict('records'):
                bh,ba=int(er['home_goals']),int(er['away_goals']);lo,hi=min(bh,ba),max(bh,ba)
                er.update(home_team=identity['home_team'],away_team=identity['away_team'],
                          scenario=name,delta=delta,
                          baseline_victory_probability=float(baseline[bh,ba]),
                          delta_victory_probability=float(matrix[bh,ba]-baseline[bh,ba]),
                          baseline_lower_draw_probability=float(baseline[lo,lo]),
                          delta_lower_draw_probability=float(matrix[lo,lo]-baseline[lo,lo]),
                          baseline_upper_draw_probability=float(baseline[hi,hi]),
                          delta_upper_draw_probability=float(matrix[hi,hi]-baseline[hi,hi]),
                          # Descriptive ratio, never a transition probability.
                          adjacent_draw_to_victory_ratio=float((matrix[lo,lo]+matrix[hi,hi])/matrix[bh,ba]),
                          baseline_adjacent_draw_to_victory_ratio=float((baseline[lo,lo]+baseline[hi,hi])/baseline[bh,ba]))
                er['delta_adjacent_draw_to_victory_ratio']=(er['adjacent_draw_to_victory_ratio']-
                                                            er['baseline_adjacent_draw_to_victory_ratio'])
                edges.append(er)
    return pd.DataFrame(rows),pd.DataFrame(edges),matrices,pd.DataFrame(skipped)


def diff_svg(before: np.ndarray, after: np.ndarray, title: str) -> str:
    step=35;left=75;top=68;cells=[]
    for h in range(11):
        for a in range(11):
            delta=float(after[h,a]-before[h,a]);strength=min(1,abs(delta)/.03)
            if delta>=0:fill=f'rgb({int(255-140*strength)},255,{int(255-140*strength)})'
            else:fill=f'rgb(255,{int(255-140*strength)},{int(255-140*strength)})'
            stroke='#3d8070' if h==a else '#4578a9' if abs(h-a)==1 else '#cbd4df'
            cells.append(f'<rect x="{left+a*step}" y="{top+h*step}" width="{step}" height="{step}" fill="{fill}" stroke="{stroke}"/>')
            if abs(delta)>=.001:cells.append(f'<text x="{left+a*step+step/2}" y="{top+h*step+22}" text-anchor="middle" font-size="9">{delta*100:+.1f}</text>')
    ticks=''.join(f'<text x="{left+k*step+step/2}" y="{top-8}" text-anchor="middle" font-size="12">{k}</text><text x="{left-12}" y="{top+k*step+22}" text-anchor="end" font-size="12">{k}</text>' for k in range(11))
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="495" height="505"><rect width="495" height="505" fill="white"/><text x="18" y="24" font-size="15">{escape(title)}</text><text x="210" y="48" font-size="12">アウェイ得点 →</text>{"".join(cells)}{ticks}<text x="75" y="478" font-size="11">緑：増加　赤：減少　枠：同点／1点差／その他　数値：pt</text></svg>'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(SOURCE)
    if len(base)!=78:raise ValueError('Expected 78 frozen prematch matches')
    scenarios,edges,matrices,skipped=compute(base)
    if len(scenarios)!=78*17 or len(edges)!=78*17*20:raise AssertionError('Missing scenario rows')
    original=scenarios[scenarios.scenario.eq('基準')]
    for k in MEASURES:
        np.testing.assert_allclose(original[k],base[k],rtol=0,atol=1e-12)
    np.testing.assert_allclose(scenarios.raw_P_H+scenarios.P_draw+scenarios.raw_P_A,1,rtol=0,atol=1e-12)
    np.testing.assert_allclose(scenarios.P_draw+scenarios.P_margin_1+scenarios.P_margin_2plus,1,rtol=0,atol=1e-12)
    scenarios.to_csv(OUT/'scenario_structure.csv',index=False,encoding='utf-8-sig')
    delta_cols=['round','match_no','home_team','away_team','scenario','delta','lambda_home','lambda_away']+[f'delta_{k}' for k in MEASURES]
    scenarios[delta_cols].to_csv(OUT/'structure_deltas.csv',index=False,encoding='utf-8-sig')
    edges.to_csv(OUT/'adjacency_sensitivity.csv',index=False,encoding='utf-8-sig')
    skipped.to_csv(OUT/'skipped_scenarios.csv',index=False,encoding='utf-8-sig')
    theory=pd.DataFrame([dict(round='理論',match_no=i,home_team=f'λH={h}',away_team=f'λA={a}',
                              lambda_home=h,lambda_away=a,final_P_D=np.nan,uncertainty_mix=np.nan)
                         for i,(h,a) in enumerate(((.8,.8),(1.5,1.5),(2.5,2.5),(2.,.8),(1.5,1.)),1)])
    theoretical,_,_,_=compute(theory,theoretical=True)
    theoretical.to_csv(OUT/'theoretical_scenarios.csv',index=False,encoding='utf-8-sig')
    changes=scenarios[(scenarios.scenario!='基準') & scenarios.delta.eq(.1)].copy()
    summary=changes.groupby('scenario').agg(mean_delta_draw=('delta_P_draw','mean'),
        median_delta_draw=('delta_P_draw','median'),min_delta_draw=('delta_P_draw','min'),
        max_delta_draw=('delta_P_draw','max'),mean_delta_00=('delta_P_0_0','mean'),
        mean_delta_11=('delta_P_1_1','mean'),mean_delta_2plus_draw=('delta_P_draw_2plus','mean'),
        mean_delta_margin1=('delta_P_margin_1','mean'),
        mean_delta_margin2plus=('delta_P_margin_2plus','mean')).reset_index()
    summary.to_csv(OUT/'scenario_summary_78.csv',index=False,encoding='utf-8-sig')
    sensitivity=scenarios[scenarios.scenario!='基準'].pivot_table(
        index=['round','match_no','home_team','away_team'],columns=['scenario','delta'],
        values='delta_P_draw').reset_index()
    sensitivity.columns=['round','match_no','home_team','away_team']+[
        f'delta_draw_{a}_{b}' for a,b in sensitivity.columns.tolist()[4:]]
    sensitivity.to_csv(OUT/'match_sensitivity_78.csv',index=False,encoding='utf-8-sig')
    # Fixed 1663 pair plus other near-raw-P(D) pairs, chosen without outcomes.
    near=[]
    for i in range(len(base)):
        for j in range(i+1,len(base)):
            a,b=base.iloc[i],base.iloc[j]
            if abs(a.raw_P_D-b.raw_P_D)<=.005:
                near.append(dict(round_A=a['round'],match_no_A=int(a.match_no),match_A=f'{a.home_team}－{a.away_team}',
                    round_B=b['round'],match_no_B=int(b.match_no),match_B=f'{b.home_team}－{b.away_team}',
                    raw_P_D_A=a.raw_P_D,raw_P_D_B=b.raw_P_D,
                    gap_P_00=abs(a.P_0_0-b.P_0_0),
                    delta_draw_H_low_A=float(changes.loc[(changes['round']==a['round'])&(changes.match_no==a.match_no)&(changes.scenario=='H低下'),'delta_P_draw'].iloc[0]),
                    delta_draw_H_low_B=float(changes.loc[(changes['round']==b['round'])&(changes.match_no==b.match_no)&(changes.scenario=='H低下'),'delta_P_draw'].iloc[0])))
    near=pd.DataFrame(near).sort_values('gap_P_00',ascending=False)
    near.to_csv(OUT/'near_draw_probability_pairs.csv',index=False,encoding='utf-8-sig')
    near_opposite=int((near.delta_draw_H_low_A*near.delta_draw_H_low_B<0).sum())
    near_spread=float(abs(near.delta_draw_H_low_A-near.delta_draw_H_low_B).max())
    (OUT/'heatmaps').mkdir(exist_ok=True)
    selected=[('toto1663',3),('toto1663',7),('toto1670',11),('toto1670',5),('toto1668',2)]
    for rid,n in selected:
        before=matrices[(rid,n,'基準',0.)]
        for name in ('H低下','双方上昇'):
            after=matrices[(rid,n,name,.2)]
            (OUT/'heatmaps'/f'{rid}_{n:02}_{name}.svg').write_text(diff_svg(before,after,f'{rid} No.{n} {name} 0.2'))
    audit=dict(source=str(SOURCE),source_sha256=sha(SOURCE),matches=78,
               scenarios_per_match=17,scenario_rows=len(scenarios),adjacency_rows=len(edges),
               skipped=len(skipped),delta_values=[.1,.2],mix_applied=False,
               result_used=False,forecast_changed=False,buyplan_changed=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    def tab(frame,columns):return '<div class="scroll">'+frame[columns].to_html(index=False,escape=True,float_format=lambda x:f'{x:.5f}')+'</div>'
    pair=near[(near.round_A=='toto1663')&(near.match_no_A==3)&(near.round_B=='toto1663')&(near.match_no_B==7)]
    if pair.empty:pair=near[(near.round_B=='toto1663')&(near.match_no_B==3)&(near.round_A=='toto1663')&(near.match_no_A==7)]
    t=theoretical[(theoretical.delta.eq(.2)) & theoretical.scenario.isin(['H低下','A低下','双方上昇','双方低下'])]
    imgs=''.join(f'<figure><img src="heatmaps/{rid}_{n:02}_{name}.svg" width="495"></figure>' for rid,n in selected for name in ('H低下','双方上昇'))
    report=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>得点構造 Phase 3</title><style>body{{font:16px/1.7 system-ui;margin:28px;color:#203047}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{border:1px solid #ccd4df;padding:5px}}th{{background:#edf3f8}}figure{{display:inline-block;margin:5px}}</style>
<h1>得点構造 Phase 3：λ変動への感応度</h1><p>試合前保存λの78試合に対し、8方向×0.1/0.2点と基準を計算。行列はPhase 1・2と同じ独立Poisson・0～10点正規化。全1,326行、1点差隣接26,520行。実際の予測誤差幅やスコア遷移確率ではない。不確実性mixは適用せず参考列に保存。試合結果・配券は使用していない。</p>
<h2>78試合の変動反応（δ=0.1、確率差）</h2>{tab(summary,['scenario','mean_delta_draw','min_delta_draw','max_delta_draw','mean_delta_00','mean_delta_11','mean_delta_2plus_draw','mean_delta_margin1','mean_delta_margin2plus'])}
<p>同点確率の増減は0-0、1-1、2-2以上のセル差分の合計。H/A別の1・2・3点以上差は<a href="scenario_structure.csv">全シナリオCSV</a>に保持。確率質量の増減は、個々の試合が勝敗から同点へ移る確率ではない。</p>
<h2>λを統制した理論比較（δ=0.2）</h2>{tab(t,['home_team','away_team','scenario','P_draw','delta_P_draw','delta_P_0_0','delta_P_1_1','delta_P_draw_2plus','delta_P_margin_1','delta_P_margin_2plus'])}
<h2>行列P(D)が近い試合の反応差</h2><p>第1663回の東京V－FC東京と水戸－浦和を同条件で比較。ほかの近接ペアも結果を使わず列挙した。行列P(D)差0.5ポイント以内の全{len(near)}組のうち、Hのλを0.1下げた場合にD増減の符号が逆になる組は{near_opposite}組、反応差の最大は{near_spread*100:.3f}ポイント。ペアは互いに独立の標本ではない。感応度の符号と大きさはλのどちらを動かすかで変わるため、「Dになりやすい」という単一順位にはしない。</p>{tab(pair,['match_A','match_B','raw_P_D_A','raw_P_D_B','gap_P_00','delta_draw_H_low_A','delta_draw_H_low_B'])}
<h2>行列差分ヒートマップ</h2><p>緑が確率質量の増加、赤が減少。枠で同点・1点差・その他を分けた。各セルは変動後−基準のポイント値。隣接する同点の確率を重複合算しない。</p>{imgs}
<h2>解釈と引き継ぎ</h2><p>同点帯別、H/A別の1点差・2点差以上、隣接勝敗セルと各同点セルの差分を独立に保存した。<a href="adjacency_sensitivity.csv">隣接表</a>の同点セルは勝敗セルごとの参照値であり、行方向に足し上げる評価量ではない。基準構造とシナリオ構造は<a href="scenario_structure.csv">構造表</a>でPhase 4へ渡せる。Phase 4は未実装。</p>
<p><a href="structure_deltas.csv">差分CSV</a> ／ <a href="match_sensitivity_78.csv">試合別感応度</a> ／ <a href="theoretical_scenarios.csv">理論CSV</a> ／ <a href="near_draw_probability_pairs.csv">近接ペア</a> ／ <a href="audit.json">監査</a></p></html>'''
    (OUT/'index.html').write_text(report)
    print(json.dumps(audit,ensure_ascii=False))
    print(summary.to_string(index=False))
    print('pair',pair.to_string(index=False))


if __name__=='__main__':main()
