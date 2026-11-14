"""Shadow A/B/C comparison using final and raw score-distribution H/D/A."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from compare_d_scoop_uncertainty import ROOT, ROUNDS, PHASE1, read_round, ranks, sha

OUT = ROOT / 'data/reports/d_scoop_phase2_score_relative'


def compare_round(round_id: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    diagnostic, provenance = read_round(round_id)
    n = diagnostic.match_no.to_numpy(dtype=int)
    final = diagnostic[['P_H','P_D','P_A']].to_numpy(dtype=float)
    score = diagnostic[['score_P_H','score_P_D','score_P_A']].to_numpy(dtype=float)
    mix = diagnostic.uncertainty_mix.to_numpy(dtype=float)
    if not np.isfinite(final).all() or not np.isfinite(score).all():
        raise ValueError(f'{round_id}: missing H/D/A probabilities')
    np.testing.assert_allclose(final.sum(axis=1),1,rtol=0,atol=1e-10)
    np.testing.assert_allclose(score.sum(axis=1),1,rtol=0,atol=1e-10)
    np.testing.assert_allclose(final,(1-mix[:,None])*score+mix[:,None]/3,rtol=0,atol=1e-10)
    if (np.maximum(final[:,0],final[:,2])<=0).any() or (np.maximum(score[:,0],score[:,2])<=0).any():
        raise ValueError(f'{round_id}: H/A denominator is zero')
    a_value = final[:,1]
    b_value = final[:,1]/np.maximum(final[:,0],final[:,2])
    c_value = score[:,1]/np.maximum(score[:,0],score[:,2])
    a_rank,b_rank,c_rank=(ranks(v,n) for v in (a_value,b_value,c_value))
    a_sel,b_sel,c_sel=(r<=4 for r in (a_rank,b_rank,c_rank))
    out=diagnostic[['round','match_no','home_team','away_team','actual_symbol','actual_D']].copy()
    for label,values in [('final_P_H',final[:,0]),('final_P_D',final[:,1]),('final_P_A',final[:,2]),
                         ('score_P_H',score[:,0]),('score_P_D',score[:,1]),('score_P_A',score[:,2]),
                         ('mix',mix),('score_A',a_value),('score_B',b_value),('score_C',c_value),
                         ('A_rank',a_rank),('B_rank',b_rank),('C_rank',c_rank),
                         ('A_selected',a_sel),('B_selected',b_sel),('C_selected',c_sel)]:
        out[label]=values
    out['final_HA_gap']=abs(final[:,0]-final[:,2])
    out['score_HA_gap']=abs(score[:,0]-score[:,2])
    out['delta_H']=final[:,0]-score[:,0]
    out['delta_D']=final[:,1]-score[:,1]
    out['delta_A']=final[:,2]-score[:,2]
    out['delta_relative_B_minus_C']=b_value-c_value
    out['rank_gain_C_over_B']=b_rank-c_rank
    out['BC_selection_change']=np.select([c_sel&~b_sel,b_sel&~c_sel],['C_only','B_only'],default='unchanged')
    out['AC_selection_change']=np.select([c_sel&~a_sel,a_sel&~c_sel],['C_only','A_only'],default='unchanged')
    out=out.sort_values('match_no').reset_index(drop=True)
    def contrast(original,alternative):
        add=alternative&~original;remove=original&~alternative
        if add.sum()!=remove.sum():raise AssertionError('Unbalanced exchange')
        gained=int((add&out.actual_D.to_numpy()).sum())
        lost=int((remove&out.actual_D.to_numpy()).sum())
        return dict(replaced=int(add.sum()),gained=gained,lost=lost,delta=gained-lost,
                    added_nos=','.join(map(str,out.match_no[add])),
                    removed_nos=','.join(map(str,out.match_no[remove])))
    ac=contrast(a_sel,c_sel);bc=contrast(b_sel,c_sel)
    cap={name:int(out.loc[out[f'{name}_selected'],'actual_D'].sum()) for name in 'ABC'}
    if cap['C']-cap['A']!=ac['delta'] or cap['C']-cap['B']!=bc['delta']:
        raise AssertionError('Capture difference does not equal exchange gain/loss')
    summary=dict(round=round_id,actual_D=int(out.actual_D.sum()),
                 **{f'{name}_selected_nos':','.join(map(str,out.match_no[out[f'{name}_selected']])) for name in 'ABC'},
                 **{f'{name}_captured_D':cap[name] for name in 'ABC'},
                 AC_replaced=ac['replaced'],AC_added_nos=ac['added_nos'],AC_removed_nos=ac['removed_nos'],
                 AC_gained_D=ac['gained'],AC_lost_D=ac['lost'],AC_delta_C_minus_A=ac['delta'],
                 BC_replaced=bc['replaced'],BC_added_nos=bc['added_nos'],BC_removed_nos=bc['removed_nos'],
                 BC_gained_D=bc['gained'],BC_lost_D=bc['lost'],BC_delta_C_minus_B=bc['delta'],
                 BC_rank_changed_matches=int((b_rank!=c_rank).sum()),
                 BC_max_absolute_rank_change=int(np.max(abs(b_rank-c_rank))),
                 B_selected_mean_final_HA_gap=float(out.loc[b_sel,'final_HA_gap'].mean()),
                 C_selected_mean_score_HA_gap=float(out.loc[c_sel,'score_HA_gap'].mean()),
                 B_selected_mean_raw_HA_gap=float(out.loc[b_sel,'score_HA_gap'].mean()),
                 C_selected_mean_final_HA_gap=float(out.loc[c_sel,'final_HA_gap'].mean()),
                 mix_min=float(mix.min()),mix_max=float(mix.max()))
    exchange=out.loc[out.BC_selection_change.ne('unchanged')].copy()
    return out,summary,exchange,provenance


def main() -> None:
    protected={str(p):sha(p) for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()}
    prior_dirs=[PHASE1,ROOT/'data/reports/d_scoop_phase2_uncertainty',
                ROOT/'data/reports/d_scoop_phase2_structure',ROOT/'data/reports/d_scoop_phase2_relative']
    prior={str(p):sha(p) for folder in prior_dirs for p in folder.rglob('*') if p.is_file()}
    details=[];summaries=[];exchanges=[];sources=[]
    for rid in ROUNDS:
        detail,summary,exchange,source=compare_round(rid)
        details.append(detail);summaries.append(summary);exchanges.append(exchange);sources.append(source)
    full=pd.concat(details,ignore_index=True)
    summary=pd.DataFrame(summaries)
    exchange=pd.concat(exchanges,ignore_index=True)
    actual_draws=full.loc[full.actual_D].sort_values(['round','match_no']).copy()
    if len(full)!=78 or len(actual_draws)!=18:raise AssertionError('Expected 78 matches and 18 draws')
    totals=dict(rounds=6,matches=78,actual_D=18,
                A_captured_D=int(summary.A_captured_D.sum()),
                B_captured_D=int(summary.B_captured_D.sum()),
                C_captured_D=int(summary.C_captured_D.sum()),
                AC_replaced=int(summary.AC_replaced.sum()),
                AC_gained_D=int(summary.AC_gained_D.sum()),AC_lost_D=int(summary.AC_lost_D.sum()),
                BC_replaced=int(summary.BC_replaced.sum()),
                BC_gained_D=int(summary.BC_gained_D.sum()),BC_lost_D=int(summary.BC_lost_D.sum()),
                BC_rank_changed_matches=int(summary.BC_rank_changed_matches.sum()),
                BC_max_absolute_rank_change=int(summary.BC_max_absolute_rank_change.max()),
                B_selected_mean_final_HA_gap=float(summary.B_selected_mean_final_HA_gap.mean()),
                C_selected_mean_score_HA_gap=float(summary.C_selected_mean_score_HA_gap.mean()),
                B_selected_mean_raw_HA_gap=float(summary.B_selected_mean_raw_HA_gap.mean()),
                C_selected_mean_final_HA_gap=float(summary.C_selected_mean_final_HA_gap.mean()))
    if totals['C_captured_D']-totals['A_captured_D']!=totals['AC_gained_D']-totals['AC_lost_D']:
        raise AssertionError('AC overall capture mismatch')
    if totals['C_captured_D']-totals['B_captured_D']!=totals['BC_gained_D']-totals['BC_lost_D']:
        raise AssertionError('BC overall capture mismatch')
    if len(exchange)!=2*totals['BC_replaced']:
        raise AssertionError('BC replacement count double counted')
    if any(sha(Path(path))!=digest for path,digest in protected.items()):raise AssertionError('Operational files changed')
    if any(sha(Path(path))!=digest for path,digest in prior.items()):raise AssertionError('Prior reports changed')
    OUT.mkdir(parents=True,exist_ok=True)
    for name,table in [('all_78_ABC.csv',full),('round_comparison.csv',summary),
                       ('BC_selection_exchanges.csv',exchange),('actual_18_draws.csv',actual_draws)]:
        table.to_csv(OUT/name,index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(
        methods={'A':'final P(D)','B':'final P(D)/max(final P(H),final P(A))',
                 'C':'score-distribution P(D)/max(score P(H),score P(A))'},
        tie_break='higher score first; exact tie by smaller match_no',
        score_matrix_after_uncertainty='not defined or reconstructed',
        result_used_for_selection=False,coefficients_fitted=False,
        sources=sources,totals=totals,prior_file_sha256=prior,production_modified=False,
        caveat='Six historical rounds with changing model versions; descriptive comparison, not independent predictive validation.'
    ),ensure_ascii=False,indent=2))
    def tab(frame):return '<div class="scroll">'+frame.to_html(index=False,escape=True,float_format=lambda v:f'{v:.6f}')+'</div>'
    display=summary[['round','actual_D','A_selected_nos','B_selected_nos','C_selected_nos',
                     'A_captured_D','B_captured_D','C_captured_D','AC_replaced','BC_replaced',
                     'AC_gained_D','AC_lost_D','BC_gained_D','BC_lost_D','BC_rank_changed_matches']]
    drawcols=['round','match_no','home_team','away_team','actual_symbol','A_rank','B_rank','C_rank',
              'A_selected','B_selected','C_selected','final_P_D','score_P_D','score_B','score_C']
    xcols=['round','match_no','home_team','away_team','actual_symbol','BC_selection_change',
           'final_P_H','final_P_D','final_P_A','score_P_H','score_P_D','score_P_A',
           'mix','delta_H','delta_D','delta_A','score_B','score_C',
           'B_rank','C_rank','final_HA_gap','score_HA_gap']
    report='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>d-scoop 得点分布由来の相対評価</title>
<style>body{font:16px/1.8 system-ui;margin:28px;color:#203147}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:6px;border:1px solid #cbd4df}th{background:#eef3f8}</style>
<h1>d-scoop：得点分布由来H/D/A相対評価</h1>
<p>試合前保存予測の6開催78試合。A＝最終P(D)、B＝最終P(D)/max(最終P(H),最終P(A))、C＝補正前得点分布のP(D)/max(得点分布P(H),得点分布P(A))。同率は小さい試合番号を優先。いずれも各開催上位4試合を結果に触れず選ぶ。</p>
<p>得点分布由来H/D/Aは保存λからPhase 1と同じ11×11行列を再現して集計。最終H/D/Aとの関係は final=(1−mix)×score+mix/3。補正後のスコア行列は作らない。B/Cの値はD発生確率ではない。</p>'''
    report+=f'''<p>6開催のD捕捉：A {totals['A_captured_D']}/18、B {totals['B_captured_D']}/18、C <b>{totals['C_captured_D']}/18</b>。A→Cで{totals['AC_replaced']}組交換し、Dを{totals['AC_gained_D']}件拾い、{totals['AC_lost_D']}件失った。B→Cで{totals['BC_replaced']}組交換し、Dを{totals['BC_gained_D']}件拾い、{totals['BC_lost_D']}件失った。B/Cの順位が変わったのは{totals['BC_rank_changed_matches']}試合、最大{totals['BC_max_absolute_rank_change']}順位。</p>'''
    report+=f'''<p>選出24試合のH/A差平均：Bの最終値{totals['B_selected_mean_final_HA_gap']:.4f}、Cの補正前値{totals['C_selected_mean_score_HA_gap']:.4f}。同一尺度での比較も、B選出の補正前差{totals['B_selected_mean_raw_HA_gap']:.4f}、C選出の補正前差{totals['C_selected_mean_score_HA_gap']:.4f}として併記する。</p>'''
    report+='''<p>結論：CはAよりDを2件、Bより1件少なく捕捉した。B→Cの2組の交換では新規D捕捉0件、失ったD1件。1665回はBが選んだNo.8 名古屋－横浜FM（D、最終P(D)28.34%）をCが外し、No.6 東京V－福岡（H、最終P(D)29.49%）を選んだ。両試合ともmix=0.25であるため、補正率の大小だけで交換を説明できない。補正前相対値はNo.6が0.615951、No.8が0.615156と僅差だが、補正後はそれぞれ0.690925、0.693636と逆転する。1668回は両試合のmix自体が異なり、No.5 柏－浦和が0.25、No.11 仙台－大宮が約0.205。補正前後の全H/D/A構成の変化が相対順位を動かした。</p>
<p>mix=0の1660・1663・1664回ではB=Cが厳密に一致する。補正がある3開催ではB/C間で15試合の順位が変わったが、上位4の交換は2組に留まる。最終P(D)の水準、H/Aの差、mixによる同率方向への圧縮をそれぞれ確認することが次の論点。今回のCは補正前のH/D/A総和を比べるだけで、0–0・1–1・2–2等の構造を個別に評価していない。得点分布の細部に追加の予測価値がないとは結論できない。</p>'''
    report+='<h2>開催別3方式</h2>'+tab(display)
    report+='<h2>BとCで採用が入れ替わった試合</h2>'+tab(exchange[xcols])
    report+='<p>delta_H/D/Aは最終確率−得点分布由来確率。各試合のmixと共に、B/C間の順位変化を確認する。個々の入れ替わりには相手試合とのスコア比較も必要で、単独のdeltaだけでは原因と断定しない。</p>'
    report+='<h2>実際にDだった18試合のA/B/C順位</h2>'+tab(actual_draws[drawcols])
    for rid,group in full.groupby('round',sort=False):
        report+='<h2>'+escape(rid)+'：全13試合</h2>'+tab(group[['match_no','home_team','away_team','actual_symbol',
                'final_P_H','final_P_D','final_P_A','score_P_H','score_P_D','score_P_A','mix',
                'score_A','score_B','score_C','A_rank','B_rank','C_rank','A_selected','B_selected','C_selected']])
    report+='''<p>Cは個別スコア0–0・1–1・2–2に追加係数を付ける方法ではない。Cの結果だけでスコア構造の詳細な利用可能性は判断できない。6開催はモデル版が異なる少数の遡及比較であり、正式なd-scoop選出式・Next Model・BuyPlanへの反映は行っていない。</p>
<p><a href="all_78_ABC.csv">78試合CSV</a> ／ <a href="round_comparison.csv">開催別CSV</a> ／ <a href="BC_selection_exchanges.csv">B/C入れ替わりCSV</a> ／ <a href="actual_18_draws.csv">実際のD18試合CSV</a> ／ <a href="audit.json">監査JSON</a></p></html>'''
    (OUT/'index.html').write_text(report)
    print(summary.to_string(index=False))
    print('TOTAL',json.dumps(totals,ensure_ascii=False))
    print(exchange[xcols].to_string(index=False))
    print(OUT/'index.html')


if __name__=='__main__':main()
