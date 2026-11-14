"""Read-only shadow comparison: P(D) versus P(D)/max(P(H),P(A))."""
from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from compare_d_scoop_uncertainty import ROOT, ROUNDS, PHASE1, read_round, ranks, sha

OUT = ROOT / 'data/reports/d_scoop_phase2_relative'


def compare_round(round_id: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    diagnostic, provenance = read_round(round_id)
    numbers = diagnostic.match_no.to_numpy(dtype=int)
    p_h = diagnostic.P_H.to_numpy(dtype=float)
    p_d = diagnostic.P_D.to_numpy(dtype=float)
    p_a = diagnostic.P_A.to_numpy(dtype=float)
    if not np.isfinite(np.c_[p_h,p_d,p_a]).all() or np.min(np.c_[p_h,p_d,p_a]) < 0:
        raise ValueError(f'{round_id}: invalid saved H/D/A')
    np.testing.assert_allclose(p_h+p_d+p_a, 1, rtol=0, atol=1e-10)
    highest_side = np.maximum(p_h,p_a)
    if np.any(highest_side <= 0):
        raise ValueError(f'{round_id}: zero H/A denominator')
    relative = p_d/highest_side
    gap = np.abs(p_h-p_a)
    # With sum(H,D,A)=1, the relative measure necessarily rewards HA balance.
    np.testing.assert_allclose(relative, 2*p_d/(1-p_d+gap), rtol=1e-12, atol=1e-12)
    rank_a,rank_b = ranks(p_d,numbers),ranks(relative,numbers)
    select_a,select_b=rank_a<=4,rank_b<=4
    detail=diagnostic[['round','match_no','home_team','away_team','actual_symbol','actual_D']].copy()
    detail['P_H']=p_h;detail['P_D']=p_d;detail['P_A']=p_a
    detail['D_relative']=relative;detail['HA_gap']=gap
    detail['A_PD_rank']=rank_a;detail['B_relative_rank']=rank_b
    detail['B_rank_gain']=rank_a-rank_b
    detail['A_selected']=select_a;detail['B_selected']=select_b
    detail['selection_change']=np.select([select_b&~select_a,select_a&~select_b],['B_only','A_only'],default='unchanged')
    detail=detail.sort_values('match_no').reset_index(drop=True)
    added=detail.loc[detail.selection_change.eq('B_only')]
    removed=detail.loc[detail.selection_change.eq('A_only')]
    if len(added)!=len(removed):raise AssertionError('Unbalanced replacements')
    a_captured=int(detail.loc[detail.A_selected,'actual_D'].sum())
    b_captured=int(detail.loc[detail.B_selected,'actual_D'].sum())
    new=int(added.actual_D.sum()); lost=int(removed.actual_D.sum())
    if b_captured-a_captured!=new-lost:raise AssertionError('Capture reconciliation failed')
    summary=dict(round=round_id,actual_D=int(detail.actual_D.sum()),
                 A_match_nos=','.join(map(str,detail.loc[detail.A_selected,'match_no'])),
                 B_match_nos=','.join(map(str,detail.loc[detail.B_selected,'match_no'])),
                 common_selected=int((select_a&select_b).sum()),
                 A_only=','.join(map(str,removed.match_no)),B_only=','.join(map(str,added.match_no)),
                 replaced_matches=len(added),A_captured_D=a_captured,B_captured_D=b_captured,
                 gained_D_by_B=new,lost_D_from_A=lost,net_capture_B_minus_A=b_captured-a_captured,
                 rank_changed_matches=int(np.count_nonzero(rank_a!=rank_b)),
                 max_absolute_rank_change=int(np.max(np.abs(rank_a-rank_b))),
                 A_selected_mean_HA_gap=float(gap[select_a].mean()),
                 B_selected_mean_HA_gap=float(gap[select_b].mean()),
                 A_selected_mean_PD=float(p_d[select_a].mean()),
                 B_selected_mean_PD=float(p_d[select_b].mean()))
    return detail,summary,detail.loc[detail.selection_change.ne('unchanged')].copy(),provenance


def main() -> None:
    protected={str(p):sha(p) for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()}
    earlier={str(p):sha(p) for folder in [PHASE1,ROOT/'data/reports/d_scoop_phase2_uncertainty',ROOT/'data/reports/d_scoop_phase2_structure']
             for p in folder.rglob('*') if p.is_file()}
    details=[];summaries=[];changed=[];provenance=[]
    for rid in ROUNDS:
        detail,summary,exchange,source=compare_round(rid)
        details.append(detail);summaries.append(summary);changed.append(exchange);provenance.append(source)
    full=pd.concat(details,ignore_index=True)
    if len(full)!=78 or int(full.actual_D.sum())!=18:raise AssertionError('Expected 78 matches, 18 draws')
    summary=pd.DataFrame(summaries)
    exchange=pd.concat(changed,ignore_index=True)
    actual_d=full.loc[full.actual_D].sort_values(['round','match_no']).copy()
    totals=dict(rounds=6,matches=78,actual_D=18,
                A_captured_D=int(summary.A_captured_D.sum()),B_captured_D=int(summary.B_captured_D.sum()),
                replaced_matches=int(summary.replaced_matches.sum()),
                gained_D_by_B=int(summary.gained_D_by_B.sum()),
                lost_D_from_A=int(summary.lost_D_from_A.sum()),
                net_capture_B_minus_A=int(summary.net_capture_B_minus_A.sum()),
                rank_changed_matches=int(summary.rank_changed_matches.sum()),
                maximum_absolute_rank_change=int(summary.max_absolute_rank_change.max()),
                A_selected_mean_HA_gap=float(summary.A_selected_mean_HA_gap.mean()),
                B_selected_mean_HA_gap=float(summary.B_selected_mean_HA_gap.mean()),
                A_selected_mean_PD=float(summary.A_selected_mean_PD.mean()),
                B_selected_mean_PD=float(summary.B_selected_mean_PD.mean()))
    if len(exchange)!=2*totals['replaced_matches'] or totals['net_capture_B_minus_A']!=totals['gained_D_by_B']-totals['lost_D_from_A']:
        raise AssertionError('Overall replacements/capture mismatch')
    if any(sha(Path(path))!=digest for path,digest in protected.items()):raise AssertionError('Operational file changed')
    if any(sha(Path(path))!=digest for path,digest in earlier.items()):raise AssertionError('Earlier reports changed')
    OUT.mkdir(exist_ok=True,parents=True)
    full.to_csv(OUT/'all_78_relative.csv',index=False)
    summary.to_csv(OUT/'round_comparison.csv',index=False)
    exchange.to_csv(OUT/'selection_exchanges.csv',index=False)
    actual_d.to_csv(OUT/'actual_18_draws.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(
        methods={'A':'final P(D) descending','B':'final P(D)/max(final P(H),final P(A)) descending'},
        tie_break='exact tie: smaller match_no first',
        relative_identity='2*P(D)/(1-P(D)+abs(P(H)-P(A)))',
        relative_score_is_probability=False,score_or_coefficients_trained=False,
        source_audits=provenance,totals=totals,production_modified=False,
        prior_reports_sha256=earlier,
        limitation='Six prior rounds and changing model versions; outcome comparisons are retrospective and exploratory.'
    ),ensure_ascii=False,indent=2))
    def table(frame):return '<div class="scroll">'+frame.to_html(index=False,escape=True,float_format=lambda v:f'{v:.5f}')+'</div>'
    view=summary[['round','actual_D','A_match_nos','B_match_nos','common_selected','A_only','B_only',
                  'A_captured_D','B_captured_D','net_capture_B_minus_A','replaced_matches',
                  'A_selected_mean_HA_gap','B_selected_mean_HA_gap']]
    draw_view=actual_d[['round','match_no','home_team','away_team','P_H','P_D','P_A','D_relative','HA_gap',
                        'A_PD_rank','B_relative_rank','A_selected','B_selected']]
    html='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>d-scoop H/A相対評価</title>
<style>body{font:16px/1.8 system-ui;margin:28px;color:#203147}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:6px;border:1px solid #cbd4df}th{background:#eef3f8}</style>
<h1>d-scoop：H/A相対評価のシャドー比較</h1>
<p>同一の保存済み最終H/D/A確率から、AはP(D)上位4、BはP(D)/max(P(H),P(A))上位4を選択。同率は試合番号が小さい方を優先。結果は選出後に採点。D_relativeはD発生確率ではない。</p>'''
    html+=f'''<p>78試合中、順位が変わった試合は{totals['rank_changed_matches']}、最大移動は{totals['maximum_absolute_rank_change']}位。6開催の4試合選出で入れ替えは{totals['replaced_matches']}組（Aのみ/Bのみを二重計上しない）。D捕捉はAが{totals['A_captured_D']}/18、Bが{totals['B_captured_D']}/18。Bで新たに拾ったDは{totals['gained_D_by_B']}、失ったDは{totals['lost_D_from_A']}、純増減{totals['net_capture_B_minus_A']:+d}。</p>'''
    html+='''<p>確率合計が1なので、D_relative＝2P(D)/(1−P(D)+HA_gap)。P(D)が同じならH/A差が小さい試合ほど相対評価が高い。従ってBはH/A拮抗を数式上すでに取り込むが、HA_gapは今回スコアに追加しない。</p>'''
    html+=f'''<p>選出24試合の平均HA_gap：A {totals['A_selected_mean_HA_gap']:.4f}、B {totals['B_selected_mean_HA_gap']:.4f}。平均P(D)：A {totals['A_selected_mean_PD']:.4f}、B {totals['B_selected_mean_PD']:.4f}。BはH/Aが接近した試合を優先し、P(D)の平均は下がった。</p>
<p>入れ替わりで第1670回のNo.2 横浜FM－岡山（D）をBが追加した。一方、第1663回No.1 清水－G大阪（D）と第1664回No.1 横浜FM－東京V（D）をBは外した。後者はP(D)が2位でもHA_gapが0.2780と大きく、相対順位が9位へ落ちた。Bの捕捉が1件減った理由はこの2失点と1追加による。</p>
<p>今回は同じ6開催を見てからの遡及比較であり、指標の予測上の優位性は確認されなかった。H/A拮抗をさらに評価する場合は、新しい試合前予測と後続結果で、P(D)が近い試合の比較から検証する価値はある。現時点でBを選出式として採用する根拠はない。</p>'''
    html+='<h2>開催別</h2>'+table(view)
    html+='<h2>入れ替わった試合の最終H/D/A構成</h2>'+table(exchange[['round','match_no','home_team','away_team','P_H','P_D','P_A','D_relative','HA_gap',
                                                                     'A_PD_rank','B_relative_rank','B_rank_gain','selection_change','actual_symbol']])
    html+='<h2>実際のD18試合：両方式の順位</h2>'+table(draw_view)
    for rid,g in full.groupby('round',sort=False):
        html+='<h2>'+escape(rid)+'：全13試合</h2>'+table(g[['match_no','home_team','away_team','P_H','P_D','P_A',
                         'D_relative','HA_gap','A_PD_rank','B_relative_rank','B_rank_gain','A_selected','B_selected','actual_symbol']])
    html+='''<p>保存予測のハッシュ、試合前生成時刻、対戦・結果の対応、Phase 1指標との一致を確認済み。今ある6開催には予測モデル版の差があり、相対評価の将来性能はこの比較だけでは確定しない。正式なd-scoopスコア、Next Model、BuyPlanは変更していない。</p>
<p><a href="all_78_relative.csv">78試合CSV</a> ／ <a href="round_comparison.csv">開催別CSV</a> ／ <a href="selection_exchanges.csv">入れ替わりCSV</a> ／ <a href="actual_18_draws.csv">実際のD18試合CSV</a> ／ <a href="audit.json">監査JSON</a></p></html>'''
    (OUT/'index.html').write_text(html)
    print(summary.to_string(index=False));print('TOTAL',json.dumps(totals,ensure_ascii=False))
    print(exchange[['round','match_no','home_team','away_team','P_H','P_D','P_A','D_relative','HA_gap',
                    'A_PD_rank','B_relative_rank','selection_change','actual_symbol']].to_string(index=False))
    print(OUT/'index.html')


if __name__=='__main__':main()
