"""Shadow comparison: contextual D placement scores with fixed H/D/A counts.

The adjustment changes only arrangement scores, never forecasts or allocations.
The fixed coefficient is a design hypothesis, not fitted to actual results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from anchor_arrangement import arrange_from_anchor
from placement_score import placement_scores as placement_probabilities
from simulate_draw_balance import exact_portfolio_probabilities

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/eval/toto_rounds/toto1663/snapshot/buyplan.csv'
ACTUAL = ROOT / 'data/eval/toto_rounds/toto1663/actual_results.csv'
GRADED = ROOT / 'data/reports/anchor_arrangement_v1/current_saved_graded.csv'
MILD = ROOT / 'data/purchase_reference/draw_spread_mild/toto1663/buyplan_draw_spread_mild.csv'
OUT = ROOT / 'data/reports/relative_draw_placement_v1/toto1663'
STRENGTH = 0.25
TICKET_COLUMNS = [f'ticket{i:02}' for i in range(1,11)]
RESULT_SYMBOLS = np.array([1,0,2])  # internal H=0,D=1,A=2


def read_tickets(path: Path, expected_matches: pd.Index) -> np.ndarray:
    frame = pd.read_csv(path).set_index('match_no').loc[expected_matches]
    symbols = frame[TICKET_COLUMNS].to_numpy(int).T
    if not np.isin(symbols,[0,1,2]).all():
        raise ValueError(f'invalid symbols in {path}')
    return np.select([symbols==1,symbols==0,symbols==2],[0,1,2]).astype(int)


def main() -> None:
    source = pd.read_csv(SOURCE).sort_values('match_no').reset_index(drop=True)
    if len(source)!=13 or set(source.match_no)!=set(range(1,14)):
        raise ValueError('Incomplete source round')
    matches = source.match_no
    p = source[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
    initial = read_tickets(SOURCE,matches)
    q,z,relative = placement_probabilities(p, strength=STRENGTH)
    # Actuals are read only after every arrangement has been generated.
    baseline,_ = arrange_from_anchor(initial,p,mode='graded')
    proposal,audit = arrange_from_anchor(initial,q,mode='graded')
    saved_graded = read_tickets(GRADED,matches)
    if not np.array_equal(baseline,saved_graded):
        raise AssertionError('Cannot reproduce the existing Anchor graded comparison')
    plans={'正式v2.2':initial,'Anchor段階型':baseline,'弱いD分散制約':read_tickets(MILD,matches),'相対D評価・段階型':proposal}
    for label,t in plans.items():
        for symbol in range(3):
            np.testing.assert_array_equal((t==symbol).sum(axis=0),(initial==symbol).sum(axis=0),err_msg=label)
        shared=(t[:,None,:]==t[None,:,:]).sum(axis=2)
        if (shared[np.triu_indices(10,1)]>11).any():
            raise AssertionError(f'distance under two: {label}')
    results = pd.read_csv(ACTUAL).set_index('match_no').loc[matches].result.to_numpy(int)
    actual = np.select([results==1,results==0,results==2],[0,1,2])
    if not np.isin(results,[0,1,2]).all():
        raise ValueError('Unresolved result')
    exact = exact_portfolio_probabilities(plans,p)
    OUT.mkdir(parents=True,exist_ok=True)
    diagnostic=source[['match_no','home_team','away_team','next_prob_home','next_prob_draw','next_prob_away']].copy()
    diagnostic['D_vs_strongest_HA_logratio']=relative
    diagnostic['D_relative_z_clipped']=z
    diagnostic['placement_home']=q[:,0]
    diagnostic['placement_draw']=q[:,1]
    diagnostic['placement_away']=q[:,2]
    diagnostic.to_csv(OUT/'match_scores.csv',index=False)
    rows=[];ticket_rows=[]
    for label,t in plans.items():
        hits=(t==actual).sum(axis=1)
        draw_counts=(t==1).sum(axis=1)
        joint=np.prod(p[np.arange(13)[None,:],t],axis=1)
        row={'strategy':label,'best_hits':int(hits.max()),'mean_hits':float(hits.mean()),
             'best_hit_rate':float(hits.max()/13),'D_counts':','.join(map(str,draw_counts)),
             'model_p_ge10':float(exact[label][0]),'model_p_ge12':float(exact[label][2]),
             'model_p_13':float(exact[label][3])}
        rows.append(row)
        for i in range(10):
            ticket_rows.append({'strategy':label,'ticket':f'ticket{i+1:02}',
                                'actual_hits':int(hits[i]),'D_count':int(draw_counts[i]),
                                'original_model_p13':float(joint[i])})
        plan=source[['match_no','home_team','away_team']].copy()
        for i,col in enumerate(TICKET_COLUMNS): plan[col]=RESULT_SYMBOLS[t[i]]
        plan.to_csv(OUT/f'{list(plans).index(label)+1:02}_buyplan.csv',index=False)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(ticket_rows).to_csv(OUT/'ticket_details.csv',index=False)
    audit.update(strength=STRENGTH,score_definition='logit(qD)=logit(pD)+0.25*clip(z,-2,2); qH/qA preserve pH/pA',
                 original_probabilities_used_for_evaluation=True,actual_results_used_in_arrangement=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    manifest={'inputs':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [SOURCE,ACTUAL,GRADED,MILD]},
              'strength':STRENGTH,'chosen_before_scoring_this_simulation':True,
              'retrospective_after_round_result_known':True,'production_changed':False,
              'same_allocations':True,'evaluation':'exact enumeration of all 3^13 outcomes under original independent-match probabilities'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    pretty=summary.copy()
    for col in ['best_hit_rate','model_p_ge10','model_p_ge12','model_p_13']:
        pretty[col]=pretty[col].map(lambda x:f'{x*100:.6f}%')
    html='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>相対D評価の配券シミュレーション toto1663</title>
<style>body{font-family:system-ui;margin:28px;color:#223;line-height:1.6}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd;padding:7px}th{background:#eef3f8}</style>
<h1>toto1663 相対D評価の配券シミュレーション</h1>
<p>元の予測確率で決まった各試合のH/D/A枚数を固定し、配券だけを変更。Dと最有力H/Aとの比を13試合内で標準化し、Dの配置用スコアを調整します。H/Aの配置用スコアは元のH:A比を保って連動します。</p>
<p>調整の強さは0.25に固定。極端な外れ値は標準値±2で止めます。13試合の相対差がない場合は調整しません。これは的中確率の再推定ではありません。モデル確率の評価には元の予測確率を使います。</p>
<p><b>注意：</b>toto1663の結果を見た後に考えた案なので、今回の実的中は探索的な事後比較です。調整値は今回の実結果で最適化していません。将来開催で固定した条件の検証が必要です。正式版と保存済みsnapshotは変更していません。</p>
<h2>比較</h2>'''+pretty.to_html(index=False)+'<h2>券別</h2>'+pd.DataFrame(ticket_rows).to_html(index=False,float_format=lambda x:f'{x:.9g}')+'<p><a href="match_scores.csv">試合別の配置用スコア</a> ／ <a href="04_buyplan.csv">新方式の配券CSV</a></p></html>'
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__': main()
