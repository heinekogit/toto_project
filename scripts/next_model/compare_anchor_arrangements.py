"""Offline comparison only. Run from repository root; never updates production."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from anchor_arrangement import arrange_from_anchor, relative_costs
from proportional_buyplan import build_proportional_tickets
from ticket_coverage_optimizer import optimize_ten_hit_coverage
from simulate_draw_balance import exact_portfolio_probabilities

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/reports/anchor_arrangement_v1'
SOURCE = ROOT / 'data/reports/next_probability_calibration/matches.csv'
CURRENT = ROOT / 'data/purchase_reference/buyplan.csv'
COLS = ['next_prob_home','next_prob_draw','next_prob_away']
TICKETS = [f'ticket{i:02}' for i in range(1,11)]
LABELS = {'current_v22':'現行v2.2（再生成）','balanced':'相対コスト均等型','graded':'Anchor中心の段階型'}


def compare(rid, frame, initial, actual=None, proven=False):
    p = frame[COLS].to_numpy(float)
    plans = {'current_v22':initial}
    audits = {}
    for mode in ['balanced','graded']:
        plans[mode], audits[mode] = arrange_from_anchor(initial,p,mode)
    exact = exact_portfolio_probabilities(plans,p)
    rows=[]
    costs=relative_costs(p)
    for name,t in plans.items():
        for s in range(3):
            np.testing.assert_array_equal((t==s).sum(axis=0),(initial==s).sum(axis=0))
        total=costs[np.arange(13),t].sum(axis=1)
        distances=(t[:,None,:]!=t[None,:,:]).sum(axis=2)[np.triu_indices(10,1)]
        row=dict(round_id=rid,strategy=name,prematch_proven=proven,
                 min_distance=int(distances.min()),mean_distance=float(distances.mean()),
                 cost_min=float(total.min()),cost_max=float(total.max()),cost_std=float(total.std()),
                 actual_best=None,actual_mean=None)
        for k,v in zip([10,11,12,13],exact[name]): row[f'model_ge{k}']=float(v)
        if actual is not None:
            hits=(t==actual).sum(axis=1)
            assert int(hits.sum()) == int((initial==actual).sum())
            row.update(actual_best=int(hits.max()),actual_mean=float(hits.mean()))
            for k in [8,9,10,11,12,13]: row[f'actual_ge{k}']=int(hits.max()>=k)
        rows.append(row)
        plan=frame[[c for c in ['match_no','home_team','away_team'] if c in frame]].copy()
        for i,col in enumerate(TICKETS): plan[col]=np.array([1,0,2])[t[i]]
        plan.to_csv(OUT/f'{rid}_{name}.csv',index=False)
        pd.DataFrame({'ticket':TICKETS,'relative_cost':total,'anchor_ratio':np.exp(-total),
                      'actual_hits':(t==actual).sum(axis=1) if actual is not None else [None]*10}).to_csv(OUT/f'{rid}_{name}_tickets.csv',index=False)
    diagnostics=frame[['match_no']+COLS].copy()
    for s,symbol in enumerate('HDA'):
        diagnostics[f'{symbol}_count']=(initial==s).sum(axis=0)
        diagnostics[f'{symbol}_relative_cost']=costs[:,s]
    diagnostics.to_csv(OUT/f'{rid}_match_costs.csv',index=False)
    (OUT/f'{rid}_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
    print(rid,[(r['strategy'],r['actual_best']) for r in rows],flush=True)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [SOURCE,CURRENT,Path(__file__),Path(__file__).with_name('anchor_arrangement.py')]}
    data=pd.read_csv(SOURCE)
    rows=[]; excluded=[]
    # Fixed rules applied to all rounds; no actual results used in arrangement.
    # Historical reconstructed probabilities are retrospective, not a forward test.
    for rid,g in data.groupby('round_id',sort=True):
        if len(g)!=13 or set(g.match_no)!=set(range(1,14)):
            excluded.append(str(rid));continue
        g=g.sort_values('match_no').reset_index(drop=True)
        p=g[COLS].to_numpy(float)
        initial,_=build_proportional_tickets(p)
        initial,_=optimize_ten_hit_coverage(initial,p)
        actual=g.actual_result.map({1:0,0:1,2:2})
        if actual.isna().any(): raise ValueError(f'{rid}: invalid actual')
        proven=bool(g.prematch_snapshot_proven.astype(str).str.lower().eq('true').all())
        rows.extend(compare(rid,g,initial,actual.to_numpy(int),proven))
    current=pd.read_csv(CURRENT).sort_values('match_no').reset_index(drop=True)
    if len(current)!=13 or set(current.match_no)!=set(range(1,14)):
        raise ValueError('Current plan must contain 13 unique matches')
    raw=current[TICKETS].to_numpy().T
    if not np.isin(raw,[1,0,2]).all(): raise ValueError('Invalid current symbols')
    initial=np.array([1,0,2])[raw.astype(int)]
    rows.extend(compare('current_saved',current,initial))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'results.csv',index=False)
    historical=df[df.actual_best.notna()]
    sections=[]
    for label,part in [('事前保存確認あり（参考評価）',historical[historical.prematch_proven]),('全履歴（参考評価）',historical)]:
        summary=part.groupby('strategy').agg(rounds=('round_id','size'),best_hits=('actual_best','mean'),mean_hits=('actual_mean','mean'),ge8_rounds=('actual_ge8','sum'),ge9_rounds=('actual_ge9','sum'),ge10_rounds=('actual_ge10','sum'),model_ge10=('model_ge10','mean'),model_ge13=('model_ge13','mean'))
        summary.to_csv(OUT/('prematch_summary.csv' if label.startswith('事前') else 'all_summary.csv'))
        sections.append(f'<h2>{label}</h2>'+summary.rename(index=LABELS).to_html(float_format=lambda x:f'{x:.6f}'))
    view=df[['round_id','strategy','actual_best','actual_mean','model_ge10','model_ge11','model_ge12','model_ge13','min_distance','cost_std']].copy()
    view['strategy']=view.strategy.map(LABELS)
    links=''.join(f'<li><a href="{rid}_{name}.csv">{rid}：{LABELS[name]} 配券CSV</a></li>' for rid in df.round_id.unique() for name in LABELS)
    html='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Anchor相対配券 比較</title>
<style>body{font-family:system-ui;margin:32px;line-height:1.7;color:#223}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccd}th{background:#eef3f8}h1{font-size:25px}</style>
<h1>Anchor相対配券の比較 — 検証用・購入対象外</h1>
<p>試合ごとのH/D/A枚数を固定し、10券への配置だけを変更。Anchorは各試合の最大確率の目。Anchor自体を購入券として固定しません。</p>
<p>変更コスト = −log(選択確率 / 最大確率)。均等型は券ごとの合計コストを平均へ近づけます。段階型は平均の0.5～1.5倍の等間隔の目標へ近づけます。この幅は未検証の設計仮説です。最低2試合差を維持し、同一試合内の目の交換で探索します。大域最適の保証はありません。</p>
<p>現行v2.2は履歴の確率から再生成した比較基準で、当時の購入券ではありません。current_savedは現在保存されている券をそのまま基準にしています。実結果は配置処理に渡していません。</p>
<p>model_ge10等は独立モデルの全3^13結果を列挙した「いずれかの券で指定的中数以上」の確率です。実測の的中率ではありません。actual_bestは実結果での最高的中数。mean_hitsは枚数固定により全方式で不変です。</p>
<p>過去開催はすべて参考評価。事前保存確認は元資料の属性を引き継いだもので、新方式や再計算確率の事前固定を保証しません。方式を選んだ後の将来開催を未使用の検証対象とし、過去の成績だけで採用しません。本番出力は変更していません。</p>'''+''.join(sections)+'<h2>開催別（13試合）</h2>'+view.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<h2>配置ファイル</h2><ul>'+links+'</ul></html>'
    grids = '<h2>現在保存された券の配置比較</h2><p>表の1=H、0=D、2=A。各行の枚数は3案で同一です。</p>'
    for name in LABELS:
        plan = pd.read_csv(OUT/f'current_saved_{name}.csv')
        grids += '<h3>'+('保存済み現行券' if name=='current_v22' else LABELS[name])+'</h3>'+plan.to_html(index=False)
    html = html.replace('</html>',grids+'</html>')
    (OUT/'index.html').write_text(html,encoding='utf-8')
    manifest=dict(source_hashes=hashes,excluded_incomplete_rounds=excluded,production_changed=False,
                  actual_results_used_in_optimization=False,steps=20000,seed=20261017,
                  graded_target_mean_multipliers=[.5,1.5],evaluation='exact independent 3^13 enumeration',
                  validation_status='retrospective exploratory; future frozen evaluation required')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if hashlib.sha256(CURRENT.read_bytes()).hexdigest()!=hashes[str(CURRENT.relative_to(ROOT))]:
        raise AssertionError('Production plan changed during comparison')

if __name__=='__main__': main()
