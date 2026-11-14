"""Offline actual-hit comparison; never writes operational purchase files."""
import itertools
import json
import numpy as np
import pandas as pd
from pathlib import Path
from shadow_v25_operational import build
from simulate_free_structural_draw import structure, PROB_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/reports/toto1665_four_double'

def main():
    source = ROOT / 'data/eval/toto_rounds/toto1665'
    f = pd.read_csv(source/'snapshot/next_predictions.csv').sort_values('match_no').reset_index(drop=True)
    p = f[PROB_COLUMNS].to_numpy()
    old = np.load(ROOT/'data/reports/v25_score_shadow/toto1665/input_and_baseline.npz')
    np.testing.assert_allclose(p, old['probabilities'], atol=0, rtol=0)
    v24 = old['baseline']
    anchor = p.argmax(1)
    features = structure(f)
    features['match_no'] = f.match_no
    features['D確率順位'] = features.P_D.rank(ascending=False, method='min')
    features['構造順位'] = features.Q_corrected.rank(ascending=False, method='min')
    features['順位合計'] = features['D確率順位'] + features['構造順位']
    eligible = features.loc[anchor != 1]
    selected = eligible.sort_values(['順位合計','P_D','match_no'], ascending=[True,False,True]).head(4).index.tolist()
    assert len(selected)==4
    double = np.tile(anchor,(16,1))
    for row,bits in zip(double,itertools.product([False,True],repeat=4)):
        for idx,bit in zip(selected,bits):
            if bit: row[idx]=1
    assert len(np.unique(double,axis=0))==16
    OUT.mkdir(parents=True,exist_ok=True)
    features['選出'] = features.index.isin(selected)
    features.to_csv(OUT/'selection.csv',index=False)
    print('Selected before scoring:', f.loc[selected,['match_no','home_team','away_team']].to_dict('records'),flush=True)
    plan=f[['match_no','home_team','away_team']+PROB_COLUMNS].copy()
    for j,t in enumerate(v24,1): plan[f'ticket{j:02}']=np.array([1,0,2])[t]
    bundle=build(f,plan)
    raw=bundle[0][[f'ticket{j:02}' for j in range(1,11)]].to_numpy().T.astype(int)
    c95=np.array([1,0,2])[raw]
    (OUT/'c95_audit.json').write_text(json.dumps(bundle[3],ensure_ascii=False,indent=2))
    # Results enter only after all choices and tickets have been finalized.
    actual=pd.read_csv(source/'actual_results.csv').sort_values('match_no').reset_index(drop=True)
    for c in ['match_no','home_team','away_team']: assert f[c].tolist()==actual[c].tolist()
    y=np.array([1,0,2])[actual.result.astype(int).to_numpy()]
    rows=[]; grids=[]
    for name,tickets in [('候補00（参考）',anchor[None,:]),('v2.4（同入力再現）',v24),('現シャドーC95（同入力再現）',c95),('候補00＋4ダブル',double)]:
        hits=(tickets==y).sum(1)
        rows.append({'方式':name,'口数':len(tickets),'金額':len(tickets)*100,'最高的中':int(hits.max()),'平均的中':float(hits.mean()),'全選択の的中率％':float(hits.mean()/13*100),'13的中口数':int((hits==13).sum()),'各口の的中数':'・'.join(map(str,hits))})
        grid=f[['match_no','home_team','away_team']].copy()
        grid['候補00']=['HDA'[v] for v in anchor]
        grid['実結果']=['HDA'[v] for v in y]
        for j,t in enumerate(tickets,1):grid[f'券{j:02}']=['HDA'[v] for v in t]
        grid.to_csv(OUT/f'grid_{len(grids)}.csv',index=False)
        grids.append('<h2>'+name+'</h2>'+grid.to_html(index=False))
    report=pd.DataFrame(rows);report.to_csv(OUT/'comparison.csv',index=False)
    detail=f[['match_no','home_team','away_team']].copy()
    detail['候補00']=['HDA'[v] for v in anchor]
    detail['購入指定']=[('HDA'[v]+'/D') if i in selected else 'HDA'[v] for i,v in enumerate(anchor)]
    detail['D確率％']=p[:,1]*100
    detail['低得点確率％']=features.low_total_le2*100
    detail['構造評価Q']=features.Q_corrected
    detail['順位合計']=features['順位合計']
    detail['実結果']=['HDA'[v] for v in y]
    detail.to_csv(OUT/'purchase.csv',index=False)
    text='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>1665回 4ダブル比較</title><style>body{font:16px/1.7 system-ui;padding:24px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px}th{background:#eef2f5}</style><h1>1665回：v2.4・C95・候補00＋4ダブル</h1><p>保存予測を共通使用。当時の正式版はv2.3のため、v2.4は既存の同入力再現券、C95は現在の実装で再現。正式ファイルは変更していない。</p><p>4ダブルは16口1,600円、比較対象は10口1,000円。最高的中数の比較には口数差がある。全選択の的中率は平均的中数÷13であり、13的中する確率ではない。1開催の結果から将来の的中率は確定できない。</p><p>選出ルール（暫定）：D確率順位＋構造評価Q順位の上位4試合。同点はD確率優先。Q＝L×[B＋(1−B)×N]。Lは合計2得点以下の確率、BはH/A勝ちの拮抗度、Nは優勢側の勝ちのうち1点差になる割合。Qは配券評価であり確率ではない。結果は選出・最適化終了後に採点のみに使用。Qの有効性を実証したものではない。</p>'''
    text+=report.to_html(index=False)+'<h2>ダブル購入指定と選出根拠</h2>'+detail.to_html(index=False)+''.join(grids)+'</html>'
    (OUT/'index.html').write_text(text)
    print(report.to_string(index=False),flush=True)
    print(detail.loc[selected].to_string(index=False),flush=True)
    print(OUT/'index.html',flush=True)

if __name__=='__main__': main()
