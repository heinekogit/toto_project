"""Offline ten-ticket illustration of fitted D probes; no production changes."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from compare_draw_predictive_features import FEATURES, fit
from simulate_free_structural_draw import ROOT, structure, sha, page, table, metrics, candidates, PROB_COLUMNS
from simulate_free_portfolio_1664 import top_joint_candidates


def main():
    out=ROOT/'data/reports/toto1670_draw_probe_tickets';out.mkdir(exist_ok=True)
    source=ROOT/'data/reports/draw_proximity_fix/corrected/toto1670/forecast_input.csv'
    train_path=ROOT/'data/reports/draw_predictive_features/matches.csv'
    protected=[source,train_path]+[p for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected}
    f=pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
    train=pd.read_csv(train_path)
    assert pd.to_datetime(train.datetime).max()<pd.to_datetime(f.datetime).min()
    assert f.match_no.tolist()==list(range(1,14)) and not train.match_id.duplicated().any()
    s=structure(f,'corrected'); test=pd.DataFrame({'logit_PD':logit(f.next_prob_draw),'Q':s.Q_corrected})
    test['draw_share_00']=s.P_00/s.score_P_D;test['draw_share_11']=s.P_11/s.score_P_D
    test['draw_share_22']=s.P_00*(s.lambda_home*s.lambda_away)**2/4/s.score_P_D
    p=f[PROB_COLUMNS].to_numpy(); modes={'raw_P_D':p.copy()};fits={}
    for name,cols in FEATURES.items():
        pd_new,info=fit(train,test,cols); fits[name]=info
        trial=p.copy();trial[:,1]=pd_new
        trial[:,[0,2]]=p[:,[0,2]]*((1-pd_new)/(1-p[:,1]))[:,None]
        np.testing.assert_allclose(trial.sum(1),1)
        np.testing.assert_allclose(trial[:,0]/trial[:,2],p[:,0]/p[:,2])
        modes[name]=trial
    summaries=[]; bodies=[]; all_probs=[]
    for name,trial in modes.items():
        tickets,_=top_joint_candidates(trial,10)
        result,strata=metrics(p,tickets,None);summaries.append(dict(mode=name,**result))
        diag=candidates(f,p,tickets,trial,None);diag.to_csv(out/f'{name}_candidates.csv',index=False)
        grid=pd.DataFrame({'No':f.match_no,'試合':f.home_team+'－'+f.away_team,'参照00':['HDA'[v] for v in p.argmax(1)]})
        for j,t in enumerate(tickets,1):grid[f'{j:02}']=['HDA'[v] for v in t]
        grid.to_csv(out/f'{name}_tickets.csv',index=False)
        pd.DataFrame(strata).to_csv(out/f'{name}_draw_strata.csv',index=False)
        probs=f[['match_no','home_team','away_team']].copy()
        for i,c in enumerate('HDA'):probs['original_'+c]=p[:,i];probs['trial_'+c]=trial[:,i]
        probs['mode']=name;all_probs.append(probs)
        bodies.append('<h2>'+name+'</h2>'+table(grid)+table(diag[['candidate','reference_only','D_count','exchanges','original_P13']]))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'comparison.csv',index=False)
    pd.concat(all_probs).to_csv(out/'trial_probabilities.csv',index=False)
    shown=summary[['mode','D_total','D_counts','P_any13','duplicate_pairs']].copy();shown.P_any13=shown.P_any13.map(lambda z:f'{z*100:.6f}%')
    body='<p>第1670回の保存入力による非固定式の購入シミュレーション。各方式01〜10の10口。00は元確率最大の参照券で購入対象外。H=ホーム勝ち、D=引き分け、A=アウェー勝ち。</p>'
    body+='<p>1660・1663・1664・1665・1668の65試合で、前回と同じ固定L2=1の試験モデルを学習。1670回の結果は使用していません。暫定条件として試験D確率以外の残りを元のH:A比で配分し、その組み合わせ確率上位10口を厳密に選択。D枠数・類似度・交換数の制約はありません。v2.5固定枚数方式ではありません。</p>'
    body+='<p>HDAの試験値はこのレポート内だけの値で、正式予測・配券スコアを上書きしていません。先の後続26試合評価では追加特徴による改善は未確認です。今回は採用推奨ではなく配置の確認です。</p>'
    body+='<h2>共通の元確率での比較</h2>'+table(shown)
    body+='<p>13的中確率は試合間独立を仮定し、元予測確率による10券の確率を合計。全券は重複なしです。元確率の上位10組がこの指標で最大になるのは選択方法上の性質で、実際の成績の優位を証明しません。</p>'
    body+=''.join(bodies)
    for path in sorted(out.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    (out/'index.html').write_text(page('第1670回：D評価試験モデルによる10口比較',body))
    assert all(sha(path)==h for path,h in hashes.items())
    (out/'audit.json').write_text(json.dumps({'training_rounds':train['round'].unique().tolist(),'fits':fits,'unchanged_sha256':hashes,'production_changed':False,'results_used_for_1670':False},ensure_ascii=False,indent=2))
    print(shown.to_string(index=False))
    print(pd.read_csv(out/'P_D_plus_score_structure_tickets.csv').to_string(index=False))
    print(out/'index.html')

if __name__=='__main__': main()
