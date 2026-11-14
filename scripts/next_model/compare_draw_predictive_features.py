"""Retrospective expanding-window D probes; never modifies production forecasts."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from audit_structural_draw_q import load_data, ROUNDS, predictive_metrics
from simulate_free_structural_draw import ROOT, structure, sha, page, table

OUT=ROOT/'data/reports/draw_predictive_features'
FEATURES={'P_D_only':['logit_PD'], 'P_D_plus_Q':['logit_PD','Q'],
          'P_D_plus_score_structure':['logit_PD','draw_share_00','draw_share_11','draw_share_22']}


def fit(train,test,columns):
    x=train[columns].to_numpy(); xt=test[columns].to_numpy()
    center=x.mean(0); scale=np.maximum(x.std(0),1e-8)
    x=np.c_[np.ones(len(x)),(x-center)/scale]; xt=np.c_[np.ones(len(xt)),(xt-center)/scale]
    y=train.actual_D.to_numpy()
    def obj(b):
        z=x@b
        return np.logaddexp(0,z).sum()-y@z+.5*(b[1:]@b[1:]), x.T@(expit(z)-y)+np.r_[0,b[1:]]
    fitted=minimize(obj,np.zeros(x.shape[1]),jac=True,method='BFGS',options={'gtol':1e-8})
    assert fitted.success or np.linalg.norm(fitted.jac)<1e-5
    return expit(xt@fitted.x),dict(features=columns,center=center.tolist(),scale=scale.tolist(),beta=fitted.x.tolist())


def auc(y,p):
    y=np.asarray(y); n1=y.sum(); n0=len(y)-n1
    if not n1 or not n0: return float('nan')
    return float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def main():
    OUT.mkdir(exist_ok=True)
    protected=[p for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected}
    data,provenance=load_data('corrected')
    extras=[]
    for rid in ROUNDS:
        f=pd.read_csv(ROOT/f'data/eval/toto_rounds/{rid}/snapshot/next_predictions.csv').sort_values('match_no').reset_index(drop=True)
        s=structure(f,'corrected')
        e=f[['match_no']].copy(); e['round']=rid
        e['P_00']=s.P_00; e['P_11']=s.P_11
        e['P_22']=s.P_00*(s.lambda_home*s.lambda_away)**2/4
        e['score_P_D']=s.score_P_D
        for k in ['00','11','22']: e['draw_share_'+k]=e['P_'+k]/e.score_P_D
        e['draw_share_33plus']=1-e[['draw_share_00','draw_share_11','draw_share_22']].sum(axis=1)
        assert (e.draw_share_33plus>=-1e-10).all()
        extras.append(e)
    data=data.merge(pd.concat(extras),on=['round','match_no'],validate='one_to_one')
    data['logit_PD']=logit(data.P_D)
    data.to_csv(OUT/'matches.csv',index=False)
    # Q is a rating, not a probability: compare its ranking using AUC only.
    ranking=[]
    for label,d in [('all_65_descriptive',data)]+list(data.groupby('round')):
        for score in ['P_D','Q']:
            ranking.append(dict(scope=label,score=score,matches=len(d),draws=int(d.actual_D.sum()),AUC=auc(d.actual_D,d[score])))
    ranking=pd.DataFrame(ranking); ranking.to_csv(OUT/'ranking.csv',index=False)
    predictions=[]; fits=[]
    for pos in (3,4):
        train=data[data['round'].isin(ROUNDS[:pos])]; test=data[data['round'].eq(ROUNDS[pos])].copy()
        assert pd.to_datetime(train.datetime).max()<pd.to_datetime(test.datetime).min()
        test['raw_P_D']=test.P_D
        for name,cols in FEATURES.items():
            test[name],info=fit(train,test,cols)
            fits.append(dict(model=name,train_rounds=ROUNDS[:pos],test_round=ROUNDS[pos],**info))
        predictions.append(test)
    pred=pd.concat(predictions); pred.to_csv(OUT/'temporal_predictions.csv',index=False)
    results=[]
    for label,d in [('pooled_26',pred)]+list(pred.groupby('round')):
        for name in ['raw_P_D']+list(FEATURES):
            results.append(dict(scope=label,model=name,matches=len(d),draws=int(d.actual_D.sum()),AUC=auc(d.actual_D,d[name]),**predictive_metrics(d.actual_D.to_numpy(),d[name].to_numpy())))
    summary=pd.DataFrame(results); summary.to_csv(OUT/'temporal_metrics.csv',index=False)
    # Paired per-match losses let reviewers see whether a few fixtures drive differences.
    loss=[]
    for _,r in pred.iterrows():
        for name in FEATURES:
            b=float(r.P_D); p=float(r[name]); y=int(r.actual_D)
            loss.append(dict(round=r['round'],match_no=r.match_no,model=name,actual_D=y,raw_P_D=b,probe_P_D=p,delta_Brier=(p-y)**2-(b-y)**2,delta_LogLoss=float(-(y*np.log(p)+(1-y)*np.log1p(-p))+(y*np.log(b)+(1-y)*np.log1p(-b)))))
    pd.DataFrame(loss).to_csv(OUT/'paired_loss_changes.csv',index=False)
    config=dict(rounds=ROUNDS,source_manifest=provenance,features=FEATURES,L2=1,temporal_fits=fits,production_changed=False,
                caveat='Retrospective feature design after historical outcomes were known. Chronological coefficient fitting is not an independent prospective holdout.',
                score_structure='Conditional shares of 0-0,1-1,2-2 among draws; 3-3+ omitted to avoid exact sum collinearity. Shares remain correlated; no claim of independent information.',
                portfolio_evaluation='Not performed: this is a binary D probe, not a complete H/D/A forecast or ticket-selection change.')
    assert all(sha(p)==h for p,h in hashes.items())
    config['unchanged_production_sha256']=hashes
    (OUT/'audit.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    pooled=summary[summary.scope.eq('pooled_26')]
    body='<p>α・選別閾値・予測確率・購入券を変更しない、複数開催の診断です。65試合の順位識別と、後続26試合の確率予測を分けて評価します。</p>'
    body+='<h2>後続開催の確率評価</h2>'+table(pooled)
    body+='<p>BrierとLog Lossは小さいほど良く、AUCは大きいほどDと非Dを順位付けできています。Qを確率と見なして誤差を計算していません。</p>'
    body+='<h2>比較方法</h2><p>元P(D)をそのまま使う基準に加え、logit(P(D))のみ、そこへQを追加、または同点スコア構成比を追加するロジスティック回帰を比較。固定L2=1で傾きだけを正則化し、標準化も学習期間のみから算出。係数・閾値を評価結果に合わせて調整していません。</p><p>構成比はP(0–0)/P(同点)、P(1–1)/P(同点)、P(2–2)/P(同点)。D確率そのものの単純な重複加算を避ける目的です。残りの3–3以上は和が1になる完全な共線性を避けるため説明変数から省略。ただし他の構成比も互いに相関があり、独立した新情報とは限りません。生のスコア確率もCSVに記録しています。</p>'
    body+='<h2>時系列分離</h2><p>1660・1663・1664で学習→1665で評価。次に1665までで学習→1668で評価。評価対象自身の結果を係数学習に使わず、試合日時の前後関係も検査しました。全65試合の保存予測は試合前生成・ハッシュ一致を確認済みです。</p><p>ただしQや今回の構造特徴の設計時には過去結果が既知でした。これは時系列で分けた事後検証であり、未見の将来開催による独立検証ではありません。5開催・後続評価26試合だけで採用・不採用を一般化できません。</p>'
    body+='<h2>開催ごとの後続評価</h2>'+table(summary[~summary.scope.eq('pooled_26')])
    body+='<h2>元P(D)とQの順位識別（記述的比較）</h2>'+table(ranking)
    body+='<h2>13的中の評価とは別</h2><p>今回は二値のD確率評価です。新しいH/A確率や10口を生成していないため、13的中確率の改善は主張しません。追加情報の有効性が確認された後も、完全なH/D/A予測と候補選択方式を明示した別検証が必要です。比較する券同士は同じ確率モデルで評価し、券ごとに自分に有利な補正確率を使って改善扱いしないことが重要です。</p>'
    for f in ['matches.csv','ranking.csv','temporal_metrics.csv','temporal_predictions.csv','paired_loss_changes.csv','audit.json']: body+=f'<p><a href="{f}">{f}</a></p>'
    (OUT/'index.html').write_text(page('D確率・Q・同点スコア構造：後続開催での比較',body))
    print(pooled.to_string(index=False));print(ranking[ranking.scope.eq('all_65_descriptive')].to_string(index=False));print(OUT/'index.html')

if __name__=='__main__':main()
