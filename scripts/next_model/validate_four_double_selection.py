"""Historical, chronological evaluation of four-double selection only."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from four_double_selection_shadow import features,select,expand,probe,VARIANTS,MIN_TRAIN,MAX_ADJUSTMENT
from simulate_free_structural_draw import PROB_COLUMNS, page, table

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/reports/four_double_selection_shadow'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected={str(p):sha(p) for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()}
    history=[]; rows=[]; allmatches=[]; ticketrows=[]; changes=[]; provenance=[]; fits=[]
    sources=sorted((ROOT/'data/eval/toto_rounds').glob('*/snapshot/next_predictions.csv'))
    for source in sources:
        rid=source.parent.parent.name; manifest_path=source.parent/'buyplan_engine_manifest.json'
        manifest=json.loads(manifest_path.read_text()); f=pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
        digest=sha(source); kickoff=pd.to_datetime(f.datetime)
        if kickoff.dt.tz is None: kickoff=kickoff.dt.tz_localize('Asia/Tokyo')
        verified=(manifest.get('output_sha256',{}).get('next_predictions.csv')==digest and
                  bool((kickoff>pd.Timestamp(manifest['created_at'])).all()))
        info=dict(round=rid,forecast=str(source),sha256=digest,created_at=manifest['created_at'],prematch_verified=verified)
        provenance.append(info)
        if not verified: info['excluded']='generation/hash check failed'; continue
        try: z=features(f)
        except ValueError as exc: info['excluded']=str(exc); continue
        assert len(z)==13 and f.match_no.tolist()==list(range(1,14))
        z['round']=rid; p=f[PROB_COLUMNS].to_numpy(); np.testing.assert_allclose(p.sum(1),1)
        scores={'A_PD':z.P_D.to_numpy()}; train=pd.concat(history,ignore_index=True) if history else pd.DataFrame()
        if len(train)>=MIN_TRAIN:
            assert pd.to_datetime(train.datetime).max()<pd.to_datetime(f.datetime).min()
            for name,cols in VARIANTS.items():
                scores[name],audit=probe(train,z,cols); fits.append(dict(test_round=rid,method=name,**audit))
        choices={name:select(z,s) for name,s in scores.items()}
        # Current results are accessed only after current choices are frozen.
        result_path=source.parent.parent/'actual_results.csv'
        actual=pd.read_csv(result_path).sort_values('match_no').reset_index(drop=True) if result_path.exists() else None
        complete=actual is not None and len(actual)==13 and actual.result.isin([0,1,2]).all()
        if complete:
            for c in ['match_no','home_team','away_team']: assert f[c].tolist()==actual[c].tolist()
            y=np.array([1,0,2])[actual.result.astype(int).to_numpy()]
            z['actual_D']=(y==1).astype(int); z['actual_symbol']=['HDA'[v] for v in y]
            info['results_sha256']=sha(result_path)
        else: y=None; z['actual_D']=np.nan; z['actual_symbol']='未確定'
        info['complete_results']=bool(complete)
        base=set(choices['A_PD'])
        for name,chosen in choices.items():
            tickets=expand(z.anchor.to_numpy(),chosen)
            fixed=np.ones(13,dtype=bool); fixed[chosen]=False
            mass=np.prod(p[np.arange(13),tickets],axis=1).sum()
            direct=np.prod(p[np.flatnonzero(fixed),z.anchor.to_numpy()[fixed]])*np.prod(p[chosen,z.anchor.to_numpy()[chosen]]+p[chosen,1])
            np.testing.assert_allclose(mass,direct,rtol=1e-12)
            capture=int((y[chosen]==1).sum()) if complete else None
            total=int((y==1).sum()) if complete else None
            hit=(tickets==y).sum(1) if complete else None
            rows.append(dict(round=rid,method=name,selected=','.join(map(str,z.match_no.iloc[chosen])),actual_D=total,
                captured_D=capture,selection_precision=capture/4 if complete else None,
                D_recall=capture/total if total else None,best_hits=int(hit.max()) if complete else None,
                anchor_hits=int((z.anchor.to_numpy()==y).sum()) if complete else None,
                model_P13=float(mass),changed_matches=len(set(chosen)-base),
                capture_delta=capture-int((y[list(base)]==1).sum()) if complete else None,
                cost=1600,comparison_eligible=len(train)>=MIN_TRAIN))
            for i,t in enumerate(tickets):
                ticketrows.append(dict(round=rid,method=name,ticket=i+1,symbols=''.join('HDA'[v] for v in t),hits=int(hit[i]) if complete else None))
            for i in range(13):
                changes.append(dict(round=rid,method=name,match_no=int(z.match_no.iloc[i]),home=f.home_team.iloc[i],away=f.away_team.iloc[i],
                    base_selected=i in base,selected=i in chosen,P_D=float(z.P_D.iloc[i]),selection_score=float(scores[name][i]),
                    correction=float(scores[name][i]-z.P_D.iloc[i]),actual_D=int(y[i]==1) if complete else None,
                    reason='P(D)順位' if name=='A_PD' else '過去開催で学習した構造残差補正（上限±1ポイント）'))
        allmatches.append(z)
        if complete: history.append(z)
    metrics=pd.DataFrame(rows); matches=pd.concat(allmatches,ignore_index=True); change=pd.DataFrame(changes)
    summary=[]
    for name,g in metrics[metrics.comparison_eligible & metrics.actual_D.notna()].groupby('method'):
        summary.append(dict(method=name,rounds=len(g),actual_D=int(g.actual_D.sum()),captured_D=int(g.captured_D.sum()),
            recall=float(g.captured_D.sum()/g.actual_D.sum()) if g.actual_D.sum() else None,
            changed_matches=int(g.changed_matches.sum()),capture_delta=int(g.capture_delta.sum()),
            improved_rounds=int((g.capture_delta>0).sum()),worsened_rounds=int((g.capture_delta<0).sum())))
    summary=pd.DataFrame(summary)
    # Diagnostic C: unused H/A is never a feature of either fitted probe.
    matches['unused_HA_band']=pd.cut(matches.unused_HA_probability,[0,.25,.30,.35,1],include_lowest=True).astype(str)
    diag=matches[matches.actual_D.notna()].groupby('unused_HA_band').agg(matches=('actual_D','size'),draws=('actual_D','sum'),mean_PD=('P_D','mean'),D_rate=('actual_D','mean')).reset_index()
    for data,name in [(metrics,'comparison'),(summary,'summary'),(matches,'features'),(change,'selection_changes'),(pd.DataFrame(ticketrows),'tickets'),(diag,'unused_HA_diagnostic')]: data.to_csv(OUT/f'{name}.csv',index=False)
    assert all(sha(Path(p))==h for p,h in protected.items())
    audit=dict(default='A_PD',adoption='P(D)上位4を維持。補正方式は事後設計の少数開催検証のため採用しない。',
        scope='shadow only',config=dict(min_train=MIN_TRAIN,max_adjustment=MAX_ADJUSTMENT,variants=VARIANTS),
        provenance=provenance,fits=fits,production_hashes=protected,probabilities_changed=False,
        score_caveat='Residualization reduces measured overlap with P(D); it does not prove independent information.',
        historical_design_caveat='Historical outcomes were known during design; temporal evaluation is retrospective, not prospective validation.')
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    body='''<p>将来v3.0向け4ダブルのシャドー検証。基準券は各試合max(H,A)。4試合にDだけ追加、16口1,600円。標準選考はA：P(D)上位4を維持。</p>
<p>B1：同点内の0–0・1–1構成比。B2：B1に低得点×H/A拮抗、優勢側1点差割合×低得点同点の相対質量を追加。高得点のDを禁止する条件はない。0–0等をそのままP(D)へ加算せず、過去開催のP(D)の二次式で説明できる部分を除く。相関が残る特徴はL2=10で係数を抑制。logit(P(D))を固定した基準として構造係数のみ学習する。順位用補正は最大±0.01。新しい予測確率として扱わない。</p>
<p>最初の3開催39試合を学習開始に必要な暫定最小数とし、以降は過去開催のみで学習。2方式・固定L2・固定上限を今回比較内で調整していない。これらは正式な最適値ではない。試合前生成時刻とハッシュを確認。現在開催の結果は選出完了後に採点のみに使用する。特徴設計時には過去結果が既知なので独立した将来検証ではない。</p>'''
    body+='<h2>共通評価期間</h2>'+table(summary)+'<h2>開催別</h2>'+table(metrics)
    body+='<h2>選考変更の成否</h2>'+table(change[change.selected != change.base_selected])
    body+='<h2>C：不採用側H/A確率の診断</h2><p>以下は記述統計であり、P(D)との交絡がある。独立した補正係数には使用していない。</p>'+table(diag)
    for rid,g in matches.groupby('round'):
        body+=f'<h2>{rid} 全試合の診断</h2>'+table(g[['match_no','home_team','away_team','P_H','P_D','P_A','P_00','P_11','P_22','P_22plus','lambda_home','lambda_away','HA_balance','P_margin1','unused_HA_probability','actual_symbol']])
        body+=table(metrics[metrics['round']==rid])+table(pd.DataFrame(ticketrows).query('round == @rid'))
    body+='<p>結論：補正方式の正式採用は保留し、4ダブル選考の標準はP(D)上位4とする。最高的中数とD捕捉数は別に評価する。Dなし開催の捕捉率は未定義として空欄。</p>'
    for name in ['comparison','summary','features','selection_changes','tickets','unused_HA_diagnostic']: body+=f'<p><a href="{name}.csv">{name}.csv</a></p>'
    body+='<p><a href="audit.json">予測生成証跡・学習係数・設定</a></p>'
    (OUT/'index.html').write_text(page('4ダブル：D選考基準の比較検証',body))
    print(summary.to_string(index=False)); print(metrics.to_string(index=False)); print(OUT/'index.html')

if __name__=='__main__':main()
