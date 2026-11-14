"""Reproduce six archived score grids and report d-scoop feasibility."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from d_scoop import diagnose,VERSION

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/reports/d_scoop_phase1'
ROUNDS=['toto1660','toto1663','toto1664','toto1665','toto1668','toto1670']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(f):return '<div class="scroll">'+f.to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</div>'
def main():
    protected={str(p):sha(p) for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()}
    OUT.mkdir(exist_ok=True,parents=True);allrows=[]; audits=[];bench=[];html_sections=[];matrices={}
    for rid in ROUNDS:
        base=ROOT/'data/eval/toto_rounds'/rid;src=base/'snapshot/next_predictions.csv';mp=base/'snapshot/buyplan_engine_manifest.json'
        m=json.loads(mp.read_text());f=pd.read_csv(src).sort_values('match_no').reset_index(drop=True)
        if m['output_sha256']['next_predictions.csv']!=sha(src):raise ValueError('Source checksum mismatch')
        dt=pd.to_datetime(f.datetime)
        if dt.dt.tz is None:dt=dt.dt.tz_localize('Asia/Tokyo')
        if not (dt>pd.Timestamp(m['created_at'])).all():raise ValueError('Post-kickoff source excluded')
        z,grid=diagnose(f);z['round']=rid;z['match_id']=f.match_id
        matrices[rid]=grid
        # Frozen historical Q comparison; not a d-scoop score or production selection.
        ph=z.score_P_H.to_numpy();pa=z.score_P_A.to_numpy()
        balance=2*np.minimum(ph,pa)/(ph+pa)
        fav_near=np.where(ph>=pa,z.P_margin_plus1/ph,z.P_margin_minus1/pa)
        from scipy.stats import poisson
        q=poisson.cdf(2,z.lambda_total)*(balance+(1-balance)*fav_near)
        z['legacy_Q']=q
        z['legacy_rank_sum']=z.P_D.rank(ascending=False,method='min')+pd.Series(q).rank(ascending=False,method='min')
        chosen_q=z.sort_values(['legacy_rank_sum','P_D','match_no'],ascending=[True,False,True]).head(4).index.to_numpy()
        z['legacy_Q_selected']=np.isin(z.index,chosen_q)
        # Actual results are appended after independent diagnostics and selection.
        ap=base/'actual_results.csv';a=pd.read_csv(ap).sort_values('match_no').reset_index(drop=True)
        for c in ['match_no','home_team','away_team']:assert f[c].tolist()==a[c].tolist()
        assert a.result.isin([0,1,2]).all()
        y=np.array([1,0,2])[a.result.astype(int).to_numpy()]
        p=z[['P_H','P_D','P_A']].to_numpy();anchor=np.where(p[:,0]>=p[:,2],0,2)
        wrong_ha=(y!=1)&(anchor!=y)
        z['actual_symbol']=['HDA'[v] for v in y];z['actual_score']=[f'{int(h)}–{int(a)}' for h,a in zip(a.home_score,a.away_score)]
        z['HA_reverse_error']=wrong_ha
        z['actual_D']=y==1
        for name,sel in [('A_PD',z.baseline_selected.to_numpy()),('B_legacy_Q_rank_sum',z.legacy_Q_selected.to_numpy())]:
            capture=int(((y==1)&sel).sum());fixed=~sel
            probability=float(np.prod(p[np.flatnonzero(fixed),anchor[fixed]])*np.prod(p[np.flatnonzero(sel),anchor[sel]]+p[sel,1]))
            best=int((anchor==y).sum()+capture)
            bench.append(dict(round=rid,method=name,selected=','.join(map(str,z.match_no[sel])),
                actual_D=int((y==1).sum()),captured_D=capture,D_precision=capture/4,D_recall=capture/int((y==1).sum()),
                anchor_hits=int((anchor==y).sum()),best_hits=best,perfect13=int(best==13),model_P13=probability,
                HA_reverse_errors=int(wrong_ha.sum()),HA_reverse_match_nos=','.join(map(str,z.match_no[wrong_ha])),
                perfect13_impossible_from_HA=bool(wrong_ha.any())))
        dest=OUT/rid;dest.mkdir(exist_ok=True)
        z.to_csv(dest/'diagnostics.csv',index=False)
        audit=dict(round=rid,forecast_sha256=sha(src),result_sha256=sha(ap),manifest_sha256=sha(mp),forecast_created_at=m['created_at'],
            archive_created_at=json.loads((src.parent/'manifest.json').read_text())['created_at'],prematch_verified=True,
            score_error_max=float(z.score_reconstruction_error.max()),HDA_error_max=float(z.HDA_reconstruction_error.max()),
            max_omitted_mass=float(z.omitted_tail_mass.max()),mix_min=float(z.uncertainty_mix.min()),mix_max=float(z.uncertainty_mix.max()),
            model=m.get('prediction_model','not recorded'))
        audits.append(audit);allrows.append(z)
        display=z[['match_no','home_team','away_team','P_D','score_P_D','PD_rank','baseline_selected','P_00','P_11','P_22','P_33plus','lambda_home','lambda_away','HA_probability_gap','P_margin1','draw_share_within_zero_or_one','actual_symbol','actual_score']]
        html_sections.append(f'<h2>{rid}：13試合の診断</h2><p><a href="{rid}/diagnostics.csv">全指標CSV</a></p>'+table(display))
    allf=pd.concat(allrows,ignore_index=True);assert not allf.match_id.duplicated().any()
    allf.to_csv(OUT/'all_diagnostics.csv',index=False);pd.DataFrame(bench).to_csv(OUT/'baseline_comparison.csv',index=False)
    np.savez_compressed(OUT/'score_matrices.npz',**matrices)
    audit=dict(version=VERSION,stage='Phase 1 only',score_formula=None,correction_adopted=False,
        matrix_shape='round key: (13,11,11); axis1=home goals; axis2=away goals',
        matrix_kind='normalized independent Poisson before HDA uncertainty correction',
        final_calibrated_score_matrix_available=False,source_audits=audits,production_unchanged=True,
        caveat='All features are deterministic functions of saved lambdas and uncertainty mix; no new external predictive information.',
        protected_hashes=protected)
    assert all(sha(Path(path))==h for path,h in protected.items())
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    text='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>d-scoop v0.1 Phase 1</title><style>body{font:16px/1.8 system-ui;margin:26px;color:#203147}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:6px;border:1px solid #ccd3dc}th{background:#eef3f8}</style><h1>d-scoop v0.1：得点分布の診断と実装調査</h1>
<p>6開催78試合の保存λから0〜10得点の11×11行列を再現。保存済み0–0・1–1・2–2および最終H/D/Aと照合。試合前生成の日時・ハッシュを確認。Next ModelとBuyPlanの動作は変更していません。</p>
<p>行列M(i,j)はPoisson(λH)×Poisson(λA)を0〜10の範囲で正規化したもの。補正前P(D)=ΣM(i,i)。最終next_prob_draw=(1−mix)×補正前P(D)+mix/3。疲労・天候の補正はH/D/Aだけに適用され、補正後のスコア行列は保存・定義されていません。補正前行列から最終P(D)を直接求めたように扱いません。</p>
<p>0–0、1–1、2–2、3–3以上、各チーム無得点、λ合計と差、H/A差、得点分布の重なり、±1点差、同点と±1点差の相対質量、得点差平均・標準偏差・全分布を診断CSVに記録。1点差の確率を試合中に追いつく確率とは扱いません。</p>
<p>d_scoop_score・順位・採用フラグは未決定（空欄）。P(D)順位と基準選出フラグは別列。Phase 1では合算係数を作りません。新方式Cは未実装のため比較成績を捏造せず、Aと以前のQ順位和案Bのみ再集計しています。</p>
<p>各構造指標はλ2個から派生し、独立した新情報ではありません。最終P(D)が同程度でも構造が違う可能性はありますが、補正に予測価値があるかは別途検証が必要です。78試合18Dのため、Phase 2は少数の事前定義した特徴群と時系列分離で調べる構成を推奨します。</p>'''
    text+='<h2>既存A・Bの再確認（Cは未決定）</h2>'+table(pd.DataFrame(bench))
    text+='<p>HA_reverse_errorsは実際にH/A決着した試合で基準H/Aが逆だった数。この外れはD追加で救済できず、1つでもある開催はどの4試合を選んでも13的中不能です。D選出の能力とは別の制約として記録します。</p>'
    text+=''.join(html_sections)+'<h2>再現性監査</h2>'+table(pd.DataFrame(audits))
    text+='<p><a href="all_diagnostics.csv">78試合全指標</a> ／ <a href="score_matrices.npz">全スコア行列</a> ／ <a href="baseline_comparison.csv">A/B比較</a> ／ <a href="audit.json">監査JSON</a></p></html>'
    (OUT/'index.html').write_text(text)
    print(pd.DataFrame(audits)[['round','score_error_max','HDA_error_max','max_omitted_mass','mix_min','mix_max']].to_string(index=False))
    print(pd.DataFrame(bench)[['round','method','captured_D','best_hits','HA_reverse_errors']].to_string(index=False));print(OUT/'index.html')

if __name__=='__main__':main()
