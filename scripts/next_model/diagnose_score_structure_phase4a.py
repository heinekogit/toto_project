"""Exploratory 78-match score diagnostics from the append-only registry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from common import ROOT,sha256
from forecast_registry import DEFAULT_ROOT,selected_matches
from analyze_score_structure import score_grid,structure

OUT=ROOT/'data/reports/score_structure_phase4a_stage1'
GROUPS={'all':'全78試合'}


def poisson_binomial_interval(probabilities):
    distribution=np.array([1.])
    for p in probabilities:distribution=np.convolve(distribution,[1-p,p])
    cdf=np.cumsum(distribution)
    return int(np.searchsorted(cdf,.025)),int(np.searchsorted(cdf,.975)),distribution


def diagnostics(registry=DEFAULT_ROOT):
    source=selected_matches(registry)
    if len(source)!=78 or source.fixture_id.duplicated().any():
        raise ValueError('Expected 78 unique registered A fixtures')
    if not source.provenance_class.eq('A').all() or source.actual_home_score.isna().any():
        raise ValueError('Only complete A originals are accepted')
    rows=[]
    for _,r in source.iterrows():
        h,a=int(r.actual_home_score),int(r.actual_away_score)
        lh,la=float(r.lambda_home),float(r.lambda_away)
        matrix,_=score_grid(lh,la);s=structure(matrix)
        raw=np.array([r.raw_prob_home,r.raw_prob_draw,r.raw_prob_away],float)
        final=np.array([r.final_prob_home,r.final_prob_draw,r.final_prob_away],float)
        np.testing.assert_allclose(raw,[np.tril(matrix,-1).sum(),np.trace(matrix),np.triu(matrix,1).sum()],rtol=0,atol=1e-10)
        np.testing.assert_allclose(raw.sum(),1,rtol=0,atol=1e-10)
        np.testing.assert_allclose(final.sum(),1,rtol=0,atol=1e-10)
        result_index=0 if h>a else 1 if h==a else 2
        y=np.eye(3)[result_index]
        actual_category=('0-0' if h==a==0 else '1-1' if h==a==1 else '2-2以上' if h==a else
                         'H1点差' if h-a==1 else 'A1点差' if a-h==1 else '2点差以上')
        records=dict(round=r.round_id,match_no=int(r.match_no),fixture_id=r.fixture_id,
            league=r.league,home_team=r.home_team,away_team=r.away_team,
            model_signature=r.model_signature,kickoff=r.kickoff,
            lambda_home=lh,lambda_away=la,actual_home=h,actual_away=a,
            actual_category=actual_category,actual_result=result_index,
            uncertainty_mix=float(r.uncertainty_mix),
            raw_P_H=raw[0],raw_P_D=raw[1],raw_P_A=raw[2],
            final_P_H=final[0],final_P_D=final[1],final_P_A=final[2],
            goal_log_likelihood=float(poisson.logpmf(h,lh)+poisson.logpmf(a,la)),
            raw_HDA_log_loss=float(-np.log(max(raw[result_index],1e-15))),
            final_HDA_log_loss=float(-np.log(max(final[result_index],1e-15))),
            raw_HDA_brier=float(((raw-y)**2).sum()),
            final_HDA_brier=float(((final-y)**2).sum()))
        for side,observed,predicted in [('home',h,lh),('away',a,la)]:
            lo,hi=int(poisson.ppf(.05,predicted)),int(poisson.ppf(.95,predicted))
            records.update({f'residual_{side}':observed-predicted,
                            f'goal_log_likelihood_{side}':float(poisson.logpmf(observed,predicted)),
                            f'pearson_{side}':(observed-predicted)/np.sqrt(predicted),
                            f'interval90_low_{side}':lo,f'interval90_high_{side}':hi,
                            f'interval90_covered_{side}':bool(lo<=observed<=hi)})
        records.update(expected_00=s['P_0_0'],expected_11=s['P_1_1'],
                       expected_22plus=s['P_draw_2plus'],expected_H1=s['P_home_1'],
                       expected_A1=s['P_away_1'],expected_margin2plus=s['P_margin_2plus'])
        rows.append(records)
    return pd.DataFrame(rows).sort_values(['round','match_no']).reset_index(drop=True)


def summarize(frame,keys):
    result=[]
    grouped=[('全体',frame)] if not keys else [(str(key),g) for key,g in frame.groupby(keys)]
    for name,g in grouped:
        for side in ('home','away'):
            observed=g[f'actual_{side}'];lam=g[f'lambda_{side}'];residual=g[f'residual_{side}']
            result.append(dict(group=name,side=side,matches=len(g),rounds=','.join(sorted(g['round'].unique())),
                predicted_goals=float(lam.sum()),actual_goals=int(observed.sum()),
                predicted_mean=float(lam.mean()),actual_mean=float(observed.mean()),
                residual_sum=float(residual.sum()),residual_mean=float(residual.mean()),
                actual_to_expected_ratio=float(observed.sum()/lam.sum()),
                mean_negative_goal_log_likelihood=float(-g[f'goal_log_likelihood_{side}'].mean()),
                pearson_dispersion=float(((residual**2/lam).sum())/len(g)),
                interval90_coverage=float(g[f'interval90_covered_{side}'].mean())))
    return pd.DataFrame(result)


def category_table(frame):
    categories=[('0-0','expected_00'),('1-1','expected_11'),('2-2以上','expected_22plus'),
                ('H1点差','expected_H1'),('A1点差','expected_A1'),('2点差以上','expected_margin2plus')]
    return pd.DataFrame([dict(category=label,matches=len(frame),expected=float(frame[column].sum()),
                              actual=int(frame.actual_category.eq(label).sum()),
                              difference_actual_minus_expected=int(frame.actual_category.eq(label).sum())-float(frame[column].sum()))
                         for label,column in categories])


def draw_table(frame,keys):
    grouped=[('全体',frame)] if not keys else [(str(key),g) for key,g in frame.groupby(keys)]
    rows=[]
    for name,g in grouped:
        p=g.raw_P_D.to_numpy(float);actual=int(g.actual_result.eq(1).sum())
        lo,hi,_=poisson_binomial_interval(p)
        rows.append(dict(group=name,matches=len(g),rounds=','.join(sorted(g['round'].unique())),
                         expected_D=float(p.sum()),actual_D=actual,difference=actual-float(p.sum()),
                         independent_sd=float(np.sqrt((p*(1-p)).sum())),
                         independent_95pct_low=lo,independent_95pct_high=hi))
    return pd.DataFrame(rows)


def hda_table(frame,keys):
    grouped=[('全体',frame)] if not keys else [(str(key),g) for key,g in frame.groupby(keys)]
    rows=[]
    for name,g in grouped:
        for stage in ('raw','final'):
            row=dict(group=name,stage=stage,matches=len(g),
                     log_loss=float(g[f'{stage}_HDA_log_loss'].mean()),
                     brier=float(g[f'{stage}_HDA_brier'].mean()))
            for side,index in [('H',0),('D',1),('A',2)]:
                row[f'mean_P_{side}']=float(g[f'{stage}_P_{side}'].mean())
                row[f'actual_rate_{side}']=float(g.actual_result.eq(index).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def goal_frequency(frame):
    rows=[]
    for side in ('home','away'):
        lam=frame[f'lambda_{side}'].to_numpy(float);actual=frame[f'actual_{side}'].to_numpy(int)
        for goals in range(7):
            expected=float(poisson.pmf(goals,lam).sum()) if goals<6 else float(poisson.sf(5,lam).sum())
            observed=int((actual==goals).sum()) if goals<6 else int((actual>=6).sum())
            rows.append(dict(side=side,goals=str(goals) if goals<6 else '6以上',matches=len(frame),
                             expected=expected,actual=observed,difference=observed-expected))
    return pd.DataFrame(rows)


def calibration_bins(frame):
    rows=[]
    for stage in ('raw','final'):
        for symbol,index in (('H',0),('D',1),('A',2)):
            probabilities=frame[f'{stage}_P_{symbol}']
            for lo,hi in ((0.,.25),(.25,.35),(.35,.45),(.45,.55),(.55,1.000001)):
                g=frame[(probabilities>=lo)&(probabilities<hi)]
                if g.empty:continue
                rows.append(dict(stage=stage,symbol=symbol,lower=lo,upper=min(hi,1.),matches=len(g),
                                 mean_predicted=float(g[f'{stage}_P_{symbol}'].mean()),
                                 actual_rate=float(g.actual_result.eq(index).mean())))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before={name:sha256(DEFAULT_ROOT/name) for name in ('forecasts.jsonl','results.jsonl')}
    frame=diagnostics();frame.to_csv(OUT/'match_diagnostics.csv',index=False,encoding='utf-8-sig')
    goals=pd.concat([summarize(frame,[]),summarize(frame,['league']),summarize(frame,['model_signature'])],ignore_index=True)
    goals.to_csv(OUT/'goal_summary.csv',index=False,encoding='utf-8-sig')
    draws=pd.concat([draw_table(frame,[]),draw_table(frame,['league']),draw_table(frame,['model_signature'])],ignore_index=True)
    draws.to_csv(OUT/'draw_frequency.csv',index=False,encoding='utf-8-sig')
    cats=category_table(frame);cats.to_csv(OUT/'score_structure_frequency.csv',index=False,encoding='utf-8-sig')
    hda=pd.concat([hda_table(frame,[]),hda_table(frame,['model_signature']),
                   hda_table(frame[frame.uncertainty_mix.gt(0)],[])],ignore_index=True)
    hda.loc[hda.index[-2:],'group']='mix適用39試合'
    hda.to_csv(OUT/'hda_evaluation.csv',index=False,encoding='utf-8-sig')
    frequency=goal_frequency(frame);frequency.to_csv(OUT/'goal_frequency.csv',index=False,encoding='utf-8-sig')
    bins=calibration_bins(frame);bins.to_csv(OUT/'calibration_bins.csv',index=False,encoding='utf-8-sig')
    after={name:sha256(DEFAULT_ROOT/name) for name in before}
    if before!=after:raise AssertionError('Registry changed during diagnosis')
    audit=dict(matches=len(frame),rounds=sorted(frame['round'].unique().tolist()),
        A_only=True,C_recomputed_used=False,registry_hashes=before,
        independent_poisson_binomial_for_draw_count=True,
        pearson_dispersion_denominator='N per side; no parameters fitted',
        goal_interval='Poisson 5th through 95th percentile, discrete conservative coverage',
        brier='sum of three squared errors, not divided by three',
        model_or_buyplan_changed=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    def tab(d,cols):return '<div class="scroll">'+d[cols].to_html(index=False,escape=True,float_format=lambda x:f'{x:.4f}')+'</div>'
    def bars(d,label):return '<div>'+''.join(f'<p>{r[label]}：予測 {r.expected:.2f}／実際 {r.actual}　<span style="display:inline-block;background:#648aaa;height:12px;width:{max(1,r.expected*13):.0f}px"></span><span style="display:inline-block;background:#d99655;height:12px;width:{max(1,r.actual*13):.0f}px"></span></p>' for _,r in d.iterrows())+'</div>'
    mix=hda[hda.group.eq('mix適用39試合')]
    html=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 4-A 第1段階</title><style>body{{font:16px/1.7 system-ui;margin:28px;color:#203047}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{border:1px solid #ccd4df;padding:6px}}th{{background:#eaf1f7}}</style><h1>Phase 4-A：予測履歴と78試合の簡易診断</h1>
<p>原本Aの6開催78試合、J1=58・J2=20。試合前保存のmanifestとハッシュを検証して登録。結果は別の追記ログに保存。診断は1実試合1予測（同じ試合に複数あれば最も早い確認済み予測）で独立再実行。後日再計算のCは含めない。</p>
<h2>得点期待値と実得点</h2>{tab(goals,['group','side','matches','predicted_goals','actual_goals','residual_sum','actual_to_expected_ratio','pearson_dispersion','interval90_coverage'])}
<p>Pearson分散は各側でΣ(得点−λ)²/λ÷N。推定パラメータを差し引いた自由度調整なし。区間は離散Poissonの5～95パーセンタイルで名目90%より保守的になり得る。1試合の残差をλ誤差とは呼ばない。</p>
<h2>Dの期待件数と実際</h2>{tab(draws,['group','matches','expected_D','actual_D','difference','independent_sd','independent_95pct_low','independent_95pct_high'])}<p>帯は各試合のraw P(D)を使った独立Poisson二項分布の約95%範囲。同一開催内の依存性は考慮していない。期待件数との差だけでD予測の系統誤差を確定しない。</p>
<h2>得点構造別の期待件数と実際</h2>{bars(cats,'category')}{tab(cats,['category','expected','actual','difference_actual_minus_expected'])}<p>橙が実際、青が予測期待件数。0-0、1-1、2-2以上、H/A各1点差、2点差以上は互いに重複しない分類。</p>
<h2>raw / final H/D/A</h2>{tab(hda,['group','stage','matches','log_loss','brier','mean_P_H','actual_rate_H','mean_P_D','actual_rate_D','mean_P_A','actual_rate_A'])}<p>mix適用39試合のraw/finalは同じ試合のペア比較。初期39試合のmix=0相当と後期39試合を横断してmixの因果効果を推定しない。</p>
<h2>得点数別頻度</h2>{tab(frequency,['side','goals','expected','actual','difference'])}
<h2>確率帯別の校正（少数標本）</h2>{tab(bins,['stage','symbol','lower','upper','matches','mean_predicted','actual_rate'])}
<p>この78試合は予測処理3系統に分かれ、診断結果は探索的。真のλ誤差、ポアソンからの追加分散、相関を安定推定したものではない。次に同一方式の試合前原本を蓄積し、結果確定時刻と入力データの利用可能時刻を記録し、時系列検証用の未使用期間を確保する。</p>
<p><a href="match_diagnostics.csv">試合別</a> ／ <a href="goal_summary.csv">得点集計</a> ／ <a href="draw_frequency.csv">D集計</a> ／ <a href="score_structure_frequency.csv">構造集計</a> ／ <a href="hda_evaluation.csv">確率評価</a> ／ <a href="calibration_bins.csv">校正表</a> ／ <a href="audit.json">監査</a></p></html>'''
    (OUT/'index.html').write_text(html)
    print(json.dumps(dict(draw_expected=float(draws.iloc[0].expected_D),draw_actual=int(draws.iloc[0].actual_D),
                          home_goal_ratio=float(goals.iloc[0].actual_to_expected_ratio),
                          away_goal_ratio=float(goals.iloc[1].actual_to_expected_ratio),
                          mix_logloss_raw=float(mix.iloc[0].log_loss),mix_logloss_final=float(mix.iloc[1].log_loss)),ensure_ascii=False))


if __name__=='__main__':main()
