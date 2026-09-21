#!/usr/bin/env python3
"""Compare weather-uncertainty hypotheses in isolated, result-censored workspaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-dir', type=Path, default=ROOT/'data/eval/simulations/toto1650_latest')
    parser.add_argument('--out', type=Path, default=ROOT/'data/eval/simulations/toto1650_weather_uncertainty_shadow')
    args = parser.parse_args()
    source = Path(json.loads((args.replay_dir/'workspace.json').read_text())['isolated_workspace'])
    if not source.exists():
        raise FileNotFoundError('The result-censored replay workspace is required; rebuild the baseline replay first.')
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = ROOT/'data/eval/toto_rounds/toto1650/snapshot'
    actual = ROOT/'data/eval/toto_rounds/toto1650/actual_results.csv'
    before = pd.read_csv(snapshot/'predictions.csv')
    ordering = pd.read_csv(snapshot/'buyplan.csv')
    norm = lambda x: unicodedata.normalize('NFKC', str(x)).strip()
    summary = []
    baseline = None
    variants = [
        ('baseline', 0.00, 0.00, False),
        ('rank1_mild_style', 0.04, 0.15, False),
        ('side_favourite_mild_style', 0.04, 0.15, True),
        ('rank1_moderate_style', 0.06, 0.15, False),
    ]
    for label, max_shift, style_tilt, side_favourite_only in variants:
        out = args.out/label
        out.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f'toto1650_{label}_', dir='/private/tmp'))
        shutil.copytree(source/'data', work/'data', ignore=shutil.ignore_patterns('reports','output_snapshots','__pycache__'))
        shutil.copytree(source/'scripts', work/'scripts', ignore=shutil.ignore_patterns('__pycache__','.venv'))
        for file in source.glob('*.py'):
            shutil.copy2(file, work/file.name)
        (work/'logs').mkdir()
        (work/'data/reports').mkdir(exist_ok=True)
        predictor = work/'scripts/11_prediction_01.py'
        rules = work/'scripts/weather_style_shadow.py'
        shutil.copy2(ROOT/'scripts/weather_style_shadow.py', rules)
        text = predictor.read_text()
        marker = 'df_pred = add_buyplan_purchase_context(df_pred)'
        assert text.count(marker) == 1
        hook = (
            'from weather_style_shadow import apply_weather_uncertainty_shadow\n'
            f'df_pred = apply_weather_uncertainty_shadow(df_pred, max_shift={max_shift}, '
            f'style_tilt={style_tilt}, side_favourite_only={side_favourite_only})\n'
        )
        predictor.write_text(text.replace(marker, hook+marker))
        env = dict(os.environ, SEASON_YEAR='2026', STATS_ASOF_DATE='2026-09-05', WEATHER_ASOF_DATE='2026-09-05',
                   ENABLE_ROUND_TYPE_DRAW_CONTROL='0', SKIP_HFA_SELF_CHECK='1', PYTHONDONTWRITEBYTECODE='1')
        def command(cmd, log, league=None):
            with (out/log).open('w') as handle:
                subprocess.run([sys.executable]+cmd, cwd=work, env=dict(env, **({'LEAGUE':league} if league else {})), stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(f'{label}: predicting J1/J2', flush=True)
        for league in ['j1','j2']:
            command(['scripts/11_prediction_01.py'], f'{league}.log', league)
        predictions = pd.concat([pd.read_csv(work/f'{l}_2026_predictions_hfa_on.csv') for l in ['j1','j2']],ignore_index=True)
        rows = []
        for _, match in ordering.iterrows():
            old = before[before.home_team.eq(match.home_team)&before.away_team.eq(match.away_team)].iloc[0]
            hits = predictions[predictions.match_id.eq(old.match_id)]
            if len(hits) != 1:
                hits = predictions[predictions.home_team.map(norm).eq(norm(match.home_team)) & predictions.away_team.map(norm).eq(norm(match.away_team)) & pd.to_datetime(predictions.datetime).dt.date.eq(pd.Timestamp(old.datetime).date())]
                hits = hits.assign(delta=(pd.to_datetime(hits.datetime)-pd.Timestamp(old.datetime)).abs()).sort_values('delta').head(1).drop(columns='delta')
            assert len(hits) == 1
            row = hits.iloc[0].copy()
            row['match_no'] = match.match_no
            row['toto_round_id'] = 1650
            row['home_team'], row['away_team'] = match.home_team, match.away_team
            rows.append(row)
        result = pd.DataFrame(rows).reset_index(drop=True)
        for filename in ['predictions.csv','predictions_buyplan_context.csv']:
            result.to_csv(out/filename,index=False,encoding='utf-8-sig')
        command(['buyplan.py','--in',str(out/'predictions.csv'),'--context-csv',str(out/'predictions_buyplan_context.csv'),'--outdir',str(out),'--toto-order-csv',str(work/'data/manual/toto節リスト.csv')], 'buyplan.log')
        command([str(ROOT/'scripts/eval/02_score_buyplan.py'),'--round','toto1650','--buyplan',str(out/'buyplan.csv'),'--actual',str(actual),'--out',str(out/'evaluation.csv'),'--history',str(out/'history.csv')], 'score.log')
        command([str(ROOT/'scripts/eval/03_build_scored_html.py'),'--round','toto1650','--buyplan',str(out/'buyplan.csv'),'--actual',str(actual),'--evaluation',str(out/'evaluation.csv'),'--out',str(out/'buyplan_scored.html')], 'html.log')
        evaluation = pd.read_csv(out/'evaluation.csv')
        probs = result[['prob_final_home','prob_final_draw','prob_final_away']].to_numpy()
        if baseline is None:
            baseline = probs.copy()
        summary.append({'variant':label,'max_shift':max_shift,'style_tilt':style_tilt,'side_favourite_only':side_favourite_only,'mean_hits':evaluation.hits.mean(),'max_hits':evaluation.hits.max(),'mean_hit_rate':evaluation.hit_rate.mean(),'max_probability_delta':float(abs(probs-baseline).max()),'mean_entropy_delta':result.shadow_entropy_delta.mean(),'affected_matches':int(result.shadow_probability_shift.gt(0).sum())})
        (out/'provenance.json').write_text(json.dumps({'shadow_only':True,'max_shift':max_shift,'style_tilt':style_tilt,'side_favourite_only':side_favourite_only,'workspace':str(work),'base_replay':str(args.replay_dir),'weather_thresholds_changed':False,'result_used_for_adjustment':False,'predictor_sha256':hashlib.sha256(predictor.read_bytes()).hexdigest(),'weather_rules_sha256':hashlib.sha256(rules.read_bytes()).hexdigest()},indent=2))
        html = out/'buyplan_scored.html'
        text = html.read_text().replace('<body>',f'<body><p><b>事後シャドー検証：荒天不確実性 {label}。本番未反映。</b>本命から減らした確率をDと反対側へ配分。雨・風の判定閾値は同じ。現存入力による事後再計算であり、試合前検証ではありません。</p>',1)
        html.write_text(text)
        print(label, summary[-1], flush=True)
    table = pd.DataFrame(summary)
    table.to_csv(args.out/'summary.csv',index=False,encoding='utf-8-sig')
    links = ''.join(f'<li><a href="{label}/buyplan_scored.html">{label}</a></li>' for label, *_ in variants)
    (args.out/'index.html').write_text(
        "<!doctype html><meta charset='utf-8'><title>toto1650 weather uncertainty shadow</title>"
        "<h1>toto1650 weather uncertainty shadow</h1>"
        "<p>荒天時に首位予測の確率を他の2結果へ再配分する、結果非参照のシャドー比較。</p>"
        + table.to_html(index=False, border=1) + '<h2>各案</h2><ul>' + links + '</ul>'
    )


if __name__ == '__main__':
    run()
