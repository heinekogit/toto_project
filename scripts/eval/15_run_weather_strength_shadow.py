#!/usr/bin/env python3
"""Compare modest weather multipliers only in isolated, result-censored workspaces."""
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
    parser.add_argument('--out', type=Path, default=ROOT/'data/eval/simulations/toto1650_weather_shadow')
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
    for multiplier in [1.0, 1.1, 1.2]:
        label = f'weather_{round(multiplier*100)}'
        out = args.out/label
        out.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f'toto1650_{label}_', dir='/private/tmp'))
        shutil.copytree(source/'data', work/'data', ignore=shutil.ignore_patterns('reports','output_snapshots','__pycache__'))
        shutil.copytree(source/'scripts', work/'scripts', ignore=shutil.ignore_patterns('__pycache__','.venv'))
        for file in source.glob('*.py'):
            shutil.copy2(file, work/file.name)
        (work/'logs').mkdir()
        (work/'data/reports').mkdir(exist_ok=True)
        # Do not alter rain/wind classification thresholds or add draw bias.
        predictor = work/'scripts/11_prediction_01.py'
        text = predictor.read_text()
        for key, value in [('HEAVY_RAIN', .15), ('STRONG_WIND', .10), ('RAIN', .05)]:
            import re
            pattern = rf'(?m)^WEATHER_PENALTY_{key} = [0-9.]+$'
            text, count = re.subn(pattern, f'WEATHER_PENALTY_{key} = {value*multiplier!r}', text)
            if count != 1:
                raise ValueError(f'weather constant not found: {key}')
        predictor.write_text(text)
        rules = work/'weather_rules.py'
        text = rules.read_text()
        marker = '    return float(value) if value.ndim == 0 else value'
        assert text.count(marker) == 1
        rules.write_text(text.replace(marker, f'    value = value * {multiplier!r}\n'+marker))
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
        summary.append({'variant':label,'multiplier':multiplier,'mean_hits':evaluation.hits.mean(),'max_hits':evaluation.hits.max(),'mean_hit_rate':evaluation.hit_rate.mean(),'max_probability_delta':float(abs(probs-baseline).max())})
        (out/'provenance.json').write_text(json.dumps({'shadow_only':True,'multiplier':multiplier,'workspace':str(work),'base_replay':str(args.replay_dir),'weather_thresholds_changed':False,'predictor_sha256':hashlib.sha256(predictor.read_bytes()).hexdigest(),'weather_rules_sha256':hashlib.sha256(rules.read_bytes()).hexdigest()},indent=2))
        html = out/'buyplan_scored.html'
        text = html.read_text().replace('<body>',f'<body><p><b>事後シャドー検証：悪天候補正 {multiplier:.1f}倍。本番未反映。</b>雨・風の判定閾値は同じ。現存入力による事後再計算であり、試合前検証ではありません。</p>',1)
        html.write_text(text)
        print(label, summary[-1], flush=True)
    pd.DataFrame(summary).to_csv(args.out/'summary.csv',index=False,encoding='utf-8-sig')


if __name__ == '__main__':
    run()
