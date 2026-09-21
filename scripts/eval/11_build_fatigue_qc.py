#!/usr/bin/env python3
"""Validate the actual production fatigue CSV, including lookup and merge coverage."""
from __future__ import annotations
import argparse
import importlib.util
import sys
from html import escape
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from fatigue_merge import add_fatigue_time_aliases


def fatigue_module():
    spec = importlib.util.spec_from_file_location('fatigue_qc_source', ROOT / 'scripts/06_calculate_fatigue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_merge(matches, fatigue):
    module = fatigue_module()
    module.validate_fatigue_output(fatigue)
    keys = ['datetime', 'home_team', 'away_team']
    if fatigue.duplicated(keys).any():
        raise ValueError('FATIGUE_QC: duplicate production merge keys')
    augmented, rescued = add_fatigue_time_aliases(matches, fatigue, keys)
    out = matches[keys].merge(augmented, on=keys, how='left', validate='one_to_one', indicator=True)
    if not out['_merge'].eq('both').all():
        raise ValueError(f'FATIGUE_QC: unmatched production fixtures={int(out["_merge"].ne("both").sum())}')
    return out.drop(columns='_merge'), rescued


def build(league, season, output_dir):
    module = fatigue_module()
    sources = []
    for kind, priority in [('upcoming', 0), ('latest_results', 1)]:
        frame = pd.read_csv(ROOT / f'data/{league}_{season}_{kind}.csv')
        frame['__source_priority'] = priority
        sources.append(frame)
    matches = module.dedupe_matches(pd.concat(sources, ignore_index=True))
    fatigue_path = ROOT / f'data/team_fatigue_scores_{league}_{season}.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{league}_{season}_fatigue_qc.csv'
    html_path = output_dir / f'{league}_{season}_fatigue_qc.html'
    try:
        fatigue = pd.read_csv(fatigue_path)
        fatigue['datetime'] = pd.to_datetime(fatigue.datetime, errors='coerce')
        out, rescued = validate_merge(matches, fatigue)
        out['production_travel_hit'] = out.travel_lookup_status.eq('hit')
        out['production_travel_km'] = out.away_travel_distance_km
        out.to_csv(csv_path, index=False, encoding='utf-8-sig')
        status = f'PASS: 本番CSV 距離参照 {len(out)}/{len(out)}、全試合マージ成功（時刻救済 {rescued}件）'
        table = out[['datetime', 'home_team', 'away_team', 'production_travel_km', 'home_rest_hours', 'away_rest_hours']].to_html(index=False, escape=True)
    except Exception as exc:
        # Replace an old PASS report too, so a failed run cannot look successful.
        status = f'FAIL: {exc}'
        pd.DataFrame([{'qc_status': 'FAIL', 'error': str(exc)}]).to_csv(csv_path, index=False, encoding='utf-8-sig')
        html_path.write_text(f'<!doctype html><meta charset="utf-8"><h1>疲労QC：不合格</h1><p>{escape(status)}</p>', encoding='utf-8')
        raise
    html_path.write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>本番疲労QC</title>
<style>body{{font-family:system-ui;margin:24px}}table{{border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:6px}}</style>
<h1>{league.upper()} {season} 本番疲労QC</h1><p>{escape(status)}</p>
<p>検査対象: {escape(str(fatigue_path))}。距離は本拠地間の代理値。外部大会の網羅性・未確認負荷はCSVに保持。</p>{table}</html>''', encoding='utf-8')
    return csv_path, html_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--leagues', nargs='+', default=['j1', 'j2'])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/reports/fatigue_qc')
    args = parser.parse_args()
    for league in args.leagues:
        _, html = build(league.lower(), args.season, args.output_dir)
        print(f'OK: {html}')


if __name__ == '__main__':
    main()
