#!/usr/bin/env python3
"""Retrospective measurement audit; does not regenerate purchase predictions."""
import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', required=True)
    parser.add_argument('--season', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('fatigue_audit', ROOT / 'scripts/06_calculate_fatigue.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = ROOT / 'data/eval/toto_rounds' / args.round / 'snapshot'
    predictions = pd.read_csv(snapshot / 'predictions.csv')
    buyplan = pd.read_csv(snapshot / 'buyplan.csv')
    matrix = module.load_travel_distances()
    events = [module.load_acl_team_events(module.ACL_SCHEDULE_CSV),
              module.load_confirmed_cup_events(ROOT / 'data'),
              module.load_external_team_events(module.EXTERNAL_MATCH_EVENTS_CSV)]
    events = pd.concat([e for e in events if not e.empty], ignore_index=True) if any(not e.empty for e in events) else pd.DataFrame()
    frames = []
    for league in sorted(set(buyplan.league.str.lower())):
        sources = []
        for kind, priority in [('upcoming', 0), ('latest_results', 1)]:
            source = pd.read_csv(ROOT / f'data/{league}_{args.season}_{kind}.csv')
            source['__source_priority'] = priority
            sources.append(source)
        matches = module.dedupe_matches(pd.concat(sources, ignore_index=True))
        frames.append(module.calculate_fatigue(matches, matrix, events))
    corrected = pd.concat(frames, ignore_index=True)
    rows = []
    for _, buy in buyplan.iterrows():
        old = predictions[(predictions.home_team == buy.home_team) & (predictions.away_team == buy.away_team)]
        if len(old) != 1:
            raise ValueError(f'non-unique snapshot fixture: {buy.match_no}')
        old = old.iloc[0]
        hit = corrected[
            corrected.home_team.map(module.canonical_team_name).eq(module.canonical_team_name(buy.home_team))
            & corrected.away_team.map(module.canonical_team_name).eq(module.canonical_team_name(buy.away_team))
            & pd.to_datetime(corrected.datetime).dt.date.eq(pd.Timestamp(old.datetime).date())]
        if len(hit) != 1:
            raise ValueError(f'non-unique corrected fixture: {buy.match_no}')
        fresh = hit.iloc[0]
        row = {'match_no': buy.match_no, 'home_team': buy.home_team, 'away_team': buy.away_team,
               'audit_basis': 'retrospective_current_results_not_prematch_backtest',
               'old_travel_status': old.get('travel_lookup_status'), 'new_travel_status': fresh.travel_lookup_status,
               'distance_proxy_km': fresh.away_travel_distance_km,
               'weather_fetched_at_saved': old.get('last_updated_at')}
        for side in ['home', 'away']:
            row[f'old_{side}_rest_score'] = old.get(f'{side}_rest_fatigue')
            row[f'new_{side}_rest_hours'] = fresh[f'{side}_rest_hours']
            row[f'new_{side}_rest_score'] = fresh[f'{side}_rest_fatigue']
        rows.append(row)
    out = ROOT / 'data/eval/observation_history/reports/conditions' / f'{args.round}_measurement_audit.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print(f'OK: {out}; travel hits={sum(r["new_travel_status"] == "hit" for r in rows)}/{len(rows)}')


if __name__ == '__main__':
    main()
