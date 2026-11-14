"""Append-only Next Model forecast and result registry; no model calculations."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import ROOT, normalize_text, sha256
from expected_goals import score_distribution
sys.path.insert(0,str(ROOT/'scripts'))
from jleague_team_names import canonical_team_name

DEFAULT_ROOT=ROOT/'data/forecast_registry'
JST=ZoneInfo('Asia/Tokyo')


def now():return datetime.now(JST).isoformat()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def fixture_id(row):
    league=str(row['league']).lower()
    date=str(pd.Timestamp(row['datetime']).date())
    return f'{league}|{date}|{normalize_text(row["home_team"])}|{normalize_text(row["away_team"])}'


def read_log(path):
    if not path.exists():return []
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_unique(root,filename,records):
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'.registry.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        path=root/filename
        existing=read_log(path)
        ids={r['event_id'] for r in existing}
        added=0
        with path.open('a',encoding='utf-8') as out:
            for record in records:
                if record['event_id'] in ids:continue
                out.write(json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
                out.flush();os.fsync(out.fileno())
                ids.add(record['event_id']);added+=1
        fcntl.flock(handle,fcntl.LOCK_UN)
    return added


def load_bundle(forecast_path,manifest_path):
    source=Path(forecast_path);mp=Path(manifest_path)
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    if manifest.get('output_sha256',{}).get(source.name)!=sha256(source):
        raise ValueError('Forecast manifest/hash mismatch')
    frame=pd.read_csv(source,encoding='utf-8-sig',low_memory=False).sort_values('match_no').reset_index(drop=True)
    if len(frame)!=13 or frame.match_no.tolist()!=list(range(1,14)):
        raise ValueError('Expected 13 numbered matches')
    created=pd.Timestamp(manifest['created_at'])
    kickoff=pd.to_datetime(frame.datetime)
    if kickoff.dt.tz is None:kickoff=kickoff.dt.tz_localize(JST)
    if not (kickoff>created).all():raise ValueError('Post-kickoff forecast refused')
    diagnostics=[]
    for _,row in frame.iterrows():
        distribution=score_distribution(float(row.final_lambda_home),float(row.final_lambda_away))
        raw=np.array([distribution[f'prob_{side}'] for side in ('home','draw','away')])
        final=np.array([row[f'next_prob_{side}'] for side in ('home','draw','away')],float)
        for goal in ('0_0','1_1','2_2'):
            if not np.isclose(float(row[f'next_prob_{goal}']),distribution[f'prob_{goal}'],atol=1e-9):
                raise ValueError('Saved scoreline probability mismatch')
        mix=row.get('next_uncertainty_calibration_mix',np.nan)
        mix=0.0 if pd.isna(mix) else float(mix)
        if not 0<=mix<=1 or not np.allclose(final,(1-mix)*raw+mix/3,rtol=0,atol=1e-8):
            raise ValueError('Saved final H/D/A mismatch')
        diagnostics.append(dict(score_P_H=raw[0],score_P_D=raw[1],score_P_A=raw[2],uncertainty_mix=mix))
    diagnostics=pd.DataFrame(diagnostics)
    return frame,diagnostics,manifest


def forecast_events(forecast_path,manifest_path,*,runtime_code_hash=None):
    frame,d,manifest=load_bundle(forecast_path,manifest_path)
    snapshot_sha=sha256(Path(forecast_path));manifest_sha=sha256(Path(manifest_path))
    events=[]
    for ix,row in frame.iterrows():
        league=str(row.get('league',str(row.match_id).split('_')[0])).lower()
        identity=dict(league=league,datetime=row.datetime,home_team=row.home_team,away_team=row.away_team)
        fid=fixture_id(identity)
        raw=[float(d.iloc[ix][f'score_P_{s}']) for s in ('H','D','A')]
        final=[float(row[f'next_prob_{s}']) for s in ('home','draw','away')]
        if not np.isclose(sum(raw),1,atol=1e-10) or not np.isclose(sum(final),1,atol=1e-10):
            raise ValueError('Invalid probabilities')
        event=dict(event_type='forecast',event_id=digest(['forecast',snapshot_sha,int(row.match_no)]),
            fixture_id=fid,round_id=manifest['round_id'],match_no=int(row.match_no),
            source_match_id=str(row.match_id),league=league,season=str(row.match_id).split('_')[1],
            home_team=str(row.home_team),away_team=str(row.away_team),kickoff=str(row.datetime),
            snapshot_id=snapshot_sha,forecast_created_at=manifest['created_at'],
            input_path=manifest.get('input'),input_sha256=manifest.get('input_sha256'),
            output_sha256=snapshot_sha,manifest_sha256=manifest_sha,
            code_version=manifest.get('code_version') or runtime_code_hash,
            code_version_source=('manifest' if manifest.get('code_version') else
                                 'same_run_code_hash' if runtime_code_hash else 'not_recorded'),
            model_version=manifest.get('prediction_model'),
            model_signature=('goal_rate_plus_style_policy' if pd.notna(row.get('goal_rate_adjustment_version'))
                else 'absence_uncertainty' if pd.notna(row.get('next_uncertainty_calibration_mix'))
                else 'early_score_model'),
            correction_settings=dict(goal_rate=row.get('goal_rate_adjustment_version') if pd.notna(row.get('goal_rate_adjustment_version')) else None,
                                     style=row.get('style_official_version') if pd.notna(row.get('style_official_version')) else None,
                                     probability_source=row.get('buyplan_probability_source') if pd.notna(row.get('buyplan_probability_source')) else None),
            lambda_home=float(row.final_lambda_home),lambda_away=float(row.final_lambda_away),
            raw_prob_home=raw[0],raw_prob_draw=raw[1],raw_prob_away=raw[2],
            uncertainty_mix=float(d.iloc[ix].uncertainty_mix),
            final_prob_home=final[0],final_prob_draw=final[1],final_prob_away=final[2],
            source_file=str(Path(forecast_path).resolve()),provenance_class='A',
            source_input_available_at=None,registered_at=now())
        events.append(event)
    return events


def register_forecast(forecast_path,manifest_path,root=DEFAULT_ROOT,*,runtime_code_hash=None):
    records=forecast_events(forecast_path,manifest_path,runtime_code_hash=runtime_code_hash)
    return dict(total=len(records),added=append_unique(Path(root),'forecasts.jsonl',records))


def result_events(result_path,forecasts):
    result_path=Path(result_path);frame=pd.read_csv(result_path,encoding='utf-8-sig')
    if not {'match_no','home_team','away_team','home_score','away_score','result'}.issubset(frame):
        raise ValueError('Result file lacks required columns')
    round_set={f['round_id'] for f in forecasts}
    if len(round_set)!=1:raise ValueError('Select one round for result registration')
    round_id=next(iter(round_set))
    indexed={int(r.match_no):r for _,r in frame.iterrows()}
    events=[]
    for f in forecasts:
        if f['match_no'] not in indexed:continue
        r=indexed[f['match_no']]
        if normalize_text(r.home_team)!=normalize_text(f['home_team']) or normalize_text(r.away_team)!=normalize_text(f['away_team']):
            raise ValueError(f'{round_id} M{f["match_no"]}: result team mismatch')
        if pd.isna(r.home_score) or pd.isna(r.away_score):continue
        h,a=int(r.home_score),int(r.away_score)
        if h<0 or a<0 or h!=float(r.home_score) or a!=float(r.away_score):raise ValueError('Invalid result score')
        outcome=1 if h>a else 0 if h==a else 2
        if int(r.result)!=outcome:raise ValueError('Result symbol/score mismatch')
        payload=[f['fixture_id'],h,a,outcome]
        events.append(dict(event_type='result',event_id=digest(['result',*payload]),
            fixture_id=f['fixture_id'],round_id=round_id,match_no=f['match_no'],
            home_score=h,away_score=a,result=outcome,
            result_known_at=None,result_recorded_at=now(),
            source_file=str(result_path.resolve()),source_sha256=sha256(result_path),
            correction_of=None))
    return events


def register_results(result_path,round_id,root=DEFAULT_ROOT):
    root=Path(root)
    forecasts=[f for f in read_log(root/'forecasts.jsonl') if f['round_id']==round_id]
    if not forecasts:raise ValueError('No registered forecast for round')
    by_fixture={f['fixture_id']:f for f in forecasts}
    records=result_events(result_path,list(by_fixture.values()))
    existing=read_log(root/'results.jsonl')
    latest={}
    for event in existing:latest[event['fixture_id']]=event
    for event in records:
        previous=latest.get(event['fixture_id'])
        if previous and previous['event_id']!=event['event_id']:
            event['correction_of']=previous['event_id']
    return dict(total=len(records),added=append_unique(root,'results.jsonl',records))


def sync_league_results(result_path,root=DEFAULT_ROOT):
    """Join completed official league results to registered forecasts, by date/clubs."""
    root=Path(root);result_path=Path(result_path)
    frame=pd.read_csv(result_path,encoding='utf-8-sig')
    frame=frame.dropna(subset=['datetime','home_score','away_score'])
    by_key={}
    for _,r in frame.iterrows():
        key=(str(pd.Timestamp(r.datetime).date()),canonical_team_name(r.home_team),
             canonical_team_name(r.away_team))
        if key in by_key and (by_key[key].home_score!=r.home_score or by_key[key].away_score!=r.away_score):
            raise ValueError(f'Conflicting league result for {key}')
        by_key[key]=r
    forecasts=read_log(root/'forecasts.jsonl')
    unique={f['fixture_id']:f for f in forecasts}
    old=read_log(root/'results.jsonl');latest={}
    for event in old:latest[event['fixture_id']]=event
    records=[]
    for f in unique.values():
        key=(str(pd.Timestamp(f['kickoff']).date()),canonical_team_name(f['home_team']),
             canonical_team_name(f['away_team']))
        if key not in by_key:continue
        r=by_key[key];h,a=int(r.home_score),int(r.away_score)
        if min(h,a)<0 or h!=float(r.home_score) or a!=float(r.away_score):
            raise ValueError(f'Invalid league result for {key}')
        outcome=1 if h>a else 0 if h==a else 2
        event_id=digest(['result',f['fixture_id'],h,a,outcome])
        previous=latest.get(f['fixture_id'])
        records.append(dict(event_type='result',event_id=event_id,
            fixture_id=f['fixture_id'],round_id=f['round_id'],match_no=f['match_no'],
            home_score=h,away_score=a,result=outcome,result_known_at=None,
            result_recorded_at=now(),source_file=str(result_path.resolve()),
            source_sha256=sha256(result_path),
            correction_of=previous['event_id'] if previous and previous['event_id']!=event_id else None))
    return dict(matched=len(records),added=append_unique(root,'results.jsonl',records))


def selected_matches(root=DEFAULT_ROOT):
    forecasts=read_log(Path(root)/'forecasts.jsonl')
    results=read_log(Path(root)/'results.jsonl')
    latest={}
    for r in results:latest[r['fixture_id']]=r
    # One real match, earliest verified prematch forecast. No outcome-based selection.
    picked={}
    for f in forecasts:
        fid=f['fixture_id']
        if fid not in picked or (f['forecast_created_at'],f['event_id']) < (picked[fid]['forecast_created_at'],picked[fid]['event_id']):
            picked[fid]=f
    return pd.DataFrame([{**f,**{f'actual_{k}':v for k,v in latest.get(fid,{}).items()}}
                         for fid,f in picked.items()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('register-forecast');a.add_argument('--forecast',required=True,type=Path);a.add_argument('--manifest',required=True,type=Path)
    b=sub.add_parser('register-results');b.add_argument('--results',required=True,type=Path);b.add_argument('--round',required=True)
    c=sub.add_parser('sync-league-results');c.add_argument('--results',required=True,type=Path)
    sub.add_parser('status')
    args=parser.parse_args()
    if args.command=='register-forecast':result=register_forecast(args.forecast,args.manifest,args.root)
    elif args.command=='register-results':result=register_results(args.results,args.round,args.root)
    elif args.command=='sync-league-results':result=sync_league_results(args.results,args.root)
    else:
        forecasts=read_log(args.root/'forecasts.jsonl');results=read_log(args.root/'results.jsonl')
        result=dict(forecast_events=len(forecasts),result_events=len(results),unique_fixtures=len({f['fixture_id'] for f in forecasts}))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
