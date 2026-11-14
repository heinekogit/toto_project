"""Standalone immutable prematch exact-ten shadow; never connected to BuyPlan."""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from common import ROOT,sha256
from forecast_registry import load_bundle
from analyze_score_structure_phase4b_stage1 import universe,top_ten,decode,SYMBOLS

DEFAULT_ROOT=ROOT/'data/forecast_registry/exact10_shadow'
CODE_VERSION='exact10-shadow-1.0'


def generate(forecast,manifest,round_id,root=DEFAULT_ROOT,*,now_time=None):
    forecast=Path(forecast).resolve();manifest=Path(manifest).resolve();root=Path(root)
    frame,_,metadata=load_bundle(forecast,manifest)
    if metadata['round_id']!=round_id:raise ValueError('round mismatch')
    source_hash=sha256(forecast);target=root/'snapshots'/round_id/source_hash
    if target.exists():
        saved=json.loads((target/'manifest.json').read_text())
        if saved['forecast_sha256']!=source_hash or saved['manifest_sha256']!=sha256(manifest):
            raise ValueError('immutable shadow directory conflicts with input')
        if sha256(target/'tickets.csv')!=saved['tickets_sha256'] or sha256(target/'probabilities.csv')!=saved['probabilities_sha256']:
            raise ValueError('immutable shadow contents changed')
        return target,False
    current=pd.Timestamp(now_time) if now_time is not None else pd.Timestamp.now(tz=ZoneInfo('Asia/Tokyo'))
    if current.tz is None:raise ValueError('current time must include timezone')
    kickoffs=pd.to_datetime(frame['datetime'])
    if kickoffs.dt.tz is None:kickoffs=kickoffs.dt.tz_localize(ZoneInfo('Asia/Tokyo'))
    if not (current<kickoffs).all():raise ValueError('new shadow generation after kickoff refused')
    p=frame[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
    if p.shape!=(13,3) or not np.allclose(p.sum(axis=1),1,atol=1e-10):raise ValueError('invalid HDA probabilities')
    mass=universe(p);indices=top_ten(mass);tickets=decode(indices)
    if len(np.unique(indices))!=10:raise AssertionError('duplicate ticket')
    target.mkdir(parents=True,exist_ok=False)
    try:
        probabilities=frame[['match_no','home_team','away_team','datetime','next_prob_home','next_prob_draw','next_prob_away']].copy()
        probabilities.to_csv(target/'probabilities.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([dict(ticket_no=i+1,symbols=''.join(SYMBOLS[row]),joint_probability=float(mass[indices[i]]),
                           **{f'match_{j+1:02}':SYMBOLS[v] for j,v in enumerate(row)}) for i,row in enumerate(tickets)]).to_csv(
                               target/'tickets.csv',index=False,encoding='utf-8-sig')
        evidence=dict(round_id=round_id,created_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            forecast_created_at=metadata['created_at'],forecast_path=str(forecast),forecast_sha256=source_hash,
            source_manifest_path=str(manifest),manifest_sha256=sha256(manifest),
            probabilities_sha256=sha256(target/'probabilities.csv'),tickets_sha256=sha256(target/'tickets.csv'),
            total_13_probability=float(mass[indices].sum()),algorithm='enumerate_3pow13_select_top10',
            code_version=CODE_VERSION,code_sha256=sha256(Path(__file__)),
            status='research_only_not_purchase',result_used=False,official_buyplan_unchanged=True)
        (target/'manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    except Exception:
        for path in target.iterdir():path.unlink()
        target.rmdir()
        raise
    return target,True


def score(snapshot,results,root=DEFAULT_ROOT):
    snapshot=Path(snapshot);results=Path(results);root=Path(root)
    meta=json.loads((snapshot/'manifest.json').read_text())
    if sha256(snapshot/'tickets.csv')!=meta['tickets_sha256'] or sha256(snapshot/'probabilities.csv')!=meta['probabilities_sha256']:
        raise ValueError('shadow snapshot changed')
    actual=pd.read_csv(results,encoding='utf-8-sig').sort_values('match_no')
    if len(actual)!=13 or actual.match_no.tolist()!=list(range(1,14)):
        raise ValueError('results must have match_no 1..13')
    for col in ('home_score','away_score'):
        if col not in actual or actual[col].isna().any():raise ValueError(f'missing {col}')
    expected=pd.read_csv(snapshot/'probabilities.csv',encoding='utf-8-sig')
    if 'home_team' in actual and 'away_team' in actual:
        if (actual.home_team.astype(str).to_numpy()!=expected.home_team.astype(str).to_numpy()).any() or (actual.away_team.astype(str).to_numpy()!=expected.away_team.astype(str).to_numpy()).any():
            raise ValueError('result team mismatch')
    outcomes=np.where(actual.home_score.to_numpy()>actual.away_score.to_numpy(),'H',
                      np.where(actual.home_score.to_numpy()==actual.away_score.to_numpy(),'D','A'))
    t=pd.read_csv(snapshot/'tickets.csv',encoding='utf-8-sig')
    hits=[sum(a==b for a,b in zip(row.symbols,outcomes)) for row in t.itertuples()]
    out=root/'scores'/meta['round_id']/meta['forecast_sha256']
    out.mkdir(parents=True,exist_ok=True)
    result_hash=sha256(results);path=out/f'{result_hash}.json'
    payload=dict(round_id=meta['round_id'],forecast_sha256=meta['forecast_sha256'],tickets_sha256=meta['tickets_sha256'],
                 result_sha256=result_hash,actual_symbols=''.join(outcomes),hits=hits,best_hits=max(hits),exact13=13 in hits,
                 score_created_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat())
    if path.exists():return path,False
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    return path,True


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--in',dest='forecast',required=True);g.add_argument('--manifest',required=True)
    g.add_argument('--round',required=True);g.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    s=sub.add_parser('score');s.add_argument('--snapshot',required=True);s.add_argument('--results',required=True)
    s.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    a=ap.parse_args();result=generate(a.forecast,a.manifest,a.round,a.root) if a.command=='generate' else score(a.snapshot,a.results,a.root)
    print(json.dumps(dict(path=str(result[0]),created=result[1]),ensure_ascii=False))

if __name__=='__main__':main()
