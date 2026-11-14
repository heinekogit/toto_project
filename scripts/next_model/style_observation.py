"""Forward-only, append-only style observation workflow (no purchase changes)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
from contextlib import contextmanager
import pandas as pd
from common import ROOT, normalize_text, sha256
from style_shadow import build_examples, predict, train, evaluate, VERSION, timestamp

STATE = ROOT / 'data/style_shadow'
JST = ZoneInfo('Asia/Tokyo')
STYLE_COLUMNS = ['team','style','metric','value','unit','selection_status','first_observed_date','first_observed_at']


def now(): return datetime.now(JST).isoformat(timespec='microseconds')


def token(at): return timestamp(at).strftime('%Y%m%dT%H%M%S%f')


@contextmanager
def locked(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield


def seal(folder, meta):
    meta['hashes']={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'}
    (folder/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))


def verify(folder):
    m=json.loads((folder/'manifest.json').read_text())
    for name,digest in m['hashes'].items():
        if sha256(folder/name)!=digest:raise ValueError(f'changed frozen file: {folder/name}')
    return m


def match_key(row):
    league=str(row.get('league',''))
    if league not in {'j1','j2'}:league=str(row.get('source_match_id',row['match_id'])).split('_')[0]
    if league not in {'j1','j2'}:raise ValueError('unknown observation league')
    return '|'.join([league,str(timestamp(row['datetime']).date()),normalize_text(row['home_team']),normalize_text(row['away_team'])])


def archive_styles(styles, season, sources=(), root=STATE, at=None):
    at=at or now()
    with locked(root):
        folder=root/'inputs'/token(at);folder.mkdir(parents=True,exist_ok=False)
        data=styles.copy();data['first_observed_date']=str(timestamp(at).date());data['first_observed_at']=at
        data.to_csv(folder/'styles.csv',index=False)
        # Keep original pages alongside accepted values for later auditing.
        import shutil
        for i,path in enumerate(sources):shutil.copyfile(path,folder/f'source_{i:03d}.html')
        seal(folder,dict(observed_at=at,season=int(season),accepted=int(data.value.notna().sum()),missing=int(data.value.isna().sum()),source_paths=[str(p) for p in sources]))
    return folder


def archives(root):
    comparisons=[];examples=[]
    for folder in sorted((root/'forecasts').glob('*')):
        if not (folder/'manifest.json').exists():continue
        verify(folder)
        c=pd.read_csv(folder/'comparison.csv',low_memory=False);e=pd.read_csv(folder/'examples.csv',low_memory=False)
        comparisons.append(c);examples.append(e)
    if not comparisons:return pd.DataFrame(),pd.DataFrame()
    # Predeclared rule: earliest actually saved prematch forecast per fixture.
    c=pd.concat(comparisons,ignore_index=True).sort_values('forecast_created_at').drop_duplicates('match_id',keep='first')
    e=pd.concat(examples,ignore_index=True).sort_values('forecast_created_at').drop_duplicates(['match_id','side'],keep='first')
    return c,e


def write_status(root,status):
    report=root/'reports';report.mkdir(exist_ok=True)
    (report/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    labels={'no_prematch_archives':'試合前予測の保存待ち','insufficient_prematch_training_data':'学習用データの蓄積待ち','time_holdout_prefers_baseline':'内部検証では補正なしを選択','selected_on_time_holdout':'比較用の補正係数を学習済み','no_nonoverlapping_time_holdout':'時系列を分離できるデータの蓄積待ち'}
    (report/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>追加スタイル前向き検証</title><h1>追加スタイル前向き検証</h1><p>正式統合が有効でも、未学習時は補正ゼロです。保存済みの試合前予測だけを採点し、未学習の一致結果は改善実証に数えません。</p>'+pd.DataFrame([{'正式統合':status.get('official_integration_enabled',False),'保存試合':status['forecast_matches'],'結果結合済み':status['completed_matches'],'次回モデル':labels.get(status['next_model_reason'],status['next_model_reason']),'補正有効':status['next_model_enabled']}]).to_html(index=False)+'<p><a href="status.json">指標と状態</a> / <a href="paired_results.csv">試合別比較</a></p>')


def _refresh(root, data_dir, at):
    ledger=root/'result_observations.csv'
    old=pd.read_csv(ledger) if ledger.exists() else pd.DataFrame(columns=['match_id','home_score','away_score','result_available_at','source_file','source_sha256'])
    latest=old.drop_duplicates('match_id',keep='last').set_index('match_id')
    additions=[]
    for path in sorted(data_dir.glob('j[12]_*_latest_results.csv')):
        d=pd.read_csv(path);lg=path.name.split('_')[0];digest=sha256(path)
        d=d.dropna(subset=['home_score','away_score','datetime'])
        for _,r in d.iterrows():
            if timestamp(r.datetime)>=timestamp(at):continue
            if any(float(r[c])<0 or float(r[c])%1 for c in ['home_score','away_score']):raise ValueError('invalid result')
            r=r.copy();r['league']=lg;k=match_key(r)
            if k in latest.index and all(float(latest.loc[k,c])==float(r[c]) for c in ['home_score','away_score']):continue
            additions.append(dict(match_id=k,home_score=int(r.home_score),away_score=int(r.away_score),result_available_at=at,source_file=str(path),source_sha256=digest))
    all_results=pd.concat([old,pd.DataFrame(additions)],ignore_index=True)
    if additions:
        temporary=ledger.with_suffix('.tmp')
        all_results.to_csv(temporary,index=False)
        temporary.replace(ledger)
    results=all_results.drop_duplicates('match_id',keep='last')
    c,e=archives(root)
    model={'version':VERSION,'cutoff':at,'enabled':False,'reason':'no_prematch_archives','train_matches':0}
    scores={};active_scores={};matched=0
    if len(c):
        scores=evaluate(c,results);matched=scores['baseline']['matches']
        active=c[c.style_model_enabled.astype(str).str.lower().eq('true') & c.style_features_available.astype(str).str.lower().eq('true')]
        active_scores=evaluate(active,results)
        model=train(e,results,at)
    models=root/'models';models.mkdir(exist_ok=True)
    model_path=models/f'{token(at)}.json'
    model_path.write_text(json.dumps(model,ensure_ascii=False,indent=2))
    # Reports score the already frozen forecasts; never re-predict completed games.
    report=root/'reports';report.mkdir(exist_ok=True)
    if len(c):
        c.merge(results,on='match_id',suffixes=('','_actual'),validate='one_to_one').to_csv(report/'paired_results.csv',index=False)
        e.to_csv(report/'examples.csv',index=False)
        results[results.match_id.isin(c.match_id)].to_csv(report/'results.csv',index=False)
        e.merge(results,on='match_id',validate='many_to_one').to_csv(report/'joined_training_rows.csv',index=False)
    else:
        pd.DataFrame(columns=['match_id','home_score','away_score']).to_csv(report/'paired_results.csv',index=False)
    policy_path=ROOT/'data/models/style_residual_deployment_v1.json'
    deployment=json.loads(policy_path.read_text()) if policy_path.exists() else {'enabled':False}
    status=dict(official_integration_enabled=deployment['enabled'],updated_at=at,forecast_matches=len(c),completed_matches=matched,
                learned_shadow_scores=active_scores,all_observation_scores=scores,
                next_model=str(model_path),next_model_enabled=model['enabled'],next_model_reason=model['reason'],
                note='All-observation scores include unchanged baselines; improvement must use frozen learned-shadow predictions on later matches.')
    write_status(root,status)
    return model,status


def refresh(root=STATE,data_dir=ROOT/'data',at=None):
    at=at or now()
    with locked(root):return _refresh(root,data_dir,at)


def record_forecast(frame, season, root=STATE,data_dir=ROOT/'data',at=None,deployment_policy=None):
    injected_time=at is not None
    at=at or now()
    with locked(root):
        model,_=_refresh(root,data_dir,at)
        selected=frame[frame.datetime.map(timestamp)>timestamp(at)].copy()
        if selected.empty:return {'status':'no_future_matches'}
        selected['source_match_id']=selected.match_id
        selected['match_id']=selected.apply(match_key,axis=1)
        selected['forecast_created_at']=at
        inputs=[]
        for folder in (root/'inputs').glob('*'):
            if not (folder/'manifest.json').exists():continue
            meta=verify(folder)
            if meta['season']==int(season) and timestamp(meta['observed_at'])<=timestamp(at):inputs.append((meta['observed_at'],folder))
        source=max(inputs,key=lambda x:timestamp(x[0]))[1] if inputs else None
        styles=pd.read_csv(source/'styles.csv') if source else pd.DataFrame(columns=STYLE_COLUMNS)
        examples=build_examples(selected,styles)
        output,details=predict(selected,examples,model)
        counts=details.groupby('match_id').available_feature_count.min()
        output['style_features_available']=output.match_id.map(counts).gt(0)
        output['style_deployment_mode']='official' if deployment_policy is not None else 'shadow'
        output['style_official_enabled']=bool(deployment_policy and deployment_policy.get('enabled'))
        finished=at if injected_time else now()
        if any(selected.datetime.map(timestamp)<=timestamp(finished)):
            raise ValueError('kickoff occurred while saving forecast; no archive created')
        folder=root/'forecasts'/token(at);folder.mkdir(parents=True,exist_ok=False)
        selected.to_csv(folder/'baseline.csv',index=False)
        styles.to_csv(folder/'styles.csv',index=False)
        output.to_csv(folder/'comparison.csv',index=False)
        details.to_csv(folder/'examples.csv',index=False)
        (folder/'model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2))
        if deployment_policy is not None:
            (folder/'deployment_policy.json').write_text(json.dumps(deployment_policy,ensure_ascii=False,indent=2))
        finished=at if injected_time else now()
        if any(selected.datetime.map(timestamp)<=timestamp(finished)):
            raise ValueError('kickoff occurred during file write; archive remains unsealed')
        seal(folder,dict(created_at=at,finished_at=finished,season=int(season),style_source=str(source) if source else None,source_age_days=(timestamp(at)-timestamp(verify(source)['observed_at'])).days if source and inputs else None,model_enabled=model['enabled'],deployment='official' if deployment_policy is not None else 'shadow'))
        status=json.loads((root/'reports/status.json').read_text())
        status['forecast_matches']=len(archives(root)[0])
        status['last_saved_forecast']=str(folder)
        status['updated_at']=finished
        write_status(root,status)
        return {'status':'saved','path':str(folder),'matches':len(output),'style_source':str(source) if source else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['refresh']);p.add_argument('--state-root',type=Path,default=STATE);p.add_argument('--data-dir',type=Path,default=ROOT/'data');a=p.parse_args()
    _,status=refresh(a.state_root,a.data_dir);print(json.dumps(status,ensure_ascii=False))

if __name__=='__main__':main()
