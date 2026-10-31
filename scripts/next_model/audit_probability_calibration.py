"""Check frozen next-model probabilities against actual outcomes, without fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/reports/next_probability_calibration'


def main():
    historical=ROOT/'data/eval/next_model/validation_v2/lab_xg_v0/predictions.csv'
    data=pd.read_csv(historical)
    snap=ROOT/'data/eval/toto_rounds/toto1660/snapshot'
    path=snap/'next_predictions.csv'
    latest=pd.read_csv(path)
    actual=pd.read_csv(snap.parent/'actual_results.csv')
    manifest=json.loads((snap/'buyplan_engine_manifest.json').read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest['output_sha256']['next_predictions.csv']!=digest:
        raise ValueError('toto1660 prediction hash differs from generator manifest')
    created=pd.Timestamp(manifest['created_at'])
    kickoff=pd.to_datetime(latest.datetime).dt.tz_localize('Asia/Tokyo')
    latest['prematch_snapshot_proven']=bool((created<kickoff).all())
    latest=latest.drop(columns=['actual_result'],errors='ignore').merge(actual[['match_no','result']],on='match_no',validate='one_to_one').rename(columns={'result':'actual_result'})
    latest['round_id']='toto1660'
    data=pd.concat([data,latest],ignore_index=True)
    if data.duplicated(['round_id','match_no']).any():raise ValueError('Duplicate matches')
    OUT.mkdir(exist_ok=True)
    summaries, classes, bins, rankings=[],[],[],[]
    edges=[0,.2,.3,.4,.5,.6,.8,1.000001]
    for population,frame in [('prematch_proven',data[data.prematch_snapshot_proven.eq(True)]),('all_reference',data)]:
        p=frame[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
        y=frame.actual_result.map({1:0,0:1,2:2}).to_numpy(int)
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-6):raise ValueError('Invalid probabilities')
        truth=np.eye(3)[y];top=p.argmax(axis=1);correct=(top==y);confidence=p.max(axis=1)
        summaries.append(dict(population=population,matches=len(frame),top_accuracy=float(correct.mean()),mean_top_probability=float(confidence.mean()),brier=float(((p-truth)**2).sum(axis=1).mean()),uniform_brier=2/3,log_loss=float(-np.log(p[np.arange(len(y)),y]).mean())))
        for i,s in enumerate('HDA'):
            classes.append(dict(population=population,symbol=s,mean_probability=float(p[:,i].mean()),actual_rate=float((y==i).mean()),actual_count=int((y==i).sum()),top_pick_count=int((top==i).sum()),top_pick_hits=int(((top==i)&correct).sum())))
            for low,high in zip(edges[:-1],edges[1:]):
                mask=(p[:,i]>=low)&(p[:,i]<high)
                if mask.any(): bins.append(dict(population=population,symbol=s,bin=f'{low:.0%}–{min(high,1):.0%}',count=int(mask.sum()),mean_probability=float(p[mask,i].mean()),actual_rate=float((y[mask]==i).mean())))
        for low,high in zip(edges[:-1],edges[1:]):
            mask=(confidence>=low)&(confidence<high)
            if mask.any():rankings.append(dict(population=population,bin=f'{low:.0%}–{min(high,1):.0%}',count=int(mask.sum()),mean_probability=float(confidence[mask].mean()),accuracy=float(correct[mask].mean())))
    for name,rows in [('summary',summaries),('outcome_rates',classes),('calibration_bins',bins),('top_confidence_bins',rankings)]:
        df=pd.DataFrame(rows);df.to_csv(OUT/f'{name}.csv',index=False);print(name);print(df[df.population=='prematch_proven'].to_string(index=False))
    data[['round_id','match_no','prematch_snapshot_proven','next_prob_home','next_prob_draw','next_prob_away','actual_result']].to_csv(OUT/'matches.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps({'historical_source':str(historical),'historical_sha256':hashlib.sha256(historical.read_bytes()).hexdigest(),'latest_sha256':digest,'latest_generator_created_at':manifest['created_at'],'latest_snapshot_created_after_matches':True,'latest_proof':'generator manifest timestamp and prediction checksum; retrospective snapshot, not independent timestamp certification','not_fitted':True},indent=2))


if __name__=='__main__':main()
