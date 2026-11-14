"""Independent prematch capture / later evaluation. Never invokes production forecasts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from phase38_ha_diagnostics import raw

def capture(path,round_id,out,asof):
 if out.exists():raise ValueError('Output exists: refuse overwrite')
 d=pd.read_csv(path);assert len(d)==13 and d.match_no.nunique()==13
 now=pd.Timestamp(asof)
 kicks=pd.to_datetime(d.datetime)
 if now.tzinfo is None:now=now.tz_localize('Asia/Tokyo')
 if kicks.dt.tz is None:kicks=kicks.dt.tz_localize('Asia/Tokyo')
 if not (kicks>now).all():raise ValueError('Capture must precede every kickoff')
 p=np.array([raw(r.final_lambda_home,r.final_lambda_away) for r in d.itertuples()]);q=d[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
 if not np.allclose(q.sum(axis=1),1):raise ValueError('Bad final probabilities')
 from expected_goals import score_distribution
 cells=[score_distribution(r.final_lambda_home,r.final_lambda_away) for r in d.itertuples()]
 z=d[[c for c in d if c.startswith(('stats_','condition_','next_uncertainty','absence_','flab_','goal_rate_')) or c in ['match_no','match_id','datetime','home_team','away_team','final_lambda_home','final_lambda_away']]].copy()
 z['round']=round_id
 for i,s in enumerate('HDA'):z['raw_'+s]=p[:,i];z['final_'+s]=q[:,i]
 z['P00']=[c['prob_0_0'] for c in cells];z['P11']=[c['prob_1_1'] for c in cells]
 z['S']=(z.P00<=z.P00.median())&(z.P11>=z.P11.median())
 out.mkdir(parents=True);z.to_csv(out/'prematch.csv',index=False)
 (out/'manifest.json').write_text(json.dumps(dict(input=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),captured_asof=now.isoformat(),protocol='phase39-fixed-1',note='asof supplied by operator; immutable source/archive timestamp evidence also required'),indent=2))

def evaluate(capture_dir,results,out):
 if out.exists():raise ValueError('Output exists')
 d=pd.read_csv(capture_dir/'prematch.csv');r=pd.read_csv(results)
 # Results must carry canonical match_id; reject name-only/date-ambiguous joins.
 d=d.merge(r[['match_id','actual_home','actual_away']],on='match_id',validate='one_to_one')
 if len(d)!=13 or d[['actual_home','actual_away']].isna().any().any():raise ValueError('Complete13 results required')
 y=np.where(d.actual_home>d.actual_away,0,np.where(d.actual_home==d.actual_away,1,2))
 for prefix in ['raw','final']:
  p=d[[prefix+'_'+s for s in 'HDA']].to_numpy(float);d[prefix+'_LL']=-np.log(p[np.arange(13),y]);d[prefix+'_Brier']=((p-np.eye(3)[y])**2).sum(axis=1)
 d['H_residual']=d.actual_home-d.final_lambda_home;d['A_residual']=d.actual_away-d.final_lambda_away
 d['goal_NLL']=-poisson.logpmf(d.actual_home,d.final_lambda_home)-poisson.logpmf(d.actual_away,d.final_lambda_away)
 out.mkdir(parents=True);d.to_csv(out/'evaluated.csv',index=False)
 summary={k:float(d[k].mean()) for k in ['raw_LL','final_LL','raw_Brier','final_Brier','H_residual','A_residual','goal_NLL']}
 summary['calibration']={s:dict(expected=float(d['final_'+s].sum()),actual=int((y==i).sum())) for i,s in enumerate('HDA')}
 summary.update(S_count=int(d.S.sum()),S_actual_D=int(((y==1)&d.S).sum()),coefficient_changes=0)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
 c=s.add_parser('capture');c.add_argument('--input',type=Path,required=True);c.add_argument('--round',required=True);c.add_argument('--out',type=Path,required=True);c.add_argument('--asof',required=True)
 e=s.add_parser('evaluate');e.add_argument('--capture',type=Path,required=True);e.add_argument('--results',type=Path,required=True);e.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.command=='capture':capture(a.input,a.round,a.out,a.asof)
 else:evaluate(a.capture,a.results,a.out)
