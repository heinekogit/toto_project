"""Read-only source audit and isolated execution of the real forecast function.
No purchase generation, historical writes or coefficient adoption.
"""
import sys,json,hashlib,html
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from forecast_pipeline import build_next_probabilities
from condition_policy import prepare_conditions
from expected_goals import score_distribution
OUT=ROOT/'data/reports/next_conditions_production'
ROUNDS=[1660,1663,1664,1665,1668,1670]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True,parents=True)
 hashes=json.loads((OUT/'protected_hashes.json').read_text())
 # Extend audit to existing purchase outputs, before any computation; never write there.
 for p in (ROOT/'data/purchase_reference').glob('*'):
  if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
 diag=pd.read_csv(ROOT/'data/reports/score_structure_phase38/match_diagnostics.csv')
 frames=[];availability=[]
 for rnd in ROUNDS:
  path=ROOT/f'data/eval/toto_rounds/toto{rnd}/snapshot/next_predictions.csv'
  d=pd.read_csv(path,low_memory=False).sort_values('match_no').reset_index(drop=True)
  t=diag[diag['round'].eq(f'toto{rnd}')].sort_values('match_no').reset_index(drop=True)
  assert list(d.match_no)==list(t.match_no)
  for i,r in d.iterrows():
   z=prepare_conditions(r);truth=t.iloc[i]
   frames.append(dict(round=rnd,match_no=r.match_no,card=truth.card,model_signature=truth.model_signature,
     lh=r.final_lambda_home,la=r.final_lambda_away,yh=truth.actual_home,ya=truth.actual_away,
     fh=z.home_total_fatigue_score,fa=z.away_total_fatigue_score,
     rh=r.get('home_rest_hours',np.nan),ra=r.get('away_rest_hours',np.nan),
     th=r.get('home_travel_fatigue',np.nan),ta=r.get('away_travel_fatigue',np.nan),
     home_status=z.home_condition_fatigue_status,away_status=z.away_condition_fatigue_status,
     coverage=r.get('external_schedule_coverage'),source_sha256=sha(path)))
  if rnd==1670:
   # Already adjusted saved features; do not fetch subsequent matches or rerun historical outlier fitting.
   prod=build_next_probabilities(d,goal_rate_adjustment=False)
   old=build_next_probabilities(d,goal_rate_adjustment=False,config_path=OUT/'previous_config.json')
   prod.to_csv(OUT/'production_execution_1670.csv',index=False,encoding='utf-8-sig')
   comp=prod[['match_no','home_team','away_team','final_lambda_home','final_lambda_away','team_condition_status']].copy()
   for side in ['home','draw','away']:
    comp['old_'+side]=old['next_prob_'+side];comp['new_'+side]=prod['next_prob_'+side]
    comp['delta_pp_'+side]=100*(comp['new_'+side]-comp['old_'+side])
   comp['old_mix']=old.next_uncertainty_calibration_mix;comp['new_mix']=prod.next_uncertainty_calibration_mix
   comp.to_csv(OUT/'production_13_comparison.csv',index=False,encoding='utf-8-sig')
   assert prod.next_uncertainty_calibration_mix.eq(0).all()
   assert np.allclose(prod[[f'next_prob_{s}' for s in ['home','draw','away']]],prod[[f'next_prob_pre_uncertainty_{s}' for s in ['home','draw','away']]],atol=1e-14)
 d=pd.DataFrame(frames);d.to_csv(OUT/'coefficient_input_audit.csv',index=False,encoding='utf-8-sig')
 # Prespecified exploratory models, no hyperparameter/score selection, no use of heldout labels in fit.
 spec=dict(train_rounds=[1660,1663,1664,1665],evaluation_rounds=[1668,1670],
  models=['saved_lambda','bias_only','bias_plus_fatigue','bias_plus_short_rest','bias_plus_travel'],
  penalty=1.,bounds=[-2,2],status='reused historical sample; temporal holdout is NOT new independent validation',
  assumptions='Poisson offsets; common attack and opposing defensive-concession coefficients; H/A intercepts',
  fatigue_scale=16,travel_load_scale=16,rest_feature='max(0,96-rest_hours)/96',production_adoption=False)
 (OUT/'coefficient_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
 d=d.dropna(subset=['fh','fa','rh','ra','th','ta']).copy()
 train=d['round'].lt(1668).to_numpy();evaluation=~train
 metrics=[];params=[];case_metrics=[]
 def features(name,frame):
  n=len(frame);x=np.zeros((2*n,4));x[:n,0]=1;x[n:,1]=1
  if name.endswith('fatigue'):h=frame.fh.to_numpy()/16;a=frame.fa.to_numpy()/16
  elif name.endswith('short_rest'):h=np.maximum(0,96-frame.rh.to_numpy())/96;a=np.maximum(0,96-frame.ra.to_numpy())/96
  elif name.endswith('travel'):h=frame.th.to_numpy()/16;a=frame.ta.to_numpy()/16
  else:return x[:,:2]
  x[:n,2]=-h;x[n:,2]=-a;x[:n,3]=a;x[n:,3]=h
  return x
 for name in spec['models']:
  tr=d[train];ev=d[evaluation];y=np.r_[tr.yh,tr.ya];offset=np.log(np.r_[tr.lh,tr.la]);x=features(name,tr)
  if name=='saved_lambda':beta=np.zeros(x.shape[1]);success=True
  else:
   def loss(b):
    eta=offset+x@b;return float((np.exp(eta)-y*eta).sum()+.5*np.dot(b,b))
   fit=minimize(loss,np.zeros(x.shape[1]),method='L-BFGS-B',bounds=[(-2,2)]*x.shape[1]);beta=fit.x;success=bool(fit.success)
  hessian=x.T@(np.exp(offset+x@beta)[:,None]*x)+np.eye(x.shape[1]);se=np.sqrt(np.diag(np.linalg.inv(hessian)))
  for j,b in enumerate(beta):params.append(dict(model=name,parameter=['home_bias','away_bias','own_attack_loss','opponent_defense_loss'][j],estimate=b,approx_se=se[j],train_matches=len(tr),evaluation_matches=len(ev),success=success,rank=int(np.linalg.matrix_rank(x)),columns=x.shape[1]))
  for subset,f in [('train',tr),('evaluation',ev)]:
   q=np.exp(np.log(np.r_[f.lh,f.la])+features(name,f)@beta);n=len(f);lh=q[:n];la=q[n:]
   p=np.array([[score_distribution(h,a)['prob_'+s] for s in ['home','draw','away']] for h,a in zip(lh,la)])
   yy=np.where(f.yh>f.ya,0,np.where(f.yh==f.ya,1,2));goals=np.r_[f.yh,f.ya]
   for j,(_,rr) in enumerate(f.iterrows()):
    case_metrics.append(dict(model=name,subset=subset,round=rr['round'],match_no=rr.match_no,card=rr.card,lambda_home=lh[j],lambda_away=la[j],LL=float(-np.log(p[j,yy[j]])),Brier=float(((p[j]-np.eye(3)[yy[j]])**2).sum())))
   metrics.append(dict(model=name,subset=subset,n=n,score_NLL=float((q-goals*np.log(q)+gammaln(goals+1)).sum()/n),LL=float(-np.log(p[np.arange(n),yy]).mean()),Brier=float(((p-np.eye(3)[yy])**2).sum(axis=1).mean())))
 pd.DataFrame(case_metrics).to_csv(OUT/'temporal_research_matches.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(params).to_csv(OUT/'research_coefficients_NOT_DEPLOYED.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(metrics).to_csv(OUT/'temporal_research_metrics.csv',index=False,encoding='utf-8-sig')
 for name in ['summary','match_comparison','portfolio_comparison','current_schedule_coverage']:
  p=ROOT/'data/reports/team_conditions_research'/f'{name}.csv'
  pd.read_csv(p).to_csv(OUT/f'archived_{name}.csv',index=False,encoding='utf-8-sig')
 unchanged={p:sha(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
 audit=dict(protected_unchanged=True,protected_hashes=hashes,executed='forecast_pipeline.build_next_probabilities',execution_matches=13,outlier_note='saved adjusted inputs reused; outlier production default unchanged',production_mix=0,uncalibrated_lambda_enabled=False,coefficient_eligible_matches=len(d),training_matches=int(train.sum()),evaluation_matches=int(evaluation.sum()))
 (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in audit.items() if k!='protected_hashes'},ensure_ascii=False,indent=2))
 print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':main()
