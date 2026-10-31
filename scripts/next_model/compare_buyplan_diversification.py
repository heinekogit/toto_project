import sys,json,math,hashlib
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path('scripts/next_model').resolve()))
from build_next_buyplan import top_joint_candidates,ranked_outcomes,select_tickets_coverage
out=Path('data/reports/buyplan_v2_comparison_toto1660');out.mkdir(exist_ok=True)
configs=[('current50',.5,0),('coverage25',.25,0),('balanced25',.25,1),('hybrid5_25',.25,5)]
def balanced(c,ranked,floor,seed):
 selected=select_tickets_coverage(c,ranked,count=seed) if seed>1 else [c[0]]
 pool=[t for t in c if t['log_probability']>=c[0]['log_probability']+math.log(floor)]
 probs=[dict(r) for r in ranked]
 while len(selected)<10:
  counts=[{s:sum(t['symbols'][i]==s for t in selected) for s in 'HDA'} for i in range(13)]
  eligible=[t for t in pool if all(sum(a==b for a,b in zip(t['symbols'],v['symbols']))<=11 for v in selected)]
  if not eligible:raise ValueError('exhausted')
  selected.append(max(eligible,key=lambda t:(sum(probs[i][s]/(counts[i][s]+1) for i,s in enumerate(t['symbols'])),t['log_probability'])))
 return selected
source=Path('data/eval/next_model/validation_v2/lab_xg_v0/predictions.csv');df=pd.read_csv(source);rows=[];excluded=[]
for rid,g in list(df.groupby('round_id'))+[('toto1660',pd.read_csv('data/purchase_reference/next_predictions.csv'))]:
 if len(g)!=13 or set(g.match_no.astype(int))!=set(range(1,14)):
  excluded.append({'round_id':rid,'rows':len(g)});continue
 g=g.sort_values('match_no').reset_index(drop=True);c=top_joint_candidates(g);ranked=[ranked_outcomes(r) for _,r in g.iterrows()];prob=g[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy();mp={'H':1,'D':0,'A':2}
 for name,floor,seed in configs:
  ts=balanced(c,ranked,floor,seed) if seed else select_tickets_coverage(c,ranked,probability_floor_ratio=floor)
  tickets=np.array([[mp[s] for s in t['symbols']] for t in ts]);counts=np.stack([(tickets==v).sum(axis=0) for v in [1,0,2]],axis=1)
  pairs=[int((tickets[i]==tickets[j]).sum()) for i in range(10) for j in range(i)]
  row=dict(round_id=rid,strategy=name,prematch_proven=bool(g.prematch_snapshot_proven.fillna(False).all()) if 'prematch_snapshot_proven' in g else False,draw_cells=int((tickets==0).sum()),joint_mass=sum(math.exp(t['log_probability']) for t in ts),marginal_mae=float(abs(counts/10-prob).mean()),same_picks_mean=float(np.mean(pairs)),same_picks_max=max(pairs))
  if rid!='toto1660':
   hits=(tickets==g.actual_result.to_numpy()).sum(axis=1);row.update(best_hits=int(hits.max()),average_hits=float(hits.mean()))
   for k in [10,11,12,13]:row[f'round_ge{k}']=int(hits.max()>=k);row[f'tickets_ge{k}']=int((hits>=k).sum())
  rows.append(row)
  plan=g[['match_no','home_team','away_team']].copy()
  for i in range(10):plan[f'ticket{i+1:02}']=tickets[i]
  plan.to_csv(out/f'{rid}_{name}.csv',index=False)
r=pd.DataFrame(rows);r.to_csv(out/'round_comparison.csv',index=False)
for label,sub in [('all_complete',r[r.round_id!='toto1660']),('prematch_proven',r[(r.round_id!='toto1660')&r.prematch_proven])]:
 summary=sub.groupby('strategy').agg(rounds=('round_id','size'),best_hits=('best_hits','mean'),average_hits=('average_hits','mean'),draw_cells=('draw_cells','mean'),joint_mass=('joint_mass','mean'),marginal_mae=('marginal_mae','mean'),same_picks_mean=('same_picks_mean','mean'),**{f'round_ge{k}':(f'round_ge{k}','sum') for k in [10,11,12,13]})
 summary.to_csv(out/f'{label}.csv');print(label);print(summary.to_string())
print('CURRENT');print(r[r.round_id=='toto1660'].to_string(index=False));print('PER ROUND');print(r[r.round_id!='toto1660'].pivot(index=['round_id','prematch_proven'],columns='strategy',values='best_hits').to_string())
(out/'manifest.json').write_text(json.dumps({'configs':configs,'candidate_limit':5000,'minimum_hamming_distance':2,'diminishing_gain':'sum p(symbol)/(existing count + 1)','hybrid_seed':'first 5 current50 tickets','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'exclusions':excluded,'production_changed':False},indent=2))
