"""Compare arrangements with fixed probability allocations, without actual-result fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from proportional_buyplan import build_proportional_tickets

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/reports/ticket_arrangement_simulation_v1'


def sample(p, count, seed):
    u=np.random.default_rng(seed).random((count,13))
    return (u[:,:,None]>p.cumsum(axis=1)[None,:,:]).sum(axis=2)


def joint(t,p):
    return np.prod(p[np.arange(13)[None,:],t],axis=1).sum()


def exact_p12(t,p):
    # >=12 hits means each actual vector differs from a ticket in at most one
    # position. Enumerate that union, counting shared vectors only once.
    neighbors=[t]
    for col in range(13):
        for delta in [1,2]:
            arr=t.copy();arr[:,col]=(arr[:,col]+delta)%3
            neighbors.append(arr)
    outcomes=np.unique(np.concatenate(neighbors),axis=0)
    return float(np.prod(p[np.arange(13)[None,:],outcomes],axis=1).sum())


def valid(t):
    same=(t[:,None,:]==t[None,:,:]).sum(axis=2)
    return bool((same[np.triu_indices(10,1)]<=11).all())


def optimize(initial,p,target,seed,steps=2000):
    rng=np.random.default_rng(seed)
    t=initial.copy()
    outcomes=sample(p,60000 if target==11 else 12000,seed+100)
    hits=(outcomes[:,None,:]==t[None,:,:]).sum(axis=2)
    def score(arr,counts):
        if target==13:return (joint(arr,p),)
        if target==12:return (exact_p12(arr,p),joint(arr,p))
        maximum=counts.max(axis=1)
        return (int((maximum>=target).sum()),joint(arr,p))
    best=score(t,hits)
    for _ in range(steps):
        i,j=rng.choice(10,2,replace=False);col=int(rng.integers(13))
        a,b=t[i,col],t[j,col]
        if a==b:continue
        candidate=t.copy();candidate[i,col],candidate[j,col]=b,a
        if not valid(candidate):continue
        new=hits.copy()
        new[:,i]+=(outcomes[:,col]==b).astype(int)-(outcomes[:,col]==a)
        new[:,j]+=(outcomes[:,col]==a).astype(int)-(outcomes[:,col]==b)
        value=score(candidate,new)
        if value>best:t,hits,best=candidate,new,value
    return t


def main():
    source=ROOT/'data/reports/next_probability_calibration/matches.csv'
    data=pd.read_csv(source)
    OUT.mkdir(exist_ok=True)
    rows=[];excluded=[]
    for rid,g in data.groupby('round_id',sort=True):
        if len(g)!=13 or set(g.match_no)!=set(range(1,14)):
            excluded.append({'round_id':rid,'matches':len(g)});continue
        g=g.sort_values('match_no').reset_index(drop=True)
        p=g[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy()
        initial,audit=build_proportional_tickets(p)
        actual=g.actual_result.map({1:0,0:1,2:2}).to_numpy(int)
        # Independent evaluation sample is never used to optimize arrangements.
        evaluation=sample(p,100000,80000+int(str(rid).removeprefix('toto')))
        plans={'current_v21':initial}
        for target in [13,10,11,12]:
            plans[f'focus_{target}']=optimize(initial,p,target,seed=13000+target)
        for name,t in plans.items():
            if not valid(t):raise ValueError('Invalid diversity')
            for s in range(3):
                np.testing.assert_array_equal((t==s).sum(axis=0),(initial==s).sum(axis=0))
            hits=(t==actual).sum(axis=1)
            simulated=(evaluation[:,None,:]==t[None,:,:]).sum(axis=2).max(axis=1)
            row=dict(round_id=rid,prematch_proven=bool(g.prematch_snapshot_proven.all()),strategy=name,
                     actual_best=int(hits.max()),actual_mean=float(hits.mean()),actual_ticket_hits=hits.tolist(),
                     exact_model_p13=float(joint(t,p)),exact_model_p_ge12=exact_p12(t,p),draw_cells=int((t==1).sum()))
            for k in [10,11,12]:
                prob=float((simulated>=k).mean())
                row[f'model_p_ge{k}']=prob;row[f'mc_se_ge{k}']=float(np.sqrt(prob*(1-prob)/len(evaluation)))
            rows.append(row)
            plan=pd.DataFrame({'match_no':g.match_no})
            for i in range(10):plan[f'ticket{i+1:02}']=np.array([1,0,2])[t[i]]
            plan.to_csv(OUT/f'{rid}_{name}.csv',index=False)
        print(rid,[(r['strategy'],r['actual_best']) for r in rows if r['round_id']==rid],flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'round_results.csv',index=False)
    for label,pop in [('prematch_proven',df[df.prematch_proven]),('all_reference',df)]:
        summary=pop.groupby('strategy').agg(rounds=('round_id','size'),best_hits=('actual_best','mean'),mean_hits=('actual_mean','mean'),model_p13=('exact_model_p13','mean'),model_p10=('model_p_ge10','mean'),model_p11=('model_p_ge11','mean'),model_p12=('exact_model_p_ge12','mean'))
        summary.to_csv(OUT/f'{label}.csv');print(label);print(summary.to_string())
    (OUT/'manifest.json').write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'optimization_steps':2000,'optimization_outcomes':{'10':12000,'11':60000,'12':'exact','13':'exact'},'evaluation_outcomes':100000,'all_match_probabilities_independent':True,'actual_results_used_for_optimization':False,'production_changed':False,'exclusions':excluded,'search':'deterministic greedy within-match swaps; no global optimality guarantee'},indent=2))


if __name__=='__main__':main()
