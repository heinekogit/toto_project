"""Individual-ticket quality probes with exact13 and draw-coverage hard floors."""
import time
import numpy as np
from fixed_exact13_shadow import mass,validate,neighbors,moved,I,J,M,N
from fixed_draw_bounds_shadow import cycle_moves,cycled,CI,CJ,CK,CM
from draw_count_cover_shadow import draw_distribution,lookup_table,signature,move_coverages


def probability_neighbors(p,t):
    pm=mass(p,t);tt=np.c_[t,np.zeros(10,dtype=int)];pad=np.vstack([p,[1,1,1]])
    pv,po=neighbors(p,t);cv,co=cycle_moves(p,t,0,13)
    pair=np.tile(pm,(len(I),1));factor=pad[M,tt[J,M]]/pad[M,tt[I,M]]*pad[N,tt[J,N]]/pad[N,tt[I,N]]
    ix=np.arange(len(I));pair[ix,I]=pm[I]*factor;pair[ix,J]=pm[J]/factor
    cyc=np.tile(pm,(len(CI),1));ix=np.arange(len(CI))
    for a,b in [(CI,CJ),(CJ,CK),(CK,CI)]:cyc[ix,a]=pm[a]*p[CM,t[b,CM]]/p[CM,t[a,CM]]
    return np.vstack([pair,cyc]),np.r_[po,co],len(I)


def objectives(probabilities,kind):
    ordered=np.sort(probabilities,axis=-1);minimum=ordered[...,0];bottom=ordered[...,:5].sum(-1);total=ordered.sum(-1)
    return np.stack([minimum,bottom,total] if kind=='Q1' else [bottom,minimum,total],axis=-1)


def lex_better(a,b):
    for av,bv in zip(a,b):
        if av>bv+1e-18:return True
        if av<bv-1e-18:return False
    return False


def optimize(p,counts,seeds_initial,total_floor,coverage_floor,kind,starts=24,cap=180,seed=171670):
    began=time.perf_counter();rng=np.random.default_rng(seed);lookup=lookup_table(draw_distribution(p))
    feasible=[t.copy() for t in seeds_initial if mass(p,t).sum()>=total_floor-1e-16 and lookup[signature(t)]>=coverage_floor-1e-14]
    if not feasible:raise ValueError('No feasible starting solution found; not proof of infeasibility')
    best=max(feasible,key=lambda t:tuple(objectives(mass(p,t),kind)));best=best.copy();bo=objectives(mass(p,best),kind);runs=[]
    for start in range(starts):
        t=feasible[start%len(feasible)].copy() if start%2==0 else best.copy()
        if start>=len(feasible):
            for _ in range(12+start%12):
                vals,ok,split=probability_neighbors(p,t)
                ok &= vals.sum(1)>=total_floor
                if coverage_floor>0:ok &= np.concatenate(move_coverages(t,lookup))>=coverage_floor-1e-14
                choices=np.flatnonzero(ok)
                if not len(choices):break
                idx=int(rng.choice(choices));t=moved(t,idx) if idx<split else cycled(t,idx-split)
        trace=[];stop=False
        for step in range(cap):
            probs=mass(p,t);current=objectives(probs,kind);cov=float(lookup[signature(t)])
            trace.append(dict(p13=float(probs.sum()),coverage=cov,minimum=float(probs.min()),bottom5=float(np.sort(probs)[:5].sum())))
            vals,ok,split=probability_neighbors(p,t);obj=objectives(vals,kind)
            ok &= vals.sum(1)>=total_floor
            if coverage_floor>0:ok &= np.concatenate(move_coverages(t,lookup))>=coverage_floor-1e-14
            order=np.lexsort(tuple(np.where(ok,obj[:,i],-np.inf) for i in [2,1,0]));idx=int(order[-1])
            if not ok[idx] or not lex_better(obj[idx],current):stop=True;break
            t=moved(t,idx) if idx<split else cycled(t,idx-split)
        validate(t,counts);pm=mass(p,t);cov=float(lookup[signature(t)])
        assert pm.sum()>=total_floor-1e-16 and cov>=coverage_floor-1e-14
        final=objectives(pm,kind)
        if lex_better(final,bo):best=t.copy();bo=final
        runs.append(dict(start=start,local_stop=stop,iterations=step+1,final_objectives=final.tolist(),final_coverage=cov,trace=trace))
    return best,dict(seconds=time.perf_counter()-began,kind=kind,total_floor=total_floor,coverage_floor=coverage_floor,starts=starts,cap=cap,seed=seed,runs=runs,global_optimum_guaranteed=False)


def quality(pm):
    a=np.sort(pm);total=a.sum()
    return dict(minimum=float(a[0]),maximum=float(a[-1]),median=float(np.median(a)),top3_sum=float(a[-3:].sum()),top3_share=float(a[-3:].sum()/total),bottom3_sum=float(a[:3].sum()),bottom3_share=float(a[:3].sum()/total),bottom5_sum=float(a[:5].sum()),bottom5_share=float(a[:5].sum()/total),max_min_ratio=float(a[-1]/a[0]),geometric_mean=float(np.exp(np.log(a).mean())))
