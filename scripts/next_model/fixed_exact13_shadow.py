"""Fixed-count shadow portfolio search. Independent-match probabilities, no Q.
B maximizes exact13 mass; C maximizes exact union coverage with an exact13 floor.
No global optimality claim. Seeded starts; deterministic local moves.
"""
from functools import lru_cache
from itertools import combinations, product
import time
import numpy as np

POWERS=3**np.arange(12,-1,-1)
MOVES=np.array([(i,j,m,n) for i,j in combinations(range(10),2)
                for m,n in [(m,13) for m in range(13)]+list(combinations(range(13),2))])
I,J,M,N=MOVES.T


def mass(p,t):return np.prod(p[np.arange(13),t],axis=1)


def validate(t,counts):
    assert t.shape==(10,13) and np.isin(t,[0,1,2]).all()
    assert len(np.unique(t,axis=0))==10
    for s in range(3):np.testing.assert_array_equal((t==s).sum(0),counts[:,s])


def neighbors(p,t):
    """All one/two-column swaps between two tickets, excluding duplicates."""
    tt=np.c_[t,np.zeros(10,dtype=int)]
    # Swapped log contributions use the destination match's probabilities.
    logs=np.log(np.vstack([p,[1,1,1]])); tm=tt[I,M];tj=tt[J,M];tn=tt[I,N];tjn=tt[J,N]
    delta=logs[M,tj]-logs[M,tm]+logs[N,tjn]-logs[N,tn]
    pm=mass(p,t); objective=pm.sum()+pm[I]*np.expm1(delta)+pm[J]*np.expm1(-delta)
    keys=t@POWERS;weights=np.r_[POWERS,0]
    change=(tj-tm)*weights[M]+(tjn-tn)*weights[N]
    ki=keys[I]+change;kj=keys[J]-change
    valid=(change!=0)&(ki!=kj)
    for k in range(10):valid &= ((I==k)|(J==k)|((ki!=keys[k])&(kj!=keys[k])))
    return objective,valid


def moved(t,idx):
    i,j,m,n=MOVES[idx];u=t.copy();positions=[m] if n==13 else [m,n]
    u[i,positions],u[j,positions]=t[j,positions],t[i,positions]
    return u


def optimize_b(p,initial,starts=24,cap=160,seed=2501670):
    started=time.perf_counter();rng=np.random.default_rng(seed);best=initial.copy();bestmass=mass(p,best).sum();runs=[]
    counts=np.stack([(initial==s).sum(0) for s in range(3)],1)
    for r in range(starts):
        if r==0:t=initial.copy()
        elif r%2:
            t=best.copy()
            for _ in range(6):
                _,ok=neighbors(p,t);t=moved(t,int(rng.choice(np.flatnonzero(ok))))
        else:
            for attempt in range(100):
                t=np.column_stack([rng.permutation(initial[:,m]) for m in range(13)])
                if len(np.unique(t,axis=0))==10:break
            else:raise RuntimeError('Unable to initialize unique tickets')
        before=float(mass(p,t).sum());trace=[before];converged=False
        for step in range(cap):
            vals,ok=neighbors(p,t);idx=int(np.argmax(np.where(ok,vals,-np.inf)))
            if vals[idx]<=trace[-1]+1e-16:converged=True;break
            t=moved(t,idx);trace.append(float(mass(p,t).sum()))
        validate(t,counts)
        if trace[-1]>bestmass:best=t.copy();bestmass=trace[-1]
        runs.append(dict(start=r,initial=before,final=trace[-1],moves=len(trace)-1,local_stop=converged,trace=trace))
    return best,dict(seconds=time.perf_counter()-started,starts=starts,cap=cap,seed=seed,runs=runs,global_optimum_guaranteed=False)


@lru_cache(maxsize=30000)
def ball(ticket,radius):
    t=np.array(ticket);center=t@POWERS;parts=[np.array([center])]
    for r in range(1,radius+1):
        pos=np.array(list(combinations(range(13),r)));offset=np.array(list(product((1,2),repeat=r)))
        old=t[pos][:,None,:];new=(old+offset[None,:,:])%3
        parts.append((center+((new-old)*POWERS[pos][:,None,:]).sum(2)).ravel())
    return np.sort(np.concatenate(parts))


def coverage(masses,t,radius):return float(masses[np.unique(np.concatenate([ball(tuple(row),radius) for row in t]))].sum())


def optimize_c(p,initial,masses,hit_threshold,retention,cap=16):
    started=time.perf_counter();radius=13-hit_threshold;t=initial.copy();floor=float(mass(p,t).sum()*retention)
    counts=np.stack([(t==s).sum(0) for s in range(3)],1)
    value=coverage(masses,t,radius);trace=[dict(p13=float(mass(p,t).sum()),coverage=value)];evaluated=0;stopped=False
    for step in range(cap):
        vals,valid=neighbors(p,t);valid &= vals>=floor-1e-16
        bestvalue=value;bestidx=None;bestp13=float(mass(p,t).sum())
        # Exact union; remove the two changed tickets, retain coverage of the other eight.
        for i,j in combinations(range(10),2):
            ids=np.flatnonzero(valid&(I==i)&(J==j))
            if not len(ids):continue
            other=np.unique(np.concatenate([ball(tuple(t[k]),radius) for k in range(10) if k not in (i,j)]))
            othermass=float(masses[other].sum())
            for idx in ids:
                u=moved(t,int(idx));added=np.union1d(ball(tuple(u[i]),radius),ball(tuple(u[j]),radius))
                added=np.setdiff1d(added,other,assume_unique=True)
                score=othermass+float(masses[added].sum());evaluated+=1
                if score>bestvalue+1e-15 or (abs(score-bestvalue)<=1e-15 and vals[idx]>bestp13+1e-16):
                    bestidx=int(idx);bestvalue=score;bestp13=vals[idx]
        if bestidx is None:stopped=True;break
        t=moved(t,bestidx);value=coverage(masses,t,radius)
        trace.append(dict(p13=float(mass(p,t).sum()),coverage=value))
    validate(t,counts);assert mass(p,t).sum()>=floor-1e-15
    return t,dict(seconds=time.perf_counter()-started,hit_threshold=hit_threshold,retention=retention,p13_floor=floor,cap=cap,evaluated_neighbors=evaluated,local_stop=stopped,trace=trace,global_optimum_guaranteed=False)
