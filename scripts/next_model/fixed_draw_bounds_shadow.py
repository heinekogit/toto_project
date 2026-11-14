"""Experimental exact13 objective with per-ticket draw bounds; no production hook."""
import time
from itertools import combinations
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, vstack
from fixed_exact13_shadow import neighbors, moved, mass, validate, I,J,M,N,POWERS

CYCLES=np.array([(i,j,k,m) for a,b,c in combinations(range(10),3) for i,j,k in [(a,b,c),(a,c,b)] for m in range(13)])
CI,CJ,CK,CM=CYCLES.T


def proposed_bounds(total,width=1):
    """Unadopted general rule: floor(mean-width), ceil(mean+width), clipped."""
    mean=total/10
    return max(0,int(np.floor(mean-width))),min(13,int(np.ceil(mean+width)))


def feasible(counts,low,high):
    """Exact integer feasibility with lazy cuts excluding identical ticket pairs."""
    if not (0<=low<=high<=13):raise ValueError('Invalid draw bounds')
    if counts.shape!=(13,3) or (counts<0).any() or not (counts.sum(1)==10).all():raise ValueError('Invalid counts')
    if not 10*low<=counts[:,1].sum()<=10*high:raise ValueError('Infeasible: total draw count outside 10*lower..10*upper')
    def idx(r,m,s):return r*39+m*3+s
    rows=[];lbs=[];ubs=[]
    for r in range(10):
        for m in range(13):rows.append([idx(r,m,s) for s in range(3)]);lbs.append(1);ubs.append(1)
    for m in range(13):
        for s in range(3):rows.append([idx(r,m,s) for r in range(10)]);lbs.append(counts[m,s]);ubs.append(counts[m,s])
    for r in range(10):rows.append([idx(r,m,1) for m in range(13)]);lbs.append(low);ubs.append(high)
    cuts=0
    for attempt in range(100):
        mat=lil_matrix((len(rows),390))
        for row,cols in enumerate(rows):mat[row,cols]=1
        result=milp(np.zeros(390),integrality=np.ones(390),bounds=Bounds(0,1),constraints=LinearConstraint(mat.tocsr(),lbs,ubs),options={'time_limit':15})
        if result.status==2:raise ValueError('Integer feasibility model proves no unique portfolio satisfies fixed counts and draw bounds')
        if result.x is None:raise RuntimeError('Feasibility solver inconclusive; counts were not modified')
        t=np.rint(result.x).reshape(10,13,3).argmax(2)
        duplicates=[(i,j) for i,j in combinations(range(10),2) if np.array_equal(t[i],t[j])]
        if not duplicates:
            validate(t,counts);assert ((t==1).sum(1)>=low).all() and ((t==1).sum(1)<=high).all()
            return t,dict(status='feasible_unique',duplicate_cuts=cuts,solver_message=result.message)
        for i,j in duplicates:
            rows.append([idx(r,m,int(t[r,m])) for r in [i,j] for m in range(13)]);lbs.append(-np.inf);ubs.append(25);cuts+=1
    raise RuntimeError('Uniqueness feasibility cut budget exhausted; no infeasibility claim')


def pair_moves(p,t,low,high):
    values,ok=neighbors(p,t);draw=(t==1).sum(1);tt=np.c_[t,np.zeros(10,dtype=int)]
    delta=(tt[J,M]==1).astype(int)-(tt[I,M]==1)+(tt[J,N]==1).astype(int)-(tt[I,N]==1)
    ok &= (draw[I]+delta>=low)&(draw[I]+delta<=high)&(draw[J]-delta>=low)&(draw[J]-delta<=high)
    return values,ok


def cycle_moves(p,t,low,high):
    keys=t@POWERS;pm=mass(p,t);dc=(t==1).sum(1);old=[t[CI,CM],t[CJ,CM],t[CK,CM]];new=[old[1],old[2],old[0]];ids=[CI,CJ,CK]
    value=np.full(len(CYCLES),pm.sum());ok=np.ones(len(CYCLES),dtype=bool);newkeys=[]
    for r,a,b in zip(ids,old,new):
        value+=pm[r]*(p[CM,b]/p[CM,a]-1)
        d=dc[r]+(b==1).astype(int)-(a==1)
        ok &= (d>=low)&(d<=high)
        newkeys.append(keys[r]+(b-a)*POWERS[CM])
    ok &= (newkeys[0]!=newkeys[1])&(newkeys[0]!=newkeys[2])&(newkeys[1]!=newkeys[2])
    ok &= (newkeys[0]!=keys[CI])|(newkeys[1]!=keys[CJ])|(newkeys[2]!=keys[CK])
    for k in range(10):
        other=(CI!=k)&(CJ!=k)&(CK!=k)
        ok &= ~other|((newkeys[0]!=keys[k])&(newkeys[1]!=keys[k])&(newkeys[2]!=keys[k]))
    return value,ok


def cycled(t,idx):
    i,j,k,m=CYCLES[idx];u=t.copy();u[i,m],u[j,m],u[k,m]=t[j,m],t[k,m],t[i,m];return u


def random_start(counts,low,high,rng,fallback):
    for _ in range(100):
        t=np.column_stack([rng.permutation(np.repeat(np.arange(3),counts[m])) for m in range(13)])
        for step in range(130):
            d=(t==1).sum(1)
            if (d>=low).all() and (d<=high).all():break
            donors=np.flatnonzero(d>high) if (d>high).any() else np.flatnonzero(d>low)
            recipients=np.flatnonzero(d<low) if (d<low).any() else np.flatnonzero(d<high)
            i=int(rng.choice(donors));j=int(rng.choice(recipients));positions=np.flatnonzero((t[i]==1)&(t[j]!=1))
            if not len(positions):continue
            m=int(rng.choice(positions));t[i,m],t[j,m]=t[j,m],t[i,m]
        if len(np.unique(t,axis=0))==10 and ((t==1).sum(1)>=low).all() and ((t==1).sum(1)<=high).all():return t
    return fallback.copy()


def optimize(p,counts,low,high,starts=64,cap=240,seed=251670):
    started=time.perf_counter();initial,proof=feasible(counts,low,high);rng=np.random.default_rng(seed)
    best=initial.copy();bestvalue=mass(p,best).sum();runs=[]
    for r in range(starts):
        if r==0:t=initial.copy()
        elif r%2==0:t=random_start(counts,low,high,rng,initial)
        else:
            t=best.copy()
            for _ in range(10):
                _,ok=pair_moves(p,t,low,high);choices=np.flatnonzero(ok)
                if len(choices):t=moved(t,int(rng.choice(choices)))
        trace=[float(mass(p,t).sum())];cycles=0;stop=False
        for step in range(cap):
            pv,po=pair_moves(p,t,low,high);cv,co=cycle_moves(p,t,low,high)
            pi=int(np.argmax(np.where(po,pv,-np.inf)));ci=int(np.argmax(np.where(co,cv,-np.inf)))
            vp=pv[pi] if po[pi] else -np.inf;vc=cv[ci] if co[ci] else -np.inf
            if max(vp,vc)<=trace[-1]+1e-16:stop=True;break
            if vc>vp:t=cycled(t,ci);cycles+=1
            else:t=moved(t,pi)
            trace.append(float(mass(p,t).sum()))
        validate(t,counts);assert ((t==1).sum(1)>=low).all() and ((t==1).sum(1)<=high).all()
        if trace[-1]>bestvalue:best=t.copy();bestvalue=trace[-1]
        runs.append(dict(start=r,final=trace[-1],moves=len(trace)-1,three_ticket_cycles=cycles,local_stop=stop,trace=trace))
    return best,dict(seconds=time.perf_counter()-started,low=low,high=high,starts=starts,cap=cap,seed=seed,feasibility=proof,runs=runs,global_optimum_guaranteed=False)
