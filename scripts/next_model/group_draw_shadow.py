"""Deterministic group-constrained exact13 search; no production integration."""
from itertools import combinations
import time
import numpy as np
from fixed_exact13_shadow import mass,validate,neighbors,moved,I,J,M,N,POWERS


def groups(d,shift=0):
    if d==0:return []
    edges=[(10*k)//d for k in range(d+1)]
    return [[(r+shift)%10 for r in range(edges[k],edges[k+1])] for k in range(d)]


def patterns(counts,circular):
    h,d,a=map(int,counts);out=[];shifts=[];allowed=np.zeros(1024,dtype=bool)
    for positions in combinations(range(10),d):
        mask=sum(1<<r for r in positions)
        options=[s for s in range(10 if circular else 1) if all(sum(r in positions for r in g)==1 for g in groups(d,s))]
        if not options:continue
        allowed[mask]=True;rest=[r for r in range(10) if r not in positions]
        for hs in combinations(rest,h):
            col=np.full(10,2,dtype=np.int8);col[list(positions)]=1;col[list(hs)]=0
            out.append(col);shifts.append(options[0])
    return np.array(out),np.array(shifts),allowed


def validate_groups(t,counts,circular):
    validate(t,counts);shifts=[]
    for m in range(13):
        d=int(counts[m,1]);options=[s for s in range(10 if circular else 1) if all(sum(t[r,m]==1 for r in g)==1 for g in groups(d,s))]
        if not options:raise AssertionError('Invalid draw groups')
        shifts.append(options[0])
    return shifts


def pair_valid(t,allowed):
    masks=((t==1)*(1<<np.arange(10))[:,None]).sum(0)
    tt=np.c_[t,np.zeros(10,dtype=int)];masks=np.r_[masks,0]
    ok=np.ones(len(I),dtype=bool)
    lookup=np.vstack([allowed,np.ones(1024,dtype=bool)])
    for pos in [M,N]:
        change=((tt[J,pos]==1).astype(int)-(tt[I,pos]==1))*((1<<I)-(1<<J))
        ok &= lookup[pos,masks[pos]+change]
    return ok


def optimize(p,counts,circular,starts=64,cap=80,warm=None):
    started=time.perf_counter();sets=[patterns(c,circular) for c in counts];allowed=np.array([s[2] for s in sets]);runs=[];best=None;bestmass=-1
    for start in range(starts):
        if start==0 and warm is not None:t=warm.copy()
        else:
            for attempt in range(100):
                n=start+attempt*starts
                t=np.column_stack([opts[(n*(2*m+1)+n*n*(m+1)+37*m)%len(opts)] for m,(opts,_,_) in enumerate(sets)])
                if len(np.unique(t,axis=0))==10:break
            else:raise ValueError('Deterministic initial generation found no unique portfolio; no infeasibility proof')
        validate_groups(t,counts,circular);trace=[float(mass(p,t).sum())];stopped=False;pair_steps=0
        for sweep in range(cap):
            changed=False
            for m in np.roll(np.arange(13),start%13):
                opts=sets[m][0];pm=mass(p,t);weights=pm/p[m,t[:,m]]
                values=p[m,opts]@weights
                keys=(t@POWERS)[None,:]+(opts-t[:,m])*POWERS[m]
                ok=np.ones(len(opts),dtype=bool)
                for i,j in combinations(range(10),2):ok &= keys[:,i]!=keys[:,j]
                idx=int(np.argmax(np.where(ok,values,-np.inf)))
                if ok[idx] and values[idx]>pm.sum()+1e-16:
                    t[:,m]=opts[idx];changed=True;trace.append(float(mass(p,t).sum()))
            vals,ok=neighbors(p,t);ok &= pair_valid(t,allowed)
            idx=int(np.argmax(np.where(ok,vals,-np.inf)))
            if ok[idx] and vals[idx]>mass(p,t).sum()+1e-16:
                t=moved(t,idx);pair_steps+=1;changed=True;trace.append(float(mass(p,t).sum()))
            if not changed:stopped=True;break
        shifts=validate_groups(t,counts,circular)
        if trace[-1]>bestmass:best=t.copy();bestmass=trace[-1]
        runs.append(dict(start=start,initial=trace[0],final=trace[-1],sweeps=sweep+1,pair_moves=pair_steps,local_stop=stopped,trace=trace,best_so_far=bestmass))
        if (start+1)%16==0:print(('G2' if circular else 'G1'),start+1,'best',bestmass,flush=True)
    return best,dict(seconds=time.perf_counter()-started,circular=circular,starts=starts,cap=cap,randomness_used=False,runs=runs,shifts=validate_groups(best,counts,circular),column_pattern_counts=[len(s[0]) for s in sets],global_optimum_guaranteed=False)
