"""Fixed-count lexicographic D-count coverage, subject to exact13 floor."""
import time
import numpy as np
from fixed_exact13_shadow import mass,validate,neighbors,moved,I,J,M,N
from fixed_draw_bounds_shadow import cycle_moves,cycled,CI,CJ,CK,CM


def draw_distribution(p):
    result=np.array([1.])
    for v in np.asarray(p)[:,1]:result=np.convolve(result,[1-v,v])
    return result


def lookup_table(dist):
    masks=np.arange(1<<14)
    return ((masks[:,None]>>np.arange(14))&1)@dist


def signature(t):return int(np.bitwise_or.reduce(1<<(t==1).sum(1)))


def move_coverages(t,lookup):
    d=(t==1).sum(1);tt=np.c_[t,np.zeros(10,dtype=int)]
    delta=(tt[J,M]==1).astype(int)-(tt[I,M]==1)+(tt[J,N]==1).astype(int)-(tt[I,N]==1)
    pmask=np.zeros(len(I),dtype=int)
    for k in range(10):pmask |= np.where((I!=k)&(J!=k),1<<d[k],0)
    pmask |= (1<<(d[I]+delta))|(1<<(d[J]-delta))
    cmask=np.zeros(len(CI),dtype=int)
    for k in range(10):cmask |= np.where((CI!=k)&(CJ!=k)&(CK!=k),1<<d[k],0)
    old=[t[CI,CM],t[CJ,CM],t[CK,CM]];new=[old[1],old[2],old[0]]
    for ix,a,b in zip([CI,CJ,CK],old,new):cmask |= 1<<(d[ix]+(b==1).astype(int)-(a==1))
    return lookup[pmask],lookup[cmask]


def better(c,p,bc,bp):return c>bc+1e-14 or (abs(c-bc)<=1e-14 and p>bp+1e-16)


def optimize(p,b,retention,starts=32,cap=240,seed=951670,warm=None):
    began=time.perf_counter();rng=np.random.default_rng(seed);floor=float(mass(p,b).sum()*retention)
    counts=np.stack([(b==s).sum(0) for s in range(3)],1);lookup=lookup_table(draw_distribution(p))
    best=b.copy();bc=float(lookup[signature(best)]);bp=float(mass(p,best).sum());runs=[];pool={tuple(r) for r in b}
    if warm is not None and mass(p,warm).sum()>=floor:
        wc=float(lookup[signature(warm)]);wp=float(mass(p,warm).sum())
        if better(wc,wp,bc,bp):best=warm.copy();bc=wc;bp=wp
    for start in range(starts):
        t=b.copy() if start%3==0 else best.copy()
        # Feasible random walks diversify starting D counts without relaxing the floor.
        walk_steps=0 if start==0 else 15+start%16
        for _ in range(walk_steps):
            pv,po=neighbors(p,t);cv,co=cycle_moves(p,t,0,13)
            picks=np.flatnonzero(np.r_[po&(pv>=floor),co&(cv>=floor)])
            if not len(picks):break
            idx=int(rng.choice(picks));t=moved(t,idx) if idx<len(pv) else cycled(t,idx-len(pv))
        validate(t,counts);assert mass(p,t).sum()>=floor-1e-16
        initial_counts=(t==1).sum(1).tolist();trace=[dict(coverage=float(lookup[signature(t)]),p13=float(mass(p,t).sum()))];stop=False
        for step in range(cap):
            pv,po=neighbors(p,t);cv,co=cycle_moves(p,t,0,13);pc,cc=move_coverages(t,lookup)
            value=np.r_[pv,cv];cov=np.r_[pc,cc];ok=np.r_[po,co]&(value>=floor)
            idx=int(np.lexsort((np.where(ok,value,-np.inf),np.where(ok,cov,-np.inf)))[-1])
            if not ok[idx] or not better(cov[idx],value[idx],trace[-1]['coverage'],trace[-1]['p13']):stop=True;break
            t=moved(t,idx) if idx<len(pv) else cycled(t,idx-len(pv))
            validate(t,counts);actual=float(mass(p,t).sum());assert actual>=floor-1e-16
            trace.append(dict(coverage=float(lookup[signature(t)]),p13=actual));pool.update(tuple(r) for r in t)
        final=trace[-1]
        if better(final['coverage'],final['p13'],bc,bp):best=t.copy();bc=final['coverage'];bp=final['p13']
        runs.append(dict(start=start,initial_D_counts=initial_counts,final_D_counts=(t==1).sum(1).tolist(),moves=len(trace)-1,local_stop=stop,trace=trace))
    return best,dict(seconds=time.perf_counter()-began,retention=retention,p13_floor=floor,starts=starts,cap=cap,seed=seed,runs=runs,global_optimum_guaranteed=False),np.array(sorted(pool))
