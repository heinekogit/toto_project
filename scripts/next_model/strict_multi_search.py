"""Full-product multi bets: D may occur only in HD/DA/HDA selections.
Single-multi families are exhaustively optimized. Mixed families use a candidate
pool MILP plus a bound on all omitted blocks to distinguish proof from search.
"""
from itertools import combinations,product
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

TYPES={'D1':(1,0),'D2':(2,0),'D3':(3,0),'T1':(0,1),'TD':(1,1),'T2':(0,2)}
COST={k:2**d*3**t for k,(d,t) in TYPES.items()}
POWERS=3**np.arange(12,-1,-1)
HA_POWERS=2**np.arange(12,-1,-1)


def expand(masks):
    options=[[s for s in range(3) if int(mask)&(1<<s)] for mask in masks]
    return np.array(list(product(*options)),dtype=np.int8)


def make_block(kind,base_id,ds,ts,ha,p):
    base=ha[base_id];masks=(1<<base).astype(int)
    for m in ds:masks[m]|=2
    for m in ts:masks[m]=7
    t=expand(masks)
    return dict(kind=kind,base_id=int(base_id),double=list(map(int,ds)),triple=list(map(int,ts)),masks=masks.tolist(),codes=(t@POWERS).tolist(),mass=float(np.prod(p[np.arange(13),t],axis=1).sum()),cost=len(t))


def enumerate_blocks(p,pool_size=256):
    start=time.perf_counter();ha_ids=np.arange(2**13);ha=((ha_ids[:,None]//HA_POWERS)%2*2).astype(np.int8)
    hp=np.prod(p[np.arange(13),ha],axis=1);ha_order=np.lexsort((ha_ids,-hp));hc=ha@POWERS
    pools={};exact={};stats={}
    for kind,(nd,nt) in TYPES.items():
        began=time.perf_counter();cost=COST[kind];single_count=10-cost;retained=[];best=None;total=0
        single_ids=ha_order[:single_count+2**nt]
        for ts in combinations(range(13),nt):
            triple_bit=sum(int(HA_POWERS[m]) for m in ts)
            bids=ha_ids[(ha_ids&triple_bit)==0];bases=ha[bids]
            remaining=[m for m in range(13) if m not in ts]
            # H/A-only singles overlap a multi iff all its non-triple positions match.
            excluded=bids[:,None]==(single_ids[None,:]&(~triple_bit))
            select=(~excluded)&(np.cumsum(~excluded,axis=1)<=single_count)
            fill=(select*hp[single_ids][None,:]).sum(1)
            for ds in combinations(remaining,nd):
                factors=p[np.arange(13),bases].copy()
                if nd:factors[:,ds]+=p[np.array(ds),1]
                if nt:factors[:,ts]=1
                weights=factors.prod(1);scores=weights+fill;total+=len(bids)
                j=int(scores.argmax())
                candidate=(float(scores[j]),int(bids[j]),tuple(ds),tuple(ts))
                if best is None or candidate[0]>best[0]:best=candidate
                n=min(pool_size,len(bids));chosen=np.argpartition(-weights,n-1)[:n]
                retained.extend((float(weights[i]),int(bids[i]),tuple(ds),tuple(ts)) for i in chosen)
                retained=sorted(retained,key=lambda z:(-z[0],z[1],z[2],z[3]))[:pool_size]
        pools[kind]=[make_block(kind,bid,ds,ts,ha,p) for _,bid,ds,ts in retained]
        _,bid,ds,ts=best;block=make_block(kind,bid,ds,ts,ha,p);covered=set(block['codes'])
        fill_ids=[int(i) for i in ha_order if int(hc[i]) not in covered][:single_count]
        exact[kind]=dict(blocks=[block],singles=fill_ids,p13=block['mass']+float(hp[fill_ids].sum()),strict_optimal=True,search='full enumeration',status='optimal',seconds=time.perf_counter()-began)
        stats[kind]=dict(enumerated_blocks=total,retained_blocks=len(retained),omitted_mass_upper_bound=float(retained[-1][0]),max_block_mass=pools[kind][0]['mass'],seconds=time.perf_counter()-began)
        print(kind,'full enumeration',total,'best10',exact[kind]['p13'],flush=True)
    return ha,hp,ha_order,pools,exact,dict(types=stats,pool_size=pool_size,seconds=time.perf_counter()-start)


def all_patterns():
    names=list(TYPES);results=[]
    def visit(i,remaining,counts):
        if i==len(names):
            if sum(counts):results.append((dict(zip(names,counts)),remaining))
            return
        for n in range(remaining//COST[names[i]]+1):visit(i+1,remaining-n*COST[names[i]],counts+[n])
    visit(0,10,[])
    return results


def solve_mixed(counts,singles,ha,hp,ha_order,pools,stats,time_limit=5.):
    began=time.perf_counter();names=[k for k in TYPES if counts[k]]
    blocks=[b for k in names for b in pools[k]]
    # At most 10-S distinct H/A outcomes can be excluded by purchased blocks.
    # Thus the best S nonoverlapping H/A singles are among the global first 10.
    single_ids=list(map(int,ha_order[:10])) if singles else []
    costs=[b['mass'] for b in blocks]+[float(hp[i]) for i in single_ids]
    rows=[];cols=[];vals=[];lb=[];ub=[]
    for k in names:
        r=len(lb);lb.append(counts[k]);ub.append(counts[k])
        for j,b in enumerate(blocks):
            if b['kind']==k:rows.append(r);cols.append(j);vals.append(1)
    r=len(lb);lb.append(singles);ub.append(singles)
    for j in range(len(single_ids)):rows.append(r);cols.append(len(blocks)+j);vals.append(1)
    incidence={}
    for j,b in enumerate(blocks):
        for code in b['codes']:incidence.setdefault(code,[]).append(j)
    for j,bid in enumerate(single_ids):incidence.setdefault(int(ha[bid]@POWERS),[]).append(len(blocks)+j)
    for variables in incidence.values():
        if len(variables)<2:continue
        r=len(lb);lb.append(0);ub.append(1)
        for j in variables:rows.append(r);cols.append(j);vals.append(1)
    mat=coo_matrix((vals,(rows,cols)),shape=(len(lb),len(costs))).tocsc()
    scale=1e9
    result=milp(-np.array(costs)*scale,integrality=np.ones(len(costs)),bounds=Bounds(0,1),constraints=LinearConstraint(mat,np.array(lb),np.array(ub)),options={'time_limit':time_limit,'mip_rel_gap':1e-9})
    # Any full-domain solution omitted from this pool contains at least one
    # block with mass no larger than that type's retained cutoff.
    excluded_ub=0.
    for missing in names:
        bound=stats['types'][missing]['omitted_mass_upper_bound']+float(hp[ha_order[:singles]].sum())
        for k in names:
            n=counts[k]-(k==missing)
            bound+=sum(b['mass'] for b in pools[k][:n])
        excluded_ub=max(excluded_ub,bound)
    if result.x is None:
        return dict(status='pool_infeasible' if result.status==2 else 'no_solution_within_budget',solver_message=result.message,strict_optimal=False,seconds=time.perf_counter()-began,omitted_upper_bound=excluded_ub)
    selected=np.flatnonzero(result.x>.5);chosen=[blocks[j] for j in selected if j<len(blocks)];ss=[single_ids[j-len(blocks)] for j in selected if j>=len(blocks)]
    for k in names:assert sum(b['kind']==k for b in chosen)==counts[k]
    assert len(ss)==singles
    codes=[c for b in chosen for c in b['codes']]+[int(ha[i]@POWERS) for i in ss]
    assert len(codes)==10 and len(set(codes))==10
    value=sum(b['mass'] for b in chosen)+float(hp[ss].sum())
    pool_bound=float(-result.mip_dual_bound/scale) if getattr(result,'mip_dual_bound',None) is not None else float('inf')
    certified=result.status==0 and excluded_ub<=value+1e-14
    return dict(blocks=chosen,singles=ss,p13=value,status='optimal' if certified else 'best_found',strict_optimal=certified,search='candidate pool integer program',pool_solver_status=int(result.status),pool_gap=float(result.mip_gap),pool_upper_bound=pool_bound,omitted_upper_bound=excluded_ub,full_upper_bound=max(value,pool_bound,excluded_ub),solver_message=result.message,seconds=time.perf_counter()-began)


def validate_purchase(solution,ha,p):
    rows=[]
    for b in solution['blocks']:
        if any(mask not in [1,4,3,6,7] for mask in b['masks']):raise AssertionError('D-only or unsupported specification')
        t=expand(b['masks']);assert len(t)==b['cost']
        rows.extend(t)
    for bid in solution['singles']:
        assert not (ha[bid]==1).any();rows.append(ha[bid])
    t=np.asarray(rows,dtype=np.int8)
    assert t.shape==(10,13) and len(np.unique(t,axis=0))==10
    np.testing.assert_allclose(np.prod(p[np.arange(13),t],axis=1).sum(),solution['p13'],rtol=1e-12)
    return t
