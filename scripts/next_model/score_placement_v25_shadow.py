"""Experimental v2.5 placement only; never changes probabilities or production.

All parameters are provisional. Symbol codes are H=0,D=1,A=2 internally.
Candidate00 is a reference, outside the 10-ticket allocation constraints.
"""
from dataclasses import asdict, dataclass
from itertools import combinations
import numpy as np
from random_remaining_v24 import allocate_per_match


@dataclass(frozen=True)
class Config:
    draw_coefficient: float = 1.2
    exchange_span: float = 4.0
    stage_weight: float = 100.0
    score_weight: float = 1.0
    late_draw_weight: float = .03
    duplicate_weight: float = .2
    run_weight: float = .01
    late_draw_threshold: float = .65
    allowed_run: int = 2
    max_steps: int = 200


def exchange_targets(row_exchanges, span):
    """Closest feasible nondecreasing integer degree sequence to a ramp.

    The prefix inequalities are the bipartite degree feasibility inequalities.
    A DP finds the exact minimum squared distance, not arbitrary rounded totals.
    """
    total = int(np.sum(row_exchanges))
    offsets = np.linspace(-span/2, span/2, 10)
    lo, hi = -span-13., span+26.
    for _ in range(70):
        mid = (lo+hi)/2
        if np.clip(mid+offsets, 0, 13).sum() < total: lo = mid
        else: hi = mid
    ideal = np.clip((lo+hi)/2+offsets,0,13)
    states = {(0,0):(0.,())}
    for slot in range(10):
        following = {}
        for (last, used),(cost,seq) in states.items():
            for k in range(last,14):
                amount = used+k
                if amount>total or amount+(9-slot)*k>total or amount+(9-slot)*13<total:continue
                # Smallest slot+1 degrees must consume the part that cannot
                # fit into the remaining largest degrees.
                if amount < total-int(np.minimum(row_exchanges,9-slot).sum()):continue
                key=(k,amount);candidate=(cost+(k-ideal[slot])**2,seq+(k,))
                if key not in following or candidate<following[key]:following[key]=candidate
        states=following
    options=[v for (last,used),v in states.items() if used==total]
    if not options:raise ValueError('No feasible exchange targets')
    _,seq=min(options)
    return np.array(seq),ideal


def initial_placement(counts, anchor, targets, losses):
    """Bipartite Havel-Hakimi construction, then deterministic symbol fill."""
    row_degrees=10-counts[np.arange(13),anchor]
    remaining=targets.copy();changed=np.zeros((10,13),dtype=bool)
    for m in sorted(range(13),key=lambda m:(-row_degrees[m],m)):
        slots=sorted(range(10),key=lambda j:(-remaining[j],j))[:row_degrees[m]]
        if any(remaining[j]<=0 for j in slots):raise AssertionError('infeasible degree construction')
        changed[slots,m]=True;remaining[slots]-=1
    if remaining.any():raise AssertionError('unfilled exchange degrees')
    t=np.tile(anchor,(10,1))
    for m in range(13):
        symbols=[s for s in sorted(range(3),key=lambda s:(losses[m,s],s)) if s!=anchor[m] for _ in range(counts[m,s])]
        slots=np.flatnonzero(changed[:,m]);t[slots,m]=symbols
    return t


def components(batch, anchor, targets, losses, cfg):
    """Normalized components; stage has a large, configurable default weight."""
    ex=(batch!=anchor).sum(axis=2)
    stage=((ex-targets)**2).mean(axis=1)/169 + np.maximum(ex[:,:-1]-ex[:,1:],0).sum(axis=1)/(9*13)
    weights=np.linspace(1,0,10)
    ticket_loss=losses[np.arange(13)[None,None,:],batch].sum(axis=2)
    positional=(ticket_loss*weights).sum(axis=1)/(13*weights.sum())
    ds=(batch==1).sum(axis=2);total=np.maximum(ds.sum(axis=1),1)
    centroid=(ds*np.linspace(0,1,10)).sum(axis=1)/total
    late=np.maximum(centroid-cfg.late_draw_threshold,0)**2
    duplicates=np.zeros(len(batch))
    for i,j in combinations(range(10),2):duplicates+=(batch[:,i]==batch[:,j]).all(axis=1)
    duplicates/=45
    width=cfg.allowed_run+1
    runs=np.zeros(len(batch))
    for j in range(11-width):runs+=(batch[:,j:j+width]==batch[:,j:j+1]).all(axis=1).sum(axis=1)
    runs/=(11-width)*13
    secondary=cfg.score_weight*positional+cfg.late_draw_weight*late+cfg.duplicate_weight*duplicates+cfg.run_weight*runs
    return cfg.stage_weight*stage,secondary,{'stage':stage,'positional_loss':positional,'late_D':late,'duplicates':duplicates,'runs':runs,'D_centroid':centroid}


def generate(probabilities, counts=None, config=None, draw_coefficients=None):
    cfg=config or Config();p=np.asarray(probabilities,dtype=float)
    validated=allocate_per_match(p)  # shape/range/normalization validation only
    raw=validated if counts is None else np.asarray(counts)
    if raw.shape!=(13,3) or not np.isfinite(raw).all() or (raw<0).any() or not np.equal(raw,np.floor(raw)).all() or not (raw.sum(axis=1)==10).all():raise ValueError('Invalid fixed counts')
    counts=raw.astype(int)
    if np.any((p==0)&(counts>0)):raise ValueError('Positive allocation to zero-probability symbol')
    values=[cfg.exchange_span,cfg.stage_weight,cfg.score_weight,cfg.late_draw_weight,cfg.duplicate_weight,cfg.run_weight,cfg.late_draw_threshold]
    if not np.isfinite(values).all() or min(values)<0 or cfg.stage_weight<=0 or cfg.score_weight<=0 or not 0<=cfg.late_draw_threshold<=1 or not 1<=cfg.allowed_run<=9 or not isinstance(cfg.allowed_run,int) or not isinstance(cfg.max_steps,int) or cfg.max_steps<1:raise ValueError('Invalid configuration')
    kd=np.full(13,cfg.draw_coefficient) if draw_coefficients is None else np.asarray(draw_coefficients,dtype=float)
    if kd.shape!=(13,) or not np.isfinite(kd).all() or (kd<=0).any():raise ValueError('Invalid D coefficients')
    anchor=np.argmax(p,axis=1);scores=p.copy();scores[:,1]*=kd
    losses=scores[np.arange(13),anchor][:,None]-scores
    targets,ideal=exchange_targets(10-counts[np.arange(13),anchor],cfg.exchange_span)
    # Same coefficient-independent initial layout across all K settings.
    base_losses=p[np.arange(13),anchor][:,None]-p
    initial=initial_placement(counts,anchor,targets,base_losses);tickets=initial.copy()
    stage,secondary,details=components(tickets[None],anchor,targets,losses,cfg)
    trace=[float(stage[0]+secondary[0])];stopped=False
    for step in range(cfg.max_steps):
        moves=[]
        for i,j in combinations(range(10),2):
            different=np.flatnonzero(tickets[i]!=tickets[j])
            moves.extend((i,j,int(m),-1) for m in different)
            delta=(tickets[i]!=anchor).astype(int)-(tickets[j]!=anchor).astype(int)
            moves.extend((i,j,int(m),int(n)) for m,n in combinations(different,2) if delta[m]*delta[n]==-1)
        if not moves:stopped=True;break
        move=np.array(moves);batch=np.repeat(tickets[None],len(moves),axis=0);idx=np.arange(len(moves))
        for column in (2,3):
            valid=move[:,column]>=0;z=idx[valid];i,j,m=move[valid,0],move[valid,1],move[valid,column]
            batch[z,i,m]=tickets[j,m];batch[z,j,m]=tickets[i,m]
        a,b,_=components(batch,anchor,targets,losses,cfg)
        best=int(np.argmin(a+b))
        improves=a[best]+b[best]<stage[0]+secondary[0]-1e-12
        if not improves:stopped=True;break
        tickets=batch[best];stage,secondary,details=components(tickets[None],anchor,targets,losses,cfg)
        trace.append(float(stage[0]+secondary[0]))
    for s in range(3):np.testing.assert_array_equal((tickets==s).sum(axis=0),counts[:,s])
    if int((tickets!=anchor).sum()) != int(targets.sum()):raise AssertionError('Exchange total changed')
    duplicates=[[i+1,j+1] for i,j in combinations(range(10),2) if np.array_equal(tickets[i],tickets[j])]
    audit={'version':'v25_score_shadow_experimental','config':asdict(cfg),'counts':counts.tolist(),
           'anchor':anchor.tolist(),'scores_HDA':scores.tolist(),'losses_HDA':losses.tolist(),'D_coefficients':kd.tolist(),
           'total_exchanges':int(targets.sum()),'target_exchanges':targets.tolist(),'ideal_exchanges':ideal.tolist(),
           'initial_tickets':initial.tolist(),'accepted_moves':len(trace)-1,'stopped_at_local_optimum':stopped,
           'objective_trace':trace,'final_components':{k:float(v[0]) for k,v in details.items()},
           'duplicate_pairs':duplicates,'duplicate_unavoidable_proven':bool(duplicates and (np.count_nonzero(counts,axis=1)==1).all()),
           'randomness_used':False,'probabilities_changed':False,'counts_preserved':True,'candidate00_consumes_counts':False,
           'global_optimum_guaranteed':False,'provisional_parameters':True}
    return tickets,audit


def diagnostics(p,tickets,audit):
    p=np.asarray(p);anchor=np.array(audit['anchor']);scores=np.array(audit['scores_HDA']);losses=np.array(audit['losses_HDA'])
    order=np.argsort(-p,axis=1,kind='stable');ranks=np.empty((13,3),dtype=int)
    np.put_along_axis(ranks,order,np.tile(np.arange(1,4),(13,1)),axis=1)
    rows=[]
    for j,t in enumerate(np.vstack([anchor,tickets])):
        rank=ranks[np.arange(13),t]
        rows.append({'candidate':f'{j:02d}','purchase':j>0,'symbols':''.join('HDA'[s] for s in t),
                     'exchanges':int((t!=anchor).sum()),'target':0 if j==0 else audit['target_exchanges'][j-1],
                     'D_count':int((t==1).sum()),'rank2':int((rank==2).sum()),'rank3':int((rank==3).sum()),
                     'joint_probability':float(np.prod(p[np.arange(13),t])),
                     'placement_value':float(scores[np.arange(13),t].sum()),
                     'exchange_loss':float(losses[np.arange(13),t].sum()),
                     'duplicate_purchase_candidates':[i+1 for i,other in enumerate(tickets) if (j==0 or i+1!=j) and np.array_equal(t,other)]})
    return rows
