"""Deterministic proposal: preserve v2.4 marginals, allocate draw-count strata.

The input probabilities are unchanged. No random numbers or actual results are
used. A Poisson-binomial draw-count distribution assumes independent fixtures.
Ticket selection is greedy within predetermined draw-count strata, not a global
optimum for winning-ticket coverage.
"""
from itertools import combinations
import math
import numpy as np
from random_remaining_v24 import allocate_per_match


def draw_count_distribution(probabilities):
    p=np.asarray(probabilities,dtype=float)
    # Reuse production input validation without changing its rounding.
    allocate_per_match(p)
    dist=np.array([1.])
    for d in p[:,1]:dist=np.convolve(dist,[1-d,d])
    return dist


def feasible_draw_degrees(rows, columns):
    rows=np.asarray(rows,dtype=int);cols=sorted(map(int,columns),reverse=True)
    if np.any(rows<0) or np.any(rows>len(cols)) or any(c<0 or c>len(rows) for c in cols):return False
    if rows.sum()!=sum(cols):return False
    return all(sum(cols[:k])<=np.minimum(rows,k).sum() for k in range(1,len(cols)+1))


def draw_quotas(counts, distribution):
    """Closest integer quotas to 10*P(K), with exact D total and feasibility.

    Dynamic programming enforces all prefix degree inequalities while visiting
    K in descending order. This gives a feasible binary D-incidence matrix.
    """
    row_d=np.asarray(counts)[:,1];n=int(np.asarray(counts)[0].sum());total=int(row_d.sum())
    target=n*np.asarray(distribution)
    states={(0,0):(0.,())}
    for k in range(len(distribution)-1,-1,-1):
        next_states={}
        for (used,draws),(cost,qs) in states.items():
            for q in range(n-used+1):
                if q and distribution[k]==0:continue
                s=draws+q*k;t=used+q
                if s>total:continue
                if any(draws+j*k>int(np.minimum(row_d,used+j).sum()) for j in range(1,q+1)):continue
                candidate=(cost+(q-target[k])**2,qs+(q,))
                key=(t,s)
                if key not in next_states or candidate<next_states[key]:next_states[key]=candidate
        states=next_states
    if (n,total) not in states:raise ValueError('No feasible draw-count quotas')
    cost,reverse=states[(n,total)];q=np.array(reverse[::-1],dtype=int)
    columns=[k for k in range(len(q)) for _ in range(q[k])]
    if not feasible_draw_degrees(row_d,columns):raise AssertionError('invalid quota degrees')
    return q,float(cost)


def best_ticket(remaining,p,k,following,previous):
    """Find the strongest unused ticket permitting completion of D marginals.

    H/A choices have no column-count restrictions. Top len(previous)+1 H/A
    assignments for each D subset suffice to find its strongest unseen ticket.
    Future ticket uniqueness is not guaranteed; forced duplicates are flagged.
    """
    n=int(remaining[0].sum());matches=len(p);best=None;fallback=None
    for positions in combinations(range(matches),k):
        mask=np.zeros(matches,dtype=bool);mask[list(positions)]=True
        if np.any(mask & (remaining[:,1]==0)) or np.any(~mask & (remaining[:,1]==n)):continue
        if not feasible_draw_degrees(remaining[:,1]-mask,following):continue
        candidates=[(0.,())]
        for i in range(matches):
            symbols=[1] if mask[i] else [s for s in (0,2) if remaining[i,s]>0 and p[i,s]>0]
            candidates=sorted((loss-math.log(p[i,s]),ticket+(s,)) for loss,ticket in candidates for s in symbols)[:len(previous)+1]
            if not candidates:break
        for candidate in candidates:
            if fallback is None or candidate<fallback:fallback=candidate
            if candidate[1] not in previous and (best is None or candidate<best):best=candidate
    chosen=best if best is not None else fallback
    if chosen is None:raise ValueError('cannot complete fixed allocations')
    return chosen[1],best is None


def generate(probabilities):
    p=np.asarray(probabilities,dtype=float);counts=allocate_per_match(p)
    dist=draw_count_distribution(p);quotas,quota_error=draw_quotas(counts,dist)
    # Most probable draw-count strata first, not artificially sorted by absolute
    # ticket probability across different draw counts. Within a stratum, pick
    # the strongest currently feasible H/D/A combination.
    order=sorted((k for k,q in enumerate(quotas) for _ in range(q)),key=lambda k:(-dist[k],k))
    remaining=counts.copy();tickets=[];forced=[]
    for j,k in enumerate(order):
        ticket,duplicate=best_ticket(remaining,p,k,order[j+1:],set(tickets))
        tickets.append(ticket);forced.append(duplicate)
        remaining[np.arange(len(p)),np.asarray(ticket)]-=1
    tickets=np.asarray(tickets,dtype=np.int8)
    if np.any(remaining):raise AssertionError('unspent allocations')
    for s in range(3):np.testing.assert_array_equal((tickets==s).sum(axis=0),counts[:,s])
    logp=np.log(p[np.arange(len(p))[None,:],tickets]).sum(axis=1)
    conditional=logp-np.log(dist[np.asarray(order)])
    return tickets,dict(version='draw_count_conditioned_proposal_v1',randomness_used=False,
        probabilities_changed=False,counts_preserved=True,allocation_HDA=counts.tolist(),
        draw_count_distribution=dist.tolist(),draw_count_quotas=quotas.tolist(),
        draw_count_order=order,quota_squared_error=quota_error,
        expected_draw_matches=float(p[:,1].sum()),rounded_draw_cells=int(counts[:,1].sum()),
        joint_probabilities=np.exp(logp).tolist(),within_draw_count_probabilities=np.exp(conditional).tolist(),
        forced_duplicate_tickets=[i+1 for i,v in enumerate(forced) if v],
        independence_assumed=True,global_portfolio_optimum_guaranteed=False,
        ticket_order='draw_count_stratum_probability_then_greedy_ticket_strength')
