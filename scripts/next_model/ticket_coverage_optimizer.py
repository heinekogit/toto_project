"""Optimize >=10-hit coverage under fixed per-match allocations."""
import numpy as np

SEED=13010
OUTCOME_COUNT=12000
SWAP_STEPS=2000


def optimize_ten_hit_coverage(initial, probabilities):
    p=np.asarray(probabilities,dtype=float)
    rng=np.random.default_rng(SEED)
    outcomes=(np.random.default_rng(SEED+100).random((OUTCOME_COUNT,13,1)) > p.cumsum(axis=1)[None,:,:]).sum(axis=2)
    tickets=np.asarray(initial,dtype=int).copy()
    hits=(outcomes[:,None,:]==tickets[None,:,:]).sum(axis=2)
    def objective(arr,counts):
        return (int((counts.max(axis=1)>=10).sum()),float(np.prod(p[np.arange(13)[None,:],arr],axis=1).sum()))
    start=best=objective(tickets,hits)
    for _ in range(SWAP_STEPS):
        i,j=rng.choice(10,2,replace=False);col=int(rng.integers(13))
        a,b=tickets[i,col],tickets[j,col]
        if a==b:continue
        candidate=tickets.copy();candidate[i,col],candidate[j,col]=b,a
        same=(candidate[:,None,:]==candidate[None,:,:]).sum(axis=2)
        if (same[np.triu_indices(10,1)]>11).any():continue
        new=hits.copy()
        new[:,i]+=(outcomes[:,col]==b).astype(int)-(outcomes[:,col]==a)
        new[:,j]+=(outcomes[:,col]==a).astype(int)-(outcomes[:,col]==b)
        score=objective(candidate,new)
        if score>best:tickets,hits,best=candidate,new,score
    for symbol in range(3):
        np.testing.assert_array_equal((tickets==symbol).sum(axis=0),(np.asarray(initial)==symbol).sum(axis=0))
    same=(tickets[:,None,:]==tickets[None,:,:]).sum(axis=2)
    return tickets,dict(target="at_least_one_ticket_ge10",seed=SEED,outcome_count=OUTCOME_COUNT,
                        swap_steps=SWAP_STEPS,training_probability_before=start[0]/OUTCOME_COUNT,
                        training_probability_after=best[0]/OUTCOME_COUNT,
                        independent_match_assumption=True,actual_results_used=False,
                        minimum_pairwise_hamming_distance=int(13-same[np.triu_indices(10,1)].max()),
                        optimality_guaranteed=False)
