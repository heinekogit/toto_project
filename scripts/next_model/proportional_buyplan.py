"""Allocate probability marginals first; arrange tickets without changing them."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def allocate_counts(probabilities, ticket_count=10):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (13, 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError('Requires 13 normalized H/D/A probability rows')
    expected = p * ticket_count
    base = np.floor(expected + 1e-10).astype(int)
    fractions = expected - base
    totals = expected.sum(axis=0)
    target = np.floor(totals + 1e-10).astype(int)
    for index in np.argsort(-(totals - target), kind='stable')[:13 * ticket_count - target.sum()]:
        target[index] += 1
    matrix = np.zeros((16, 39))
    for row in range(13):
        matrix[row, row*3:row*3+3] = 1
    for col in range(3):
        matrix[13+col, col::3] = 1
    required = np.r_[ticket_count-base.sum(axis=1), target-base.sum(axis=0)]
    result = milp((1-2*fractions).ravel(), integrality=np.ones(39),
                  bounds=Bounds(np.zeros(39), (fractions > 1e-9).astype(int).ravel()),
                  constraints=LinearConstraint(matrix, required, required))
    if not result.success:
        raise ValueError(f'Probability rounding failed: {result.message}')
    counts = base + np.rint(result.x).astype(int).reshape(13,3)
    if not (counts.sum(axis=1) == ticket_count).all() or not np.array_equal(counts.sum(axis=0), target):
        raise ValueError('Allocation invariant failed')
    return counts


def build_proportional_tickets(probabilities, seed=20261010, trials=2000, ticket_count=10, fixed_tickets=None):
    p = np.asarray(probabilities, dtype=float)
    counts = allocate_counts(p, ticket_count=ticket_count)
    fixed = np.empty((0, 13), dtype=int) if fixed_tickets is None else np.asarray(fixed_tickets, dtype=int)
    if fixed.ndim != 2 or fixed.shape[1] != 13 or not np.isin(fixed, [0,1,2]).all():
        raise ValueError('Invalid fixed tickets')
    rng = np.random.default_rng(seed)
    best, best_key = None, None
    for _ in range(trials):
        tickets = np.stack([rng.permutation(np.repeat(np.arange(3), row)) for row in counts], axis=1)
        combined = np.concatenate([fixed, tickets])
        same = (combined[:,None,:] == combined[None,:,:]).sum(axis=2)
        pairs = same[np.triu_indices(len(combined),1)]
        logs = np.log(np.maximum(p[np.arange(13)[None,:], tickets], 1e-300)).sum(axis=1)
        key = (int((pairs >= 12).sum()), int(pairs.max()), float(np.var((tickets==1).sum(axis=1))), float(np.var(logs)))
        if best_key is None or key < best_key:
            best, best_key = tickets.copy(), key
    # Impossibly concentrated inputs must be reported, never changed silently.
    combined = np.concatenate([fixed, best])
    pairs = (combined[:,None,:] == combined[None,:,:]).sum(axis=2)[np.triu_indices(len(combined),1)]
    audit = {'generator':'probability_allocation_v1', 'seed':seed, 'trials':trials,
             'allocation_HDA':counts.tolist(), 'totals_HDA':counts.sum(axis=0).tolist(),
             'expected_totals_HDA':(p*ticket_count).sum(axis=0).tolist(),
             'minimum_pairwise_hamming_distance':int(13-pairs.max()),
             'diversity_constraint_satisfied':bool(pairs.max()<=11),
             'probability_floor_used':False, 'fixed_draw_ratio_used':False}
    return best, audit
