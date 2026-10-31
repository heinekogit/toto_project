"""Rearrange fixed H/D/A allocations using relative probability costs.

Internal symbols: H=0, D=1, A=2. No actual results are accepted.
"""
import numpy as np


def relative_costs(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Requires finite, strictly positive normalized H/D/A probabilities')
    return -np.log(p / p.max(axis=1, keepdims=True))


def arrange_from_anchor(initial, probabilities, mode='balanced', steps=20000, seed=20261017):
    """Greedy within-match swaps; preserves every match's symbol counts.

    balanced targets equal total costs. graded targets 0.5..1.5 times mean
    cost (fixed design hypothesis). Neither forces an Anchor ticket. A minimum
    pairwise Hamming distance of two is required; infeasibility is not relaxed.
    """
    costs = relative_costs(probabilities)
    raw = np.asarray(initial)
    if raw.ndim != 2 or raw.shape[1] != len(costs) or len(raw) < 2 or not np.isin(raw, [0, 1, 2]).all():
        raise ValueError('Invalid ticket matrix')
    if mode not in {'balanced', 'graded'} or not isinstance(steps, int) or steps < 0:
        raise ValueError('Invalid mode or steps')
    t = raw.astype(int).copy()
    pairs = np.triu_indices(len(t), 1)
    def distance(arr):
        return int((arr[:, None, :] != arr[None, :, :]).sum(axis=2)[pairs].min())
    if distance(t) < 2:
        raise ValueError('Initial tickets must differ in at least two matches')
    cols = np.arange(len(costs))
    totals = costs[cols, t].sum(axis=1)
    target = np.full(len(t), totals.mean()) if mode == 'balanced' else np.linspace(.5, 1.5, len(t)) * totals.mean()
    def objective(values):
        return float(np.mean((np.sort(values) - target)**2))
    before = objective(totals)
    best = before
    accepted = 0
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i, j = rng.choice(len(t), 2, replace=False)
        col = int(rng.integers(len(costs)))
        a, b = t[i, col], t[j, col]
        if a == b:
            continue
        delta = costs[col, b] - costs[col, a]
        new_totals = totals.copy()
        new_totals[i] += delta
        new_totals[j] -= delta
        value = objective(new_totals)
        if value >= best - 1e-12:
            continue
        t[i, col], t[j, col] = b, a
        if distance(t) < 2:
            t[i, col], t[j, col] = a, b
            continue
        totals, best = new_totals, value
        accepted += 1
    t = t[np.argsort(costs[cols, t].sum(axis=1), kind='stable')]
    if any(not np.array_equal((t == s).sum(axis=0), (raw == s).sum(axis=0)) for s in range(3)):
        raise AssertionError('Allocation changed')
    totals = costs[cols, t].sum(axis=1)
    return t, dict(mode=mode, steps=steps, seed=seed, accepted_swaps=accepted,
                   objective_before=before, objective_after=best, target_costs=target.tolist(),
                   ticket_costs=totals.tolist(), anchor_probability_ratios=np.exp(-totals).tolist(),
                   anchor_symbols=np.argmax(probabilities, axis=1).tolist(),
                   minimum_pairwise_hamming_distance=distance(t), counts_preserved=True,
                   global_optimum_guaranteed=False)


def rearrange_buyplan(plan, mode='balanced', **kwargs):
    """Adapter for BuyPlan DataFrames (official result symbols H=1,D=0,A=2).

    Returns a copy plus audit; never writes files or changes probability/counts.
    Input row order is preserved, so callers may use their existing match order.
    """
    columns = [f'ticket{i:02d}' for i in range(1, 11)]
    if len(plan) != 13 or set(plan['match_no']) != set(range(1,14)):
        raise ValueError('Requires 13 distinct match numbers')
    values = plan[columns].to_numpy().T
    if not np.isin(values, [0,1,2]).all():
        raise ValueError('Invalid BuyPlan symbols')
    p = plan[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
    tickets, audit = arrange_from_anchor(np.array([1,0,2])[values.astype(int)],p,mode,**kwargs)
    result = plan.copy(deep=True)
    result[columns] = np.array([1,0,2])[tickets.T]
    return result, audit
