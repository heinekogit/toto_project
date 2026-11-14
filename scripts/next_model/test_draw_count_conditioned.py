import itertools
import math
import numpy as np
import unittest
from draw_count_conditioned import (
    best_ticket, draw_count_distribution, draw_quotas,
    feasible_draw_degrees, generate,
)


def test_distribution_matches_binomial_and_extreme_outcomes():
    p = np.tile([.4, .25, .35], (13, 1))
    expected = [math.comb(13, k)*.25**k*.75**(13-k) for k in range(14)]
    np.testing.assert_allclose(draw_count_distribution(p), expected)
    for symbol in range(3):
        p = np.zeros((13, 3)); p[:, symbol] = 1
        tickets, audit = generate(p)
        assert (tickets == symbol).all()
        assert audit['draw_count_quotas'][13 if symbol == 1 else 0] == 10
        assert len(audit['forced_duplicate_tickets']) == 9


def test_quota_dp_matches_exhaustive_feasible_optimum():
    counts = np.array([[1, 2, 1], [2, 1, 1], [2, 0, 2]])
    distribution = np.array([.2, .4, .3, .1])
    q, cost = draw_quotas(counts, distribution)
    feasible = []
    for columns in itertools.product(range(4), repeat=4):
        if feasible_draw_degrees(counts[:, 1], columns):
            quota = np.bincount(columns, minlength=4)
            feasible.append(float(((quota-4*distribution)**2).sum()))
    assert math.isclose(cost, min(feasible))
    assert q.sum() == 4 and q @ np.arange(4) == 3
    assert not feasible_draw_degrees([2, 0, 0], [2, 0])


def test_greedy_choice_matches_exhaustive_ticket_search():
    counts = np.array([[1, 1, 1], [2, 1, 0], [0, 1, 2]])
    p = np.array([[.5, .2, .3], [.7, .3, 0], [0, .4, .6]])
    previous = {(0, 1, 2)}
    ticket, duplicate = best_ticket(counts, p, 1, [1, 1], previous)
    options = []
    for t in itertools.product(range(3), repeat=3):
        if t.count(1) != 1 or t in previous: continue
        if any(counts[i, s] == 0 or p[i, s] == 0 for i, s in enumerate(t)): continue
        if not feasible_draw_degrees(counts[:, 1] - (np.array(t) == 1), [1, 1]): continue
        options.append((-sum(math.log(p[i, s]) for i, s in enumerate(t)), t))
    assert ticket == min(options)[1] and not duplicate


def test_reproducibility_marginals_and_invariant_average_hits():
    p = np.array([[.2+i*.015, .25, .55-i*.015] for i in range(13)])
    original = p.copy()
    a, audit = generate(p); b, audit2 = generate(p)
    np.testing.assert_array_equal(p, original)
    np.testing.assert_array_equal(a, b)
    assert audit == audit2
    counts = np.array(audit['allocation_HDA'])
    for s in range(3):
        np.testing.assert_array_equal((a == s).sum(axis=0), counts[:, s])
    for outcome in (np.zeros(13, dtype=int), np.ones(13, dtype=int), np.arange(13)%3):
        assert math.isclose((a == outcome).sum(axis=1).mean(), counts[np.arange(13), outcome].sum()/10)


def test_invalid_probabilities_rejected():
    with unittest.TestCase().assertRaises(ValueError): generate(np.ones((13, 3)))


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in list(globals().items()) if name.startswith("test_"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
