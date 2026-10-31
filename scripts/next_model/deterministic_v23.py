"""Deterministic v2.3 ticket placement with fixed per-match H/D/A counts."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from anchor_arrangement import relative_costs
from proportional_buyplan import allocate_counts


def _row_candidates(counts: np.ndarray) -> np.ndarray:
    count_h, count_d, count_a = map(int, counts)
    ticket_count = count_h + count_d + count_a
    rows = []
    for home in combinations(range(ticket_count), count_h):
        remaining = tuple(i for i in range(ticket_count) if i not in home)
        for draw in combinations(remaining, count_d):
            row = np.full(ticket_count, 2, dtype=np.int8)
            row[list(home)] = 0
            row[list(draw)] = 1
            rows.append(row)
    return np.asarray(rows, dtype=np.int8)


def build_deterministic_tickets(probabilities: np.ndarray) -> tuple[np.ndarray, dict]:
    """Choose each match's placement by exhaustive, ordered comparison."""
    p = np.asarray(probabilities, dtype=float)
    counts = allocate_counts(p)
    ticket_count = int(counts.sum(axis=1)[0])
    pair_i, pair_j = np.triu_indices(ticket_count, 1)
    same = np.zeros(len(pair_i), dtype=np.int16)
    draw_totals = np.zeros(ticket_count, dtype=np.int16)
    log_totals = np.zeros(ticket_count, dtype=float)
    tickets = np.empty((ticket_count, len(counts)), dtype=np.int8)
    for match, allocation in enumerate(counts):
        options = _row_candidates(allocation)
        next_same = same[None, :] + (options[:, pair_i] == options[:, pair_j])
        next_draws = draw_totals[None, :] + (options == 1)
        next_logs = log_totals[None, :] + np.log(p[match, options])
        # Lexicographic objective: distinct tickets, then spread D and joint
        # probability. Candidate order breaks exact ties without randomness.
        keys = (
            np.var(next_logs, axis=1),
            np.var(next_draws, axis=1),
            np.sum(next_same.astype(np.int32) ** 2, axis=1),
            np.max(next_same, axis=1),
        )
        best = int(np.lexsort(keys)[0])
        tickets[:, match] = options[best]
        same = next_same[best]
        draw_totals = next_draws[best]
        log_totals = next_logs[best]
    minimum_distance = int(len(counts) - same.max())
    if minimum_distance < 2:
        raise ValueError("Deterministic v2.3 placement cannot satisfy minimum ticket distance")
    return tickets, {
        "generator": "deterministic_v23",
        "allocation_HDA": counts.tolist(),
        "totals_HDA": counts.sum(axis=0).tolist(),
        "expected_totals_HDA": (p * ticket_count).sum(axis=0).tolist(),
        "minimum_pairwise_hamming_distance": minimum_distance,
        "diversity_constraint_satisfied": True,
        "probability_floor_used": False,
        "fixed_draw_ratio_used": False,
        "randomness_used": False,
    }


def arrange_deterministically(initial: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, dict]:
    """Best improving pair swap until a local optimum; never change counts."""
    costs = relative_costs(scores)
    tickets = np.asarray(initial, dtype=np.int8).copy()
    n, matches = tickets.shape
    pairs = list(combinations(range(n), 2))
    target = np.linspace(.5, 1.5, n) * costs[np.arange(matches)[None, :], tickets].sum(axis=1).mean()

    def objective(arr):
        totals = costs[np.arange(matches)[None, :], arr].sum(axis=1)
        return float(np.mean((np.sort(totals) - target) ** 2))

    def distance(arr):
        return min(int(np.count_nonzero(arr[i] != arr[j])) for i, j in pairs)

    before = objective(tickets)
    current = before
    swaps = 0
    while True:
        best_value = current - 1e-12
        best_swap = None
        for match in range(matches):
            for i, j in pairs:
                if tickets[i, match] == tickets[j, match]:
                    continue
                tickets[i, match], tickets[j, match] = tickets[j, match], tickets[i, match]
                if distance(tickets) >= 2:
                    value = objective(tickets)
                    if value < best_value:
                        best_value, best_swap = value, (match, i, j)
                tickets[i, match], tickets[j, match] = tickets[j, match], tickets[i, match]
        if best_swap is None:
            break
        match, i, j = best_swap
        tickets[i, match], tickets[j, match] = tickets[j, match], tickets[i, match]
        current = best_value
        swaps += 1
    totals = costs[np.arange(matches)[None, :], tickets].sum(axis=1)
    tickets = tickets[np.argsort(totals, kind="stable")]
    totals = costs[np.arange(matches)[None, :], tickets].sum(axis=1)
    if any(not np.array_equal((tickets == symbol).sum(axis=0), (initial == symbol).sum(axis=0)) for symbol in range(3)):
        raise AssertionError("Allocation changed")
    return tickets, {
        "mode": "graded", "algorithm": "deterministic_best_improvement",
        "randomness_used": False, "accepted_swaps": swaps,
        "objective_before": before, "objective_after": current,
        "target_costs": target.tolist(), "ticket_costs": totals.tolist(),
        "anchor_probability_ratios": np.exp(-totals).tolist(),
        "anchor_symbols": np.argmax(scores, axis=1).tolist(),
        "minimum_pairwise_hamming_distance": distance(tickets),
        "counts_preserved": True, "global_optimum_guaranteed": False,
    }
