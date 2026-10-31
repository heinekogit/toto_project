"""BuyPlan v2.4: independent largest-remainder counts and remaining-count draw."""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np

SYMBOLS = "HDA"


def allocate_per_match(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (13, 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Requires 13 normalized H/D/A probability rows")
    counts = np.floor(p * 10 + 1e-10).astype(int)
    for row in range(13):
        expected = p[row] * 10
        remaining = 10 - int(counts[row].sum())
        if remaining < 0 or remaining > 3:
            raise ValueError("Invalid largest-remainder allocation")
        order = sorted(range(3), key=lambda i: (-(expected[i] - counts[row, i]), i))
        for symbol in order[:remaining]:
            counts[row, symbol] += 1
    if not (counts.sum(axis=1) == 10).all():
        raise AssertionError("Per-match allocation failed")
    return counts


def prediction_fingerprint(frame, round_id: str) -> str:
    ordered = frame.sort_values("match_no")
    columns = ("next_prob_home", "next_prob_draw", "next_prob_away")
    payload = {
        "version": "v2.4", "round_id": round_id,
        "matches": [
            {"match_no": int(row.match_no), "home_team": str(row.home_team),
             "away_team": str(row.away_team),
             "probabilities_HDA": [float(getattr(row, col)).hex() for col in columns]}
            for row in ordered.itertuples(index=False)
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate(probabilities: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    counts = allocate_per_match(probabilities)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    rng = random.Random(seed)
    tickets = np.empty((10, 13), dtype=np.int8)
    for match, allocation in enumerate(counts):
        remaining = allocation.astype(int).tolist()
        for candidate in range(10):
            roll = rng.randrange(10 - candidate)
            cumulative = 0
            for symbol, available in enumerate(remaining):
                cumulative += available
                if roll < cumulative:
                    tickets[candidate, match] = symbol
                    remaining[symbol] -= 1
                    break
        if remaining != [0, 0, 0]:
            raise AssertionError("Allocation not exhausted")
    for symbol in range(3):
        np.testing.assert_array_equal((tickets == symbol).sum(axis=0), counts[:, symbol])
    keys = [tuple(map(int, row)) for row in tickets]
    duplicates = []
    for left in range(10):
        for right in range(left + 1, 10):
            if keys[left] == keys[right]:
                duplicates.append([left + 1, right + 1])
    audit = {
        "generator": "random_remaining_v24", "seed": seed,
        "rng": "Python random.Random; randrange(remaining ticket count)",
        "allocation_method": "per_match_largest_remainder_HDA_tie_order",
        "allocation_HDA": counts.tolist(), "totals_HDA": counts.sum(axis=0).tolist(),
        "ticket_display_order": "draw_order_01_to_10", "randomness_used": True,
        "D_score_adjustment_used": False, "cost_adjustment_used": False,
        "duplicate_pairs": duplicates, "duplicate_count": len(duplicates),
        "unique_ticket_count": len(set(keys)),
    }
    return tickets, audit


def seed_from_fingerprint(fingerprint: str) -> int:
    return int(fingerprint, 16)
