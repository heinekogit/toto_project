"""Compare v2.2 with a draw-spread arrangement without changing production."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from proportional_buyplan import build_proportional_tickets
from ticket_coverage_optimizer import OUTCOME_COUNT, SEED, SWAP_STEPS, optimize_ten_hit_coverage


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/reports/draw_balance_simulation_v1"
SOURCE = ROOT / "data/reports/next_probability_calibration/matches.csv"
SYMBOLS = np.array([1, 0, 2])


def draw_spread_limit(tickets, square_slack):
    total = int((tickets == 1).sum())
    base, remainder = divmod(total, len(tickets))
    minimum_square_sum = (len(tickets) - remainder) * base**2 + remainder * (base + 1) ** 2
    return minimum_square_sum + square_slack


def is_valid(tickets, square_limit, maximum_draws):
    draws = (tickets == 1).sum(axis=1)
    if draws.min() < 1 or draws.max() > maximum_draws or int(np.sum(draws**2)) > square_limit:
        return False
    shared = (tickets[:, None, :] == tickets[None, :, :]).sum(axis=2)
    return bool((shared[np.triu_indices(len(tickets), 1)] <= 11).all())


def optimize_balanced(initial, probabilities, maximum_draws, square_slack):
    """Use the v2.2 swaps and objective, rejecting concentrated draw profiles."""
    p = np.asarray(probabilities, dtype=float)
    tickets = np.asarray(initial, dtype=int).copy()
    square_limit = draw_spread_limit(tickets, square_slack)
    if not is_valid(tickets, square_limit, maximum_draws):
        raise ValueError("Initial arrangement fails the draw-spread constraints")
    rng = np.random.default_rng(SEED)
    outcomes = (
        np.random.default_rng(SEED + 100).random((OUTCOME_COUNT, 13, 1))
        > p.cumsum(axis=1)[None, :, :]
    ).sum(axis=2)
    hits = (outcomes[:, None, :] == tickets[None, :, :]).sum(axis=2)

    def score(arrangement, counts):
        exact = np.prod(p[np.arange(13)[None, :], arrangement], axis=1).sum()
        return int((counts.max(axis=1) >= 10).sum()), float(exact)

    best = score(tickets, hits)
    for _ in range(SWAP_STEPS):
        i, j = rng.choice(10, 2, replace=False)
        col = int(rng.integers(13))
        a, b = tickets[i, col], tickets[j, col]
        if a == b:
            continue
        candidate = tickets.copy()
        candidate[i, col], candidate[j, col] = b, a
        if not is_valid(candidate, square_limit, maximum_draws):
            continue
        new_hits = hits.copy()
        new_hits[:, i] += (outcomes[:, col] == b).astype(int) - (outcomes[:, col] == a)
        new_hits[:, j] += (outcomes[:, col] == a).astype(int) - (outcomes[:, col] == b)
        value = score(candidate, new_hits)
        if value > best:
            tickets, hits, best = candidate, new_hits, value
    return tickets


def exact_portfolio_probabilities(tickets_by_strategy, probabilities):
    """Enumerate all 3**13 results, so evaluation has no sampling error."""
    totals = {name: np.zeros(4) for name in tickets_by_strategy}
    count = 3**13
    for start in range(0, count, 100_000):
        ids = np.arange(start, min(start + 100_000, count), dtype=np.int64)
        results = np.empty((len(ids), 13), dtype=np.int8)
        for col in range(12, -1, -1):
            results[:, col] = ids % 3
            ids //= 3
        weights = np.prod(probabilities[np.arange(13)[None, :], results], axis=1)
        for name, tickets in tickets_by_strategy.items():
            best_hits = (results[:, None, :] == tickets[None, :, :]).sum(axis=2).max(axis=1)
            for index, threshold in enumerate((10, 11, 12, 13)):
                totals[name][index] += weights[best_hits >= threshold].sum()
    return totals


def summarize(round_id, probabilities, actual=None, prematch=None):
    initial, _ = build_proportional_tickets(probabilities)
    plans = {
        "v2.2": optimize_ten_hit_coverage(initial, probabilities)[0],
        "draw_spread_strict": optimize_balanced(initial, probabilities, 6, 20),
        "draw_spread_mild": optimize_balanced(initial, probabilities, 7, 40),
    }
    expected = [(initial == symbol).sum(axis=0) for symbol in range(3)]
    for name, tickets in plans.items():
        for symbol in range(3):
            np.testing.assert_array_equal((tickets == symbol).sum(axis=0), expected[symbol])
        shared = (tickets[:, None, :] == tickets[None, :, :]).sum(axis=2)
        if (shared[np.triu_indices(10, 1)] > 11).any():
            raise ValueError(f"{round_id}: {name} violates ticket diversity")
    probabilities_exact = exact_portfolio_probabilities(plans, probabilities)
    rows = []
    for name, tickets in plans.items():
        plan = pd.DataFrame({"match_no": np.arange(1, 14)})
        for i in range(10):
            plan[f"ticket{i + 1:02d}"] = SYMBOLS[tickets[i]]
        plan.to_csv(OUT / f"{round_id}_{name.replace('.', '')}.csv", index=False)
        draw_counts = (tickets == 1).sum(axis=1)
        result = {
            "round_id": round_id,
            "prematch_snapshot_proven": prematch,
            "strategy": name,
            "draw_counts": json.dumps(sorted(draw_counts.tolist())),
            "draw_square_sum": int(np.sum(draw_counts**2)),
            "zero_h_tickets": int(((tickets == 0).sum(axis=1) == 0).sum()),
            "zero_d_tickets": int((draw_counts == 0).sum()),
            "zero_a_tickets": int(((tickets == 2).sum(axis=1) == 0).sum()),
            "model_p_ge10": float(probabilities_exact[name][0]),
            "model_p_ge11": float(probabilities_exact[name][1]),
            "model_p_ge12": float(probabilities_exact[name][2]),
            "model_p_13": float(probabilities_exact[name][3]),
        }
        if actual is not None:
            result["actual_best_hits"] = int((tickets == actual).sum(axis=1).max())
        rows.append(result)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    rows = []
    for round_id, frame in data.groupby("round_id", sort=True):
        if len(frame) != 13 or not frame.prematch_snapshot_proven.all():
            continue
        frame = frame.sort_values("match_no")
        p = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
        actual = frame.actual_result.map({1: 0, 0: 1, 2: 2}).to_numpy(int)
        rows.extend(summarize(round_id, p, actual, True))
        print(round_id, "done", flush=True)

    current = pd.read_csv(ROOT / "data/purchase_reference/buyplan.csv").sort_values("match_no")
    p = current[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
    rows.extend(summarize("toto1663", p, None, True))
    official = current[[f"ticket{i:02d}" for i in range(1, 11)]].to_numpy(int).T
    official = np.select([official == 1, official == 0, official == 2], [0, 1, 2])
    generated = pd.read_csv(OUT / "toto1663_v22.csv")[[f"ticket{i:02d}" for i in range(1, 11)]].to_numpy(int).T
    generated = np.select([generated == 1, generated == 0, generated == 2], [0, 1, 2])
    if not np.array_equal(official, generated):
        raise ValueError("Failed to reproduce the current v2.2 plan")
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "current_buyplan_sha256": hashlib.sha256((ROOT / "data/purchase_reference/buyplan.csv").read_bytes()).hexdigest(),
        "strategies": {
            "v2.2": "production algorithm reproduced from fixed per-match allocations",
            "draw_spread_strict": "D per ticket 1..6; draw-count square sum at most balanced minimum + 20",
            "draw_spread_mild": "D per ticket 1..7; draw-count square sum at most balanced minimum + 40",
        },
        "optimization_outcomes": OUTCOME_COUNT,
        "swap_steps": SWAP_STEPS,
        "optimization_seed": SEED,
        "evaluation": "exact enumeration of all 3^13 outcome combinations under independent match probabilities",
        "actual_results_used_in_optimization": False,
        "production_changed": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["round_id", "strategy", "draw_counts", "model_p_ge10", "model_p_13", "actual_best_hits"]].to_string(index=False))


if __name__ == "__main__":
    main()
