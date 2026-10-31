#!/usr/bin/env python3
"""Compare free-marginal ten-ticket portfolios on frozen toto1664 forecasts.

All selection uses frozen probabilities and deterministic model-drawn scenarios.
Actual results are loaded only after both experimental portfolios are fixed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, sha256


SOURCE = ROOT / "data/eval/toto_rounds/toto1664"
OUT = ROOT / "data/reports/toto1664_free_portfolio"
SIDES = ("home", "draw", "away")
STORED_TO_INTERNAL = np.array([1, 0, 2])  # stored result 1=H, 0=D, 2=A
INTERNAL_TO_STORED = np.array([1, 0, 2])
CANDIDATE_COUNT = 5000
SCENARIOS = 20000
FIRST_PRIZE_FLOOR = 0.8
SEED = 1664001


def top_joint_candidates(probabilities: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k Cartesian products under independent match probabilities."""
    prefixes = [(1.0, ())]
    for row in probabilities:
        expanded = [(mass * float(row[symbol]), ticket + (symbol,))
                    for mass, ticket in prefixes for symbol in range(3)]
        expanded.sort(key=lambda item: (-item[0], item[1]))
        prefixes = expanded[:count]
    return np.asarray([ticket for _, ticket in prefixes], dtype=np.uint8), np.asarray([mass for mass, _ in prefixes])


def model_scenarios(probabilities: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    uniform = rng.random((SCENARIOS, 13))
    return (uniform[:, :, None] > probabilities.cumsum(axis=1)[None, :, :]).sum(axis=2).astype(np.uint8)


def packed_hit_masks(candidates: np.ndarray, outcomes: np.ndarray) -> tuple[list[int], list[int]]:
    eleven, twelve = [], []
    for ticket in candidates:
        hits = (outcomes == ticket).sum(axis=1)
        eleven.append(int.from_bytes(np.packbits(hits >= 11).tobytes(), "big"))
        twelve.append(int.from_bytes(np.packbits(hits >= 12).tobytes(), "big"))
    return eleven, twelve


def coverage_search(candidates: np.ndarray, masses: np.ndarray, probabilities: np.ndarray) -> list[int]:
    scenarios = model_scenarios(probabilities)
    eleven, twelve = packed_hit_masks(candidates, scenarios)
    selected = list(range(10))
    minimum_mass = FIRST_PRIZE_FLOOR * float(masses[:10].sum())
    for _ in range(20):
        chosen_mass = float(masses[selected].sum())
        current_11 = 0
        current_12 = 0
        for index in selected:
            current_11 |= eleven[index]
            current_12 |= twelve[index]
        current_score = current_11.bit_count() + current_12.bit_count()
        best_score = current_score
        best_swap = None
        chosen = set(selected)
        for remove_position, old in enumerate(selected):
            retained_11 = 0
            retained_12 = 0
            for index in selected:
                if index != old:
                    retained_11 |= eleven[index]
                    retained_12 |= twelve[index]
            for new in range(len(candidates)):
                if new in chosen or chosen_mass - masses[old] + masses[new] < minimum_mass:
                    continue
                score = (retained_11 | eleven[new]).bit_count() + (retained_12 | twelve[new]).bit_count()
                if score > best_score:
                    best_score = score
                    best_swap = (remove_position, new)
        if best_swap is None:
            break
        selected[best_swap[0]] = best_swap[1]
    return sorted(selected, key=lambda index: (-masses[index], index))


def exact_model_coverage(portfolios: dict[str, np.ndarray], probabilities: np.ndarray) -> dict:
    totals = {name: np.zeros(3, dtype=float) for name in portfolios}
    for start in range(0, 3**13, 50000):
        stop = min(start + 50000, 3**13)
        code = np.arange(start, stop, dtype=np.int64)
        outcomes = np.empty((len(code), 13), dtype=np.uint8)
        for column in range(12, -1, -1):
            outcomes[:, column] = code % 3
            code //= 3
        weights = np.prod(probabilities[np.arange(13)[None, :], outcomes], axis=1)
        for name, tickets in portfolios.items():
            best_hits = np.zeros(len(outcomes), dtype=np.uint8)
            for ticket in tickets:
                best_hits = np.maximum(best_hits, (outcomes == ticket).sum(axis=1))
            totals[name] += np.array([weights[best_hits >= threshold].sum() for threshold in (13, 12, 11)])
    return {name: {f"prob_any_{threshold}": float(value)
                   for threshold, value in zip((13, 12, 11), total)}
            for name, total in totals.items()}


def stored_portfolio(path: Path, first_ticket: int) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig").sort_values("match_no")
    values = frame[[f"ticket{i:02d}" for i in range(first_ticket, first_ticket + 10)]].to_numpy(dtype=int).T
    if values.shape != (10, 13) or not np.isin(values, [0, 1, 2]).all():
        raise ValueError(f"invalid frozen portfolio: {path}")
    return STORED_TO_INTERNAL[values]


def main() -> None:
    snapshot = SOURCE / "snapshot"
    forecast_path = snapshot / "next_predictions.csv"
    forecast = pd.read_csv(forecast_path, encoding="utf-8-sig", low_memory=False).sort_values("match_no").reset_index(drop=True)
    probabilities = forecast[[f"next_prob_{side}" for side in SIDES]].to_numpy(dtype=float)
    if probabilities.shape != (13, 3) or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("invalid frozen next probabilities")
    candidates, masses = top_joint_candidates(probabilities, CANDIDATE_COUNT)
    chosen = coverage_search(candidates, masses, probabilities)
    portfolios = {
        "official_v23": stored_portfolio(snapshot / "buyplan.csv", 1),
        "shadow_A": stored_portfolio(snapshot / "shadow_buyplan.csv", 11),
        "shadow_B": stored_portfolio(snapshot / "shadow_buyplan_b.csv", 21),
        "top10_joint": candidates[:10],
        "free_coverage_80pct_first": candidates[chosen],
    }
    model_coverage = exact_model_coverage(portfolios, probabilities)
    # Outcomes are not read until both experimental portfolios are finalized.
    actual_path = SOURCE / "actual_results.csv"
    actual = pd.read_csv(actual_path, encoding="utf-8-sig").sort_values("match_no")
    result = STORED_TO_INTERNAL[actual.result.to_numpy(dtype=int)]
    report = {
        "round_id": "toto1664",
        "method": "frozen next_prob probabilities; exact independent-match model coverage; free coverage selected from top 5000 joint tickets with 20,000 deterministic model-drawn scenarios, objective P(any >=11)+P(any >=12), exact first-prize probability >=80% of top10; actual results used only to score",
        "forecast_sha256": sha256(forecast_path),
        "actual_results_sha256": sha256(actual_path),
        "candidate_count": CANDIDATE_COUNT, "selection_scenarios": SCENARIOS,
        "seed": SEED, "first_prize_floor_fraction": FIRST_PRIZE_FLOOR,
        "caution": "These are frozen toto1664 probabilities, not recalculated current fatigue/absence/weather probabilities.",
        "portfolios": {},
    }
    rows = []
    for name, tickets in portfolios.items():
        hits = (tickets == result).sum(axis=1)
        ticket_masses = np.prod(probabilities[np.arange(13)[None, :], tickets], axis=1)
        counts = np.stack([(tickets == symbol).sum(axis=0) for symbol in range(3)], axis=1)
        distances = (tickets[:, None, :] != tickets[None, :, :]).sum(axis=2)
        report["portfolios"][name] = {
            **model_coverage[name],
            "actual_ticket_hits": hits.tolist(),
            "actual_mean_hits": float(hits.mean()),
            "actual_max_hits": int(hits.max()),
            "ticket_joint_probabilities": ticket_masses.tolist(),
            "draws_per_ticket": (tickets == 1).sum(axis=1).tolist(),
            "minimum_pairwise_distance": int(distances[np.triu_indices(10, 1)].min()),
            "per_match_HDA_counts": counts.tolist(),
        }
        for match_index, source in forecast.iterrows():
            rows.append({"portfolio": name, "match_no": int(source.match_no),
                         "home_team": source.home_team, "away_team": source.away_team,
                         **{f"ticket{i+1:02d}": int(INTERNAL_TO_STORED[tickets[i, match_index]]) for i in range(10)}})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "portfolios.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({name: {key: entry[key] for key in ("prob_any_13", "prob_any_12", "prob_any_11", "actual_mean_hits", "actual_max_hits", "draws_per_ticket")}
                      for name, entry in report["portfolios"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
