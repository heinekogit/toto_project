"""Experimental ten-ticket portfolio with free per-match H/D/A counts.

The forecast probabilities are never changed. Selection uses only prematch
probabilities and fixed Monte Carlo scenarios; actual results are not inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from build_next_buyplan import top_joint_candidates


TOP_CANDIDATES = 3000
MODEL_SAMPLED_CANDIDATES = 5000
SCENARIOS = 5000
SEED = 230031
SYMBOLS = {"H": 0, "D": 1, "A": 2}
TO_STORED = np.array([1, 0, 2], dtype=np.uint8)


def build_shadow(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if len(frame) != 13 or frame["match_no"].astype(int).tolist() != list(range(1, 14)):
        raise ValueError("shadow C requires 13 matches in toto order")
    probabilities = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("invalid next H/D/A probabilities")

    # Combine the most likely full tickets with samples from the unchanged
    # H/D/A distribution. D appears only when the forecast assigns it mass.
    pool: dict[tuple[int, ...], None] = {}
    for item in top_joint_candidates(frame, limit=TOP_CANDIDATES):
        pool.setdefault(tuple(SYMBOLS[s] for s in item["symbols"]), None)
    candidate_rng = np.random.default_rng(SEED + 1)
    candidate_uniforms = candidate_rng.random((MODEL_SAMPLED_CANDIDATES, 13))
    sampled = (candidate_uniforms[:, :, None] > probabilities.cumsum(axis=1)[None, :, :]).sum(axis=2)
    for ticket in sampled:
        pool.setdefault(tuple(int(value) for value in ticket), None)
    candidates = np.asarray(list(pool), dtype=np.uint8)
    if len(candidates) < 10:
        raise ValueError("shadow C candidate frontier is too small")
    ticket_mass = np.prod(probabilities[np.arange(13)[None, :], candidates], axis=1)
    rng = np.random.default_rng(SEED)
    uniforms = rng.random((SCENARIOS, 13))
    outcomes = (uniforms[:, :, None] > probabilities.cumsum(axis=1)[None, :, :]).sum(axis=2).astype(np.uint8)
    hit_counts = np.zeros((len(candidates), SCENARIOS), dtype=np.uint8)
    for match in range(13):
        hit_counts += candidates[:, None, match] == outcomes[None, :, match]
    cover11 = hit_counts >= 11
    cover12 = hit_counts >= 12
    del hit_counts

    chosen: list[int] = []
    covered11 = np.zeros(SCENARIOS, dtype=bool)
    covered12 = np.zeros(SCENARIOS, dtype=bool)
    # Every candidate competes on the same forecast-based coverage score.
    # No ticket is required to contain a particular number of draws.
    for _ in range(10):
        eligible = np.ones(len(candidates), dtype=bool)
        eligible[chosen] = False
        if not eligible.any():
            raise ValueError("shadow C has no eligible ticket")
        # Marginal coverage under the shadow probability model. The
        # first-prize term gives joint probability a small explicit weight.
        score = ((cover11 & ~covered11).mean(axis=1)
                 + 4.0 * (cover12 & ~covered12).mean(axis=1)
                 + 12.0 * ticket_mass)
        score[~eligible] = -np.inf
        selected = int(np.argmax(score))
        chosen.append(selected)
        covered11 |= cover11[selected]
        covered12 |= cover12[selected]
    chosen.sort(key=lambda index: (-ticket_mass[index], index))
    tickets = candidates[chosen]
    result = frame.copy()
    for ticket_no, ticket in enumerate(tickets, start=31):
        result[f"ticket{ticket_no:02d}"] = TO_STORED[ticket]
    audit = {
        "generator": "free_coverage_shadow_c_v4",
        "status": "non-purchase shadow C",
        "version": "C: 採用数自由・Dカバー型",
        "ticket_count": 10,
        "probability_source": "current next_prob_* unchanged",
        "candidate_source": "top joint tickets plus draws sampled from unchanged next_prob_*",
        "top_joint_candidates": TOP_CANDIDATES,
        "model_sampled_candidates": MODEL_SAMPLED_CANDIDATES,
        "candidate_seed": SEED + 1,
        "candidate_count_after_dedup": len(candidates),
        "selection_scenarios": SCENARIOS,
        "selection_seed": SEED,
        "coverage_weights_11_12_13": [1.0, 4.0, 12.0],
        "draw_count_policy": "none; selected by modeled 11/12/13 coverage",
        "draws_per_ticket": (tickets == 1).sum(axis=1).tolist(),
        "sampled_prob_any_11": float(covered11.mean()),
        "sampled_prob_any_12": float(covered12.mean()),
    }
    return result, audit
