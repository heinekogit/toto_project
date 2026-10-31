#!/usr/bin/env python3
"""Diagnostic H/A-closeness D boost sweep; never changes production forecasts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ROOT, sha256
from simulate_free_portfolio_1664 import top_joint_candidates


FROZEN = ROOT / "data/eval/toto_rounds/toto1664/snapshot/next_predictions.csv"
ACTUAL = ROOT / "data/eval/toto_rounds/toto1664/actual_results.csv"
HISTORY = ROOT / "data/eval/next_model/correction_history_v1/matches.csv"
OUT = ROOT / "data/reports/toto1664_draw_boost_sweep"
BETAS = tuple(round(i * 0.05, 2) for i in range(11))
SIDES = ("home", "draw", "away")
RESULT_INDEX = {1: 0, 0: 1, 2: 2}


def boost(probabilities: np.ndarray, beta: float) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("invalid H/D/A probabilities")
    if beta < 0:
        raise ValueError("this experiment only tests positive D boosts")
    closeness = 1 - np.abs(probabilities[:, 0] - probabilities[:, 2]) / (
        probabilities[:, 0] + probabilities[:, 2])
    result = probabilities.copy()
    result[:, 1] *= np.exp(beta * closeness)
    return result / result.sum(axis=1, keepdims=True)


def main() -> None:
    frozen = pd.read_csv(FROZEN, encoding="utf-8-sig").sort_values("match_no")
    actual = pd.read_csv(ACTUAL, encoding="utf-8-sig").sort_values("match_no")
    history = pd.read_csv(HISTORY, encoding="utf-8-sig")
    if not np.array_equal(frozen.match_no.to_numpy(), actual.match_no.to_numpy()):
        raise ValueError("frozen forecast and actual match numbers do not align")
    base = frozen[[f"next_prob_{side}" for side in SIDES]].to_numpy(dtype=float)
    truth = actual.result.map(RESULT_INDEX).to_numpy(dtype=int)
    rows = []
    for beta in BETAS:
        adjusted = boost(base, beta)
        tickets, masses = top_joint_candidates(adjusted, 10)
        hits = (tickets == truth).sum(axis=1)
        draw_counts = (tickets == 1).sum(axis=1)
        row = {
            "beta": beta,
            "draw_cells_in_10_tickets": int(draw_counts.sum()),
            "draws_per_ticket": draw_counts.tolist(),
            "actual_ticket_hits": hits.tolist(),
            "actual_mean_hits": float(hits.mean()),
            "actual_max_hits": int(hits.max()),
            "actual_tickets_ge7": int((hits >= 7).sum()),
            "boosted_model_prob_any_13": float(masses.sum()),
        }
        for split in ("fit", "validation"):
            subset = history.loc[history.split.eq(split)]
            model = subset[[f"next_prob_{side}" for side in SIDES]].to_numpy(dtype=float)
            outcomes = subset.actual_result.map(RESULT_INDEX).to_numpy(dtype=int)
            candidate = boost(model, beta)
            row[f"{split}_logloss"] = float(-np.log(candidate[np.arange(len(outcomes)), outcomes]).mean())
            row[f"{split}_predicted_draw_rate"] = float(candidate[:, 1].mean())
            row[f"{split}_actual_draw_rate"] = float((outcomes == 1).mean())
        rows.append(row)
    report = {
        "method": "experimental D odds multiplier exp(beta * (1 - abs(pH-pA)/(pH+pA))), then renormalize H/D/A; independent top-ten joint portfolio; no production changes",
        "frozen_forecast_sha256": sha256(FROZEN),
        "actual_results_sha256": sha256(ACTUAL),
        "history_sha256": sha256(HISTORY),
        "important": "Frozen toto1664 probabilities predate current fatigue/absence/weather correction; historical 49/39 comparison uses today's next_prob replay. No beta is selected using toto1664 results.",
        "actual_draw_matches": actual.loc[actual.result.eq(0), "match_no"].astype(int).tolist(),
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{key: value for key, value in row.items() if not isinstance(value, list)}
                  for row in rows]).to_csv(OUT / "sweep.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([{key: row[key] for key in ("beta", "draw_cells_in_10_tickets", "actual_mean_hits", "actual_max_hits", "fit_logloss", "validation_logloss")}
                        for row in rows]).to_string(index=False))


if __name__ == "__main__":
    main()
