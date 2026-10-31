#!/usr/bin/env python3
"""Compare frozen prematch score-model forecasts before/after uncertainty correction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_next_buyplan import build_round
from common import ROOT, match_key
from run_next_buyplan import build_next_probabilities


ROUNDS = (1650, 1653, 1654, 1660, 1663, 1664)
OUT = ROOT / "data/reports/next_uncertainty_simulation"
SIDES = ("home", "draw", "away")
RESULT_INDEX = {1: 0, 0: 1, 2: 2}


def evaluate_round(round_number: int) -> dict:
    source = ROOT / "data/eval/toto_rounds" / f"toto{round_number}"
    predictions = pd.read_csv(source / "snapshot/predictions.csv", encoding="utf-8-sig", low_memory=False)
    actual = pd.read_csv(source / "actual_results.csv", encoding="utf-8-sig")
    predictions = predictions.copy()
    predictions["__key"] = [match_key(h, a) for h, a in zip(predictions.home_team, predictions.away_team)]
    if predictions["__key"].duplicated().any():
        raise ValueError(f"toto{round_number}: duplicate prediction match")
    indexed = predictions.set_index("__key")
    rows = []
    for match in actual.sort_values("match_no").itertuples(index=False):
        key = match_key(match.home_team, match.away_team)
        if key not in indexed.index:
            raise ValueError(f"toto{round_number}: prediction missing for {key}")
        row = indexed.loc[key].copy()
        row["match_no"] = int(match.match_no)
        rows.append(row)
    if len(rows) != 13:
        raise ValueError(f"toto{round_number}: requires 13 matches")
    forecast = build_next_probabilities(pd.DataFrame(rows).reset_index(drop=True))
    actual_index = np.array([RESULT_INDEX[int(value)] for value in actual.sort_values("match_no").result])
    actual_result = actual.sort_values("match_no").result.to_numpy(dtype=int)
    output = {"round": f"toto{round_number}", "matches": 13,
              "mean_uncertainty_mix": float(forecast.next_uncertainty_calibration_mix.mean()),
              "weather_known_matches": int(forecast.next_uncertainty_weather_known.sum())}
    for version, prefix in (("before", "next_prob_pre_uncertainty"), ("after", "next_prob")):
        probabilities = forecast[[f"{prefix}_{side}" for side in SIDES]].to_numpy(dtype=float)
        output[f"{version}_logloss"] = float(-np.log(probabilities[np.arange(13), actual_index]).mean())
        one_hot = np.eye(3)[actual_index]
        output[f"{version}_brier"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        frame = forecast.copy()
        for side in SIDES:
            frame[f"next_prob_{side}"] = frame[f"{prefix}_{side}"]
        plan, _ = build_round(frame, strategy="relative_draw_graded_v23")
        hits = [int(np.sum(plan[f"ticket{i:02d}"].to_numpy(dtype=int) == actual_result))
                for i in range(1, 11)]
        output[f"{version}_ticket_hits"] = hits
        output[f"{version}_ticket_mean_hits"] = float(np.mean(hits))
        output[f"{version}_ticket_max_hits"] = max(hits)
    return output


def main() -> None:
    results = [evaluate_round(number) for number in ROUNDS]
    report = {"method": "frozen prematch snapshots; same v2.3 BuyPlan generator and actual results",
              "calibration": "fatigue_weather_uniform_v1; weights from legacy forecast (candidate transfer)",
              "rounds": results}
    for label, subset in (("fit", results[:3]), ("heldout", results[3:]), ("all", results)):
        report[label] = {
            "matches": 13 * len(subset),
            **{f"{version}_{metric}": float(np.mean([row[f"{version}_{metric}"] for row in subset]))
               for version in ("before", "after")
               for metric in ("logloss", "brier", "ticket_mean_hits", "ticket_max_hits")},
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(results).to_csv(OUT / "rounds.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({key: report[key] for key in ("fit", "heldout", "all")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
