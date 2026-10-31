#!/usr/bin/env python3
"""Compare frozen prematch BuyPlans with and without the absence correction."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from build_next_buyplan import build_round
from common import ROOT, match_key
from run_next_buyplan import build_next_probabilities


ROUNDS = (1650, 1653, 1654, 1660, 1663, 1664)
OUT = ROOT / "data/reports/next_absence_simulation"
SIDES = ("home", "draw", "away")
RESULT_INDEX = {1: 0, 0: 1, 2: 2}


def evaluate_round(round_number: int) -> dict:
    source = ROOT / "data/eval/toto_rounds" / f"toto{round_number}"
    predictions = pd.read_csv(source / "snapshot/predictions.csv", encoding="utf-8-sig", low_memory=False).copy()
    actual = pd.read_csv(source / "actual_results.csv", encoding="utf-8-sig").sort_values("match_no")
    predictions["__key"] = [match_key(h, a) for h, a in zip(predictions.home_team, predictions.away_team)]
    if predictions["__key"].duplicated().any():
        raise ValueError(f"toto{round_number}: duplicate prediction match")
    indexed = predictions.set_index("__key")
    rows = []
    for match in actual.itertuples(index=False):
        key = match_key(match.home_team, match.away_team)
        if key not in indexed.index:
            raise ValueError(f"toto{round_number}: prediction missing for {key}")
        row = indexed.loc[key].copy()
        row["match_no"] = int(match.match_no)
        rows.append(row)
    if len(rows) != 13:
        raise ValueError(f"toto{round_number}: requires 13 matches")
    forecast = build_next_probabilities(pd.DataFrame(rows).reset_index(drop=True))
    actual_index = np.array([RESULT_INDEX[int(value)] for value in actual.result])
    actual_result = actual.result.to_numpy(dtype=int)
    mix = forecast.next_uncertainty_calibration_mix.to_numpy(dtype=float)
    after = forecast[[f"next_prob_{side}" for side in SIDES]].to_numpy(dtype=float)
    raw_before = forecast[[f"next_prob_pre_absence_{side}" for side in SIDES]].to_numpy(dtype=float)
    before = (1 - mix[:, None]) * raw_before + mix[:, None] / 3.0
    output = {"round": f"toto{round_number}", "matches": 13,
              "known_absence_home": int((~forecast.absence_input_missing_home).sum()),
              "known_absence_away": int((~forecast.absence_input_missing_away).sum()),
              "mean_absence_lambda_delta_home": float(forecast.absence_lambda_delta_home.mean()),
              "mean_absence_lambda_delta_away": float(forecast.absence_lambda_delta_away.mean())}
    for version, probabilities in (("before", before), ("after", after)):
        output[f"{version}_logloss"] = float(-np.log(probabilities[np.arange(13), actual_index]).mean())
        one_hot = np.eye(3)[actual_index]
        output[f"{version}_brier"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        frame = forecast.copy()
        for index, side in enumerate(SIDES):
            frame[f"next_prob_{side}"] = probabilities[:, index]
        plan, _ = build_round(frame, strategy="relative_draw_graded_v23")
        hits = [int(np.sum(plan[f"ticket{i:02d}"].to_numpy(dtype=int) == actual_result))
                for i in range(1, 11)]
        output[f"{version}_ticket_hits"] = hits
        output[f"{version}_ticket_mean_hits"] = float(np.mean(hits))
        output[f"{version}_ticket_max_hits"] = max(hits)
    return output


def main() -> None:
    results = [evaluate_round(number) for number in ROUNDS]
    report = {"method": "frozen prematch snapshots; both paths retain the same fatigue/weather correction and v2.3 BuyPlan generator",
              "correction": "absence_attack_defense_lambda_v1; own attack absence reduces own lambda, own defense absence raises opponent lambda; 0.25 goals per known impact unit, capped at impact 0.5",
              "rounds": results}
    for label, subset in (("earlier", results[:3]), ("later", results[3:]), ("all", results)):
        report[label] = {
            "matches": 13 * len(subset),
            **{f"{version}_{metric}": float(np.mean([row[f"{version}_{metric}"] for row in subset]))
               for version in ("before", "after")
               for metric in ("logloss", "brier", "ticket_mean_hits", "ticket_max_hits")},
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(results).to_csv(OUT / "rounds.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({key: report[key] for key in ("earlier", "later", "all")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
