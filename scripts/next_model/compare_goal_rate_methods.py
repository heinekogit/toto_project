#!/usr/bin/env python3
"""Frozen-prematch comparison of unchanged, surprise-weighted, and recent goal rates."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT
from forecast_pipeline import build_next_probabilities
from goal_rate_adjustment import load_history, rates_for_match

sys.path.insert(0, str(ROOT / "scripts"))
from jleague_team_names import canonical_team_name

ROUNDS = (1660, 1663, 1664, 1665, 1668)
TRAIN = (1660, 1663, 1664)
TEST = (1665, 1668)
OUT = ROOT / "data/reports/goal_rate_methods"


def selected_source(round_no: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = ROOT / f"data/eval/toto_rounds/toto{round_no}"
    saved = pd.read_csv(root / "snapshot/next_predictions.csv", encoding="utf-8-sig", low_memory=False)
    full = pd.read_csv(root / "snapshot/predictions.csv", encoding="utf-8-sig", low_memory=False)
    source = full.set_index("match_id").loc[saved.match_id].reset_index()
    actual = pd.read_csv(root / "actual_results.csv", encoding="utf-8-sig")
    actual = actual.set_index("match_no").loc[saved.match_no].reset_index()
    return source, saved, actual


def alter_source(source: pd.DataFrame, method: str, parameter: float,
                 histories: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
    changed = source.copy()
    audits = []
    for i, row in source.iterrows():
        league = str(row.league)
        season = str(row.match_id).split("_")[1]
        history = histories.setdefault((league, season), load_history(league, season))
        kickoff = pd.Timestamp(row.datetime)
        record = {"match_id": row.match_id, "method": method, "parameter": parameter}
        for side in ("home", "away"):
            team = canonical_team_name(row[f"{side}_team"])
            for metric, kind in (("得点数", "for"), ("失点数", "against")):
                stats_col = f"stats_1試合平均{metric}_{side}"
                rates = rates_for_match(history, team, kickoff, kind, scale=parameter if method == "B" else 2.0)
                record[f"{side}_{kind}_n"] = rates["n"]
                record[f"{side}_{kind}_ordinary"] = rates["ordinary"]
                record[f"{side}_{kind}_recent5"] = rates["recent5"]
                record[f"{side}_{kind}_weighted"] = rates["weighted"]
                record[f"{side}_{kind}_downweighted"] = rates["downweighted"]
                if rates["n"] < 8 or not math.isfinite(rates["ordinary"]):
                    continue
                if method == "B":
                    delta = rates["weighted"] - rates["ordinary"]
                elif method == "C":
                    delta = parameter * (rates["recent5"] - rates["ordinary"])
                else:
                    delta = 0.0
                old = pd.to_numeric(row.get(stats_col), errors="coerce")
                if pd.notna(old):
                    changed.at[i, stats_col] = max(0.05, float(old) + delta)
                    record[f"{side}_{kind}_stats_old"] = float(old)
                    record[f"{side}_{kind}_stats_new"] = float(changed.at[i, stats_col])
        audits.append(record)
    return changed, audits


def score(predictions: pd.DataFrame, actual: pd.DataFrame) -> dict:
    probs = predictions[["next_prob_draw", "next_prob_home", "next_prob_away"]].to_numpy(float)
    result = actual.result.to_numpy(int)
    goals = actual[["home_score", "away_score"]].to_numpy(float)
    lambdas = predictions[["final_lambda_home", "final_lambda_away"]].to_numpy(float)
    rank = np.argsort(-probs, axis=1)
    return {"n": len(result), "top1": int((rank[:, 0] == result).sum()),
            "top2": int((rank[:, 1] == result).sum()), "top3": int((rank[:, 2] == result).sum()),
            "brier": float(np.mean(np.sum((probs - np.eye(3)[result]) ** 2, axis=1))),
            "log_loss": float(-np.log(np.clip(probs[np.arange(len(result)), result], 1e-12, 1)).mean()),
            "goal_mae": float(np.abs(lambdas - goals).mean())}


def main() -> None:
    histories: dict[tuple[str, str], pd.DataFrame] = {}
    records = []
    audits = []
    details = []
    for round_no in ROUNDS:
        source, saved, actual = selected_source(round_no)
        # Recompute all methods through the same current code; old saved forecasts
        # differ from current code for some earlier rounds.
        choices = [("A", 0.0, build_next_probabilities(source, goal_rate_adjustment=False), [])]
        for method, parameters in (("B", (1.5, 2.0, 2.5, 3.0)),
                                   ("C", (0.25, 0.5, 0.75))):
            for parameter in parameters:
                altered, audit = alter_source(source, method, parameter, histories)
                forecast = build_next_probabilities(altered, goal_rate_adjustment=False)
                choices.append((method, parameter, forecast, audit))
        for method, parameter, forecast, audit in choices:
            item = {"round": round_no, "method": method, "parameter": parameter, **score(forecast, actual)}
            records.append(item)
            for j, (_, row) in enumerate(forecast.iterrows()):
                details.append({"round": round_no, "match_no": int(saved.iloc[j].match_no),
                    "match_id": saved.iloc[j].match_id, "method": method, "parameter": parameter,
                    "actual_result": int(actual.iloc[j].result),
                    "actual_home_score": float(actual.iloc[j].home_score),
                    "actual_away_score": float(actual.iloc[j].away_score),
                    "lambda_home": float(row.final_lambda_home), "lambda_away": float(row.final_lambda_away),
                    **{f"prob_{side}": float(row[f"next_prob_{side}"]) for side in ("home", "draw", "away")},
                    "any_downweighted": any(value > 0 for key, value in audit[j].items()
                                            if key.endswith("_downweighted")) if audit else False})
            if audit:
                for row in audit:
                    row.update(round_no=round_no)
                audits.extend(audit)
    frame = pd.DataFrame(records)
    train = frame[frame["round"].isin(TRAIN)].groupby(["method", "parameter"], as_index=False).agg(
        n=("n", "sum"), brier_sum=("brier", lambda s: float(np.sum(s * 13))),
        log_loss_sum=("log_loss", lambda s: float(np.sum(s * 13))))
    train["brier"] = train.brier_sum / train.n
    selected = {"A": 0.0}
    for method in ("B", "C"):
        selected[method] = float(train[train.method == method].sort_values("log_loss_sum").iloc[0].parameter)
    summary = {}
    for split, rounds in (("train", TRAIN), ("test", TEST)):
        summary[split] = {}
        for method, parameter in selected.items():
            part = frame[(frame["round"].isin(rounds)) & (frame.method == method) &
                         (frame.parameter == parameter)]
            n = int(part.n.sum())
            summary[split][method] = {"parameter": parameter, "n": n,
                "top1": int(part.top1.sum()), "top2": int(part.top2.sum()), "top3": int(part.top3.sum()),
                **{k: float(np.average(part[k], weights=part.n)) for k in ("brier", "log_loss", "goal_mae")}}
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "round_metrics.csv", index=False)
    detail = pd.DataFrame(details)
    detail.to_csv(OUT / "match_predictions.csv", index=False)
    selected_detail = detail.loc[detail.apply(
        lambda row: row.parameter == selected[row.method], axis=1)].copy()
    symbols = np.array(["D", "H", "A"])
    probabilities = selected_detail[["prob_draw", "prob_home", "prob_away"]].to_numpy(float)
    outcomes = selected_detail.actual_result.to_numpy(int)
    selected_detail["actual_symbol"] = symbols[outcomes]
    selected_detail["predicted_symbol"] = symbols[probabilities.argmax(axis=1)]
    selected_detail["actual_probability"] = probabilities[np.arange(len(outcomes)), outcomes]
    selected_detail["match_brier"] = np.sum((probabilities - np.eye(3)[outcomes]) ** 2, axis=1)
    selected_detail.to_csv(OUT / "probabilities_vs_actual.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "rate_audit.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"train_rounds": TRAIN, "test_rounds": TEST,
        "selected_parameters": selected, "metrics": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_parameters": selected, "metrics": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
