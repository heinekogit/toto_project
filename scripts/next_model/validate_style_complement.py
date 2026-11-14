#!/usr/bin/env python3
"""Exploratory, prematch-only test of Football LAB style as an xG complement."""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import ROOT
from expected_goals import base_lambdas_v1, score_distribution
from lab_matchup import apply_lab_matchup


ROUNDS = ("toto1649", "toto1650", "toto1653", "toto1654", "toto1663", "toto1668")
STYLES = ("short_counter", "long_counter", "enemy_possession", "own_possession",
          "left_attack", "center_attack", "right_attack")
ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
OUT = ROOT / "data/reports/style_complement_validation"


def norm(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def load_round(round_id: str) -> pd.DataFrame:
    base = ROOT / "data/eval/toto_rounds" / round_id
    predictions = pd.read_csv(base / "snapshot/predictions.csv", low_memory=False)
    actual = pd.read_csv(base / "actual_results.csv")
    predictions["key"] = [norm(h) + "|" + norm(a) for h, a in zip(predictions.home_team, predictions.away_team)]
    actual["key"] = [norm(h) + "|" + norm(a) for h, a in zip(actual.home_team, actual.away_team)]
    if predictions.key.duplicated().any():
        raise ValueError(f"{round_id}: duplicate prediction pairs")
    merged = actual[["key", "match_no", "home_score", "away_score", "result"]].merge(
        predictions, on="key", suffixes=("_actual", ""), validate="one_to_one")
    if merged.empty:
        raise ValueError(f"{round_id}: no matched predictions")
    if round_id == "toto1668":
        # This snapshot was sealed at 20:15 on 31 Oct, after the 31 Oct matches.
        merged = merged[pd.to_datetime(merged.datetime) > pd.Timestamp("2026-10-31 20:15:18")]
    rows = []
    for _, row in merged.iterrows():
        bh, ba, _ = base_lambdas_v1(row)
        lab = apply_lab_matchup(row, bh, ba)
        own = {"home": lab["final_lambda_home"], "away": lab["final_lambda_away"]}
        for side in ("home", "away"):
            opponent = "away" if side == "home" else "home"
            features = [float(row[f"flab_style_{style}_index_{side}"]) for style in STYLES]
            features += [float(row[f"flab_style_{style}_index_{opponent}"]) for style in STYLES]
            features += [float(row[f"flab_style_{style}_shot_rate_{side}"]) for style in STYLES]
            features += [float(row[f"flab_style_{style}_shot_rate_{opponent}"]) for style in STYLES]
            rows.append({"round_id": round_id, "match_no": int(row.match_no), "side": side,
                         "home_team": norm(row.home_team), "away_team": norm(row.away_team),
                         "actual_goals": int(row[f"{side}_score_actual"]),
                         "actual_result": int(row.result), "base_lambda": own[side],
                         **{f"f{i}": value for i, value in enumerate(features)}})
    return pd.DataFrame(rows)


def fit(train: pd.DataFrame, alpha: float | None, feature_count: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = [f"f{i}" for i in range(feature_count)]
    features = train[columns].to_numpy(float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x = (features - mean) / scale
    if alpha is None:
        intercept = math.log(train.actual_goals.sum() / train.base_lambda.sum())
        return np.array([intercept]), mean, scale
    y = train.actual_goals.to_numpy(float)
    offset = np.log(train.base_lambda.to_numpy(float))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = np.clip(offset + beta[0] + x @ beta[1:], -8, 8)
        rate = np.exp(linear)
        loss = np.mean(rate - y * linear) + alpha * np.dot(beta[1:], beta[1:]) / 2
        gradient = np.r_[np.mean(rate - y), x.T @ (rate - y) / len(y) + alpha * beta[1:]]
        return float(loss), gradient

    result = minimize(objective, np.zeros(feature_count + 1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, mean, scale


def predict(frame: pd.DataFrame, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    beta, mean, scale = fitted
    features = frame[[f"f{i}" for i in range(len(mean))]].to_numpy(float)
    adjustment = beta[0] + (features - mean) / scale @ beta[1:] if len(beta) > 1 else beta[0]
    return np.clip(frame.base_lambda.to_numpy(float) * np.exp(adjustment), 0.15, 4.5)


def assess(frame: pd.DataFrame, lambdas: np.ndarray) -> tuple[dict, list[dict]]:
    detail = frame[["round_id", "match_no", "side", "home_team", "away_team",
                    "actual_goals", "actual_result"]].copy()
    detail["lambda"] = lambdas
    games = []
    for (_, _), group in detail.groupby(["round_id", "match_no"], sort=False):
        if set(group.side) != {"home", "away"}:
            raise ValueError("missing side")
        h = group[group.side == "home"].iloc[0]
        a = group[group.side == "away"].iloc[0]
        probs = score_distribution(h["lambda"], a["lambda"])
        p = np.array([probs["prob_home"], probs["prob_draw"], probs["prob_away"]])
        outcome = {1: 0, 0: 1, 2: 2}[int(h.actual_result)]
        games.append({"round_id": h.round_id, "match_no": int(h.match_no),
                      "home_team": h.home_team, "away_team": h.away_team,
                      "lambda_home": h["lambda"], "lambda_away": a["lambda"],
                      "actual_home": int(h.actual_goals), "actual_away": int(a.actual_goals),
                      "prob_home": p[0], "prob_draw": p[1], "prob_away": p[2],
                      "log_loss": -math.log(max(p[outcome], 1e-12)),
                      "brier": float(np.sum((p - np.eye(3)[outcome]) ** 2))})
    g = pd.DataFrame(games)
    summary = {"matches": len(g),
               "goal_mae": float(np.mean(np.abs(g.lambda_home - g.actual_home) +
                                          np.abs(g.lambda_away - g.actual_away)) / 2),
               "log_loss": float(g.log_loss.mean()), "brier": float(g.brier.mean())}
    return summary, games


def main() -> None:
    frames = {name: load_round(name) for name in ROUNDS}
    train_names = ROUNDS[:4]
    cv = {}
    selected = {}
    for model, feature_count in (("xg_plus_style_index", 14), ("xg_plus_style_index_rate", 28)):
        cv[model] = []
        for alpha in ALPHAS:
            losses = []
            for start in (2, 3):
                train = pd.concat([frames[name] for name in train_names[:start]], ignore_index=True)
                validation = frames[train_names[start]]
                predicted = predict(validation, fit(train, alpha, feature_count))
                losses.append(assess(validation, predicted)[0]["log_loss"])
            cv[model].append({"alpha": alpha, "mean_log_loss": float(np.mean(losses))})
        selected[model] = min(cv[model], key=lambda item: item["mean_log_loss"])["alpha"]
    summaries = []
    all_details = []
    for test_name, prior in (("toto1663", train_names), ("toto1668", ROUNDS[:5])):
        train = pd.concat([frames[name] for name in prior], ignore_index=True)
        test = frames[test_name]
        for model in ("xg_raw", "xg_calibrated", "xg_plus_style_index", "xg_plus_style_index_rate"):
            lam = (test.base_lambda.to_numpy(float) if model == "xg_raw" else
                   predict(test, fit(train, None if model == "xg_calibrated" else selected[model],
                                     28 if model == "xg_plus_style_index_rate" else 14)))
            summary, details = assess(test, lam)
            summaries.append({"test_round": test_name, "model": model, **summary})
            for detail in details:
                all_details.append({"model": model, **detail})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_details).to_csv(OUT / "matches.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"train_rounds": list(train_names),
        "test_rounds": ["toto1663", "toto1668"], "style_features": list(STYLES),
        "alpha_cv": cv, "selected_alpha": selected, "scores": summaries,
        "notes": "toto1668 only 4 Nov-1 games after snapshot seal; baseline is xG+LAB lambda, no absence/weather/goal-rate adjustment"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_alpha": selected, "scores": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
