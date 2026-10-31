#!/usr/bin/env python3
"""Fit capped, nonnegative fatigue/weather uncertainty weights on archived forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from uncertainty_calibration import adjust, signals

ROOT = Path(__file__).resolve().parents[1]
LABEL = {1: 0, 0: 1, 2: 2}  # stored 1/0/2 -> H/D/A


def load_history(path):
    source = pd.read_csv(path)
    payloads = [json.loads(value) for value in source.prediction_payload_json]
    required = ["home_total_fatigue_score", "away_total_fatigue_score", "weather_missing", "is_rain", "is_heavy_rain", "is_strong_wind", "__weather_asof"]
    missing = [i for i, p in enumerate(payloads) if any(p.get(name) is None for name in required)]
    if missing:
        raise ValueError(f"archived pre-match signal missing in {len(missing)} rows")
    if any(p["weather_missing"] for p in payloads):
        raise ValueError("weather missing; do not treat unknown weather as clear")
    observed_day = pd.to_datetime(source.datetime).dt.normalize()
    weather_asof = pd.to_datetime([str(p["__weather_asof"]) for p in payloads], format="%Y%m%d")
    if (weather_asof > observed_day).any():
        raise ValueError("future weather snapshot detected")
    source = source[source.actual_result.notna()].copy()
    payloads = [json.loads(value) for value in source.prediction_payload_json]
    fatigue, weather = signals(
        [p["home_total_fatigue_score"] for p in payloads],
        [p["away_total_fatigue_score"] for p in payloads],
        [p["is_rain"] for p in payloads],
        [p["is_heavy_rain"] for p in payloads],
        [p["is_strong_wind"] for p in payloads],
    )
    source["fatigue_signal"] = fatigue
    source["weather_signal"] = weather
    source["actual_index"] = pd.to_numeric(source.actual_result).map(LABEL)
    if source.actual_index.isna().any():
        raise ValueError("unknown actual result")
    source["round_sort"] = source.round_id.astype(str).str.extract(r"(\d+)")[0].astype(int)
    return source.sort_values(["round_sort", "datetime", "match_no"]).reset_index(drop=True)


def score(prob, actual):
    onehot = np.eye(3)[actual]
    return {
        "logloss": float(-np.log(np.clip(prob[np.arange(len(actual)), actual], 1e-12, 1)).mean()),
        "brier": float(np.square(prob - onehot).sum(axis=1).mean()),
        "accuracy": float((prob.argmax(axis=1) == actual).mean()),
    }


def evaluate(frame, train_ratio=0.75, max_mix=0.25):
    rounds = frame.round_sort.drop_duplicates().tolist()
    if len(rounds) < 4:
        raise ValueError("at least four historical rounds are required")
    train_rounds = rounds[:max(2, min(len(rounds) - 1, int(len(rounds) * train_ratio)))]
    train = frame[frame.round_sort.isin(train_rounds)]
    valid = frame[~frame.round_sort.isin(train_rounds)]
    if len(valid) < 20:
        raise ValueError("at least 20 later matches are required for validation")

    def arrays(part):
        p = part[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        y = part.actual_index.to_numpy(dtype=int)
        f = part.fatigue_signal.to_numpy(dtype=float)
        w = part.weather_signal.to_numpy(dtype=float)
        return p, y, f, w

    p_train, y_train, f_train, w_train = arrays(train)
    p_valid, y_valid, f_valid, w_valid = arrays(valid)

    def objective(weights):
        adjusted, _ = adjust(p_train, f_train, w_train, weights[0], weights[1], max_mix)
        return score(adjusted, y_train)["logloss"]

    fitted = minimize(objective, [0.03, 0.03], bounds=[(0, max_mix), (0, max_mix)], method="L-BFGS-B")
    if not fitted.success:
        raise RuntimeError(f"calibration failed: {fitted.message}")
    adjusted, mix = adjust(p_valid, f_valid, w_valid, *fitted.x, max_mix)
    baseline = score(p_valid, y_valid)
    candidate = score(adjusted, y_valid)
    return {
        "train_rounds": train_rounds, "validation_rounds": rounds[len(train_rounds):],
        "train_rows": len(train), "validation_rows": len(valid),
        "fatigue_weight": float(fitted.x[0]), "weather_weight": float(fitted.x[1]),
        "max_mix": max_mix, "baseline": baseline, "candidate": candidate,
        "mean_validation_mix": float(mix.mean()), "max_validation_mix": float(mix.max()),
        "delta_logloss": candidate["logloss"] - baseline["logloss"],
        "status": "held-out validation of monotone uncertainty calibration",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "data/eval/observation_history/matches.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/reports/uncertainty_calibration_2026.json")
    parser.add_argument("--export-model", type=Path, help="Write the validated coefficients for prediction runs")
    args = parser.parse_args()
    report = evaluate(load_history(args.history))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.export_model:
        if report["delta_logloss"] >= 0:
            raise ValueError("held-out logloss did not improve; model not exported")
        model = {
            "version": "fatigue_weather_uniform_v1",
            "fatigue_weight": report["fatigue_weight"],
            "weather_weight": report["weather_weight"],
            "max_mix": report["max_mix"],
            "fit_rounds": report["train_rounds"],
            "validation_rounds": report["validation_rounds"],
            "validation_logloss_baseline": report["baseline"]["logloss"],
            "validation_logloss_candidate": report["candidate"]["logloss"],
            "source": str(args.history.resolve()),
        }
        args.export_model.parent.mkdir(parents=True, exist_ok=True)
        args.export_model.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
