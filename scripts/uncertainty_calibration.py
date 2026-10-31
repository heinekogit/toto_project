"""Monotone uncertainty calibration for final H/D/A probabilities.

The correction mixes the existing forecast with the uniform distribution.
It cannot favour a particular H/D/A symbol and cannot reduce entropy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signals(home_fatigue, away_fatigue, rain, heavy_rain, strong_wind,
            rain_signal=0.35, heavy_rain_signal=1.0, strong_wind_signal=0.65):
    if not all(np.isfinite(value) and value >= 0 for value in
               (rain_signal, heavy_rain_signal, strong_wind_signal)):
        raise ValueError("weather signal coefficients must be finite and nonnegative")
    fatigue = np.clip((np.asarray(home_fatigue, dtype=float) + np.asarray(away_fatigue, dtype=float)) / 16.0, 0.0, 1.0)
    weather = np.clip(np.where(heavy_rain, heavy_rain_signal, np.where(rain, rain_signal, 0.0))
                      + np.where(strong_wind, strong_wind_signal, 0.0), 0.0, 1.0)
    return fatigue, weather


def adjust(probabilities, fatigue, weather, fatigue_weight, weather_weight, max_mix=0.25):
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3 or not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("invalid H/D/A probabilities")
    if min(fatigue_weight, weather_weight) < 0 or max_mix < 0 or max_mix > 1:
        raise ValueError("uncertainty weights must be nonnegative and max_mix in [0,1]")
    f = np.asarray(fatigue, dtype=float)
    w = np.asarray(weather, dtype=float)
    if len(f) != len(probs) or len(w) != len(probs) or not np.isfinite(f).all() or not np.isfinite(w).all():
        raise ValueError("missing uncertainty signals")
    f_effect = np.clip(fatigue_weight * f, 0.0, 1.0)
    w_effect = np.clip(weather_weight * w, 0.0, 1.0)
    mix = np.minimum(1.0 - (1.0 - f_effect) * (1.0 - w_effect), max_mix)
    adjusted = (1.0 - mix[:, None]) * probs + mix[:, None] / 3.0
    return adjusted, mix


def apply_to_frame(frame, fatigue_weight, weather_weight, max_mix=0.25,
                   probability_prefix="prob_final", audit_prefix="uncertainty",
                   rain_signal=0.35, heavy_rain_signal=1.0, strong_wind_signal=0.65):
    """Apply once after other probability fusion, preserving model components."""
    out = frame.copy()
    if out.empty:
        return out
    mix_column = f"{audit_prefix}_calibration_mix"
    if mix_column in out.columns:
        raise ValueError("uncertainty calibration already applied")
    required = [f"{probability_prefix}_{name}" for name in ("home", "draw", "away")] + [
                "home_total_fatigue_score", "away_total_fatigue_score",
                "is_rain", "is_heavy_rain", "is_strong_wind", "weather_missing"]
    missing = [name for name in required if name not in out.columns]
    if missing:
        raise ValueError(f"uncertainty inputs missing: {missing}")
    fatigue_inputs = out[["home_total_fatigue_score", "away_total_fatigue_score"]].apply(pd.to_numeric, errors="coerce")
    if fatigue_inputs.isna().any().any():
        raise ValueError("unknown fatigue cannot be treated as zero")
    def boolean_column(name, allow_unknown=False):
        values = out[name].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if values.isna().any() and not allow_unknown:
            raise ValueError(f"unknown {name}")
        return values
    weather_missing = boolean_column("weather_missing")
    rain = boolean_column("is_rain", allow_unknown=True)
    heavy = boolean_column("is_heavy_rain", allow_unknown=True)
    wind = boolean_column("is_strong_wind", allow_unknown=True)
    weather_known = ~weather_missing & rain.notna() & heavy.notna() & wind.notna()
    # Unknown weather has no weather adjustment; it is explicitly flagged.
    # Fatigue remains usable independently.
    rain = rain.fillna(False).to_numpy(dtype=bool)
    heavy = heavy.fillna(False).to_numpy(dtype=bool)
    wind = wind.fillna(False).to_numpy(dtype=bool)
    fatigue, weather = signals(
        fatigue_inputs.iloc[:, 0], fatigue_inputs.iloc[:, 1],
        rain, heavy, wind, rain_signal, heavy_rain_signal, strong_wind_signal,
    )
    weather = np.where(weather_known.to_numpy(dtype=bool), weather, 0.0)
    names = ["home", "draw", "away"]
    columns = [f"{probability_prefix}_{name}" for name in names]
    before = out[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    after, mix = adjust(before, fatigue, weather, fatigue_weight, weather_weight, max_mix)
    out[f"{audit_prefix}_calibration_version"] = "fatigue_weather_uniform_v1"
    out[f"{audit_prefix}_fatigue_signal"] = fatigue
    out[f"{audit_prefix}_weather_signal"] = weather
    out[f"{audit_prefix}_weather_known"] = weather_known.to_numpy(dtype=bool)
    out[mix_column] = mix
    for i, name in enumerate(names):
        before_column = (f"prob_pre_uncertainty_{name}" if probability_prefix == "prob_final"
                         else f"{probability_prefix}_pre_uncertainty_{name}")
        out[before_column] = before[:, i]
        out[f"{probability_prefix}_{name}"] = after[:, i]
    if probability_prefix == "prob_final":
        for column, i in [("prob_home_win", 0), ("prob_draw", 1), ("prob_away_win", 2),
                          ("prob_home", 0), ("prob_away", 2)]:
            if column in out:
                out[column] = after[:, i]
    if not np.array_equal(before.argmax(axis=1), after.argmax(axis=1)):
        raise AssertionError("uncertainty adjustment changed H/D/A ranking")
    entropy_before = -(before * np.log(np.clip(before, 1e-12, 1))).sum(axis=1)
    entropy_after = -(after * np.log(np.clip(after, 1e-12, 1))).sum(axis=1)
    if (entropy_after + 1e-12 < entropy_before).any():
        raise AssertionError("uncertainty adjustment reduced entropy")
    out[f"{audit_prefix}_entropy_delta"] = entropy_after - entropy_before
    return out
