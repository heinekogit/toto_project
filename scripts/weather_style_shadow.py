"""Weather uncertainty hypothesis for shadow evaluation only.

Adverse weather lowers confidence in the pre-match favourite. Probability
removed from rank 1 is shared by draw and the other side; it is never assigned
to draw alone. Playing style only tilts that split.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

VERSION = "weather_uncertainty_v1"
FEATURES = {
    "technical_dependency": [
        ("stats_1試合平均パス数", 300.0, 650.0),
        ("flab_possession_rate", 35.0, 65.0),
        ("stats_1試合平均ドリブル数", 5.0, 20.0),
    ],
    "direct_resilience": [
        ("flab_style_long_counter_index", 20.0, 70.0),
        ("stats_1試合平均空中戦勝利数", 10.0, 30.0),
        ("stats_空中戦勝率", 40.0, 60.0),
    ],
}


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def flag(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    return None


def _score(row, side, group):
    values, evidence = [], []
    for field, low, high in FEATURES[group]:
        raw = number(row.get(f"{field}_{side}"))
        evidence.append(f"{field}={'unknown' if raw is None else f'{raw:g}'}")
        if raw is not None:
            values.append(float(np.clip((raw - low) / (high - low), 0.0, 1.0)))
    # Partial feature sets would silently change the score's meaning.
    value = sum(values) / len(values) if len(values) == len(FEATURES[group]) else None
    return value, "; ".join(evidence)


def style_profile(row, side):
    technical, technical_evidence = _score(row, side, "technical_dependency")
    direct, direct_evidence = _score(row, side, "direct_resilience")
    return {
        "technical_dependency": technical,
        "direct_resilience": direct,
        "evidence": technical_evidence + "; " + direct_evidence,
    }


def _entropy(values):
    return -sum(value * math.log(value) for value in values if value > 0)


def apply_weather_uncertainty_shadow(
    frame, max_shift=0.06, style_tilt=0.15, side_favourite_only=False
):
    """Move up to max_shift from rank 1 to both alternatives.

    Heavy rain severity is 1.0, ordinary rain 0.35 and strong wind 0.65;
    combined severity is capped at 1.0. Style changes at most 15 percentage
    points of how moved mass is split between draw and the opposing side.
    """
    if not 0.0 <= max_shift <= 0.08:
        raise ValueError("max_shift must be in [0, 0.08]")
    if not 0.0 <= style_tilt <= 0.20:
        raise ValueError("style_tilt must be in [0, 0.20]")
    if "shadow_version" in frame and frame["shadow_version"].notna().any():
        raise ValueError("weather uncertainty shadow is already applied")

    out = frame.copy()
    labels = ["home", "draw", "away"]
    symbols = {"home": "H", "draw": "D", "away": "A"}
    toto_symbols = {"home": "1", "draw": "0", "away": "2"}

    for idx, row in out.iterrows():
        baseline = {label: number(row.get(f"prob_final_{label}")) for label in labels}
        if any(value is None or not 0 <= value <= 1 for value in baseline.values()):
            raise ValueError("invalid baseline probability")
        if abs(sum(baseline.values()) - 1.0) > 1e-6:
            raise ValueError("baseline probabilities do not sum to one")

        profiles = {side: style_profile(row, side) for side in ["home", "away"]}
        for side, profile in profiles.items():
            for key, value in profile.items():
                out.loc[idx, f"shadow_{side}_{key}"] = value

        rain, heavy, wind = [flag(row.get(key)) for key in ["is_rain", "is_heavy_rain", "is_strong_wind"]]
        weather_known = flag(row.get("weather_missing")) is False
        weather_assessable = weather_known and rain is not None and heavy is not None and wind is not None
        rain_severity = 1.0 if heavy is True else 0.35 if rain is True else 0.0
        wind_severity = 0.65 if wind is True else 0.0
        severity = min(1.0, rain_severity + wind_severity) if weather_assessable else 0.0

        favourite = max(labels, key=lambda label: baseline[label])
        top_probability = baseline[favourite]
        confidence_factor = float(np.clip((top_probability - 1 / 3) / (2 / 3), 0.0, 1.0))
        shift = min(max_shift * severity * confidence_factor, max(0.0, top_probability - 1 / 3))
        if side_favourite_only and favourite == "draw":
            shift = 0.0
        adjusted = baseline.copy()
        draw_share, style_assessable, style_reason = None, False, "not_applicable"

        if shift > 0:
            alternatives = [label for label in labels if label != favourite]
            alternative_total = sum(baseline[label] for label in alternatives)
            shares = {label: baseline[label] / alternative_total for label in alternatives}

            # For a side favourite, uncertainty benefits both D and opponent.
            if favourite in {"home", "away"}:
                other = "away" if favourite == "home" else "home"
                fav_profile, other_profile = profiles[favourite], profiles[other]
                values = [fav_profile["technical_dependency"], fav_profile["direct_resilience"],
                          other_profile["technical_dependency"], other_profile["direct_resilience"]]
                if all(value is not None for value in values):
                    style_assessable = True
                    edge = 0.5 * (fav_profile["technical_dependency"] - other_profile["technical_dependency"])
                    edge += 0.5 * (other_profile["direct_resilience"] - fav_profile["direct_resilience"])
                    shares[other] = float(np.clip(shares[other] + style_tilt * edge, 0.20, 0.80))
                    shares["draw"] = 1.0 - shares[other]
                    style_reason = f"opponent_tilt_edge={edge:.4f}"
                draw_share = shares["draw"]

            adjusted[favourite] -= shift
            for label in alternatives:
                adjusted[label] += shift * shares[label]

        entropy_before = _entropy(baseline.values())
        entropy_after = _entropy(adjusted.values())
        if entropy_after + 1e-12 < entropy_before:
            raise AssertionError("uncertainty shadow reduced entropy")

        metadata = {
            "version": VERSION, "weather_assessable": weather_assessable,
            "side_favourite_only": side_favourite_only,
            "weather_severity": severity, "favourite": favourite,
            "baseline_top_probability": top_probability, "confidence_factor": confidence_factor,
            "probability_shift": shift, "draw_share_of_shift": draw_share,
            "style_assessable": style_assessable, "style_reason": style_reason,
            "entropy_before": entropy_before, "entropy_after": entropy_after,
            "entropy_delta": entropy_after - entropy_before,
        }
        for key, value in metadata.items():
            out.loc[idx, f"shadow_{key}"] = value
        for label, value in adjusted.items():
            out.loc[idx, f"shadow_baseline_{label}"] = baseline[label]
            out.loc[idx, f"shadow_adjusted_{label}"] = value

        if shift == 0:
            continue
        # This is a terminal recalibration. Preserve BASE/LAB/MAIN/blend columns
        # so the source models remain auditable, and update only the final
        # probabilities consumed by purchase-context and BuyPlan generation.
        for label, value in adjusted.items():
            for column in [f"prob_final_{label}", f"final_prob_{label}"]:
                if column in out:
                    out.loc[idx, column] = value
        for column, value in [("prob_home_win", adjusted["home"]), ("prob_draw", adjusted["draw"]),
                              ("prob_away_win", adjusted["away"]), ("prob_home", adjusted["home"]),
                              ("prob_away", adjusted["away"])]:
            if column in out:
                out.loc[idx, column] = value
        result = max(labels, key=lambda label: adjusted[label])
        for column in ["predicted_result", "predicted_result_main", "final_result", "argmax_result",
                       "predicted_highest_prob_result"]:
            if column in out:
                out.loc[idx, column] = symbols[result]
        if "predicted_result_main_symbol" in out:
            out.loc[idx, "predicted_result_main_symbol"] = toto_symbols[result]
    return out


def apply_weather_style_shadow(frame, rain_weight=0.0, wind_weight=0.0):
    """Compatibility wrapper for older shadow commands."""
    return apply_weather_uncertainty_shadow(frame, max_shift=max(rain_weight, wind_weight))
