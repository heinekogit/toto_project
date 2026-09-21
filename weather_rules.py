"""Shared, side-effect-free weather classification and penalty rules."""

from __future__ import annotations

import numpy as np

WIND_SPEED_UNIT = "km/h"
STRONG_WIND_THRESHOLD_KMH = 28.8  # 8 m/s


def strong_wind_from_kmh(value):
    try:
        return bool(float(value) >= STRONG_WIND_THRESHOLD_KMH)
    except (TypeError, ValueError):
        return False


def adverse_weather_penalty(is_rain, is_heavy_rain, is_strong_wind):
    """Heavy rain supersedes ordinary rain; wind remains independently additive."""
    rain = np.asarray(is_rain, dtype=bool)
    heavy = np.asarray(is_heavy_rain, dtype=bool)
    wind = np.asarray(is_strong_wind, dtype=bool)
    value = np.where(heavy, 0.8, np.where(rain, 0.45, 0.0)) + np.where(wind, 0.45, 0.0)
    return float(value) if value.ndim == 0 else value
