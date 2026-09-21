#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_builder import build_features_for_match, hourly_json_to_df
from weather_rules import adverse_weather_penalty


class WeatherRulesTest(unittest.TestCase):
    def _features(self, wind_kmh):
        hourly = pd.DataFrame([{
            "datetime": pd.Timestamp("2026-09-02 19:00"),
            "temperature_2m": 25.0,
            "precipitation": 0.0,
            "wind_speed_10m": wind_kmh,
            "weather_code": 0,
        }])
        return build_features_for_match(
            {"kickoff_jst": pd.Timestamp("2026-09-02 19:00")}, hourly
        )

    def test_strong_wind_threshold_is_8ms_in_kmh(self):
        self.assertEqual(self._features(8.0)["is_strong_wind"], 0)
        self.assertEqual(self._features(28.7)["is_strong_wind"], 0)
        self.assertEqual(self._features(28.8)["is_strong_wind"], 1)
        self.assertEqual(self._features(28.8)["wind_speed_unit"], "km/h")

    def test_heavy_rain_supersedes_ordinary_rain(self):
        self.assertAlmostEqual(adverse_weather_penalty(True, True, False), 0.8)
        self.assertAlmostEqual(adverse_weather_penalty(True, True, True), 1.25)

    def test_missing_rain_is_not_dry(self):
        hourly = pd.DataFrame([{'datetime': pd.Timestamp('2026-09-06 19:00'),
                               'temperature_2m': 25, 'precipitation': float('nan'),
                               'wind_speed_10m': 3, 'weather_code': None}])
        features = build_features_for_match({'kickoff_jst': pd.Timestamp('2026-09-06 19:00')}, hourly)
        self.assertIsNone(features['is_rain'])
        self.assertEqual(features['weather_quality_status'], 'partial_match_window')

    def test_rain_after_kickoff_is_visible(self):
        hourly = pd.DataFrame({'datetime': pd.date_range('2026-09-06 19:00', periods=3, freq='h'),
                               'temperature_2m': [25]*3, 'precipitation': [0, 6, 7],
                               'wind_speed_10m': [3]*3, 'weather_code': [0]*3})
        features = build_features_for_match({'kickoff_jst': pd.Timestamp('2026-09-06 19:00')}, hourly)
        self.assertEqual(features['is_rain'], 0)
        self.assertTrue(features['adverse_weather_during_match'])
        self.assertEqual(features['weather_quality_status'], 'complete')

    def test_out_of_window_cannot_be_success(self):
        hourly = pd.DataFrame([{'datetime': pd.Timestamp('2026-09-05'), 'temperature_2m': 25,
                               'precipitation': 0, 'wind_speed_10m': 0, 'weather_code': 0}])
        features = build_features_for_match({'kickoff_jst': pd.Timestamp('2026-09-06')}, hourly)
        self.assertNotIn('is_rain', features)

    def test_unit_conversion_and_unknown_rejection(self):
        raw = {'hourly_units': {'wind_speed_10m': 'm/s'},
               'hourly': {'time': ['2026-09-06T19:00'], 'wind_speed_10m': [8]}}
        self.assertAlmostEqual(hourly_json_to_df(raw).wind_speed_10m.iloc[0], 28.8)
        raw['hourly_units']['wind_speed_10m'] = 'unknown'
        with self.assertRaises(ValueError):
            hourly_json_to_df(raw)


if __name__ == "__main__":
    unittest.main()
