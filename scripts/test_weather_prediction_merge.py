"""Exercise production merge functions without running the training script."""
import ast
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class WeatherPredictionMergeTest(unittest.TestCase):
    def load_functions(self, tmp):
        tree = ast.parse(Path(__file__).with_name('11_prediction_01.py').read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ['merge_weather_cache', 'normalize_weather_cache_columns']]
        scope = {'pd': pd, 'np': np, 'os': os, 'MERGE_QC_DIR': tmp,
                 'STRONG_WIND_THRESHOLD_KMH': 28.8, 'WIND_SPEED_UNIT': 'km/h',
                 'WEATHER_DEFAULT_TEMPERATURE': 20, 'WEATHER_DEFAULT_WIND_SPEED': 0,
                 '_ensure_merge_qc_dir': lambda: None,
                 '_log_df_key_health': lambda *args: None,
                 '_add_weather_match_id_aliases': lambda df, weather: (weather, 0)}
        exec(compile(ast.Module(body=selected, type_ignores=[]), '<production-weather>', 'exec'), scope)
        return scope

    def test_future_stale_and_partial_weather_not_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope = self.load_functions(tmp)
            now = pd.Timestamp.now(tz='UTC')
            left = pd.DataFrame({'match_id': ['fresh', 'stale', 'partial'],
                                 'datetime': [(now + pd.Timedelta(days=2)).tz_convert('Asia/Tokyo').tz_localize(None)] * 3})
            weather = pd.DataFrame({'match_id': left.match_id, 'is_rain': ['False', 'True', 'True'],
                                    'is_heavy_rain': [False, True, True], 'is_strong_wind': [False, True, None],
                                    'last_updated_at': [now.isoformat(), (now - pd.Timedelta(days=2)).isoformat(), now.isoformat()]})
            result = scope['merge_weather_cache'](left, weather, 'test')
            self.assertEqual(result.weather_missing.tolist(), [False, True, True])
            self.assertEqual(result.is_rain.tolist(), [False, False, False])

    def test_declared_ms_is_converted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope = self.load_functions(tmp)
            frame = pd.DataFrame({'wind_speed': [8.0], 'wind_speed_unit': ['m/s']})
            result = scope['normalize_weather_cache_columns'](frame)
            result = scope['normalize_weather_cache_columns'](result)
            self.assertAlmostEqual(result.wind_speed.iloc[0], 28.8)
            self.assertTrue(result.is_strong_wind.iloc[0])

    def test_absent_flag_column_is_missing_and_provenance_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope = self.load_functions(tmp)
            now = pd.Timestamp.now(tz='UTC')
            left = pd.DataFrame({'match_id': ['m'], 'datetime': [now + pd.Timedelta(days=1)]})
            weather = pd.DataFrame({'match_id': ['m'], 'is_rain': [True], 'last_updated_at': [now.isoformat()], 'weather_data_kind': ['forecast']})
            out = scope['merge_weather_cache'](left, weather, 'partial')
            self.assertTrue(out.weather_missing.iloc[0])
            self.assertEqual(out.last_updated_at.iloc[0], now.isoformat())
            self.assertEqual(out.weather_data_kind.iloc[0], 'forecast')


if __name__ == '__main__':
    unittest.main()
