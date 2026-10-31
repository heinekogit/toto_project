import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data' / 'reports'))
from build_round_views import _fill_if_missing


class FillMissingTests(unittest.TestCase):
    def test_missing_string_source_leaves_float_column_unchanged(self):
        frame = pd.DataFrame({'datetime_weather': [float('nan')] * 4})
        before = frame.copy()
        _fill_if_missing(frame, 'datetime_weather', pd.Series([None] * 4, dtype='str'))
        pd.testing.assert_frame_equal(frame, before)

    def test_text_can_fill_empty_numeric_column(self):
        frame = pd.DataFrame({'datetime_weather': [float('nan')] * 2})
        _fill_if_missing(frame, 'datetime_weather', pd.Series(['2026-10-08 19:00', None], dtype='str'))
        self.assertEqual(frame.datetime_weather.iloc[0], '2026-10-08 19:00')
        self.assertTrue(pd.isna(frame.datetime_weather.iloc[1]))

    def test_preserves_existing_values_and_aligns_index(self):
        frame = pd.DataFrame({'stadium': ['existing', None]}, index=[4, 2])
        _fill_if_missing(frame, 'stadium', pd.Series(['fill', 'do not overwrite'], index=[2, 4]))
        self.assertEqual(frame.stadium.tolist(), ['existing', 'fill'])

    def test_numeric_fill_stays_numeric(self):
        frame = pd.DataFrame({'temperature': [20.0, float('nan')]})
        _fill_if_missing(frame, 'temperature', pd.Series([99.0, 21.5]))
        self.assertEqual(frame.temperature.tolist(), [20.0, 21.5])
        self.assertEqual(str(frame.temperature.dtype), 'float64')

    def test_boolean_fill_and_absent_column(self):
        frame = pd.DataFrame({'weather_missing': [float('nan')]})
        _fill_if_missing(frame, 'weather_missing', pd.Series([True], dtype='boolean'))
        self.assertEqual(frame.weather_missing.iloc[0], True)
        _fill_if_missing(frame, 'absent', pd.Series([True]))
        self.assertNotIn('absent', frame)


if __name__ == '__main__':
    unittest.main()
