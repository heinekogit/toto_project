import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_weather_features as fetch


class WeatherFetchPipelineTest(unittest.TestCase):
    def run_fetch(self, tmp, missing_coordinate=False):
        args = SimpleNamespace(stadiums='unused', stadiums_sheet=None, matches='unused', matches_sheet=None,
                               out=str(Path(tmp)/'weather.csv'), cache_dir=tmp, lookahead_days=None)
        matches = pd.DataFrame({'match_id': ['a', 'b'], 'kickoff_jst': [pd.Timestamp('2026-09-11 19:00')]*2,
                                'stadium_name': ['A','B'], 'lat': [35, None if missing_coordinate else 36], 'lon': [139, 140]})
        acquired = datetime.now(timezone.utc).isoformat()
        client = Mock()
        client.fetch_hourly_range.return_value = [{'_acquisition': {'fetched_at': acquired, 'data_kind': 'forecast'},
            'hourly_units': {'wind_speed_10m':'km/h'}, 'hourly': {'time': ['2026-09-11T19:00','2026-09-11T20:00','2026-09-11T21:00'],
            'temperature_2m':[25]*3,'precipitation':[0,6,0],'wind_speed_10m':[3]*3,'weather_code':[0]*3}}]
        with patch.object(fetch,'parse_args',return_value=args), patch.object(fetch,'setup_logger'), \
             patch.object(fetch,'load_stadiums'), patch.object(fetch,'load_matches'), \
             patch.object(fetch,'merge_matches_with_stadiums',return_value=(matches,'key')), \
             patch.object(fetch,'OpenMeteoClient',return_value=client), patch.dict(os.environ,{'WEATHER_MIN_SUCCESS_RATIO':'1.0'}):
            fetch.main()
        return pd.read_csv(args.out), acquired

    def test_complete_forecast_preserves_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, acquired = self.run_fetch(tmp)
            self.assertTrue(out.weather_fetch_ok.eq(1).all())
            self.assertTrue(out.last_updated_at.eq(acquired).all())
            self.assertTrue(out.adverse_weather_during_match.all())

    def test_coordinate_failure_included_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'below threshold'):
                self.run_fetch(tmp, missing_coordinate=True)


if __name__ == '__main__':
    unittest.main()
