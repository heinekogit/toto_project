import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_meteo_client import OpenMeteoClient
from update_weather_cache import build_weather_row
import pandas as pd


class CacheFreshnessTest(unittest.TestCase):
    def test_csv_updater_keeps_missing_flags_and_acquisition_time(self):
        client = Mock()
        acquired = datetime.now(timezone.utc).isoformat()
        client.fetch_hourly_range.return_value = [{
            '_acquisition': {'fetched_at': acquired, 'data_kind': 'forecast'},
            'hourly_units': {'wind_speed_10m': 'km/h'},
            'hourly': {'time': ['2026-09-06T19:00'], 'temperature_2m': [25],
                       'precipitation': [None], 'wind_speed_10m': [3], 'weather_code': [None]}}]
        row = {'match_id': 'm', 'datetime': pd.Timestamp('2026-09-06 19:00'),
               'stadium': 's', 'lat': 35, 'lon': 139}
        result = build_weather_row(client, row)
        self.assertTrue(pd.isna(result['is_rain']))
        self.assertEqual(result['weather_fetch_ok'], 0)
        self.assertEqual(result['last_updated_at'], acquired)

    def test_fresh_reused_legacy_and_expired_refreshed(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = OpenMeteoClient(tmp, sleep_sec=0)
            path = Path(client._cache_path('2026-09-06', 35, 139))
            raw = {'hourly': {'time': ['2026-09-06T19:00']}}
            response = Mock()
            response.json.return_value = raw
            for meta, expected_calls in [({}, 1), ({'fetched_at': (datetime.now(timezone.utc)-timedelta(hours=7)).isoformat(), 'endpoint': client.base_url}, 1),
                                         ({'fetched_at': datetime.now(timezone.utc).isoformat(), 'endpoint': client.base_url}, 0)]:
                path.write_text(json.dumps({**raw, '_acquisition': meta}))
                with patch('open_meteo_client.requests.get', return_value=response) as get:
                    data = client.fetch_hourly_by_date('2026-09-06', 35, 139)
                    self.assertEqual(get.call_count, expected_calls)
                    self.assertIn('fetched_at', data['_acquisition'])

    def test_failed_refresh_does_not_return_stale_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = OpenMeteoClient(tmp, sleep_sec=0, retries=2)
            Path(client._cache_path('2026-09-06', 35, 139)).write_text('{}')
            with patch('open_meteo_client.requests.get', side_effect=RuntimeError('offline')) as get:
                with self.assertRaises(RuntimeError):
                    client.fetch_hourly_by_date('2026-09-06', 35, 139)
                self.assertEqual(get.call_count, 2)


if __name__ == '__main__':
    unittest.main()
