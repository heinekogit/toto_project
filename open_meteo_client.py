import json
import os
import time
from datetime import datetime, timedelta
import requests


class OpenMeteoClient:
    def __init__(
        self,
        cache_dir,
        base_url="https://api.open-meteo.com/v1/forecast",
        sleep_sec=0.5,
        retries=3,
        timeout=10,
    ):
        self.cache_dir = os.path.abspath(cache_dir)
        self.base_url = base_url
        self.sleep_sec = sleep_sec
        self.retries = retries
        self.timeout = timeout
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, date_str, lat, lon):
        lat_str = f"{lat:.6f}"
        lon_str = f"{lon:.6f}"
        filename = f"{date_str}_{lat_str}_{lon_str}.json"
        return os.path.join(self.cache_dir, filename)

    def fetch_hourly_by_date(self, date_str, lat, lon):
        cache_path = self._cache_path(date_str, lat, lon)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "Asia/Tokyo",
            "start_date": date_str,
            "end_date": date_str,
        }

        last_err = None
        for _ in range(self.retries):
            try:
                resp = requests.get(self.base_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                time.sleep(self.sleep_sec)
                return data
            except Exception as e:
                print(f"[openmeteo] fetch_hourly_by_date ERROR: {repr(e)}")
                raise

        raise RuntimeError(f"Open-Meteo取得失敗: {last_err}")

    def fetch_hourly_range(self, kickoff_dt, lat, lon, hours_before=2, hours_after=2):
        start_dt = kickoff_dt - timedelta(hours=hours_before)
        end_dt = kickoff_dt + timedelta(hours=hours_after)
        dates = []
        cur = start_dt.date()
        while cur <= end_dt.date():
            dates.append(cur.strftime("%Y-%m-%d"))
            cur = cur + timedelta(days=1)

        datasets = []
        for date_str in dates:
            datasets.append(self.fetch_hourly_by_date(date_str, lat, lon))
        return datasets
