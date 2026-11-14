import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from acl_schedule import classify_distance, normalize_acl_schedule


class AclScheduleTest(unittest.TestCase):
    def test_memo_does_not_hide_missing_venue(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "acl.csv"
            pd.DataFrame([{"match_date": "2026/4/15", "home_team": "G大阪",
                           "away_team": "相手", "city": "未登録", "country": "タイ",
                           "memo": "long"}]).to_csv(path, index=False)
            with patch("acl_schedule.load_jleague_home_coordinates", return_value={"G大阪": (34.8, 135.5)}), \
                 patch("acl_schedule.load_acl_venue_coordinates", return_value={}):
                with self.assertRaisesRegex(ValueError, "venue coordinates missing"):
                    normalize_acl_schedule(path)

    def test_production_schedule_rejects_inconsistent_venue(self):
        schedule = Path(__file__).parents[1] / "data/manual/acl_schedule.csv"
        with self.assertRaisesRegex(ValueError, "吹田, サウジアラビア"):
            normalize_acl_schedule(schedule)

    def test_confirmed_schedule_keeps_calculated_travel_and_home_zero(self):
        schedule = Path(__file__).parents[1] / "data/manual/acl_schedule.csv"
        frame = pd.read_csv(schedule, encoding="utf-8-sig")
        frame = frame[frame.status.eq("confirmed")]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "confirmed.csv"
            frame.to_csv(path, index=False)
            normalized = normalize_acl_schedule(path)
        self.assertEqual(len(normalized), len(frame))
        self.assertEqual(len(normalized), 18)
        self.assertTrue(normalized.loc[normalized.venue_type.eq("away"), "distance_km"].gt(0).all())
        self.assertTrue(normalized.loc[normalized.venue_type.eq("home"), "distance_km"].eq(0).all())

    def test_distance_bands_keep_legacy_scale(self):
        self.assertEqual(classify_distance(999.9), ("short", 3.0))
        self.assertEqual(classify_distance(1000), ("medium", 4.0))
        self.assertEqual(classify_distance(2500), ("long", 5.0))
        self.assertEqual(classify_distance(5000), ("long_haul", 7.0))

    def test_new_schema_identifies_jleague_club_and_calculates(self):
        frame = pd.DataFrame([{
            "match_date": "2026/9/15", "competition": "ACL Elite",
            "home_team": "大田", "away_team": "京都", "city": "大田広域市",
            "country": "韓国", "memo": "第1節", "memo2": "",
        }])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "acl.csv"
            frame.to_csv(path, index=False)
            with patch("acl_schedule.load_jleague_home_coordinates", return_value={"京都": (35.0, 135.5)}), \
                 patch("acl_schedule.load_acl_venue_coordinates", return_value={("大田広域市", "韓国"): (36.35, 127.38)}):
                out = normalize_acl_schedule(path).iloc[0]
        self.assertEqual(out.team, "京都")
        self.assertEqual(out.opponent, "大田")
        self.assertEqual(out.venue_type, "away")
        self.assertEqual(out.travel_type, "short")
        self.assertEqual(out.fatigue_grade, 3.0)
        self.assertGreater(out.distance_km, 0)

    def test_historical_memo_is_an_override(self):
        frame = pd.DataFrame([{
            "match_date": "2026/4/15", "home_team": "G大阪", "away_team": "バンコクU",
            "city": "バンコク", "country": "タイ", "memo": "long",
        }])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "acl.csv"
            frame.to_csv(path, index=False)
            with patch("acl_schedule.load_jleague_home_coordinates", return_value={"G大阪": (34.8, 135.5)}), \
                 patch("acl_schedule.load_acl_venue_coordinates", return_value={("バンコク", "タイ"): (13.75, 100.5)}):
                out = normalize_acl_schedule(path).iloc[0]
        self.assertEqual(out.travel_type, "long")
        self.assertEqual(out.fatigue_grade, 5.0)


if __name__ == "__main__":
    unittest.main()
