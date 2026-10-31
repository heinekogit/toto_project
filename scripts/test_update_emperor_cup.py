import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from update_emperor_cup import parse_results
from test_fatigue_components import _load_module


class EmperorCupTests(unittest.TestCase):
    def payload(self):
        return {"matchScheduleList": {"competitionName": "天皇杯", "matchSchedule": [{
            "matchNumber": "1", "matchDateJpn": "2026/09/23", "matchTimeJpn": "19:00",
            "homeTeamName": "ジェフユナイテッド千葉", "awayTeamName": "浦和レッズ",
            "matchStatus": "試合終了", "score": {"homeScore": "1", "awayScore": "1", "exMatch": True},
        }]}}

    def test_finished_draw_with_pk_is_confirmed_and_changes_rest(self):
        events = parse_results(self.payload(), 2026)
        self.assertEqual(events.team.tolist(), ["千葉", "浦和"])
        self.assertEqual(events.extra_time_minutes.tolist(), [30, 30])
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            events.to_csv(path, index=False)
            loaded = module.load_external_team_events(path)
        matches = pd.DataFrame([dict(match_id="m", datetime=pd.Timestamp("2026-09-26 19:00"), home_team="千葉", away_team="浦和")])
        result = module.calculate_fatigue(matches, pd.DataFrame(), loaded).iloc[0]
        self.assertEqual(result.home_rest_hours, 72)
        self.assertEqual(result.home_external_matches_last14d, 1)
        self.assertGreater(result.home_rest_fatigue, 0)
        detail = json.loads(result.home_external_events_json)[0]
        self.assertEqual(detail["extra_time_minutes"], 30)
        self.assertEqual(detail["minutes_status"], "unknown")
        self.assertTrue(result.home_external_load_unknown)

    def test_unfinished_postponed_and_undecided_are_not_appearances(self):
        for status in ["", "試合前", "延期", "試合中", "中止"]:
            payload = self.payload()
            payload["matchScheduleList"]["matchSchedule"][0]["matchStatus"] = status
            self.assertTrue(parse_results(payload, 2026).empty)

    def test_bad_completed_data_fails_and_duplicate_fails(self):
        for key, value in [("matchTimeJpn", "未定"), ("homeTeamName", "No.2の勝者"), ("score", {})]:
            payload = self.payload()
            payload["matchScheduleList"]["matchSchedule"][0][key] = value
            with self.assertRaises(ValueError):
                parse_results(payload, 2026)
        payload = self.payload()
        payload["matchScheduleList"]["matchSchedule"] *= 2
        with self.assertRaises(ValueError):
            parse_results(payload, 2026)

    def test_next_january_final_and_unknown_extra_time(self):
        payload = self.payload()
        row = payload["matchScheduleList"]["matchSchedule"][0]
        row["matchDateJpn"] = "2027/01/01"
        del row["score"]["exMatch"]
        self.assertTrue(parse_results(payload, 2026).extra_time_minutes.isna().all())


if __name__ == "__main__":
    unittest.main()
