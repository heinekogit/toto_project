import os
import sys
import unittest

import pandas as pd

SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from jleague_matches import build_match_id, parse_match_datetimes, validate_match_identity


class TestJleagueMatches(unittest.TestCase):
    def test_datetime_uses_kickoff_when_available(self):
        actual = parse_match_datetimes(pd.Series(["26/08/07(金)"]), pd.Series(["19:25"]))
        self.assertEqual(actual.iloc[0], pd.Timestamp("2026-08-07 19:25"))

    def test_datetime_falls_back_to_date_when_kickoff_is_undecided(self):
        actual = parse_match_datetimes(pd.Series(["27/02/13(土)"]), pd.Series(["未定"]))
        self.assertEqual(actual.iloc[0], pd.Timestamp("2027-02-13 00:00"))

    def test_match_id_falls_back_to_round_and_card_when_date_is_undecided(self):
        actual = build_match_id("j1", "2026", "第２０節第２日", pd.NaT, "京都", "岡山")
        self.assertEqual(actual, "j1_2026_round_第20節第2日_京都_岡山")

    def test_schedule_and_results_generate_same_fallback_id(self):
        args = ("j1", "2026", "第２０節第２日", pd.NaT, "京都", "岡山")
        self.assertEqual(build_match_id(*args), build_match_id(*args))

    def test_validation_rejects_duplicate_ids(self):
        df = pd.DataFrame({"match_id": ["same", "same"], "datetime": [pd.NaT, pd.NaT]})
        with self.assertRaisesRegex(RuntimeError, "duplicate_rows=2"):
            validate_match_identity(df, label="test")


if __name__ == "__main__":
    unittest.main()
