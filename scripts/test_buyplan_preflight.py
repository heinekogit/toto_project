import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "next_model"))
from preflight import check_inputs


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.order = pd.DataFrame([dict(match_no=1, match_date="10月10日")])
        row = dict(match_no=1, home_team="H", away_team="A", datetime="2026-10-10 14:00",
                   stats_asof="2026-10-09", flab_snapshot_date=20261009,
                   prob_final_home=0.45, prob_final_draw=0.25, prob_final_away=0.30)
        for side in ("home", "away"):
            for field in ("stats_1試合平均得点数", "stats_1試合平均失点数", "stats_ゴール期待値",
                          "stats_被ゴール期待値", "stats_1試合平均被ゴール期待値",
                          "flab_expected_for_xg", "flab_expected_against_xg"):
                row[f"{field}_{side}"] = 1.5
        self.rows = pd.DataFrame([row])

    def check(self):
        return check_inputs(self.rows, self.order, "toto9999", as_of="2026-10-09 12:00")

    def test_valid(self):
        self.assertEqual(self.check()["status"], "passed")

    def test_wrong_date_same_teams(self):
        self.rows.loc[0, "datetime"] = "2026-10-11 14:00"
        with self.assertRaisesRegex(ValueError, "開催日が不一致"):
            self.check()

    def test_old_year_and_finished_match(self):
        self.rows.loc[0, "datetime"] = "2025-10-10 14:00"
        with self.assertRaisesRegex(ValueError, "実行後7日以内"):
            self.check()

    def test_missing_stale_future_dates(self):
        for value in (None, "2026-09-19", "2026-10-11"):
            with self.subTest(value=value):
                self.rows.loc[0, "stats_asof"] = value
                with self.assertRaisesRegex(ValueError, "stats_asof"):
                    self.check()

    def test_missing_lab_provenance(self):
        self.rows = self.rows.drop(columns="flab_snapshot_date")
        with self.assertRaisesRegex(ValueError, "flab_snapshot_date"):
            self.check()

    def test_invalid_model_inputs(self):
        for value in (None, float("inf"), "(3位)", 0, -1):
            with self.subTest(value=value):
                self.rows["stats_ゴール期待値_home"] = value
                with self.assertRaisesRegex(ValueError, "stats_ゴール期待値_home"):
                    self.check()

    def test_legacy_final_probabilities_are_not_required(self):
        self.rows.loc[0, "prob_final_draw"] = 0.35
        self.assertEqual(self.check()["status"], "passed")
