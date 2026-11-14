"""The condition-S display must reproduce the official BuyPlan tickets exactly."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from condition_s_buyplan_shadow import SYMBOLS, TICKETS, build_view, write


ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "data/purchase_reference"


class ConditionSBuyPlanShadowTest(unittest.TestCase):
    def test_saved_tickets_unchanged_in_shadow(self):
        plan = pd.read_csv(SAVED / "buyplan.csv")
        forecast = pd.read_csv(SAVED / "next_predictions.csv")
        original = plan[TICKETS].copy(deep=True)
        view, audit = build_view(plan, forecast)
        self.assertEqual(audit["S_count"], 1)
        self.assertEqual(len(view), 13)
        for number, column in enumerate(TICKETS, 1):
            self.assertEqual(view[f"候補{number:02d}"].tolist(),
                             [SYMBOLS[int(value)] for value in original[column]])
        pd.testing.assert_frame_equal(plan[TICKETS], original)
        with tempfile.TemporaryDirectory() as directory:
            write(plan, forecast, "toto1670", Path(directory))
            html = (Path(directory) / "shadow_condition_s_buyplan.html").read_text()
            self.assertIn("正式BuyPlan v2.4", html)
            self.assertIn("研究用v4.0配券ではありません", html)
            exported = pd.read_csv(Path(directory) / "shadow_condition_s_buyplan.csv")
            self.assertEqual(exported["条件S"].eq("該当").sum(), 1)

    def test_match_identity_mismatch_fails(self):
        plan = pd.read_csv(SAVED / "buyplan.csv")
        forecast = pd.read_csv(SAVED / "next_predictions.csv")
        forecast.loc[0, "home_team"] = "別チーム"
        with self.assertRaisesRegex(ValueError, "match identity mismatch"):
            build_view(plan, forecast)


if __name__ == "__main__":
    unittest.main()
