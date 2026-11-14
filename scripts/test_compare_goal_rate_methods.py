import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "next_model"))
from goal_rate_adjustment import rates_for_match


class GoalRateComparisonTest(unittest.TestCase):
    def test_future_and_same_time_results_are_excluded(self):
        rows = []
        for day in range(1, 11):
            rows.append({"datetime": pd.Timestamp(f"2026-10-{day:02d} 14:00"),
                         "home_key": "長崎", "away_key": "広島",
                         "home_score": 1.0, "away_score": 0.0})
        rows.append({"datetime": pd.Timestamp("2026-10-11 14:00"),
                     "home_key": "長崎", "away_key": "広島",
                     "home_score": 9.0, "away_score": 0.0})
        history = pd.DataFrame(rows)
        rate = rates_for_match(history, "長崎", pd.Timestamp("2026-10-11 14:00"), "for", scale=1.5)
        self.assertEqual(rate["n"], 10)
        self.assertEqual(rate["ordinary"], 1.0)


if __name__ == "__main__":
    unittest.main()
