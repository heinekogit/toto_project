import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "next_model"))
from shadow_free_coverage import build_shadow


class ShadowFreeCoverageTest(unittest.TestCase):
    def test_produces_ten_distinct_tickets_without_changing_probabilities(self):
        frame = pd.DataFrame({
            "match_no": range(1, 14),
            "next_prob_home": [0.45] * 13,
            "next_prob_draw": [0.27] * 13,
            "next_prob_away": [0.28] * 13,
            "home_acl_fatigue": [0.0] * 13,
            "away_acl_fatigue": [0.0] * 13,
            "final_lambda_total": [2.5] * 13,
        })
        original = frame.copy(deep=True)
        result, audit = build_shadow(frame)
        columns = [f"ticket{i:02d}" for i in range(31, 41)]
        tickets = [tuple(result[column]) for column in columns]
        self.assertEqual(len(set(tickets)), 10)
        self.assertTrue(np.isin(result[columns].to_numpy(), [0, 1, 2]).all())
        self.assertEqual(audit["draw_count_policy"], "none; selected by modeled 11/12/13 coverage")
        self.assertEqual(audit["candidate_source"], "top joint tickets plus draws sampled from unchanged next_prob_*")
        self.assertNotIn("tilt_factors_search_only", audit)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(audit["probability_source"], "current next_prob_* unchanged")
        other, other_audit = build_shadow(frame)
        pd.testing.assert_frame_equal(result, other)
        self.assertEqual(audit, other_audit)

    def test_acl_does_not_change_shadow_probability(self):
        frame = pd.DataFrame({
            "match_no": range(1, 14),
            "next_prob_home": [0.55] * 13,
            "next_prob_draw": [0.25] * 13,
            "next_prob_away": [0.20] * 13,
            "home_acl_fatigue": [8.0] + [0.0] * 12,
            "away_acl_fatigue": [0.0] * 13,
            "final_lambda_total": [2.4] * 13,
        })
        _, audit = build_shadow(frame)
        self.assertNotIn("acl_draw_log_odds_weight", audit)
        self.assertNotIn("draw_probability_after_by_match", audit)
        self.assertEqual(frame.loc[0, "next_prob_draw"], 0.25)


if __name__ == "__main__":
    unittest.main()
