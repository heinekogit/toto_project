import itertools
import unittest

import numpy as np
import pandas as pd

from expected_goals import score_distribution
from simulate_free_structural_draw import structure, placement_scores, metrics, proximity_components
from simulate_free_portfolio_1664 import top_joint_candidates


class StructuralDrawTest(unittest.TestCase):
    def test_one_sided_one_goal_wins_support_draw_without_opposite_wins(self):
        # Strong favourite, almost no one-goal underdog outcomes.
        b, nh, na, near, combined = proximity_components([.8], [.05], [.72], [.001])
        self.assertAlmostEqual(near[0], .9)
        self.assertGreater(combined[0], .9)
        self.assertLess(b[0], .12)
        # Removing favourite one-goal mass, at the same H/A balance, must
        # reduce draw proximity. The former balance-only score missed this.
        _, _, _, far, combined_far = proximity_components([.8], [.05], [.08], [.001])
        self.assertLess(far[0], near[0])
        self.assertLess(combined_far[0], combined[0])

    def test_proximity_is_symmetric_under_home_away_exchange(self):
        first = proximity_components([.8], [.05], [.72], [.001])
        second = proximity_components([.05], [.8], [.001], [.72])
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[3], second[3])
        np.testing.assert_array_equal(first[4], second[4])

    def test_low_total_and_balance_act_separately(self):
        rows = []
        for h, a in [(1.3, 1.3), (.7, .7), (1.8, .8)]:
            d = score_distribution(h, a)
            row = {"final_lambda_home": h, "final_lambda_away": a}
            for s in ("home", "draw", "away", "0_0", "1_1", "2_2"):
                row["next_prob_" + s] = d["prob_" + s]
            rows.append(row)
        f = structure(pd.DataFrame(rows))
        self.assertGreater(f.D_structure_support[1], f.D_structure_support[0])
        self.assertAlmostEqual(f.low_total_le2[0], f.low_total_le2[2])
        self.assertGreater(f.D_structure_support[0], f.D_structure_support[2])

    def test_structural_score_can_change_choice_without_changing_forecast(self):
        p = np.array([[.40, .28, .32], [.25, .50, .25]])
        before = p.copy()
        s = placement_scores(p, np.array([.6, .6]), 1.0)
        self.assertGreater(s[0, 1], s[0, 2])
        np.testing.assert_array_equal(p, before)
        np.testing.assert_array_equal(s[:, [0, 2]], p[:, [0, 2]])
        self.assertEqual(s[1, 1], p[1, 1])

    def test_exact_selection_matches_exhaustive_small_case(self):
        scores = np.array([[.4, .41, .25], [.3, .5, .4], [.8, .4, .3]])
        tickets, masses = top_joint_candidates(scores, 10)
        exhaustive = sorted(((-float(np.prod(scores[np.arange(3), t])), t)
                             for t in itertools.product(range(3), repeat=3)))[:10]
        np.testing.assert_array_equal(tickets, [t for _, t in exhaustive])
        np.testing.assert_allclose(masses, [-m for m, _ in exhaustive])

    def test_duplicate_exact_result_tickets_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics(np.full((13, 3), 1 / 3), np.zeros((10, 13), dtype=int), None)


if __name__ == '__main__':
    unittest.main()
