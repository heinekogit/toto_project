import unittest
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "next_model"))
from build_next_buyplan import build_round, build_shadow_round


class NextBuyPlanTest(unittest.TestCase):
    def test_v23_reproduces_frozen_round_without_actual_results(self):
        snapshot = Path(__file__).resolve().parents[1] / "data/eval/toto_rounds/toto1663/snapshot"
        frame = pd.read_csv(snapshot / "next_predictions.csv")
        prior = pd.read_csv(snapshot / "buyplan.csv").sort_values("match_no")
        with patch("ticket_coverage_optimizer.optimize_ten_hit_coverage", side_effect=AssertionError("v2.3 must not call the >=10 optimizer")):
            plan, audit = build_round(frame, strategy="relative_draw_graded_v23")
        columns = [f"ticket{i:02d}" for i in range(1, 11)]
        for value in (0, 1, 2):
            np.testing.assert_array_equal(
                (plan[columns] == value).sum(axis=1),
                (prior[columns] == value).sum(axis=1),
            )
        self.assertNotIn("optimizer", audit["allocation"])
        self.assertFalse(audit["allocation"]["randomness_used"])
        self.assertFalse(audit["placement_score"]["randomness_used"])
        self.assertEqual(audit["placement_score"]["algorithm"], "deterministic_best_improvement")
        self.assertEqual(audit["ticket_display_order"], "placement_score_joint_descending")
        self.assertEqual(audit["placement_score"]["strength"], .50)
        self.assertTrue(audit["placement_score"]["counts_preserved"])
        self.assertGreaterEqual(audit["minimum_pairwise_hamming_distance"], 2)
        self.assertFalse(audit["placement_score"]["actual_results_used"])

    def test_v22_reorders_tickets_without_changing_portfolio(self):
        frame = pd.DataFrame([{
            "match_no": index + 1, "home_team": f"H{index}", "away_team": f"A{index}",
            "league": "j1", "next_prob_home": .46 if index % 2 else .38,
            "next_prob_draw": .24 if index % 2 else .27,
            "next_prob_away": .30 if index % 2 else .35,
            "final_lambda_home": 1.4, "final_lambda_away": 1.0,
            "next_prob_0_0": .08, "next_prob_1_1": .10, "next_prob_2_2": .03,
            "lab_basis_hint": "basis_balanced",
        } for index in range(13)])
        plan, audit = build_round(frame, strategy="probability_allocation_ge10_v1")
        from proportional_buyplan import build_proportional_tickets
        from ticket_coverage_optimizer import optimize_ten_hit_coverage
        p = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy()
        initial, _ = build_proportional_tickets(p)
        raw, _ = optimize_ten_hit_coverage(initial, p)
        displayed = plan[[f"ticket{i:02d}" for i in range(1, 11)]].to_numpy().T
        displayed = np.select([displayed == 1, displayed == 0, displayed == 2], [0, 1, 2])
        self.assertEqual(sorted(map(tuple, displayed)), sorted(map(tuple, raw)))
        joint = np.prod(p[np.arange(13)[None, :], displayed], axis=1)
        self.assertTrue(np.all(joint[:-1] >= joint[1:] - 1e-18))
        np.testing.assert_allclose(joint, audit["ticket_joint_probabilities"])
        self.assertEqual(audit["ticket_display_order"], "joint_probability_descending")

    def test_ticket01_is_argmax_and_no_fixed_draw_quota(self):
        rows = []
        for index in range(13):
            rows.append({
                "match_no": index + 1, "home_team": f"H{index}", "away_team": f"A{index}",
                "league": "j1", "next_prob_home": 0.50 if index else 0.34,
                "next_prob_draw": 0.20 if index else 0.35,
                "next_prob_away": 0.30 if index else 0.31,
                "final_lambda_home": 1.4, "final_lambda_away": 1.0,
                "next_prob_0_0": 0.08, "next_prob_1_1": 0.10, "next_prob_2_2": 0.03,
                "lab_basis_hint": "basis_balanced",
            })
        plan, audit = build_round(pd.DataFrame(rows))
        self.assertEqual(int(plan.loc[0, "ticket01"]), 0)
        self.assertTrue((plan.loc[1:, "ticket01"] == 1).all())
        self.assertFalse(audit["fixed_symbol_ratio_used"])
        self.assertFalse(audit["target_draw_count_used"])
        self.assertEqual(len({tuple(plan[f"ticket{i:02d}"]) for i in range(1, 11)}), 10)
        self.assertLessEqual(audit["pairwise_same_pick_max"], 11)

    def test_coverage_v1_respects_probability_floor(self):
        rows = []
        for index in range(13):
            rows.append({
                "match_no": index + 1, "home_team": f"H{index}", "away_team": f"A{index}",
                "league": "j1", "next_prob_home": 0.40, "next_prob_draw": 0.33,
                "next_prob_away": 0.27, "final_lambda_home": 1.2, "final_lambda_away": 1.1,
                "next_prob_0_0": 0.09, "next_prob_1_1": 0.11, "next_prob_2_2": 0.03,
                "lab_basis_hint": "basis_balanced",
            })
        _, audit = build_round(pd.DataFrame(rows), strategy="coverage_v1")
        top = audit["ticket_joint_probabilities"][0]
        self.assertTrue(all(value >= top * 0.50 - 1e-15 for value in audit["ticket_joint_probabilities"]))
        self.assertEqual(audit["coverage_objective"], "probability_weighted_new_symbols")

    def test_shadow_adds_five_distinct_non_purchase_tickets(self):
        rows = []
        for index in range(13):
            rows.append({
                "match_no": index + 1, "home_team": f"H{index}", "away_team": f"A{index}",
                "league": "j1", "next_prob_home": 0.40, "next_prob_draw": 0.33,
                "next_prob_away": 0.27, "final_lambda_home": 1.2, "final_lambda_away": 1.1,
                "next_prob_0_0": 0.09, "next_prob_1_1": 0.11, "next_prob_2_2": 0.03,
                "lab_basis_hint": "basis_balanced",
            })
        frame = pd.DataFrame(rows)
        official, _ = build_round(frame, strategy="coverage_v1")
        shadow, audit = build_shadow_round(frame)
        official_tickets = {tuple(official[f"ticket{i:02d}"]) for i in range(1, 11)}
        shadow_tickets = {tuple(shadow[f"ticket{i:02d}"]) for i in range(11, 16)}
        self.assertEqual(len(shadow_tickets), 5)
        self.assertTrue(official_tickets.isdisjoint(shadow_tickets))
        self.assertTrue(audit["official_tickets_unchanged"])
        self.assertTrue(all(value >= 0.35 - 1e-12 for value in audit["shadow_ticket_probability_ratios"]))
        self.assertGreaterEqual(audit["minimum_pairwise_hamming_distance"], 2)


if __name__ == "__main__":
    unittest.main()
