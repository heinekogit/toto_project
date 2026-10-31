import sys
import unittest
from pathlib import Path

import pandas as pd

NEXT_MODEL = Path(__file__).parent / "next_model"
sys.path.insert(0, str(NEXT_MODEL))

from build_validation_dataset import select_matches
from draw_filter import classify_draw, classify_draw_v1, classify_draw_v2
from expected_goals import base_lambdas, base_lambdas_v1, score_distribution
from lab_matchup import apply_lab_matchup, apply_lab_matchup_v1


class NextModelTest(unittest.TestCase):
    def test_draw_filter_v2_legacy_structure_has_limited_authority(self):
        row = {
            "next_prob_home": 0.38, "next_prob_draw": 0.26, "next_prob_away": 0.36,
            "final_lambda_home": 1.20, "final_lambda_away": 1.10,
            "next_prob_0_0": 0.10, "next_prob_1_1": 0.12, "next_prob_2_2": 0.03,
            "lab_basis_hint": "split_side",
        }
        self.assertEqual(classify_draw_v2(row).draw_class, "D-cover")
        row["lab_basis_hint"] = "flat_draw_trap"
        self.assertEqual(classify_draw_v2(row).draw_class, "D-reject")

    def test_draw_filter_v1_does_not_use_legacy_labels(self):
        row = {
            "next_prob_home": 0.36, "next_prob_draw": 0.30, "next_prob_away": 0.34,
            "final_lambda_home": 1.05, "final_lambda_away": 1.00,
            "next_prob_0_0": 0.12, "next_prob_1_1": 0.13, "next_prob_2_2": 0.03,
            "match_purchase_type": "j1_false_draw_watch", "lab_basis_hint": "draw_compressed",
        }
        first = classify_draw_v1(row)
        row["match_purchase_type"] = "anything_else"
        row["lab_basis_hint"] = "anything_else"
        second = classify_draw_v1(row)
        self.assertEqual(first, second)
        self.assertIn(first.draw_class, {"D-core", "D-cover"})

    def test_lab_shot_matchup_uses_numeric_shot_evidence(self):
        row = {
            "flab_expected_for_xg_home": 1.44, "flab_expected_against_xg_away": 1.0,
            "flab_expected_for_xg_away": 1.0, "flab_expected_against_xg_home": 1.0,
            "flab_chance_shots_home": 10.0, "flab_chance_shot_conversion_home": 10.0,
            "flab_chance_allowed_shots_away": 10.0,
            "flab_chance_allowed_shot_conversion_away": 10.0,
            "flab_chance_shots_away": 10.0, "flab_chance_shot_conversion_away": 10.0,
            "flab_chance_allowed_shots_home": 10.0,
            "flab_chance_allowed_shot_conversion_home": 10.0,
        }
        result = apply_lab_matchup_v1(row, 1.0, 1.0, weight=0.25)
        self.assertAlmostEqual(result["lab_xg_matchup_lambda_home"], 1.2)
        self.assertAlmostEqual(result["lab_shot_matchup_lambda_home"], 1.0)
        self.assertAlmostEqual(result["lab_matchup_lambda_home"], 1.2 ** 0.5)

    def test_lab_matchup_is_auditable_log_blend(self):
        row = {
            "flab_expected_for_xg_home": 1.44,
            "flab_expected_against_xg_away": 1.00,
            "flab_expected_for_xg_away": 0.81,
            "flab_expected_against_xg_home": 1.00,
        }
        result = apply_lab_matchup(row, 1.0, 1.0, weight=0.25)
        self.assertAlmostEqual(result["lab_matchup_lambda_home"], 1.2)
        self.assertAlmostEqual(result["lab_matchup_lambda_away"], 0.9)
        self.assertAlmostEqual(result["final_lambda_home"], 1.2 ** 0.25)
        self.assertAlmostEqual(result["final_lambda_away"], 0.9 ** 0.25)

    def test_select_matches_uses_normalized_team_pair_and_actual_order(self):
        homes = [f"H{i}" for i in range(1, 14)]
        aways = [f"A{i}" for i in range(1, 14)]
        predictions = pd.DataFrame({
            "match_id": list(reversed([f"m{i}" for i in range(1, 14)])),
            "datetime": ["2026-01-01"] * 13, "league": ["J1"] * 13,
            "home_team": list(reversed(homes[:-1] + ["Ｈ１３"])),
            "away_team": list(reversed(aways[:-1] + ["Ａ１３"])),
        })
        actual = pd.DataFrame({
            "match_no": range(1, 14), "home_team": homes, "away_team": aways,
            "home_score": [1] * 13, "away_score": [0] * 13, "result": [1] * 13,
        })
        selected = select_matches(predictions, actual, "test")
        self.assertEqual(selected["match_id"].tolist(), [f"m{i}" for i in range(1, 14)])
        self.assertEqual(selected["actual_result"].tolist(), [1] * 13)

    def test_base_lambdas_use_attack_against_opponent_defence(self):
        row = pd.Series({
            "stats_1試合平均得点数_home": 2.0,
            "stats_1試合平均失点数_away": 0.5,
            "stats_1試合平均得点数_away": 0.5,
            "stats_1試合平均失点数_home": 2.0,
        })
        home, away, _ = base_lambdas(row)
        self.assertAlmostEqual(home, 1.0)
        self.assertAlmostEqual(away, 1.0)

    def test_score_distribution_is_normalized(self):
        result = score_distribution(1.4, 1.1)
        self.assertAlmostEqual(result["prob_home"] + result["prob_draw"] + result["prob_away"], 1.0)
        self.assertGreater(result["grid_mass_before_normalization"], 0.999)

    def test_v1_converts_cumulative_xg_to_per_game(self):
        row = pd.Series({
            "stats_1試合平均得点数_home": 1.0, "stats_1試合平均失点数_home": 1.0,
            "stats_1試合平均得点数_away": 1.0, "stats_1試合平均失点数_away": 1.0,
            "stats_ゴール期待値_home": 10.0, "stats_被ゴール期待値_home": 10.0,
            "stats_1試合平均被ゴール期待値_home": 1.0,
            "stats_ゴール期待値_away": 20.0, "stats_被ゴール期待値_away": 10.0,
            "stats_1試合平均被ゴール期待値_away": 1.0,
        })
        home, away, reason = base_lambdas_v1(row)
        self.assertAlmostEqual(home, 1.0)
        self.assertAlmostEqual(away, 1.5 ** 0.5)
        self.assertIn("games=10", reason)

    def test_draw_filter_only_returns_metadata(self):
        row = {
            "next_prob_home": 0.31, "next_prob_draw": 0.40, "next_prob_away": 0.29,
            "base_lambda_total": 2.1, "next_prob_0_0": 0.12,
            "next_prob_1_1": 0.15, "next_prob_2_2": 0.03,
            "match_purchase_type": "neutral", "lab_basis_hint": "basis_balanced",
        }
        before = dict(row)
        assessment = classify_draw(row)
        self.assertEqual(assessment.draw_class, "D-core")
        self.assertEqual(row, before)

    def test_false_draw_is_classified_without_replacing_prediction(self):
        row = {
            "next_prob_home": 0.30, "next_prob_draw": 0.39, "next_prob_away": 0.31,
            "base_lambda_total": 2.0, "next_prob_0_0": 0.10,
            "next_prob_1_1": 0.14, "next_prob_2_2": 0.03,
            "match_purchase_type": "j1_false_draw_watch", "lab_basis_hint": "draw_compressed",
        }
        assessment = classify_draw(row)
        self.assertEqual(assessment.draw_class, "D-reject")
        self.assertEqual(max(("H", "D", "A"), key=lambda s: row[{"H": "next_prob_home", "D": "next_prob_draw", "A": "next_prob_away"}[s]]), "D")


if __name__ == "__main__":
    unittest.main()
