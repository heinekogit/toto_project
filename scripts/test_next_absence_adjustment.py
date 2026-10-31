import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "next_model"))
from absence_adjustment import apply_absence_to_lambdas


class AbsenceAdjustmentTests(unittest.TestCase):
    def test_known_absence_lowers_own_goal_expectation(self):
        row = {"absence_missing_home": False, "absence_missing_away": False,
               "absence_effective_attack_home": 0.2, "absence_effective_defense_home": 0.1,
               "absence_effective_attack_away": 0.0, "absence_effective_defense_away": 0.0}
        result = apply_absence_to_lambdas(row, 1.4, 1.1)
        self.assertAlmostEqual(result["final_lambda_home"], 1.35)
        self.assertAlmostEqual(result["final_lambda_away"], 1.125)

    def test_missing_absence_does_not_use_imputed_baseline(self):
        row = {"absence_missing_home": True, "absence_missing_away": False,
               "absence_effective_attack_home": 0.03, "absence_effective_defense_home": 0.02,
               "absence_effective_attack_away": 0.0, "absence_effective_defense_away": 0.0}
        result = apply_absence_to_lambdas(row, 1.4, 1.1)
        self.assertEqual(result["absence_attack_used_home"], 0.0)
        self.assertEqual(result["absence_defense_used_home"], 0.0)
        self.assertAlmostEqual(result["final_lambda_home"], 1.4)

    def test_known_missing_impact_fails(self):
        row = {"absence_missing_home": False, "absence_missing_away": False,
               "absence_effective_attack_away": 0.0, "absence_effective_defense_away": 0.0}
        with self.assertRaisesRegex(ValueError, "absence_effective_attack_home"):
            apply_absence_to_lambdas(row, 1.4, 1.1)


if __name__ == "__main__":
    unittest.main()
