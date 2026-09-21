import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parent / "eval" / "06_evaluate_d_filter_candidates.py"
SPEC = importlib.util.spec_from_file_location("evaluate_d_filter_candidates", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DFilterCandidateTest(unittest.TestCase):
    def test_false_draw_filter_uses_rank2_side(self):
        row = pd.Series(
            {
                "predicted_result": 0,
                "rank2_symbol": 2,
                "match_purchase_type": "j1_false_draw_watch",
            }
        )
        result, changed, reason = MODULE.apply_policy(row, "false_draw_only")
        self.assertEqual(result, 2)
        self.assertTrue(changed)
        self.assertIn("rank2_side", reason)

    def test_false_draw_filter_does_not_change_other_type(self):
        row = pd.Series(
            {
                "predicted_result": 0,
                "rank2_symbol": 1,
                "match_purchase_type": "chaos_side_watch",
            }
        )
        result, changed, _ = MODULE.apply_policy(row, "false_draw_only")
        self.assertEqual(result, 0)
        self.assertFalse(changed)

    def test_topology_guard_keeps_strong_draw(self):
        row = pd.Series(
            {
                "predicted_result": 0,
                "rank2_symbol": 1,
                "p_home": 0.28,
                "p_draw": 0.44,
                "p_away": 0.28,
                "lab_draw_tension_score": 0.75,
                "lab_stall_compactness_score": 0.70,
                "lab_stall_weight": 0.50,
                "lab_volatility_score": 0.20,
            }
        )
        result, changed, reason = MODULE.apply_policy(row, "topology_guard")
        self.assertEqual(result, 0)
        self.assertFalse(changed)
        self.assertIn("keep_strong_D", reason)

    def test_topology_guard_changes_weak_draw(self):
        row = pd.Series(
            {
                "predicted_result": 0,
                "rank2_symbol": 1,
                "p_home": 0.34,
                "p_draw": 0.36,
                "p_away": 0.30,
                "lab_draw_tension_score": 0.50,
                "lab_stall_compactness_score": 0.50,
                "lab_stall_weight": 0.35,
                "lab_volatility_score": 0.40,
            }
        )
        result, changed, reason = MODULE.apply_policy(row, "topology_guard")
        self.assertEqual(result, 1)
        self.assertTrue(changed)
        self.assertIn("filter_weak_D", reason)


if __name__ == "__main__":
    unittest.main()
