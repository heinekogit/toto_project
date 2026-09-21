import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).parent / "eval/18_build_shadow_buyplan_candidates.py"
SPEC = importlib.util.spec_from_file_location("shadow_buyplans", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ShadowBuyplanCandidatesTest(unittest.TestCase):
    def test_reference_for_another_round_is_rejected(self):
        order = pd.DataFrame({"match_no": range(1, 14), "home_team": [f"H{i}" for i in range(13)],
                              "away_team": [f"A{i}" for i in range(13)]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = order.copy(); wrong.loc[0, "home_team"] = "WRONG"
            wrong.assign(match_id="x", datetime="2026-01-01").to_csv(root / "predictions.csv", index=False)
            order.assign(**{f"ticket{i:02d}": 1 for i in range(1, 11)}).to_csv(root / "buyplan.csv", index=False)
            with self.assertRaises(ValueError):
                MODULE.validate_reference(root, order, "toto-test")

    def test_ticket_table_contains_ten_candidates(self):
        plan = pd.DataFrame({"match_no": [1], "home_team": ["H"], "away_team": ["A"],
                             **{f"ticket{i:02d}": [i % 3] for i in range(1, 11)}})
        html = MODULE.ticket_table("test", plan)
        self.assertIn("候補01", html)
        self.assertIn("候補10", html)

    def test_ticket_table_highlights_differences_from_current(self):
        current = pd.DataFrame({"match_no": [1], "home_team": ["H"], "away_team": ["A"],
                                **{f"ticket{i:02d}": [1] for i in range(1, 11)}})
        shadow = current.copy()
        shadow.loc[0, "ticket03"] = 0
        html = MODULE.ticket_table("shadow", shadow, current)
        self.assertIn('class="changed"', html)
        self.assertIn("現行との差分: 1セル", html)
        self.assertIn("現行: 1", html)


if __name__ == "__main__":
    unittest.main()
