"""Exercise date logic without importing the prediction script's live pipeline."""
import ast
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).with_name("11_prediction_01.py")


def date_factor():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = {"_safe_float_value", "_absence_uncertainty_weeks", "_absence_date_factor"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == 3
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    scope = {"pd": pd}
    exec(compile(module, str(SCRIPT), "exec"), scope)
    return scope["_absence_date_factor"]


class AbsenceReturnDateTest(unittest.TestCase):
    def test_card_ends_on_return_date(self):
        factor = date_factor()
        start, returned = "2026-10-24", "2026-10-31"
        self.assertEqual(factor("2026-10-23", start, returned, 1, "card"), 0.0)
        self.assertEqual(factor("2026-10-30", start, returned, 1, "card"), 1.0)
        self.assertEqual(factor("2026-10-31", start, returned, 1, "card"), 0.0)
        self.assertEqual(factor("2026-11-01", start, returned, 1, "card"), 0.0)

    def test_injury_keeps_existing_fade(self):
        factor = date_factor()
        start, returned = "2026-10-01", "2026-10-29"
        self.assertEqual(factor("2026-10-29", start, returned, 4, "injury"), 0.5)
        self.assertGreater(factor("2026-10-30", start, returned, 4, "injury"), 0.0)
        self.assertEqual(factor("2026-11-05", start, returned, 4, "injury"), 0.0)


if __name__ == "__main__":
    unittest.main()
