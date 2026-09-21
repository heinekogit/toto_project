import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parent / "eval" / "19_build_market_alignment.py"
SPEC = importlib.util.spec_from_file_location("market_alignment", MODULE_PATH)
market_alignment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(market_alignment)


class MarketAlignmentTest(unittest.TestCase):
    def test_build_rows_maps_toto_symbols(self):
        shares = [{"1": 0.60, "0": 0.25, "2": 0.15} for _ in range(13)]
        shares[1] = {"1": 0.20, "0": 0.30, "2": 0.50}
        actual = pd.DataFrame(
            {
                "match_no": range(1, 14),
                "home_team": [f"H{i}" for i in range(1, 14)],
                "away_team": [f"A{i}" for i in range(1, 14)],
                "result": [1, 0] + [1] * 11,
                "status": ["OK"] * 13,
            }
        )
        rows = market_alignment.build_match_rows("toto9999", shares, actual)
        self.assertEqual(int(rows.loc[0, "market_favorite_hit"]), 1)
        self.assertEqual(int(rows.loc[1, "market_favorite_hit"]), 0)
        self.assertAlmostEqual(float(rows.loc[1, "actual_market_share"]), 0.30)
        self.assertEqual(int(rows.loc[0, "strong_favorite_60"]), 1)

    def test_rejects_incomplete_results(self):
        shares = [{"1": 0.40, "0": 0.30, "2": 0.30} for _ in range(13)]
        actual = pd.DataFrame({"match_no": [1], "result": [1], "status": ["OK"]})
        with self.assertRaisesRegex(ValueError, "expected 13"):
            market_alignment.build_match_rows("toto9999", shares, actual)


if __name__ == "__main__":
    unittest.main()
