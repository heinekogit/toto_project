import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from validate_pattern_matchup import EXPECTED_HEADERS, PATTERNS, affinity, dataset, read_patterns


class PatternMatchupTests(unittest.TestCase):
    def test_parser_excludes_penalties_and_combines_set_pieces(self):
        table = [EXPECTED_HEADERS, ["横浜FM", 2, 1, 3, 4, 5, 6, 7, 8, 9, 10]]
        text = '<title>2026/27 J1 失点パターン</title><em class="uiDate">2026.10.26 update</em>'
        text += f"arrayToDataTable({table!r});arrayToDataTable({table!r});"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patterns.html"
            path.write_text(text)
            date, values, totals = read_patterns(path)
        self.assertEqual(date, pd.Timestamp("2026-10-26"))
        np.testing.assert_array_equal(values["横浜FM"], [4, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(totals["横浜FM"] - values["横浜FM"].sum(), 2)

    def feature_row(self, scored, conceded):
        return {**{f"for_{n}": scored[i] for i, n in enumerate(PATTERNS)},
                **{f"against_{n}": conceded[i] for i, n in enumerate(PATTERNS)},
                **{f"pool_{n}": 1 / 8 for n in PATTERNS}}

    def test_league_average_profile_has_no_matchup_advantage(self):
        frame = pd.DataFrame([self.feature_row([2]*8, [20, 0, 0, 0, 0, 0, 0, 0]),
                              self.feature_row([20, 0, 0, 0, 0, 0, 0, 0], [2]*8)])
        np.testing.assert_allclose(affinity(frame, 10), 0, atol=1e-12)

    def test_opponent_profile_changes_the_interaction(self):
        frame = pd.DataFrame([self.feature_row([20, 0, 0, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0, 0, 0]),
                              self.feature_row([20, 0, 0, 0, 0, 0, 0, 0], [0, 20, 0, 0, 0, 0, 0, 0])])
        a = affinity(frame, 10)
        self.assertGreater(a[0], 0)
        self.assertLess(a[1], 0)

    def test_same_day_and_future_snapshots_cannot_enter_prediction(self):
        metrics = pd.DataFrame({"flab_expected_for_xg": [1.5, 1.0],
                                "flab_expected_against_xg": [1.0, 1.5]}, index=["a", "b"])
        common = {"league": "j1", "goals": {"a": np.ones(8), "b": np.ones(8)},
                  "losses": {"a": np.ones(8), "b": np.ones(8)}, "pool": np.ones(8)/8,
                  "metrics": metrics, "asof": pd.Timestamp("2026-10-26")}
        sources = [{**common, "snapshot": day.replace("-", ""), "date": pd.Timestamp(day)}
                   for day in ("2026-10-30", "2026-10-31", "2026-11-02")]
        history = pd.DataFrame([{"match_id": "m", "datetime": pd.Timestamp("2026-10-31 15:00"),
                                 "home_key": "a", "away_key": "b", "home_score": 2, "away_score": 1}])
        frame, excluded = dataset(sources, {"j1": history})
        self.assertFalse(excluded)
        self.assertEqual(set(frame.snapshot), {"20261030"})


if __name__ == "__main__":
    unittest.main()
