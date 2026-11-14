import unittest

import numpy as np
import pandas as pd

from shadow_v40_step4 import build


class ShadowV40Test(unittest.TestCase):
    def test_independent_tickets_and_original_probability_mass(self):
        p = np.tile([0.5, 0.3, 0.2], (13, 1))
        f = pd.DataFrame(dict(match_no=range(1, 14), home_team=[f"H{i}" for i in range(13)],
                              away_team=[f"A{i}" for i in range(13)],
                              next_prob_home=p[:, 0], next_prob_draw=p[:, 1], next_prob_away=p[:, 2],
                              next_prob_0_0=np.arange(13)/100, next_prob_1_1=np.arange(12, -1, -1)/100))
        o = f[["match_no", "home_team", "away_team", "next_prob_home", "next_prob_draw", "next_prob_away"]].copy()
        for i in range(1, 11):
            o[f"ticket{i:02d}"] = 1 if i <= 5 else 2
        original = o.copy(deep=True)
        shadow, audit = build(f, o)
        self.assertTrue(o.equals(original))
        self.assertEqual(len(set(audit["shadow_tickets"])), 10)
        self.assertEqual(len(shadow), 13)
        self.assertEqual(audit["condition_S_matches"], list(range(1, 8)))
        self.assertAlmostEqual(audit["official_final_model_prob13"], 5 * 0.5**13 + 5 * 0.2**13)
        self.assertEqual(audit["shadow_tickets"], build(f, o)[1]["shadow_tickets"])


if __name__ == "__main__":
    unittest.main()
