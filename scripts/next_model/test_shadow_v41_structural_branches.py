import unittest

import numpy as np
import pandas as pd

from shadow_v41_structural_branches import evaluate


class StructuralBranchesTest(unittest.TestCase):
    def test_d_is_placed_in_each_required_match_without_changing_probabilities(self):
        f=pd.DataFrame(dict(match_no=range(1,14),home_team=["H"]*13,away_team=["A"]*13,
                            final_lambda_home=[1.3]*13,final_lambda_away=[1.2]*13,
                            next_prob_home=[.4]*13,next_prob_draw=[.3]*13,next_prob_away=[.3]*13))
        original=f.copy(deep=True)
        matches,tickets,summary=evaluate(f)
        self.assertTrue(f.equals(original))
        self.assertEqual(len(summary),11)
        for k in range(11):
            group=tickets[tickets.structural_top_k.eq(k)]
            self.assertEqual(len(group),10)
            self.assertEqual(group.symbols.nunique(),10)
            for n in matches.head(k).match_no:
                self.assertIn("D",set(group[f"match_{int(n):02d}"]))
        self.assertTrue(np.all(summary.P_loss_vs_unrestricted>=-1e-15))


if __name__=="__main__":
    unittest.main()
