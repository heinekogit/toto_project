import unittest

import numpy as np
import pandas as pd

from shadow_v41_score_neighbors import build, neighbor_state


class ScoreNeighborShadowTest(unittest.TestCase):
    def test_score_adjacency_and_ticket_uniqueness(self):
        f = pd.DataFrame(dict(match_no=range(1,14),home_team=["H"]*13,away_team=["A"]*13,
                              final_lambda_home=[1.4]*13,final_lambda_away=[1.1]*13,
                              next_prob_home=[0.4]*13,next_prob_draw=[0.3]*13,next_prob_away=[0.3]*13))
        p, q, edges = neighbor_state(f)
        self.assertTrue(np.allclose(q.sum(axis=1),1))
        self.assertTrue((q[:,1]>p[:,1]).all())
        one_zero = edges[(edges.match_no==1)&edges.score.eq("1-0")].iloc[0]
        two_one = edges[(edges.match_no==1)&edges.score.eq("2-1")].iloc[0]
        three_two = edges[(edges.match_no==1)&edges.score.eq("3-2")].iloc[0]
        self.assertEqual((one_zero.lower_draw,one_zero.upper_draw),("0-0","1-1"))
        self.assertEqual((two_one.lower_draw,two_one.upper_draw),("1-1","2-2"))
        self.assertEqual((three_two.lower_draw,three_two.upper_draw),("2-2","3-3"))
        self.assertAlmostEqual(one_zero.hypothetical_to_lower+one_zero.hypothetical_to_upper,one_zero.hypothetical_transfer)
        self.assertFalse(edges.score.eq("2-0").any())
        self.assertTrue((edges.hypothetical_transfer>=0).all())
        _, _, tickets, summary = build(f)
        self.assertEqual(len(summary),4)
        self.assertTrue((tickets.groupby("state_weight").symbols.nunique()==10).all())


if __name__=="__main__":
    unittest.main()
