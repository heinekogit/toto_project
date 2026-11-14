import unittest
import numpy as np
import pandas as pd
from simulate_gated_structural_draw import Gate, eligibility
from simulate_free_structural_draw import placement_scores


class GateTest(unittest.TestCase):
    def test_gate_branches_and_boundary(self):
        f=pd.DataFrame({
            'lambda_home':[.9,1.5,1.2,1.5,1.5,1.5],
            'lambda_away':[.1,1.0,1.2,1.,1.,1.],
            'low_total_le2':[.92,.55,.55,.49,.55,.5],
            'full_win_balance':[.1,.2,.95,.95,.2,.2],
            'favoured_one_goal_share':[.5,.65,.3,.65,.59,.6]})
        mask,_=eligibility(f,Gate())
        # 00 mode; one-sided near win; balance alone; high-total exclusion;
        # near below cutoff; inclusive boundary.
        np.testing.assert_array_equal(mask,[True,True,True,False,False,True])

    def test_excluded_draw_is_not_banned_or_boosted(self):
        p=np.array([[.25,.5,.25],[.4,.28,.32]])
        scores=placement_scores(p,np.array([0.,.5]),2.)
        np.testing.assert_array_equal(scores[0],p[0])
        self.assertEqual(int(scores[0].argmax()),1)
        self.assertGreater(scores[1,1],p[1,1])


if __name__=='__main__':unittest.main()
