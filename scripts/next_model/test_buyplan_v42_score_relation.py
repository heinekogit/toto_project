import unittest

import numpy as np
import pandas as pd

from analyze_score_structure import score_grid
from buyplan_v42_score_relation import current, draw_candidates, tickets


class BuyPlanV42Test(unittest.TestCase):
    def test_score_regions_and_d_branch_candidates(self):
        matrix,_=score_grid(1.4,1.2)
        f=pd.DataFrame(dict(match_no=range(1,14),home_team=['H']*13,away_team=['A']*13,
                            final_lambda_home=[1.4]*13,final_lambda_away=[1.2]*13,
                            next_prob_home=[.4]*13,next_prob_draw=[.3]*13,next_prob_away=[.3]*13,
                            next_prob_0_0=[matrix[0,0]]*13,next_prob_1_1=[matrix[1,1]]*13))
        copy=f.copy(deep=True)
        details=current(f)
        self.assertTrue(f.equals(copy))
        self.assertTrue(np.allclose(details[['raw_score_P_H','raw_score_P_D','raw_score_P_A']].sum(axis=1),1))
        self.assertTrue((details.home_one_goal>0).all())
        self.assertTrue((details.away_one_goal>0).all())
        d=draw_candidates(f)
        for i,row in d.iterrows():
            self.assertEqual(row.best_D_ticket[i],'D')
            self.assertEqual((row.D_count_floor,row.D_count_ceil),(3,3))
        plan=tickets(f)
        self.assertEqual(plan.symbols.nunique(),10)


if __name__=='__main__':unittest.main()
