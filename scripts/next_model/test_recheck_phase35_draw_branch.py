import unittest

import numpy as np
import pandas as pd

from recheck_phase35_draw_branch import update_probabilities,rank_tickets


class DrawBranchRecheckTests(unittest.TestCase):
    def test_target_result_is_not_used_to_make_probabilities(self):
        target=pd.DataFrame([dict(round='r',match_no=1,rare=True,broad_only=False,near_HA=False,
                                  baseline_P_H=.45,baseline_P_D=.27,baseline_P_A=.28,actual_symbol='D')])
        train=pd.DataFrame([dict(round='old',rare=True,broad_only=False,near_HA=False,actual_symbol='D')])
        first,_=update_probabilities(target,train,3.,False)
        target.loc[0,'actual_symbol']='H'
        second,_=update_probabilities(target,train,3.,False)
        np.testing.assert_array_equal(first,second)
        self.assertGreater(first[0,1],.27)

    def test_ten_tickets_unique_without_D_quota(self):
        p=np.tile(np.array([.46,.25,.29]),(13,1))
        tickets,_=rank_tickets(p)
        self.assertEqual(tickets.shape,(10,13))
        self.assertEqual(len({tuple(x) for x in tickets}),10)
        self.assertEqual(int((tickets==1).sum()),0)


if __name__=='__main__':unittest.main()
