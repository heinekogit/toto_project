import unittest
import numpy as np
import pandas as pd
from buyplan_v30 import build

class V30Tests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'match_no':range(1,14),'home_team':[f'H{i}' for i in range(13)],
            'away_team':[f'A{i}' for i in range(13)],'next_prob_home':.3,'next_prob_draw':.4,'next_prob_away':.3})
    def test_draw_top_still_uses_HA_anchor_and_deterministic_ties(self):
        plan,diag,audit=build(self.frame().iloc[::-1])
        self.assertEqual(audit['selected_match_nos'],[1,2,3,4])
        self.assertTrue(plan.candidate00.eq(1).all())
        self.assertTrue(plan.ticket01.eq(plan.candidate00).all())
        self.assertEqual(len(set(diag.symbols)),16)
        self.assertAlmostEqual(audit['model_P13'],.7**4*.3**9)
        self.assertEqual(int(diag.D_count.sum()),32)
    def test_invalid_inputs_rejected(self):
        f=self.frame(); f.loc[0,'next_prob_home']=.4
        with self.assertRaises(ValueError):build(f)
        with self.assertRaises(ValueError):build(self.frame().iloc[:12])
    def test_result_columns_do_not_affect_output(self):
        f=self.frame(); p,d,a=build(f)
        f['result']=0;f['home_score']=99;f['away_score']=0
        p2,d2,a2=build(f)
        pd.testing.assert_frame_equal(p,p2);pd.testing.assert_frame_equal(d,d2)
        self.assertEqual(a,a2)

if __name__=='__main__':unittest.main()
