import unittest
import numpy as np
import pandas as pd
from expected_goals import score_distribution
from d_scoop import diagnose

class DiagnosticTests(unittest.TestCase):
    def frame(self,h=1.2,a=1.7):
        d=score_distribution(h,a); mix=.2
        return pd.DataFrame([dict(match_no=i,home_team=f'H{i}',away_team=f'A{i}',
            final_lambda_home=h,final_lambda_away=a,next_uncertainty_calibration_mix=mix,
            next_prob_home=(1-mix)*d['prob_home']+mix/3,
            next_prob_draw=(1-mix)*d['prob_draw']+mix/3,
            next_prob_away=(1-mix)*d['prob_away']+mix/3,
            next_prob_0_0=d['prob_0_0'],next_prob_1_1=d['prob_1_1'],next_prob_2_2=d['prob_2_2']) for i in range(1,14)])
    def test_home_away_swap_preserves_draw_and_reverses_margin(self):
        z,m=diagnose(self.frame()); r,mr=diagnose(self.frame(1.7,1.2))
        np.testing.assert_allclose(m,mr.transpose(0,2,1))
        np.testing.assert_allclose(z.P_D,r.P_D)
        np.testing.assert_allclose(z.P_margin_plus1,r.P_margin_minus1)
        np.testing.assert_allclose(z.goal_difference_mean,-r.goal_difference_mean)
        np.testing.assert_allclose(z.P_00+z.P_11+z.P_22+z.P_33plus,z.score_P_D)
        np.testing.assert_allclose(z.filter(regex=r'^goal_difference_[+-]').sum(axis=1),1)
        self.assertTrue(z.d_scoop_score.isna().all())
        self.assertTrue(z.d_scoop_selected.isna().all())
    def test_saved_values_checked_and_results_ignored(self):
        f=self.frame(); z,m=diagnose(f); f['result']=0
        other,mo=diagnose(f);pd.testing.assert_frame_equal(z,other)
        f.loc[0,'next_prob_0_0']+=.01
        with self.assertRaises(ValueError):diagnose(f)

if __name__=='__main__': unittest.main()
