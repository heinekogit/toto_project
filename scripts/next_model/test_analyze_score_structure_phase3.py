import unittest

import numpy as np
import pandas as pd

from analyze_score_structure import score_grid, structure
from analyze_score_structure_phase3 import compute, MEASURES


class Phase3Test(unittest.TestCase):
    def test_zero_delta_and_probability_partition(self):
        frame=pd.DataFrame([dict(round='test',match_no=1,home_team='H',away_team='A',
                                 lambda_home=1.5,lambda_away=.8,final_P_D=.27,
                                 uncertainty_mix=.2)])
        scenarios,edges,_,skipped=compute(frame)
        self.assertEqual(len(scenarios),17)
        self.assertEqual(len(edges),17*20)
        self.assertTrue(skipped.empty)
        baseline=structure(score_grid(1.5,.8)[0])
        zero=scenarios.loc[scenarios.scenario.eq('基準')].iloc[0]
        for name in MEASURES:
            self.assertAlmostEqual(zero[name],baseline[name])
            self.assertAlmostEqual(zero[f'delta_{name}'],0)
        self.assertTrue(np.allclose(scenarios.raw_P_H+scenarios.P_draw+scenarios.raw_P_A,1))
        self.assertTrue(np.allclose(scenarios.P_draw+scenarios.P_margin_1+scenarios.P_margin_2plus,1))
        self.assertAlmostEqual(scenarios.final_P_D_reference.iloc[0],.27)

    def test_home_away_swap_symmetry(self):
        for h,a in ((.8,1.5),(2.5,2.5),(2.,.8)):
            x=structure(score_grid(h,a)[0]);y=structure(score_grid(a,h)[0])
            self.assertAlmostEqual(x['P_draw'],y['P_draw'])
            for margin in ('1','2','3plus'):
                self.assertAlmostEqual(x[f'P_home_{margin}'],y[f'P_away_{margin}'])
                self.assertAlmostEqual(x[f'P_away_{margin}'],y[f'P_home_{margin}'])

    def test_invalid_lambda_reported_not_clipped(self):
        frame=pd.DataFrame([dict(round='test',match_no=1,home_team='H',away_team='A',
                                 lambda_home=.15,lambda_away=1.,final_P_D=.3,
                                 uncertainty_mix=0)])
        scenarios,_,_,skipped=compute(frame)
        self.assertGreater(len(skipped),0)
        self.assertNotIn(('H低下',.2),set(zip(scenarios.scenario,scenarios.delta)))
        self.assertIn('nonpositive_lambda',set(skipped.reason))


if __name__=='__main__':unittest.main()
