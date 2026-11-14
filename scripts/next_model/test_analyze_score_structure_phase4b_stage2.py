import unittest
import numpy as np
from analyze_score_structure_phase4b_stage2 import shift,scenario,portfolio_probability

class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.p=np.tile([.45,.25,.30],(13,1))
    def test_probability_transfer(self):
        for symbol in range(3):
            for direction in (-1,1):
                q=shift(self.p,2,symbol,.03*direction)
                np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-12)
                self.assertTrue(np.all(q>=0))
                self.assertAlmostEqual(q[2,symbol]-self.p[2,symbol],.03*direction)
                np.testing.assert_array_equal(q[1],self.p[1])
    def test_scenarios_and_fixed_tickets(self):
        tickets=np.tile([0]*13,(10,1))
        baseline=portfolio_probability(self.p,tickets)
        for kind in ['D上昇','D低下','最高確率過信','最高確率過小評価','H一斉上昇','独立ランダム','共通方向ランダム']:
            q=scenario(self.p,kind,.05,np.random.default_rng(1))
            np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-12)
            self.assertTrue(np.all(q>=0))
            self.assertTrue(np.isfinite(portfolio_probability(q,tickets)))
        self.assertAlmostEqual(baseline,10*.45**13)

if __name__=='__main__':unittest.main()
