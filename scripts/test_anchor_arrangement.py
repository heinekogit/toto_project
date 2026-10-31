import unittest
import numpy as np
from next_model.anchor_arrangement import arrange_from_anchor, relative_costs, rearrange_buyplan
import pandas as pd


class AnchorTests(unittest.TestCase):
    def test_relative_relationship(self):
        c = relative_costs([[.4,.2,.4],[.7,.2,.1]])
        self.assertAlmostEqual(np.exp(-c[0,1]), .5)
        self.assertAlmostEqual(np.exp(-c[1,1]), 2/7)

    def test_fixed_allocations_and_invariants(self):
        rng = np.random.default_rng(3)
        p = rng.dirichlet([3,2,3], size=13)
        t = rng.integers(0,3,(10,13))
        original = t.copy()
        actual = rng.integers(0,3,13)
        for mode in ['balanced','graded']:
            out, audit = arrange_from_anchor(t,p,mode,steps=1500)
            np.testing.assert_array_equal(t, original)
            for s in range(3):
                np.testing.assert_array_equal((out==s).sum(axis=0),(t==s).sum(axis=0))
            self.assertEqual(int((out==actual).sum()), int((t==actual).sum()))
            self.assertGreaterEqual(audit['minimum_pairwise_hamming_distance'],2)
            self.assertLessEqual(audit['objective_after'],audit['objective_before'] + 1e-12)
            np.testing.assert_array_equal(out, arrange_from_anchor(t,p,mode,steps=1500)[0])
            ratio=np.prod(p[np.arange(13),out],axis=1)/np.prod(p.max(axis=1))
            np.testing.assert_allclose(ratio,audit['anchor_probability_ratios'])

    def test_buyplan_adapter(self):
        rng = np.random.default_rng(17)
        symbols = rng.integers(0,3,(10,13))
        p = rng.dirichlet([3,2,3],size=13)
        plan = pd.DataFrame(p,columns=['next_prob_home','next_prob_draw','next_prob_away'])
        plan['match_no'] = range(1,14)
        plan['home_team'] = 'preserved'
        cols = [f'ticket{i:02}' for i in range(1,11)]
        for i,c in enumerate(cols): plan[c] = np.array([1,0,2])[symbols[i]]
        before = plan.copy(deep=True)
        result,_ = rearrange_buyplan(plan,steps=100)
        pd.testing.assert_frame_equal(plan,before)
        pd.testing.assert_frame_equal(result.drop(columns=cols),plan.drop(columns=cols))
        for s in [0,1,2]:
            np.testing.assert_array_equal((result[cols]==s).sum(axis=1),(plan[cols]==s).sum(axis=1))

    def test_invalid(self):
        for p in [[[0,.5,.5]], [[float('nan'),.5,.5]], [[.4,.4,.4]]]:
            with self.assertRaises(ValueError): relative_costs(p)
        with self.assertRaises(ValueError):
            arrange_from_anchor(np.zeros((10,13)),np.full((13,3),1/3))

if __name__ == '__main__': unittest.main()
