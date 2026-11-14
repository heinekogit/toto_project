import unittest
import numpy as np
from audit_draw_portfolio import ExactPortfolio, ball, optimize_fixed, optimize_free
from random_remaining_v24 import generate


class PortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=np.array([[.3+i*.01,.26,.44-i*.01] for i in range(13)])
        cls.e=ExactPortfolio(cls.p)
        cls.t,_=generate(cls.p,0)

    def test_ball_matches_full_enumeration(self):
        ticket=tuple(self.t[0])
        exact=np.flatnonzero((self.e.outcomes==ticket).sum(axis=1)>=10)
        np.testing.assert_array_equal(ball(ticket),exact)
        self.assertEqual(len(exact),2627)

    def test_union_metrics_and_duplicate_invariance(self):
        covered=np.unique(np.concatenate([ball(tuple(t)) for t in self.t]))
        m=self.e.metrics(self.t)
        self.assertAlmostEqual(m['p10'],float(self.e.mass[covered].sum()),places=13)
        one=np.tile(self.t[0],(10,1));n=self.e.metrics(one)
        expected=float(np.prod(self.p[np.arange(13),self.t[0]]))
        self.assertAlmostEqual(n['p13'],expected,places=14)
        self.assertAlmostEqual(n['expected_max_hits'],float(self.p[np.arange(13),self.t[0]].sum()),places=12)

    def test_fixed_search_coverage_and_constraints(self):
        for restricted in (False,True):
            t,a=optimize_fixed(self.t,self.e,preserve_draw_counts=restricted,max_steps=2)
            self.assertTrue(np.all(np.diff(a['p10_trace'])>0))
            self.assertAlmostEqual(self.e.metrics(t)['p10'],a['p10_trace'][-1],places=12)
            for s in range(3):np.testing.assert_array_equal((t==s).sum(axis=0),(self.t==s).sum(axis=0))
            if restricted:np.testing.assert_array_equal((t==1).sum(axis=1),(self.t==1).sum(axis=1))

    def test_guard_preserves_each_draw_count_coverage(self):
        t,a=optimize_fixed(self.t,self.e,max_steps=2,protect_draw_strata=True)
        self.assertTrue(np.all(np.array(a['final_K_coverage'])>=np.array(a['initial_K_coverage'])-1e-13))
        self.assertAlmostEqual(self.e.metrics(t)['p10'],a['p10_trace'][-1],places=12)

    def test_free_search_exact_union(self):
        t,a=optimize_free(self.t,self.e,max_steps=2)
        self.assertTrue(np.all(np.diff(a['p10_trace'])>0))
        self.assertAlmostEqual(self.e.metrics(t)['p10'],a['p10_trace'][-1],places=12)


if __name__=='__main__':unittest.main()
