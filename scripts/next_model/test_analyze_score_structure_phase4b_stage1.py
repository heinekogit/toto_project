import unittest
import numpy as np
from analyze_score_structure_phase4b_stage1 import N,decode,encode,universe,top_ten,score_tickets

class ExactPortfolioTests(unittest.TestCase):
    def test_space_and_round_trip(self):
        self.assertEqual(N,1594323)
        codes=np.array([0,1,3,N-1])
        np.testing.assert_array_equal(encode(decode(codes)),codes)
    def test_exact_top_ten_by_full_sort(self):
        rng=np.random.default_rng(20261111)
        p=rng.dirichlet([2,2,2],size=13)
        mass=universe(p)
        got=top_ten(mass)
        expected=np.lexsort((np.arange(N),-mass))[:10]
        np.testing.assert_array_equal(got,expected)
        np.testing.assert_allclose(score_tickets(decode(got),p),mass[got],rtol=1e-12)
    def test_uniform_ties_are_deterministic(self):
        p=np.full((13,3),1/3)
        got=top_ten(universe(p))
        np.testing.assert_array_equal(got,np.arange(10))

if __name__=='__main__':unittest.main()
