import unittest
import numpy as np
from draw_count_cover_shadow import draw_distribution,lookup_table,signature,move_coverages,optimize,better
from fixed_exact13_shadow import moved,mass
from fixed_draw_bounds_shadow import cycled

class DrawCountCoverTests(unittest.TestCase):
    def test_distribution(self):
        from math import comb
        p=np.tile([.35,.3,.35],(13,1));d=draw_distribution(p)
        np.testing.assert_allclose(d,[comb(13,k)*.3**k*.7**(13-k) for k in range(14)])
        self.assertAlmostEqual(d.sum(),1);self.assertAlmostEqual(d@np.arange(14),3.9)
    def test_vector_coverage(self):
        rng=np.random.default_rng(2);p=rng.dirichlet([2,2,2],13);t=rng.integers(0,3,(10,13));lookup=lookup_table(draw_distribution(p));a,b=move_coverages(t,lookup)
        for values,apply in [(a,moved),(b,cycled)]:
            for idx in rng.choice(len(values),150,replace=False):self.assertAlmostEqual(values[idx],lookup[signature(apply(t,int(idx)))],places=14)
    def test_floor_and_lexicographic_trace(self):
        rng=np.random.default_rng(3);p=rng.dirichlet([3,3,3],13);t=rng.integers(0,3,(10,13))
        result,audit,_=optimize(p,t,.95,starts=3,cap=10,seed=1)
        self.assertGreaterEqual(mass(p,result).sum(),.95*mass(p,t).sum()-1e-16)
        for run in audit['runs']:
            for x,y in zip(run['trace'],run['trace'][1:]):self.assertTrue(better(y['coverage'],y['p13'],x['coverage'],x['p13']))
            for x in run['trace']:self.assertGreaterEqual(x['p13'],audit['p13_floor']-1e-16)

if __name__=='__main__':unittest.main()
