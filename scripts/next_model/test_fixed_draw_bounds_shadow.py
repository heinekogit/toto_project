import unittest
import numpy as np
from fixed_draw_bounds_shadow import pair_moves,cycle_moves,cycled,feasible,proposed_bounds,optimize
from fixed_exact13_shadow import mass,moved,validate

class DrawBoundsTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(91);self.p=self.rng.dirichlet([3,3,3],13);self.t=self.rng.integers(0,3,(10,13));self.counts=np.stack([(self.t==s).sum(0) for s in range(3)],1)
    def test_pair_and_cycle_scores_and_constraints(self):
        for fn,apply in [(pair_moves,moved),(cycle_moves,cycled)]:
            values,valid=fn(self.p,self.t,2,5)
            for idx in self.rng.choice(len(values),150,replace=False):
                u=apply(self.t,int(idx));self.assertAlmostEqual(values[idx],mass(self.p,u).sum(),places=15)
                # Existing unmodified rows can be out of bounds in this random test.
                changed=np.flatnonzero((u!=self.t).any(1));d=(u==1).sum(1)
                if valid[idx]:
                    self.assertTrue((d[changed]>=2).all() and (d[changed]<=5).all());self.assertEqual(len(np.unique(u,axis=0)),10)
                for s in range(3):np.testing.assert_array_equal((u==s).sum(0),self.counts[:,s])
    def test_feasible_and_search(self):
        t,_=feasible(self.counts,1,6);validate(t,self.counts)
        a,info=optimize(self.p,self.counts,1,6,starts=4,cap=20,seed=7)
        b,_=optimize(self.p,self.counts,1,6,starts=4,cap=20,seed=7)
        np.testing.assert_array_equal(a,b);validate(a,self.counts)
        self.assertTrue(((a==1).sum(1)>=1).all() and ((a==1).sum(1)<=6).all())
        for run in info['runs']:self.assertTrue((np.diff(run['trace'])>0).all())
    def test_infeasible_total(self):
        counts=np.tile([10,0,0],(13,1))
        with self.assertRaisesRegex(ValueError,'total draw'):feasible(counts,2,5)
    def test_duplicate_impossibility(self):
        counts=np.tile([10,0,0],(13,1))
        with self.assertRaisesRegex(ValueError,'no unique portfolio'):feasible(counts,0,0)
    def test_generalization(self):
        self.assertEqual(proposed_bounds(38,1),(2,5));self.assertEqual(proposed_bounds(38,2),(1,6))
        self.assertEqual(proposed_bounds(5,1)[0],0)
        for total in range(131):
            lo,hi=proposed_bounds(total);self.assertTrue(0<=lo<=hi<=13 and 10*lo<=total<=10*hi)

if __name__=='__main__':unittest.main()
