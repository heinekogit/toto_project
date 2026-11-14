import unittest
import numpy as np
from fixed_exact13_shadow import neighbors,moved,mass,ball,POWERS,optimize_b,validate

class FixedExact13Tests(unittest.TestCase):
    def test_neighbor_objective_and_uniqueness(self):
        rng=np.random.default_rng(73);p=rng.dirichlet([2,2,2],13);t=rng.integers(0,3,(10,13))
        values,valid=neighbors(p,t)
        for idx in rng.choice(len(values),120,replace=False):
            u=moved(t,int(idx));self.assertAlmostEqual(values[idx],mass(p,u).sum(),places=15)
            changed=not np.array_equal(u,t)
            self.assertEqual(bool(valid[idx]),changed and len(np.unique(u,axis=0))==10)
            for s in range(3):np.testing.assert_array_equal((t==s).sum(0),(u==s).sum(0))
    def test_duplicate_creation_rejected(self):
        rng=np.random.default_rng(9);t=rng.integers(0,3,(10,13));t[1]=t[0];t[1,0]=(t[0,0]+1)%3;t[2]=t[0];t[2,1]=(t[0,1]+1)%3
        p=np.full((13,3),1/3);_,valid=neighbors(p,t)
        from fixed_exact13_shadow import MOVES
        idx=np.flatnonzero((MOVES==[1,2,0,13]).all(1))[0]
        self.assertFalse(valid[idx])
    def test_neighborhoods(self):
        t=np.arange(13)%3
        for radius,size in [(1,27),(2,339)]:
            codes=ball(tuple(t),radius);self.assertEqual(len(codes),size);self.assertEqual(len(np.unique(codes)),size)
            outcomes=(codes[:,None]//POWERS)%3
            self.assertTrue(((outcomes!=t).sum(1)<=radius).all())
    def test_b_monotonic_reproducible(self):
        rng=np.random.default_rng(2);p=rng.dirichlet([3,3,3],13);t=rng.integers(0,3,(10,13));counts=np.stack([(t==s).sum(0) for s in range(3)],1)
        a,info=optimize_b(p,t,starts=3,cap=10);b,_=optimize_b(p,t,starts=3,cap=10)
        validate(a,counts);np.testing.assert_array_equal(a,b);self.assertGreaterEqual(mass(p,a).sum(),mass(p,t).sum())
        for run in info['runs']:self.assertTrue((np.diff(run['trace'])>0).all())

if __name__=='__main__':unittest.main()
