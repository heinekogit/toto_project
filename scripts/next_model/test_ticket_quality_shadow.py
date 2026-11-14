import unittest
import numpy as np
from ticket_quality_shadow import probability_neighbors,objectives,quality,optimize,lex_better
from fixed_exact13_shadow import moved,mass
from fixed_draw_bounds_shadow import cycled
from draw_count_cover_shadow import lookup_table,draw_distribution,signature

class TicketQualityTests(unittest.TestCase):
    def test_neighbor_probabilities(self):
        rng=np.random.default_rng(17);p=rng.dirichlet([3,3,3],13);t=rng.integers(0,3,(10,13));vals,ok,split=probability_neighbors(p,t)
        for idx in rng.choice(len(vals),160,replace=False):
            u=moved(t,int(idx)) if idx<split else cycled(t,int(idx-split))
            np.testing.assert_allclose(vals[idx],mass(p,u),rtol=1e-13)
    def test_quality(self):
        x=np.arange(1,11,dtype=float);q=quality(x)
        self.assertEqual(q['bottom5_sum'],15);self.assertEqual(q['top3_sum'],27);self.assertEqual(q['bottom3_sum'],6);self.assertEqual(q['median'],5.5)
        np.testing.assert_array_equal(objectives(x,'Q1'),[1,15,55]);np.testing.assert_array_equal(objectives(x,'Q2'),[15,1,55])
    def test_hard_floors_and_improvement(self):
        rng=np.random.default_rng(7);p=rng.dirichlet([3,3,3],13);t=rng.integers(0,3,(10,13));counts=np.stack([(t==s).sum(0) for s in range(3)],1)
        lower=mass(p,t).sum()*.95;lookup=lookup_table(draw_distribution(p));cf=lookup[signature(t)]*.9
        for kind in ['Q1','Q2','Q3']:
            u,a=optimize(p,counts,[t],lower,cf,kind,starts=3,cap=12,seed=2)
            self.assertGreaterEqual(mass(p,u).sum(),lower-1e-16);self.assertGreaterEqual(lookup[signature(u)],cf-1e-14)
            for run in a['runs']:
                for x in run['trace']:self.assertTrue(x['p13']>=lower-1e-16 and x['coverage']>=cf-1e-14)
    def test_infeasible_start_not_proof(self):
        p=np.full((13,3),1/3);t=np.zeros((10,13),dtype=int)
        with self.assertRaisesRegex(ValueError,'not proof'):optimize(p,np.zeros((13,3)),[t],1.,1.,'Q1')

if __name__=='__main__':unittest.main()
