import unittest
import numpy as np
from analyze_score_structure_phase4b_stage1 import N,POWERS,universe,decode
from analyze_score_structure_phase4b_stage3 import constrained,draw_distribution

class DrawOpportunityTests(unittest.TestCase):
    def test_exact_constrained_against_full_sort(self):
        p=np.random.default_rng(431).dirichlet([3,2,3],size=13)
        mass=universe(p);codes=np.arange(N)
        draw=np.zeros(N,dtype=bool)
        for power in POWERS:draw|=((codes//power)%3)==1
        for k in (0,1,2,5,10):
            selected,n=constrained(mass,draw,k)
            self.assertEqual(len(set(selected)),10)
            self.assertGreaterEqual(n,k)
            best=float(mass[selected].sum())
            # Independently sort the complete 1,594,323-combination universe by score.
            no=codes[~draw];yes=codes[draw]
            no=no[np.lexsort((no,-mass[no]))[:10]]
            yes=yes[np.lexsort((yes,-mass[yes]))[:10]]
            brute=max(float(mass[np.r_[no[:10-m],yes[:m]]].sum()) for m in range(k,11))
            self.assertAlmostEqual(best,brute,places=14)
    def test_draw_count_mass(self):
        p=np.tile([.4,.25,.35],(13,1))
        d=draw_distribution(p)
        self.assertAlmostEqual(float(d.sum()),1)
        self.assertAlmostEqual(float(d[0]),.75**13)
        self.assertAlmostEqual(float(np.arange(14)@d),3.25)

if __name__=='__main__':unittest.main()
