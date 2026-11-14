import unittest

import numpy as np

from phase35_conditional_branches import probabilities, scenarios


class Phase35Tests(unittest.TestCase):
    def test_total_and_ratio_axes_are_separate(self):
        h,a=1.5,1.0
        cases={(name,eps):(hh,aa,ts,rs) for name,_,eps,hh,aa,ts,rs in scenarios(h,a)}
        self.assertEqual(len(cases),25)
        for eps in (.05,.10):
            for name in ('総得点低下','総得点上昇'):
                hh,aa,_,_=cases[(name,eps)]
                self.assertAlmostEqual(hh/aa,h/a)
            for name in ('H相対上昇','A相対上昇'):
                hh,aa,_,_=cases[(name,eps)]
                self.assertAlmostEqual(hh+aa,h+a)

    def test_probabilities_and_mix(self):
        matrix,structure,raw,final,_=probabilities(1.5,1.0,.12)
        self.assertAlmostEqual(matrix.sum(),1)
        self.assertAlmostEqual(raw.sum(),1)
        np.testing.assert_allclose(final,.88*raw+.12/3,rtol=0,atol=1e-12)
        self.assertAlmostEqual(structure['P_draw'],raw[1])


if __name__=='__main__': unittest.main()
