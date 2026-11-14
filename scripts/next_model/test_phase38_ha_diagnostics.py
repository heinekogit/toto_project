import unittest
import numpy as np
import pandas as pd
from phase38_ha_diagnostics import pb_interval,wilson,metrics,raw,SOURCE

class Phase38Tests(unittest.TestCase):
 def test_conditional_expectation(self):
  d=pd.read_csv(SOURCE);g=d[d.actual_symbol!='D'];q=g[['baseline_P_H','baseline_P_A']].min(axis=1)/(g.baseline_P_H+g.baseline_P_A)
  mean,lo,hi=pb_interval(q)
  self.assertAlmostEqual(mean,24.319427151,places=9);self.assertEqual((lo,hi),(17,32))
  self.assertEqual(int((g.actual_symbol!=g.baseline_symbol).sum()),23)
 def test_probability_metrics(self):
  p=np.eye(3);m=metrics(p,[0,1,2]);self.assertEqual(m['log_loss'],0);self.assertEqual(m['brier'],0)
  self.assertEqual(pb_interval([0,1]),(1.,1,1))
 def test_coupled_distribution_and_mix(self):
  for h,a in [(1.5,1.5),(1.,2.),(2.,1.)]:
   p=raw(h,a);q=raw(a,h);np.testing.assert_allclose(p,q[::-1],atol=1e-12)
   self.assertAlmostEqual(p.sum(),1)
   mixed=.75*p+.25/3
   self.assertAlmostEqual(mixed[0]-mixed[2],.75*(p[0]-p[2]))
 def test_wilson_bounds(self):
  lo,hi=wilson(0,2);self.assertAlmostEqual(lo,0);self.assertGreater(hi,.6)

if __name__=='__main__':unittest.main()
