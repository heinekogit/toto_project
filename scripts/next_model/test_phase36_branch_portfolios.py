import unittest
import numpy as np
import pandas as pd
from phase36_branch_portfolios import SOURCE,FEATURES,select_candidates,top_codes,stress,generate
from analyze_score_structure_phase4b_stage1 import universe,top_ten,decode

class Phase36Tests(unittest.TestCase):
 def test_heap_matches_full_result_space(self):
  p=np.random.default_rng(401).dirichlet([3,2,3],13)
  np.testing.assert_array_equal(top_codes(p,[[0,1,2]]*13,10),top_ten(universe(p)))
 def test_selectors_no_result_leakage_and_B_C_equal(self):
  d=pd.read_csv(SOURCE)
  for _,g in d.groupby('round'):
   for k in (3,4,5):
    self.assertEqual(set(select_candidates(g,'B',k)),set(select_candidates(g,'C',k)))
    for m in 'ABC':
     shuffled=g.copy()
     for col in ['actual_symbol','actual_D','actual_home','actual_away']:shuffled[col]=0
     np.testing.assert_array_equal(select_candidates(g,m,k),select_candidates(shuffled,m,k))
 def test_stress_odds_and_portfolio_reproducible(self):
  g=pd.read_csv(SOURCE,usecols=FEATURES).query('round == "toto1660"').sort_values('match_no')
  p=g[['baseline_P_H','baseline_P_D','baseline_P_A']].to_numpy()
  q=stress(p,1)
  self.assertAlmostEqual(q[0,1]/(1-q[0,1]),2*p[0,1]/(1-p[0,1]))
  self.assertAlmostEqual(q[0,0]/q[0,2],p[0,0]/p[0,2]);np.testing.assert_allclose(q[1:],p[1:])
  sel=select_candidates(g,'B',3)
  x,_,_,_=generate(p,sel);y,_,_,_=generate(p,sel[::-1])
  np.testing.assert_array_equal(x,y);self.assertEqual(len(set(x)),10)
  t=decode(x)
  self.assertFalse((t[:,[i for i in range(13) if i not in sel]]==1).any())

if __name__=='__main__':unittest.main()
