import unittest
import numpy as np
from phase37_constraint_audit import Engine,state,prune,branch_pool,DMASK,FULLMASK
from phase36_branch_portfolios import top_codes
from prototype_buyplan_v40 import ticket_probabilities
from analyze_score_structure_phase4b_stage1 import decode,encode

class AuditTests(unittest.TestCase):
 def setUp(self):
  self.p=np.random.default_rng(37).dirichlet([3,2,3],13);self.e=Engine(self.p)
 def test_exact_mass_and_top(self):
  for mask,f in [(0,1.),(5,1.5),(9,2.)]:
   q=state(self.p,mask,f);ids=np.array([0,1,34,545,1594322])
   np.testing.assert_allclose(self.e.mass(mask,f,ids),ticket_probabilities(q,decode(ids)),rtol=1e-13)
   np.testing.assert_array_equal(self.e.top(mask,f,FULLMASK)[:10],top_codes(q,[[0,1,2]]*13,10))
 def test_lossless_pattern_dominance(self):
  allowedmask=7;support=7;ids=self.e.allowed(allowedmask);pool=self.e.full(allowedmask,support)
  for pat in range(8):
   all_ids=ids[(DMASK[ids]&support)==pat];kept=pool[(DMASK[pool]&support)==pat]
   self.assertEqual(len(kept),10)
   omitted=np.setdiff1d(all_ids,kept)
   self.assertGreaterEqual(self.e.base[kept].min(),self.e.base[omitted].max())
   for mask in range(8):
    self.assertGreaterEqual(self.e.mass(mask,2.,kept).min()+1e-18,self.e.mass(mask,2.,omitted).max())
 def test_unrestricted_no_stress_is_exact_top10(self):
  ans=self.e.solve([0,1,2],[0,1,2,3], 'U',30,1.,False,[0])
  np.testing.assert_array_equal(ans['codes'],self.e.top(0,1.,FULLMASK)[:10])
 def test_S_branch_is_not_purchase_constraint(self):
  pool=branch_pool(self.p,[0,4],1)
  self.assertTrue(((DMASK[pool]&(FULLMASK^1))==0).all())
  self.assertTrue((DMASK[pool]==0).any());self.assertTrue((DMASK[pool]==1).any())

if __name__=='__main__':unittest.main()
