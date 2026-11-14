import unittest,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from phase39_input_audit import games_candidates,loss
from uncertainty_calibration import adjust
from phase39_future_check import capture,evaluate
class AuditTests(unittest.TestCase):
 def test_rounding_not_identification(self):
  self.assertEqual(round(14/1.2),12)
  self.assertIn(11,games_candidates(14,1.2,'floor'))
  self.assertNotIn(11,games_candidates(14,1.2))
 def test_mix_loss_direction(self):
  p=np.array([[.6,.25,.15]]);q,m=adjust(p,[1],[0],.25,.25)
  self.assertGreater(loss(q[0],0)[0],loss(p[0],0)[0]);self.assertLess(loss(q[0],2)[0],loss(p[0],2)[0]);self.assertAlmostEqual(q.sum(),1)
 def test_future_capture_evaluate_and_refuse_overwrite(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'input.csv';d=pd.DataFrame(dict(match_no=range(1,14),match_id=[str(i) for i in range(13)],datetime=['2030-01-01 15:00']*13,final_lambda_home=[1.2]*13,final_lambda_away=[1.1]*13,next_prob_home=[.4]*13,next_prob_draw=[.3]*13,next_prob_away=[.3]*13));d.to_csv(p,index=False)
   capture(p,'test',root/'capture','2029-12-31');r=root/'results.csv';pd.DataFrame(dict(match_id=range(13),actual_home=[1]*13,actual_away=[0]*13)).to_csv(r,index=False)
   evaluate(root/'capture',r,root/'eval');self.assertTrue((root/'eval/summary.json').exists())
   with self.assertRaises(ValueError):capture(p,'test',root/'capture','2029-12-31')
   with self.assertRaises(ValueError):capture(p,'test',root/'late','2030-01-02')
if __name__=='__main__':unittest.main()
