import unittest
import numpy as np
import pandas as pd
from diagnose_score_structure_phase4a_stage2 import hypergeom_null,ranked,discrimination

class Stage2Tests(unittest.TestCase):
    def test_hypergeometric_exact_and_mean(self):
        p=hypergeom_null([2,3,4,4,1,4],4)
        self.assertAlmostEqual(float(p.sum()),1)
        self.assertAlmostEqual(float(np.arange(len(p))@p),18*4/13)
        self.assertAlmostEqual(float(hypergeom_null([2,3,4,4,1,4],13)[18]),1)
    def test_rank_ties_by_match_number(self):
        d=pd.DataFrame({'round':['t']*3,'match_no':[3,1,2], 'raw_P_D':[.3,.3,.2], 'final_P_D':[.3,.3,.2], 'actual_result':[1,0,0]})
        got=ranked(d).set_index('match_no')
        self.assertEqual(got.raw_rank.to_dict(),{1:1,2:3,3:2})
    def test_discrimination_perfect(self):
        d=pd.DataFrame({'actual_result':[1,0,0], 'raw_P_D':[.9,.2,.1]})
        v=discrimination(d,'raw_P_D')
        self.assertEqual(v['ROC_AUC'],1)
        self.assertEqual(v['average_precision'],1)

if __name__=='__main__':unittest.main()
