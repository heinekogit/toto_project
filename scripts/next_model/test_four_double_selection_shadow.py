import unittest
import numpy as np
import pandas as pd
from four_double_selection_shadow import select, expand, probe, features, VARIANTS, MAX_ADJUSTMENT
from pathlib import Path

class FourDoubleTests(unittest.TestCase):
    def test_anchor_is_not_reversed_and_all_sixteen_are_unique(self):
        anchor=np.array([0,2]*6+[0]); selected=np.array([0,3,7,12])
        t=expand(anchor,selected)
        self.assertEqual(len(np.unique(t,axis=0)),16)
        self.assertTrue(any(np.array_equal(row,anchor) for row in t))
        for i in range(13):
            if i in selected:
                self.assertEqual(set(t[:,i]),{1,int(anchor[i])})
                self.assertEqual(int((t[:,i]==1).sum()),8)
            else: self.assertTrue((t[:,i]==anchor[i]).all())

    def test_current_outcomes_and_unused_side_diagnostic_cannot_change_selection(self):
        root=Path(__file__).resolve().parents[2]
        frames=[]
        for rid in ['toto1660','toto1663','toto1664','toto1665']:
            base=root/'data/eval/toto_rounds'/rid
            z=features(pd.read_csv(base/'snapshot/next_predictions.csv').sort_values('match_no').reset_index(drop=True))
            z['round']=rid
            z['actual_D']=pd.read_csv(base/'actual_results.csv').sort_values('match_no').result.eq(0).astype(int).to_numpy()
            frames.append(z)
        train=pd.concat(frames[:3],ignore_index=True); test=frames[3]
        alternate=test.copy(); alternate['actual_D']=1-alternate.actual_D
        alternate['unused_HA_probability']=.999
        np.testing.assert_array_equal(select(test),select(alternate))
        for cols in VARIANTS.values():
            before,_=probe(train,test,cols); after,_=probe(train,alternate,cols)
            np.testing.assert_array_equal(before,after)
            self.assertLessEqual(np.abs(before-test.P_D.to_numpy()).max(),MAX_ADJUSTMENT+1e-15)

if __name__=='__main__': unittest.main()
