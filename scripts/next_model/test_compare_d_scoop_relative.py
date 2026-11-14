import unittest

import numpy as np

from compare_d_scoop_relative import compare_round


class RelativeComparisonTests(unittest.TestCase):
    def test_saved_probabilities_and_exchange_accounting(self):
        captured_a=captured_b=gained=lost=replacements=0
        for rid in ('toto1660','toto1663','toto1664','toto1665','toto1668','toto1670'):
            frame,summary,exchange,_=compare_round(rid)
            np.testing.assert_allclose(frame.P_H+frame.P_D+frame.P_A,1,atol=1e-10)
            np.testing.assert_allclose(frame.D_relative,frame.P_D/np.maximum(frame.P_H,frame.P_A))
            self.assertEqual(int(frame.A_selected.sum()),4)
            self.assertEqual(int(frame.B_selected.sum()),4)
            self.assertEqual(len(exchange),2*summary['replaced_matches'])
            captured_a+=summary['A_captured_D'];captured_b+=summary['B_captured_D']
            gained+=summary['gained_D_by_B'];lost+=summary['lost_D_from_A']
            replacements+=summary['replaced_matches']
        self.assertEqual((captured_a,captured_b,gained,lost,replacements),(6,5,1,2,10))
        self.assertEqual(captured_b-captured_a,gained-lost)

    def test_same_PD_rewards_smaller_HA_gap(self):
        pd=.28
        balanced=2*pd/(1-pd+.02)
        lopsided=2*pd/(1-pd+.20)
        self.assertGreater(balanced,lopsided)


if __name__=='__main__': unittest.main()
