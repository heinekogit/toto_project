import unittest

import numpy as np

from compare_d_scoop_score_relative import compare_round


class ScoreRelativeTests(unittest.TestCase):
    def test_three_methods_and_mix_controls(self):
        cap={'A':0,'B':0,'C':0};ac=bc=gained=lost=0
        for rid in ('toto1660','toto1663','toto1664','toto1665','toto1668','toto1670'):
            frame,summary,exchange,_=compare_round(rid)
            for name in 'ABC':
                self.assertEqual(int(frame[f'{name}_selected'].sum()),4)
                cap[name]+=summary[f'{name}_captured_D']
            np.testing.assert_allclose(frame.score_C,frame.score_P_D/np.maximum(frame.score_P_H,frame.score_P_A))
            np.testing.assert_allclose(frame[['final_P_H','final_P_D','final_P_A']],
                (1-frame.mix.to_numpy()[:,None])*frame[['score_P_H','score_P_D','score_P_A']].to_numpy()+frame.mix.to_numpy()[:,None]/3)
            self.assertEqual(len(exchange),2*summary['BC_replaced'])
            if rid in ('toto1660','toto1663','toto1664'):
                np.testing.assert_array_equal(frame.B_rank,frame.C_rank)
            ac+=summary['AC_replaced'];bc+=summary['BC_replaced']
            gained+=summary['BC_gained_D'];lost+=summary['BC_lost_D']
        self.assertEqual((cap,ac,bc,gained,lost),({'A':6,'B':5,'C':4},9,2,0,1))
        self.assertEqual(cap['C']-cap['B'],gained-lost)


if __name__=='__main__':unittest.main()
