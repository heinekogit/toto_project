import unittest

import numpy as np

from compare_v42_portfolios import optimize, value


class CompareV42PortfoliosTest(unittest.TestCase):
    def test_fixed_D_counts_and_ticket_support(self):
        p=np.tile([.45,.28,.27],(13,1))
        target=np.full(13,3)
        m,score=optimize(p,target,target,starts=2,steps=500,required_counts=tuple(range(1,7)),
                         initial_counts=target,max_ticket_D=7)
        self.assertTrue(np.array_equal((m==1).sum(axis=0),target))
        self.assertTrue(set(range(1,7)).issubset(set((m==1).sum(axis=1))))
        self.assertLessEqual(int((m==1).sum(axis=1).max()),7)
        self.assertEqual(len({tuple(row) for row in m}),10)
        self.assertAlmostEqual(score,value(m,p))


if __name__=='__main__':unittest.main()
