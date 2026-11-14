import unittest
from itertools import combinations
import numpy as np
from strict_multi_search import TYPES,COST,POWERS,HA_POWERS,expand,make_block,all_patterns,validate_purchase

class StrictMultiTests(unittest.TestCase):
    def test_full_product_and_probability(self):
        rng=np.random.default_rng(1670);p=rng.dirichlet([4,3,4],13);ids=np.arange(8192);ha=((ids[:,None]//HA_POWERS)%2*2).astype(np.int8)
        for kind,(nd,nt) in TYPES.items():
            for _ in range(15):
                positions=rng.choice(13,nd+nt,replace=False);ds=positions[:nd];ts=positions[nd:];bid=int(rng.integers(8192))
                b=make_block(kind,bid,ds,ts,ha,p);t=expand(b['masks'])
                self.assertEqual(len(t),COST[kind]);self.assertEqual(len(np.unique(t,axis=0)),len(t));self.assertNotIn(2,b['masks'])
                expected=np.prod([sum(p[m,s] for s in range(3) if mask&(1<<s)) for m,mask in enumerate(b['masks'])])
                self.assertAlmostEqual(expected,b['mass'],places=15)
    def test_all_budget_patterns(self):
        patterns=all_patterns();self.assertEqual(len(patterns),30)
        for counts,singles in patterns:self.assertEqual(sum(COST[k]*v for k,v in counts.items())+singles,10)
        for kind in TYPES:self.assertTrue(any(c[kind]==1 and sum(c.values())==1 for c,s in patterns))
    def test_D_only_prohibited_and_overlaps(self):
        p=np.full((13,3),1/3);ids=np.arange(8192);ha=((ids[:,None]//HA_POWERS)%2*2).astype(np.int8)
        b=make_block('D1',0,[0],[],ha,p)
        s=dict(blocks=[b],singles=list(range(1,9)),p13=10/3**13)
        t=validate_purchase(s,ha,p);self.assertEqual(len(t),10)
        b['masks'][0]=2
        with self.assertRaisesRegex(AssertionError,'D-only'):validate_purchase(s,ha,p)
    def test_single_overlap_characterization(self):
        ids=np.arange(8192);ha=((ids[:,None]//HA_POWERS)%2*2).astype(np.int8);p=np.full((13,3),1/3)
        for ts in [[],[3],[3,8]]:
            mask=sum(int(HA_POWERS[m]) for m in ts);baseid=170&~mask;b=make_block('test',baseid,[1],ts,ha,p)
            actual=set(b['codes']);truth=np.array([int(row@POWERS) in actual for row in ha])
            np.testing.assert_array_equal(truth,(ids&~mask)==baseid)

if __name__=='__main__':unittest.main()
