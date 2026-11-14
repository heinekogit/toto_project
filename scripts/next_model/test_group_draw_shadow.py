import unittest
import numpy as np
from group_draw_shadow import groups,patterns,pair_valid,validate_groups
from fixed_exact13_shadow import moved,MOVES

class GroupDrawTests(unittest.TestCase):
    def test_partitions(self):
        self.assertEqual(groups(3),[[0,1,2],[3,4,5],[6,7,8,9]])
        for d in range(1,11):
            for shift in range(10):
                g=groups(d,shift);self.assertEqual(sorted(sum(g,[])),list(range(10)))
                self.assertLessEqual(max(map(len,g))-min(map(len,g)),1)
        self.assertEqual(groups(0),[])
    def test_complete_patterns(self):
        for circular in [False,True]:
            opts,shifts,allowed=patterns([4,3,3],circular)
            self.assertEqual(len(opts),len(np.unique(opts,axis=0)))
            for s,n in enumerate([4,3,3]):self.assertTrue(((opts==s).sum(1)==n).all())
            for t,shift in zip(opts,shifts):
                self.assertTrue(all(sum(t[r]==1 for r in g)==1 for g in groups(3,int(shift))))
                self.assertTrue(allowed[int((t==1)@(1<<np.arange(10)))])
        fixed=set(map(tuple,patterns([4,3,3],False)[0]));circ=set(map(tuple,patterns([4,3,3],True)[0]));self.assertTrue(fixed<circ)
    def test_edge_counts(self):
        for d in [0,10]:
            opts,_,_=patterns([10-d,d,0],True);self.assertEqual(len(opts),1)
    def test_pair_filter(self):
        for circular in [False,True]:
            opts,_,a=patterns([4,3,3],circular);t=np.column_stack([opts[(m*127+15)%len(opts)] for m in range(13)])
            allowed=np.tile(a,(13,1));ok=pair_valid(t,allowed)
            for idx in range(0,len(MOVES),17):
                u=moved(t,idx);masks=((u==1)*(1<<np.arange(10))[:,None]).sum(0)
                self.assertEqual(bool(ok[idx]),bool(a[masks].all()))

if __name__=='__main__':unittest.main()
