import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from score_placement_v25_shadow import Config,generate,diagnostics,exchange_targets
from random_remaining_v24 import allocate_per_match


class ShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=Path(__file__).resolve().parents[2]/'data/eval/toto_rounds/toto1668/snapshot/next_predictions.csv'
        cls.p=pd.read_csv(source).sort_values('match_no')[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy()

    def test_fixed_counts_reference_and_reproducibility(self):
        p=self.p.copy();counts=allocate_per_match(p);original=counts.copy()
        t,a=generate(p,counts);t2,a2=generate(p,counts)
        np.testing.assert_array_equal(p,self.p);np.testing.assert_array_equal(counts,original)
        np.testing.assert_array_equal(t,t2);self.assertEqual(a,a2)
        for s in range(3):np.testing.assert_array_equal((t==s).sum(axis=0),counts[:,s])
        anchor=np.argmax(p,axis=1);np.testing.assert_array_equal(a['anchor'],anchor)
        self.assertEqual(a['total_exchanges'],int((10-counts[np.arange(13),anchor]).sum()))
        np.testing.assert_array_equal((t!=anchor).sum(axis=1),a['target_exchanges'])
        self.assertFalse(a['duplicate_pairs'])

    def test_coefficient_changes_placement_not_anchor_or_probability(self):
        t1,a1=generate(self.p,config=Config(draw_coefficient=1))
        t2,a2=generate(self.p,config=Config(draw_coefficient=1.4))
        np.testing.assert_array_equal(a1['initial_tickets'],a2['initial_tickets'])
        self.assertTrue(np.any(t1!=t2));self.assertTrue(np.any((t1==1)!=(t2==1)))
        self.assertEqual(a1['anchor'],a2['anchor']);self.assertEqual(a1['counts'],a2['counts'])
        self.assertEqual(a1['target_exchanges'],a2['target_exchanges'])
        rows=diagnostics(self.p,t2,a2)
        self.assertEqual(len(rows),11);self.assertFalse(rows[0]['purchase'])
        for row,t in zip(rows[1:],t2):
            self.assertAlmostEqual(row['joint_probability'],float(np.prod(self.p[np.arange(13),t])),places=15)
        self.assertTrue((np.array(a2['losses_HDA'])<0).any())

    def test_target_feasibility_and_configurable_span(self):
        degrees=10-allocate_per_match(self.p)[np.arange(13),self.p.argmax(axis=1)]
        targets=[]
        for span in (0,2,4,8,30):
            e,_=exchange_targets(degrees,span);targets.append(tuple(e))
            self.assertEqual(e.sum(),degrees.sum());self.assertTrue(np.all(np.diff(e)>=0))
            for k in range(1,11):self.assertLessEqual(e[-k:].sum(),np.minimum(degrees,k).sum())
        self.assertGreater(len(set(targets)),1)

    def test_anchor_draw_and_forced_duplicates(self):
        p=np.tile([.2,.6,.2],(13,1));counts=np.tile([0,10,0],(13,1))
        t,a=generate(p,counts)
        self.assertTrue((t==1).all());self.assertEqual(a['total_exchanges'],0)
        self.assertEqual(len(a['duplicate_pairs']),45);self.assertTrue(a['duplicate_unavoidable_proven'])

    def test_custom_counts_and_per_match_coefficients(self):
        counts=allocate_per_match(self.p);counts[0,0]-=1;counts[0,2]+=1
        kd=np.linspace(1,1.4,13);t,a=generate(self.p,counts,draw_coefficients=kd)
        np.testing.assert_allclose(np.array(a['scores_HDA'])[:,1],self.p[:,1]*kd)
        for s in range(3):np.testing.assert_array_equal((t==s).sum(axis=0),counts[:,s])

    def test_bad_input(self):
        with self.assertRaises(ValueError):generate(self.p,config=Config(draw_coefficient=-1))
        with self.assertRaises(ValueError):generate(self.p,np.ones((13,3)))
        with self.assertRaises(ValueError):generate(self.p,config=Config(max_steps=0))


if __name__=='__main__':unittest.main()
