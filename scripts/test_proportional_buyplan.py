import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / 'next_model'))
from proportional_buyplan import allocate_counts, build_proportional_tickets
import pandas as pd
from build_next_buyplan import build_round, build_shadow_round
from ticket_coverage_optimizer import optimize_ten_hit_coverage


class ProportionalTests(unittest.TestCase):
    def test_ge10_optimizer_preserves_allocations_and_improves_objective(self):
        p=np.tile([.45,.25,.30],(13,1))
        initial,_=build_proportional_tickets(p)
        tickets,audit=optimize_ten_hit_coverage(initial,p)
        for symbol in range(3):
            np.testing.assert_array_equal((initial==symbol).sum(axis=0),(tickets==symbol).sum(axis=0))
        self.assertGreaterEqual(audit['training_probability_after'],audit['training_probability_before'])
        self.assertGreaterEqual(audit['minimum_pairwise_hamming_distance'],2)
        repeated,_=optimize_ten_hit_coverage(initial,p)
        np.testing.assert_array_equal(tickets,repeated)

    def test_shadow_uses_actual_proportional_official_tickets(self):
        frame = pd.DataFrame([dict(match_no=i+1,home_team=f'H{i}',away_team=f'A{i}',league='j1',
                                  next_prob_home=.45,next_prob_draw=.25,next_prob_away=.30,
                                  final_lambda_home=1.4,final_lambda_away=1.0,
                                  next_prob_0_0=.08,next_prob_1_1=.10,next_prob_2_2=.03,
                                  lab_basis_hint='basis_balanced') for i in range(13)])
        plan,_=build_round(frame,strategy='probability_allocation_v1')
        before=plan.copy()
        shadow,audit=build_shadow_round(frame,official_plan=plan,strategy='probability_allocation_v1')
        pd.testing.assert_frame_equal(plan,before)
        tickets=[tuple(plan[f'ticket{i:02}']) for i in range(1,11)]+[tuple(shadow[f'ticket{i:02}']) for i in range(11,16)]
        self.assertEqual(len(set(tickets)),15)
        self.assertTrue(all(sum(a!=b for a,b in zip(tickets[i],tickets[j]))>=2 for i in range(15) for j in range(i)))
        self.assertIsNone(audit['shadow_probability_floor_ratio'])
        expected = allocate_counts(frame[['next_prob_home','next_prob_draw','next_prob_away']],ticket_count=5)
        cols=[f'ticket{i:02}' for i in range(11,16)]
        for index,result in enumerate([1,0,2]):
            np.testing.assert_array_equal(shadow[cols].eq(result).sum(axis=1),expected[:,index])

    def test_rounding_preserves_rows_and_probability_totals(self):
        rng = np.random.default_rng(12)
        for _ in range(20):
            p = rng.dirichlet([2,2,2], 13)
            c = allocate_counts(p)
            self.assertTrue((c.sum(axis=1)==10).all())
            self.assertTrue((abs(c-10*p)<1+1e-8).all())
            self.assertTrue((abs(c.sum(axis=0)-(10*p).sum(axis=0))<1).all())

    def test_arrangement_never_changes_allocations(self):
        p = np.tile([.45,.25,.30], (13,1))
        t,a = build_proportional_tickets(p, trials=300)
        c = allocate_counts(p)
        for s in range(3):
            np.testing.assert_array_equal((t==s).sum(axis=0),c[:,s])
        self.assertIn(int((t==1).sum()),[32,33])
        self.assertEqual(len(set(map(tuple,t))),10)
        self.assertTrue(a['diversity_constraint_satisfied'])

    def test_impossible_diversity_does_not_change_probabilities(self):
        p = np.tile([1.,0.,0.],(13,1))
        t,a = build_proportional_tickets(p,trials=2)
        self.assertTrue((t==0).all())
        self.assertFalse(a['diversity_constraint_satisfied'])

    def test_invalid_inputs_rejected(self):
        for p in [np.tile([.5,.5,.5],(13,1)),np.tile([-.1,.5,.6],(13,1)),np.tile([np.nan,.5,.5],(13,1))]:
            with self.assertRaises(ValueError): allocate_counts(p)

if __name__=='__main__': unittest.main()
