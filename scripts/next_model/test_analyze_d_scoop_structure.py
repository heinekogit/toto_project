import unittest
from pathlib import Path

import pandas as pd

from analyze_d_scoop_structure import FEATURES, CALIPER, pair_close, boundary_pairs, score_type


class StructureAuditTests(unittest.TestCase):
    def test_close_pairs_are_unique_and_within_caliper(self):
        source=Path(__file__).resolve().parents[2]/'data/reports/d_scoop_phase1/all_diagnostics.csv'
        f=pd.read_csv(source)
        pairs=pair_close(f,within_round=True)
        self.assertEqual(len(pairs),18)
        self.assertTrue(pairs.P_D_gap.le(CALIPER+1e-15).all())
        self.assertTrue(pairs.D_round.eq(pairs.nonD_round).all())
        self.assertFalse(pairs[['D_round','D_match_no']].duplicated().any())
        self.assertFalse(pairs[['nonD_round','nonD_match_no']].duplicated().any())
        for field in FEATURES:
            expected=pairs[f'D_{field}']-pairs[f'nonD_{field}']
            pd.testing.assert_series_equal(expected,pairs[f'delta_{field}'],check_names=False)

    def test_boundary_and_score_classification(self):
        source=Path(__file__).resolve().parents[2]/'data/reports/d_scoop_phase1/all_diagnostics.csv'
        f=pd.read_csv(source)
        pairs=boundary_pairs(f)
        self.assertEqual(len(pairs),10)
        self.assertTrue(pairs.missed_D_rank.between(5,6).all())
        self.assertTrue(pairs.selected_nonD_rank.le(4).all())
        self.assertEqual(score_type('3–3'),'3–3以上')
        self.assertEqual(score_type('1–2'),'非D')


if __name__=='__main__': unittest.main()
