import unittest
import numpy as np
from compare_d_scoop_uncertainty import ranks, compare_round


class Phase2Tests(unittest.TestCase):
    def test_ties_break_by_match_number(self):
        numbers = np.array([9, 3, 7, 1])
        values = np.array([.28, .28, .30, .28])
        np.testing.assert_array_equal(ranks(values, numbers), [4, 3, 1, 2])

    def test_real_exchanges_are_counted_once(self):
        for round_id in ('toto1660','toto1663','toto1664','toto1665','toto1668'):
            _, summary, changes, _ = compare_round(round_id)
            self.assertEqual(summary['replaced_matches'], 0)
            self.assertEqual(len(changes), 0)
        detail, summary, changes, _ = compare_round('toto1670')
        self.assertEqual(summary['replaced_matches'], 1)
        self.assertEqual(len(changes), 2)
        self.assertEqual({c['match_no'] for c in changes}, {2, 3})
        self.assertTrue(all(c['actual_symbol'] == 'D' for c in changes))
        self.assertEqual(summary['capture_delta_B_minus_A'], 0)
        self.assertEqual(int(detail['A_selected'].sum()), 4)
        self.assertEqual(int(detail['B_selected'].sum()), 4)


if __name__ == '__main__': unittest.main()
