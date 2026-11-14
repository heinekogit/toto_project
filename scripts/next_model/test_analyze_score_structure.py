import unittest

import numpy as np

from analyze_score_structure import adjacency, score_grid, structure, distance_rows


class ScoreStructureTest(unittest.TestCase):
    def test_partition_and_nonduplicated_draw_mass(self):
        matrix, retained = score_grid(1.5, 1.5)
        s = structure(matrix)
        self.assertGreater(retained, .99)
        self.assertAlmostEqual(s['P_draw'] + s['P_margin_1'] + s['P_margin_2plus'], 1)
        self.assertAlmostEqual(s['P_draw_neighbor_union'], s['P_draw'])
        self.assertEqual(len(adjacency(matrix, 'test', 1)), 20)
        self.assertAlmostEqual(distance_rows(matrix, 'test', 1).probability.sum(), 1)

    def test_one_goal_neighbors_and_distance(self):
        matrix, _ = score_grid(.8, .8)
        a = adjacency(matrix, 'test', 1)
        one_zero = a.loc[(a.home_goals == 1) & (a.away_goals == 0)].iloc[0]
        self.assertEqual((one_zero.adjacent_lower_draw, one_zero.adjacent_upper_draw), ('0-0', '1-1'))
        self.assertAlmostEqual(one_zero.lower_draw_probability, matrix[0, 0])
        self.assertAlmostEqual(one_zero.upper_draw_probability, matrix[1, 1])
        d = distance_rows(matrix, 'test', 1)
        self.assertEqual(int(d.loc[d.margin.eq(2), 'minimum_goal_changes_to_draw'].iloc[0]), 2)

    def test_low_and_high_equal_lambdas(self):
        low = structure(score_grid(.8, .8)[0])
        high = structure(score_grid(2.5, 2.5)[0])
        self.assertGreater(low['P_draw'], high['P_draw'])
        self.assertGreater(low['P_0_0'], high['P_0_0'])
        self.assertLess(low['P_draw_2plus'], high['P_draw_2plus'])
        self.assertLess(low['P_margin_2plus'], high['P_margin_2plus'])


if __name__ == '__main__':
    unittest.main()
