import unittest

import pandas as pd

from football_lab_merge import merge_football_lab_fixtures


class FixtureMergeTests(unittest.TestCase):
    def merge(self, target, sources):
        return merge_football_lab_fixtures(pd.DataFrame({'match_id': target}), pd.DataFrame({
            'match_id': sources, 'flab_chance_shots_home': list(range(10, 10 + len(sources))),
        }))

    def test_actual_incidents_and_values_preserved(self):
        targets = ['j1_2026_08071926_横浜FM_鹿島', 'j2_2026_08081450_札幌_徳島']
        sources = ['j1_2026_08071925_横浜FM_鹿島', 'j2_2026_08081445_札幌_徳島']
        out = self.merge(targets, sources)
        self.assertEqual(out.match_id.tolist(), targets)
        self.assertEqual(out.flab_match_source_id.tolist(), sources)
        self.assertEqual(out.flab_kickoff_delta_minutes.tolist(), [1, 5])
        self.assertEqual(out.flab_chance_shots_home.tolist(), [10, 11])

    def test_exact_wins_over_nearby(self):
        mid = 'j1_2026_08071926_横浜FM_鹿島'
        out = self.merge([mid], [mid, 'j1_2026_08071925_横浜FM_鹿島'])
        self.assertEqual(out.flab_match_method.iloc[0], 'exact')

    def test_normalized_names_and_ten_minute_boundary(self):
        out = self.merge(['j1_2026_08071910_Ｇ大阪_東京Ｖ'], ['j1_2026_08071900_G大阪_東京V'])
        self.assertEqual(out.flab_match_method.iloc[0], 'kickoff_alias')

    def test_other_date_season_league_direction_and_excess_drift_do_not_match(self):
        target = 'j1_2026_08071926_横浜FM_鹿島'
        for source in ['j1_2026_08061926_横浜FM_鹿島', 'j1_2025_08071926_横浜FM_鹿島',
                       'j2_2026_08071926_横浜FM_鹿島', 'j1_2026_08071926_鹿島_横浜FM',
                       'j1_2026_08071915_横浜FM_鹿島', 'malformed']:
            with self.subTest(source=source):
                out = self.merge([target], [source])
                self.assertEqual(out.flab_match_method.iloc[0], 'unmatched')
                self.assertTrue(pd.isna(out.flab_chance_shots_home.iloc[0]))

    def test_ambiguous_match_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.merge(['j1_2026_08071926_横浜FM_鹿島'],
                       ['j1_2026_08071925_横浜FM_鹿島', 'j1_2026_08071927_横浜FM_鹿島'])

    def test_nearest_kickoff_wins_over_older_alias(self):
        target = 'j2_2026_10111310_山形_仙台'
        sources = ['j2_2026_10111300_山形_仙台', 'j2_2026_10111305_山形_仙台']
        out = self.merge([target], sources)
        self.assertEqual(out.flab_match_source_id.iloc[0], sources[1])
        self.assertEqual(out.flab_kickoff_delta_minutes.iloc[0], 5)

    def test_source_cannot_be_reused(self):
        for second in ['1925', '1927']:
            with self.assertRaisesRegex(ValueError, 'source'):
                self.merge(['j1_2026_08071926_横浜FM_鹿島', f'j1_2026_0807{second}_横浜FM_鹿島'],
                           ['j1_2026_08071925_横浜FM_鹿島'])

    def test_duplicate_and_missing_ids_rejected(self):
        for targets in [[None], [''], ['x', 'x']]:
            with self.assertRaisesRegex(ValueError, 'match_id'):
                self.merge(targets, ['x'])


if __name__ == '__main__':
    unittest.main()
