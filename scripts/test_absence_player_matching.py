import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from build_absence_impact import _player_key, check_active_unmatched, check_player_master_freshness, match_player


class PlayerMatchingTest(unittest.TestCase):
    def test_variant_character_matches_exactly(self):
        players = pd.DataFrame({'player_name': ['山﨑 大地', '岩﨑 博', '髙田 颯也']})
        players['player_key'] = players.player_name.map(_player_key)
        for query, expected in [('山崎大地', '山﨑 大地'), ('岩崎博', '岩﨑 博'), ('高田颯也', '髙田 颯也')]:
            row, status, _ = match_player(players, query)
            self.assertEqual(status, 'matched')
            self.assertEqual(row.player_name, expected)

    def test_similar_names_are_not_aliases(self):
        self.assertNotEqual(_player_key('行友祐翔'), _player_key('行友翔哉'))
        row, status, _ = match_player(pd.DataFrame(columns=['player_key']), '川島康暉')
        self.assertIsNone(row)
        self.assertEqual(status, 'not_found')

    def test_production_rejects_mixed_age_player_master(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'players.csv'
            pd.DataFrame({'club_key': ['a', 'b'],
                          'fetched_at': ['2026-10-26 10:00:00', '2026-02-22 10:00:00']}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, '期限外'):
                check_player_master_freshness(path, '2026-10-26', 1)
            pd.DataFrame({'club_key': ['a', 'b'],
                          'fetched_at': ['2026-10-26 10:00:00', '2026-10-25 10:00:00']}).to_csv(path, index=False)
            check_player_master_freshness(path, '2026-10-26', 1)

    def test_production_rejects_unmatched_upcoming_absence(self):
        impacts = pd.DataFrame({
            'team': ['a', 'b'], 'player_name': ['new', 'old'],
            'match_status': ['not_found', 'not_found'],
            'start_date': ['2026-10-27', '2026-07-01'],
            'expected_return_date': ['2026-11-05', '2026-09-01'],
        })
        with self.assertRaisesRegex(ValueError, 'a:new'):
            check_active_unmatched(impacts, '2026-10-26')
        check_active_unmatched(impacts.iloc[[1]], '2026-10-26')


if __name__ == '__main__':
    unittest.main()
