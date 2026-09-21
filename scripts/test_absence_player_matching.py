import unittest
import pandas as pd
from build_absence_impact import _player_key, match_player


class PlayerMatchingTest(unittest.TestCase):
    def test_variant_character_matches_exactly(self):
        players = pd.DataFrame({'player_name': ['山﨑 大地', '岩﨑 博']})
        players['player_key'] = players.player_name.map(_player_key)
        for query, expected in [('山崎大地', '山﨑 大地'), ('岩崎博', '岩﨑 博')]:
            row, status, _ = match_player(players, query)
            self.assertEqual(status, 'matched')
            self.assertEqual(row.player_name, expected)

    def test_similar_names_are_not_aliases(self):
        self.assertNotEqual(_player_key('行友祐翔'), _player_key('行友翔哉'))
        row, status, _ = match_player(pd.DataFrame(columns=['player_key']), '川島康暉')
        self.assertIsNone(row)
        self.assertEqual(status, 'not_found')


if __name__ == '__main__':
    unittest.main()
