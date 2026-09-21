import unittest
from fetch_jleague_club_players import parse_current_player_page


class CurrentPlayerPageTest(unittest.TestCase):
    def test_current_table_keeps_name_and_stats_separate(self):
        row = '''<tr><th class="o-table__cell--player"><p class="o-table__player-position">DF 4</p><a class="o-table__player-name-link">岩﨑 博</a><span>HG</span></th><td class="o-table__cell--date-of-birth">2002/10/11</td><td class="o-table__cell--height-weight">186 / 80</td><td class="o-table__cell--number-of-games-played">5</td><td class="o-table__cell--goals-scored">0</td></tr>'''
        df = parse_current_player_page('<table><tbody>' + row + row + '</tbody></table>')
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0].player_name, '岩﨑 博')
        self.assertEqual(df.iloc[0]['出場 試合数 ※2'], '5')
        self.assertEqual(df.iloc[0].position, 'DF')

    def test_empty_response_is_not_a_valid_master(self):
        with self.assertRaises(RuntimeError):
            parse_current_player_page('<html>Not found</html>')


if __name__ == '__main__':
    unittest.main()
