import os
import sys
import unittest

SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from jleague_standings import parse_official_standings


def standings_html(rows):
    body = "".join(
        f"<tr><td>{rank}</td><td>{team}</td><td>{points}</td><td>{games}</td>"
        "<td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td></tr>"
        for rank, team, points, games in rows
    )
    return (
        "<table><thead><tr><th>順位</th><th>クラブ</th><th>勝点</th><th>試合数</th>"
        "<th>勝</th><th>分</th><th>負</th><th>得点</th><th>失点</th>"
        "<th>得失点</th><th>直近5試合</th></tr></thead><tbody>" + body + "</tbody></table>"
    )


class TestOfficialStandings(unittest.TestCase):
    def test_preseason_placeholders_become_neutral(self):
        df = parse_official_standings(
            standings_html([("-", "鹿島アントラーズ", "-", "-")]),
            season="2026", league="j1", fetched_date="20260722",
        )
        self.assertEqual(df.loc[0, "preseason"], 1)
        self.assertEqual(df.loc[0, "勝点"], 0)
        self.assertTrue(df.loc[0, "順位"] != df.loc[0, "順位"])
        self.assertEqual(df.loc[0, "competition_key"], "regular_2026_27")

    def test_active_values_are_numeric(self):
        df = parse_official_standings(
            standings_html([("1", "鹿島アントラーズ", "3", "1")]),
            season="2026", league="j1", fetched_date="20260810",
        )
        self.assertEqual(df.loc[0, "preseason"], 0)
        self.assertEqual(df.loc[0, "順位"], 1)
        self.assertEqual(df.loc[0, "勝点"], 3)


if __name__ == "__main__":
    unittest.main()
