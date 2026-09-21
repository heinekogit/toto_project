import unittest

from fetch_football_lab_html import (
    build_season_url,
    extract_title,
    football_lab_season_label,
    validate_season_html,
)
from build_football_lab_snapshot import parse_title_info


class FootballLabSeasonUrlTest(unittest.TestCase):
    def test_converts_j1_legacy_url_and_preserves_data(self):
        url = "https://www.football-lab.jp/summary/team_ranking/j1001?year=100&data=expected"
        self.assertEqual(
            build_season_url(url, "j1", 2026),
            "https://www.football-lab.jp/summary/team_ranking/j1?year=2026&data=expected",
        )

    def test_converts_j2_legacy_url(self):
        url = "https://www.football-lab.jp/summary/team_style/j1002?year=100"
        self.assertEqual(
            build_season_url(url, "j2", 2026),
            "https://www.football-lab.jp/summary/team_style/j2?year=2026",
        )

    def test_season_title_validation(self):
        body = "<html><head><title>リーグサマリー：2026/27 J1 順位表</title></head></html>"
        self.assertEqual(football_lab_season_label(2026), "2026/27")
        self.assertIn("2026/27", extract_title(body))
        self.assertIn("2026/27", validate_season_html(body, 2026, "https://example.test"))

    def test_rejects_special_season_title(self):
        body = "<title>リーグサマリー：2026特別 J1百年構想リーグ 順位表</title>"
        with self.assertRaisesRegex(RuntimeError, "season mismatch"):
            validate_season_html(body, 2026, "https://example.test")

    def test_snapshot_parser_accepts_regular_season_title(self):
        season, league, page = parse_title_info(
            "リーグサマリー:2026/27 J1 チャンスビルディングポイント パスポイント ランキング | Football LAB",
            "j1",
        )
        self.assertEqual((season, league), ("2026", "j1"))
        self.assertIn("パスポイント", page)


if __name__ == "__main__":
    unittest.main()
