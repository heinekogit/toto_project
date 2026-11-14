import unittest

from fetch_football_lab_html import (
    Target,
    supplement_team_style_targets,
    build_season_url,
    extract_title,
    football_lab_season_label,
    validate_season_html,
)
from build_football_lab_snapshot import parse_title_info


class TeamStyleCoverageTest(unittest.TestCase):
    def test_adds_seven_j1_pages_once_and_preserves_originals(self):
        original = Target("j1", "チームスタイル指標", "ショートカウンター",
                          "https://www.football-lab.jp/summary/team_style/j1001?year=100")
        targets = supplement_team_style_targets([original])
        self.assertEqual(len(targets), 8)
        self.assertEqual(targets[0], original)
        self.assertEqual(supplement_team_style_targets(targets), targets)
        for target in targets[1:]:
            self.assertIn("year=2026", build_season_url(target.url, "j1", 2026))

    def test_does_not_expand_j2_or_non_style_workbooks(self):
        for league, category in [("j2", "チームスタイル指標"), ("j1", "チームデータ")]:
            targets = [Target(league, category, "test", "https://example.test")]
            self.assertEqual(supplement_team_style_targets(targets), targets)

    def test_existing_page_under_alternate_name_is_not_duplicated(self):
        targets = [Target("j1", "チームスタイル指標", "別表記",
                          "https://www.football-lab.jp/summary/team_style/j1?data=66")]
        supplemented = supplement_team_style_targets(targets)
        self.assertEqual(len(supplemented), 7)
        self.assertEqual(supplemented[0], targets[0])


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
