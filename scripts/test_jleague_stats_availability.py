import os
import sys
import tempfile
import unittest

import pandas as pd

SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from jleague_stats_availability import (
    find_latest_stats_fallback,
    is_unavailable_page,
    validate_stats_fallback,
)


class TestStatsAvailability(unittest.TestCase):
    def test_detects_current_not_found_page(self):
        html = "<html><head><title>お探しのページは見つかりませんでした</title></head></html>"
        self.assertTrue(is_unavailable_page(html))

    def test_detects_nextjs_404_shell(self):
        html = '<html id="__next_error__"><script>"NEXT_HTTP_ERROR_FALLBACK;404"</script></html>'
        self.assertTrue(is_unavailable_page(html))

    def test_detects_season_stats_not_published_page(self):
        html = (
            "<html><head><title>J1 クラブスタッツ</title></head>"
            "<body><p>当該シーズンのスタッツデータはございません。</p></body></html>"
        )
        self.assertTrue(is_unavailable_page(html))

    def test_does_not_classify_active_page_as_unavailable(self):
        html = "<html><head><title>J1 チームスタッツ</title></head><body>ランキング</body></html>"
        self.assertFalse(is_unavailable_page(html))

    def test_validates_nonempty_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "stats.csv")
            pd.DataFrame({"team_name": ["鹿島"], "シュート": [10.0]}).to_csv(path, index=False)
            result = validate_stats_fallback(path)
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["filled_values"], 1)

    def test_rejects_empty_value_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "stats.csv")
            pd.DataFrame({"team_name": ["鹿島"], "シュート": [None]}).to_csv(path, index=False)
            self.assertIsNone(validate_stats_fallback(path))

    def test_prefers_current_file_over_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data")
            snapshots = os.path.join(data_dir, "stats_snapshots")
            os.makedirs(snapshots)
            current = os.path.join(data_dir, "team_master_stats_j1_2026.csv")
            snapshot = os.path.join(snapshots, "team_master_stats_j1_2026_asof_20260530.csv")
            pd.DataFrame({"team_name": ["鹿島"], "x": [1]}).to_csv(current, index=False)
            pd.DataFrame({"team_name": ["鹿島"], "x": [2]}).to_csv(snapshot, index=False)
            result = find_latest_stats_fallback(temp_dir, "j1", "2026")
            self.assertEqual(result["path"], current)


if __name__ == "__main__":
    unittest.main()
