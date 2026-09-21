import unittest

from build_batch_log_report import parse_log, render_html


class TestBatchLogReport(unittest.TestCase):
    def test_rankmot_skip_in_old_logs_requires_attention(self):
        parsed = parse_log([
            "[MERGE_QC][INFO] rankmot_future: 対象リーグとのチーム一致なし (target_teams=20, external_teams=20); 対象外データとしてスキップ",
            "[RESULT] prediction.py : OK",
        ])
        self.assertIn("順位情報の結合不一致", [i["name"] for i in parsed["top_issues"]])
        self.assertIn(">要確認</span>", render_html(parsed, "/tmp/test.log", "test"))

    def test_warning_and_error_status(self):
        for lines, status in [
            (["[RESULT] prediction.py : OK"], "概ね正常"),
            (["[WARN] check data", "[RESULT] prediction.py : OK"], "要確認"),
            (["[WARN] check data", "[RESULT] prediction.py : ERROR"], "ERRORあり"),
            (["[MERGE_QC][WARN] rankmot_future: 順位情報の未結合 matched=19/20"], "要確認"),
        ]:
            with self.subTest(status=status):
                self.assertIn(f">{status}</span>", render_html(parse_log(lines), "/tmp/test.log", "test"))

    def test_keeps_same_step_separately_for_each_league(self):
        parsed = parse_log(
            [
                "=== League: j1 / Season: 2026 ===\n",
                "[STEP] update.py : stats\n",
                "[RESULT] update.py : ERROR\n",
                "=== League: j2 / Season: 2026 ===\n",
                "[STEP] update.py : stats\n",
                "[RESULT] update.py : OK\n",
            ]
        )
        self.assertEqual(
            [(step["league"], step["name"], step["result"]) for step in parsed["steps"]],
            [("j1", "update.py", "ERROR"), ("j2", "update.py", "OK")],
        )
        report = render_html(parsed, "/tmp/run.log", "report")
        self.assertIn("<td>j1</td>", report)
        self.assertIn("<td>j2</td>", report)

    def test_summary_error_count_is_not_an_error_line(self):
        parsed = parse_log(["=== SUMMARY ===\n", "OK: 14\n", "ERROR: 1\n"])
        self.assertEqual(parsed["errors"], [])
        self.assertNotIn("実行時エラー", [issue["name"] for issue in parsed["top_issues"]])

    def test_probability_0429_is_not_rate_limit(self):
        parsed = parse_log(["[MAX_PROB_DIST] p25=0.429 max=0.464\n"])
        self.assertNotIn("レート制限", [issue["name"] for issue in parsed["top_issues"]])

    def test_http_429_is_rate_limit(self):
        parsed = parse_log(["request failed: HTTP 429 Too Many Requests\n"])
        self.assertIn("レート制限", [issue["name"] for issue in parsed["top_issues"]])

    def test_calibration_skip_is_not_runtime_error(self):
        parsed = parse_log(["[CALIBRATION][WARN] skipped due to error: 'prob_home_win_cal'\n"])
        self.assertNotIn("実行時エラー", [issue["name"] for issue in parsed["top_issues"]])


if __name__ == "__main__":
    unittest.main()
