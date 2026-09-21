import unittest
import os
import sys

SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from jleague_competition import resolve_competition_ids


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class TestJleagueCompetitionResolver(unittest.TestCase):
    def test_resolves_single_matching_competition(self):
        calls = []

        def fake_post(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(
                {"error": False, "messages": [], "data": [
                    {"selectValue": 725, "parentValue": 1}
                ]}
            )

        actual = resolve_competition_ids("2026", "1", post_func=fake_post)

        self.assertEqual(actual, "725")
        self.assertEqual(
            calls[0][1]["data"],
            {"competition_year": "2026", "competition_frame_id": "1"},
        )

    def test_override_skips_lookup(self):
        def unexpected_post(*args, **kwargs):
            raise AssertionError("lookup must not run when override is set")

        actual = resolve_competition_ids(
            "2026", "1", override="900, 901,900", post_func=unexpected_post
        )
        self.assertEqual(actual, "900,901")

    def test_rejects_empty_candidates(self):
        def fake_post(*args, **kwargs):
            return FakeResponse({"error": False, "messages": [], "data": []})

        with self.assertRaisesRegex(RuntimeError, "一意に自動解決"):
            resolve_competition_ids("2026", "1", post_func=fake_post)

    def test_rejects_multiple_candidates(self):
        def fake_post(*args, **kwargs):
            return FakeResponse(
                {"error": False, "messages": [], "data": [
                    {"selectValue": 725, "parentValue": 1},
                    {"selectValue": 999, "parentValue": 1},
                ]}
            )

        with self.assertRaisesRegex(RuntimeError, "一意に自動解決"):
            resolve_competition_ids("2026", "1", post_func=fake_post)

    def test_rejects_wrong_parent_frame(self):
        def fake_post(*args, **kwargs):
            return FakeResponse(
                {"error": False, "messages": [], "data": [
                    {"selectValue": 727, "parentValue": 2}
                ]}
            )

        with self.assertRaisesRegex(RuntimeError, r"candidates=\[\]"):
            resolve_competition_ids("2026", "1", post_func=fake_post)


if __name__ == "__main__":
    unittest.main()
