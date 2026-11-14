"""Prospective condition-S capture and result-link invariants."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

from condition_s_registry import VERSION, capture, classify, observations, record_deadline, record_failure, status, sync_results
from forecast_registry import append_unique, read_log


ROOT = Path(__file__).resolve().parents[2]
FORECAST = ROOT / "data/purchase_reference/next_predictions.csv"
MANIFEST = ROOT / "data/purchase_reference/manifest.json"


class ConditionSRegistryTest(unittest.TestCase):
    def test_inclusive_unrounded_boundaries(self):
        p00 = np.array([0.123456789012, 0.2, 0.3], dtype=float)
        p11 = np.array([0.3, 0.2, 0.1], dtype=float)
        flags, median00, median11 = classify(p00, p11)
        self.assertEqual(flags.tolist(), [True, True, False])
        self.assertEqual((median00, median11), (0.2, 0.2))

    def test_saved_snapshot_definition_and_idempotence(self):
        records = observations(FORECAST, MANIFEST)
        self.assertEqual(len(records), 13)
        frame = pd.read_csv(FORECAST).sort_values("match_no")
        median00 = frame["next_prob_0_0"].median()
        median11 = frame["next_prob_1_1"].median()
        expected = [int(r.match_no) for _, r in frame.iterrows()
                    if r.next_prob_0_0 <= median00 and r.next_prob_1_1 >= median11]
        self.assertEqual([r["match_no"] for r in records if r["condition_S"]], expected)
        self.assertTrue(all(r["condition_version"] == VERSION for r in records))
        self.assertTrue(all(r["provenance_class"] == "A" for r in records))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("condition_s_registry.now", return_value=records[0]["forecast_created_at"]):
                self.assertEqual(capture(FORECAST, MANIFEST, root=root)["added"], 13)
                self.assertEqual(capture(FORECAST, MANIFEST, root=root)["added"], 0)
            with self.assertRaisesRegex(ValueError, "after first kickoff"):
                capture(FORECAST, MANIFEST, root=root)
            self.assertEqual(status(root=root)["forecasts"], 13)
            self.assertEqual(status(root=root)["selected_snapshots"][0]["purchase_eligibility"], "unknown")
            deadline = pd.Timestamp(records[0]["kickoff"])
            if deadline.tzinfo is None:
                deadline = deadline.tz_localize("Asia/Tokyo")
            record_deadline(records[0]["round_id"], deadline.isoformat(), "test fixture", root=root)
            self.assertEqual(status(root=root)["selected_snapshots"][0]["purchase_eligibility"], "before_deadline")
            self.assertTrue(all(r["input_sha256"] for r in read_log(root / "condition_s_forecasts.jsonl")))
            record_failure("capture", "test failure", round_id="toto-test", root=root)
            self.assertEqual(status(root=root)["failure_events"], 1)

    def test_result_link_is_separate_and_idempotent(self):
        records = observations(FORECAST, MANIFEST)
        target = next(r for r in records if r["condition_S"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("condition_s_registry.now", return_value=records[0]["forecast_created_at"]):
                capture(FORECAST, MANIFEST, root=root)
            self.assertEqual(sync_results(root=root)["matched"], 0)
            result = dict(event_id="test_result", fixture_id=target["fixture_id"],
                          result=0, home_score=1, away_score=1,
                          result_recorded_at="2026-11-14T00:00:00+09:00")
            append_unique(root, "results.jsonl", [result])
            self.assertEqual(sync_results(root=root)["added"], 1)
            self.assertEqual(sync_results(root=root)["added"], 0)
            linked = read_log(root / "condition_s_results.jsonl")
            self.assertEqual(linked[0]["condition_s_event_id"], target["event_id"])
            self.assertTrue(linked[0]["actual_D"])
            metrics = [g for g in status(root=root)["evaluation"] if g["condition_S"]]
            self.assertEqual(metrics[0]["n"], 1)
            self.assertEqual(metrics[0]["actual_D"], 1)
            self.assertAlmostEqual(metrics[0]["expected_D"], target["final_prob_draw"])
            corrected = dict(result, event_id="test_correction", result=1, home_score=2, away_score=1,
                             result_recorded_at="2026-11-15T00:00:00+09:00")
            append_unique(root, "results.jsonl", [corrected])
            self.assertEqual(sync_results(root=root)["added"], 1)
            self.assertEqual(sync_results(root=root)["added"], 0)
            self.assertEqual(len(read_log(root / "condition_s_results.jsonl")), 2)
            metrics = [g for g in status(root=root)["evaluation"] if g["condition_S"]]
            self.assertEqual(metrics[0]["actual_D"], 0)


if __name__ == "__main__":
    unittest.main()
