import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "next_model"))
from build_next_buyplan import build_round
from random_remaining_v24 import allocate_per_match, generate, prediction_fingerprint, seed_from_fingerprint
from run_next_buyplan import write_html


class BuyPlanV24Test(unittest.TestCase):
    def test_largest_remainder_is_independent_for_each_match(self):
        p = np.tile([0.51, 0.26, 0.23], (13, 1))
        counts = allocate_per_match(p)
        np.testing.assert_array_equal(counts, np.tile([5, 3, 2], (13, 1)))
        self.assertEqual(counts.sum(axis=0).tolist(), [65, 39, 26])
        tie = np.tile([0.35, 0.35, 0.30], (13, 1))
        np.testing.assert_array_equal(allocate_per_match(tie)[0], [4, 3, 3])

    def test_reproducible_output_and_display_contract(self):
        source = Path(__file__).resolve().parents[1] / "data/eval/toto_rounds/toto1664/snapshot/next_predictions.csv"
        frame = pd.read_csv(source, encoding="utf-8-sig")
        fingerprint = prediction_fingerprint(frame, "toto1664")
        seed = seed_from_fingerprint(fingerprint)
        plan, audit = build_round(frame, strategy="random_remaining_v24", seed=seed)
        same, same_audit = build_round(frame, strategy="random_remaining_v24", seed=seed)
        pd.testing.assert_frame_equal(plan, same)
        self.assertEqual(audit["allocation"], same_audit["allocation"])
        self.assertEqual(len(plan), 13)
        self.assertEqual(set(plan.match_no), set(range(1, 14)))
        self.assertNotIn("ticket00", plan)
        self.assertNotIn("placement_score", audit)
        self.assertEqual(audit["ticket_display_order"], "draw_order_01_to_10")
        columns = [f"ticket{i:02d}" for i in range(1, 11)]
        self.assertTrue(all(column in plan for column in columns))
        counts = np.asarray(audit["allocation"]["allocation_HDA"])
        for symbol, value in enumerate([1, 0, 2]):
            np.testing.assert_array_equal((plan[columns] == value).sum(axis=1), counts[:, symbol])
        changed = frame.copy()
        changed.loc[0, "next_prob_home"] += 0.001
        changed.loc[0, "next_prob_away"] -= 0.001
        self.assertNotEqual(fingerprint, prediction_fingerprint(changed, "toto1664"))
        with TemporaryDirectory() as directory:
            html = Path(directory) / "buyplan.html"
            write_html(plan, "toto1664", html, generator_audit=audit["allocation"])
            page = html.read_text(encoding="utf-8")
            self.assertIn("BuyPlan v2.4", page)
            self.assertNotIn("候補00", page)
            self.assertNotIn("配券スコア", page)

    def test_duplicate_detection_without_repair(self):
        p = np.tile([1.0, 0.0, 0.0], (13, 1))
        tickets, audit = generate(p, 42)
        self.assertEqual(audit["duplicate_count"], 45)
        self.assertEqual(audit["unique_ticket_count"], 1)
        self.assertTrue((tickets == 0).all())


if __name__ == "__main__":
    unittest.main()
