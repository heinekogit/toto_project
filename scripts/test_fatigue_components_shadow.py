import unittest

from fatigue_components_shadow import finite_carry


class FatigueComponentsShadowTest(unittest.TestCase):
    def test_carry_uses_each_cause_once(self):
        history = [
            {"recovery": 1, "travel": 10, "external": 100},
            {"recovery": 2, "travel": 20, "external": 200},
            {"recovery": 3, "travel": 30, "external": 300},
        ]
        out = finite_carry(history)
        self.assertAlmostEqual(out["recovery"], 3*.35 + 2*.20 + 1*.10)
        self.assertAlmostEqual(out["travel"], 30*.35 + 20*.20 + 10*.10)
        self.assertAlmostEqual(out["external"], 300*.35 + 200*.20 + 100*.10)

    def test_old_aggregate_is_not_an_input(self):
        history = [{"recovery": 0, "travel": 4, "external": 0, "aggregate": 99}]
        self.assertAlmostEqual(finite_carry(history)["travel"], 1.4)

    def test_only_three_latest_raw_events_are_used(self):
        history = [{"recovery": value, "travel": 0, "external": 0} for value in [100, 1, 2, 3]]
        self.assertAlmostEqual(finite_carry(history)["recovery"], 3*.35 + 2*.20 + 1*.10)


if __name__ == "__main__":
    unittest.main()
