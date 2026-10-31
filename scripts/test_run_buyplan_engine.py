import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_buyplan_engine import command_for


class BuyPlanEngineSwitchTest(unittest.TestCase):
    def args(self, engine):
        return argparse.Namespace(
            engine=engine, input_csv=Path("predictions.csv"), outdir=Path("out"),
            round="", logical_season=None, toto_order_csv=None, context_csv=None,
        )

    def test_legacy_command_remains_explicitly_available(self):
        command = command_for(self.args("legacy"))
        self.assertTrue(command[1].endswith("buyplan.py"))

    def test_next_requires_round_identity(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            command_for(self.args("next"))

    def test_next_command_enables_controlled_replacement(self):
        args = self.args("next")
        args.round = "toto1654"
        args.logical_season = 2027
        command = command_for(args)
        self.assertIn("--replace", command)


if __name__ == "__main__":
    unittest.main()
