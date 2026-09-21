import ast
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "11_prediction_01.py"


def load_acl_away_travel_types() -> set[str]:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "ACL_AWAY_TRAVEL_TYPES" for target in node.targets):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("ACL_AWAY_TRAVEL_TYPES not found")


class AclTravelTypesTest(unittest.TestCase):
    def test_existing_csv_long_types_receive_away_bonus(self):
        travel_types = load_acl_away_travel_types()
        self.assertIn("short", travel_types)
        self.assertIn("medium", travel_types)
        self.assertIn("long", travel_types)
        self.assertIn("long_haul", travel_types)

    def test_home_and_no_travel_do_not_receive_away_bonus(self):
        travel_types = load_acl_away_travel_types()
        self.assertNotIn("home", travel_types)
        self.assertNotIn("none", travel_types)


if __name__ == "__main__":
    unittest.main()
