import importlib.util
import json
import unittest
from pathlib import Path


def load_feedback():
    path = Path(__file__).parent / 'eval/13_build_condition_feedback.py'
    spec = importlib.util.spec_from_file_location('feedback_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConditionFeedbackTest(unittest.TestCase):
    def row(self, result):
        buy = {'p_home': .275, 'p_draw': .449, 'p_away': .276,
               **{f'ticket{i:02}': 2 if i <= 6 else 0 for i in range(1, 11)}}
        return {'round_id': 'test', 'match_no': 1, 'match_id': 'm', 'datetime': '2026-09-06',
                'league': 'J1', 'home_team': 'H', 'away_team': 'A', 'actual_result': result,
                'prediction_payload_json': json.dumps({'weather_missing': False, 'is_heavy_rain': True,
                                                       'is_strong_wind': False}),
                'buyplan_payload_json': json.dumps(buy)}

    def test_pre_match_flags_do_not_depend_on_result(self):
        module = load_feedback()
        home = module.analyze_row(self.row(1))
        draw = module.analyze_row(self.row(0))
        self.assertEqual(home['pre_match_flags'], draw['pre_match_flags'])
        self.assertEqual(home['measurement_issues'], draw['measurement_issues'])
        self.assertTrue(home['portfolio_uncovered'])
        self.assertFalse(draw['portfolio_uncovered'])
        self.assertEqual(home['actual_probability_rank'], 3)
        self.assertFalse(home['low_probability_result_25pct'])
        self.assertIsNone(home['short_rest_96h'])
        self.assertAlmostEqual(home['omitted_probability_mass'], .275)


if __name__ == '__main__':
    unittest.main()
