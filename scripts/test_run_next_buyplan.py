import sys
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "next_model"))
from run_next_buyplan import select_round, build_next_probabilities, build_comparison_shadow, build_anchor_shadow
from build_next_buyplan import build_round
from expected_goals import base_lambdas_v1, score_distribution
from lab_matchup import apply_lab_matchup
from absence_adjustment import apply_absence_to_lambdas


class RunNextBuyPlanTest(unittest.TestCase):
    def test_condition_weights_are_shared_and_configurable_before_ticket_selection(self):
        snapshot = Path(__file__).resolve().parents[1] / 'data/eval/toto_rounds/toto1664/snapshot/next_predictions.csv'
        selected = pd.read_csv(snapshot, encoding='utf-8-sig').iloc[:1]
        default = build_next_probabilities(selected)
        with TemporaryDirectory() as directory:
            config = Path(directory) / 'conditions.json'
            config.write_text(json.dumps({
                'version': 'fatigue_weather_uniform_v1',
                'absence_goals_per_impact': 0.0,
                'fatigue_weight': 0.0,
                'weather_weight': 0.0,
                'max_mix': 0.25,
            }), encoding='utf-8')
            unadjusted = build_next_probabilities(selected, config_path=config)
        self.assertEqual(unadjusted.loc[0, 'next_uncertainty_calibration_mix'], 0.0)
        self.assertAlmostEqual(unadjusted.loc[0, 'next_prob_home'], unadjusted.loc[0, 'next_prob_pre_absence_home'])
        self.assertNotAlmostEqual(default.loc[0, 'next_prob_home'], unadjusted.loc[0, 'next_prob_home'])
        self.assertEqual(default.loc[0, 'condition_absence_goals_per_impact'], 0.25)

    def test_purchase_probabilities_come_from_score_distribution(self):
        snapshot = Path(__file__).resolve().parents[1] / 'data/eval/toto_rounds/toto1664/snapshot/next_predictions.csv'
        selected = pd.read_csv(snapshot, encoding='utf-8-sig')
        output = build_next_probabilities(selected)
        self.assertTrue(output['buyplan_probability_source'].eq('next_score_distribution_HDA + absence_attack_defense_lambda_v1 + fatigue_weather_uniform_v1').all())
        for idx, source in selected.iterrows():
            base_home, base_away, _ = base_lambdas_v1(source)
            lab = apply_lab_matchup(source, base_home, base_away)
            baseline = score_distribution(lab['final_lambda_home'], lab['final_lambda_away'])
            absence = apply_absence_to_lambdas(source, lab['final_lambda_home'], lab['final_lambda_away'])
            distribution = score_distribution(absence['final_lambda_home'], absence['final_lambda_away'])
            for side in ('home', 'draw', 'away'):
                self.assertAlmostEqual(output.loc[idx, f'next_prob_pre_absence_{side}'], baseline[f'prob_{side}'])
                self.assertAlmostEqual(output.loc[idx, f'next_prob_pre_uncertainty_{side}'], distribution[f'prob_{side}'])
                expected = ((1 - output.loc[idx, 'next_uncertainty_calibration_mix']) * distribution[f'prob_{side}']
                            + output.loc[idx, 'next_uncertainty_calibration_mix'] / 3)
                self.assertAlmostEqual(output.loc[idx, f'next_prob_{side}'], expected)
        self.assertNotAlmostEqual(output.loc[0, 'next_prob_home'], selected.loc[0, 'prob_final_home'])

    def test_default_applies_uncertainty_to_score_forecast(self):
        snapshot = Path(__file__).resolve().parents[1] / 'data/eval/toto_rounds/toto1664/snapshot/next_predictions.csv'
        selected = pd.read_csv(snapshot, encoding='utf-8-sig').iloc[:1]
        output = build_next_probabilities(selected)
        self.assertAlmostEqual(output.loc[0, 'next_prob_pre_absence_home'], 0.5040519005115558)
        self.assertNotAlmostEqual(output.loc[0, 'next_prob_home'], 0.5040519005115558)
        self.assertEqual(output.loc[0, 'buyplan_probability_source'],
                         'next_score_distribution_HDA + absence_attack_defense_lambda_v1 + fatigue_weather_uniform_v1')

    def test_prediction_final_probabilities_are_not_required(self):
        snapshot = Path(__file__).resolve().parents[1] / 'data/eval/toto_rounds/toto1664/snapshot/next_predictions.csv'
        selected = pd.read_csv(snapshot, encoding='utf-8-sig').iloc[:1].copy()
        selected.loc[selected.index[0], 'prob_final_draw'] = float('nan')
        output = build_next_probabilities(selected)
        self.assertAlmostEqual(output.loc[0, 'next_prob_pre_absence_home'], 0.5040519005115558)

    def test_shadow_ab_keep_their_ten_ticket_algorithms(self):
        frame = pd.DataFrame([dict(match_no=i+1, home_team=f'H{i}',away_team=f'A{i}',league='j1',
                                  next_prob_home=.40,next_prob_draw=.33,next_prob_away=.27,
                                  final_lambda_home=1.2,final_lambda_away=1.1,
                                  next_prob_0_0=.09,next_prob_1_1=.11,next_prob_2_2=.03,
                                  lab_basis_hint='basis_balanced') for i in range(13)])
        reference,_=build_round(frame,strategy='probability_allocation_ge10_v1')
        shadow,audit=build_comparison_shadow(frame)
        for i in range(1,11):
            pd.testing.assert_series_equal(reference[f'ticket{i:02d}'],shadow[f'ticket{i+10:02d}'],check_names=False)
        self.assertEqual(audit['shadow_ticket_count'],10)
        self.assertTrue(audit['cross_version_duplicates_allowed'])
        self.assertEqual(audit['version'],'v2.2')
        anchor_reference,_=build_round(frame,strategy='anchor_graded_shadow')
        shadow_b,audit_b=build_anchor_shadow(frame)
        for i in range(1,11):
            pd.testing.assert_series_equal(anchor_reference[f'ticket{i:02d}'],shadow_b[f'ticket{i+20:02d}'],check_names=False)
        self.assertEqual(audit_b['version'],'Anchor段階型')
    def test_round_mismatch_stops(self):
        order = pd.DataFrame({
            "match_no": range(1, 14),
            "home_team": [f"H{i}" for i in range(13)],
            "away_team": [f"A{i}" for i in range(13)],
        })
        predictions = order.iloc[:-1][["home_team", "away_team"]].copy()
        with self.assertRaisesRegex(ValueError, "another/incomplete round"):
            select_round(predictions, order, "toto9999")


if __name__ == "__main__":
    unittest.main()
