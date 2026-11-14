import unittest
import numpy as np
from team_conditions_research import *
from absence_adjustment import apply_absence_to_lambdas

class TeamConditionsTests(unittest.TestCase):
    def row(self,h=14,a=2):
        return dict(home_team='H',away_team='A',home_total_fatigue_score=h,away_total_fatigue_score=a,final_lambda_home=1.3,final_lambda_away=1.3,weather_missing=False,temp_kickoff=25,precip_kickoff=5,wind_kickoff=15)
    def params(self):
        return FatigueParameters(.01,.005,'SYNTHETIC TEST ONLY; not fitted or recommended')
    def test_swapped_fatigue_changes_direction(self):
        x=research_forecast(self.row(),fatigue_parameters=self.params());y=research_forecast(self.row(2,14),fatigue_parameters=self.params())
        self.assertEqual(x['fatigue_sum'],y['fatigue_sum']);self.assertEqual(x['fatigue_difference'],-y['fatigue_difference'])
        self.assertLess(x['lambda_home'],x['lambda_away']);self.assertGreater(y['lambda_home'],y['lambda_away'])
        np.testing.assert_allclose(x['probabilities'],y['probabilities'][::-1]);self.assertAlmostEqual(sum(x['probabilities']),1)
    def test_both_fatigued_not_one(self):
        x=research_forecast(self.row(14,14),fatigue_parameters=self.params());y=research_forecast(self.row(14,0),fatigue_parameters=self.params())
        self.assertNotEqual(x['lambda_away'],y['lambda_away']);self.assertAlmostEqual(x['lambda_home'],x['lambda_away'])
    def test_missing_not_zero(self):
        x=read_inputs(self.row(None,2));self.assertIsNone(x.home.fatigue);self.assertIsNone(x.fatigue_sum)
        with self.assertRaises(ValueError):research_forecast(self.row(None,2),fatigue_parameters=self.params())
    def test_no_coefficients_no_directional_adjustment(self):
        x=research_forecast(self.row());self.assertEqual(x['effects'],[]);self.assertAlmostEqual(x['lambda_home'],1.3);self.assertTrue(x['pending'])
    def test_absence_direction_and_no_reapplication(self):
        r=dict(absence_missing_home=False,absence_missing_away=False,absence_effective_attack_home=.4,absence_effective_defense_home=.2,absence_effective_attack_away=0,absence_effective_defense_away=0)
        x=apply_absence_to_lambdas(r,1.3,1.3);self.assertAlmostEqual(x['final_lambda_home'],1.2);self.assertAlmostEqual(x['final_lambda_away'],1.35)
        with self.assertRaises(ValueError):apply_team_effects(1.2,1.35,[Effect('absence','home',-.1,0,'duplicate')])
    def test_double_counting_rejected(self):
        for cause in ['fatigue','travel']:
            with self.assertRaises(ValueError):apply_research_mix([.4,.3,.3],fatigue_effect=.25,directional_causes=[cause])
        with self.assertRaises(ValueError):apply_team_effects(1,1,[Effect('fatigue','home',-.1,0,'test'),Effect('travel','home',-.1,0,'test')])
    def test_weather_no_unfounded_difference(self):
        with self.assertRaises(ValueError):research_forecast(self.row(),include_weather=True)
        spec=dict(precip_kickoff=dict(reference=0,attack_slope=-.01,defense_slope=0,evidence='synthetic test only'))
        x=research_forecast(self.row(),include_weather=True,weather_sensitivities={'home':spec,'away':spec});self.assertAlmostEqual(x['lambda_home'],x['lambda_away'])
        zero=dict(precip_kickoff=dict(reference=0,attack_slope=0,defense_slope=0,evidence='synthetic test only'))
        y=research_forecast(self.row(),include_weather=True,weather_sensitivities={'home':spec,'away':zero});self.assertLess(y['lambda_home'],y['lambda_away'])
    def test_timestamp_audit(self):
        self.assertEqual(observed_before_kickoff(None,'2026-11-01'), 'unknown')
        self.assertEqual(observed_before_kickoff('2026-10-31','2026-11-01'), 'prematch')
        self.assertEqual(observed_before_kickoff('2026-11-02','2026-11-01'), 'not_prematch')
    def test_mix_comparison(self):
        p=np.array([.5,.3,.2]);q,m=apply_research_mix(p,fatigue_effect=.25,strength=.5)
        self.assertAlmostEqual(m,.125);np.testing.assert_allclose(q,.875*p+.125/3)
    def test_external_unknown_blocks_fatigue(self):
        r=self.row();r['home_external_load_unknown']=True
        with self.assertRaises(ValueError):research_forecast(r,fatigue_parameters=self.params())
if __name__=='__main__':unittest.main()
