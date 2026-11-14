"""Production integration contracts; no purchase or archive writes."""
import sys,json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from forecast_pipeline import build_next_probabilities
from condition_policy import prepare_conditions,team_lambda,final_policy
from expected_goals import score_distribution
from uncertainty_calibration import apply_to_frame
ROOT=Path(__file__).resolve().parents[2]
class ProductionConditionsTest(unittest.TestCase):
 def setUp(self):
  self.rows=pd.read_csv(ROOT/'data/eval/toto_rounds/toto1670/snapshot/next_predictions.csv',low_memory=False).iloc[:2].copy()
  self.cfg=json.loads((ROOT/'data/models/next_conditions_v2.json').read_text())
 def test_default_is_raw_preserves_absence_and_lab(self):
  out=build_next_probabilities(self.rows,goal_rate_adjustment=False)
  self.assertTrue(out.next_uncertainty_calibration_mix.eq(0).all())
  self.assertTrue(out.team_lambda_delta_home.eq(0).all())
  self.assertTrue(out.team_lambda_delta_away.eq(0).all())
  for _,r in out.iterrows():
   d=score_distribution(r.final_lambda_home,r.final_lambda_away)
   np.testing.assert_allclose([r['next_prob_'+k] for k in ['home','draw','away']],[d['prob_'+k] for k in ['home','draw','away']],rtol=0,atol=1e-14)
  self.assertTrue(out.absence_adjustment_version.eq('absence_attack_defense_lambda_v1').all())
 def test_legacy_configuration_reproduces_previous_mix(self):
  out=build_next_probabilities(self.rows,goal_rate_adjustment=False)
  raw=out.drop(columns=[c for c in out if c.startswith('next_uncertainty_')]).copy()
  for k in ['home','draw','away']:raw['next_prob_'+k]=raw['next_prob_pre_uncertainty_'+k]
  expected=apply_to_frame(raw,.25,.25,probability_prefix='next_prob',audit_prefix='next_uncertainty')
  old=build_next_probabilities(self.rows,goal_rate_adjustment=False,config_path=ROOT/'data/models/uncertainty_calibration_v1.json')
  for k in ['home','draw','away']:
   np.testing.assert_allclose(expected['next_prob_'+k],old['next_prob_'+k],atol=1e-14)
   np.testing.assert_allclose(old['next_prob_'+k],out['next_prob_legacy_mix_'+k],atol=1e-14)
 def test_missing_does_not_turn_into_zero(self):
  r=self.rows.iloc[0].copy();r['home_rest_hours']=np.nan;r['home_total_fatigue_score']=0.;r['weather_missing']=True;r['absence_missing_home']=True
  z=prepare_conditions(r)
  self.assertTrue(pd.isna(z.home_total_fatigue_score));self.assertTrue(pd.isna(z.is_rain));self.assertTrue(pd.isna(z.absence_effective_attack_home))
  out=build_next_probabilities(pd.DataFrame([r]),goal_rate_adjustment=False)
  self.assertTrue(pd.isna(out.iloc[0].next_uncertainty_fatigue_signal));self.assertTrue(out.iloc[0].absence_input_missing_home)
 def test_uncalibrated_activation_refused(self):
  self.cfg['team_lambda']['enabled']=True
  with self.assertRaises(ValueError):team_lambda(self.rows.iloc[0],1.3,1.3,self.cfg)
 def test_invalid_strength_refused(self):
  out=build_next_probabilities(self.rows,goal_rate_adjustment=False)
  for x in [-.1,1.1,float('nan')]:
   self.cfg['mix_strength']=x
   with self.assertRaises(ValueError):final_policy(out,self.cfg)
if __name__=='__main__':unittest.main()
