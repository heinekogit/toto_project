import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from style_official import apply_official,promote
from style_observation import archive_styles,verify
from style_shadow import VERSION,build_examples,predict
from random_remaining_v24 import prediction_fingerprint

class OfficialStyleTest(unittest.TestCase):
    def game(self):
        return pd.DataFrame([dict(match_id='j1_2026_test',league='j1',match_no=1,datetime='2027-01-02 14:00',forecast_created_at='2027-01-01 10:00',home_team='A',away_team='B',final_lambda_home=1.4,final_lambda_away=1.1,next_prob_home=.4,next_prob_draw=.3,next_prob_away=.3,next_uncertainty_calibration_mix=.2)])
    def styles(self):
        return pd.DataFrame([dict(team='A',style='ラインブレイクラン',metric='指数',value=60,selection_status='accepted_overview',first_observed_date='2026-12-31')])
    def comparison(self,model=None):
        g=self.game();c,_=predict(g,build_examples(g,self.styles()),model)
        c['source_match_id']=c.match_id
        return c
    def model(self):
        return dict(version=VERSION,enabled=True,cutoff='2027-01-01 09:00',columns=['f|own|ラインブレイクラン|指数'],mean=[50],scale=[10],beta=[.1],reason='synthetic_test_model')
    def test_zero_start_preserves_probabilities_and_purchase_seed(self):
        g=self.game();o=promote(g,self.comparison(),{'enabled':True})
        for s in ['home','draw','away']:
            self.assertEqual(g[f'next_prob_{s}'].iloc[0],o[f'next_prob_{s}'].iloc[0])
        self.assertEqual(prediction_fingerprint(g,'test'),prediction_fingerprint(o,'test'))
        self.assertFalse(o.style_official_model_enabled.iloc[0])
    def test_trained_correction_is_used_by_official_probability_columns(self):
        g=self.game();c=self.comparison(self.model());o=promote(g,c,{'enabled':True})
        self.assertAlmostEqual(o.final_lambda_home.iloc[0],1.4*np.exp(.1))
        self.assertEqual(o.next_prob_home.iloc[0],c.shadow_prob_home.iloc[0])
        self.assertEqual(o.pre_style_prob_home.iloc[0],g.next_prob_home.iloc[0])
        self.assertAlmostEqual(o.final_lambda_total.iloc[0],o.final_lambda_home.iloc[0]+1.1)
        from expected_goals import score_distribution
        d=score_distribution(o.final_lambda_home.iloc[0],1.1)
        self.assertAlmostEqual(o.next_prob_0_0.iloc[0],d['prob_0_0'])
        self.assertAlmostEqual(o.next_prob_home.iloc[0],.8*o.next_prob_pre_uncertainty_home.iloc[0]+.2/3)
    def test_policy_rollback_uses_exact_baseline(self):
        g=self.game();o=promote(g,self.comparison(self.model()),{'enabled':False})
        self.assertEqual(o.final_lambda_home.iloc[0],1.4)
        self.assertEqual(o.next_prob_home.iloc[0],.4)
        self.assertEqual(o.style_official_status.iloc[0],'disabled_by_policy')
    def test_double_application_rejected(self):
        o=promote(self.game(),self.comparison(),{'enabled':True})
        with self.assertRaisesRegex(ValueError,'already applied'):promote(o,self.comparison(),{'enabled':True})
    def test_freeze_and_official_apply_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'state';data=Path(directory)/'data';data.mkdir()
            config=Path(directory)/'policy.json';config.write_text(json.dumps({'version':'style_residual_deployment_v1','enabled':True}))
            archive_styles(self.styles(),2026,root=root,at='2027-01-01 08:00')
            first,audit=apply_official(self.game(),2026,config,root,data,at='2027-01-01 10:00')
            self.assertEqual(first.next_prob_home.iloc[0],.4)
            verify(Path(audit['path']))
            with patch('style_observation.train',return_value=self.model()):
                second,audit=apply_official(self.game(),2026,config,root,data,at='2027-01-01 11:00')
            self.assertGreater(second.final_lambda_home.iloc[0],1.4)
            self.assertEqual(audit['corrected_matches'],1)
            folder=Path(audit['path']);manifest=verify(folder)
            self.assertEqual(manifest['deployment'],'official')
            self.assertEqual(pd.read_csv(folder/'baseline.csv').final_lambda_home.iloc[0],1.4)
            self.assertTrue(json.loads((folder/'deployment_policy.json').read_text())['enabled'])

if __name__=='__main__':unittest.main()
