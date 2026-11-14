import unittest
import numpy as np
import pandas as pd
from style_shadow import VERSION,build_examples,corrected,predict,train,evaluate,timestamp

class StyleShadowTest(unittest.TestCase):
    def game(self):
        return pd.DataFrame([dict(match_id='x',datetime='2026-11-05 14:00',forecast_created_at='2026-11-04 12:00',home_team='A',away_team='B',final_lambda_home=1.4,final_lambda_away=1.1,next_prob_home=.4,next_prob_draw=.3,next_prob_away=.3,next_uncertainty_calibration_mix=.2)])
    def styles(self):
        return pd.DataFrame([dict(team='A',style='ラインブレイクラン',metric='指数',value=60,selection_status='accepted_overview',first_observed_date='2026-11-03')])
    def test_no_model_preserves_exact_probabilities(self):
        g=self.game();e=build_examples(g,self.styles());o,d=predict(g,e)
        self.assertEqual(o.shadow_prob_home.iloc[0],.4)
        self.assertEqual(o.shadow_lambda_home.iloc[0],1.4)
        self.assertEqual(d.contributions_json.tolist(),['{}','{}'])
    def test_sources_must_exist_at_forecast_creation_not_just_kickoff(self):
        g=self.game();g['forecast_created_at']='2026-11-03 23:59'
        e=build_examples(g,self.styles())
        self.assertTrue(e.filter(regex=r'^f\|').isna().all().all())
        g['forecast_created_at']='2026-11-04 00:00'
        self.assertEqual(build_examples(g,self.styles())['f|own|ラインブレイクラン|指数'].iloc[0],60)
    def test_postmatch_prediction_is_rejected(self):
        g=self.game();g['forecast_created_at']='2026-11-06'
        with self.assertRaisesRegex(ValueError,'before kickoff'):build_examples(g,self.styles())
    def test_ambiguous_value_not_consumed(self):
        s=self.styles();s['selection_status']='missing_ambiguous'
        self.assertTrue(build_examples(self.game(),s).filter(regex=r'^f\|').isna().all().all())
    def model(self):
        return dict(version=VERSION,enabled=True,cutoff='2026-11-04',columns=['f|own|ラインブレイクラン|指数'],mean=[50],scale=[10],beta=[.1],reason='test_only')
    def test_missing_features_have_zero_effect_and_keep_uncertainty(self):
        g=self.game();e=build_examples(g,self.styles());o,d=predict(g,e,self.model())
        self.assertAlmostEqual(o.shadow_lambda_home.iloc[0],1.4*np.exp(.1))
        self.assertEqual(o.shadow_lambda_away.iloc[0],1.1)
        from expected_goals import score_distribution
        p=score_distribution(1.4*np.exp(.1),1.1)
        self.assertAlmostEqual(o.shadow_prob_home.iloc[0],.8*p['prob_home']+.2/3)
        self.assertAlmostEqual(o.filter(regex='^shadow_prob_').sum(axis=1).iloc[0],1)
    def test_future_model_is_rejected(self):
        m=self.model();m['cutoff']='2026-11-05'
        with self.assertRaisesRegex(ValueError,'after this forecast'):corrected(build_examples(self.game(),self.styles()),m)
    def training(self):
        examples=[];results=[]
        for i in range(40):
            day=pd.Timestamp('2027-01-01')+pd.Timedelta(days=i)
            for side in ['home','away']:
                examples.append(dict(match_id=str(i),side=side,datetime=str(day+pd.Timedelta(hours=14)),forecast_created_at=str(day+pd.Timedelta(hours=10)),baseline_lambda=2.,source_available_at='2026-12-01',feature_version=VERSION,**{'f|signal':float(i%2)}))
            results.append(dict(match_id=str(i),home_score=4*(i%2),away_score=4*(i%2),result_available_at=str(day+pd.Timedelta(hours=17))))
        return pd.DataFrame(examples),pd.DataFrame(results)
    def test_residual_fit_learns_known_synthetic_signal(self):
        e,r=self.training();m=train(e,r,'2027-02-15')
        self.assertTrue(m['enabled']);self.assertGreater(m['beta'][0],0)
        self.assertLess(min(v['goal_nll'] for v in m['validation']),m['validation_baseline_nll'])
    def test_results_after_cutoff_are_excluded(self):
        e,r=self.training();m=train(e,r,'2027-01-04')
        self.assertFalse(m['enabled']);self.assertEqual(m['train_matches'],3)
    def test_all_missing_is_not_trainable(self):
        e,r=self.training();e['f|signal']=np.nan
        self.assertFalse(train(e,r,'2027-02-15')['enabled'])
    def test_no_residual_signal_selects_baseline(self):
        e,r=self.training();r['home_score']=2;r['away_score']=2
        m=train(e,r,'2027-02-15')
        self.assertFalse(m['enabled']);self.assertEqual(m['reason'],'time_holdout_prefers_baseline')
    def test_future_features_do_not_change_fit_or_scaling(self):
        e,r=self.training();m=train(e,r,'2027-01-31')
        e.loc[e.match_id.astype(int)>=30,'f|signal']=1000000
        other=train(e,r,'2027-01-31')
        self.assertEqual(m,other)
    def test_duplicate_source_rows_are_rejected(self):
        s=self.styles()
        with self.assertRaisesRegex(ValueError,'duplicate style'):build_examples(self.game(),pd.concat([s,s]))
    def test_evaluation_pairs_same_games(self):
        g=self.game();o,_=predict(g,build_examples(g,self.styles()))
        scores=evaluate(o,pd.DataFrame([dict(match_id='x',home_score=2,away_score=1)]))
        self.assertEqual(scores['baseline'],scores['shadow'])

if __name__=='__main__':unittest.main()
