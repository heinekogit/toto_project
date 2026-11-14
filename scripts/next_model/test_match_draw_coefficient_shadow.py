import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from expected_goals import score_distribution
from match_draw_coefficient_shadow import extract,fit_reference,coefficients,FeatureConfig


class CoefficientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.frame=pd.read_csv(root/'data/eval/toto_rounds/toto1668/snapshot/next_predictions.csv')
        cls.features=extract(cls.frame);cls.reference=fit_reference(cls.features)

    def test_reconstruction_and_probability_accounting(self):
        f=self.features
        np.testing.assert_allclose(f.P_00+f.P_11+f.P_22plus,f.score_P_D,atol=1e-14)
        self.assertLess(f.HDA_reconciliation_error.max(),1e-10)
        self.assertTrue((f.balanced_near_given_non_draw<=f.P_margin1/(1-f.score_P_D)+1e-14).all())

    def test_no_substitution_on_missing_or_inconsistent_inputs(self):
        with self.assertRaises(ValueError):extract(self.frame.drop(columns=['final_lambda_home']))
        broken=self.frame.copy();broken.loc[0,'next_prob_0_0']+=.01
        with self.assertRaises(ValueError):extract(broken)

    def test_fixed_reference_is_independent_of_batch_and_outcomes(self):
        full=coefficients(self.features,self.reference)
        one=coefficients(self.features.iloc[[3]],self.reference)
        self.assertEqual(full.K_D.iloc[3],one.K_D.iloc[0])
        extra=self.features.copy();extra['actual_result']=np.arange(len(extra))%3
        self.assertEqual(fit_reference(extra),self.reference)

    def test_d_probability_and_feature_overlap_removed_at_reference(self):
        a=coefficients(self.features,self.reference)
        self.assertLess(abs(np.dot(a.zero_residual,a.P_D-a.P_D.mean())),1e-10)
        self.assertLess(abs(np.dot(a.zero_residual,a.near_residual)),1e-10)

    def test_same_final_draw_probability_different_score_structure(self):
        rows=[]
        for lh,la in [(1.3,1.3),(.8,1.8)]:
            d=score_distribution(lh,la);mix=(.28-d['prob_draw'])/(1/3-d['prob_draw'])
            r=dict(final_lambda_home=lh,final_lambda_away=la,next_uncertainty_calibration_mix=mix)
            for side in ['home','draw','away']:r['next_prob_'+side]=(1-mix)*d['prob_'+side]+mix/3
            for score in ['0_0','1_1','2_2']:r['next_prob_'+score]=d['prob_'+score]
            rows.append(r)
        f=extract(pd.DataFrame(rows));a=coefficients(f,self.reference)
        self.assertAlmostEqual(a.P_D.iloc[0],a.P_D.iloc[1],places=14)
        self.assertGreater(abs(a.K_D.iloc[0]-a.K_D.iloc[1]),1e-3)

    def test_bounds_neutral_strength_and_input_unchanged(self):
        original=self.features.copy(deep=True)
        a=coefficients(self.features,self.reference,FeatureConfig(strength=0))
        np.testing.assert_array_equal(a.K_D,np.ones(len(a)))
        b=coefficients(self.features,self.reference,FeatureConfig(lower=.99,upper=1.01))
        self.assertTrue(b.K_D.between(.99,1.01).all())
        pd.testing.assert_frame_equal(self.features,original)


if __name__=='__main__':unittest.main()
