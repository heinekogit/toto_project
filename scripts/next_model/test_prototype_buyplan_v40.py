import unittest
import numpy as np
import pandas as pd
from common import ROOT
from prototype_buyplan_v40 import final_state,greedy_relative_robust,portfolio
from analyze_score_structure_phase4b_stage1 import encode,decode

class V40PrototypeTests(unittest.TestCase):
    def test_phase3_final_mix_reproduces_saved_forecast(self):
        phase=pd.read_csv(ROOT/'data/reports/score_structure_phase3/scenario_structure.csv',encoding='utf-8-sig')
        forecast=pd.read_csv(ROOT/'data/eval/toto_rounds/toto1670/snapshot/next_predictions.csv',encoding='utf-8-sig').sort_values('match_no')
        _,p=final_state(phase[phase['round']=='toto1670'],'基準',0.)
        expected=forecast[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
        np.testing.assert_allclose(p,expected,rtol=0,atol=1e-10)
        _,q=final_state(phase[phase['round']=='toto1670'],'H低下・A上昇',.2)
        np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-12)
        self.assertTrue(np.all(q>=0))
    def test_robust_search_keeps_baseline_feasible(self):
        rng=np.random.default_rng(22)
        pool=np.arange(40)
        vals=rng.uniform(.01,1,size=(5,40))
        opt=np.maximum(vals.max(axis=1)*10,1)
        baseline=pool[:10]
        result=greedy_relative_robust(pool,vals,opt,baseline)
        self.assertEqual(len(set(result)),10)
        index=[int(np.where(pool==x)[0][0]) for x in result]
        self.assertGreaterEqual(float((vals[:,index]/opt[:,None]).sum(axis=1).min())+1e-12,
                                float((vals[:,baseline]/opt[:,None]).sum(axis=1).min()))

if __name__=='__main__':unittest.main()
