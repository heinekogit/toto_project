import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from style_observation import archive_styles, record_forecast, refresh, verify
from style_input_collection import assemble
from common import ROOT

class ObservationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'state';self.data=Path(self.temp.name)/'data';self.data.mkdir()
    def styles(self):
        return pd.DataFrame([dict(team=t,style='ラインブレイクラン',metric='指数',value=v,unit='',selection_status='accepted_overview') for t,v in [('A',70),('B',30)]])
    def predictions(self,n=1,start='2027-01-02'):
        rows=[]
        for i in range(n):
            dt=pd.Timestamp(start)+pd.Timedelta(days=i,hours=14)
            rows.append(dict(match_id=f'j1_2026_{i}',league='j1',datetime=str(dt),home_team='A',away_team='B',home_score=np.nan,away_score=np.nan,final_lambda_home=2.,final_lambda_away=2.,next_prob_home=.4,next_prob_draw=.2,next_prob_away=.4,next_uncertainty_calibration_mix=0.))
        return pd.DataFrame(rows)
    def results(self,predictions,h=4,a=0):
        r=predictions.copy();r['home_score']=h;r['away_score']=a
        r['datetime']=pd.to_datetime(r.datetime)+pd.Timedelta(minutes=3)
        r.to_csv(self.data/'j1_2026_latest_results.csv',index=False)
    def test_archive_and_earliest_prediction_are_frozen(self):
        archive_styles(self.styles(),2026,root=self.root,at='2027-01-01 09:00')
        p=self.predictions();before=p.copy(deep=True)
        first=record_forecast(p,2026,self.root,self.data,at='2027-01-01 10:00')
        pd.testing.assert_frame_equal(p,before)
        p['final_lambda_home']=3.
        record_forecast(p,2026,self.root,self.data,at='2027-01-01 11:00')
        self.results(p)
        _,s=refresh(self.root,self.data,at='2027-01-03 10:00')
        self.assertEqual(s['completed_matches'],1)
        paired=pd.read_csv(self.root/'reports/paired_results.csv')
        self.assertEqual(paired.final_lambda_home.iloc[0],2.)
        ledger=pd.read_csv(self.root/'result_observations.csv');available=ledger.result_available_at.iloc[0]
        refresh(self.root,self.data,at='2027-01-03 11:00')
        self.assertEqual(pd.read_csv(self.root/'result_observations.csv').result_available_at.iloc[0],available)
        path=Path(first['path']);(path/'baseline.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed frozen'):verify(path)
    def test_no_styles_preserves_and_records_baseline(self):
        saved=record_forecast(self.predictions(),2026,self.root,self.data,at='2027-01-01 10:00')
        self.assertIsNone(saved['style_source'])
        c=pd.read_csv(Path(saved['path'])/'comparison.csv')
        self.assertEqual(c.shadow_prob_home.iloc[0],.4)
        self.assertFalse(c.style_features_available.iloc[0])
    def test_only_past_results_train_then_later_frozen_prediction_scored(self):
        archive_styles(self.styles(),2026,root=self.root,at='2027-01-01 09:00')
        p=self.predictions(24)
        for i in range(24):
            day=pd.Timestamp(p.iloc[i].datetime).normalize()
            record_forecast(p.iloc[[i]],2026,self.root,self.data,at=str(day+pd.Timedelta(hours=10)))
            self.results(p.iloc[:i+1])
            refresh(self.root,self.data,at=str(day+pd.Timedelta(hours=18)))
        future=self.predictions(1,start='2027-01-28')
        saved=record_forecast(future,2026,self.root,self.data,at='2027-01-27 10:00')
        model=json.loads((Path(saved['path'])/'model.json').read_text())
        self.assertTrue(model['enabled']);self.assertEqual(model['train_matches'],24)
        allgames=pd.concat([p,future],ignore_index=True);self.results(allgames)
        _,status=refresh(self.root,self.data,at='2027-01-29 10:00')
        # Four earlier rolling predictions plus the final unseen fixture.
        self.assertEqual(status['learned_shadow_scores']['shadow']['matches'],5)
        self.assertNotIn('j1|2027-01-28|A|B',model['training_match_ids'])
        self.assertEqual(status['completed_matches'],25)
        model_after=json.loads((Path(saved['path'])/'model.json').read_text())
        self.assertEqual(model,model_after)
    def test_actual_archived_pages_reproduce_selection(self):
        folder=ROOT/'data/reports/style_value_investigation/sources'
        if not folder.exists():self.skipTest('local investigation fixtures absent')
        d,paths=assemble(ROOT/'external_contents/20261103/j1',lambda slug,code:folder/f'{slug}_{code}.html')
        self.assertEqual(len(d),840);self.assertEqual(d.value.isna().sum(),7)
        expected=pd.read_csv(ROOT/'data/reports/style_value_investigation/validation_inputs.csv')
        cols=['team','style','metric']
        a=d.set_index(cols).value.sort_index();b=expected.set_index(cols).value.sort_index()
        pd.testing.assert_series_equal(a,b)

if __name__=='__main__':unittest.main()
