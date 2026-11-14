import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from common import ROOT,sha256
from forecast_registry import register_forecast,register_results,read_log,selected_matches,load_bundle


class ForecastRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.registry=self.root/'registry'
        sample=ROOT/'data/eval/toto_rounds/toto1660'
        self.forecast=self.root/'next_predictions.csv'
        shutil.copyfile(sample/'snapshot/next_predictions.csv',self.forecast)
        original=json.loads((sample/'snapshot/buyplan_engine_manifest.json').read_text())
        self.manifest=self.root/'manifest.json'
        original['output_sha256']['next_predictions.csv']=sha256(self.forecast)
        self.manifest.write_text(json.dumps(original))
        self.results=self.root/'actual_results.csv'
        shutil.copyfile(sample/'actual_results.csv',self.results)

    def test_idempotence_separate_results_and_correction_history(self):
        self.assertEqual(register_forecast(self.forecast,self.manifest,self.registry)['added'],13)
        self.assertEqual(register_forecast(self.forecast,self.manifest,self.registry)['added'],0)
        initial=sha256(self.registry/'forecasts.jsonl')
        self.assertEqual(register_results(self.results,'toto1660',self.registry)['added'],13)
        self.assertEqual(register_results(self.results,'toto1660',self.registry)['added'],0)
        self.assertEqual(sha256(self.registry/'forecasts.jsonl'),initial)
        d=pd.read_csv(self.results);d.loc[0,'home_score']=2;d.loc[0,'away_score']=0;d.loc[0,'result']=1
        d.to_csv(self.results,index=False)
        self.assertEqual(register_results(self.results,'toto1660',self.registry)['added'],1)
        events=read_log(self.registry/'results.jsonl')
        self.assertEqual(len(events),14)
        self.assertEqual(events[-1]['correction_of'],events[0]['event_id'])
        self.assertEqual(len(selected_matches(self.registry)),13)
        self.assertEqual(sha256(self.registry/'forecasts.jsonl'),initial)

    def test_multiple_snapshots_one_fixture(self):
        register_forecast(self.forecast,self.manifest,self.registry)
        d=pd.read_csv(self.forecast);d['observation_note']='second snapshot';d.to_csv(self.forecast,index=False)
        m=json.loads(self.manifest.read_text());m['created_at']='2026-10-10T12:00:00+09:00'
        m['output_sha256']['next_predictions.csv']=sha256(self.forecast)
        self.manifest.write_text(json.dumps(m))
        self.assertEqual(register_forecast(self.forecast,self.manifest,self.registry)['added'],13)
        self.assertEqual(len(read_log(self.registry/'forecasts.jsonl')),26)
        selected=selected_matches(self.registry)
        self.assertEqual(len(selected),13)
        self.assertTrue(selected.forecast_created_at.eq('2026-10-10T11:11:18+09:00').all())

    def test_postkickoff_rejected(self):
        m=json.loads(self.manifest.read_text());m['created_at']='2026-10-11T20:00:00+09:00'
        self.manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'Post-kickoff'):
            load_bundle(self.forecast,self.manifest)
        self.assertFalse((self.registry/'forecasts.jsonl').exists())


if __name__=='__main__':unittest.main()
