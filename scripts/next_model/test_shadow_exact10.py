import tempfile
import unittest
from pathlib import Path
import pandas as pd
from shadow_exact10 import generate,score
from common import ROOT,sha256

class ShadowExact10Tests(unittest.TestCase):
    def test_immutable_generate_and_separate_score(self):
        source=ROOT/'data/eval/toto_rounds/toto1670/snapshot'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            snapshot,created=generate(source/'next_predictions.csv',source/'buyplan_engine_manifest.json','toto1670',root,now_time='2026-11-07T09:00:00+09:00')
            self.assertTrue(created)
            original=sha256(snapshot/'tickets.csv')
            again,created=generate(source/'next_predictions.csv',source/'buyplan_engine_manifest.json','toto1670',root,now_time='2026-11-07T09:00:00+09:00')
            self.assertFalse(created);self.assertEqual(again,snapshot)
            pd.DataFrame({'match_no':range(1,14),'home_score':[1]*13,'away_score':[0]*13}).to_csv(root/'test_results.csv',index=False)
            scored,created=score(snapshot,root/'test_results.csv',root)
            self.assertTrue(created);self.assertTrue(scored.exists())
            self.assertEqual(sha256(snapshot/'tickets.csv'),original)
            _,created=score(snapshot,root/'test_results.csv',root)
            self.assertFalse(created)
    def test_reject_new_generation_after_kickoff(self):
        source=ROOT/'data/eval/toto_rounds/toto1670/snapshot'
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'after kickoff'):
                generate(source/'next_predictions.csv',source/'buyplan_engine_manifest.json','toto1670',Path(temp),now_time='2026-11-08T21:00:00+09:00')
    def test_reject_wrong_round(self):
        source=ROOT/'data/eval/toto_rounds/toto1670/snapshot'
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                generate(source/'next_predictions.csv',source/'buyplan_engine_manifest.json','toto1668',Path(temp))

if __name__=='__main__':unittest.main()
