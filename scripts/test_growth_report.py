import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import pandas as pd

spec=importlib.util.spec_from_file_location('growth_report',Path(__file__).parent/'eval/20_build_growth_report.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class GrowthReportTests(unittest.TestCase):
    def fixture(self,root):
        snap=root/'snapshot';snap.mkdir()
        frame=pd.DataFrame([dict(match_no=i,home_team=f'H{i}',away_team=f'A{i}',
                                 next_prob_home=.5,next_prob_draw=.2,next_prob_away=.3,
                                 **{f'ticket{j:02}':1 for j in range(1,11)}) for i in range(1,14)])
        frame.to_csv(snap/'buyplan.csv',index=False)
        pd.DataFrame([dict(match_no=i,home_team=f'H{i}',away_team=f'A{i}',result=1 if i<=2 else None,status='OK' if i<=2 else 'PENDING') for i in range(1,14)]).to_csv(root/'actual_results.csv',index=False)
        (snap/'buyplan_engine_manifest.json').write_text(json.dumps({'generator':{'generator':'probability_allocation_ge10_v1'}}))

    def test_pending_matches_not_counted_as_misses_and_version_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            result=mod.analyze('toto1',root)
            self.assertEqual(result['version'],'v2.2')
            self.assertEqual(result['resolved'],2)
            self.assertEqual(result['official']['accuracy'],1)
            self.assertEqual(result['official']['best_hits'],2)
            self.assertIn('途中結果',mod.render_round(result))

    def test_v23_snapshot_is_reported_as_v23(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            manifest=root/'snapshot/buyplan_engine_manifest.json'
            manifest.write_text(json.dumps({'generator':{'generator':'relative_draw_graded_v23'},
                                            'shadow_observation':{'generator':'probability_allocation_ge10_v1'},
                                            'shadow_b_observation':{'generator':'anchor_graded_shadow'}}))
            base=pd.read_csv(root/'snapshot/buyplan.csv')
            shadow_b=base.rename(columns={f'ticket{i:02d}':f'ticket{i+20:02d}' for i in range(1,11)})
            shadow_b.to_csv(root/'snapshot/shadow_buyplan_b.csv',index=False)
            result=mod.analyze('toto1',root)
            self.assertEqual(result['version'],'v2.3')
            self.assertEqual(result['shadow_b_version'],'B：Anchor段階型')
            self.assertEqual(result['shadow_b']['ticket_count'],10)
            self.assertIn('シャドーB各口',mod.render_round(result))

    def test_mismatched_actual_card_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            actual=pd.read_csv(root/'actual_results.csv');actual.loc[0,'home_team']='WRONG';actual.to_csv(root/'actual_results.csv',index=False)
            with self.assertRaisesRegex(ValueError,'カード不一致'):mod.analyze('toto1',root)

    def test_shadow_c_is_scored_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            manifest=root/'snapshot/buyplan_engine_manifest.json'
            manifest.write_text(json.dumps({'generator':{'generator':'relative_draw_graded_v23'},
                                            'shadow_c_observation':{'generator':'free_coverage_shadow_c_v1'}}))
            base=pd.read_csv(root/'snapshot/buyplan.csv')
            shadow_c=base.rename(columns={f'ticket{i:02d}':f'ticket{i+30:02d}' for i in range(1,11)})
            shadow_c.to_csv(root/'snapshot/shadow_buyplan_c.csv',index=False)
            result=mod.analyze('toto1',root)
            self.assertEqual(result['shadow_c_version'],'C：採用数自由・Dカバー型 v1')
            self.assertEqual(result['shadow_c']['ticket_count'],10)
            self.assertIn('シャドーC各口',mod.render_round(result))

    def test_probability_schema_and_invalid_probability(self):
        self.assertEqual(mod.probabilities({'next_prob_home':.5,'next_prob_draw':.2,'next_prob_away':.3}),[.5,.2,.3])
        self.assertIsNone(mod.probabilities({'next_prob_home':.5,'next_prob_draw':.5,'next_prob_away':.3}))


if __name__=='__main__':unittest.main()
