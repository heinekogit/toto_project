"""Regression checks using production functions without running prediction on import."""
import ast
import contextlib
import io
import tempfile
import unittest
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_scope():
    tree = ast.parse((ROOT / 'scripts/11_prediction_01.py').read_text())
    names = {'_normalize_team_text', 'canonical_team_name', 'normalize_team_series', 'merge_external_stats'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'TEAM_NAME_ALIAS_RAW_MAP', 'TEAM_NAME_ALIAS_MAP'} for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'TEAM_NAME_ALIAS_MAP.update':
            nodes.append(node)
    scope = {'pd': pd, 'unicodedata': unicodedata}
    def merge(left, right, *, stage, **kwargs):
        return left.merge(right, how='left', **kwargs)
    scope['audited_left_merge'] = merge
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<production-rankmot>', 'exec'), scope)
    return scope


class RankmotMergeTest(unittest.TestCase):
    def test_current_j1_j2_rankings_join_every_fixture(self):
        scope = load_scope()
        for league in ('j1', 'j2'):
            with self.subTest(league=league):
                matches = pd.read_csv(ROOT / f'data/{league}_2026_upcoming.csv')
                rankings = ROOT / f'data/{league}_2026_motivation.csv'
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = scope['merge_external_stats'](matches, rankings, merge_col_prefix='rankmot_', stage_label='rankmot_future')
                self.assertEqual(len(result), len(matches))
                for side in ('home', 'away'):
                    self.assertTrue(result[f'rankmot_rank_latest_{side}'].notna().all())
                self.assertNotIn('[WARN]', output.getvalue())
                self.assertNotIn('スキップ', output.getvalue())

    def test_different_clubs_and_unknown_suffix_are_not_conflated(self):
        canonical = load_scope()['canonical_team_name']
        self.assertEqual(canonical('アビスパ福岡　福岡'), canonical('福岡'))
        self.assertNotEqual(canonical('栃木ＳＣ'), canonical('栃木Ｃ'))
        self.assertNotEqual(canonical('未知のクラブ 福岡'), canonical('福岡'))

    def test_partial_and_zero_coverage_are_warned(self):
        scope = load_scope()
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / 'rankings.csv'
            for team in ('福岡', '未知'):
                pd.DataFrame({'team_name': [team], 'rank_latest': [1]}).to_csv(csv, index=False)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    scope['merge_external_stats'](pd.DataFrame({'home_team': ['福岡'], 'away_team': ['柏']}), csv, merge_col_prefix='rankmot_', stage_label='rankmot_future')
                self.assertIn('[MERGE_QC][WARN] rankmot_future: 順位情報の未結合', output.getvalue())


if __name__ == '__main__':
    unittest.main()
