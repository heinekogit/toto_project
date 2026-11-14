#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fatigue_merge import add_fatigue_time_aliases, validate_prediction_fatigue


def _load_module():
    path = Path(__file__).with_name("06_calculate_fatigue.py")
    spec = importlib.util.spec_from_file_location("fatigue_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FatigueComponentTest(unittest.TestCase):
    def test_unconfirmed_archive_does_not_block_confirmed_events(self):
        module = _load_module()
        path = Path(__file__).parents[1] / "data/manual/acl_schedule.csv"
        events = module.load_acl_team_events(path)
        self.assertEqual(len(events), 40)
        self.assertEqual(events.status.value_counts().to_dict(), {"scheduled": 22, "confirmed": 18})
        self.assertTrue(events.event_load.notna().all())

    def test_recent_unverified_completed_fixture_is_reported(self):
        module = _load_module()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "acl.csv"
            pd.DataFrame([{"match_date": "2026/11/03", "home_team": "柏", "away_team": "大田",
                           "city": "柏", "country": "日本", "status": "unverified", "source": ""}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "実施済みACL日程の確認が必要"):
                module.load_acl_team_events(path, as_of_date="2026/11/13")

    def test_original_schedule_without_added_gate_columns_is_usable(self):
        module = _load_module()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "acl.csv"
            pd.DataFrame([{"match_date": "2026/11/03", "competition": "ACL Elite", "home_team": "柏", "away_team": "大田",
                           "city": "柏", "country": "日本"},
                          {"match_date": "2026/11/24", "competition": "ACL Elite", "home_team": "京都", "away_team": "浦項",
                           "city": "京都", "country": "日本"}]).to_csv(path, index=False)
            events = module.load_acl_team_events(path, as_of_date="2026/11/13")
        self.assertEqual(events.status.tolist(), ["ledger", "scheduled"])
        self.assertTrue(events.source.notna().all())

    def test_acl_unknown_distance_stops_instead_of_zero(self):
        module = _load_module()
        for distance in [None, "unknown", float("inf"), -1]:
            with self.subTest(distance=distance), TemporaryDirectory() as directory:
                path = Path(directory) / "acl.csv"
                pd.DataFrame([{"match_date": "2026/10/13", "team": "鹿島",
                               "fatigue_grade": 7, "distance_km": distance,
                               "competition": "ACL", "status": "confirmed",
                               "source": "official"}]).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "ACL移動距離"):
                    module.load_acl_team_events(path)

    def test_acl_normalization_error_is_not_swallowed(self):
        module = _load_module()
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            path = Path(directory) / "acl.csv"
            pd.DataFrame([{"match_date": "2026/11/24", "home_team": "鹿島", "away_team": "相手", "city": "鹿嶋", "country": "日本"}]).to_csv(path, index=False)
            with patch.object(module, "normalize_acl_schedule", side_effect=ValueError("coordinates missing")):
                with self.assertRaisesRegex(ValueError, "coordinates missing"):
                    module.load_acl_team_events(path)

    def test_acl_uses_confirmed_appearance_and_generic_travel_load(self):
        module = _load_module()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "acl.csv"
            pd.DataFrame([
                {"match_date": "2026/10/13", "team": "鹿島", "fatigue_grade": 7,
                 "distance_km": 2000, "competition": "ACL", "status": "confirmed", "source": "official-result"},
                {"match_date": "2026/11/20", "team": "鹿島", "fatigue_grade": 7,
                 "distance_km": 2000, "competition": "ACL", "status": "scheduled", "source": ""},
            ]).to_csv(path, index=False)
            events = module.load_acl_team_events(path)
        self.assertEqual(len(events), 2)
        self.assertEqual(events.iloc[0].event_load, 2000 * module.TRAVEL_DISTANCE_WEIGHT)
        self.assertEqual(events.iloc[0].status, "confirmed")

    def test_missing_distance_is_explicitly_held_not_known_zero(self):
        module = _load_module()
        matches = pd.DataFrame([{'match_id': 'm', 'datetime': pd.Timestamp('2026-09-06'), 'home_team': 'H', 'away_team': 'A'}])
        result = module.calculate_fatigue(matches, pd.DataFrame())
        module.validate_fatigue_output(result)
        validate_prediction_fatigue(result)
        self.assertTrue(pd.isna(result.iloc[0].away_fatigue_score))
        self.assertEqual(result.iloc[0].away_fatigue_status, 'partial')
        corrupted = result.copy()
        corrupted['away_fatigue_score'] = 0.0
        with self.assertRaisesRegex(ValueError, 'FATIGUE_INPUT'):
            validate_prediction_fatigue(corrupted)

    def test_old_csv_is_rejected_even_if_scores_are_present(self):
        old = pd.DataFrame([{'home_fatigue_score': 1, 'away_fatigue_score': 4}])
        with self.assertRaisesRegex(ValueError, 'FATIGUE_INPUT'):
            validate_prediction_fatigue(old)

    def test_confirmed_extra_time_and_unknown_rotation_are_separate(self):
        import json
        module = _load_module()
        matches = pd.DataFrame([{'match_id': 'm', 'datetime': pd.Timestamp('2026-09-06 19:00'), 'home_team': '京都', 'away_team': '鹿島'}])
        events = pd.DataFrame([{'datetime': pd.Timestamp('2026-09-02 19:00'), 'team': '京都', 'event_type': 'cup',
                                'status': 'confirmed', 'minutes_played': 120, 'event_load': float('nan')}])
        result = module.calculate_fatigue(matches, pd.DataFrame(), events).iloc[0]
        detail = json.loads(result.home_external_events_json)[0]
        self.assertEqual(detail['extra_time_minutes'], 30)
        self.assertEqual(detail['extra_time_status'], 'confirmed')
        self.assertEqual(detail['rotation_status'], 'unknown')
        self.assertTrue(result.home_external_load_unknown)

    def test_scheduled_and_actual_kickoff_do_not_double_count(self):
        module = _load_module()
        rows = pd.DataFrame([
            {'match_id': 'scheduled', 'datetime': '2026-09-02 19:00', 'home_team': '東京V', 'away_team': 'Ｇ大阪', 'home_score': None, 'away_score': None},
            {'match_id': 'actual', 'datetime': '2026-09-02 19:03', 'home_team': '東京ヴェルディ', 'away_team': 'G大阪', 'home_score': 1, 'away_score': 0},
        ])
        result = module.dedupe_matches(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].match_id, 'actual')

    def test_components_preserve_total_and_expose_travel(self):
        module = _load_module()
        matches = pd.DataFrame([
            {"match_id": "m1", "datetime": pd.Timestamp("2026-08-29 19:00"),
             "home_team": "東京V", "away_team": "鹿島"},
            {"match_id": "m2", "datetime": pd.Timestamp("2026-09-02 19:00"),
             "home_team": "水戸", "away_team": "鹿島"},
        ])
        matrix = pd.DataFrame([[100.0]], index=["鹿島"], columns=["水戸"])
        result = module.calculate_fatigue(matches, matrix)
        target = result[result.match_id.eq("m2")].iloc[0]
        self.assertEqual(target.travel_lookup_status, "hit")
        self.assertAlmostEqual(target.away_rest_fatigue, 0.0)
        self.assertTrue(pd.isna(target.away_recent_load_carry))
        self.assertAlmostEqual(target.away_travel_fatigue, 0.5)
        self.assertAlmostEqual(target.away_away_condition_penalty, 4.0)
        component_total = (
            target.away_rest_fatigue + target.away_recent_load_carry
            + target.away_travel_fatigue + target.away_away_condition_penalty
        )
        self.assertTrue(pd.isna(target.away_fatigue_score))
        self.assertEqual(target.away_fatigue_known_subtotal, 4.5)

    def test_full_team_names_resolve_to_short_names(self):
        module = _load_module()
        matches = pd.DataFrame([{
            "match_id": "m1", "datetime": pd.Timestamp("2026-08-29 19:00"),
            "home_team": "東京V", "away_team": "鹿島",
        }])
        matrix = pd.DataFrame([[100.0]], index=["鹿島アントラーズ"], columns=["東京ヴェルディ"])
        target = module.calculate_fatigue(matches, matrix).iloc[0]
        self.assertEqual(target.travel_lookup_status, "hit")
        self.assertTrue(pd.isna(target.away_fatigue_score))
        self.assertEqual(target.away_fatigue_known_subtotal, 4.5)

    def test_missing_team_is_still_reported(self):
        module = _load_module()
        matches = pd.DataFrame([{'match_id': 'm', 'datetime': pd.Timestamp('2026-09-06'),
                                 'home_team': 'unknown', 'away_team': '鹿島'}])
        result = module.calculate_fatigue(matches, pd.DataFrame()).iloc[0]
        self.assertEqual(result.travel_lookup_status, 'matrix_empty')
        self.assertTrue(pd.isna(result.away_travel_distance_km))

    def test_rest_is_continuous_across_kickoff_minutes(self):
        module = _load_module()
        self.assertLess(abs(module.calc_rest_fatigue(95.9/24) - module.calc_rest_fatigue(96.1/24)), .01)

    def test_cup_appearance_affects_rest_without_inventing_load(self):
        module = _load_module()
        matches = pd.DataFrame([{'match_id': 'm', 'datetime': pd.Timestamp('2026-09-06 19:00'),
                                 'home_team': '京都', 'away_team': '鹿島'}])
        events = pd.DataFrame([{'datetime': pd.Timestamp('2026-09-02 19:00'), 'team': '京都サンガF.C.',
                                'event_type': 'league_cup', 'event_load': float('nan')},
                               {'datetime': pd.Timestamp('2026-09-09 19:00'), 'team': '京都',
                                'event_type': 'league_cup', 'event_load': 8}])
        result = module.calculate_fatigue(matches, pd.DataFrame(), events).iloc[0]
        self.assertEqual(result.home_rest_hours, 96)
        self.assertEqual(result.home_external_matches_last14d, 1)
        self.assertTrue(result.home_external_load_unknown)
        self.assertEqual(result.home_matches_last7d, 1)

    def test_time_alias_rescues_unique_same_matchup_only(self):
        left = pd.DataFrame([{
            "datetime": pd.Timestamp("2026-09-02 18:30"),
            "home_team": "京都", "away_team": "岡山",
        }])
        right = pd.DataFrame([{
            "datetime": pd.Timestamp("2026-09-02 18:33"),
            "home_team": "京都", "away_team": "岡山",
            "home_fatigue_score": 1.2, "away_fatigue_score": 5.6,
        }])
        keys = ["datetime", "home_team", "away_team"]
        augmented, rescued = add_fatigue_time_aliases(left, right, keys)
        self.assertEqual(rescued, 1)
        merged = left.merge(augmented, on=keys, how="left")
        self.assertEqual(merged.iloc[0].away_fatigue_score, 5.6)


if __name__ == "__main__":
    unittest.main()
