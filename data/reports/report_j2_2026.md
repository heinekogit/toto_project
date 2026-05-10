# 解析内訳レポート (j2 2026)

## 入力ファイル
- csv_prev: /Users/dev_tomo/Desktop/tt_prj_restart/data/j2_2025_latest_results.csv
- prev_final_elo_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/j2_2025_final_elo.csv
- csv_season: /Users/dev_tomo/Desktop/tt_prj_restart/data/j2_2026_upcoming.csv
- team_master_stats_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/stats_snapshots/team_master_stats_j2_2026_asof_20260508.csv
- stats_asof: 2026-05-08
- absence_impact_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/absence_snapshots/absences_with_impact_asof_20260508.csv
- absence_asof: 20260508
- team_management_master_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/manual/team_management_master.csv
- team_motivation_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/j2_2026_motivation.csv
- team_travel_distances_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/team_travel_distances.csv
- team_fatigue_scores_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/team_fatigue_scores_j2_2026.csv
- acl_schedule_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/manual/acl_schedule.csv
- weather_cache_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/weather_snapshots/weather_features_j2_2026_asof_20260508.csv
- weather_asof: 20260508

## パラメータ
- INITIAL_ELO: 1500
- ELO_UPDATE_HOME_ADVANTAGE: 0.0
- HFA_ELO: 12.0
- ENABLE_HFA: 1
- ENABLE_MATCHUP_BIAS: 0
- MATCHUP_BIAS_COEF: 60.0
- HOME_ADV_ELO_COEF: 60.0
- HOME_ADV_PROFILE_DIFF_CLIP: 0.8
- ELO_DIFF_TEMPERATURE: 1.35
- ELO_D_VALUE: 550.0
- HFA_PROB_WEIGHT: 0.35
- MULTINOM_ELO_DIFF_SIGN: 1
- MULTINOM_SWAP_HA_OUTPUT: 0
- J1_WIN_PROB_CAP: 0.68
- GOAL_SCALING_FACTOR: 0.01
- FATIGUE_GOAL_SCALING: 0.01
- AWAY_PROB_MULTIPLIER: 1.05
- ACL_EFFECTIVE_DAYS: 5
- ENABLE_ROUND_TYPE_DRAW_CONTROL: 0
- ROUND_TYPE_DRAW_REL_THRESHOLD: 0.008
- ROUND_TYPE_DRAW_SHARE_THRESHOLD: 0.3
- ROUND_TYPE_DRAW_HEAVY_AVG: 0.335
- ROUND_TYPE_DRAW_LIGHT_AVG: 0.325
- ROUND_TYPE_DRAW_BOOST: 0.015
- RANK_MOTIVATION_GOAL_SCALING: 0.01
- WEATHER_PENALTY_HEAVY_RAIN: 0.15
- WEATHER_PENALTY_RAIN: 0.05
- WEATHER_PENALTY_STRONG_WIND: 0.1
- STATS_ASOF_DATE: 
- STATS_SNAPSHOT_NAME: 

## サマリ
- league: j2
- season_year: 2026
- generated_at: 2026-05-09T01:10:21
- all_teams: 26
- results_rows: 454
- future_matches: 16
- finished_matches: 74
- accuracy: 39.19% (29/74)

## 出力
- predictions: /Users/dev_tomo/Desktop/tt_prj_restart/j2_2026_predictions_hfa_on.csv
- backtest: /Users/dev_tomo/Desktop/tt_prj_restart/backtest_j2_2026.csv
- report_json: /Users/dev_tomo/Desktop/tt_prj_restart/data/reports/report_j2_2026.json
