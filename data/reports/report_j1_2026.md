# 解析内訳レポート (j1 2026)

## 入力ファイル
- csv_prev: /Users/dev_tomo/Desktop/tt_prj_restart/data/j1_2025_latest_results.csv
- prev_final_elo_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/j1_2025_final_elo.csv
- csv_season: /Users/dev_tomo/Desktop/tt_prj_restart/data/j1_2026_upcoming.csv
- team_master_stats_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/stats_snapshots/team_master_stats_j1_2026_asof_20260228.csv
- stats_asof: 2026-02-28
- absence_impact_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/absence_snapshots/absences_with_impact_asof_20260228.csv
- absence_asof: 20260228
- team_management_master_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/manual/team_management_master.csv
- team_motivation_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/j1_2026_motivation.csv
- team_travel_distances_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/team_travel_distances.csv
- team_fatigue_scores_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/team_fatigue_scores_j1_2026.csv
- weather_cache_csv: /Users/dev_tomo/Desktop/tt_prj_restart/data/weather_snapshots/weather_features_j1_2026_asof_20260228.csv
- weather_asof: 20260228

## パラメータ
- INITIAL_ELO: 1500
- ELO_UPDATE_HOME_ADVANTAGE: 0.0
- HFA_ELO: 35.0
- ENABLE_HFA: True
- HOME_ADV_ELO_COEF: 60.0
- HOME_ADV_PROFILE_DIFF_CLIP: 0.8
- HFA_ABS_MAX: 60.0
- HFA_DATA_QUALITY_MULT: 0.3
- HFA_STATS_MISSING_MULT: 0.0
- ELO_DIFF_TEMPERATURE: 1.35
- J1_WIN_PROB_CAP: 0.68
- GOAL_SCALING_FACTOR: 0.01
- FATIGUE_GOAL_SCALING: 0.01
- RANK_MOTIVATION_GOAL_SCALING: 0.01
- WEATHER_PENALTY_HEAVY_RAIN: 0.15
- WEATHER_PENALTY_RAIN: 0.05
- WEATHER_PENALTY_STRONG_WIND: 0.1
- STATS_ASOF_DATE: 2026-02-28
- STATS_SNAPSHOT_NAME: 

## サマリ
- league: j1
- season_year: 2026
- generated_at: 2026-02-28T08:24:05
- all_teams: 24
- results_rows: 380
- future_matches: 179
- finished_matches: 0
- accuracy: None% (0/0)

## 出力
- predictions: /Users/dev_tomo/Desktop/tt_prj_restart/j1_2026_predictions.csv
- backtest: /Users/dev_tomo/Desktop/tt_prj_restart/backtest_j1_2026.csv
- report_json: /Users/dev_tomo/Desktop/tt_prj_restart/data/reports/report_j1_2026.json
