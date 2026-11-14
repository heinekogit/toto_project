# 2026-11-14 疲労マージ修正の確認

- 原因：`team_fatigue_scores_j1_2026.csv` は `conditions_v3` の `partial` 状態と欠損スコアを区別して保存していたが、`11_prediction_01.py` のマージ対象列に `home_fatigue_status`・`away_fatigue_status` が含まれていなかった。`validate_prediction_fatigue` が正当な欠損を「ラベルなし」と判定した。
- 修正：両チームの状態列をマージする。スコアのゼロ補完や検証関数の緩和は行わない。
- J1：2026-11-14の入力で `11_prediction_01.py` が終了コード0。予測CSVは240行。ホーム `partial` 27行、アウェー `partial` 33行のスコア欠損を状態付きで保持。
- J2：同条件で終了コード0。予測CSVは230行。実行ログは `/tmp/j2_prediction_fatigue_fix_20261114.log`。
- `scripts/test_fatigue_components.py`：16件成功。欠損距離や旧形式の拒否も維持。
- 週次全体：この環境では `data.j-league.or.jp` と `www.jleague.jp` の疎通検査が失敗し、開始前に停止。したがって週次バッチ全体の成功は未確認。`logs/run_batch_weekly_report.html` はこの疎通エラーを示す。
