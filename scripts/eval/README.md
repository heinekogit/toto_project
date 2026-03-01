# Evaluate Round Tools

## 目的
- 購入時点スナップショット保存
- 結果確定後の実結果抽出
- buyplan 候補の採点（命中数/13）
- 採点済み HTML 出力

## 一括実行
```bash
SEASON_YEAR=2026 ROUND=round02 ./scripts/run_evaluate_round.sh 2>&1 | tee logs/run_evaluate_round_round02.log
```

## 個別実行
```bash
scripts/.venv/bin/python scripts/eval/00_snapshot_purchase.py --round round02
scripts/.venv/bin/python scripts/eval/01_export_actual_results.py --round round02 --season 2026
scripts/.venv/bin/python scripts/eval/02_score_buyplan.py --round round02
scripts/.venv/bin/python scripts/eval/03_build_scored_html.py --round round02
```

## 生成物
- `data/eval/rounds/round02/snapshot/predictions.csv`
- `data/eval/rounds/round02/snapshot/buyplan.csv`
- `data/eval/rounds/round02/actual_results.csv`
- `data/eval/rounds/round02/evaluation.csv`
- `data/eval/rounds/round02/buyplan_scored.html`
- `data/eval/candidate_history.csv`

## スクリプト機能とCSV入出力一覧

### 0) `scripts/run_evaluate_round.sh`（一括実行）
- 機能:
  - 評価パイプライン（snapshot -> actual抽出 -> 採点 -> scored HTML）を順番に実行する。
- 読み取りCSV:
  - 直接はなし（下記4本のPythonスクリプトに引き渡し）。
- 書き出しCSV:
  - 直接はなし（下記4本のPythonスクリプトで生成）。
- 代表的な出力先:
  - `data/eval/rounds/{round}/...`
  - `data/eval/candidate_history.csv`

### 1) `scripts/eval/00_snapshot_purchase.py`
- 機能:
  - 購入時点の参照データを `data/eval/rounds/{round}/snapshot` に退避する。
  - 出力先が既に存在する場合は安全停止する。
- 読み取りCSV:
  - 必須: `data/purchase_reference/predictions.csv`
  - 必須: `data/purchase_reference/buyplan.csv`
- 書き出しCSV:
  - `data/eval/rounds/{round}/snapshot/predictions.csv`
  - `data/eval/rounds/{round}/snapshot/buyplan.csv`

### 2) `scripts/eval/01_export_actual_results.py`
- 機能:
  - snapshotの `predictions.csv` を基準に、対象13試合の実結果を突合して `actual_results.csv` を作る。
  - 内部で `scripts/01_update_match_results.py` を呼び、J1/J2結果CSVを更新してから突合する。
- 読み取りCSV:
  - 必須: `data/eval/rounds/{round}/snapshot/predictions.csv`
  - 参照: `data/{league}_{season}_latest_results.csv`
  - 参照(フォールバック): `data/{league}_{season}_upcoming.csv`
- 書き出しCSV:
  - `data/eval/rounds/{round}/actual_results.csv`
- 間接的に更新されるCSV（内部実行スクリプト由来）:
  - `data/j1_{season}_latest_results.csv`
  - `data/j2_{season}_latest_results.csv`

### 3) `scripts/eval/02_score_buyplan.py`
- 機能:
  - `buyplan.csv` の候補列（ticket/候補01-10）を実結果と照合し、候補ごとの命中数を採点する。
  - ラウンド別の履歴 `candidate_history.csv` を追記/更新する。
- 読み取りCSV:
  - 必須: `data/eval/rounds/{round}/snapshot/buyplan.csv`
  - 必須: `data/eval/rounds/{round}/actual_results.csv`
  - 任意（存在時）: `data/eval/candidate_history.csv`
- 書き出しCSV:
  - `data/eval/rounds/{round}/evaluation.csv`
  - `data/eval/candidate_history.csv`

### 4) `scripts/eval/03_build_scored_html.py`
- 機能:
  - 採点済みの買い目一覧HTMLを生成する（CSVは生成しない）。
- 読み取りCSV:
  - 必須: `data/eval/rounds/{round}/snapshot/buyplan.csv`
  - 必須: `data/eval/rounds/{round}/actual_results.csv`
  - 任意（存在時）: `data/eval/rounds/{round}/evaluation.csv`
- 書き出しCSV:
  - なし（`data/eval/rounds/{round}/buyplan_scored.html` を出力）
