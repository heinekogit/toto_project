# 試合前予測レジストリ

`forecasts.jsonl` と `results.jsonl` は独立した追記専用イベントログ。既存の購入用CSVや予測原本は変更しない。

## 予測イベント

1行1試合・1スナップショット。`event_id` は予測出力SHA-256と試合番号から決定的に生成し、再登録を除外する。`fixture_id` はリーグ、試合日、正規化したホーム・アウェイ名から作る。`round_id` と `match_no` も残す。同じ試合の別時刻予測は別イベントだが、診断の標準選択は検証済み試合前予測のうち最も早いもの1件。

保存値：予測作成時刻、入力と出力のパス・ハッシュ、manifestハッシュ、コードハッシュが利用可能な場合のコード識別、モデル表記、補正監査欄、λ、行列由来raw H/D/A、mix、final H/D/A。`source_input_available_at` は現在のmanifestに情報がないため `null`。過去の `code_version` も未記録なら `null` とし、推測の版番号で埋めない。今後の通常実行では予測生成直後に主要コードファイルのハッシュを登録する。

試合前判定：manifestの予測ファイルハッシュを照合し、manifest作成時刻が全試合のキックオフより前であることを確認。保存得点確率、raw H/D/A、final H/D/Aの整合性も検査する。これらは**生成時刻の証拠**であり、すべての入力ソースの公開・利用可能時刻まで保証するものではない。今後は入力ごとのas-ofをmanifestへ保存する必要がある。

## 結果イベント

`results.jsonl` に別途追記。得点・H/D/A・出所ファイルハッシュ・登録日時を保持する。古い結果を訂正する場合は新イベントに `correction_of` を記録し、古い行は残す。過去の結果ファイルには実際の結果公表時刻がないため、`result_known_at` は `null`。登録日時を公表日時として扱わない。

## 操作

通常の `run_next_buyplan.py` が予測・配券ファイルとmanifestを完成させた後、独立した登録関数を呼ぶ。登録失敗は `[FORECAST_REGISTRY][ERROR]` と表示し、既に生成された予測・配券を変更しない。

```sh
./scripts/.venv/bin/python scripts/next_model/forecast_registry.py register-forecast \
  --forecast data/purchase_reference/next_predictions.csv \
  --manifest data/purchase_reference/manifest.json
./scripts/.venv/bin/python scripts/next_model/forecast_registry.py status
```

通常の `01_update_match_results.py` は結果CSV保存後、登録済み予測との結果照合を行う。失敗は `[FORECAST_REGISTRY][ERROR]` と表示する。再試行は次のコマンドを使用する。

```sh
./scripts/.venv/bin/python scripts/next_model/forecast_registry.py sync-league-results \
  --results data/j1_2026_latest_results.csv
```

toto結果CSVからの登録・訂正は `register-results --results ... --round totoXXXX`。後日再計算したC分類のλは、このレジストリのA原本へ取り込まない。診断は `diagnose_score_structure_phase4a.py` を単独実行し、レジストリのハッシュ不変を確認する。
