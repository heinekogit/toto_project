# 数学的最適10口の独立シャドー保存

試合前に、Next Modelの保存済み `next_predictions.csv` と同じ生成時の `buyplan_engine_manifest.json` を指定する。試合番号13件、H/D/A和、保存ハッシュ、予測生成日時とシャドー生成時刻が全キックオフ前であることを検査する。試合開始後の新規生成は拒否する。実結果は生成に渡さない。

```sh
./scripts/.venv/bin/python scripts/next_model/shadow_exact10.py generate \
  --in data/purchase_reference/next_predictions.csv \
  --manifest data/purchase_reference/manifest.json \
  --round totoXXXX
```

`--manifest` は `next_predictions.csv` の SHA-256 を `output_sha256` に持つ**予測・配券生成時のmanifest**を指定する。購入スナップショットの場合は `snapshot/buyplan_engine_manifest.json`。ファイル名だけでなくハッシュを照合する。生成物は `data/forecast_registry/exact10_shadow/snapshots/<開催回>/<予測SHA>/` に、`tickets.csv`、`probabilities.csv`、`manifest.json` として保存。予測SHAで一意にし、再実行時は既存ファイルのハッシュを検証して再利用し、変更・上書きしない。券には研究用と明記され、本番の購入処理には接続しない。

結果確定後は、`match_no,home_score,away_score` の13行CSVを用意する。チーム名列を含める場合は保存予測と一致を検査する。

```sh
./scripts/.venv/bin/python scripts/next_model/shadow_exact10.py score \
  --snapshot data/forecast_registry/exact10_shadow/snapshots/totoXXXX/<予測SHA> \
  --results <13試合の確定結果CSV>
```

照合は別の `scores/<開催回>/<予測SHA>/<結果SHA>.json` に保存し、予測と券は変更しない。結果に訂正があれば新しい結果SHAで別の照合を保存する。未使用の新開催で運用検証した後に集計する。自動購入・正式版切替は行わない。
