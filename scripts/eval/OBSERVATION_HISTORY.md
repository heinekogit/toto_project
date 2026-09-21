# 新シーズン観測履歴

正式な prediction / buyplan の判断を変えず、後日の再検証に必要な情報を保存する仕組み。

## 試合前

購入用 `predictions.csv` と `buyplan.csv` の生成後、toto開催回をIDとしてsnapshotを保存する。

```bash
./scripts/.venv/bin/python scripts/eval/00_snapshot_purchase.py \
  --round toto1644 \
  --srcdir data/purchase_reference \
  --logical-season 2027 \
  --prediction-season 2026 \
  --j1-source data/output_snapshots/j1_2026/20260723_230938_predictions_candidate_j1_2026_predictions_hfa_on.csv \
  --j2-source data/output_snapshots/j2_2026/20260723_230943_predictions_candidate_j2_2026_predictions_hfa_on.csv \
  --football-lab-snapshot 20260802
```

同じ開催回のsnapshotが既に存在する場合は上書きせず安全停止する。保存先の `manifest.json` には
入力ファイルのSHA-256と、論理シーズン・予測ファイルシーズン・元snapshotを記録する。

## 試合後

通常の採点を実行する。

```bash
SEASON_YEAR=2027 ROUND=toto1644 ./scripts/run_evaluate_round.sh
```

この処理の末尾で観測履歴も自動更新される。

## 累積出力

- `data/eval/observation_history/matches.csv`
  - 1試合1行
  - H/D/A確率、既存分類、実結果、10券の記号と的中、rescue/all-sameを保持
  - prediction・buyplan・actualの全列をJSON payloadとして保持
- `data/eval/observation_history/round_summaries.csv`
  - 1開催回1行
  - H/D/A実績、確率品質、portfolio cover、rescueなどを保持
- `data/eval/observation_history/provenance/{round}.json`
  - 累積時に使用した3ファイルのパスとSHA-256を保持
- `data/eval/observation_history/market_alignment_matches.csv`
  - ルート直下に保存したtoto PDFの投票率と実結果を1試合1行で保持
  - 市場本命、実結果支持率、強本命の失敗、市場サプライズ度を記録
- `data/eval/observation_history/market_alignment_rounds.csv`
  - 市場本命的中率、実結果平均支持率、市場対数損失などを1開催回1行で保持
- `data/eval/observation_history/provenance/{round}_market_alignment.json`
  - 使用したPDFと実結果CSVのパス・SHA-256を保持

同じ開催回を再実行した場合は `(round_id, match_no)` で置換し、二重追加しない。
観測履歴は正式ロジックの入力には使用しない。

## 市場整合度（開催後のみ）

`toto{回号}.pdf` がプロジェクト直下にある場合、通常の採点処理が投票率を自動抽出し、
確定した実結果と照合する。PDFがない回はこの工程だけをスキップする。

このデータは開催回が事前情報に沿ったかを後から比較するためのもので、prediction、
buyplan、モデル学習には渡さない。蓄積期間が終わるまでは固定ラベルを付けず、生の指標を保存する。
