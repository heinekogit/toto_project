# 追加チームスタイルの比較用モデル

`style_shadow.py` は補正係数の学習・評価を担うモデルです。2026-11-03から、`style_official.py` を介して正式予測にも統合しています。入力は現行 `forecast_pipeline.build_next_probabilities` で計算済みの予測です。BuyPlanの配券は扱いません。

## 計算仕様

基準の最終λ（LAB・欠場等の補正後）を固定し、各チームの得点をPoisson回帰で学習します。

`log(λ_shadow) = log(λ_baseline) + Σ β_j × z_j`

- 追加7項目の42指標を自チーム・相手チーム別に使います。
- 既存の攻撃指数と相手守備指数の交互作用を4つ追加します：ロングカウンター×ハイブロック、ショートカウンター×ハイプレッシング、敵陣保持×ローブロック、自陣保持×カウンタープレス。
- 指数の50を中心、10を単位とした積を作ります。係数の符号も大きさも手で設定しません。各組合せは検証仮説で、効果が実証された関係ではありません。
- λの基準値は再学習せず、切片も追加しません。基準の残差を説明する設計ですが、xGと指標の重複を完全に排除できるという保証ではありません。
- 標準化の平均・分散は学習期間だけから計算します。欠損に関係する項は寄与を0にし、元の観測値を平均やゼロで埋めません。
- 補正後のλは既存モデルの範囲0.15～4.5に収めます。補正が0なら基準λを厳密に保持します。
- 確率は同じPoisson計算から再算出し、基準予測の `next_uncertainty_calibration_mix` による疲労・天候の均等混合を再適用します。補正なしでは基準確率をそのまま返します。

## 学習と採否

実行下限は20試合・3日以上です。これは精度の裏付けとなる標本数ではありません。日付順に先頭約70%を学習、残りを内部評価に使い、L2正則化0.1・1・10を比較します。内部評価の得点負対数尤度が基準より良い案だけを採用し、条件を満たさなければ補正を無効にします。選択後は学習締切以前の全対象で再学習します。

内部評価は係数選択用です。精度改善の主張には、その後の未使用試合で独立評価が必要です。

## 日付と保存条件

- 時刻未指定の日時は日本時間です。
- 指標の取得日しか分からない場合、その翌日0時から使用可能とします。公開ページの更新日から過去へ遡らせません。
- 指標はキックオフではなく、予測を保存した時刻までに利用可能だったものに限定します。
- 予測時刻は試合開始より前、実結果の利用可能時刻は開始より後である必要があります。
- 係数学習には `result_available_at < cutoff` の実結果だけを使います。モデルの締切より前の予測にそのモデルを使うことも拒否します。
- 基準予測と追加指標の実際の保存履歴は利用側で正しく指定する必要があります。過去データを今再計算したものに、過去の予測時刻を与えてはいけません。

## 実行例

予測CSVには `match_id,datetime,forecast_created_at,home_team,away_team,final_lambda_home,final_lambda_away,next_prob_home,next_prob_draw,next_prob_away,next_uncertainty_calibration_mix` が必要です。既存攻撃指数列がない場合、対応する交互作用は欠損になります。

```bash
./scripts/.venv/bin/python scripts/next_model/style_shadow.py predict \
  --predictions /path/to/prematch_next_predictions.csv \
  --styles data/reports/style_value_investigation/validation_inputs.csv \
  --out-dir /path/to/shadow_snapshot
```

`--model /path/to/model.json` を指定すると学習済み補正を使用します。未指定の場合は未学習として基準を保持します。保存時刻の列がない場合だけ、実際の保存時刻を `--forecast-created-at` で指定してください。

出力：

- `comparison.csv`：基準・補正後のλ、確率、差分、モデル状態。
- `examples.csv`：試合片側ごとの入力特徴、利用可能時刻、各項の寄与、補正前後のlog差。

学習用の `examples.csv` は複数回の試合前保存分を結合します。同一試合・同一sideの重複は拒否します。結果CSVには `match_id,home_score,away_score,result_available_at` が必要です。

```bash
./scripts/.venv/bin/python scripts/next_model/style_shadow.py train \
  --examples /path/to/archived_examples.csv \
  --results /path/to/completed_results_with_availability.csv \
  --cutoff '2026-12-01 12:00:00+09:00' \
  --model-out /path/to/model.json

./scripts/.venv/bin/python scripts/next_model/style_shadow.py evaluate \
  --comparison /path/to/shadow_snapshot/comparison.csv \
  --results /path/to/results.csv \
  --out /path/to/scores.json
```

評価は同一match_idで対になった完了試合について、得点MAE・H/D/A Log Loss・Brierを両方式で算出します。IDが予定時刻と実開始時刻で異なる場合は、試合日・正規化チーム名を確認して明示的に対応付けます。自動で曖昧な結合はしません。

## 今回の確認

13件の自動テスト：時刻制約、欠損、重複、未来データ排除、既知の合成信号の学習、効果なしの基準選択、確率の再校正、同一試合の評価。

`data/reports/style_shadow_implementation/control` は、保存予測後に開催された1668回の4試合を使った動作確認です。追加指標は当時未取得のため全て除外され、未学習モデルの出力は元予測と一致しました。精度改善の検証ではありません。実データの補正係数はまだ学習していません。

## 通常運用への接続（2026-11-03）

通常の実行入口に接続済みです。新しい定期実行サービスは追加していません。従来のコマンドを実行した際に以下が動きます。

| 通常の操作 | 追加された処理 |
|---|---|
| `fetch_football_lab_html.py --season-year 2026` | J1追加7ページに加え、20チームの上部一覧を取得。候補が残る項目のみ追加照合。採用値・欠損値と原本を固定保存 |
| `run_next_buyplan.py`（既存の正式予測入口） | 配券前に、補正前の予測と指標・使用モデルを固定保存。統合設定が有効なら補正後のλ・確率を正式出力へ反映。未学習時は補正ゼロ |
| `01_update_match_results.py` | 結果CSV保存後に、保存済み予測へ実結果を結合・評価 |
| `run_evaluate_round.sh` | 結果結合と評価を再更新。手動再実行しても同じ結果を重複蓄積しない |

保存先は `data/style_shadow/` です。

- `inputs/<取得時刻>/`：採用指標・原本HTML・ハッシュ。観測時刻が分かる新しい保存分は、その時刻以降の予測に使えます。日付しかない旧資料は従来どおり翌日から利用可能です。
- `forecasts/<保存時刻>/`：正式予測、追加指標、比較用予測、入力特徴と寄与、使用したモデルを保存。通常の予測ファイルを上書きしても、この記録は変更しません。
- `result_observations.csv`：実結果を初めて確認した時刻を記録。結果が訂正された場合は新しい観測を追加し、過去の観測を保持します。
- `models/`：学習時点ごとの係数と採用判断。試合前に保存したモデルは、その試合の結果を知った後に差し替えません。
- `reports/index.html`：保存数・結果結合数・次回モデルの状態。
- `reports/examples.csv` と `results.csv`：最初に保存した予測の特徴と、結合できる最新結果。`joined_training_rows.csv` は結合内容の確認用。
- `reports/paired_results.csv`：試合前に固定した基準・比較予測と実結果。

同一試合は「リーグ・試合日・正規化したホーム／アウェイ名」で照合し、予定時刻と実開始時刻の数分差を吸収します。複数の予測を保存しても、評価には**最初に保存された試合前予測**を使います。試合日が変わる延期試合は自動で結合しません。

モデルの学習対象は、その時点までに観測された実結果だけです。新しい結果を取り込んだ後、次回予測時にも学習し直します。学習に使った試合へ新しいモデルを遡及適用せず、学習後に保存した後続試合の予測を評価します。係数未学習、指標未取得、内部検証で補正なしが選択された場合は基準予測を保持します。

全観測のスコアと、学習済み比較モデルを試合前に使えた試合だけのスコアは分けて記録します。未学習時の一致結果を「精度改善の実証」と数えません。

状態更新を単独で再実行する場合：

```bash
./scripts/.venv/bin/python scripts/next_model/style_observation.py refresh
```

正式統合では、予測の固定保存や設定検証に失敗した場合は配券出力前にエラーで停止します。黙って異なるモデルへ切り替えません。正常時は正式出力先の `style_observation.json` と `manifest.json` に適用状況を記録します。結果結合の失敗も結果更新ログに表示され、上記の単独コマンドで再実行できます。

取得した照合用チームページは `external_contents/<日付>/j1/style_observation_sources/` に分離し、既存の正式LAB集計処理が読み込むページを変えないようにしています。

## 正式版への統合開始（2026-11-03）

設定：`data/models/style_residual_deployment_v1.json` の `enabled: true`。次回の通常予測生成から有効です。既に保存した予測や購入券は再生成しません。

1. 現行モデルから補正前のλ・確率を計算。
2. その時点までの結果だけで係数を学習・更新し、基準を上回らない場合は補正なしを選択。未学習も補正ゼロ。
3. 同じ入力で追加指標を評価し、正式な `final_lambda_*` と `next_prob_*` へ反映してから配券へ渡す。
4. `pre_style_lambda_*`、`pre_style_prob_*`、差分、使用モデル、設定、原本を保存。正式版の誤差と補正前の誤差を同じ試合で追跡。

ロールバック：設定の `enabled` を `false` にすると、次の予測生成は補正前の値を正式採用します。観測・学習・比較用の記録は継続します。過去の保存予測を変更しません。

学習済みモデルがない現状では、統合設定が有効でも予測確率・配券用seedは元のままです。正式な購入券の生成方式はBuyPlan v2.4を維持します。
