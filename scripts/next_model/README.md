# Next score-model workspace

## New BuyPlan research direction

The base research policy is [BuyPlan v4.0](../../BUYPLAN_V4_0.md). The latest generated research portfolio is [BuyPlan v4.2, a provisional comparison baseline](../../BUYPLAN_V4_2.md), not an adopted or purchase-ready version. It keeps sampled D placements and overlays ten H/A patterns; later ideas should be compared with that frozen result. Its objective remains at least one exact 13-match ticket among ten distinct tickets. The old v3.0 four-double / top-four-D proposal is abandoned and retained only as historical evidence. Phase 4-A and Phase 4-B are research stages, not BuyPlan version numbers. The currently running BuyPlan program is labelled v2.4; its fixed allocation is not a design premise for v4.0/v4.2. Historical sections below describe the versions and experiments that existed at their own dates; they are not the current research specification.

## Shared prematch conditions

`forecast_pipeline.py` produces the single `next_prob_home/draw/away` forecast
before any purchase strategy runs. Official v2.3 and shadows A/B/C consume the
same result; a future variable-ticket generator should consume the same frame.
Ticket generators must not apply weather, fatigue or absence corrections again.

The configurable defaults are in
`data/models/uncertainty_calibration_v1.json`: `absence_goals_per_impact`,
`fatigue_weight`, `weather_weight`, `max_mix`, and the rain/heavy-rain/wind
signal coefficients. The absence coefficient moves
goal expectations according to attacking/defensive absences. Fatigue (including
confirmed extra fixtures and travel) and weather currently mix H/D/A toward
uniform, increasing uncertainty without favoring a particular result. These
are different operations; a positive draw-specific weather or fatigue effect
would need a separately fitted model, not a change to the ticket generator.
The output records pre-absence and pre-uncertainty probabilities, signal
strengths, applied mix and the coefficients. Missing weather contributes
no weather signal; missing absence information contributes no directional
adjustment. Keep frozen purchase snapshots intact when comparing coefficients;
generate candidates in a separate output directory.

## Production BuyPlan v2.3

The next prediction model builds match goal expectations from scoring and xG
rates, adjusts them with Football LAB xG, then applies a bounded absence
correction: missing attackers reduce their team's goal expectation, and missing
defenders raise the opponent's. The score distribution produces
H/D/A probabilities. A bounded fatigue/weather uncertainty correction
then produces `next_prob_home/draw/away`, which BuyPlan uses for ticket
allocations and candidate probabilities. The unadjusted probabilities are
saved as `next_prob_pre_uncertainty_*`; the values before absence correction
are saved as `next_prob_pre_absence_*`. Unknown absence inputs make no
directional correction and are flagged. The fatigue/weather coefficients were fitted
for the older forecast and need further tuning for this score model. The
frozen-round simulation in `data/reports/next_uncertainty_simulation/comparison.json`
found worse heldout log loss and lower maximum ticket hits with these coefficients.
The absence comparison is in `data/reports/next_absence_simulation/comparison.json`;
its current coefficient also worsened heldout metrics and needs tuning.
The replayable input history for tuning these existing corrections is
`data/eval/next_model/correction_history_v1/matches.csv`. Build it with
`./scripts/.venv/bin/python scripts/next_model/build_correction_history.py`.
It joins frozen prematch prediction inputs to results for 88 matches: 49 in
the earlier fit period and 39 in the later validation period. The table keeps
fatigue, weather and absence inputs, missing-data flags, pre/post correction
probabilities, and outcomes. Its manifest records source hashes and the
snapshot times. Building this history does not fit or change production
coefficients. Some component fatigue columns were absent in older snapshots;
the total fatigue score used by the current correction is present.
The first time-ordered coefficient fit is reproducible with
`./scripts/.venv/bin/python scripts/next_model/fit_correction_history.py`;
its report is `data/reports/next_correction_fit_v1/comparison.json`. The
fit-only candidate worsened log loss on the later 39 matches, so it was not
applied to production.
Scoreline probabilities also
feed the D classifier. Frozen purchase snapshots retain the probability source
recorded when they were created.

The production CLI `run_next_buyplan.py` now uses `random_remaining_v24`.
Each match receives ten H/D/A slots by largest remainder independently of
the other matches. An input-derived seed drives remaining-count proportional
draws into tickets 01--10. Ticket numbers are draw order. No candidate 00,
D placement score, cost optimization, or final sorting is used. Exact duplicate
tickets are reported without repair. The seed and prediction fingerprint are
stored in the generator manifest. Actual results are never supplied to the
generator. The full specification is in `ドキュメント類/BuyPlan_v2.4_正式仕様.txt`.

The official output is labelled v2.4. Three non-purchase, ten-ticket shadows
are generated from the same predictions:

- Shadow A: previous official v2.2, `shadow_buyplan.csv` / `.html`, ticket11..20.
- Shadow B: Anchor graded without D-score adjustment, `shadow_buyplan_b.csv`
  / `.html`, ticket21..30.
- Shadow C: free-coverage observation, `shadow_buyplan_c.csv` / `.html`,
  ticket31..40; its per-match H/D/A allocation counts may differ.

The snapshot and growth report save and score both separately. Overlap between
the official portfolio and shadows is allowed. Existing frozen purchase
snapshots and their scored histories must not be rewritten.
The v2.2 generator `probability_allocation_ge10_v1` remains available for
historical comparison. The HTML separately shows each ticket's model 13-hit
probability calculated from the unchanged original forecasts.

This directory originally provided an isolated diagnostic workspace. It now
also contains the official next BuyPlan pipeline described above. Its original
purpose was to freeze eight-round D-validation evidence and build an auditable
Lab-free score-model baseline; the following sections record that diagnostic work.

## Commands

```bash
./scripts/.venv/bin/python scripts/next_model/build_validation_dataset.py
./scripts/.venv/bin/python scripts/next_model/evaluate_next_model.py
./scripts/.venv/bin/python scripts/next_model/audit_stats_units.py
./scripts/.venv/bin/python scripts/next_model/evaluate_next_model.py --model v1 \
  --out data/eval/next_model/validation_v1/base_lambda_v1
./scripts/.venv/bin/python scripts/next_model/evaluate_next_model.py --model v2 \
  --out data/eval/next_model/validation_v1/lab_matchup_v0
./scripts/.venv/bin/python scripts/next_model/build_validation_v2.py
./scripts/.venv/bin/python scripts/next_model/audit_validation_v2.py
./scripts/.venv/bin/python scripts/next_model/evaluate_next_model.py \
  --input data/eval/next_model/validation_v2/matches.csv --model v3 \
  --out data/eval/next_model/validation_v2/lab_xg_shot_v1
./scripts/.venv/bin/python -m unittest scripts/test_next_model.py
```

The validation builder refuses to overwrite an existing output directory. A
new schema or source population must use a new versioned output path.

## Current contract

`validation_v1/matches.csv` contains:

- source identity and actual results;
- current base/Lab/final probabilities;
- the legacy D classifications and their evidence;
- production state inputs reserved for base lambda work;
- Football Lab source fields reserved for the later matchup adjustment.

The manifest records SHA-256 hashes for every prediction snapshot and actual
result source. Matches absent from a frozen prediction snapshot are excluded
explicitly rather than reconstructed using post-match data.

## Baseline v0

`base_lambda_goals_xga_geomean_v0` is diagnostic only. It uses the geometric
mean of a team's per-game scoring rate and its opponent's per-game conceding
rate. It does not use Lab fields or historical xG fields. The latter are
present in the frozen table, but their units and aggregation windows must be
verified before they can be used as lambda inputs.

This baseline is not a candidate for production. It establishes the opposite
edge of the current over-D behavior and verifies the score-distribution and
evaluation plumbing before matchup adjustments and the four-way D classifier
are implemented.

## D-filter boundary

`draw_filter.py` consumes the raw score distribution and H/D/A probabilities.
It only emits `D-core`, `D-cover`, `D-uncertain`, `D-reject`, or
`not_applicable` metadata. It must not alter lambda values, probabilities, or
the raw predicted symbol. Version v0 contains explicit observation hypotheses;
it is not a fitted or production-ready classifier.

## Baseline v1

The official J.League `ゴール期待値` and `被ゴール期待値` fields are cumulative,
while `1試合平均被ゴール期待値` is per-game. V1 reconstructs games played by
rounding cumulative xGA / per-game xGA inside the same frozen row, then converts
cumulative attacking xG to xG per game. Attack and defence each use an equal,
explicit blend of goals and xG before opponent matchup. This is still a
Lab-free diagnostic baseline and is not tuned to produce any target draw count.

## Lab matchup v0

The first correction layer uses only Football Lab's complete per-match
`expected_for_xg` and `expected_against_xg` fields. It forms an independent
matchup estimate with a geometric mean, then applies 25% of the log-space
difference from baseline v1. The weight is an explicit a-priori diagnostic
setting, not fitted against results or a target H/D/A distribution.

Every prediction row retains the base lambda, raw Lab matchup lambda, log
delta, applied multiplier, and final lambda. Legacy Lab H/D/A probabilities,
draw labels, AGI/KAGI (partially missing), and shot fields (stored as rank text
by the historical snapshot parser) are not used. The parser has since been
corrected, but frozen validation v1 is intentionally unchanged. The D filter remains downstream metadata and
cannot change these values or the raw prediction.

## Validation v2 and shot candidate

Validation v2 reparses the saved Football LAB HTML at each historical
prediction point with the corrected value/rank alignment. It contains 101
rows and 40 numeric chance value/rank columns with no missing cells; v1 remains
unchanged as an audit parent.

The `lab_xg_shot_matchup_logblend_v1` diagnostic candidate combines the xG
matchup estimate with a shot-volume/conversion goal-rate estimate. It is kept
as a rejected experiment: on the 49 prematch-proven rows it preserved accuracy
but worsened Brier and log loss versus the xG-only Lab correction. It must not
replace `lab_xg_matchup_logblend_v0` without a new independent hypothesis.

## D filter v1

`draw_filter_score_structure_v1` is applied after the xG-only Lab correction.
It uses only the new probability and score-distribution fields and never
changes lambda, H/D/A probabilities, or the raw prediction. Legacy purchase
types and Lab basis labels are reported as diagnostic strata but are not
classifier inputs. Its initial thresholds are frozen design hypotheses, not
fitted to the validation outcomes. Initial prematch-proven results show useful
draw rejection but insufficient D-core recall, so v1 is not production-ready.

V2 gives historical structure only limited authority: `draw_compressed` and
`flat_draw_trap` can reject a D candidate, while `split_side` can promote an
uncertain case to D-cover but never to D-core. Reports separate the
counterfactual effect of making D-core the primary pick from the deliberately
harsher diagnostic that also treats D-cover as primary; production BuyPlan is
expected to use D-cover as portfolio coverage instead.

## Next BuyPlan v0

`build_next_buyplan.py` is the first implementation candidate for the future
official BuyPlan. Ticket 01 is the exact maximum joint-probability combination.
Tickets 02-10 are the highest-probability candidates that keep a minimum
pairwise Hamming distance of two matches. There are no fixed H/D/A ratios,
target draw count, world-name rules, or symbol forcing. D-filter classes are
retained for audit only in v0.

The historical comparison deliberately reports both ticket quality and
portfolio breadth. Initial prematch-proven results improve ticket01, best, and
average hits versus the legacy BuyPlan, but cover fewer distinct match outcomes.
This is therefore an implementation baseline, not yet the accepted official
generator.

Next BuyPlan v1 retains the exact same ticket01 and admits only combinations
whose joint probability is at least 50% of ticket01. Within that frontier it
greedily adds probability-weighted outcome coverage, still enforcing the
two-match pairwise distance. This is a probability-loss budget, not a target
symbol allocation; draw cells remain an unconstrained output.

## Production-shaped acceptance interface

`run_next_buyplan.py` accepts the ordinary prediction CSV plus an explicit
toto round/order, fails closed unless all 13 pairs match, removes the older
`prob_final_*`/`prob_main_*` forecasts and labels at the BuyPlan boundary,
recomputes `next_prob_home/draw/away` from the score model, and writes `next_predictions.csv`, `buyplan.csv`,
`buyplan.html`, and a manifest. `scripts/run_buyplan_engine.py` is the explicit
engine switch. Its `next` route now produces v2.4; the legacy route remains
available only by explicitly passing `--engine legacy`.
Example:

```bash
./scripts/.venv/bin/python scripts/run_buyplan_engine.py --engine next \
  --in data/purchase_reference/predictions.csv --round toto1654 \
  --logical-season 2027 --outdir data/eval/next_model/integration/toto1654
```

The official run also writes shadow A/B/C CSV and HTML files with non-purchase
tickets 11--40. They remain separate from official tickets 01--10.
It also writes `shadow_condition_s_buyplan.html` and `.csv`: a research display
of the unchanged official v2.4 tickets 01--10 with each match's frozen condition-S
flag. This is not a v4.0 ticket generator. The observation archive copies both
files when present.

## Live preflight (2026-10-03)

The live runner checks kickoff dates against the selected order and the next
seven days, stats/LAB snapshot age (default seven days), and finite positive
inputs before replacing any output. `--check-only` validates without writing;
`--max-age-days` sets an explicit recorded freshness limit. Older predictions
must be regenerated to carry `flab_snapshot_date` provenance. Historical
research should use the evaluation functions, not the live CLI.

Purchase snapshots preserve the generator manifest as
`buyplan_engine_manifest.json`; their own file hashes are in `manifest.json`.
New generator manifests hash the generated output files. Snapshot creation
rejects changed inputs/outputs and mismatched round/season metadata.

## Anchor相対配券の比較（2026-10-17、未採用）

`anchor_arrangement.py` は枚数決定後の配置専用関数です。各試合の元確率から
`-log(p / max(p))` を求め、同一試合内の2券の目を交換します。
各試合のH/D/A枚数、総券数、最低2試合差を維持します。Anchor券の固定はしません。

- `balanced`: 券別の合計相対コストを均等化。
- `graded`: 平均コストの0.5～1.5倍の等間隔目標へ配置。幅は未検証の固定仮説。
- 再現可能な貪欲探索（seed=20261017、20,000提案）。大域最適保証なし。
- ゼロ確率は無限コストになるため明示的に拒否します。

既存BuyPlanへの接続口（返すだけで書き込みません）:

```python
from anchor_arrangement import rearrange_buyplan
comparison_plan, audit = rearrange_buyplan(plan, mode="balanced")
```

比較の再実行:

```sh
./scripts/.venv/bin/python scripts/next_model/compare_anchor_arrangements.py
./scripts/.venv/bin/python -m unittest discover -s scripts -p test_anchor_arrangement.py
```

出力: `data/reports/anchor_arrangement_v1/index.html`。現行v2.2の再生成と2案を
同じ枚数で比較。現在保存されたBuyPlanも別枠で比較します。現在券に実結果を
結び付けたり、開催回を推測したりはしません。10試合の不完全開催は除外します。
モデル評価は独立を仮定した全3^13結果の厳密列挙、実測は開催別の最高的中と
8/9/10的中以上到達数。平均的中は枚数固定で不変です。

実結果は生成・探索に使いません。過去確率は再計算資料なので、元snapshotに
事前保存保証があっても新方式の前向き検証にはなりません。すべて探索的参考評価とし、
方式を選択・固定した後の将来開催で検証してください。本番のデフォルトは変更していません。

## 正式版v2.2の候補表示順

正式版 `probability_allocation_ge10_v1` は、10券の最適化後に各券の
13的中モデル確率（各試合の選択確率の積）が高い順へ候補01～10を付けます。
同率時は最適化後の元の並びを維持します。組み合わせ、試合ごとの配置枚数、
10券全体のモデル確率は変わりません。生成記録には旧インデックスとの対応を
`ticket_original_indices` として保存し、HTMLに各候補の確率とD本数を表示します。
既に保存した開催回のsnapshotは番号を変えません。

## 追加チームスタイルの前向き観測運用

LAB取得・正式予測生成・結果更新の既存入口から、指標と予測の固定保存、結果結合、過去分での学習、後続試合での評価を実行します。2026-11-03に正式予測への統合を有効化しました。未学習時は補正ゼロ、学習済み補正は次の正式予測に反映します。
仕様・保存先・再実行方法は [STYLE_SHADOW.md](STYLE_SHADOW.md) を参照。
運用状態は `data/style_shadow/reports/index.html`（予想・検証ダッシュボードからリンク）で確認できます。
