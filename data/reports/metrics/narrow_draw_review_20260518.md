# Narrow Draw Review 2026-05-18

## Purpose

- Review where `NARROW_DRAW_OVERRIDE` is still concentrated.
- Separate:
  - branches that are just noisy but still acceptable,
  - branches that are structurally weak,
  - branches where removing `NARROW_DRAW_OVERRIDE` alone would not fix the final label because `predicted_result_main` is already `D`.

## Current Baseline

- J1 snapshot:
  - `data/output_snapshots/j1_2026/20260517_222000_backtest_candidate_backtest_j1_2026.csv`
- J2 snapshot:
  - `data/output_snapshots/j2_2026/20260518_003359_backtest_candidate_backtest_j2_2026.csv`

## J1 Narrow Draw

- total: `38`
- hits: `17`
- hit rate: `44.74%`

Largest branches:

1. `neutral × draw_compressed`
   - `21 matches`
   - `11 hits`
   - `52.38%`
   - Still large, but not obviously broken.

2. `signal_conflict × draw_compressed`
   - `7 matches`
   - `4 hits`
   - `57.14%`
   - Acceptable for now.

3. `signal_conflict × basis_balanced`
   - `5 matches`
   - `0 hits`
   - `0.00%`
   - Actuals: `H,H,H,A,H`
   - Important note:
     - In all five cases, `argmax_result=D` and `predicted_result_main=D`.
     - This means removing `NARROW_DRAW_OVERRIDE` alone would not fix these cases.
   - Interpretation:
     - This is a `main`-layer draw bias problem, not only an override problem.

4. `neutral × split_side`
   - `2 matches`
   - `0 hits`
   - `0.00%`
   - Actuals: both `A`
   - Same note:
     - `argmax_result=D`, `predicted_result_main=D`.
   - Interpretation:
     - Also a `main`-side issue.

J1 conclusion:

- The biggest remaining J1 draw problem is not `NARROW_DRAW_OVERRIDE` itself.
- The weak branches are already `D` before the override stage.
- Friday target should be:
  - review `signal_conflict × basis_balanced`,
  - review `neutral × split_side`,
  - trace them from `main` rather than from post-override code only.

## J2 Narrow Draw

- total: `40`
- hits: `12`
- hit rate: `30.00%`

Largest branches:

1. `neutral × draw_compressed`
   - `7 matches`
   - `1 hit`
   - `14.29%`
   - Actuals: `A,A,D,A,A,A,A`
   - Internal shape:
     - `predicted_result_main` is always `D`
     - `argmax_result` is mixed: `H,H,D,H,D,A,H`
   - Interpretation:
     - Very weak branch.
     - Simply removing narrow draw is not enough in most rows, because `main` is already `D`.

2. `away_strong × draw_compressed`
   - `7 matches`
   - `2 hits`
   - `28.57%`
   - Actuals lean `A`.
   - Mixed internal shape; not as clean as the branch above.

3. `away_strong × basis_balanced`
   - `6 matches`
   - `2 hits`
   - `33.33%`

4. `signal_conflict × basis_balanced`
   - `3 matches`
   - `0 hits`
   - `0.00%`
   - Actuals: all `A`
   - Internal shape:
     - `predicted_result_main` is `D` for all three
     - `argmax_result` is `A,A,H`
   - Interpretation:
     - Also a `main`-side draw issue, not just an override issue.

J2 conclusion:

- The worst current branch is `neutral × draw_compressed`.
- `signal_conflict × basis_balanced` is small but clearly weak.
- As with J1, the bad rows are mostly already `D` at `main`.
- Friday target should be:
  - inspect `main` draw gating for these two branches first,
  - avoid touching `away_strong` branches until the `main`-layer issue is better understood.

## Working Recommendation

Do not add new broad restore rules yet.

Preferred next order:

1. J2 `neutral × draw_compressed`
2. J2 `signal_conflict × basis_balanced`
3. J1 `signal_conflict × basis_balanced`
4. J1 `neutral × split_side`

Reason:

- These are the branches where draw is weak even before post-main overrides.
- Improving them at the `main` layer will be cleaner than stacking more local restore logic.
