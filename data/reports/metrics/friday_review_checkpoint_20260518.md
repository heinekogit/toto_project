# Friday Review Checkpoint 2026-05-18

## Current Baseline

- J1 backtest snapshot:
  - `data/output_snapshots/j1_2026/20260518_231726_backtest_candidate_backtest_j1_2026.csv`
  - `82 / 170 = 48.24%`
- J2 backtest snapshot:
  - `data/output_snapshots/j2_2026/20260518_230820_backtest_candidate_backtest_j2_2026.csv`
  - `51 / 84 = 60.71%`

## Latest J1 Gains

- `neutral × split_side`
  - `D -> A` on 3 matches
  - all 3 became hits
- `signal_conflict × basis_balanced`
  - narrow `home restore` for `away_relegation` cases
  - `D -> H` on 2 matches
  - both became hits

## J1 Remaining Risk

Main unresolved branch:

- `signal_conflict × basis_balanced`
  - `9 matches`
  - `3 hits`
  - `33.33%`

Current branch detail:

- still wrong as `D`
  - `千葉-東京V`
  - `清水-長崎`
  - `柏-川崎F`
  - `水戸-東京V`
- wrong as `A`
  - `清水-京都`
  - `京都-岡山`

Interpretation:

- this branch is no longer a single-direction problem
- `H / A / D` all exist
- broad new restore rules are now high-risk

Recommendation:

- do not widen J1 `signal_conflict × basis_balanced` further before Friday
- keep current logic as review baseline
- inspect this branch in the Friday report as a residual weak spot, not as an immediate patch target

## J1 Weak Branch Order

From current snapshot:

1. `close_match × draw_compressed`
   - `1 match`, `0 hits`
2. `signal_conflict × split_side`
   - `1 match`, `0 hits`
3. `away_strong × split_side`
   - `2 matches`, `0 hits`
4. `home_strong × draw_compressed`
   - `13 matches`, `3 hits`, `23.08%`
5. `signal_conflict × basis_balanced`
   - `9 matches`, `3 hits`, `33.33%`

Practical note:

- only `signal_conflict × basis_balanced` has enough volume to matter immediately
- the smaller zero-hit branches should be reported, but not aggressively tuned yet

## J2 Current Read

- J2 is now in a much safer state than J1
- current weakest meaningful branch:
  - `away_strong × draw_compressed`
  - `11 matches`, `5 hits`, `45.45%`
- `neutral × draw_compressed` is already stabilized:
  - `14 matches`, `10 hits`, `71.43%`

Recommendation:

- freeze J2 logic for now
- use current J2 as the Friday reference version

## Friday Review Focus

1. Use the two snapshots above as the official baseline.
2. Review `decision_reason` concentration:
   - J1 still depends heavily on `BASE_FUSION`, `NARROW_DRAW_OVERRIDE`, `J1_AWAY_RESTORE_OVERRIDE`
   - J2 is materially cleaner than before
3. Treat J1 `signal_conflict × basis_balanced` as the main unresolved branch.
4. Avoid further broad rule additions unless the next evidence is match-level and one-directional.
