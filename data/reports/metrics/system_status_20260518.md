# System Status 2026-05-18

## Scope

- Goal: stabilize the current `base + lab + main + override` pipeline and keep Friday review focused on current production-like behavior.
- Baseline snapshots:
  - J1 backtest: `data/output_snapshots/j1_2026/20260518_231306_backtest_candidate_backtest_j1_2026.csv`
  - J2 backtest: `data/output_snapshots/j2_2026/20260518_230820_backtest_candidate_backtest_j2_2026.csv`

## Current Accuracy

| League | Matches | Correct | Accuracy |
|---|---:|---:|---:|
| J1 2026 | 170 | 80 | 47.06% |
| J2 2026 | 84 | 51 | 60.71% |

Reference:
- `data/reports/report_j1_2026.json`
- `data/reports/report_j2_2026.json`

## Current Decision Shape

### J1

- `BASE_FUSION`: 83
- `NARROW_DRAW_OVERRIDE`: 40
- `J1_AWAY_RESTORE_OVERRIDE`: 36
- `J1_HOME_RESTORE_OVERRIDE`: 7
- `J1_SIGNAL_CONFLICT_HOME_RESTORE`: 3

Interpretation:
- J1 is now materially driven by `BASE_FUSION` plus the two restore families.
- `NARROW_DRAW_OVERRIDE` is still large enough to remain a Friday review target.

### J2

- `NARROW_DRAW_OVERRIDE`: 33
- `BASE_FUSION`: 24
- `J2_HOME_BALANCED_RESTORE`: 5
- `J2_NEUTRAL_HOME_RESTORE`: 4
- `J2_HOME_DRAW_RESTORE`: 3

Interpretation:
- J2 restore logic has been simplified without changing results.
- `home_balanced` and `home_draw` are already consolidated.

## J2 Simplification Progress

- Before merge:
  - `decision_reason` kinds: 22
  - restore-specific kinds: 20
- After `home_draw` merge:
  - `decision_reason` kinds: 20
  - restore-specific kinds: 18
- After `home_balanced` merge:
  - `decision_reason` kinds: 18
  - restore-specific kinds: 16

Reference:
- `data/reports/metrics/j2_2026_reason_simplification_summary.csv`
- `data/reports/metrics/j2_2026_reason_group_mapping.csv`
- `data/reports/metrics/j2_2026_reason_count_compare.csv`

## Working Interpretation

- Core architecture is functionally connected.
- J1 and J2 are both in a usable validation state.
- Remaining work is now mostly:
  - explanation cleanup,
  - targeted review of large draw-related branches,
  - final report packaging for next-match evaluation.
- Latest J1 gain came from `neutral × split_side` in `main`, where 3 common matches moved `D -> A` and all 3 became hits.

## Friday Review Targets

1. Re-run J1/J2 latest backtests and confirm no regression from the baseline snapshots above.
2. Review `NARROW_DRAW_OVERRIDE` concentration in both leagues, especially J1 `signal_conflict × basis_balanced`.
3. Decide whether `neutral_draw` and `neutral_balanced` in J2 should remain split or stay as-is for the season handoff.
4. Prepare next-match review from current production outputs, not from the older 60%-complete version.
