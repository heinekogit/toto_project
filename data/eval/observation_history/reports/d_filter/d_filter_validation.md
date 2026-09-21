# Dフィルター観測評価

prediction本体には未接続。確定結果に対するcounterfactual評価のみを行う。

## 累積

| policy | 変更 | Hへ/Aへ | rescue | harm | net | 変更精度 | 的中率 before → after |
|---|---:|---:|---:|---:|---:|---:|---:|
| false_draw_only | 44 | 23/21 | 23 | 9 | +14 | 52.3% | 27.7% → 41.6% |
| false_draw_plus_trap | 58 | 31/27 | 33 | 11 | +22 | 56.9% | 27.7% → 49.5% |
| topology_guard | 57 | 30/27 | 34 | 11 | +23 | 59.6% | 27.7% → 50.5% |

## 節別

| 開催回 | policy | 変更 | rescue | harm | net | before → after |
|---|---|---:|---:|---:|---:|---:|
| toto1644 | false_draw_only | 8 | 5 | 1 | +4 | 15.4% → 46.2% |
| toto1644 | false_draw_plus_trap | 10 | 7 | 1 | +6 | 15.4% → 61.5% |
| toto1644 | topology_guard | 7 | 6 | 1 | +5 | 15.4% → 53.8% |
| toto1645 | false_draw_only | 2 | 1 | 1 | +0 | 46.2% → 46.2% |
| toto1645 | false_draw_plus_trap | 3 | 1 | 2 | -1 | 46.2% → 38.5% |
| toto1645 | topology_guard | 3 | 1 | 2 | -1 | 46.2% → 38.5% |
| toto1647 | false_draw_only | 6 | 3 | 1 | +2 | 23.1% → 38.5% |
| toto1647 | false_draw_plus_trap | 8 | 5 | 1 | +4 | 23.1% → 53.8% |
| toto1647 | topology_guard | 8 | 5 | 1 | +4 | 23.1% → 53.8% |
| toto1648 | false_draw_only | 3 | 3 | 0 | +3 | 30.8% → 53.8% |
| toto1648 | false_draw_plus_trap | 8 | 6 | 1 | +5 | 30.8% → 69.2% |
| toto1648 | topology_guard | 9 | 7 | 1 | +6 | 30.8% → 76.9% |
| toto1649 | false_draw_only | 5 | 3 | 1 | +2 | 20.0% → 40.0% |
| toto1649 | false_draw_plus_trap | 5 | 3 | 1 | +2 | 20.0% → 40.0% |
| toto1649 | topology_guard | 5 | 3 | 1 | +2 | 20.0% → 40.0% |
| toto1650 | false_draw_only | 7 | 2 | 2 | +0 | 23.1% → 23.1% |
| toto1650 | false_draw_plus_trap | 9 | 4 | 2 | +2 | 23.1% → 38.5% |
| toto1650 | topology_guard | 9 | 4 | 2 | +2 | 23.1% → 38.5% |
| toto1653 | false_draw_only | 9 | 5 | 2 | +3 | 23.1% → 46.2% |
| toto1653 | false_draw_plus_trap | 10 | 5 | 2 | +3 | 23.1% → 46.2% |
| toto1653 | topology_guard | 10 | 5 | 2 | +3 | 23.1% → 46.2% |
| toto1654 | false_draw_only | 4 | 1 | 1 | +0 | 38.5% → 38.5% |
| toto1654 | false_draw_plus_trap | 5 | 2 | 1 | +1 | 38.5% → 46.2% |
| toto1654 | topology_guard | 6 | 3 | 1 | +2 | 38.5% → 53.8% |

## policy定義

- `false_draw_only`: J1/J2のfalse_draw_watchだけをrank2側へ変更。
- `false_draw_plus_trap`: false_draw_watchにdraw_trap系を加える。
- `topology_guard`: 強いD根拠を満たすものだけDを保持し、それ以外をrank2側へ変更。

正式採用条件は、複数節で net rescue が正、harmが許容範囲、リーグ別でも再現すること。
