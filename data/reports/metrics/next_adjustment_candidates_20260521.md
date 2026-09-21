# Next Adjustment Candidates

## Buyplan
- `primary=D, secondary=A, cluster=draw_core, basis=draw_compressed`: avg `H/D/A = 1.5/1.0/7.5` over `2` matches
- `primary=D, secondary=A, cluster=low_tempo_draw, basis=draw_compressed`: avg `H/D/A = 0.0/7.0/3.0` over `1` matches
- `primary=D, secondary=A, cluster=low_tempo_draw, basis=flat_draw_trap`: avg `H/D/A = 2.0/6.0/2.0` over `1` matches
- `primary=D, secondary=H, cluster=hold_core, basis=basis_balanced`: avg `H/D/A = 1.0/8.0/1.0` over `1` matches

## Predictor
- `J2 / mixed_close`: pred_draw_rate `0.562`, actual_draw_rate `0.167`, draw_miss_to_side_rate `0.458`, side_miss_to_draw_rate `0.062` (`48` matches)
- `J1 / draw_core`: pred_draw_rate `0.592`, actual_draw_rate `0.368`, draw_miss_to_side_rate `0.395`, side_miss_to_draw_rate `0.171` (`76` matches)
- `J2 / soft_draw`: pred_draw_rate `0.353`, actual_draw_rate `0.176`, draw_miss_to_side_rate `0.294`, side_miss_to_draw_rate `0.118` (`17` matches)
- `J1 / soft_draw`: pred_draw_rate `0.333`, actual_draw_rate `0.167`, draw_miss_to_side_rate `0.278`, side_miss_to_draw_rate `0.111` (`18` matches)
- `J2 / swing_close`: pred_draw_rate `0.231`, actual_draw_rate `0.077`, draw_miss_to_side_rate `0.231`, side_miss_to_draw_rate `0.077` (`13` matches)
- `J1 / mixed_close`: pred_draw_rate `0.171`, actual_draw_rate `0.343`, draw_miss_to_side_rate `0.086`, side_miss_to_draw_rate `0.257` (`35` matches)
- `J1 / swing_close`: pred_draw_rate `0.043`, actual_draw_rate `0.130`, draw_miss_to_side_rate `0.043`, side_miss_to_draw_rate `0.130` (`23` matches)
- `J2 / soft_swing`: pred_draw_rate `0.000`, actual_draw_rate `1.000`, draw_miss_to_side_rate `0.000`, side_miss_to_draw_rate `1.000` (`2` matches)
- `J1 / undiff_close`: pred_draw_rate `0.000`, actual_draw_rate `0.500`, draw_miss_to_side_rate `0.000`, side_miss_to_draw_rate `0.500` (`2` matches)
- `J1 / soft_swing`: pred_draw_rate `0.000`, actual_draw_rate `0.250`, draw_miss_to_side_rate `0.000`, side_miss_to_draw_rate `0.250` (`16` matches)

## Priority
- `high` `buyplan` D->A side bias / draw_core / draw_compressed: matches=2, avg H=1.5, D=1.0, A=7.5
- `high` `buyplan` D->A side bias / low_tempo_draw / draw_compressed: matches=1, avg H=0.0, D=7.0, A=3.0
- `high` `predictor` pred D overused / J2 / mixed_close: matches=48, draw_miss_to_side_rate=0.458, actual_draw_rate=0.167