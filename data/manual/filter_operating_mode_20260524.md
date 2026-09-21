# Filter Operating Mode 2026-05-24

## Current policy

- Keep `j1_draw_rescue_watch` as a single interpretation type for now.
- Keep `j2_false_draw_watch` and `j2_draw_trap` separated.
- Do not further split J1 draw-rescue cases until more new-season samples accumulate.
- Keep buyplan connected to the latest interpretation filters.

## Reason

- Current J1 draw-heavy matches are still too homogeneous to cleanly split into finer subtypes.
- Forcing a `live/soft` split now would be arbitrary.
- J2 already shows a meaningful separation between false-D watch and draw-trap behavior.

## What to monitor next

- `j1_draw_rescue_watch`
- `j2_false_draw_watch`
- `j2_draw_trap`

For each type, evaluate:

- actual H/D/A distribution
- main hit rate
- rescue / harm / net rescue
- overlap with all-same ticket concentration
- whether subtype boundaries become stable enough for promotion/refinement

## Promotion rule

- Do not promote a finer subtype on intuition alone.
- Revisit only after several new-season rounds accumulate enough samples.
