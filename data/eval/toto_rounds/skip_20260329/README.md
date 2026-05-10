# Skip Record 2026-03-29

- status: skip
- date: 2026-03-29
- purchase_executed: no
- scope: prediction_only
- official_parallel_count: excluded
- reason: representative week; toto card included J2/J3 only, and J3 is outside the current prediction/buyplan coverage

## Notes

- This run is treated as an operational test or observation only.
- Do not include this run in baseline vs caution performance counting.
- Rebuild `data/purchase_reference/predictions.csv` fresh before the next real purchase cycle.
- `data/purchase_reference/` was not snapshotted here because the current contents still point to an older round (`toto_round_id: 第8節` in the active buyplan HTML).

## Suggested Follow-up

- If the actual toto round number for this skipped card is needed later, add it to the directory name or this note.
- If prediction artifacts for this skip run were saved elsewhere, place copies under this directory explicitly.
