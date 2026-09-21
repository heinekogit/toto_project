#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/scripts/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

SEASON_YEAR="${SEASON_YEAR:-2026}"
PREDICTION_SEASON_YEAR="${PREDICTION_SEASON_YEAR:-2026}"
ROUND="${ROUND:-round02}"
PURCHASE_DIR="${PURCHASE_DIR:-$ROOT_DIR/data/purchase_reference}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/data/eval}"

if [[ "$ROUND" == toto* ]]; then
  ROUND_DIR="$EVAL_ROOT/toto_rounds/$ROUND"
else
  ROUND_DIR="$EVAL_ROOT/rounds/$ROUND"
fi
SNAPSHOT_DIR="$ROUND_DIR/snapshot"

expected_round_no=""
if [[ "$ROUND" == round* ]]; then
  expected_round_no="${ROUND#round}"
  expected_round_no="${expected_round_no##0}"
  if [[ -z "$expected_round_no" ]]; then
    expected_round_no="0"
  fi
fi

validate_snapshot_round() {
  local pred_csv="$1"
  "$PYTHON" - "$pred_csv" "$expected_round_no" <<'PY'
import csv
import re
import sys
import unicodedata

pred_csv = sys.argv[1]
expected_arg = sys.argv[2].strip()
expected = int(expected_arg) if expected_arg else None

def normalize_league(value):
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    if text in {"j1", "j1リーグ", "明治安田j1", "明治安田j1リーグ"}:
        return "j1"
    if text in {"j2", "j2リーグ", "明治安田j2", "明治安田j2リーグ"}:
        return "j2"
    return ""

with open(pred_csv, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rounds = set()
    rounds_by_league = {}
    seen = 0
    for row in reader:
        seen += 1
        raw = (row.get("節") or "").strip()
        if not raw:
            continue
        text = unicodedata.normalize("NFKC", raw)
        m = re.search(r"第?\s*([0-9]+)\s*節", text)
        if m:
            round_no = int(m.group(1))
            rounds.add(round_no)
            league = normalize_league(row.get("league"))
            if league:
                rounds_by_league.setdefault(league, set()).add(round_no)
        if seen >= 200:
            break

if not rounds:
    print(f"[WARN] 節列から節番号を抽出できませんでした: {pred_csv}")
    sys.exit(0)

bad_leagues = {lg: vals for lg, vals in rounds_by_league.items() if len(vals) > 1}
if bad_leagues:
    detail = ", ".join(f"{lg}={sorted(vals)}" for lg, vals in sorted(bad_leagues.items()))
    print(f"ERROR: 同一リーグ内でsnapshotの節番号が複数あります: {detail} ({pred_csv})")
    sys.exit(1)

if len(rounds) == 1:
    actual = next(iter(rounds))
    if expected is None:
        print(f"[OK] snapshot節番号チェック: 保存ID={sys.argv[2] or 'toto'} / snapshot=第{actual}節")
        sys.exit(0)
    if actual != expected:
        print(
            "ERROR: ROUNDとsnapshotの節番号が一致しません: "
            f"ROUND=round{expected:02d}, snapshot=第{actual}節 ({pred_csv})"
        )
        sys.exit(1)
    print(f"[OK] snapshot節番号チェック: round{expected:02d} == 第{actual}節")
    sys.exit(0)

if rounds_by_league:
    detail = ", ".join(
        f"{lg}=第{next(iter(sorted(vals)))}節"
        for lg, vals in sorted(rounds_by_league.items())
    )
else:
    detail = ",".join(f"第{x}節" for x in sorted(rounds))

print(
    "[OK] snapshot mixed節チェック: "
    f"{detail} / ROUND={expected if expected is not None else 'toto'} は保存用IDとして扱います"
)
PY
}

echo "==> snapshot"
if [[ -d "$SNAPSHOT_DIR" ]]; then
  echo "[INFO] 既存snapshotを再利用: $SNAPSHOT_DIR"
else
  "$PYTHON" "$ROOT_DIR/scripts/eval/00_snapshot_purchase.py" \
    --round "$ROUND" \
    --srcdir "$PURCHASE_DIR" \
    --outdir "$SNAPSHOT_DIR" \
    --logical-season "$SEASON_YEAR" \
    --prediction-season "$PREDICTION_SEASON_YEAR"
fi
validate_snapshot_round "$SNAPSHOT_DIR/predictions.csv"

echo "==> align snapshot buyplan"
"$PYTHON" "$ROOT_DIR/scripts/eval/00_align_snapshot_buyplan.py" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --python "$PYTHON"

echo "==> export actual results"
"$PYTHON" "$ROOT_DIR/scripts/eval/01_export_actual_results.py" \
  --round "$ROUND" \
  --season "$SEASON_YEAR" \
  --result-season "$PREDICTION_SEASON_YEAR" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --out "$ROUND_DIR/actual_results.csv" \
  --python "$PYTHON"

echo "==> score buyplan"
"$PYTHON" "$ROOT_DIR/scripts/eval/02_score_buyplan.py" \
  --round "$ROUND" \
  --buyplan "$SNAPSHOT_DIR/buyplan.csv" \
  --actual "$ROUND_DIR/actual_results.csv" \
  --out "$ROUND_DIR/evaluation.csv" \
  --history "$EVAL_ROOT/candidate_history.csv"

echo "==> build scored html"
"$PYTHON" "$ROOT_DIR/scripts/eval/03_build_scored_html.py" \
  --round "$ROUND" \
  --buyplan "$SNAPSHOT_DIR/buyplan.csv" \
  --actual "$ROUND_DIR/actual_results.csv" \
  --evaluation "$ROUND_DIR/evaluation.csv" \
  --out "$ROUND_DIR/buyplan_scored.html"

echo "==> accumulate observations"
"$PYTHON" "$ROOT_DIR/scripts/eval/04_accumulate_observations.py" \
  --round "$ROUND" \
  --round-dir "$ROUND_DIR" \
  --out-dir "$EVAL_ROOT/observation_history"

echo "==> accumulate post-match market alignment"
"$PYTHON" "$ROOT_DIR/scripts/eval/19_build_market_alignment.py" \
  --round "$ROUND" \
  --round-dir "$ROUND_DIR" \
  --history-dir "$EVAL_ROOT/observation_history"

echo "==> rebuild cross-round observation report"
"$PYTHON" "$ROOT_DIR/scripts/eval/05_build_observation_report.py" \
  --history-dir "$EVAL_ROOT/observation_history"

echo "==> evaluate observation-only D filters"
"$PYTHON" "$ROOT_DIR/scripts/eval/06_evaluate_d_filter_candidates.py" \
  --history-dir "$EVAL_ROOT/observation_history"

echo "==> condition and upset feedback"
"$PYTHON" "$ROOT_DIR/scripts/eval/13_build_condition_feedback.py" \
  --history-dir "$EVAL_ROOT/observation_history"

SHADOW_PARENT="$EVAL_ROOT/d_filter_shadow/$SEASON_YEAR/$ROUND"
if [[ -d "$SHADOW_PARENT" ]]; then
  echo "==> score frozen pre-match D-filter shadow"
  "$PYTHON" "$ROOT_DIR/scripts/eval/08_score_d_filter_shadow.py" \
    --round "$ROUND" \
    --logical-season "$SEASON_YEAR" \
    --shadow-root "$EVAL_ROOT/d_filter_shadow" \
    --actual "$ROUND_DIR/actual_results.csv"
else
  echo "[INFO] D-filter shadow採点をスキップ: pre-match snapshotなし ($SHADOW_PARENT)"
fi

echo "==> rebuild scored index"
if [[ "$ROUND" == toto* ]]; then
  "$PYTHON" "$ROOT_DIR/data/reports/build_buyplan_scored_index.py" \
    --rounds-dir "$EVAL_ROOT/toto_rounds" \
    --kind toto \
    --season "$SEASON_YEAR" \
    --toto-order-csv "$ROOT_DIR/data/manual/toto節リスト.csv" \
    --out "$EVAL_ROOT/toto_rounds/buyplan_scored_index.html"
else
  "$PYTHON" "$ROOT_DIR/data/reports/build_buyplan_scored_index.py" \
    --rounds-dir "$EVAL_ROOT/rounds" \
    --kind round \
    --out "$EVAL_ROOT/rounds/buyplan_scored_index.html"
fi

echo "完了: $ROUND_DIR"
