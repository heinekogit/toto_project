#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/scripts/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

SEASON_YEAR="${SEASON_YEAR:-2026}"
ROUND="${ROUND:-round02}"
PURCHASE_DIR="${PURCHASE_DIR:-$ROOT_DIR/data/purchase_reference}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/data/eval}"

ROUND_DIR="$EVAL_ROOT/rounds/$ROUND"
SNAPSHOT_DIR="$ROUND_DIR/snapshot"

echo "==> snapshot"
"$PYTHON" "$ROOT_DIR/scripts/eval/00_snapshot_purchase.py" \
  --round "$ROUND" \
  --srcdir "$PURCHASE_DIR" \
  --outdir "$SNAPSHOT_DIR"

echo "==> export actual results"
"$PYTHON" "$ROOT_DIR/scripts/eval/01_export_actual_results.py" \
  --round "$ROUND" \
  --season "$SEASON_YEAR" \
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

echo "完了: $ROUND_DIR"

