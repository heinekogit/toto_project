#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/scripts/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

PREDICTION_SEASON="${PREDICTION_SEASON:-2026}"
LOGICAL_SEASON="${LOGICAL_SEASON:-2027}"
ROUND_NO="${ROUND_NO:-1}"
PURCHASE_DIR="${PURCHASE_DIR:-$ROOT_DIR/data/purchase_reference}"
TOTO_ORDER_CSV="${TOTO_ORDER_CSV:-$ROOT_DIR/data/manual/toto節リスト.csv}"

echo "==> purchase reference predictions"
"$PYTHON" "$ROOT_DIR/scripts/build_purchase_reference_predictions.py" \
  --season "$PREDICTION_SEASON" \
  --logical-season "$LOGICAL_SEASON" \
  --round "$ROUND_NO" \
  --out "$PURCHASE_DIR/predictions.csv" \
  --out-context "$PURCHASE_DIR/predictions_buyplan_context.csv"

echo "==> buyplan"
"$PYTHON" "$ROOT_DIR/buyplan.py" \
  --in "$PURCHASE_DIR/predictions.csv" \
  --outdir "$PURCHASE_DIR" \
  --toto-order-csv "$TOTO_ORDER_CSV"

echo "==> immutable observation archive"
"$PYTHON" "$ROOT_DIR/scripts/archive_buyplan_observation.py" \
  --logical-season "$LOGICAL_SEASON" \
  --prediction-season "$PREDICTION_SEASON" \
  --round "$ROUND_NO" \
  --srcdir "$PURCHASE_DIR"

echo "完了: $PURCHASE_DIR/buyplan.html"
