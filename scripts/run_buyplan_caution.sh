#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/scripts/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

IN_CSV="${IN_CSV:-$ROOT_DIR/data/purchase_reference/predictions.csv}"
OUTDIR="${OUTDIR:-$ROOT_DIR/data/purchase_reference}"
TOTO_ORDER_CSV="${TOTO_ORDER_CSV:-$ROOT_DIR/data/manual/toto節リスト.csv}"
OUTPUT_NAME="${OUTPUT_NAME:-buyplan_caution}"
HISTORY_DIR="${HISTORY_DIR:-$OUTDIR/history_caution}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "==> buyplan_caution"
echo "PYTHON=$PYTHON"
echo "IN_CSV=$IN_CSV"
echo "OUTDIR=$OUTDIR"
echo "TOTO_ORDER_CSV=$TOTO_ORDER_CSV"
echo "OUTPUT_NAME=$OUTPUT_NAME"
echo "HISTORY_DIR=$HISTORY_DIR"

"$PYTHON" "$ROOT_DIR/buyplan.py" \
  --in "$IN_CSV" \
  --outdir "$OUTDIR" \
  --name "$OUTPUT_NAME" \
  --toto-order-csv "$TOTO_ORDER_CSV" \
  "$@"

mkdir -p "$HISTORY_DIR"
cp "$OUTDIR/${OUTPUT_NAME}.csv" "$HISTORY_DIR/${STAMP}_${OUTPUT_NAME}.csv"
cp "$OUTDIR/${OUTPUT_NAME}.html" "$HISTORY_DIR/${STAMP}_${OUTPUT_NAME}.html"

echo "完了: $OUTDIR/${OUTPUT_NAME}.csv"
echo "完了: $OUTDIR/${OUTPUT_NAME}.html"
echo "履歴: $HISTORY_DIR/${STAMP}_${OUTPUT_NAME}.csv"
echo "履歴: $HISTORY_DIR/${STAMP}_${OUTPUT_NAME}.html"
