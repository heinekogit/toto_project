#!/usr/bin/env bash
# Rebuild human-readable validation from saved purchases/results only.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/scripts/.venv/bin/python}"
ROUND="${ROUND:?ROUND=toto1660 のように開催を指定してください}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/data/eval}"
"$PYTHON" "$ROOT_DIR/scripts/eval/20_build_growth_report.py" --round "$ROUND" --eval-root "$EVAL_ROOT"
