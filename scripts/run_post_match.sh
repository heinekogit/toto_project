#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/scripts/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

SEASON="${SEASON:-2026}"
LEAGUE="${LEAGUE:-both}"   # j1 | j2 | both
ROUND="${ROUND:-2}"
FORCE_FLAG="${FORCE_FLAG:-0}"

CMD=(
  "$PYTHON" "$ROOT_DIR/scripts/post_match_snapshot.py"
  --season "$SEASON"
  --league "$LEAGUE"
  --round "$ROUND"
)

if [[ "$FORCE_FLAG" == "1" ]]; then
  CMD+=(--force)
fi

echo "==> ${CMD[*]}"
"${CMD[@]}"

