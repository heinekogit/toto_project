#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/scripts/.venv/bin/python}"

SEASON_YEAR="${SEASON_YEAR:-2025}"
LEAGUES="${LEAGUES:-j1 j2}"
STATS_ASOF_DATE="${STATS_ASOF_DATE:-$(date +%F)}"
WEATHER_ASOF_DATE="${WEATHER_ASOF_DATE:-$STATS_ASOF_DATE}"
WEATHER_SNAPSHOT_DIR="${WEATHER_SNAPSHOT_DIR:-$ROOT_DIR/data/weather_snapshots}"
STADIUMS_FILE="${STADIUMS_FILE:-$ROOT_DIR/data/manual/stadiums.csv}"
ABSENCES_CSV="${ABSENCES_CSV:-$ROOT_DIR/data/manual/欠場管理リスト.csv}"
PLAYERS_CSV="${PLAYERS_CSV:-$ROOT_DIR/data/manual/jleague_club_players.csv}"
ABSENCE_OUT_DIR="${ABSENCE_OUT_DIR:-$ROOT_DIR/data/manual}"
ABSENCE_SNAPSHOT_DIR="${ABSENCE_SNAPSHOT_DIR:-$ROOT_DIR/data/absence_snapshots}"
WEATHER_LOOKAHEAD_DAYS="${WEATHER_LOOKAHEAD_DAYS:-7}"
RUN_RANKINGS_UPDATE="${RUN_RANKINGS_UPDATE:-1}"
GENERATE_LOG_HTML="${GENERATE_LOG_HTML:-1}"
BATCH_LOG_PATH="${BATCH_LOG_PATH:-$ROOT_DIR/logs/run_batch_matchday.log}"
LOG_REPORT_HTML="${LOG_REPORT_HTML:-$ROOT_DIR/logs/run_batch_matchday_report.html}"

run_step () {
  echo "==> $*"
  local env_args=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do
    env_args+=("$1")
    shift
  done
  if [[ ${#env_args[@]} -gt 0 ]]; then
    env "${env_args[@]}" "$@"
  else
    "$@"
  fi
}

on_exit () {
  if [[ "$GENERATE_LOG_HTML" == "1" ]]; then
    if [[ -f "$BATCH_LOG_PATH" ]]; then
      "$PYTHON" "$ROOT_DIR/scripts/build_batch_log_report.py" \
        --input "$BATCH_LOG_PATH" \
        --output "$LOG_REPORT_HTML" \
        --title "試合直前バッチ ログ警告レポート" || true
    else
      echo "[WARN] ログHTML生成をスキップ: ログ未検出 ($BATCH_LOG_PATH)"
    fi
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi
trap on_exit EXIT

# 欠場影響を先に更新（予測特徴量で利用）
if [[ -f "$ABSENCES_CSV" && -f "$PLAYERS_CSV" ]]; then
  run_step "$PYTHON" "$ROOT_DIR/scripts/build_absence_impact.py" \
    "$ABSENCES_CSV" "$PLAYERS_CSV" --out-dir "$ABSENCE_OUT_DIR" --season "$SEASON_YEAR" \
    --asof-date "$STATS_ASOF_DATE" --snapshot-dir "$ABSENCE_SNAPSHOT_DIR"
else
  echo "[WARN] build_absence_impact.py をスキップ: 入力CSV不足 (absences=$ABSENCES_CSV players=$PLAYERS_CSV)"
fi

for LEAGUE in $LEAGUES; do
  echo "=== League: $LEAGUE / Season: $SEASON_YEAR ==="

  MATCHES_FILE="$ROOT_DIR/data/${LEAGUE}_${SEASON_YEAR}_upcoming.csv"
  WEATHER_OUT="$ROOT_DIR/data/manual/weather_features_${LEAGUE}_${SEASON_YEAR}.csv"
  WEATHER_ASOF_KEY="$(echo "$WEATHER_ASOF_DATE" | tr -cd '0-9')"
  WEATHER_SNAPSHOT_OUT="$WEATHER_SNAPSHOT_DIR/weather_features_${LEAGUE}_${SEASON_YEAR}_asof_${WEATHER_ASOF_KEY}.csv"
  mkdir -p "$WEATHER_SNAPSHOT_DIR"

  # 試合直前の再計算（週次以降の更新差分を吸収）
  if [[ "$RUN_RANKINGS_UPDATE" == "1" ]]; then
    run_step LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/02_update_rankings.py"
  fi
  run_step LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/01_update_match_schedule.py"
  run_step LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/06_calculate_fatigue.py"
  run_step LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/03_rankings_motivation.py"

  run_step WEATHER_ASOF_DATE="$WEATHER_ASOF_DATE" "$PYTHON" "$ROOT_DIR/fetch_weather_features.py" \
    --stadiums "$STADIUMS_FILE" \
    --matches "$MATCHES_FILE" \
    --lookahead-days "$WEATHER_LOOKAHEAD_DAYS" \
    --out "$WEATHER_SNAPSHOT_OUT"

  # 互換維持: 従来の固定パスも最新スナップショットで更新
  run_step cp "$WEATHER_SNAPSHOT_OUT" "$WEATHER_OUT"

  run_step LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" STATS_ASOF_DATE="$STATS_ASOF_DATE" "$PYTHON" "$ROOT_DIR/scripts/11_prediction_01.py"
done

echo "完了"
