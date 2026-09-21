#!/usr/bin/env bash
set -uo pipefail

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
    return $?
  fi
  "$@"
  return $?
}

describe_step () {
  local script_name="$1"
  local purpose="$2"
  echo "[STEP] ${script_name} : ${purpose}"
}

FAILED_STEPS=0
OK_STEPS=0
ERROR_STEPS=0

preflight_check () {
  local host="$1"
  local url="https://${host}"
  # DNS/TCP 到達性のみ確認する（HTTPステータスは判定に使わない）
  if curl --connect-timeout 5 --max-time 8 -s "$url" >/dev/null 2>&1; then
    echo "[PREFLIGHT] ${host} : OK"
    return 0
  fi
  echo "[PREFLIGHT] ${host} : ERROR"
  return 1
}

run_step_safe () {
  local step_name="$1"
  shift
  if run_step "$@"; then
    OK_STEPS=$((OK_STEPS + 1))
    echo "[INFO] ${step_name} : success"
    echo "[RESULT] ${step_name} : OK"
    return 0
  else
    local rc=$?
    FAILED_STEPS=$((FAILED_STEPS + 1))
    ERROR_STEPS=$((ERROR_STEPS + 1))
    echo "[ERROR] ${step_name} : failed (exit=${rc})"
    echo "[WARN] ${step_name} : 続行します"
    echo "[RESULT] ${step_name} : ERROR"
    return "$rc"
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

preflight_ok=0
if preflight_check "data.j-league.or.jp"; then
  preflight_ok=1
fi
if preflight_check "www.jleague.jp"; then
  preflight_ok=1
fi
if [[ "$preflight_ok" -eq 0 ]]; then
  echo "FATAL: Network/DNS unavailable"
  exit 1
fi

# 欠場影響を先に更新（予測特徴量で利用）
if [[ -f "$ABSENCES_CSV" && -f "$PLAYERS_CSV" ]]; then
  describe_step "build_absence_impact.py" "欠場管理CSVと選手マスタから欠場影響CSVを更新"
  run_step_safe "build_absence_impact.py" "$PYTHON" "$ROOT_DIR/scripts/build_absence_impact.py" \
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
    describe_step "02_update_rankings.py" "順位表CSVを更新（新シーズン未公開時は失敗を記録して続行）"
    run_step_safe "02_update_rankings.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/02_update_rankings.py"
  fi
  describe_step "01_update_match_schedule.py" "今後日程CSVを更新（新シーズンの大会IDを自動解決）"
  run_step_safe "01_update_match_schedule.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/01_update_match_schedule.py"
  describe_step "06_calculate_fatigue.py" "最新日程から疲労スコアCSVを更新"
  run_step_safe "06_calculate_fatigue.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/06_calculate_fatigue.py" || exit 1
  describe_step "11_build_fatigue_qc.py" "本番疲労CSVの距離参照・マージを検査（不備は停止）"
  run_step_safe "11_build_fatigue_qc.py" "$PYTHON" "$ROOT_DIR/scripts/eval/11_build_fatigue_qc.py" \
    --season "$SEASON_YEAR" --leagues "$LEAGUE" || exit 1
  describe_step "03_rankings_motivation.py" "最新順位からモチベーション関連指標を更新"
  run_step_safe "03_rankings_motivation.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/03_rankings_motivation.py"

  describe_step "fetch_weather_features.py" "直近試合の天候特徴量スナップショットを更新"
  if run_step_safe "fetch_weather_features.py" WEATHER_ASOF_DATE="$WEATHER_ASOF_DATE" "$PYTHON" "$ROOT_DIR/fetch_weather_features.py" \
    --stadiums "$STADIUMS_FILE" \
    --matches "$MATCHES_FILE" \
    --lookahead-days "$WEATHER_LOOKAHEAD_DAYS" \
    --out "$WEATHER_SNAPSHOT_OUT"; then
    # 互換維持: 取得に成功したスナップショットだけを固定パスへ反映
    if [[ -f "$WEATHER_SNAPSHOT_OUT" ]]; then
      run_step_safe "weather_features_latest_copy" cp "$WEATHER_SNAPSHOT_OUT" "$WEATHER_OUT"
    else
      echo "[WARN] weather_features_latest_copy : 成功扱いですが出力未生成のため既存ファイルを維持します"
      echo "[RESULT] weather_features_latest_copy : SKIP"
      exit 1
    fi
  else
    echo "[WARN] weather_features_latest_copy : 天候取得失敗のため既存ファイルを維持します"
    echo "[RESULT] weather_features_latest_copy : SKIP"
    exit 1
  fi

  describe_step "11_prediction_01.py" "最新の取得済みデータで予測CSVを更新（節タイプ補正は無効化）"
  run_step_safe "11_prediction_01.py" ENABLE_ROUND_TYPE_DRAW_CONTROL="0" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" STATS_ASOF_DATE="$STATS_ASOF_DATE" "$PYTHON" "$ROOT_DIR/scripts/11_prediction_01.py" || exit 1
  PREDICTIONS_FILE="$ROOT_DIR/${LEAGUE}_${SEASON_YEAR}_predictions_hfa_on.csv"
  if [[ -f "$PREDICTIONS_FILE" ]]; then
    describe_step "12_build_weather_rule_shadow.py" "天候単位・大雨排他・欠損状態をQC（予測の追加変更なし）"
    run_step_safe "12_build_weather_rule_shadow.py" "$PYTHON" "$ROOT_DIR/scripts/eval/12_build_weather_rule_shadow.py" \
      --predictions "$PREDICTIONS_FILE" --label "${LEAGUE}_${SEASON_YEAR}_latest"
  fi
done

if [[ "$FAILED_STEPS" -gt 0 ]]; then
  echo "[WARN] 完了（一部STEP失敗: ${FAILED_STEPS}）"
else
  echo "[INFO] 完了（全STEP成功）"
fi
echo "=== SUMMARY ==="
echo "OK: ${OK_STEPS}"
echo "ERROR: ${ERROR_STEPS}"

if [[ "$GENERATE_LOG_HTML" == "1" ]]; then
  if [[ -f "$BATCH_LOG_PATH" ]]; then
    run_step "$PYTHON" "$ROOT_DIR/scripts/build_batch_log_report.py" \
      --input "$BATCH_LOG_PATH" \
      --output "$LOG_REPORT_HTML" \
      --title "試合直前バッチ ログ警告レポート"
  else
    echo "[WARN] ログHTML生成をスキップ: ログ未検出 ($BATCH_LOG_PATH)"
  fi
fi
