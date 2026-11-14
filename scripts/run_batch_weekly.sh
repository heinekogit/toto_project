#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/scripts/.venv/bin/python}"

SEASON_YEAR="${SEASON_YEAR:-2025}"
LEAGUES="${LEAGUES:-j1 j2}"
RUN_WEATHER="${RUN_WEATHER:-0}"
STATS_ASOF_DATE="${STATS_ASOF_DATE:-$(date +%F)}"
WEATHER_ASOF_DATE="${WEATHER_ASOF_DATE:-$STATS_ASOF_DATE}"
WEATHER_SNAPSHOT_DIR="${WEATHER_SNAPSHOT_DIR:-$ROOT_DIR/data/weather_snapshots}"
STADIUMS_FILE="${STADIUMS_FILE:-$ROOT_DIR/data/manual/stadiums.csv}"
# 最新の台帳エクスポートを全件置換入力に使用。旧台帳とのマージなし。
ABSENCES_CSV="${ABSENCES_CSV:-$ROOT_DIR/data/manual/temporary_absense/absence-ledger-partial.csv}"
PLAYERS_CSV="${PLAYERS_CSV:-$ROOT_DIR/data/manual/jleague_club_players.csv}"
PLAYERS_SUMMARY_JSON="${PLAYERS_SUMMARY_JSON:-$ROOT_DIR/data/manual/jleague_club_players_summary.json}"
PLAYER_FETCH_SLEEP_SEC="${PLAYER_FETCH_SLEEP_SEC:-1.5}"
ABSENCE_OUT_DIR="${ABSENCE_OUT_DIR:-$ROOT_DIR/data/manual}"
ABSENCE_SNAPSHOT_DIR="${ABSENCE_SNAPSHOT_DIR:-$ROOT_DIR/data/absence_snapshots}"
GENERATE_LOG_HTML="${GENERATE_LOG_HTML:-1}"
BATCH_LOG_PATH="${BATCH_LOG_PATH:-$ROOT_DIR/logs/run_batch_weekly.log}"
LOG_REPORT_HTML="${LOG_REPORT_HTML:-$ROOT_DIR/logs/run_batch_weekly_report.html}"

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

# 欠場影響を先に更新（予測特徴量で利用）。名簿取得失敗時は古い名簿で続行しない。
describe_step "fetch_jleague_club_players.py" "公式選手名簿を全クラブ取得"
run_step_safe "fetch_jleague_club_players.py" "$PYTHON" "$ROOT_DIR/scripts/fetch_jleague_club_players.py" \
  --out "$PLAYERS_CSV" --summary-out "$PLAYERS_SUMMARY_JSON" \
  --min-clubs 60 --sleep-sec "$PLAYER_FETCH_SLEEP_SEC" || exit 1
if [[ -f "$ABSENCES_CSV" && -f "$PLAYERS_CSV" ]]; then
  describe_step "build_absence_impact.py" "欠場管理CSVと選手マスタから欠場影響CSVを更新"
  run_step_safe "build_absence_impact.py" "$PYTHON" "$ROOT_DIR/scripts/build_absence_impact.py" \
    "$ABSENCES_CSV" "$PLAYERS_CSV" --out-dir "$ABSENCE_OUT_DIR" --season "$SEASON_YEAR" \
    --asof-date "$STATS_ASOF_DATE" --snapshot-dir "$ABSENCE_SNAPSHOT_DIR" \
    --max-player-age-days 1 --reject-active-unmatched || exit 1
else
  echo "[ERROR] build_absence_impact.py 入力CSV不足 (absences=$ABSENCES_CSV players=$PLAYERS_CSV)"
  exit 1
fi

describe_step "update_emperor_cup.py" "JFA公式の天皇杯終了試合を更新（疲労計算用）"
# Stop on failure rather than silently calculate with stale cup results.
cup_failed_before="$FAILED_STEPS"
run_step_safe "update_emperor_cup.py" "$PYTHON" "$ROOT_DIR/scripts/update_emperor_cup.py" --season "$SEASON_YEAR"
if [[ "$FAILED_STEPS" -gt "$cup_failed_before" ]]; then
  echo "[ERROR] 天皇杯の更新に失敗しました。週次処理を停止します。"
  exit 1
fi

for LEAGUE in $LEAGUES; do
  echo "=== League: $LEAGUE / Season: $SEASON_YEAR ==="

  # 週の初め：結果・順位・スタッツ
  describe_step "01_update_match_results.py" "試合結果CSVを更新（終了試合のスコア反映）"
  if run_step_safe "01_update_match_results.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/01_update_match_results.py"; then
    # Append-only research observation. No prediction or purchase input is changed.
    RESULT_CSV="$ROOT_DIR/data/${LEAGUE}_${SEASON_YEAR}_latest_results.csv"
    if [[ -f "$RESULT_CSV" ]]; then
      if research_output=$("$PYTHON" "$ROOT_DIR/scripts/next_model/forecast_registry.py" sync-league-results --results "$RESULT_CSV" 2>&1); then
        echo "$research_output"
      else
        research_rc=$?
        echo "[WARN] forecast registry result sync failed (exit=${research_rc}, source=${RESULT_CSV}): ${research_output}"
        "$PYTHON" "$ROOT_DIR/scripts/next_model/condition_s_registry.py" record-failure \
          --stage league-result-sync --round "${LEAGUE}_${SEASON_YEAR}" --source "$RESULT_CSV" --reason "exit=${research_rc}: ${research_output}" || true
      fi
      if research_output=$("$PYTHON" "$ROOT_DIR/scripts/next_model/condition_s_registry.py" sync-results 2>&1); then
        echo "$research_output"
      else
        research_rc=$?
        echo "[WARN] condition S result sync failed (exit=${research_rc}, source=${RESULT_CSV}): ${research_output}"
        "$PYTHON" "$ROOT_DIR/scripts/next_model/condition_s_registry.py" record-failure \
          --stage condition-s-result-sync --round "${LEAGUE}_${SEASON_YEAR}" --source "$RESULT_CSV" --reason "exit=${research_rc}: ${research_output}" || true
      fi
    fi
  fi
  describe_step "02_update_rankings.py" "順位表CSVを更新（勝点・得失点など）"
  run_step_safe "02_update_rankings.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/02_update_rankings.py"
  describe_step "20_update_stats.py" "チームスタッツCSVを更新（予測特徴量の基礎データ）"
  run_step_safe "20_update_stats.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" STATS_ASOF_DATE="$STATS_ASOF_DATE" "$PYTHON" "$ROOT_DIR/scripts/20_update_stats.py"

  # 週の中盤：日程・疲労・モチベーション
  describe_step "01_update_match_schedule.py" "今後日程CSVを更新（試合日時・対戦カード）"
  run_step_safe "01_update_match_schedule.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/01_update_match_schedule.py"
  describe_step "06_calculate_fatigue.py" "疲労スコアCSVを作成/更新（連戦・移動の影響）"
  run_step_safe "06_calculate_fatigue.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/06_calculate_fatigue.py" || exit 1
  describe_step "11_build_fatigue_qc.py" "本番疲労CSVの距離参照・マージを検査（不備は停止）"
  run_step_safe "11_build_fatigue_qc.py" "$PYTHON" "$ROOT_DIR/scripts/eval/11_build_fatigue_qc.py" \
    --season "$SEASON_YEAR" --leagues "$LEAGUE" || exit 1
  describe_step "03_rankings_motivation.py" "モチベーション関連指標を更新（順位文脈の補助特徴量）"
  run_step_safe "03_rankings_motivation.py" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" "$PYTHON" "$ROOT_DIR/scripts/03_rankings_motivation.py"


  # 試合直前：天候取得（任意）
  if [[ "$RUN_WEATHER" == "1" ]]; then
    MATCHES_FILE="$ROOT_DIR/data/${LEAGUE}_${SEASON_YEAR}_upcoming.csv"
    OUT_FILE="$ROOT_DIR/data/manual/weather_features_${LEAGUE}_${SEASON_YEAR}.csv"
    WEATHER_ASOF_KEY="$(echo "$WEATHER_ASOF_DATE" | tr -cd '0-9')"
    SNAPSHOT_OUT="$WEATHER_SNAPSHOT_DIR/weather_features_${LEAGUE}_${SEASON_YEAR}_asof_${WEATHER_ASOF_KEY}.csv"
    mkdir -p "$WEATHER_SNAPSHOT_DIR"
    describe_step "fetch_weather_features.py" "天候特徴量CSVを取得/更新（雨・風など）"
    if run_step_safe "fetch_weather_features.py" WEATHER_ASOF_DATE="$WEATHER_ASOF_DATE" "$PYTHON" "$ROOT_DIR/fetch_weather_features.py" \
      --stadiums "$STADIUMS_FILE" \
      --matches "$MATCHES_FILE" \
      --lookahead-days "${WEATHER_LOOKAHEAD_DAYS:-7}" \
      --out "$SNAPSHOT_OUT"; then
      if [[ -f "$SNAPSHOT_OUT" ]]; then
        run_step_safe "weather_features_latest_copy" cp "$SNAPSHOT_OUT" "$OUT_FILE"
      else
        echo "[ERROR] 天候取得の出力がありません。予測を停止します。"
        exit 1
      fi
    else
      echo "[ERROR] 天候取得失敗。予測を停止します。"
      exit 1
    fi
  fi
  describe_step "11_prediction_01.py" "予測CSVを更新（節タイプ補正は無効化）"
  run_step_safe "11_prediction_01.py" ENABLE_ROUND_TYPE_DRAW_CONTROL="0" LEAGUE="$LEAGUE" SEASON_YEAR="$SEASON_YEAR" STATS_ASOF_DATE="$STATS_ASOF_DATE" ABSENCE_ASOF_DATE="$STATS_ASOF_DATE" ABSENCE_SNAPSHOT_DIR="$ABSENCE_SNAPSHOT_DIR" ABSENCE_REQUIRE_EXACT_ASOF="1" "$PYTHON" "$ROOT_DIR/scripts/11_prediction_01.py" || exit 1
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
      --title "週次バッチ ログ警告レポート"
  else
    echo "[WARN] ログHTML生成をスキップ: ログ未検出 ($BATCH_LOG_PATH)"
  fi
fi
