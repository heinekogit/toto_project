#   ■スクリプトの機能概要
#   j1_2024_results.csv：2024年シーズンの全試合（終了分）※任意
#   j1_20xx_upcoming.csv：対象シーズン（終了試合＋未開催試合）
#       → 終了試合は学習素材として、未開催試合に予測を行う。
#   j1_2025_predictions.csv（出力：）
#       → 未開催試合に予測勝敗と確率を付与し、結果をCSV出力。
#
#   ■予測ロジック（シンプルElo風）
#   各チームの基本強さは、対象シーズンの終了試合からElo風スコアを構築。
#   直前シーズンのデータがある場合は学習素材として追加で使用。
#   ホーム補正あり。
#   未開催の試合のみ予測対象。
#   =================================================



import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import poisson
import json
import re
import hashlib
import subprocess
import pickle
from datetime import datetime
import unicodedata

# パスはスクリプト起点で固定
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MANUAL_DIR = os.path.join(DATA_DIR, "manual")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
STATS_SNAPSHOT_DIR = os.path.join(DATA_DIR, "stats_snapshots")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
SEASON_YEAR = int(os.environ.get("SEASON_YEAR", "2025"))
TOTO_ROUND_ID = os.environ.get("TOTO_ROUND_ID", "").strip()
ROUND_NO_ENV = os.environ.get("ROUND_NO", "").strip()
STATS_ASOF_DATE = os.environ.get("STATS_ASOF_DATE", "").strip()
STATS_SNAPSHOT_NAME = os.environ.get("STATS_SNAPSHOT_NAME", "").strip()
WEATHER_ASOF_DATE = os.environ.get("WEATHER_ASOF_DATE", STATS_ASOF_DATE).strip()
WEATHER_SNAPSHOT_NAME = os.environ.get("WEATHER_SNAPSHOT_NAME", "").strip()
WEATHER_SNAPSHOT_DIR = os.environ.get("WEATHER_SNAPSHOT_DIR", os.path.join(DATA_DIR, "weather_snapshots"))
ABSENCE_ASOF_DATE = os.environ.get("ABSENCE_ASOF_DATE", STATS_ASOF_DATE).strip()
ABSENCE_SNAPSHOT_NAME = os.environ.get("ABSENCE_SNAPSHOT_NAME", "").strip()
ABSENCE_SNAPSHOT_DIR = os.environ.get("ABSENCE_SNAPSHOT_DIR", os.path.join(DATA_DIR, "absence_snapshots"))
TOTO_ORDER_CSV = os.environ.get("TOTO_ORDER_CSV", os.path.join(MANUAL_DIR, "toto並び順.csv"))
RAW_CLI_ARGS = sys.argv[1:]
CLI_ARGS = set(RAW_CLI_ARGS)


def _get_env_int(name, default):
    raw = os.environ.get(name, str(default))
    try:
        return int(str(raw).strip())
    except Exception:
        print(f"[CONFIG][WARN] invalid int env {name}={raw!r}; fallback={default}")
        return int(default)


def _env_flag(name, default=0):
    return _get_env_int(name, default) == 1


def _get_cli_int_arg(flag, default):
    try:
        idx = RAW_CLI_ARGS.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(RAW_CLI_ARGS):
        return default
    try:
        return int(str(RAW_CLI_ARGS[idx + 1]).strip())
    except Exception:
        return default


FORCE_RECALC = ("--force" in CLI_ARGS) or _env_flag("FORCE_RECALC", 0)
SELF_CHECK_HFA = ("--self-check-hfa" in CLI_ARGS) or _env_flag("SELF_CHECK_HFA", 0)
SKIP_HFA_SELF_CHECK = ("--skip-hfa-self-check" in CLI_ARGS) or _env_flag("SKIP_HFA_SELF_CHECK", 0)
DUMP_DECISION = ("--dump-decision" in CLI_ARGS) or _env_flag("DUMP_DECISION", 0)
HFA_TRACE_N = max(1, _get_cli_int_arg("--hfa-trace-n", _get_env_int("HFA_TRACE_N", 5)))
DECISION_RULE_DESC = "argmax(prob_home_win, prob_draw, prob_away_win)"
MERGE_QC_DIR = os.path.join(REPORT_DIR, "merge_qc", f"{LEAGUE}_{SEASON_YEAR}")
PREV_SEASON_YEAR = SEASON_YEAR - 1
csv_prev = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_results.csv")
if not os.path.exists(csv_prev):
    csv_prev_latest = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_latest_results.csv")
    if os.path.exists(csv_prev_latest):
        csv_prev = csv_prev_latest
prev_final_elo_csv = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_final_elo.csv")
csv_season = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_upcoming.csv")
csv_season_latest = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_latest_results.csv")
ENABLE_HFA_INT = _get_env_int("ENABLE_HFA", 1)
if ENABLE_HFA_INT not in (0, 1):
    print(f"[CONFIG][WARN] ENABLE_HFA should be 0/1 but got {ENABLE_HFA_INT}; coerced to {1 if ENABLE_HFA_INT else 0}")
    ENABLE_HFA_INT = 1 if ENABLE_HFA_INT else 0
ENABLE_HFA = ENABLE_HFA_INT == 1
hfa_suffix = "hfa_on" if ENABLE_HFA else "hfa_off"
LEGACY_OUTPUT_CSV = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions.csv")
output_csv_default = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions_{hfa_suffix}.csv")
output_csv = os.environ.get("OUTPUT_PRED_CSV", "").strip() or output_csv_default
backtest_output_csv = os.path.join(BASE_DIR, f"backtest_{LEAGUE}_{SEASON_YEAR}.csv")

def pick_non_empty_csv_path(candidates, required_cols=None):
    required_cols = required_cols or []
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"[PATH] 空CSVのためスキップ: {path}")
                continue
            if required_cols and not set(required_cols).issubset(df.columns):
                print(f"[PATH] 必須列不足のためスキップ: {path}")
                continue
            print(f"[PATH] 採用: {path} (rows={len(df)})")
            return path
        except Exception as e:
            print(f"[PATH] 読み込み失敗のためスキップ: {path} ({e})")
            continue
    return candidates[-1] if candidates else None


def _norm_key_text(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v)).replace("　", " ").strip()
    s = s.replace(" ", "").replace("・", "")
    return s.upper()


def _build_match_merge_key(df):
    out = df.copy()
    dt = pd.to_datetime(out.get("datetime"), errors="coerce")
    out["_dt_key"] = dt.dt.strftime("%Y-%m-%d %H:%M")
    out["_home_key"] = out.get("home_team", pd.Series(index=out.index, dtype="object")).map(_norm_key_text)
    out["_away_key"] = out.get("away_team", pd.Series(index=out.index, dtype="object")).map(_norm_key_text)
    out["_match_merge_key"] = out["_dt_key"].fillna("") + "|" + out["_home_key"].fillna("") + "|" + out["_away_key"].fillna("")
    return out


def enrich_scores_from_latest_results(df_season, latest_results_csv):
    if df_season is None or df_season.empty:
        return df_season
    if not os.path.exists(latest_results_csv):
        print(f"[SCORE_ENRICH] skip: latest_results not found ({latest_results_csv})")
        return df_season
    try:
        latest = pd.read_csv(latest_results_csv)
    except Exception as e:
        print(f"[SCORE_ENRICH][WARN] failed to read latest_results: {e}")
        return df_season
    required = {"home_team", "away_team"}
    if not required.issubset(latest.columns):
        print("[SCORE_ENRICH][WARN] latest_results missing required columns")
        return df_season
    work = df_season.copy()
    before_scored = int(
        pd.to_numeric(work.get("home_score"), errors="coerce").notna()
        .mul(pd.to_numeric(work.get("away_score"), errors="coerce").notna())
        .sum()
    )
    for col in ["home_score", "away_score"]:
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = pd.to_numeric(work[col], errors="coerce")
        if col not in latest.columns:
            latest[col] = pd.NA
        latest[col] = pd.to_numeric(latest[col], errors="coerce")

    # 1) match_id優先
    filled_by_match_id = 0
    if "match_id" in work.columns and "match_id" in latest.columns:
        right = latest[["match_id", "home_score", "away_score"]].dropna(subset=["match_id"]).drop_duplicates(
            subset=["match_id"], keep="last"
        )
        m = work.merge(right, on="match_id", how="left", suffixes=("", "__latest"))
        for col in ["home_score", "away_score"]:
            pre_na = m[col].isna()
            m[col] = m[col].fillna(m[f"{col}__latest"])
            filled_by_match_id += int(pre_na.sum() - m[col].isna().sum())
        work = m.drop(columns=["home_score__latest", "away_score__latest"], errors="ignore")

    # 2) datetime+home+awayで補完
    left = _build_match_merge_key(work)
    right = _build_match_merge_key(latest)
    right = right[["_match_merge_key", "home_score", "away_score"]].drop_duplicates(subset=["_match_merge_key"], keep="last")
    m2 = left.merge(right, on="_match_merge_key", how="left", suffixes=("", "__latest2"))
    filled_by_key = 0
    for col in ["home_score", "away_score"]:
        pre_na = m2[col].isna()
        m2[col] = m2[col].fillna(m2[f"{col}__latest2"])
        filled_by_key += int(pre_na.sum() - m2[col].isna().sum())
    work = m2.drop(
        columns=[
            "home_score__latest2",
            "away_score__latest2",
            "_dt_key",
            "_home_key",
            "_away_key",
            "_match_merge_key",
        ],
        errors="ignore",
    )

    after_scored = int(
        pd.to_numeric(work.get("home_score"), errors="coerce").notna()
        .mul(pd.to_numeric(work.get("away_score"), errors="coerce").notna())
        .sum()
    )
    print(
        f"[SCORE_ENRICH] source={latest_results_csv} "
        f"filled_by_match_id={filled_by_match_id} filled_by_key={filled_by_key} "
        f"scored_rows={before_scored}->{after_scored}"
    )
    return work


def _asof_key(value):
    if not value:
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def resolve_team_master_stats_csv():
    # 1) 明示指定（ファイル名）
    if STATS_SNAPSHOT_NAME:
        candidate = STATS_SNAPSHOT_NAME
        if not os.path.isabs(candidate):
            candidate = os.path.join(STATS_SNAPSHOT_DIR, candidate)
        if os.path.exists(candidate):
            print(f"[PATH] stats snapshot(明示指定)を採用: {candidate}")
            return candidate, _asof_key(os.path.basename(candidate))
        print(f"[PATH][WARN] 指定snapshotが見つかりません: {candidate}")

    # 2) snapshots から解決（asof指定があれば <= asof の最新）
    if os.path.isdir(STATS_SNAPSHOT_DIR):
        prefix = f"team_master_stats_{LEAGUE}_{SEASON_YEAR}_asof_"
        candidates = []
        for fn in os.listdir(STATS_SNAPSHOT_DIR):
            if not (fn.startswith(prefix) and fn.endswith(".csv")):
                continue
            asof = fn[len(prefix) : -4]
            if asof and asof.isdigit():
                path = os.path.join(STATS_SNAPSHOT_DIR, fn)
                candidates.append((asof, path))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            target_asof = _asof_key(STATS_ASOF_DATE)
            if target_asof:
                le_targets = [x for x in candidates if x[0] <= target_asof]
                if le_targets:
                    chosen = le_targets[-1]
                    print(f"[PATH] stats snapshot(asof={target_asof})を採用: {chosen[1]}")
                    return chosen[1], chosen[0]
                print(f"[PATH][WARN] asof={target_asof} 以下のsnapshotがないため最新を使用します。")
            chosen = candidates[-1]
            print(f"[PATH] stats snapshot(最新)を採用: {chosen[1]}")
            return chosen[1], chosen[0]

    # 3) 従来フォールバック
    fallback = pick_non_empty_csv_path(
        [
            os.path.join(DATA_DIR, f"team_master_stats_{LEAGUE}_{SEASON_YEAR}.csv"),
            os.path.join(DATA_DIR, f"team_master_stats_{SEASON_YEAR}.csv"),
            os.path.join(DATA_DIR, "team_master_stats.csv"),
        ],
        required_cols=["team_name"],
    )
    return fallback, ""


def resolve_weather_cache_csv():
    # 1) 明示指定
    if WEATHER_SNAPSHOT_NAME:
        candidate = WEATHER_SNAPSHOT_NAME
        if not os.path.isabs(candidate):
            candidate = os.path.join(WEATHER_SNAPSHOT_DIR, candidate)
        if os.path.exists(candidate):
            print(f"[PATH] weather snapshot(明示指定)を採用: {candidate}")
            return candidate, _asof_key(os.path.basename(candidate))
        print(f"[PATH][WARN] 指定weather snapshotが見つかりません: {candidate}")

    # 2) snapshots から解決（asof指定があれば <= asof）
    if os.path.isdir(WEATHER_SNAPSHOT_DIR):
        prefix = f"weather_features_{LEAGUE}_{SEASON_YEAR}_asof_"
        candidates = []
        for fn in os.listdir(WEATHER_SNAPSHOT_DIR):
            if not (fn.startswith(prefix) and fn.endswith(".csv")):
                continue
            asof = fn[len(prefix) : -4]
            if asof and asof.isdigit():
                candidates.append((asof, os.path.join(WEATHER_SNAPSHOT_DIR, fn)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            target_asof = _asof_key(WEATHER_ASOF_DATE)
            if target_asof:
                le_targets = [x for x in candidates if x[0] <= target_asof]
                if le_targets:
                    chosen = le_targets[-1]
                    print(f"[PATH] weather snapshot(asof={target_asof})を採用: {chosen[1]}")
                    return chosen[1], chosen[0]
                print(f"[PATH][WARN] weather asof={target_asof} 以下のsnapshotがないため最新を使用します。")
            chosen = candidates[-1]
            print(f"[PATH] weather snapshot(最新)を採用: {chosen[1]}")
            return chosen[1], chosen[0]

    # 3) 従来フォールバック
    fallback = pick_non_empty_csv_path(
        [
            os.path.join(MANUAL_DIR, f"weather_features_{LEAGUE}_{SEASON_YEAR}.csv"),
            os.path.join(DATA_DIR, f"weather_features_{LEAGUE}_{SEASON_YEAR}.csv"),
            os.path.join(MANUAL_DIR, "weather_cache.csv"),
            os.path.join(DATA_DIR, "weather_cache.csv"),
        ],
        required_cols=["match_id"],
    )
    return fallback, ""


def resolve_absence_impact_csv():
    # 1) 明示指定
    if ABSENCE_SNAPSHOT_NAME:
        candidate = ABSENCE_SNAPSHOT_NAME
        if not os.path.isabs(candidate):
            candidate = os.path.join(ABSENCE_SNAPSHOT_DIR, candidate)
        if os.path.exists(candidate):
            print(f"[PATH] absence snapshot(明示指定)を採用: {candidate}")
            return candidate, _asof_key(os.path.basename(candidate))
        print(f"[PATH][WARN] 指定absence snapshotが見つかりません: {candidate}")

    # 2) snapshots から解決
    # 許容ファイル名:
    # - absences_with_impact_asof_YYYYMMDD.csv
    # - absences_with_impact_<season>_asof_YYYYMMDD.csv
    if os.path.isdir(ABSENCE_SNAPSHOT_DIR):
        candidates = []
        pat1 = re.compile(r"^absences_with_impact_asof_(\d{8})\.csv$")
        pat2 = re.compile(r"^absences_with_impact_(\d{4})_asof_(\d{8})\.csv$")
        for fn in os.listdir(ABSENCE_SNAPSHOT_DIR):
            m1 = pat1.match(fn)
            m2 = pat2.match(fn)
            if m1:
                candidates.append((m1.group(1), os.path.join(ABSENCE_SNAPSHOT_DIR, fn)))
                continue
            if m2 and int(m2.group(1)) == int(SEASON_YEAR):
                candidates.append((m2.group(2), os.path.join(ABSENCE_SNAPSHOT_DIR, fn)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            target_asof = _asof_key(ABSENCE_ASOF_DATE)
            if target_asof:
                le_targets = [x for x in candidates if x[0] <= target_asof]
                if le_targets:
                    chosen = le_targets[-1]
                    print(f"[PATH] absence snapshot(asof={target_asof})を採用: {chosen[1]}")
                    return chosen[1], chosen[0]
                print(f"[PATH][WARN] absence asof={target_asof} 以下のsnapshotがないため最新を使用します。")
            chosen = candidates[-1]
            print(f"[PATH] absence snapshot(最新)を採用: {chosen[1]}")
            return chosen[1], chosen[0]

    # 3) 従来フォールバック
    fallback = pick_non_empty_csv_path(
        [
            os.path.join(MANUAL_DIR, "absences_with_impact.csv"),
            os.path.join(DATA_DIR, "absences_with_impact.csv"),
        ],
        required_cols=["team", "round_start", "impact_total"],
    )
    return fallback, ""


# 追加するパス（非空ファイルを優先）
team_master_stats_csv, stats_asof_key = resolve_team_master_stats_csv()
if not stats_asof_key:
    stats_asof_key = _asof_key(STATS_ASOF_DATE) or datetime.now().strftime("%Y%m%d")
STATS_ASOF_LABEL = (
    f"{stats_asof_key[:4]}-{stats_asof_key[4:6]}-{stats_asof_key[6:8]}"
    if len(stats_asof_key) >= 8
    else stats_asof_key
)
team_management_master_csv = pick_non_empty_csv_path(
    [
        os.path.join(MANUAL_DIR, "team_management_master.csv"),
        os.path.join(DATA_DIR, "team_management_master.csv"),
    ],
    required_cols=["team_name"],
)
absence_impact_csv, absence_asof_key = resolve_absence_impact_csv()
team_motivation_csv = pick_non_empty_csv_path(
    [
        os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_motivation.csv"),
        os.path.join(DATA_DIR, "team_motivation_master.csv"),
    ],
    required_cols=["team_name"],
)
team_travel_distances_csv = os.path.join(MANUAL_DIR, "team_travel_distances.csv")
if not os.path.exists(team_travel_distances_csv):
    team_travel_distances_csv = os.path.join(DATA_DIR, "team_travel_distances.csv")
team_fatigue_scores_csv = os.path.join(DATA_DIR, f"team_fatigue_scores_{LEAGUE}_{SEASON_YEAR}.csv")
if not os.path.exists(team_fatigue_scores_csv):
    team_fatigue_scores_csv = os.path.join(DATA_DIR, f"team_fatigue_scores_{SEASON_YEAR}.csv")
if not os.path.exists(team_fatigue_scores_csv):
    team_fatigue_scores_csv = os.path.join(DATA_DIR, "team_fatigue_scores.csv")

# 天候キャッシュは as-of 付きsnapshotを優先し、なければ従来ファイルへフォールバック
weather_cache_csv, weather_asof_key = resolve_weather_cache_csv()
J2_ALLOWED_TEAMS_CSV = os.environ.get(
    "J2_ALLOWED_TEAMS_CSV",
    os.path.join(MANUAL_DIR, f"j2_allowed_teams_{SEASON_YEAR}.csv"),
)

# Elo-like 初期スコア
INITIAL_ELO = 1500
ELO_UPDATE_HOME_ADVANTAGE = float(os.environ.get("ELO_UPDATE_HOME_ADVANTAGE", "0"))
GOAL_SCALING_FACTOR = 0.01
FATIGUE_GOAL_SCALING = 0.01
RANK_MOTIVATION_GOAL_SCALING = float(os.environ.get("RANK_MOTIVATION_GOAL_SCALING", "0.01"))
ABSENCE_IMPACT_GOAL_SCALING = float(os.environ.get("ABSENCE_IMPACT_GOAL_SCALING", "0.25"))
# 欠場データ欠損時のベースライン（観測バイアス緩和）
ABSENCE_BASELINE_TOTAL = float(os.environ.get("ABSENCE_BASELINE_TOTAL", "0.05"))
ABSENCE_BASELINE_ATTACK = float(os.environ.get("ABSENCE_BASELINE_ATTACK", "0.03"))
ABSENCE_BASELINE_DEFENSE = float(os.environ.get("ABSENCE_BASELINE_DEFENSE", "0.02"))
# 欠場影響の過補正防止
ABSENCE_IMPACT_CAP_TOTAL = float(os.environ.get("ABSENCE_IMPACT_CAP_TOTAL", "0.25"))
WEATHER_PENALTY_HEAVY_RAIN = 0.15
WEATHER_PENALTY_STRONG_WIND = 0.10
WEATHER_PENALTY_RAIN = 0.05
D_INTERCEPT = -1.2
D_SCALE = 1.5
DRAW_PROB_THRESHOLD = float(os.environ.get("DRAW_PROB_THRESHOLD", "0.24"))
DRAW_BALANCE_THRESHOLD = 0.10
HOME_ADV_ELO_COEF = float(os.environ.get("HOME_ADV_ELO_COEF", "60"))
HFA_ELO = float(os.environ.get("HFA_ELO", "35"))
HOME_ADV_PROFILE_DIFF_CLIP = float(os.environ.get("HOME_ADV_PROFILE_DIFF_CLIP", "0.8"))
# HFAは固定定数（デフォルト35）。試合固有バイアスは別スイッチで分離する。
ENABLE_MATCHUP_BIAS = _env_flag("ENABLE_MATCHUP_BIAS", 0)
MATCHUP_BIAS_COEF = float(os.environ.get("MATCHUP_BIAS_COEF", str(HOME_ADV_ELO_COEF)))
ELO_DIFF_TEMPERATURE = float(os.environ.get("ELO_DIFF_TEMPERATURE", "1.35"))
ELO_DIFF_SCALE = float(os.environ.get("ELO_DIFF_SCALE", "1.00"))
ELO_DRAW_BASE = float(os.environ.get("ELO_DRAW_BASE", "0.33"))
# base主導で調整できるよう、既定は0（必要時のみ環境変数で有効化）
ELO_DRAW_BUMP = float(os.environ.get("ELO_DRAW_BUMP", "0.00"))
ELO_DRAW_SENSITIVITY = float(os.environ.get("ELO_DRAW_SENSITIVITY", "400"))
ELO_DRAW_DIFF_SCALE = float(os.environ.get("ELO_DRAW_DIFF_SCALE", "1.00"))
ELO_DRAW_MIN = float(os.environ.get("ELO_DRAW_MIN", "0.10"))
ELO_DRAW_MAX = float(os.environ.get("ELO_DRAW_MAX", "0.38"))
DRAW_DECAY_SCALE = float(os.environ.get("DRAW_DECAY_SCALE", "320.0"))
# draw確率はPoisson由来とElo由来をブレンド（1.0=Poissonのみ, 0.0=Eloのみ）
DRAW_BLEND_WEIGHT = float(os.environ.get("DRAW_BLEND_WEIGHT", "0.60"))
DRAW_ASSIGN_BY_EXPECTATION = _env_flag("DRAW_ASSIGN_BY_EXPECTATION", 1)
# 期待ドロー件数の倍率（確率自体は変更せず、D割当件数のみ調整）
DRAW_EXPECTATION_MULTIPLIER = float(os.environ.get("DRAW_EXPECTATION_MULTIPLIER", "1.0"))
# Poisson格子の打ち切り誤差を抑えるための設定
POISSON_GRID_MIN_K = int(os.environ.get("POISSON_GRID_MIN_K", "10"))
POISSON_GRID_MAX_K = int(os.environ.get("POISSON_GRID_MAX_K", "20"))
POISSON_TAIL_EPS = float(os.environ.get("POISSON_TAIL_EPS", "1e-6"))
MISSING_WARN_THRESHOLD = float(os.environ.get("MISSING_WARN_THRESHOLD", "0.05"))
DEBUG_ELO_PROB = _env_flag("DEBUG_ELO_PROB", 0)
DEBUG_MATCH_ID = os.environ.get("DEBUG_MATCH_ID", "").strip()
J1_WIN_PROB_CAP = float(os.environ.get("J1_WIN_PROB_CAP", "0.68"))
PROB_FALLBACK = (0.397, 0.251, 0.353)
HFA_APPLY_COUNTER = {"applied": 0, "skipped": 0, "reason_counts": {}}
HDA_MODEL_MODE = os.environ.get("HDA_MODEL_MODE", "multinom").strip().lower()
if HDA_MODEL_MODE not in {"legacy", "multinom"}:
    print(f"[CONFIG][WARN] invalid HDA_MODEL_MODE={HDA_MODEL_MODE!r}; fallback='legacy'")
    HDA_MODEL_MODE = "legacy"
HDA_FEATURE_PROFILE = os.environ.get("HDA_FEATURE_PROFILE", "").strip()
_hda_model_default_profile = (
    os.path.join(DATA_DIR, "models", f"hda_multinom_train2025_{LEAGUE}__{HDA_FEATURE_PROFILE}.joblib")
    if HDA_FEATURE_PROFILE
    else ""
)
_hda_model_default_league = os.path.join(DATA_DIR, "models", f"hda_multinom_train2025_{LEAGUE}.joblib")
_hda_model_default_legacy = os.path.join(DATA_DIR, "models", "hda_multinom_train2025.joblib")
HDA_MODEL_PATH = os.environ.get("HDA_MODEL_PATH", "").strip() or (_hda_model_default_profile or _hda_model_default_league)
HDA_MODEL_BUNDLE = None
HDA_MODEL_MODE_EFFECTIVE = HDA_MODEL_MODE


def _softmax_rows(logits):
    arr = np.asarray(logits, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    z = arr - np.max(arr, axis=1, keepdims=True)
    ez = np.exp(z)
    den = np.sum(ez, axis=1, keepdims=True)
    den = np.where(den <= 0, 1.0, den)
    return ez / den


def _load_hda_model_bundle(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    required = {"type", "classes", "feature_names", "coef", "intercept", "feature_mean", "feature_std"}
    if not isinstance(bundle, dict) or not required.issubset(bundle.keys()):
        raise RuntimeError(f"invalid model bundle keys: required={sorted(required)}")
    if bundle.get("type") != "softmax_linear":
        raise RuntimeError(f"unsupported model bundle type: {bundle.get('type')!r}")
    classes = [str(c).upper() for c in bundle["classes"]]
    if not {"H", "D", "A"}.issubset(classes):
        raise RuntimeError(f"classes must include H/D/A but got {classes}")
    bundle["classes"] = classes
    bundle["feature_names"] = [str(c) for c in bundle["feature_names"]]
    bundle["coef"] = np.asarray(bundle["coef"], dtype=float)
    bundle["intercept"] = np.asarray(bundle["intercept"], dtype=float)
    bundle["feature_mean"] = np.asarray(bundle["feature_mean"], dtype=float)
    bundle["feature_std"] = np.asarray(bundle["feature_std"], dtype=float)
    return bundle


def _log_model_config():
    if HDA_MODEL_BUNDLE is None:
        print(
            f"[MODEL_CONFIG] league={LEAGUE} class_weight=unavailable "
            f"alpha=unavailable baseline_eps=unavailable"
        )
        return
    cw = HDA_MODEL_BUNDLE.get("class_weight", "unknown")
    alpha = HDA_MODEL_BUNDLE.get("class_weight_alpha", "unknown")
    baseline_eps = HDA_MODEL_BUNDLE.get("baseline_eps", "unknown")
    print(
        f"[MODEL_CONFIG] league={LEAGUE} class_weight={cw} "
        f"alpha={alpha} baseline_eps={baseline_eps}"
    )


def _init_hda_model():
    global HDA_MODEL_BUNDLE, HDA_MODEL_MODE_EFFECTIVE
    if HDA_MODEL_MODE != "multinom":
        HDA_MODEL_MODE_EFFECTIVE = "legacy"
        HDA_MODEL_BUNDLE = None
        _log_model_config()
        return
    model_candidates = []
    if HDA_MODEL_PATH:
        model_candidates.append(HDA_MODEL_PATH)
    if _hda_model_default_profile and _hda_model_default_profile not in model_candidates:
        model_candidates.append(_hda_model_default_profile)
    if _hda_model_default_league not in model_candidates:
        model_candidates.append(_hda_model_default_league)
    if _hda_model_default_legacy not in model_candidates:
        model_candidates.append(_hda_model_default_legacy)
    model_path = next((p for p in model_candidates if p and os.path.exists(p)), "")
    if not model_path or not os.path.exists(model_path):
        print(f"[CONFIG][WARN] HDA_MODEL_MODE=multinom ですがモデル未検出のため legacy にフォールバック: {HDA_MODEL_PATH}")
        HDA_MODEL_MODE_EFFECTIVE = "legacy"
        HDA_MODEL_BUNDLE = None
        _log_model_config()
        return
    try:
        HDA_MODEL_BUNDLE = _load_hda_model_bundle(model_path)
        globals()["HDA_MODEL_PATH"] = model_path
        HDA_MODEL_MODE_EFFECTIVE = "multinom"
        _log_model_config()
    except Exception as e:
        print(f"[CONFIG][WARN] multinomモデル読み込み失敗のため legacy にフォールバック: {e}")
        HDA_MODEL_MODE_EFFECTIVE = "legacy"
        HDA_MODEL_BUNDLE = None
        _log_model_config()


def _predict_hda_multinom_probs(elo_diff_for_prob):
    if HDA_MODEL_BUNDLE is None:
        raise RuntimeError("HDA_MODEL_BUNDLE is not loaded")
    feat_values = {
        "elo_diff_for_prob": float(elo_diff_for_prob),
        "abs_elo_diff_for_prob": float(abs(elo_diff_for_prob)),
        # multinomモードでは旧draw調整係数を使わず、生の差分由来のみを使う
        "d_scaled": float(abs(elo_diff_for_prob)),
        "abs_d_scaled": float(abs(elo_diff_for_prob)),
    }
    feature_names = HDA_MODEL_BUNDLE["feature_names"]
    x = np.array([feat_values.get(name, 0.0) for name in feature_names], dtype=float)
    mu = HDA_MODEL_BUNDLE["feature_mean"]
    sigma = HDA_MODEL_BUNDLE["feature_std"]
    sigma = np.where(np.abs(sigma) < 1e-12, 1.0, sigma)
    x_std = (x - mu) / sigma
    logits = x_std.dot(HDA_MODEL_BUNDLE["coef"].T) + HDA_MODEL_BUNDLE["intercept"]
    probs = _softmax_rows(logits)[0]
    cls_to_prob = {c: float(p) for c, p in zip(HDA_MODEL_BUNDLE["classes"], probs)}
    ph = float(cls_to_prob.get("H", 0.0))
    pdw = float(cls_to_prob.get("D", 0.0))
    pa = float(cls_to_prob.get("A", 0.0))
    return _normalize_probs(ph, pdw, pa), feat_values


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _short_col_list(cols, head=50, tail=20):
    cols = list(cols)
    if len(cols) <= head + tail:
        return cols
    return cols[:head] + ["..."] + cols[-tail:]


def _update_hfa_apply_counter(reason, applied):
    if applied:
        HFA_APPLY_COUNTER["applied"] += 1
    else:
        HFA_APPLY_COUNTER["skipped"] += 1
    rc = HFA_APPLY_COUNTER["reason_counts"]
    rc[reason] = int(rc.get(reason, 0)) + 1


def _detect_probability_columns(df_a, df_b):
    prob_candidates = [
        ("prob_home", "prob_draw", "prob_away"),
        ("prob_home_win", "prob_draw", "prob_away_win"),
        ("p_home", "p_draw", "p_away"),
    ]
    for c_home, c_draw, c_away in prob_candidates:
        if {c_home, c_draw, c_away}.issubset(df_a.columns) and {c_home, c_draw, c_away}.issubset(df_b.columns):
            return [c_home, c_draw, c_away]
    cols_a = _short_col_list(df_a.columns)
    cols_b = _short_col_list(df_b.columns)
    print(
        "[ERROR] probability columns not found. "
        f"available_columns_a={cols_a} available_columns_b={cols_b}"
    )
    raise RuntimeError("probability columns not found")


def _build_hfa_aligned_dataframe(df_a, df_b, value_cols):
    shared_cols = [c for c in value_cols if c in df_a.columns and c in df_b.columns]
    if not shared_cols:
        return pd.DataFrame(), [], "none"
    if "match_id" in df_a.columns and "match_id" in df_b.columns:
        merged = df_a[["match_id"] + shared_cols].merge(
            df_b[["match_id"] + shared_cols], on="match_id", how="inner", suffixes=("_a", "_b")
        )
        return merged, shared_cols, "match_id"
    key_cols = ["datetime", "home_team", "away_team"]
    if set(key_cols).issubset(df_a.columns) and set(key_cols).issubset(df_b.columns):
        merged = df_a[key_cols + shared_cols].merge(
            df_b[key_cols + shared_cols], on=key_cols, how="inner", suffixes=("_a", "_b")
        )
        return merged, shared_cols, "datetime+home+away"
    min_len = min(len(df_a), len(df_b))
    data = {}
    for col in shared_cols:
        data[f"{col}_a"] = df_a[col].head(min_len)
        data[f"{col}_b"] = df_b[col].head(min_len)
    return pd.DataFrame(data), shared_cols, "row_index"


def _compute_max_abs_diff(merged, col_name):
    left = f"{col_name}_a"
    right = f"{col_name}_b"
    if left not in merged.columns or right not in merged.columns:
        return None
    d = (pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(merged[right], errors="coerce")).abs()
    return float(d.max(skipna=True))


def _compute_diff_stats(merged, col_name):
    left = f"{col_name}_a"
    right = f"{col_name}_b"
    if left not in merged.columns or right not in merged.columns:
        return None, None
    d = (pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(merged[right], errors="coerce")).abs()
    max_abs = float(d.max(skipna=True))
    num_diff = int((d > 1e-12).sum())
    return max_abs, num_diff


def _log_hfa_intermediate_trace(df_a, df_b):
    compare_cols = [
        "elo_diff_for_prob",
        "elo_diff_scaled",
        "elo_diff",
        "elo_diff_raw",
        "d_scaled",
        "elo_diff_before_hfa",
        "elo_diff_after_hfa",
    ]
    merged, shared_cols, key_name = _build_hfa_aligned_dataframe(df_a, df_b, compare_cols)
    if merged.empty:
        print("[HFA_TRACE] key=none max_abs_diff=NA num_rows_with_any_diff=0")
        return

    primary = next(
        (c for c in ["elo_diff_for_prob", "elo_diff_scaled", "elo_diff", "elo_diff_raw"] if c in shared_cols),
        None,
    )
    if primary:
        max_abs, num_diff = _compute_diff_stats(merged, primary)
        print(
            f"[HFA_TRACE] key={primary} align={key_name} "
            f"max_abs_diff={max_abs:.6f} num_rows_with_any_diff={num_diff}"
        )
    else:
        print(f"[HFA_TRACE] key=none align={key_name} max_abs_diff=NA num_rows_with_any_diff=0")

    if "d_scaled" in shared_cols:
        max_abs, num_diff = _compute_diff_stats(merged, "d_scaled")
        print(
            f"[HFA_TRACE] key=d_scaled align={key_name} "
            f"max_abs_diff={max_abs:.6f} num_rows_with_any_diff={num_diff}"
        )

    if "elo_diff_before_hfa" in shared_cols and "elo_diff_after_hfa" in shared_cols:
        added_a = (
            pd.to_numeric(merged["elo_diff_after_hfa_a"], errors="coerce")
            - pd.to_numeric(merged["elo_diff_before_hfa_a"], errors="coerce")
        )
        added_b = (
            pd.to_numeric(merged["elo_diff_after_hfa_b"], errors="coerce")
            - pd.to_numeric(merged["elo_diff_before_hfa_b"], errors="coerce")
        )
        d_added = (added_a - added_b).abs()
        max_abs_added = float(d_added.max(skipna=True))
        max_abs_added_on = float(added_a.abs().max(skipna=True))
        max_abs_added_off = float(added_b.abs().max(skipna=True))
        num_added_diff = int((d_added > 1e-12).sum())
        print(
            f"[HFA_TRACE] max_abs_hfa_added={max_abs_added:.6f} "
            f"num_rows_with_any_diff={num_added_diff}"
        )
        print(
            f"[HFA_TRACE] max_abs_hfa_added_on={max_abs_added_on:.6f} "
            f"max_abs_hfa_added_off={max_abs_added_off:.6f}"
        )
        print(
            "[HFA_TRACE] formula=max_abs_hfa_added=max(abs("
            "(elo_diff_after_hfa-elo_diff_before_hfa)_ON - "
            "(elo_diff_after_hfa-elo_diff_before_hfa)_OFF))"
        )

    # Representative per-row trace: sort by |diff(elo_diff_for_prob)| desc and print top-N
    row_cols = [
        "elo_diff_before_hfa",
        "hfa_added_to_diff",
        "elo_diff_after_hfa",
        "elo_diff_for_prob",
        "d_scaled",
    ]
    if "match_id" in df_a.columns and "match_id" in df_b.columns:
        left_cols = ["match_id"] + [c for c in ["home_team", "away_team"] if c in df_a.columns] + [c for c in row_cols if c in df_a.columns]
        right_cols = ["match_id"] + [c for c in row_cols if c in df_b.columns]
        row_df = df_a[left_cols].merge(df_b[right_cols], on="match_id", how="inner", suffixes=("_a", "_b"))
        row_align = "match_id"
    elif set(["datetime", "home_team", "away_team"]).issubset(df_a.columns) and set(["datetime", "home_team", "away_team"]).issubset(df_b.columns):
        key_cols = ["datetime", "home_team", "away_team"]
        left_cols = key_cols + [c for c in row_cols if c in df_a.columns]
        right_cols = key_cols + [c for c in row_cols if c in df_b.columns]
        row_df = df_a[left_cols].merge(df_b[right_cols], on=key_cols, how="inner", suffixes=("_a", "_b"))
        row_align = "datetime+home+away"
    else:
        min_len = min(len(df_a), len(df_b))
        row_align = "row_index"
        row_df = pd.DataFrame({"row_index": np.arange(min_len)})
        if "home_team" in df_a.columns:
            row_df["home_team"] = df_a["home_team"].head(min_len).values
        if "away_team" in df_a.columns:
            row_df["away_team"] = df_a["away_team"].head(min_len).values
        for c in row_cols:
            if c in df_a.columns:
                row_df[f"{c}_a"] = df_a[c].head(min_len).values
            if c in df_b.columns:
                row_df[f"{c}_b"] = df_b[c].head(min_len).values

    if not row_df.empty:
        if "elo_diff_for_prob_a" in row_df.columns and "elo_diff_for_prob_b" in row_df.columns:
            row_df["__diff_elo_for_prob"] = (
                pd.to_numeric(row_df["elo_diff_for_prob_a"], errors="coerce")
                - pd.to_numeric(row_df["elo_diff_for_prob_b"], errors="coerce")
            )
        elif "elo_diff_used_for_prob_a" in row_df.columns and "elo_diff_used_for_prob_b" in row_df.columns:
            row_df["__diff_elo_for_prob"] = (
                pd.to_numeric(row_df["elo_diff_used_for_prob_a"], errors="coerce")
                - pd.to_numeric(row_df["elo_diff_used_for_prob_b"], errors="coerce")
            )
        else:
            row_df["__diff_elo_for_prob"] = 0.0
        if "d_scaled_a" in row_df.columns and "d_scaled_b" in row_df.columns:
            row_df["__diff_d_scaled"] = (
                pd.to_numeric(row_df["d_scaled_a"], errors="coerce")
                - pd.to_numeric(row_df["d_scaled_b"], errors="coerce")
            )
        else:
            row_df["__diff_d_scaled"] = np.nan
        top_rows = row_df.reindex(row_df["__diff_elo_for_prob"].abs().sort_values(ascending=False).index).head(int(HFA_TRACE_N))
        for _, r in top_rows.iterrows():
            match_id = str(r.get("match_id", r.get("row_index", "")))
            home = str(r.get("home_team", ""))
            away = str(r.get("away_team", ""))
            print(
                f"[HFA_TRACE_ROW] align={row_align} match_id={match_id} home={home} away={away} "
                f"elo_before={pd.to_numeric(r.get('elo_diff_before_hfa_a'), errors='coerce'):.4f} "
                f"hfa_added={pd.to_numeric(r.get('hfa_added_to_diff_a'), errors='coerce'):.4f} "
                f"elo_after={pd.to_numeric(r.get('elo_diff_after_hfa_a'), errors='coerce'):.4f} "
                f"elo_for_prob={pd.to_numeric(r.get('elo_diff_for_prob_a'), errors='coerce'):.4f} "
                f"d_scaled={pd.to_numeric(r.get('d_scaled_a'), errors='coerce'):.4f} "
                f"diff_elo_for_prob={pd.to_numeric(r.get('__diff_elo_for_prob'), errors='coerce'):.4f} "
                f"diff_d_scaled={pd.to_numeric(r.get('__diff_d_scaled'), errors='coerce'):.4f}"
            )


def _compare_hfa_probability_files(path_a, path_b, label_a="HFA_ON", label_b="HFA_OFF"):
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    _log_hfa_intermediate_trace(df_a, df_b)

    sha_a = _sha1_file(path_a)
    sha_b = _sha1_file(path_b)
    print(f"[HFA_SELF_CHECK:SHA1] {label_a}={sha_a}")
    print(f"[HFA_SELF_CHECK:SHA1] {label_b}={sha_b}")
    if sha_a == sha_b:
        print("[ERROR] HFA_ON and HFA_OFF outputs are identical (sha1 match)")
        raise RuntimeError("HFA self-check failed: identical outputs by sha1")

    intermediate_candidates = [
        "elo_diff_for_prob",
        "elo_diff",
        "elo_diff_scaled",
        "elo_diff_raw",
    ]
    aligned_intermediate, shared_intermediate, inter_key = _build_hfa_aligned_dataframe(
        df_a, df_b, intermediate_candidates + ["d_scaled"]
    )
    elo_col = next((c for c in intermediate_candidates if c in shared_intermediate), None)
    max_diff_elo = _compute_max_abs_diff(aligned_intermediate, elo_col) if elo_col else None
    max_diff_d_scaled = _compute_max_abs_diff(aligned_intermediate, "d_scaled") if "d_scaled" in shared_intermediate else None
    elo_part = (
        f"max_abs_diff_{elo_col}={max_diff_elo:.6f}" if (elo_col and max_diff_elo is not None)
        else "max_abs_diff_elo_diff_for_prob=NA"
    )
    d_scaled_part = (
        f"max_abs_diff_d_scaled={max_diff_d_scaled:.6f}" if max_diff_d_scaled is not None
        else "max_abs_diff_d_scaled=NA"
    )
    print(f"[HFA_COMPARE_INTERMEDIATE] key={inter_key} {elo_part} {d_scaled_part}")

    prob_cols = _detect_probability_columns(df_a, df_b)
    merged, _, prob_key = _build_hfa_aligned_dataframe(df_a, df_b, prob_cols)
    if merged.empty:
        raise RuntimeError("[ERROR] HFA compare merge returned 0 rows")
    home_col, draw_col, away_col = prob_cols
    d_home = (
        pd.to_numeric(merged[f"{home_col}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{home_col}_b"], errors="coerce")
    ).abs()
    d_draw = (
        pd.to_numeric(merged[f"{draw_col}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{draw_col}_b"], errors="coerce")
    ).abs()
    d_away = (
        pd.to_numeric(merged[f"{away_col}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{away_col}_b"], errors="coerce")
    ).abs()
    any_diff = (d_home > 1e-12) | (d_draw > 1e-12) | (d_away > 1e-12)
    num_rows_with_any_diff = int(any_diff.sum())
    print(
        f"[HFA_SELF_CHECK] key={prob_key} cols={prob_cols} "
        f"max_abs_diff_prob_home={float(d_home.max(skipna=True)):.6f} "
        f"max_abs_diff_prob_draw={float(d_draw.max(skipna=True)):.6f} "
        f"max_abs_diff_prob_away={float(d_away.max(skipna=True)):.6f} "
        f"num_rows_with_any_diff={num_rows_with_any_diff}"
    )
    if num_rows_with_any_diff == 0:
        print("[ERROR] HFA_ON and HFA_OFF prediction CSVs are identical (no probability differences detected)")
        raise RuntimeError("HFA self-check failed: no probability differences")


def _run_hfa_self_check_generation():
    args = [a for a in RAW_CLI_ARGS if a not in {"--self-check-hfa", "--skip-hfa-self-check"}]
    on_path = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions_hfa_on.csv")
    off_path = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions_hfa_off.csv")
    script_path = os.path.abspath(__file__)
    print("[SELF_CHECK] force recalculation active; cache reuse disabled")
    for mode, out_path, label in [(1, on_path, "HFA_ON"), (0, off_path, "HFA_OFF")]:
        env = os.environ.copy()
        env["ENABLE_HFA"] = str(mode)
        env["OUTPUT_PRED_CSV"] = out_path
        env["SKIP_HFA_SELF_CHECK"] = "1"
        cmd = [sys.executable, script_path] + args
        if "--force" not in cmd:
            cmd.append("--force")
        print(f"[HFA_SELF_CHECK] generating {label}: out={out_path}")
        subprocess.run(cmd, check=True, cwd=BASE_DIR, env=env)
        meta_path = f"{out_path}.meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                c = meta.get("hfa_apply_count", {})
                print(
                    f"[HFA_APPLY_COUNT] label={label} applied={int(c.get('applied', 0))} "
                    f"skipped={int(c.get('skipped', 0))} "
                    f"reason_counts={json.dumps(c.get('reason_counts', {}), ensure_ascii=False, sort_keys=True)}"
                )
            except Exception as e:
                print(f"[HFA_APPLY_COUNT][WARN] label={label} meta read failed: {e}")
        else:
            print(f"[HFA_APPLY_COUNT][WARN] label={label} meta not found: {meta_path}")
    _compare_hfa_probability_files(on_path, off_path, label_a="HFA_ON", label_b="HFA_OFF")


def log_run_config():
    print(
        "[CONFIG] "
        f"HDA_MODEL_MODE={HDA_MODEL_MODE} HDA_FEATURE_PROFILE={HDA_FEATURE_PROFILE or 'default'} "
        f"HDA_MODEL_EFFECTIVE={HDA_MODEL_MODE_EFFECTIVE} HDA_MODEL_PATH={HDA_MODEL_PATH} "
        f"ENABLE_HFA={ENABLE_HFA_INT} HFA_ELO={HFA_ELO:.2f} HFA_BASE_APPLIED={HFA_ELO if ENABLE_HFA else 0.0:.2f} "
        f"ENABLE_MATCHUP_BIAS={int(ENABLE_MATCHUP_BIAS)} MATCHUP_BIAS_COEF={MATCHUP_BIAS_COEF:.3f} "
        f"OUT={output_csv} FORCE={int(FORCE_RECALC)} "
        f"ELO_DIFF_SCALE={ELO_DIFF_SCALE:.3f} DRAW_DECAY_SCALE={DRAW_DECAY_SCALE:.1f} "
        f"DRAW_BLEND_WEIGHT={DRAW_BLEND_WEIGHT:.3f} ELO_DRAW_MIN={ELO_DRAW_MIN:.3f} "
        f"ELO_DRAW_MAX={ELO_DRAW_MAX:.3f} ELO_DRAW_BASE={ELO_DRAW_BASE:.3f} "
        f"ELO_DRAW_BUMP={ELO_DRAW_BUMP:.3f} ELO_DRAW_DIFF_SCALE={ELO_DRAW_DIFF_SCALE:.3f} "
        f"DRAW_ASSIGN_BY_EXPECTATION={int(DRAW_ASSIGN_BY_EXPECTATION)} "
        f"DRAW_EXPECTATION_MULTIPLIER={DRAW_EXPECTATION_MULTIPLIER:.3f}"
    )


def log_hfa_apply_path():
    print("[HFA_APPLY_PATH] active_path=compute_probabilities_and_result:elo_diff_for_prob (single source of HFA addition)")


_init_hda_model()
log_run_config()
log_hfa_apply_path()
if SELF_CHECK_HFA and (not SKIP_HFA_SELF_CHECK):
    _run_hfa_self_check_generation()
    sys.exit(0)

TEAM_NAME_ALIAS_RAW_MAP = {
    "G大阪": "G大阪",
    "ガンバ大阪": "G大阪",
    "C大阪": "C大阪",
    "セレッソ大阪": "C大阪",
    "横浜FM": "横浜FM",
    "横浜Fマリノス": "横浜FM",
    "横浜F・マリノス": "横浜FM",
    "横浜FC": "横浜FC",
    "FC東京": "FC東京",
    "FCTOKYO": "FC東京",
    "川崎F": "川崎F",
    "川崎フロンターレ": "川崎F",
    "東京V": "東京V",
    "東京ヴェルディ": "東京V",
    "湘南": "湘南",
    "湘南ベルマーレ": "湘南",
    "神戸": "神戸",
    "ヴィッセル神戸": "神戸",
    "名古屋": "名古屋",
    "名古屋グランパス": "名古屋",
    "浦和": "浦和",
    "浦和レッズ": "浦和",
    "広島": "広島",
    "サンフレッチェ広島": "広島",
    "福岡": "福岡",
    "アビスパ福岡": "福岡",
    "清水": "清水",
    "清水エスパルス": "清水",
    "新潟": "新潟",
    "アルビレックス新潟": "新潟",
    "千葉": "千葉",
    "ジェフユナイテッド千葉": "千葉",
    "鹿島": "鹿島",
    "鹿島アントラーズ": "鹿島",
    "柏": "柏",
    "柏レイソル": "柏",
    "水戸": "水戸",
    "水戸ホーリーホック": "水戸",
    "長崎": "長崎",
    "V・ファーレン長崎": "長崎",
    "Ｖ・ファーレン長崎": "長崎",
    "町田": "町田",
    "FC町田ゼルビア": "町田",
    "岡山": "岡山",
    "ファジアーノ岡山": "岡山",
    "京都": "京都",
    "京都サンガFC": "京都",
    "京都サンガF.C.": "京都",
    "鳥栖": "鳥栖",
    "サガン鳥栖": "鳥栖",
    "仙台": "仙台",
    "ベガルタ仙台": "仙台",
    "秋田": "秋田",
    "ブラウブリッツ秋田": "秋田",
    "山形": "山形",
    "モンテディオ山形": "山形",
    "いわき": "いわき",
    "いわきFC": "いわき",
    "いわきＦＣ": "いわき",
    "大宮": "大宮",
    "RB大宮アルディージャ": "大宮",
    "ＲＢ大宮アルディージャ": "大宮",
    "甲府": "甲府",
    "ヴァンフォーレ甲府": "甲府",
    "札幌": "札幌",
    "北海道コンサドーレ札幌": "札幌",
    "八戸": "八戸",
    "ヴァンラーレ八戸": "八戸",
    "磐田": "磐田",
    "ジュビロ磐田": "磐田",
    "藤枝": "藤枝",
    "藤枝MYFC": "藤枝",
    "藤枝ＭＹＦＣ": "藤枝",
    "栃木C": "栃木C",
    "栃木Ｃ": "栃木C",
    "栃木SC": "栃木C",
    "栃木ＳＣ": "栃木C",
    "栃木シティ": "栃木C",
    "富山": "富山",
    "カターレ富山": "富山",
    "今治": "今治",
    "FC今治": "今治",
    "ＦＣ今治": "今治",
    "徳島": "徳島",
    "徳島ヴォルティス": "徳島",
    "山口": "山口",
    "レノファ山口FC": "山口",
    "レノファ山口ＦＣ": "山口",
    "熊本": "熊本",
    "ロアッソ熊本": "熊本",
    "大分": "大分",
    "大分トリニータ": "大分",
    "宮崎": "宮崎",
    "テゲバジャーロ宮崎": "宮崎",
    "愛媛FC": "愛媛",
    "愛媛ＦＣ": "愛媛",
    "愛媛": "愛媛",
    "FC琉球": "琉球",
    "琉球": "琉球",
}


def _normalize_team_text(text):
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("　", " ").strip()
    s = s.replace("Ｆ", "F").replace("Ｃ", "C").replace("Ｖ", "V")
    s = s.upper()
    s = s.replace(" ", "").replace("・", "").replace(".", "")
    return s


TEAM_NAME_ALIAS_MAP = {
    _normalize_team_text(k): _normalize_team_text(v)
    for k, v in TEAM_NAME_ALIAS_RAW_MAP.items()
}

# J2(2026特別大会)では未公開が続くため、予測入力から除外するフィジカル系指標
J2_EXCLUDED_STATS_BASE_NAMES = [
    "1試合平均走行距離",
    "1試合平均スプリント回数",
    "1試合平均Atスプリント回数",
    "1試合平均Mtスプリント回数",
    "1試合平均Dtスプリント回数",
    "1試合平均ポゼッション時の走行距離",
    "1試合平均ポゼッション時のスプリント回数",
]

# 勝敗判定
def get_result(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return "H"
    elif home_score < away_score:
        return "A"
    else:
        return "D"

# Eloスコアから平均得点期待値を算出
def calculate_expected_goals(
    elo_diff,
    home_xg_stats=None,
    away_xg_stats=None,
    home_travel_distance=0,
    away_travel_distance=0,
    home_fatigue_score=None,
    away_fatigue_score=None,
    home_rank_motivation_score=None,
    away_rank_motivation_score=None,
    home_absence_impact=None,
    away_absence_impact=None,
    weather_flags=None,
):
    # 確率変換と同じ elo_diff を利用して期待得点を算出する
    temp = max(1e-6, float(ELO_DIFF_TEMPERATURE))
    adjusted_elo_diff = float(elo_diff) / temp
    elo_home_expected_goals = 1.5 + adjusted_elo_diff * GOAL_SCALING_FACTOR
    elo_away_expected_goals = 1.5 - adjusted_elo_diff * GOAL_SCALING_FACTOR
    
    # デフォルト値を設定
    home_hybrid_expected_goals = elo_home_expected_goals
    away_hybrid_expected_goals = elo_away_expected_goals

    # xGスタッツが利用可能であればハイブリッド評価を適用（NaNは未指定扱い）
    if pd.notna(home_xg_stats) and pd.notna(away_xg_stats):
        home_hybrid_expected_goals = (elo_home_expected_goals * 0.7) + (home_xg_stats * 0.3)
        away_hybrid_expected_goals = (elo_away_expected_goals * 0.7) + (away_xg_stats * 0.3)
    
    # 疲労度スコアがあれば控えめに期待値を補正（NaNは未指定扱い）
    if pd.notna(home_fatigue_score):
        home_hybrid_expected_goals -= home_fatigue_score * FATIGUE_GOAL_SCALING
    if pd.notna(away_fatigue_score):
        away_hybrid_expected_goals -= away_fatigue_score * FATIGUE_GOAL_SCALING

    # 順位推移由来のモチベーションを控えめに反映（NaNは未指定扱い）
    if pd.notna(home_rank_motivation_score):
        home_hybrid_expected_goals += home_rank_motivation_score * RANK_MOTIVATION_GOAL_SCALING
    if pd.notna(away_rank_motivation_score):
        away_hybrid_expected_goals += away_rank_motivation_score * RANK_MOTIVATION_GOAL_SCALING

    # 欠場影響（チーム内重みの合算）を控えめに減点
    if pd.notna(home_absence_impact):
        home_hybrid_expected_goals -= float(home_absence_impact) * ABSENCE_IMPACT_GOAL_SCALING
    if pd.notna(away_absence_impact):
        away_hybrid_expected_goals -= float(away_absence_impact) * ABSENCE_IMPACT_GOAL_SCALING

    # 天候フラグによる控えめな補正（両チームに同率で適用）
    if weather_flags:
        penalty = 0.0
        if weather_flags.get("is_heavy_rain"):
            penalty += WEATHER_PENALTY_HEAVY_RAIN
        elif weather_flags.get("is_rain"):
            penalty += WEATHER_PENALTY_RAIN
        if weather_flags.get("is_strong_wind"):
            penalty += WEATHER_PENALTY_STRONG_WIND
        if penalty > 0:
            home_hybrid_expected_goals -= penalty
            away_hybrid_expected_goals -= penalty
    
    # 負の得点期待値にならないように調整
    home_hybrid_expected_goals = max(0.1, home_hybrid_expected_goals)
    away_hybrid_expected_goals = max(0.1, away_hybrid_expected_goals)
    
    return home_hybrid_expected_goals, away_hybrid_expected_goals


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def apply_draw_separation(prob_home_win, prob_draw, prob_away_win):
    # Poisson由来のraw draw確率を分離補正し、D過大を抑える。
    # Home/Awayの比率はPoissonの相対関係をそのまま保持する。
    pD_poisson = prob_draw
    pD = _sigmoid(D_INTERCEPT + D_SCALE * pD_poisson)

    ha_sum = prob_home_win + prob_away_win
    if ha_sum > 0:
        pH = (1.0 - pD) * (prob_home_win / ha_sum)
        pA = (1.0 - pD) * (prob_away_win / ha_sum)
    else:
        pH = (1.0 - pD) * 0.5
        pA = (1.0 - pD) * 0.5

    # 1試合分の計算例:
    # before(Poisson正規化後): pH=0.24, pD_poisson=0.53, pA=0.23
    # pD=sigmoid(-1.2 + 1.5*0.53)=0.400
    # after: pH=(1-0.400)*(0.24/0.47)=0.306, pA=(1-0.400)*(0.23/0.47)=0.294
    # sum=1.000
    return pH, pD, pA


def _resolve_draw_sensitivity(sensitivity):
    # 旧設定(<=1)は互換のため逆数変換して実効スケールを確保する。
    # 例: 0.0002 -> 5000
    s = float(sensitivity)
    if s <= 0:
        return 120.0
    if s <= 1.0:
        return 1.0 / s
    return s


def calibrate_draw_probability(prob_home_win, prob_draw, prob_away_win, elo_diff):
    # drawは「Eloで上書き」せず、Poisson由来とElo由来をブレンドする
    d_scaled = abs(float(elo_diff)) * float(ELO_DRAW_DIFF_SCALE)
    decay_scale = float(DRAW_DECAY_SCALE) if float(DRAW_DECAY_SCALE) > 0 else 120.0
    p_draw_elo = float(ELO_DRAW_MIN) + (float(ELO_DRAW_MAX) - float(ELO_DRAW_MIN)) * float(
        np.exp(-d_scaled / decay_scale)
    )
    p_draw_elo = float(np.clip(p_draw_elo, ELO_DRAW_MIN, ELO_DRAW_MAX))
    p_draw_poi = float(np.clip(float(prob_draw), 0.0, 1.0))
    blend_w = float(np.clip(DRAW_BLEND_WEIGHT, 0.0, 1.0))
    p_draw_raw = float(np.clip((blend_w * p_draw_poi) + ((1.0 - blend_w) * p_draw_elo), 0.0, 1.0))

    ha_sum = float(prob_home_win) + float(prob_away_win)
    if ha_sum > 0:
        p_home = (1.0 - p_draw_raw) * (float(prob_home_win) / ha_sum)
        p_away = (1.0 - p_draw_raw) * (float(prob_away_win) / ha_sum)
    else:
        p_home = (1.0 - p_draw_raw) * 0.5
        p_away = (1.0 - p_draw_raw) * 0.5

    p_home, p_draw_raw, p_away = _normalize_probs(p_home, p_draw_raw, p_away)
    return p_home, p_draw_raw, p_away, d_scaled, p_draw_poi, p_draw_elo


def build_elo_context(
    home_elo,
    away_elo,
    home_advantage_diff,
    stats_home_missing=False,
    stats_away_missing=False,
    data_quality_warn=False,
):
    profile_diff_raw = float(home_advantage_diff)
    profile_diff_clipped = float(np.clip(profile_diff_raw, -HOME_ADV_PROFILE_DIFF_CLIP, HOME_ADV_PROFILE_DIFF_CLIP))
    matchup_bias = 0.0
    if ENABLE_MATCHUP_BIAS:
        matchup_bias = float(profile_diff_clipped) * float(MATCHUP_BIAS_COEF)
    elo_diff_before_hfa = float(home_elo) - float(away_elo) + float(matchup_bias)
    base_hfa = float(HFA_ELO) if ENABLE_HFA else 0.0
    hfa_mult = 1.0
    applied_hfa = float(base_hfa)
    elo_diff_raw = float(elo_diff_before_hfa) + float(applied_hfa)
    elo_diff = float(elo_diff_raw) * float(ELO_DIFF_SCALE)
    expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    return {
        "hfa_enabled": bool(ENABLE_HFA),
        "matchup_bias_enabled": bool(ENABLE_MATCHUP_BIAS),
        "matchup_bias_coef": float(MATCHUP_BIAS_COEF),
        "matchup_bias": float(matchup_bias),
        "home_advantage_profile_diff_raw": profile_diff_raw,
        "home_advantage_profile_diff_clipped": profile_diff_clipped,
        "elo_diff_before_hfa": float(elo_diff_before_hfa),
        "elo_diff_after_hfa": float(elo_diff_raw),
        "base_hfa": float(base_hfa),
        "hfa_mult": float(hfa_mult),
        "applied_hfa": float(applied_hfa),
        "elo_diff_raw": float(elo_diff_raw),
        "elo_diff_scaled": float(elo_diff),
        "elo_diff": float(elo_diff),
        "expected_home": float(expected_home),
    }


def log_prob_summary(df, label):
    required = {"prob_draw", "predicted_result"}
    if not required.issubset(df.columns):
        return
    rows = int(len(df))
    if rows == 0:
        print(f"[{label}] rows=0 ELO_DIFF_SCALE={ELO_DIFF_SCALE:.2f}")
        return
    avg_draw = float(df["prob_draw"].mean())
    sum_draw = float(df["prob_draw"].sum())
    d_count = int((df["predicted_result"].astype(str) == "D").sum())
    actual_d_rate = None
    if "actual_result" in df.columns:
        actual = df["actual_result"].astype(str).str.upper()
        valid = actual.isin(["H", "D", "A"])
        if int(valid.sum()) > 0:
            actual_d_rate = float((actual[valid] == "D").mean())
    elif {"home_score", "away_score"}.issubset(df.columns):
        hs = pd.to_numeric(df["home_score"], errors="coerce")
        aw = pd.to_numeric(df["away_score"], errors="coerce")
        valid = hs.notna() & aw.notna()
        if int(valid.sum()) > 0:
            actual_d_rate = float((hs[valid] == aw[valid]).mean())
    draw_diff_text = ""
    if actual_d_rate is not None:
        draw_diff_pp = (avg_draw - float(actual_d_rate)) * 100.0
        draw_diff_text = f" actual_D_rate={actual_d_rate:.3f} draw_diff_pp={draw_diff_pp:.2f}"
    if HDA_MODEL_MODE_EFFECTIVE == "multinom":
        print(
            f"[{label}] rows={rows} avg_prob_draw={avg_draw:.3f} "
            f"sum_prob_draw={sum_draw:.3f} predicted_D_count={d_count} "
            f"ELO_DIFF_SCALE={ELO_DIFF_SCALE:.2f}{draw_diff_text} "
            "legacy_draw_adjustment=disabled"
        )
    else:
        print(
            f"[{label}] rows={rows} avg_prob_draw={avg_draw:.3f} "
            f"sum_prob_draw={sum_draw:.3f} predicted_D_count={d_count} "
            f"ELO_DIFF_SCALE={ELO_DIFF_SCALE:.2f}{draw_diff_text} "
            f"DRAW_DECAY_SCALE={DRAW_DECAY_SCALE:.1f} "
            f"ELO_DRAW_DIFF_SCALE={ELO_DRAW_DIFF_SCALE:.3f} "
            f"ELO_DRAW_MIN={ELO_DRAW_MIN:.3f} ELO_DRAW_MAX={ELO_DRAW_MAX:.3f} "
            f"DRAW_BLEND_WEIGHT={DRAW_BLEND_WEIGHT:.3f}"
        )


def log_prob_draw_distribution(df, label):
    if "prob_draw" not in df.columns or df.empty:
        print(f"[PROB_DRAW_DIST:{label}] unavailable")
        return
    s = pd.to_numeric(df["prob_draw"], errors="coerce").dropna()
    if s.empty:
        print(f"[PROB_DRAW_DIST:{label}] unavailable")
        return
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"[PROB_DRAW_DIST:{label}] rows={len(s)} min={float(s.min()):.3f} "
        f"p05={float(q.loc[0.05]):.3f} p25={float(q.loc[0.25]):.3f} "
        f"p50={float(q.loc[0.5]):.3f} p75={float(q.loc[0.75]):.3f} "
        f"p95={float(q.loc[0.95]):.3f} max={float(s.max()):.3f}"
    )


def log_prob_distribution(df, label, col):
    if col not in df.columns or df.empty:
        print(f"[PROB_DIST:{label}] col={col} unavailable")
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        print(f"[PROB_DIST:{label}] col={col} unavailable")
        return
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"[PROB_DIST:{label}] col={col} rows={len(s)} "
        f"min={float(s.min()):.3f} p05={float(q.loc[0.05]):.3f} p25={float(q.loc[0.25]):.3f} "
        f"p50={float(q.loc[0.5]):.3f} p75={float(q.loc[0.75]):.3f} p95={float(q.loc[0.95]):.3f} "
        f"max={float(s.max()):.3f}"
    )


def log_max_prob_distribution(df, label):
    candidates = [
        ("prob_home", "prob_draw", "prob_away"),
        ("prob_home_win", "prob_draw", "prob_away_win"),
    ]
    cols = None
    for c in candidates:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None or df.empty:
        print(f"[MAX_PROB_DIST:{label}] unavailable")
        return
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    mx = pd.concat([ph, pdw, pa], axis=1).max(axis=1).dropna()
    if mx.empty:
        print(f"[MAX_PROB_DIST:{label}] unavailable")
        return
    q = mx.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"[MAX_PROB_DIST:{label}] rows={len(mx)} "
        f"min={float(mx.min()):.3f} p05={float(q.loc[0.05]):.3f} p25={float(q.loc[0.25]):.3f} "
        f"p50={float(q.loc[0.5]):.3f} p75={float(q.loc[0.75]):.3f} p95={float(q.loc[0.95]):.3f} "
        f"max={float(mx.max()):.3f}"
    )


def _feature_series_for_name(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    if name == "abs_elo_diff_for_prob" and "elo_diff_for_prob" in df.columns:
        return pd.to_numeric(df["elo_diff_for_prob"], errors="coerce").abs()
    if name in {"d_scaled", "abs_d_scaled"} and "elo_diff_for_prob" in df.columns:
        base = pd.to_numeric(df["elo_diff_for_prob"], errors="coerce").abs()
        return base
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")


def log_multinom_feature_distribution(df, label):
    if HDA_MODEL_MODE_EFFECTIVE != "multinom" or HDA_MODEL_BUNDLE is None:
        return
    if df is None or df.empty:
        print(f"[FEATURE_DIST:{label}] unavailable")
        return
    names = list(HDA_MODEL_BUNDLE.get("feature_names", []))
    mu = np.asarray(HDA_MODEL_BUNDLE.get("feature_mean", []), dtype=float)
    sigma = np.asarray(HDA_MODEL_BUNDLE.get("feature_std", []), dtype=float)
    for i, name in enumerate(names):
        s = _feature_series_for_name(df, name)
        rows = int(len(s))
        missing = int(s.isna().sum())
        valid = s.dropna()
        if valid.empty:
            print(f"[FEATURE_DIST:{label}] col={name} rows={rows} missing={missing} unique=0 unavailable")
            continue
        q = valid.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(
            f"[FEATURE_DIST:{label}] col={name} rows={rows} missing={missing} unique={int(valid.nunique())} "
            f"min={float(valid.min()):.3f} p05={float(q.loc[0.05]):.3f} p25={float(q.loc[0.25]):.3f} "
            f"p50={float(q.loc[0.5]):.3f} p75={float(q.loc[0.75]):.3f} p95={float(q.loc[0.95]):.3f} "
            f"max={float(valid.max()):.3f}"
        )
        if i < len(mu) and i < len(sigma):
            den = 1.0 if abs(float(sigma[i])) < 1e-12 else float(sigma[i])
            s_std = (valid - float(mu[i])) / den
            q2 = s_std.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            print(
                f"[FEATURE_DIST_STD:{label}] col={name} rows={int(len(s_std))} "
                f"min={float(s_std.min()):.3f} p05={float(q2.loc[0.05]):.3f} p25={float(q2.loc[0.25]):.3f} "
                f"p50={float(q2.loc[0.5]):.3f} p75={float(q2.loc[0.75]):.3f} p95={float(q2.loc[0.95]):.3f} "
                f"max={float(s_std.max()):.3f}"
            )


def log_actual_hda_ratio(df, label):
    if df is None or df.empty:
        print(f"[ACTUAL_HDA:{label}] unavailable")
        return
    if "actual_result" in df.columns:
        actual = df["actual_result"].astype(str).str.upper()
    elif {"home_score", "away_score"}.issubset(df.columns):
        hs = pd.to_numeric(df["home_score"], errors="coerce")
        aw = pd.to_numeric(df["away_score"], errors="coerce")
        actual = pd.Series(np.where(hs > aw, "H", np.where(hs < aw, "A", "D")), index=df.index)
        actual = actual.where(hs.notna() & aw.notna(), pd.NA).astype("object")
    else:
        print(f"[ACTUAL_HDA:{label}] unavailable")
        return
    actual = actual[actual.isin(["H", "D", "A"])]
    if actual.empty:
        print(f"[ACTUAL_HDA:{label}] unavailable")
        return
    total = int(len(actual))
    h = int((actual == "H").sum())
    d = int((actual == "D").sum())
    a = int((actual == "A").sum())
    print(
        f"[ACTUAL_HDA:{label}] rows={total} H={100.0*h/total:.1f}% ({h}) "
        f"D={100.0*d/total:.1f}% ({d}) A={100.0*a/total:.1f}% ({a})"
    )


def _calc_hda_dist_from_series(series):
    s = pd.Series(series, dtype="object").astype(str).str.upper()
    s = s[s.isin(["H", "D", "A"])]
    n = int(len(s))
    h = int((s == "H").sum())
    d = int((s == "D").sum())
    a = int((s == "A").sum())
    hp = (100.0 * h / n) if n > 0 else 0.0
    dp = (100.0 * d / n) if n > 0 else 0.0
    ap = (100.0 * a / n) if n > 0 else 0.0
    return {"rows": n, "H_cnt": h, "D_cnt": d, "A_cnt": a, "H_pct": hp, "D_pct": dp, "A_pct": ap}


def log_pred_dist(df, label, scope="all"):
    if df is None or df.empty:
        print(f"[PRED_DIST:{label}] scope={scope} unavailable")
        return
    col = "final_result" if "final_result" in df.columns else "predicted_result"
    if col not in df.columns:
        print(f"[PRED_DIST:{label}] scope={scope} unavailable")
        return
    dist = _calc_hda_dist_from_series(df[col])
    if dist["rows"] <= 0:
        print(f"[PRED_DIST:{label}] scope={scope} unavailable")
        return
    print(
        f"[PRED_DIST:{label}] scope={scope} rows={dist['rows']} "
        f"H={dist['H_pct']:.1f}% ({dist['H_cnt']}) "
        f"D={dist['D_pct']:.1f}% ({dist['D_cnt']}) "
        f"A={dist['A_pct']:.1f}% ({dist['A_cnt']})"
    )


def log_draw_argmax_stats(df, label, threshold=0.23):
    candidates = [
        ("prob_home", "prob_draw", "prob_away"),
        ("prob_home_win", "prob_draw", "prob_away_win"),
    ]
    cols = None
    for c in candidates:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None or df.empty:
        print(f"[DRAW_ARGMAX:{label}] unavailable")
        return
    ch, cd, ca = cols
    ph = pd.to_numeric(df[ch], errors="coerce")
    pdw = pd.to_numeric(df[cd], errors="coerce")
    pa = pd.to_numeric(df[ca], errors="coerce")
    valid = ph.notna() & pdw.notna() & pa.notna()
    if int(valid.sum()) == 0:
        print(f"[DRAW_ARGMAX:{label}] unavailable")
        return
    cnt_argmax = int(((pdw >= ph) & (pdw >= pa) & valid).sum())
    cnt_threshold = int(((pdw >= float(threshold)) & valid).sum())
    print(
        f"[DRAW_ARGMAX:{label}] rows={int(valid.sum())} "
        f"prob_draw_argmax_count={cnt_argmax} prob_draw_ge_{threshold:.2f}_count={cnt_threshold}"
    )


def _parse_round_no_env(value):
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"([0-9]+)", unicodedata.normalize("NFKC", s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _resolve_round_filter(df):
    if df is None or df.empty:
        return pd.Series([], dtype=bool), "empty_df", True

    if TOTO_ROUND_ID:
        if "toto_round_id" in df.columns:
            mask = df["toto_round_id"].astype(str).str.strip() == TOTO_ROUND_ID
            return mask, f"toto_round_id={TOTO_ROUND_ID}", False
        print(f"[WARN] TOTO_ROUND_IDが指定されていますが 'toto_round_id' 列がありません: {TOTO_ROUND_ID}")

    round_no = _parse_round_no_env(ROUND_NO_ENV)
    if round_no is not None:
        if "節" in df.columns:
            mask = df["節"].map(extract_round_number).astype("Int64") == int(round_no)
            return mask.fillna(False), f"round_no={int(round_no)} (from 節)", False
        if "round" in df.columns:
            mask = df["round"].map(extract_round_number).astype("Int64") == int(round_no)
            return mask.fillna(False), f"round_no={int(round_no)} (from round)", False
        print(f"[WARN] ROUND_NOが指定されていますが '節'/'round' 列がありません: {ROUND_NO_ENV}")

    print("[WARN] フィルタ未指定のため全件集計")
    return pd.Series([True] * len(df), index=df.index), "ALL", True


def _load_toto_targets():
    if not os.path.exists(TOTO_ORDER_CSV):
        return pd.DataFrame()
    try:
        src = pd.read_csv(TOTO_ORDER_CSV, header=None, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] toto並び順CSVの読み込み失敗: {TOTO_ORDER_CSV} ({e})")
        return pd.DataFrame()
    if src.empty:
        return pd.DataFrame()
    # 想定: col0=match_no, col1=home_team, col2='vs', col3=away_team
    if src.shape[1] < 4:
        print(f"[WARN] toto並び順CSVの列数が不足: {TOTO_ORDER_CSV}")
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "match_no": pd.to_numeric(src.iloc[:, 0], errors="coerce"),
            "home_team": src.iloc[:, 1].astype(str),
            "away_team": src.iloc[:, 3].astype(str),
        }
    )
    out = out.dropna(subset=["match_no"]).copy()
    out["match_no"] = out["match_no"].astype(int)
    out["_home_key"] = normalize_team_series(out["home_team"])
    out["_away_key"] = normalize_team_series(out["away_team"])
    out = out.dropna(subset=["_home_key", "_away_key"])
    out["_pair_key"] = out["_home_key"].astype(str) + "||" + out["_away_key"].astype(str)
    out = out.drop_duplicates(subset=["_pair_key"], keep="first")
    return out


def _resolve_toto_target_filter(df):
    if df is None or df.empty:
        return None, None
    targets = _load_toto_targets()
    if targets.empty:
        return None, None
    if not {"home_team", "away_team"}.issubset(df.columns):
        print("[WARN] toto並び順フィルタを適用できません（home_team/away_team列不足）")
        return None, None
    work = df.copy()
    work["_home_key"] = normalize_team_series(work["home_team"])
    work["_away_key"] = normalize_team_series(work["away_team"])
    work["_pair_key"] = work["_home_key"].astype(str) + "||" + work["_away_key"].astype(str)
    target_keys = set(targets["_pair_key"].astype(str).tolist())
    mask = work["_pair_key"].isin(target_keys)
    return mask, f"toto_order_csv={os.path.basename(TOTO_ORDER_CSV)}"


def calc_hda_ratio(series_of_HDA) -> dict:
    if series_of_HDA is None:
        return {
            "H": {"count": 0, "pct": 0.0},
            "D": {"count": 0, "pct": 0.0},
            "A": {"count": 0, "pct": 0.0},
            "total": 0,
        }
    s = pd.Series(series_of_HDA).astype(str).str.upper().str.strip()
    valid = s[s.isin(["H", "D", "A"])]
    total = int(len(valid))
    h = int((valid == "H").sum())
    d = int((valid == "D").sum())
    a = int((valid == "A").sum())
    denom = total if total > 0 else 1
    return {
        "H": {"count": h, "pct": (h * 100.0 / denom) if total > 0 else 0.0},
        "D": {"count": d, "pct": (d * 100.0 / denom) if total > 0 else 0.0},
        "A": {"count": a, "pct": (a * 100.0 / denom) if total > 0 else 0.0},
        "total": total,
    }


def _enrich_scores_from_results(df_pred_filtered, df_results):
    if df_results is None or df_results.empty or df_pred_filtered.empty:
        return df_pred_filtered

    out = df_pred_filtered.copy()
    res = df_results.copy()
    for col in ["home_score", "away_score"]:
        if col not in out.columns:
            out[col] = pd.NA

    # 1) match_id があれば最優先で突合
    if "match_id" in out.columns and "match_id" in res.columns:
        right = res[["match_id", "home_score", "away_score"]].copy()
        right = right.dropna(subset=["match_id"]).drop_duplicates(subset=["match_id"], keep="last")
        merged = out.merge(right, on="match_id", how="left", suffixes=("", "__res"))
        for col in ["home_score", "away_score"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(
                pd.to_numeric(merged[f"{col}__res"], errors="coerce")
            )
        return merged.drop(columns=["home_score__res", "away_score__res"], errors="ignore")

    # 2) datetime + home_team + away_team
    key_cols = {"datetime", "home_team", "away_team"}
    if key_cols.issubset(set(out.columns)) and key_cols.issubset(set(res.columns)):
        left = out.copy()
        right = res.copy()
        left["_dt_key"] = pd.to_datetime(left["datetime"], errors="coerce")
        right["_dt_key"] = pd.to_datetime(right["datetime"], errors="coerce")
        left["_home_key"] = normalize_team_series(left["home_team"]) if "normalize_team_series" in globals() else left["home_team"].astype(str)
        left["_away_key"] = normalize_team_series(left["away_team"]) if "normalize_team_series" in globals() else left["away_team"].astype(str)
        right["_home_key"] = normalize_team_series(right["home_team"]) if "normalize_team_series" in globals() else right["home_team"].astype(str)
        right["_away_key"] = normalize_team_series(right["away_team"]) if "normalize_team_series" in globals() else right["away_team"].astype(str)
        right = right[["_dt_key", "_home_key", "_away_key", "home_score", "away_score"]].drop_duplicates(
            subset=["_dt_key", "_home_key", "_away_key"], keep="last"
        )
        merged = left.merge(right, on=["_dt_key", "_home_key", "_away_key"], how="left", suffixes=("", "__res"))
        for col in ["home_score", "away_score"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(
                pd.to_numeric(merged[f"{col}__res"], errors="coerce")
            )
        return merged.drop(
            columns=["home_score__res", "away_score__res", "_dt_key", "_home_key", "_away_key"],
            errors="ignore",
        )

    return out


def _write_round_summary_csv(filter_label, pred_ratio, actual_ratio):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(filter_label)).strip("._")
    if not safe:
        safe = "all"
    out_path = os.path.join(MERGE_QC_DIR, f"round_summary_{safe}.csv")
    os.makedirs(MERGE_QC_DIR, exist_ok=True)
    rows = [
        {
            "kind": "pred",
            "H_cnt": pred_ratio["H"]["count"],
            "D_cnt": pred_ratio["D"]["count"],
            "A_cnt": pred_ratio["A"]["count"],
            "total": pred_ratio["total"],
            "H_pct": pred_ratio["H"]["pct"],
            "D_pct": pred_ratio["D"]["pct"],
            "A_pct": pred_ratio["A"]["pct"],
            "filter": filter_label,
        }
    ]
    if actual_ratio is not None:
        rows.append(
            {
                "kind": "actual",
                "H_cnt": actual_ratio["H"]["count"],
                "D_cnt": actual_ratio["D"]["count"],
                "A_cnt": actual_ratio["A"]["count"],
                "total": actual_ratio["total"],
                "H_pct": actual_ratio["H"]["pct"],
                "D_pct": actual_ratio["D"]["pct"],
                "A_pct": actual_ratio["A"]["pct"],
                "filter": filter_label,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[ROUND_SUMMARY_CSV] saved={out_path}")


def summarize_round_hda(df_pred, df_results=None, round_filter_label="auto"):
    if df_pred is None or df_pred.empty:
        print("[ROUND_SUMMARY] filter=empty rows=0")
        print("[PRED_RATIO] H=0.0% (0) D=0.0% (0) A=0.0% (0)")
        print("[ACTUAL_RATIO] unavailable (scores not found or not finished)")
        return

    mask, filter_label = _resolve_toto_target_filter(df_pred)
    if mask is None:
        mask, filter_label, _ = _resolve_round_filter(df_pred)
    pred_filtered = df_pred.loc[mask].copy()
    rows = int(len(pred_filtered))
    if filter_label.startswith("toto_order_csv=") and rows != 13:
        print(f"[WARN] toto対象の抽出件数が13ではありません: rows={rows} (LEAGUE={LEAGUE})")
    print(f"[ROUND_SUMMARY] filter={filter_label} rows={rows}")

    pred_ratio = calc_hda_ratio(pred_filtered.get("predicted_result", pd.Series(dtype="object")))
    print(
        f"[PRED_RATIO] H={pred_ratio['H']['pct']:.1f}% ({pred_ratio['H']['count']}) "
        f"D={pred_ratio['D']['pct']:.1f}% ({pred_ratio['D']['count']}) "
        f"A={pred_ratio['A']['pct']:.1f}% ({pred_ratio['A']['count']})"
    )

    if rows == 0:
        print("[ACTUAL_RATIO] unavailable (scores not found or not finished)")
        _write_round_summary_csv(filter_label, pred_ratio, None)
        return

    actual_source = _enrich_scores_from_results(pred_filtered, df_results)
    hs = pd.to_numeric(actual_source.get("home_score"), errors="coerce")
    aw = pd.to_numeric(actual_source.get("away_score"), errors="coerce")
    actual_result = pd.Series(
        [get_result(h, a) for h, a in zip(hs.tolist(), aw.tolist())],
        index=actual_source.index,
        dtype="object",
    )
    resolved_count = int(actual_result.notna().sum())
    if resolved_count != rows:
        print("[ACTUAL_RATIO] unavailable (scores not found or not finished)")
        _write_round_summary_csv(filter_label, pred_ratio, None)
        return

    actual_ratio = calc_hda_ratio(actual_result)
    print(
        f"[ACTUAL_RATIO] H={actual_ratio['H']['pct']:.1f}% ({actual_ratio['H']['count']}) "
        f"D={actual_ratio['D']['pct']:.1f}% ({actual_ratio['D']['count']}) "
        f"A={actual_ratio['A']['pct']:.1f}% ({actual_ratio['A']['count']})"
    )
    _write_round_summary_csv(filter_label, pred_ratio, actual_ratio)


def log_decision_rule_once():
    if getattr(log_decision_rule_once, "_done", False):
        return
    if DRAW_ASSIGN_BY_EXPECTATION:
        rule_desc = "FORCE_DRAW_BY_EXPECTATION_ASSIGN > ARGMAX"
    else:
        rule_desc = "ARGMAX"
    print(f"[DECISION_RULE] {rule_desc}")
    log_decision_rule_once._done = True


def dump_decision_artifacts(df, label="pred", threshold=0.25):
    if not DUMP_DECISION:
        return
    required = {"prob_home_win", "prob_draw", "prob_away_win", "predicted_result"}
    if not required.issubset(df.columns):
        print(f"[DUMP_DECISION][WARN] required_columns_missing label={label} need={sorted(required)}")
        return

    os.makedirs(MERGE_QC_DIR, exist_ok=True)
    work = df.copy()
    work["prob_home_win"] = pd.to_numeric(work["prob_home_win"], errors="coerce")
    work["prob_draw"] = pd.to_numeric(work["prob_draw"], errors="coerce")
    work["prob_away_win"] = pd.to_numeric(work["prob_away_win"], errors="coerce")
    work["draw_gap"] = work[["prob_home_win", "prob_away_win"]].max(axis=1) - work["prob_draw"]

    keep_cols = [c for c in ["match_id", "match_no", "league", "節", "home_team", "away_team"] if c in work.columns]
    keep_cols += ["prob_home_win", "prob_draw", "prob_away_win", "predicted_result", "draw_gap"]
    for c in [
        "decision_reason",
        "argmax_result",
        "argmax_raw_result",
        "argmax_max_prob",
        "argmax_raw_max_prob",
        "d_scaled",
        "elo_diff_for_prob",
        "decision_draw_expectation_multiplier",
        "decision_draw_assign_enabled",
    ]:
        if c in work.columns:
            keep_cols.append(c)

    full_csv = os.path.join(MERGE_QC_DIR, f"decision_scores_{label}.csv")
    work[keep_cols].to_csv(full_csv, index=False, encoding="utf-8-sig")

    cond = (work["prob_draw"] >= float(threshold)) & (work["predicted_result"].astype(str).str.upper() != "D")
    top50 = work.loc[cond, keep_cols].sort_values(["draw_gap", "prob_draw"], ascending=[True, False]).head(50)
    top50_csv = os.path.join(MERGE_QC_DIR, f"decision_draw_candidates_{label}_top50.csv")
    top50.to_csv(top50_csv, index=False, encoding="utf-8-sig")
    print(
        f"[DUMP_DECISION] label={label} total={len(work)} "
        f"cond=(prob_draw>={threshold} and predicted_result!=D) matched={int(cond.sum())} "
        f"saved_full={full_csv} saved_top50={top50_csv}"
    )


def compute_row_quality_flags(row):
    stats_home_missing = pd.isna(row.get("stats_ゴール期待値_home"))
    stats_away_missing = pd.isna(row.get("stats_ゴール期待値_away"))
    mgmt_home_col = _pick_first_non_na_value(
        row,
        [
            "management_recent_injuries_suspensions_count_home",
            "management_recent_injuries_suspensions_count",
        ],
    )
    mgmt_away_col = _pick_first_non_na_value(row, ["management_recent_injuries_suspensions_count_away"])
    management_missing = pd.isna(mgmt_home_col) or pd.isna(mgmt_away_col)
    weather_missing = bool(row.get("weather_missing")) if pd.notna(row.get("weather_missing")) else False
    data_quality_warn = bool(weather_missing or stats_home_missing or stats_away_missing or management_missing)
    return {
        "weather_missing": bool(weather_missing),
        "stats_home_missing": bool(stats_home_missing),
        "stats_away_missing": bool(stats_away_missing),
        "management_missing": bool(management_missing),
        "data_quality_warn": bool(data_quality_warn),
    }


def _bool_like(v):
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _safe_float_value(v, default=0.0):
    n = pd.to_numeric(v, errors="coerce")
    if pd.isna(n):
        return float(default)
    return float(n)


def compute_effective_absence_impacts(row):
    # raw値（欠損は0扱い）
    raw_total_home = _safe_float_value(row.get("absence_impact_total_home"), 0.0)
    raw_attack_home = _safe_float_value(row.get("absence_impact_attack_home"), 0.0)
    raw_defense_home = _safe_float_value(row.get("absence_impact_defense_home"), 0.0)
    raw_players_home = _safe_float_value(row.get("absence_players_count_home"), 0.0)

    raw_total_away = _safe_float_value(row.get("absence_impact_total_away"), 0.0)
    raw_attack_away = _safe_float_value(row.get("absence_impact_attack_away"), 0.0)
    raw_defense_away = _safe_float_value(row.get("absence_impact_defense_away"), 0.0)
    raw_players_away = _safe_float_value(row.get("absence_players_count_away"), 0.0)

    # 欠損判定（指定要件: count=0 かつ total=0、かつ既存欠損フラグをORで加味）
    missing_home = (raw_players_home <= 0.0 and raw_total_home <= 0.0)
    missing_away = (raw_players_away <= 0.0 and raw_total_away <= 0.0)

    if "absence_missing_home" in row.index:
        missing_home = bool(missing_home or _bool_like(row.get("absence_missing_home")))
    if "absence_missing_away" in row.index:
        missing_away = bool(missing_away or _bool_like(row.get("absence_missing_away")))

    # 既存フラグ（例: management_missing / data_quality_warn）を補助的にOR
    for flag_col in ["management_missing", "data_quality_warn"]:
        if flag_col in row.index:
            flag_val = _bool_like(row.get(flag_col))
            missing_home = bool(missing_home or flag_val)
            missing_away = bool(missing_away or flag_val)

    # 欠損ならベースライン、そうでなければrawを使用
    eff_total_home = ABSENCE_BASELINE_TOTAL if missing_home else raw_total_home
    eff_attack_home = ABSENCE_BASELINE_ATTACK if missing_home else raw_attack_home
    eff_defense_home = ABSENCE_BASELINE_DEFENSE if missing_home else raw_defense_home

    eff_total_away = ABSENCE_BASELINE_TOTAL if missing_away else raw_total_away
    eff_attack_away = ABSENCE_BASELINE_ATTACK if missing_away else raw_attack_away
    eff_defense_away = ABSENCE_BASELINE_DEFENSE if missing_away else raw_defense_away

    # 過補正防止のcap（totalのみ）
    cap_applied_home = eff_total_home > ABSENCE_IMPACT_CAP_TOTAL
    cap_applied_away = eff_total_away > ABSENCE_IMPACT_CAP_TOTAL
    eff_total_home = min(eff_total_home, ABSENCE_IMPACT_CAP_TOTAL)
    eff_total_away = min(eff_total_away, ABSENCE_IMPACT_CAP_TOTAL)

    # 念のため下限
    eff_total_home = max(0.0, eff_total_home)
    eff_total_away = max(0.0, eff_total_away)
    eff_attack_home = max(0.0, eff_attack_home)
    eff_attack_away = max(0.0, eff_attack_away)
    eff_defense_home = max(0.0, eff_defense_home)
    eff_defense_away = max(0.0, eff_defense_away)

    return {
        "absence_missing_home": bool(missing_home),
        "absence_missing_away": bool(missing_away),
        "absence_effective_total_home": float(eff_total_home),
        "absence_effective_attack_home": float(eff_attack_home),
        "absence_effective_defense_home": float(eff_defense_home),
        "absence_effective_total_away": float(eff_total_away),
        "absence_effective_attack_away": float(eff_attack_away),
        "absence_effective_defense_away": float(eff_defense_away),
        "absence_cap_applied_home": bool(cap_applied_home),
        "absence_cap_applied_away": bool(cap_applied_away),
    }


def log_absence_effective_summary(df, label):
    required = {
        "absence_missing_home",
        "absence_missing_away",
        "absence_effective_total_home",
        "absence_effective_total_away",
    }
    if not required.issubset(set(df.columns)):
        return
    rows = int(len(df))
    if rows == 0:
        return
    miss_home = int(pd.Series(df["absence_missing_home"]).fillna(False).astype(bool).sum())
    miss_away = int(pd.Series(df["absence_missing_away"]).fillna(False).astype(bool).sum())
    miss_any = int(
        (pd.Series(df["absence_missing_home"]).fillna(False).astype(bool) |
         pd.Series(df["absence_missing_away"]).fillna(False).astype(bool)).sum()
    )
    eff_home = pd.to_numeric(df["absence_effective_total_home"], errors="coerce").fillna(0.0)
    eff_away = pd.to_numeric(df["absence_effective_total_away"], errors="coerce").fillna(0.0)
    cap_home = int(pd.Series(df.get("absence_cap_applied_home", False)).fillna(False).astype(bool).sum())
    cap_away = int(pd.Series(df.get("absence_cap_applied_away", False)).fillna(False).astype(bool).sum())
    print(
        f"[ABSENCE_EFFECTIVE][{label}] rows={rows} "
        f"missing_home_count={miss_home} missing_away_count={miss_away} missing_any_row_count={miss_any} "
        f"effective_total_home_avg={eff_home.mean():.4f} effective_total_home_max={eff_home.max():.4f} "
        f"effective_total_away_avg={eff_away.mean():.4f} effective_total_away_max={eff_away.max():.4f} "
        f"cap_applied_home_count={cap_home} cap_applied_away_count={cap_away} "
        f"baseline_total={ABSENCE_BASELINE_TOTAL:.4f} cap_total={ABSENCE_IMPACT_CAP_TOTAL:.4f}"
    )

# ポアソン分布を用いて勝敗確率を計算
def predict_poisson_probabilities(
    elo_diff,
    home_xg_stats=None,
    away_xg_stats=None,
    home_travel_distance=0,
    away_travel_distance=0,
    home_fatigue_score=None,
    away_fatigue_score=None,
    home_rank_motivation_score=None,
    away_rank_motivation_score=None,
    home_absence_impact=None,
    away_absence_impact=None,
    weather_flags=None,
    max_goals=10,
):
    home_expected_goals, away_expected_goals = calculate_expected_goals(
        elo_diff,
        home_xg_stats,
        away_xg_stats,
        home_travel_distance,
        away_travel_distance,
        home_fatigue_score,
        away_fatigue_score,
        home_rank_motivation_score,
        away_rank_motivation_score,
        home_absence_impact,
        away_absence_impact,
        weather_flags,
    )

    # 打ち切りによる歪みを減らすため、十分な格子上限Kを動的に決める
    k = max(int(max_goals), int(POISSON_GRID_MIN_K))
    max_k_cap = max(k, int(POISSON_GRID_MAX_K))
    tail_eps = max(1e-12, float(POISSON_TAIL_EPS))
    while k < max_k_cap:
        in_grid_mass = float(poisson.cdf(k, home_expected_goals)) * float(poisson.cdf(k, away_expected_goals))
        tail_mass = 1.0 - in_grid_mass
        if tail_mass <= tail_eps:
            break
        k += 1

    prob_home_win = 0.0
    prob_draw = 0.0
    prob_away_win = 0.0

    for i in range(k + 1):  # ホームチームの得点
        for j in range(k + 1):  # アウェイチームの得点
            prob = poisson.pmf(i, home_expected_goals) * poisson.pmf(j, away_expected_goals)
            if i > j:
                prob_home_win += prob
            elif i == j:
                prob_draw += prob
            else:
                prob_away_win += prob
    
    # 確率の合計が1になるように正規化（max_goalsを小さくした場合に必要）
    total_prob = prob_home_win + prob_draw + prob_away_win
    if total_prob > 0:
        prob_home_win /= total_prob
        prob_draw /= total_prob
        prob_away_win /= total_prob

    return prob_home_win, prob_draw, prob_away_win


def compute_probabilities_and_result(
    match_id,
    home_elo,
    away_elo,
    home_advantage_diff,
    home_xg_stats=None,
    away_xg_stats=None,
    home_travel_distance=0,
    away_travel_distance=0,
    home_fatigue_score=None,
    away_fatigue_score=None,
    home_rank_motivation_score=None,
    away_rank_motivation_score=None,
    home_absence_impact=None,
    away_absence_impact=None,
    weather_flags=None,
    stats_home_missing=False,
    stats_away_missing=False,
    data_quality_warn=False,
):
    """予想/バックテスト共通: 確率算出→丸め→結果判定を一元化する。"""
    elo_ctx = build_elo_context(
        home_elo=home_elo,
        away_elo=away_elo,
        home_advantage_diff=home_advantage_diff,
        stats_home_missing=stats_home_missing,
        stats_away_missing=stats_away_missing,
        data_quality_warn=data_quality_warn,
    )

    # HFAの単一適用点: 確率計算に渡す差分（elo_diff_for_prob）をここで確定させる
    elo_diff_before_hfa = float(elo_ctx["elo_diff_before_hfa"])
    applied_hfa = float(elo_ctx["applied_hfa"]) if ENABLE_HFA else 0.0
    elo_diff_after_hfa = float(elo_diff_before_hfa + applied_hfa)
    elo_diff_for_prob = float(elo_diff_after_hfa) * float(ELO_DIFF_SCALE)

    if not ENABLE_HFA:
        _update_hfa_apply_counter("ENABLE_HFA=0", applied=False)
    elif float(HFA_ELO) <= 0.0:
        _update_hfa_apply_counter("HFA_ELO<=0", applied=False)
    elif abs(applied_hfa) > 1e-12:
        _update_hfa_apply_counter("applied", applied=True)
    else:
        _update_hfa_apply_counter("applied_hfa_zero_other", applied=False)

    if HDA_MODEL_MODE_EFFECTIVE == "multinom" and HDA_MODEL_BUNDLE is not None:
        (prob_home_win, prob_draw, prob_away_win), model_feats = _predict_hda_multinom_probs(elo_diff_for_prob)
        draw_model_input = float(model_feats.get("d_scaled", abs(elo_diff_for_prob)))
        draw_poi = float("nan")
        draw_elo = float("nan")
    else:
        prob_home_win, prob_draw, prob_away_win = predict_poisson_probabilities(
            elo_diff_for_prob,
            home_xg_stats,
            away_xg_stats,
            home_travel_distance,
            away_travel_distance,
            home_fatigue_score,
            away_fatigue_score,
            home_rank_motivation_score,
            away_rank_motivation_score,
            home_absence_impact,
            away_absence_impact,
            weather_flags,
        )
        prob_home_win, prob_draw, prob_away_win, draw_model_input, draw_poi, draw_elo = calibrate_draw_probability(
            prob_home_win,
            prob_draw,
            prob_away_win,
            elo_diff_for_prob,
        )
    sum_before_round = prob_home_win + prob_draw + prob_away_win
    if not np.isclose(sum_before_round, 1.0, atol=1e-6):
        print(
            f"[PROB_QC][WARN] match_id={match_id} prob_sum={sum_before_round:.9f} "
            f"(home={prob_home_win:.6f}, draw={prob_draw:.6f}, away={prob_away_win:.6f})"
        )

    if elo_diff_for_prob > 0 and prob_home_win < prob_away_win:
        print(
            f"[PROB_QC][WARN] match_id={match_id} elo_diff_for_prob={elo_diff_for_prob:.4f} "
            f"なのに prob_home({prob_home_win:.4f}) < prob_away({prob_away_win:.4f})"
        )

    predicted_result, decision_reason, decision_metrics = decide_result(
        prob_home_win, prob_draw, prob_away_win
    )

    expected_home_for_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff_for_prob / 400.0))

    debug_row = {
        "match_id": match_id,
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "home_advantage_diff_input": float(home_advantage_diff),
        "hfa_enabled": elo_ctx["hfa_enabled"],
        "matchup_bias_enabled": elo_ctx["matchup_bias_enabled"],
        "matchup_bias_coef": elo_ctx["matchup_bias_coef"],
        "matchup_bias": elo_ctx["matchup_bias"],
        "home_advantage_profile_diff_raw": elo_ctx["home_advantage_profile_diff_raw"],
        "home_advantage_profile_diff_clipped": elo_ctx["home_advantage_profile_diff_clipped"],
        "HFA_base": elo_ctx["base_hfa"],
        "HFA_multiplier": elo_ctx["hfa_mult"],
        "hfa_clip_min": float("nan"),
        "hfa_clip_max": float("nan"),
        "elo_diff_scale_factor": float(ELO_DIFF_SCALE),
        "hfa_added_to_diff": applied_hfa,
        "HFA_applied": applied_hfa,
        "elo_diff_before_hfa": elo_diff_before_hfa,
        "elo_diff_after_hfa": elo_diff_after_hfa,
        "elo_diff_raw": elo_diff_after_hfa,
        "elo_diff_scaled": elo_diff_for_prob,
        "elo_diff_for_prob": elo_diff_for_prob,
        "elo_diff": elo_diff_for_prob,
        "expected_home": expected_home_for_prob,
        "draw_model_input": draw_model_input,
        "draw_model_output": prob_draw,
        "draw_model_poi": draw_poi,
        "draw_model_elo": draw_elo,
        "hda_model_mode_effective": HDA_MODEL_MODE_EFFECTIVE,
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "predicted_result": predicted_result,
        "decision_reason": decision_reason,
        "argmax_result": decision_metrics.get("argmax_result"),
        "argmax_max_prob": decision_metrics.get("argmax_max_prob"),
    }

    if DEBUG_ELO_PROB or (DEBUG_MATCH_ID and str(match_id) == DEBUG_MATCH_ID):
        print(
            "[ELO_DEBUG] "
            f"match_id={debug_row['match_id']} "
            f"home_elo={debug_row['home_elo']:.2f} away_elo={debug_row['away_elo']:.2f} "
            f"HFA={debug_row['HFA_applied']:.4f} "
            f"elo_diff_raw={debug_row['elo_diff_raw']:.4f} elo_diff={debug_row['elo_diff']:.4f} "
            f"expected_home={debug_row['expected_home']:.4f} "
            f"draw(poi/elo/blend)=({debug_row['draw_model_poi']:.3f}/{debug_row['draw_model_elo']:.3f}/{debug_row['draw_model_output']:.3f}) "
            f"probs=({prob_home_win:.3f},{prob_draw:.3f},{prob_away_win:.3f}) "
            f"result={predicted_result}"
        )

    return prob_home_win, prob_draw, prob_away_win, predicted_result, debug_row


def decide_result(
    prob_home_win,
    prob_draw,
    prob_away_win,
    force_draw=False,
    force_reason=None,
):
    if pd.isna(prob_home_win) or pd.isna(prob_draw) or pd.isna(prob_away_win):
        return None, "UNDECIDED_NAN", {"argmax_result": None, "argmax_max_prob": None}
    ph = float(prob_home_win)
    pdw = float(prob_draw)
    pa = float(prob_away_win)
    argmax_max = max(ph, pdw, pa)
    # Base rule: calibrated H/D/A probabilities の argmax
    if ph >= pdw and ph >= pa:
        argmax_result = "H"
    elif pa >= ph and pa >= pdw:
        argmax_result = "A"
    else:
        argmax_result = "D"

    if force_draw:
        reason = force_reason or "FORCE_DRAW_BY_RULE"
        return "D", reason, {"argmax_result": argmax_result, "argmax_max_prob": float(argmax_max)}
    return argmax_result, "ARGMAX", {"argmax_result": argmax_result, "argmax_max_prob": float(argmax_max)}


def decide_predicted_result(
    prob_home_win,
    prob_draw,
    prob_away_win,
):
    # 互換ラッパー: 決定ロジックは decide_result() に集約
    decided, _, _ = decide_result(prob_home_win, prob_draw, prob_away_win)
    return decided


def assign_draw_results_by_expectation(df, output_col="predicted_result"):
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if not required_cols.issubset(df.columns):
        return df

    out = df.copy()
    out["__base_pred"] = out.apply(
        lambda r: decide_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])[0],
        axis=1,
    )
    out["decision_reason"] = out.apply(
        lambda r: decide_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])[1],
        axis=1,
    )

    def _assign_block(block: pd.DataFrame, round_label: str) -> pd.DataFrame:
        b = block.copy()
        valid = b["prob_draw"].notna() & b["prob_home_win"].notna() & b["prob_away_win"].notna()
        valid_count = int(valid.sum())
        block_count = int(len(b))
        if valid_count == 0:
            b[output_col] = b["__base_pred"]
            print(
                f"[DRAW_ASSIGN] round={round_label} matches={block_count} "
                f"Expected_draws_raw=0.00 Expected_draws_scaled=0.00 "
                f"target_draw_count=0 Assigned_D=0 overwrite_targets=[]"
            )
            return b

        expected_draws_raw = float(b.loc[valid, "prob_draw"].sum())
        expected_draws_scaled = expected_draws_raw * DRAW_EXPECTATION_MULTIPLIER
        target_draw_count = int(round(expected_draws_scaled))
        target_draw_count = max(0, min(target_draw_count, valid_count))

        top_draw = (
            b.loc[valid]
            .sort_values("prob_draw", ascending=False)
            .head(target_draw_count)
        )
        draw_idx = top_draw.index
        draw_target_col = "match_no" if "match_no" in b.columns else ("match_id" if "match_id" in b.columns else None)
        if draw_target_col:
            draw_targets = top_draw[draw_target_col].astype(str).tolist()
        else:
            draw_targets = [str(i) for i in draw_idx.tolist()]
        draw_targets_txt = "[" + ",".join(draw_targets) + "]"

        draw_idx_set = set(draw_idx.tolist())
        b[output_col] = b.index.map(lambda idx: "D" if idx in draw_idx_set else b.at[idx, "__base_pred"])
        b["decision_reason"] = b.index.map(
            lambda idx: "FORCE_DRAW_BY_EXPECTATION_ASSIGN" if idx in draw_idx_set else b.at[idx, "decision_reason"]
        )
        assigned_d = int((b.loc[valid, output_col] == "D").sum())
        print(
            f"[DRAW_ASSIGN] round={round_label} matches={block_count} "
            f"Expected_draws_raw={expected_draws_raw:.2f} Expected_draws_scaled={expected_draws_scaled:.2f} "
            f"target_draw_count={target_draw_count} "
            f"Assigned_D={assigned_d} overwrite_targets={draw_targets_txt}"
        )
        return b

    # 節単位でのみ割当。優先: toto_round_id -> round -> 節（第n節抽出）
    group_col = None
    if "toto_round_id" in out.columns:
        group_col = "toto_round_id"
    elif "round" in out.columns:
        group_col = "round"
    elif "節" in out.columns:
        def _to_round_label(v):
            if pd.isna(v):
                return pd.NA
            s = unicodedata.normalize("NFKC", str(v))
            m = re.search(r"第\s*([0-9]+)\s*節", s)
            if m:
                return f"第{int(m.group(1))}節"
            return pd.NA
        out["__round_group"] = out["節"].map(_to_round_label)
        if out["__round_group"].notna().any():
            group_col = "__round_group"

    if group_col is None:
        out = _assign_block(out, "ALL")
    else:
        pieces = []
        for g, block in out.groupby(group_col, dropna=False, sort=False):
            label = str(g) if pd.notna(g) and str(g).strip() else "NA"
            pieces.append(_assign_block(block, label))
        out = pd.concat(pieces, axis=0).sort_index()

    # 最終結果列を output_col に一本化しつつ、互換列 predicted_result も同期する
    if output_col in out.columns and output_col != "predicted_result":
        out["predicted_result"] = out[output_col]
    out = out.drop(columns=["__base_pred", "__round_group"], errors="ignore")
    return out


def recalculate_predicted_result(df, output_col="predicted_result"):
    out = df.copy()
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if not required_cols.issubset(out.columns):
        return out
    def _decide_triplet(row):
        return decide_result(row["prob_home_win"], row["prob_draw"], row["prob_away_win"])

    decided = out.apply(_decide_triplet, axis=1)
    out[output_col] = decided.map(lambda x: x[0])
    out["final_result"] = out[output_col]
    out["argmax_result"] = decided.map(lambda x: x[2].get("argmax_result"))
    out["argmax_max_prob"] = decided.map(lambda x: x[2].get("argmax_max_prob"))
    if "decision_reason" not in out.columns:
        out["decision_reason"] = decided.map(lambda x: x[1])
    return out


def recalculate_predicted_highest_prob_result(df, output_col="predicted_highest_prob_result"):
    out = df.copy()
    required_cols = {"prob_home_win_raw", "prob_draw_raw", "prob_away_win_raw"}
    if not required_cols.issubset(out.columns):
        return out
    out[output_col] = out.apply(
        lambda r: decide_result(r["prob_home_win_raw"], r["prob_draw_raw"], r["prob_away_win_raw"])[0],
        axis=1,
    )
    out["argmax_raw_result"] = out[output_col]
    out["argmax_raw_max_prob"] = out[["prob_home_win_raw", "prob_draw_raw", "prob_away_win_raw"]].max(axis=1)
    return out


def sync_and_validate_prediction_results(df, label, raise_on_error=True):
    out = df.copy()
    if "final_result" not in out.columns and "predicted_result" in out.columns:
        out["final_result"] = out["predicted_result"]
    if "predicted_result" not in out.columns and "final_result" in out.columns:
        out["predicted_result"] = out["final_result"]
    if "predicted_result" in out.columns and "final_result" in out.columns:
        mismatch = out["predicted_result"].astype(str) != out["final_result"].astype(str)
        m = int(mismatch.sum())
        if m > 0:
            sample_cols = [c for c in ["match_id", "match_no", "home_team", "away_team", "predicted_result", "final_result"] if c in out.columns]
            sample = out.loc[mismatch, sample_cols].head(10).to_dict(orient="records")
            msg = f"[CONSISTENCY][ERROR:{label}] predicted_result!=final_result rows={m} sample={sample}"
            print(msg)
            if raise_on_error:
                raise RuntimeError(msg)
        out["predicted_result"] = out["final_result"]

    if "decision_reason" in out.columns and "final_result" in out.columns:
        force_mask = out["decision_reason"].astype(str).str.contains("FORCE_DRAW", na=False)
        bad_force = force_mask & (out["final_result"].astype(str) != "D")
        bad = int(bad_force.sum())
        if bad > 0:
            sample_cols = [c for c in ["match_id", "match_no", "home_team", "away_team", "decision_reason", "final_result"] if c in out.columns]
            sample = out.loc[bad_force, sample_cols].head(10).to_dict(orient="records")
            msg = f"[CONSISTENCY][ERROR:{label}] FORCE_DRAW reason but final_result!=D rows={bad} sample={sample}"
            print(msg)
            if raise_on_error:
                raise RuntimeError(msg)

    if "argmax_raw_result" in out.columns and "final_result" in out.columns:
        diff = int((out["argmax_raw_result"].astype(str) != out["final_result"].astype(str)).sum())
        print(f"[CONSISTENCY:{label}] raw_vs_final_diff_rows={diff}")
    return out


def log_prediction_consistency(df, label):
    required = {
        "prob_home_win_raw",
        "prob_draw_raw",
        "prob_away_win_raw",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "predicted_result",
        "predicted_highest_prob_result",
    }
    if not required.issubset(df.columns):
        print(f"[PRED_CHECK:{label}] required_columns_missing")
        return

    work = df.copy()
    work["_raw_argmax"] = work.apply(
        lambda r: decide_result(r["prob_home_win_raw"], r["prob_draw_raw"], r["prob_away_win_raw"])[0],
        axis=1,
    )
    work["_cal_argmax"] = work.apply(
        lambda r: decide_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])[0],
        axis=1,
    )

    rows = len(work)
    if rows == 0:
        print(f"[PRED_CHECK:{label}] rows=0")
        return

    match_id_col = "match_id" if "match_id" in work.columns else None
    raw_vs_cal_match = (work["_raw_argmax"] == work["_cal_argmax"])
    pred_vs_highest_match = (work["predicted_result"] == work["predicted_highest_prob_result"])
    delta_draw = pd.to_numeric(work["prob_draw"], errors="coerce") - pd.to_numeric(work["prob_draw_raw"], errors="coerce")

    print(
        f"[PRED_CHECK:{label}] raw_argmax_vs_cal_argmax_match_rate={raw_vs_cal_match.mean()*100:.1f}% "
        f"pred_vs_highest_match_rate={pred_vs_highest_match.mean()*100:.1f}%"
    )
    print(
        f"[PRED_CHECK:{label}] draw_delta(mean/max/plus_count)="
        f"{delta_draw.mean(skipna=True):.6f}/{delta_draw.max(skipna=True):.6f}/{int((delta_draw > 0).sum())}"
    )

    pred = work["predicted_result"].astype(str)
    h_rate = (pred == "H").mean() * 100
    d_rate = (pred == "D").mean() * 100
    a_rate = (pred == "A").mean() * 100
    print(f"[PRED_CHECK:{label}] predicted_result_ratio(H/D/A)={h_rate:.1f}%/{d_rate:.1f}%/{a_rate:.1f}%")

    mismatch_highest = work[work["predicted_highest_prob_result"] != work["_raw_argmax"]]
    mismatch_pred = work[work["predicted_result"] != work["_cal_argmax"]]
    if len(mismatch_highest) > 0:
        mids = mismatch_highest[match_id_col].astype(str).tolist() if match_id_col else mismatch_highest.index.astype(str).tolist()
        print(f"[PRED_CHECK:{label}][WARN] highest_vs_raw_argmax_mismatch={len(mismatch_highest)} match_ids={mids}")
    else:
        print(f"[PRED_CHECK:{label}] highest_vs_raw_argmax_mismatch=0")
    if len(mismatch_pred) > 0:
        mids = mismatch_pred[match_id_col].astype(str).tolist() if match_id_col else mismatch_pred.index.astype(str).tolist()
        print(f"[PRED_CHECK:{label}][WARN] predicted_vs_cal_argmax_mismatch={len(mismatch_pred)} match_ids={mids}")
    else:
        print(f"[PRED_CHECK:{label}] predicted_vs_cal_argmax_mismatch=0")


def _normalize_probs(ph, pdw, pa):
    arr = np.array([ph, pdw, pa], dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    arr = arr / s
    return float(arr[0]), float(arr[1]), float(arr[2])


def sanitize_prob_triplet(ph, pdw, pa, fallback=PROB_FALLBACK):
    arr = np.array([ph, pdw, pa], dtype=float)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return _normalize_probs(*fallback)
    arr = np.clip(arr, 0.0, 1.0)
    s = arr.sum()
    if s <= 0:
        return _normalize_probs(*fallback)
    return float(arr[0] / s), float(arr[1] / s), float(arr[2] / s)


def calibrate_probabilities(p_home, p_draw, p_away, league, cap_j1=J1_WIN_PROB_CAP):
    p_home, p_draw, p_away = sanitize_prob_triplet(p_home, p_draw, p_away)
    if pd.isna(league) or str(league).strip() == "":
        lg = str(LEAGUE).strip().upper()
    else:
        lg = str(league).strip().upper()
    if lg != "J1":
        return p_home, p_draw, p_away

    # J1のみ、勝ち確率（H/A）の過剰確信を抑える
    if p_home > cap_j1:
        delta = p_home - cap_j1
        p_home = cap_j1
        p_draw += delta
    if p_away > cap_j1:
        delta = p_away - cap_j1
        p_away = cap_j1
        p_draw += delta

    p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)
    # 念のため cap を再適用（数値誤差吸収）
    p_home = min(p_home, cap_j1)
    p_away = min(p_away, cap_j1)
    p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)
    return p_home, p_draw, p_away


def predict_elo_probabilities_with_home_advantage(
    home_elo,
    away_elo,
    home_advantage_diff,
    home_adv_coef=HOME_ADV_ELO_COEF,
    draw_base=ELO_DRAW_BASE,
    draw_sensitivity=ELO_DRAW_SENSITIVITY,
    draw_min=ELO_DRAW_MIN,
    draw_max=ELO_DRAW_MAX,
):
    # home_advantage_diff を Elo に反映して有効Elo差を作る
    elo_home_eff = float(home_elo) + float(home_advantage_diff) * float(home_adv_coef)
    elo_away_eff = float(away_elo)
    elo_diff = elo_home_eff - elo_away_eff

    # Elo期待値（2値）を先に計算
    p_home_two_way = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_away_two_way = 1.0 - p_home_two_way

    # Elo差が大きいほど draw を下げる簡易モデル
    p_draw = float(draw_base) - abs(float(elo_diff)) * float(draw_sensitivity)
    p_draw = max(float(draw_min), min(float(draw_max), p_draw))

    # 残りを home/away に按分し、最後に正規化
    remaining = 1.0 - p_draw
    p_home = remaining * p_home_two_way
    p_away = remaining * p_away_two_way
    return _normalize_probs(p_home, p_draw, p_away)


def build_home_away_profile_map(results_df):
    finished = results_df.dropna(subset=["home_score", "away_score"]).copy()
    if finished.empty:
        return {}, {}

    home = pd.DataFrame(
        {
            "team": finished["home_team"],
            "venue": "home",
            "gf": pd.to_numeric(finished["home_score"], errors="coerce"),
            "ga": pd.to_numeric(finished["away_score"], errors="coerce"),
        }
    )
    away = pd.DataFrame(
        {
            "team": finished["away_team"],
            "venue": "away",
            "gf": pd.to_numeric(finished["away_score"], errors="coerce"),
            "ga": pd.to_numeric(finished["home_score"], errors="coerce"),
        }
    )
    rows = pd.concat([home, away], ignore_index=True).dropna(subset=["gf", "ga"])
    rows["is_win"] = (rows["gf"] > rows["ga"]).astype(int)
    rows["is_draw"] = (rows["gf"] == rows["ga"]).astype(int)
    rows["points"] = rows["is_win"] * 3 + rows["is_draw"]

    g = rows.groupby(["team", "venue"], as_index=False).agg(matches=("team", "size"), points=("points", "sum"))
    g["ppm"] = g["points"] / g["matches"]

    home_map = g[g["venue"] == "home"].set_index("team")["ppm"].to_dict()
    away_map = g[g["venue"] == "away"].set_index("team")["ppm"].to_dict()
    return home_map, away_map


def calc_home_advantage_diff(home_team, away_team, home_ppm_map, away_ppm_map):
    home_ppm = float(home_ppm_map.get(home_team, 0.0))
    away_ppm = float(away_ppm_map.get(away_team, 0.0))
    diff = home_ppm - away_ppm
    return diff, diff > 0


def load_j2_allowed_teams():
    if LEAGUE != "j2":
        return None

    # 1) 明示ファイル優先
    if os.path.exists(J2_ALLOWED_TEAMS_CSV):
        try:
            df = pd.read_csv(J2_ALLOWED_TEAMS_CSV)
            col = "team_name" if "team_name" in df.columns else df.columns[0]
            teams = set(canonical_team_name(v) for v in df[col].dropna().astype(str))
            teams = {t for t in teams if t}
            if teams:
                print(f"J2許可チームを {J2_ALLOWED_TEAMS_CSV} から読み込みました: {len(teams)}")
                return teams
        except Exception as e:
            print(f"警告: J2許可チームCSVの読み込みに失敗しました: {e}")

    # 2) フォールバック: 前年J2結果から推定
    fallback_csv = os.path.join(DATA_DIR, f"j2_{PREV_SEASON_YEAR}_latest_results.csv")
    if os.path.exists(fallback_csv):
        try:
            df = pd.read_csv(fallback_csv)
            teams = set(canonical_team_name(v) for v in df["home_team"].dropna().astype(str)) | set(
                canonical_team_name(v) for v in df["away_team"].dropna().astype(str)
            )
            teams = {t for t in teams if t}
            if teams:
                print(f"J2許可チームを前年データから推定しました: {len(teams)} ({fallback_csv})")
                return teams
        except Exception as e:
            print(f"警告: 前年J2結果からの許可チーム推定に失敗しました: {e}")

    print("警告: J2許可チームを特定できませんでした。全カードを予測対象にします。")
    return None


def canonical_team_name(name):
    if pd.isna(name):
        return None
    text = _normalize_team_text(name)
    # 代表表記へ寄せる
    text = TEAM_NAME_ALIAS_MAP.get(text, text)
    return text


def normalize_team_series(series):
    return series.map(canonical_team_name)


def drop_j2_excluded_stats_columns(df):
    out = df.copy()
    if LEAGUE != "j2":
        return out
    drop_cols = []
    for base_name in J2_EXCLUDED_STATS_BASE_NAMES:
        drop_cols.append(f"stats_{base_name}_home")
        drop_cols.append(f"stats_{base_name}_away")
    existing = [c for c in drop_cols if c in out.columns]
    if existing:
        out = out.drop(columns=existing, errors="ignore")
        print(f"[INFO] J2除外指標を予測入力から削除: cols={len(existing)}")
    return out


def extract_round_number(v):
    if pd.isna(v):
        return pd.NA
    s = str(v)
    s = unicodedata.normalize("NFKC", s)
    m = re.search(r"第\s*([0-9]+)\s*節", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return pd.NA
    m2 = re.search(r"([0-9]+)", s)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return pd.NA
    return pd.NA


def _safe_numeric(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def load_absence_impact_team_round_map(absence_csv_path, match_round_numbers):
    if (not absence_csv_path) or (not os.path.exists(absence_csv_path)):
        print("[ABSENCE] 欠場影響CSVが見つからないためスキップします。")
        return pd.DataFrame()
    try:
        src = pd.read_csv(absence_csv_path)
    except Exception as e:
        print(f"[ABSENCE][WARN] 欠場影響CSVの読み込みに失敗: {e}")
        return pd.DataFrame()

    if src.empty:
        print("[ABSENCE] 欠場影響CSVが空のためスキップします。")
        return pd.DataFrame()

    required = {"team", "round_start"}
    if not required.issubset(set(src.columns)):
        print(f"[ABSENCE][WARN] 必須列不足: need={required}, have={set(src.columns)}")
        return pd.DataFrame()

    work = src.copy()
    if "season" not in work.columns:
        work["season"] = int(SEASON_YEAR)
    work["season"] = _safe_numeric(work["season"], default=int(SEASON_YEAR)).astype("Int64")
    work["round_start"] = _safe_numeric(work["round_start"]).astype("Int64")
    work["expected_rounds"] = _safe_numeric(work.get("expected_rounds", 1), default=1).astype("Int64")
    work.loc[work["expected_rounds"] <= 0, "expected_rounds"] = 1

    # 影響列が無い場合は weight から代用
    if "impact_total" not in work.columns:
        wm = _safe_numeric(work.get("weight_minutes", 0))
        wa = _safe_numeric(work.get("weight_attack", 0))
        wd = _safe_numeric(work.get("weight_defense", 0))
        work["impact_total"] = 0.6 * wm + 0.2 * wa + 0.2 * wd
    if "impact_attack" not in work.columns:
        work["impact_attack"] = _safe_numeric(work.get("weight_attack", 0))
    if "impact_defense" not in work.columns:
        work["impact_defense"] = _safe_numeric(work.get("weight_defense", 0))

    work["impact_total"] = _safe_numeric(work["impact_total"])
    work["impact_attack"] = _safe_numeric(work["impact_attack"])
    work["impact_defense"] = _safe_numeric(work["impact_defense"])
    if "availability" in work.columns:
        # returned は影響0扱い
        av = work["availability"].astype(str).str.lower().str.strip()
        work.loc[av.isin(["returned", "return", "available", "fit"]), ["impact_total", "impact_attack", "impact_defense"]] = 0.0

    work["team_name"] = normalize_team_series(work["team"].astype(str))
    work["_merge_team_name"] = normalize_team_series(work["team_name"])

    target_rounds = sorted({int(x) for x in match_round_numbers if pd.notna(x)})
    if not target_rounds:
        print("[ABSENCE][WARN] 対象節が特定できないため欠場影響を無効化します。")
        return pd.DataFrame()
    min_r = min(target_rounds)
    max_r = max(target_rounds)

    expanded_rows = []
    for _, r in work.iterrows():
        if pd.isna(r["round_start"]) or pd.isna(r["season"]) or pd.isna(r["_merge_team_name"]):
            continue
        start_r = int(r["round_start"])
        span = int(r["expected_rounds"]) if pd.notna(r["expected_rounds"]) else 1
        end_r = start_r + max(span, 1) - 1
        # 予測対象節へクリップ
        s = max(start_r, min_r)
        e = min(end_r, max_r)
        if s > e:
            continue
        for rr in range(s, e + 1):
            expanded_rows.append(
                {
                    "season": int(r["season"]),
                    "_merge_team_name": r["_merge_team_name"],
                    "round_no": rr,
                    "absence_impact_total": float(r["impact_total"]),
                    "absence_impact_attack": float(r["impact_attack"]),
                    "absence_impact_defense": float(r["impact_defense"]),
                    "absence_players_count": 1,
                }
            )

    if not expanded_rows:
        print("[ABSENCE] 対象節に有効な欠場行がありません。")
        return pd.DataFrame()

    out = pd.DataFrame(expanded_rows)
    out = (
        out.groupby(["season", "_merge_team_name", "round_no"], as_index=False)
        .agg(
            absence_impact_total=("absence_impact_total", "sum"),
            absence_impact_attack=("absence_impact_attack", "sum"),
            absence_impact_defense=("absence_impact_defense", "sum"),
            absence_players_count=("absence_players_count", "sum"),
        )
    )
    print(
        f"[ABSENCE] 取り込み完了: src_rows={len(work)}, expanded={len(expanded_rows)}, team_round_rows={len(out)}"
    )
    return out


def merge_absence_impacts(df, absence_map_df, stage_label):
    if absence_map_df is None or absence_map_df.empty:
        out = df.copy()
        for c in [
            "absence_impact_total_home", "absence_impact_attack_home", "absence_impact_defense_home", "absence_players_count_home",
            "absence_impact_total_away", "absence_impact_attack_away", "absence_impact_defense_away", "absence_players_count_away",
        ]:
            if c not in out.columns:
                out[c] = 0.0
        return out

    out = df.copy()
    out["_round_no"] = out["節"].map(extract_round_number).astype("Int64")
    out["_season"] = int(SEASON_YEAR)
    out["_merge_home_team"] = normalize_team_series(out["home_team"])
    out["_merge_away_team"] = normalize_team_series(out["away_team"])

    home_map = absence_map_df.rename(
        columns={
            "absence_impact_total": "absence_impact_total_home",
            "absence_impact_attack": "absence_impact_attack_home",
            "absence_impact_defense": "absence_impact_defense_home",
            "absence_players_count": "absence_players_count_home",
        }
    )
    away_map = absence_map_df.rename(
        columns={
            "absence_impact_total": "absence_impact_total_away",
            "absence_impact_attack": "absence_impact_attack_away",
            "absence_impact_defense": "absence_impact_defense_away",
            "absence_players_count": "absence_players_count_away",
        }
    )

    out = audited_left_merge(
        out,
        home_map[
            ["season", "_merge_team_name", "round_no", "absence_impact_total_home", "absence_impact_attack_home", "absence_impact_defense_home", "absence_players_count_home"]
        ],
        stage=f"{stage_label}_home",
        left_on=["_season", "_merge_home_team", "_round_no"],
        right_on=["season", "_merge_team_name", "round_no"],
        validate="many_to_one",
    )
    out = out.drop(columns=["season", "_merge_team_name", "round_no"], errors="ignore")

    out = audited_left_merge(
        out,
        away_map[
            ["season", "_merge_team_name", "round_no", "absence_impact_total_away", "absence_impact_attack_away", "absence_impact_defense_away", "absence_players_count_away"]
        ],
        stage=f"{stage_label}_away",
        left_on=["_season", "_merge_away_team", "_round_no"],
        right_on=["season", "_merge_team_name", "round_no"],
        validate="many_to_one",
    )
    out = out.drop(columns=["season", "_merge_team_name", "round_no"], errors="ignore")

    for c in [
        "absence_impact_total_home", "absence_impact_attack_home", "absence_impact_defense_home", "absence_players_count_home",
        "absence_impact_total_away", "absence_impact_attack_away", "absence_impact_defense_away", "absence_players_count_away",
    ]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out = out.drop(columns=["_merge_home_team", "_merge_away_team", "_round_no", "_season"], errors="ignore")
    return out


def normalize_travel_distance_matrix(df):
    if df.empty:
        return df
    out = df.copy()
    out.index = [canonical_team_name(x) for x in out.index]
    out.columns = [canonical_team_name(x) for x in out.columns]
    out = out.groupby(level=0).mean(numeric_only=True)
    out = out.T.groupby(level=0).mean(numeric_only=True).T
    return out


def merge_weather_cache(df, weather_cache_df, stage):
    _ensure_merge_qc_dir()
    _log_df_key_health(stage, "left_before", df, ["match_id"])
    _log_df_key_health(stage, "right", weather_cache_df, ["match_id"])

    merged = pd.merge(
        df,
        weather_cache_df,
        how="left",
        on="match_id",
        validate="many_to_one",
        indicator="_merge_weather",
        suffixes=("", "_weather"),
    )
    counts = merged["_merge_weather"].value_counts(dropna=False).to_dict()
    print(f"[MERGE_QC] {stage} indicator={counts}")

    left_only = merged[merged["_merge_weather"] == "left_only"].copy()
    if not left_only.empty:
        left_only_path = os.path.join(MERGE_QC_DIR, f"{stage}_left_only.csv")
        left_only.to_csv(left_only_path, index=False, encoding="utf-8-sig")
        print(f"[MERGE_QC][WARN] {stage}: left_only={len(left_only)} -> {left_only_path}")
        show_cols = [c for c in ["match_id", "home_team", "away_team"] if c in left_only.columns]
        if show_cols:
            print(left_only[show_cols].head(10).to_string(index=False))
    else:
        print(f"[MERGE_QC] {stage}: left_only=0")

    weather_cols = [c for c in ["is_rain", "is_heavy_rain", "is_strong_wind"] if c in merged.columns]
    if not weather_cols:
        merged["is_rain"] = pd.NA
        merged["is_heavy_rain"] = pd.NA
        merged["is_strong_wind"] = pd.NA
        weather_cols = ["is_rain", "is_heavy_rain", "is_strong_wind"]

    merged["weather_missing"] = (merged["_merge_weather"] == "left_only") | merged[weather_cols].isna().all(axis=1)
    for col in weather_cols:
        merged[col] = merged[col].fillna(False).astype(bool)

    return merged.drop(columns=["_merge_weather"], errors="ignore")


def normalize_weather_cache_columns(df):
    """新旧の天候CSV列名を予測側の期待列へ正規化する。"""
    out = df.copy()
    # キックオフ時刻 / スタジアム名
    if "datetime" not in out.columns and "kickoff_jst" in out.columns:
        out["datetime"] = out["kickoff_jst"]
    if "stadium" not in out.columns and "stadium_name" in out.columns:
        out["stadium"] = out["stadium_name"]

    # 気温 / 風速
    if "temperature" not in out.columns and "temp_kickoff" in out.columns:
        out["temperature"] = out["temp_kickoff"]
    if "wind_speed" not in out.columns and "wind_kickoff" in out.columns:
        out["wind_speed"] = out["wind_kickoff"]

    # 取得時刻（なければ空列を作る）
    if "last_updated_at" not in out.columns:
        out["last_updated_at"] = pd.NA

    return out


def load_allowed_teams():
    env_allowed_csv = os.environ.get("ALLOWED_TEAMS_CSV")
    csv_candidates = []
    if env_allowed_csv:
        csv_candidates.append(env_allowed_csv)
    csv_candidates.append(os.path.join(MANUAL_DIR, f"{LEAGUE}_allowed_teams_{SEASON_YEAR}.csv"))
    csv_candidates.append(os.path.join(MANUAL_DIR, f"{LEAGUE}_allowed_teams.csv"))

    for allowed_csv in csv_candidates:
        if not allowed_csv or not os.path.exists(allowed_csv):
            continue
        try:
            df = pd.read_csv(allowed_csv)
            col = "team_name" if "team_name" in df.columns else df.columns[0]
            teams = set(canonical_team_name(v) for v in df[col].dropna().astype(str))
            teams = {t for t in teams if t}
            if teams:
                print(f"{LEAGUE.upper()}許可チームを {allowed_csv} から読み込みました: {len(teams)}")
                return teams
        except Exception as e:
            print(f"警告: 許可チームCSVの読み込みに失敗しました ({allowed_csv}): {e}")

    # J1は team_master_stats を優先して許可チームを推定する
    if LEAGUE == "j1":
        def estimate_top_teams_from_results(results_csv, top_n=3):
            try:
                df = pd.read_csv(results_csv)
            except Exception:
                return set()
            required = {"home_team", "away_team", "home_score", "away_score"}
            if not required.issubset(df.columns):
                return set()
            df = df.dropna(subset=["home_score", "away_score"])
            if df.empty:
                return set()
            teams = sorted(set(df["home_team"].astype(str)) | set(df["away_team"].astype(str)))
            pts = {t: 0 for t in teams}
            gd = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            for _, r in df.iterrows():
                h = str(r["home_team"]).strip()
                a = str(r["away_team"]).strip()
                hs = int(r["home_score"])
                aw = int(r["away_score"])
                gf[h] += hs
                gf[a] += aw
                gd[h] += hs - aw
                gd[a] += aw - hs
                if hs > aw:
                    pts[h] += 3
                elif hs < aw:
                    pts[a] += 3
                else:
                    pts[h] += 1
                    pts[a] += 1
            rank = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
            return set(canonical_team_name(t) for t in rank[:top_n])

        stats_candidates = [team_master_stats_csv, os.path.join(DATA_DIR, "team_master_stats.csv")]
        for stats_csv in stats_candidates:
            if not os.path.exists(stats_csv):
                continue
            try:
                stats_df = pd.read_csv(stats_csv)
                if "team_name" not in stats_df.columns:
                    continue
                teams = set(canonical_team_name(v) for v in stats_df["team_name"].dropna().astype(str))
                teams = {t for t in teams if t}
                if teams:
                    j2_prev_csv = os.path.join(DATA_DIR, f"j2_{PREV_SEASON_YEAR}_latest_results.csv")
                    promoted = estimate_top_teams_from_results(j2_prev_csv, top_n=3)
                    if promoted:
                        teams |= promoted
                        print(
                            f"J1許可チームを team_master_stats + 前年J2上位から推定しました: "
                            f"{len(teams)} ({stats_csv})"
                        )
                    else:
                        print(f"J1許可チームを team_master_stats から推定しました: {len(teams)} ({stats_csv})")
                    return teams
            except Exception as e:
                print(f"警告: team_master_stats からの許可チーム推定に失敗しました ({stats_csv}): {e}")

    if LEAGUE == "j2":
        # 2026特別大会（J2/J3混在）ではリーグ外除外を無効化し、日程側の定義に従う
        if SEASON_YEAR >= 2026 and (not _env_flag("ENABLE_J2_STRICT_FILTER", 0)):
            print("J2許可チームフィルタを無効化します（2026特別大会モード）。")
            return None
        return load_j2_allowed_teams()

    print(f"警告: {LEAGUE.upper()}許可チームを特定できませんでした。全カードを予測対象にします。")
    return None

# Elo更新（簡易式）
def update_elo(elo_home, elo_away, result, k=20):
    # predict_eloはここでは使わないが、Elo更新のために仮のp_homeを計算
    elo_diff = elo_home + ELO_UPDATE_HOME_ADVANTAGE - elo_away
    p_home = 1 / (1 + 10 ** (-elo_diff / 400))
    s_home = {"H": 1, "D": 0.5, "A": 0}[result]
    delta = k * (s_home - p_home)
    return elo_home + delta, elo_away - delta


def sort_results_for_elo(df):
    if "datetime" in df.columns:
        out = df.copy()
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        sort_cols = ["datetime"]
        if "match_id" in out.columns:
            sort_cols.append("match_id")
        return out.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return df.reset_index(drop=True)


def compute_elo_map_from_results(results_df, base_elo_map=None):
    elo_map = {} if base_elo_map is None else {k: float(v) for k, v in base_elo_map.items()}
    ordered = sort_results_for_elo(results_df)
    for _, row in ordered.iterrows():
        home = row.get("home_team")
        away = row.get("away_team")
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.isna(home) or pd.isna(away):
            continue
        home = str(home).strip()
        away = str(away).strip()
        if home not in elo_map:
            elo_map[home] = INITIAL_ELO
        if away not in elo_map:
            elo_map[away] = INITIAL_ELO
        result = get_result(hs, as_)
        if result:
            elo_map[home], elo_map[away] = update_elo(elo_map[home], elo_map[away], result)
    return elo_map


def load_or_build_prev_final_elo(df_prev_results):
    if FORCE_RECALC:
        print(f"[FORCE] 前年最終ELOキャッシュを再利用しません: {prev_final_elo_csv}")
    elif os.path.exists(prev_final_elo_csv):
        try:
            prev_elo_df = pd.read_csv(prev_final_elo_csv)
            if {"team", "elo"}.issubset(prev_elo_df.columns):
                elo_map = {
                    str(r["team"]).strip(): float(r["elo"])
                    for _, r in prev_elo_df[["team", "elo"]].dropna(subset=["team"]).iterrows()
                }
                if elo_map:
                    print(f"前年最終ELOを読み込みました: {prev_final_elo_csv} ({len(elo_map)}チーム)")
                    return elo_map
        except Exception as e:
            print(f"警告: 前年最終ELOの読み込みに失敗しました: {e}")

    if df_prev_results.empty:
        print("前年最終ELOは未作成（前年結果データなし）。")
        return {}

    elo_map = compute_elo_map_from_results(df_prev_results)
    try:
        out_df = pd.DataFrame(
            [{"team": team, "elo": round(score, 6)} for team, score in sorted(elo_map.items(), key=lambda x: x[0])]
        )
        out_df.to_csv(prev_final_elo_csv, index=False, encoding="utf-8-sig")
        print(f"前年最終ELOを作成しました: {prev_final_elo_csv} ({len(out_df)}チーム)")
    except Exception as e:
        print(f"警告: 前年最終ELOの保存に失敗しました: {e}")
    return elo_map


def _ensure_merge_qc_dir():
    os.makedirs(MERGE_QC_DIR, exist_ok=True)


def _to_key_list(on=None, left_on=None):
    if on is not None:
        return [on] if isinstance(on, str) else list(on)
    if left_on is not None:
        return [left_on] if isinstance(left_on, str) else list(left_on)
    return []


def _key_stats(df, keys):
    if not keys or any(k not in df.columns for k in keys):
        return {"rows": len(df), "unique_keys": None, "duplicate_keys": None}
    subset = df[keys]
    return {
        "rows": len(df),
        "unique_keys": int(subset.drop_duplicates().shape[0]),
        "duplicate_keys": int(subset.duplicated().sum()),
    }


def _log_df_key_health(stage, side, df, keys):
    stats = _key_stats(df, keys)
    if stats["unique_keys"] is None:
        print(f"[MERGE_QC] {stage} {side}: rows={stats['rows']} keys={keys} (missing key columns)")
        return
    print(
        f"[MERGE_QC] {stage} {side}: rows={stats['rows']} "
        f"unique_keys={stats['unique_keys']} duplicate_keys={stats['duplicate_keys']} keys={keys}"
    )


def audited_left_merge(
    left_df,
    right_df,
    stage,
    on=None,
    left_on=None,
    right_on=None,
    validate=None,
    suffixes=("", "_r"),
):
    _ensure_merge_qc_dir()
    left_keys = _to_key_list(on=on, left_on=left_on)
    right_keys = _to_key_list(on=on, left_on=right_on)

    _log_df_key_health(stage, "left_before", left_df, left_keys)
    _log_df_key_health(stage, "right", right_df, right_keys)
    before_rows = len(left_df)

    missing_left_keys = [k for k in left_keys if k not in left_df.columns]
    missing_right_keys = [k for k in right_keys if k not in right_df.columns]
    if missing_left_keys or missing_right_keys:
        print(
            f"[MERGE_QC][WARN] {stage}: mergeキー不足のため結合をスキップ "
            f"(missing_left={missing_left_keys}, missing_right={missing_right_keys})"
        )
        fallback = left_df.copy()
        for c in right_df.columns:
            if c not in fallback.columns and c not in right_keys:
                fallback[c] = pd.NA
        return fallback

    indicator_col = f"_merge_{stage}"
    merged = pd.merge(
        left_df,
        right_df,
        how="left",
        on=on,
        left_on=left_on,
        right_on=right_on,
        validate=validate,
        indicator=indicator_col,
        suffixes=suffixes,
    )

    after_rows = len(merged)
    counts = merged[indicator_col].value_counts(dropna=False).to_dict()
    print(
        f"[MERGE_QC] {stage} result: rows_before={before_rows} rows_after={after_rows} "
        f"delta={after_rows - before_rows} validate={validate} indicator={counts}"
    )
    if after_rows > before_rows:
        print(f"[MERGE_QC][WARN] {stage}: row増殖を検知しました（右側キー重複の可能性）")
    if after_rows < before_rows:
        print(f"[MERGE_QC][WARN] {stage}: row減少を検知しました（merge条件を要確認）")

    left_only = merged[merged[indicator_col] == "left_only"].copy()
    if not left_only.empty:
        print(f"[MERGE_QC][WARN] {stage}: left_only={len(left_only)}")
        show_cols = [c for c in ["match_id", "home_team", "away_team"] if c in left_only.columns]
        show_cols += [c for c in left_keys if c in left_only.columns and c not in show_cols]
        show_cols += [c for c in right_keys if c in left_only.columns and c not in show_cols]
        preview_cols = show_cols[:10] if show_cols else left_only.columns[:10].tolist()
        print(left_only[preview_cols].head(10).to_string(index=False))
        left_only_path = os.path.join(MERGE_QC_DIR, f"{stage}_left_only.csv")
        left_only.to_csv(left_only_path, index=False, encoding="utf-8-sig")
        print(f"[MERGE_QC] {stage}: left_only CSV保存 -> {left_only_path}")
    else:
        print(f"[MERGE_QC] {stage}: left_only=0")

    return merged.drop(columns=[indicator_col], errors="ignore")


def report_missing_rates(df, stage, threshold=MISSING_WARN_THRESHOLD):
    if len(df) == 0:
        print(f"[MISSING_QC] {stage}: rows=0 のため欠損率をスキップ")
        return
    target_groups = {
        "weather": ["is_rain", "is_heavy_rain", "is_strong_wind", "weather_missing"],
        "stats": ["stats_ゴール期待値_home", "stats_ゴール期待値_away", "stats_home_missing", "stats_away_missing"],
        "management": [
            "management_recent_injuries_suspensions_count_home",
            "management_recent_injuries_suspensions_count",
            "management_recent_injuries_suspensions_count_away",
            "management_missing",
        ],
        "quality": ["data_quality_warn"],
    }
    existing = []
    for cols in target_groups.values():
        existing.extend([c for c in cols if c in df.columns and c not in existing])
    if not existing:
        print(f"[MISSING_QC] {stage}: 対象カラムなし")
        return
    for col in existing:
        miss_rate = float(df[col].isna().mean())
        level = "WARN" if miss_rate > threshold else "INFO"
        print(f"[MISSING_QC][{level}] {stage} {col}: missing_rate={miss_rate:.2%}")


def export_team_name_diff(matches_df, stats_csv_path):
    _ensure_merge_qc_dir()
    try:
        stats_df = pd.read_csv(stats_csv_path)
    except Exception as e:
        print(f"[MERGE_QC][WARN] チーム名差分の算出失敗: {e}")
        return
    if "team_name" not in stats_df.columns:
        print("[MERGE_QC][WARN] stats側に team_name 列がないため差分をスキップ")
        return

    match_teams = set(normalize_team_series(matches_df["home_team"].dropna())) | set(
        normalize_team_series(matches_df["away_team"].dropna())
    )
    stats_teams = set(normalize_team_series(stats_df["team_name"].dropna()))
    only_matches = sorted(match_teams - stats_teams)

    diff_df = pd.DataFrame({"team_name_in_matches_only": only_matches})
    diff_path = os.path.join(MERGE_QC_DIR, "team_name_diff_matches_vs_stats.csv")
    diff_df.to_csv(diff_path, index=False, encoding="utf-8-sig")
    print(
        f"[MERGE_QC] チーム名差分: matches_only={len(only_matches)} "
        f"CSV保存 -> {diff_path}"
    )

    only_matches_norm = sorted(only_matches)
    diff_norm_df = pd.DataFrame({"team_name_in_matches_only_after_canonical": only_matches_norm})
    diff_norm_path = os.path.join(MERGE_QC_DIR, "team_name_diff_matches_vs_stats_after_canonical.csv")
    diff_norm_df.to_csv(diff_norm_path, index=False, encoding="utf-8-sig")
    print(
        f"[MERGE_QC] 正規化後チーム名差分: matches_only={len(only_matches_norm)} "
        f"CSV保存 -> {diff_norm_path}"
    )


def _pick_first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _pick_first_non_na_value(row, candidates):
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None


def add_data_quality_flags(df):
    out = df.copy()

    if "weather_missing" not in out.columns:
        weather_cols = [c for c in ["is_rain", "is_heavy_rain", "is_strong_wind"] if c in out.columns]
        if weather_cols:
            out["weather_missing"] = out[weather_cols].isna().all(axis=1)
        else:
            out["weather_missing"] = True

    home_stats_col = _pick_first_existing(out, ["stats_ゴール期待値_home", "stats_ゴール期待値"])
    away_stats_col = _pick_first_existing(out, ["stats_ゴール期待値_away"])
    out["stats_home_missing"] = out[home_stats_col].isna() if home_stats_col else True
    out["stats_away_missing"] = out[away_stats_col].isna() if away_stats_col else True

    mgmt_home_col = _pick_first_existing(
        out,
        ["management_recent_injuries_suspensions_count_home", "management_recent_injuries_suspensions_count"],
    )
    mgmt_away_col = _pick_first_existing(out, ["management_recent_injuries_suspensions_count_away"])
    if mgmt_home_col and mgmt_away_col:
        out["management_missing"] = out[[mgmt_home_col, mgmt_away_col]].isna().any(axis=1)
    elif mgmt_home_col:
        out["management_missing"] = out[mgmt_home_col].isna()
    elif mgmt_away_col:
        out["management_missing"] = out[mgmt_away_col].isna()
    else:
        out["management_missing"] = True

    out["data_quality_warn"] = out[
        ["weather_missing", "stats_home_missing", "stats_away_missing", "management_missing"]
    ].any(axis=1)
    return out


def drop_internal_output_columns(df):
    out = df.copy()
    drop_exact = {
        "team_name",
        "_merge_team_name",
        "datetime_r",
        "datetime_x",
        "datetime_y",
        "_merge_home_team",
        "_merge_away_team",
    }
    drop_cols = [c for c in out.columns if c in drop_exact or c.startswith("_merge_")]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    return out

# 外部スタッツデータをマージする関数
def merge_external_stats(
    df,
    stats_csv_path,
    team_name_col="team_name",
    merge_col_prefix="",
    stage_label="external",
):
    try:
        external_stats = pd.read_csv(stats_csv_path)

        if team_name_col != "team_name":
            external_stats = external_stats.rename(columns={team_name_col: "team_name"})

        external_stats["team_name"] = external_stats["team_name"].astype(str).str.strip()
        external_stats["_merge_team_name"] = normalize_team_series(external_stats["team_name"])

        # マージ対象カラム抽出
        stats_cols = [
            col for col in external_stats.columns
            if col not in ["team_name", "_merge_team_name", "match_date"]
        ]

        # 正規化キー重複を解消（例: 栃木SC / 栃木Ｃ / 栃木シティ -> 栃木C）
        dup_count = int(external_stats.duplicated(subset=["_merge_team_name"]).sum())
        if dup_count:
            print(
                f"[MERGE_QC][WARN] {stage_label}: 右側重複キー={dup_count} "
                "-> 情報量が多い行を優先して重複排除"
            )
            # 非欠損が多い行を優先し、同率は後勝ち（最新行）にする
            external_stats["_non_na_score"] = external_stats[stats_cols].notna().sum(axis=1)
            external_stats = (
                external_stats
                .sort_values(by=["_merge_team_name", "_non_na_score"], ascending=[True, False], kind="mergesort")
                .drop_duplicates(subset=["_merge_team_name"], keep="first")
                .drop(columns=["_non_na_score"], errors="ignore")
            )

        # HOME 用データ作成
        home_stats = external_stats.copy()
        home_rename = {
            col: f"{merge_col_prefix}{col}_home"
            for col in stats_cols
        }
        home_stats = home_stats.rename(columns=home_rename)

        # AWAY 用データ作成
        away_stats = external_stats.copy()
        away_rename = {
            col: f"{merge_col_prefix}{col}_away"
            for col in stats_cols
        }
        away_stats = away_stats.rename(columns=away_rename)

        out = df.copy()
        out["_merge_home_team"] = normalize_team_series(out["home_team"])
        out["_merge_away_team"] = normalize_team_series(out["away_team"])

        # HOME merge
        out = audited_left_merge(
            out,
            home_stats[["_merge_team_name"] + list(home_rename.values())],
            stage=f"{stage_label}_home",
            left_on="_merge_home_team",
            right_on="_merge_team_name",
            validate="many_to_one",
        )
        out = out.drop(columns=["_merge_team_name"], errors="ignore")

        # AWAY merge
        out = audited_left_merge(
            out,
            away_stats[["_merge_team_name"] + list(away_rename.values())],
            stage=f"{stage_label}_away",
            left_on="_merge_away_team",
            right_on="_merge_team_name",
            validate="many_to_one",
        )
        out = out.drop(columns=["_merge_team_name"], errors="ignore")

        out = out.drop(columns=["_merge_home_team", "_merge_away_team"], errors="ignore")
        return out

    except FileNotFoundError:
        print(f"警告: 外部スタッツファイル '{stats_csv_path}' が見つかりません。スキップ。")
        return df

    except Exception as e:
        print(f"エラー: 外部スタッツのマージ中にエラー発生: {e}")
        raise

# 前年結果読み込み（任意）
if os.path.exists(csv_prev):
    df_prev = pd.read_csv(csv_prev)
    print(f"前年データを読み込みました: {csv_prev}")
else:
    df_prev = pd.DataFrame(columns=["home_team", "away_team", "home_score", "away_score"])
    print("前年データは未使用（ファイルなし）。")

df_2025 = pd.read_csv(csv_season)
df_2025 = enrich_scores_from_latest_results(df_2025, csv_season_latest)

# datetime 列の補完（date列がある場合）
if "datetime" not in df_2025.columns and "date" in df_2025.columns:
    df_2025["datetime"] = pd.to_datetime(df_2025["date"], errors="coerce")
if "datetime" in df_2025.columns:
    df_2025["datetime"] = pd.to_datetime(df_2025["datetime"], errors="coerce")

# スコア付き行のみ抽出（素材化）
df_prev = df_prev.dropna(subset=["home_score", "away_score"])
df_2025_finished = df_2025.dropna(subset=["home_score", "away_score"])
df_2025_future = df_2025[df_2025["home_score"].isna()]
_ensure_merge_qc_dir()
export_team_name_diff(df_2025, team_master_stats_csv)

# 疲労度スコア（試合単位）をマージ
try:
    fatigue_scores_df = pd.read_csv(team_fatigue_scores_csv)
    if "datetime" in fatigue_scores_df.columns:
        fatigue_scores_df["datetime"] = pd.to_datetime(fatigue_scores_df["datetime"], errors="coerce")

    merge_keys = ["datetime", "home_team", "away_team"]
    fatigue_cols = ["home_fatigue_score", "away_fatigue_score"]
    fatigue_merge_df = fatigue_scores_df[merge_keys + fatigue_cols].copy()
    dup = int(fatigue_merge_df.duplicated(subset=merge_keys).sum())
    if dup:
        print(f"[MERGE_QC][WARN] fatigue: 右側重複キー={dup} -> 最後の行を採用して重複排除")
        fatigue_merge_df = fatigue_merge_df.drop_duplicates(subset=merge_keys, keep="last")

    df_2025_future = audited_left_merge(
        df_2025_future,
        fatigue_merge_df,
        stage="fatigue_future",
        on=merge_keys,
        validate="one_to_one",
    )
    df_2025_finished = audited_left_merge(
        df_2025_finished,
        fatigue_merge_df,
        stage="fatigue_finished",
        on=merge_keys,
        validate="one_to_one",
    )
    report_missing_rates(df_2025_future, "after_fatigue_future")
    report_missing_rates(df_2025_finished, "after_fatigue_finished")
except FileNotFoundError:
    print(f"警告: 疲労度ファイル '{team_fatigue_scores_csv}' が見つかりませんでした。スキップします。")
except Exception as e:
    print(f"エラー: 疲労度のマージ中にエラーが発生しました: {e}")
    raise

# 天候キャッシュ（match_idキー）をマージ
if os.path.exists(weather_cache_csv):
    try:
        weather_df = pd.read_csv(weather_cache_csv)
        weather_df = normalize_weather_cache_columns(weather_df)
        required_weather_cols = ["match_id", "is_rain", "is_heavy_rain", "is_strong_wind"]
        for c in required_weather_cols:
            if c not in weather_df.columns:
                weather_df[c] = pd.NA
        keep_cols = [c for c in ["match_id", "datetime", "stadium", "is_rain", "is_heavy_rain", "is_strong_wind", "temperature", "wind_speed", "last_updated_at"] if c in weather_df.columns]
        weather_merge_df = weather_df[keep_cols].copy()
        weather_dup = int(weather_merge_df.duplicated(subset=["match_id"]).sum())
        if weather_dup:
            print(f"[MERGE_QC][WARN] weather_cache: 右側重複キー={weather_dup} -> 最後の行を採用")
            weather_merge_df = weather_merge_df.drop_duplicates(subset=["match_id"], keep="last")

        df_2025_future = merge_weather_cache(df_2025_future, weather_merge_df, stage="weather_cache_future")
        df_2025_finished = merge_weather_cache(df_2025_finished, weather_merge_df, stage="weather_cache_finished")
        report_missing_rates(df_2025_future, "after_weather_cache_future")
        report_missing_rates(df_2025_finished, "after_weather_cache_finished")
        print(f"天候キャッシュを {weather_cache_csv} から読み込みました。")
    except Exception as e:
        print(f"エラー: 天候キャッシュのマージ中にエラーが発生しました: {e}")
        raise
else:
    print(f"天候キャッシュが見つかりません: {weather_cache_csv}")
    df_2025_future["is_rain"] = False
    df_2025_future["is_heavy_rain"] = False
    df_2025_future["is_strong_wind"] = False
    df_2025_future["weather_missing"] = True
    df_2025_finished["is_rain"] = False
    df_2025_finished["is_heavy_rain"] = False
    df_2025_finished["is_strong_wind"] = False
    df_2025_finished["weather_missing"] = True

# team_master_stats.csv をマージ
df_2025_future = merge_external_stats(
    df_2025_future,
    team_master_stats_csv,
    team_name_col="team_name",
    merge_col_prefix="stats_",
    stage_label="stats_future",
)
df_2025_finished = merge_external_stats(
    df_2025_finished,
    team_master_stats_csv,
    team_name_col="team_name",
    merge_col_prefix="stats_",
    stage_label="stats_finished",
)
df_2025_future = drop_j2_excluded_stats_columns(df_2025_future)
df_2025_finished = drop_j2_excluded_stats_columns(df_2025_finished)
report_missing_rates(df_2025_future, "after_stats_future")
report_missing_rates(df_2025_finished, "after_stats_finished")

# team_management_master.csv をマージ
df_2025_future = merge_external_stats(
    df_2025_future,
    team_management_master_csv,
    team_name_col="team_name",
    merge_col_prefix="management_",
    stage_label="management_future",
)
df_2025_finished = merge_external_stats(
    df_2025_finished,
    team_management_master_csv,
    team_name_col="team_name",
    merge_col_prefix="management_",
    stage_label="management_finished",
)
report_missing_rates(df_2025_future, "after_management_future")
report_missing_rates(df_2025_finished, "after_management_finished")

# ランキング推移由来モチベーションをマージ
df_2025_future = merge_external_stats(
    df_2025_future,
    team_motivation_csv,
    team_name_col="team_name",
    merge_col_prefix="rankmot_",
    stage_label="rankmot_future",
)
df_2025_finished = merge_external_stats(
    df_2025_finished,
    team_motivation_csv,
    team_name_col="team_name",
    merge_col_prefix="rankmot_",
    stage_label="rankmot_finished",
)
report_missing_rates(df_2025_future, "after_rankmot_future")
report_missing_rates(df_2025_finished, "after_rankmot_finished")

# 欠場影響（absences_with_impact.csv）を節×チームでマージ
match_rounds = set()
if "節" in df_2025_future.columns:
    match_rounds |= set(df_2025_future["節"].map(extract_round_number).dropna().astype(int).tolist())
if "節" in df_2025_finished.columns:
    match_rounds |= set(df_2025_finished["節"].map(extract_round_number).dropna().astype(int).tolist())
absence_map_df = load_absence_impact_team_round_map(absence_impact_csv, match_rounds)
df_2025_future = merge_absence_impacts(df_2025_future, absence_map_df, stage_label="absence_future")
df_2025_finished = merge_absence_impacts(df_2025_finished, absence_map_df, stage_label="absence_finished")
report_missing_rates(df_2025_future, "after_absence_future")
report_missing_rates(df_2025_finished, "after_absence_finished")

# team_travel_distances.csv を読み込み、データフレームとして準備 (行列形式)
# これはルックアップテーブルとして使用
try:
    travel_distances_df = pd.read_csv(team_travel_distances_csv, sep='	')
    travel_distances_df = travel_distances_df.set_index('ホーム　／　アウェイ')
    travel_distances_df = normalize_travel_distance_matrix(travel_distances_df)
    print(f"移動距離データを {team_travel_distances_csv} から読み込みました。")
except FileNotFoundError:
    print(f"警告: 移動距離ファイル '{team_travel_distances_csv}' が見つかりませんでした。移動距離データは使用されません。")
    travel_distances_df = pd.DataFrame() # ファイルがない場合は空のDataFrameを設定
except Exception as e:
    print(f"警告: 移動距離データの読み込み中にエラーが発生しました: {e}。移動距離データは使用されません。")
    travel_distances_df = pd.DataFrame()


# Elo初期化
all_teams = set(df_2025["home_team"].tolist() + df_2025["away_team"].tolist())
if not df_prev.empty:
    all_teams |= set(df_prev["home_team"].tolist() + df_prev["away_team"].tolist())

# 前年最終ELOを初期値として使用（未存在チームのみ1500）
prev_final_elo_map = load_or_build_prev_final_elo(df_prev)
elo_base = {team: float(prev_final_elo_map.get(team, INITIAL_ELO)) for team in all_teams}

# 予測用Elo（当年終了済みを反映）
elo_for_prediction = dict(elo_base)
df_2025_finished_for_elo = sort_results_for_elo(df_2025_finished)
for _, row in df_2025_finished_for_elo.iterrows():
    home, away = row["home_team"], row["away_team"]
    hs, as_ = row["home_score"], row["away_score"]
    if home not in elo_for_prediction:
        elo_for_prediction[home] = INITIAL_ELO
    if away not in elo_for_prediction:
        elo_for_prediction[away] = INITIAL_ELO
    result = get_result(hs, as_)
    if result:
        elo_for_prediction[home], elo_for_prediction[away] = update_elo(
            elo_for_prediction[home], elo_for_prediction[away], result
        )

df_all_results = pd.concat([df_prev, df_2025_finished], ignore_index=True)

# STEP1: チーム別ホーム/アウェイ成績を作成
home_ppm_map, away_ppm_map = build_home_away_profile_map(df_all_results)
allowed_teams = load_allowed_teams()

# 予測
predictions = []
elo_debug_rows = []
for _, row in df_2025_future.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    if home not in elo_for_prediction:
        elo_for_prediction[home] = INITIAL_ELO
    if away not in elo_for_prediction:
        elo_for_prediction[away] = INITIAL_ELO
    home_advantage_profile_diff, _ = calc_home_advantage_diff(
        home, away, home_ppm_map, away_ppm_map
    )

    home_canon = canonical_team_name(home)
    away_canon = canonical_team_name(away)
    if allowed_teams is not None and (home_canon not in allowed_teams or away_canon not in allowed_teams):
        # 対象リーグ外カードは出力対象外
        continue

    # team_master_stats からゴール期待値（xG）を取得。存在しない場合はNone
    home_xg_stats = row.get("stats_ゴール期待値_home")
    away_xg_stats = row.get("stats_ゴール期待値_away")

    # 移動距離を取得。存在しない場合は0
    home_travel_distance = 0
    away_travel_distance = 0
    if not travel_distances_df.empty:
        # travel_distances_dfは行列形式なので、locで直接アクセス
        home_key = canonical_team_name(home)
        away_key = canonical_team_name(away)
        if home_key in travel_distances_df.index and away_key in travel_distances_df.columns:
            home_travel_distance = travel_distances_df.loc[home_key, away_key]
        if away_key in travel_distances_df.index and home_key in travel_distances_df.columns:
            away_travel_distance = travel_distances_df.loc[away_key, home_key]

    # 疲労度スコアを取得。存在しない場合はNone
    home_fatigue_score = row.get("home_fatigue_score")
    away_fatigue_score = row.get("away_fatigue_score")
    home_rank_motivation_score = _pick_first_non_na_value(
        row,
        ["rankmot_motivation_score_3w", "rankmot_motivation_score_5w"],
    )
    away_rank_motivation_score = _pick_first_non_na_value(
        row,
        ["rankmot_motivation_score_3w_away", "rankmot_motivation_score_5w_away"],
    )
    absence_effective = compute_effective_absence_impacts(row)
    home_absence_impact = absence_effective["absence_effective_total_home"]
    away_absence_impact = absence_effective["absence_effective_total_away"]

    # 天候フラグを取得
    weather_flags = {
        "is_rain": bool(row.get("is_rain")) if pd.notna(row.get("is_rain")) else False,
        "is_heavy_rain": bool(row.get("is_heavy_rain")) if pd.notna(row.get("is_heavy_rain")) else False,
        "is_strong_wind": bool(row.get("is_strong_wind")) if pd.notna(row.get("is_strong_wind")) else False,
    }

    quality_flags = compute_row_quality_flags(row)

    # 共通ロジックで確率計算（raw）
    prob_home_win_raw, prob_draw_raw, prob_away_win_raw, _, debug_row = compute_probabilities_and_result(
        row.get("match_id"),
        elo_for_prediction[home],
        elo_for_prediction[away],
        home_advantage_profile_diff,
        home_xg_stats,
        away_xg_stats,
        home_travel_distance,
        away_travel_distance,
        home_fatigue_score,
        away_fatigue_score,
        home_rank_motivation_score,
        away_rank_motivation_score,
        home_absence_impact,
        away_absence_impact,
        weather_flags,
        quality_flags["stats_home_missing"],
        quality_flags["stats_away_missing"],
        quality_flags["data_quality_warn"],
    )
    if HDA_MODEL_MODE_EFFECTIVE == "multinom":
        prob_home_win, prob_draw, prob_away_win = _normalize_probs(
            prob_home_win_raw, prob_draw_raw, prob_away_win_raw
        )
    else:
        prob_home_win, prob_draw, prob_away_win = calibrate_probabilities(
            prob_home_win_raw,
            prob_draw_raw,
            prob_away_win_raw,
            row.get("league", LEAGUE),
        )
    prob_home_win, prob_draw, prob_away_win = _normalize_probs(prob_home_win, prob_draw, prob_away_win)
    predicted_result, decision_reason, decision_metrics = decide_result(prob_home_win, prob_draw, prob_away_win)
    argmax_result = decision_metrics.get("argmax_result")
    argmax_max_prob = decision_metrics.get("argmax_max_prob")
    predicted_highest_prob_result, _, raw_decision_metrics = decide_result(
        prob_home_win_raw, prob_draw_raw, prob_away_win_raw
    )
    argmax_max_prob_raw = raw_decision_metrics.get("argmax_max_prob")
    if not np.isclose(prob_home_win + prob_draw + prob_away_win, 1.0, atol=1e-6):
        print(
            f"[PROB_QC][WARN] match_id={row.get('match_id')} calibrated_prob_sum="
            f"{(prob_home_win + prob_draw + prob_away_win):.9f}"
        )

    debug_row.update(
        {
            "prob_home_win_raw": prob_home_win_raw,
            "prob_draw_raw": prob_draw_raw,
            "prob_away_win_raw": prob_away_win_raw,
            "prob_home_win_cal": prob_home_win,
            "prob_draw_cal": prob_draw,
            "prob_away_win_cal": prob_away_win,
            "predicted_result_cal": predicted_result,
            "decision_reason_cal": decision_reason,
            "argmax_result_cal": argmax_result,
            "argmax_max_prob_cal": argmax_max_prob,
        }
    )
    elo_debug_rows.append({**debug_row, "phase": "prediction"})

    # profile差分と確率入力elo差分は別管理する（上書きしない）
    home_advantage_diff = home_advantage_profile_diff
    is_home_advantage_positive = home_advantage_diff > 0

    predictions.append({
        **row.to_dict(),
        **absence_effective,
        "stats_asof": STATS_ASOF_LABEL,
        "stats_source_csv": os.path.basename(team_master_stats_csv) if team_master_stats_csv else "",
        "home_elo": round(elo_for_prediction[home]),
        "away_elo": round(elo_for_prediction[away]),
        "home_advantage_diff": round(home_advantage_diff, 4),
        "home_advantage_profile_diff": round(home_advantage_profile_diff, 4),
        "hfa_applied_elo": round(debug_row["HFA_applied"], 4),
        "hfa_added_to_diff": round(debug_row["hfa_added_to_diff"], 4),
        "hfa_clip_min": round(debug_row["hfa_clip_min"], 4),
        "hfa_clip_max": round(debug_row["hfa_clip_max"], 4),
        "elo_diff_scale_factor": round(debug_row["elo_diff_scale_factor"], 4),
        "elo_diff_before_hfa": round(debug_row["elo_diff_before_hfa"], 4),
        "elo_diff_after_hfa": round(debug_row["elo_diff_after_hfa"], 4),
        "elo_diff_scaled": round(debug_row["elo_diff_scaled"], 4),
        "elo_diff_for_prob": round(debug_row["elo_diff"], 4),
        "elo_diff_used_for_prob": round(debug_row["elo_diff_for_prob"], 4),
        "expected_home_two_way": round(debug_row["expected_home"], 4),
        "is_home_advantage_positive": bool(is_home_advantage_positive),
        "prob_home_win_raw": prob_home_win_raw,
        "prob_draw_raw": prob_draw_raw,
        "prob_away_win_raw": prob_away_win_raw,
        "prob_home_raw": prob_home_win_raw,
        "prob_away_raw": prob_away_win_raw,
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "prob_home": prob_home_win,
        "prob_away": prob_away_win,
        "final_result": predicted_result,
        "predicted_result": predicted_result,
        "decision_reason": decision_reason,
        "argmax_result": argmax_result,
        "argmax_max_prob": argmax_max_prob,
        "argmax_raw_result": predicted_highest_prob_result,
        "argmax_raw_max_prob": argmax_max_prob_raw,
        "d_scaled": debug_row.get("draw_model_input"),
        "decision_draw_expectation_multiplier": DRAW_EXPECTATION_MULTIPLIER,
        "decision_draw_assign_enabled": bool(DRAW_ASSIGN_BY_EXPECTATION),
        "predicted_highest_prob_result": predicted_highest_prob_result,
    })

# 保存
df_pred = pd.DataFrame(predictions)
df_pred = add_data_quality_flags(df_pred)
df_pred = recalculate_predicted_result(df_pred, "predicted_result")
df_pred = recalculate_predicted_highest_prob_result(df_pred, "predicted_highest_prob_result")
if DRAW_ASSIGN_BY_EXPECTATION:
    # 最終ラベルは「調整後確率」をベースに、節単位の期待ドロー数へ合わせてDを付与する
    df_pred = assign_draw_results_by_expectation(df_pred, "final_result")
else:
    print("[DRAW_ASSIGN] disabled (DRAW_ASSIGN_BY_EXPECTATION=0)")
df_pred = sync_and_validate_prediction_results(df_pred, "PRED", raise_on_error=True)
log_decision_rule_once()
log_pred_dist(df_pred, "PRED", scope="all")
try:
    round_mask_pred, round_label_pred, _ = _resolve_round_filter(df_pred)
    if len(round_mask_pred) == len(df_pred) and int(round_mask_pred.sum()) > 0:
        log_pred_dist(df_pred.loc[round_mask_pred], "PRED", scope=f"round:{round_label_pred}")
except Exception as e:
    print(f"[PRED_DIST:PRED][WARN] round scope unavailable: {e}")
log_prediction_consistency(df_pred, "PRED")
log_prob_summary(df_pred, "PRED_SUMMARY")
log_prob_draw_distribution(df_pred, "PRED")
log_draw_argmax_stats(df_pred, "PRED")
log_actual_hda_ratio(df_pred, "PRED")
log_absence_effective_summary(df_pred, "PRED")
summarize_round_hda(df_pred, df_results=df_2025, round_filter_label="auto")
dump_decision_artifacts(df_pred, label="pred", threshold=0.25)
df_pred = drop_internal_output_columns(df_pred)
report_missing_rates(df_pred, "final_predictions_df")
df_pred.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"予測結果を {output_csv} に出力しました。")
if output_csv != LEGACY_OUTPUT_CSV:
    df_pred.to_csv(LEGACY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[LEGACY_ALIAS] 互換出力を更新しました: {LEGACY_OUTPUT_CSV}")

# --- 2025年終了済み試合のバックテストと的中率計算 ---
backtest_results = []
correct_predictions = 0
total_finished_games = 0

# バックテストはリーク防止：時系列で「予測→採点→Elo更新」
elo_for_backtest = dict(elo_base)
for _, row in sort_results_for_elo(df_2025_finished).iterrows():
    home, away = row["home_team"], row["away_team"]
    hs, as_ = row["home_score"], row["away_score"]
    if home not in elo_for_backtest:
        elo_for_backtest[home] = INITIAL_ELO
    if away not in elo_for_backtest:
        elo_for_backtest[away] = INITIAL_ELO
    
    # 実際の試合結果
    actual_result = get_result(hs, as_)
    home_advantage_profile_diff, _ = calc_home_advantage_diff(
        home, away, home_ppm_map, away_ppm_map
    )

    home_canon = canonical_team_name(home)
    away_canon = canonical_team_name(away)
    if allowed_teams is not None and (home_canon not in allowed_teams or away_canon not in allowed_teams):
        # 対象リーグ外カードは出力対象外
        continue

    # team_master_stats からゴール期待値（xG）を取得。存在しない場合はNone
    home_xg_stats = row.get("stats_ゴール期待値_home")
    away_xg_stats = row.get("stats_ゴール期待値_away")

    # 移動距離を取得。存在しない場合は0
    home_travel_distance = 0
    away_travel_distance = 0
    if not travel_distances_df.empty:
        home_key = canonical_team_name(home)
        away_key = canonical_team_name(away)
        if home_key in travel_distances_df.index and away_key in travel_distances_df.columns:
            home_travel_distance = travel_distances_df.loc[home_key, away_key]
        if away_key in travel_distances_df.index and home_key in travel_distances_df.columns:
            away_travel_distance = travel_distances_df.loc[away_key, home_key]

    # 疲労度スコアを取得。存在しない場合はNone
    home_fatigue_score = row.get("home_fatigue_score")
    away_fatigue_score = row.get("away_fatigue_score")
    home_rank_motivation_score = _pick_first_non_na_value(
        row,
        ["rankmot_motivation_score_3w", "rankmot_motivation_score_5w"],
    )
    away_rank_motivation_score = _pick_first_non_na_value(
        row,
        ["rankmot_motivation_score_3w_away", "rankmot_motivation_score_5w_away"],
    )
    absence_effective = compute_effective_absence_impacts(row)
    home_absence_impact = absence_effective["absence_effective_total_home"]
    away_absence_impact = absence_effective["absence_effective_total_away"]

    # 天候フラグを取得
    weather_flags = {
        "is_rain": bool(row.get("is_rain")) if pd.notna(row.get("is_rain")) else False,
        "is_heavy_rain": bool(row.get("is_heavy_rain")) if pd.notna(row.get("is_heavy_rain")) else False,
        "is_strong_wind": bool(row.get("is_strong_wind")) if pd.notna(row.get("is_strong_wind")) else False,
    }

    quality_flags = compute_row_quality_flags(row)

    # 共通ロジックで確率計算（raw）
    prob_home_win_raw, prob_draw_raw, prob_away_win_raw, _, debug_row = compute_probabilities_and_result(
        row.get("match_id"),
        elo_for_backtest[home],
        elo_for_backtest[away],
        home_advantage_profile_diff,
        home_xg_stats,
        away_xg_stats,
        home_travel_distance,
        away_travel_distance,
        home_fatigue_score,
        away_fatigue_score,
        home_rank_motivation_score,
        away_rank_motivation_score,
        home_absence_impact,
        away_absence_impact,
        weather_flags,
        quality_flags["stats_home_missing"],
        quality_flags["stats_away_missing"],
        quality_flags["data_quality_warn"],
    )
    if HDA_MODEL_MODE_EFFECTIVE == "multinom":
        prob_home_win, prob_draw, prob_away_win = _normalize_probs(
            prob_home_win_raw, prob_draw_raw, prob_away_win_raw
        )
    else:
        prob_home_win, prob_draw, prob_away_win = calibrate_probabilities(
            prob_home_win_raw,
            prob_draw_raw,
            prob_away_win_raw,
            row.get("league", LEAGUE),
        )
    prob_home_win, prob_draw, prob_away_win = _normalize_probs(prob_home_win, prob_draw, prob_away_win)
    predicted_label, decision_reason_bt, decision_metrics_bt = decide_result(prob_home_win, prob_draw, prob_away_win)
    argmax_result_bt = decision_metrics_bt.get("argmax_result")
    argmax_max_prob_bt = decision_metrics_bt.get("argmax_max_prob")
    predicted_highest_prob_result, _, raw_decision_metrics_bt = decide_result(
        prob_home_win_raw, prob_draw_raw, prob_away_win_raw
    )
    argmax_max_prob_raw_bt = raw_decision_metrics_bt.get("argmax_max_prob")
    if not np.isclose(prob_home_win + prob_draw + prob_away_win, 1.0, atol=1e-6):
        print(
            f"[PROB_QC][WARN] match_id={row.get('match_id')} calibrated_prob_sum="
            f"{(prob_home_win + prob_draw + prob_away_win):.9f}"
        )
    debug_row.update(
        {
            "prob_home_win_raw": prob_home_win_raw,
            "prob_draw_raw": prob_draw_raw,
            "prob_away_win_raw": prob_away_win_raw,
            "prob_home_win_cal": prob_home_win,
            "prob_draw_cal": prob_draw,
            "prob_away_win_cal": prob_away_win,
            "predicted_result_cal": predicted_label,
            "decision_reason_cal": decision_reason_bt,
            "argmax_result_cal": argmax_result_bt,
            "argmax_max_prob_cal": argmax_max_prob_bt,
        }
    )
    elo_debug_rows.append({**debug_row, "phase": "backtest"})
    home_advantage_diff = home_advantage_profile_diff
    is_home_advantage_positive = home_advantage_diff > 0
    is_correct = (predicted_label == actual_result) if actual_result else False
    
    backtest_results.append({
        **row.to_dict(),
        **absence_effective,
        "stats_asof": STATS_ASOF_LABEL,
        "stats_source_csv": os.path.basename(team_master_stats_csv) if team_master_stats_csv else "",
        "home_elo_at_prediction": round(elo_for_backtest[home]), # 予測時のEloスコアを記録
        "away_elo_at_prediction": round(elo_for_backtest[away]),
        "home_advantage_diff": round(home_advantage_diff, 4),
        "home_advantage_profile_diff": round(home_advantage_profile_diff, 4),
        "hfa_applied_elo": round(debug_row["HFA_applied"], 4),
        "hfa_added_to_diff": round(debug_row["hfa_added_to_diff"], 4),
        "hfa_clip_min": round(debug_row["hfa_clip_min"], 4),
        "hfa_clip_max": round(debug_row["hfa_clip_max"], 4),
        "elo_diff_scale_factor": round(debug_row["elo_diff_scale_factor"], 4),
        "elo_diff_before_hfa": round(debug_row["elo_diff_before_hfa"], 4),
        "elo_diff_after_hfa": round(debug_row["elo_diff_after_hfa"], 4),
        "elo_diff_scaled": round(debug_row["elo_diff_scaled"], 4),
        "elo_diff_for_prob": round(debug_row["elo_diff"], 4),
        "elo_diff_used_for_prob": round(debug_row["elo_diff_for_prob"], 4),
        "expected_home_two_way": round(debug_row["expected_home"], 4),
        "is_home_advantage_positive": bool(is_home_advantage_positive),
        "prob_home_win_raw": prob_home_win_raw,
        "prob_draw_raw": prob_draw_raw,
        "prob_away_win_raw": prob_away_win_raw,
        "prob_home_raw": prob_home_win_raw,
        "prob_away_raw": prob_away_win_raw,
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "prob_home": prob_home_win,
        "prob_away": prob_away_win,
        "final_result": predicted_label,
        "predicted_result": predicted_label,
        "decision_reason": decision_reason_bt,
        "argmax_result": argmax_result_bt,
        "argmax_max_prob": argmax_max_prob_bt,
        "argmax_raw_result": predicted_highest_prob_result,
        "argmax_raw_max_prob": argmax_max_prob_raw_bt,
        "d_scaled": debug_row.get("draw_model_input"),
        "decision_draw_expectation_multiplier": DRAW_EXPECTATION_MULTIPLIER,
        "decision_draw_assign_enabled": bool(DRAW_ASSIGN_BY_EXPECTATION),
        "predicted_highest_prob_result": predicted_highest_prob_result,
        "actual_result": actual_result,
        "is_correct": is_correct
    })

    if actual_result: # 実際の試合結果がある場合のみ的中率に含める
        total_finished_games += 1
        if is_correct:
            correct_predictions += 1
        elo_for_backtest[home], elo_for_backtest[away] = update_elo(
            elo_for_backtest[home], elo_for_backtest[away], actual_result
        )

df_backtest = pd.DataFrame(backtest_results)
if "stats_asof" not in df_backtest.columns:
    df_backtest["stats_asof"] = pd.Series(dtype="object")
if "stats_source_csv" not in df_backtest.columns:
    df_backtest["stats_source_csv"] = pd.Series(dtype="object")
df_backtest = add_data_quality_flags(df_backtest)
df_backtest = recalculate_predicted_result(df_backtest, "predicted_result")
df_backtest = recalculate_predicted_highest_prob_result(df_backtest, "predicted_highest_prob_result")
if DRAW_ASSIGN_BY_EXPECTATION:
    df_backtest = assign_draw_results_by_expectation(df_backtest, "final_result")
df_backtest = sync_and_validate_prediction_results(df_backtest, "BACKTEST", raise_on_error=True)
log_pred_dist(df_backtest, "BACKTEST", scope="all")
log_prediction_consistency(df_backtest, "BACKTEST")
log_prob_summary(df_backtest, "BACKTEST_SUMMARY")
log_multinom_feature_distribution(df_backtest, "BACKTEST")
log_prob_draw_distribution(df_backtest, "BACKTEST")
log_prob_distribution(df_backtest, "BACKTEST", "prob_home")
log_prob_distribution(df_backtest, "BACKTEST", "prob_draw")
log_prob_distribution(df_backtest, "BACKTEST", "prob_away")
log_max_prob_distribution(df_backtest, "BACKTEST")
log_draw_argmax_stats(df_backtest, "BACKTEST")
log_actual_hda_ratio(df_backtest, "BACKTEST")
log_absence_effective_summary(df_backtest, "BACKTEST")
if "actual_result" in df_backtest.columns and "predicted_result" in df_backtest.columns:
    df_backtest["is_correct"] = df_backtest["actual_result"] == df_backtest["predicted_result"]
    finished_mask = df_backtest["actual_result"].notna()
    total_finished_games = int(finished_mask.sum())
    correct_predictions = int(df_backtest.loc[finished_mask, "is_correct"].sum())
df_backtest = drop_internal_output_columns(df_backtest)
report_missing_rates(df_backtest, "final_backtest_df")
df_backtest.to_csv(backtest_output_csv, index=False, encoding="utf-8-sig")
print(f"2025年終了済み試合のバックテスト結果を {backtest_output_csv} に出力しました。")
print(
    f"[HFA_APPLY_COUNT] applied={HFA_APPLY_COUNTER['applied']} "
    f"skipped={HFA_APPLY_COUNTER['skipped']} "
    f"reason_counts={json.dumps(HFA_APPLY_COUNTER['reason_counts'], ensure_ascii=False, sort_keys=True)}"
)
if ENABLE_HFA and float(HFA_ELO) > 0:
    evaluated = int(HFA_APPLY_COUNTER["applied"]) + int(HFA_APPLY_COUNTER["skipped"])
    if evaluated <= 0:
        print("[ERROR] HFA apply counter has zero evaluated rows under ENABLE_HFA=1 and HFA_ELO>0")
        raise RuntimeError("HFA apply counter invalid: no evaluated rows")

try:
    run_meta = {
        "output_csv": output_csv,
        "league": LEAGUE,
        "season_year": int(SEASON_YEAR),
        "enable_hfa": int(ENABLE_HFA_INT),
        "hfa_elo": float(HFA_ELO),
        "pred_rows": int(len(df_pred)) if "df_pred" in globals() else 0,
        "hfa_apply_count": {
            "applied": int(HFA_APPLY_COUNTER["applied"]),
            "skipped": int(HFA_APPLY_COUNTER["skipped"]),
            "reason_counts": HFA_APPLY_COUNTER["reason_counts"],
        },
    }
    meta_path = f"{output_csv}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[RUN_META] saved={meta_path}")
except Exception as e:
    print(f"[RUN_META][WARN] failed to write meta: {e}")

if elo_debug_rows:
    debug_df = pd.DataFrame(elo_debug_rows)
    debug_csv = os.path.join(REPORT_DIR, f"elo_prob_debug_{LEAGUE}_{SEASON_YEAR}.csv")
    debug_df.to_csv(debug_csv, index=False, encoding="utf-8-sig")
    print(f"Elo確率デバッグCSVを出力しました: {debug_csv}")

# 的中率の表示
if total_finished_games > 0:
    accuracy = (correct_predictions / total_finished_games) * 100
    print(f"2025年シーズン現時点までの累計的中率: {accuracy:.2f}% ({correct_predictions}/{total_finished_games})")
else:
    print("2025年シーズンの終了済み試合がありません。")


# --- 解析内訳レポート出力 ---
def build_report():
    global df_pred, df_backtest
    if "df_pred" not in globals():
        df_pred = pd.DataFrame()
    if "df_backtest" not in globals():
        df_backtest = pd.DataFrame()
    inputs = {
        "csv_prev": csv_prev if os.path.exists(csv_prev) else None,
        "prev_final_elo_csv": prev_final_elo_csv if os.path.exists(prev_final_elo_csv) else None,
        "csv_season": csv_season,
        "team_master_stats_csv": team_master_stats_csv,
        "stats_asof": STATS_ASOF_LABEL,
        "absence_impact_csv": absence_impact_csv if absence_impact_csv and os.path.exists(absence_impact_csv) else None,
        "absence_asof": absence_asof_key if absence_asof_key else None,
        "team_management_master_csv": team_management_master_csv if os.path.exists(team_management_master_csv) else None,
        "team_motivation_csv": team_motivation_csv if os.path.exists(team_motivation_csv) else None,
        "team_travel_distances_csv": team_travel_distances_csv if os.path.exists(team_travel_distances_csv) else None,
        "team_fatigue_scores_csv": team_fatigue_scores_csv if os.path.exists(team_fatigue_scores_csv) else None,
        "weather_cache_csv": weather_cache_csv if os.path.exists(weather_cache_csv) else None,
        "weather_asof": weather_asof_key if weather_asof_key else None,
    }

    params = {
        "INITIAL_ELO": INITIAL_ELO,
        "ELO_UPDATE_HOME_ADVANTAGE": ELO_UPDATE_HOME_ADVANTAGE,
        "HFA_ELO": HFA_ELO,
        "ENABLE_HFA": ENABLE_HFA_INT,
        "ENABLE_MATCHUP_BIAS": int(ENABLE_MATCHUP_BIAS),
        "MATCHUP_BIAS_COEF": MATCHUP_BIAS_COEF,
        "HOME_ADV_ELO_COEF": HOME_ADV_ELO_COEF,
        "HOME_ADV_PROFILE_DIFF_CLIP": HOME_ADV_PROFILE_DIFF_CLIP,
        "ELO_DIFF_TEMPERATURE": ELO_DIFF_TEMPERATURE,
        "J1_WIN_PROB_CAP": J1_WIN_PROB_CAP,
        "GOAL_SCALING_FACTOR": GOAL_SCALING_FACTOR,
        "FATIGUE_GOAL_SCALING": FATIGUE_GOAL_SCALING,
        "RANK_MOTIVATION_GOAL_SCALING": RANK_MOTIVATION_GOAL_SCALING,
        "WEATHER_PENALTY_HEAVY_RAIN": WEATHER_PENALTY_HEAVY_RAIN,
        "WEATHER_PENALTY_RAIN": WEATHER_PENALTY_RAIN,
        "WEATHER_PENALTY_STRONG_WIND": WEATHER_PENALTY_STRONG_WIND,
        "STATS_ASOF_DATE": STATS_ASOF_DATE,
        "STATS_SNAPSHOT_NAME": STATS_SNAPSHOT_NAME,
    }

    summary = {
        "league": LEAGUE,
        "season_year": SEASON_YEAR,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "all_teams": len(all_teams),
            "results_rows": len(df_all_results),
            "future_matches": len(df_2025_future),
            "finished_matches": len(df_2025_finished),
        },
        "accuracy": {
            "finished_games": total_finished_games,
            "correct": correct_predictions,
            "accuracy_pct": round(accuracy, 2) if total_finished_games > 0 else None,
        },
    }

    desired_pred_cols = [
        "match_id",
        "home_team",
        "away_team",
        "stats_asof",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "final_result",
        "predicted_result",
        "predicted_highest_prob_result",
        "argmax_raw_result",
    ]
    if "predicted_result" not in df_pred.columns:
        df_pred = recalculate_predicted_result(df_pred, "predicted_result")
    if "predicted_highest_prob_result" not in df_pred.columns:
        df_pred = recalculate_predicted_highest_prob_result(df_pred, "predicted_highest_prob_result")
    pred_cols = [c for c in desired_pred_cols if c in df_pred.columns]
    pred_list = df_pred[pred_cols].to_dict(orient="records") if not df_pred.empty else []

    desired_backtest_cols = [
        "match_id",
        "home_team",
        "away_team",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "final_result",
        "predicted_result",
        "predicted_highest_prob_result",
        "argmax_raw_result",
        "actual_result",
        "is_correct",
    ]
    if "predicted_result" not in df_backtest.columns:
        df_backtest = recalculate_predicted_result(df_backtest, "predicted_result")
    if "predicted_highest_prob_result" not in df_backtest.columns:
        df_backtest = recalculate_predicted_highest_prob_result(df_backtest, "predicted_highest_prob_result")
    backtest_cols = [c for c in desired_backtest_cols if c in df_backtest.columns]
    backtest_list = df_backtest[backtest_cols].to_dict(orient="records") if not df_backtest.empty else []

    report = {
        "inputs": inputs,
        "parameters": params,
        "summary": summary,
        "predictions": pred_list,
        "backtest": backtest_list,
    }
    return report


def write_report():
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = build_report()
    report_json = os.path.join(REPORT_DIR, f"report_{LEAGUE}_{SEASON_YEAR}.json")
    report_md = os.path.join(REPORT_DIR, f"report_{LEAGUE}_{SEASON_YEAR}.md")

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"# 解析内訳レポート ({LEAGUE} {SEASON_YEAR})")
    lines.append("")
    lines.append("## 入力ファイル")
    for k, v in report["inputs"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## パラメータ")
    for k, v in report["parameters"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## サマリ")
    summary = report["summary"]
    lines.append(f"- league: {summary['league']}")
    lines.append(f"- season_year: {summary['season_year']}")
    lines.append(f"- generated_at: {summary['generated_at']}")
    counts = summary["counts"]
    lines.append(f"- all_teams: {counts['all_teams']}")
    lines.append(f"- results_rows: {counts['results_rows']}")
    lines.append(f"- future_matches: {counts['future_matches']}")
    lines.append(f"- finished_matches: {counts['finished_matches']}")
    acc = summary["accuracy"]
    lines.append(f"- accuracy: {acc['accuracy_pct']}% ({acc['correct']}/{acc['finished_games']})")
    lines.append("")
    lines.append("## 出力")
    lines.append(f"- predictions: {output_csv}")
    lines.append(f"- backtest: {backtest_output_csv}")
    lines.append(f"- report_json: {report_json}")
    lines.append("")

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"解析内訳レポートを出力しました: {report_json}")


write_report()
