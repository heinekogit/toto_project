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
import pandas as pd
import numpy as np
from scipy.stats import poisson
import json
import re
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
STATS_ASOF_DATE = os.environ.get("STATS_ASOF_DATE", "").strip()
STATS_SNAPSHOT_NAME = os.environ.get("STATS_SNAPSHOT_NAME", "").strip()
WEATHER_ASOF_DATE = os.environ.get("WEATHER_ASOF_DATE", STATS_ASOF_DATE).strip()
WEATHER_SNAPSHOT_NAME = os.environ.get("WEATHER_SNAPSHOT_NAME", "").strip()
WEATHER_SNAPSHOT_DIR = os.environ.get("WEATHER_SNAPSHOT_DIR", os.path.join(DATA_DIR, "weather_snapshots"))
ABSENCE_ASOF_DATE = os.environ.get("ABSENCE_ASOF_DATE", STATS_ASOF_DATE).strip()
ABSENCE_SNAPSHOT_NAME = os.environ.get("ABSENCE_SNAPSHOT_NAME", "").strip()
ABSENCE_SNAPSHOT_DIR = os.environ.get("ABSENCE_SNAPSHOT_DIR", os.path.join(DATA_DIR, "absence_snapshots"))
MERGE_QC_DIR = os.path.join(REPORT_DIR, "merge_qc", f"{LEAGUE}_{SEASON_YEAR}")
PREV_SEASON_YEAR = SEASON_YEAR - 1
csv_prev = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_results.csv")
if not os.path.exists(csv_prev):
    csv_prev_latest = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_latest_results.csv")
    if os.path.exists(csv_prev_latest):
        csv_prev = csv_prev_latest
prev_final_elo_csv = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_final_elo.csv")
csv_season = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_upcoming.csv")
output_csv = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions.csv")
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
ENABLE_HFA = os.environ.get("ENABLE_HFA", "1") == "1"
HOME_ADV_PROFILE_DIFF_CLIP = float(os.environ.get("HOME_ADV_PROFILE_DIFF_CLIP", "0.8"))
HFA_ABS_MAX = float(os.environ.get("HFA_ABS_MAX", "60"))
HFA_DATA_QUALITY_MULT = float(os.environ.get("HFA_DATA_QUALITY_MULT", "0.3"))
HFA_STATS_MISSING_MULT = float(os.environ.get("HFA_STATS_MISSING_MULT", "0.0"))
ELO_DIFF_TEMPERATURE = float(os.environ.get("ELO_DIFF_TEMPERATURE", "1.35"))
ELO_DIFF_SCALE = float(os.environ.get("ELO_DIFF_SCALE", "1.00"))
ELO_DRAW_BASE = float(os.environ.get("ELO_DRAW_BASE", "0.33"))
# base主導で調整できるよう、既定は0（必要時のみ環境変数で有効化）
ELO_DRAW_BUMP = float(os.environ.get("ELO_DRAW_BUMP", "0.00"))
ELO_DRAW_SENSITIVITY = float(os.environ.get("ELO_DRAW_SENSITIVITY", "400"))
ELO_DRAW_DIFF_SCALE = float(os.environ.get("ELO_DRAW_DIFF_SCALE", "1.00"))
ELO_DRAW_MIN = float(os.environ.get("ELO_DRAW_MIN", "0.10"))
ELO_DRAW_MAX = float(os.environ.get("ELO_DRAW_MAX", "0.33"))
DRAW_DECAY_SCALE = float(os.environ.get("DRAW_DECAY_SCALE", "120.0"))
# draw確率はPoisson由来とElo由来をブレンド（1.0=Poissonのみ, 0.0=Eloのみ）
DRAW_BLEND_WEIGHT = float(os.environ.get("DRAW_BLEND_WEIGHT", "0.75"))
DRAW_ASSIGN_BY_EXPECTATION = os.environ.get("DRAW_ASSIGN_BY_EXPECTATION", "1") == "1"
# 期待ドロー件数の倍率（確率自体は変更せず、D割当件数のみ調整）
DRAW_EXPECTATION_MULTIPLIER = float(os.environ.get("DRAW_EXPECTATION_MULTIPLIER", "1.0"))
# Poisson格子の打ち切り誤差を抑えるための設定
POISSON_GRID_MIN_K = int(os.environ.get("POISSON_GRID_MIN_K", "10"))
POISSON_GRID_MAX_K = int(os.environ.get("POISSON_GRID_MAX_K", "20"))
POISSON_TAIL_EPS = float(os.environ.get("POISSON_TAIL_EPS", "1e-6"))
MISSING_WARN_THRESHOLD = float(os.environ.get("MISSING_WARN_THRESHOLD", "0.05"))
DEBUG_ELO_PROB = os.environ.get("DEBUG_ELO_PROB", "0") == "1"
DEBUG_MATCH_ID = os.environ.get("DEBUG_MATCH_ID", "").strip()
J1_WIN_PROB_CAP = float(os.environ.get("J1_WIN_PROB_CAP", "0.68"))
PROB_FALLBACK = (0.397, 0.251, 0.353)

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
    base_hfa = 0.0
    if ENABLE_HFA:
        base_hfa = float(HFA_ELO) + profile_diff_clipped * float(HOME_ADV_ELO_COEF)
        base_hfa = float(np.clip(base_hfa, -HFA_ABS_MAX, HFA_ABS_MAX))
    hfa_mult = 1.0
    if bool(stats_home_missing) or bool(stats_away_missing):
        hfa_mult = float(HFA_STATS_MISSING_MULT)
    elif bool(data_quality_warn):
        hfa_mult = float(HFA_DATA_QUALITY_MULT)
    applied_hfa = float(base_hfa) * float(hfa_mult)

    elo_diff_raw = (float(home_elo) - float(away_elo)) + applied_hfa
    elo_diff = float(elo_diff_raw) * float(ELO_DIFF_SCALE)
    expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    return {
        "hfa_enabled": bool(ENABLE_HFA),
        "home_advantage_profile_diff_raw": profile_diff_raw,
        "home_advantage_profile_diff_clipped": profile_diff_clipped,
        "base_hfa": float(base_hfa),
        "hfa_mult": float(hfa_mult),
        "applied_hfa": float(applied_hfa),
        "elo_diff_raw": float(elo_diff_raw),
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
    print(
        f"[{label}] rows={rows} avg_prob_draw={avg_draw:.3f} "
        f"sum_prob_draw={sum_draw:.3f} predicted_D_count={d_count} "
        f"ELO_DIFF_SCALE={ELO_DIFF_SCALE:.2f} "
        f"DRAW_DECAY_SCALE={DRAW_DECAY_SCALE:.1f} "
        f"ELO_DRAW_DIFF_SCALE={ELO_DRAW_DIFF_SCALE:.3f} "
        f"ELO_DRAW_MIN={ELO_DRAW_MIN:.3f} ELO_DRAW_MAX={ELO_DRAW_MAX:.3f} "
        f"DRAW_BLEND_WEIGHT={DRAW_BLEND_WEIGHT:.3f}"
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

    prob_home_win, prob_draw, prob_away_win = predict_poisson_probabilities(
        elo_ctx["elo_diff"],
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
    sum_before_round = prob_home_win + prob_draw + prob_away_win
    if not np.isclose(sum_before_round, 1.0, atol=1e-6):
        print(
            f"[PROB_QC][WARN] match_id={match_id} prob_sum={sum_before_round:.9f} "
            f"(home={prob_home_win:.6f}, draw={prob_draw:.6f}, away={prob_away_win:.6f})"
        )

    if elo_ctx["elo_diff"] > 0 and prob_home_win < prob_away_win:
        print(
            f"[PROB_QC][WARN] match_id={match_id} elo_diff={elo_ctx['elo_diff']:.4f} "
            f"なのに prob_home({prob_home_win:.4f}) < prob_away({prob_away_win:.4f})"
        )

    prob_home_win, prob_draw, prob_away_win, draw_model_input, draw_poi, draw_elo = calibrate_draw_probability(
        prob_home_win,
        prob_draw,
        prob_away_win,
        elo_ctx["elo_diff"],
    )

    predicted_result = decide_predicted_result(prob_home_win, prob_draw, prob_away_win)

    debug_row = {
        "match_id": match_id,
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "home_advantage_diff_input": float(home_advantage_diff),
        "hfa_enabled": elo_ctx["hfa_enabled"],
        "home_advantage_profile_diff_raw": elo_ctx["home_advantage_profile_diff_raw"],
        "home_advantage_profile_diff_clipped": elo_ctx["home_advantage_profile_diff_clipped"],
        "HFA_base": elo_ctx["base_hfa"],
        "HFA_multiplier": elo_ctx["hfa_mult"],
        "HFA_applied": elo_ctx["applied_hfa"],
        "elo_diff_raw": elo_ctx["elo_diff_raw"],
        "elo_diff": elo_ctx["elo_diff"],
        "expected_home": elo_ctx["expected_home"],
        "draw_model_input": draw_model_input,
        "draw_model_output": prob_draw,
        "draw_model_poi": draw_poi,
        "draw_model_elo": draw_elo,
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "predicted_result": predicted_result,
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


def decide_predicted_result(
    prob_home_win,
    prob_draw,
    prob_away_win,
):
    if pd.isna(prob_home_win) or pd.isna(prob_draw) or pd.isna(prob_away_win):
        return None
    if prob_home_win >= prob_draw and prob_home_win >= prob_away_win:
        return "H"
    if prob_away_win >= prob_home_win and prob_away_win >= prob_draw:
        return "A"
    return "D"


def assign_draw_results_by_expectation(df, output_col="predicted_result"):
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if not required_cols.issubset(df.columns):
        return df

    out = df.copy()
    out["__base_pred"] = out.apply(
        lambda r: decide_predicted_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"]),
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

    out = out.drop(columns=["__base_pred", "__round_group"], errors="ignore")
    return out


def recalculate_predicted_result(df, output_col="predicted_result"):
    out = df.copy()
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if not required_cols.issubset(out.columns):
        return out
    out[output_col] = out.apply(
        lambda r: decide_predicted_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"]),
        axis=1,
    )
    return out


def recalculate_predicted_highest_prob_result(df, output_col="predicted_highest_prob_result"):
    out = df.copy()
    required_cols = {"prob_home_win_raw", "prob_draw_raw", "prob_away_win_raw"}
    if not required_cols.issubset(out.columns):
        return out
    out[output_col] = out.apply(
        lambda r: decide_predicted_result(r["prob_home_win_raw"], r["prob_draw_raw"], r["prob_away_win_raw"]),
        axis=1,
    )
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
        lambda r: decide_predicted_result(r["prob_home_win_raw"], r["prob_draw_raw"], r["prob_away_win_raw"]),
        axis=1,
    )
    work["_cal_argmax"] = work.apply(
        lambda r: decide_predicted_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"]),
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
        if SEASON_YEAR >= 2026 and os.environ.get("ENABLE_J2_STRICT_FILTER", "0") != "1":
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
    if os.path.exists(prev_final_elo_csv):
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
    prob_home_win, prob_draw, prob_away_win = calibrate_probabilities(
        prob_home_win_raw,
        prob_draw_raw,
        prob_away_win_raw,
        row.get("league", LEAGUE),
    )
    prob_home_win, prob_draw, prob_away_win = _normalize_probs(prob_home_win, prob_draw, prob_away_win)
    predicted_result = decide_predicted_result(prob_home_win, prob_draw, prob_away_win)
    predicted_highest_prob_result = decide_predicted_result(prob_home_win_raw, prob_draw_raw, prob_away_win_raw)
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
        "elo_diff_for_prob": round(debug_row["elo_diff"], 4),
        "expected_home_two_way": round(debug_row["expected_home"], 4),
        "is_home_advantage_positive": bool(is_home_advantage_positive),
        "prob_home_win_raw": prob_home_win_raw,
        "prob_draw_raw": prob_draw_raw,
        "prob_away_win_raw": prob_away_win_raw,
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "predicted_result": predicted_result,
        "predicted_highest_prob_result": predicted_highest_prob_result,
    })

# 保存
df_pred = pd.DataFrame(predictions)
df_pred = add_data_quality_flags(df_pred)
df_pred = recalculate_predicted_result(df_pred, "predicted_result")
df_pred = recalculate_predicted_highest_prob_result(df_pred, "predicted_highest_prob_result")
if DRAW_ASSIGN_BY_EXPECTATION:
    # 最終ラベルは「調整後確率」をベースに、節単位の期待ドロー数へ合わせてDを付与する
    df_pred = assign_draw_results_by_expectation(df_pred, "predicted_result")
else:
    print("[DRAW_ASSIGN] disabled (DRAW_ASSIGN_BY_EXPECTATION=0)")
log_prediction_consistency(df_pred, "PRED")
log_prob_summary(df_pred, "PRED_SUMMARY")
log_absence_effective_summary(df_pred, "PRED")
df_pred = drop_internal_output_columns(df_pred)
report_missing_rates(df_pred, "final_predictions_df")
df_pred.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"予測結果を {output_csv} に出力しました。")

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
    prob_home_win, prob_draw, prob_away_win = calibrate_probabilities(
        prob_home_win_raw,
        prob_draw_raw,
        prob_away_win_raw,
        row.get("league", LEAGUE),
    )
    prob_home_win, prob_draw, prob_away_win = _normalize_probs(prob_home_win, prob_draw, prob_away_win)
    predicted_label = decide_predicted_result(prob_home_win, prob_draw, prob_away_win)
    predicted_highest_prob_result = decide_predicted_result(prob_home_win_raw, prob_draw_raw, prob_away_win_raw)
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
        "elo_diff_for_prob": round(debug_row["elo_diff"], 4),
        "expected_home_two_way": round(debug_row["expected_home"], 4),
        "is_home_advantage_positive": bool(is_home_advantage_positive),
        "prob_home_win_raw": prob_home_win_raw,
        "prob_draw_raw": prob_draw_raw,
        "prob_away_win_raw": prob_away_win_raw,
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "predicted_result": predicted_label,
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
log_prediction_consistency(df_backtest, "BACKTEST")
log_prob_summary(df_backtest, "BACKTEST_SUMMARY")
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
        "ENABLE_HFA": ENABLE_HFA,
        "HOME_ADV_ELO_COEF": HOME_ADV_ELO_COEF,
        "HOME_ADV_PROFILE_DIFF_CLIP": HOME_ADV_PROFILE_DIFF_CLIP,
        "HFA_ABS_MAX": HFA_ABS_MAX,
        "HFA_DATA_QUALITY_MULT": HFA_DATA_QUALITY_MULT,
        "HFA_STATS_MISSING_MULT": HFA_STATS_MISSING_MULT,
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
        "predicted_result",
        "predicted_highest_prob_result",
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
        "predicted_result",
        "predicted_highest_prob_result",
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
