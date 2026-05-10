import os
import pandas as pd
import numpy as np
from scipy.stats import poisson
from pandas.errors import EmptyDataError


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MANUAL_DIR = os.path.join(DATA_DIR, "manual")

LEAGUE = os.environ.get("LEAGUE", "j1").lower()
SEASON_YEAR = int(os.environ.get("SEASON_YEAR", "2025"))
PREV_SEASON_YEAR = SEASON_YEAR - 1

RESULTS_CSV = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_latest_results.csv")
STATS_CSV = os.path.join(DATA_DIR, f"team_master_stats_{LEAGUE}_{SEASON_YEAR}.csv")
if not os.path.exists(STATS_CSV):
    STATS_CSV = os.path.join(DATA_DIR, f"team_master_stats_{SEASON_YEAR}.csv")
if not os.path.exists(STATS_CSV):
    STATS_CSV = os.path.join(DATA_DIR, "team_master_stats.csv")

FATIGUE_CSV = os.path.join(DATA_DIR, f"team_fatigue_scores_{LEAGUE}_{SEASON_YEAR}.csv")
if not os.path.exists(FATIGUE_CSV):
    FATIGUE_CSV = os.path.join(DATA_DIR, f"team_fatigue_scores_{SEASON_YEAR}.csv")
if not os.path.exists(FATIGUE_CSV):
    FATIGUE_CSV = os.path.join(DATA_DIR, "team_fatigue_scores.csv")

TRAVEL_CSV = os.path.join(MANUAL_DIR, "team_travel_distances.csv")
if not os.path.exists(TRAVEL_CSV):
    TRAVEL_CSV = os.path.join(DATA_DIR, "team_travel_distances.csv")

WEATHER_CANDIDATES = [
    os.path.join(MANUAL_DIR, f"weather_features_{LEAGUE}_{SEASON_YEAR}.csv"),
    os.path.join(MANUAL_DIR, "weather_features.csv"),
    os.path.join(DATA_DIR, f"weather_features_{LEAGUE}_{SEASON_YEAR}.csv"),
    os.path.join(DATA_DIR, "weather_features.csv"),
]
WEATHER_CSV = next((p for p in WEATHER_CANDIDATES if os.path.exists(p)), None)
J2_ALLOWED_TEAMS_CSV = os.environ.get(
    "J2_ALLOWED_TEAMS_CSV",
    os.path.join(MANUAL_DIR, f"j2_allowed_teams_{SEASON_YEAR}.csv"),
)

OUTPUT_CSV = os.path.join(BASE_DIR, f"backtest_{LEAGUE}_{SEASON_YEAR}_rounds.csv")

# パラメータ
INITIAL_ELO = 1500
HOME_ADVANTAGE = 50
GOAL_SCALING_FACTOR = 0.01
FATIGUE_GOAL_SCALING = 0.01
WEATHER_PENALTY_HEAVY_RAIN = 0.15
WEATHER_PENALTY_STRONG_WIND = 0.10
WEATHER_PENALTY_RAIN = 0.05
D_INTERCEPT = -1.2
D_SCALE = 1.5
DRAW_PROB_THRESHOLD = 0.45
DRAW_BALANCE_THRESHOLD = 0.10
HOME_ADV_ELO_COEF = float(os.environ.get("HOME_ADV_ELO_COEF", "60"))
ELO_DRAW_BASE = float(os.environ.get("ELO_DRAW_BASE", "0.26"))
ELO_DRAW_SENSITIVITY = float(os.environ.get("ELO_DRAW_SENSITIVITY", "0.0002"))
ELO_DRAW_MIN = float(os.environ.get("ELO_DRAW_MIN", "0.10"))
ELO_DRAW_MAX = float(os.environ.get("ELO_DRAW_MAX", "0.35"))


NAME_MAP = {
    "Ｇ大阪": "ガンバ大阪",
    "Ｃ大阪": "セレッソ大阪",
    "横浜FM": "横浜Ｆ・マリノス",
    "横浜ＦＭ": "横浜Ｆ・マリノス",
    "横浜FC": "横浜ＦＣ",
    "FC東京": "ＦＣ東京",
    "川崎Ｆ": "川崎フロンターレ",
    "東京Ｖ": "東京ヴェルディ",
    "湘南": "湘南ベルマーレ",
    "神戸": "ヴィッセル神戸",
    "名古屋": "名古屋グランパス",
    "浦和": "浦和レッズ",
    "広島": "サンフレッチェ広島",
    "福岡": "アビスパ福岡",
    "清水": "清水エスパルス",
    "新潟": "アルビレックス新潟",
    "鹿島": "鹿島アントラーズ",
    "柏": "柏レイソル",
    "町田": "ＦＣ町田ゼルビア",
    "岡山": "ファジアーノ岡山",
    "京都": "京都サンガF.C.",
}


def get_result(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return "H"
    elif home_score < away_score:
        return "A"
    else:
        return "D"


def update_elo(elo_home, elo_away, result, k=20):
    elo_diff = elo_home + HOME_ADVANTAGE - elo_away
    p_home = 1 / (1 + 10 ** (-elo_diff / 400))
    s_home = {"H": 1, "D": 0.5, "A": 0}[result]
    delta = k * (s_home - p_home)
    return elo_home + delta, elo_away - delta


def calculate_expected_goals(
    elo_home,
    elo_away,
    home_xg=None,
    away_xg=None,
    home_travel=0,
    away_travel=0,
    home_fatigue=None,
    away_fatigue=None,
    weather_flags=None,
):
    elo_home_xg = 1.5 + (elo_home - elo_away + HOME_ADVANTAGE) * GOAL_SCALING_FACTOR
    elo_away_xg = 1.5 + (elo_away - elo_home - HOME_ADVANTAGE) * GOAL_SCALING_FACTOR
    home_goal = elo_home_xg
    away_goal = elo_away_xg
    if home_xg is not None and away_xg is not None:
        home_goal = (elo_home_xg * 0.7) + (home_xg * 0.3)
        away_goal = (elo_away_xg * 0.7) + (away_xg * 0.3)
    if home_fatigue is not None:
        home_goal -= home_fatigue * FATIGUE_GOAL_SCALING
    if away_fatigue is not None:
        away_goal -= away_fatigue * FATIGUE_GOAL_SCALING

    if weather_flags:
        penalty = 0.0
        if weather_flags.get("is_heavy_rain"):
            penalty += WEATHER_PENALTY_HEAVY_RAIN
        elif weather_flags.get("is_rain"):
            penalty += WEATHER_PENALTY_RAIN
        if weather_flags.get("is_strong_wind"):
            penalty += WEATHER_PENALTY_STRONG_WIND
        if penalty > 0:
            home_goal -= penalty
            away_goal -= penalty

    return max(0.1, home_goal), max(0.1, away_goal)


def predict_probs(
    elo_home,
    elo_away,
    home_xg=None,
    away_xg=None,
    home_travel=0,
    away_travel=0,
    home_fatigue=None,
    away_fatigue=None,
    weather_flags=None,
    max_goals=5,
):
    hg, ag = calculate_expected_goals(
        elo_home,
        elo_away,
        home_xg,
        away_xg,
        home_travel,
        away_travel,
        home_fatigue,
        away_fatigue,
        weather_flags,
    )
    ph = pdw = pa = 0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, hg) * poisson.pmf(j, ag)
            if i > j:
                ph += p
            elif i == j:
                pdw += p
            else:
                pa += p
    total = ph + pdw + pa
    if total > 0:
        ph, pdw, pa = ph / total, pdw / total, pa / total
    return apply_draw_separation(ph, pdw, pa)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def apply_draw_separation(prob_home_win, prob_draw, prob_away_win):
    # Poissonのスコア分布は維持し、Dのみ分離補正で過大化を抑制する。
    # H/AはPoisson由来の比率を使って再配分するため強弱関係は保たれる。
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
    # after: pH=0.306, pD=0.400, pA=0.294 (sum=1.000)
    return pH, pD, pA


def decide_predicted_result(
    prob_home_win,
    prob_draw,
    prob_away_win,
    draw_prob_threshold=DRAW_PROB_THRESHOLD,
    draw_balance_threshold=DRAW_BALANCE_THRESHOLD,
):
    if pd.isna(prob_home_win) or pd.isna(prob_draw) or pd.isna(prob_away_win):
        return None
    if prob_draw >= draw_prob_threshold and abs(prob_home_win - prob_away_win) <= draw_balance_threshold:
        return "D"
    return "H" if prob_home_win >= prob_away_win else "A"


def recalculate_predicted_result(df, output_col="predicted_result"):
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if not required_cols.issubset(df.columns):
        return df
    out = df.copy()
    out["__base_pred"] = out.apply(
        lambda r: decide_predicted_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"]),
        axis=1,
    )

    def _assign_block(block: pd.DataFrame, group_label: str) -> pd.DataFrame:
        b = block.copy()
        valid = b["prob_draw"].notna() & b["prob_home_win"].notna() & b["prob_away_win"].notna()
        valid_count = int(valid.sum())
        if valid_count == 0:
            b[output_col] = b["__base_pred"]
            print(f"[DRAW_ASSIGN] group={group_label} Expected_draws=0.00, Assigned_D=0")
            return b

        expected_draws = float(b.loc[valid, "prob_draw"].sum())
        target_draw_count = 0 if expected_draws < 0.5 else int(round(expected_draws))
        target_draw_count = max(0, min(target_draw_count, valid_count))

        draw_idx = (
            b.loc[valid]
            .sort_values("prob_draw", ascending=False)
            .head(target_draw_count)
            .index
        )
        draw_idx_set = set(draw_idx.tolist())
        b[output_col] = b.index.map(lambda idx: "D" if idx in draw_idx_set else b.at[idx, "__base_pred"])
        assigned_d = int((b.loc[valid, output_col] == "D").sum())
        print(f"[DRAW_ASSIGN] group={group_label} Expected_draws={expected_draws:.2f}, Assigned_D={assigned_d}")
        return b

    group_col = None
    for c in ["toto_round_id", "節", "round", "round_id"]:
        if c in out.columns:
            group_col = c
            break

    if group_col is None:
        out = _assign_block(out, "ALL")
    else:
        pieces = []
        for g, block in out.groupby(group_col, dropna=False, sort=False):
            label = str(g) if pd.notna(g) else "NA"
            pieces.append(_assign_block(block, label))
        out = pd.concat(pieces, axis=0).sort_index()

    out = out.drop(columns=["__base_pred"], errors="ignore")
    return out


def _normalize_probs(ph, pdw, pa):
    arr = np.array([ph, pdw, pa], dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    arr = arr / s
    return float(arr[0]), float(arr[1]), float(arr[2])


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

    # 2026特別大会（J2/J3混在）は既定で厳格フィルタを無効化し、
    # 日程データ上のカードをそのまま予測対象にする。
    if SEASON_YEAR >= 2026 and os.environ.get("ENABLE_J2_STRICT_FILTER", "0") != "1":
        print("J2許可チームフィルタを無効化します（2026特別大会モード）。")
        return None

    if os.path.exists(J2_ALLOWED_TEAMS_CSV):
        try:
            df = pd.read_csv(J2_ALLOWED_TEAMS_CSV)
            col = "team_name" if "team_name" in df.columns else df.columns[0]
            teams = set(df[col].dropna().astype(str).str.strip())
            if teams:
                # 2026特別大会では実カード側の参加チーム数を下回る許可リストは不整合とみなし無効化
                if SEASON_YEAR >= 2026 and os.path.exists(RESULTS_CSV):
                    try:
                        cur = pd.read_csv(RESULTS_CSV)
                        cur_teams = set(cur.get("home_team", pd.Series(dtype="object")).dropna().astype(str).str.strip())
                        cur_teams |= set(cur.get("away_team", pd.Series(dtype="object")).dropna().astype(str).str.strip())
                        cur_teams = {t for t in cur_teams if t}
                        if cur_teams and len(teams) < len(cur_teams) and os.environ.get("ENABLE_J2_STRICT_FILTER_FORCE", "0") != "1":
                            print(
                                f"警告: J2許可チームCSV({len(teams)}チーム)が実カードのチーム数({len(cur_teams)}チーム)より少ないため、"
                                "2026特別大会モードとしてフィルタを無効化します。"
                            )
                            return None
                    except Exception as e:
                        print(f"警告: J2許可チーム整合性チェックに失敗しました: {e}")
                print(f"J2許可チームを {J2_ALLOWED_TEAMS_CSV} から読み込みました: {len(teams)}")
                return teams
        except Exception as e:
            print(f"警告: J2許可チームCSVの読み込みに失敗しました: {e}")

    fallback_csv = os.path.join(DATA_DIR, f"j2_{PREV_SEASON_YEAR}_latest_results.csv")
    if os.path.exists(fallback_csv):
        try:
            df = pd.read_csv(fallback_csv)
            teams = set(df["home_team"].dropna().astype(str).str.strip()) | set(
                df["away_team"].dropna().astype(str).str.strip()
            )
            if teams:
                print(f"J2許可チームを前年データから推定しました: {len(teams)} ({fallback_csv})")
                return teams
        except Exception as e:
            print(f"警告: 前年J2結果からの許可チーム推定に失敗しました: {e}")

    print("警告: J2許可チームを特定できませんでした。全カードを予測対象にします。")
    return None


def main():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"結果データが見つかりません: {RESULTS_CSV}")

    results = pd.read_csv(RESULTS_CSV)
    results["round_num"] = results["節"].astype(str).str.extract(r"第(\d+)節").astype(float)
    results = results.dropna(subset=["round_num"])
    results["round_num"] = results["round_num"].astype(int)

    stats_df = pd.read_csv(STATS_CSV)
    stats_map = stats_df.set_index("team_name")["ゴール期待値"].to_dict()

    travel_df = pd.read_csv(TRAVEL_CSV, sep="\t").set_index("ホーム　／　アウェイ")

    fatigue_map = {}
    if os.path.exists(FATIGUE_CSV):
        fatigue_df = pd.read_csv(FATIGUE_CSV)
        if "match_id" in fatigue_df.columns:
            dedup = fatigue_df.drop_duplicates(subset=["match_id"], keep="last")
            fatigue_map = dedup.set_index("match_id")[["home_fatigue_score", "away_fatigue_score"]].to_dict("index")

    weather_map = {}
    if WEATHER_CSV:
        try:
            weather_df = pd.read_csv(WEATHER_CSV)
            if "match_id" in weather_df.columns:
                weather_map = weather_df.set_index("match_id")[["is_rain", "is_heavy_rain", "is_strong_wind"]].to_dict("index")
        except EmptyDataError:
            print(f"警告: 天候ファイルが空のためスキップします: {WEATHER_CSV}")

    all_rounds = sorted(results["round_num"].unique())
    records = []
    home_ppm_map, away_ppm_map = build_home_away_profile_map(results)
    j2_allowed_teams = load_j2_allowed_teams()

    # Elo初期化
    teams = set(results["home_team"]).union(set(results["away_team"]))
    elo = {t: INITIAL_ELO for t in teams}

    for rnd in all_rounds:
        train = results[results["round_num"] < rnd]
        target = results[results["round_num"] == rnd]

        # Elo更新（過去のみ）
        for _, row in train.iterrows():
            res = get_result(row["home_score"], row["away_score"])
            if res:
                elo[row["home_team"]], elo[row["away_team"]] = update_elo(
                    elo[row["home_team"]], elo[row["away_team"]], res
                )

        # 予測
        for _, row in target.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_advantage_diff, is_home_advantage_positive = calc_home_advantage_diff(
                home, away, home_ppm_map, away_ppm_map
            )

            home_full = NAME_MAP.get(home, home)
            away_full = NAME_MAP.get(away, away)

            home_xg = stats_map.get(home_full)
            away_xg = stats_map.get(away_full)

            home_travel = 0
            away_travel = 0
            if home_full in travel_df.index and away_full in travel_df.columns:
                home_travel = travel_df.loc[home_full, away_full]
            if away_full in travel_df.index and home_full in travel_df.columns:
                away_travel = travel_df.loc[away_full, home_full]

            fatigue = fatigue_map.get(row.get("match_id"), {})
            home_fatigue = fatigue.get("home_fatigue_score")
            away_fatigue = fatigue.get("away_fatigue_score")

            weather_flags = weather_map.get(row.get("match_id"), {})

            if LEAGUE == "j2" and j2_allowed_teams is not None and (home not in j2_allowed_teams or away not in j2_allowed_teams):
                ph, pdw, pa = np.nan, np.nan, np.nan
                pred = None
                actual = get_result(row["home_score"], row["away_score"])
                is_correct = False
            else:
                ph, pdw, pa = predict_probs(
                    elo[home],
                    elo[away],
                    home_xg,
                    away_xg,
                    home_travel,
                    away_travel,
                    home_fatigue,
                    away_fatigue,
                    weather_flags,
                )
                ph, pdw, pa = predict_elo_probabilities_with_home_advantage(
                    home_elo=elo[home],
                    away_elo=elo[away],
                    home_advantage_diff=home_advantage_diff,
                    home_adv_coef=HOME_ADV_ELO_COEF,
                )

                pred = decide_predicted_result(ph, pdw, pa)
                actual = get_result(row["home_score"], row["away_score"])
                is_correct = (pred == actual) if actual else False

            records.append(
                {
                    "節": row["節"],
                    "match_id": row.get("match_id"),
                    "datetime": row.get("datetime"),
                    "home_team": home,
                    "away_team": away,
                    "home_score": row.get("home_score"),
                    "away_score": row.get("away_score"),
                    "home_advantage_diff": round(home_advantage_diff, 4),
                    "is_home_advantage_positive": bool(is_home_advantage_positive),
                    "prob_home_win": round(ph, 3),
                    "prob_draw": round(pdw, 3),
                    "prob_away_win": round(pa, 3),
                    "predicted_highest_prob_result": pred,
                    "actual_result": actual,
                    "is_correct": is_correct,
                }
            )

    df_out = pd.DataFrame(records)
    df_out = recalculate_predicted_result(df_out, "predicted_result")
    if "predicted_result" in df_out.columns:
        df_out["predicted_highest_prob_result"] = df_out["predicted_result"]
    if "actual_result" in df_out.columns and "predicted_result" in df_out.columns:
        df_out["is_correct"] = df_out["actual_result"] == df_out["predicted_result"]
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"出力: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
