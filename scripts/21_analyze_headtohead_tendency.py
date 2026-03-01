import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


SEASON_YEAR = os.environ.get("SEASON_YEAR", "2025")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
INPUT_CSV = os.environ.get("INPUT_CSV", f"backtest_{LEAGUE}_{SEASON_YEAR}_rounds.csv")
HALF_LIFE_DAYS = float(os.environ.get("HALF_LIFE_DAYS", "180"))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "reports", "metrics")


@dataclass
class MatchRow:
    datetime: pd.Timestamp
    team: str
    opponent: str
    venue: str
    gf: float
    ga: float


def classify_tendency(points_per_match: float, gd_per_match: float, matches: int) -> str:
    if matches < 2:
        return "判定保留(試合数不足)"
    if points_per_match >= 2.0 and gd_per_match >= 0.5:
        return "得意"
    if points_per_match <= 1.0 and gd_per_match <= -0.3:
        return "苦手"
    return "互角/中間"


def confidence_label(matches: int) -> str:
    if matches >= 7:
        return "high"
    if matches >= 4:
        return "medium"
    return "low"


def recency_weight(series_dt: pd.Series, half_life_days: float) -> np.ndarray:
    dt = pd.to_datetime(series_dt, errors="coerce")
    max_dt = dt.max()
    if pd.isna(max_dt):
        return np.ones(len(series_dt), dtype=float)
    age_days = (max_dt - dt).dt.total_seconds().fillna(0) / 86400.0
    decay = np.log(2) / half_life_days
    return np.exp(-decay * age_days.to_numpy())


def load_match_rows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["datetime", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["datetime", "home_score", "away_score", "home_team", "away_team"])

    rows = []
    for _, r in df.iterrows():
        rows.append(
            MatchRow(
                datetime=r["datetime"],
                team=str(r["home_team"]).strip(),
                opponent=str(r["away_team"]).strip(),
                venue="home",
                gf=float(r["home_score"]),
                ga=float(r["away_score"]),
            )
        )
        rows.append(
            MatchRow(
                datetime=r["datetime"],
                team=str(r["away_team"]).strip(),
                opponent=str(r["home_team"]).strip(),
                venue="away",
                gf=float(r["away_score"]),
                ga=float(r["home_score"]),
            )
        )
    out = pd.DataFrame([x.__dict__ for x in rows])
    out["is_win"] = (out["gf"] > out["ga"]).astype(int)
    out["is_draw"] = (out["gf"] == out["ga"]).astype(int)
    out["is_loss"] = (out["gf"] < out["ga"]).astype(int)
    out["points"] = out["is_win"] * 3 + out["is_draw"] * 1
    out["goal_diff"] = out["gf"] - out["ga"]
    return out


def aggregate_team_vs_opponent(df: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    records = []
    for (team, opponent), g in df.groupby(["team", "opponent"], sort=True):
        matches = int(len(g))
        wins = int(g["is_win"].sum())
        draws = int(g["is_draw"].sum())
        losses = int(g["is_loss"].sum())
        points = float(g["points"].sum())
        gd = float(g["goal_diff"].sum())

        weights = recency_weight(g["datetime"], half_life_days=half_life_days)
        weighted_ppm = float(np.average(g["points"].to_numpy(), weights=weights))
        weighted_gdpm = float(np.average(g["goal_diff"].to_numpy(), weights=weights))

        home = g[g["venue"] == "home"]
        away = g[g["venue"] == "away"]
        home_ppm = float(home["points"].mean()) if len(home) else np.nan
        away_ppm = float(away["points"].mean()) if len(away) else np.nan

        points_per_match = points / matches if matches else np.nan
        gd_per_match = gd / matches if matches else np.nan
        tendency = classify_tendency(points_per_match, gd_per_match, matches)

        records.append(
            {
                "team": team,
                "opponent": opponent,
                "matches": matches,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "points": points,
                "goal_diff": gd,
                "points_per_match": round(points_per_match, 3),
                "gd_per_match": round(gd_per_match, 3),
                "home_points_per_match": round(home_ppm, 3) if pd.notna(home_ppm) else np.nan,
                "away_points_per_match": round(away_ppm, 3) if pd.notna(away_ppm) else np.nan,
                "weighted_points_per_match": round(weighted_ppm, 3),
                "weighted_gd_per_match": round(weighted_gdpm, 3),
                "tendency": tendency,
                "confidence": confidence_label(matches),
            }
        )
    out = pd.DataFrame(records)
    return out.sort_values(["team", "points_per_match", "gd_per_match"], ascending=[True, False, False]).reset_index(drop=True)


def build_team_summary(df_pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, g in df_pair.groupby("team", sort=True):
        g_valid = g[g["matches"] >= 2].copy()
        if g_valid.empty:
            rows.append(
                {
                    "team": team,
                    "strongest_opponent": "",
                    "strongest_ppm": np.nan,
                    "weakest_opponent": "",
                    "weakest_ppm": np.nan,
                }
            )
            continue
        strongest = g_valid.sort_values(["points_per_match", "gd_per_match"], ascending=[False, False]).iloc[0]
        weakest = g_valid.sort_values(["points_per_match", "gd_per_match"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "team": team,
                "strongest_opponent": strongest["opponent"],
                "strongest_ppm": strongest["points_per_match"],
                "weakest_opponent": weakest["opponent"],
                "weakest_ppm": weakest["points_per_match"],
            }
        )
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_path = INPUT_CSV if os.path.isabs(INPUT_CSV) else os.path.join(BASE_DIR, INPUT_CSV)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

    df_match = load_match_rows(input_path)
    df_pair = aggregate_team_vs_opponent(df_match, half_life_days=HALF_LIFE_DAYS)
    df_summary = build_team_summary(df_pair)

    pair_path = os.path.join(OUTPUT_DIR, f"head_to_head_tendency_{LEAGUE}_{SEASON_YEAR}.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"head_to_head_tendency_summary_{LEAGUE}_{SEASON_YEAR}.csv")
    df_pair.to_csv(pair_path, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"出力: {pair_path}")
    print(f"出力: {summary_path}")
    print(f"件数: {len(df_pair)} 対戦ペア(チーム視点)")


if __name__ == "__main__":
    main()
