import os
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["datetime", "home_team", "away_team", "home_score", "away_score"]


def _validate_input_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"入力CSVに必要列が不足しています: {missing}")


def _build_team_match_rows(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    _validate_input_columns(base)

    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
    base["home_score"] = pd.to_numeric(base["home_score"], errors="coerce")
    base["away_score"] = pd.to_numeric(base["away_score"], errors="coerce")
    base["home_team"] = base["home_team"].astype(str).str.strip()
    base["away_team"] = base["away_team"].astype(str).str.strip()

    # 試合結果が確定している行のみ集計対象
    base = base.dropna(subset=["datetime", "home_score", "away_score", "home_team", "away_team"])

    home = pd.DataFrame(
        {
            "team": base["home_team"],
            "venue": "home",
            "gf": base["home_score"],
            "ga": base["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "team": base["away_team"],
            "venue": "away",
            "gf": base["away_score"],
            "ga": base["home_score"],
        }
    )
    rows = pd.concat([home, away], ignore_index=True)
    rows["is_win"] = (rows["gf"] > rows["ga"]).astype(int)
    rows["is_draw"] = (rows["gf"] == rows["ga"]).astype(int)
    rows["is_loss"] = (rows["gf"] < rows["ga"]).astype(int)
    rows["points"] = rows["is_win"] * 3 + rows["is_draw"]
    return rows


def build_home_away_team_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    試合結果DataFrameから、チーム別のホーム/アウェイ成績指標を返す。
    必須列: datetime, home_team, away_team, home_score, away_score
    """
    rows = _build_team_match_rows(df)
    grouped = (
        rows.groupby(["team", "venue"], as_index=False)
        .agg(
            matches=("venue", "size"),
            points=("points", "sum"),
            wins=("is_win", "sum"),
            draws=("is_draw", "sum"),
            losses=("is_loss", "sum"),
        )
        .copy()
    )

    grouped["points_per_match"] = grouped["points"] / grouped["matches"]
    grouped["win_rate"] = grouped["wins"] / grouped["matches"]
    grouped["draw_rate"] = grouped["draws"] / grouped["matches"]
    grouped["loss_rate"] = grouped["losses"] / grouped["matches"]

    home = grouped[grouped["venue"] == "home"].set_index("team")
    away = grouped[grouped["venue"] == "away"].set_index("team")
    teams = pd.Index(sorted(set(home.index) | set(away.index)), name="team")

    out = pd.DataFrame(index=teams)
    out["home_points_per_match"] = home.reindex(teams)["points_per_match"]
    out["away_points_per_match"] = away.reindex(teams)["points_per_match"]

    out["home_win_rate"] = home.reindex(teams)["win_rate"]
    out["home_draw_rate"] = home.reindex(teams)["draw_rate"]
    out["home_loss_rate"] = home.reindex(teams)["loss_rate"]

    out["away_win_rate"] = away.reindex(teams)["win_rate"]
    out["away_draw_rate"] = away.reindex(teams)["draw_rate"]
    out["away_loss_rate"] = away.reindex(teams)["loss_rate"]

    out["home_matches"] = home.reindex(teams)["matches"]
    out["away_matches"] = away.reindex(teams)["matches"]

    out = out.fillna(0.0).reset_index()

    float_cols = [c for c in out.columns if c not in ["team", "home_matches", "away_matches"]]
    out[float_cols] = out[float_cols].round(4)
    out["home_matches"] = out["home_matches"].astype(int)
    out["away_matches"] = out["away_matches"].astype(int)

    # 要件互換: 列名バリエーション（winrate表記）も同時に持たせる
    out["home_winrate"] = out["home_win_rate"]
    out["home_drawrate"] = out["home_draw_rate"]
    out["home_lossrate"] = out["home_loss_rate"]
    out["away_winrate"] = out["away_win_rate"]
    out["away_drawrate"] = out["away_draw_rate"]
    out["away_lossrate"] = out["away_loss_rate"]
    return out


def build_home_away_team_profile_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return build_home_away_team_profile(df)


def main() -> None:
    season = os.environ.get("SEASON_YEAR", "2026")
    league = os.environ.get("LEAGUE", "j1").lower()
    input_csv = os.environ.get("INPUT_CSV", f"data/{league}_{season}_latest_results.csv")
    output_csv = os.environ.get("OUTPUT_CSV", f"data/{league}_{season}_home_away_profile.csv")

    profile = build_home_away_team_profile_from_csv(input_csv)
    profile.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"出力: {output_csv}")
    print(f"teams: {len(profile)}")


if __name__ == "__main__":
    main()
