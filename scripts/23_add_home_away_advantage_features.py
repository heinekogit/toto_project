import os
import pandas as pd


REQUIRED_MATCH_COLUMNS = ["home_team", "away_team"]
REQUIRED_PROFILE_COLUMNS = ["team", "home_points_per_match", "away_points_per_match"]


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}に必要列が不足しています: {missing}")


def add_home_away_advantage_features(
    matches_df: pd.DataFrame,
    team_profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    STEP1のチーム別ホーム/アウェイ成績を使って、
    試合一覧に home_advantage_diff と is_home_advantage_positive を付与する。
    """
    _validate_columns(matches_df, REQUIRED_MATCH_COLUMNS, "matches_df")
    _validate_columns(team_profile_df, REQUIRED_PROFILE_COLUMNS, "team_profile_df")

    matches = matches_df.copy()
    profile = team_profile_df.copy()

    matches["home_team"] = matches["home_team"].astype(str).str.strip()
    matches["away_team"] = matches["away_team"].astype(str).str.strip()
    profile["team"] = profile["team"].astype(str).str.strip()

    home_lookup = profile[["team", "home_points_per_match"]].rename(
        columns={"team": "home_team", "home_points_per_match": "_home_ppm"}
    )
    away_lookup = profile[["team", "away_points_per_match"]].rename(
        columns={"team": "away_team", "away_points_per_match": "_away_ppm"}
    )

    out = matches.merge(home_lookup, on="home_team", how="left")
    out = out.merge(away_lookup, on="away_team", how="left")

    out["_home_ppm"] = pd.to_numeric(out["_home_ppm"], errors="coerce").fillna(0.0)
    out["_away_ppm"] = pd.to_numeric(out["_away_ppm"], errors="coerce").fillna(0.0)

    out["home_advantage_diff"] = (out["_home_ppm"] - out["_away_ppm"]).round(4)
    out["is_home_advantage_positive"] = out["home_advantage_diff"] > 0

    return out.drop(columns=["_home_ppm", "_away_ppm"])


def add_home_away_advantage_features_from_csv(
    matches_csv_path: str,
    team_profile_csv_path: str,
) -> pd.DataFrame:
    matches_df = pd.read_csv(matches_csv_path)
    team_profile_df = pd.read_csv(team_profile_csv_path)
    return add_home_away_advantage_features(matches_df, team_profile_df)


def main() -> None:
    season = os.environ.get("SEASON_YEAR", "2025")
    league = os.environ.get("LEAGUE", "j1").lower()
    input_matches_csv = os.environ.get("INPUT_MATCHES_CSV", f"backtest_{league}_{season}_rounds.csv")
    input_profile_csv = os.environ.get("INPUT_PROFILE_CSV", f"data/{league}_{season}_home_away_profile.csv")
    output_csv = os.environ.get("OUTPUT_CSV", f"data/{league}_{season}_matches_with_home_away_features.csv")

    out_df = add_home_away_advantage_features_from_csv(
        matches_csv_path=input_matches_csv,
        team_profile_csv_path=input_profile_csv,
    )
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"出力: {output_csv}")
    print(f"rows: {len(out_df)}")


if __name__ == "__main__":
    main()
