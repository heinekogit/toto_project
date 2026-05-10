#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def _safe_mean(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _safe_median(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").median())


def build_outputs(backtest_csv: Path, out_dir: Path, league: str, season_year: int):
    df = pd.read_csv(backtest_csv)
    if "argmax_result" not in df.columns:
        raise SystemExit(f"argmax_result not found: {backtest_csv}")
    if "actual_result" not in df.columns:
        raise SystemExit(f"actual_result not found: {backtest_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    league_key = league.lower()

    detail_mask = (df["actual_result"] == "A") & (df["argmax_result"] == "H")
    detail_cols = [
        "match_id",
        "節",
        "datetime",
        "home_team",
        "away_team",
        "actual_result",
        "argmax_result",
        "argmax_max_prob",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "elo_diff_before_hfa",
        "hfa_added_to_diff",
        "elo_diff_after_hfa",
        "elo_diff_for_prob",
        "home_advantage_diff",
        "home_advantage_profile_diff",
    ]
    detail_cols = [c for c in detail_cols if c in df.columns]
    detail_df = df.loc[detail_mask, detail_cols].copy()
    detail_df = detail_df.sort_values(
        by="argmax_max_prob" if "argmax_max_prob" in detail_df.columns else detail_cols[0],
        ascending=False,
    )

    groups = {
        "all": df.index == df.index,
        "actual_A_all": df["actual_result"] == "A",
        "actual_A_predH": (df["actual_result"] == "A") & (df["argmax_result"] == "H"),
        "actual_A_predA": (df["actual_result"] == "A") & (df["argmax_result"] == "A"),
        "actual_D_all": df["actual_result"] == "D",
        "actual_D_predH": (df["actual_result"] == "D") & (df["argmax_result"] == "H"),
        "pred_H_hit": (df["actual_result"] == "H") & (df["argmax_result"] == "H"),
    }
    metric_cols = [
        "argmax_max_prob",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "elo_diff_before_hfa",
        "hfa_added_to_diff",
        "elo_diff_after_hfa",
        "elo_diff_for_prob",
        "home_advantage_diff",
        "home_advantage_profile_diff",
    ]

    rows = []
    for name, mask in groups.items():
        sub = df.loc[mask].copy()
        row = {
            "segment": name,
            "matches": int(len(sub)),
        }
        for col in metric_cols:
            if col in df.columns:
                row[f"{col}_mean"] = _safe_mean(sub, col)
                row[f"{col}_median"] = _safe_median(sub, col)
        rows.append(row)
    summary_df = pd.DataFrame(rows)

    summary_path = out_dir / f"{league_key}_{season_year}_home_bias_summary.csv"
    detail_path = out_dir / f"{league_key}_{season_year}_home_bias_detail.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    top = summary_df.loc[summary_df["segment"] == "actual_A_predH"].iloc[0]
    print(
        f"[HOME_BIAS_DIAG] league={league.upper()} season={season_year} "
        f"actual_A_predH={int(top['matches'])} "
        f"elo_before={top.get('elo_diff_before_hfa_mean')} "
        f"hfa_add={top.get('hfa_added_to_diff_mean')} "
        f"elo_after={top.get('elo_diff_after_hfa_mean')} "
        f"elo_for_prob={top.get('elo_diff_for_prob_mean')}"
    )
    print(f"[HOME_BIAS_DIAG_SAVE] summary={summary_path} detail={detail_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--league", default="j2")
    parser.add_argument("--season-year", type=int, default=2026)
    args = parser.parse_args()

    build_outputs(
        backtest_csv=Path(args.backtest_csv),
        out_dir=Path(args.out_dir),
        league=args.league,
        season_year=args.season_year,
    )


if __name__ == "__main__":
    main()
