#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/Users/dev_tomo/Desktop/tt_prj_restart")
METRICS_DIR = BASE_DIR / "data" / "reports" / "metrics"


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_backtest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty csv: {path}")
    return df


def _build_match_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["elo_diff_for_prob_num"] = _safe_num(out["elo_diff_for_prob"])
    out["xg_diff"] = _safe_num(out["stats_ゴール期待値_home"]) - _safe_num(out["stats_ゴール期待値_away"])
    out["rank_gap"] = _safe_num(out["rankmot_rank_latest_home"]) - _safe_num(out["rankmot_rank_latest_away"])

    out["sig_away_elo"] = out["elo_diff_for_prob_num"] <= -40.0
    out["sig_away_xg"] = out["xg_diff"] <= -1.0
    out["sig_away_rank"] = out["rank_gap"] >= 3.0
    out["sig_home_elo"] = out["elo_diff_for_prob_num"] >= 40.0
    out["sig_home_xg"] = out["xg_diff"] >= 1.0
    out["sig_home_rank"] = out["rank_gap"] <= -3.0

    out["away_signal_count"] = (
        out[["sig_away_elo", "sig_away_xg", "sig_away_rank"]].fillna(False).sum(axis=1).astype(int)
    )
    out["home_signal_count"] = (
        out[["sig_home_elo", "sig_home_xg", "sig_home_rank"]].fillna(False).sum(axis=1).astype(int)
    )

    close_mask = (
        out["elo_diff_for_prob_num"].abs() <= 20.0
    ) & (
        out["xg_diff"].abs() <= 1.0
    ) & (
        out["rank_gap"].abs() <= 2.0
    )
    away_strong_mask = out["away_signal_count"] >= 2
    home_strong_mask = out["home_signal_count"] >= 2
    conflict_mask = (
        (out["away_signal_count"] >= 1) & (out["home_signal_count"] >= 1)
    ) & ~away_strong_mask & ~home_strong_mask & ~close_mask

    out["match_type"] = "neutral"
    out.loc[close_mask, "match_type"] = "close_match"
    out.loc[conflict_mask, "match_type"] = "signal_conflict"
    out.loc[away_strong_mask, "match_type"] = "away_strong"
    out.loc[home_strong_mask, "match_type"] = "home_strong"

    out["hit"] = (
        out["predicted_result"].astype(str).str.upper() == out["actual_result"].astype(str).str.upper()
    ).astype(int)
    return out


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_type, part in df.groupby("match_type", dropna=False):
        pred = part["predicted_result"].astype(str).str.upper()
        act = part["actual_result"].astype(str).str.upper()
        rows.append(
            {
                "match_type": match_type,
                "matches": int(len(part)),
                "hits": int(part["hit"].sum()),
                "hit_rate": float(part["hit"].mean()),
                "pred_H": int((pred == "H").sum()),
                "pred_D": int((pred == "D").sum()),
                "pred_A": int((pred == "A").sum()),
                "act_H": int((act == "H").sum()),
                "act_D": int((act == "D").sum()),
                "act_A": int((act == "A").sum()),
                "elo_diff_mean": float(_safe_num(part["elo_diff_for_prob_num"]).mean()),
                "xg_diff_mean": float(_safe_num(part["xg_diff"]).mean()),
                "rank_gap_mean": float(_safe_num(part["rank_gap"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["matches", "hit_rate"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True, choices=["j1", "j2"])
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--input-csv", default="")
    args = parser.parse_args()

    input_path = Path(args.input_csv) if args.input_csv else BASE_DIR / f"backtest_{args.league}_{args.season}.csv"
    detail_path = METRICS_DIR / f"match_type_detail_{args.league}_{args.season}.csv"
    summary_path = METRICS_DIR / f"match_type_summary_{args.league}_{args.season}.csv"

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    detail = _build_match_types(_load_backtest(input_path))
    summary = _build_summary(detail)

    keep_cols = [
        "match_type",
        "league",
        "節",
        "match_id",
        "home_team",
        "away_team",
        "actual_result",
        "predicted_result",
        "hit",
        "elo_diff_for_prob_num",
        "xg_diff",
        "rank_gap",
        "away_signal_count",
        "home_signal_count",
        "sig_away_elo",
        "sig_away_xg",
        "sig_away_rank",
        "sig_home_elo",
        "sig_home_xg",
        "sig_home_rank",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "decision_reason",
    ]
    detail[[c for c in keep_cols if c in detail.columns]].to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(summary.to_csv(index=False))
    print(f"[OK] detail={detail_path}")
    print(f"[OK] summary={summary_path}")


if __name__ == "__main__":
    main()
