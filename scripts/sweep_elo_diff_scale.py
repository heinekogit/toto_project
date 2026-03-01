import argparse
import os
import subprocess
import sys

import pandas as pd


def summarize(csv_path: str):
    df = pd.read_csv(csv_path)
    required = {"prob_draw", "predicted_result"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"required columns missing in {csv_path}: {required}")
    rows = len(df)
    avg_draw = float(df["prob_draw"].mean()) if rows else 0.0
    sum_draw = float(df["prob_draw"].sum()) if rows else 0.0
    d_count = int((df["predicted_result"].astype(str) == "D").sum()) if rows else 0
    return rows, avg_draw, sum_draw, d_count


def run_once(scale: float, league: str, season: str):
    env = os.environ.copy()
    env["LEAGUE"] = league
    env["SEASON_YEAR"] = season
    env["ELO_DIFF_SCALE"] = f"{scale:.2f}"
    cmd = [sys.executable, "scripts/11_prediction_01.py"]
    proc = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

    out_csv = f"{league}_{season}_predictions.csv"
    rows, avg_draw, sum_draw, d_count = summarize(out_csv)
    pred_summary = ""
    for line in proc.stdout.splitlines():
        if line.startswith("[PRED_SUMMARY]"):
            pred_summary = line
            break
    print(
        f"scale={scale:.2f} rows={rows} avg_prob_draw={avg_draw:.3f} "
        f"sum_prob_draw={sum_draw:.3f} predicted_D={d_count}"
    )
    if pred_summary:
        print(f"  {pred_summary}")


def main():
    parser = argparse.ArgumentParser(description="Sweep ELO_DIFF_SCALE and report draw stats.")
    parser.add_argument("--league", default="j1")
    parser.add_argument("--season", default="2026")
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[1.00, 0.90, 0.80, 0.70, 0.60],
    )
    args = parser.parse_args()

    print(f"league={args.league} season={args.season}")
    for s in args.scales:
        run_once(float(s), args.league, str(args.season))


if __name__ == "__main__":
    main()
