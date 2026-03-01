import argparse
import os
import subprocess
import sys

import pandas as pd


def summarize(csv_path: str):
    df = pd.read_csv(csv_path)
    if "prob_draw" not in df.columns or "predicted_result" not in df.columns:
        raise RuntimeError(f"required columns missing in {csv_path}")
    rows = len(df)
    avg_draw = float(df["prob_draw"].mean()) if rows else 0.0
    sum_draw = float(df["prob_draw"].sum()) if rows else 0.0
    d_count = int((df["predicted_result"].astype(str) == "D").sum()) if rows else 0
    return rows, avg_draw, sum_draw, d_count


def run_once(
    league: str,
    season: str,
    base: float,
    bump: float,
    sensitivity: float,
    elo_diff_scale: float,
    draw_diff_scale: float,
):
    env = os.environ.copy()
    env["LEAGUE"] = league
    env["SEASON_YEAR"] = season
    env["ELO_DRAW_BASE"] = str(base)
    env["ELO_DRAW_BUMP"] = str(bump)
    env["ELO_DRAW_SENSITIVITY"] = str(sensitivity)
    env["ELO_DIFF_SCALE"] = str(elo_diff_scale)
    env["ELO_DRAW_DIFF_SCALE"] = str(draw_diff_scale)

    cmd = [sys.executable, "scripts/11_prediction_01.py"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

    csv_path = f"{league}_{season}_predictions.csv"
    rows, avg_draw, sum_draw, d_count = summarize(csv_path)

    pred_summary = ""
    for line in proc.stdout.splitlines():
        if line.startswith("[PRED_SUMMARY]"):
            pred_summary = line
            break

    print(
        f"base={base:.3f} bump={bump:.3f} sens={sensitivity:g} diff_scale={draw_diff_scale:.2f} "
        f"rows={rows} avg_prob_draw={avg_draw:.3f} sum_prob_draw={sum_draw:.3f} predicted_D={d_count}"
    )
    if pred_summary:
        print(f"  {pred_summary}")
    return {
        "base": base,
        "bump": bump,
        "sensitivity": sensitivity,
        "diff_scale": draw_diff_scale,
        "rows": rows,
        "avg_prob_draw": avg_draw,
        "sum_prob_draw": sum_draw,
        "predicted_D": d_count,
    }


def main():
    p = argparse.ArgumentParser(description="Sweep draw model params and print summary.")
    p.add_argument("--league", default="j1")
    p.add_argument("--season", default="2026")
    p.add_argument("--bases", nargs="+", type=float, default=[0.28, 0.32, 0.36, 0.40])
    p.add_argument("--bumps", nargs="+", type=float, default=[0.08, 0.12, 0.16])
    p.add_argument("--sensitivities", nargs="+", type=float, default=[180, 240, 320, 420])
    p.add_argument("--elo-diff-scale", type=float, default=1.0)
    p.add_argument("--draw-diff-scales", nargs="+", type=float, default=[1.0, 0.85, 0.70])
    p.add_argument("--target-min", type=float, default=0.28)
    p.add_argument("--target-max", type=float, default=0.30)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    print(
        f"league={args.league} season={args.season} "
        f"ELO_DIFF_SCALE={args.elo_diff_scale} draw_diff_scales={args.draw_diff_scales}"
    )
    all_results = []
    for ds in args.draw_diff_scales:
        for b in args.bases:
            for bump in args.bumps:
                for s in args.sensitivities:
                    row = run_once(
                        args.league,
                        str(args.season),
                        float(b),
                        float(bump),
                        float(s),
                        float(args.elo_diff_scale),
                        float(ds),
                    )
                    all_results.append(row)

    target_mid = (args.target_min + args.target_max) / 2.0
    in_range = [
        r
        for r in all_results
        if args.target_min <= r["avg_prob_draw"] <= args.target_max
    ]
    in_range = sorted(
        in_range,
        key=lambda r: (abs(r["avg_prob_draw"] - target_mid), r["predicted_D"]),
    )
    print("")
    print(
        f"[TOP_CANDIDATES] target_avg_prob_draw={args.target_min:.2f}..{args.target_max:.2f} "
        f"matches={len(in_range)}"
    )
    for r in in_range[: args.top_k]:
        print(
            "  "
            f"base={r['base']:.3f} bump={r['bump']:.3f} sens={r['sensitivity']:g} diff_scale={r['diff_scale']:.2f} "
            f"avg_prob_draw={r['avg_prob_draw']:.3f} sum_prob_draw={r['sum_prob_draw']:.3f} "
            f"predicted_D={r['predicted_D']}"
        )


if __name__ == "__main__":
    main()
