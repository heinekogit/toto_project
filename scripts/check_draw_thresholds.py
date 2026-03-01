import argparse
from pathlib import Path

import pandas as pd


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def decide_row(ph: float, pdw: float, pa: float, thr: float) -> str:
    if pd.isna(ph) or pd.isna(pdw) or pd.isna(pa):
        return ""
    if pdw >= thr:
        return "D"
    if ph >= pdw and ph >= pa:
        return "H"
    if pdw >= ph and pdw >= pa:
        return "D"
    return "A"


def main():
    p = argparse.ArgumentParser(description="Check D count under multiple DRAW thresholds.")
    p.add_argument("--csv", default="data/purchase_reference/predictions.csv")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.24, 0.28, 0.30, 0.32])
    args = p.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)

    ph_col = pick_col(df, ["prob_home_win", "p_home"])
    pd_col = pick_col(df, ["prob_draw", "p_draw"])
    pa_col = pick_col(df, ["prob_away_win", "p_away"])
    if not ph_col or not pd_col or not pa_col:
        raise SystemExit(
            f"required cols missing: need one of "
            f"prob_home_win/p_home, prob_draw/p_draw, prob_away_win/p_away in {path}"
        )

    n = len(df)
    sum_draw = float(df[pd_col].sum())
    print(f"file={path}")
    print(f"rows={n}")
    print(f"sum(prob_draw)={sum_draw:.3f} (expected D count ~= {sum_draw:.1f})")
    print("")

    for thr in args.thresholds:
        pred = df.apply(lambda r: decide_row(r[ph_col], r[pd_col], r[pa_col], thr), axis=1)
        d_count = int((pred == "D").sum())
        print(f"thr={thr:.2f} -> D count={d_count}")


if __name__ == "__main__":
    main()
