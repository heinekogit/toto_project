import argparse
import os
import numpy as np
import pandas as pd


CLASS_ORDER = ["H", "D", "A"]
PROB_COLS = {"H": "prob_home_win", "D": "prob_draw", "A": "prob_away_win"}


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search D decision thresholds on alpha-calibrated probabilities.")
    parser.add_argument("--input", required=True, help="Input CSV with actual_result and prob_* columns")
    parser.add_argument("--out-dir", default="data/reports/metrics_draw_shrink", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.30, help="Draw shrink alpha")
    parser.add_argument("--d-min", type=float, default=0.25)
    parser.add_argument("--d-max", type=float, default=0.40)
    parser.add_argument("--d-step", type=float, default=0.01)
    parser.add_argument("--gap-min", type=float, default=0.05)
    parser.add_argument("--gap-max", type=float, default=0.15)
    parser.add_argument("--gap-step", type=float, default=0.01)
    parser.add_argument("--d-rate-min", type=float, default=0.10)
    parser.add_argument("--d-rate-max", type=float, default=0.40)
    parser.add_argument("--actual-d-rate", type=float, default=0.255263)
    return parser.parse_args()


def frange(start, stop, step):
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(n)]


def apply_draw_shrinkage(df, alpha):
    out = df.copy()
    p_h = out[PROB_COLS["H"]].astype(float).to_numpy()
    p_d = out[PROB_COLS["D"]].astype(float).to_numpy() * alpha
    p_a = out[PROB_COLS["A"]].astype(float).to_numpy()
    total = p_h + p_d + p_a
    total = np.where(total <= 0, 1.0, total)
    out[PROB_COLS["H"]] = p_h / total
    out[PROB_COLS["D"]] = p_d / total
    out[PROB_COLS["A"]] = p_a / total
    return out


def apply_rule(df, d_th, ha_gap_th):
    out = df.copy()
    cond_d = (out[PROB_COLS["D"]] >= d_th) & ((out[PROB_COLS["H"]] - out[PROB_COLS["A"]]).abs() <= ha_gap_th)
    out["predicted_result"] = np.where(cond_d, "D", np.where(out[PROB_COLS["H"]] >= out[PROB_COLS["A"]], "H", "A"))
    return out


def accuracy(df):
    return float((df["predicted_result"] == df["actual_result"]).mean())


def pred_rates(df):
    vc = df["predicted_result"].value_counts(normalize=True).reindex(CLASS_ORDER).fillna(0.0)
    return float(vc["H"]), float(vc["D"]), float(vc["A"])


def confusion_matrix(df):
    return pd.crosstab(
        pd.Categorical(df["actual_result"], categories=CLASS_ORDER, ordered=True),
        pd.Categorical(df["predicted_result"], categories=CLASS_ORDER, ordered=True),
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    need = {"actual_result", "prob_home_win", "prob_draw", "prob_away_win"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(need - set(df.columns))}")
    df = df.dropna(subset=["actual_result", "prob_home_win", "prob_draw", "prob_away_win"]).copy()
    df_cal = apply_draw_shrinkage(df, args.alpha)

    rows = []
    for d_th in frange(args.d_min, args.d_max, args.d_step):
        for gap_th in frange(args.gap_min, args.gap_max, args.gap_step):
            dfr = apply_rule(df_cal, d_th, gap_th)
            acc = accuracy(dfr)
            p_h, p_d, p_a = pred_rates(dfr)
            rows.append(
                {
                    "d_th": d_th,
                    "ha_gap_th": gap_th,
                    "accuracy": acc,
                    "pred_H": p_h,
                    "pred_D": p_d,
                    "pred_A": p_a,
                    "d_rate_gap_from_actual": p_d - args.actual_d_rate,
                }
            )
    grid = pd.DataFrame(rows).sort_values(["accuracy", "pred_D"], ascending=[False, True]).reset_index(drop=True)

    grid_path = os.path.join(args.out_dir, "threshold_grid_evaluation.csv")
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")

    filtered = grid[(grid["pred_D"] >= args.d_rate_min) & (grid["pred_D"] <= args.d_rate_max)].copy()
    if filtered.empty:
        filtered = grid.copy()

    best = filtered.sort_values(["accuracy", "d_rate_gap_from_actual"], ascending=[False, True], key=lambda s: s.abs() if s.name == "d_rate_gap_from_actual" else s).iloc[0]
    best_d_th = float(best["d_th"])
    best_gap = float(best["ha_gap_th"])

    best_df = apply_rule(df_cal, best_d_th, best_gap)
    best_cm = confusion_matrix(best_df)
    best_metrics = pd.DataFrame(
        [
            {"metric": "alpha", "value": args.alpha},
            {"metric": "d_th", "value": best_d_th},
            {"metric": "ha_gap_th", "value": best_gap},
            {"metric": "accuracy", "value": float(best["accuracy"])},
            {"metric": "pred_H", "value": float(best["pred_H"])},
            {"metric": "pred_D", "value": float(best["pred_D"])},
            {"metric": "pred_A", "value": float(best["pred_A"])},
        ]
    )

    best_metrics_path = os.path.join(args.out_dir, "best_rule_metrics.csv")
    best_cm_path = os.path.join(args.out_dir, "best_rule_confusion_matrix.csv")
    legacy_cm_path = os.path.join(args.out_dir, "confusion_matrix.csv")
    best_metrics.to_csv(best_metrics_path, index=False, encoding="utf-8-sig")
    best_cm.to_csv(best_cm_path, encoding="utf-8-sig")
    best_cm.to_csv(legacy_cm_path, encoding="utf-8-sig")

    # candidates for reporting
    top_acc = filtered.sort_values(["accuracy", "d_rate_gap_from_actual"], ascending=[False, True], key=lambda s: s.abs() if s.name == "d_rate_gap_from_actual" else s).head(5)
    near_best = filtered[filtered["accuracy"] >= float(best["accuracy"]) - 0.005].copy()
    if near_best.empty:
        near_best = filtered.copy()
    balanced = near_best.sort_values("d_rate_gap_from_actual", key=lambda s: s.abs()).head(1)

    print("== Top accuracy candidates (filtered by pred_D range) ==")
    print(top_acc.to_string(index=False))
    print("\n== Balanced near-best candidate ==")
    print(balanced.to_string(index=False))
    print("\n== Best rule metrics ==")
    print(best_metrics.to_string(index=False))
    print("\n== Best rule confusion matrix (actual x predicted) ==")
    print(best_cm.to_string())
    print(f"\nSaved: {grid_path}")
    print(f"Saved: {best_metrics_path}")
    print(f"Saved: {best_cm_path}")
    print(f"Saved: {legacy_cm_path}")


if __name__ == "__main__":
    main()
