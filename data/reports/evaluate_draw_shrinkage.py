import argparse
import os
import numpy as np
import pandas as pd


CLASS_ORDER = ["H", "D", "A"]
PROB_COLS = {
    "H": "prob_home_win",
    "D": "prob_draw",
    "A": "prob_away_win",
}
DRAW_PROB_THRESHOLD = 0.45
DRAW_BALANCE_THRESHOLD = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate draw shrinkage calibration.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--out-dir", default="data/reports/metrics_draw_shrink", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.45, help="Primary alpha for detailed report")
    parser.add_argument("--alpha-min", type=float, default=0.35, help="Grid search min alpha")
    parser.add_argument("--alpha-max", type=float, default=0.60, help="Grid search max alpha")
    parser.add_argument("--alpha-step", type=float, default=0.01, help="Grid search step")
    return parser.parse_args()


def decide_predicted_result(prob_home_win, prob_draw, prob_away_win):
    if pd.isna(prob_home_win) or pd.isna(prob_draw) or pd.isna(prob_away_win):
        return None
    if prob_draw >= DRAW_PROB_THRESHOLD and abs(prob_home_win - prob_away_win) <= DRAW_BALANCE_THRESHOLD:
        return "D"
    return "H" if prob_home_win >= prob_away_win else "A"


def apply_draw_shrinkage(df, alpha):
    out = df.copy()
    p_h = out[PROB_COLS["H"]].astype(float).to_numpy()
    p_d = out[PROB_COLS["D"]].astype(float).to_numpy()
    p_a = out[PROB_COLS["A"]].astype(float).to_numpy()

    p_d = p_d * alpha
    total = p_h + p_d + p_a
    total = np.where(total <= 0, 1.0, total)
    p_h = p_h / total
    p_d = p_d / total
    p_a = p_a / total

    out[PROB_COLS["H"]] = p_h
    out[PROB_COLS["D"]] = p_d
    out[PROB_COLS["A"]] = p_a
    out["predicted_result"] = out.apply(
        lambda r: decide_predicted_result(r[PROB_COLS["H"]], r[PROB_COLS["D"]], r[PROB_COLS["A"]]),
        axis=1,
    )
    return out


def compute_metrics(df):
    y = df["actual_result"].astype(str)
    p = df[[PROB_COLS["H"], PROB_COLS["D"], PROB_COLS["A"]]].astype(float).to_numpy()
    p = np.clip(p, 1e-15, 1 - 1e-15)
    p = p / p.sum(axis=1, keepdims=True)

    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_idx = y.map(class_to_idx).to_numpy()
    y_one_hot = np.eye(3)[y_idx]
    pred = df["predicted_result"].astype(str)

    accuracy = float((pred == y).mean())
    log_loss = float(-np.mean(np.log(p[np.arange(len(df)), y_idx])))
    brier_3class = float(np.mean(np.sum((p - y_one_hot) ** 2, axis=1)))
    return accuracy, log_loss, brier_3class


def class_balance(df):
    actual_rate = (
        df["actual_result"].value_counts(normalize=True).reindex(CLASS_ORDER).fillna(0).rename("actual_rate")
    )
    mean_pred = pd.Series(
        {c: float(df[PROB_COLS[c]].mean()) for c in CLASS_ORDER},
        name="mean_pred_prob",
    )
    out = pd.concat([actual_rate, mean_pred], axis=1)
    out["diff_pred_minus_actual"] = out["mean_pred_prob"] - out["actual_rate"]
    out.index.name = "class"
    return out.reset_index()


def confusion_matrix(df):
    return pd.crosstab(
        pd.Categorical(df["actual_result"], categories=CLASS_ORDER, ordered=True),
        pd.Categorical(df["predicted_result"], categories=CLASS_ORDER, ordered=True),
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    )


def make_alpha_grid(alpha_min, alpha_max, alpha_step):
    count = int(round((alpha_max - alpha_min) / alpha_step)) + 1
    vals = [round(alpha_min + i * alpha_step, 6) for i in range(count)]
    return [a for a in vals if alpha_min - 1e-9 <= a <= alpha_max + 1e-9]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_raw = pd.read_csv(args.input)
    need = {"actual_result", "prob_home_win", "prob_draw", "prob_away_win"}
    if not need.issubset(df_raw.columns):
        raise ValueError(f"Missing required columns: {sorted(need - set(df_raw.columns))}")
    df_raw = df_raw.dropna(subset=["actual_result", "prob_home_win", "prob_draw", "prob_away_win"]).copy()

    grid_rows = []
    for alpha in make_alpha_grid(args.alpha_min, args.alpha_max, args.alpha_step):
        dfa = apply_draw_shrinkage(df_raw, alpha)
        acc, ll, bs = compute_metrics(dfa)
        grid_rows.append(
            {
                "alpha": alpha,
                "accuracy": acc,
                "log_loss_3class": ll,
                "brier_score_3class": bs,
                "mean_prob_home_win": float(dfa[PROB_COLS["H"]].mean()),
                "mean_prob_draw": float(dfa[PROB_COLS["D"]].mean()),
                "mean_prob_away_win": float(dfa[PROB_COLS["A"]].mean()),
            }
        )
    grid_df = pd.DataFrame(grid_rows).sort_values("alpha").reset_index(drop=True)
    best = grid_df.sort_values("log_loss_3class", ascending=True).iloc[0]
    best_alpha = float(best["alpha"])

    # detailed report for requested alpha
    alpha_df = apply_draw_shrinkage(df_raw, args.alpha)
    acc, ll, bs = compute_metrics(alpha_df)
    alpha_metrics_df = pd.DataFrame(
        [
            {"metric": "alpha", "value": args.alpha},
            {"metric": "accuracy", "value": acc},
            {"metric": "log_loss_3class", "value": ll},
            {"metric": "brier_score_3class", "value": bs},
        ]
    )
    alpha_balance_df = class_balance(alpha_df)
    alpha_cm_df = confusion_matrix(alpha_df)

    # detailed report for best alpha
    best_df = apply_draw_shrinkage(df_raw, best_alpha)
    best_balance_df = class_balance(best_df)
    best_cm_df = confusion_matrix(best_df)

    grid_path = os.path.join(args.out_dir, "alpha_grid_evaluation.csv")
    alpha_metrics_path = os.path.join(args.out_dir, f"alpha_{args.alpha:.2f}_metrics.csv")
    alpha_balance_path = os.path.join(args.out_dir, f"alpha_{args.alpha:.2f}_class_balance.csv")
    alpha_cm_path = os.path.join(args.out_dir, f"alpha_{args.alpha:.2f}_confusion_matrix.csv")
    best_balance_path = os.path.join(args.out_dir, f"alpha_{best_alpha:.2f}_best_class_balance.csv")
    best_cm_path = os.path.join(args.out_dir, f"alpha_{best_alpha:.2f}_best_confusion_matrix.csv")

    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")
    alpha_metrics_df.to_csv(alpha_metrics_path, index=False, encoding="utf-8-sig")
    alpha_balance_df.to_csv(alpha_balance_path, index=False, encoding="utf-8-sig")
    alpha_cm_df.to_csv(alpha_cm_path, encoding="utf-8-sig")
    best_balance_df.to_csv(best_balance_path, index=False, encoding="utf-8-sig")
    best_cm_df.to_csv(best_cm_path, encoding="utf-8-sig")

    print("== Alpha Grid (head) ==")
    print(grid_df.head(10).to_string(index=False))
    print("\n== Alpha Grid (best by log loss) ==")
    print(best.to_frame().T.to_string(index=False))

    print(f"\n== Detailed alpha={args.alpha:.2f} metrics ==")
    print(alpha_metrics_df.to_string(index=False))
    print("\n== Detailed class balance ==")
    print(alpha_balance_df.to_string(index=False))
    print("\n== Detailed confusion matrix (actual x predicted) ==")
    print(alpha_cm_df.to_string())

    print(f"\nSaved: {grid_path}")
    print(f"Saved: {alpha_metrics_path}")
    print(f"Saved: {alpha_balance_path}")
    print(f"Saved: {alpha_cm_path}")
    print(f"Saved: {best_balance_path}")
    print(f"Saved: {best_cm_path}")


if __name__ == "__main__":
    main()
