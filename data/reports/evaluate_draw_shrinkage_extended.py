import argparse
import os
import numpy as np
import pandas as pd


CLASS_ORDER = ["H", "D", "A"]
PROB_COLS = {"H": "prob_home_win", "D": "prob_draw", "A": "prob_away_win"}
DRAW_PROB_THRESHOLD = 0.45
DRAW_BALANCE_THRESHOLD = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description="Extended alpha search for draw shrinkage.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--out-dir", default="data/reports/metrics_draw_shrink", help="Output directory")
    parser.add_argument("--alpha-min", type=float, default=0.10)
    parser.add_argument("--alpha-max", type=float, default=0.35)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--logloss-window", type=float, default=0.01)
    parser.add_argument("--d-gap-target", type=float, default=0.03)
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
    p_d = out[PROB_COLS["D"]].astype(float).to_numpy() * alpha
    p_a = out[PROB_COLS["A"]].astype(float).to_numpy()

    total = p_h + p_d + p_a
    total = np.where(total <= 0, 1.0, total)
    out[PROB_COLS["H"]] = p_h / total
    out[PROB_COLS["D"]] = p_d / total
    out[PROB_COLS["A"]] = p_a / total

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
    log_loss_3class = float(-np.mean(np.log(p[np.arange(len(df)), y_idx])))
    brier_score_3class = float(np.mean(np.sum((p - y_one_hot) ** 2, axis=1)))
    return accuracy, log_loss_3class, brier_score_3class


def class_balance(df):
    actual_rate = df["actual_result"].value_counts(normalize=True).reindex(CLASS_ORDER).fillna(0.0)
    mean_pred = pd.Series({c: float(df[PROB_COLS[c]].mean()) for c in CLASS_ORDER})
    out = pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "actual_rate": [actual_rate[c] for c in CLASS_ORDER],
            "mean_pred_prob": [mean_pred[c] for c in CLASS_ORDER],
        }
    )
    out["diff_pred_minus_actual"] = out["mean_pred_prob"] - out["actual_rate"]
    return out


def confusion_matrix(df):
    return pd.crosstab(
        pd.Categorical(df["actual_result"], categories=CLASS_ORDER, ordered=True),
        pd.Categorical(df["predicted_result"], categories=CLASS_ORDER, ordered=True),
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    )


def alpha_values(alpha_min, alpha_max, alpha_step):
    n = int(round((alpha_max - alpha_min) / alpha_step)) + 1
    vals = [round(alpha_min + i * alpha_step, 6) for i in range(n)]
    return [a for a in vals if alpha_min - 1e-9 <= a <= alpha_max + 1e-9]


def build_grid(df_raw, alpha_min, alpha_max, alpha_step):
    rows = []
    actual = df_raw["actual_result"].value_counts(normalize=True).reindex(CLASS_ORDER).fillna(0.0)
    for alpha in alpha_values(alpha_min, alpha_max, alpha_step):
        dfa = apply_draw_shrinkage(df_raw, alpha)
        acc, ll, bs = compute_metrics(dfa)
        mean_h = float(dfa[PROB_COLS["H"]].mean())
        mean_d = float(dfa[PROB_COLS["D"]].mean())
        mean_a = float(dfa[PROB_COLS["A"]].mean())
        d_gap = mean_d - float(actual["D"])
        h_gap = mean_h - float(actual["H"])
        a_gap = mean_a - float(actual["A"])
        rows.append(
            {
                "alpha": alpha,
                "accuracy": acc,
                "log_loss_3class": ll,
                "brier_score_3class": bs,
                "actual_rate_H": float(actual["H"]),
                "actual_rate_D": float(actual["D"]),
                "actual_rate_A": float(actual["A"]),
                "mean_pred_H": mean_h,
                "mean_pred_D": mean_d,
                "mean_pred_A": mean_a,
                "h_mean_gap": h_gap,
                "d_mean_gap": d_gap,
                "a_mean_gap": a_gap,
                "ha_gap_abs_max": max(abs(h_gap), abs(a_gap)),
            }
        )
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def choose_candidates(grid_df, logloss_window, d_gap_target):
    min_ll = float(grid_df["log_loss_3class"].min())
    vicinity = grid_df[grid_df["log_loss_3class"] <= min_ll + logloss_window].copy()
    if vicinity.empty:
        vicinity = grid_df.copy()

    preferred = vicinity[vicinity["d_mean_gap"].abs() <= d_gap_target].copy()
    if preferred.empty:
        preferred = vicinity.copy()

    preferred = preferred.sort_values(
        ["d_mean_gap", "ha_gap_abs_max", "log_loss_3class"],
        key=lambda s: s.abs() if s.name == "d_mean_gap" else s,
        ascending=True,
    ).reset_index(drop=True)

    best = preferred.iloc[0]
    rest = preferred[preferred["alpha"] != best["alpha"]]
    if rest.empty:
        rest = grid_df[grid_df["alpha"] != best["alpha"]].copy()
        rest = rest.sort_values(["log_loss_3class", "ha_gap_abs_max", "d_mean_gap"], key=lambda s: s.abs() if s.name == "d_mean_gap" else s)
    second = rest.iloc[0]
    return best, second


def save_candidate_outputs(df_raw, alpha, out_dir, prefix):
    dfa = apply_draw_shrinkage(df_raw, alpha)
    acc, ll, bs = compute_metrics(dfa)
    metrics_df = pd.DataFrame(
        [
            {"metric": "alpha", "value": float(alpha)},
            {"metric": "accuracy", "value": acc},
            {"metric": "log_loss_3class", "value": ll},
            {"metric": "brier_score_3class", "value": bs},
        ]
    )
    balance_df = class_balance(dfa)
    cm_df = confusion_matrix(dfa)

    metrics_path = os.path.join(out_dir, f"{prefix}_metrics.csv")
    balance_path = os.path.join(out_dir, f"{prefix}_class_balance.csv")
    cm_path = os.path.join(out_dir, f"{prefix}_confusion_matrix.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    balance_df.to_csv(balance_path, index=False, encoding="utf-8-sig")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")
    return metrics_path, balance_path, cm_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_raw = pd.read_csv(args.input)
    required = {"actual_result", "prob_home_win", "prob_draw", "prob_away_win"}
    if not required.issubset(df_raw.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df_raw.columns))}")
    df_raw = df_raw.dropna(subset=["actual_result", "prob_home_win", "prob_draw", "prob_away_win"]).copy()

    grid_df = build_grid(df_raw, args.alpha_min, args.alpha_max, args.alpha_step)
    best, second = choose_candidates(grid_df, args.logloss_window, args.d_gap_target)

    grid_path = os.path.join(args.out_dir, "alpha_grid_evaluation_extended.csv")
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")

    best_paths = save_candidate_outputs(df_raw, float(best["alpha"]), args.out_dir, "best_alpha")
    second_paths = save_candidate_outputs(df_raw, float(second["alpha"]), args.out_dir, "second_best_alpha")

    print("== Alpha Grid (top by log loss) ==")
    print(grid_df.sort_values("log_loss_3class").head(12).to_string(index=False))
    print("\n== Recommended alpha ==")
    print(best.to_frame().T.to_string(index=False))
    print("\n== Second-best alpha ==")
    print(second.to_frame().T.to_string(index=False))
    print(f"\nSaved: {grid_path}")
    print(f"Saved: {best_paths[0]}")
    print(f"Saved: {best_paths[1]}")
    print(f"Saved: {best_paths[2]}")
    print(f"Saved: {second_paths[0]}")
    print(f"Saved: {second_paths[1]}")
    print(f"Saved: {second_paths[2]}")


if __name__ == "__main__":
    main()
