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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3-class probability quality.")
    parser.add_argument("--input", required=True, help="Input CSV with actual/prob columns")
    parser.add_argument("--out-dir", default="data/reports/metrics", help="Output directory for CSVs")
    return parser.parse_args()


def ensure_predicted_result(df):
    if "predicted_result" in df.columns:
        return df
    out = df.copy()
    missing = [PROB_COLS[c] for c in CLASS_ORDER if PROB_COLS[c] not in out.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    prob_matrix = out[[PROB_COLS["H"], PROB_COLS["D"], PROB_COLS["A"]]].to_numpy()
    idx = np.argmax(prob_matrix, axis=1)
    out["predicted_result"] = [CLASS_ORDER[i] for i in idx]
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

    return pd.DataFrame(
        [
            {"metric": "accuracy", "value": accuracy},
            {"metric": "log_loss_3class", "value": log_loss},
            {"metric": "brier_score_3class", "value": brier_3class},
        ]
    )


def compute_class_balance(df):
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


def compute_calibration(df, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = [f"[{edges[i]:.1f},{edges[i + 1]:.1f}]" for i in range(bins)]
    rows = []
    for c in CLASS_ORDER:
        prob_col = PROB_COLS[c]
        tmp = pd.DataFrame(
            {
                "prob": df[prob_col].astype(float),
                "actual": (df["actual_result"] == c).astype(int),
            }
        )
        tmp["bin"] = pd.cut(tmp["prob"], bins=edges, labels=labels, include_lowest=True)
        g = tmp.groupby("bin", observed=False)
        table = g.agg(
            count=("prob", "size"),
            mean_pred_prob=("prob", "mean"),
            actual_rate=("actual", "mean"),
        ).reset_index()
        table["class"] = c
        table = table[["class", "bin", "count", "mean_pred_prob", "actual_rate"]]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def compute_confusion_matrix(df):
    cm = pd.crosstab(
        pd.Categorical(df["actual_result"], categories=CLASS_ORDER, ordered=True),
        pd.Categorical(df["predicted_result"], categories=CLASS_ORDER, ordered=True),
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    )
    return cm


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"actual_result", "prob_home_win", "prob_draw", "prob_away_win"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    df = df.dropna(subset=["actual_result", "prob_home_win", "prob_draw", "prob_away_win"]).copy()
    df = ensure_predicted_result(df)

    metrics_df = compute_metrics(df)
    balance_df = compute_class_balance(df)
    calib_df = compute_calibration(df, bins=10)
    cm_df = compute_confusion_matrix(df)

    metrics_path = os.path.join(args.out_dir, "metrics_summary.csv")
    balance_path = os.path.join(args.out_dir, "class_balance_vs_mean_prob.csv")
    calib_path = os.path.join(args.out_dir, "calibration_table_10bins.csv")
    cm_path = os.path.join(args.out_dir, "confusion_matrix.csv")

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    balance_df.to_csv(balance_path, index=False, encoding="utf-8-sig")
    calib_df.to_csv(calib_path, index=False, encoding="utf-8-sig")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    print("== Metrics ==")
    print(metrics_df.to_string(index=False))
    print("\n== Class Balance vs Mean Pred Prob ==")
    print(balance_df.to_string(index=False))
    print("\n== Confusion Matrix (actual x predicted) ==")
    print(cm_df.to_string())
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {balance_path}")
    print(f"Saved: {calib_path}")
    print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()
