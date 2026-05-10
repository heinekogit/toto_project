#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
LABEL_TO_IDX = {"H": 0, "D": 1, "A": 2}


@dataclass(frozen=True)
class ParamSet:
    draw_boost: float
    temperature: float
    home_shift: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "draw_boost": float(self.draw_boost),
            "temperature": float(self.temperature),
            "home_shift": float(self.home_shift),
        }


def parse_float_grid(text: str) -> List[float]:
    vals = [s.strip() for s in text.split(",") if s.strip()]
    if not vals:
        raise ValueError(f"empty grid: {text}")
    return [float(v) for v in vals]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward tuning on existing backtest probabilities (post-calibration)."
    )
    p.add_argument(
        "--inputs",
        default="backtest_j1_2025.csv,backtest_j2_2025.csv,backtest_j1_2026.csv,backtest_j2_2026.csv",
        help="Comma-separated input CSV paths (must include 節, actual_result, prob_* columns).",
    )
    p.add_argument("--outdir", default="data/reports/metrics", help="Output directory")
    p.add_argument("--min-train-round", type=int, default=5, help="Minimum training rounds before first fold")
    p.add_argument("--draw-boost-grid", default="1.00,1.02,1.04,1.06,1.08,1.10")
    p.add_argument("--temperature-grid", default="0.95,1.00,1.05,1.10")
    p.add_argument("--home-shift-grid", default="-0.02,-0.01,0.00,0.01,0.02")
    p.add_argument(
        "--objective",
        choices=["logloss", "brier", "hybrid"],
        default="hybrid",
        help="Metric for selecting params on each training fold",
    )
    p.add_argument(
        "--label",
        default="walkforward_hda_tune",
        help="Output file prefix",
    )
    return p.parse_args()


def extract_round_no(val: object) -> Optional[int]:
    if pd.isna(val):
        return None
    s = str(val).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    import re

    m = re.search(r"第\s*(\d+)\s*節", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", s)
    return int(m.group(1)) if m else None


def load_df(paths: Sequence[str]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] skip missing input: {path}")
            continue
        df = pd.read_csv(path)
        df["_source_file"] = path.name
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No readable input CSV found.")
    df = pd.concat(dfs, ignore_index=True)

    required = ["actual_result", "prob_home_win", "prob_draw", "prob_away_win", "節"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}")

    out = df.copy()
    out["actual_result"] = out["actual_result"].astype(str).str.upper()
    out = out[out["actual_result"].isin(["H", "D", "A"])].copy()
    out["_round_no"] = out["節"].apply(extract_round_no)
    out = out[out["_round_no"].notna()].copy()
    out["_round_no"] = out["_round_no"].astype(int)

    for c in ["prob_home_win", "prob_draw", "prob_away_win"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["prob_home_win", "prob_draw", "prob_away_win"]).copy()

    s = out[["prob_home_win", "prob_draw", "prob_away_win"]].sum(axis=1)
    out = out[s > 0].copy()
    out[["prob_home_win", "prob_draw", "prob_away_win"]] = out[
        ["prob_home_win", "prob_draw", "prob_away_win"]
    ].div(s[s > 0], axis=0)
    return out


def apply_transform(base_probs: np.ndarray, p: ParamSet) -> np.ndarray:
    probs = np.clip(base_probs.copy(), EPS, 1.0)
    probs[:, 1] *= p.draw_boost

    # Home/Away shift: move mass between H and A while keeping D untouched first.
    probs[:, 0] = np.clip(probs[:, 0] + p.home_shift, EPS, None)
    probs[:, 2] = np.clip(probs[:, 2] - p.home_shift, EPS, None)

    probs = probs / probs.sum(axis=1, keepdims=True)

    # Temperature scaling (T>1 flatter, T<1 sharper)
    inv_t = 1.0 / max(float(p.temperature), EPS)
    probs = np.power(np.clip(probs, EPS, 1.0), inv_t)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def metrics(y_idx: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    probs = np.clip(probs, EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    n = len(y_idx)
    ll = float(-np.mean(np.log(probs[np.arange(n), y_idx])))

    y_onehot = np.zeros((n, 3), dtype=float)
    y_onehot[np.arange(n), y_idx] = 1.0
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    pred_idx = np.argmax(probs, axis=1)
    acc = float(np.mean(pred_idx == y_idx))

    return {
        "N": int(n),
        "logloss": ll,
        "brier": brier,
        "accuracy": acc,
    }


def objective_value(m: Dict[str, float], key: str) -> float:
    if key == "logloss":
        return float(m["logloss"])
    if key == "brier":
        return float(m["brier"])
    # hybrid: logloss + weak brier penalty
    return float(m["logloss"] + 0.25 * m["brier"])


def make_param_grid(args: argparse.Namespace) -> List[ParamSet]:
    draw_grid = parse_float_grid(args.draw_boost_grid)
    temp_grid = parse_float_grid(args.temperature_grid)
    shift_grid = parse_float_grid(args.home_shift_grid)
    out: List[ParamSet] = []
    for db in draw_grid:
        for t in temp_grid:
            for hs in shift_grid:
                out.append(ParamSet(draw_boost=float(db), temperature=float(t), home_shift=float(hs)))
    return out


def y_from_df(df: pd.DataFrame) -> np.ndarray:
    return df["actual_result"].map(LABEL_TO_IDX).astype(int).to_numpy()


def probs_from_df(df: pd.DataFrame) -> np.ndarray:
    return df[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy(dtype=float)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = [x.strip() for x in str(args.inputs).split(",") if x.strip()]
    df = load_df(paths)

    rounds = sorted(df["_round_no"].unique().tolist())
    if not rounds:
        raise RuntimeError("No valid round rows found.")

    grid = make_param_grid(args)
    base_param = ParamSet(draw_boost=1.0, temperature=1.0, home_shift=0.0)

    fold_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    param_rows: List[Dict[str, object]] = []

    for test_round in rounds:
        train = df[df["_round_no"] < test_round].copy()
        test = df[df["_round_no"] == test_round].copy()
        if len(train) == 0 or len(test) == 0:
            continue
        train_rounds = sorted(train["_round_no"].unique().tolist())
        if len(train_rounds) < int(args.min_train_round):
            continue

        y_tr = y_from_df(train)
        y_te = y_from_df(test)
        p_tr_base = probs_from_df(train)
        p_te_base = probs_from_df(test)

        best_param: Optional[ParamSet] = None
        best_obj = float("inf")
        best_train_metrics: Optional[Dict[str, float]] = None

        for ps in grid:
            p_tr = apply_transform(p_tr_base, ps)
            m_tr = metrics(y_tr, p_tr)
            obj = objective_value(m_tr, args.objective)
            param_rows.append(
                {
                    "test_round": int(test_round),
                    "train_round_min": int(min(train_rounds)),
                    "train_round_max": int(max(train_rounds)),
                    "draw_boost": ps.draw_boost,
                    "temperature": ps.temperature,
                    "home_shift": ps.home_shift,
                    "train_N": m_tr["N"],
                    "train_logloss": m_tr["logloss"],
                    "train_brier": m_tr["brier"],
                    "train_accuracy": m_tr["accuracy"],
                    "objective": obj,
                }
            )
            if obj < best_obj:
                best_obj = obj
                best_param = ps
                best_train_metrics = m_tr

        if best_param is None or best_train_metrics is None:
            continue

        p_te_best = apply_transform(p_te_base, best_param)
        p_te_base = apply_transform(p_te_base, base_param)

        m_te_best = metrics(y_te, p_te_best)
        m_te_base = metrics(y_te, p_te_base)

        fold_rows.append(
            {
                "test_round": int(test_round),
                "train_round_min": int(min(train_rounds)),
                "train_round_max": int(max(train_rounds)),
                "test_N": int(m_te_best["N"]),
                "selected_draw_boost": best_param.draw_boost,
                "selected_temperature": best_param.temperature,
                "selected_home_shift": best_param.home_shift,
                "test_logloss_selected": m_te_best["logloss"],
                "test_brier_selected": m_te_best["brier"],
                "test_accuracy_selected": m_te_best["accuracy"],
                "test_logloss_baseline": m_te_base["logloss"],
                "test_brier_baseline": m_te_base["brier"],
                "test_accuracy_baseline": m_te_base["accuracy"],
                "delta_logloss": m_te_best["logloss"] - m_te_base["logloss"],
                "delta_brier": m_te_best["brier"] - m_te_base["brier"],
                "delta_accuracy": m_te_best["accuracy"] - m_te_base["accuracy"],
            }
        )

        best_rows.append(
            {
                "test_round": int(test_round),
                "train_round_min": int(min(train_rounds)),
                "train_round_max": int(max(train_rounds)),
                "selected_draw_boost": best_param.draw_boost,
                "selected_temperature": best_param.temperature,
                "selected_home_shift": best_param.home_shift,
                "train_objective": best_obj,
                "train_logloss": best_train_metrics["logloss"],
                "train_brier": best_train_metrics["brier"],
                "train_accuracy": best_train_metrics["accuracy"],
            }
        )

    if not fold_rows:
        raise RuntimeError("No folds evaluated. Adjust --min-train-round or input data.")

    fold_df = pd.DataFrame(fold_rows).sort_values("test_round")
    best_df = pd.DataFrame(best_rows).sort_values("test_round")
    param_df = pd.DataFrame(param_rows).sort_values(["test_round", "objective"]) 

    summary = {
        "fold_count": int(len(fold_df)),
        "test_total_N": int(fold_df["test_N"].sum()),
        "mean_test_logloss_selected": float(fold_df["test_logloss_selected"].mean()),
        "mean_test_brier_selected": float(fold_df["test_brier_selected"].mean()),
        "mean_test_accuracy_selected": float(fold_df["test_accuracy_selected"].mean()),
        "mean_test_logloss_baseline": float(fold_df["test_logloss_baseline"].mean()),
        "mean_test_brier_baseline": float(fold_df["test_brier_baseline"].mean()),
        "mean_test_accuracy_baseline": float(fold_df["test_accuracy_baseline"].mean()),
        "mean_delta_logloss": float(fold_df["delta_logloss"].mean()),
        "mean_delta_brier": float(fold_df["delta_brier"].mean()),
        "mean_delta_accuracy": float(fold_df["delta_accuracy"].mean()),
    }

    # Recommend median of selected params for robust next-run default.
    rec = {
        "draw_boost": float(best_df["selected_draw_boost"].median()),
        "temperature": float(best_df["selected_temperature"].median()),
        "home_shift": float(best_df["selected_home_shift"].median()),
    }

    prefix = args.label
    fold_path = outdir / f"{prefix}_fold_metrics.csv"
    best_path = outdir / f"{prefix}_best_params_by_fold.csv"
    param_path = outdir / f"{prefix}_param_train_scores.csv"
    summary_path = outdir / f"{prefix}_summary.json"

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    param_df.to_csv(param_path, index=False, encoding="utf-8-sig")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": args.objective,
                "inputs": paths,
                "min_train_round": int(args.min_train_round),
                "grid": {
                    "draw_boost": parse_float_grid(args.draw_boost_grid),
                    "temperature": parse_float_grid(args.temperature_grid),
                    "home_shift": parse_float_grid(args.home_shift_grid),
                },
                "summary": summary,
                "recommended_params": rec,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[WALKFORWARD_SUMMARY] folds={summary['fold_count']} N={summary['test_total_N']} "
        f"ll_sel={summary['mean_test_logloss_selected']:.6f} ll_base={summary['mean_test_logloss_baseline']:.6f} "
        f"delta_ll={summary['mean_delta_logloss']:.6f} "
        f"acc_sel={summary['mean_test_accuracy_selected']:.4f} acc_base={summary['mean_test_accuracy_baseline']:.4f}"
    )
    print(
        f"[WALKFORWARD_RECOMMENDED] draw_boost={rec['draw_boost']:.3f} "
        f"temperature={rec['temperature']:.3f} home_shift={rec['home_shift']:.3f}"
    )
    print(f"[OK] {fold_path}")
    print(f"[OK] {best_path}")
    print(f"[OK] {param_path}")
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()
