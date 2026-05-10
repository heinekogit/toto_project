#!/usr/bin/env python3
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
CLASS_ORDER = ["H", "D", "A"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_ORDER)}
L2_LAMBDA = 1e-3
H_MAX = 0.60


def parse_args():
    p = argparse.ArgumentParser(description="Train 1D multinomial calibration model from elo_diff_for_prob.")
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--league", choices=["j1", "j2"], required=True)
    p.add_argument("--dataset", default="rounds", choices=["rounds", "backtest"])
    p.add_argument("--l2-lambda", type=float, default=L2_LAMBDA)
    p.add_argument("--h-max", type=float, default=H_MAX)
    p.add_argument("--class-weight", default="balanced", choices=["none", "balanced"])
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--out", default="")
    return p.parse_args()


def get_result(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    h = float(home_score)
    a = float(away_score)
    if h > a:
        return "H"
    if h < a:
        return "A"
    return "D"


def _norm_team(v):
    if pd.isna(v):
        return ""
    return str(v).strip().replace(" ", "").replace("　", "")


def _snapshot_backtest_files(league, season):
    snap_dir = DATA_DIR / "output_snapshots" / f"{league}_{season}"
    if not snap_dir.exists():
        return []
    return sorted(snap_dir.glob(f"*_backtest*_backtest_{league}_{season}.csv"))


def _load_canonical_snapshot_backtest(league, season):
    files = _snapshot_backtest_files(league, season)
    if not files:
        return None, ""
    parts = []
    for p in files:
        try:
            sdf = pd.read_csv(p)
        except Exception as e:
            print(f"[CALIB_SNAPSHOT][WARN] skip unreadable snapshot: {p} ({e})")
            continue
        if sdf.empty:
            continue
        sdf = sdf.copy()
        sdf["__src_ts"] = p.stem[:15]
        parts.append(sdf)
    if not parts:
        return None, ""
    all_df = pd.concat(parts, ignore_index=True, sort=False)
    all_df["__filled_n"] = all_df.notna().sum(axis=1)
    if "match_id" in all_df.columns and all_df["match_id"].notna().any():
        key_cols = ["match_id"]
    else:
        key_cols = [c for c in ["league", "datetime", "home_team", "away_team"] if c in all_df.columns]
    if key_cols:
        all_df = all_df.sort_values(key_cols + ["__filled_n", "__src_ts"], ascending=[True] * len(key_cols) + [False, False])
        all_df = all_df.drop_duplicates(key_cols, keep="first")
    all_df = all_df.drop(columns=["__filled_n", "__src_ts"], errors="ignore")
    src_desc = f"snapshot_canonical(files={len(files)} rows={len(all_df)})"
    return all_df, src_desc


def _load_backtest_reference_df(league, season):
    use_snapshot = str(os.environ.get("TRAIN_USE_OUTPUT_SNAPSHOT", "1")).strip() == "1"
    if use_snapshot:
        snap_df, snap_desc = _load_canonical_snapshot_backtest(league, season)
        if snap_df is not None and not snap_df.empty:
            print(f"[CALIB_SCHEMA] backtest_reference={snap_desc}")
            return snap_df, snap_desc
        print("[CALIB_SCHEMA][WARN] snapshot reference unavailable; fallback to base backtest csv")
    fallback_paths = [ROOT / f"backtest_{league}_{season}.csv", DATA_DIR / f"backtest_{league}_{season}.csv"]
    for p in fallback_paths:
        if p.exists():
            return pd.read_csv(p), str(p)
    raise RuntimeError("backtest fallback source not found")


def enrich_elo_diff_if_missing(df, league, season):
    if "elo_diff_for_prob" in df.columns and pd.to_numeric(df["elo_diff_for_prob"], errors="coerce").notna().any():
        return df
    src, src_path = _load_backtest_reference_df(league, season)
    if "elo_diff_for_prob" not in src.columns:
        raise RuntimeError(f"elo_diff_for_prob not found in fallback source: {src_path}")

    work = df.copy()
    if "match_id" in work.columns and "match_id" in src.columns:
        m = work.merge(src[["match_id", "elo_diff_for_prob"]].drop_duplicates("match_id"), on="match_id", how="left")
        if pd.to_numeric(m["elo_diff_for_prob"], errors="coerce").notna().sum() > 0:
            return m

    for t in (work, src):
        t["_dt"] = pd.to_datetime(t.get("datetime"), errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        t["_home"] = t.get("home_team", pd.Series(index=t.index, dtype="object")).map(_norm_team)
        t["_away"] = t.get("away_team", pd.Series(index=t.index, dtype="object")).map(_norm_team)
    m = work.merge(
        src[["_dt", "_home", "_away", "elo_diff_for_prob"]].drop_duplicates(["_dt", "_home", "_away"]),
        on=["_dt", "_home", "_away"],
        how="left",
    )
    return m.drop(columns=["_dt", "_home", "_away"], errors="ignore")


def enrich_diff_raw_no_hfa(df, league, season):
    if "diff_raw_no_hfa" in df.columns and pd.to_numeric(df["diff_raw_no_hfa"], errors="coerce").notna().any():
        return df
    work = df.copy()
    # まず同一行にhome/away eloがあれば直接作成
    if {"home_elo_at_prediction", "away_elo_at_prediction"}.issubset(work.columns):
        h = pd.to_numeric(work["home_elo_at_prediction"], errors="coerce")
        a = pd.to_numeric(work["away_elo_at_prediction"], errors="coerce")
        d = h - a
        if d.notna().sum() > 0:
            work["diff_raw_no_hfa"] = d
            return work
    # fallback: root/data の backtest_{league}_{season}.csv から補完
    src, _ = _load_backtest_reference_df(league, season)
    if {"home_elo_at_prediction", "away_elo_at_prediction"}.issubset(src.columns):
        src = src.copy()
        src["diff_raw_no_hfa"] = pd.to_numeric(src["home_elo_at_prediction"], errors="coerce") - pd.to_numeric(
            src["away_elo_at_prediction"], errors="coerce"
        )
    elif "elo_diff_before_hfa" in src.columns:
        src = src.copy()
        src["diff_raw_no_hfa"] = pd.to_numeric(src["elo_diff_before_hfa"], errors="coerce")
    else:
        raise RuntimeError("fallback source has no columns to compute diff_raw_no_hfa")

    if "match_id" in work.columns and "match_id" in src.columns:
        m = work.merge(src[["match_id", "diff_raw_no_hfa"]].drop_duplicates("match_id"), on="match_id", how="left")
        if pd.to_numeric(m["diff_raw_no_hfa"], errors="coerce").notna().sum() > 0:
            return m

    for t in (work, src):
        t["_dt"] = pd.to_datetime(t.get("datetime"), errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        t["_home"] = t.get("home_team", pd.Series(index=t.index, dtype="object")).map(_norm_team)
        t["_away"] = t.get("away_team", pd.Series(index=t.index, dtype="object")).map(_norm_team)
    m = work.merge(
        src[["_dt", "_home", "_away", "diff_raw_no_hfa"]].drop_duplicates(["_dt", "_home", "_away"]),
        on=["_dt", "_home", "_away"],
        how="left",
    )
    return m.drop(columns=["_dt", "_home", "_away"], errors="ignore")


def load_dataset(league, season, dataset):
    base_name = f"backtest_{league}_{season}_rounds.csv" if dataset == "rounds" else f"backtest_{league}_{season}.csv"
    candidates = [DATA_DIR / base_name, ROOT / base_name]
    if dataset == "rounds":
        path = None
    else:
        path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError(candidates[0])
    df = pd.read_csv(path)
    if "actual_result" not in df.columns:
        if {"home_score", "away_score"}.issubset(df.columns):
            df["actual_result"] = [
                get_result(h, a) for h, a in zip(df["home_score"].tolist(), df["away_score"].tolist())
            ]
        else:
            raise RuntimeError("actual_result/home_score/away_score not found")
    df = df[df["actual_result"].isin(CLASS_ORDER)].copy()
    df = enrich_diff_raw_no_hfa(df, league, season)
    df["diff_raw_no_hfa"] = pd.to_numeric(df["diff_raw_no_hfa"], errors="coerce")
    df = df.dropna(subset=["diff_raw_no_hfa", "actual_result"]).copy()
    if df.empty:
        raise RuntimeError("no rows after preprocessing")
    return df


def softmax(z):
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _balanced_class_weights(y_idx):
    n = len(y_idx)
    k = len(CLASS_ORDER)
    counts = np.bincount(y_idx, minlength=k).astype(float)
    counts = np.where(counts <= 0, 1.0, counts)
    return np.array([n / (k * counts[i]) for i in range(k)], dtype=float)


def constrained_scores(diff, k, b0, h, bD):
    x = np.asarray(diff, dtype=float).reshape(-1)
    s_h = (k * x) + b0 + h
    s_d = np.full_like(x, float(bD), dtype=float)
    s_a = (-k * x) + b0 - h
    return np.stack([s_h, s_d, s_a], axis=1)


def fit_constrained_model(diff, y_idx, class_weight_mode="balanced", maxiter=2000, l2_lambda=L2_LAMBDA, h_max=H_MAX):
    x = np.asarray(diff, dtype=float).reshape(-1)
    n = len(x)
    if class_weight_mode == "balanced":
        cw = _balanced_class_weights(y_idx)
    else:
        cw = np.ones(len(CLASS_ORDER), dtype=float)
    sample_w = cw[y_idx]

    lam = float(l2_lambda)
    h_lim = abs(float(h_max))

    def unpack(theta):
        # theta = [theta_k, b0, h, bD], k = exp(theta_k) to enforce k>=0
        theta_k, b0, h, bD = [float(v) for v in theta]
        k = float(np.exp(theta_k))
        return k, b0, h, bD

    def objective(theta):
        k, b0, h, bD = unpack(theta)
        logits = constrained_scores(x, k, b0, h, bD)
        probs = softmax(logits)
        p = np.clip(probs[np.arange(n), y_idx], 1e-15, 1.0)
        nll = float(np.sum(sample_w * (-np.log(p))) / max(1.0, float(np.sum(sample_w))))
        reg = lam * (k * k + b0 * b0 + h * h + bD * bD)
        return nll + reg

    theta0 = np.array([np.log(0.01), 0.0, 0.0, 0.0], dtype=float)
    # theta_k,b0,bDは自由。hのみ境界を付与
    bounds = [(None, None), (None, None), (-h_lim, h_lim), (None, None)]
    res = minimize(objective, x0=theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
    k, b0, h, bD = unpack(res.x)
    return {"k": k, "b0": b0, "h": h, "bD": bD, "opt_result": res}


def predict_constrained_probs(diff, params):
    logits = constrained_scores(diff, params["k"], params["b0"], params["h"], params["bD"])
    return softmax(logits)


def logloss(y_idx, probs, eps=1e-15):
    p = np.clip(probs[np.arange(len(y_idx)), y_idx], eps, 1.0)
    return float(-np.mean(np.log(p)))


def elo_sign_check_from_probs(df, probs, classes):
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    i_h = cls_to_idx["H"]
    i_a = cls_to_idx["A"]
    d = pd.to_numeric(df["diff_raw_no_hfa"], errors="coerce").to_numpy()
    ph = probs[:, i_h]
    pa = probs[:, i_a]
    neg = d < 0
    pos = d > 0
    vneg = int(np.sum((ph > pa) & neg))
    vpos = int(np.sum((ph < pa) & pos))
    return {
        "neg_cases": int(np.sum(neg)),
        "neg_violated": vneg,
        "neg_rate": float(vneg / max(1, np.sum(neg))),
        "pos_cases": int(np.sum(pos)),
        "pos_violated": vpos,
        "pos_rate": float(vpos / max(1, np.sum(pos))),
    }


def main():
    args = parse_args()
    df = load_dataset(args.league, args.season, args.dataset)
    x_raw = df["diff_raw_no_hfa"].to_numpy(dtype=float)
    y_lbl = df["actual_result"].astype(str).str.upper().to_numpy()
    y = np.array([CLASS_TO_INDEX[v] for v in y_lbl], dtype=int)

    fitted = fit_constrained_model(
        x_raw,
        y,
        class_weight_mode=args.class_weight,
        maxiter=int(args.max_iter),
        l2_lambda=float(args.l2_lambda),
        h_max=float(args.h_max),
    )
    params = {"k": float(fitted["k"]), "b0": float(fitted["b0"]), "h": float(fitted["h"]), "bD": float(fitted["bD"])}
    train_backend = "constrained_softmax_1d_scipy"
    # 互換用に線形softmax表現へ展開
    coef = np.array([[params["k"]], [0.0], [-params["k"]]], dtype=float)
    intercept = np.array([params["b0"] + params["h"], params["bD"], params["b0"] - params["h"]], dtype=float)
    mu = np.array([0.0], dtype=float)
    sigma = np.array([1.0], dtype=float)

    bundle = {
        "type": "constrained_softmax_1d",
        "classes": CLASS_ORDER,
        "feature_names": ["diff_raw_no_hfa"],
        "coef": coef,
        "intercept": intercept,
        "feature_mean": mu.astype(float),
        "feature_std": sigma.astype(float),
        "constrained_params": params,
        "l2_lambda": float(args.l2_lambda),
        "h_max": float(args.h_max),
        "train_rows": int(len(df)),
        "model_source": train_backend,
        "league": args.league,
        "season": int(args.season),
        "class_weight": args.class_weight,
        "C": None,
        "diff_definition": "diff_raw_no_hfa",
    }

    probs = predict_constrained_probs(x_raw, params)
    pred_idx = probs.argmax(axis=1)
    acc = float(np.mean(pred_idx == y))
    ll = logloss(y, probs)
    sign = elo_sign_check_from_probs(df, probs, CLASS_ORDER)
    pred_counts = {c: int(np.sum(pred_idx == CLASS_TO_INDEX[c])) for c in CLASS_ORDER}
    actual_counts = {c: int(np.sum(y == CLASS_TO_INDEX[c])) for c in CLASS_ORDER}

    out = args.out.strip() if args.out else str(MODEL_DIR / f"hda_multinom_calibrated_{args.league}.joblib")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f)

    print(
        f"[CALIB_TRAIN] league={args.league} season={args.season} rows={len(df)} "
        f"dataset={args.dataset} class_weight={args.class_weight} backend={train_backend} "
        f"l2_lambda={args.l2_lambda} h_max={args.h_max} diff_definition=diff_raw_no_hfa"
    )
    print(
        f"[CALIB_PARAMS] k={params['k']:.6f} b0={params['b0']:.6f} h={params['h']:.6f} bD={params['bD']:.6f} "
        f"opt_success={bool(fitted['opt_result'].success)} iters={int(getattr(fitted['opt_result'], 'nit', -1))}"
    )
    print(f"[CALIB_EVAL] accuracy={acc:.4f} logloss={ll:.6f}")
    print(
        f"[CALIB_HDA] pred=H:{pred_counts['H']} D:{pred_counts['D']} A:{pred_counts['A']} "
        f"actual=H:{actual_counts['H']} D:{actual_counts['D']} A:{actual_counts['A']}"
    )
    print(
        f"[CALIB_SIGN_CHECK] neg_cases={sign['neg_cases']} violated={sign['neg_violated']} rate={sign['neg_rate']*100:.2f}% "
        f"pos_cases={sign['pos_cases']} violated={sign['pos_violated']} rate={sign['pos_rate']*100:.2f}%"
    )
    probe = np.array([-200.0, -25.0, 0.0, 25.0, 200.0], dtype=float)
    p_probe = predict_constrained_probs(probe, params)
    for d, pr in zip(probe.tolist(), p_probe.tolist()):
        print(
            f"[CALIB_PROBE] diff={d:+.1f} pH={float(pr[0]):.6f} pD={float(pr[1]):.6f} pA={float(pr[2]):.6f}"
        )
    print(f"[CALIB_OUT] model={out}")


if __name__ == "__main__":
    main()
