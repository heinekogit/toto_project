#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"

CLASS_ORDER = ["H", "D", "A"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_ORDER)}
INDEX_TO_CLASS = {i: c for c, i in CLASS_TO_INDEX.items()}
LEAGUE_FEATURE_CANDIDATES = {
    "j1": ["elo_diff_for_prob", "abs_elo_diff_for_prob", "d_scaled", "abs_d_scaled"],
    "j2": ["elo_diff_for_prob", "abs_elo_diff_for_prob"],
}
FEATURE_PROFILES = {
    "j1_full": ["elo_diff_for_prob", "abs_elo_diff_for_prob", "d_scaled", "abs_d_scaled"],
    "j1_core": ["elo_diff_for_prob", "abs_elo_diff_for_prob"],
    "j2_core": ["elo_diff_for_prob", "abs_elo_diff_for_prob"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Train 3-class softmax model for H/D/A probabilities.")
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--league", choices=["j1", "j2", "both"], default="both")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--l2", type=float, default=0.3)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--gtol", type=float, default=1e-10)
    p.add_argument("--ftol", type=float, default=1e-12)
    p.add_argument("--min-warn-iters", type=int, default=50)
    p.add_argument("--grid-l2", default="0.01,0.03,0.1,0.3,1.0,3.0")
    p.add_argument("--class-weight", choices=["auto", "none", "balanced"], default="auto")
    p.add_argument("--class-weight-alpha", type=float, default=None)
    p.add_argument("--baseline-eps", type=float, default=1e-6)
    p.add_argument("--feature-profile", choices=["auto", "j1_full", "j1_core", "j2_core"], default="auto")
    p.add_argument("--eval-backtest", action="store_true")
    p.add_argument("--out", default="")
    return p.parse_args()


def parse_round_no(v):
    if pd.isna(v):
        return np.nan
    s = str(v).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    m = re.search(r"第\s*(\d+)\s*節", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


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


def _read_first_existing(paths):
    for p in paths:
        if p.exists():
            return pd.read_csv(p), p
    raise FileNotFoundError(f"none of paths exist: {[str(p) for p in paths]}")


def _norm_team(v):
    if pd.isna(v):
        return ""
    return str(v).strip().replace(" ", "").replace("　", "")


def enrich_elo_diff_if_missing(df, league, season):
    if "elo_diff_for_prob" in df.columns and pd.to_numeric(df["elo_diff_for_prob"], errors="coerce").notna().any():
        return df

    fallback_paths = [
        ROOT / f"backtest_{league}_{season}.csv",
        DATA_DIR / f"backtest_{league}_{season}.csv",
    ]
    src, src_path = _read_first_existing(fallback_paths)
    if "elo_diff_for_prob" not in src.columns:
        raise RuntimeError(f"elo_diff_for_prob not found in fallback source: {src_path}")

    work = df.copy()
    got = False
    if "match_id" in work.columns and "match_id" in src.columns:
        merged = work.merge(src[["match_id", "elo_diff_for_prob"]].drop_duplicates("match_id"), on="match_id", how="left")
        if pd.to_numeric(merged["elo_diff_for_prob"], errors="coerce").notna().sum() > 0:
            work = merged
            got = True

    if not got:
        for t in (work, src):
            t["_dt"] = pd.to_datetime(t.get("datetime"), errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            t["_home"] = t.get("home_team", pd.Series(index=t.index, dtype="object")).map(_norm_team)
            t["_away"] = t.get("away_team", pd.Series(index=t.index, dtype="object")).map(_norm_team)
        merged = work.merge(
            src[["_dt", "_home", "_away", "elo_diff_for_prob"]].drop_duplicates(["_dt", "_home", "_away"]),
            on=["_dt", "_home", "_away"],
            how="left",
        )
        work = merged.drop(columns=["_dt", "_home", "_away"], errors="ignore")

    hit = int(pd.to_numeric(work.get("elo_diff_for_prob"), errors="coerce").notna().sum())
    print(f"[TRAIN_SCHEMA] elo_diff fallback source={src_path} matched_rows={hit}/{len(work)}")
    if hit == 0:
        raise RuntimeError("failed to enrich elo_diff_for_prob")
    return work


def load_league_dataset(league, season):
    rounds_paths = [
        DATA_DIR / f"backtest_{league}_{season}_rounds.csv",
        ROOT / f"backtest_{league}_{season}_rounds.csv",
    ]
    df, src_path = _read_first_existing(rounds_paths)
    df = enrich_elo_diff_if_missing(df, league, season)

    if "actual_result" not in df.columns:
        df["actual_result"] = df.apply(lambda r: get_result(r.get("home_score"), r.get("away_score")), axis=1)
    else:
        missing = df["actual_result"].isna() | (df["actual_result"].astype(str).str.strip() == "")
        if missing.any():
            df.loc[missing, "actual_result"] = df.loc[missing].apply(
                lambda r: get_result(r.get("home_score"), r.get("away_score")), axis=1
            )

    df["actual_result"] = df["actual_result"].astype(str).str.upper()
    df = df[df["actual_result"].isin(CLASS_ORDER)].copy()
    df["elo_diff_for_prob"] = pd.to_numeric(df["elo_diff_for_prob"], errors="coerce")
    df = df.dropna(subset=["elo_diff_for_prob"]).copy()

    df["abs_elo_diff_for_prob"] = df["elo_diff_for_prob"].abs()
    if "d_scaled" in df.columns:
        df["d_scaled"] = pd.to_numeric(df["d_scaled"], errors="coerce")
    else:
        # 旧draw補正に依存しない派生特徴として作成
        df["d_scaled"] = df["abs_elo_diff_for_prob"]
    if "abs_d_scaled" in df.columns:
        df["abs_d_scaled"] = pd.to_numeric(df["abs_d_scaled"], errors="coerce")
    else:
        df["abs_d_scaled"] = pd.to_numeric(df["d_scaled"], errors="coerce").abs()

    df["round_no"] = df.get("節", pd.Series(index=df.index)).map(parse_round_no)
    df["dt_sort"] = pd.to_datetime(df.get("datetime"), errors="coerce")
    df["league"] = league
    df = df.sort_values(["round_no", "dt_sort", "match_id" if "match_id" in df.columns else "home_team"], na_position="last")
    print(f"[TRAIN_SCHEMA] league={league} source={src_path} rows={len(df)}")
    return df


def one_hot(y_idx, k):
    out = np.zeros((len(y_idx), k), dtype=float)
    out[np.arange(len(y_idx)), y_idx] = 1.0
    return out


def labels_to_indices(labels):
    s = pd.Series(labels).astype(str).str.upper()
    bad = sorted(set([x for x in s.tolist() if x not in CLASS_TO_INDEX]))
    if bad:
        raise RuntimeError(f"unknown labels in y: {bad}")
    return s.map(CLASS_TO_INDEX).to_numpy(dtype=int)


def _idx_dist(y_idx):
    y = np.asarray(y_idx, dtype=int)
    n = int(len(y))
    out = {}
    for i, c in enumerate(CLASS_ORDER):
        cnt = int((y == i).sum())
        pct = (100.0 * cnt / n) if n > 0 else 0.0
        out[c] = {"count": cnt, "pct": pct}
    return out, n


def log_y_dist(y_idx, split_label, league, feature_profile):
    dist, n = _idx_dist(y_idx)
    print(
        f"[Y_DIST] league={league} feature_profile={feature_profile} split={split_label} rows={n} "
        f"H={dist['H']['pct']:.1f}% ({dist['H']['count']}) "
        f"D={dist['D']['pct']:.1f}% ({dist['D']['count']}) "
        f"A={dist['A']['pct']:.1f}% ({dist['A']['count']})"
    )


def log_pred_dist(pred_idx, split_label, league, feature_profile, collapse_threshold=95.0):
    dist, n = _idx_dist(pred_idx)
    print(
        f"[PRED_DIST] league={league} feature_profile={feature_profile} split={split_label} rows={n} "
        f"pred_H={dist['H']['pct']:.1f}% ({dist['H']['count']}) "
        f"pred_D={dist['D']['pct']:.1f}% ({dist['D']['count']}) "
        f"pred_A={dist['A']['pct']:.1f}% ({dist['A']['count']})"
    )
    max_pct = max(dist[c]["pct"] for c in CLASS_ORDER) if n > 0 else 0.0
    if max_pct >= float(collapse_threshold):
        print(
            f"[WARN_PRED_COLLAPSE] league={league} feature_profile={feature_profile} split={split_label} "
            f"max_class_share_pct={max_pct:.1f} threshold={float(collapse_threshold):.1f}"
        )


def log_logits_stats(logits, split_label, league, feature_profile, l2=None):
    if logits is None or len(logits) == 0:
        print(f"[LOGITS_STATS] league={league} feature_profile={feature_profile} split={split_label} unavailable")
        return
    sfx = f" l2={float(l2):.6f}" if l2 is not None else ""
    arr = np.asarray(logits, dtype=float)
    q50 = np.median(arr, axis=0)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    max_abs = float(np.max(np.abs(arr)))
    print(
        f"[LOGITS_STATS] league={league} feature_profile={feature_profile} split={split_label}{sfx} "
        f"H(min/p50/max)={mins[0]:.3f}/{q50[0]:.3f}/{maxs[0]:.3f} "
        f"D(min/p50/max)={mins[1]:.3f}/{q50[1]:.3f}/{maxs[1]:.3f} "
        f"A(min/p50/max)={mins[2]:.3f}/{q50[2]:.3f}/{maxs[2]:.3f} "
        f"max_abs={max_abs:.3f}"
    )
    if max_abs > 20.0:
        print(
            f"[WARN_LOGITS_SATURATION] league={league} feature_profile={feature_profile} split={split_label}{sfx} "
            f"max_abs_logit={max_abs:.3f} (>20)"
        )


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return e / s


def fit_softmax(
    X,
    y_idx,
    l2=1.0,
    maxiter=1000,
    gtol=1e-10,
    ftol=1e-12,
    min_total_iters=50,
    class_weight_mode="none",
    class_weight_alpha=1.0,
):
    n, p = X.shape
    k = len(CLASS_ORDER)

    if str(class_weight_mode) == "balanced":
        class_counts = np.bincount(y_idx, minlength=k).astype(float)
        class_counts = np.where(class_counts <= 0, 1.0, class_counts)
        class_w_bal = class_counts.sum() / (k * class_counts)
        alpha = float(class_weight_alpha)
        alpha = max(0.0, min(1.0, alpha))
        class_w = 1.0 + alpha * (class_w_bal - 1.0)
        sample_w = class_w[y_idx]
    else:
        class_w = np.ones(k, dtype=float)
        sample_w = np.ones(len(y_idx), dtype=float)

    Y = one_hot(y_idx, k)
    theta0 = np.zeros((k, p + 1), dtype=float)

    def unpack(theta):
        t = theta.reshape(k, p + 1)
        return t[:, :p], t[:, p]

    def objective(theta):
        W, b = unpack(theta)
        logits = X.dot(W.T) + b
        P = softmax(logits)
        eps = 1e-12
        ce = -np.sum(sample_w[:, None] * Y * np.log(np.clip(P, eps, 1.0))) / np.sum(sample_w)
        reg = 0.5 * float(l2) * np.sum(W * W)
        return ce + reg

    def gradient(theta):
        W, b = unpack(theta)
        logits = X.dot(W.T) + b
        P = softmax(logits)
        R = (P - Y) * sample_w[:, None]
        den = np.sum(sample_w)
        gW = R.T.dot(X) / den + float(l2) * W
        gb = np.sum(R, axis=0) / den
        return np.concatenate([gW, gb[:, None]], axis=1).ravel()

    iter_counter = {"i": 0}

    def callback(theta):
        iter_counter["i"] += 1
        if iter_counter["i"] % 10 == 0:
            print(f"[TRAIN_LOSS] iter={iter_counter['i']} loss={float(objective(theta)):.6f}")

    res = minimize(
        objective,
        theta0.ravel(),
        jac=gradient,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": float(ftol)},
    )
    if not res.success:
        print(f"[TRAIN_WARN] optimizer status={res.status} message={res.message}")
    W, b = unpack(res.x)
    total_iters = int(getattr(res, "nit", -1))

    if total_iters < int(min_total_iters):
        # 反復不足時は全バッチGDで追加最適化し、薄い学習を避ける
        den = np.sum(sample_w)
        epochs = max(0, int(min_total_iters) - max(total_iters, 0))
        lr = 0.05
        for e in range(epochs):
            logits = X.dot(W.T) + b
            P = softmax(logits)
            R = (P - Y) * sample_w[:, None]
            gW = R.T.dot(X) / den + float(l2) * W
            gb = np.sum(R, axis=0) / den
            W = W - lr * gW
            b = b - lr * gb
            cur_iter = max(total_iters, 0) + e + 1
            if cur_iter % 10 == 0:
                logits_cur = X.dot(W.T) + b
                P_cur = softmax(logits_cur)
                eps = 1e-12
                ce_cur = -np.sum(sample_w[:, None] * Y * np.log(np.clip(P_cur, eps, 1.0))) / den
                reg_cur = 0.5 * float(l2) * np.sum(W * W)
                print(f"[TRAIN_LOSS] iter={cur_iter} loss={float(ce_cur + reg_cur):.6f}")
        total_iters = max(total_iters, 0) + epochs
    logits_final = X.dot(W.T) + b
    P_final = softmax(logits_final)
    eps = 1e-12
    ce_final = -np.sum(sample_w[:, None] * Y * np.log(np.clip(P_final, eps, 1.0))) / np.sum(sample_w)
    reg_final = 0.5 * float(l2) * np.sum(W * W)
    final_loss = float(ce_final + reg_final)
    R_final = (P_final - Y) * sample_w[:, None]
    den_final = np.sum(sample_w)
    gW_final = R_final.T.dot(X) / den_final + float(l2) * W
    gb_final = np.sum(R_final, axis=0) / den_final
    grad_norm = float(np.linalg.norm(np.concatenate([gW_final, gb_final[:, None]], axis=1).ravel(), ord=2))
    opt_info = {
        "status": int(res.status),
        "success": bool(res.success),
        "nit": int(total_iters),
        "final_loss": final_loss,
        "grad_norm": grad_norm,
        "message": str(getattr(res, "message", "")),
        "gtol": float(gtol),
        "ftol": float(ftol),
        "class_weight_mode": str(class_weight_mode),
        "class_weight_alpha": float(class_weight_alpha),
        "class_weights": [float(x) for x in class_w.tolist()],
    }
    return W, b, opt_info


def predict_proba(X, W, b):
    return softmax(X.dot(W.T) + b)


def logloss(y_idx, probs):
    eps = 1e-12
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], eps, 1.0))))


def confusion(y_true_idx, y_pred_idx):
    k = len(CLASS_ORDER)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true_idx, y_pred_idx):
        cm[int(t), int(p)] += 1
    return cm


def summarize_probs(df_eval):
    ph = pd.to_numeric(df_eval["prob_home"], errors="coerce")
    pdw = pd.to_numeric(df_eval["prob_draw"], errors="coerce")
    pa = pd.to_numeric(df_eval["prob_away"], errors="coerce")
    valid = ph.notna() & pdw.notna() & pa.notna()
    argmax_d = int(((pdw >= ph) & (pdw >= pa) & valid).sum())
    ge_023 = int(((pdw >= 0.23) & valid).sum())
    return int(valid.sum()), argmax_d, ge_023


def log_prob_dist_from_array(name, arr):
    s = pd.Series(np.asarray(arr, dtype=float))
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        print(f"[{name}] unavailable")
        return
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"[{name}] rows={len(s)} min={float(s.min()):.3f} p05={float(q.loc[0.05]):.3f} "
        f"p25={float(q.loc[0.25]):.3f} p50={float(q.loc[0.5]):.3f} "
        f"p75={float(q.loc[0.75]):.3f} p95={float(q.loc[0.95]):.3f} max={float(s.max()):.3f}"
    )


def summarize_draw_stats(prob_draw, y_idx):
    s = pd.Series(np.asarray(prob_draw, dtype=float))
    q = s.quantile([0.5, 0.95])
    avg = float(s.mean())
    actual_d = float(np.mean(np.asarray(y_idx, dtype=int) == 1))
    return {
        "draw_min": float(s.min()),
        "draw_p50": float(q.loc[0.5]),
        "draw_p95": float(q.loc[0.95]),
        "draw_max": float(s.max()),
        "avg_prob_draw": avg,
        "actual_d_rate": actual_d,
        "draw_diff_pp": float((avg - actual_d) * 100.0),
    }


def _ratio_from_labels(labels):
    s = pd.Series(labels).astype(str).str.upper()
    s = s[s.isin(CLASS_ORDER)]
    n = int(len(s))
    if n == 0:
        return {"H_cnt": 0, "D_cnt": 0, "A_cnt": 0, "H_pct": 0.0, "D_pct": 0.0, "A_pct": 0.0, "rows": 0}
    h = int((s == "H").sum())
    d = int((s == "D").sum())
    a = int((s == "A").sum())
    return {
        "H_cnt": h,
        "D_cnt": d,
        "A_cnt": a,
        "H_pct": 100.0 * h / n,
        "D_pct": 100.0 * d / n,
        "A_pct": 100.0 * a / n,
        "rows": n,
    }


def _prob_dist_dict(arr):
    s = pd.Series(np.asarray(arr, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": float(s.min()),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "max": float(s.max()),
        "rows": int(len(s)),
    }


def evaluate_on_backtest_df(df, W, b, mu, sigma, feature_names, league, feature_profile):
    X_raw = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    X = (X_raw - mu) / np.where(np.abs(sigma) < 1e-12, 1.0, sigma)
    logits = X.dot(W.T) + b
    probs = softmax(logits)
    pred_idx = np.argmax(probs, axis=1)
    pred_lbl = [CLASS_ORDER[int(i)] for i in pred_idx]
    actual_lbl = df["actual_result"].astype(str).str.upper().tolist()

    actual_ratio = _ratio_from_labels(actual_lbl)
    pred_ratio = _ratio_from_labels(pred_lbl)
    y_idx = labels_to_indices(actual_lbl)
    draw_stats = summarize_draw_stats(probs[:, 1], y_idx)
    draw_dist = _prob_dist_dict(probs[:, 1])
    max_dist = _prob_dist_dict(np.max(probs, axis=1))

    print(f"[MODEL_EVAL] league={league} feature_profile={feature_profile} rows={len(df)}")
    print(f"[CLASS_MAP] order={CLASS_ORDER} mapping={CLASS_TO_INDEX}")
    log_y_dist(y_idx, "eval_actual", league, feature_profile)
    log_pred_dist(pred_idx, "eval_pred", league, feature_profile)
    log_logits_stats(logits, "eval", league, feature_profile)
    print(
        f"[ACTUAL_HDA:EVAL] league={league} feature_profile={feature_profile} "
        f"H={actual_ratio['H_pct']:.1f}% ({actual_ratio['H_cnt']}) "
        f"D={actual_ratio['D_pct']:.1f}% ({actual_ratio['D_cnt']}) "
        f"A={actual_ratio['A_pct']:.1f}% ({actual_ratio['A_cnt']})"
    )
    print(
        f"[PRED_HDA:EVAL] league={league} feature_profile={feature_profile} "
        f"H={pred_ratio['H_pct']:.1f}% ({pred_ratio['H_cnt']}) "
        f"D={pred_ratio['D_pct']:.1f}% ({pred_ratio['D_cnt']}) "
        f"A={pred_ratio['A_pct']:.1f}% ({pred_ratio['A_cnt']})"
    )
    print(
        f"[DRAW_EVAL] league={league} feature_profile={feature_profile} "
        f"avg_prob_draw={draw_stats['avg_prob_draw']:.3f} actual_D_rate={draw_stats['actual_d_rate']:.3f} "
        f"draw_diff_pp={draw_stats['draw_diff_pp']:.2f}"
    )
    if draw_dist is not None:
        print(
            f"[PROB_DRAW_DIST:EVAL] league={league} feature_profile={feature_profile} "
            f"min={draw_dist['min']:.3f} p05={draw_dist['p05']:.3f} p25={draw_dist['p25']:.3f} "
            f"p50={draw_dist['p50']:.3f} p75={draw_dist['p75']:.3f} p95={draw_dist['p95']:.3f} max={draw_dist['max']:.3f}"
        )
    if max_dist is not None:
        print(
            f"[MAX_PROB_DIST:EVAL] league={league} feature_profile={feature_profile} "
            f"min={max_dist['min']:.3f} p05={max_dist['p05']:.3f} p25={max_dist['p25']:.3f} "
            f"p50={max_dist['p50']:.3f} p75={max_dist['p75']:.3f} p95={max_dist['p95']:.3f} max={max_dist['max']:.3f}"
        )
    return {
        "league": league,
        "feature_profile": feature_profile,
        "rows": int(len(df)),
        "avg_prob_draw": float(draw_stats["avg_prob_draw"]),
        "actual_D_rate": float(draw_stats["actual_d_rate"]),
        "draw_diff_pp": float(draw_stats["draw_diff_pp"]),
    }


def maybe_print_j1_profile_diff():
    p_full = MODEL_DIR / "hda_eval_summary_j1__j1_full.json"
    p_core = MODEL_DIR / "hda_eval_summary_j1__j1_core.json"
    if not (p_full.exists() and p_core.exists()):
        return
    try:
        full = json.loads(p_full.read_text(encoding="utf-8"))
        core = json.loads(p_core.read_text(encoding="utf-8"))
        d_full = float(full.get("draw_diff_pp", np.nan))
        d_core = float(core.get("draw_diff_pp", np.nan))
        delta = d_full - d_core
        print(
            f"[J1_PROFILE_DIFF] draw_diff_pp_full={d_full:.2f} "
            f"draw_diff_pp_core={d_core:.2f} delta_pp={delta:.2f}"
        )
    except Exception as e:
        print(f"[WARN] failed to build J1_PROFILE_DIFF: {e}")


def resolve_profile_for_league(args, league):
    if args.feature_profile == "auto":
        return "j1_full" if league == "j1" else "j2_core"
    if league == "j1" and args.feature_profile not in {"j1_full", "j1_core"}:
        raise RuntimeError(f"invalid feature_profile for j1: {args.feature_profile}")
    if league == "j2" and args.feature_profile != "j2_core":
        raise RuntimeError(f"invalid feature_profile for j2: {args.feature_profile}")
    return str(args.feature_profile)


def resolve_class_weight_for_league(args, league):
    # CLI明示があれば優先、未指定(auto/None)はリーグ別デフォルト
    if str(args.class_weight) == "auto":
        mode = "balanced"
    else:
        mode = str(args.class_weight)
    if args.class_weight_alpha is None:
        alpha = 0.7 if league == "j1" else 1.0
    else:
        alpha = float(args.class_weight_alpha)
    alpha = max(0.0, min(1.0, alpha))
    return mode, alpha


def select_feature_names(df, league, feature_profile):
    required = FEATURE_PROFILES[str(feature_profile)]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        preview = list(df.columns[:80])
        print(f"[ERROR] missing required features: {missing_cols} available_columns={preview}")
        raise RuntimeError("missing required features")
    no_values = [c for c in required if pd.to_numeric(df[c], errors="coerce").notna().sum() == 0]
    if no_values:
        preview = list(df.columns[:80])
        print(f"[ERROR] missing required features: {no_values} available_columns={preview}")
        raise RuntimeError("missing required features (all NaN)")
    return list(required)


def resolve_model_out_path(args, league, feature_profile):
    if args.out:
        out = Path(args.out).expanduser()
        if args.league == "both":
            suffix = f"_{league}__{feature_profile}{out.suffix or '.joblib'}"
            return out.with_name(out.stem + suffix)
        return out
    return MODEL_DIR / f"hda_multinom_train{args.season}_{league}__{feature_profile}.joblib"


def train_one_league(args, league):
    df = load_league_dataset(league, args.season).reset_index(drop=True)
    feature_profile = resolve_profile_for_league(args, league)
    class_weight_mode_eff, class_weight_alpha_eff = resolve_class_weight_for_league(args, league)
    print(
        f"[CLASS_WEIGHT_CONFIG] league={league} class_weight={class_weight_mode_eff} "
        f"alpha={class_weight_alpha_eff:.3f} (cli_weight={args.class_weight} cli_alpha={args.class_weight_alpha})"
    )
    feature_names = select_feature_names(df, league, feature_profile)
    print(f"[FEATURE_SET] league={league} feature_profile={feature_profile} features={feature_names}")

    X_raw = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = labels_to_indices(df["actual_result"])

    n = len(df)
    split = max(1, min(n - 1, int(n * float(args.train_ratio))))
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"[CLASS_MAP] order={CLASS_ORDER} mapping={CLASS_TO_INDEX}")
    log_y_dist(y_train, "train", league, feature_profile)
    log_y_dist(y_test, "valid", league, feature_profile)

    mu = X_train_raw.mean(axis=0)
    sigma = X_train_raw.std(axis=0)
    sigma = np.where(np.abs(sigma) < 1e-12, 1.0, sigma)
    X_train = (X_train_raw - mu) / sigma
    X_test = (X_test_raw - mu) / sigma

    l2_grid = []
    for x in str(args.grid_l2).split(","):
        x = x.strip()
        if x:
            l2_grid.append(float(x))
    if float(args.l2) not in l2_grid:
        l2_grid = [float(args.l2)] + l2_grid
    l2_grid = sorted(set(l2_grid))

    grid_rows = []
    candidate_payloads = []

    for l2 in l2_grid:
        print(f"[GRID] league={league} feature_profile={feature_profile} start l2={l2:.6f}")
        W, b, opt_info = fit_softmax(
            X_train,
            y_train,
            l2=float(l2),
            maxiter=int(args.maxiter),
            gtol=float(args.gtol),
            ftol=float(args.ftol),
            min_total_iters=int(args.min_warn_iters),
            class_weight_mode=class_weight_mode_eff,
            class_weight_alpha=float(class_weight_alpha_eff),
        )
        print(
            f"[TRAIN_OPT] league={league} feature_profile={feature_profile} l2={l2:.6f} status={opt_info['status']} success={int(opt_info['success'])} "
            f"iters={opt_info['nit']} final_loss={opt_info['final_loss']:.6f} grad_norm={opt_info['grad_norm']:.6e} "
            f"gtol={opt_info['gtol']:.1e} ftol={opt_info['ftol']:.1e} "
            f"class_weight={opt_info['class_weight_mode']} alpha={opt_info['class_weight_alpha']:.3f} "
            f"class_w={opt_info['class_weights']}"
        )

        p_test = predict_proba(X_test, W, b)
        pred_test = np.argmax(p_test, axis=1)
        acc = float(np.mean(pred_test == y_test))
        ll = logloss(y_test, p_test)
        logits_test = X_test.dot(W.T) + b
        log_logits_stats(logits_test, "valid", league, feature_profile, l2=l2)
        log_pred_dist(pred_test, "valid_pred", league, feature_profile)
        draw_stats = summarize_draw_stats(p_test[:, 1], y_test)
        print(
            f"[GRID_EVAL] league={league} feature_profile={feature_profile} l2={l2:.6f} accuracy={acc:.4f} logloss={ll:.6f} "
            f"avg_prob_draw={draw_stats['avg_prob_draw']:.3f} actual_D_rate={draw_stats['actual_d_rate']:.3f} "
            f"draw_diff_pp={draw_stats['draw_diff_pp']:.2f}"
        )
        print(
            f"[PROB_DRAW_DIST:BACKTEST] league={league} feature_profile={feature_profile} l2={l2:.6f} min={draw_stats['draw_min']:.3f} "
            f"p50={draw_stats['draw_p50']:.3f} p95={draw_stats['draw_p95']:.3f} max={draw_stats['draw_max']:.3f}"
        )
        row = {
            "league": league,
            "feature_profile": feature_profile,
            "l2": float(l2),
            "class_weight": str(class_weight_mode_eff),
            "class_weight_alpha": float(class_weight_alpha_eff),
            "accuracy": acc,
            "logloss": ll,
            **draw_stats,
            "iters": int(opt_info["nit"]),
            "final_loss": float(opt_info["final_loss"]),
            "grad_norm": float(opt_info["grad_norm"]),
        }
        grid_rows.append(row)
        candidate_payloads.append(
            {
                "l2": float(l2),
                "W": W,
                "b": b,
                "opt_info": opt_info,
                "acc": float(acc),
                "ll": float(ll),
            }
        )

    if not candidate_payloads:
        raise RuntimeError(f"grid search failed: league={league}")

    # Primary metric: minimum logloss. Tie-breaker: higher accuracy, then smaller l2.
    best_candidate = min(candidate_payloads, key=lambda x: (x["ll"], -x["acc"], x["l2"]))
    best_l2 = float(best_candidate["l2"])
    W = best_candidate["W"]
    b = best_candidate["b"]
    opt_info = best_candidate["opt_info"]
    acc = float(best_candidate["acc"])
    ll = float(best_candidate["ll"])
    print(f"[GRID_BEST] league={league} feature_profile={feature_profile} l2={best_l2:.6f} logloss={ll:.6f} accuracy={acc:.4f}")

    p_test = predict_proba(X_test, W, b)
    pred_test = np.argmax(p_test, axis=1)
    logits_test_best = X_test.dot(W.T) + b
    log_logits_stats(logits_test_best, "valid_best", league, feature_profile, l2=best_l2)
    log_pred_dist(pred_test, "valid_best_pred", league, feature_profile)
    cm = confusion(y_test, pred_test)
    print(
        f"[TRAIN_EVAL] league={league} feature_profile={feature_profile} rows={n} train={len(y_train)} test={len(y_test)} "
        f"accuracy={acc:.4f} logloss={ll:.6f}"
    )
    print("[TRAIN_EVAL] confusion_matrix rows=actual(H,D,A) cols=pred(H,D,A)")
    print(f"[TRAIN_EVAL] league={league} feature_profile={feature_profile} confusion={cm.tolist()}")
    cm_named = {
        INDEX_TO_CLASS[i]: {INDEX_TO_CLASS[j]: int(cm[i, j]) for j in range(len(CLASS_ORDER))}
        for i in range(len(CLASS_ORDER))
    }
    print(f"[TRAIN_EVAL] league={league} feature_profile={feature_profile} confusion_named={json.dumps(cm_named, ensure_ascii=False)}")

    # Baseline: always predict H with epsilon smoothing on 3-class probs.
    baseline_pred = np.zeros_like(y_test, dtype=int)
    baseline_probs = np.zeros((len(y_test), len(CLASS_ORDER)), dtype=float)
    eps = float(args.baseline_eps)
    eps = max(1e-12, min(1e-2, eps))
    baseline_probs[:, CLASS_TO_INDEX["H"]] = 1.0 - eps
    baseline_probs[:, CLASS_TO_INDEX["D"]] = eps / 2.0
    baseline_probs[:, CLASS_TO_INDEX["A"]] = eps / 2.0
    baseline_acc = float(np.mean(baseline_pred == y_test))
    baseline_ll = logloss(y_test, baseline_probs)
    print(
        f"[BASELINE_CONFIG] league={league} feature_profile={feature_profile} "
        f"type=always_H_smoothed eps={eps:.1e} formula=P(H)=1-eps,P(D)=eps/2,P(A)=eps/2"
    )
    print(
        f"[BASELINE] league={league} feature_profile={feature_profile} type=always_H "
        f"accuracy={baseline_acc:.4f} logloss={baseline_ll:.6f}"
    )

    X_train_zero = np.zeros_like(X_train)
    X_test_zero = np.zeros_like(X_test)
    W0, b0, opt0 = fit_softmax(
        X_train_zero,
        y_train,
        l2=float(best_l2),
        maxiter=int(args.maxiter),
        gtol=float(args.gtol),
        ftol=float(args.ftol),
        min_total_iters=int(args.min_warn_iters),
        class_weight_mode=class_weight_mode_eff,
        class_weight_alpha=float(class_weight_alpha_eff),
    )
    p_test0 = predict_proba(X_test_zero, W0, b0)
    pred_test0 = np.argmax(p_test0, axis=1)
    acc0 = float(np.mean(pred_test0 == y_test))
    ll0 = logloss(y_test, p_test0)
    print(
        f"[INTERCEPT_ONLY_EVAL] league={league} feature_profile={feature_profile} rows={n} train={len(y_train)} test={len(y_test)} "
        f"accuracy={acc0:.4f} logloss={ll0:.6f}"
    )
    print(
        f"[INTERCEPT_ONLY_OPT] league={league} feature_profile={feature_profile} status={opt0['status']} success={int(opt0['success'])} "
        f"iters={opt0['nit']} final_loss={opt0['final_loss']:.6f} grad_norm={opt0['grad_norm']:.6e}"
    )
    log_prob_dist_from_array(f"PROB_DRAW_DIST:INTERCEPT_ONLY:{league}:{feature_profile}", p_test0[:, 1])
    if abs(ll - ll0) <= 0.005 and abs(acc - acc0) <= 0.01:
        print(
            f"[WARN] league={league} feature_profile={feature_profile} model is close to intercept-only baseline "
            f"(delta_logloss={abs(ll-ll0):.6f}, delta_acc={abs(acc-acc0):.4f})"
        )

    p_all = predict_proba((X_raw - mu) / sigma, W, b)
    out = df.copy()
    out["prob_home"] = p_all[:, 0]
    out["prob_draw"] = p_all[:, 1]
    out["prob_away"] = p_all[:, 2]
    rows, argmax_d, ge023 = summarize_probs(out)
    print(
        f"[TRAIN_PROB_CHECK] league={league} feature_profile={feature_profile} "
        f"rows={rows} prob_draw_argmax_count={argmax_d} prob_draw_ge_0.23_count={ge023}"
    )

    grid_df = pd.DataFrame(grid_rows).sort_values(["logloss", "accuracy"], ascending=[True, False])
    min_ll = float(grid_df["logloss"].astype(float).min())
    if abs(float(ll) - min_ll) > 1e-12:
        raise RuntimeError(
            f"[GRID_BEST_MISMATCH] league={league} feature_profile={feature_profile} "
            f"best_logloss={ll:.12f} grid_min_logloss={min_ll:.12f}"
        )
    print(
        f"[GRID_BEST_CHECK] league={league} feature_profile={feature_profile} "
        f"best_logloss={ll:.6f} grid_min_logloss={min_ll:.6f} status=ok"
    )
    grid_csv = MODEL_DIR / f"hda_multinom_grid_{args.season}_{league}__{feature_profile}.csv"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    print(f"[GRID_OUT] league={league} feature_profile={feature_profile} csv={grid_csv}")

    bundle = {
        "type": "softmax_linear",
        "classes": CLASS_ORDER,
        "feature_names": feature_names,
        "coef": W,
        "intercept": b,
        "feature_mean": mu,
        "feature_std": sigma,
        "season": int(args.season),
        "league": league,
        "feature_profile": feature_profile,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "metrics": {"accuracy": acc, "logloss": ll, "confusion": cm.tolist()},
        "selected_l2": float(best_l2),
        "class_weight": str(class_weight_mode_eff),
        "class_weight_alpha": float(class_weight_alpha_eff),
        "baseline_eps": float(args.baseline_eps),
        "optimizer": opt_info,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = resolve_model_out_path(args, league, feature_profile)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[TRAIN_OUT] league={league} feature_profile={feature_profile} model={out_path}")

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, ensure_ascii=False, indent=2)
    print(f"[TRAIN_OUT] league={league} feature_profile={feature_profile} metrics={metrics_path}")

    if args.eval_backtest:
        eval_summary = evaluate_on_backtest_df(df, W, b, mu, sigma, feature_names, league, feature_profile)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        eval_path = MODEL_DIR / f"hda_eval_summary_{league}__{feature_profile}.json"
        eval_path.write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[MODEL_EVAL_OUT] league={league} feature_profile={feature_profile} path={eval_path}")
        maybe_print_j1_profile_diff()


def main():
    args = parse_args()
    targets = [args.league] if args.league in {"j1", "j2"} else ["j1", "j2"]
    for lg in targets:
        train_one_league(args, lg)


if __name__ == "__main__":
    main()
