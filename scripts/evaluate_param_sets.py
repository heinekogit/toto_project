#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "scripts" / ".venv" / "bin" / "python"
PRED_SCRIPT = ROOT / "scripts" / "11_prediction_01.py"

PROB_COL_SETS: List[Tuple[str, str, str]] = [
    ("prob_home_win", "prob_draw", "prob_away_win"),
    ("prob_home", "prob_draw", "prob_away"),
    ("p_home", "p_draw", "p_away"),
]

PARAM_SETS = {
    "baseline": {
        "DRAW_BLEND_WEIGHT": "0.750",
        "ELO_DRAW_MAX": "0.330",
        "DRAW_DECAY_SCALE": "120.0",
    },
    "current": {
        "DRAW_BLEND_WEIGHT": "0.600",
        "ELO_DRAW_MAX": "0.380",
        "DRAW_DECAY_SCALE": "320.0",
    },
}

COMMON_ENV = {
    "ELO_DIFF_SCALE": "1.0",
    "ELO_DRAW_MIN": "0.100",
    "ELO_DRAW_DIFF_SCALE": "1.0",
    "DRAW_ASSIGN_BY_EXPECTATION": "1",
    "ENABLE_HFA": "0",
}


@dataclass
class DatasetConfig:
    name: str
    season_year: int
    season_label: str
    round_min: Optional[int] = None
    round_max: Optional[int] = None


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "2025_rounds": DatasetConfig(name="2025_rounds", season_year=2025, season_label="{league}_2025_rounds"),
    "2026_r1to4": DatasetConfig(name="2026_r1to4", season_year=2026, season_label="{league}_2026_r1to4", round_min=1, round_max=4),
    "2026_r3to4": DatasetConfig(name="2026_r3to4", season_year=2026, season_label="{league}_2026_r3to4", round_min=3, round_max=4),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate old/new parameter sets with probabilistic metrics.")
    p.add_argument("--league", required=True, choices=["j1", "j2"])
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    p.add_argument("--skip-run", action="store_true", help="Skip model execution and only evaluate existing snapshots.")
    p.add_argument("--keep-snapshots", action="store_true", help="Keep per-set backtest snapshots.")
    return p.parse_args()


def _extract_round_no(val: object) -> Optional[int]:
    if pd.isna(val):
        return None
    s = str(val)
    trans = str.maketrans("０１２３４５６７８９", "0123456789")
    s = s.translate(trans)
    import re

    m = re.search(r"第\s*(\d+)\s*節", s)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", s)
    return int(m.group(1)) if m else None


def _result_from_scores(home_score: object, away_score: object) -> Optional[str]:
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    try:
        h = float(home_score)
        a = float(away_score)
    except Exception:
        return None
    if h > a:
        return "H"
    if h < a:
        return "A"
    return "D"


def _detect_prob_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    for cols in PROB_COL_SETS:
        if all(c in df.columns for c in cols):
            return cols
    available = ", ".join(df.columns[:50])
    if len(df.columns) > 50:
        available += " ... " + ", ".join(df.columns[-20:])
    raise RuntimeError(f"[ERROR] probability columns not found. available_columns={available}")


def _ensure_actual_result(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "actual_result" not in out.columns:
        out["actual_result"] = np.nan
    if {"home_score", "away_score"}.issubset(out.columns):
        need = out["actual_result"].isna() | (out["actual_result"].astype(str).str.strip() == "")
        if need.any():
            out.loc[need, "actual_result"] = out.loc[need].apply(
                lambda r: _result_from_scores(r.get("home_score"), r.get("away_score")), axis=1
            )
    out["actual_result"] = out["actual_result"].astype(str).str.upper().where(out["actual_result"].notna(), np.nan)
    return out


def _apply_dataset_filter(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_actual_result(out)
    out = out[out["actual_result"].isin(["H", "D", "A"])].copy()

    if cfg.round_min is not None and cfg.round_max is not None:
        if "節" not in out.columns:
            raise RuntimeError("[ERROR] round filter requested but '節' column not found.")
        out["_round_no"] = out["節"].apply(_extract_round_no)
        out = out[(out["_round_no"] >= cfg.round_min) & (out["_round_no"] <= cfg.round_max)].copy()
    return out


def _safe_logloss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    probs = np.clip(probs, eps, 1.0 - eps)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(row_sums == 0, 1.0, row_sums)
    idx = y_true.astype(int)
    return float(-np.mean(np.log(probs[np.arange(len(probs)), idx])))


def _multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    k = probs.shape[1]
    y_onehot = np.zeros((len(y_true), k), dtype=float)
    y_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def _hda_ratio(series: pd.Series) -> Tuple[float, float, float, int, int, int]:
    s = series.astype(str).str.upper()
    total = len(s)
    if total == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0
    h = int((s == "H").sum())
    d = int((s == "D").sum())
    a = int((s == "A").sum())
    return (100.0 * h / total, 100.0 * d / total, 100.0 * a / total, h, d, a)


def _metrics_for_df(df: pd.DataFrame) -> Dict[str, object]:
    p_home, p_draw, p_away = _detect_prob_cols(df)
    work = df.copy()
    work[p_home] = pd.to_numeric(work[p_home], errors="coerce")
    work[p_draw] = pd.to_numeric(work[p_draw], errors="coerce")
    work[p_away] = pd.to_numeric(work[p_away], errors="coerce")
    work = work.dropna(subset=[p_home, p_draw, p_away, "actual_result"]).copy()

    label_to_idx = {"H": 0, "D": 1, "A": 2}
    y = work["actual_result"].map(label_to_idx)
    work = work[y.notna()].copy()
    y = y[y.notna()].astype(int).to_numpy()

    probs = work[[p_home, p_draw, p_away]].to_numpy(dtype=float)
    sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(sums == 0, 1.0, sums)

    argmax_idx = np.argmax(probs, axis=1)
    idx_to_label = np.array(["H", "D", "A"])
    pred_argmax = pd.Series(idx_to_label[argmax_idx], index=work.index)

    logloss = _safe_logloss(y, probs)
    brier = _multiclass_brier(y, probs)
    acc = float(np.mean(argmax_idx == y)) if len(y) else float("nan")

    ph, pdw, pa, ch, cd, ca = _hda_ratio(pred_argmax)
    ah, ad, aa, ahc, adc, aac = _hda_ratio(work["actual_result"])

    return {
        "N": int(len(work)),
        "logloss": logloss,
        "brier": brier,
        "top1_accuracy": acc,
        "pred_H_pct": ph,
        "pred_D_pct": pdw,
        "pred_A_pct": pa,
        "pred_H_cnt": ch,
        "pred_D_cnt": cd,
        "pred_A_cnt": ca,
        "actual_H_pct": ah,
        "actual_D_pct": ad,
        "actual_A_pct": aa,
        "actual_H_cnt": ahc,
        "actual_D_cnt": adc,
        "actual_A_cnt": aac,
        "prob_cols": f"{p_home},{p_draw},{p_away}",
    }


def _draw_reliability(df: pd.DataFrame, set_name: str) -> pd.DataFrame:
    p_home, p_draw, p_away = _detect_prob_cols(df)
    work = df.copy()
    work[p_draw] = pd.to_numeric(work[p_draw], errors="coerce")
    work = work.dropna(subset=[p_draw, "actual_result"]).copy()
    work["is_draw"] = (work["actual_result"].astype(str).str.upper() == "D").astype(int)

    edges = np.arange(0.0, 0.5000001, 0.05)
    rows = []
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (work[p_draw] >= left) & (work[p_draw] <= right)
            label = f"[{left:.2f},{right:.2f}]"
        else:
            mask = (work[p_draw] >= left) & (work[p_draw] < right)
            label = f"[{left:.2f},{right:.2f})"
        b = work.loc[mask]
        rows.append(
            {
                "set": set_name,
                "bin": label,
                "bin_left": left,
                "bin_right": right,
                "count": int(len(b)),
                "avg_pred_draw": float(b[p_draw].mean()) if len(b) else float("nan"),
                "actual_draw_rate": float(b["is_draw"].mean()) if len(b) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out["gap"] = out["actual_draw_rate"] - out["avg_pred_draw"]
    return out


def _run_prediction(league: str, season_year: int, set_name: str, out_dir: Path) -> Path:
    out_pred = out_dir / f"{league}_{season_year}_predictions_{set_name}.csv"
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update(PARAM_SETS[set_name])
    env["LEAGUE"] = league
    env["SEASON_YEAR"] = str(season_year)
    env["OUTPUT_PRED_CSV"] = str(out_pred)

    cmd = [str(PYTHON if PYTHON.exists() else Path(sys.executable)), str(PRED_SCRIPT), "--force", "--skip-hfa-self-check"]
    print(
        "[EVAL_RUN] "
        f"set={set_name} league={league} season={season_year} "
        f"ENABLE_HFA={env['ENABLE_HFA']} DRAW_BLEND_WEIGHT={env['DRAW_BLEND_WEIGHT']} "
        f"ELO_DRAW_MAX={env['ELO_DRAW_MAX']} DRAW_DECAY_SCALE={env['DRAW_DECAY_SCALE']}"
    )
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    backtest_src = ROOT / f"backtest_{league}_{season_year}.csv"
    if not backtest_src.exists():
        raise RuntimeError(f"[ERROR] backtest output not found: {backtest_src}")

    out_bt = out_dir / f"backtest_{league}_{season_year}_{set_name}.csv"
    df_bt = pd.read_csv(backtest_src)
    df_bt.to_csv(out_bt, index=False, encoding="utf-8-sig")
    print(f"[EVAL_RUN] snapshot_backtest={out_bt} rows={len(df_bt)}")
    return out_bt


def _swap_upcoming_with_latest_if_needed(league: str, cfg: DatasetConfig) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if cfg.season_year != 2025:
        return None, None, None
    upcoming = ROOT / "data" / f"{league}_{cfg.season_year}_upcoming.csv"
    latest = ROOT / "data" / f"{league}_{cfg.season_year}_latest_results.csv"
    if not latest.exists():
        raise RuntimeError(f"[ERROR] latest results CSV not found for 2025 evaluation: {latest}")
    backup = None
    if upcoming.exists():
        backup = upcoming.with_suffix(upcoming.suffix + ".eval_bak")
        if backup.exists():
            backup.unlink()
        upcoming.rename(backup)
    df_latest = pd.read_csv(latest)
    df_latest.to_csv(upcoming, index=False, encoding="utf-8-sig")
    print(f"[EVAL_PREP] 2025 run uses latest_results as season source: {latest} -> {upcoming}")
    return upcoming, latest, backup


def _restore_upcoming_after_swap(upcoming: Optional[Path], backup: Optional[Path]) -> None:
    if upcoming is None:
        return
    if upcoming.exists():
        upcoming.unlink()
    if backup is not None and backup.exists():
        backup.rename(upcoming)
    print(f"[EVAL_PREP] restored season source: {upcoming}")


def _load_snapshot(league: str, season_year: int, set_name: str, out_dir: Path) -> Path:
    p = out_dir / f"backtest_{league}_{season_year}_{set_name}.csv"
    if not p.exists():
        raise RuntimeError(f"[ERROR] snapshot not found for --skip-run: {p}")
    return p


def _build_html(metrics: pd.DataFrame, rel: pd.DataFrame, out_html: Path, title: str) -> None:
    style = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; }
      table { border-collapse: collapse; margin-bottom: 24px; min-width: 980px; }
      th, td { border: 1px solid #d0d0d0; padding: 6px 8px; font-size: 13px; }
      th { background: #f4f6f8; text-align: left; }
      h1, h2 { margin: 8px 0 12px; }
      .muted { color: #666; font-size: 12px; }
    </style>
    """
    html = ["<html><head><meta charset='utf-8'>", style, "</head><body>"]
    html.append(f"<h1>{title}</h1>")
    html.append("<h2>metrics_summary</h2>")
    html.append(metrics.to_html(index=False, na_rep=""))
    html.append("<h2>reliability_draw</h2>")
    html.append(rel.to_html(index=False, na_rep=""))
    html.append("<p class='muted'>Generated by scripts/evaluate_param_sets.py</p>")
    html.append("</body></html>")
    out_html.write_text("\n".join(html), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    season_label = cfg.season_label.format(league=args.league)
    out_dir = ROOT / "data" / "reports" / "metrics" / season_label
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshots: Dict[str, Path] = {}
    upcoming_swapped = None
    backup = None
    try:
        if not args.skip_run:
            upcoming_swapped, _, backup = _swap_upcoming_with_latest_if_needed(args.league, cfg)
        for set_name in ("baseline", "current"):
            if args.skip_run:
                snapshots[set_name] = _load_snapshot(args.league, cfg.season_year, set_name, out_dir)
            else:
                snapshots[set_name] = _run_prediction(args.league, cfg.season_year, set_name, out_dir)
    finally:
        _restore_upcoming_after_swap(upcoming_swapped, backup)

    metrics_rows: List[Dict[str, object]] = []
    rel_rows: List[pd.DataFrame] = []

    for set_name, path in snapshots.items():
        df = pd.read_csv(path)
        print(f"[EVAL_SCHEMA] set={set_name} rows_raw={len(df)} columns={len(df.columns)}")
        print(f"[EVAL_SCHEMA] set={set_name} available_columns={list(df.columns[:40])}{' ...' if len(df.columns) > 40 else ''}")

        filtered = _apply_dataset_filter(df, cfg)
        if filtered.empty:
            raise RuntimeError(
                f"[ERROR] filtered dataset is empty: set={set_name} league={args.league} dataset={args.dataset}"
            )
        m = _metrics_for_df(filtered)
        m.update(
            {
                "set": set_name,
                "league": args.league,
                "dataset": args.dataset,
                "season_year": cfg.season_year,
                "season_label": season_label,
            }
        )
        metrics_rows.append(m)
        rel_rows.append(_draw_reliability(filtered, set_name))
        print(
            f"[EVAL_METRICS] set={set_name} N={m['N']} logloss={m['logloss']:.6f} "
            f"brier={m['brier']:.6f} top1={m['top1_accuracy']:.4f}"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    rel_df = pd.concat(rel_rows, ignore_index=True)

    metrics_csv = out_dir / "metrics_summary.csv"
    rel_csv = out_dir / "reliability_draw.csv"
    html_path = out_dir / "report.html"

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    rel_df.to_csv(rel_csv, index=False, encoding="utf-8-sig")
    _build_html(metrics_df, rel_df, html_path, f"Param Set Eval: {args.league.upper()} {args.dataset}")

    print(f"[EVAL_OUT] metrics_summary={metrics_csv}")
    print(f"[EVAL_OUT] reliability_draw={rel_csv}")
    print(f"[EVAL_OUT] report_html={html_path}")

    if not args.keep_snapshots:
        for p in snapshots.values():
            if p.exists():
                p.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
