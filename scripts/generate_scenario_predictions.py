#!/usr/bin/env python3
import argparse
import glob
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT = "data/purchase_reference/predictions.csv"
DEFAULT_OUTPUT_DIR = "data/purchase_reference"
PROB_CLIP_MIN = 0.02
PROB_CLIP_MAX = 0.90


def _pick_prob_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    if {"prob_home_win", "prob_draw", "prob_away_win"}.issubset(df.columns):
        return "prob_home_win", "prob_draw", "prob_away_win"
    if {"p_home", "p_draw", "p_away"}.issubset(df.columns):
        return "p_home", "p_draw", "p_away"
    return "", "", ""


def _normalize_probs(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    arr = np.array([p_h, p_d, p_a], dtype=float)
    arr = np.nan_to_num(arr, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0)
    arr = np.clip(arr, PROB_CLIP_MIN, PROB_CLIP_MAX)
    s = float(arr.sum())
    if s <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    arr /= s
    return float(arr[0]), float(arr[1]), float(arr[2])


def _argmax_result(p_h: float, p_d: float, p_a: float) -> str:
    if p_h >= p_d and p_h >= p_a:
        return "H"
    if p_d >= p_h and p_d >= p_a:
        return "D"
    return "A"


def _apply_slight_upset(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    arr = np.array([p_h, p_d, p_a], dtype=float)
    best_idx = int(np.argmax(arr))
    best = float(arr[best_idx])
    if best > 0.55:
        arr[best_idx] -= 0.05
        others = [i for i in [0, 1, 2] if i != best_idx]
        arr[others[0]] += 0.03
        arr[others[1]] += 0.02
    return _normalize_probs(float(arr[0]), float(arr[1]), float(arr[2]))


def _apply_strong_upset(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    arr = np.array([p_h, p_d, p_a], dtype=float)
    best_idx = int(np.argmax(arr))
    best = float(arr[best_idx])
    if best > 0.60:
        arr[best_idx] -= 0.10
        others = [i for i in [0, 1, 2] if i != best_idx]
        arr[others[0]] += 0.06
        arr[others[1]] += 0.04
    return _normalize_probs(float(arr[0]), float(arr[1]), float(arr[2]))


def _apply_draw_bias(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    return _normalize_probs(p_h - 0.025, p_d + 0.05, p_a - 0.025)


def _apply_home_bias(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    return _normalize_probs(p_h + 0.05, p_d - 0.025, p_a - 0.025)


def _apply_away_bias(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    return _normalize_probs(p_h - 0.025, p_d - 0.025, p_a + 0.05)


def _apply_tight_match(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    if abs(p_h - p_a) < 0.12:
        return _apply_slight_upset(p_h, p_d, p_a)
    return _normalize_probs(p_h, p_d, p_a)


def _apply_chaos(p_h: float, p_d: float, p_a: float, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-0.02, 0.02, size=3)
    return _normalize_probs(p_h + float(noise[0]), p_d + float(noise[1]), p_a + float(noise[2]))


def _apply_conservative(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    # 指示仕様: max(prob) > 0.60 は固定。それ以外も Base と同じ。
    return _normalize_probs(p_h, p_d, p_a)


def _apply_balanced(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    arr = np.array([p_h, p_d, p_a], dtype=float)
    best_idx = int(np.argmax(arr))
    best = float(arr[best_idx])
    if best > 0.60:
        delta = best - 0.60
        arr[best_idx] = 0.60
        other_idx = [i for i in [0, 1, 2] if i != best_idx]
        other_sum = float(arr[other_idx[0]] + arr[other_idx[1]])
        if other_sum > 0:
            arr[other_idx[0]] += delta * (arr[other_idx[0]] / other_sum)
            arr[other_idx[1]] += delta * (arr[other_idx[1]] / other_sum)
        else:
            arr[other_idx[0]] += delta / 2.0
            arr[other_idx[1]] += delta / 2.0
    return _normalize_probs(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class ScenarioDef:
    scenario_id: str
    scenario_name: str
    filename_suffix: str
    transform: Callable[[float, float, float, int], Tuple[float, float, float]]


SCENARIOS: Dict[str, ScenarioDef] = {
    "01": ScenarioDef("01", "Base", "base", lambda h, d, a, seed: _normalize_probs(h, d, a)),
    "02": ScenarioDef("02", "Slight Upset", "slight_upset", lambda h, d, a, seed: _apply_slight_upset(h, d, a)),
    "03": ScenarioDef("03", "Strong Upset", "strong_upset", lambda h, d, a, seed: _apply_strong_upset(h, d, a)),
    "04": ScenarioDef("04", "Draw Bias", "draw_bias", lambda h, d, a, seed: _apply_draw_bias(h, d, a)),
    "05": ScenarioDef("05", "Home Bias", "home_bias", lambda h, d, a, seed: _apply_home_bias(h, d, a)),
    "06": ScenarioDef("06", "Away Bias", "away_bias", lambda h, d, a, seed: _apply_away_bias(h, d, a)),
    "07": ScenarioDef("07", "Tight Match", "tight_match", lambda h, d, a, seed: _apply_tight_match(h, d, a)),
    "08": ScenarioDef("08", "Chaos", "chaos", lambda h, d, a, seed: _apply_chaos(h, d, a, seed)),
    "09": ScenarioDef("09", "Conservative", "conservative", lambda h, d, a, seed: _apply_conservative(h, d, a)),
    "10": ScenarioDef("10", "Balanced", "balanced", lambda h, d, a, seed: _apply_balanced(h, d, a)),
}


def _cleanup_old_outputs(output_dir: str) -> None:
    old_files = glob.glob(os.path.join(output_dir, "predictions_scenario_*.csv"))
    for f in old_files:
        os.remove(f)


def _build_scenario_df(
    base_df: pd.DataFrame,
    ph_col: str,
    pd_col: str,
    pa_col: str,
    scenario: ScenarioDef,
) -> pd.DataFrame:
    out = base_df.copy()
    if "predicted_result" in out.columns:
        out["predicted_result_raw"] = out["predicted_result"]
    if "predicted_highest_prob_result" in out.columns:
        out["predicted_highest_prob_result_raw"] = out["predicted_highest_prob_result"]

    for idx, row in out.iterrows():
        p_h = float(pd.to_numeric(row.get(ph_col), errors="coerce"))
        p_d = float(pd.to_numeric(row.get(pd_col), errors="coerce"))
        p_a = float(pd.to_numeric(row.get(pa_col), errors="coerce"))
        seed = int(row.get("match_no", idx + 1)) * 100 + int(scenario.scenario_id)
        h2, d2, a2 = scenario.transform(p_h, p_d, p_a, seed)
        out.at[idx, ph_col] = h2
        out.at[idx, pd_col] = d2
        out.at[idx, pa_col] = a2
        out.at[idx, "predicted_result"] = _argmax_result(h2, d2, a2)
        out.at[idx, "predicted_highest_prob_result"] = out.at[idx, "predicted_result"]

    out["scenario_id"] = scenario.scenario_id
    out["scenario_name"] = scenario.scenario_name
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 10 scenario-adjusted predictions CSV files")
    p.add_argument("--in", dest="in_csv", default=DEFAULT_INPUT, help="input predictions.csv")
    p.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="output directory")
    p.add_argument("--clean", action="store_true", help="remove old predictions_scenario_*.csv before writing")
    args = p.parse_args()

    in_csv = os.path.abspath(args.in_csv)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(in_csv)
    ph_col, pd_col, pa_col = _pick_prob_columns(df)
    if not ph_col:
        raise RuntimeError("確率列が見つかりません。必要列: prob_home_win/prob_draw/prob_away_win または p_home/p_draw/p_away")
    if "match_no" not in df.columns:
        df = df.copy()
        df["match_no"] = np.arange(1, len(df) + 1)

    if args.clean:
        _cleanup_old_outputs(outdir)

    for sid in [f"{i:02d}" for i in range(1, 11)]:
        scenario = SCENARIOS[sid]
        out_df = _build_scenario_df(df, ph_col, pd_col, pa_col, scenario)
        out_name = f"predictions_scenario_{scenario.scenario_id}_{scenario.filename_suffix}.csv"
        out_path = os.path.join(outdir, out_name)
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
