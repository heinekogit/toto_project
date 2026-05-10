#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd

import buyplan


DEFAULT_ROUNDS_DIR = os.path.join(ROOT_DIR, "data", "eval", "rounds")
DEFAULT_OUT_ROOT = os.path.join(ROOT_DIR, "data", "purchase_reference", "backtest")
DEFAULT_CURRENT_PREDICTIONS = os.path.join(ROOT_DIR, "data", "purchase_reference", "predictions.csv")


def _align_predictions_to_actual(pred_df: pd.DataFrame, actual_df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    if "home_team" not in pred_df.columns or "away_team" not in pred_df.columns:
        raise ValueError("snapshot predictions に home_team/away_team がありません")

    src = pred_df.copy()
    src["_home_key"] = src["home_team"].map(buyplan._norm_team_key)
    src["_away_key"] = src["away_team"].map(buyplan._norm_team_key)
    key_cols = ["_home_key", "_away_key"]
    dup = int(src.duplicated(subset=key_cols, keep="first").sum())
    if dup:
        warnings.append(f"predictions側で重複カードが {dup} 件あるため、先頭行を採用します。")
        src = src.drop_duplicates(subset=key_cols, keep="first")
    src_map = {(r["_home_key"], r["_away_key"]): r for _, r in src.iterrows()}

    order = actual_df.copy()
    order["_home_key"] = order["home_team"].map(buyplan._norm_team_key)
    order["_away_key"] = order["away_team"].map(buyplan._norm_team_key)
    rows = []
    miss = 0
    for _, o in order.sort_values("match_no").iterrows():
        key = (o["_home_key"], o["_away_key"])
        if key not in src_map:
            raise ValueError(
                f"snapshot predictions に actual 対象カードがありません: "
                f"{o['home_team']} vs {o['away_team']} (match_no={int(o['match_no'])})"
            )
        row = dict(src_map[key])
        row["match_no"] = int(o["match_no"])
        if "league" in o:
            row["league"] = o["league"]
        row["home_team"] = o["home_team"]
        row["away_team"] = o["away_team"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("match_no").reset_index(drop=True)
    out = out.drop(columns=["_home_key", "_away_key"], errors="ignore")
    return out


def _discover_rounds(rounds_dir: str) -> List[str]:
    names: List[str] = []
    if not os.path.isdir(rounds_dir):
        return names
    for name in sorted(os.listdir(rounds_dir)):
        if not re.fullmatch(r"round\d{2}", name):
            continue
        pred_csv = os.path.join(rounds_dir, name, "snapshot", "predictions.csv")
        actual_csv = os.path.join(rounds_dir, name, "actual_results.csv")
        if os.path.exists(pred_csv) and os.path.exists(actual_csv):
            names.append(name)
    return names


def _build_backtest(
    round_name: str,
    rounds_dir: str,
    out_root: str,
    predictions_override: str | None = None,
) -> str:
    warnings: List[str] = []
    round_dir = os.path.join(rounds_dir, round_name)
    snapshot_pred_csv = os.path.join(round_dir, "snapshot", "predictions.csv")
    pred_csv = predictions_override or snapshot_pred_csv
    actual_csv = os.path.join(round_dir, "actual_results.csv")
    outdir = os.path.join(out_root, round_name)
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"predictions.csv not found: {pred_csv}")
    df = pd.read_csv(pred_csv)
    actual_df = buyplan._load_actual_results_df(actual_csv)
    df = _align_predictions_to_actual(df, actual_df, warnings)
    df = buyplan._normalize_match_no(df, warnings)
    df = buyplan._dedupe_match_no(df, warnings)
    plans = buyplan._build_match_plans(df, warnings, base_mode=buyplan.BUYPLAN_BASE_MODE)
    tickets, flip_descs, scenario_defs, ticket_stats, scenario_result_labels = buyplan._generate_tickets_by_scenario(df, plans, warnings)

    buyplan._write_buyplan_csv(
        plans=plans,
        tickets=tickets,
        out_csv=os.path.join(outdir, "buyplan.csv"),
        scenario_defs=scenario_defs,
        scenario_result_labels=scenario_result_labels,
    )
    buyplan._write_buyplan_html(
        plans=plans,
        tickets=tickets,
        flip_descs=flip_descs,
        warnings=warnings,
        out_html=os.path.join(outdir, "buyplan.html"),
        input_csv=pred_csv,
        outdir=outdir,
        ticket_stats=ticket_stats,
        scenario_defs=scenario_defs,
    )
    _, scored_html = buyplan._write_buyplan_scored_outputs(
        plans=plans,
        tickets=tickets,
        outdir=outdir,
        actual_df=actual_df,
        actual_csv_path=actual_csv,
    )
    return scored_html


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build current buyplan backtests for historical rounds")
    p.add_argument("--round", action="append", dest="rounds", default=[], help="round02 のように指定。複数回指定可")
    p.add_argument("--rounds-dir", default=DEFAULT_ROUNDS_DIR, help="既定: data/eval/rounds")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="既定: data/purchase_reference/backtest")
    p.add_argument(
        "--use-current-predictions",
        action="store_true",
        help="snapshot/predictions.csv ではなく data/purchase_reference/predictions.csv を使う",
    )
    p.add_argument(
        "--current-predictions",
        default=DEFAULT_CURRENT_PREDICTIONS,
        help="--use-current-predictions 時に使う predictions.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rounds = args.rounds or _discover_rounds(args.rounds_dir)
    if not rounds:
        raise SystemExit("backtest対象のroundが見つかりません")
    predictions_override = None
    if args.use_current_predictions:
        if len(rounds) != 1:
            raise SystemExit("--use-current-predictions は --round を1つだけ指定して使ってください")
        predictions_override = os.path.abspath(args.current_predictions)

    for round_name in rounds:
        try:
            scored_html = _build_backtest(round_name, args.rounds_dir, args.out_root, predictions_override=predictions_override)
            print(f"[OK] {scored_html}")
        except Exception as e:
            print(f"[SKIP] {round_name} {e}")


if __name__ == "__main__":
    main()
