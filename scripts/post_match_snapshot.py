#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_PURCHASE_DIR = os.path.join(DATA_DIR, "purchase_reference")
DEFAULT_OUT_ROOT = os.path.join(DATA_DIR, "verification")


def normalize_text(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v))
    return s.strip()


def normalize_league(v):
    s = normalize_text(v).upper().replace(" ", "")
    if "J1" in s:
        return "J1"
    if "J2" in s:
        return "J2"
    return s


def extract_round_num(v):
    s = normalize_text(v)
    m = re.search(r"第\s*([0-9]+)\s*節", s)
    if not m:
        return None
    return int(m.group(1))


def ensure_dir(path, force=False):
    if os.path.exists(path):
        if not force:
            raise RuntimeError(f"出力先が既に存在します（--forceで上書き可）: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def run_update_results(script_python, season, league):
    env = os.environ.copy()
    env["SEASON_YEAR"] = str(season)
    env["LEAGUE"] = league.lower()
    cmd = [script_python, os.path.join(ROOT_DIR, "scripts", "01_update_match_results.py")]
    print(f"[RUN] {' '.join(cmd)} (LEAGUE={league.lower()} SEASON_YEAR={season})")
    cp = subprocess.run(cmd, env=env, cwd=ROOT_DIR)
    if cp.returncode != 0:
        raise RuntimeError(f"01_update_match_results.py failed: league={league} season={season} rc={cp.returncode}")


def load_results_source(season, league):
    cands = [
        os.path.join(DATA_DIR, f"{league.lower()}_{season}_latest_results.csv"),
        os.path.join(DATA_DIR, f"{league.lower()}_{season}_upcoming.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if not df.empty:
                return p, df
    raise FileNotFoundError(f"結果CSVが見つかりません: {cands}")


def build_actual_results(df_results, league, round_num):
    if "節" not in df_results.columns:
        raise ValueError("結果CSVに '節' 列がありません")
    df = df_results.copy()
    df["round_num"] = df["節"].map(extract_round_num)
    df = df[df["round_num"] == int(round_num)].copy()
    if df.empty:
        return df

    needed = ["match_id", "datetime", "home_team", "away_team", "home_score", "away_score"]
    for col in needed:
        if col not in df.columns:
            df[col] = pd.NA

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    def to_result(row):
        hs = row["home_score"]
        aw = row["away_score"]
        if pd.isna(hs) or pd.isna(aw):
            return pd.NA
        if hs > aw:
            return "H"
        if hs < aw:
            return "A"
        return "D"

    df["result_1x2"] = df.apply(to_result, axis=1)
    df["league"] = league
    out_cols = ["league", "節", "round_num", "match_id", "datetime", "home_team", "away_team", "home_score", "away_score", "result_1x2"]
    return df[out_cols].reset_index(drop=True)


def safe_copy(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False


def find_ticket_columns(df_buyplan):
    cols = []
    for c in df_buyplan.columns:
        cc = c.lower()
        if re.fullmatch(r"ticket\d+", cc):
            cols.append(c)
        elif re.fullmatch(r"候補\d+", normalize_text(c)):
            cols.append(c)
    return cols


def to_mark_102(result_hda):
    if pd.isna(result_hda):
        return pd.NA
    if result_hda == "H":
        return "1"
    if result_hda == "D":
        return "0"
    if result_hda == "A":
        return "2"
    return pd.NA


def build_evaluation(pred_round, buyplan_round, actual_round):
    # predicted_result 精度
    join_cols = ["match_id"]
    if "match_id" in pred_round.columns and "match_id" in actual_round.columns:
        pred_eval = pred_round.merge(actual_round[["match_id", "result_1x2"]], on="match_id", how="left")
    else:
        pred_eval = pred_round.merge(
            actual_round[["home_team", "away_team", "result_1x2"]],
            on=["home_team", "away_team"],
            how="left",
        )

    pred_eval["pred_ok"] = (
        pred_eval["predicted_result"].astype(str).str.upper() == pred_eval["result_1x2"].astype(str).str.upper()
    )
    pred_eval_valid = pred_eval[pred_eval["result_1x2"].notna()].copy()
    pred_total = int(len(pred_eval_valid))
    pred_hits = int(pred_eval_valid["pred_ok"].sum()) if pred_total > 0 else 0
    pred_acc = (pred_hits / pred_total) if pred_total else None

    # buyplan 的中分布
    buy = buyplan_round.copy()
    actual_key = actual_round[["home_team", "away_team", "result_1x2"]].copy()
    buy = buy.merge(actual_key, on=["home_team", "away_team"], how="left")
    buy["actual_mark"] = buy["result_1x2"].map(to_mark_102)

    ticket_cols = find_ticket_columns(buy)
    hits_per_ticket = {}
    distribution = {}
    unresolved = int(buy["actual_mark"].isna().sum())
    for col in ticket_cols:
        p = buy[col].astype(str).str.strip()
        ok = (p == buy["actual_mark"].astype(str)) & buy["actual_mark"].notna()
        hits = int(ok.sum())
        hits_per_ticket[col] = hits
        distribution[str(hits)] = distribution.get(str(hits), 0) + 1

    any_full_hit = any(v == int(buy["actual_mark"].notna().sum()) for v in hits_per_ticket.values()) if ticket_cols else False

    return {
        "predicted_result_eval": {
            "evaluated_matches": pred_total,
            "hits": pred_hits,
            "accuracy": pred_acc,
        },
        "buyplan_eval": {
            "ticket_columns": ticket_cols,
            "hits_per_ticket": hits_per_ticket,
            "hit_distribution": distribution,
            "any_full_hit": any_full_hit,
            "unresolved_matches": unresolved,
        },
    }, pred_eval


def parse_args():
    p = argparse.ArgumentParser(description="Post-match snapshot and round evaluation")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--league", choices=["j1", "j2", "both"], required=True)
    p.add_argument("--round", type=int, required=True, dest="round_num")
    p.add_argument("--purchase-dir", default=DEFAULT_PURCHASE_DIR)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--python", default=os.path.join(ROOT_DIR, "scripts", ".venv", "bin", "python"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    leagues = ["j1", "j2"] if args.league == "both" else [args.league]
    round_label = f"round{args.round_num:02d}"

    predictions_path = os.path.join(args.purchase_dir, "predictions.csv")
    buyplan_path = os.path.join(args.purchase_dir, "buyplan.csv")
    buyplan_html_path = os.path.join(args.purchase_dir, "buyplan.html")

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"predictions.csv not found: {predictions_path}")
    if not os.path.exists(buyplan_path):
        raise FileNotFoundError(f"buyplan.csv not found: {buyplan_path}")

    df_pred_all = pd.read_csv(predictions_path)
    df_buy_all = pd.read_csv(buyplan_path)
    df_pred_all["league_norm"] = df_pred_all.get("league", pd.Series(dtype=str)).map(normalize_league)
    if "節" in df_pred_all.columns:
        df_pred_all["round_num"] = df_pred_all["節"].map(extract_round_num)
    else:
        df_pred_all["round_num"] = pd.NA
    if "league" in df_buy_all.columns:
        df_buy_all["league_norm"] = df_buy_all["league"].map(normalize_league)
    else:
        df_buy_all["league_norm"] = pd.NA

    saved_paths = []
    for lg in leagues:
        lg_upper = lg.upper()
        out_dir = os.path.join(args.out_root, str(args.season), lg, round_label)
        ensure_dir(out_dir, force=args.force)

        # 1) 結果更新
        run_update_results(args.python, args.season, lg)

        # 2) 実結果抽出
        result_src_path, df_results = load_results_source(args.season, lg)
        actual_round = build_actual_results(df_results, lg_upper, args.round_num)
        actual_csv = os.path.join(out_dir, "actual_results.csv")
        actual_round.to_csv(actual_csv, index=False, encoding="utf-8-sig")
        saved_paths.append(actual_csv)

        # 3) purchase_reference スナップショット
        pred_raw_dst = os.path.join(out_dir, "predictions.source.csv")
        buy_raw_dst = os.path.join(out_dir, "buyplan.source.csv")
        shutil.copy2(predictions_path, pred_raw_dst)
        shutil.copy2(buyplan_path, buy_raw_dst)
        saved_paths.extend([pred_raw_dst, buy_raw_dst])
        if os.path.exists(buyplan_html_path):
            dst = os.path.join(out_dir, "buyplan.source.html")
            shutil.copy2(buyplan_html_path, dst)
            saved_paths.append(dst)

        # round+league 抽出版
        pred_round = df_pred_all[
            (df_pred_all["league_norm"] == lg_upper) & (df_pred_all["round_num"] == int(args.round_num))
        ].copy()
        if pred_round.empty:
            print(f"[WARN] {lg_upper} round={args.round_num} の predictions 抽出が0件")

        buy_round = df_buy_all[df_buy_all["league_norm"] == lg_upper].copy()
        if "match_no" in pred_round.columns and "match_no" in buy_round.columns and not pred_round.empty:
            mnos = set(pd.to_numeric(pred_round["match_no"], errors="coerce").dropna().astype(int))
            buy_round = buy_round[pd.to_numeric(buy_round["match_no"], errors="coerce").astype("Int64").isin(mnos)]
        elif {"home_team", "away_team"}.issubset(buy_round.columns) and {"home_team", "away_team"}.issubset(pred_round.columns):
            key = set(zip(pred_round["home_team"].astype(str), pred_round["away_team"].astype(str)))
            pair_series = list(zip(buy_round["home_team"].astype(str), buy_round["away_team"].astype(str)))
            buy_round = buy_round[[p in key for p in pair_series]]

        pred_round_csv = os.path.join(out_dir, "predictions.round.csv")
        buy_round_csv = os.path.join(out_dir, "buyplan.round.csv")
        pred_round.to_csv(pred_round_csv, index=False, encoding="utf-8-sig")
        buy_round.to_csv(buy_round_csv, index=False, encoding="utf-8-sig")
        saved_paths.extend([pred_round_csv, buy_round_csv])

        # 4) evaluation
        eval_obj, pred_eval = build_evaluation(pred_round, buy_round, actual_round)
        eval_obj.update(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "season": args.season,
                "league": lg_upper,
                "round": args.round_num,
                "round_label": round_label,
                "source_files": {
                    "result_source": result_src_path,
                    "predictions_source": predictions_path,
                    "buyplan_source": buyplan_path,
                },
                "counts": {
                    "actual_round_rows": int(len(actual_round)),
                    "pred_round_rows": int(len(pred_round)),
                    "buyplan_round_rows": int(len(buy_round)),
                },
            }
        )
        eval_json = os.path.join(out_dir, "evaluation.json")
        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(eval_obj, f, ensure_ascii=False, indent=2)
        saved_paths.append(eval_json)

        eval_csv = os.path.join(out_dir, "evaluation_detail.csv")
        pred_eval.to_csv(eval_csv, index=False, encoding="utf-8-sig")
        saved_paths.append(eval_csv)

    print("\n=== SAVED FILES ===")
    for p in saved_paths:
        print(p)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
