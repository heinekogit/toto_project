#!/usr/bin/env python3
import argparse
import os
import re
import sys

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    p = argparse.ArgumentParser(description="buyplan採点")
    p.add_argument("--round", required=True, help="round02")
    p.add_argument("--buyplan", default=None, help="既定: data/eval/rounds/{round}/snapshot/buyplan.csv")
    p.add_argument("--actual", default=None, help="既定: data/eval/rounds/{round}/actual_results.csv")
    p.add_argument("--out", default=None, help="既定: data/eval/rounds/{round}/evaluation.csv")
    p.add_argument("--history", default=os.path.join(ROOT_DIR, "data", "eval", "candidate_history.csv"))
    return p.parse_args()


def detect_candidate_columns(df):
    found = []
    for c in df.columns:
        s = str(c)
        m = re.search(r"(?:ticket|候補)\s*0*(10|[1-9])", s, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            found.append((idx, c))
    found = sorted(set(found), key=lambda x: x[0])
    if not found:
        raise RuntimeError("buyplan.csv から候補列を検出できません")
    return found


def upsert_history(history_path, round_id, scores_map, total):
    cols = ["round_id"] + [f"cand{i:02d}" for i in range(1, 11)] + ["total"]
    if os.path.exists(history_path):
        h = pd.read_csv(history_path)
    else:
        h = pd.DataFrame(columns=cols)

    row = {"round_id": round_id, "total": int(total)}
    for i in range(1, 11):
        row[f"cand{i:02d}"] = int(scores_map.get(f"cand{i:02d}", 0))

    if "round_id" in h.columns and (h["round_id"] == round_id).any():
        h = h[h["round_id"] != round_id].copy()
    h = pd.concat([h, pd.DataFrame([row])], ignore_index=True)
    h = h.sort_values("round_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    h.to_csv(history_path, index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    round_id = args.round
    buyplan_path = args.buyplan or os.path.join(ROOT_DIR, "data", "eval", "rounds", round_id, "snapshot", "buyplan.csv")
    actual_path = args.actual or os.path.join(ROOT_DIR, "data", "eval", "rounds", round_id, "actual_results.csv")
    out_csv = args.out or os.path.join(ROOT_DIR, "data", "eval", "rounds", round_id, "evaluation.csv")

    if not os.path.exists(buyplan_path):
        raise FileNotFoundError(f"buyplan.csv not found: {buyplan_path}")
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"actual_results.csv not found: {actual_path}")

    buy = pd.read_csv(buyplan_path)
    actual = pd.read_csv(actual_path)

    for col in ["match_no", "home_team", "away_team"]:
        if col not in buy.columns:
            raise ValueError(f"buyplan.csv 必須列不足: {col}")
    for col in ["match_no", "result"]:
        if col not in actual.columns:
            raise ValueError(f"actual_results.csv 必須列不足: {col}")

    cand_cols = detect_candidate_columns(buy)
    score_df = buy.merge(actual[["match_no", "result"]], on="match_no", how="inner")
    if len(score_df) != 13:
        raise RuntimeError(f"採点対象試合が13ではありません: {len(score_df)}")

    rows = []
    score_map = {}
    total = len(score_df)
    for idx, col in cand_cols:
        cid = f"cand{idx:02d}"
        pred = score_df[col].astype(str).str.strip()
        actual_mark = score_df["result"].astype(str).str.strip()
        ok = pred == actual_mark
        hits = int(ok.sum())
        miss_nos = score_df.loc[~ok, "match_no"].astype(str).tolist()
        score_map[cid] = hits
        rows.append(
            {
                "round_id": round_id,
                "candidate_id": cid,
                "hits": hits,
                "total": int(total),
                "hit_rate": hits / total if total else 0.0,
                "miss_match_nos": " ".join(miss_nos),
            }
        )

    ev = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ev.to_csv(out_csv, index=False, encoding="utf-8-sig")

    upsert_history(args.history, round_id, score_map, total)

    print(f"[OK] evaluation saved: {out_csv}")
    print(f"[OK] history updated: {args.history}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
