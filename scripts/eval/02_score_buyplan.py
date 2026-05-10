#!/usr/bin/env python3
import argparse
import os
import re
import sys

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def eval_base_dir(round_id: str) -> str:
    if str(round_id).startswith("toto"):
        return os.path.join(ROOT_DIR, "data", "eval", "toto_rounds", round_id)
    return os.path.join(ROOT_DIR, "data", "eval", "rounds", round_id)


def parse_args():
    p = argparse.ArgumentParser(description="buyplan採点")
    p.add_argument("--round", required=True, help="round02 / toto1608")
    p.add_argument("--buyplan", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/snapshot/buyplan.csv")
    p.add_argument("--actual", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/actual_results.csv")
    p.add_argument("--out", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/evaluation.csv")
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


def normalize_result_series(series):
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    ok = vals.isin([0, 1, 2])
    out.loc[ok] = vals.loc[ok].astype(int).astype(str)
    return out


def main():
    args = parse_args()
    round_id = args.round
    base_dir = eval_base_dir(round_id)
    buyplan_path = args.buyplan or os.path.join(base_dir, "snapshot", "buyplan.csv")
    actual_path = args.actual or os.path.join(base_dir, "actual_results.csv")
    out_csv = args.out or os.path.join(base_dir, "evaluation.csv")

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
    score_df = buy.merge(actual[["match_no", "result"]], on="match_no", how="left")
    if len(score_df) != len(buy):
        raise RuntimeError(f"採点対象試合数が buyplan と一致しません: buy={len(buy)} scored={len(score_df)}")
    score_df["result"] = normalize_result_series(score_df["result"])
    resolved_mask = score_df["result"].isin(["0", "1", "2"])
    resolved_df = score_df[resolved_mask].copy()

    rows = []
    score_map = {}
    total = len(resolved_df)
    actual_mark = resolved_df["result"].astype(str).str.strip()
    actual_draw_mask = actual_mark == "0"
    actual_draw_total = int(actual_draw_mask.sum())
    for idx, col in cand_cols:
        cid = f"cand{idx:02d}"
        pred = resolved_df[col].astype(str).str.strip()
        ok = pred == actual_mark
        hits = int(ok.sum())
        miss_nos = resolved_df.loc[~ok, "match_no"].astype(str).tolist()
        pred_draw_mask = pred == "0"
        draw_pred_count = int(pred_draw_mask.sum())
        draw_hit_count = int((pred_draw_mask & actual_draw_mask).sum())
        draw_precision = (draw_hit_count / draw_pred_count) if draw_pred_count else 0.0
        draw_recall = (draw_hit_count / actual_draw_total) if actual_draw_total else 0.0
        score_map[cid] = hits
        rows.append(
            {
                "round_id": round_id,
                "candidate_id": cid,
                "hits": hits,
                "total": int(total),
                "hit_rate": hits / total if total else 0.0,
                "actual_draw_total": actual_draw_total,
                "draw_pred_count": draw_pred_count,
                "draw_hit_count": draw_hit_count,
                "draw_precision": draw_precision,
                "draw_recall": draw_recall,
                "resolved_total": int(total),
                "pending_total": int(len(score_df) - total),
                "miss_match_nos": " ".join(miss_nos),
            }
        )

    ev = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ev.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if total == len(score_df):
        upsert_history(args.history, round_id, score_map, total)
        print(f"[OK] history updated: {args.history}")
    else:
        print(
            f"[WARN] 未解決試合があるため history 更新をスキップ: resolved={total} total={len(score_df)}"
        )

    print(f"[OK] evaluation saved: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
