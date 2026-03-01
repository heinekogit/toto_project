#!/usr/bin/env python3
import argparse
import html
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


EPS = 1e-15

LABELS = ["H", "D", "A"]
PROB_KEYS = ["prob_home", "prob_draw", "prob_away"]
PRIORITY = {"H": 0, "A": 1, "D": 2}  # tie-break: H -> A -> D

CANDIDATES = {
    "prob_home": ["prob_home_win", "prob_home", "p_home", "home_prob"],
    "prob_draw": ["prob_draw", "p_draw", "draw_prob"],
    "prob_away": ["prob_away_win", "prob_away", "p_away", "away_prob"],
    "elo_diff": ["elo_diff_for_prob", "elo_diff", "elo_diff_raw", "home_advantage_diff"],
    "match_id": ["match_id", "id", "game_id", "fixture_id"],
    "round": ["round", "round_id", "節", "section", "matchday"],
    "actual": ["actual_result", "actual", "result", "final_result"],
}


def resolve_column(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def normalize_label(v) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    if s in {"H", "1", "HOME", "HOME_WIN"}:
        return "H"
    if s in {"D", "0", "DRAW", "X"}:
        return "D"
    if s in {"A", "2", "AWAY", "AWAY_WIN"}:
        return "A"
    return None


def argmax_label(p_h: float, p_d: float, p_a: float) -> str:
    if p_h >= p_d and p_h >= p_a:
        return "H"
    if p_a >= p_h and p_a >= p_d:
        return "A"
    return "D"


def rank_labels(p_h: float, p_d: float, p_a: float):
    arr = [("H", p_h), ("D", p_d), ("A", p_a)]
    arr = sorted(arr, key=lambda x: (-x[1], PRIORITY[x[0]]))
    return arr


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if int(m.sum()) < 2:
        return np.nan
    a2 = a[m]
    b2 = b[m]
    if float(a2.std(ddof=0)) == 0.0 or float(b2.std(ddof=0)) == 0.0:
        return np.nan
    return float(a2.corr(b2))


def compute_brier_3class(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    y_h = (df["actual_norm"] == "H").astype(float)
    y_d = (df["actual_norm"] == "D").astype(float)
    y_a = (df["actual_norm"] == "A").astype(float)
    v = (
        (df["prob_home_win"] - y_h) ** 2
        + (df["prob_draw"] - y_d) ** 2
        + (df["prob_away_win"] - y_a) ** 2
    )
    return float(v.mean())


def format_pct(v):
    if pd.isna(v):
        return "-"
    return f"{float(v) * 100:.1f}%"


def build_html_report(df_match: pd.DataFrame, df_round: pd.DataFrame, out_path: str):
    overall = df_round[df_round["round_key"] == "__ALL__"]
    if overall.empty:
        overall_row = None
    else:
        overall_row = overall.iloc[0]

    round_rows = df_round[df_round["round_key"] != "__ALL__"].copy()

    warnings = []
    if overall_row is not None:
        c = overall_row.get("corr_eloabs_pdraw", np.nan)
        if pd.notna(c) and float(c) >= 0:
            warnings.append(f"corr_eloabs_pdraw が非負です（{float(c):.4f}）")

    low_conf = round_rows.sort_values("avg_maxP", ascending=True).head(5)
    high_conf = round_rows.sort_values("avg_maxP", ascending=False).head(5)
    high_close = round_rows.sort_values("close_rate", ascending=False).head(5)

    surprise_rows = pd.DataFrame()
    if "surprise" in df_match.columns:
        surprise_rows = df_match[df_match["surprise"].notna()].sort_values("surprise", ascending=False).head(20)

    html_parts = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html lang='ja'><head><meta charset='utf-8'>")
    html_parts.append("<title>Diagnostics Report</title>")
    html_parts.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;margin:24px;color:#222;}"
        "table{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:18px;}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;}"
        "th{background:#f5f5f5;}"
        ".warn{background:#fff3cd;border:1px solid #ffe69c;padding:10px;margin:10px 0;}"
        ".ok{background:#eef9f0;border:1px solid #b8e2c0;padding:10px;margin:10px 0;}"
        ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}"
        "</style></head><body>"
    )
    html_parts.append("<h2>予想診断レポート</h2>")

    if overall_row is None:
        html_parts.append("<div class='warn'>全体サマリを計算できませんでした。</div>")
    else:
        html_parts.append("<h3>全体サマリ</h3>")
        html_parts.append("<table><tbody>")
        pairs = [
            ("n_matches", overall_row.get("n_matches")),
            ("acc", format_pct(overall_row.get("acc"))),
            ("brier", overall_row.get("brier")),
            ("logloss", overall_row.get("logloss")),
            ("avg_maxP", overall_row.get("avg_maxP")),
            ("avg_margin12", overall_row.get("avg_margin12")),
            ("close_rate", format_pct(overall_row.get("close_rate"))),
            ("pred_rate_H/D/A", f"{format_pct(overall_row.get('pred_rate_H'))} / {format_pct(overall_row.get('pred_rate_D'))} / {format_pct(overall_row.get('pred_rate_A'))}"),
            ("actual_rate_H/D/A", f"{format_pct(overall_row.get('actual_rate_H'))} / {format_pct(overall_row.get('actual_rate_D'))} / {format_pct(overall_row.get('actual_rate_A'))}"),
            ("corr_eloabs_pdraw", overall_row.get("corr_eloabs_pdraw")),
            ("expected_correct_proxy", overall_row.get("expected_correct_proxy")),
            ("delta_hits_proxy", overall_row.get("delta_hits_proxy")),
        ]
        for k, v in pairs:
            html_parts.append(f"<tr><th>{html.escape(str(k))}</th><td class='mono'>{html.escape(str(v))}</td></tr>")
        html_parts.append("</tbody></table>")

    html_parts.append("<h3>異常検知</h3>")
    if warnings:
        for w in warnings:
            html_parts.append(f"<div class='warn'>WARN: {html.escape(w)}</div>")
    else:
        html_parts.append("<div class='ok'>WARN 条件は検出されませんでした。</div>")

    html_parts.append("<h4>avg_maxP 低い節（不確実性高）</h4>")
    html_parts.append(low_conf.to_html(index=False, border=0, classes="tbl", escape=True))
    html_parts.append("<h4>avg_maxP 高い節（過信候補）</h4>")
    html_parts.append(high_conf.to_html(index=False, border=0, classes="tbl", escape=True))
    html_parts.append("<h4>close_rate 高い節（拮抗だらけ）</h4>")
    html_parts.append(high_close.to_html(index=False, border=0, classes="tbl", escape=True))

    if not surprise_rows.empty:
        cols = [c for c in ["round_key", "match_id", "pred_label", "actual_norm", "p_actual", "surprise", "maxP", "margin12"] if c in surprise_rows.columns]
        html_parts.append("<h4>surprise 上位試合</h4>")
        html_parts.append(surprise_rows[cols].to_html(index=False, border=0, classes="tbl", escape=True))

    html_parts.append("<h3>節別テーブル</h3>")
    html_parts.append(round_rows.to_html(index=False, border=0, classes="tbl", escape=True))
    html_parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def main():
    ap = argparse.ArgumentParser(description="Prediction diagnostics from a predictions/backtest CSV")
    ap.add_argument("--input", required=True, help="input csv path")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--close-margin", type=float, default=0.03)
    args = ap.parse_args()

    in_csv = os.path.abspath(args.input)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(in_csv)

    col_map = {
        "prob_home": resolve_column(df, CANDIDATES["prob_home"]),
        "prob_draw": resolve_column(df, CANDIDATES["prob_draw"]),
        "prob_away": resolve_column(df, CANDIDATES["prob_away"]),
        "elo_diff": resolve_column(df, CANDIDATES["elo_diff"]),
        "match_id": resolve_column(df, CANDIDATES["match_id"]),
        "round": resolve_column(df, CANDIDATES["round"]),
        "actual": resolve_column(df, CANDIDATES["actual"]),
    }

    missing_probs = [k for k in ["prob_home", "prob_draw", "prob_away"] if col_map[k] is None]
    if missing_probs:
        raise RuntimeError(f"必須確率列が見つかりません: {missing_probs}")

    out = df.copy()
    out["prob_home_win"] = pd.to_numeric(out[col_map["prob_home"]], errors="coerce")
    out["prob_draw"] = pd.to_numeric(out[col_map["prob_draw"]], errors="coerce")
    out["prob_away_win"] = pd.to_numeric(out[col_map["prob_away"]], errors="coerce")
    out["elo_diff_for_prob"] = pd.to_numeric(out[col_map["elo_diff"]], errors="coerce") if col_map["elo_diff"] else np.nan
    out["match_id"] = out[col_map["match_id"]] if col_map["match_id"] else np.arange(1, len(out) + 1)
    out["round_key"] = out[col_map["round"]].astype(str) if col_map["round"] else "__ALL__"

    ranked = out.apply(
        lambda r: rank_labels(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])
        if pd.notna(r["prob_home_win"]) and pd.notna(r["prob_draw"]) and pd.notna(r["prob_away_win"])
        else [("H", np.nan), ("D", np.nan), ("A", np.nan)],
        axis=1,
    )
    out["P1_label"] = ranked.map(lambda x: x[0][0])
    out["P2_label"] = ranked.map(lambda x: x[1][0])
    out["P3_label"] = ranked.map(lambda x: x[2][0])
    out["P1_prob"] = ranked.map(lambda x: x[0][1])
    out["P2_prob"] = ranked.map(lambda x: x[1][1])
    out["P3_prob"] = ranked.map(lambda x: x[2][1])
    out["maxP"] = out[["prob_home_win", "prob_draw", "prob_away_win"]].max(axis=1)
    out["margin12"] = out["P1_prob"] - out["P2_prob"]
    out["is_close"] = out["margin12"] < float(args.close_margin)
    out["pred_label"] = out.apply(
        lambda r: argmax_label(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])
        if pd.notna(r["prob_home_win"]) and pd.notna(r["prob_draw"]) and pd.notna(r["prob_away_win"])
        else None,
        axis=1,
    )
    out["elo_abs"] = out["elo_diff_for_prob"].abs()
    out["p_draw"] = out["prob_draw"]

    if col_map["actual"]:
        out["actual_norm"] = out[col_map["actual"]].map(normalize_label)
        out["is_correct"] = (out["pred_label"] == out["actual_norm"]).astype(float)

        def _p_actual(r):
            if r["actual_norm"] == "H":
                return r["prob_home_win"]
            if r["actual_norm"] == "D":
                return r["prob_draw"]
            if r["actual_norm"] == "A":
                return r["prob_away_win"]
            return np.nan

        out["p_actual"] = out.apply(_p_actual, axis=1)
        out["surprise"] = -np.log(out["p_actual"].clip(EPS, 1.0))
    else:
        out["actual_norm"] = None
        out["is_correct"] = np.nan
        out["p_actual"] = np.nan
        out["surprise"] = np.nan

    group_keys = ["__ALL__"] if col_map["round"] is None else ["__ALL__"] + list(pd.unique(out["round_key"]))
    round_rows: List[Dict[str, object]] = []
    for g in group_keys:
        sub = out if g == "__ALL__" else out[out["round_key"] == g]
        n = int(len(sub))
        row: Dict[str, object] = {"round_key": g, "n_matches": n}
        row["avg_maxP"] = float(sub["maxP"].mean()) if n else np.nan
        row["avg_margin12"] = float(sub["margin12"].mean()) if n else np.nan
        row["close_rate"] = float(sub["is_close"].mean()) if n else np.nan
        row["pred_rate_H"] = float((sub["pred_label"] == "H").mean()) if n else np.nan
        row["pred_rate_D"] = float((sub["pred_label"] == "D").mean()) if n else np.nan
        row["pred_rate_A"] = float((sub["pred_label"] == "A").mean()) if n else np.nan
        row["avg_p_draw"] = float(sub["p_draw"].mean()) if n else np.nan
        row["corr_eloabs_pdraw"] = safe_corr(sub["elo_abs"], sub["p_draw"])

        valid_actual = sub[sub["actual_norm"].isin(LABELS)].copy()
        if valid_actual.empty:
            row["acc"] = np.nan
            row["actual_rate_H"] = np.nan
            row["actual_rate_D"] = np.nan
            row["actual_rate_A"] = np.nan
            row["brier"] = np.nan
            row["logloss"] = np.nan
            row["expected_correct_proxy"] = np.nan
            row["delta_hits_proxy"] = np.nan
        else:
            row["acc"] = float((valid_actual["pred_label"] == valid_actual["actual_norm"]).mean())
            row["actual_rate_H"] = float((valid_actual["actual_norm"] == "H").mean())
            row["actual_rate_D"] = float((valid_actual["actual_norm"] == "D").mean())
            row["actual_rate_A"] = float((valid_actual["actual_norm"] == "A").mean())
            row["brier"] = compute_brier_3class(valid_actual)
            row["logloss"] = float((-np.log(valid_actual["p_actual"].clip(EPS, 1.0))).mean())
            row["expected_correct_proxy"] = float(valid_actual["maxP"].sum() / len(valid_actual))
            row["delta_hits_proxy"] = float((valid_actual["pred_label"] == valid_actual["actual_norm"]).sum() - valid_actual["maxP"].sum())
        round_rows.append(row)

    df_round = pd.DataFrame(round_rows)

    match_out = os.path.join(outdir, "diagnostics_matches.csv")
    round_out = os.path.join(outdir, "diagnostics_rounds.csv")
    html_out = os.path.join(outdir, "diagnostics_report.html")

    out.to_csv(match_out, index=False, encoding="utf-8-sig")
    df_round.to_csv(round_out, index=False, encoding="utf-8-sig")
    build_html_report(out, df_round, html_out)

    print("[DONE] diagnostics generated")
    print(f"  input: {in_csv}")
    print(f"  matches: {match_out}")
    print(f"  rounds: {round_out}")
    print(f"  html: {html_out}")
    print("[USED_COLUMNS]")
    for k, v in col_map.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
