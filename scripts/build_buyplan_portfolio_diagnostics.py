#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


TICKET_RE = re.compile(r"^ticket\d+$")


def _load_buyplan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ticket_cols = [c for c in df.columns if TICKET_RE.match(c)]
    if not ticket_cols:
        raise ValueError("no ticket columns found")
    df["match_no"] = pd.to_numeric(df["match_no"], errors="coerce").astype("Int64")
    for col in ticket_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _load_actual(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["match_no"] = pd.to_numeric(df["match_no"], errors="coerce").astype("Int64")
    df["actual_result"] = pd.to_numeric(df["result"], errors="coerce").astype("Int64")
    return df[["match_no", "result", "actual_result"]]


def _safe_label(value: object) -> str:
    if pd.isna(value):
        return ""
    ivalue = int(value)
    return {0: "D", 1: "H", 2: "A"}.get(ivalue, str(ivalue))


def _ticket_ids(ticket_cols: list[str]) -> list[str]:
    return [f"cand{int(col.replace('ticket', '')):02d}" for col in ticket_cols]


def build_diagnostics(buyplan_csv: Path, actual_csv: Path, outdir: Path, suffix: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    buyplan = _load_buyplan(buyplan_csv)
    actual = _load_actual(actual_csv)
    df = buyplan.merge(actual, on="match_no", how="left")
    df = df[df["actual_result"].isin([0, 1, 2])].copy()

    ticket_cols = [c for c in df.columns if TICKET_RE.match(c)]
    ticket_ids = _ticket_ids(ticket_cols)
    ticket_map = dict(zip(ticket_cols, ticket_ids))

    hit_cols: list[str] = []
    for col in ticket_cols:
        hit_col = f"{col}_hit"
        df[hit_col] = (df[col] == df["actual_result"]).astype(int)
        hit_cols.append(hit_col)

    df["portfolio_hit_count"] = df[hit_cols].sum(axis=1)
    df["portfolio_hit_any"] = (df["portfolio_hit_count"] > 0).astype(int)
    df["all_miss"] = (df["portfolio_hit_count"] == 0).astype(int)
    df["same_symbol_count"] = df[ticket_cols].nunique(axis=1, dropna=True)
    df["all_same_symbol"] = (df["same_symbol_count"] == 1).astype(int)
    df["symbols"] = df[ticket_cols].apply(
        lambda row: "".join(sorted({_safe_label(v) for v in row if not pd.isna(v)})),
        axis=1,
    )
    df["actual_label"] = df["actual_result"].map({0: "D", 1: "H", 2: "A"})
    df["away_lock_fail"] = ((df["all_same_symbol"] == 1) & (df[ticket_cols[0]] == 2) & (df["actual_result"] != 2)).astype(int)
    df["draw_lock_fail"] = ((df["all_same_symbol"] == 1) & (df[ticket_cols[0]] == 0) & (df["actual_result"] != 0)).astype(int)
    df["home_lock_fail"] = ((df["all_same_symbol"] == 1) & (df[ticket_cols[0]] == 1) & (df["actual_result"] != 1)).astype(int)

    match_cols = [
        "match_no",
        "league",
        "home_team",
        "away_team",
        "actual_label",
        "portfolio_hit_count",
        "portfolio_hit_any",
        "all_miss",
        "all_same_symbol",
        "symbols",
        "shape_cluster_label",
        "shape_overlap_subtype",
        "lab_cluster_label",
        "lab_overlap_subtype",
        "lab_basis_hint",
        "context_risk_level",
        "context_ticket_guidance",
        "topology_asymmetry_score",
        "admission_draw_score",
        "admission_escape_draw_score",
        "admission_escape_home_score",
        "admission_escape_away_score",
        "p_home",
        "p_draw",
        "p_away",
    ] + ticket_cols
    match_df = df[match_cols].copy()

    pairwise_rows: list[dict[str, object]] = []
    for i, col_a in enumerate(ticket_cols):
        for col_b in ticket_cols[i + 1 :]:
            same_pick = (df[col_a] == df[col_b]).astype(int)
            same_hit = ((df[f"{col_a}_hit"] == 1) & (df[f"{col_b}_hit"] == 1)).astype(int)
            same_miss = ((df[f"{col_a}_hit"] == 0) & (df[f"{col_b}_hit"] == 0)).astype(int)
            rescue_a = ((df[f"{col_a}_hit"] == 1) & (df[f"{col_b}_hit"] == 0)).astype(int)
            rescue_b = ((df[f"{col_a}_hit"] == 0) & (df[f"{col_b}_hit"] == 1)).astype(int)
            phi = pd.Series(df[f"{col_a}_hit"]).corr(pd.Series(df[f"{col_b}_hit"]))
            pairwise_rows.append(
                {
                    "ticket_a": ticket_map[col_a],
                    "ticket_b": ticket_map[col_b],
                    "same_pick_rate": float(same_pick.mean()),
                    "same_hit_count": int(same_hit.sum()),
                    "same_miss_count": int(same_miss.sum()),
                    "correlated_failure_rate": float(same_miss.mean()),
                    "rescue_a_only": int(rescue_a.sum()),
                    "rescue_b_only": int(rescue_b.sum()),
                    "hit_miss_phi": 0.0 if pd.isna(phi) else float(phi),
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)

    ticket_rows: list[dict[str, object]] = []
    best_hit = max(int(df[f"{col}_hit"].sum()) for col in ticket_cols)
    best_cols = [col for col in ticket_cols if int(df[f"{col}_hit"].sum()) == best_hit]
    best_ref_col = best_cols[0]
    for col in ticket_cols:
        hit_col = f"{col}_hit"
        other_hit_cols = [f"{other}_hit" for other in ticket_cols if other != col]
        unique_hits = ((df[hit_col] == 1) & (df[other_hit_cols].sum(axis=1) == 0)).sum()
        with_all = int(df["portfolio_hit_any"].sum())
        without_this = int((df[other_hit_cols].sum(axis=1) > 0).sum())
        draw_pred = int((df[col] == 0).sum())
        draw_hit = int(((df[col] == 0) & (df["actual_result"] == 0)).sum())
        rescue_vs_base = int(((df[hit_col] == 1) & (df["ticket01_hit"] == 0)).sum())
        same_as_best = int((df[col] == df[best_ref_col]).sum())
        ticket_rows.append(
            {
                "ticket": ticket_map[col],
                "hits": int(df[hit_col].sum()),
                "hit_rate": float(df[hit_col].mean()),
                "unique_hits_vs_portfolio": int(unique_hits),
                "leave_one_out_cover_drop": int(with_all - without_this),
                "rescue_vs_base_ticket01": rescue_vs_base,
                "draw_pred_count": draw_pred,
                "draw_hit_count": draw_hit,
                "same_as_best_ticket": same_as_best,
            }
        )
    ticket_df = pd.DataFrame(ticket_rows).sort_values(["hits", "ticket"], ascending=[False, True])

    cluster_rows: list[dict[str, object]] = []
    for cluster_col, overlap_col in [
        ("shape_cluster_label", "shape_overlap_subtype"),
        ("lab_cluster_label", "lab_overlap_subtype"),
    ]:
        group = (
            df.groupby([cluster_col, overlap_col], dropna=False)
            .agg(
                matches=("match_no", "count"),
                all_same_matches=("all_same_symbol", "sum"),
                all_miss_matches=("all_miss", "sum"),
                portfolio_hit_rate=("portfolio_hit_any", "mean"),
                avg_topology_asymmetry=("topology_asymmetry_score", "mean"),
                avg_admission_draw=("admission_draw_score", "mean"),
                avg_escape_draw=("admission_escape_draw_score", "mean"),
                avg_escape_home=("admission_escape_home_score", "mean"),
                avg_escape_away=("admission_escape_away_score", "mean"),
                avg_p_draw=("p_draw", "mean"),
            )
            .reset_index()
        )
        group.insert(0, "cluster_type", cluster_col.replace("_cluster_label", ""))
        group = group.rename(columns={cluster_col: "cluster_label", overlap_col: "overlap_subtype"})
        cluster_rows.append(group)
    cluster_df = pd.concat(cluster_rows, ignore_index=True)

    summary_lines = [
        "# BuyPlan Portfolio Diagnostics Current",
        "",
        f"- resolved matches: {len(df)}",
        f"- portfolio cover hits: {int(df['portfolio_hit_any'].sum())}/{len(df)}",
        f"- all-miss matches: {int(df['all_miss'].sum())}",
        f"- all-same-symbol matches: {int(df['all_same_symbol'].sum())}",
        f"- away-lock failures: {int(df['away_lock_fail'].sum())}",
        f"- draw-lock failures: {int(df['draw_lock_fail'].sum())}",
        f"- home-lock failures: {int(df['home_lock_fail'].sum())}",
        "",
        "## Top Findings",
        "",
    ]
    for row in ticket_df.head(5).itertuples(index=False):
        summary_lines.append(
            f"- {row.ticket}: hits={row.hits}, unique_hits={row.unique_hits_vs_portfolio}, "
            f"leave_one_out_drop={row.leave_one_out_cover_drop}, rescue_vs_base={row.rescue_vs_base_ticket01}"
        )

    summary_lines.extend(["", "## All-Miss Matches", ""])
    for row in match_df[match_df["all_miss"] == 1].itertuples(index=False):
        summary_lines.append(
            f"- M{int(row.match_no):02d} {row.home_team}-{row.away_team}: "
            f"all_same={bool(row.all_same_symbol)}, symbols={row.symbols}, actual={row.actual_label}, "
            f"shape={row.shape_cluster_label}, basis={row.lab_basis_hint}"
        )

    summary_path = outdir / f"buyplan_portfolio_diagnostics_{suffix}.md"
    pairwise_path = outdir / f"buyplan_portfolio_pairwise_{suffix}.csv"
    ticket_path = outdir / f"buyplan_portfolio_ticket_marginal_{suffix}.csv"
    match_path = outdir / f"buyplan_portfolio_match_diagnostics_{suffix}.csv"
    cluster_path = outdir / f"buyplan_portfolio_cluster_coverage_{suffix}.csv"

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    pairwise_df.to_csv(pairwise_path, index=False)
    ticket_df.to_csv(ticket_path, index=False)
    match_df.to_csv(match_path, index=False)
    cluster_df.to_csv(cluster_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buyplan", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--suffix", default="current")
    args = parser.parse_args()

    build_diagnostics(
        buyplan_csv=Path(args.buyplan),
        actual_csv=Path(args.actual),
        outdir=Path(args.outdir),
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
