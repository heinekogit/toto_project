#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "data", "reports")
if REPORTS_DIR not in sys.path:
    sys.path.insert(0, REPORTS_DIR)

import pandas as pd

import buyplan
from report_view_utils import write_html_table


DEFAULT_ROUNDS_DIR = os.path.join(ROOT_DIR, "data", "eval", "rounds")
DEFAULT_OUT_ROOT = os.path.join(ROOT_DIR, "data", "purchase_reference", "backtest")
DEFAULT_EXPERIMENT_OUT_ROOT = os.path.join(ROOT_DIR, "data", "purchase_reference", "current_predictions_experiment")
DEFAULT_CURRENT_PREDICTIONS = os.path.join(ROOT_DIR, "data", "purchase_reference", "predictions.csv")
ROLE_FAMILY_TICKETS = {
    "ticket02",
    "ticket03",
    "ticket05",
    "ticket06",
    "ticket07",
    "ticket08",
    "ticket09",
    "ticket10",
}
ROUND_SUMMARY_DESC_MAP = {
    "round": "節ID",
    "topology_mode": "節topology",
    "topology_draw_cluster_count": "draw cluster数",
    "topology_swing_cluster_count": "swing cluster数",
    "topology_hold_cluster_count": "hold cluster数",
    "ticket01_hits": "本線ticket01の的中数",
    "ticket01_total": "試合数",
    "ticket01_hit_rate": "本線ticket01の的中率",
    "best_ticket": "この節の最高hit ticket",
    "best_hits": "最高hit ticketの的中数",
    "best_hit_rate": "最高hit ticketの的中率",
    "unique_ticket_count": "最終券面のユニーク枚数",
    "duplicate_count": "重複枚数",
    "avg_ticket_distance": "券面どうしの平均距離",
    "draw_family_ticket_count": "draw familyのslot数",
    "draw_family_hit_rate": "draw family合算的中率",
    "draw_family_base_net_gain": "draw familyの対ticket01 net gain",
    "swing_family_ticket_count": "swing familyのslot数",
    "swing_family_hit_rate": "swing family合算的中率",
    "swing_family_base_net_gain": "swing familyの対ticket01 net gain",
}
ROUND_DASHBOARD_DESC_MAP = {
    "round": "節ID",
    "topology_mode": "節topology",
    "ticket01_hit_rate": "本線ticket01の的中率",
    "best_ticket": "この節の最高hit ticket",
    "best_hit_rate": "最高hit ticketの的中率",
    "draw_family_base_net_gain": "draw familyの対ticket01 net gain",
    "swing_family_base_net_gain": "swing familyの対ticket01 net gain",
    "unique_ticket_count": "最終券面のユニーク枚数",
    "duplicate_count": "重複枚数",
    "dashboard_note": "運用向けの一言判定",
}


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
        ticket_stats=ticket_stats,
        actual_csv_path=actual_csv,
    )
    return scored_html


def _ticket_distance(a: List[str], b: List[str]) -> int:
    return sum(1 for x, y in zip(a, b) if str(x) != str(y))


def _load_round_ticket_stats(round_name: str, out_root: str) -> Dict[str, float]:
    buyplan_csv = os.path.join(out_root, round_name, "buyplan.csv")
    if not os.path.exists(buyplan_csv):
        return {
            "unique_ticket_count": 0,
            "duplicate_count": 0,
            "avg_ticket_distance": 0.0,
        }
    df = pd.read_csv(buyplan_csv)
    ticket_cols = [f"ticket{i:02d}" for i in range(1, 11) if f"ticket{i:02d}" in df.columns]
    if not ticket_cols:
        return {
            "unique_ticket_count": 0,
            "duplicate_count": 0,
            "avg_ticket_distance": 0.0,
        }
    tickets = [df[col].fillna("").astype(str).tolist() for col in ticket_cols]
    unique_count = len({tuple(ticket) for ticket in tickets})
    pairwise = [
        _ticket_distance(tickets[i], tickets[j])
        for i in range(len(tickets))
        for j in range(i + 1, len(tickets))
    ]
    return {
        "unique_ticket_count": unique_count,
        "duplicate_count": max(0, len(tickets) - unique_count),
        "avg_ticket_distance": round(float(sum(pairwise)) / float(len(pairwise)), 3) if pairwise else 0.0,
    }


def _round_dashboard_note(row: Dict[str, object]) -> str:
    topology = str(row.get("topology_mode", "unknown") or "unknown")
    best_ticket = str(row.get("best_ticket", ""))
    draw_net = float(row.get("draw_family_base_net_gain", 0) or 0)
    swing_net = float(row.get("swing_family_base_net_gain", 0) or 0)
    duplicate_count = int(row.get("duplicate_count", 0) or 0)

    if duplicate_count > 0:
        return "重複あり"
    if topology == "swing_heavy":
        if swing_net >= 2:
            return "swing familyが効いた節"
        if draw_net > swing_net:
            return "draw family寄りの戻し節"
        return "swing優勢だが伸び弱め"
    if topology == "balanced":
        if draw_net <= 0 and swing_net <= 0:
            return "base依存が強い節"
        if draw_net > swing_net:
            return "draw familyが相対優勢"
        if swing_net > draw_net:
            return "swing familyが相対優勢"
        return "family差が小さい節"
    if topology == "draw_heavy":
        if draw_net >= 2:
            return "draw familyが効いた節"
        return "draw heavyだが差は限定"
    if best_ticket == "ticket01":
        return "base優位"
    return "要確認"


def _write_role_summary(rounds: List[str], out_root: str) -> tuple[str, str, str, str, str, str, str, str, str]:
    rows: List[Dict[str, object]] = []
    agg: Dict[str, Dict[str, float]] = {}
    required_cols = {
        "ticket",
        "hits",
        "total",
        "hit_rate",
        "draw_count",
        "sway_count",
        "role_family",
        "role_mode",
        "role_unique_only",
        "role_same_family",
        "role_diff_family",
        "topology_mode",
        "topology_draw_cluster_count",
        "topology_swing_cluster_count",
        "topology_hold_cluster_count",
    }
    for round_name in rounds:
        summary_csv = os.path.join(out_root, round_name, "buyplan_scored_summary.csv")
        scored_csv = os.path.join(out_root, round_name, "buyplan_scored.csv")
        if not os.path.exists(summary_csv):
            continue
        df = pd.read_csv(summary_csv)
        scored_df = pd.read_csv(scored_csv) if os.path.exists(scored_csv) else pd.DataFrame()
        if not required_cols.issubset(set(df.columns)):
            continue
        base_hit_col = "ticket01_is_hit"
        base_hit_series = (
            scored_df[base_hit_col].fillna(False).astype(bool)
            if not scored_df.empty and base_hit_col in scored_df.columns
            else pd.Series(dtype=bool)
        )
        for _, rec in df.iterrows():
            ticket = str(rec.get("ticket", "")).strip()
            if ticket not in ROLE_FAMILY_TICKETS:
                continue
            role_family = str(rec.get("role_family", "none") or "none")
            ticket_hit_col = f"{ticket}_is_hit"
            recovered_vs_base = 0
            lost_vs_base = 0
            same_family_hit_shared = 0
            same_family_hit_unique = 0
            if (
                not scored_df.empty
                and ticket_hit_col in scored_df.columns
                and not base_hit_series.empty
            ):
                ticket_hit_series = scored_df[ticket_hit_col].fillna(False).astype(bool)
                recovered_vs_base = int((ticket_hit_series & ~base_hit_series).sum())
                lost_vs_base = int((~ticket_hit_series & base_hit_series).sum())
                peer_cols = [
                    f"ticket{peer:02d}_is_hit"
                    for peer in (
                        [2, 5, 6, 8, 9] if role_family == "draw"
                        else [3, 7, 10] if role_family == "swing"
                        else []
                    )
                    if f"ticket{peer:02d}" != ticket and f"ticket{peer:02d}_is_hit" in scored_df.columns
                ]
                if peer_cols:
                    peer_hit_count = scored_df[peer_cols].fillna(False).astype(bool).sum(axis=1)
                    same_family_hit_unique = int((ticket_hit_series & (peer_hit_count == 0)).sum())
                    same_family_hit_shared = int((ticket_hit_series & (peer_hit_count >= 1)).sum())
                else:
                    same_family_hit_unique = int(ticket_hit_series.sum())
            row = {
                "round": round_name,
                "ticket": ticket,
                "hits": int(rec.get("hits", 0)),
                "total": int(rec.get("total", 0)),
                "hit_rate": float(rec.get("hit_rate", 0.0)),
                "draw_count": int(float(rec.get("draw_count", 0) or 0)),
                "sway_count": int(float(rec.get("sway_count", 0) or 0)),
                "role_family": role_family,
                "role_mode": str(rec.get("role_mode", "none") or "none"),
                "role_unique_only": int(float(rec.get("role_unique_only", 0) or 0)),
                "role_same_family": int(float(rec.get("role_same_family", 0) or 0)),
                "role_diff_family": int(float(rec.get("role_diff_family", 0) or 0)),
                "base_miss_recovery": recovered_vs_base,
                "base_hit_loss": lost_vs_base,
                "base_net_gain": recovered_vs_base - lost_vs_base,
                "family_hit_unique": same_family_hit_unique,
                "family_hit_shared": same_family_hit_shared,
                "topology_mode": str(rec.get("topology_mode", "unknown") or "unknown"),
                "topology_draw_cluster_count": int(float(rec.get("topology_draw_cluster_count", 0) or 0)),
                "topology_swing_cluster_count": int(float(rec.get("topology_swing_cluster_count", 0) or 0)),
                "topology_hold_cluster_count": int(float(rec.get("topology_hold_cluster_count", 0) or 0)),
                "cluster_changes": str(rec.get("cluster_changes", "")),
                "basis_changes": str(rec.get("basis_changes", "")),
            }
            rows.append(row)
            item = agg.setdefault(ticket, {"hits": 0.0, "total": 0.0, "draw_count": 0.0, "sway_count": 0.0, "role_unique_only": 0.0, "role_same_family": 0.0, "role_diff_family": 0.0, "base_miss_recovery": 0.0, "base_hit_loss": 0.0, "base_net_gain": 0.0, "family_hit_unique": 0.0, "family_hit_shared": 0.0, "rounds": 0.0, "mode_skip": 0.0, "mode_active": 0.0})
            item["hits"] += row["hits"]
            item["total"] += row["total"]
            item["draw_count"] += row["draw_count"]
            item["sway_count"] += row["sway_count"]
            item["role_unique_only"] += row["role_unique_only"]
            item["role_same_family"] += row["role_same_family"]
            item["role_diff_family"] += row["role_diff_family"]
            item["base_miss_recovery"] += row["base_miss_recovery"]
            item["base_hit_loss"] += row["base_hit_loss"]
            item["base_net_gain"] += row["base_net_gain"]
            item["family_hit_unique"] += row["family_hit_unique"]
            item["family_hit_shared"] += row["family_hit_shared"]
            if row["role_mode"] == "skip":
                item["mode_skip"] += 1
            elif row["role_mode"] == "active":
                item["mode_active"] += 1
            item["rounds"] += 1

    detail_df = pd.DataFrame(rows)
    detail_csv = os.path.join(out_root, "buyplan_role_detail.csv")
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8")

    summary_rows: List[Dict[str, object]] = []
    for ticket in sorted(agg):
        item = agg[ticket]
        rounds_count = int(item["rounds"])
        total = int(item["total"])
        hits = int(item["hits"])
        summary_rows.append(
            {
                "ticket": ticket,
                "hits": hits,
                "total": total,
                "hit_rate": (hits / total) if total else 0.0,
                "avg_draw_count": (item["draw_count"] / rounds_count) if rounds_count else 0.0,
                "avg_sway_count": (item["sway_count"] / rounds_count) if rounds_count else 0.0,
                "avg_role_unique_only": (item["role_unique_only"] / rounds_count) if rounds_count else 0.0,
                "avg_role_same_family": (item["role_same_family"] / rounds_count) if rounds_count else 0.0,
                "avg_role_diff_family": (item["role_diff_family"] / rounds_count) if rounds_count else 0.0,
                "avg_base_miss_recovery": (item["base_miss_recovery"] / rounds_count) if rounds_count else 0.0,
                "avg_base_hit_loss": (item["base_hit_loss"] / rounds_count) if rounds_count else 0.0,
                "avg_base_net_gain": (item["base_net_gain"] / rounds_count) if rounds_count else 0.0,
                "avg_family_hit_unique": (item["family_hit_unique"] / rounds_count) if rounds_count else 0.0,
                "avg_family_hit_shared": (item["family_hit_shared"] / rounds_count) if rounds_count else 0.0,
                "base_miss_recovery_rate": (item["base_miss_recovery"] / total) if total else 0.0,
                "mode_skip_rounds": int(item["mode_skip"]),
                "mode_active_rounds": int(item["mode_active"]),
                "rounds": rounds_count,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_root, "buyplan_role_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    mode_rows: List[Dict[str, object]] = []
    if not detail_df.empty and "role_mode" in detail_df.columns:
        grouped = (
            detail_df.groupby(["ticket", "role_family", "role_mode"], dropna=False)[
                ["hits", "total", "draw_count", "sway_count", "role_unique_only", "role_same_family", "role_diff_family", "base_miss_recovery", "base_hit_loss", "base_net_gain", "family_hit_unique", "family_hit_shared"]
            ]
            .sum()
            .reset_index()
        )
        round_counts = (
            detail_df.groupby(["ticket", "role_family", "role_mode"], dropna=False)["round"]
            .count()
            .reset_index(name="rounds")
        )
        mode_df = grouped.merge(round_counts, on=["ticket", "role_family", "role_mode"], how="left")
        for _, rec in mode_df.iterrows():
            rounds_count = int(rec.get("rounds", 0) or 0)
            total = int(rec.get("total", 0) or 0)
            hits = int(rec.get("hits", 0) or 0)
            mode_rows.append(
                {
                    "ticket": str(rec.get("ticket", "")),
                    "role_family": str(rec.get("role_family", "none") or "none"),
                    "role_mode": str(rec.get("role_mode", "none") or "none"),
                    "hits": hits,
                    "total": total,
                    "hit_rate": (hits / total) if total else 0.0,
                    "avg_draw_count": (float(rec.get("draw_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_sway_count": (float(rec.get("sway_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_role_unique_only": (float(rec.get("role_unique_only", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_role_same_family": (float(rec.get("role_same_family", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_role_diff_family": (float(rec.get("role_diff_family", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_base_miss_recovery": (float(rec.get("base_miss_recovery", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_base_hit_loss": (float(rec.get("base_hit_loss", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_base_net_gain": (float(rec.get("base_net_gain", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_family_hit_unique": (float(rec.get("family_hit_unique", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_family_hit_shared": (float(rec.get("family_hit_shared", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "rounds": rounds_count,
                }
            )
    mode_summary_df = pd.DataFrame(mode_rows)
    mode_summary_csv = os.path.join(out_root, "buyplan_role_mode_summary.csv")
    mode_summary_df.to_csv(mode_summary_csv, index=False, encoding="utf-8")

    topology_rows: List[Dict[str, object]] = []
    if not detail_df.empty and {"ticket", "topology_mode"}.issubset(set(detail_df.columns)):
        grouped = (
            detail_df.groupby(["ticket", "topology_mode"], dropna=False)[
                [
                    "hits",
                    "total",
                    "draw_count",
                    "sway_count",
                    "role_unique_only",
                    "role_same_family",
                    "role_diff_family",
                    "base_miss_recovery",
                    "base_hit_loss",
                    "base_net_gain",
                    "family_hit_unique",
                    "family_hit_shared",
                    "topology_draw_cluster_count",
                    "topology_swing_cluster_count",
                    "topology_hold_cluster_count",
                ]
            ]
            .sum()
            .reset_index()
        )
        round_counts = (
            detail_df.groupby(["ticket", "topology_mode"], dropna=False)["round"]
            .count()
            .reset_index(name="rounds")
        )
        topology_df = grouped.merge(round_counts, on=["ticket", "topology_mode"], how="left")
        for _, rec in topology_df.iterrows():
            rounds_count = int(rec.get("rounds", 0) or 0)
            total = int(rec.get("total", 0) or 0)
            hits = int(rec.get("hits", 0) or 0)
            topology_rows.append(
                {
                    "ticket": str(rec.get("ticket", "")),
                    "topology_mode": str(rec.get("topology_mode", "unknown") or "unknown"),
                    "hits": hits,
                    "total": total,
                    "hit_rate": (hits / total) if total else 0.0,
                    "avg_draw_count": (float(rec.get("draw_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_sway_count": (float(rec.get("sway_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_role_unique_only": (float(rec.get("role_unique_only", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_role_same_family": (float(rec.get("role_same_family", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_role_diff_family": (float(rec.get("role_diff_family", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_base_miss_recovery": (float(rec.get("base_miss_recovery", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_base_hit_loss": (float(rec.get("base_hit_loss", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_base_net_gain": (float(rec.get("base_net_gain", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_family_hit_unique": (float(rec.get("family_hit_unique", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_family_hit_shared": (float(rec.get("family_hit_shared", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_topology_draw_cluster_count": (float(rec.get("topology_draw_cluster_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_topology_swing_cluster_count": (float(rec.get("topology_swing_cluster_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "avg_topology_hold_cluster_count": (float(rec.get("topology_hold_cluster_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                    "rounds": rounds_count,
                }
            )
    topology_summary_df = pd.DataFrame(topology_rows)
    topology_summary_csv = os.path.join(out_root, "buyplan_role_topology_summary.csv")
    topology_summary_df.to_csv(topology_summary_csv, index=False, encoding="utf-8")

    family_detail_rows: List[Dict[str, object]] = []
    family_summary_rows: List[Dict[str, object]] = []
    family_topology_rows: List[Dict[str, object]] = []
    if not detail_df.empty and "role_family" in detail_df.columns:
        family_df = detail_df[detail_df["role_family"].isin(["draw", "swing"])].copy()
        if not family_df.empty:
            family_round = (
                family_df.groupby(["round", "role_family", "topology_mode"], dropna=False)[
                    [
                        "hits",
                        "total",
                        "draw_count",
                        "sway_count",
                        "role_unique_only",
                        "role_same_family",
                        "role_diff_family",
                        "base_miss_recovery",
                        "base_hit_loss",
                        "base_net_gain",
                        "family_hit_unique",
                        "family_hit_shared",
                    ]
                ]
                .sum()
                .reset_index()
            )
            family_round["ticket_count"] = (
                family_df.groupby(["round", "role_family", "topology_mode"], dropna=False)["ticket"]
                .count()
                .reset_index(drop=True)
            )
            family_detail_rows = family_round.to_dict("records")

            family_grouped = (
                family_round.groupby(["role_family"], dropna=False)[
                    [
                        "hits",
                        "total",
                        "draw_count",
                        "sway_count",
                        "role_unique_only",
                        "role_same_family",
                        "role_diff_family",
                        "base_miss_recovery",
                        "base_hit_loss",
                        "base_net_gain",
                        "family_hit_unique",
                        "family_hit_shared",
                        "ticket_count",
                    ]
                ]
                .sum()
                .reset_index()
            )
            family_round_counts = (
                family_round.groupby(["role_family"], dropna=False)["round"]
                .count()
                .reset_index(name="rounds")
            )
            family_summary_df = family_grouped.merge(family_round_counts, on=["role_family"], how="left")
            for _, rec in family_summary_df.iterrows():
                rounds_count = int(rec.get("rounds", 0) or 0)
                total = int(rec.get("total", 0) or 0)
                hits = int(rec.get("hits", 0) or 0)
                family_summary_rows.append(
                    {
                        "role_family": str(rec.get("role_family", "none") or "none"),
                        "hits": hits,
                        "total": total,
                        "hit_rate": (hits / total) if total else 0.0,
                        "avg_ticket_count": (float(rec.get("ticket_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_draw_count": (float(rec.get("draw_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_sway_count": (float(rec.get("sway_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_role_unique_only": (float(rec.get("role_unique_only", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_role_same_family": (float(rec.get("role_same_family", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_role_diff_family": (float(rec.get("role_diff_family", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_base_miss_recovery": (float(rec.get("base_miss_recovery", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_base_hit_loss": (float(rec.get("base_hit_loss", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_base_net_gain": (float(rec.get("base_net_gain", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_family_hit_unique": (float(rec.get("family_hit_unique", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_family_hit_shared": (float(rec.get("family_hit_shared", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "rounds": rounds_count,
                    }
                )

            family_topology_grouped = (
                family_round.groupby(["role_family", "topology_mode"], dropna=False)[
                    [
                        "hits",
                        "total",
                        "draw_count",
                        "sway_count",
                        "role_unique_only",
                        "role_same_family",
                        "role_diff_family",
                        "base_miss_recovery",
                        "base_hit_loss",
                        "base_net_gain",
                        "family_hit_unique",
                        "family_hit_shared",
                        "ticket_count",
                    ]
                ]
                .sum()
                .reset_index()
            )
            family_topology_round_counts = (
                family_round.groupby(["role_family", "topology_mode"], dropna=False)["round"]
                .count()
                .reset_index(name="rounds")
            )
            family_topology_df = family_topology_grouped.merge(
                family_topology_round_counts,
                on=["role_family", "topology_mode"],
                how="left",
            )
            for _, rec in family_topology_df.iterrows():
                rounds_count = int(rec.get("rounds", 0) or 0)
                total = int(rec.get("total", 0) or 0)
                hits = int(rec.get("hits", 0) or 0)
                family_topology_rows.append(
                    {
                        "role_family": str(rec.get("role_family", "none") or "none"),
                        "topology_mode": str(rec.get("topology_mode", "unknown") or "unknown"),
                        "hits": hits,
                        "total": total,
                        "hit_rate": (hits / total) if total else 0.0,
                        "avg_ticket_count": (float(rec.get("ticket_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_draw_count": (float(rec.get("draw_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_sway_count": (float(rec.get("sway_count", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_role_unique_only": (float(rec.get("role_unique_only", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_role_same_family": (float(rec.get("role_same_family", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_role_diff_family": (float(rec.get("role_diff_family", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_base_miss_recovery": (float(rec.get("base_miss_recovery", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_base_hit_loss": (float(rec.get("base_hit_loss", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_base_net_gain": (float(rec.get("base_net_gain", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_family_hit_unique": (float(rec.get("family_hit_unique", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "avg_family_hit_shared": (float(rec.get("family_hit_shared", 0) or 0) / rounds_count) if rounds_count else 0.0,
                        "rounds": rounds_count,
                    }
                )

    family_detail_csv = os.path.join(out_root, "buyplan_family_detail.csv")
    pd.DataFrame(family_detail_rows).to_csv(family_detail_csv, index=False, encoding="utf-8")
    family_summary_csv = os.path.join(out_root, "buyplan_family_summary.csv")
    pd.DataFrame(family_summary_rows).to_csv(family_summary_csv, index=False, encoding="utf-8")
    family_topology_csv = os.path.join(out_root, "buyplan_family_topology_summary.csv")
    pd.DataFrame(family_topology_rows).to_csv(family_topology_csv, index=False, encoding="utf-8")

    round_summary_rows: List[Dict[str, object]] = []
    family_detail_df = pd.DataFrame(family_detail_rows)
    for round_name in rounds:
        summary_csv_path = os.path.join(out_root, round_name, "buyplan_scored_summary.csv")
        if not os.path.exists(summary_csv_path):
            continue
        round_summary_df = pd.read_csv(summary_csv_path)
        if round_summary_df.empty or "ticket" not in round_summary_df.columns:
            continue
        base_row_df = round_summary_df[round_summary_df["ticket"].astype(str) == "ticket01"]
        if base_row_df.empty:
            continue
        base_row = base_row_df.iloc[0]
        best_row = round_summary_df.sort_values(["hits", "hit_rate", "ticket"], ascending=[False, False, True]).iloc[0]
        draw_family_row = None
        swing_family_row = None
        if not family_detail_df.empty:
            draw_sub = family_detail_df[
                (family_detail_df["round"].astype(str) == round_name) &
                (family_detail_df["role_family"].astype(str) == "draw")
            ]
            if not draw_sub.empty:
                draw_family_row = draw_sub.iloc[0]
            swing_sub = family_detail_df[
                (family_detail_df["round"].astype(str) == round_name) &
                (family_detail_df["role_family"].astype(str) == "swing")
            ]
            if not swing_sub.empty:
                swing_family_row = swing_sub.iloc[0]
        ticket_stats = _load_round_ticket_stats(round_name, out_root)
        round_summary_rows.append(
            {
                "round": round_name,
                "topology_mode": str(base_row.get("topology_mode", "unknown") or "unknown"),
                "topology_draw_cluster_count": int(float(base_row.get("topology_draw_cluster_count", 0) or 0)),
                "topology_swing_cluster_count": int(float(base_row.get("topology_swing_cluster_count", 0) or 0)),
                "topology_hold_cluster_count": int(float(base_row.get("topology_hold_cluster_count", 0) or 0)),
                "ticket01_hits": int(float(base_row.get("hits", 0) or 0)),
                "ticket01_total": int(float(base_row.get("total", 0) or 0)),
                "ticket01_hit_rate": float(base_row.get("hit_rate", 0.0) or 0.0),
                "best_ticket": str(best_row.get("ticket", "")),
                "best_hits": int(float(best_row.get("hits", 0) or 0)),
                "best_hit_rate": float(best_row.get("hit_rate", 0.0) or 0.0),
                "unique_ticket_count": int(ticket_stats["unique_ticket_count"]),
                "duplicate_count": int(ticket_stats["duplicate_count"]),
                "avg_ticket_distance": float(ticket_stats["avg_ticket_distance"]),
                "draw_family_ticket_count": int(float(draw_family_row.get("ticket_count", 0) or 0)) if draw_family_row is not None else 0,
                "draw_family_hit_rate": (
                    float(draw_family_row.get("hits", 0) or 0) / float(draw_family_row.get("total", 0) or 0)
                    if draw_family_row is not None and float(draw_family_row.get("total", 0) or 0) > 0
                    else 0.0
                ),
                "draw_family_base_net_gain": int(float(draw_family_row.get("base_net_gain", 0) or 0)) if draw_family_row is not None else 0,
                "swing_family_ticket_count": int(float(swing_family_row.get("ticket_count", 0) or 0)) if swing_family_row is not None else 0,
                "swing_family_hit_rate": (
                    float(swing_family_row.get("hits", 0) or 0) / float(swing_family_row.get("total", 0) or 0)
                    if swing_family_row is not None and float(swing_family_row.get("total", 0) or 0) > 0
                    else 0.0
                ),
                "swing_family_base_net_gain": int(float(swing_family_row.get("base_net_gain", 0) or 0)) if swing_family_row is not None else 0,
            }
        )

    round_summary_df = pd.DataFrame(round_summary_rows)
    round_summary_csv = os.path.join(out_root, "buyplan_round_summary.csv")
    round_summary_df.to_csv(round_summary_csv, index=False, encoding="utf-8")
    round_summary_html = os.path.join(out_root, "buyplan_round_summary.html")
    write_html_table(round_summary_df, "buyplan round summary", ROUND_SUMMARY_DESC_MAP, round_summary_html)
    dashboard_df = round_summary_df.copy()
    if not dashboard_df.empty:
        dashboard_df["dashboard_note"] = [
            _round_dashboard_note(rec)
            for rec in dashboard_df.to_dict("records")
        ]
        dashboard_df = dashboard_df[
            [
                "round",
                "topology_mode",
                "ticket01_hit_rate",
                "best_ticket",
                "best_hit_rate",
                "draw_family_base_net_gain",
                "swing_family_base_net_gain",
                "unique_ticket_count",
                "duplicate_count",
                "dashboard_note",
            ]
        ].sort_values("round")
    dashboard_csv = os.path.join(out_root, "buyplan_round_dashboard.csv")
    dashboard_df.to_csv(dashboard_csv, index=False, encoding="utf-8")
    dashboard_html = os.path.join(out_root, "buyplan_round_dashboard.html")
    write_html_table(dashboard_df, "buyplan round dashboard", ROUND_DASHBOARD_DESC_MAP, dashboard_html)
    return (
        summary_csv,
        mode_summary_csv,
        topology_summary_csv,
        family_summary_csv,
        family_topology_csv,
        round_summary_csv,
        round_summary_html,
        dashboard_csv,
        dashboard_html,
    )


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
    out_root = os.path.abspath(args.out_root)
    if args.use_current_predictions:
        if len(rounds) != 1:
            raise SystemExit("--use-current-predictions は --round を1つだけ指定して使ってください")
        predictions_override = os.path.abspath(args.current_predictions)
        if out_root == os.path.abspath(DEFAULT_OUT_ROOT):
            out_root = DEFAULT_EXPERIMENT_OUT_ROOT
            print(f"[INFO] current predictions experiment 出力先に切り替え: {out_root}")

    for round_name in rounds:
        try:
            scored_html = _build_backtest(round_name, args.rounds_dir, out_root, predictions_override=predictions_override)
            print(f"[OK] {scored_html}")
        except Exception as e:
            print(f"[SKIP] {round_name} {e}")
    (
        role_summary_csv,
        mode_summary_csv,
        topology_summary_csv,
        family_summary_csv,
        family_topology_csv,
        round_summary_csv,
        round_summary_html,
        round_dashboard_csv,
        round_dashboard_html,
    ) = _write_role_summary(rounds, out_root)
    print(f"[OK] {role_summary_csv}")
    print(f"[OK] {mode_summary_csv}")
    print(f"[OK] {topology_summary_csv}")
    print(f"[OK] {family_summary_csv}")
    print(f"[OK] {family_topology_csv}")
    print(f"[OK] {round_summary_csv}")
    print(f"[OK] {round_summary_html}")
    print(f"[OK] {round_dashboard_csv}")
    print(f"[OK] {round_dashboard_html}")


if __name__ == "__main__":
    main()
