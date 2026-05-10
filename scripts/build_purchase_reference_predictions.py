#!/usr/bin/env python3
import argparse
import os
import re
import unicodedata
from typing import List

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT_CSV = os.path.join(ROOT_DIR, "data", "purchase_reference", "predictions.csv")
DEFAULT_OUT_CONTEXT_CSV = os.path.join(ROOT_DIR, "data", "purchase_reference", "predictions_buyplan_context.csv")

CONFIDENCE_THRESHOLDS = {
    ("j1", "1"): 0.48,
    ("j1", "0"): 0.38,
    ("j1", "2"): 0.40,
    ("j2", "1"): 0.38,
}


def extract_round_number(value) -> int | None:
    if pd.isna(value):
        return None
    s = unicodedata.normalize("NFKC", str(value))
    m = re.search(r"第\s*([0-9]+)\s*節", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_prediction_csv(path: str, league_label: str, round_no: int) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"predictions csv not found: {path}")
    # Keep empty-string branch outputs distinct from NaN so H/D/A sparse variant columns
    # survive the J1/J2 concat and later CSV roundtrip unchanged.
    df = pd.read_csv(path, keep_default_na=False)
    if "節" not in df.columns:
        raise ValueError(f"'節' 列がありません: {path}")
    out = df.copy()
    out["__round_no"] = out["節"].map(extract_round_number)
    out = out[out["__round_no"] == int(round_no)].copy()
    out = out.drop(columns=["__round_no"], errors="ignore")
    if out.empty:
        raise RuntimeError(f"{league_label} round{round_no:02d} の予想行がありません: {path}")
    return out


def align_prediction_columns(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    ordered_columns: List[str] = []
    for df in frames:
        for col in df.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
    aligned: List[pd.DataFrame] = []
    for df in frames:
        aligned.append(df.reindex(columns=ordered_columns))
    return aligned


def sort_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "datetime" in out.columns:
        out["__dt_sort"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        out["__dt_sort"] = pd.NaT
    sort_cols: List[str] = ["__dt_sort"]
    for col in ["league", "home_team", "away_team", "match_id"]:
        if col in out.columns:
            sort_cols.append(col)
    out = out.sort_values(sort_cols, kind="mergesort", na_position="last").drop(columns=["__dt_sort"], errors="ignore")
    out = out.reset_index(drop=True)
    return out


def _result_to_symbol(value: object) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    if s in {"H", "1"}:
        return "1"
    if s in {"D", "0", "DRAW"}:
        return "0"
    if s in {"A", "2"}:
        return "2"
    return ""


def _symbol_to_label(symbol: str) -> str:
    return {"1": "home", "0": "draw", "2": "away"}.get(symbol, "")


def _safe_float(value: object, default: float = 0.0) -> float:
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return default
    return float(x)


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value or "").strip()
    if text.lower() == "nan":
        return ""
    return text


def _confidence_threshold(league: str, symbol: str) -> float | None:
    return CONFIDENCE_THRESHOLDS.get((str(league or "").strip().lower(), str(symbol or "").strip()))


def _build_context_row(row: pd.Series) -> dict:
    p_home = _safe_float(row.get("prob_home_win"), 1.0 / 3.0)
    p_draw = _safe_float(row.get("prob_draw"), 1.0 / 3.0)
    p_away = _safe_float(row.get("prob_away_win"), 1.0 / 3.0)
    ranked = sorted([("1", p_home), ("0", p_draw), ("2", p_away)], key=lambda kv: (-kv[1], kv[0]))

    primary = _result_to_symbol(row.get("predicted_result")) or ranked[0][0]
    secondary = next((sym for sym, _ in ranked if sym != primary), ranked[1][0] if len(ranked) > 1 else "")
    primary_prob = dict(ranked).get(primary, ranked[0][1])
    secondary_prob = dict(ranked).get(secondary, ranked[1][1] if len(ranked) > 1 else 0.0)
    prob_gap = float(primary_prob - secondary_prob)
    league = _safe_text(row.get("league", "")).lower()
    draw_gap = float(max(p_home, p_away) - p_draw)
    trusted_score = _safe_float(row.get("max_prob_cal"), primary_prob)
    confidence_threshold = _safe_float(row.get("confidence_threshold"), float("nan"))
    if pd.isna(confidence_threshold):
        confidence_threshold = _confidence_threshold(league, primary)
    confidence_hit_raw = row.get("threshold_hit_flag")
    if str(confidence_hit_raw).strip().lower() in {"true", "1"}:
        confidence_hit = True
    elif str(confidence_hit_raw).strip().lower() in {"false", "0"}:
        confidence_hit = False
    else:
        confidence_hit = bool(confidence_threshold is not None and not pd.isna(confidence_threshold) and primary_prob >= float(confidence_threshold))
    confidence_level_raw = _safe_text(row.get("confidence_class", "")).lower()
    if confidence_level_raw in {"trusted", "watch", "fragile", "unscored"}:
        confidence_level = confidence_level_raw
    elif confidence_threshold is None or pd.isna(confidence_threshold):
        confidence_level = "unscored"
    elif confidence_hit:
        confidence_level = "trusted"
    elif primary_prob >= 0.35:
        confidence_level = "watch"
    else:
        confidence_level = "fragile"

    flags_raw = _safe_text(row.get("match_type_flags", ""))
    flags = [x for x in flags_raw.split(",") if x]
    primary_flag = _safe_text(row.get("match_type_primary", ""))
    draw_risk = str(row.get("draw_risk_flag", "")).strip().lower() in {"true", "1"}
    lab_style_conflict = str(row.get("match_type_lab_style_conflict", "")).strip().lower() in {"true", "1"}
    lab_low_event = str(row.get("match_type_lab_low_event", "")).strip().lower() in {"true", "1"}
    lab_edge = _safe_float(row.get("match_type_lab_matchup_edge"), 0.0)
    type_c = _result_to_symbol(row.get("predicted_result_type_c"))
    type_b = _result_to_symbol(row.get("predicted_result_type_b"))
    confidence_level_detail = confidence_level
    j2_trusted_soft = bool(
        league == "j2"
        and confidence_level == "trusted"
        and (
            trusted_score < 0.38
            or draw_risk
            or draw_gap <= 0.02
        )
    )
    if league == "j2" and confidence_level == "trusted":
        confidence_level_detail = "trusted_soft" if j2_trusted_soft else "trusted_strong"

    if primary == "0" or draw_risk or lab_style_conflict:
        risk_level = "draw_watch"
    elif confidence_level == "trusted" and primary in {"1", "2"}:
        risk_level = "fixed"
    elif confidence_level == "fragile":
        risk_level = "volatile"
    elif prob_gap >= 0.18 and primary in {"1", "2"}:
        risk_level = "fixed"
    elif prob_gap <= 0.04:
        risk_level = "volatile"
    else:
        risk_level = "caution"
    if j2_trusted_soft and risk_level == "fixed":
        risk_level = "caution"

    if confidence_level == "trusted":
        if primary == "0":
            ticket_guidance = "main_ok"
        else:
            ticket_guidance = "main_only"
    elif primary == "0":
        ticket_guidance = "main_ok"
    elif draw_risk and type_c == "0":
        ticket_guidance = "draw_cover"
    elif type_b and type_b != primary and ("lab_away_matchup" in flags or "lab_home_matchup" in flags):
        ticket_guidance = "lab_cover"
    elif draw_risk or lab_style_conflict or lab_low_event:
        ticket_guidance = "avoid_main"
    else:
        ticket_guidance = "main_only"
    if j2_trusted_soft:
        if primary == "0":
            ticket_guidance = "main_ok"
        elif draw_risk or lab_style_conflict or lab_low_event:
            ticket_guidance = "avoid_main"
        else:
            ticket_guidance = "main_ok"

    summary_parts: List[str] = []
    if primary_flag:
        summary_parts.append(primary_flag)
    for marker in ["draw_risk", "lab_style_conflict", "lab_low_event", "lab_away_matchup", "lab_home_matchup", "away_strong", "home_strong"]:
        if marker in flags and marker not in summary_parts:
            summary_parts.append(marker)
    if not summary_parts:
        summary_parts.append("neutral")
    if ("lab_away_matchup" in flags or "lab_home_matchup" in flags) and abs(lab_edge) > 0:
        summary_parts.append(f"lab_edge={lab_edge:.1f}")
    if confidence_threshold is not None and not pd.isna(confidence_threshold):
        threshold_label = f"{confidence_threshold:.2f}".rstrip("0").rstrip(".")
        summary_parts.append(f"conf={confidence_level_detail}@{threshold_label}")
    else:
        summary_parts.append(f"conf={confidence_level_detail}")

    return {
        "節": row.get("節", ""),
        "league": row.get("league", ""),
        "match_id": row.get("match_id", ""),
        "datetime": row.get("datetime", ""),
        "home_team": row.get("home_team", ""),
        "away_team": row.get("away_team", ""),
        "primary_pick_symbol": primary,
        "secondary_pick_symbol": secondary,
        "primary_pick_label": _symbol_to_label(primary),
        "secondary_pick_label": _symbol_to_label(secondary),
        "primary_prob": primary_prob,
        "secondary_prob": secondary_prob,
        "prob_gap": prob_gap,
        "confidence_level": confidence_level,
        "confidence_level_detail": confidence_level_detail,
        "confidence_threshold": confidence_threshold,
        "confidence_hit": confidence_hit,
        "risk_level": risk_level,
        "ticket_guidance": ticket_guidance,
        "decision_summary": " + ".join(summary_parts),
        "predicted_result": row.get("predicted_result", ""),
        "predicted_result_type_a": row.get("predicted_result_type_a", ""),
        "predicted_result_type_b": row.get("predicted_result_type_b", ""),
        "predicted_result_type_c": row.get("predicted_result_type_c", ""),
        "match_type_primary": row.get("match_type_primary", ""),
        "match_type_flags": flags_raw,
        "draw_risk_flag": row.get("draw_risk_flag", ""),
        "match_type_lab_matchup_edge": row.get("match_type_lab_matchup_edge", ""),
        "match_type_lab_style_conflict": row.get("match_type_lab_style_conflict", ""),
        "match_type_lab_low_event": row.get("match_type_lab_low_event", ""),
        "type_adjust_note_a": row.get("type_adjust_note_a", ""),
        "type_adjust_note_b": row.get("type_adjust_note_b", ""),
        "type_adjust_note_c": row.get("type_adjust_note_c", ""),
    }


def build_buyplan_context(df: pd.DataFrame) -> pd.DataFrame:
    rows = [_build_context_row(row) for _, row in df.iterrows()]
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="purchase_reference/predictions.csv を節固定で再生成")
    p.add_argument("--season", type=int, required=True, help="例: 2026")
    p.add_argument("--round", type=int, default=None, dest="round_no", help="J1/J2共通の節番号")
    p.add_argument("--j1-round", type=int, default=None, help="J1 の節番号を個別指定")
    p.add_argument("--j2-round", type=int, default=None, help="J2 の節番号を個別指定")
    p.add_argument("--j1", default=None, help="既定: j1_{season}_predictions.csv")
    p.add_argument("--j2", default=None, help="既定: j2_{season}_predictions.csv")
    p.add_argument("--out", default=DEFAULT_OUT_CSV, help="既定: data/purchase_reference/predictions.csv")
    p.add_argument(
        "--out-context",
        default=DEFAULT_OUT_CONTEXT_CSV,
        help="既定: data/purchase_reference/predictions_buyplan_context.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.round_no is None and args.j1_round is None and args.j2_round is None:
        raise RuntimeError("--round または --j1-round/--j2-round の指定が必要です")
    j1_path = args.j1 or os.path.join(ROOT_DIR, f"j1_{args.season}_predictions.csv")
    j2_path = args.j2 or os.path.join(ROOT_DIR, f"j2_{args.season}_predictions.csv")
    out_csv = os.path.abspath(args.out)
    out_context_csv = os.path.abspath(args.out_context)
    j1_round = int(args.j1_round if args.j1_round is not None else args.round_no)
    j2_round = int(args.j2_round if args.j2_round is not None else args.round_no)

    j1_df = load_prediction_csv(j1_path, "J1", j1_round)
    j2_df = load_prediction_csv(j2_path, "J2", j2_round)
    j1_df, j2_df = align_prediction_columns([j1_df, j2_df])
    out = sort_predictions(pd.concat([j1_df, j2_df], ignore_index=True))

    outdir = os.path.dirname(out_csv) or "."
    os.makedirs(outdir, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    context_df = build_buyplan_context(out)
    context_dir = os.path.dirname(out_context_csv) or "."
    os.makedirs(context_dir, exist_ok=True)
    context_df.to_csv(out_context_csv, index=False, encoding="utf-8-sig")

    round_labels = sorted({str(v) for v in out.get("節", pd.Series(dtype=str)).dropna().unique().tolist()})
    print(
        f"[OK] {out_csv} rows={len(out)} "
        f"j1={len(j1_df)}(round={j1_round}) j2={len(j2_df)}(round={j2_round}) labels={round_labels}"
    )
    print(f"[OK] {out_context_csv} rows={len(context_df)}")


if __name__ == "__main__":
    main()
