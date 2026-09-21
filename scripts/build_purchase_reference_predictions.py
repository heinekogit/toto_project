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
DEFAULT_TOTO_ORDER_CSV = os.path.join(ROOT_DIR, "data", "manual", "toto節リスト.csv")
DEFAULT_TOTO_ROUND_MAP_CSV = os.path.join(ROOT_DIR, "data", "manual", "toto_round_map_2026.csv")
LEGACY_VARIANT_COLS = {
    "predicted_result_type_a",
    "predicted_result_type_b",
    "predicted_result_type_c",
    "type_adjust_note_a",
    "type_adjust_note_b",
    "type_adjust_note_c",
}


def _latest_candidate_prediction_path(league: str, season: int) -> str:
    snap_dir = os.path.join(ROOT_DIR, "data", "output_snapshots", f"{league}_{season}")
    if not os.path.isdir(snap_dir):
        return ""
    candidates = []
    for name in os.listdir(snap_dir):
        if not (
            name.endswith(f"predictions_candidate_{league}_{season}_predictions_hfa_on.csv")
            or name.endswith(f"predictions_candidate_{league}_{season}_predictions.csv")
        ):
            continue
        path = os.path.join(snap_dir, name)
        if os.path.isfile(path):
            candidates.append(path)
    if not candidates:
        return ""
    def _ts_key(path: str) -> tuple[str, float, str]:
        name = os.path.basename(path)
        m = re.match(r"^(\d{8}_\d{6})_", name)
        ts = m.group(1) if m else ""
        return (ts, os.path.getmtime(path), name)

    candidates.sort(key=_ts_key)
    return candidates[-1]


def resolve_prediction_input_path(explicit_path: str | None, league: str, season: int) -> str:
    if explicit_path:
        return explicit_path
    latest_candidate = _latest_candidate_prediction_path(league, season)
    if latest_candidate:
        return latest_candidate
    return os.path.join(ROOT_DIR, f"{league}_{season}_predictions.csv")

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


def _round_label_variants(round_no: int) -> list[str]:
    n = int(round_no)
    variants = [
        f"第{n}節",
        f"第{n:02d}節",
        f"第{n}節第1日",
        f"第{n}節第2日",
        f"第{n}節第3日",
    ]
    # 2026 の特殊リーグ期間では J1/J2 round19 が第1戦、round20 が第2戦に相当する。
    playoff_map = {
        19: "第1戦",
        20: "第2戦",
    }
    if n in playoff_map:
        base = playoff_map[n]
        variants.extend(
            [
                base,
                f"{base}第1日",
                f"{base}第2日",
            ]
        )
    normalized = []
    seen = set()
    for item in variants:
        norm = unicodedata.normalize("NFKC", item)
        if norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


def resolve_toto_round_id(season: int, round_no: int) -> str:
    season_s = str(int(season))
    round_s = str(int(round_no))

    if os.path.exists(DEFAULT_TOTO_ORDER_CSV):
        try:
            order_df = pd.read_csv(DEFAULT_TOTO_ORDER_CSV, dtype=str, encoding="utf-8-sig")
            if {"season", "toto_round", "J1_round"}.issubset(order_df.columns):
                hit = order_df[
                    order_df["season"].astype(str).str.strip().eq(season_s)
                    & order_df["J1_round"].astype(str).str.strip().eq(round_s)
                ]
                if not hit.empty:
                    toto_round = str(hit["toto_round"].dropna().iloc[0]).strip()
                    if toto_round:
                        return f"toto{toto_round}"
        except Exception:
            pass

    if os.path.exists(DEFAULT_TOTO_ROUND_MAP_CSV):
        try:
            map_df = pd.read_csv(DEFAULT_TOTO_ROUND_MAP_CSV, dtype=str, encoding="utf-8-sig")
            if {"season", "j1_round", "toto_round"}.issubset(map_df.columns):
                hit = map_df[
                    map_df["season"].astype(str).str.strip().eq(season_s)
                    & map_df["j1_round"].astype(str).str.strip().eq(round_s)
                ]
                if not hit.empty:
                    toto_round = str(hit["toto_round"].dropna().iloc[0]).strip()
                    if toto_round:
                        return f"toto{toto_round}"
        except Exception:
            pass

    return f"toto{int(round_no)}"


def load_prediction_csv(
    path: str,
    league_label: str,
    season: int,
    round_no: int,
    logical_season: int | None = None,
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"predictions csv not found: {path}")
    # Keep empty-string branch outputs distinct from NaN so H/D/A sparse variant columns
    # survive the J1/J2 concat and later CSV roundtrip unchanged.
    df = pd.read_csv(path, keep_default_na=False)
    if "節" not in df.columns:
        raise ValueError(f"'節' 列がありません: {path}")
    out = df.copy()
    out["__round_label_norm"] = out["節"].map(lambda v: unicodedata.normalize("NFKC", str(v or "")))
    out["__round_no"] = out["節"].map(extract_round_number)
    target_variants = set(_round_label_variants(round_no))
    out = out[
        (out["__round_no"] == int(round_no))
        | (out["__round_label_norm"].isin(target_variants))
    ].copy()
    out = out.drop(columns=["__round_no", "__round_label_norm"], errors="ignore")
    if out.empty:
        raise RuntimeError(f"{league_label} round{round_no:02d} の予想行がありません: {path}")
    out["toto_round_id"] = resolve_toto_round_id(
        logical_season if logical_season is not None else season,
        round_no,
    )
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


def drop_legacy_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_cols = [
        c
        for c in out.columns
        if c in LEGACY_VARIANT_COLS
        or c.startswith("type_a_")
        or c.startswith("type_b_")
        or c.startswith("type_c_")
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
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


def _build_purchase_layer(row: pd.Series, p_home: float, p_draw: float, p_away: float, pred_symbol: str) -> dict:
    ranked = sorted([("1", p_home), ("0", p_draw), ("2", p_away)], key=lambda kv: (-kv[1], kv[0]))
    rank1_symbol = str(ranked[0][0])
    rank2_symbol = str(ranked[1][0])
    rank3_symbol = str(ranked[2][0])
    rank1_prob = float(ranked[0][1])
    rank2_prob = float(ranked[1][1])
    rank3_prob = float(ranked[2][1])
    best_symbol = rank1_symbol
    second_symbol = next((sym for sym, _ in ranked if sym != best_symbol), "")
    best_prob = float(ranked[0][1])
    second_prob = float(ranked[1][1] if len(ranked) > 1 else 0.0)
    top_gap = float(best_prob - second_prob)
    draw_gap = float(max(p_home, p_away) - p_draw)
    best_side_symbol = "1" if p_home >= p_away else "2"
    pred_main_symbol = _result_to_symbol(row.get("predicted_result_main"))
    purchase_type = _safe_text(row.get("match_purchase_type")).lower()
    draw_core_score = _safe_float(row.get("draw_core_score"), float("nan"))
    swing_close_score = _safe_float(row.get("swing_close_score"), float("nan"))
    profile = _safe_text(row.get("close_split_profile", "")).lower()

    if pred_symbol == "0":
        false_draw_watch_flag = purchase_type in {"j1_false_draw_watch", "j2_false_draw_watch"}
        strong_draw_core_flag = bool(
            not false_draw_watch_flag
            and p_draw >= 0.452
            and top_gap >= 0.175
            and draw_core_score >= 0.86
            and profile in {"draw_core", "soft_draw"}
            and _safe_float(row.get("lab_scenario_entropy_score"), float("nan")) <= 0.44
            and _safe_float(row.get("lab_stall_compactness_score"), float("nan")) >= 0.71
            and _safe_float(row.get("lab_draw_tension_score"), float("nan")) >= 0.82
        )
        draw_core_flag = bool(
            strong_draw_core_flag
        )
        if _safe_text(row.get("league", "")).upper() == "J1":
            draw_core_flag = False
        draw_cut_flag = bool(
            (top_gap <= 0.025)
            or (draw_gap >= -0.030)
            or swing_close_score >= 0.58
            or profile in {"soft_swing", "swing_close", "undiff_close"}
        )
        if false_draw_watch_flag:
            return {
                "admission_policy": "rank1_draw_watch",
                "rank1_symbol": rank1_symbol,
                "rank2_symbol": rank2_symbol,
                "rank3_symbol": rank3_symbol,
                "rank1_prob": rank1_prob,
                "rank2_prob": rank2_prob,
                "rank3_prob": rank3_prob,
                "draw_core_flag": False,
                "draw_purchase_tier": "watch",
                "primary_pick_symbol": "0",
                "secondary_pick_symbol": best_side_symbol,
                "risk_level": "draw_watch",
                "ticket_guidance": "draw_watch_split",
                "purchase_summary": f"draw watch split: pd={p_draw:.3f} gap={draw_gap:.3f} top_gap={top_gap:.3f}",
            }
        if draw_core_flag:
            return {
                "admission_policy": "rank1_draw_core",
                "rank1_symbol": rank1_symbol,
                "rank2_symbol": rank2_symbol,
                "rank3_symbol": rank3_symbol,
                "rank1_prob": rank1_prob,
                "rank2_prob": rank2_prob,
                "rank3_prob": rank3_prob,
                "draw_core_flag": True,
                "draw_purchase_tier": "core",
                "primary_pick_symbol": "0",
                "secondary_pick_symbol": best_side_symbol,
                "risk_level": "low",
                "ticket_guidance": "draw_core_keep",
                "purchase_summary": f"draw core keep: pd={p_draw:.3f} gap={draw_gap:.3f} top_gap={top_gap:.3f}",
            }
        if draw_cut_flag:
            return {
                "admission_policy": "rank2_draw_cut",
                "rank1_symbol": rank1_symbol,
                "rank2_symbol": rank2_symbol,
                "rank3_symbol": rank3_symbol,
                "rank1_prob": rank1_prob,
                "rank2_prob": rank2_prob,
                "rank3_prob": rank3_prob,
                "draw_core_flag": False,
                "draw_purchase_tier": "cut",
                "primary_pick_symbol": best_side_symbol,
                "secondary_pick_symbol": "0",
                "risk_level": "high",
                "ticket_guidance": "draw_cut_to_side",
                "purchase_summary": f"weak draw cut: pd={p_draw:.3f} gap={draw_gap:.3f} top_gap={top_gap:.3f}",
            }
        return {
            "admission_policy": "rank1_draw_rescue",
            "rank1_symbol": rank1_symbol,
            "rank2_symbol": rank2_symbol,
            "rank3_symbol": rank3_symbol,
            "rank1_prob": rank1_prob,
            "rank2_prob": rank2_prob,
            "rank3_prob": rank3_prob,
            "draw_core_flag": False,
            "draw_purchase_tier": "rescue",
            "primary_pick_symbol": "0",
            "secondary_pick_symbol": best_side_symbol,
            "risk_level": "medium",
            "ticket_guidance": "draw_rescue_split",
            "purchase_summary": f"draw rescue split: pd={p_draw:.3f} gap={draw_gap:.3f} top_gap={top_gap:.3f}",
        }

    side_secondary_symbol = next((sym for sym, _ in ranked if sym not in {pred_symbol, best_symbol}), "")
    if not side_secondary_symbol:
        side_secondary_symbol = next((sym for sym, _ in ranked if sym != pred_symbol), "")
    if rank1_symbol == "0" and pred_symbol in {"1", "2"} and pred_symbol == rank2_symbol:
        admission_policy = "rank2_side_escape"
        side_primary_symbol = rank2_symbol
        side_secondary_symbol = rank3_symbol
    else:
        admission_policy = "rank1_side_cover"
        side_primary_symbol = rank1_symbol
        side_secondary_symbol = rank2_symbol
    max_prob = max(p_home, p_draw, p_away)
    risk_level = "low" if (max_prob >= 0.45 and top_gap >= 0.08) else "medium"
    ticket_guidance = "side_with_cover"
    return {
        "admission_policy": admission_policy,
        "rank1_symbol": rank1_symbol,
        "rank2_symbol": rank2_symbol,
        "rank3_symbol": rank3_symbol,
        "rank1_prob": rank1_prob,
        "rank2_prob": rank2_prob,
        "rank3_prob": rank3_prob,
        "draw_core_flag": False,
        "draw_purchase_tier": "side",
        "primary_pick_symbol": side_primary_symbol,
        "secondary_pick_symbol": side_secondary_symbol,
        "risk_level": risk_level,
        "ticket_guidance": ticket_guidance,
        "purchase_summary": (
            f"side ranks r1={rank1_symbol} r2={rank2_symbol} r3={rank3_symbol}: "
            f"top_gap={top_gap:.3f} max_prob={max_prob:.3f}"
        ),
    }


def _build_context_row(row: pd.Series) -> dict:
    p_home = _safe_float(row.get("prob_home_win"), 1.0 / 3.0)
    p_draw = _safe_float(row.get("prob_draw"), 1.0 / 3.0)
    p_away = _safe_float(row.get("prob_away_win"), 1.0 / 3.0)
    ranked = sorted([("1", p_home), ("0", p_draw), ("2", p_away)], key=lambda kv: (-kv[1], kv[0]))

    pred_symbol = _result_to_symbol(row.get("predicted_result")) or ranked[0][0]
    purchase_layer = _build_purchase_layer(row, p_home, p_draw, p_away, pred_symbol)
    primary = str(purchase_layer["primary_pick_symbol"] or pred_symbol or ranked[0][0])
    secondary = str(
        purchase_layer["secondary_pick_symbol"]
        or next((sym for sym, _ in ranked if sym != primary), ranked[1][0] if len(ranked) > 1 else "")
    )
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
    lab_hold_weight = _safe_float(row.get("lab_hold_weight"), float("nan"))
    lab_stall_weight = _safe_float(row.get("lab_stall_weight"), float("nan"))
    lab_flip_weight = _safe_float(row.get("lab_flip_weight"), float("nan"))
    lab_volatility_score = _safe_float(row.get("lab_volatility_score"), float("nan"))
    lab_draw_tension_score = _safe_float(row.get("lab_draw_tension_score"), float("nan"))
    lab_dynamic_swing_score = _safe_float(row.get("lab_dynamic_swing_score"), float("nan"))
    lab_tempo_band = _safe_text(row.get("lab_tempo_band", ""))
    lab_control_band = _safe_text(row.get("lab_control_band", ""))
    lab_pressure_band = _safe_text(row.get("lab_pressure_band", ""))
    lab_matchup_profile = _safe_text(row.get("lab_matchup_profile", ""))
    lab_basis_hint = _safe_text(row.get("lab_basis_hint", ""))
    lab_state_notes = _safe_text(row.get("lab_state_notes", ""))
    has_lab_distribution = not any(pd.isna(v) for v in [lab_hold_weight, lab_stall_weight, lab_flip_weight, lab_volatility_score, lab_draw_tension_score])
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

    risk_level = str(purchase_layer["risk_level"])
    ticket_guidance = str(purchase_layer["ticket_guidance"])

    summary_parts: List[str] = []
    summary_parts.append(str(purchase_layer["purchase_summary"]))
    if has_lab_distribution:
        dominant = max(
            [("hold", lab_hold_weight), ("stall", lab_stall_weight), ("flip", lab_flip_weight)],
            key=lambda kv: kv[1],
        )[0]
        summary_parts.append(f"lab={dominant}")
        summary_parts.append(f"hold={lab_hold_weight:.2f}")
        summary_parts.append(f"stall={lab_stall_weight:.2f}")
        summary_parts.append(f"flip={lab_flip_weight:.2f}")
        summary_parts.append(f"vol={lab_volatility_score:.2f}")
        summary_parts.append(f"dt={lab_draw_tension_score:.2f}")
        if lab_tempo_band:
            summary_parts.append(f"tempo={lab_tempo_band}")
        if lab_matchup_profile and lab_matchup_profile != "balanced":
            summary_parts.append(f"profile={lab_matchup_profile}")
        if lab_basis_hint and lab_basis_hint != "basis_balanced":
            summary_parts.append(f"basis={lab_basis_hint}")
        if lab_control_band and lab_control_band != "neutral":
            summary_parts.append(f"control={lab_control_band}")
        if lab_pressure_band and lab_pressure_band != "neutral":
            summary_parts.append(f"pressure={lab_pressure_band}")
        if lab_state_notes:
            summary_parts.append(lab_state_notes)
    else:
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
        "draw_core_flag": bool(purchase_layer["draw_core_flag"]),
        "draw_purchase_tier": str(purchase_layer["draw_purchase_tier"]),
        "admission_policy": str(purchase_layer.get("admission_policy", "")),
        "rank1_symbol": str(purchase_layer.get("rank1_symbol", ranked[0][0])),
        "rank2_symbol": str(purchase_layer.get("rank2_symbol", ranked[1][0] if len(ranked) > 1 else "")),
        "rank3_symbol": str(purchase_layer.get("rank3_symbol", ranked[2][0] if len(ranked) > 2 else "")),
        "rank1_prob": float(purchase_layer.get("rank1_prob", ranked[0][1])),
        "rank2_prob": float(purchase_layer.get("rank2_prob", ranked[1][1] if len(ranked) > 1 else 0.0)),
        "rank3_prob": float(purchase_layer.get("rank3_prob", ranked[2][1] if len(ranked) > 2 else 0.0)),
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
        "match_purchase_type": row.get("match_purchase_type", ""),
        "match_purchase_subtype": row.get("match_purchase_subtype", ""),
        "purchase_type_confidence": row.get("purchase_type_confidence", ""),
        "purchase_type_reason": row.get("purchase_type_reason", ""),
        "anchor_candidate_flag": row.get("anchor_candidate_flag", ""),
        "anchor_candidate_score": row.get("anchor_candidate_score", ""),
        "anchor_candidate_reason": row.get("anchor_candidate_reason", ""),
        "draw_core_candidate_flag": row.get("draw_core_candidate_flag", ""),
        "draw_core_candidate_score": row.get("draw_core_candidate_score", ""),
        "draw_core_candidate_reason": row.get("draw_core_candidate_reason", ""),
        "away_overread_draw_cover_flag": row.get("away_overread_draw_cover_flag", ""),
        "away_overread_draw_cover_score": row.get("away_overread_draw_cover_score", ""),
        "away_overread_draw_cover_reason": row.get("away_overread_draw_cover_reason", ""),
        "legacy_direct_override_applied": row.get("legacy_direct_override_applied", ""),
        "legacy_direct_override_reason": row.get("legacy_direct_override_reason", ""),
        "predicted_result": row.get("predicted_result", ""),
        "state_strength_home": row.get("state_strength_base_home", ""),
        "state_attack_home": row.get("state_attack_level_home", ""),
        "state_defense_home": row.get("state_defense_level_home", ""),
        "state_form_home": row.get("state_form_level_home", ""),
        "state_availability_home": row.get("state_availability_level_home", ""),
        "state_fatigue_home": row.get("state_fatigue_level_home", ""),
        "state_motivation_home": row.get("state_motivation_level_home", ""),
        "state_stability_home": row.get("state_stability_level_home", ""),
        "state_strength_away": row.get("state_strength_base_away", ""),
        "state_attack_away": row.get("state_attack_level_away", ""),
        "state_defense_away": row.get("state_defense_level_away", ""),
        "state_form_away": row.get("state_form_level_away", ""),
        "state_availability_away": row.get("state_availability_level_away", ""),
        "state_fatigue_away": row.get("state_fatigue_level_away", ""),
        "state_motivation_away": row.get("state_motivation_level_away", ""),
        "state_stability_away": row.get("state_stability_level_away", ""),
        "lab_hold_weight": row.get("lab_hold_weight", ""),
        "lab_stall_weight": row.get("lab_stall_weight", ""),
        "lab_flip_weight": row.get("lab_flip_weight", ""),
        "lab_volatility_score": row.get("lab_volatility_score", ""),
        "lab_draw_tension_score": row.get("lab_draw_tension_score", ""),
        "lab_dynamic_swing_score": row.get("lab_dynamic_swing_score", ""),
        "lab_hold_foundation_score": row.get("lab_hold_foundation_score", ""),
        "lab_stall_compactness_score": row.get("lab_stall_compactness_score", ""),
        "lab_flip_dislocation_score": row.get("lab_flip_dislocation_score", ""),
        "lab_side_path_home_score": row.get("lab_side_path_home_score", ""),
        "lab_side_path_away_score": row.get("lab_side_path_away_score", ""),
        "lab_draw_support_score": row.get("lab_draw_support_score", ""),
        "lab_dispersion_score": row.get("lab_dispersion_score", ""),
        "lab_home_path_score": row.get("lab_home_path_score", ""),
        "lab_away_path_score": row.get("lab_away_path_score", ""),
        "lab_scenario_entropy_score": row.get("lab_scenario_entropy_score", ""),
        "lab_draw_path_score": row.get("lab_draw_path_score", ""),
        "lab_mix_home": row.get("lab_mix_home", ""),
        "lab_mix_draw": row.get("lab_mix_draw", ""),
        "lab_mix_away": row.get("lab_mix_away", ""),
        "lab_tempo_band": row.get("lab_tempo_band", ""),
        "lab_control_band": row.get("lab_control_band", ""),
        "lab_pressure_band": row.get("lab_pressure_band", ""),
        "lab_matchup_profile": row.get("lab_matchup_profile", ""),
        "lab_basis_hint": row.get("lab_basis_hint", ""),
        "lab_state_notes": row.get("lab_state_notes", ""),
        "match_type_primary": row.get("match_type_primary", ""),
        "match_type_flags": flags_raw,
        "draw_risk_flag": row.get("draw_risk_flag", ""),
        "match_type_lab_matchup_edge": row.get("match_type_lab_matchup_edge", ""),
        "match_type_lab_style_conflict": row.get("match_type_lab_style_conflict", ""),
        "match_type_lab_low_event": row.get("match_type_lab_low_event", ""),
    }


def build_buyplan_context(df: pd.DataFrame) -> pd.DataFrame:
    rows = [_build_context_row(row) for _, row in df.iterrows()]
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="purchase_reference/predictions.csv を節固定で再生成")
    p.add_argument("--season", type=int, required=True, help="例: 2026")
    p.add_argument(
        "--logical-season",
        type=int,
        default=None,
        help="toto節リスト上のseason。移行日程など予測seasonと異なる場合に指定",
    )
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
    j1_path = resolve_prediction_input_path(args.j1, "j1", args.season)
    j2_path = resolve_prediction_input_path(args.j2, "j2", args.season)
    out_csv = os.path.abspath(args.out)
    out_context_csv = os.path.abspath(args.out_context)
    j1_round = int(args.j1_round if args.j1_round is not None else args.round_no)
    j2_round = int(args.j2_round if args.j2_round is not None else args.round_no)

    j1_df = load_prediction_csv(j1_path, "J1", args.season, j1_round, args.logical_season)
    j2_df = load_prediction_csv(j2_path, "J2", args.season, j2_round, args.logical_season)
    # toto節リストはJ1_roundを開催回キーとして持つ。リーグごとの節番号が異なる回でも、
    # J2側を自身の節番号で再解決せず、J1側で確定した同一開催回IDへ統一する。
    toto_round_id = resolve_toto_round_id(
        args.logical_season if args.logical_season is not None else args.season,
        j1_round,
    )
    j1_df["toto_round_id"] = toto_round_id
    j2_df["toto_round_id"] = toto_round_id
    j1_df, j2_df = align_prediction_columns([j1_df, j2_df])
    out = sort_predictions(pd.concat([j1_df, j2_df], ignore_index=True))
    out = drop_legacy_variant_columns(out)

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
    print(f"[TOTO_ROUND] mixed-round unified={toto_round_id}")
    print(f"[SOURCE] j1={j1_path}")
    print(f"[SOURCE] j2={j2_path}")
    print(f"[OK] {out_context_csv} rows={len(context_df)}")


if __name__ == "__main__":
    main()
