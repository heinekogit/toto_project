#!/usr/bin/env python3
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKTEST_ROOT = os.path.join(ROOT_DIR, "data", "purchase_reference", "backtest")
EVAL_ROOT = os.path.join(ROOT_DIR, "data", "eval", "rounds")
J1_MASTER = os.path.join(ROOT_DIR, "backtest_j1_2026.csv")
J2_MASTER = os.path.join(ROOT_DIR, "backtest_j2_2026.csv")
ACTIVE_POLICIES = {"home_reversal_watch", "j2_draw_trap"}
OBSERVATION_FLAGS = [
    "away_overread_draw_cover_flag",
    "anchor_candidate_flag",
    "draw_core_candidate_flag",
]


def _safe_text(value: object) -> str:
    if isinstance(value, pd.Series):
        value = value.iloc[0] if not value.empty else ""
    if pd.isna(value):
        return ""
    text = str(value or "").strip()
    if text.lower() == "nan":
        return ""
    return text


def _safe_float(value: object, default: float = float("nan")) -> float:
    if isinstance(value, pd.Series):
        value = value.iloc[0] if not value.empty else default
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return default
    return float(x)


def _safe_bool(value: object) -> bool:
    if isinstance(value, pd.Series):
        value = value.iloc[0] if not value.empty else False
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)


def _sym(value: object) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    if s in {"H", "1", "HOME"}:
        return "1"
    if s in {"D", "0", "DRAW"}:
        return "0"
    if s in {"A", "2", "AWAY"}:
        return "2"
    return ""


def _norm_key(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


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


def compute_prob_shape(ph: float, pdw: float, pa: float) -> dict:
    items = [("H", ph), ("D", pdw), ("A", pa)]
    ranked = sorted(items, key=lambda kv: (-kv[1], kv[0]))
    best_label, best_prob = ranked[0]
    second_label, second_prob = ranked[1]
    return {
        "best_label": best_label,
        "second_label": second_label,
        "top_gap": float(best_prob - second_prob),
    }


def build_match_purchase_context(row: pd.Series) -> dict:
    ph = _safe_float(row.get("prob_home_win"))
    pdw = _safe_float(row.get("prob_draw"))
    pa = _safe_float(row.get("prob_away_win"))
    if not (math.isfinite(ph) and math.isfinite(pdw) and math.isfinite(pa)):
        return {
            "match_purchase_type": "",
            "match_purchase_subtype": "",
            "purchase_type_confidence": "",
            "purchase_type_reason": "",
            "draw_purchase_tier": "",
            "primary_pick_symbol": "",
            "secondary_pick_symbol": "",
            "anchor_candidate_flag": False,
            "anchor_candidate_score": 0.0,
            "anchor_candidate_reason": "",
            "draw_core_candidate_flag": False,
            "draw_core_candidate_score": 0.0,
            "draw_core_candidate_reason": "",
            "away_overread_draw_cover_flag": False,
            "away_overread_draw_cover_score": 0.0,
            "away_overread_draw_cover_reason": "",
        }

    pred = _safe_text(row.get("predicted_result")).upper()
    prob_shape = compute_prob_shape(ph, pdw, pa)
    top_gap = float(prob_shape["top_gap"])
    draw_gap = float(max(ph, pa) - pdw)
    best_label = str(prob_shape["best_label"])
    second_label = str(prob_shape["second_label"])

    best_side_label = "H" if ph >= pa else "A"
    best_side_symbol = _sym(best_side_label)
    pred_symbol = _sym(pred)
    second_symbol = _sym(second_label)
    hold_weight = _safe_float(row.get("lab_hold_weight"))
    stall_weight = _safe_float(row.get("lab_stall_weight"))
    flip_weight = _safe_float(row.get("lab_flip_weight"))
    mix_home = _safe_float(row.get("lab_mix_home"))
    mix_draw = _safe_float(row.get("lab_mix_draw"))
    mix_away = _safe_float(row.get("lab_mix_away"))
    entropy_score = _safe_float(row.get("lab_scenario_entropy_score"))
    volatility_score = _safe_float(row.get("lab_volatility_score"))
    draw_tension_score = _safe_float(row.get("lab_draw_tension_score"))
    compactness_score = _safe_float(row.get("lab_stall_compactness_score"))
    flip_dislocation_score = _safe_float(row.get("lab_flip_dislocation_score"))
    home_path_score = _safe_float(row.get("lab_home_path_score"))
    away_path_score = _safe_float(row.get("lab_away_path_score"))
    path_gap = abs(home_path_score - away_path_score) if math.isfinite(home_path_score) and math.isfinite(away_path_score) else 0.0
    side_gap = abs(ph - pa)

    def conf_label(score: float) -> str:
        if score >= 0.72:
            return "high"
        if score >= 0.56:
            return "medium"
        return "low"

    purchase_type = "chaos_spread"
    purchase_subtype = "entropy_high"
    purchase_conf = 0.40
    purchase_reason = f"fallback entropy={entropy_score:.3f} vol={volatility_score:.3f} flip={flip_weight:.3f}"

    if (
        stall_weight >= 0.44
        and draw_tension_score >= 0.58
        and compactness_score >= 0.52
        and pdw >= max(ph, pa) - 0.01
        and top_gap <= 0.05
    ):
        purchase_type = "compressed_draw"
        purchase_subtype = "stall_compact"
        purchase_conf = min(0.95, 0.40 * stall_weight + 0.30 * draw_tension_score + 0.30 * compactness_score)
        purchase_reason = f"stall={stall_weight:.3f} draw_tension={draw_tension_score:.3f} compact={compactness_score:.3f}"
    elif (
        max(mix_home, mix_away) >= 0.40
        and hold_weight >= 0.34
        and top_gap >= 0.06
        and side_gap >= 0.05
        and volatility_score <= 0.28
    ):
        purchase_type = "stable_directional"
        purchase_subtype = "home_path" if mix_home >= mix_away else "away_path"
        purchase_conf = min(0.95, 0.40 * max(mix_home, mix_away) + 0.35 * hold_weight + 0.25 * top_gap * 4.0)
        purchase_reason = f"mix={max(mix_home, mix_away):.3f} hold={hold_weight:.3f} top_gap={top_gap:.3f}"
    elif (
        flip_weight >= 0.33
        and volatility_score >= 0.24
        and top_gap <= 0.08
        and path_gap >= 0.08
    ):
        purchase_type = "asymmetric_swing"
        purchase_subtype = "home_swing" if home_path_score >= away_path_score else "away_swing"
        purchase_conf = min(0.95, 0.45 * flip_weight + 0.30 * volatility_score + 0.25 * min(1.0, path_gap))
        purchase_reason = f"flip={flip_weight:.3f} vol={volatility_score:.3f} path_gap={path_gap:.3f}"
    elif (
        pdw >= max(ph, pa) - 0.015
        and top_gap <= 0.03
        and draw_tension_score >= 0.52
        and (flip_dislocation_score >= 0.18 or volatility_score >= 0.22)
    ):
        purchase_type = "draw_trap"
        purchase_subtype = "weak_draw_side_bias"
        purchase_conf = min(0.95, 0.40 * draw_tension_score + 0.30 * max(flip_dislocation_score, volatility_score) + 0.30 * (0.03 - min(0.03, top_gap)) / 0.03)
        purchase_reason = f"top_gap={top_gap:.3f} draw_tension={draw_tension_score:.3f} flip_dislocation={flip_dislocation_score:.3f}"
    elif (
        entropy_score >= 0.46
        and volatility_score >= 0.24
        and top_gap <= 0.05
        and flip_weight >= 0.28
    ):
        purchase_type = "chaos_spread"
        purchase_subtype = "entropy_flip"
        purchase_conf = min(0.95, 0.35 * entropy_score + 0.35 * volatility_score + 0.30 * flip_weight)
        purchase_reason = f"entropy={entropy_score:.3f} vol={volatility_score:.3f} flip={flip_weight:.3f}"

    draw_core_score = _safe_float(row.get("draw_core_score"))
    swing_close_score = _safe_float(row.get("swing_close_score"))
    profile = _safe_text(row.get("close_split_profile")).lower()
    draw_purchase_tier = "side"
    primary_pick_symbol = pred_symbol or _sym(best_label)
    secondary_pick_symbol = second_symbol if top_gap < 0.08 else ""
    if pred == "D":
        draw_core_flag = bool(
            (pdw >= 0.40 and top_gap >= 0.06)
            or draw_core_score >= 0.64
            or (profile == "draw_core" and draw_gap <= -0.05 and top_gap >= 0.05)
        )
        draw_cut_flag = bool(
            (top_gap <= 0.025)
            or (draw_gap >= -0.030)
            or swing_close_score >= 0.58
            or profile in {"soft_swing", "swing_close", "undiff_close"}
        )
        if draw_core_flag:
            draw_purchase_tier = "core"
            primary_pick_symbol = "0"
            secondary_pick_symbol = best_side_symbol
        elif draw_cut_flag:
            draw_purchase_tier = "cut"
            primary_pick_symbol = best_side_symbol
            secondary_pick_symbol = "0"
        else:
            draw_purchase_tier = "rescue"
            primary_pick_symbol = "0"
            secondary_pick_symbol = best_side_symbol

    draw_core_candidate_score = (
        0.38 * max(0.0, min(1.0, mix_draw))
        + 0.22 * max(0.0, min(1.0, pdw))
        + 0.18 * max(0.0, min(1.0, draw_tension_score))
        + 0.12 * max(0.0, min(1.0, compactness_score))
        + 0.10 * max(0.0, 1.0 - min(1.0, path_gap * 2.0))
    )
    draw_core_candidate_flag = bool(
        pdw >= 0.36 and mix_draw >= 0.34 and draw_tension_score >= 0.54 and compactness_score >= 0.46
    )
    draw_core_candidate_reason = (
        f"pd={pdw:.3f}|mix_draw={mix_draw:.3f}|draw_tension={draw_tension_score:.3f}"
        f"|compact={compactness_score:.3f}|path_gap={path_gap:.3f}"
    )

    max_mix = max(mix_home, mix_draw, mix_away)
    mix_best_label = "H" if mix_home >= max(mix_draw, mix_away) else ("A" if mix_away >= max(mix_home, mix_draw) else "D")
    anchor_candidate_score = (
        0.30 * max(0.0, min(1.0, hold_weight))
        + 0.20 * max(0.0, min(1.0, top_gap * 4.0))
        + 0.18 * max(0.0, 1.0 - min(1.0, entropy_score))
        + 0.16 * max(0.0, 1.0 - min(1.0, volatility_score))
        + 0.16 * max(0.0, 1.0 - min(1.0, draw_tension_score))
    )
    anchor_candidate_flag = bool(
        pred in {"H", "A"}
        and mix_best_label == pred
        and hold_weight >= 0.34
        and top_gap >= 0.05
        and entropy_score <= 0.45
        and volatility_score <= 0.32
        and draw_tension_score <= 0.44
    )
    anchor_candidate_reason = (
        f"pred={pred}|mix_best={mix_best_label}|hold={hold_weight:.3f}|top_gap={top_gap:.3f}"
        f"|entropy={entropy_score:.3f}|vol={volatility_score:.3f}|draw_tension={draw_tension_score:.3f}"
    )

    away_overread_draw_cover_score = (
        0.20 * max(0.0, min(1.0, pdw))
        + 0.14 * max(0.0, min(1.0, 1.0 - pa))
        + 0.17 * max(0.0, min(1.0, flip_dislocation_score))
        + 0.17 * max(0.0, min(1.0, entropy_score))
        + 0.12 * max(0.0, min(1.0, away_path_score))
        + 0.10 * max(0.0, min(1.0, side_gap))
        + 0.10 * max(0.0, min(1.0, top_gap * 5.0))
    )
    away_overread_draw_cover_flag = bool(
        pred == "A"
        and pdw >= 0.34
        and pa <= 0.43
        and side_gap >= 0.14
        and top_gap >= 0.05
        and flip_dislocation_score >= 0.24
        and entropy_score >= 0.50
        and away_path_score >= 0.56
    )
    away_overread_draw_cover_reason = (
        f"pred=A|pd={pdw:.3f}|pa={pa:.3f}|side_gap={side_gap:.3f}|top_gap={top_gap:.3f}"
        f"|flip={flip_dislocation_score:.3f}|entropy={entropy_score:.3f}|away_path={away_path_score:.3f}"
    )

    return {
        "match_purchase_type": purchase_type,
        "match_purchase_subtype": purchase_subtype,
        "purchase_type_confidence": conf_label(purchase_conf),
        "purchase_type_reason": purchase_reason,
        "draw_purchase_tier": draw_purchase_tier,
        "primary_pick_symbol": primary_pick_symbol,
        "secondary_pick_symbol": secondary_pick_symbol,
        "anchor_candidate_flag": anchor_candidate_flag,
        "anchor_candidate_score": anchor_candidate_score,
        "anchor_candidate_reason": anchor_candidate_reason,
        "draw_core_candidate_flag": draw_core_candidate_flag,
        "draw_core_candidate_score": draw_core_candidate_score,
        "draw_core_candidate_reason": draw_core_candidate_reason,
        "away_overread_draw_cover_flag": away_overread_draw_cover_flag,
        "away_overread_draw_cover_score": away_overread_draw_cover_score,
        "away_overread_draw_cover_reason": away_overread_draw_cover_reason,
    }


def load_master_predictions() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for league, path in [("J1", J1_MASTER), ("J2", J2_MASTER)]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, keep_default_na=False)
        df["league"] = league.upper()
        df["home_team_key"] = df["home_team"].map(_norm_key)
        df["away_team_key"] = df["away_team"].map(_norm_key)
        df["round_no"] = df["節"].map(extract_round_number)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if not REQUIRED_SNAPSHOT_COLS.issubset(set(out.columns)):
        ctx = out.apply(build_match_purchase_context, axis=1, result_type="expand")
        ctx = ctx[[c for c in ctx.columns if c not in out.columns]]
        out = pd.concat([out, ctx], axis=1)
    return out


def load_snapshot_predictions(path: str, round_no: int) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    if "league" in df.columns:
        df["league"] = df["league"].astype(str).str.upper()
    df["home_team_key"] = df["home_team"].map(_norm_key)
    df["away_team_key"] = df["away_team"].map(_norm_key)
    df["round_no"] = round_no
    if not REQUIRED_SNAPSHOT_COLS.issubset(set(df.columns)):
        ctx = df.apply(build_match_purchase_context, axis=1, result_type="expand")
        ctx = ctx[[c for c in ctx.columns if c not in df.columns]]
        df = pd.concat([df, ctx], axis=1)
    return df


@dataclass
class RoundPaths:
    round_name: str
    round_no: int
    buyplan_csv: str
    scored_csv: str
    snapshot_csv: str


REQUIRED_SNAPSHOT_COLS = {
    "match_purchase_type",
    "match_purchase_subtype",
    "purchase_type_confidence",
    "purchase_type_reason",
    "lab_hold_weight",
    "lab_stall_weight",
    "lab_flip_weight",
    "lab_mix_home",
    "lab_mix_draw",
    "lab_mix_away",
    "anchor_candidate_flag",
    "anchor_candidate_score",
    "anchor_candidate_reason",
    "draw_core_candidate_flag",
    "draw_core_candidate_score",
    "draw_core_candidate_reason",
    "away_overread_draw_cover_flag",
    "away_overread_draw_cover_score",
    "away_overread_draw_cover_reason",
}
MIN_SNAPSHOT_COLS = {
    "home_team",
    "away_team",
    "predicted_result",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
}


def snapshot_is_usable(path: str) -> bool:
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    cols = set(df.columns)
    return MIN_SNAPSHOT_COLS.issubset(cols)


def iter_rounds() -> Iterable[RoundPaths]:
    for name in sorted(os.listdir(BACKTEST_ROOT)):
        if not re.fullmatch(r"round\d+", name):
            continue
        buyplan_csv = os.path.join(BACKTEST_ROOT, name, "buyplan.csv")
        scored_csv = os.path.join(BACKTEST_ROOT, name, "buyplan_scored.csv")
        snapshot_csv = os.path.join(EVAL_ROOT, name, "snapshot", "predictions.csv")
        if not (os.path.exists(buyplan_csv) and os.path.exists(scored_csv) and os.path.exists(snapshot_csv)):
            continue
        if not snapshot_is_usable(snapshot_csv):
            continue
        yield RoundPaths(name, int(name.replace("round", "")), buyplan_csv, scored_csv, snapshot_csv)


def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    def _agg(group: pd.DataFrame) -> pd.Series:
        actual_counts = group["actual_symbol"].value_counts()
        return pd.Series(
            {
                "matches": int(len(group)),
                "actual_h_count": int(actual_counts.get("1", 0)),
                "actual_d_count": int(actual_counts.get("0", 0)),
                "actual_a_count": int(actual_counts.get("2", 0)),
                "actual_h_rate": float((group["actual_symbol"] == "1").mean()),
                "actual_d_rate": float((group["actual_symbol"] == "0").mean()),
                "actual_a_rate": float((group["actual_symbol"] == "2").mean()),
                "main_hit_count": int(group["main_hit"].sum()),
                "main_hit_rate": float(group["main_hit"].mean()),
                "buyplan_change_count": int(group["buyplan_change_count"].sum()),
                "avg_buyplan_change_count": float(group["buyplan_change_count"].mean()),
                "rescue_count": int(group["rescue_count"].sum()),
                "harm_count": int(group["harm_count"].sum()),
                "net_rescue": int(group["net_rescue"].sum()),
                "avg_net_rescue": float(group["net_rescue"].mean()),
                "all_same_count": int(group["all_same_exact"].sum()),
                "all_same_rate": float(group["all_same_exact"].mean()),
                "all_same_side_count": int(group["all_same_side"].sum()),
                "all_same_side_rate": float(group["all_same_side"].mean()),
            }
        )

    out = df.groupby(group_cols, dropna=False).apply(_agg).reset_index()
    return out.sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    master = load_master_predictions()

    rows: List[dict] = []
    ticket_cols = [f"ticket{i:02d}" for i in range(1, 11)]
    ticket_cols = [c for c in ticket_cols if True]
    for paths in iter_rounds():
        buyplan_df = pd.read_csv(paths.buyplan_csv)
        buyplan_df["league"] = buyplan_df["league"].astype(str).str.upper()
        buyplan_df["home_team_key"] = buyplan_df["home_team"].map(_norm_key)
        buyplan_df["away_team_key"] = buyplan_df["away_team"].map(_norm_key)
        scored_df = pd.read_csv(paths.scored_csv)
        snapshot_df = load_snapshot_predictions(paths.snapshot_csv, paths.round_no)
        merged = buyplan_df.merge(
            scored_df[["match_no", "actual_symbol", "actual_result"] + [c for c in scored_df.columns if c.endswith("_is_hit")]],
            on="match_no",
            how="left",
        )
        merged = merged.merge(
            snapshot_df,
            on=["league", "home_team_key", "away_team_key"],
            how="left",
            suffixes=("", "_snapshot"),
        )
        if not master.empty:
            round_master = master[master["round_no"] == paths.round_no].copy()
            merged = merged.merge(
                round_master,
                on=["league", "home_team_key", "away_team_key"],
                how="left",
                suffixes=("", "_master"),
            )
            for col in [
                "lab_hold_weight",
                "lab_stall_weight",
                "lab_flip_weight",
                "lab_mix_home",
                "lab_mix_draw",
                "lab_mix_away",
                "lab_scenario_entropy_score",
                "lab_volatility_score",
                "lab_draw_tension_score",
                "lab_stall_compactness_score",
                "lab_flip_dislocation_score",
                "lab_home_path_score",
                "lab_away_path_score",
                "match_purchase_type",
                "match_purchase_subtype",
                "purchase_type_confidence",
                "purchase_type_reason",
                "draw_purchase_tier",
                "primary_pick_symbol",
                "secondary_pick_symbol",
                "predicted_result_main",
                "predicted_result",
                "anchor_candidate_flag",
                "anchor_candidate_score",
                "anchor_candidate_reason",
                "draw_core_candidate_flag",
                "draw_core_candidate_score",
                "draw_core_candidate_reason",
                "away_overread_draw_cover_flag",
                "away_overread_draw_cover_score",
                "away_overread_draw_cover_reason",
            ]:
                master_col = f"{col}_master"
                if master_col in merged.columns:
                    valid_master = merged[master_col].notna() & (merged[master_col].astype(str) != "")
                    merged[col] = merged[master_col].where(valid_master, merged.get(col))
        for _, row in merged.iterrows():
            actual_symbol = _sym(row.get("actual_symbol") or row.get("actual_result"))
            main_symbol = _sym(row.get("predicted_result_main") or row.get("predicted_result"))
            ticket01 = _sym(row.get("ticket01"))
            ticket_picks = [_sym(row.get(col)) for col in ticket_cols]
            ticket_picks = [pick for pick in ticket_picks if pick]
            unique_ticket_picks = sorted(set(ticket_picks))
            all_same_exact = bool(ticket_picks and len(unique_ticket_picks) == 1)
            all_same_side = bool(all_same_exact and unique_ticket_picks[0] in {"1", "2"})
            change_count = 0
            rescue_count = 0
            harm_count = 0
            base_hit = int(actual_symbol != "" and ticket01 == actual_symbol)
            for idx in range(2, 11):
                col = f"ticket{idx:02d}"
                pick = _sym(row.get(col))
                if not pick:
                    continue
                if pick != ticket01:
                    change_count += 1
                if pick != ticket01 and actual_symbol:
                    if pick == actual_symbol and ticket01 != actual_symbol:
                        rescue_count += 1
                    elif pick != actual_symbol and ticket01 == actual_symbol:
                        harm_count += 1
            rows.append(
                {
                    "round": paths.round_name,
                    "round_no": paths.round_no,
                    "match_no": int(row["match_no"]),
                    "league": _safe_text(row.get("league")),
                    "home_team": _safe_text(row.get("home_team")),
                    "away_team": _safe_text(row.get("away_team")),
                    "match_purchase_type": _safe_text(row.get("match_purchase_type")) or "unclassified",
                    "match_purchase_subtype": _safe_text(row.get("match_purchase_subtype")) or "unclassified",
                    "purchase_type_confidence": _safe_text(row.get("purchase_type_confidence")) or "unknown",
                    "purchase_type_reason": _safe_text(row.get("purchase_type_reason")),
                    "policy_status": "active_policy"
                    if _safe_text(row.get("match_purchase_type")) in ACTIVE_POLICIES
                    else "observed_type",
                    "draw_purchase_tier": _safe_text(row.get("draw_purchase_tier")),
                    "predicted_result_main": _safe_text(row.get("predicted_result_main")),
                    "predicted_result": _safe_text(row.get("predicted_result")),
                    "main_symbol": main_symbol,
                    "actual_symbol": actual_symbol,
                    "actual_result": _safe_text(row.get("actual_result")),
                    "main_hit": int(actual_symbol != "" and main_symbol == actual_symbol),
                    "ticket01_symbol": ticket01,
                    "ticket01_hit": base_hit,
                    "buyplan_change_count": int(change_count),
                    "rescue_count": int(rescue_count),
                    "harm_count": int(harm_count),
                    "net_rescue": int(rescue_count - harm_count),
                    "all_same_exact": int(all_same_exact),
                    "all_same_side": int(all_same_side),
                    "anchor_candidate_flag": int(_safe_bool(row.get("anchor_candidate_flag"))),
                    "anchor_candidate_score": _safe_float(row.get("anchor_candidate_score")),
                    "anchor_candidate_reason": _safe_text(row.get("anchor_candidate_reason")),
                    "draw_core_candidate_flag": int(_safe_bool(row.get("draw_core_candidate_flag"))),
                    "draw_core_candidate_score": _safe_float(row.get("draw_core_candidate_score")),
                    "draw_core_candidate_reason": _safe_text(row.get("draw_core_candidate_reason")),
                    "away_overread_draw_cover_flag": int(_safe_bool(row.get("away_overread_draw_cover_flag"))),
                    "away_overread_draw_cover_score": _safe_float(row.get("away_overread_draw_cover_score")),
                    "away_overread_draw_cover_reason": _safe_text(row.get("away_overread_draw_cover_reason")),
                }
            )

    detail_df = pd.DataFrame(rows)
    outdir = BACKTEST_ROOT
    os.makedirs(outdir, exist_ok=True)

    summary_df = aggregate(detail_df, ["match_purchase_type"])
    subtype_df = aggregate(detail_df, ["match_purchase_type", "match_purchase_subtype"])
    league_df = aggregate(detail_df, ["league", "match_purchase_type"])
    confidence_df = aggregate(detail_df, ["purchase_type_confidence", "match_purchase_type"])
    policy_df = aggregate(detail_df, ["policy_status", "match_purchase_type"])

    flag_rows: List[pd.DataFrame] = []
    for flag in OBSERVATION_FLAGS:
        flagged = detail_df[detail_df[flag] == 1].copy()
        if flagged.empty:
            continue
        agg = aggregate(flagged, ["league"])
        agg.insert(0, "flag_name", flag)
        agg["avg_flag_score"] = float("nan")
        score_col = flag.replace("_flag", "_score")
        if score_col in flagged.columns:
            score_map = flagged.groupby("league")[score_col].mean()
            agg["avg_flag_score"] = agg["league"].map(score_map)
        flag_rows.append(agg)
    flag_df = pd.concat(flag_rows, ignore_index=True) if flag_rows else pd.DataFrame()

    flag_summary_rows: List[dict] = []
    for flag in OBSERVATION_FLAGS:
        flagged = detail_df[detail_df[flag] == 1].copy()
        score_col = flag.replace("_flag", "_score")
        reason_col = flag.replace("_flag", "_reason")
        flag_summary_rows.append(
            {
                "flag_name": flag,
                "matches": int(len(flagged)),
                "actual_h_count": int((flagged["actual_symbol"] == "1").sum()),
                "actual_d_count": int((flagged["actual_symbol"] == "0").sum()),
                "actual_a_count": int((flagged["actual_symbol"] == "2").sum()),
                "actual_h_rate": float((flagged["actual_symbol"] == "1").mean()) if not flagged.empty else float("nan"),
                "actual_d_rate": float((flagged["actual_symbol"] == "0").mean()) if not flagged.empty else float("nan"),
                "actual_a_rate": float((flagged["actual_symbol"] == "2").mean()) if not flagged.empty else float("nan"),
                "main_hit_rate": float(flagged["main_hit"].mean()) if not flagged.empty else float("nan"),
                "buyplan_change_count": int(flagged["buyplan_change_count"].sum()),
                "rescue_count": int(flagged["rescue_count"].sum()),
                "harm_count": int(flagged["harm_count"].sum()),
                "net_rescue": int(flagged["net_rescue"].sum()),
                "all_same_count": int(flagged["all_same_exact"].sum()),
                "all_same_side_count": int(flagged["all_same_side"].sum()),
                "avg_flag_score": float(flagged[score_col].mean()) if score_col in flagged.columns and not flagged.empty else float("nan"),
                "sample_reason": _safe_text(flagged[reason_col].iloc[0]) if reason_col in flagged.columns and not flagged.empty else "",
            }
        )
    flag_summary_df = pd.DataFrame(flag_summary_rows)

    all_same_rows: List[dict] = []
    all_same_df = detail_df[detail_df["all_same_exact"] == 1].copy()
    for group_name, group_df in [
        ("all_same_exact", all_same_df),
        ("all_same_side", detail_df[detail_df["all_same_side"] == 1].copy()),
    ]:
        for flag in OBSERVATION_FLAGS:
            overlap = group_df[group_df[flag] == 1].copy()
            all_same_rows.append(
                {
                    "overlap_group": group_name,
                    "target_name": flag,
                    "target_kind": "flag",
                    "overlap_count": int(len(overlap)),
                    "overlap_rate_within_group": float(len(overlap) / len(group_df)) if len(group_df) else float("nan"),
                    "main_hit_rate": float(overlap["main_hit"].mean()) if not overlap.empty else float("nan"),
                    "actual_h_rate": float((overlap["actual_symbol"] == "1").mean()) if not overlap.empty else float("nan"),
                    "actual_d_rate": float((overlap["actual_symbol"] == "0").mean()) if not overlap.empty else float("nan"),
                    "actual_a_rate": float((overlap["actual_symbol"] == "2").mean()) if not overlap.empty else float("nan"),
                }
            )
        for purchase_type in sorted(detail_df["match_purchase_type"].dropna().unique().tolist()):
            overlap = group_df[group_df["match_purchase_type"] == purchase_type].copy()
            all_same_rows.append(
                {
                    "overlap_group": group_name,
                    "target_name": purchase_type,
                    "target_kind": "purchase_type",
                    "overlap_count": int(len(overlap)),
                    "overlap_rate_within_group": float(len(overlap) / len(group_df)) if len(group_df) else float("nan"),
                    "main_hit_rate": float(overlap["main_hit"].mean()) if not overlap.empty else float("nan"),
                    "actual_h_rate": float((overlap["actual_symbol"] == "1").mean()) if not overlap.empty else float("nan"),
                    "actual_d_rate": float((overlap["actual_symbol"] == "0").mean()) if not overlap.empty else float("nan"),
                    "actual_a_rate": float((overlap["actual_symbol"] == "2").mean()) if not overlap.empty else float("nan"),
                }
            )
    all_same_overlap_df = pd.DataFrame(all_same_rows)

    summary_path = os.path.join(outdir, "purchase_type_summary.csv")
    detail_path = os.path.join(outdir, "purchase_type_detail.csv")
    league_path = os.path.join(outdir, "purchase_type_by_league.csv")
    confidence_path = os.path.join(outdir, "purchase_type_by_confidence.csv")
    policy_path = os.path.join(outdir, "purchase_type_policy_summary.csv")
    flag_path = os.path.join(outdir, "purchase_type_by_flag.csv")
    flag_summary_path = os.path.join(outdir, "purchase_type_flag_summary.csv")
    all_same_path = os.path.join(outdir, "purchase_type_all_same_overlap.csv")

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    subtype_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    league_df.to_csv(league_path, index=False, encoding="utf-8-sig")
    confidence_df.to_csv(confidence_path, index=False, encoding="utf-8-sig")
    policy_df.to_csv(policy_path, index=False, encoding="utf-8-sig")
    flag_df.to_csv(flag_path, index=False, encoding="utf-8-sig")
    flag_summary_df.to_csv(flag_summary_path, index=False, encoding="utf-8-sig")
    all_same_overlap_df.to_csv(all_same_path, index=False, encoding="utf-8-sig")

    print(f"[OK] {summary_path}")
    print(f"[OK] {detail_path}")
    print(f"[OK] {league_path}")
    print(f"[OK] {confidence_path}")
    print(f"[OK] {policy_path}")
    print(f"[OK] {flag_path}")
    print(f"[OK] {flag_summary_path}")
    print(f"[OK] {all_same_path}")
    print(f"[INFO] rows={len(detail_df)} rounds={detail_df['round'].nunique() if not detail_df.empty else 0}")


if __name__ == "__main__":
    main()
