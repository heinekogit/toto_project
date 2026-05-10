#!/usr/bin/env python3
import argparse
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


SYMBOLS = ["1", "0", "2"]  # home/draw/away
REQUIRED_MATCH_COUNT = 13
REQUIRED_TICKET_COUNT = 10
UNIQUE_VARIATION_TOPK = int(os.environ.get("BUYPLAN_UNIQUE_TOPK", "8"))
DEBUG_PROBS = os.environ.get("BUYPLAN_DEBUG_PROBS", "0") == "1"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_TOTO_ORDER_CSV = os.path.join(BASE_DIR, "data", "manual", "toto節リスト.csv")
LEGACY_TOTO_ORDER_CSV = os.path.join(BASE_DIR, "data", "manual", "toto並び順.csv")
LOCK_BY_PBEST_MIN = float(os.environ.get("LOCK_BY_PBEST_MIN", "0.62"))
LOCK_BY_MARGIN_MIN = float(os.environ.get("LOCK_BY_MARGIN_MIN", "0.18"))
BUYPLAN_2AXIS_DRAW_ENABLE = os.environ.get("BUYPLAN_2AXIS_DRAW_ENABLE", "1") == "1"
BUYPLAN_2AXIS_C_DIFF = float(os.environ.get("BUYPLAN_2AXIS_C_DIFF", "0.15"))
BUYPLAN_2AXIS_D_MIN = float(os.environ.get("BUYPLAN_2AXIS_D_MIN", "0.26"))
BUYPLAN_2AXIS_W_DRAW = float(os.environ.get("BUYPLAN_2AXIS_W_DRAW", "0.35"))
BUYPLAN_2AXIS_MAX_STRONG = float(os.environ.get("BUYPLAN_2AXIS_MAX_STRONG", "0.58"))
BUYPLAN_2AXIS_CAP_DEFAULT = int(os.environ.get("BUYPLAN_2AXIS_CAP_D_MATCHES", "3"))
BUYPLAN_2AXIS_CAP_LOCK = int(os.environ.get("BUYPLAN_2AXIS_CAP_D_MATCHES_LOCK", "2"))
BUYPLAN_2AXIS_CAP_PROB = int(os.environ.get("BUYPLAN_2AXIS_CAP_D_MATCHES_PROB", "3"))
BUYPLAN_2AXIS_CAP_EXP = int(os.environ.get("BUYPLAN_2AXIS_CAP_D_MATCHES_EXP", "4"))
BUYPLAN_2AXIS_W_DRAW_LOCK = float(os.environ.get("BUYPLAN_2AXIS_W_DRAW_LOCK", str(BUYPLAN_2AXIS_W_DRAW)))
BUYPLAN_2AXIS_W_DRAW_PROB = float(os.environ.get("BUYPLAN_2AXIS_W_DRAW_PROB", str(BUYPLAN_2AXIS_W_DRAW)))
BUYPLAN_2AXIS_W_DRAW_EXP = float(os.environ.get("BUYPLAN_2AXIS_W_DRAW_EXP", str(BUYPLAN_2AXIS_W_DRAW)))
DRAW_BOOST = float(os.environ.get("BUYPLAN_DRAW_BOOST", "1.08"))
DRAW_BOOST_CLOSE = float(os.environ.get("BUYPLAN_DRAW_BOOST_CLOSE", "1.12"))
DRAW_BOOST_MARGIN_MAX = float(os.environ.get("BUYPLAN_DRAW_BOOST_MARGIN_MAX", "0.03"))
MARGIN_D_MAX = float(os.environ.get("BUYPLAN_MARGIN_D_MAX", "0.08"))
ENTROPY_MIN = float(os.environ.get("BUYPLAN_ENTROPY_MIN", "1.05"))
BEST_MAX_FOR_D = float(os.environ.get("BUYPLAN_BEST_MAX_FOR_D", "0.48"))
DRAW_MATCH_CAP = int(os.environ.get("BUYPLAN_DRAW_MATCH_CAP", "4"))
ZERO_RATIO_CAP = float(os.environ.get("BUYPLAN_ZERO_RATIO_CAP", "0.33"))
PER_MATCH_SAME_SYMBOL_CAP = int(os.environ.get("BUYPLAN_PER_MATCH_SAME_SYMBOL_CAP", "8"))
BUYPLAN_BALANCE_T_DRAW = float(os.environ.get("BUYPLAN_BALANCE_T_DRAW", "0.06"))
BUYPLAN_BALANCE_D_MIN = float(os.environ.get("BUYPLAN_BALANCE_D_MIN", "0.20"))
LOCK02_MARGIN_THRESHOLD = float(os.environ.get("BUYPLAN_LOCK02_MARGIN_THRESHOLD", "0.04"))
LOCK03_MARGIN_THRESHOLD = float(os.environ.get("BUYPLAN_LOCK03_MARGIN_THRESHOLD", "0.06"))
LOCK02_MAX_FLIPS = int(os.environ.get("BUYPLAN_LOCK02_MAX_FLIPS", "2"))
LOCK03_MAX_FLIPS = int(os.environ.get("BUYPLAN_LOCK03_MAX_FLIPS", "2"))
ENABLE_EXTREME_MARGIN_RELEASE = os.environ.get("BUYPLAN_ENABLE_EXTREME_MARGIN_RELEASE", "1") == "1"
EXTREME_MARGIN_RELEASE_THRESHOLD = float(os.environ.get("BUYPLAN_EXTREME_MARGIN_RELEASE_THRESHOLD", "0.025"))
EXTREME_MARGIN_RELEASE_MIN_ALT_TICKETS = int(os.environ.get("BUYPLAN_EXTREME_MARGIN_RELEASE_MIN_ALT_TICKETS", "2"))
STRONG_BREAK_MARGIN_THRESHOLD = float(os.environ.get("BUYPLAN_STRONG_BREAK_MARGIN_THRESHOLD", "0.10"))
STRONG_BREAK_MARGIN_MIN = float(os.environ.get("BUYPLAN_STRONG_BREAK_MARGIN_MIN", "0.04"))
STRONG_BREAK_MARGIN_MAX = float(os.environ.get("BUYPLAN_STRONG_BREAK_MARGIN_MAX", "0.10"))
STRONG_BREAK_TARGET_COUNT = int(os.environ.get("BUYPLAN_STRONG_BREAK_TARGET_COUNT", "2"))
STRONG_BREAK_W_MARGIN = float(os.environ.get("BUYPLAN_STRONG_BREAK_W_MARGIN", "0.35"))
STRONG_BREAK_W_AWAY = float(os.environ.get("BUYPLAN_STRONG_BREAK_W_AWAY", "0.30"))
STRONG_BREAK_W_SECOND = float(os.environ.get("BUYPLAN_STRONG_BREAK_W_SECOND", "0.20"))
STRONG_BREAK_W_VOLATILITY = float(os.environ.get("BUYPLAN_STRONG_BREAK_W_VOLATILITY", "0.15"))
WEAK_DRAW_MARGIN = float(os.environ.get("BUYPLAN_WEAK_DRAW_MARGIN", "0.10"))
WEAK_DRAW_ENTROPY_MIN = float(os.environ.get("BUYPLAN_WEAK_DRAW_ENTROPY_MIN", "1.09"))
J2_WEAK_DRAW_MARGIN_MAX = float(os.environ.get("BUYPLAN_J2_WEAK_DRAW_MARGIN_MAX", "0.050"))
J2_WEAK_DRAW_PD_MIN = float(os.environ.get("BUYPLAN_J2_WEAK_DRAW_PD_MIN", "0.33"))
J2_WEAK_DRAW_ENTROPY_MIN = float(os.environ.get("BUYPLAN_J2_WEAK_DRAW_ENTROPY_MIN", "1.08"))
J2_WEAK_DRAW_RATIO31_MIN = float(os.environ.get("BUYPLAN_J2_WEAK_DRAW_RATIO31_MIN", "0.70"))
DRAW_SCORE_W_CLOSENESS = float(os.environ.get("BUYPLAN_DRAW_SCORE_W_CLOSENESS", "0.45"))
DRAW_SCORE_W_ENTROPY = float(os.environ.get("BUYPLAN_DRAW_SCORE_W_ENTROPY", "0.25"))
DRAW_SCORE_W_MARGIN = float(os.environ.get("BUYPLAN_DRAW_SCORE_W_MARGIN", "0.30"))
DRAW_SCORE_MIN_LOCK = float(os.environ.get("BUYPLAN_DRAW_SCORE_MIN_LOCK", "0.42"))
DRAW_SCORE_MIN_PROB = float(os.environ.get("BUYPLAN_DRAW_SCORE_MIN_PROB", "0.36"))
DRAW_SCORE_MIN_EXP = float(os.environ.get("BUYPLAN_DRAW_SCORE_MIN_EXP", "0.30"))
TARGET_DRAW_MIN_LOCK = int(os.environ.get("BUYPLAN_TARGET_DRAW_MIN_LOCK", "2"))
TARGET_DRAW_MAX_LOCK = int(os.environ.get("BUYPLAN_TARGET_DRAW_MAX_LOCK", "3"))
TARGET_DRAW_MIN_PROB = int(os.environ.get("BUYPLAN_TARGET_DRAW_MIN_PROB", "3"))
TARGET_DRAW_MAX_PROB = int(os.environ.get("BUYPLAN_TARGET_DRAW_MAX_PROB", "4"))
TARGET_DRAW_MIN_EXP = int(os.environ.get("BUYPLAN_TARGET_DRAW_MIN_EXP", "4"))
TARGET_DRAW_MAX_EXP = int(os.environ.get("BUYPLAN_TARGET_DRAW_MAX_EXP", "5"))
REL_CLOSE_RATIO_TO_TOP_MIN = float(os.environ.get("BUYPLAN_REL_CLOSE_RATIO_TO_TOP_MIN", "0.80"))
REL_CLOSE_GAP_TO_ABOVE_MAX = float(os.environ.get("BUYPLAN_REL_CLOSE_GAP_TO_ABOVE_MAX", "0.06"))
REL_CLOSE_SPREAD_MAX = float(os.environ.get("BUYPLAN_REL_CLOSE_SPREAD_MAX", "0.12"))
RELATIVE_SCORE_ALPHA = float(os.environ.get("BUYPLAN_RELATIVE_SCORE_ALPHA", "0.50"))
BUYPLAN_BASE_MODE = os.environ.get("BUYPLAN_BASE_MODE", "shape_relative").strip().lower()
if BUYPLAN_BASE_MODE not in {"balance_ha", "tri_argmax", "topgap", "shape_relative"}:
    BUYPLAN_BASE_MODE = "shape_relative"
BASE_TOP_GAP_STRONG = float(os.environ.get("BUYPLAN_BASE_TOP_GAP_STRONG", "0.08"))
BASE_TOP_GAP_MID = float(os.environ.get("BUYPLAN_BASE_TOP_GAP_MID", "0.04"))
SMALL_GAP_TOP_GAP_MAX = float(os.environ.get("BUYPLAN_SMALL_GAP_TOP_GAP_MAX", "0.02"))
SMALL_GAP_CLOSENESS_MIN = float(os.environ.get("BUYPLAN_SMALL_GAP_CLOSENESS_MIN", "0.95"))
SMALL_GAP_ENTROPY_MIN = float(os.environ.get("BUYPLAN_SMALL_GAP_ENTROPY_MIN", "1.09"))
WEAK_DRAW_TOP_GAP_MAX = float(os.environ.get("BUYPLAN_WEAK_DRAW_TOP_GAP_MAX", "0.05"))
SHAPE_STRENGTH_STRONG = float(os.environ.get("BUYPLAN_SHAPE_STRENGTH_STRONG", "0.33"))
SHAPE_STRENGTH_MID = float(os.environ.get("BUYPLAN_SHAPE_STRENGTH_MID", "0.23"))
SHAPE_DRAW_RATIO31_MIN = float(os.environ.get("BUYPLAN_SHAPE_DRAW_RATIO31_MIN", "0.72"))
SHAPE_DRAW_SPREAD_MAX = float(os.environ.get("BUYPLAN_SHAPE_DRAW_SPREAD_MAX", "0.16"))
SHAPE_DRAW_ENTROPY_MIN = float(os.environ.get("BUYPLAN_SHAPE_DRAW_ENTROPY_MIN", "1.02"))
SMALL_GAP_STRENGTH_MAX = float(os.environ.get("BUYPLAN_SMALL_GAP_STRENGTH_MAX", "0.22"))
SMALL_GAP_RATIO31_MIN = float(os.environ.get("BUYPLAN_SMALL_GAP_RATIO31_MIN", "0.85"))
WEAK_DRAW_STRENGTH_MAX = float(os.environ.get("BUYPLAN_WEAK_DRAW_STRENGTH_MAX", "0.28"))
WEAK_DRAW_RATIO31_MIN = float(os.environ.get("BUYPLAN_WEAK_DRAW_RATIO31_MIN", "0.75"))
DRAW_BRANCH_HA_DIFF_MAX = float(os.environ.get("BUYPLAN_DRAW_BRANCH_HA_DIFF_MAX", "0.12"))
DRAW_BRANCH_RATIO31_MIN = float(os.environ.get("BUYPLAN_DRAW_BRANCH_RATIO31_MIN", "0.78"))
DRAW_BRANCH_ENTROPY_MIN = float(os.environ.get("BUYPLAN_DRAW_BRANCH_ENTROPY_MIN", "1.05"))
DRAW_BRANCH_PD_MIN = float(os.environ.get("BUYPLAN_DRAW_BRANCH_PD_MIN", "0.30"))
AWAY_VALUE_HA_DIFF_MAX = float(os.environ.get("BUYPLAN_AWAY_VALUE_HA_DIFF_MAX", "0.10"))
AWAY_VALUE_TOP_GAP_MAX = float(os.environ.get("BUYPLAN_AWAY_VALUE_TOP_GAP_MAX", "0.08"))
AWAY_VALUE_RATIO_TO_TOP_MIN = float(os.environ.get("BUYPLAN_AWAY_VALUE_RATIO_TO_TOP_MIN", "0.78"))
AWAY_VALUE_ENTROPY_MIN = float(os.environ.get("BUYPLAN_AWAY_VALUE_ENTROPY_MIN", "1.02"))
AWAY_VALUE_PD_MIN = float(os.environ.get("BUYPLAN_AWAY_VALUE_PD_MIN", "0.22"))
ENABLE_SMALL_GAP_RULE = os.environ.get("BUYPLAN_ENABLE_SMALL_GAP_RULE", "0") == "1"
ENABLE_WEAK_DRAW_APPLY = os.environ.get("BUYPLAN_ENABLE_WEAK_DRAW_APPLY", "1") == "1"
MAX_WEAK_DRAW_PER_MATCH = int(os.environ.get("BUYPLAN_MAX_WEAK_DRAW_PER_MATCH", "2"))
MAX_FLIPS_PER_MATCH_PROB = int(os.environ.get("BUYPLAN_MAX_FLIPS_PER_MATCH_PROB", "1"))
MAX_FLIPS_PER_MATCH_EXP = int(os.environ.get("BUYPLAN_MAX_FLIPS_PER_MATCH_EXP", "2"))
ENABLE_SAME_SYMBOL_CAP = os.environ.get("BUYPLAN_ENABLE_SAME_SYMBOL_CAP", "0") == "1"
ENABLE_FINAL_RANGE = os.environ.get("BUYPLAN_ENABLE_FINAL_RANGE", "0") == "1"
ENABLE_GRADUAL_SWAY = os.environ.get("BUYPLAN_ENABLE_GRADUAL_SWAY", "1") == "1"
ENABLE_PROB_DRAW_FLOOR = os.environ.get("BUYPLAN_ENABLE_PROB_DRAW_FLOOR", "1") == "1"
BUYPLAN_USE_PREDICTED_RESULT_BASE = os.environ.get("BUYPLAN_USE_PREDICTED_RESULT_BASE", "1") == "1"
SWAY_DEGREE_TABLE_RAW = os.environ.get(
    "BUYPLAN_SWAY_DEGREE_TABLE",
    "04:2,05:2,06:3,07:3",
)
ALL_SAME_SECOND_RATIO_TABLE_RAW = os.environ.get(
    "BUYPLAN_ALL_SAME_SECOND_RATIO_TABLE",
    "08:0.40,09:0.55,10:0.70",
)


def _parse_sway_degree_table(raw: str) -> Dict[int, int]:
    table: Dict[int, int] = {}
    for part in str(raw).split(","):
        token = part.strip()
        if not token or ":" not in token:
            continue
        k, v = token.split(":", 1)
        try:
            ticket_no = int(k.strip())
            degree = int(v.strip())
        except ValueError:
            continue
        if 4 <= ticket_no <= 7:
            table[ticket_no] = max(0, min(6, degree))
    if not table:
        table = {4: 2, 5: 2, 6: 3, 7: 3}
    return table


def _degree_to_profile(degree: int) -> Tuple[int, int]:
    d = max(0, int(degree))
    if d == 0:
        return (0, 0)
    if d == 1:
        return (1, 0)
    if d == 2:
        return (2, 0)
    if d == 3:
        return (3, 1)
    if d == 4:
        return (4, 1)
    # degree 5+ : stronger sway
    return (5, 2)


SWAY_DEGREE_TABLE = _parse_sway_degree_table(SWAY_DEGREE_TABLE_RAW)


def _parse_all_same_second_ratio_table(raw: str) -> Dict[int, float]:
    table: Dict[int, float] = {}
    for part in str(raw).split(","):
        token = part.strip()
        if not token or ":" not in token:
            continue
        k, v = token.split(":", 1)
        try:
            ticket_no = int(k.strip())
            ratio = float(v.strip())
        except ValueError:
            continue
        if 8 <= ticket_no <= 10:
            table[ticket_no] = max(0.0, min(1.0, ratio))
    # fallback defaults if parse fails
    if not table:
        table = {8: 0.40, 9: 0.55, 10: 0.70}
    return table


ALL_SAME_SECOND_RATIO_TABLE = _parse_all_same_second_ratio_table(ALL_SAME_SECOND_RATIO_TABLE_RAW)


def _predicted_result_to_symbol(value: object) -> str:
    token = _safe_text(value, "").upper()
    mapping = {
        "H": "1",
        "D": "0",
        "A": "2",
        "1": "1",
        "0": "0",
        "2": "2",
    }
    return mapping.get(token, "")


@dataclass(frozen=True)
class ScenarioDef:
    scenario_id: str
    scenario_name: str
    scenario_note: str


SCENARIO_DEFS: List[ScenarioDef] = [
    ScenarioDef("01", "System 01", "予想システム指定枠（main argmax）"),
    ScenarioDef("02", "System 02", "LAB展開反映票（type_b: hold/stall/flip を反映した別勝敗票）"),
    ScenarioDef("03", "System 03", "LAB Dグラデーション（type_c: 02準拠で弱いD揺れを補う派生票）"),
    ScenarioDef("04", "Prob Main", "本線寄り確率券（base維持優先、軽いsecondのみ）"),
    ScenarioDef("05", "Prob Draw", "D軸確率券（draw_cover優先、D本数3-4を維持）"),
    ScenarioDef("06", "Prob J2 Draw", "J2拮抗D補助券（ticket06のみdraw floor適用）"),
    ScenarioDef("07", "Prob Away", "A反転補助券（away_value候補優先、不要Dは抑制）"),
    ScenarioDef("08", "Prob Close Draw", "接戦D券（draw_risk / close系を優先してD寄せ）"),
    ScenarioDef("09", "Exp 01", "実験（best/second/third）"),
    ScenarioDef("10", "Exp 02", "実験（best/second/third）"),
]

SCENARIO_JA_BY_ID: Dict[str, str] = {
    "01": "予想指定",
    "02": "LAB展開",
    "03": "Dグラデ",
    "04": "本線確率",
    "05": "D軸確率",
    "06": "J2 D補助",
    "07": "A補助",
    "08": "接戦D",
    "09": "実験",
    "10": "実験",
}


@dataclass
class MatchPlan:
    match_no: int
    toto_round_id: str
    league: str
    home_team: str
    away_team: str
    status: str
    base_pick: str
    best: str
    second: str
    third: str
    margin: Optional[float]
    reason: str = ""
    p_home: float = 1.0 / 3.0
    p_draw: float = 1.0 / 3.0
    p_away: float = 1.0 / 3.0
    p_best: float = 1.0 / 3.0
    p_second: float = 1.0 / 3.0
    prob_best_pick: str = "1"
    prob_second_pick: str = "0"
    prob_margin: Optional[float] = None
    draw_eligible: bool = False
    draw_promoted: bool = False
    base_from_predicted: bool = False
    diff_ha: float = 0.0
    closeness: float = 0.0
    closeness_effective: float = 0.0
    d_weight: float = 0.0
    buyplan_choice: str = "1"
    buyplan_reason: str = "ARGMAX_BASE"
    entropy: float = 0.0
    draw_candidate: bool = False
    draw_candidate_reason: str = ""
    draw_score: float = -1.0
    d_score_close: float = 0.0
    d_score_stall: float = 0.0
    d_score_total: float = 0.0
    weak_draw_candidate: bool = False
    draw_branch_candidate: bool = False
    draw_branch_score: float = 0.0
    away_value_candidate: bool = False
    away_value_score: float = 0.0
    top_gap: float = 0.0
    spread: float = 0.0
    ratio21: float = 0.0
    ratio31: float = 0.0
    ratio32: float = 0.0
    norm_entropy: float = 0.0
    strength_score: float = 0.0
    flab_trial_flag: str = ""
    flab_trial_score: float = 0.0
    match_type_flags: str = ""
    match_type_primary: str = ""
    lab_matchup_edge: float = 0.0
    lab_style_conflict: bool = False
    lab_low_event: bool = False
    context_primary_pick: str = ""
    context_secondary_pick: str = ""
    context_risk_level: str = ""
    context_ticket_guidance: str = ""
    context_decision_summary: str = ""


def _warn(warnings: List[str], msg: str) -> None:
    warnings.append(msg)
    print(f"[WARN] {msg}")


def _pick_prob_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    if {"prob_home_win", "prob_draw", "prob_away_win"}.issubset(df.columns):
        return "prob_home_win", "prob_draw", "prob_away_win"
    if {"p_home", "p_draw", "p_away"}.issubset(df.columns):
        return "p_home", "p_draw", "p_away"
    return "", "", ""


def _to_symbol(index: int) -> str:
    return SYMBOLS[index]


def _result_to_symbol(v: object) -> str:
    s = _safe_text(v, "").upper()
    if s in {"H", "1"}:
        return "1"
    if s in {"D", "0", "DRAW"}:
        return "0"
    if s in {"A", "2"}:
        return "2"
    return ""


def _safe_prob(v: object) -> float:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return 1.0 / 3.0
    return float(x)


def _load_buyplan_context_df(context_csv: str, warnings: List[str]) -> pd.DataFrame:
    path = os.path.abspath(context_csv)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        _warn(warnings, f"context csv を読み込めません: {path} ({e})")
        return pd.DataFrame()
    required_any = [{"match_id"}, {"home_team", "away_team"}]
    if not any(cols.issubset(df.columns) for cols in required_any):
        _warn(warnings, f"context csv にキー列がありません: {path}")
        return pd.DataFrame()
    print(f"[INFO] buyplan context を読込: {path} rows={len(df)}")
    return df


def _merge_buyplan_context(pred_df: pd.DataFrame, context_df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    if context_df is None or context_df.empty:
        return pred_df
    work = pred_df.copy()
    ctx = context_df.copy()
    key_cols = [
        "primary_pick_symbol",
        "secondary_pick_symbol",
        "risk_level",
        "ticket_guidance",
        "decision_summary",
    ]
    if "match_id" in pred_df.columns and "match_id" in ctx.columns:
        available_ctx_cols = [c for c in ["match_id"] + key_cols if c in ctx.columns]
        ctx = ctx[available_ctx_cols].copy()
    else:
        available_ctx_cols = [c for c in ["league", "home_team", "away_team"] + key_cols if c in ctx.columns]
        ctx = ctx[available_ctx_cols].copy()
    if "match_id" in work.columns and "match_id" in ctx.columns:
        merged = work.merge(ctx.drop_duplicates(subset=["match_id"], keep="first"), on="match_id", how="left")
    else:
        join_cols = [c for c in ["league", "home_team", "away_team"] if c in work.columns and c in ctx.columns]
        if not {"home_team", "away_team"}.issubset(join_cols):
            _warn(warnings, "context csv を結合できません（match_id/home_team/away_team不足）")
            return work
        merged = work.merge(ctx.drop_duplicates(subset=join_cols, keep="first"), on=join_cols, how="left")
    hit_count = int(merged["decision_summary"].notna().sum()) if "decision_summary" in merged.columns else 0
    print(f"[INFO] buyplan context を結合: matched={hit_count}/{len(merged)}")
    return merged


def _normalize_probs3(p_h: float, p_d: float, p_a: float) -> Tuple[float, float, float]:
    vals = [max(0.0, min(1.0, float(p_h))), max(0.0, min(1.0, float(p_d))), max(0.0, min(1.0, float(p_a)))]
    s = vals[0] + vals[1] + vals[2]
    if s <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return vals[0] / s, vals[1] / s, vals[2] / s


def _symbol_from_probs(p_h: float, p_d: float, p_a: float) -> str:
    if p_h >= p_d and p_h >= p_a:
        return "1"
    if p_d >= p_h and p_d >= p_a:
        return "0"
    return "2"


def _result_label_from_symbol(sym: str) -> str:
    if sym == "1":
        return "H"
    if sym == "0":
        return "D"
    return "A"


def _round_draw_pressure_count(plans: List["MatchPlan"]) -> int:
    count = 0
    for p in plans:
        if getattr(p, "status", "") != "OK":
            continue
        if getattr(p, "prob_best_pick", "") == "0":
            count += 1
            continue
        if getattr(p, "prob_second_pick", "") == "0":
            count += 1
            continue
        if bool(getattr(p, "draw_candidate", False)):
            count += 1
            continue
        try:
            if float(getattr(p, "p_draw", 0.0)) >= 0.30:
                count += 1
        except Exception:
            pass
    return count


def _candidate_draw_priority_label(draw_count: int, draw_pressure_count: int) -> str:
    if draw_pressure_count >= 3 and draw_count == 0:
        return "低優先"
    if draw_pressure_count >= 4 and draw_count <= 1:
        return "注意"
    return "通常"


def _draw_insertion_score(plan: "MatchPlan", current_symbol: str) -> float:
    if getattr(plan, "status", "") != "OK":
        return -1.0
    if str(current_symbol) == "0":
        return -1.0
    if not bool(getattr(plan, "draw_candidate", False) or getattr(plan, "weak_draw_candidate", False)):
        return -1.0
    draw_paths = {str(getattr(plan, "best", "")), str(getattr(plan, "second", "")), str(getattr(plan, "third", ""))}
    prob_paths = {str(getattr(plan, "prob_best_pick", "")), str(getattr(plan, "prob_second_pick", ""))}
    if "0" not in draw_paths and "0" not in prob_paths and not bool(getattr(plan, "draw_candidate", False)):
        return -1.0
    score = float(getattr(plan, "p_draw", 0.0))
    if bool(getattr(plan, "draw_candidate", False)):
        score += 0.050
    if str(getattr(plan, "prob_best_pick", "")) == "0":
        score += 0.030
    elif str(getattr(plan, "prob_second_pick", "")) == "0":
        score += 0.020
    if str(getattr(plan, "second", "")) == "0":
        score += 0.015
    if str(getattr(plan, "third", "")) == "0":
        score += 0.010
    try:
        score += min(float(getattr(plan, "draw_score", 0.0)), 1.0) * 0.005
    except Exception:
        pass
    return score


def _inject_minimum_draws(
    plans: List["MatchPlan"],
    tickets: List[List[str]],
    flip_descs: List[str],
    warnings: List[str],
) -> Tuple[int, List[str]]:
    draw_pressure_count = _round_draw_pressure_count(plans)
    if draw_pressure_count < 3:
        return 0, []

    changed = 0
    change_logs: List[str] = []
    existing_keys = {_ticket_key(t): idx for idx, t in enumerate(tickets)}
    for ti in range(3, len(tickets)):
        if ti == 9:
            # 候補10は攻め券として保持し、最低D補完の対象外にする。
            continue
        ticket = tickets[ti]
        if _symbol_count(ticket, "0") > 0:
            continue
        candidates: List[Tuple[float, int]] = []
        for mi, plan in enumerate(plans):
            score = _draw_insertion_score(plan, ticket[mi])
            if score >= 0.0:
                candidates.append((score, mi))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        applied = False
        for score, mi in candidates:
            new_ticket = list(ticket)
            before = str(new_ticket[mi])
            new_ticket[mi] = "0"
            new_key = _ticket_key(new_ticket)
            owner = existing_keys.get(new_key)
            if owner is not None and owner != ti:
                continue
            old_key = _ticket_key(ticket)
            if old_key in existing_keys and existing_keys.get(old_key) == ti:
                existing_keys.pop(old_key, None)
            tickets[ti] = new_ticket
            existing_keys[new_key] = ti
            note = (
                f"min_draw_insert:M{plans[mi].match_no:02d}:{before}->0"
                f"(score={score:.3f},reason={plans[mi].draw_candidate_reason or 'draw_path'})"
            )
            if ti < len(flip_descs):
                base_desc = flip_descs[ti]
                flip_descs[ti] = f"{base_desc}; {note}" if base_desc else note
            change_logs.append(f"ticket={ti+1:02d} {note}")
            changed += 1
            applied = True
            break
        if not applied:
            _warn(
                warnings,
                f"ticket{ti+1:02d}: D=0 のままです（節内D気配={draw_pressure_count}だがユニーク維持のため差し込み失敗）",
            )
    return changed, change_logs


def _build_system_ticket(df: pd.DataFrame, plans: List[MatchPlan], col: str, fallback_col: str) -> Tuple[List[str], str]:
    ticket: List[str] = []
    if df is None or df.empty:
        return [p.base_pick for p in plans], f"system_fixed:{col}:fallback_empty_df"

    work = df.copy()
    if "match_no" in work.columns:
        work["match_no"] = pd.to_numeric(work["match_no"], errors="coerce").astype("Int64")
    by_match_no = {}
    by_match_key = {}
    if "match_no" in work.columns:
        for _, row in work.dropna(subset=["match_no"]).drop_duplicates(subset=["match_no"], keep="first").iterrows():
            by_match_no[int(row["match_no"])] = row
    if {"home_team", "away_team"}.issubset(work.columns):
        keyed = work.copy()
        keyed["_home_key"] = keyed["home_team"].map(_norm_team_key)
        keyed["_away_key"] = keyed["away_team"].map(_norm_team_key)
        if "league" in keyed.columns:
            keyed["_league_key"] = keyed["league"].map(lambda v: _safe_text(v, "").upper())
        else:
            keyed["_league_key"] = ""
        for _, row in keyed.drop_duplicates(
            subset=["_league_key", "_home_key", "_away_key"], keep="first"
        ).iterrows():
            by_match_key[(row["_league_key"], row["_home_key"], row["_away_key"])] = row

    fallback_used = 0
    primary_used = 0
    for plan in plans:
        row = by_match_no.get(int(plan.match_no))
        if row is None:
            row = by_match_key.get(
                (
                    _safe_text(plan.league, "").upper(),
                    _norm_team_key(plan.home_team),
                    _norm_team_key(plan.away_team),
                )
            )
        sym = ""
        if row is not None and col in row.index:
            sym = _result_to_symbol(row.get(col))
            if sym:
                primary_used += 1
        if not sym and row is not None and fallback_col in row.index:
            sym = _result_to_symbol(row.get(fallback_col))
            if sym:
                fallback_used += 1
        if not sym:
            sym = plan.base_pick
            fallback_used += 1
        ticket.append(sym)

    return ticket, f"system_fixed:{col}:primary={primary_used},fallback={fallback_used}"


def _select_with_fallback(
    low_order: List[int],
    preferred_abs_idx: Optional[int],
    used: set,
    banned: set,
) -> Optional[int]:
    if preferred_abs_idx is None:
        return None
    if preferred_abs_idx not in low_order:
        return None
    start = low_order.index(preferred_abs_idx)
    for pos in range(start, len(low_order)):
        cand = low_order[pos]
        if cand not in used and cand not in banned:
            return cand
    for pos in range(0, start):
        cand = low_order[pos]
        if cand not in used and cand not in banned:
            return cand
    return None


def _to_outcome_label(index: int) -> str:
    if index == 0:
        return "home"
    if index == 1:
        return "draw"
    return "away"


def _rank_outcomes(p_h: float, p_d: float, p_a: float) -> List[str]:
    vals = [p_h, p_d, p_a]
    ranked = sorted(range(3), key=lambda i: (-vals[i], i))
    return [_to_outcome_label(ranked[0]), _to_outcome_label(ranked[1]), _to_outcome_label(ranked[2])]


def _entropy_3way(p_h: float, p_d: float, p_a: float) -> float:
    vals = [max(1e-12, float(p_h)), max(1e-12, float(p_d)), max(1e-12, float(p_a))]
    return float(-(vals[0] * math.log(vals[0]) + vals[1] * math.log(vals[1]) + vals[2] * math.log(vals[2])))


def compute_closeness_2axis(p_h: float, p_a: float, c_diff: float = BUYPLAN_2AXIS_C_DIFF) -> float:
    diff = abs(float(p_h) - float(p_a))
    denom = max(float(c_diff), 1e-9)
    closeness = 1.0 - (diff / denom)
    if closeness < 0.0:
        return 0.0
    if closeness > 1.0:
        return 1.0
    return float(closeness)


def _argmax_symbol(p_h: float, p_d: float, p_a: float) -> str:
    return _symbol_from_probs(p_h, p_d, p_a)


def apply_draw_bias_in_buyplan(
    p_h: float,
    p_d: float,
    p_a: float,
    closeness_score: float,
    config: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    cfg = config or {}
    d_min = float(cfg.get("d_min", BUYPLAN_2AXIS_D_MIN))
    max_strong = float(cfg.get("max_strong", BUYPLAN_2AXIS_MAX_STRONG))
    w_draw = float(cfg.get("w_draw", BUYPLAN_2AXIS_W_DRAW))
    base_choice = _argmax_symbol(p_h, p_d, p_a)
    reason = "ARGMAX_BASE"
    closeness_effective = float(closeness_score)
    if float(p_d) < d_min:
        closeness_effective = 0.0
        reason = "GATE_PD_MIN"
    if max(float(p_h), float(p_a)) >= max_strong:
        closeness_effective = 0.0
        reason = "GATE_MAX_STRONG"
    d_weight = float(closeness_effective * w_draw)
    score_h = float(p_h)
    score_d = float(p_d + d_weight)
    score_a = float(p_a)
    buyplan_choice = _argmax_symbol(score_h, score_d, score_a)
    if buyplan_choice == "0" and d_weight > 0:
        reason = "2AXIS_CLOSE_DRAW"
    elif d_weight > 0:
        reason = "2AXIS_APPLIED_NO_SWITCH"
    return {
        "base_choice": base_choice,
        "buyplan_choice": buyplan_choice,
        "closeness_effective": float(closeness_effective),
        "d_weight": float(d_weight),
        "score_h": score_h,
        "score_d": score_d,
        "score_a": score_a,
        "reason": reason,
    }


def _evaluate_draw_candidate(p_h: float, p_d: float, p_a: float) -> Dict[str, object]:
    probs = {"1": float(p_h), "0": float(p_d), "2": float(p_a)}
    ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    best_sym, best_prob = ranked[0]
    second_sym, second_prob = ranked[1]
    third_prob = float(ranked[2][1])
    margin = float(best_prob - second_prob)
    spread = float(best_prob - third_prob)
    ent = _entropy_3way(p_h, p_d, p_a)
    diff_ha = abs(float(p_h) - float(p_a))
    ratio21 = float(second_prob / max(best_prob, 1e-12))
    ratio31 = float(third_prob / max(best_prob, 1e-12))
    ratio32 = float(third_prob / max(second_prob, 1e-12))

    # draw候補は3値形状で判定（2軸差分のみには依存しない）
    ok = bool(
        ratio31 >= float(SHAPE_DRAW_RATIO31_MIN)
        and spread <= float(SHAPE_DRAW_SPREAD_MAX)
        and ent >= float(SHAPE_DRAW_ENTROPY_MIN)
    )
    reason = "shape_close_3way" if ok else "shape_not_close_3way"
    closeness = max(0.0, min(1.0, 1.0 - diff_ha))
    ent_max = math.log(3.0)
    norm_entropy = max(0.0, min(1.0, float(ent / ent_max))) if ent_max > 0 else 0.0
    norm_margin = max(0.0, min(1.0, float(margin)))
    strength_score = float(
        0.45 * (1.0 - ratio21)
        + 0.35 * (1.0 - ratio31)
        + 0.20 * (1.0 - norm_entropy)
    )
    score = float(closeness + 0.5 * float(p_d))
    return {
        "ok": bool(ok),
        "best_sym": best_sym,
        "second_sym": second_sym,
        "third_sym": ranked[2][0],
        "best_prob": float(best_prob),
        "second_prob": float(second_prob),
        "margin": margin,
        "spread": spread,
        "entropy": float(ent),
        "ratio21": float(ratio21),
        "ratio31": float(ratio31),
        "ratio32": float(ratio32),
        "diff_ha": float(diff_ha),
        "closeness": float(closeness),
        "norm_entropy": float(norm_entropy),
        "strength_score": float(strength_score),
        "norm_margin": float(norm_margin),
        "score": score,
        "reason": reason,
    }


def _evaluate_draw_branch_candidate(
    p_h: float,
    p_d: float,
    p_a: float,
    draw_eval: Dict[str, object],
) -> Dict[str, object]:
    best_sym = str(draw_eval.get("best_sym", ""))
    diff_ha = abs(float(p_h) - float(p_a))
    ratio31 = float(draw_eval.get("ratio31", 0.0))
    entropy = float(draw_eval.get("entropy", 0.0))
    p_draw = float(p_d)
    ok = bool(
        best_sym == "0"
        and diff_ha <= float(DRAW_BRANCH_HA_DIFF_MAX)
        and ratio31 >= float(DRAW_BRANCH_RATIO31_MIN)
        and entropy >= float(DRAW_BRANCH_ENTROPY_MIN)
        and p_draw >= float(DRAW_BRANCH_PD_MIN)
    )
    diff_score = 1.0 - min(1.0, diff_ha / max(float(DRAW_BRANCH_HA_DIFF_MAX), 1e-9))
    ratio_score = min(1.0, ratio31 / max(float(DRAW_BRANCH_RATIO31_MIN), 1e-9))
    entropy_score = min(1.0, entropy / math.log(3.0))
    pd_score = min(1.0, p_draw / max(float(DRAW_BRANCH_PD_MIN), 1e-9))
    score = float(0.40 * diff_score + 0.25 * ratio_score + 0.20 * entropy_score + 0.15 * pd_score)
    return {
        "ok": ok,
        "score": score,
        "diff_ha": diff_ha,
        "ratio31": ratio31,
        "entropy": entropy,
        "p_draw": p_draw,
    }


def _evaluate_away_value_candidate(
    p_h: float,
    p_d: float,
    p_a: float,
    draw_eval: Dict[str, object],
) -> Dict[str, object]:
    ranked = sorted([("1", float(p_h)), ("0", float(p_d)), ("2", float(p_a))], key=lambda kv: (-kv[1], kv[0]))
    best_sym = str(ranked[0][0])
    top_prob = float(ranked[0][1])
    away_prob = float(p_a)
    diff_ha = max(0.0, float(p_h) - float(p_a))
    top_gap = max(0.0, top_prob - away_prob)
    ratio_to_top = float(away_prob / max(top_prob, 1e-12))
    entropy = float(draw_eval.get("entropy", 0.0))
    ok = bool(
        best_sym != "2"
        and float(p_h) >= float(p_a)
        and diff_ha <= float(AWAY_VALUE_HA_DIFF_MAX)
        and top_gap <= float(AWAY_VALUE_TOP_GAP_MAX)
        and ratio_to_top >= float(AWAY_VALUE_RATIO_TO_TOP_MIN)
        and entropy >= float(AWAY_VALUE_ENTROPY_MIN)
        and float(p_d) >= float(AWAY_VALUE_PD_MIN)
    )
    diff_score = 1.0 - min(1.0, diff_ha / max(float(AWAY_VALUE_HA_DIFF_MAX), 1e-9))
    gap_score = 1.0 - min(1.0, top_gap / max(float(AWAY_VALUE_TOP_GAP_MAX), 1e-9))
    ratio_score = min(1.0, ratio_to_top / max(float(AWAY_VALUE_RATIO_TO_TOP_MIN), 1e-9))
    entropy_score = min(1.0, entropy / math.log(3.0))
    pd_score = min(1.0, float(p_d) / max(float(AWAY_VALUE_PD_MIN), 1e-9))
    score = float(0.28 * diff_score + 0.28 * gap_score + 0.24 * ratio_score + 0.10 * entropy_score + 0.10 * pd_score)
    return {
        "ok": ok,
        "score": score,
        "diff_ha": diff_ha,
        "top_gap": top_gap,
        "ratio_to_top": ratio_to_top,
        "entropy": entropy,
        "p_draw": float(p_d),
    }


def _prob_by_symbol(plan: MatchPlan) -> Dict[str, float]:
    return {"1": float(plan.p_home), "0": float(plan.p_draw), "2": float(plan.p_away)}


def _relative_eval_map(plan: MatchPlan) -> Dict[str, Dict[str, float]]:
    probs = _prob_by_symbol(plan)
    ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    top_prob = float(ranked[0][1])
    spread = float(ranked[0][1] - ranked[2][1])
    rank_of = {sym: i + 1 for i, (sym, _) in enumerate(ranked)}
    rel: Dict[str, Dict[str, float]] = {}
    for sym in ["1", "0", "2"]:
        r = int(rank_of[sym])
        prob = float(probs[sym])
        above_prob = float(ranked[r - 2][1]) if r > 1 else prob
        below_prob = float(ranked[r][1]) if r < 3 else prob
        rel[sym] = {
            "prob": prob,
            "rank": float(r),
            "top_prob": top_prob,
            "ratio_to_top": float(prob / max(top_prob, 1e-12)),
            "gap_to_above": float(max(0.0, above_prob - prob)),
            "gap_to_below": float(max(0.0, prob - below_prob)),
            "spread": spread,
        }
    return rel


def _is_close_branch(plan: MatchPlan, sym: str) -> bool:
    rel = _relative_eval_map(plan).get(sym, {})
    if not rel:
        return False
    if int(rel.get("rank", 9.0)) <= 1:
        return False
    return bool(
        float(rel.get("ratio_to_top", 0.0)) >= float(REL_CLOSE_RATIO_TO_TOP_MIN)
        or float(rel.get("gap_to_above", 999.0)) <= float(REL_CLOSE_GAP_TO_ABOVE_MAX)
        or float(rel.get("spread", 999.0)) <= float(REL_CLOSE_SPREAD_MAX)
    )


def _relative_branch_score(plan: MatchPlan, sym: str) -> float:
    rel = _relative_eval_map(plan).get(sym, {})
    prob = float(rel.get("prob", 0.0))
    ratio_to_top = float(rel.get("ratio_to_top", 0.0))
    alpha = max(0.0, min(1.0, float(RELATIVE_SCORE_ALPHA)))
    return float(prob * (alpha + (1.0 - alpha) * ratio_to_top))


def _log_repair_drop(ticket_no: int, plan: MatchPlan, from_symbol: str, to_symbol: str, reason: str) -> None:
    rel = _relative_eval_map(plan)
    r_from = rel.get(from_symbol, {})
    r_to = rel.get(to_symbol, {})
    print(
        f"[BUYPLAN_REPAIR_DROP] ticket={ticket_no:02d} match_no={plan.match_no:02d} "
        f"from_symbol={from_symbol} to_symbol={to_symbol} "
        f"from_prob={float(r_from.get('prob', 0.0)):.4f} to_prob={float(r_to.get('prob', 0.0)):.4f} "
        f"from_ratio_to_top={float(r_from.get('ratio_to_top', 0.0)):.4f} "
        f"to_ratio_to_top={float(r_to.get('ratio_to_top', 0.0)):.4f} reason={reason}"
    )


def _log_repair_keep(ticket_no: int, plan: MatchPlan, sym: str, reason: str = "close_branch_protected") -> None:
    rel = _relative_eval_map(plan).get(sym, {})
    print(
        f"[BUYPLAN_REPAIR_KEEP] ticket={ticket_no:02d} match_no={plan.match_no:02d} "
        f"symbol={sym} prob={float(rel.get('prob', 0.0)):.4f} "
        f"ratio_to_top={float(rel.get('ratio_to_top', 0.0)):.4f} reason={reason}"
    )


def generate_patterns(matches: List[Dict[str, object]]) -> List[Dict[str, object]]:
    n = len(matches)
    if n == 0:
        return []

    prepared = []
    for i, m in enumerate(matches):
        p_h = _safe_prob(m.get("prob_home"))
        p_d = _safe_prob(m.get("prob_draw"))
        p_a = _safe_prob(m.get("prob_away"))
        p_h, p_d, p_a = _normalize_probs3(p_h, p_d, p_a)
        ranked = _rank_outcomes(p_h, p_d, p_a)
        prepared.append(
            {
                "idx": i,
                "match_id": m.get("match_id", i),
                "p1": ranked[0],
                "p2": ranked[1],
                "p3": ranked[2],
                "maxP": max(p_h, p_d, p_a),
            }
        )

    # 低自信順
    low = sorted(range(n), key=lambda i: (prepared[i]["maxP"], i))
    # 高自信順
    high = sorted(range(n), key=lambda i: (-prepared[i]["maxP"], i))
    protected = set(high[:2])

    mid_idx_pos = n // 2
    mid_pair_pos = [((n - 1) // 2), ((n - 1 + 1) // 2)]
    mid_pair_abs_unique = []
    for pos in mid_pair_pos:
        abs_idx = low[pos] if 0 <= pos < n else None
        if abs_idx is not None and abs_idx not in mid_pair_abs_unique:
            mid_pair_abs_unique.append(abs_idx)

    def low_abs(pos: int) -> Optional[int]:
        if pos < 0 or pos >= n:
            return None
        return low[pos]

    def apply_pattern(change_specs: List[Tuple[Optional[int], str]], banned: Optional[set] = None) -> Dict[int, str]:
        used = set()
        change_map: Dict[int, str] = {}
        banned_set = banned or set()
        for preferred_abs_idx, to_rank in change_specs:
            picked = _select_with_fallback(low, preferred_abs_idx, used, banned_set)
            if picked is None:
                continue
            used.add(picked)
            change_map[picked] = to_rank
        return change_map

    pattern_specs: List[Tuple[List[Tuple[Optional[int], str]], set]] = [
        ([], set()),  # No.1
        ([(low_abs(0), "p2")], set()),  # No.2
        ([(low_abs(0), "p2"), (low_abs(1), "p2")], set()),  # No.3
        ([(low_abs(0), "p2")], set()),  # No.4
        ([(low_abs(0), "p2"), (low_abs(1), "p2")], set()),  # No.5
        ([(low_abs(0), "p2"), (low_abs(1), "p2"), (low_abs(2), "p2")], set()),  # No.6
        ([(low_abs(0), "p3"), (low_abs(1), "p2")], set()),  # No.7
        ([(low_abs(0), "p3"), (low_abs(1), "p3"), (low_abs(2), "p2")], set()),  # No.8
        ([(low_abs(0), "p3"), (low_abs(1), "p3"), (low_abs(mid_idx_pos), "p3"), (low_abs(2), "p2")], set()),  # No.9
        (
            [
                (low_abs(0), "p3"),
                (low_abs(1), "p3"),
                *[(x, "p3") for x in mid_pair_abs_unique],
                (low_abs(2), "p2"),
            ],
            protected,  # No.10
        ),
    ]

    patterns: List[Dict[str, object]] = []
    for p_no, (change_specs, banned) in enumerate(pattern_specs, start=1):
        change_map = apply_pattern(change_specs, banned=banned)
        picks = []
        for i, m in enumerate(prepared):
            selected = m["p1"]
            if i in change_map:
                selected = m[change_map[i]]
            picks.append({"match_id": m["match_id"], "selected": selected})
        patterns.append({"pattern_no": p_no, "picks": picks})
    return patterns


def _apply_scenario_probs(
    p_h: float,
    p_d: float,
    p_a: float,
    scenario_id: str,
    match_no: int,
) -> Tuple[float, float, float]:
    p_h, p_d, p_a = _normalize_probs3(p_h, p_d, p_a)

    if scenario_id == "01":
        return p_h, p_d, p_a

    if scenario_id == "02":
        vals = [p_h, p_d, p_a]
        best = max(range(3), key=lambda i: vals[i])
        if vals[best] > 0.55:
            vals[best] -= 0.05
            others = [i for i in [0, 1, 2] if i != best]
            vals[others[0]] += 0.03
            vals[others[1]] += 0.02
        return _normalize_probs3(vals[0], vals[1], vals[2])

    if scenario_id == "03":
        vals = [p_h, p_d, p_a]
        best = max(range(3), key=lambda i: vals[i])
        if vals[best] > 0.60:
            vals[best] -= 0.10
            others = [i for i in [0, 1, 2] if i != best]
            vals[others[0]] += 0.06
            vals[others[1]] += 0.04
        return _normalize_probs3(vals[0], vals[1], vals[2])

    if scenario_id == "04":
        return _normalize_probs3(p_h - 0.025, p_d + 0.05, p_a - 0.025)

    if scenario_id == "05":
        return _normalize_probs3(p_h + 0.05, p_d - 0.025, p_a - 0.025)

    if scenario_id == "06":
        return _normalize_probs3(p_h - 0.025, p_d - 0.025, p_a + 0.05)

    if scenario_id == "07":
        if abs(p_h - p_a) < 0.12:
            return _apply_scenario_probs(p_h, p_d, p_a, "02", match_no)
        return p_h, p_d, p_a

    if scenario_id == "08":
        # Deterministic tiny noise by match_no (no random dependency).
        step = ((match_no * 37) % 5 - 2) * 0.01
        return _normalize_probs3(p_h + step, p_d - step / 2.0, p_a - step / 2.0)

    if scenario_id == "09":
        if max(p_h, p_d, p_a) > 0.60:
            return p_h, p_d, p_a
        return p_h, p_d, p_a

    if scenario_id == "10":
        vals = [p_h, p_d, p_a]
        best = max(range(3), key=lambda i: vals[i])
        if vals[best] > 0.60:
            delta = vals[best] - 0.60
            vals[best] = 0.60
            others = [i for i in [0, 1, 2] if i != best]
            s_other = vals[others[0]] + vals[others[1]]
            if s_other > 0:
                vals[others[0]] += delta * (vals[others[0]] / s_other)
                vals[others[1]] += delta * (vals[others[1]] / s_other)
            else:
                vals[others[0]] += delta / 2.0
                vals[others[1]] += delta / 2.0
        return _normalize_probs3(vals[0], vals[1], vals[2])

    return p_h, p_d, p_a


def _normalize_match_no(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    out = df.copy()
    if "match_no" not in out.columns:
        _warn(warnings, "match_no がないため、入力順で 1..13 を割り当てます。")
        out = out.reset_index(drop=True)
        out["match_no"] = out.index + 1
    out["match_no"] = pd.to_numeric(out["match_no"], errors="coerce")
    missing = out["match_no"].isna().sum()
    if missing:
        _warn(warnings, f"match_no が数値化できない行が {missing} 件あります。入力順で補完します。")
        idx = out["match_no"].isna()
        out.loc[idx, "match_no"] = out[idx].index + 1
    out["match_no"] = out["match_no"].astype(int)
    return out


def _dedupe_match_no(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    dup_count = int(df.duplicated(subset=["match_no"], keep="first").sum())
    if dup_count:
        _warn(warnings, f"重複する match_no が {dup_count} 件あります。先頭行を採用します。")
    return df.drop_duplicates(subset=["match_no"], keep="first")


def _safe_text(v: object, default: str = "") -> str:
    if pd.isna(v):
        return default
    s = str(v).strip()
    return s if s else default


def _norm_team_key(v: object) -> str:
    s = unicodedata.normalize("NFKC", _safe_text(v, ""))
    s = s.replace("　", " ").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("・", "").replace(".", "").replace("･", "")
    s = s.upper()
    team_alias = {
        "FC東京": "FC東京",
        "FC今治": "今治",
        "SC相模原": "相模原",
        "RB大宮": "大宮",
        "RB大宮アルディージャ": "大宮",
        "横浜FC": "横浜FC",
        "横浜FM": "横浜FM",
        "川崎F": "川崎F",
        "東京V": "東京V",
        "C大阪": "C大阪",
        "G大阪": "G大阪",
    }
    return team_alias.get(s, s)


def _resolve_toto_order_csv_path(arg_path: str) -> str:
    if isinstance(arg_path, str) and arg_path.strip().lower() in {"none", "off", "disable", "disabled", "false", "no"}:
        return ""
    if arg_path:
        return os.path.abspath(arg_path)
    # 既定ファイルを最優先
    if os.path.exists(DEFAULT_TOTO_ORDER_CSV):
        return DEFAULT_TOTO_ORDER_CSV
    if os.path.exists(LEGACY_TOTO_ORDER_CSV):
        return LEGACY_TOTO_ORDER_CSV
    # ファイル名ゆらぎ（結合文字違い）に備えて曖昧探索
    manual_dir = Path(BASE_DIR) / "data" / "manual"
    if manual_dir.exists():
        candidates = sorted(manual_dir.glob("toto*リスト*.csv"))
        if candidates:
            return str(candidates[0])
        candidates = sorted(manual_dir.glob("toto*順*.csv"))
        if candidates:
            return str(candidates[0])
    return DEFAULT_TOTO_ORDER_CSV


def _extract_round_number(text: object) -> Optional[int]:
    s = unicodedata.normalize("NFKC", _safe_text(text, ""))
    m = re.search(r"第\s*(\d+)\s*節", s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return None


def _detect_prediction_context(pred_df: Optional[pd.DataFrame]) -> Dict[str, Optional[int]]:
    context: Dict[str, Optional[int]] = {"j1_round": None, "toto_round": None}
    if pred_df is None or pred_df.empty:
        return context

    for col in ("toto_round", "toto_round_id"):
        if col in pred_df.columns:
            vals = pred_df[col].dropna().astype(str).str.extract(r"(\d+)")[0].dropna()
            if not vals.empty:
                context["toto_round"] = int(vals.mode().iloc[0])
                break

    round_col = next((c for c in ("節", "section", "round") if c in pred_df.columns), None)
    if round_col is None:
        return context

    scope = pred_df
    if "league" in pred_df.columns:
        j1_scope = pred_df[pred_df["league"].astype(str).str.lower() == "j1"]
        if not j1_scope.empty:
            scope = j1_scope
    vals = scope[round_col].map(_extract_round_number).dropna()
    if not vals.empty:
        context["j1_round"] = int(vals.mode().iloc[0])
    return context


def _coerce_toto_order_columns(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip(): c for c in raw.columns}
    named_required = {"match_no", "home_team", "away_team"}
    if named_required.issubset(cols):
        keep = ["match_no", "home_team", "away_team"]
        for opt in ("season", "toto_round", "J1_round", "j1_round", "match_date"):
            if opt in cols:
                keep.append(opt)
        out = raw[[cols[c] for c in keep]].copy()
        if "J1_round" in out.columns:
            out = out.rename(columns={"J1_round": "j1_round"})
        return out

    if len(raw.columns) >= 4 and all(isinstance(c, int) for c in raw.columns):
        out = raw.iloc[:, :4].copy()
        out.columns = ["match_no", "home_team", "vs", "away_team"]
        return out
    return pd.DataFrame()


def _select_toto_order_rows(df: pd.DataFrame, pred_df: Optional[pd.DataFrame], warnings: List[str]) -> pd.DataFrame:
    if df.empty or "toto_round" not in df.columns:
        return df

    work = df.copy()
    work["toto_round"] = pd.to_numeric(work["toto_round"], errors="coerce")
    work = work.dropna(subset=["toto_round"]).copy()
    if work.empty:
        return work
    work["toto_round"] = work["toto_round"].astype(int)
    if "j1_round" in work.columns:
        work["j1_round"] = pd.to_numeric(work["j1_round"], errors="coerce").astype("Int64")

    unique_rounds = sorted(work["toto_round"].unique().tolist())
    if len(unique_rounds) <= 1:
        return work

    context = _detect_prediction_context(pred_df)
    target_toto_round = context.get("toto_round")
    if target_toto_round is not None:
        picked = work[work["toto_round"] == target_toto_round].copy()
        if not picked.empty:
            print(f"[INFO] toto並び順を開催回で選択: toto{target_toto_round}")
            return picked

    pred_keys: Set[Tuple[str, str]] = set()
    if pred_df is not None and not pred_df.empty and {"home_team", "away_team"}.issubset(pred_df.columns):
        tmp = pred_df[["home_team", "away_team"]].copy()
        tmp["_home_key"] = tmp["home_team"].map(_norm_team_key)
        tmp["_away_key"] = tmp["away_team"].map(_norm_team_key)
        pred_keys = set(zip(tmp["_home_key"], tmp["_away_key"]))

    scored: List[Tuple[int, int, int]] = []
    target_j1_round = context.get("j1_round")
    for toto_round, grp in work.groupby("toto_round", dropna=False):
        g = grp.copy()
        g["_home_key"] = g["home_team"].map(_norm_team_key)
        g["_away_key"] = g["away_team"].map(_norm_team_key)
        overlap = sum((h, a) in pred_keys for h, a in zip(g["_home_key"], g["_away_key"])) if pred_keys else 0
        round_bonus = 0
        if target_j1_round is not None and "j1_round" in g.columns and g["j1_round"].notna().any():
            round_bonus = 1 if int(g["j1_round"].dropna().mode().iloc[0]) == target_j1_round else 0
        scored.append((int(toto_round), overlap, round_bonus))

    scored.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    best_toto_round, best_overlap, best_bonus = scored[0]
    tie_count = sum(1 for _, overlap, bonus in scored if (overlap, bonus) == (best_overlap, best_bonus))
    if best_overlap <= 0 and best_bonus > 0:
        print(f"[INFO] toto並び順を節情報で選択: toto{best_toto_round}")
    elif best_overlap <= 0:
        _warn(warnings, "toto並び順CSVの開催回を自動判定できませんでした。最大の開催回を採用します。")
    elif tie_count > 1:
        _warn(warnings, f"toto並び順CSVの開催回候補が複数同率です。toto{best_toto_round} を採用します。")
    else:
        print(f"[INFO] toto並び順を自動選択: toto{best_toto_round} (overlap={best_overlap})")
    return work[work["toto_round"] == best_toto_round].copy()


def _load_toto_order_df(csv_path: str, warnings: List[str], pred_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if not csv_path or not os.path.exists(csv_path):
        _warn(warnings, f"toto並び順CSVが見つかりません: {csv_path}")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])
    try:
        raw = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        _warn(warnings, f"toto並び順CSVを読み込めません: {csv_path} ({e})")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])

    if raw.empty:
        _warn(warnings, f"toto並び順CSVが空です: {csv_path}")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])

    df = _coerce_toto_order_columns(raw)
    if df.empty:
        try:
            raw = pd.read_csv(csv_path, header=None, dtype=str, encoding="utf-8-sig")
            df = _coerce_toto_order_columns(raw)
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        _warn(warnings, f"toto並び順CSVの列構造を解釈できません: {csv_path}")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])

    df = _select_toto_order_rows(df, pred_df, warnings)
    df["match_no"] = pd.to_numeric(df["match_no"], errors="coerce")
    df = df.dropna(subset=["match_no", "home_team", "away_team"]).copy()
    if df.empty:
        _warn(warnings, f"toto並び順CSVに有効行がありません: {csv_path}")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])
    df["match_no"] = df["match_no"].astype(int)
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df["_home_key"] = df["home_team"].map(_norm_team_key)
    df["_away_key"] = df["away_team"].map(_norm_team_key)
    dup = int(df.duplicated(subset=["_home_key", "_away_key"], keep="first").sum())
    if dup:
        _warn(warnings, f"toto並び順CSVで重複カードが {dup} 件あり、先頭を採用しました。")
        df = df.drop_duplicates(subset=["_home_key", "_away_key"], keep="first")
    df = df.sort_values("match_no").reset_index(drop=True)
    return df[["match_no", "home_team", "away_team", "_home_key", "_away_key"]]

def _apply_toto_match_order(df: pd.DataFrame, order_df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    if order_df is None or order_df.empty:
        return df
    if "home_team" not in df.columns or "away_team" not in df.columns:
        _warn(warnings, "home_team/away_team が無いため、toto並び順照合をスキップします。")
        return df

    src = df.copy()
    src["_home_key"] = src["home_team"].map(_norm_team_key)
    src["_away_key"] = src["away_team"].map(_norm_team_key)
    key_cols = ["_home_key", "_away_key"]
    dup = int(src.duplicated(subset=key_cols, keep="first").sum())
    if dup:
        _warn(warnings, f"predictions側で重複カードが {dup} 件あるため、先頭行を採用します。")
        src = src.drop_duplicates(subset=key_cols, keep="first")
    src_map = {(r["_home_key"], r["_away_key"]): r for _, r in src.iterrows()}

    rows = []
    mapped = 0
    miss = 0
    for _, o in order_df.iterrows():
        key = (o["_home_key"], o["_away_key"])
        if key in src_map:
            row = dict(src_map[key])
            row["match_no"] = int(o["match_no"])
            rows.append(row)
            mapped += 1
        else:
            # 未一致カードはNO_DATA補完（HOME_XX/AWAY_XX ではなく実カード名で保持）
            row = {"match_no": int(o["match_no"]), "home_team": o["home_team"], "away_team": o["away_team"]}
            if "league" in src.columns:
                row["league"] = "UNKNOWN"
            if "status" in src.columns:
                row["status"] = "NO_DATA"
            rows.append(row)
            miss += 1

    out = pd.DataFrame(rows)
    out = out.sort_values("match_no").reset_index(drop=True)
    extra = max(0, len(src) - mapped)

    if mapped:
        print(f"[INFO] toto並び順を適用: matched={mapped}")
    if miss:
        _warn(warnings, f"toto並び順で未一致のカードが {miss} 件あります。未一致カードは NO_DATA 補完します。")
    if extra:
        _warn(warnings, f"predictions側にtoto対象外カードが {extra} 件あります。toto13試合以外は無視します。")
    out = out.drop(columns=["_home_key", "_away_key"], errors="ignore")
    return out


def _build_toto_diff_report(pred_df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
    if order_df is None or order_df.empty:
        return pd.DataFrame(columns=["diff_type", "match_no", "home_team", "away_team", "league"])

    src = pred_df.copy()
    if "home_team" not in src.columns or "away_team" not in src.columns:
        return pd.DataFrame(columns=["diff_type", "match_no", "home_team", "away_team", "league"])

    src["home_team"] = src["home_team"].astype(str).str.strip()
    src["away_team"] = src["away_team"].astype(str).str.strip()
    src["_home_key"] = src["home_team"].map(_norm_team_key)
    src["_away_key"] = src["away_team"].map(_norm_team_key)
    src = src.drop_duplicates(subset=["_home_key", "_away_key"], keep="first")

    order = order_df.copy()
    order_keys = {(r["_home_key"], r["_away_key"]) for _, r in order.iterrows()}
    src_keys = {(r["_home_key"], r["_away_key"]) for _, r in src.iterrows()}

    rows: List[Dict[str, object]] = []
    for _, r in order.iterrows():
        key = (r["_home_key"], r["_away_key"])
        if key in src_keys:
            rows.append(
                {
                    "diff_type": "matched",
                    "match_no": int(r["match_no"]),
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "league": "",
                }
            )
        else:
            rows.append(
                {
                    "diff_type": "missing_in_predictions",
                    "match_no": int(r["match_no"]),
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "league": "",
                }
            )

    for _, r in src.iterrows():
        key = (r["_home_key"], r["_away_key"])
        if key not in order_keys:
            rows.append(
                {
                    "diff_type": "missing_in_toto_order",
                    "match_no": "",
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "league": _safe_text(r.get("league"), ""),
                }
            )

    out = pd.DataFrame(rows)
    order_map = {"missing_in_predictions": 0, "missing_in_toto_order": 1, "matched": 2}
    out["_ord"] = out["diff_type"].map(order_map).fillna(9)
    out = out.sort_values(["_ord", "match_no", "home_team", "away_team"]).drop(columns=["_ord"])
    return out


def _normalize_league(value: object, warnings: List[str], match_no: int) -> str:
    raw = _safe_text(value, "UNKNOWN")
    if raw == "UNKNOWN":
        return raw
    s = raw.upper().replace("　", "").replace(" ", "")
    # Jが重複している壊れ値を救済（例: JJ1 -> J1）
    s = re.sub(r"^J+", "J", s)
    if s in {"J1", "J2"}:
        return s
    # 余計な記号混入ケースを緩く吸収
    if "1" in s and "J" in s:
        _warn(warnings, f"M{match_no:02d} league異常値 '{raw}' を 'J1' に補正しました。")
        return "J1"
    if "2" in s and "J" in s:
        _warn(warnings, f"M{match_no:02d} league異常値 '{raw}' を 'J2' に補正しました。")
        return "J2"
    _warn(warnings, f"M{match_no:02d} league異常値 '{raw}' を 'UNKNOWN' として扱います。")
    return "UNKNOWN"


def _derive_round_id(df: pd.DataFrame, input_csv: str) -> str:
    if "toto_round_id" in df.columns and len(df):
        v = _safe_text(df["toto_round_id"].iloc[0], "")
        if v:
            return v
    if "節" in df.columns and len(df):
        raw = _safe_text(df["節"].iloc[0], "")
        m = re.search(r"(第\d+[節戦])", raw)
        if m:
            return m.group(1)
        if raw:
            return raw
    return os.path.basename(input_csv) or "UNKNOWN"


def _build_match_plans(df: pd.DataFrame, warnings: List[str], base_mode: str = BUYPLAN_BASE_MODE) -> List[MatchPlan]:
    ph_col, pd_col, pa_col = _pick_prob_columns(df)
    if not ph_col:
        _warn(warnings, "確率カラムが見つかりません。status=OK でもフォールバックで処理します。")

    round_id_default = "UNKNOWN"
    plans: List[MatchPlan] = []

    for m in range(1, REQUIRED_MATCH_COUNT + 1):
        row_df = df[df["match_no"] == m]
        if row_df.empty:
            _warn(warnings, f"第{m}試合が存在しないため NO_DATA で補完します。")
            plans.append(
                MatchPlan(
                    match_no=m,
                    toto_round_id=round_id_default,
                    league="UNKNOWN",
                    home_team=f"HOME_{m:02d}",
                    away_team=f"AWAY_{m:02d}",
                    status="NO_DATA",
                    base_pick="1",
                    best="1",
                    second="0",
                    third="2",
                    margin=None,
                    reason="missing_match_no",
                )
            )
            continue

        row = row_df.iloc[0]
        has_status_col = "status" in row_df.columns
        status_raw = _safe_text(row["status"], "") if has_status_col else ""
        round_id = _safe_text(row["toto_round_id"], round_id_default) if "toto_round_id" in row_df.columns else round_id_default
        league = _normalize_league(row["league"], warnings, m) if "league" in row_df.columns else "UNKNOWN"
        home = _safe_text(row["home_team"], f"HOME_{m:02d}") if "home_team" in row_df.columns else f"HOME_{m:02d}"
        away = _safe_text(row["away_team"], f"AWAY_{m:02d}") if "away_team" in row_df.columns else f"AWAY_{m:02d}"

        inferred_ok = False
        if ph_col:
            probs_for_status = [
                pd.to_numeric(row.get(ph_col), errors="coerce"),
                pd.to_numeric(row.get(pd_col), errors="coerce"),
                pd.to_numeric(row.get(pa_col), errors="coerce"),
            ]
            inferred_ok = not any(pd.isna(x) for x in probs_for_status)
        elif "predicted_result" in row_df.columns:
            inferred_ok = bool(_safe_text(row.get("predicted_result"), ""))

        status = status_raw.upper() if status_raw else ("OK" if inferred_ok else "NO_DATA")
        if status not in {"OK", "NO_DATA"}:
            status = "OK" if inferred_ok else "NO_DATA"

        if status != "OK":
            plans.append(
                MatchPlan(
                    match_no=m,
                    toto_round_id=round_id,
                    league=league,
                    home_team=home,
                    away_team=away,
                    status="NO_DATA",
                    base_pick="1",
                    best="1",
                    second="0",
                    third="2",
                    margin=None,
                    reason="status_no_data",
                )
            )
            continue

        if not ph_col:
            plans.append(
                MatchPlan(
                    match_no=m,
                    toto_round_id=round_id,
                    league=league,
                    home_team=home,
                    away_team=away,
                    status="NO_DATA",
                    base_pick="1",
                    best="1",
                    second="0",
                    third="2",
                    margin=None,
                    reason="no_probability_columns",
                )
            )
            continue

        probs = [
            pd.to_numeric(row.get(ph_col), errors="coerce"),
            pd.to_numeric(row.get(pd_col), errors="coerce"),
            pd.to_numeric(row.get(pa_col), errors="coerce"),
        ]
        if any(pd.isna(x) for x in probs):
            _warn(warnings, f"第{m}試合の確率に欠損があるため NO_DATA として処理します。")
            plans.append(
                MatchPlan(
                    match_no=m,
                    toto_round_id=round_id,
                    league=league,
                    home_team=home,
                    away_team=away,
                    status="NO_DATA",
                    base_pick="1",
                    best="1",
                    second="0",
                    third="2",
                    margin=None,
                    reason="probability_missing",
                )
            )
            continue

        p_h, p_d, p_a = _normalize_probs3(float(probs[0]), float(probs[1]), float(probs[2]))
        # draw補正（確率計算後、順位決定前）
        pre_ranked = sorted([float(p_h), float(p_d), float(p_a)], reverse=True)
        pre_margin = float(pre_ranked[0] - pre_ranked[1]) if len(pre_ranked) >= 2 else 0.0
        draw_boost_used = float(DRAW_BOOST_CLOSE) if pre_margin <= float(DRAW_BOOST_MARGIN_MAX) else float(DRAW_BOOST)
        p_d = float(p_d) * draw_boost_used
        p_h, p_d, p_a = _normalize_probs3(float(p_h), float(p_d), float(p_a))
        probs = [p_h, p_d, p_a]
        draw_eval = _evaluate_draw_candidate(p_h, p_d, p_a)
        draw_branch_eval = _evaluate_draw_branch_candidate(p_h, p_d, p_a, draw_eval)
        away_value_eval = _evaluate_away_value_candidate(p_h, p_d, p_a, draw_eval)
        prob_by_symbol = {"1": float(p_h), "0": float(p_d), "2": float(p_a)}
        ranked_symbols = [str(draw_eval["best_sym"]), str(draw_eval["second_sym"]), str(draw_eval["third_sym"])]
        best_symbol_prob = ranked_symbols[0]
        second_symbol_prob = ranked_symbols[1]
        third_symbol_prob = ranked_symbols[2]
        top_gap = float(draw_eval["margin"])
        spread_value = float(draw_eval.get("spread", 0.0))
        diff_ha_base = abs(float(p_h) - float(p_a))

        strength_score = float(draw_eval.get("strength_score", 0.0))
        ratio21 = float(draw_eval.get("ratio21", 0.0))
        ratio31 = float(draw_eval.get("ratio31", 0.0))
        ratio32 = float(draw_eval.get("ratio32", 0.0))
        norm_entropy = float(draw_eval.get("norm_entropy", 0.0))
        flab_trial_flag = _safe_text(row.get("flab_trial_flag"), "").upper()
        flab_trial_score_raw = pd.to_numeric(row.get("flab_trial_score"), errors="coerce")
        flab_trial_score = 0.0 if pd.isna(flab_trial_score_raw) else float(flab_trial_score_raw)
        match_type_flags = _safe_text(row.get("match_type_flags"), "")
        match_type_primary = _safe_text(row.get("match_type_primary"), "")
        lab_matchup_edge_raw = pd.to_numeric(row.get("match_type_lab_matchup_edge"), errors="coerce")
        lab_matchup_edge = 0.0 if pd.isna(lab_matchup_edge_raw) else float(lab_matchup_edge_raw)
        lab_style_conflict = bool(str(row.get("match_type_lab_style_conflict", "")).strip().lower() == "true")
        lab_low_event = bool(str(row.get("match_type_lab_low_event", "")).strip().lower() == "true")
        d_score_close_raw = pd.to_numeric(row.get("d_score_close"), errors="coerce")
        d_score_stall_raw = pd.to_numeric(row.get("d_score_stall"), errors="coerce")
        d_score_total_raw = pd.to_numeric(row.get("d_score_total"), errors="coerce")
        d_score_close = 0.0 if pd.isna(d_score_close_raw) else float(d_score_close_raw)
        d_score_stall = 0.0 if pd.isna(d_score_stall_raw) else float(d_score_stall_raw)
        d_score_total = 0.0 if pd.isna(d_score_total_raw) else float(d_score_total_raw)
        context_primary_pick = _safe_text(row.get("primary_pick_symbol"), "")
        context_secondary_pick = _safe_text(row.get("secondary_pick_symbol"), "")
        context_risk_level = _safe_text(row.get("risk_level"), "")
        context_ticket_guidance = _safe_text(row.get("ticket_guidance"), "")
        context_decision_summary = _safe_text(row.get("decision_summary"), "")
        if base_mode == "tri_argmax":
            base_pick = best_symbol_prob
            decision_reason = "TRI_ARGMAX"
        elif base_mode == "topgap":
            base_pick = best_symbol_prob
            if top_gap >= BASE_TOP_GAP_STRONG:
                decision_reason = "TOPGAP_STRONG"
            elif top_gap >= BASE_TOP_GAP_MID:
                decision_reason = "TOPGAP_MID"
            else:
                decision_reason = "TOPGAP_SMALL"
        elif base_mode == "shape_relative":
            base_pick = best_symbol_prob
            if strength_score >= float(SHAPE_STRENGTH_STRONG):
                decision_reason = "SHAPE_STRONG_BEST"
            elif strength_score >= float(SHAPE_STRENGTH_MID):
                decision_reason = "SHAPE_MID_BEST"
            else:
                decision_reason = "SHAPE_WEAK_BEST"
        else:
            if diff_ha_base >= BUYPLAN_BALANCE_T_DRAW:
                base_pick = "1" if p_h > p_a else "2"
                decision_reason = "BALANCE_HA"
            else:
                base_pick = "0"
                decision_reason = "BALANCE_DRAW"
            if base_pick == "0" and float(p_d) < BUYPLAN_BALANCE_D_MIN:
                base_pick = "1" if p_h > p_a else "2"
                decision_reason = "DRAW_BLOCK_PD"

        base_from_predicted = False
        predicted_base_pick = _predicted_result_to_symbol(row.get("predicted_result"))
        if BUYPLAN_USE_PREDICTED_RESULT_BASE and predicted_base_pick in SYMBOLS:
            base_pick = predicted_base_pick
            decision_reason = "PREDICTED_RESULT_BASE"
            base_from_predicted = True

        # 候補枝は常に確率順位を母体にしつつ、baseだけ predicted_result で上書き可能
        alt_symbols = [sym for sym in ranked_symbols if sym != base_pick]
        for sym in SYMBOLS:
            if sym != base_pick and sym not in alt_symbols:
                alt_symbols.append(sym)
        second_symbol = alt_symbols[0]
        third_symbol = alt_symbols[1]
        best_symbol = base_pick
        best_p = float(prob_by_symbol.get(best_symbol, 1.0 / 3.0))
        second_p = float(prob_by_symbol.get(second_symbol, 1.0 / 3.0))
        margin_value = abs(float(best_p) - float(second_p))

        diff_ha = float(draw_eval["diff_ha"])
        closeness = compute_closeness_2axis(p_h, p_a)
        weak_draw_candidate = bool(
            (not bool(draw_eval["ok"]))
            and strength_score <= float(WEAK_DRAW_STRENGTH_MAX)
            and ratio31 >= float(WEAK_DRAW_RATIO31_MIN)
            and float(draw_eval["entropy"]) >= float(WEAK_DRAW_ENTROPY_MIN)
        )
        # J2の薄差カードは、draw候補ゲートを通らなくても Prob/Exp だけで弱いD候補として扱う。
        # 01〜03 の固定票には触れず、拮抗J2の D 配分だけ少し広げる。
        j2_weak_draw_candidate = bool(
            league == "J2"
            and (not bool(draw_eval["ok"]))
            and second_symbol_prob == "0"
            and top_gap <= float(J2_WEAK_DRAW_MARGIN_MAX)
            and float(p_d) >= float(J2_WEAK_DRAW_PD_MIN)
            and float(draw_eval["entropy"]) >= float(J2_WEAK_DRAW_ENTROPY_MIN)
            and ratio31 >= float(J2_WEAK_DRAW_RATIO31_MIN)
        )
        weak_draw_candidate = bool(weak_draw_candidate or j2_weak_draw_candidate)

        draw_bias = apply_draw_bias_in_buyplan(
            p_h,
            p_d,
            p_a,
            closeness,
            config={
                "d_min": BUYPLAN_BALANCE_D_MIN,
                "max_strong": BUYPLAN_2AXIS_MAX_STRONG,
                "w_draw": BUYPLAN_2AXIS_W_DRAW,
            },
        )
        if DEBUG_PROBS:
            print(
                f"[BUYPLAN_DEBUG] M{m:02d} {home} vs {away} "
                f"probs=[H:{float(probs[0]):.6f}, D:{float(probs[1]):.6f}, A:{float(probs[2]):.6f}] "
                f"base={base_pick} best={best_symbol} second={second_symbol} third={third_symbol} "
                f"using=({ph_col},{pd_col},{pa_col})"
            )
        print(
            f"[BUYPLAN_MATCH] M{m:02d} {home} vs {away} "
            f"draw_boost={draw_boost_used:.3f} pre_margin={pre_margin:.4f} "
            f"pH={p_h:.4f} pD={p_d:.4f} pA={p_a:.4f} best={draw_eval['best_sym']} second={draw_eval['second_sym']} "
            f"margin={draw_eval['margin']:.4f} spread={draw_eval['spread']:.4f} entropy={draw_eval['entropy']:.4f} "
            f"r21={ratio21:.4f} r31={ratio31:.4f} strength={strength_score:.4f} "
            f"closeness={draw_eval['closeness']:.4f} score={draw_eval['score']:.4f} "
            f"draw_candidate={int(bool(draw_eval['ok']))} weak_draw_candidate={int(weak_draw_candidate)} "
            f"draw_branch_candidate={int(bool(draw_branch_eval['ok']))} draw_branch_score={float(draw_branch_eval['score']):.4f} "
            f"away_value_candidate={int(bool(away_value_eval['ok']))} away_value_score={float(away_value_eval['score']):.4f} "
            f"reason={draw_eval['reason']} flab_trial_flag={flab_trial_flag or '-'}"
        )
        match_id = _safe_text(row.get("match_id"), f"match_no_{m:02d}")
        print(
            f"[BUYPLAN_BASE] match_id={match_id} "
            f"pH={p_h:.4f} pA={p_a:.4f} pD={p_d:.4f} diff={diff_ha_base:.4f} "
            f"base_pick={base_pick} decision_reason={decision_reason}"
        )
        print(
            f"[BUYPLAN_BASE_COMPARE] mode={base_mode} match_no={m:02d} "
            f"pH={p_h:.4f} pD={p_d:.4f} pA={p_a:.4f} best={best_symbol_prob} second={second_symbol_prob} "
            f"top_gap={top_gap:.4f} spread={spread_value:.4f} entropy={float(draw_eval['entropy']):.4f} "
            f"r21={ratio21:.4f} r31={ratio31:.4f} strength={strength_score:.4f} "
            f"closeness={closeness:.4f} base_pick={base_pick} reason={decision_reason}"
        )
        tmp_plan = MatchPlan(
            match_no=m,
            toto_round_id=round_id,
            league=league,
            home_team=home,
            away_team=away,
            status="OK",
            base_pick=base_pick,
            best=best_symbol,
            second=second_symbol,
            third=third_symbol,
            margin=margin_value,
            p_home=p_h,
            p_draw=p_d,
            p_away=p_a,
        )
        rel_eval = _relative_eval_map(tmp_plan)
        for sym in ["1", "0", "2"]:
            rs = rel_eval[sym]
            print(
                f"[BUYPLAN_RELATIVE_EVAL] match_no={m:02d} symbol={sym} prob={rs['prob']:.4f} "
                f"rank={int(rs['rank'])} top_prob={rs['top_prob']:.4f} ratio_to_top={rs['ratio_to_top']:.4f} "
                f"gap_to_above={rs['gap_to_above']:.4f} gap_to_below={rs['gap_to_below']:.4f} "
                f"is_close_branch={int(_is_close_branch(tmp_plan, sym))}"
            )
        plans.append(
            MatchPlan(
                match_no=m,
                toto_round_id=round_id,
                league=league,
                home_team=home,
                away_team=away,
                status="OK",
                base_pick=base_pick,
                best=best_symbol,
                second=second_symbol,
                third=third_symbol,
                margin=margin_value,
                p_home=p_h,
                p_draw=p_d,
                p_away=p_a,
                p_best=float(best_p),
                p_second=float(second_p),
                prob_best_pick=best_symbol,
                prob_second_pick=second_symbol,
                prob_margin=margin_value,
                base_from_predicted=base_from_predicted,
                diff_ha=float(diff_ha),
                closeness=float(closeness),
                closeness_effective=float(draw_bias["closeness_effective"]),
                d_weight=float(draw_bias["d_weight"]),
                buyplan_choice=str(draw_bias["buyplan_choice"]),
                buyplan_reason=str(draw_bias["reason"]),
                entropy=float(draw_eval["entropy"]),
                draw_candidate=bool(draw_eval["ok"]),
                draw_candidate_reason=str(draw_eval["reason"]),
                draw_score=float(draw_eval["score"]),
                d_score_close=float(d_score_close),
                d_score_stall=float(d_score_stall),
                d_score_total=float(d_score_total),
                weak_draw_candidate=weak_draw_candidate,
                draw_branch_candidate=bool(draw_branch_eval["ok"]),
                draw_branch_score=float(draw_branch_eval["score"]),
                away_value_candidate=bool(away_value_eval["ok"]),
                away_value_score=float(away_value_eval["score"]),
                top_gap=float(top_gap),
                spread=float(spread_value),
                ratio21=float(ratio21),
                ratio31=float(ratio31),
                ratio32=float(ratio32),
                norm_entropy=float(norm_entropy),
                strength_score=float(strength_score),
                flab_trial_flag=str(flab_trial_flag),
                flab_trial_score=float(flab_trial_score),
                match_type_flags=str(match_type_flags),
                match_type_primary=str(match_type_primary),
                lab_matchup_edge=float(lab_matchup_edge),
                lab_style_conflict=bool(lab_style_conflict),
                lab_low_event=bool(lab_low_event),
                context_primary_pick=str(context_primary_pick),
                context_secondary_pick=str(context_secondary_pick),
                context_risk_level=str(context_risk_level),
                context_ticket_guidance=str(context_ticket_guidance),
                context_decision_summary=str(context_decision_summary),
                reason=decision_reason,
            )
        )

    return plans


def _ticket_key(values: List[str]) -> str:
    return "".join(values)


def _build_ticket_from_flips(plans: List[MatchPlan], flips: Dict[int, int]) -> List[str]:
    # flip value: 1 -> second, 2 -> third, 3 -> draw(0)
    ticket = [p.base_pick for p in plans]
    for idx, which in flips.items():
        if which == 1:
            ticket[idx] = plans[idx].second
        elif which == 2:
            ticket[idx] = plans[idx].third
        elif which == 3:
            ticket[idx] = "0"
    return ticket


def _flip_desc(plans: List[MatchPlan], flips: Dict[int, int]) -> str:
    if not flips:
        return "base"
    parts = []
    for idx in sorted(flips.keys()):
        p = plans[idx]
        before = p.base_pick
        after = p.second if flips[idx] == 1 else (p.third if flips[idx] == 2 else "0")
        parts.append(f"M{p.match_no:02d}:{before}->{after}")
    return ", ".join(parts)


def _mode_for_ticket_index(ticket_index_zero_based: int) -> str:
    idx = ticket_index_zero_based + 1
    if 1 <= idx <= 3:
        return "lock_strict"
    if 4 <= idx <= 8:
        return "prob_faithful"
    return "experimental"


def _target_draw_range_for_mode(mode: str) -> Tuple[int, int]:
    if mode == "lock_strict":
        return int(TARGET_DRAW_MIN_LOCK), int(TARGET_DRAW_MAX_LOCK)
    if mode == "prob_faithful":
        return int(TARGET_DRAW_MIN_PROB), int(TARGET_DRAW_MAX_PROB)
    return int(TARGET_DRAW_MIN_EXP), int(TARGET_DRAW_MAX_EXP)


def _symbol_count(ticket: List[str], sym: str) -> int:
    return sum(1 for x in ticket if x == sym)


def _ratio_str(v: int, total: int) -> str:
    if total <= 0:
        return "0.000"
    return f"{(v / total):.3f}"


def _generate_tickets(plans: List[MatchPlan], warnings: List[str]) -> Tuple[List[List[str]], List[str], Dict[str, int]]:
    tickets: List[List[str]] = []
    descs: List[str] = []
    stats = {
        "duplicate_skips": 0,
        "generated": 0,
        "attempted_candidates": 0,
        "unique_ticket_count": 0,
        "duplicate_count": 0,
        "locked_count": 0,
        "second_zero_matches_all_ok": 0,
        "second_zero_matches": 0,
        "second_zero_applied": 0,
        "total_zero_count": 0,
        "total_one_count": 0,
        "total_two_count": 0,
        "draw_bias_fired_total": 0,
        "draw_bias_fired_lock_strict": 0,
        "draw_bias_fired_prob_faithful": 0,
        "draw_bias_fired_experimental": 0,
        "draw_bias_fired_pd_avg": 0.0,
        "draw_gate_candidate_matches": 0,
        "draw_gate_candidate_rate": "0.000",
        "unique_repair_count": 0,
        "same_symbol_cap_adjust_count": 0,
        "zero_cap_adjust_count": 0,
        "zero_cap_adjust_matches": "",
        "same_symbol_cap_adjust_matches": "",
        "unique_repair_matches": "",
        "lock02_flips": 0,
        "lock03_flips": 0,
        "draw_distribution_adjust_count": 0,
        "draw_distribution_adjust_matches": "",
        "extreme_margin_release_count": 0,
        "extreme_margin_release_matches": "",
        "weak_draw_count": 0,
        "weak_draw_matches": "",
        "weak_draw_selected_for_exp": "",
        "weak_draw_selected_for_distribution": "",
        "unique_before_duplicate_count": 0,
        "unique_after_duplicate_count": 0,
        "unique_repair_fallback_used": False,
        "unique_repair_protected_cells_skipped": 0,
    }

    mode_stats: Dict[str, Dict[str, float]] = {
        "lock_strict": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
        "prob_faithful": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
        "experimental": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
    }

    ok_all = [i for i, p in enumerate(plans) if p.status == "OK" and p.prob_margin is not None]
    sorted_all = sorted(
        ok_all,
        key=lambda i: (
            float(plans[i].strength_score),
            float(plans[i].spread),
            -float(plans[i].entropy),
            -float(plans[i].ratio31),
            i,
        ),
    )
    sorted_non_lock = list(sorted_all)
    stats["second_zero_matches_all_ok"] = sum(1 for i in ok_all if plans[i].second == "0")
    stats["second_zero_matches"] = sum(1 for i in sorted_non_lock if plans[i].second == "0")

    draw_candidates_all = [i for i in ok_all if plans[i].draw_candidate]
    weak_draw_candidates_all = [i for i in ok_all if plans[i].weak_draw_candidate]
    draw_branch_candidates_all = [i for i in ok_all if plans[i].draw_branch_candidate]
    away_value_candidates_all = [i for i in ok_all if plans[i].away_value_candidate]
    stats["draw_gate_candidate_matches"] = int(len(draw_candidates_all))
    stats["draw_gate_candidate_rate"] = _ratio_str(int(len(draw_candidates_all)), len(ok_all))
    stats["weak_draw_count"] = int(len(weak_draw_candidates_all))
    stats["weak_draw_matches"] = "; ".join([f"M{plans[i].match_no:02d}" for i in weak_draw_candidates_all][:30])

    mode_by_ticket = [_mode_for_ticket_index(i) for i in range(REQUIRED_TICKET_COUNT)]
    lock_ticket_indices = {i for i, mode in enumerate(mode_by_ticket) if mode == "lock_strict"}
    immutable_ticket_indices = {0}
    non_lock_ticket_indices = [i for i in range(REQUIRED_TICKET_COUNT) if i not in immutable_ticket_indices]
    branch_audit_baseline: Dict[Tuple[str, int], Tuple[int, int, int, int]] = {}
    # 04..10（prob/exp）横断で、同一試合への揺らし集中を抑える
    gradual_flip_usage_by_match: Dict[int, int] = {}

    def _log_stage_symbol_totals(stage: str) -> None:
        if not tickets or not plans:
            print(f"[BUYPLAN_STAGE_COUNTS] stage={stage} H=0 D=0 A=0 total_cells=0")
            return
        one_count = sum(_symbol_count(t, "1") for t in tickets)
        zero_count = sum(_symbol_count(t, "0") for t in tickets)
        two_count = sum(_symbol_count(t, "2") for t in tickets)
        total_cells = len(tickets) * len(plans)
        print(
            f"[BUYPLAN_STAGE_COUNTS] stage={stage} "
            f"H={one_count} D={zero_count} A={two_count} total_cells={total_cells}"
        )

    def _log_stage_match_counts(stage: str) -> None:
        if not tickets or not plans:
            print(f"[BUYPLAN_STAGE_MATCH_COUNTS] stage={stage} rows=0")
            return
        for mi, p in enumerate(plans):
            col = [t[mi] for t in tickets if mi < len(t)]
            h = sum(1 for x in col if x == "1")
            d = sum(1 for x in col if x == "0")
            a = sum(1 for x in col if x == "2")
            dominant = "1"
            dominant_n = h
            if d > dominant_n:
                dominant, dominant_n = "0", d
            if a > dominant_n:
                dominant, dominant_n = "2", a
            locked = int(dominant_n >= int(round(len(tickets) * 0.8)))
            print(
                f"[BUYPLAN_STAGE_MATCH_COUNTS] stage={stage} match_no={p.match_no:02d} "
                f"H={h} D={d} A={a} dominant={dominant} dominant_count={dominant_n} "
                f"locked80={locked}"
            )

    def _log_stage_branch_audit(stage: str) -> None:
        if not tickets or not plans:
            print(f"[BUYPLAN_BRANCH_AUDIT] stage={stage} rows=0")
            return
        mode_order = ["lock_strict", "prob_faithful", "experimental"]
        ticket_modes = mode_by_ticket[: len(tickets)]
        for mode in mode_order:
            idxs = [i for i, m in enumerate(ticket_modes) if m == mode]
            if not idxs:
                continue
            for mi, p in enumerate(plans):
                best_n = 0
                second_n = 0
                third_n = 0
                other_n = 0
                for ti in idxs:
                    sym = str(tickets[ti][mi])
                    if sym == str(p.best):
                        best_n += 1
                    elif sym == str(p.second):
                        second_n += 1
                    elif sym == str(p.third):
                        third_n += 1
                    else:
                        other_n += 1
                print(
                    f"[BUYPLAN_BRANCH_AUDIT] stage={stage} scenario={mode} "
                    f"match_no={p.match_no:02d} best={best_n} second={second_n} third={third_n} other={other_n}"
                )
                key = (mode, int(p.match_no))
                if stage == "after_ticket_generation":
                    branch_audit_baseline[key] = (best_n, second_n, third_n, other_n)
                else:
                    b_best, b_second, b_third, b_other = branch_audit_baseline.get(key, (0, 0, 0, 0))
                    print(
                        f"[BUYPLAN_BRANCH_DELTA] stage={stage} scenario={mode} "
                        f"match_no={p.match_no:02d} d_best={best_n - b_best} "
                        f"d_second={second_n - b_second} d_third={third_n - b_third} d_other={other_n - b_other}"
                    )

    def _log_ticket_branch_summary(stage: str) -> None:
        if not tickets or not plans:
            print(f"[BUYPLAN_TICKET_BRANCH] stage={stage} rows=0")
            return
        for ti, t in enumerate(tickets):
            mode = _mode_for_ticket_index(ti)
            best_n = 0
            second_n = 0
            third_n = 0
            other_n = 0
            for mi, p in enumerate(plans):
                sym = str(t[mi])
                if sym == str(p.best):
                    best_n += 1
                elif sym == str(p.second):
                    second_n += 1
                elif sym == str(p.third):
                    third_n += 1
                else:
                    other_n += 1
            print(
                f"[BUYPLAN_TICKET_BRANCH] stage={stage} ticket={ti+1:02d} mode={mode} "
                f"best={best_n} second={second_n} third={third_n} other={other_n}"
            )

    def _alt_symbols_by_preference(plan: MatchPlan) -> List[str]:
        order = [plan.best, plan.second, plan.third]
        out = []
        for s in order + ["0", "1", "2"]:
            if s in {"0", "1", "2"} and s not in out:
                out.append(s)
        return out

    def _best_non_draw_prob_alt(plan: MatchPlan, current: str) -> str:
        preferred = "1" if float(plan.p_home) >= float(plan.p_away) else "2"
        if preferred != current:
            return preferred
        return "2" if preferred == "1" else "1"

    def _context_guidance(plan: MatchPlan) -> str:
        return str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()

    def _context_risk(plan: MatchPlan) -> str:
        return str(getattr(plan, "context_risk_level", "") or "").strip().lower()

    def _context_draw_prefer(plan: MatchPlan) -> bool:
        guidance = _context_guidance(plan)
        risk = _context_risk(plan)
        return bool(guidance in {"draw_cover", "avoid_main"} or risk == "draw_watch")

    def _context_keep_main(plan: MatchPlan) -> bool:
        guidance = _context_guidance(plan)
        risk = _context_risk(plan)
        return bool(guidance == "main_only" or risk == "fixed")

    def _context_lab_cover(plan: MatchPlan) -> bool:
        return _context_guidance(plan) == "lab_cover"

    def _context_flip_ready(plan: MatchPlan) -> bool:
        risk = _context_risk(plan)
        return bool(risk in {"volatile", "caution", "draw_watch"} or _context_draw_prefer(plan) or _context_lab_cover(plan))

    def _match_priority_indices() -> List[int]:
        return sorted(
            ok_all,
            key=lambda i: (
                0 if plans[i].second == "0" else 1,
                0 if plans[i].draw_branch_candidate else 1,
                0 if plans[i].away_value_candidate else 1,
                -float(getattr(plans[i], "d_score_total", 0.0)),
                -float(getattr(plans[i], "d_score_stall", 0.0)),
                -float(plans[i].draw_branch_score),
                -float(plans[i].away_value_score),
                float(plans[i].strength_score),
                float(plans[i].spread),
                -float(plans[i].entropy),
                -float(plans[i].ratio31),
                i,
            ),
        )

    def _probexp_match_priority_indices() -> List[int]:
        return sorted(
            ok_all,
            key=lambda i: (
                0 if _context_draw_prefer(plans[i]) else 1,
                0 if _allow_draw_for_plan(plans[i]) else 1,
                0 if _context_flip_ready(plans[i]) else 1,
                1 if _context_keep_main(plans[i]) else 0,
                -float(getattr(plans[i], "d_score_total", 0.0)),
                -float(getattr(plans[i], "d_score_stall", 0.0)),
                -float(getattr(plans[i], "d_score_close", 0.0)),
                0 if plans[i].second == "0" else 1,
                0 if plans[i].draw_candidate else 1,
                0 if plans[i].weak_draw_candidate else 1,
                -float(plans[i].draw_score),
                -float(plans[i].p_draw),
                float(plans[i].strength_score),
                float(plans[i].spread),
                -float(plans[i].entropy),
                i,
            ),
        )

    def _is_strong_plan(p: MatchPlan) -> bool:
        # 強試合はロック準拠票や最大探索で原則触らない
        m = float(p.margin if p.margin is not None else 0.0)
        return bool(
            m >= float(BASE_TOP_GAP_STRONG)
            or float(p.strength_score) >= float(SHAPE_STRENGTH_STRONG)
            or max(float(p.p_home), float(p.p_draw), float(p.p_away)) >= float(BUYPLAN_2AXIS_MAX_STRONG)
        )

    def _build_base_ticket(flips: Dict[int, int]) -> List[str]:
        return _build_ticket_from_flips(plans, flips)

    def _allow_draw_for_plan(p: MatchPlan) -> bool:
        return bool(p.draw_candidate or p.weak_draw_candidate)

    def _draw_priority_components(p: MatchPlan) -> Tuple[float, float, float, float, float]:
        return (
            float(getattr(p, "d_score_total", 0.0)),
            float(getattr(p, "d_score_stall", 0.0)),
            float(getattr(p, "d_score_close", 0.0)),
            float(getattr(p, "draw_score", 0.0)),
            float(getattr(p, "p_draw", 0.0)),
        )

    def _draw_add_sort_key(p: MatchPlan, mi: int) -> Tuple[float, float, float, float, float, float, int]:
        return (
            -float(getattr(p, "d_score_total", 0.0)),
            -float(getattr(p, "d_score_stall", 0.0)),
            -float(getattr(p, "d_score_close", 0.0)),
            -float(getattr(p, "draw_score", 0.0)),
            -float(getattr(p, "p_draw", 0.0)),
            float(p.margin if p.margin is not None else 999.0),
            mi,
        )

    def _draw_remove_sort_key(p: MatchPlan, mi: int) -> Tuple[float, float, float, float, float, float, int]:
        return (
            float(getattr(p, "d_score_total", 0.0)),
            float(getattr(p, "d_score_stall", 0.0)),
            float(getattr(p, "d_score_close", 0.0)),
            float(getattr(p, "draw_score", 0.0)),
            float(getattr(p, "p_draw", 0.0)),
            float(p.margin if p.margin is not None else 999.0),
            mi,
        )

    def _best_release_alt(plan: MatchPlan, current: str, prefer_third: bool = False) -> str:
        ordered = [plan.third, plan.second] if prefer_third else [plan.second, plan.third]
        for alt in ordered:
            if alt == current:
                continue
            if alt == "0" and (not _allow_draw_for_plan(plan)):
                continue
            if alt in {"0", "1", "2"}:
                return alt
        return current

    def _enforce_ticket_draw_range(
        ti: int,
        phase: str,
        protected_cells: set,
        allow_protected: bool,
    ) -> Tuple[int, List[str]]:
        mode = mode_by_ticket[ti]
        min_d, max_d = _target_draw_range_for_mode(mode)
        t = tickets[ti]
        changes: List[str] = []
        changed = 0

        # reduce if too many draws
        while _symbol_count(t, "0") > max_d:
            cands = []
            for mi, sym in enumerate(t):
                if sym != "0":
                    continue
                if (ti, mi) in protected_cells and not allow_protected:
                    continue
                p = plans[mi]
                if mode != "lock_strict" and _is_close_branch(p, "0"):
                    _log_repair_keep(ti + 1, p, "0", "close_branch_protected_draw_range_reduce")
                    continue
                alts = sorted(
                    [s for s in [p.second, p.third, p.base_pick] if s != "0"],
                    key=lambda s: (-_relative_branch_score(p, s), s),
                )
                repl = alts[0] if alts else p.base_pick
                cands.append((_draw_remove_sort_key(p, mi), repl))
            if not cands:
                break
            cands = sorted(cands, key=lambda x: (x[0], x[1]))
            _, repl = cands[0]
            mi = cands[0][0][-1]
            before = t[mi]
            t[mi] = repl
            changed += 1
            changes.append(f"{phase}:ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->{repl}")
            _log_repair_drop(ti + 1, plans[mi], before, repl, f"{phase}_reduce")

        # add if too few draws
        while _symbol_count(t, "0") < min_d:
            cands = []
            for mi, sym in enumerate(t):
                if sym == "0":
                    continue
                if (ti, mi) in protected_cells and not allow_protected:
                    continue
                p = plans[mi]
                if not _allow_draw_for_plan(p):
                    continue
                if mode != "lock_strict" and _is_close_branch(p, sym):
                    _log_repair_keep(ti + 1, p, sym, "close_branch_protected_draw_range_add_scan")
                cands.append(_draw_add_sort_key(p, mi))
            if not cands:
                break
            cands = sorted(cands)
            mi = cands[0][-1]
            before = t[mi]
            t[mi] = "0"
            changed += 1
            changes.append(f"{phase}:ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->0")
            _log_repair_drop(ti + 1, plans[mi], before, "0", f"{phase}_add")

        # lock_strict は draw 範囲を必ず満たす（保護よりレンジ優先）
        if mode == "lock_strict":
            while _symbol_count(t, "0") > max_d:
                removable = []
                for mi, sym in enumerate(t):
                    if sym != "0":
                        continue
                    p = plans[mi]
                    alts = [s for s in [p.second, p.third, p.base_pick] if s != "0"]
                    repl = alts[0] if alts else ("1" if p.p_home >= p.p_away else "2")
                    removable.append((_draw_remove_sort_key(p, mi), repl))
                if not removable:
                    break
                removable = sorted(removable, key=lambda x: (x[0], x[1]))
                _, repl = removable[0]
                mi = removable[0][0][-1]
                before = t[mi]
                t[mi] = repl
                changed += 1
                changes.append(f"{phase}:ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->{repl}")
                _log_repair_drop(ti + 1, plans[mi], before, repl, f"{phase}_force_reduce")
            while _symbol_count(t, "0") < min_d:
                addable = []
                for mi, sym in enumerate(t):
                    if sym == "0":
                        continue
                    p = plans[mi]
                    if not _allow_draw_for_plan(p):
                        continue
                    addable.append(_draw_add_sort_key(p, mi))
                if not addable:
                    break
                addable = sorted(addable)
                mi = addable[0][-1]
                before = t[mi]
                t[mi] = "0"
                changed += 1
                changes.append(f"{phase}:ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->0")
                _log_repair_drop(ti + 1, plans[mi], before, "0", f"{phase}_force_add")
        tickets[ti] = t
        return changed, changes

    def _apply_exp_weak_draw(ticket: List[str], ticket_no: int, mode: str) -> Tuple[List[str], Optional[int]]:
        if mode != "experimental":
            return ticket, None
        strong_zero_count = 0
        for mi, p in enumerate(plans):
            if p.draw_candidate and ticket[mi] == "0":
                strong_zero_count += 1
        if strong_zero_count > 0:
            return ticket, None
        weak_pool = []
        for mi in weak_draw_candidates_all:
            if ticket[mi] == "0":
                continue
            p = plans[mi]
            weak_pool.append(
                (
                    -float(getattr(p, "d_score_total", 0.0)),
                    -float(getattr(p, "d_score_stall", 0.0)),
                    float(p.margin if p.margin is not None else 999.0),
                    -float(p.entropy),
                    -float(p.p_draw),
                    mi,
                )
            )
        if not weak_pool:
            return ticket, None
        weak_pool = sorted(weak_pool, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
        mi = weak_pool[0][5]
        t = list(ticket)
        before = t[mi]
        t[mi] = "0"
        print(
            f"[BUYPLAN_EXP_WEAK_DRAW] ticket={ticket_no:02d} match=M{plans[mi].match_no:02d} "
            f"{before}->0 margin={float(plans[mi].margin if plans[mi].margin is not None else 0.0):.4f}"
        )
        return t, mi

    def _gradual_flip_kind(plan: MatchPlan, strength: int) -> int:
        # 1=second, 2=third
        if plan.away_value_candidate:
            if str(plan.second) == "2":
                return 1
            if str(plan.third) == "2":
                return 2
        if plan.draw_branch_candidate and strength >= 1:
            if plan.third != plan.base_pick and float(plan.draw_branch_score) >= 0.72:
                return 2
            return 1
        if strength <= 0:
            return 1
        if float(plan.ratio31) >= 0.86 and float(plan.entropy) >= 1.08:
            return 2
        if strength >= 2 and float(plan.ratio31) >= 0.78:
            return 2
        return 1

    def _strong_break_flip_kind(plan: MatchPlan) -> int:
        if str(plan.third) != str(plan.base_pick):
            return 2
        if str(plan.second) != str(plan.base_pick):
            return 1
        return 0

    def _strong_break_away_bias(plan: MatchPlan) -> float:
        if str(plan.base_pick) == "1" and str(plan.third) == "2":
            return 1.0
        if str(plan.base_pick) == "1" and str(plan.second) == "0":
            return 0.8
        if str(plan.base_pick) == "0" and str(plan.third) == "2":
            return 0.6
        return 0.3

    def _strong_break_norm_margin(plan: MatchPlan) -> float:
        margin = float(plan.margin if plan.margin is not None else 0.0)
        return max(0.0, min(1.0, 1.0 - abs(margin - 0.065) / 0.065))

    def _strong_break_close_second(plan: MatchPlan) -> float:
        margin = float(plan.margin if plan.margin is not None else 1.0)
        return max(0.0, min(1.0, 1.0 - margin))

    def _strong_break_targets() -> List[int]:
        scores: List[Tuple[float, int]] = []
        for i in ok_all:
            p = plans[i]
            margin = float(p.margin if p.margin is not None else 0.0)
            if margin < float(STRONG_BREAK_MARGIN_MIN) or margin > float(STRONG_BREAK_MARGIN_MAX):
                continue
            if bool(p.draw_candidate or p.weak_draw_candidate):
                continue
            if _strong_break_flip_kind(p) != 2:
                continue
            score = (
                float(STRONG_BREAK_W_MARGIN) * _strong_break_norm_margin(p)
                + float(STRONG_BREAK_W_AWAY) * _strong_break_away_bias(p)
                + float(STRONG_BREAK_W_SECOND) * _strong_break_close_second(p)
                + float(STRONG_BREAK_W_VOLATILITY) * float(max(0.0, min(1.0, p.norm_entropy)))
            )
            scores.append((score, i))
        scores = sorted(scores, key=lambda x: (-x[0], x[1]))
        targets: List[int] = []
        used_teams = set()
        for _, i in scores:
            p = plans[i]
            team_key = tuple(sorted([str(p.home_team), str(p.away_team)]))
            if team_key in used_teams:
                continue
            targets.append(i)
            used_teams.add(team_key)
            if len(targets) >= max(0, int(STRONG_BREAK_TARGET_COUNT)):
                break
        return targets

    strong_break_targets = set(_strong_break_targets())

    def _build_gradual_ticket(ticket_index: int) -> Tuple[List[str], Dict[int, int], str]:
        # ticket_index: 0-based, target for 04..10 is 3..9
        src = _probexp_match_priority_indices()
        per_match_cap = int(MAX_FLIPS_PER_MATCH_PROB) if ticket_index <= 7 else int(MAX_FLIPS_PER_MATCH_EXP)
        if ticket_index == 3:
            # 候補04: 本線寄り。keep_main を優先し、軽い second だけ許す。
            src = sorted(
                ok_all,
                key=lambda i: (
                    1 if _context_keep_main(plans[i]) else 0,
                    0 if _context_draw_prefer(plans[i]) else 1,
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    float(plans[i].strength_score),
                    i,
                ),
            )
        elif ticket_index == 6:
            # 候補07: A反転補助。away_value を最優先にし、非A文脈の 0 を抑える。
            src = sorted(
                ok_all,
                key=lambda i: (
                    0 if plans[i].away_value_candidate else 1,
                    0 if _context_lab_cover(plans[i]) else 1,
                    0 if _context_flip_ready(plans[i]) else 1,
                    1 if _context_draw_prefer(plans[i]) else 0,
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    -float(getattr(plans[i], "away_value_score", 0.0)),
                    -float(getattr(plans[i], "draw_branch_score", 0.0)),
                    i,
                ),
            )
        elif ticket_index == 7:
            # 候補08: draw多め仮説（best->second中心、draw候補優先）
            src = sorted(
                ok_all,
                key=lambda i: (
                    0 if _context_draw_prefer(plans[i]) else 1,
                    0 if _allow_draw_for_plan(plans[i]) else 1,
                    0 if _context_flip_ready(plans[i]) else 1,
                    1 if _context_keep_main(plans[i]) else 0,
                    -float(getattr(plans[i], "d_score_total", 0.0)),
                    -float(getattr(plans[i], "d_score_stall", 0.0)),
                    -float(getattr(plans[i], "d_score_close", 0.0)),
                    0 if plans[i].second == "0" else 1,
                    0 if plans[i].draw_branch_candidate else 1,
                    0 if plans[i].away_value_candidate else 1,
                    -float(plans[i].draw_branch_score),
                    -float(plans[i].away_value_score),
                    -float(plans[i].draw_score),
                    float(plans[i].strength_score),
                    i,
                ),
            )
        elif ticket_index == 8:
            # 候補09: 接戦 second 専任
            src = sorted(
                [i for i in ok_all if i not in strong_break_targets],
                key=lambda i: (
                    0 if _context_draw_prefer(plans[i]) else 1,
                    0 if _allow_draw_for_plan(plans[i]) else 1,
                    0 if _context_flip_ready(plans[i]) else 1,
                    1 if _context_keep_main(plans[i]) else 0,
                    -float(getattr(plans[i], "d_score_total", 0.0)),
                    -float(getattr(plans[i], "d_score_stall", 0.0)),
                    -float(getattr(plans[i], "d_score_close", 0.0)),
                    0 if plans[i].second == "0" else 1,
                    0 if float(plans[i].margin if plans[i].margin is not None else 999.0) < float(LOCK03_MARGIN_THRESHOLD) else 1,
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    float(plans[i].prob_margin if plans[i].prob_margin is not None else (plans[i].margin if plans[i].margin is not None else 999.0)),
                    i,
                ),
            )
        elif ticket_index == 9:
            # 候補10: LAB反転優先の strong_break 探索
            src = sorted(
                ok_all,
                key=lambda i: (
                    0 if _context_lab_cover(plans[i]) else 1,
                    0 if i in strong_break_targets else 1,
                    1 if _context_keep_main(plans[i]) else 0,
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    -float(getattr(plans[i], "away_value_score", 0.0)),
                    -float(getattr(plans[i], "draw_branch_score", 0.0)),
                    i,
                ),
            )
        # default level profile for candidate04..10
        profile: Dict[int, Tuple[int, int]] = {
            3: (0, 0),  # almost base
            4: (1, 0),  # +1 second
            5: (2, 0),  # +2 second
            6: (3, 1),  # +3 with up to 1 third
            7: (4, 1),  # +4 with up to 1 third（draw仮説）
            8: (4, 0),  # +4 second only（接戦 second専任）
            9: (6, 2),  # +6 with up to 2 third（最大探索）
        }
        # configurable sway degree table for candidate04..07
        for ticket_no, degree in SWAY_DEGREE_TABLE.items():
            idx = int(ticket_no) - 1
            if 3 <= idx <= 6:
                profile[idx] = _degree_to_profile(int(degree))
        # 04〜06は second-only を維持。07・08のみ third を最大1つまで許可する。
        profile[3] = (profile.get(3, (0, 0))[0], 0)
        profile[4] = (profile.get(4, (0, 0))[0], 0)
        profile[5] = (profile.get(5, (0, 0))[0], 0)
        profile[6] = (profile.get(6, (0, 0))[0], min(1, profile.get(6, (0, 0))[1]))
        profile[7] = (profile.get(7, (0, 0))[0], min(1, profile.get(7, (0, 0))[1]))
        profile[3] = (1, 0)  # ticket04: 本線寄りに固定
        profile[4] = (2, 0)  # ticket05: D軸の second-only
        profile[5] = (3, 0)  # ticket06: J2 D補助の second-only
        profile[6] = (3, 1)  # ticket07: A補助で軽い third を許可
        profile[7] = (4, 1)  # ticket08: 接戦Dを厚め
        n_flip, n_third_target = profile.get(ticket_index, (0, 0))
        flips: Dict[int, int] = {}
        third_used = 0
        for mi in src:
            if len(flips) >= n_flip:
                break
            if gradual_flip_usage_by_match.get(mi, 0) >= max(0, int(per_match_cap)):
                continue
            p = plans[mi]
            if p.status != "OK":
                continue
            if ticket_index == 9:
                kind = _strong_break_flip_kind(p)
                if mi not in strong_break_targets:
                    continue
            elif ticket_index == 8:
                if p.margin is None or float(p.margin) >= float(LOCK03_MARGIN_THRESHOLD):
                    continue
                if str(p.second) == str(p.base_pick):
                    continue
                kind = 1
            else:
                kind = _gradual_flip_kind(p, n_third_target - third_used)
                if ticket_index in {6, 7} and kind == 2:
                    if _is_strong_plan(p):
                        kind = 1
                    elif mi in strong_break_targets:
                        kind = 1
                    elif p.margin is None or float(p.margin) > max(float(LOCK03_MARGIN_THRESHOLD), 0.08):
                        kind = 1
                    elif float(p.ratio31) < 0.72:
                        kind = 1
            if kind <= 0:
                continue
            if kind == 2 and third_used >= n_third_target:
                kind = 1
            if kind == 2 and p.third == p.base_pick:
                kind = 1
            if kind == 1 and p.second == p.base_pick:
                if p.third != p.base_pick and third_used < n_third_target:
                    kind = 2
                else:
                    continue
            if ticket_index == 9 and kind != 2:
                continue
            flips[mi] = kind
            if kind == 2:
                third_used += 1
        ticket = _build_base_ticket(flips)
        if ticket_index == 7:
            desc = "draw_hypothesis_08"
        elif ticket_index == 8:
            desc = "close_reversal_09"
        elif ticket_index == 9:
            desc = "strong_break_10"
        else:
            desc = f"gradual_sway_{ticket_index+1:02d}"
        return ticket, flips, desc

    def record_mode_stats(mode: str, ticket: List[str], flips: Dict[int, int]) -> None:
        m = mode_stats[mode]
        m["tickets"] += 1
        m["zero_count"] += _symbol_count(ticket, "0")
        m["flip_count"] += len(flips)
        for idx in flips.keys():
            pm = plans[idx].prob_margin
            if pm is not None:
                m["margin_sum"] += float(pm)
                m["margin_n"] += 1

    def candidate_flip_maps(mode: str) -> List[Dict[int, int]]:
        if mode == "lock_strict":
            src = sorted_non_lock
            cands = [{}]
            if len(src) >= 1:
                cands.append({src[0]: 1})
            if len(src) >= 2:
                cands.append({src[0]: 1, src[1]: 1})
                cands.append({src[1]: 1})
            if len(src) >= 3:
                cands.append({src[2]: 1})
            return cands

        if mode == "prob_faithful":
            src = sorted_all
            cands: List[Dict[int, int]] = []
            if len(src) >= 2:
                n = len(src)
                anchors = sorted(set([0, max(0, n // 3), max(0, n // 2), max(0, (2 * n) // 3), max(0, n - 2)]))
                for a in anchors:
                    b = a + 1 if a + 1 < n else a - 1
                    if b >= 0:
                        cands.append({src[a]: 1, src[b]: 1})
                    c = a + 2 if a + 2 < n else None
                    if c is not None:
                        cands.append({src[a]: 1, src[b]: 1, src[c]: 1})
            dedup = []
            seen_keys = set()
            for fm in cands:
                key = tuple(sorted(fm.items()))
                if key not in seen_keys:
                    seen_keys.add(key)
                    dedup.append(fm)
            return dedup

        src = _match_priority_indices()
        cands = [{}]
        if len(src) >= 1:
            cands.append({src[0]: 2})
            cands.append({src[0]: 1})
        if len(src) >= 2:
            cands.append({src[0]: 2, src[1]: 1})
            cands.append({src[0]: 1, src[1]: 2})
            cands.append({src[0]: 1, src[1]: 1})
        if len(src) >= 3:
            cands.append({src[0]: 1, src[1]: 1, src[2]: 2})
        return cands

    pool_by_mode = {
        "lock_strict": candidate_flip_maps("lock_strict"),
        "prob_faithful": candidate_flip_maps("prob_faithful"),
        "experimental": candidate_flip_maps("experimental"),
    }
    pool_pos = {"lock_strict": 0, "prob_faithful": 0, "experimental": 0}
    seen = set()
    exp_ticket_indices: List[int] = []
    lock02_flipped_matches: set[int] = set()

    for t_idx in range(REQUIRED_TICKET_COUNT):
        # Lock01は完全immutable（base固定）
        if t_idx == 0:
            base_ticket = [p.base_pick for p in plans]
            tickets.append(base_ticket)
            descs.append("lock_base_immutable")
            seen.add(_ticket_key(base_ticket))
            mode = mode_by_ticket[t_idx]
            record_mode_stats(mode, base_ticket, {})
            print("[BUYPLAN_LOCK_IMMUTABLE] ticket=01 reason=lock_base_protected")
            continue
        # Lock02/03はmargin閾値でsecond採用して固定バリエーションを作る。
        if t_idx in lock_ticket_indices:
            ticket = [p.base_pick for p in plans]
            flips = 0
            threshold = LOCK02_MARGIN_THRESHOLD if t_idx == 1 else LOCK03_MARGIN_THRESHOLD
            max_flips = LOCK02_MAX_FLIPS if t_idx == 1 else LOCK03_MAX_FLIPS
            if t_idx == 1:
                # ticket02: margin小を second に寄せる軽い保険券
                candidates: List[Tuple[float, float, int]] = []
                for mi, p in enumerate(plans):
                    if p.status != "OK" or p.margin is None:
                        continue
                    if _is_strong_plan(p):
                        continue
                    if float(p.margin) >= float(threshold):
                        continue
                    if str(p.second) == str(p.base_pick):
                        continue
                    candidates.append(
                        (
                            0 if str(p.second) == "0" else 1,
                            -float(getattr(p, "d_score_total", 0.0)),
                            -float(getattr(p, "d_score_stall", 0.0)),
                            float(p.margin),
                            float(p.prob_margin if p.prob_margin is not None else p.margin),
                            mi,
                        )
                    )
                candidates = sorted(candidates, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
                for _, _, _, _, _, mi in candidates[: max(0, int(max_flips))]:
                    p = plans[mi]
                    ticket[mi] = p.second
                    if ticket[mi] != p.base_pick:
                        flips += 1
                        lock02_flipped_matches.add(mi)
            else:
                # ticket03: draw寄り試合だけを 0 に寄せる軽い保険券
                candidates: List[Tuple[int, int, float, float, int]] = []
                for mi, p in enumerate(plans):
                    if p.status != "OK" or p.margin is None:
                        continue
                    if _is_strong_plan(p):
                        continue
                    if mi in lock02_flipped_matches:
                        continue
                    if str(p.base_pick) == "0":
                        continue
                    if str(p.second) != "0":
                        continue
                    margin_v = float(p.margin)
                    d_first_rank = 0 if str(p.prob_best_pick) == "0" else 1
                    draw_rank = 0 if p.draw_candidate else 1
                    tight_rank = 0 if margin_v < float(threshold) else 1
                    if d_first_rank == 1 and draw_rank == 1 and tight_rank == 1:
                        continue
                    candidates.append(
                        (
                            d_first_rank,
                            draw_rank,
                            -float(getattr(p, "d_score_total", 0.0)),
                            -float(getattr(p, "d_score_stall", 0.0)),
                            margin_v,
                            -float(p.draw_score),
                            mi,
                        )
                    )
                candidates = sorted(candidates, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6]))
                for _, _, _, _, _, _, mi in candidates[: max(0, int(max_flips))]:
                    p = plans[mi]
                    ticket[mi] = "0"
                    if ticket[mi] != p.base_pick:
                        flips += 1
            tickets.append(ticket)
            descs.append(f"lock_margin_variation_lt_{threshold:.2f}")
            seen.add(_ticket_key(ticket))
            mode = mode_by_ticket[t_idx]
            record_mode_stats(mode, ticket, {})
            if t_idx == 1:
                stats["lock02_flips"] = int(flips)
            if t_idx == 2:
                stats["lock03_flips"] = int(flips)
            print(
                f"[BUYPLAN_LOCK_VARIATION] ticket={t_idx+1:02d} margin_threshold={threshold:.2f} "
                f"max_flips={int(max_flips)} flips={flips} "
                f"strategy={'second_guard' if t_idx == 1 else 'draw_guard'}"
            )
            continue

        mode = mode_by_ticket[t_idx]
        if ENABLE_GRADUAL_SWAY and 3 <= t_idx <= 9:
            t, flips_map, desc = _build_gradual_ticket(t_idx)
            key = _ticket_key(t)
            if key in seen:
                # minimal fallback to keep uniqueness
                repaired = False
                for mi in _probexp_match_priority_indices():
                    p = plans[mi]
                    for alt in [p.second, p.third]:
                        if alt == t[mi]:
                            continue
                        cand = list(t)
                        cand[mi] = alt
                        k2 = _ticket_key(cand)
                        if k2 not in seen:
                            t = cand
                            key = k2
                            repaired = True
                            break
                    if repaired:
                        break
            seen.add(key)
            tickets.append(t)
            descs.append(desc)
            record_mode_stats(mode, t, flips_map)
            for mi in flips_map.keys():
                gradual_flip_usage_by_match[mi] = gradual_flip_usage_by_match.get(mi, 0) + 1
            print(
                f"[BUYPLAN_GRADUAL_SWAY] ticket={t_idx+1:02d} mode={mode} "
                f"flips={len(flips_map)} third_flips={sum(1 for v in flips_map.values() if v==2)}"
            )
            continue

        if mode == "experimental":
            ticket = [p.base_pick for p in plans]
            tickets.append(ticket)
            descs.append("exp_pending_coverage")
            seen.add(_ticket_key(ticket))
            exp_ticket_indices.append(t_idx)
            stats["generated"] += 1
            continue

        pool = pool_by_mode[mode]
        chosen_ticket: Optional[List[str]] = None
        chosen_desc = "base"
        chosen_flips: Dict[int, int] = {}

        while pool_pos[mode] < len(pool):
            flips = pool[pool_pos[mode]]
            pool_pos[mode] += 1
            if mode == "experimental" and sum(1 for v in flips.values() if v == 2) > 2:
                continue
            stats["attempted_candidates"] += 1
            t = _build_base_ticket(flips)
            t, weak_idx = _apply_exp_weak_draw(t, t_idx + 1, mode)
            key = _ticket_key(t)
            if key in seen:
                stats["duplicate_skips"] += 1
                continue
            seen.add(key)
            chosen_ticket = t
            chosen_desc = _flip_desc(plans, flips)
            chosen_flips = flips
            stats["generated"] += 1
            stats["second_zero_applied"] += sum(1 for idx, which in flips.items() if which == 1 and plans[idx].second == "0")
            if weak_idx is not None:
                w = f"ticket={t_idx+1:02d} M{plans[weak_idx].match_no:02d}"
                stats["weak_draw_selected_for_exp"] = (
                    f"{stats['weak_draw_selected_for_exp']}; {w}".strip("; ").strip()
                )
            break

        if chosen_ticket is None:
            fallback = _build_base_ticket({})
            fallback, weak_idx = _apply_exp_weak_draw(fallback, t_idx + 1, mode)
            key = _ticket_key(fallback)
            if key in seen:
                repaired = False
                for i in _probexp_match_priority_indices():
                    for sym in _alt_symbols_by_preference(plans[i]):
                        if sym == fallback[i]:
                            continue
                        cand = list(fallback)
                        cand[i] = sym
                        k2 = _ticket_key(cand)
                        if k2 not in seen:
                            fallback = cand
                            key = k2
                            repaired = True
                            break
                    if repaired:
                        break
                if not repaired:
                    stats["duplicate_skips"] += 1
            if key not in seen:
                seen.add(key)
                chosen_ticket = fallback
                chosen_desc = "base"
                chosen_flips = {}
                stats["generated"] += 1
                if weak_idx is not None:
                    w = f"ticket={t_idx+1:02d} M{plans[weak_idx].match_no:02d}"
                    stats["weak_draw_selected_for_exp"] = (
                        f"{stats['weak_draw_selected_for_exp']}; {w}".strip("; ").strip()
                    )

        if chosen_ticket is None:
            chosen_ticket = [p.base_pick for p in plans]
            chosen_desc = "base"
            chosen_flips = {}

        tickets.append(chosen_ticket)
        descs.append(chosen_desc)
        record_mode_stats(mode, chosen_ticket, chosen_flips)

    # Experimental tickets: cover over-fixed branches observed in Lock+Prob
    if exp_ticket_indices:
        coverage_ticket_indices = [
            i for i, m in enumerate(mode_by_ticket[: len(tickets)]) if m in {"lock_strict", "prob_faithful"}
        ]
        coverage_total = max(1, len(coverage_ticket_indices))
        dominant_threshold = max(1, int(math.ceil(REQUIRED_TICKET_COUNT * 0.8)))
        symbol_counts_by_match: List[Dict[str, int]] = []
        for mi in range(len(plans)):
            cnt = {"1": 0, "0": 0, "2": 0}
            for ti in coverage_ticket_indices:
                sym = tickets[ti][mi]
                cnt[sym] = cnt.get(sym, 0) + 1
            symbol_counts_by_match.append(cnt)

        exp_candidates: List[Tuple[float, int, str, str, int, float, str]] = []
        for mi, p in enumerate(plans):
            if p.status != "OK":
                continue
            cnt = symbol_counts_by_match[mi]
            dominant_symbol, dominant_count = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
            prob_by_symbol = {"1": float(p.p_home), "0": float(p.p_draw), "2": float(p.p_away)}
            max_p = max(prob_by_symbol.values())
            is_over_fixed = int(dominant_count) >= dominant_threshold
            is_high_conf = max_p >= 0.62
            if (not is_over_fixed) and (not is_high_conf):
                continue
            for branch_symbol in ["1", "0", "2"]:
                if branch_symbol == dominant_symbol:
                    continue
                branch_prob = float(prob_by_symbol.get(branch_symbol, 0.0))
                if branch_prob < 0.18:
                    continue
                branch_count = int(cnt.get(branch_symbol, 0))
                dominant_rate = float(dominant_count) / float(coverage_total)
                branch_rate = float(branch_count) / float(coverage_total)
                relative_score = _relative_branch_score(p, branch_symbol)
                score = (dominant_rate - branch_rate) * relative_score
                reason = "dominant_coverage"
                if branch_symbol == "0":
                    score += 0.12 * float(getattr(p, "d_score_total", 0.0))
                    score += 0.05 * float(getattr(p, "d_score_stall", 0.0))
                if is_high_conf:
                    score += 0.05
                    reason = "high_maxp_priority"
                print(
                    f"[BUYPLAN_EXP_CANDIDATE] match_no={p.match_no:02d} dominant_symbol={dominant_symbol} "
                    f"dominant_count={dominant_count} branch_symbol={branch_symbol} "
                    f"branch_prob={branch_prob:.4f} score={score:.6f}"
                )
                exp_candidates.append(
                    (float(score), mi, dominant_symbol, branch_symbol, dominant_count, branch_prob, reason)
                )

        exp_candidates = sorted(exp_candidates, key=lambda x: (-x[0], -x[4], -x[5], x[1], x[3]))
        used_matches = set()
        for ti in exp_ticket_indices:
            mode = mode_by_ticket[ti]
            ticket = [p.base_pick for p in plans]
            pick = None
            for cand in exp_candidates:
                _, mi, dominant_symbol, branch_symbol, _, _, reason = cand
                if mi in used_matches:
                    continue
                if ticket[mi] == branch_symbol:
                    continue
                pick = cand
                used_matches.add(mi)
                break
            if pick is not None:
                _, mi, _, branch_symbol, _, _, reason = pick
                from_symbol = ticket[mi]
                ticket[mi] = branch_symbol
                descs[ti] = f"exp_bias_cover_M{plans[mi].match_no:02d}"
                print(
                    f"[BUYPLAN_EXP_PICK] ticket={ti+1:02d} match_no={plans[mi].match_no:02d} "
                    f"from_symbol={from_symbol} to_symbol={branch_symbol} reason={reason}"
                )
                record_mode_stats(mode, ticket, {mi: 1})
            else:
                descs[ti] = "exp_bias_cover_none"
                record_mode_stats(mode, ticket, {})
            tickets[ti] = ticket

    # small-gap rule: avoid all-10 same symbol on near-tied matches
    if ENABLE_SMALL_GAP_RULE:
        for mi, p in enumerate(plans):
            if p.status != "OK":
                continue
            trig = bool(
                float(p.strength_score) <= float(SMALL_GAP_STRENGTH_MAX)
                or float(p.ratio31) >= float(SMALL_GAP_RATIO31_MIN)
                or float(p.entropy) >= float(SMALL_GAP_ENTROPY_MIN)
            )
            if not trig:
                continue
            counts_before = {"1": 0, "0": 0, "2": 0}
            for t in tickets:
                counts_before[t[mi]] = counts_before.get(t[mi], 0) + 1
            dominant_before, dom_n_before = max(counts_before.items(), key=lambda kv: (kv[1], kv[0]))
            changed = 0
            second_sym = str(p.prob_second_pick)
            third_sym = str(p.third)
            mutable = [ti for ti in range(len(tickets)) if _mode_for_ticket_index(ti) in {"prob_faithful", "experimental"}]
            if dom_n_before >= len(tickets):
                for ti in mutable:
                    if tickets[ti][mi] == dominant_before and second_sym != dominant_before:
                        tickets[ti][mi] = second_sym
                        changed += 1
                        break
            counts_mid = {"1": 0, "0": 0, "2": 0}
            for t in tickets:
                counts_mid[t[mi]] = counts_mid.get(t[mi], 0) + 1
            if float(p.strength_score) <= float(SMALL_GAP_STRENGTH_MAX) and counts_mid.get(third_sym, 0) == 0 and third_sym in {"0", "1", "2"}:
                for ti in mutable:
                    if tickets[ti][mi] != third_sym:
                        tickets[ti][mi] = third_sym
                        changed += 1
                        break
            counts_after = {"1": 0, "0": 0, "2": 0}
            for t in tickets:
                counts_after[t[mi]] = counts_after.get(t[mi], 0) + 1
            dominant_after, dom_n_after = max(counts_after.items(), key=lambda kv: (kv[1], kv[0]))
            if changed > 0:
                cond = (
                    f"strength={float(p.strength_score):.4f},ratio31={float(p.ratio31):.4f},entropy={float(p.entropy):.4f}"
                )
                print(
                    f"[BUYPLAN_SMALL_GAP_RULE] match_no={p.match_no:02d} triggered=1 condition={cond} "
                    f"dominant_before={dominant_before}:{dom_n_before} dominant_after={dominant_after}:{dom_n_after}"
                )
    else:
        print("[BUYPLAN_SMALL_GAP_RULE] skipped=true")

    # weak_draw_candidate utilization at ticket_generation stage
    if ENABLE_WEAK_DRAW_APPLY:
        for mi, p in enumerate(plans):
            if p.status != "OK":
                continue
            weak_ok_generic = bool(
                p.weak_draw_candidate
                and float(p.strength_score) <= float(WEAK_DRAW_STRENGTH_MAX)
                and float(p.ratio31) >= float(WEAK_DRAW_RATIO31_MIN)
                and float(p.entropy) >= float(WEAK_DRAW_ENTROPY_MIN)
            )
            weak_ok_j2 = bool(
                str(p.league).upper() == "J2"
                and bool(p.weak_draw_candidate)
                and (not bool(p.draw_candidate))
                and float(p.top_gap) <= float(J2_WEAK_DRAW_MARGIN_MAX)
                and float(p.p_draw) >= float(J2_WEAK_DRAW_PD_MIN)
                and float(p.entropy) >= float(J2_WEAK_DRAW_ENTROPY_MIN)
                and float(p.ratio31) >= float(J2_WEAK_DRAW_RATIO31_MIN)
            )
            weak_ok = bool(weak_ok_generic or weak_ok_j2)
            if not weak_ok:
                continue
            draw_now = sum(1 for t in tickets if t[mi] == "0")
            target_draw = max(1, int(MAX_WEAK_DRAW_PER_MATCH))
            if weak_ok_j2:
                # J2の弱いD候補は、Prob/Exp にだけ 0 を1本上積みできる余地を残す。
                target_draw = max(target_draw, 3)
            if draw_now >= target_draw:
                print(
                    f"[BUYPLAN_WEAK_DRAW_APPLY] match_no={p.match_no:02d} applied=0 "
                    f"reason=already_reached_cap count={draw_now} target={target_draw}"
                )
                continue
            applied_count = 0
            for ti in [3, 4, 5, 6, 7, 8, 9]:
                if ti >= len(tickets) or ti in immutable_ticket_indices:
                    continue
                if draw_now >= target_draw:
                    break
                before = tickets[ti][mi]
                if before == "0":
                    continue
                tickets[ti][mi] = "0"
                applied_count += 1
                draw_now += 1
                print(
                    f"[BUYPLAN_WEAK_DRAW_APPLY] match_no={p.match_no:02d} applied=1 ticket={ti+1:02d} "
                    f"{before}->0 reason=weak_draw_shape count={draw_now}/{target_draw}"
                )
                w = f"ticket={ti+1:02d} M{p.match_no:02d}"
                stats["weak_draw_selected_for_distribution"] = (
                    f"{stats['weak_draw_selected_for_distribution']}; {w}".strip("; ").strip()
                )
            if applied_count == 0:
                print(
                    f"[BUYPLAN_WEAK_DRAW_APPLY] match_no={p.match_no:02d} applied=0 reason=no_mutable_ticket"
                )
    else:
        print("[BUYPLAN_WEAK_DRAW_APPLY] skipped=true")

    extreme_margin_changes: List[str] = []
    if ENABLE_EXTREME_MARGIN_RELEASE:
        target_alt = max(1, int(EXTREME_MARGIN_RELEASE_MIN_ALT_TICKETS))
        for mi, p in enumerate(plans):
            if p.status != "OK" or p.margin is None:
                continue
            if _is_strong_plan(p):
                continue
            if float(p.margin) > float(EXTREME_MARGIN_RELEASE_THRESHOLD):
                continue
            alt_now = sum(1 for ti in range(1, len(tickets)) if tickets[ti][mi] != p.base_pick)
            if alt_now >= target_alt:
                continue
            ordered_ticket_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            changed = 0
            for ti in ordered_ticket_indices:
                if ti >= len(tickets) or ti in immutable_ticket_indices:
                    continue
                current = tickets[ti][mi]
                if current != p.base_pick:
                    continue
                prefer_third = ti >= 7 and str(p.third) != str(p.base_pick) and alt_now > 0
                alt = _best_release_alt(p, current, prefer_third=prefer_third)
                if alt == current:
                    continue
                before = tickets[ti][mi]
                tickets[ti][mi] = alt
                alt_now += 1
                changed += 1
                change_msg = (
                    f"ticket={ti+1:02d} M{p.match_no:02d} {before}->{alt} "
                    f"margin={float(p.margin):.4f}"
                )
                extreme_margin_changes.append(change_msg)
                print(f"[BUYPLAN_EXTREME_MARGIN_RELEASE] {change_msg}")
                _log_repair_drop(ti + 1, p, before, alt, "extreme_margin_release")
                if alt_now >= target_alt:
                    break
            if changed == 0:
                print(
                    f"[BUYPLAN_EXTREME_MARGIN_RELEASE] match_no={p.match_no:02d} "
                    f"changed=0 margin={float(p.margin):.4f} reason=no_mutable_alt"
                )
    else:
        print("[BUYPLAN_EXTREME_MARGIN_RELEASE] skipped=true")
    stats["extreme_margin_release_count"] = int(len(extreme_margin_changes))
    stats["extreme_margin_release_matches"] = "; ".join(extreme_margin_changes[:30])

    # all_same_flag trial sway: force second branch on tickets 07-10 by configured ratio
    all_same_match_indices: List[int] = []
    dominant_symbol_by_match: Dict[int, str] = {}
    for mi, p in enumerate(plans):
        if p.status != "OK":
            continue
        col = [t[mi] for t in tickets if mi < len(t)]
        if not col:
            continue
        counts = {"1": 0, "0": 0, "2": 0}
        for sym in col:
            counts[sym] = counts.get(sym, 0) + 1
        dominant_symbol, dominant_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        if dominant_count >= len(tickets):
            all_same_match_indices.append(mi)
            dominant_symbol_by_match[mi] = dominant_symbol

    if all_same_match_indices:
        ordered = sorted(
            all_same_match_indices,
            key=lambda i: (
                float(plans[i].strength_score),
                float(plans[i].margin if plans[i].margin is not None else 999.0),
                i,
            ),
        )
        for ticket_no in [8, 9, 10]:
            ti = ticket_no - 1
            if ti >= len(tickets):
                continue
            ratio = float(ALL_SAME_SECOND_RATIO_TABLE.get(ticket_no, 0.0))
            target = int(math.floor(len(ordered) * ratio + 0.5))
            target = max(0, min(len(ordered), target))
            flips = 0
            touched: List[str] = []
            if target > 0:
                for mi in ordered:
                    if flips >= target:
                        break
                    p = plans[mi]
                    before = tickets[ti][mi]
                    if before != dominant_symbol_by_match.get(mi):
                        continue
                    after = str(p.second)
                    if after == before:
                        continue
                    tickets[ti][mi] = after
                    flips += 1
                    touched.append(f"M{p.match_no:02d}:{before}->{after}")
            print(
                f"[BUYPLAN_ALL_SAME_SWAY] ticket={ticket_no:02d} ratio={ratio:.2f} "
                f"all_same_matches={len(ordered)} target={target} flips={flips} "
                f"matches={'; '.join(touched)}"
            )
    else:
        print("[BUYPLAN_ALL_SAME_SWAY] skipped=true reason=no_all_same_match")

    _log_stage_symbol_totals("after_ticket_generation")
    _log_stage_match_counts("after_ticket_generation")
    _log_stage_branch_audit("after_ticket_generation")
    _log_ticket_branch_summary("after_ticket_generation")

    # 1) draw_distribution は無効化（明示的にスキップ）
    draw_distribution_changes: List[str] = []
    draw_distribution_locked_cells = set()
    stats["draw_distribution_adjust_count"] = 0
    stats["draw_distribution_adjust_matches"] = ""
    print("[BUYPLAN_DRAW_DISTRIBUTION] adjust_count=0 skipped=true")
    _log_stage_symbol_totals("after_draw_distribution")
    _log_stage_match_counts("after_draw_distribution")
    _log_stage_branch_audit("after_draw_distribution")
    _log_ticket_branch_summary("after_draw_distribution")

    # 2) same_symbol_cap（票内バランスのみ）
    same_cap = max(1, int(PER_MATCH_SAME_SYMBOL_CAP))
    cap_adjust = 0
    same_symbol_changes: List[str] = []
    if ENABLE_SAME_SYMBOL_CAP:
        for ti in non_lock_ticket_indices:
            t = tickets[ti]
            step = 0
            while step < 20:
                cnt = {"1": 0, "0": 0, "2": 0}
                for sym in t:
                    cnt[sym] = cnt.get(sym, 0) + 1
                sym_over = None
                over_count = 0
                for sym in ["1", "0", "2"]:
                    over = cnt.get(sym, 0) - same_cap
                    if over > over_count:
                        over_count = over
                        sym_over = sym
                if not sym_over or over_count <= 0:
                    break
                candidates = []
                for mi, current in enumerate(t):
                    if current != sym_over:
                        continue
                    p = plans[mi]
                    if p.status != "OK":
                        continue
                    if _is_close_branch(p, current):
                        _log_repair_keep(ti + 1, p, current)
                        continue
                    if p.margin is not None and float(p.margin) > float(LOCK_BY_MARGIN_MIN):
                        continue
                    candidates.append((float(p.margin if p.margin is not None else 999.0), -float(p.draw_score), mi))
                if not candidates:
                    break
                candidates = sorted(candidates, key=lambda x: (x[0], x[1], x[2]))
                changed = False
                for _, __, mi in candidates:
                    p = plans[mi]
                    alts = sorted(
                        [p.second, p.third],
                        key=lambda s: (-_relative_branch_score(p, s), s),
                    )
                    for alt in alts:
                        if alt == t[mi]:
                            continue
                        if alt == "0" and (not _allow_draw_for_plan(p)):
                            continue
                        before = t[mi]
                        t[mi] = alt
                        cap_adjust += 1
                        changed = True
                        change_msg = f"ticket={ti+1:02d} M{p.match_no:02d} {before}->{alt}"
                        same_symbol_changes.append(change_msg)
                        print(f"[BUYPLAN_SAME_SYMBOL_ADJUST] {change_msg}")
                        _log_repair_drop(ti + 1, p, before, alt, "same_symbol_cap")
                        break
                    if changed:
                        break
                if not changed:
                    break
                step += 1
            tickets[ti] = t
    else:
        print("[BUYPLAN_SAME_SYMBOL_ADJUST] skipped=true")
    stats["same_symbol_cap_adjust_count"] = int(cap_adjust)
    print(f"[BUYPLAN_SYMBOL_CAP] adjust_count={cap_adjust}")
    _log_stage_symbol_totals("after_same_symbol_cap")
    _log_stage_match_counts("after_same_symbol_cap")
    _log_stage_branch_audit("after_same_symbol_cap")
    _log_ticket_branch_summary("after_same_symbol_cap")

    # 3) 最終ユニーク保証（最後に1回）
    repair_count = 0
    unique_repair_changes: List[str] = []
    unique_repair_fallback_used = False
    protected_cells_skipped = 0
    before_dup = len(tickets) - len({_ticket_key(t) for t in tickets})
    stats["unique_before_duplicate_count"] = int(max(0, before_dup))
    seen_final = {}
    for ti, t in enumerate(tickets):
        key = _ticket_key(t)
        if key not in seen_final:
            seen_final[key] = ti
            continue
        if ti in immutable_ticket_indices:
            continue
        repaired = False
        # pass1: draw_distributionで触った保護セルを避ける
        for mi in _match_priority_indices():
            if (ti, mi) in draw_distribution_locked_cells:
                protected_cells_skipped += 1
                continue
            p = plans[mi]
            if _is_close_branch(p, t[mi]):
                _log_repair_keep(ti + 1, p, t[mi])
                continue
            for alt in sorted([p.second, p.third], key=lambda s: (-_relative_branch_score(p, s), s)):
                if alt == t[mi]:
                    continue
                if alt == "0" and (not _allow_draw_for_plan(p)):
                    continue
                cand = list(t)
                cand[mi] = alt
                k2 = _ticket_key(cand)
                if k2 not in seen_final:
                    before_sym = t[mi]
                    tickets[ti] = cand
                    seen_final[k2] = ti
                    repaired = True
                    repair_count += 1
                    unique_repair_changes.append(f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before_sym}->{alt}")
                    print(
                        f"[BUYPLAN_UNIQUE_REPAIR] ticket={ti+1:02d} match=M{plans[mi].match_no:02d} "
                        f"{before_sym}->{alt} reason=duplicate_avoid"
                    )
                    _log_repair_drop(ti + 1, p, before_sym, alt, "unique_repair")
                    break
            if repaired:
                break
        # pass2 fallback: 保護セルも含めて最終解決
        if not repaired:
            for mi in _match_priority_indices():
                p = plans[mi]
                if _is_close_branch(p, t[mi]):
                    _log_repair_keep(ti + 1, p, t[mi], "close_branch_protected_fallback")
                    continue
                for alt in sorted([p.second, p.third], key=lambda s: (-_relative_branch_score(p, s), s)):
                    if alt == t[mi]:
                        continue
                    if alt == "0" and (not _allow_draw_for_plan(p)):
                        continue
                    cand = list(t)
                    cand[mi] = alt
                    k2 = _ticket_key(cand)
                    if k2 not in seen_final:
                        before_sym = t[mi]
                        tickets[ti] = cand
                        seen_final[k2] = ti
                        repaired = True
                        repair_count += 1
                        unique_repair_fallback_used = True
                        unique_repair_changes.append(f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before_sym}->{alt}")
                        print(
                            f"[BUYPLAN_UNIQUE_REPAIR] ticket={ti+1:02d} match=M{plans[mi].match_no:02d} "
                            f"{before_sym}->{alt} reason=duplicate_avoid_fallback"
                        )
                        _log_repair_drop(ti + 1, p, before_sym, alt, "unique_repair_fallback")
                        break
                if repaired:
                    break
        if not repaired:
            _warn(warnings, f"ticket{ti+1:02d} の重複を解消できませんでした（探索不足）")

    _log_stage_symbol_totals("after_unique_repair")
    _log_stage_match_counts("after_unique_repair")
    _log_stage_branch_audit("after_unique_repair")
    _log_ticket_branch_summary("after_unique_repair")

    # 4) 10口不足時は探索補充
    while len(tickets) < REQUIRED_TICKET_COUNT:
        base = [p.base_pick for p in plans]
        for mi in _match_priority_indices():
            for alt in _alt_symbols_by_preference(plans[mi]):
                cand = list(base)
                cand[mi] = alt
                k = _ticket_key(cand)
                if k not in {_ticket_key(x) for x in tickets}:
                    tickets.append(cand)
                    descs.append("auto_explore_fill")
                    print(f"[BUYPLAN_FILL] ticket={len(tickets):02d} match=M{plans[mi].match_no:02d} sym={alt}")
                    break
            if len(tickets) >= REQUIRED_TICKET_COUNT:
                break
        if len(tickets) < REQUIRED_TICKET_COUNT:
            break

    if len(tickets) < REQUIRED_TICKET_COUNT:
        _warn(warnings, f"重複なしで {REQUIRED_TICKET_COUNT} 口を作れず {len(tickets)} 口になりました。")

    if len(descs) < len(tickets):
        descs.extend(["base"] * (len(tickets) - len(descs)))
    descs = descs[:len(tickets)]

    # final range enforcement after unique repair
    final_range_changes: List[str] = []
    if ENABLE_FINAL_RANGE:
        for ti in range(len(tickets)):
            if ti in immutable_ticket_indices:
                continue
            c, cc = _enforce_ticket_draw_range(
                ti=ti,
                phase="final_range",
                protected_cells=draw_distribution_locked_cells,
                allow_protected=True,
            )
            if c > 0:
                final_range_changes.extend(cc)
        if final_range_changes:
            for msg in final_range_changes:
                print(f"[BUYPLAN_FINAL_RANGE_ADJUST] {msg}")
    else:
        print("[BUYPLAN_FINAL_RANGE_ADJUST] skipped=true")
    _log_stage_symbol_totals("after_final_range")
    _log_stage_match_counts("after_final_range")
    _log_stage_branch_audit("after_final_range")
    _log_ticket_branch_summary("after_final_range")

    # post-range dedupe: keep draw range constraints
    if ENABLE_FINAL_RANGE:
        seen_post = {}
        for ti, t in enumerate(tickets):
            key = _ticket_key(t)
            if key not in seen_post:
                seen_post[key] = ti
                continue
            if ti in immutable_ticket_indices:
                continue
            repaired_post = False
            mode = mode_by_ticket[ti]
            min_d, max_d = _target_draw_range_for_mode(mode)
            for mi in _match_priority_indices():
                p = plans[mi]
                if _is_close_branch(p, t[mi]):
                    _log_repair_keep(ti + 1, p, t[mi], "close_branch_protected_post_range")
                    continue
                for alt in sorted([p.second, p.third], key=lambda s: (-_relative_branch_score(p, s), s)):
                    if alt == t[mi]:
                        continue
                    if alt == "0" and (not _allow_draw_for_plan(p)):
                        continue
                    cand = list(t)
                    cand[mi] = alt
                    draw_n = _symbol_count(cand, "0")
                    if draw_n < min_d or draw_n > max_d:
                        continue
                    k2 = _ticket_key(cand)
                    if k2 not in seen_post:
                        before = t[mi]
                        tickets[ti] = cand
                        seen_post[k2] = ti
                        repaired_post = True
                        repair_count += 1
                        unique_repair_changes.append(f"ticket={ti+1:02d} M{p.match_no:02d} {before}->{alt}")
                        print(
                            f"[BUYPLAN_UNIQUE_REPAIR] ticket={ti+1:02d} match=M{p.match_no:02d} "
                            f"{before}->{alt} reason=post_range_duplicate_avoid"
                        )
                        _log_repair_drop(ti + 1, p, before, alt, "post_range_unique_repair")
                        break
                if repaired_post:
                    break

    stats["unique_repair_count"] = int(repair_count)
    stats["unique_ticket_count"] = len({_ticket_key(t) for t in tickets})
    stats["duplicate_count"] = max(0, len(tickets) - stats["unique_ticket_count"])
    stats["unique_after_duplicate_count"] = int(stats["duplicate_count"])
    stats["unique_repair_fallback_used"] = bool(unique_repair_fallback_used)
    stats["unique_repair_protected_cells_skipped"] = int(protected_cells_skipped)
    stats["zero_cap_adjust_count"] = int(len(draw_distribution_changes))
    stats["zero_cap_adjust_matches"] = "; ".join(draw_distribution_changes[:30])
    stats["draw_distribution_adjust_matches"] = "; ".join(draw_distribution_changes[:30])
    stats["same_symbol_cap_adjust_matches"] = "; ".join(same_symbol_changes[:30])
    stats["unique_repair_matches"] = "; ".join(unique_repair_changes[:30])
    print(
        f"[BUYPLAN_UNIQUE_REPAIR] before_duplicate_count={stats['unique_before_duplicate_count']} "
        f"after_duplicate_count={stats['unique_after_duplicate_count']} "
        f"repair_matches={stats['unique_repair_matches']} "
        f"fallback_used={str(stats['unique_repair_fallback_used']).lower()} "
        f"protected_cells_skipped={stats['unique_repair_protected_cells_skipped']}"
    )

    final_mode_stats: Dict[str, Dict[str, float]] = {
        "lock_strict": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
        "prob_faithful": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
        "experimental": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
    }
    for ti, ticket in enumerate(tickets):
        mode = mode_by_ticket[ti]
        ms = final_mode_stats[mode]
        ms["tickets"] += 1
        ms["zero_count"] += _symbol_count(ticket, "0")
        for mi, sym in enumerate(ticket):
            base = str(plans[mi].base_pick)
            if sym == base:
                continue
            ms["flip_count"] += 1
            pm = plans[mi].prob_margin
            if pm is not None:
                ms["margin_sum"] += float(pm)
                ms["margin_n"] += 1

    for mode in ["lock_strict", "prob_faithful", "experimental"]:
        ms = final_mode_stats[mode]
        avg_margin = (ms["margin_sum"] / ms["margin_n"]) if ms["margin_n"] > 0 else 0.0
        stats[f"{mode}_zero_count"] = int(ms["zero_count"])
        stats[f"{mode}_flip_count"] = int(ms["flip_count"])
        stats[f"{mode}_avg_margin"] = float(avg_margin)
        stats[f"{mode}_tickets"] = int(ms["tickets"])

    total_cells = len(tickets) * len(plans) if tickets and plans else 0
    one_count = sum(_symbol_count(t, "1") for t in tickets)
    zero_count = sum(_symbol_count(t, "0") for t in tickets)
    two_count = sum(_symbol_count(t, "2") for t in tickets)
    stats["total_one_count"] = one_count
    stats["total_zero_count"] = zero_count
    stats["total_two_count"] = two_count
    stats["total_ratio_1"] = _ratio_str(one_count, total_cells)
    stats["total_ratio_0"] = _ratio_str(zero_count, total_cells)
    stats["total_ratio_2"] = _ratio_str(two_count, total_cells)
    stats["draw_bias_fired_pd_avg"] = 0.0
    return tickets[:REQUIRED_TICKET_COUNT], descs[:REQUIRED_TICKET_COUNT], stats


def _scenario_desc(base_ticket: List[str], ticket: List[str]) -> str:
    diffs = []
    for idx, (b, t) in enumerate(zip(base_ticket, ticket), start=1):
        if b != t:
            diffs.append(f"M{idx:02d}:{b}->{t}")
    if not diffs:
        return "変更なし（基準）"
    return ", ".join(diffs)


def _generate_tickets_by_scenario(
    df: pd.DataFrame,
    plans: List[MatchPlan],
    warnings: List[str],
) -> Tuple[List[List[str]], List[str], List[ScenarioDef], Dict[str, int], List[List[str]]]:
    tickets, flip_descs, stats = _generate_tickets(plans, warnings)

    def _refresh_final_ticket_stats() -> None:
        final_mode_stats: Dict[str, Dict[str, float]] = {
            "lock_strict": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
            "prob_faithful": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
            "experimental": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
        }
        for ti, ticket in enumerate(tickets):
            mode = _mode_for_ticket_index(ti)
            ms = final_mode_stats[mode]
            ms["tickets"] += 1
            ms["zero_count"] += _symbol_count(ticket, "0")
            for mi, sym in enumerate(ticket):
                base = str(plans[mi].base_pick)
                if sym == base:
                    continue
                ms["flip_count"] += 1
                pm = plans[mi].prob_margin
                if pm is not None:
                    ms["margin_sum"] += float(pm)
                    ms["margin_n"] += 1

        for mode in ["lock_strict", "prob_faithful", "experimental"]:
            ms = final_mode_stats[mode]
            avg_margin = (ms["margin_sum"] / ms["margin_n"]) if ms["margin_n"] > 0 else 0.0
            stats[f"{mode}_zero_count"] = int(ms["zero_count"])
            stats[f"{mode}_flip_count"] = int(ms["flip_count"])
            stats[f"{mode}_avg_margin"] = float(avg_margin)
            stats[f"{mode}_tickets"] = int(ms["tickets"])

        total_cells = len(tickets) * len(plans) if tickets and plans else 0
        one_count = sum(_symbol_count(t, "1") for t in tickets)
        zero_count = sum(_symbol_count(t, "0") for t in tickets)
        two_count = sum(_symbol_count(t, "2") for t in tickets)
        stats["total_one_count"] = one_count
        stats["total_zero_count"] = zero_count
        stats["total_two_count"] = two_count
        stats["total_ratio_1"] = _ratio_str(one_count, total_cells)
        stats["total_ratio_0"] = _ratio_str(zero_count, total_cells)
        stats["total_ratio_2"] = _ratio_str(two_count, total_cells)

    def _apply_post_system_weak_draw() -> None:
        applied_logs: List[str] = []
        for mi, p in enumerate(plans):
            weak_ok_j2 = bool(
                str(p.league).upper() == "J2"
                and bool(p.weak_draw_candidate)
                and (not bool(p.draw_candidate))
                and float(p.top_gap) <= float(J2_WEAK_DRAW_MARGIN_MAX)
                and float(p.p_draw) >= float(J2_WEAK_DRAW_PD_MIN)
                and float(p.entropy) >= float(J2_WEAK_DRAW_ENTROPY_MIN)
                and float(p.ratio31) >= float(J2_WEAK_DRAW_RATIO31_MIN)
            )
            if not weak_ok_j2:
                continue
            draw_now = sum(1 for t in tickets if t[mi] == "0")
            target_draw = 3
            if draw_now >= target_draw:
                continue
            for ti in [3, 4, 5, 6, 7, 8, 9]:
                if draw_now >= target_draw:
                    break
                before = tickets[ti][mi]
                if before == "0":
                    continue
                cand = list(tickets[ti])
                cand[mi] = "0"
                cand_key = _ticket_key(cand)
                if any(j != ti and _ticket_key(tickets[j]) == cand_key for j in range(len(tickets))):
                    continue
                tickets[ti] = cand
                draw_now += 1
                log = f"ticket={ti+1:02d} M{p.match_no:02d}"
                applied_logs.append(log)
                print(
                    f"[BUYPLAN_POST_SYSTEM_WEAK_DRAW] match_no={p.match_no:02d} ticket={ti+1:02d} "
                    f"{before}->0 count={draw_now}/{target_draw}"
                )
        if applied_logs:
            merged = "; ".join(applied_logs)
            existing = str(stats.get("weak_draw_selected_for_distribution", "")).strip()
            stats["weak_draw_selected_for_distribution"] = "; ".join(
                [s for s in [existing, merged] if s]
            )

    def _apply_candidate_ticket_repairs() -> None:
        # 05/07/10 の後段補修を context ベースへ寄せる。
        # - draw_cover / avoid_main / draw_watch: 0 復元を許す
        # - main_only / fixed: 不要な flip を戻す
        # - lab_cover: 10 の反転を残す
        repair_logs: List[str] = []
        def _best_non_draw_prob_alt_local(plan: MatchPlan, current: str) -> str:
            preferred = "1" if float(plan.p_home) >= float(plan.p_away) else "2"
            if preferred != current:
                return preferred
            return "2" if preferred == "1" else "1"
        def _context_guidance_local(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()
        def _context_risk_local(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_risk_level", "") or "").strip().lower()
        def _ticket_draw_ok(plan: MatchPlan) -> bool:
            guidance = _context_guidance_local(plan)
            risk = _context_risk_local(plan)
            return bool(
                plan.draw_candidate
                or plan.weak_draw_candidate
                or guidance in {"draw_cover", "avoid_main"}
                or risk == "draw_watch"
            )
        def _must_keep_main(plan: MatchPlan) -> bool:
            guidance = _context_guidance_local(plan)
            risk = _context_risk_local(plan)
            return bool(guidance == "main_only" or risk == "fixed")
        def _prefer_lab_cover(plan: MatchPlan) -> bool:
            return _context_guidance_local(plan) == "lab_cover"
        def _draw_restore_ok(plan: MatchPlan) -> bool:
            guidance = _context_guidance_local(plan)
            risk = _context_risk_local(plan)
            return bool(
                _ticket_draw_ok(plan)
                and (
                    guidance in {"draw_cover", "avoid_main", "lab_cover"}
                    or risk == "draw_watch"
                    or str(plan.prob_best_pick) == "0"
                )
            )
        def _is_lab_reverse_target(plan: MatchPlan) -> bool:
            flags = str(getattr(plan, "match_type_flags", ""))
            has_lab_matchup = ("lab_away_matchup" in flags) or ("lab_home_matchup" in flags)
            if not has_lab_matchup:
                return False
            if bool(getattr(plan, "lab_style_conflict", False) or getattr(plan, "lab_low_event", False)):
                return False
            return abs(float(getattr(plan, "lab_matchup_edge", 0.0))) >= 6.0
        for ti in [4, 6, 9]:
            if ti >= len(tickets):
                continue
            ticket = tickets[ti]
            for mi, p in enumerate(plans):
                if p.status != "OK":
                    continue
                current = str(ticket[mi])
                if (
                    ti == 6
                    and current == "2"
                    and _draw_restore_ok(p)
                    and not bool(p.away_value_candidate)
                ):
                    ticket[mi] = "0"
                    repair_logs.append(
                        f"ticket={ti+1:02d} M{p.match_no:02d} 2->0 reason=context_draw_restore"
                    )
                    current = str(ticket[mi])
                if ti == 9 and current != str(p.buyplan_choice):
                    if _must_keep_main(p):
                        ticket[mi] = str(p.buyplan_choice)
                        repair_logs.append(
                            f"ticket={ti+1:02d} M{p.match_no:02d} {current}->{p.buyplan_choice} reason=context_main_only_reset"
                        )
                        current = str(ticket[mi])
                    elif not (_prefer_lab_cover(p) and _is_lab_reverse_target(p)):
                        ticket[mi] = str(p.buyplan_choice)
                        repair_logs.append(
                            f"ticket={ti+1:02d} M{p.match_no:02d} {current}->{p.buyplan_choice} reason=context_non_lab_reset"
                        )
                        current = str(ticket[mi])
                if current == "0" and not _ticket_draw_ok(p):
                    alt = _best_non_draw_prob_alt_local(p, current)
                    if alt != current:
                        ticket[mi] = alt
                        repair_logs.append(f"ticket={ti+1:02d} M{p.match_no:02d} 0->{alt} reason=context_non_draw_prune")
                if (
                    ti in {4, 9}
                    and current == "2"
                    and _draw_restore_ok(p)
                    and not bool(p.away_value_candidate)
                ):
                    ticket[mi] = "0"
                    repair_logs.append(f"ticket={ti+1:02d} M{p.match_no:02d} 2->0 reason=context_draw_top_restore")
            tickets[ti] = ticket
        if repair_logs:
            merged = "; ".join(repair_logs)
            stats["candidate_ticket_repairs"] = merged
            for log in repair_logs:
                print(f"[BUYPLAN_CANDIDATE_REPAIR] {log}")

    def _normalize_ticket05_draw_count() -> None:
        if len(tickets) <= 4:
            return
        ti = 4  # ticket05
        ticket = list(tickets[ti])
        min_d, max_d = _target_draw_range_for_mode(_mode_for_ticket_index(ti))

        def _guidance(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()

        if _symbol_count(ticket, "0") < min_d:
            addable: List[Tuple[int, int, float, float, float, int]] = []
            for mi, current in enumerate(ticket):
                if current == "0":
                    continue
                p = plans[mi]
                if p.status != "OK":
                    continue
                if not bool(p.draw_candidate or p.weak_draw_candidate):
                    continue
                addable.append(
                    (
                        0 if _guidance(p) in {"draw_cover", "lab_cover", "avoid_main"} else 1,
                        0 if str(p.second) == "0" else 1,
                        -float(getattr(p, "d_score_total", 0.0)),
                        -float(getattr(p, "d_score_stall", 0.0)),
                        float(p.margin if p.margin is not None else 999.0),
                        mi,
                    )
                )
            addable = sorted(addable)
            for _, _, _, _, _, mi in addable:
                if _symbol_count(ticket, "0") >= min_d:
                    break
                p = plans[mi]
                before = ticket[mi]
                cand = list(ticket)
                cand[mi] = "0"
                cand_key = _ticket_key(cand)
                if any(j != ti and _ticket_key(tickets[j]) == cand_key for j in range(len(tickets))):
                    continue
                ticket = cand
                print(
                    f"[BUYPLAN_TICKET05_DRAW_RAISE] match_no={p.match_no:02d} "
                    f"{before}->0 draw_count={_symbol_count(ticket, '0')}"
                )

        if _symbol_count(ticket, "0") <= max_d:
            tickets[ti] = ticket
            return

        removable: List[Tuple[int, int, float, float, float, int]] = []
        for mi, current in enumerate(ticket):
            if current != "0":
                continue
            p = plans[mi]
            if p.status != "OK":
                continue
            removable.append(
                (
                    0 if str(p.base_pick) != "0" else 1,
                    0 if _guidance(p) not in {"draw_cover", "lab_cover"} else 1,
                    float(getattr(p, "d_score_total", 0.0)),
                    float(getattr(p, "d_score_stall", 0.0)),
                    -float(p.margin if p.margin is not None else 999.0),
                    mi,
                )
            )
        removable = sorted(removable)
        for _, _, _, _, _, mi in removable:
            if _symbol_count(ticket, "0") <= max_d:
                break
            p = plans[mi]
            before = ticket[mi]
            alts = [str(p.buyplan_choice), str(p.base_pick), str(p.second), str(p.third)]
            chosen = None
            for alt in alts:
                if alt not in {"1", "2"}:
                    continue
                cand = list(ticket)
                cand[mi] = alt
                cand_key = _ticket_key(cand)
                if any(j != ti and _ticket_key(tickets[j]) == cand_key for j in range(len(tickets))):
                    continue
                chosen = alt
                ticket = cand
                print(
                    f"[BUYPLAN_TICKET05_DRAW_TRIM] match_no={p.match_no:02d} "
                    f"{before}->{alt} draw_count={_symbol_count(ticket, '0')}"
                )
                break
            if chosen is None:
                continue
        tickets[ti] = ticket

    def _normalize_ticket08_close_draw_count() -> None:
        if len(tickets) <= 7:
            return
        ti = 7  # ticket08
        ticket = list(tickets[ti])
        min_d, max_d = _target_draw_range_for_mode(_mode_for_ticket_index(ti))

        def _guidance(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()

        while _symbol_count(ticket, "0") < min_d:
            addable: List[Tuple[int, int, float, float, float, float, int]] = []
            for mi, current in enumerate(ticket):
                if current == "0":
                    continue
                p = plans[mi]
                if p.status != "OK":
                    continue
                if not bool(p.draw_candidate or p.weak_draw_candidate):
                    continue
                addable.append(
                    (
                        0 if _guidance(p) in {"draw_cover", "avoid_main", "lab_cover"} else 1,
                        0 if _is_close_branch(p, "0") else 1,
                        -float(getattr(p, "d_score_total", 0.0)),
                        -float(getattr(p, "d_score_close", 0.0)),
                        -float(getattr(p, "d_score_stall", 0.0)),
                        float(p.margin if p.margin is not None else 999.0),
                        mi,
                    )
                )
            if not addable:
                break
            applied = False
            for _, _, _, _, _, _, mi in sorted(addable):
                p = plans[mi]
                before = ticket[mi]
                cand = list(ticket)
                cand[mi] = "0"
                if _symbol_count(cand, "0") > max_d:
                    continue
                cand_key = _ticket_key(cand)
                if any(j != ti and _ticket_key(tickets[j]) == cand_key for j in range(len(tickets))):
                    continue
                ticket = cand
                applied = True
                print(
                    f"[BUYPLAN_TICKET08_CLOSE_DRAW_RAISE] match_no={p.match_no:02d} "
                    f"{before}->0 draw_count={_symbol_count(ticket, '0')}"
                )
                break
            if not applied:
                break
        tickets[ti] = ticket

    def _raise_prob_ticket_draw_floor() -> None:
        if not ENABLE_PROB_DRAW_FLOOR:
            return
        target_indices = [5]  # ticket06

        def _guidance(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()

        def _risk(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_risk_level", "") or "").strip().lower()

        def _draw_ok(plan: MatchPlan) -> bool:
            return bool(getattr(plan, "draw_candidate", False) or getattr(plan, "weak_draw_candidate", False))

        for ti in target_indices:
            if ti >= len(tickets):
                continue
            ticket = list(tickets[ti])
            min_d, max_d = _target_draw_range_for_mode(_mode_for_ticket_index(ti))
            while _symbol_count(ticket, "0") < min_d:
                candidates: List[Tuple[int, int, float, float, float, int]] = []
                for mi, current in enumerate(ticket):
                    if current == "0":
                        continue
                    p = plans[mi]
                    if p.status != "OK":
                        continue
                    if not _draw_ok(p):
                        continue
                    candidates.append(
                        (
                            0 if _guidance(p) in {"draw_cover", "avoid_main", "lab_cover"} else 1,
                            0 if _risk(p) == "draw_watch" else 1,
                            -float(getattr(p, "d_score_total", 0.0)),
                            -float(getattr(p, "d_score_stall", 0.0)),
                            float(p.margin if p.margin is not None else 999.0),
                            mi,
                        )
                    )
                if not candidates:
                    break
                candidates = sorted(candidates)
                applied = False
                for _, _, _, _, _, mi in candidates:
                    p = plans[mi]
                    before = ticket[mi]
                    cand = list(ticket)
                    cand[mi] = "0"
                    draw_n = _symbol_count(cand, "0")
                    if draw_n > max_d:
                        continue
                    cand_key = _ticket_key(cand)
                    if any(j != ti and _ticket_key(tickets[j]) == cand_key for j in range(len(tickets))):
                        continue
                    ticket = cand
                    applied = True
                    print(
                        f"[BUYPLAN_PROB_DRAW_ADD] ticket={ti+1:02d} match_no={p.match_no:02d} "
                        f"{before}->0 draw_count={draw_n}"
                    )
                    break
                if not applied:
                    break
            tickets[ti] = ticket

    def _resolve_post_context_duplicates() -> None:
        def _dup_guidance(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()

        def _dup_risk(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_risk_level", "") or "").strip().lower()

        def _dup_draw_prefer(plan: MatchPlan) -> bool:
            return bool(_dup_guidance(plan) in {"draw_cover", "avoid_main"} or _dup_risk(plan) == "draw_watch")

        def _dup_keep_main(plan: MatchPlan) -> bool:
            return bool(_dup_guidance(plan) == "main_only" or _dup_risk(plan) == "fixed")

        def _dup_flip_ready(plan: MatchPlan) -> bool:
            return bool(
                _dup_risk(plan) in {"volatile", "caution", "draw_watch"}
                or _dup_draw_prefer(plan)
                or _dup_guidance(plan) == "lab_cover"
            )

        def _dup_allow_draw(plan: MatchPlan) -> bool:
            return bool(getattr(plan, "draw_candidate", False) or getattr(plan, "weak_draw_candidate", False))

        def _dup_priority_indices() -> List[int]:
            ok_local = [i for i, p in enumerate(plans) if p.status == "OK"]
            return sorted(
                ok_local,
                key=lambda i: (
                    0 if _dup_draw_prefer(plans[i]) else 1,
                    0 if _dup_allow_draw(plans[i]) else 1,
                    0 if _dup_flip_ready(plans[i]) else 1,
                    1 if _dup_keep_main(plans[i]) else 0,
                    -float(getattr(plans[i], "d_score_total", 0.0)),
                    -float(getattr(plans[i], "d_score_stall", 0.0)),
                    -float(getattr(plans[i], "d_score_close", 0.0)),
                    0 if plans[i].second == "0" else 1,
                    0 if plans[i].draw_candidate else 1,
                    0 if plans[i].weak_draw_candidate else 1,
                    -float(plans[i].draw_score),
                    -float(plans[i].p_draw),
                    float(plans[i].strength_score),
                    float(plans[i].spread),
                    -float(plans[i].entropy),
                    i,
                ),
            )

        seen: Dict[str, int] = {}
        for ti, ticket in enumerate(tickets):
            key = _ticket_key(ticket)
            if key not in seen:
                seen[key] = ti
                continue
            if ti == 0:
                continue
            mode = _mode_for_ticket_index(ti)
            min_d, max_d = _target_draw_range_for_mode(mode)
            repaired = False
            for mi in _dup_priority_indices():
                p = plans[mi]
                if p.status != "OK":
                    continue
                current = ticket[mi]
                alt_order = [str(p.second), str(p.third), str(p.buyplan_choice), str(p.base_pick)]
                for alt in alt_order:
                    if alt == current or alt not in {"0", "1", "2"}:
                        continue
                    if alt == "0" and not _dup_allow_draw(p):
                        continue
                    cand = list(ticket)
                    cand[mi] = alt
                    draw_n = _symbol_count(cand, "0")
                    if draw_n < min_d or draw_n > max_d:
                        continue
                    cand_key = _ticket_key(cand)
                    if any(j != ti and _ticket_key(tickets[j]) == cand_key for j in range(len(tickets))):
                        continue
                    tickets[ti] = cand
                    key = cand_key
                    seen[key] = ti
                    repaired = True
                    print(
                        f"[BUYPLAN_POST_DUPLICATE_REPAIR] ticket={ti+1:02d} match_no={p.match_no:02d} "
                        f"{current}->{alt}"
                    )
                    break
                if repaired:
                    break

    def _separate_system_variant_duplicates() -> None:
        if len(tickets) < 3:
            return
        if not any(_ticket_key(tickets[j]) == _ticket_key(tickets[2]) for j in range(2)):
            return
        def _guidance(plan: MatchPlan) -> str:
            return str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()
        ordered_match_indices = sorted(
            range(len(plans)),
            key=lambda mi: (
                0 if _guidance(plans[mi]) == "lab_cover" else 1,
                0 if _guidance(plans[mi]) in {"draw_cover", "avoid_main"} else 1,
                float(plans[mi].margin if plans[mi].margin is not None else 999.0),
                -float(plans[mi].draw_score),
                int(plans[mi].match_no),
            ),
        )
        for mi in ordered_match_indices:
            p = plans[mi]
            current = tickets[2][mi]
            alts = sorted(
                [sym for sym in [p.third, p.second] if sym and sym != current],
                key=lambda s: (-_relative_branch_score(p, s), s),
            )
            for alt in alts:
                cand = list(tickets[2])
                cand[mi] = alt
                cand_key = _ticket_key(cand)
                if any(j != 2 and _ticket_key(tickets[j]) == cand_key for j in range(len(tickets))):
                    continue
                tickets[2] = cand
                change_note = f"system_variant_separate:M{p.match_no:02d}:{current}->{alt}"
                base_desc = flip_descs[2] if 2 < len(flip_descs) else ""
                if 2 < len(flip_descs):
                    flip_descs[2] = f"{base_desc}; {change_note}" if base_desc else change_note
                print(
                    f"[BUYPLAN_SYSTEM_VARIANT_SEPARATE] ticket=03 match_no={p.match_no:02d} "
                    f"{current}->{alt}"
                )
                return

    def _apply_unseen_symbol_fill_for_exp_tickets() -> None:
        if len(tickets) < 10:
            return
        source_tickets = tickets[:8]
        ticket09 = list(tickets[8])
        ticket10 = list(tickets[9])
        changed09 = 0
        changed10 = 0
        for mi, p in enumerate(plans):
            if p.status != "OK":
                continue
            seen_symbols = {str(t[mi]) for t in source_tickets if mi < len(t)}
            unseen = [sym for sym in ["1", "0", "2"] if sym not in seen_symbols]
            if not unseen:
                fill09 = str(p.buyplan_choice)
                fill10 = str(p.buyplan_choice)
            elif len(unseen) == 1:
                fill09 = unseen[0]
                fill10 = unseen[0]
            else:
                probs = {
                    "1": float(p.p_home),
                    "0": float(p.p_draw),
                    "2": float(p.p_away),
                }
                ranked = sorted(unseen, key=lambda sym: (-probs[sym], sym))
                fill09 = ranked[0]
                fill10 = ranked[-1]
            if ticket09[mi] != fill09:
                changed09 += 1
            if ticket10[mi] != fill10:
                changed10 += 1
            ticket09[mi] = fill09
            ticket10[mi] = fill10
        tickets[8] = ticket09
        tickets[9] = ticket10
        if len(flip_descs) > 8:
            flip_descs[8] = "unseen_fill_09"
        if len(flip_descs) > 9:
            flip_descs[9] = "unseen_fill_10"
        print(
            f"[BUYPLAN_UNSEEN_FILL] ticket09_changed={changed09} "
            f"ticket10_changed={changed10} source_tickets=01-08"
        )

    system_specs = [
        ("predicted_result_main", "predicted_result"),
        ("predicted_result_type_b", "predicted_result_main"),
        ("predicted_result_type_c", "predicted_result_type_b"),
    ]
    for idx, (col, fallback_col) in enumerate(system_specs):
        system_ticket, desc = _build_system_ticket(df, plans, col, fallback_col)
        if idx < len(tickets):
            tickets[idx] = system_ticket
            flip_descs[idx] = desc
        else:
            tickets.append(system_ticket)
            flip_descs.append(desc)

    _separate_system_variant_duplicates()
    _apply_post_system_weak_draw()
    _apply_candidate_ticket_repairs()
    _normalize_ticket05_draw_count()
    _normalize_ticket08_close_draw_count()
    _raise_prob_ticket_draw_floor()
    _apply_unseen_symbol_fill_for_exp_tickets()
    _resolve_post_context_duplicates()
    _refresh_final_ticket_stats()

    if tickets:
        unique_count = len({_ticket_key(t) for t in tickets})
        stats["unique_ticket_count"] = unique_count
        stats["duplicate_count"] = max(0, len(tickets) - unique_count)

    injected_draw_count, injected_draw_logs = _inject_minimum_draws(plans, tickets, flip_descs, warnings)
    stats["min_draw_insert_count"] = int(injected_draw_count)
    stats["min_draw_insert_logs"] = "; ".join(injected_draw_logs[:30])
    _refresh_final_ticket_stats()
    if tickets:
        unique_count = len({_ticket_key(t) for t in tickets})
        stats["unique_ticket_count"] = unique_count
        stats["duplicate_count"] = max(0, len(tickets) - unique_count)

    pred_base_used = sum(1 for p in plans if p.status == "OK" and p.base_from_predicted)
    pred_base_fallback = sum(1 for p in plans if p.status == "OK" and not p.base_from_predicted)
    _warn(
        warnings,
        f"基準票ソース: predicted_result採用={pred_base_used}, best_symbolフォールバック={pred_base_fallback}",
    )
    for mode, label in [
        ("lock_strict", "LOCK遵守"),
        ("prob_faithful", "確率忠実"),
        ("experimental", "実験"),
    ]:
        _warn(
            warnings,
            f"{label}: tickets={stats.get(f'{mode}_tickets', 0)}, "
            f"0採用件数={stats.get(f'{mode}_zero_count', 0)}, "
            f"平均margin={stats.get(f'{mode}_avg_margin', 0.0):.3f}, "
            f"総flip数={stats.get(f'{mode}_flip_count', 0)}",
        )
    _warn(
        warnings,
        "探索ログ: "
        f"second_pickが0の試合数(全OK)={stats.get('second_zero_matches_all_ok', 0)}, "
        f"second_pickが0の試合数(探索対象)={stats.get('second_zero_matches', 0)}, "
        f"探索枠でsecond=0を採用した件数={stats.get('second_zero_applied', 0)}",
    )
    _warn(
        warnings,
        "system fixed tickets: "
        "ticket01=predicted_result_main, ticket02=predicted_result_type_b, ticket03=predicted_result_type_c(fallback=type_b)",
    )
    _warn(
        warnings,
        "全体比率: "
        f"0総出現回数={stats.get('total_zero_count', 0)}, "
        f"1/0/2比率={stats.get('total_ratio_1', '0.000')}/"
        f"{stats.get('total_ratio_0', '0.000')}/"
        f"{stats.get('total_ratio_2', '0.000')}",
    )
    _warn(
        warnings,
        "draw_distribution: "
        f"adjust_count={stats.get('draw_distribution_adjust_count', 0)}",
    )
    _warn(
        warnings,
        "[BUYPLAN_EXTREME_MARGIN_RELEASE] "
        f"count={stats.get('extreme_margin_release_count', 0)} "
        f"matches={stats.get('extreme_margin_release_matches', '')}",
    )
    _warn(
        warnings,
        "draw候補ゲート: "
        f"candidate_matches={stats.get('draw_gate_candidate_matches', 0)} "
        f"rate={stats.get('draw_gate_candidate_rate', '0.000')}",
    )
    _warn(
        warnings,
        "[BUYPLAN_WEAK_DRAW_CAND] "
        f"matches={stats.get('weak_draw_matches', '')} "
        f"selected_for_exp={stats.get('weak_draw_selected_for_exp', '')} "
        f"selected_for_distribution={stats.get('weak_draw_selected_for_distribution', '')}",
    )
    _warn(
        warnings,
        "[BUYPLAN_DRAW_POLICY] "
        f"strong_draw_count={stats.get('draw_gate_candidate_matches', 0)} "
        f"weak_draw_count={stats.get('weak_draw_count', 0)} "
        f"final_zero_count={stats.get('total_zero_count', 0)}",
    )
    _warn(
        warnings,
        f"[BUYPLAN_TARGET_DRAW] scenario=Lock target_draw_count_range={TARGET_DRAW_MIN_LOCK}..{TARGET_DRAW_MAX_LOCK}",
    )
    _warn(
        warnings,
        f"[BUYPLAN_TARGET_DRAW] scenario=Prob target_draw_count_range={TARGET_DRAW_MIN_PROB}..{TARGET_DRAW_MAX_PROB}",
    )
    _warn(
        warnings,
        f"[BUYPLAN_TARGET_DRAW] scenario=Exp target_draw_count_range={TARGET_DRAW_MIN_EXP}..{TARGET_DRAW_MAX_EXP}",
    )
    _warn(
        warnings,
        "[BUYPLAN_MIN_DRAW_INSERT] "
        f"count={stats.get('min_draw_insert_count', 0)} "
        f"logs={stats.get('min_draw_insert_logs', '')}",
    )
    _warn(
        warnings,
        "重複/偏り調整: "
        f"draw_distribution_adjust_count={stats.get('draw_distribution_adjust_count', 0)} "
        f"unique_repair_count={stats.get('unique_repair_count', 0)} "
        f"same_symbol_cap_adjust_count={stats.get('same_symbol_cap_adjust_count', 0)} "
        f"unique_ticket_count={stats.get('unique_ticket_count', 0)} "
        f"duplicate_count={stats.get('duplicate_count', 0)}",
    )
    if stats.get("draw_distribution_adjust_matches", ""):
        _warn(warnings, f"draw_distribution_adjust_matches={stats.get('draw_distribution_adjust_matches')}")
    if stats.get("same_symbol_cap_adjust_matches", ""):
        _warn(warnings, f"same_symbol_cap_adjust_matches={stats.get('same_symbol_cap_adjust_matches')}")
    if stats.get("unique_repair_matches", ""):
        _warn(warnings, f"unique_repair_matches={stats.get('unique_repair_matches')}")
    if stats.get("duplicate_count", 0) != 0:
        _warn(warnings, "duplicate_count が 0 ではありません（要確認）")
    if tickets:
        _warn(warnings, "ticket01-03 are immutable system prediction tickets")
        print("[BUYPLAN_SYSTEM01_CHECK] immutable=true all_match=true")
        if len(tickets) >= 3:
            _warn(
                warnings,
                "ticket02-03 are system prediction variants "
                f"(desc={flip_descs[1] if len(flip_descs) > 1 else ''} / {flip_descs[2] if len(flip_descs) > 2 else ''})",
            )
        for ti in range(3, len(tickets)):
            mode = _mode_for_ticket_index(ti)
            dmin, dmax = _target_draw_range_for_mode(mode)
            draw_n = _symbol_count(tickets[ti], "0")
            in_range = int(dmin <= draw_n <= dmax)
            print(
                f"[BUYPLAN_RANGE_CHECK] ticket={ti+1:02d} mode={mode} "
                f"draw_count={draw_n} range={dmin}..{dmax} in_range={in_range}"
            )

    # 同一試合で10口同値を検出
    if tickets:
        for mi, p in enumerate(plans):
            vals = [t[mi] for t in tickets if mi < len(t)]
            if p.status != "OK":
                continue
            if vals and len(set(vals)) == 1:
                _warn(
                    warnings,
                    f"M{p.match_no:02d} が全口同値です: symbol={vals[0]} "
                    f"(pH={p.p_home:.3f} pD={p.p_draw:.3f} pA={p.p_away:.3f} margin={float(p.margin or 0.0):.3f})",
                )
    if len(tickets) < REQUIRED_TICKET_COUNT:
        _warn(warnings, f"候補不足: ユニーク口が {len(tickets)} / {REQUIRED_TICKET_COUNT} です。")

    scenario_result_labels: List[List[str]] = []
    for t in tickets:
        scenario_result_labels.append([_result_label_from_symbol(x) for x in t])

    return tickets, flip_descs, SCENARIO_DEFS, stats, scenario_result_labels


def _write_buyplan_csv(
    plans: List[MatchPlan],
    tickets: List[List[str]],
    out_csv: str,
    scenario_defs: List[ScenarioDef],
    scenario_result_labels: List[List[str]],
) -> None:
    rows = []
    for i, p in enumerate(plans):
        row = {
            "match_no": p.match_no,
            "home_team": p.home_team,
            "away_team": p.away_team,
            "league": p.league,
            "context_primary_pick": p.context_primary_pick,
            "context_secondary_pick": p.context_secondary_pick,
            "context_risk_level": p.context_risk_level,
            "context_ticket_guidance": p.context_ticket_guidance,
            "context_decision_summary": p.context_decision_summary,
            "p_home": p.p_home,
            "p_draw": p.p_draw,
            "p_away": p.p_away,
            "diff_ha": p.diff_ha,
            "margin_best_second": float(p.margin) if p.margin is not None else "",
            "entropy": p.entropy,
            "draw_candidate": p.draw_candidate,
            "draw_candidate_reason": p.draw_candidate_reason,
            "draw_score": p.draw_score,
            "draw_branch_candidate": p.draw_branch_candidate,
            "draw_branch_score": p.draw_branch_score,
            "away_value_candidate": p.away_value_candidate,
            "away_value_score": p.away_value_score,
            "flab_trial_flag": p.flab_trial_flag,
            "flab_trial_score": p.flab_trial_score,
            "closeness_2axis": p.closeness,
            "closeness_effective_base": p.closeness_effective,
            "d_weight_base": p.d_weight,
            "buyplan_base_choice": p.buyplan_choice,
            "buyplan_base_reason": p.buyplan_reason,
        }
        for t_idx in range(REQUIRED_TICKET_COUNT):
            idx = t_idx + 1
            col = f"ticket{idx:02d}"
            row[col] = tickets[t_idx][i] if t_idx < len(tickets) else ""
            sdef = scenario_defs[t_idx] if t_idx < len(scenario_defs) else ScenarioDef(f"{idx:02d}", "N/A", "")
            row[f"scenario_id_{idx:02d}"] = sdef.scenario_id
            row[f"scenario_name_{idx:02d}"] = sdef.scenario_name
            row[f"scenario_note_{idx:02d}"] = sdef.scenario_note
            sres = scenario_result_labels[t_idx][i] if t_idx < len(scenario_result_labels) and i < len(scenario_result_labels[t_idx]) else ""
            row[f"scenario_predicted_result_{idx:02d}"] = sres
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


def _write_buyplan_html(
    plans: List[MatchPlan],
    tickets: List[List[str]],
    flip_descs: List[str],
    warnings: List[str],
    out_html: str,
    input_csv: str,
    outdir: str,
    ticket_stats: Dict[str, int],
    scenario_defs: List[ScenarioDef],
) -> None:
    no_data = [p for p in plans if p.status != "OK"]
    round_id = plans[0].toto_round_id if plans else "UNKNOWN"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_basename = Path(out_html).stem.strip()
    title_label = "BuyPlan_caution" if html_basename == "buyplan_caution" else "BuyPlan"
    scenario_meta = " / ".join(
        [
            f"候補{i + 1:02d}:{s.scenario_name}（{SCENARIO_JA_BY_ID.get(s.scenario_id, '未定義')}）"
            for i, s in enumerate(scenario_defs[:REQUIRED_TICKET_COUNT])
        ]
    )

    html = []
    html.append("<!doctype html>")
    html.append("<html lang='ja'>")
    html.append("<head>")
    html.append("<meta charset='utf-8'>")
    html.append(f"<title>{title_label}（toto 13試合）</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui,-apple-system,sans-serif;margin:20px;line-height:1.45;color:#111;}")
    html.append(".header{border:1px solid #ddd;background:#fafafa;padding:12px 14px;margin-bottom:12px;}")
    html.append(".title{font-size:22px;font-weight:700;margin-bottom:6px;}")
    html.append(".sub{font-size:12px;color:#444;display:flex;flex-wrap:wrap;gap:10px 18px;margin-bottom:6px;}")
    html.append(".note{font-size:12px;color:#555;}")
    html.append(".table-wrap{overflow-x:auto;border:1px solid #ddd;}")
    html.append("table{border-collapse:collapse;width:100%;font-size:12px;min-width:1500px;}")
    html.append("th,td{border:1px solid #ddd;padding:6px;text-align:center;}")
    html.append("th{background:#f5f5f5;position:sticky;top:0;z-index:1;}")
    html.append("tbody tr:nth-child(even){background:#fcfcfc;}")
    html.append(".left{text-align:left;white-space:nowrap;}")
    html.append(".match{font-weight:700;}")
    html.append(".pick{font-size:14px;font-weight:700;min-width:34px;}")
    html.append(".cand-head{min-width:58px;}")
    html.append(".warn{background:#fff5f5;border:1px solid #f0b7b7;padding:10px;margin-bottom:14px;}")
    html.append(".info{background:#f7faff;border:1px solid #bcd3ff;padding:10px;margin-bottom:14px;}")
    html.append(".footer{font-size:11px;color:#555;margin-top:10px;}")
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<div class='header'>")
    html.append(f"<div class='title'>{title_label}（toto 13試合）</div>")
    html.append("<div class='sub'>")
    html.append(f"<span>toto_round_id: <b>{round_id}</b></span>")
    html.append("<span>scenario: <b>候補ごとに固定割当</b></span>")
    html.append(f"<span>scenario一覧: <b>{scenario_meta}</b></span>")
    html.append(f"<span>生成日時: <b>{generated_at}</b></span>")
    html.append(f"<span>入力: <b>{input_csv}</b></span>")
    html.append(f"<span>出力先: <b>{outdir}</b></span>")
    html.append("</div>")
    html.append("<div class='note'>記号: 1=ホーム勝ち / 0=引分 / 2=アウェイ勝ち</div>")
    unique_count = int(ticket_stats.get("unique_ticket_count", len(tickets)))
    duplicate_count = int(ticket_stats.get("duplicate_count", max(0, len(tickets) - unique_count)))
    if len(tickets) >= REQUIRED_TICKET_COUNT:
        html.append("<div class='note'>10口はユニーク生成（重複なし）</div>")
    else:
        html.append(
            f"<div class='note'>ユニーク口が不足: {len(tickets)} / {REQUIRED_TICKET_COUNT}（重複禁止ルールを維持）</div>"
        )
    html.append(f"<div class='note'>unique_ticket_count={unique_count} / duplicate_count={duplicate_count}</div>")
    html.append("</div>")

    draw_pressure_count = _round_draw_pressure_count(plans)
    scenario_draw_counts: Dict[int, int] = {}
    scenario_priority_labels: Dict[int, str] = {}
    for t_idx in range(REQUIRED_TICKET_COUNT):
        idx = t_idx + 1
        d_count = int(sum(1 for ticket in tickets[t_idx] if str(ticket).strip() == "0")) if t_idx < len(tickets) else 0
        scenario_draw_counts[idx] = d_count
        scenario_priority_labels[idx] = _candidate_draw_priority_label(d_count, draw_pressure_count)
    zero_draw_candidates = [idx for idx, d_count in scenario_draw_counts.items() if d_count == 0]
    draw_count_parts = [f"候補{idx:02d}=D{scenario_draw_counts.get(idx, 0)}本" for idx in range(1, REQUIRED_TICKET_COUNT + 1)]
    html.append(f"<div class='note'>候補別D本数: {' / '.join(draw_count_parts)}</div>")
    priority_parts = [
        f"候補{idx:02d}={scenario_priority_labels.get(idx, '通常')}" for idx in range(1, REQUIRED_TICKET_COUNT + 1)
    ]
    html.append(
        f"<div class='note'>節内D気配={draw_pressure_count}試合 / 候補優先度: {' / '.join(priority_parts)}</div>"
    )
    if zero_draw_candidates:
        zero_text = " / ".join(f"候補{idx:02d}" for idx in zero_draw_candidates)
        html.append(
            f"<div class='warn'><b>Dなし候補に注意</b><br>{zero_text} は D=0本です。"
            " 引分が複数出る節では上限が下がりやすいため、主力扱いしない前提で確認してください。</div>"
        )

    if no_data:
        html.append("<div class='warn'><b>⚠ 情報不足（NO_DATA）</b><ul>")
        for p in no_data:
            html.append(
                f"<li>M{p.match_no:02d} ({p.league}) {p.home_team} vs {p.away_team} : フォールバック=1固定</li>"
            )
        html.append("</ul></div>")

    if warnings:
        html.append("<div class='info'><b>警告ログ</b><ul>")
        for w in warnings:
            html.append(f"<li>{w}</li>")
        html.append("</ul></div>")

    html.append("<div class='table-wrap'><table>")
    html.append(
        "<thead><tr>"
        "<th class='match'>match_no</th><th class='left'>league</th><th class='left'>home_team</th><th class='left'>away_team</th>"
        "<th class='left'>context</th><th class='left'>best/second</th>"
    )
    for t_idx in range(REQUIRED_TICKET_COUNT):
        idx = t_idx + 1
        label = f"候補{idx:02d}"
        if t_idx == 0:
            label += "(基準)"
        sname = scenario_defs[t_idx].scenario_name if t_idx < len(scenario_defs) else "N/A"
        sid = scenario_defs[t_idx].scenario_id if t_idx < len(scenario_defs) else ""
        sja = SCENARIO_JA_BY_ID.get(sid, "未定義")
        priority = scenario_priority_labels.get(idx, "通常")
        html.append(
            f"<th class='cand-head'>{label}<br><small>{sname}（{sja}） / D{scenario_draw_counts.get(idx, 0)} / {priority}</small></th>"
        )
    html.append("</tr></thead><tbody>")
    for i, p in enumerate(plans):
        html.append("<tr>")
        html.append(f"<td class='match'>{p.match_no}</td>")
        html.append(f"<td class='left'>{p.league}</td>")
        html.append(f"<td class='left'>{p.home_team}</td>")
        html.append(f"<td class='left'>{p.away_team}</td>")
        context_text = ""
        if any([p.context_primary_pick, p.context_secondary_pick, p.context_risk_level, p.context_ticket_guidance, p.context_decision_summary]):
            context_parts = []
            if p.context_primary_pick:
                context_parts.append(f"本命={p.context_primary_pick}")
            if p.context_secondary_pick:
                context_parts.append(f"次点={p.context_secondary_pick}")
            if p.context_risk_level:
                context_parts.append(f"risk={p.context_risk_level}")
            if p.context_ticket_guidance:
                context_parts.append(f"guide={p.context_ticket_guidance}")
            if p.context_decision_summary:
                context_parts.append(p.context_decision_summary)
            context_text = " / ".join(context_parts)
        html.append(f"<td class='left'>{context_text}</td>")
        best_second = (
            f"best={p.prob_best_pick} second={p.prob_second_pick} third={p.third} "
            f"(base={p.base_pick}) "
            f"margin={float(p.margin) if p.margin is not None else 0.0:.3f} "
            f"entropy={float(p.entropy):.3f} "
            f"draw_cand={int(bool(p.draw_candidate))}"
            if p.status == "OK"
            else ""
        )
        html.append(f"<td class='left'>{best_second}</td>")
        for t_idx in range(REQUIRED_TICKET_COUNT):
            val = tickets[t_idx][i] if t_idx < len(tickets) else ""
            if val and p.status != "OK":
                val = f"{val}*"
            html.append(f"<td class='pick'>{val}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")

    html.append("<h3>候補の変更点（flip_desc）</h3>")
    html.append("<table>")
    html.append("<thead><tr><th>ticket</th><th class='left'>scenario_name</th><th class='left'>description</th></tr></thead><tbody>")
    for t_idx in range(REQUIRED_TICKET_COUNT):
        d = flip_descs[t_idx] if t_idx < len(flip_descs) else "（未生成）"
        sname = scenario_defs[t_idx].scenario_name if t_idx < len(scenario_defs) else "N/A"
        sid = scenario_defs[t_idx].scenario_id if t_idx < len(scenario_defs) else ""
        sja = SCENARIO_JA_BY_ID.get(sid, "未定義")
        html.append(
            f"<tr><td>候補{t_idx + 1:02d}</td><td class='left'>{sname}（{sja}）</td><td class='left'>{d}</td></tr>"
        )
    html.append("</tbody></table>")
    html.append(
        "<div class='footer'>"
        f"warnings={len(warnings)} / duplicate_skips={ticket_stats.get('duplicate_skips', 0)} / "
        f"generated_variants={ticket_stats.get('generated', 0)} / "
        f"unique_ticket_count={ticket_stats.get('unique_ticket_count', len(tickets))} / "
        f"duplicate_count={ticket_stats.get('duplicate_count', 0)}"
        "</div>"
    )

    html.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def _load_actual_results_df(actual_csv: str) -> pd.DataFrame:
    df = pd.read_csv(actual_csv)
    if "match_no" not in df.columns:
        raise RuntimeError(f"actual csv missing required column: match_no ({actual_csv})")
    result_col = None
    for c in ["actual_result", "result", "actual"]:
        if c in df.columns:
            result_col = c
            break
    if result_col is None:
        raise RuntimeError(f"actual csv missing result column (actual_result/result/actual): {actual_csv}")
    base_cols = ["match_no", result_col]
    extra_cols = [c for c in ["league", "home_team", "away_team", "match_id", "datetime"] if c in df.columns]
    out = df[base_cols + extra_cols].copy()
    out["match_no"] = pd.to_numeric(out["match_no"], errors="coerce")
    out = out.dropna(subset=["match_no"]).copy()
    out["match_no"] = out["match_no"].astype(int)
    sym_map = {"H": "1", "D": "0", "A": "2", "1": "1", "0": "0", "2": "2"}
    out["actual_symbol"] = out[result_col].astype(str).str.strip().map(sym_map)
    out = out.dropna(subset=["actual_symbol"]).copy()
    out["actual_result"] = out["actual_symbol"].map(_result_label_from_symbol)
    keep_cols = ["match_no", "actual_symbol", "actual_result"]
    for c in extra_cols:
        if c in out.columns:
            keep_cols.append(c)
    return out[keep_cols]


def _validate_actual_alignment(plans: List[MatchPlan], actual_df: pd.DataFrame, actual_csv_path: str = "") -> None:
    if actual_df.empty:
        raise RuntimeError(f"actual csv is empty: {actual_csv_path}")
    actual_map = {int(r["match_no"]): r for _, r in actual_df.iterrows()}
    mismatches: List[str] = []
    for p in plans:
        row = actual_map.get(int(p.match_no))
        if row is None:
            mismatches.append(f"M{p.match_no:02d}:missing")
            continue
        if "home_team" in actual_df.columns and str(row.get("home_team", "")) != str(p.home_team):
            mismatches.append(
                f"M{p.match_no:02d}:home {p.home_team}!={row.get('home_team', '')}"
            )
            continue
        if "away_team" in actual_df.columns and str(row.get("away_team", "")) != str(p.away_team):
            mismatches.append(
                f"M{p.match_no:02d}:away {p.away_team}!={row.get('away_team', '')}"
            )
            continue
        if "league" in actual_df.columns and str(row.get("league", "")) not in {"", str(p.league)}:
            mismatches.append(
                f"M{p.match_no:02d}:league {p.league}!={row.get('league', '')}"
            )
            continue
    if mismatches:
        sample = "; ".join(mismatches[:5])
        raise RuntimeError(
            f"actual csv does not align with buyplan matches: {len(mismatches)} mismatch(es) "
            f"[{sample}] source={actual_csv_path}"
        )


def _build_buyplan_round_nav_items(outdir: str) -> List[Tuple[str, str, bool]]:
    outdir_abs = os.path.abspath(outdir)
    rounds_dir = os.path.join(BASE_DIR, "data", "eval", "rounds")
    items = [("現在", "buyplan_scored.html", True)]
    if not os.path.isdir(rounds_dir):
        return items
    for name in sorted(os.listdir(rounds_dir)):
        if not re.fullmatch(r"round\d{2}", name):
            continue
        html_path = os.path.join(rounds_dir, name, "buyplan_scored.html")
        if not os.path.exists(html_path):
            continue
        try:
            num = int(name.replace("round", ""))
        except ValueError:
            continue
        rel = os.path.relpath(html_path, outdir_abs)
        items.append((f"第{num:02d}節", rel, False))
    return items


def _buyplan_simulation_index_link(outdir: str) -> str:
    outdir_abs = os.path.abspath(outdir)
    purchase_dir = os.path.join(BASE_DIR, "data", "purchase_reference")
    backtest_root = os.path.join(purchase_dir, "backtest")
    index_path = os.path.join(purchase_dir, "buyplan_simulation.html")
    if outdir_abs == purchase_dir or outdir_abs.startswith(backtest_root + os.sep):
        return os.path.relpath(index_path, outdir_abs)
    return ""


def _write_buyplan_scored_outputs(
    plans: List[MatchPlan],
    tickets: List[List[str]],
    outdir: str,
    actual_df: pd.DataFrame,
    actual_csv_path: str = "",
    output_name: str = "buyplan",
) -> Tuple[str, str]:
    _validate_actual_alignment(plans, actual_df, actual_csv_path=actual_csv_path)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: List[Dict[str, object]] = []
    actual_map = {int(r["match_no"]): (str(r["actual_symbol"]), str(r["actual_result"])) for _, r in actual_df.iterrows()}

    for i, p in enumerate(plans):
        a_sym, a_label = actual_map.get(int(p.match_no), ("", ""))
        best_second = ""
        if p.status == "OK":
            best_second = (
                f"best={p.prob_best_pick} second={p.prob_second_pick} third={p.third} "
                f"(base={p.base_pick}) margin={float(p.margin) if p.margin is not None else 0.0:.3f}"
            )
        row: Dict[str, object] = {
            "match_no": p.match_no,
            "league": p.league,
            "home_team": p.home_team,
            "away_team": p.away_team,
            "best_second": best_second,
            "actual_symbol": a_sym,
            "actual_result": a_label,
        }
        for t_idx in range(REQUIRED_TICKET_COUNT):
            ticket_col = f"ticket{t_idx+1:02d}"
            pred = tickets[t_idx][i] if t_idx < len(tickets) else ""
            row[ticket_col] = pred
            row[f"{ticket_col}_is_hit"] = bool(a_sym) and (str(pred) == str(a_sym))
        rows.append(row)

    scored_df = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    n = len(scored_df)
    for t_idx in range(REQUIRED_TICKET_COUNT):
        ticket_col = f"ticket{t_idx+1:02d}"
        hit_col = f"{ticket_col}_is_hit"
        hits = int(pd.to_numeric(scored_df[hit_col], errors="coerce").fillna(False).astype(bool).sum())
        summary_rows.append(
            {
                "ticket": ticket_col,
                "hits": hits,
                "total": n,
                "hit_rate": (hits / n) if n else 0.0,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    draw_count_map: Dict[int, int] = {}
    for t_idx in range(REQUIRED_TICKET_COUNT):
        ticket_col = f"ticket{t_idx+1:02d}"
        if ticket_col in scored_df.columns:
            draw_count_map[t_idx + 1] = int((scored_df[ticket_col].astype(str) == "0").sum())
        else:
            draw_count_map[t_idx + 1] = 0
    draw_pressure_count = _round_draw_pressure_count(plans)
    priority_label_map: Dict[int, str] = {
        idx: _candidate_draw_priority_label(draw_count_map.get(idx, 0), draw_pressure_count)
        for idx in range(1, REQUIRED_TICKET_COUNT + 1)
    }
    zero_draw_candidates = [idx for idx, d_count in draw_count_map.items() if d_count == 0]
    # 揺らし個数: ticket01との差分件数（実際に券面でどれだけ変えたか）
    sway_count_map: Dict[int, int] = {}
    if "ticket01" in scored_df.columns:
        base_series = scored_df["ticket01"].astype(str)
        for t_idx in range(REQUIRED_TICKET_COUNT):
            tcol = f"ticket{t_idx+1:02d}"
            if tcol in scored_df.columns:
                sway_count_map[t_idx + 1] = int((scored_df[tcol].astype(str) != base_series).sum())
            else:
                sway_count_map[t_idx + 1] = 0

    safe_output_name = (output_name or "buyplan").strip() or "buyplan"
    out_scored_csv = os.path.join(outdir, f"{safe_output_name}_scored.csv")
    out_scored_summary_csv = os.path.join(outdir, f"{safe_output_name}_scored_summary.csv")
    outdir_abs = os.path.abspath(outdir)
    purchase_dir = os.path.join(BASE_DIR, "data", "purchase_reference")
    backtest_root = os.path.join(purchase_dir, "backtest")
    use_simulation_name = outdir_abs == purchase_dir or outdir_abs.startswith(backtest_root + os.sep)
    if use_simulation_name and safe_output_name == "buyplan":
        out_scored_html_name = "buyplan_simulation_current.html" if outdir_abs == purchase_dir else "buyplan_simulation.html"
    else:
        out_scored_html_name = f"{safe_output_name}_scored.html"
    out_scored_html = os.path.join(outdir, out_scored_html_name)
    scored_df.to_csv(out_scored_csv, index=False, encoding="utf-8")
    summary_df.to_csv(out_scored_summary_csv, index=False, encoding="utf-8")

    html: List[str] = []
    html.append("<!doctype html>")
    html.append("<html lang='ja'><head><meta charset='utf-8'>")
    html.append("<title>buyplanシミュレーション</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui,-apple-system,sans-serif;margin:20px;}")
    html.append("table{border-collapse:collapse;width:100%;font-size:12px;}")
    html.append("th,td{border:1px solid #ddd;padding:6px;text-align:center;}")
    html.append("th{background:#f5f5f5;}")
    html.append(".left{text-align:left;white-space:nowrap;}")
    html.append(".ok{color:#0a7d27;font-weight:700;}")
    html.append(".ng{color:#b3261e;font-weight:700;}")
    html.append(".nav{margin:10px 0 14px;padding:10px 12px;border:1px solid #ddd;background:#fafafa;display:flex;gap:10px;align-items:center;flex-wrap:wrap;}")
    html.append(".nav label{font-weight:700;}")
    html.append(".nav select{font:inherit;padding:4px 8px;min-width:180px;}")
    html.append("</style></head><body>")
    html.append("<h2>buyplanシミュレーション</h2>")
    html.append(f"<p>更新日時: <b>{generated_at}</b></p>")
    if actual_csv_path:
        html.append(f"<p>actual source: <b>{actual_csv_path}</b></p>")
    index_link = _buyplan_simulation_index_link(outdir)
    if index_link:
        html.append(f"<p><a href='{index_link}'>buyplanシミュレーション一覧へ</a></p>")
    else:
        nav_items = _build_buyplan_round_nav_items(outdir)
        if nav_items:
            html.append("<div class='nav'>")
            html.append("<label for='round-select'>節を選択</label>")
            html.append("<select id='round-select' onchange=\"if(this.value){window.location.href=this.value;}\">")
            for label, rel, selected in nav_items:
                sel = " selected" if selected else ""
                html.append(f"<option value='{rel}'{sel}>{label}</option>")
            html.append("</select>")
            html.append("</div>")

    best_row = summary_df.sort_values(["hits", "ticket"], ascending=[False, True]).head(1)
    best_text = "-"
    if not best_row.empty:
        r = best_row.iloc[0]
        best_text = f"{r['ticket']} {int(r['hits'])}/{int(r['total'])}"
    avg_hit_rate = float(summary_df["hit_rate"].mean()) if not summary_df.empty else 0.0
    html.append(f"<p>best: <b>{best_text}</b> / avg_hit: <b>{avg_hit_rate:.1%}</b></p>")
    draw_count_parts = [f"候補{idx:02d}=D{draw_count_map.get(idx, 0)}本" for idx in range(1, REQUIRED_TICKET_COUNT + 1)]
    html.append(f"<p>候補別D本数: <b>{' / '.join(draw_count_parts)}</b></p>")
    priority_parts = [f"候補{idx:02d}={priority_label_map.get(idx, '通常')}" for idx in range(1, REQUIRED_TICKET_COUNT + 1)]
    html.append(f"<p>節内D気配: <b>{draw_pressure_count}試合</b> / 候補優先度: <b>{' / '.join(priority_parts)}</b></p>")
    if zero_draw_candidates:
        zero_text = " / ".join(f"候補{idx:02d}" for idx in zero_draw_candidates)
        html.append(
            f"<p style='color:#b3261e;'><b>Dなし候補に注意:</b> {zero_text} は D=0本です。"
            " 引分が複数出る節では上限が下がりやすいです。</p>"
        )

    html.append("<h3>Match Detail</h3><table><thead><tr>")
    html.extend(
        [
            "<th>match_no</th>",
            "<th>league</th>",
            "<th>home_team</th>",
            "<th>away_team</th>",
            "<th>best/second</th>",
            "<th>actual</th>",
        ]
    )
    for t_idx in range(REQUIRED_TICKET_COUNT):
        html.append(f"<th>ticket{t_idx+1:02d}</th>")
    html.append("</tr>")
    html.append("<tr>")
    html.append("<th colspan='6'>ラベル</th>")
    ticket_labels = {
        1: "LOCK遵守",
        2: "LOCK遵守",
        3: "LOCK遵守",
        4: "探索 小",
        5: "探索 中小",
        6: "探索 中",
        7: "探索 中大",
        8: "探索 大",
        9: "実験 接戦反転",
        10: "実験 strong_break",
    }
    for t_idx in range(REQUIRED_TICKET_COUNT):
        n = sway_count_map.get(t_idx + 1, 0)
        d_count = draw_count_map.get(t_idx + 1, 0)
        priority = priority_label_map.get(t_idx + 1, "通常")
        html.append(f"<th>{ticket_labels.get(t_idx+1, '')}<br><small>揺らし{n} / D{d_count} / {priority}</small></th>")
    html.append("</tr></thead><tbody>")
    for _, r in scored_df.iterrows():
        html.append("<tr>")
        html.append(f"<td>{int(r['match_no'])}</td>")
        html.append(f"<td>{r['league']}</td>")
        html.append(f"<td class='left'>{r['home_team']}</td>")
        html.append(f"<td class='left'>{r['away_team']}</td>")
        html.append(f"<td class='left'>{r['best_second']}</td>")
        html.append(f"<td>{r['actual_symbol']}</td>")
        for t_idx in range(REQUIRED_TICKET_COUNT):
            ticket_col = f"ticket{t_idx+1:02d}"
            hit_col = f"{ticket_col}_is_hit"
            cls = "ok" if bool(r[hit_col]) else "ng"
            html.append(f"<td class='{cls}'>{r[ticket_col]}</td>")
        html.append("</tr>")
    html.append("</tbody><tfoot>")
    html.append("<tr>")
    html.append("<td colspan='6' class='left'><b>hits</b></td>")
    hit_map = {str(r["ticket"]): int(r["hits"]) for _, r in summary_df.iterrows()}
    rate_map = {str(r["ticket"]): float(r["hit_rate"]) for _, r in summary_df.iterrows()}
    for t_idx in range(REQUIRED_TICKET_COUNT):
        tcol = f"ticket{t_idx+1:02d}"
        html.append(f"<td><b>{hit_map.get(tcol, 0)}</b></td>")
    html.append("</tr>")
    html.append("<tr>")
    html.append("<td colspan='6' class='left'><b>hit_rate</b></td>")
    for t_idx in range(REQUIRED_TICKET_COUNT):
        tcol = f"ticket{t_idx+1:02d}"
        html.append(f"<td><b>{rate_map.get(tcol, 0.0):.1%}</b></td>")
    html.append("</tr>")
    html.append("</tfoot></table></body></html>")

    with open(out_scored_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_scored_csv, out_scored_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toto buy plan (single x10) from predictions.csv")
    parser.add_argument("--in", dest="in_csv", required=True, help="input predictions.csv path")
    parser.add_argument("--scenario", default="", help="deprecated: ignored (scenario is fixed per 候補01..10)")
    parser.add_argument("--scenario-file", default="", help="deprecated: ignored (scenario is fixed per 候補01..10)")
    parser.add_argument("--scenario-dir", default="", help="deprecated: ignored (scenario is fixed per 候補01..10)")
    parser.add_argument("--outdir", default="", help="output directory (default: input file dir)")
    parser.add_argument(
        "--name",
        default="buyplan",
        help="output basename (default: buyplan). 例: buyplan_caution",
    )
    parser.add_argument(
        "--actual-csv",
        default="",
        help="採点用実結果CSV（match_no + actual_result/result/actual）。指定時に buyplan_scored.* を出力",
    )
    parser.add_argument(
        "--toto-order-csv",
        default="",
        help="toto対象リストCSV（既定: data/manual/toto節リスト.csv、旧 data/manual/toto並び順.csv も可）",
    )
    parser.add_argument(
        "--context-csv",
        default="",
        help="buyplan用補助CSV（既定: 入力CSVと同じディレクトリの predictions_buyplan_context.csv）",
    )
    args = parser.parse_args()

    warnings: List[str] = []
    print(
        "[BUYPLAN_CONFIG] "
        f"2AXIS_ENABLE={int(BUYPLAN_2AXIS_DRAW_ENABLE)} "
        f"BASE_MODE={BUYPLAN_BASE_MODE} BASE_TOP_GAP(strong/mid)={BASE_TOP_GAP_STRONG:.3f}/{BASE_TOP_GAP_MID:.3f} "
        f"SHAPE_STRENGTH(strong/mid)={SHAPE_STRENGTH_STRONG:.3f}/{SHAPE_STRENGTH_MID:.3f} "
        f"SHAPE_DRAW(r31/spread/entropy)={SHAPE_DRAW_RATIO31_MIN:.3f}/{SHAPE_DRAW_SPREAD_MAX:.3f}/{SHAPE_DRAW_ENTROPY_MIN:.3f} "
        f"HA_DRAW_MARGIN={BUYPLAN_BALANCE_T_DRAW:.3f} DRAW_MIN={BUYPLAN_BALANCE_D_MIN:.3f} "
        f"SMALL_GAP_RULE(strength/ratio31/entropy)={SMALL_GAP_STRENGTH_MAX:.3f}/{SMALL_GAP_RATIO31_MIN:.3f}/{SMALL_GAP_ENTROPY_MIN:.3f} "
        f"WEAK_DRAW_RULE(strength/ratio31/entropy)={WEAK_DRAW_STRENGTH_MAX:.3f}/{WEAK_DRAW_RATIO31_MIN:.3f}/{WEAK_DRAW_ENTROPY_MIN:.3f} "
        f"REL_CLOSE(ratio/gap/spread)={REL_CLOSE_RATIO_TO_TOP_MIN:.2f}/{REL_CLOSE_GAP_TO_ABOVE_MAX:.2f}/{REL_CLOSE_SPREAD_MAX:.2f} "
        f"REL_SCORE_ALPHA={RELATIVE_SCORE_ALPHA:.2f} "
        f"LOCK02_MARGIN={LOCK02_MARGIN_THRESHOLD:.3f} LOCK03_MARGIN={LOCK03_MARGIN_THRESHOLD:.3f} "
        f"LOCK_MAX_FLIPS(02/03)={LOCK02_MAX_FLIPS}/{LOCK03_MAX_FLIPS} "
        f"STRONG_BREAK_MARGIN={STRONG_BREAK_MARGIN_THRESHOLD:.3f} "
        f"EXTREME_MARGIN_RELEASE(enable/th/min_alt)="
        f"{int(ENABLE_EXTREME_MARGIN_RELEASE)}/{EXTREME_MARGIN_RELEASE_THRESHOLD:.3f}/{EXTREME_MARGIN_RELEASE_MIN_ALT_TICKETS} "
        f"WEAK_DRAW_MARGIN={WEAK_DRAW_MARGIN:.3f} WEAK_DRAW_ENTROPY_MIN={WEAK_DRAW_ENTROPY_MIN:.3f} "
        f"TARGET_DRAW_RANGE(lock/prob/exp)={TARGET_DRAW_MIN_LOCK}-{TARGET_DRAW_MAX_LOCK}/"
        f"{TARGET_DRAW_MIN_PROB}-{TARGET_DRAW_MAX_PROB}/"
        f"{TARGET_DRAW_MIN_EXP}-{TARGET_DRAW_MAX_EXP} "
        f"C_DIFF={BUYPLAN_2AXIS_C_DIFF:.3f} D_MIN={BUYPLAN_2AXIS_D_MIN:.3f} "
        f"W_DRAW(base/lock/prob/exp)={BUYPLAN_2AXIS_W_DRAW:.3f}/{BUYPLAN_2AXIS_W_DRAW_LOCK:.3f}/{BUYPLAN_2AXIS_W_DRAW_PROB:.3f}/{BUYPLAN_2AXIS_W_DRAW_EXP:.3f} "
        f"MAX_STRONG={BUYPLAN_2AXIS_MAX_STRONG:.3f} "
        f"CAP_D_MATCHES(default/lock/prob/exp)={BUYPLAN_2AXIS_CAP_DEFAULT}/{BUYPLAN_2AXIS_CAP_LOCK}/{BUYPLAN_2AXIS_CAP_PROB}/{BUYPLAN_2AXIS_CAP_EXP} "
        f"DRAW_BOOST(base/close,margin_max)={DRAW_BOOST:.3f}/{DRAW_BOOST_CLOSE:.3f},{DRAW_BOOST_MARGIN_MAX:.3f} "
        f"MARGIN_D_MAX={MARGIN_D_MAX:.3f} ENTROPY_MIN={ENTROPY_MIN:.3f} BEST_MAX_FOR_D={BEST_MAX_FOR_D:.3f} "
        f"DRAW_MATCH_CAP={DRAW_MATCH_CAP} ZERO_RATIO_CAP={ZERO_RATIO_CAP:.3f} "
        f"PER_MATCH_SAME_SYMBOL_CAP={PER_MATCH_SAME_SYMBOL_CAP} "
        f"SWAY_DEGREE(04/05/06/07)="
        f"{SWAY_DEGREE_TABLE.get(4, 0)}/"
        f"{SWAY_DEGREE_TABLE.get(5, 1)}/"
        f"{SWAY_DEGREE_TABLE.get(6, 2)}/"
        f"{SWAY_DEGREE_TABLE.get(7, 3)} "
        f"ALL_SAME_SECOND_RATIO(08/09/10)="
        f"{ALL_SAME_SECOND_RATIO_TABLE.get(8, 0.0):.2f}/"
        f"{ALL_SAME_SECOND_RATIO_TABLE.get(9, 0.0):.2f}/"
        f"{ALL_SAME_SECOND_RATIO_TABLE.get(10, 0.0):.2f} "
        f"MAX_FLIPS_PER_MATCH(prob/exp)={MAX_FLIPS_PER_MATCH_PROB}/{MAX_FLIPS_PER_MATCH_EXP} "
        f"ENABLE_LAYER(small_gap/weak_draw/same_cap/final_range)="
        f"{int(ENABLE_SMALL_GAP_RULE)}/{int(ENABLE_WEAK_DRAW_APPLY)}/{int(ENABLE_SAME_SYMBOL_CAP)}/{int(ENABLE_FINAL_RANGE)} "
        f"ENABLE_GRADUAL_SWAY={int(ENABLE_GRADUAL_SWAY)}"
    )
    in_csv = os.path.abspath((args.in_csv or "").strip())
    if args.scenario or args.scenario_file or args.scenario_dir:
        _warn(warnings, "--scenario / --scenario-file / --scenario-dir は非推奨です。候補01〜10の内部シナリオを使用します。")
    outdir = args.outdir.strip() if isinstance(args.outdir, str) else ""
    if not outdir:
        outdir = os.path.dirname(os.path.abspath(in_csv)) or "."
    os.makedirs(outdir, exist_ok=True)
    output_name = (args.name.strip() if isinstance(args.name, str) else "") or "buyplan"

    try:
        df = pd.read_csv(in_csv)
    except Exception as e:
        print(f"[ERROR] 入力CSVを読み込めません: {in_csv} ({e})")
        return
    context_csv = args.context_csv.strip() if isinstance(args.context_csv, str) else ""
    if not context_csv:
        context_csv = os.path.join(os.path.dirname(in_csv), "predictions_buyplan_context.csv")
    context_df = _load_buyplan_context_df(context_csv, warnings)
    df = _merge_buyplan_context(df, context_df, warnings)

    toto_order_csv = _resolve_toto_order_csv_path(args.toto_order_csv.strip() if isinstance(args.toto_order_csv, str) else "")
    order_df = _load_toto_order_df(toto_order_csv, warnings, df)
    diff_df = _build_toto_diff_report(df, order_df)
    if not diff_df.empty:
        diff_csv = os.path.join(outdir, f"{output_name}_toto_diff.csv")
        diff_df.to_csv(diff_csv, index=False, encoding="utf-8-sig")
        miss_pred = int((diff_df["diff_type"] == "missing_in_predictions").sum())
        miss_order = int((diff_df["diff_type"] == "missing_in_toto_order").sum())
        if miss_pred or miss_order:
            _warn(
                warnings,
                f"toto差分を出力しました: {diff_csv} "
                f"(missing_in_predictions={miss_pred}, missing_in_toto_order={miss_order})",
            )
        else:
            print(f"[INFO] toto差分チェック: 不一致なし ({diff_csv})")
    df = _apply_toto_match_order(df, order_df, warnings)

    if len(df) != REQUIRED_MATCH_COUNT:
        _warn(warnings, f"入力行数が {REQUIRED_MATCH_COUNT} ではありません: {len(df)}")

    if "toto_round_id" not in df.columns:
        df = df.copy()
        df["toto_round_id"] = _derive_round_id(df, in_csv)

    df = _normalize_match_no(df, warnings)
    df = _dedupe_match_no(df, warnings)
    plans = _build_match_plans(df, warnings, base_mode=BUYPLAN_BASE_MODE)
    tickets, flip_descs, scenario_defs, ticket_stats, scenario_result_labels = _generate_tickets_by_scenario(df, plans, warnings)

    out_csv = os.path.join(outdir, f"{output_name}.csv")
    out_html = os.path.join(outdir, f"{output_name}.html")
    _write_buyplan_csv(
        plans=plans,
        tickets=tickets,
        out_csv=out_csv,
        scenario_defs=scenario_defs,
        scenario_result_labels=scenario_result_labels,
    )
    _write_buyplan_html(
        plans=plans,
        tickets=tickets,
        flip_descs=flip_descs,
        warnings=warnings,
        out_html=out_html,
        input_csv=in_csv,
        outdir=outdir,
        ticket_stats=ticket_stats,
        scenario_defs=scenario_defs,
    )

    actual_csv = os.path.abspath(args.actual_csv.strip()) if isinstance(args.actual_csv, str) and args.actual_csv.strip() else ""
    if actual_csv:
        try:
            actual_df = _load_actual_results_df(actual_csv)
            out_scored_csv, out_scored_html = _write_buyplan_scored_outputs(
                plans=plans,
                tickets=tickets,
                outdir=outdir,
                actual_df=actual_df,
                actual_csv_path=actual_csv,
                output_name=output_name,
            )
            print(f"[OK] {out_scored_csv}")
            print(f"[OK] {os.path.join(outdir, f'{output_name}_scored_summary.csv')}")
            print(f"[OK] {out_scored_html}")
        except Exception as e:
            print(f"[ERROR] scored出力に失敗: {actual_csv} ({e})")

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_html}")


if __name__ == "__main__":
    main()
