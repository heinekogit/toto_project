#!/usr/bin/env python3
import argparse
from collections import Counter
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
BUYPLAN_ALLOCATOR_ENGINE = os.environ.get("BUYPLAN_ALLOCATOR_ENGINE", "world_topology_v1").strip().lower()
BUYPLAN_DRAW_TARGET_RATIO = float(os.environ.get("BUYPLAN_DRAW_TARGET_RATIO", "0.300"))
BUYPLAN_DRAW_MIN_RATIO = float(os.environ.get("BUYPLAN_DRAW_MIN_RATIO", "0.269"))
BUYPLAN_DRAW_MAX_RATIO = float(os.environ.get("BUYPLAN_DRAW_MAX_RATIO", "0.323"))
BUYPLAN_TICKET_DRAW_MIN = int(os.environ.get("BUYPLAN_TICKET_DRAW_MIN", "2"))
BUYPLAN_TICKET_DRAW_MAX = int(os.environ.get("BUYPLAN_TICKET_DRAW_MAX", "6"))
BUYPLAN_HIGH_PD_FLOOR = float(os.environ.get("BUYPLAN_HIGH_PD_FLOOR", "0.35"))
BUYPLAN_HIGH_PD_MIN_COVER = int(os.environ.get("BUYPLAN_HIGH_PD_MIN_COVER", "2"))
ENABLE_PROB_DRAW_FLOOR = os.environ.get("BUYPLAN_ENABLE_PROB_DRAW_FLOOR", "1") == "1"
BUYPLAN_USE_PREDICTED_RESULT_BASE = os.environ.get("BUYPLAN_USE_PREDICTED_RESULT_BASE", "0") == "1"
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


def _normalize_symbol_token(value: object) -> str:
    token = _safe_text(value, "").upper()
    mapped = _predicted_result_to_symbol(token)
    if mapped:
        return mapped
    num = pd.to_numeric(value, errors="coerce")
    if pd.notna(num):
        try:
            token = str(int(float(num)))
        except (TypeError, ValueError):
            token = ""
        mapped = _predicted_result_to_symbol(token)
        if mapped:
            return mapped
    return ""


@dataclass(frozen=True)
class ScenarioDef:
    scenario_id: str
    scenario_name: str
    scenario_note: str


SCENARIO_DEFS: List[ScenarioDef] = [
    ScenarioDef("01", "Main", "本線"),
    ScenarioDef("02", "Draw Compression", "draw cluster を優先して吸収する券"),
    ScenarioDef("03", "Volatility Swing", "swing cluster を優先して吸収する券"),
    ScenarioDef("04", "Hold Guard", "hold core を維持するガード券"),
    ScenarioDef("05", "Draw Pressure", "draw pressure を厚めに取る券"),
    ScenarioDef("06", "J2 Draw Pressure", "J2 の draw pressure を拾う券"),
    ScenarioDef("07", "Away Swing", "away 寄り swing を拾う券"),
    ScenarioDef("08", "Close Draw", "close / low-tempo draw を拾う券"),
    ScenarioDef("09", "Low Tempo Cover", "low-tempo draw を補完する券"),
    ScenarioDef("10", "Blend Wide", "draw と swing を混ぜる探索券"),
]

SCENARIO_JA_BY_ID: Dict[str, str] = {
    "01": "本線",
    "02": "draw compression",
    "03": "volatility swing",
    "04": "hold guard",
    "05": "draw pressure",
    "06": "J2 draw pressure",
    "07": "away swing",
    "08": "close draw",
    "09": "low-tempo cover",
    "10": "blend wide",
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
    context_admission_policy: str = ""
    context_rank1_pick: str = ""
    context_rank2_pick: str = ""
    context_rank3_pick: str = ""
    context_rank1_prob: float = 0.0
    context_rank2_prob: float = 0.0
    context_rank3_prob: float = 0.0
    context_risk_level: str = ""
    context_ticket_guidance: str = ""
    context_decision_summary: str = ""
    context_draw_core_flag: bool = False
    context_draw_purchase_tier: str = ""
    context_match_purchase_type: str = ""
    context_match_purchase_subtype: str = ""
    context_purchase_type_confidence: str = ""
    context_purchase_type_reason: str = ""
    context_anchor_candidate_flag: bool = False
    context_draw_core_candidate_flag: bool = False
    lab_hold_weight: float = 0.0
    lab_stall_weight: float = 0.0
    lab_flip_weight: float = 0.0
    lab_volatility_score: float = 0.0
    lab_draw_tension_score: float = 0.0
    lab_dynamic_swing_score: float = 0.0
    lab_hold_foundation_score: float = 0.0
    lab_stall_compactness_score: float = 0.0
    lab_flip_dislocation_score: float = 0.0
    lab_home_path_score: float = 0.0
    lab_away_path_score: float = 0.0
    lab_scenario_entropy_score: float = 0.0
    lab_draw_path_score: float = 0.0
    lab_mix_home: float = 0.0
    lab_mix_draw: float = 0.0
    lab_mix_away: float = 0.0
    lab_tempo_band: str = ""
    lab_control_band: str = ""
    lab_pressure_band: str = ""
    lab_matchup_profile: str = ""
    lab_basis_hint: str = ""
    lab_state_notes: str = ""
    shape_hold_strength: float = 0.0
    shape_draw_compression: float = 0.0
    shape_swing_instability: float = 0.0
    shape_path_imbalance: float = 0.0
    shape_entropy: float = 0.0
    shape_directionality: float = 0.0


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


def _safe_float(v: object, default: float = 0.0) -> float:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return float(default)
    return float(x)


def _context_rank_order(
    admission_policy: str,
    rank1: str,
    rank2: str,
    rank3: str,
    fallback_primary: str,
    fallback_secondary: str,
) -> Tuple[str, List[str]]:
    policy = str(admission_policy or "").strip().lower()
    r1 = _normalize_symbol_token(rank1)
    r2 = _normalize_symbol_token(rank2)
    r3 = _normalize_symbol_token(rank3)
    fp = _normalize_symbol_token(fallback_primary)
    fs = _normalize_symbol_token(fallback_secondary)
    if policy == "rank1_draw_core":
        base = r1 or fp
        alts = [sym for sym in [r2, r3, fs] if sym and sym != base]
    elif policy == "rank1_draw_watch":
        base = r1 or fp
        alts = [sym for sym in [r2, r3, fs] if sym and sym != base]
    elif policy == "rank2_draw_cut":
        base = r2 or fp
        alts = [sym for sym in [r1, r3, fs] if sym and sym != base]
    elif policy == "rank1_draw_rescue":
        base = r1 or fp
        alts = [sym for sym in [r2, r3, fs] if sym and sym != base]
    elif policy == "rank2_side_escape":
        base = r2 or fp
        alts = [sym for sym in [r3, r1, fs] if sym and sym != base]
    elif policy == "rank1_side_cover":
        base = r1 or fp
        alts = [sym for sym in [r2, r3, fs] if sym and sym != base]
    else:
        base = fp
        alts = [sym for sym in [fs, r1, r2, r3] if sym and sym != base]
    seen: List[str] = []
    for sym in alts:
        if sym not in seen:
            seen.append(sym)
    return base, seen


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
        "match_purchase_type",
        "match_purchase_subtype",
        "purchase_type_confidence",
        "purchase_type_reason",
        "admission_policy",
        "rank1_symbol",
        "rank2_symbol",
        "rank3_symbol",
        "rank1_prob",
        "rank2_prob",
        "rank3_prob",
        "anchor_candidate_flag",
        "draw_core_candidate_flag",
        "draw_core_flag",
        "draw_purchase_tier",
        "primary_pick_symbol",
        "secondary_pick_symbol",
        "risk_level",
        "ticket_guidance",
        "decision_summary",
        "lab_hold_weight",
        "lab_stall_weight",
        "lab_flip_weight",
        "lab_volatility_score",
        "lab_draw_tension_score",
        "lab_dynamic_swing_score",
        "lab_tempo_band",
        "lab_control_band",
        "lab_pressure_band",
        "lab_state_notes",
    ]
    if "match_id" in pred_df.columns and "match_id" in ctx.columns:
        available_ctx_cols = [c for c in ["match_id"] + key_cols if c in ctx.columns]
        ctx = ctx[available_ctx_cols].copy()
    else:
        available_ctx_cols = [c for c in ["league", "home_team", "away_team"] + key_cols if c in ctx.columns]
        ctx = ctx[available_ctx_cols].copy()
    if "match_id" in work.columns and "match_id" in ctx.columns:
        join_cols = ["match_id"]
        overlap_cols = [c for c in ctx.columns if c in work.columns and c not in join_cols]
        if overlap_cols:
            work = work.drop(columns=overlap_cols, errors="ignore")
        merged = work.merge(ctx.drop_duplicates(subset=["match_id"], keep="first"), on="match_id", how="left")
    else:
        join_cols = [c for c in ["league", "home_team", "away_team"] if c in work.columns and c in ctx.columns]
        if not {"home_team", "away_team"}.issubset(join_cols):
            _warn(warnings, "context csv を結合できません（match_id/home_team/away_team不足）")
            return work
        overlap_cols = [c for c in ctx.columns if c in work.columns and c not in join_cols]
        if overlap_cols:
            work = work.drop(columns=overlap_cols, errors="ignore")
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
        context_admission_policy = _safe_text(row.get("admission_policy"), "")
        context_rank1_pick = _safe_text(row.get("rank1_symbol"), "")
        context_rank2_pick = _safe_text(row.get("rank2_symbol"), "")
        context_rank3_pick = _safe_text(row.get("rank3_symbol"), "")
        context_rank1_prob = _safe_float(row.get("rank1_prob"), 0.0)
        context_rank2_prob = _safe_float(row.get("rank2_prob"), 0.0)
        context_rank3_prob = _safe_float(row.get("rank3_prob"), 0.0)
        context_risk_level = _safe_text(row.get("risk_level"), "")
        context_ticket_guidance = _safe_text(row.get("ticket_guidance"), "")
        context_decision_summary = _safe_text(row.get("decision_summary"), "")
        context_match_purchase_type = _safe_text(row.get("match_purchase_type"), "")
        context_match_purchase_subtype = _safe_text(row.get("match_purchase_subtype"), "")
        context_purchase_type_confidence = _safe_text(row.get("purchase_type_confidence"), "")
        context_purchase_type_reason = _safe_text(row.get("purchase_type_reason"), "")
        context_anchor_candidate_flag = bool(str(row.get("anchor_candidate_flag", "")).strip().lower() in {"true", "1"})
        context_draw_core_candidate_flag = bool(str(row.get("draw_core_candidate_flag", "")).strip().lower() in {"true", "1"})
        context_draw_core_flag = bool(str(row.get("draw_core_flag", "")).strip().lower() in {"true", "1"})
        context_draw_purchase_tier = _safe_text(row.get("draw_purchase_tier"), "")
        lab_hold_weight = _safe_float(row.get("lab_hold_weight"), 0.0)
        lab_stall_weight = _safe_float(row.get("lab_stall_weight"), 0.0)
        lab_flip_weight = _safe_float(row.get("lab_flip_weight"), 0.0)
        lab_volatility_score = _safe_float(row.get("lab_volatility_score"), 0.0)
        lab_draw_tension_score = _safe_float(row.get("lab_draw_tension_score"), 0.0)
        lab_dynamic_swing_score = _safe_float(row.get("lab_dynamic_swing_score"), 0.0)
        lab_hold_foundation_score = _safe_float(row.get("lab_hold_foundation_score"), 0.0)
        lab_stall_compactness_score = _safe_float(row.get("lab_stall_compactness_score"), 0.0)
        lab_flip_dislocation_score = _safe_float(row.get("lab_flip_dislocation_score"), 0.0)
        lab_home_path_score = _safe_float(row.get("lab_home_path_score"), 0.0)
        lab_away_path_score = _safe_float(row.get("lab_away_path_score"), 0.0)
        lab_scenario_entropy_score = _safe_float(row.get("lab_scenario_entropy_score"), 0.0)
        lab_draw_path_score = _safe_float(row.get("lab_draw_path_score"), 0.0)
        lab_mix_home = _safe_float(row.get("lab_mix_home"), 0.0)
        lab_mix_draw = _safe_float(row.get("lab_mix_draw"), 0.0)
        lab_mix_away = _safe_float(row.get("lab_mix_away"), 0.0)
        lab_tempo_band = _safe_text(row.get("lab_tempo_band"), "")
        lab_control_band = _safe_text(row.get("lab_control_band"), "")
        lab_pressure_band = _safe_text(row.get("lab_pressure_band"), "")
        lab_matchup_profile = _safe_text(row.get("lab_matchup_profile"), "")
        lab_basis_hint = _safe_text(row.get("lab_basis_hint"), "")
        lab_state_notes = _safe_text(row.get("lab_state_notes"), "")
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
        context_primary_pick = _normalize_symbol_token(row.get("primary_pick_symbol"))
        context_secondary_pick = _normalize_symbol_token(row.get("secondary_pick_symbol"))
        context_admission_policy = _safe_text(row.get("admission_policy"), "")
        context_rank1_pick = _normalize_symbol_token(row.get("rank1_symbol"))
        context_rank2_pick = _normalize_symbol_token(row.get("rank2_symbol"))
        context_rank3_pick = _normalize_symbol_token(row.get("rank3_symbol"))
        context_rank1_prob = _safe_float(row.get("rank1_prob"), 0.0)
        context_rank2_prob = _safe_float(row.get("rank2_prob"), 0.0)
        context_rank3_prob = _safe_float(row.get("rank3_prob"), 0.0)
        predicted_base_pick = _predicted_result_to_symbol(row.get("predicted_result_main"))
        if predicted_base_pick not in SYMBOLS:
            predicted_base_pick = _predicted_result_to_symbol(row.get("predicted_result"))
        if BUYPLAN_USE_PREDICTED_RESULT_BASE and predicted_base_pick in SYMBOLS and context_primary_pick not in SYMBOLS:
            base_pick = predicted_base_pick
            decision_reason = "PREDICTED_RESULT_BASE"
            base_from_predicted = True

        context_draw_tier = _safe_text(row.get("draw_purchase_tier"), "").strip().lower()
        context_policy_base, context_policy_alts = _context_rank_order(
            context_admission_policy,
            context_rank1_pick,
            context_rank2_pick,
            context_rank3_pick,
            context_primary_pick,
            context_secondary_pick,
        )
        if context_draw_tier == "cut" and context_policy_base in {"1", "2"}:
            base_pick = context_policy_base
            decision_reason = "CONTEXT_DRAW_CUT"
            base_from_predicted = False
        elif context_draw_tier == "core" and context_policy_base == "0":
            base_pick = "0"
            decision_reason = "CONTEXT_DRAW_CORE"
            base_from_predicted = False
        elif context_draw_tier == "rescue" and context_policy_base == "0":
            base_pick = "0"
            decision_reason = "CONTEXT_DRAW_RESCUE"
            base_from_predicted = False
        elif context_draw_tier == "side" and context_policy_base in {"1", "2"}:
            base_pick = context_policy_base
            decision_reason = "CONTEXT_SIDE_PRIMARY"
            base_from_predicted = False

        # 候補枝は常に確率順位を母体にしつつ、baseだけ predicted_result で上書き可能
        alt_symbols = [sym for sym in ranked_symbols if sym != base_pick]
        for sym in SYMBOLS:
            if sym != base_pick and sym not in alt_symbols:
                alt_symbols.append(sym)
        for preferred_sym in context_policy_alts:
            if preferred_sym in SYMBOLS and preferred_sym != base_pick:
                alt_symbols = [preferred_sym] + [sym for sym in alt_symbols if sym != preferred_sym]
        if context_draw_tier == "cut" and "0" in alt_symbols and "0" != base_pick:
            alt_symbols = ["0"] + [sym for sym in alt_symbols if sym != "0"]
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
                context_admission_policy=str(context_admission_policy),
                context_rank1_pick=str(context_rank1_pick),
                context_rank2_pick=str(context_rank2_pick),
                context_rank3_pick=str(context_rank3_pick),
                context_rank1_prob=float(context_rank1_prob),
                context_rank2_prob=float(context_rank2_prob),
                context_rank3_prob=float(context_rank3_prob),
                context_risk_level=str(context_risk_level),
                context_ticket_guidance=str(context_ticket_guidance),
                context_decision_summary=str(context_decision_summary),
                context_match_purchase_type=str(context_match_purchase_type),
                context_match_purchase_subtype=str(context_match_purchase_subtype),
                context_purchase_type_confidence=str(context_purchase_type_confidence),
                context_purchase_type_reason=str(context_purchase_type_reason),
                context_anchor_candidate_flag=bool(context_anchor_candidate_flag),
                context_draw_core_candidate_flag=bool(context_draw_core_candidate_flag),
                context_draw_core_flag=bool(context_draw_core_flag),
                context_draw_purchase_tier=str(context_draw_purchase_tier),
                lab_hold_weight=float(lab_hold_weight),
                lab_stall_weight=float(lab_stall_weight),
                lab_flip_weight=float(lab_flip_weight),
                lab_volatility_score=float(lab_volatility_score),
                lab_draw_tension_score=float(lab_draw_tension_score),
                lab_dynamic_swing_score=float(lab_dynamic_swing_score),
                lab_hold_foundation_score=float(lab_hold_foundation_score),
                lab_stall_compactness_score=float(lab_stall_compactness_score),
                lab_flip_dislocation_score=float(lab_flip_dislocation_score),
                lab_home_path_score=float(lab_home_path_score),
                lab_away_path_score=float(lab_away_path_score),
                lab_scenario_entropy_score=float(lab_scenario_entropy_score),
                lab_draw_path_score=float(lab_draw_path_score),
                lab_mix_home=float(lab_mix_home),
                lab_mix_draw=float(lab_mix_draw),
                lab_mix_away=float(lab_mix_away),
                lab_tempo_band=str(lab_tempo_band),
                lab_control_band=str(lab_control_band),
                lab_pressure_band=str(lab_pressure_band),
                lab_matchup_profile=str(lab_matchup_profile),
                lab_basis_hint=str(lab_basis_hint),
                lab_state_notes=str(lab_state_notes),
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
        tier = str(getattr(plan, "context_draw_purchase_tier", "") or "").strip().lower()
        return bool(
            guidance in {"draw_cover", "avoid_main", "draw_core_keep", "draw_rescue_split"}
            or tier in {"core", "rescue"}
            or risk == "draw_watch"
        )

    def _context_keep_main(plan: MatchPlan) -> bool:
        guidance = _context_guidance(plan)
        risk = _context_risk(plan)
        tier = str(getattr(plan, "context_draw_purchase_tier", "") or "").strip().lower()
        return bool(guidance in {"main_only", "draw_core_keep"} or tier == "core" or risk == "fixed")

    def _context_lab_cover(plan: MatchPlan) -> bool:
        return _context_guidance(plan) == "lab_cover"

    def _context_flip_ready(plan: MatchPlan) -> bool:
        guidance = _context_guidance(plan)
        risk = _context_risk(plan)
        tier = str(getattr(plan, "context_draw_purchase_tier", "") or "").strip().lower()
        return bool(
            risk in {"volatile", "caution", "draw_watch"}
            or guidance == "draw_cut_to_side"
            or tier == "cut"
            or _context_draw_prefer(plan)
            or _context_lab_cover(plan)
        )

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
                    if ticket_no == 9:
                        cluster_name = _cluster_label(p)
                        if cluster_name not in {"low_tempo_draw", "hold_core"}:
                            continue
                        if cluster_name == "hold_core" and float(p.p_draw) < 0.36:
                            continue
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
    stats: Dict[str, int | float | str] = {}
    ok_indices = [i for i, p in enumerate(plans) if p.status == "OK"]

    def _clip01(value: float) -> float:
        return float(min(max(value, 0.0), 1.0))

    def _draw_pressure_score(plan: MatchPlan) -> float:
        if str(plan.league).upper() == "J2":
            score = (
                0.42 * float(plan.lab_stall_weight)
                + 0.10 * float(plan.lab_stall_compactness_score)
                + 0.34 * float(plan.lab_draw_tension_score)
                + 0.10 * float(plan.p_draw)
                + 0.04 * float(plan.d_score_total)
            )
        else:
            score = (
                0.32 * float(plan.lab_stall_weight)
                + 0.20 * float(plan.lab_stall_compactness_score)
                + 0.38 * float(plan.lab_draw_tension_score)
                + 0.12 * float(plan.p_draw)
                + 0.08 * float(plan.d_score_total)
            )
            score += 0.04 * float(plan.lab_scenario_entropy_score)
        if bool(plan.lab_low_event):
            score += 0.04
        if str(plan.lab_tempo_band).lower() == "low":
            score += 0.03
        return float(min(max(score, 0.0), 1.0))

    def _swing_score(plan: MatchPlan) -> float:
        if str(plan.league).upper() == "J2":
            score = (
                0.30 * float(plan.lab_flip_weight)
                + 0.14 * float(plan.lab_flip_dislocation_score)
                + 0.24 * float(plan.lab_volatility_score)
                + 0.22 * float(plan.lab_dynamic_swing_score)
                + 0.10 * float(plan.away_value_score)
            )
        else:
            score = (
                0.24 * float(plan.lab_flip_weight)
                + 0.22 * float(plan.lab_flip_dislocation_score)
                + 0.20 * float(plan.lab_volatility_score)
                + 0.18 * float(plan.lab_dynamic_swing_score)
                + 0.10 * float(plan.lab_scenario_entropy_score)
                + 0.06 * float(plan.away_value_score)
            )
        if str(plan.lab_pressure_band).lower() != "neutral":
            score += 0.03
        if str(plan.lab_control_band).lower() != "neutral":
            score += 0.02
        return float(min(max(score, 0.0), 1.0))

    def _hold_lock_score(plan: MatchPlan) -> float:
        if str(plan.league).upper() == "J2":
            score = (
                0.48 * float(plan.lab_hold_weight)
                + 0.10 * float(plan.lab_hold_foundation_score)
                + 0.20 * float(plan.strength_score)
                + 0.12 * float(plan.p_best)
                + 0.10 * float(plan.margin if plan.margin is not None else 0.0)
            )
        else:
            score = (
                0.34 * float(plan.lab_hold_weight)
                + 0.22 * float(plan.lab_hold_foundation_score)
                + 0.14 * float(plan.strength_score)
                + 0.10 * float(plan.p_best)
                + 0.08 * float(plan.margin if plan.margin is not None else 0.0)
                + 0.06 * max(float(plan.lab_home_path_score), float(plan.lab_away_path_score))
            )
        score -= 0.10 * float(plan.lab_volatility_score)
        score -= 0.08 * float(plan.lab_dynamic_swing_score)
        if str(plan.league).upper() != "J2":
            score -= 0.06 * float(plan.lab_scenario_entropy_score)
        return float(min(max(score, 0.0), 1.0))

    def _path_imbalance_score(plan: MatchPlan) -> float:
        return _clip01(abs(float(plan.lab_home_path_score) - float(plan.lab_away_path_score)))

    def _entropy_score(plan: MatchPlan) -> float:
        return _clip01((float(plan.entropy) + float(plan.lab_scenario_entropy_score)) / 2.0)

    def _directionality_score(plan: MatchPlan) -> float:
        best_gap = abs(float(plan.p_home) - float(plan.p_away))
        path_gap = abs(float(plan.lab_home_path_score) - float(plan.lab_away_path_score))
        return _clip01(0.55 * best_gap + 0.45 * path_gap)

    def _assign_shape_scores() -> None:
        for plan in plans:
            if plan.status != "OK":
                plan.shape_hold_strength = 0.0
                plan.shape_draw_compression = 0.0
                plan.shape_swing_instability = 0.0
                plan.shape_path_imbalance = 0.0
                plan.shape_entropy = 0.0
                plan.shape_directionality = 0.0
                continue
            plan.shape_hold_strength = _hold_lock_score(plan)
            plan.shape_draw_compression = _draw_pressure_score(plan)
            plan.shape_swing_instability = _swing_score(plan)
            plan.shape_path_imbalance = _path_imbalance_score(plan)
            plan.shape_entropy = _entropy_score(plan)
            plan.shape_directionality = _directionality_score(plan)

    _assign_shape_scores()

    def _draw_ready(plan: MatchPlan) -> bool:
        return bool(
            float(plan.shape_draw_compression) >= 0.07
            or float(plan.p_draw) >= 0.30
            or str(plan.second) == "0"
            or str(plan.third) == "0"
        )

    def _swing_ready(plan: MatchPlan) -> bool:
        return bool(float(plan.shape_swing_instability) >= 0.09)

    def _hold_locked(plan: MatchPlan) -> bool:
        return bool(float(plan.shape_hold_strength) >= 0.12)

    def _j2_low_path_low_dispersion_flag(plan: MatchPlan) -> int:
        if str(plan.league).upper() != "J2":
            return 0
        profile = str(getattr(plan, "lab_matchup_profile", "")).lower()
        if profile not in {"stall_shape", "balanced"}:
            return 0
        side_path = float(
            min(
                1.0,
                max(
                    0.0,
                    (
                        max(float(plan.lab_home_path_score), float(plan.lab_away_path_score))
                        + (0.55 * float(plan.lab_hold_foundation_score) + 0.45 * abs(float(plan.lab_home_path_score) - float(plan.lab_away_path_score)))
                        + (1.0 - 0.60 * float(plan.lab_draw_path_score))
                    ) / 3.0,
                ),
            )
        )
        dispersion = float(
            min(
                1.0,
                max(
                    0.0,
                    (float(plan.lab_flip_dislocation_score) + float(plan.lab_scenario_entropy_score)) / 2.0,
                ),
            )
        )
        return 1 if (side_path < 0.55 and dispersion < 0.55) else 0

    def _round_topology() -> Dict[str, object]:
        draw_count = sum(1 for i in ok_indices if float(plans[i].shape_draw_compression) >= 0.08)
        swing_count = sum(1 for i in ok_indices if float(plans[i].shape_swing_instability) >= 0.10)
        hold_count = sum(1 for i in ok_indices if float(plans[i].shape_hold_strength) >= 0.12)
        if draw_count >= max(swing_count + 1, 4):
            mode = "draw_heavy"
        elif swing_count >= max(draw_count + 1, 3):
            mode = "swing_heavy"
        else:
            mode = "balanced"
        return {
            "mode": mode,
            "draw_cluster_count": int(draw_count),
            "swing_cluster_count": int(swing_count),
            "hold_cluster_count": int(hold_count),
        }

    def _cluster_label(plan: MatchPlan) -> str:
        profile = str(getattr(plan, "lab_matchup_profile", "")).lower()
        if profile == "low_tempo_draw":
            return "low_tempo_draw"
        if _is_j1_split_side(plan):
            return "hold_core"
        if profile == "stall_shape" and str(plan.league).upper() == "J1":
            if float(plan.shape_draw_compression) >= 0.68 and float(plan.p_draw) >= 0.40:
                return "draw_core"
            return "hold_core"
        if profile in {
            "directional_stall",
            "swing_watch",
            "draw_anchor",
            "stall_side_preserve",
            "mid_stall_side_preserve",
            "home_control_watch",
            "j2_draw_stall_anchor",
            "j2_flat_draw_trap",
            "j2_home_stall_preserve",
            "j1_pressure_draw_watch",
            "j1_floor_recovery_watch",
            "j2_flip_conflict_watch",
            "j2_home_pressure_stall_preserve",
            "j2_low_dyn_home_stall_preserve",
            "j2_neutral_flip_draw_watch",
            "j2_tense_stall_watch",
        }:
            return "hold_core"
        draw_score = float(plan.shape_draw_compression)
        swing_score = float(plan.shape_swing_instability)
        hold_score = float(plan.shape_hold_strength)
        if draw_score >= 0.08 and swing_score >= 0.10:
            return "overlap"
        if hold_score >= 0.12 and draw_score < 0.08 and swing_score < 0.09:
            return "hold_core"
        if draw_score >= 0.08:
            return "draw_core"
        if swing_score >= 0.10:
            return "swing_core"
        return "neutral"

    def _is_protected_profile(plan: MatchPlan) -> bool:
        profile = str(getattr(plan, "lab_matchup_profile", "")).lower()
        if _is_j1_split_side(plan):
            return True
        return profile in {
            "swing_watch",
            "directional_stall",
            "draw_anchor",
            "stall_side_preserve",
            "mid_stall_side_preserve",
            "home_control_watch",
            "j2_draw_stall_anchor",
            "j2_flat_draw_trap",
            "j2_home_stall_preserve",
            "j1_pressure_draw_watch",
            "j1_floor_recovery_watch",
            "j2_flip_conflict_watch",
            "j2_home_pressure_stall_preserve",
            "j2_low_dyn_home_stall_preserve",
            "j2_neutral_flip_draw_watch",
            "j2_tense_stall_watch",
        } or (profile == "stall_shape" and str(plan.league).upper() == "J1")

    def _overlap_subtype(plan: MatchPlan) -> str:
        if _cluster_label(plan) != "overlap":
            return "non_overlap"
        primary = str(plan.context_primary_pick or plan.buyplan_choice or plan.best)
        control = str(plan.lab_control_band).lower()
        pressure = str(plan.lab_pressure_band).lower()
        primary_prob = float(plan.p_best)
        if primary == "2" and (control == "home" or pressure == "home") and primary_prob <= 0.38:
            return "counterflow_away"
        if primary == "1" and (control == "away" or pressure == "away") and primary_prob <= 0.38:
            return "counterflow_home"
        if (
            str(plan.lab_tempo_band).lower() == "low"
            and float(plan.shape_draw_compression) >= 0.74
            and float(plan.shape_swing_instability) <= 0.34
            and float(plan.lab_flip_weight) <= 0.30
        ):
            return "low_tempo_draw"
        return "general_overlap"

    def _basis_hint(plan: MatchPlan) -> str:
        hint = str(getattr(plan, "lab_basis_hint", "")).strip().lower()
        return hint or "basis_balanced"

    def _is_j1_split_side(plan: MatchPlan) -> bool:
        return str(plan.league).upper() == "J1" and _basis_hint(plan) == "split_side"

    def _best_flip_symbol(plan: MatchPlan, prefer_away: bool = False) -> str:
        ranked = []
        pressure_band = str(plan.lab_pressure_band).lower()
        control_band = str(plan.lab_control_band).lower()
        if prefer_away or pressure_band == "away" or control_band == "away":
            ranked.append("2")
        elif pressure_band == "home" or control_band == "home":
            ranked.append("1")
        ranked.extend([str(plan.second), str(plan.third), str(plan.best), str(plan.base_pick)])
        for sym in ranked:
            if sym in {"1", "0", "2"} and sym != str(plan.base_pick):
                if sym == "0" and not _draw_ready(plan):
                    continue
                return sym
        return str(plan.base_pick)

    def _rank_indices(label: str) -> List[int]:
        if label == "stall":
            order = sorted(
                ok_indices,
                key=lambda i: (
                    _j2_low_path_low_dispersion_flag(plans[i]),
                    -float(plans[i].shape_draw_compression),
                    -float(plans[i].shape_entropy),
                    -float(plans[i].p_draw),
                    -float(plans[i].lab_stall_weight),
                    float(plans[i].shape_directionality),
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    i,
                ),
            )
        elif label == "draw_tension":
            order = sorted(
                ok_indices,
                key=lambda i: (
                    _j2_low_path_low_dispersion_flag(plans[i]),
                    -float(plans[i].shape_draw_compression),
                    -float(plans[i].shape_entropy),
                    -float(plans[i].p_draw),
                    -float(plans[i].shape_path_imbalance),
                    -float(plans[i].lab_stall_weight),
                    float(plans[i].shape_directionality),
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    i,
                ),
            )
        elif label == "flip":
            order = sorted(
                ok_indices,
                key=lambda i: (
                    _j2_low_path_low_dispersion_flag(plans[i]),
                    -float(plans[i].shape_swing_instability),
                    -float(plans[i].shape_path_imbalance),
                    -float(plans[i].lab_flip_weight),
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    i,
                ),
            )
        elif label == "volatility":
            order = sorted(
                ok_indices,
                key=lambda i: (
                    -float(plans[i].shape_swing_instability),
                    -float(plans[i].shape_entropy),
                    -float(plans[i].lab_volatility_score),
                    -float(plans[i].lab_flip_weight),
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    i,
                ),
            )
        elif label == "hold":
            order = sorted(
                ok_indices,
                key=lambda i: (
                    -float(plans[i].shape_hold_strength),
                    -float(plans[i].shape_directionality),
                    -float(plans[i].lab_hold_weight),
                    float(plans[i].margin if plans[i].margin is not None else 999.0),
                    i,
                ),
            )
        elif label == "away_flip":
            order = sorted(
                ok_indices,
                key=lambda i: (
                    _j2_low_path_low_dispersion_flag(plans[i]),
                    0 if float(plans[i].p_away) >= float(plans[i].p_home) else 1,
                    -float(plans[i].shape_swing_instability),
                    -float(plans[i].shape_path_imbalance),
                    -float(plans[i].lab_flip_weight),
                    -float(plans[i].p_away),
                    i,
                ),
            )
        elif label == "j2_draw":
            order = sorted(
                ok_indices,
                key=lambda i: (
                    0 if str(plans[i].league).upper() == "J2" else 1,
                    _j2_low_path_low_dispersion_flag(plans[i]),
                    -float(plans[i].shape_draw_compression),
                    -float(plans[i].shape_entropy),
                    -float(plans[i].lab_stall_weight),
                    -float(plans[i].p_draw),
                    i,
                ),
            )
        else:
            order = list(ok_indices)
        return order

    top_map = {
        "stall": _rank_indices("stall")[:5],
        "draw_tension": _rank_indices("draw_tension")[:5],
        "flip": _rank_indices("flip")[:5],
        "volatility": _rank_indices("volatility")[:5],
        "dynamic_swing": sorted(
            ok_indices,
            key=lambda i: (
                -float(plans[i].shape_swing_instability),
                -float(plans[i].shape_path_imbalance),
                -float(plans[i].shape_entropy),
                i,
            ),
        )[:5],
    }
    for key, idxs in top_map.items():
        parts = []
        for i in idxs:
            p = plans[i]
            if key == "stall":
                score = float(p.shape_draw_compression)
            elif key == "draw_tension":
                score = float(p.shape_draw_compression)
            elif key == "flip":
                score = float(p.shape_swing_instability)
            elif key == "dynamic_swing":
                score = float(p.shape_swing_instability)
            else:
                score = float(p.shape_swing_instability)
            parts.append(f"M{p.match_no:02d}:{p.home_team}-{p.away_team}:{score:.3f}")
        label_map = {
            "stall": "shape_draw_compression",
            "draw_tension": "shape_draw_compression_tension",
            "flip": "shape_swing_instability",
            "volatility": "shape_swing_instability_volatility",
            "dynamic_swing": "shape_swing_instability_dynamic",
        }
        _warn(warnings, f"{label_map.get(key, key)}上位: {' / '.join(parts) if parts else '-'}")

    topology = _round_topology()
    cluster_map: Dict[str, List[int]] = {
        k: [] for k in ["draw_core", "low_tempo_draw", "swing_core", "overlap", "hold_core", "neutral"]
    }
    for i in ok_indices:
        cluster_map[_cluster_label(plans[i])].append(i)
    stats["topology_mode"] = str(topology["mode"])
    stats["topology_draw_cluster_count"] = int(topology["draw_cluster_count"])
    stats["topology_swing_cluster_count"] = int(topology["swing_cluster_count"])
    stats["topology_hold_cluster_count"] = int(topology["hold_cluster_count"])
    for label, idxs in cluster_map.items():
        stats[f"cluster_{label}_count"] = int(len(idxs))
    _warn(
        warnings,
        "round_topology="
        f"{topology['mode']} draw_cluster={int(topology['draw_cluster_count'])} "
        f"swing_cluster={int(topology['swing_cluster_count'])} "
        f"hold_cluster={int(topology['hold_cluster_count'])}",
    )
    _warn(
        warnings,
        "round_clusters="
        f"draw_core={len(cluster_map['draw_core'])} "
        f"low_tempo_draw={len(cluster_map['low_tempo_draw'])} "
        f"overlap={len(cluster_map['overlap'])} "
        f"swing_core={len(cluster_map['swing_core'])} "
        f"hold_core={len(cluster_map['hold_core'])} "
        f"neutral={len(cluster_map['neutral'])}",
    )

    j2_ok_indices = [i for i in ok_indices if str(plans[i].league).upper() == "J2"]
    j2_draw_ready_indices = [
        i
        for i in j2_ok_indices
        if _draw_ready(plans[i]) and _cluster_label(plans[i]) in {"draw_core", "neutral", "low_tempo_draw"}
    ]
    stats["j2_match_count"] = int(len(j2_ok_indices))
    stats["j2_draw_ready_count"] = int(len(j2_draw_ready_indices))
    if len(j2_ok_indices) == 0 or len(j2_draw_ready_indices) <= 1:
        ticket06_mode = "skip"
    else:
        ticket06_mode = "active"
    stats["ticket_06_mode"] = ticket06_mode
    _warn(
        warnings,
        f"round_j2_context=matches={len(j2_ok_indices)} draw_ready={len(j2_draw_ready_indices)} mode06={ticket06_mode}",
    )
    ticket06_enabled = ticket06_mode != "skip"

    def _exclude_ticket06(indices: List[int]) -> List[int]:
        if ticket06_enabled:
            return list(indices)
        return [idx for idx in indices if idx != 5]

    portfolio_target_map: Dict[str, Dict[str, Dict[str, int]]] = {
        "draw_heavy": {
            "02": {"max_draw": 3, "max_flip": 0},
            "03": {"max_draw": 3, "max_flip": 0},
            "04": {"max_draw": 0, "max_flip": 2},
            "05": {"max_draw": 0, "max_flip": 3},
            "06": {
                "max_draw": 2,
                "max_flip": 0,
                "allowed_draw_clusters": ["draw_core", "low_tempo_draw"],
                "cluster_caps": {"draw_core": 1, "overlap": 0},
            },
            "07": {"max_draw": 0, "max_flip": 3},
            "08": {"max_draw": 2, "max_flip": 0},
            "09": {"max_draw": 1, "max_flip": 0},
            "10": {"max_draw": 1, "max_flip": 3},
        },
        "swing_heavy": {
            "02": {"max_draw": 0, "max_flip": 5},
            "03": {"max_draw": 0, "max_flip": 4},
            "04": {"max_draw": 0, "max_flip": 2},
            "05": {"max_draw": 2, "max_flip": 1},
            "06": {
                "max_draw": 0,
                "max_flip": 0,
                "allowed_draw_clusters": ["neutral"],
                "cluster_caps": {"neutral": 1, "overlap": 0},
            },
            "07": {"max_draw": 0, "max_flip": 5},
            "08": {"max_draw": 0, "max_flip": 0, "allowed_draw_clusters": ["neutral", "low_tempo_draw"]},
            "09": {"max_draw": 0, "max_flip": 0, "allowed_draw_clusters": ["low_tempo_draw"]},
            "10": {"max_draw": 0, "max_flip": 4},
        },
        "balanced": {
            "02": {
                "max_draw": 1,
                "max_flip": 0,
                "allowed_draw_clusters": ["neutral"],
                "cluster_caps": {"neutral": 2, "draw_core": 0, "overlap": 0},
            },
            "03": {"max_draw": 0, "max_flip": 5},
            "04": {"max_draw": 0, "max_flip": 2},
            "05": {
                "max_draw": 3,
                "max_flip": 1,
                "allowed_draw_clusters": ["draw_core"],
                "cluster_caps": {"draw_core": 2, "neutral": 0, "overlap": 0},
            },
            "06": {
                "max_draw": 1,
                "max_flip": 0,
                "allowed_draw_clusters": ["draw_core", "neutral"],
                "cluster_caps": {"neutral": 1, "draw_core": 1, "overlap": 0},
            },
            "07": {"max_draw": 0, "max_flip": 1},
            "08": {"max_draw": 0, "max_flip": 0},
            "09": {"max_draw": 0, "max_flip": 0},
            "10": {"max_draw": 0, "max_flip": 1},
        },
    }
    portfolio_target = portfolio_target_map.get(str(topology["mode"]), portfolio_target_map["balanced"])
    stats["portfolio_target_mode"] = str(topology["mode"])
    stats["portfolio_target_summary"] = " / ".join(
        f"{sid}=D{cfg['max_draw']}:F{cfg['max_flip']}" for sid, cfg in sorted(portfolio_target.items())
    )
    stats["portfolio_target_cluster_summary"] = " / ".join(
        f"{sid}="
        f"{','.join(portfolio_target[sid].get('allowed_draw_clusters', [])) or '-'}"
        for sid in sorted(portfolio_target)
        if portfolio_target[sid].get("allowed_draw_clusters")
    )
    _warn(warnings, f"portfolio_target={stats['portfolio_target_summary']}")
    if stats["portfolio_target_cluster_summary"]:
        _warn(warnings, f"portfolio_target_clusters={stats['portfolio_target_cluster_summary']}")

    portfolio_objective_config_map: Dict[str, Dict[str, object]] = {
        "draw_heavy": {
            "weights": {
                "topology_fit": 1.00,
                "diversification_min": 0.55,
                "diversification_avg": 0.18,
                "cluster_penalty": -0.22,
                "correlated_penalty": -0.26,
                "coverage_gain": 0.34,
                "saturation_penalty": -0.30,
                "ticket_budget_penalty": -0.24,
                "overlap_bonus": 0.12,
                "rescue_uniqueness": 0.18,
            },
            "cluster_role_targets": {
                "draw_core": {"draw": 2},
                "low_tempo_draw": {"draw": 2},
                "neutral": {"draw": 1},
                "overlap": {"swing": 1, "draw": 1},
                "hold_core": {"draw": 1},
            },
            "role_alt_caps": {"draw": 2, "swing": 1, "other": 1, "base": 0},
        },
        "swing_heavy": {
            "weights": {
                "topology_fit": 1.00,
                "diversification_min": 0.60,
                "diversification_avg": 0.20,
                "cluster_penalty": -0.20,
                "correlated_penalty": -0.28,
                "coverage_gain": 0.30,
                "saturation_penalty": -0.34,
                "ticket_budget_penalty": -0.20,
                "overlap_bonus": 0.18,
                "rescue_uniqueness": 0.20,
            },
            "cluster_role_targets": {
                "draw_core": {"draw": 1},
                "low_tempo_draw": {"draw": 1},
                "neutral": {"draw": 1},
                "overlap": {"swing": 2},
                "hold_core": {"swing": 1},
            },
            "role_alt_caps": {"draw": 1, "swing": 2, "other": 1, "base": 0},
        },
        "balanced": {
            "weights": {
                "topology_fit": 1.00,
                "diversification_min": 0.55,
                "diversification_avg": 0.18,
                "cluster_penalty": -0.22,
                "correlated_penalty": -0.26,
                "coverage_gain": 0.32,
                "saturation_penalty": -0.32,
                "ticket_budget_penalty": -0.28,
                "overlap_bonus": 0.15,
                "rescue_uniqueness": 0.18,
            },
            "cluster_role_targets": {
                "draw_core": {"draw": 2},
                "low_tempo_draw": {"draw": 1},
                "neutral": {"draw": 1},
                "overlap": {"swing": 2, "draw": 1},
                "hold_core": {"draw": 1},
            },
            "role_alt_caps": {"draw": 1, "swing": 1, "other": 1, "base": 0},
        },
    }
    portfolio_objective_config = portfolio_objective_config_map.get(
        str(topology["mode"]), portfolio_objective_config_map["balanced"]
    )
    portfolio_objective_weights = dict(portfolio_objective_config.get("weights", {}))
    portfolio_cluster_role_targets = dict(portfolio_objective_config.get("cluster_role_targets", {}))
    portfolio_role_alt_caps = dict(portfolio_objective_config.get("role_alt_caps", {}))
    stats["portfolio_objective_weights"] = " / ".join(
        f"{k}={float(v):.2f}" for k, v in sorted(portfolio_objective_weights.items())
    )
    stats["portfolio_objective_cluster_targets"] = " / ".join(
        f"{cluster}:{','.join(f'{role}={int(count)}' for role, count in sorted(role_map.items()))}"
        for cluster, role_map in sorted(portfolio_cluster_role_targets.items())
    )
    _warn(warnings, f"portfolio_objective_weights={stats['portfolio_objective_weights']}")
    _warn(warnings, f"portfolio_objective_cluster_targets={stats['portfolio_objective_cluster_targets']}")

    def _target_draw(scenario_id: str, default: int) -> int:
        return int(portfolio_target.get(str(scenario_id), {}).get("max_draw", default))

    def _target_flip(scenario_id: str, default: int) -> int:
        return int(portfolio_target.get(str(scenario_id), {}).get("max_flip", default))

    def _target_draw_scaled(scenario_id: str, default: int) -> int:
        value = int(_target_draw(scenario_id, default))
        if str(scenario_id) == "06":
            if ticket06_mode == "skip":
                value = 0
            elif str(topology["mode"]) == "swing_heavy":
                value = 1
            elif len(j2_draw_ready_indices) == 0 or len(j2_ok_indices) <= 2:
                value = 1
            elif len(j2_draw_ready_indices) == 1:
                value = 1 if len(j2_ok_indices) <= 3 else 2
            else:
                value = max(2, value)
                if len(j2_ok_indices) >= 5 and str(topology["mode"]) in {"draw_heavy", "balanced"}:
                    value = 3
            value = max(0, min(3, value))
        return int(value)

    def _target_cluster_caps(scenario_id: str, default: Optional[Dict[str, int]] = None) -> Optional[Dict[str, int]]:
        merged: Dict[str, int] = dict(default or {})
        override = portfolio_target.get(str(scenario_id), {}).get("cluster_caps")
        if isinstance(override, dict):
            for key, value in override.items():
                try:
                    merged[str(key)] = int(value)
                except Exception:
                    continue
        return merged or None

    def _target_allowed_clusters(
        scenario_id: str,
        key: str,
        default: Optional[List[str]] = None,
    ) -> Optional[set[str]]:
        if str(scenario_id) == "06" and key == "allowed_draw_clusters":
            if ticket06_mode == "skip":
                return set()
            if len(j2_draw_ready_indices) == 0:
                return {"neutral"}
            if len(j2_draw_ready_indices) == 1:
                return {"neutral", "draw_core"}
            return {"draw_core", "low_tempo_draw"}
        value = portfolio_target.get(str(scenario_id), {}).get(key, default)
        if not value:
            return None
        return {str(item) for item in value}

    base_ticket = [str(p.base_pick) if p.status == "OK" else "1" for p in plans]

    def _draw_admission_score(plan: MatchPlan) -> float:
        return min(
            1.0,
            0.64 * float(plan.shape_draw_compression)
            + 0.20 * float(plan.shape_entropy)
            + 0.16 * float(plan.p_draw),
        )

    def _hold_admission_score(plan: MatchPlan) -> float:
        return min(
            1.0,
            0.62 * float(plan.shape_hold_strength)
            + 0.24 * float(plan.shape_directionality)
            + 0.14 * float(max(plan.p_home, plan.p_away)),
        )

    def _away_admission_score(plan: MatchPlan) -> float:
        return min(
            1.0,
            0.44 * float(plan.shape_swing_instability)
            + 0.30 * float(plan.shape_path_imbalance)
            + 0.14 * float(plan.shape_directionality)
            + 0.12 * float(plan.p_away),
        )

    def _topology_asymmetry_score(plan: MatchPlan) -> float:
        home_path = float(plan.lab_home_path_score)
        away_path = float(plan.lab_away_path_score)
        side_gap = abs(home_path - away_path)
        directional_gap = abs(float(plan.p_home) - float(plan.p_away))
        return min(
            1.0,
            0.38 * side_gap
            + 0.24 * float(plan.shape_directionality)
            + 0.20 * float(plan.shape_path_imbalance)
            + 0.18 * directional_gap,
        )

    def _admission_escape_score(plan: MatchPlan, target_symbol: str) -> float:
        target_symbol = str(target_symbol)
        asym = float(_topology_asymmetry_score(plan))
        draw_adm = float(_draw_admission_score(plan))
        hold_adm = float(_hold_admission_score(plan))
        away_adm = float(_away_admission_score(plan))
        home_side_score = min(
            1.0,
            0.42 * float(plan.shape_swing_instability)
            + 0.24 * float(plan.shape_path_imbalance)
            + 0.18 * float(plan.shape_directionality)
            + 0.16 * float(plan.p_home),
        )
        if target_symbol == "0":
            return min(
                1.0,
                0.58 * draw_adm
                + 0.22 * (1.0 - asym)
                + 0.20 * float(plan.lab_draw_tension_score),
            )
        if target_symbol == "2":
            return min(
                1.0,
                0.52 * away_adm
                + 0.26 * asym
                + 0.12 * float(plan.lab_away_path_score)
                + 0.10 * (1.0 - draw_adm),
            )
        if target_symbol == "1":
            return min(
                1.0,
                0.46 * home_side_score
                + 0.24 * asym
                + 0.18 * float(plan.lab_home_path_score)
                + 0.12 * (1.0 - hold_adm),
            )
        return 0.0

    def _make_ticket(
        scenario_id: str,
        name: str,
        draw_indices: List[int],
        flip_indices: List[int],
        *,
        max_draw_changes: int = 0,
        max_flip_changes: int = 0,
        prefer_away: bool = False,
        j2_only: bool = False,
        cluster_caps: Optional[Dict[str, int]] = None,
        allow_low_tempo_draw: bool = False,
        allowed_draw_clusters: Optional[set[str]] = None,
        allowed_flip_clusters: Optional[set[str]] = None,
    ) -> Tuple[List[str], str]:
        ticket = list(base_ticket)
        logs: List[str] = []
        basis_hint_counts: Dict[str, int] = {}
        draw_done = 0
        flip_done = 0
        cluster_change_counts: Dict[str, int] = {}
        overlap_subtype_counts: Dict[str, int] = {}
        draw_mix_gate = {
            "02": 0.16,
            "05": 0.15,
            "06": 0.17,
            "08": 0.18,
            "09": 0.19,
        }.get(str(scenario_id), 0.0)
        hold_mix_gate = 0.12 if str(scenario_id) == "04" else 0.0
        away_mix_gate = {
            "07": 0.11,
            "10": 0.10,
        }.get(str(scenario_id), 0.0)
        for i in draw_indices:
            p = plans[i]
            league_upper = str(p.league).upper()
            if p.status != "OK":
                continue
            if _is_protected_profile(p) and str(scenario_id) != "04":
                continue
            if j2_only and str(p.league).upper() != "J2":
                continue
            if draw_done >= max_draw_changes:
                break
            cluster_label = _cluster_label(p)
            overlap_subtype = _overlap_subtype(p)
            escape_draw_ready = bool(
                cluster_label == "draw_core"
                and float(_topology_asymmetry_score(p)) <= 0.22
                and float(_admission_escape_score(p, "0")) >= 0.70
            )
            if allowed_draw_clusters and cluster_label not in allowed_draw_clusters:
                if not (str(scenario_id) == "02" and escape_draw_ready):
                    continue
            if cluster_caps and cluster_label in cluster_caps and int(cluster_change_counts.get(cluster_label, 0)) >= int(cluster_caps[cluster_label]):
                continue
            if cluster_label == "low_tempo_draw" and not allow_low_tempo_draw:
                continue
            if overlap_subtype in {"counterflow_away", "counterflow_home"}:
                continue
            if overlap_subtype == "low_tempo_draw":
                continue
            if cluster_label == "low_tempo_draw":
                if not (
                    str(p.best) in {"1", "2"}
                    and str(p.second) == "0"
                    and float(p.shape_draw_compression) >= 0.70
                    and float(p.shape_entropy) >= 0.40
                    and float(p.p_draw) >= 0.30
                    and float(p.shape_directionality) <= 0.32
                ):
                    continue
            if str(scenario_id) == "08":
                if cluster_label not in {"low_tempo_draw", "neutral", "draw_core"}:
                    continue
                if not (
                    float(p.shape_draw_compression) >= 0.080
                    and float(p.shape_entropy) >= 0.95
                    and float(p.p_draw) >= 0.30
                    and float(p.shape_directionality) <= 0.34
                ):
                    continue
            if str(scenario_id) == "09":
                is_low_tempo = cluster_label == "low_tempo_draw"
                is_j2_cover = league_upper == "J2" and float(p.shape_draw_compression) >= 0.082 and float(p.p_draw) >= 0.31
                fallback_cover = (
                    cluster_label == "neutral"
                    and float(p.shape_draw_compression) >= 0.086
                    and float(p.shape_entropy) >= 1.02
                    and float(p.p_draw) >= 0.33
                    and float(p.shape_directionality) <= 0.22
                )
                if not (is_low_tempo or is_j2_cover or fallback_cover or escape_draw_ready):
                    continue
                if not (
                    float(p.shape_entropy) >= 0.96
                    and float(p.shape_directionality) <= 0.30
                ):
                    continue
            if str(scenario_id) == "06":
                if cluster_label not in {"draw_core", "neutral", "low_tempo_draw"}:
                    continue
                if not (
                    league_upper == "J2"
                    and float(p.shape_draw_compression) >= 0.072
                    and float(p.shape_entropy) >= 0.98
                    and float(p.p_draw) >= 0.30
                    and float(p.shape_directionality) <= 0.32
                ):
                    continue
            if _hold_locked(p):
                continue
            if not _draw_ready(p):
                continue
            if draw_mix_gate > 0.0 and (league_upper == "J1" or str(scenario_id) == "06"):
                if float(_draw_admission_score(p)) < draw_mix_gate:
                    continue
            if str(scenario_id) in {"02", "05", "08", "09"}:
                if float(_admission_escape_score(p, "0")) < 0.34:
                    continue
            if ticket[i] == "0":
                continue
            target_symbol = "0"
            if cluster_label == "low_tempo_draw":
                target_symbol = str(p.second)
            ticket[i] = target_symbol
            draw_done += 1
            cluster_change_counts[cluster_label] = int(cluster_change_counts.get(cluster_label, 0)) + 1
            if cluster_label == "overlap":
                overlap_subtype_counts[overlap_subtype] = int(overlap_subtype_counts.get(overlap_subtype, 0)) + 1
            basis_hint_counts[_basis_hint(p)] = int(basis_hint_counts.get(_basis_hint(p), 0)) + 1
            logs.append(f"M{p.match_no:02d}:{base_ticket[i]}->{target_symbol}")
        for i in flip_indices:
            p = plans[i]
            league_upper = str(p.league).upper()
            if p.status != "OK":
                continue
            if _is_protected_profile(p) and str(scenario_id) != "04":
                continue
            if j2_only and str(p.league).upper() != "J2":
                continue
            if flip_done >= max_flip_changes:
                break
            cluster_label = _cluster_label(p)
            overlap_subtype = _overlap_subtype(p)
            if allowed_flip_clusters and cluster_label not in allowed_flip_clusters:
                continue
            if cluster_caps and cluster_label in cluster_caps and int(cluster_change_counts.get(cluster_label, 0)) >= int(cluster_caps[cluster_label]):
                continue
            if cluster_label == "low_tempo_draw":
                continue
            if overlap_subtype in {"counterflow_away", "counterflow_home"}:
                continue
            if _hold_locked(p) and not prefer_away and str(scenario_id) not in {"04", "05"}:
                continue
            if not _swing_ready(p) and not prefer_away and str(scenario_id) not in {"04", "05"}:
                continue
            if hold_mix_gate > 0.0 and league_upper == "J1":
                if float(_hold_admission_score(p)) < hold_mix_gate:
                    continue
            if str(scenario_id) == "05" and str(ticket[i]) == "0":
                alt = str(p.second) if str(p.second) in {"1", "2"} else _best_flip_symbol(p, prefer_away=prefer_away)
            else:
                alt = _best_flip_symbol(p, prefer_away=prefer_away)
            if away_mix_gate > 0.0 and league_upper == "J1" and alt == "2":
                if float(_away_admission_score(p)) < away_mix_gate:
                    continue
            if str(scenario_id) in {"03", "05", "07", "10"}:
                if float(_admission_escape_score(p, alt)) < 0.31:
                    continue
            if overlap_subtype == "low_tempo_draw" and alt == "0":
                continue
            if alt == ticket[i]:
                continue
            if alt == "0" and not _draw_ready(p):
                continue
            ticket[i] = alt
            flip_done += 1
            cluster_change_counts[cluster_label] = int(cluster_change_counts.get(cluster_label, 0)) + 1
            if cluster_label == "overlap":
                overlap_subtype_counts[overlap_subtype] = int(overlap_subtype_counts.get(overlap_subtype, 0)) + 1
            basis_hint_counts[_basis_hint(p)] = int(basis_hint_counts.get(_basis_hint(p), 0)) + 1
            logs.append(f"M{p.match_no:02d}:{base_ticket[i]}->{alt}")
        change_count = int(draw_done + flip_done)
        stats[f"ticket_{scenario_id}_shape_change_count"] = change_count
        stats[f"ticket_{scenario_id}_lab_change_count"] = change_count
        stats[f"ticket_{scenario_id}_draw_count"] = int(_symbol_count(ticket, "0"))
        stats[f"ticket_{scenario_id}_diff_count"] = int(sum(1 for a, b in zip(ticket, base_ticket) if a != b))
        cluster_summary_parts: List[str] = []
        for cluster_name in ["draw_core", "low_tempo_draw", "overlap", "swing_core", "neutral", "hold_core"]:
            cluster_count = int(cluster_change_counts.get(cluster_name, 0))
            stats[f"ticket_{scenario_id}_cluster_{cluster_name}_changes"] = cluster_count
            cluster_summary_parts.append(f"{cluster_name}={cluster_count}")
        for hint in ["side_strong", "draw_compressed", "flat_draw_trap", "split_side", "basis_balanced"]:
            stats[f"ticket_{scenario_id}_basis_{hint}_changes"] = int(basis_hint_counts.get(hint, 0))
        overlap_summary = " ".join(
            f"{name}={int(overlap_subtype_counts.get(name, 0))}"
            for name in ["counterflow_away", "counterflow_home", "low_tempo_draw", "general_overlap"]
        )
        basis_summary = " ".join(
            f"{hint}={int(basis_hint_counts.get(hint, 0))}"
            for hint in ["side_strong", "draw_compressed", "flat_draw_trap", "split_side", "basis_balanced"]
        )
        print(
            f"[BUYPLAN_SCENARIO_TICKET] ticket={scenario_id} name={name} "
            f"shape_changes={change_count} draw_count={int(_symbol_count(ticket, '0'))} "
            f"diff_count={int(sum(1 for a, b in zip(ticket, base_ticket) if a != b))} "
            f"clusters={' '.join(cluster_summary_parts)} "
            f"overlap_subtypes={overlap_summary} "
            f"basis_hints={basis_summary} "
            f"changes={'; '.join(logs) if logs else 'none'}"
        )
        return ticket, ("base" if not logs else "; ".join(logs))

    tickets: List[List[str]] = []
    flip_descs: List[str] = []
    scenario_defs: List[ScenarioDef] = [ScenarioDef("01", "Main", "本線")]

    tickets.append(list(base_ticket))
    flip_descs.append("main_base")
    for cluster_name in ["draw_core", "low_tempo_draw", "overlap", "swing_core", "neutral", "hold_core"]:
        stats[f"ticket_01_cluster_{cluster_name}_changes"] = 0
    for hint in ["side_strong", "draw_compressed", "flat_draw_trap", "split_side", "basis_balanced"]:
        stats[f"ticket_01_basis_{hint}_changes"] = 0
    stats["ticket_01_shape_change_count"] = 0
    stats["ticket_01_lab_change_count"] = 0
    stats["ticket_01_diff_count"] = 0
    stats["ticket_01_draw_count"] = int(_symbol_count(base_ticket, "0"))

    close_draw_indices = sorted(
        ok_indices,
        key=lambda i: (
            -float(plans[i].shape_draw_compression),
            -float(plans[i].shape_entropy),
            -float(plans[i].p_draw),
            -float(plans[i].shape_directionality),
            float(abs(plans[i].p_home - plans[i].p_away)),
            -float(plans[i].shape_path_imbalance),
            i,
        ),
    )
    def _merge_unique(*parts: List[int]) -> List[int]:
        out: List[int] = []
        for seq in parts:
            for idx in seq:
                if idx not in out:
                    out.append(idx)
        return out

    hold_tail = list(reversed(_rank_indices("hold")))
    draw_cluster_indices = cluster_map["draw_core"] + [i for i in _rank_indices("draw_tension") if i not in cluster_map["draw_core"] and i not in cluster_map["overlap"]]
    swing_cluster_indices = cluster_map["swing_core"] + [i for i in _rank_indices("flip") if i not in cluster_map["swing_core"] and i not in cluster_map["overlap"]]
    hold_cluster_indices = cluster_map["hold_core"] + [i for i in hold_tail if i not in cluster_map["hold_core"]]
    neutral_draw_indices = sorted(
        cluster_map["neutral"],
        key=lambda i: (
            -float(plans[i].shape_draw_compression),
            -float(plans[i].shape_entropy),
            -float(plans[i].p_draw),
            -float(plans[i].shape_directionality),
            float(abs(plans[i].p_home - plans[i].p_away)),
            -float(plans[i].shape_path_imbalance),
            i,
        ),
    )
    neutral_swing_indices = sorted(
        cluster_map["neutral"],
        key=lambda i: (
            -float(plans[i].shape_swing_instability),
            -float(plans[i].shape_path_imbalance),
            -float(plans[i].shape_entropy),
            i,
        ),
    )
    overlap_swing_first = sorted(
        cluster_map["overlap"],
        key=lambda i: (
            -float(plans[i].shape_swing_instability),
            -float(plans[i].shape_draw_compression),
            -float(plans[i].shape_entropy),
            i,
        ),
    )
    overlap_counterflow_indices = [
        i for i in overlap_swing_first if _overlap_subtype(plans[i]) in {"counterflow_away", "counterflow_home"}
    ]
    overlap_drawlike_indices = [i for i in overlap_swing_first if _overlap_subtype(plans[i]) == "low_tempo_draw"]
    overlap_general_indices = [i for i in overlap_swing_first if _overlap_subtype(plans[i]) == "general_overlap"]
    overlap_guarded_indices = [
        i
        for i in overlap_general_indices
        if (
            float(plans[i].shape_swing_instability) >= 0.40
            and float(plans[i].shape_path_imbalance) >= 0.20
            and max(float(plans[i].shape_directionality), float(plans[i].shape_entropy)) >= 0.24
            and (
                str(plans[i].lab_pressure_band).lower() != "neutral"
                or str(plans[i].lab_control_band).lower() != "neutral"
            )
        )
    ]
    overlap_guarded_match_nos = {int(plans[i].match_no) for i in overlap_guarded_indices}
    stats["cluster_overlap_counterflow_count"] = int(len(overlap_counterflow_indices))
    stats["cluster_overlap_drawlike_count"] = int(len(overlap_drawlike_indices))
    stats["cluster_overlap_general_count"] = int(len(overlap_general_indices))
    stats["swing_ready_count"] = int(len(cluster_map["swing_core"]))
    stats["overlap_guarded_count"] = int(len(overlap_guarded_indices))
    _warn(
        warnings,
        "round_overlap_subtypes="
        f"counterflow={len(overlap_counterflow_indices)} "
        f"low_tempo_draw={len(overlap_drawlike_indices)} "
        f"general={len(overlap_general_indices)}",
    )
    _warn(
        warnings,
        f"round_swing_context=swing_ready={len(cluster_map['swing_core'])} overlap_guarded={len(overlap_guarded_indices)}",
    )
    away_swing_indices = [i for i in _rank_indices("away_flip") if i in overlap_guarded_indices or i in cluster_map["swing_core"]] + [i for i in _rank_indices("away_flip") if i not in overlap_guarded_indices and i not in cluster_map["swing_core"]]

    def _target_flip_scaled(scenario_id: str, default: int) -> int:
        value = int(_target_flip(scenario_id, default))
        sid = str(scenario_id)
        if sid not in {"03", "07", "10"}:
            return value
        swing_ready_count = int(stats.get("swing_ready_count", 0) or 0)
        overlap_guarded_count = int(stats.get("overlap_guarded_count", 0) or 0)
        topo_mode = str(topology["mode"])
        if topo_mode == "balanced":
            if sid == "03":
                if swing_ready_count <= 2 and overlap_guarded_count == 0:
                    value = max(4, value - 1)
                elif swing_ready_count >= 5 or overlap_guarded_count >= 1:
                    value = max(value, 5)
            elif sid in {"07", "10"}:
                if swing_ready_count <= 2 and overlap_guarded_count == 0:
                    value = 1
                elif swing_ready_count <= 4 and overlap_guarded_count == 0:
                    value = min(value, 2)
                else:
                    value = min(max(value, 2), 3)
        elif topo_mode == "swing_heavy":
            if sid == "03":
                if swing_ready_count >= 6 and overlap_guarded_count >= 1:
                    value = max(2, value - 1)
                elif swing_ready_count <= 3 and overlap_guarded_count == 0:
                    value = max(value, 4)
            elif sid in {"07", "10"}:
                if swing_ready_count <= 3 and overlap_guarded_count == 0:
                    value = min(value, 4)
                elif swing_ready_count >= 6 or overlap_guarded_count >= 2:
                    value = max(value, 5)
        return int(max(0, min(5, value)))

    stats["portfolio_target_effective_summary"] = (
        f"06={ticket06_mode}:D{_target_draw_scaled('06', 2)} / "
        f"06C={','.join(sorted(_target_allowed_clusters('06', 'allowed_draw_clusters') or []))} / "
        f"02C={','.join(sorted(_target_allowed_clusters('02', 'allowed_draw_clusters') or []))} / "
        f"05C={','.join(sorted(_target_allowed_clusters('05', 'allowed_draw_clusters') or []))} / "
        f"08=D{_target_draw('08', 2)} / "
        f"09=D{_target_draw('09', 1)} / "
        f"03=F{_target_flip_scaled('03', 4)} / "
        f"07=F{_target_flip_scaled('07', 4)} / "
        f"10=F{_target_flip_scaled('10', 4)}"
    )
    _warn(warnings, f"portfolio_target_effective={stats['portfolio_target_effective_summary']}")

    low_tempo_draw_indices = sorted(
        cluster_map["low_tempo_draw"],
        key=lambda i: (
            -float(plans[i].shape_draw_compression),
            -float(plans[i].shape_entropy),
            -float(plans[i].p_draw),
            float(plans[i].shape_directionality),
            i,
        ),
    )
    pure_draw_core = [i for i in draw_cluster_indices if i not in cluster_map["overlap"] and i not in cluster_map["low_tempo_draw"]]
    pure_swing_core = [i for i in swing_cluster_indices if i not in cluster_map["overlap"] and i not in cluster_map["low_tempo_draw"]]
    j1_draw_escape_indices = sorted(
        [
            i
            for i in pure_draw_core
            if str(plans[i].league).upper() == "J1"
            and str(plans[i].base_pick) == "2"
            and float(_topology_asymmetry_score(plans[i])) <= 0.22
            and float(_admission_escape_score(plans[i], "0")) >= 0.70
        ],
        key=lambda i: (
            float(_topology_asymmetry_score(plans[i])),
            -float(_admission_escape_score(plans[i], "0")),
            -float(plans[i].p_draw),
            i,
        ),
    )
    def _only_j1(indices: List[int]) -> List[int]:
        return [i for i in indices if str(plans[i].league).upper() == "J1"]
    draw_cluster_a = _merge_unique(neutral_draw_indices, pure_draw_core, close_draw_indices)
    draw_cluster_b = _merge_unique(neutral_draw_indices[1:], pure_draw_core, close_draw_indices[1:])
    swing_cluster_a = _merge_unique(overlap_guarded_indices[:1], pure_swing_core, neutral_swing_indices, _rank_indices("flip"))
    swing_cluster_b = _merge_unique(overlap_guarded_indices[1:], away_swing_indices, neutral_swing_indices, _rank_indices("volatility"))
    close_draw_mix = _merge_unique(neutral_draw_indices[:3], pure_draw_core[:3], close_draw_indices[:4])
    blend_draw_a = _merge_unique(neutral_draw_indices[:3], pure_draw_core[:2], close_draw_indices[:3])
    blend_draw_b = _merge_unique(neutral_draw_indices[1:5], pure_draw_core[2:5], close_draw_indices[2:6])
    blend_swing_a = _merge_unique(overlap_guarded_indices[:1], pure_swing_core[:3], neutral_swing_indices[:3], _rank_indices("flip")[:3])
    blend_swing_b = _merge_unique(overlap_guarded_indices[1:2], pure_swing_core[1:5], neutral_swing_indices[1:5], _rank_indices("volatility")[1:5])
    hold_guard_indices = _merge_unique(hold_cluster_indices[:2], close_draw_indices[:2], neutral_swing_indices[:1])
    pressure_return_indices = _merge_unique(pure_draw_core[:2], hold_cluster_indices[:2], neutral_swing_indices[:2], close_draw_indices[:1])
    close_draw_fallback_indices = _merge_unique(close_draw_indices[:3], neutral_draw_indices[:2], j1_draw_escape_indices[:2], pure_draw_core[:1])
    # ticket09 は low-tempo の補完に責務を絞る。
    # neutral / J2 draw_core をここで抱えると、後段で役割過多になりやすい。
    low_tempo_cover_indices = _merge_unique(
        low_tempo_draw_indices[:3],
    )
    neutral_draw_recipe = _merge_unique(neutral_draw_indices[:4], close_draw_indices[:3])
    tension_draw_recipe = _merge_unique(pure_draw_core[:4], close_draw_indices[:2], neutral_draw_indices[2:6])
    close_draw_recipe = _merge_unique(low_tempo_draw_indices[:2], close_draw_fallback_indices, neutral_draw_indices[2:4])
    j1_draw_cluster_a = _merge_unique(_only_j1(neutral_draw_indices), _only_j1(pure_draw_core), _only_j1(close_draw_indices))
    j1_draw_cluster_b = _merge_unique(_only_j1(neutral_draw_indices[1:]), _only_j1(pure_draw_core), _only_j1(close_draw_indices[1:]))
    j1_neutral_draw_recipe = _merge_unique(_only_j1(neutral_draw_indices[:4]), j1_draw_escape_indices[:2], _only_j1(close_draw_indices[:3]))
    j1_tension_draw_recipe = _merge_unique(j1_draw_escape_indices[:3], _only_j1(pure_draw_core[:4]), _only_j1(close_draw_indices[:2]), _only_j1(neutral_draw_indices[2:6]))
    j1_close_draw_recipe = _merge_unique(
        _only_j1(low_tempo_draw_indices[:2]),
        _only_j1(close_draw_fallback_indices),
        _only_j1(neutral_draw_indices[2:4]),
    )
    j1_pressure_draw_recipe = _merge_unique(j1_draw_escape_indices[:3], _only_j1(pure_draw_core[:5]), _only_j1(neutral_draw_indices[2:6]), _only_j1(close_draw_indices[:1]))

    if str(topology["mode"]) == "draw_heavy":
        recipe_specs = [
            ("02", "Draw Compression A", "draw-heavy close/neutral draw compression cluster", j1_neutral_draw_recipe, [], _target_draw("02", 3), _target_flip("02", 0), False, False, {"overlap": 0, "draw_core": 1}),
            ("03", "Draw Compression B", "draw-heavy neutral+draw-core compression allocation", j1_draw_cluster_b, [], _target_draw("03", 3), _target_flip_scaled("03", 0), False, False, {"overlap": 0, "draw_core": 2}),
            ("04", "Hold Guard", "draw-heavy round guard ticket", [], hold_guard_indices, _target_draw("04", 0), _target_flip("04", 2), False, False, {"overlap": 0}),
            ("05", "Volatility Swing Cover", "draw-heavy round swing hedge", [], swing_cluster_a, _target_draw("05", 0), _target_flip("05", 3), False, False, {"overlap": 1}),
            ("06", "J2 Draw Pressure", "J2 draw pressure cluster", _rank_indices("j2_draw"), [], _target_draw_scaled("06", 2), _target_flip("06", 0), False, True, {"overlap": 0, "draw_core": 1}),
            ("07", "Away Swing Cover", "draw-heavy away swing hedge", [], swing_cluster_b, _target_draw("07", 0), _target_flip_scaled("07", 3), True, False, {"overlap": 1}),
            ("08", "Close Draw", "draw-heavy close/low-tempo draw cluster", j1_close_draw_recipe, [], _target_draw("08", 2), _target_flip("08", 0), False, False, {"overlap": 0, "draw_core": 1, "low_tempo_draw": 1}, True),
            ("09", "Low Tempo Cover", "draw-heavy low-tempo + J2 draw cover", low_tempo_cover_indices, [], _target_draw("09", 1), _target_flip("09", 0), False, False, {"overlap": 0, "low_tempo_draw": 1}, True),
            ("10", "Blend Wide", "draw+swing blend", blend_draw_b, blend_swing_b, _target_draw("10", 1), _target_flip_scaled("10", 3), True, False, {"overlap": 1, "draw_core": 1}),
        ]
    elif str(topology["mode"]) == "swing_heavy":
        recipe_specs = [
            ("02", "Volatility Swing A", "swing-heavy overlap-swing allocation A", [], swing_cluster_a, _target_draw("02", 0), _target_flip("02", 5), False, False, {"overlap": 2}),
            ("03", "Volatility Swing B", "swing-heavy overlap-swing allocation B", [], swing_cluster_b, _target_draw("03", 0), _target_flip_scaled("03", 4), True, False, {"overlap": 1}),
            ("04", "Hold Guard", "swing-heavy round guard ticket", [], hold_guard_indices, _target_draw("04", 0), _target_flip("04", 2), False, False, {"overlap": 0}),
            ("05", "Draw Pressure Cover", "swing-heavy draw-core tension hedge", j1_pressure_draw_recipe, pressure_return_indices, _target_draw("05", 3), _target_flip("05", 1), False, False, {"overlap": 0, "draw_core": 3, "neutral": 1}),
            ("06", "J2 Draw Pressure", "J2 draw pressure cluster", _rank_indices("j2_draw"), [], _target_draw_scaled("06", 3), _target_flip("06", 0), False, True, {"overlap": 1}),
            ("07", "Volatility Cluster", "swing-heavy volatility cluster", [], _rank_indices("volatility"), _target_draw("07", 0), _target_flip_scaled("07", 4), False, False, {"overlap": 1}),
            ("08", "Close Draw", "swing-heavy close/low-tempo draw cluster", j1_close_draw_recipe, [], _target_draw("08", 2), _target_flip("08", 0), False, False, {"overlap": 0, "low_tempo_draw": 1}, True),
            ("09", "Low Tempo Cover", "swing-heavy low-tempo + J2 draw cover", low_tempo_cover_indices, [], _target_draw("09", 1), _target_flip("09", 0), False, False, {"overlap": 0, "low_tempo_draw": 1}, True),
            ("10", "Blend Wide", "flip+draw blend", blend_draw_b, blend_swing_b, _target_draw("10", 1), _target_flip_scaled("10", 4), True, False, {"overlap": 1}),
        ]
    else:
        recipe_specs = [
            ("02", "Draw Compression", "balanced close/neutral draw compression cluster", j1_neutral_draw_recipe, [], _target_draw("02", 3), _target_flip("02", 0), False, False, {"overlap": 0, "draw_core": 1}),
            ("03", "Volatility Swing", "balanced round swing cluster", [], swing_cluster_a, _target_draw("03", 0), _target_flip_scaled("03", 4), False, False, {"overlap": 2}),
            ("04", "Hold Guard", "balanced round guard ticket", [], hold_guard_indices, _target_draw("04", 0), _target_flip("04", 2), False, False, {"overlap": 0}),
            ("05", "Draw Pressure", "balanced draw-core tension cluster", j1_pressure_draw_recipe, pressure_return_indices, _target_draw("05", 3), _target_flip("05", 1), False, False, {"overlap": 0, "draw_core": 3, "neutral": 1}),
            ("06", "J2 Draw Pressure", "J2 draw pressure cluster", _rank_indices("j2_draw"), [], _target_draw_scaled("06", 3), _target_flip("06", 0), False, True, {"overlap": 1}),
            ("07", "Away Swing", "balanced round away swing", [], swing_cluster_b, _target_draw("07", 0), _target_flip_scaled("07", 4), True, False, {"overlap": 1}),
            ("08", "Close Draw", "balanced close/low-tempo draw cluster", j1_close_draw_recipe, [], _target_draw("08", 2), _target_flip("08", 0), False, False, {"overlap": 0, "low_tempo_draw": 1}, True),
            ("09", "Low Tempo Cover", "balanced low-tempo + J2 draw cover", low_tempo_cover_indices, [], _target_draw("09", 1), _target_flip("09", 0), False, False, {"overlap": 0, "low_tempo_draw": 1}, True),
            ("10", "Blend Wide", "balanced draw+swing blend", blend_draw_b, blend_swing_b, _target_draw("10", 2), _target_flip_scaled("10", 4), True, False, {"overlap": 1}),
        ]

    for spec in recipe_specs:
        if len(spec) == 10:
            scenario_id, scenario_name, scenario_note, draw_idx, flip_idx, max_draw, max_flip, prefer_away, j2_only, cluster_caps = spec
            allow_low_tempo_draw = False
        else:
            scenario_id, scenario_name, scenario_note, draw_idx, flip_idx, max_draw, max_flip, prefer_away, j2_only, cluster_caps, allow_low_tempo_draw = spec
        merged_cluster_caps = _target_cluster_caps(scenario_id, cluster_caps)
        allowed_draw_clusters = _target_allowed_clusters(scenario_id, "allowed_draw_clusters")
        allowed_flip_clusters = _target_allowed_clusters(scenario_id, "allowed_flip_clusters")
        ticket, desc = _make_ticket(
            scenario_id,
            scenario_name,
            draw_idx,
            flip_idx,
            max_draw_changes=max_draw,
            max_flip_changes=max_flip,
            prefer_away=prefer_away,
            j2_only=j2_only,
            cluster_caps=merged_cluster_caps,
            allow_low_tempo_draw=allow_low_tempo_draw,
            allowed_draw_clusters=allowed_draw_clusters,
            allowed_flip_clusters=allowed_flip_clusters,
        )
        tickets.append(ticket)
        flip_descs.append(desc)
        scenario_defs.append(ScenarioDef(scenario_id, scenario_name, scenario_note))

    protected_home_rescue_slots: set[tuple[int, int]] = set()

    def _apply_escape_distribution() -> None:
        changes: List[str] = []
        escape_targets = [
            {"ticket_idx": 1, "symbol": "0", "label": "escape_draw_a"},  # ticket02
            {"ticket_idx": 4, "symbol": "0", "label": "escape_draw_b"},  # ticket05
            {"ticket_idx": 7, "symbol": "0", "label": "escape_draw_c"},  # ticket08
            {"ticket_idx": 8, "symbol": "0", "label": "escape_draw_d"},  # ticket09
        ]
        escape_pool = sorted(
            [
                i
                for i in ok_indices
                if str(plans[i].league).upper() == "J1"
                and _cluster_label(plans[i]) == "draw_core"
                and str(plans[i].base_pick) == "2"
                and float(_topology_asymmetry_score(plans[i])) <= 0.22
                and float(_admission_escape_score(plans[i], "0")) >= 0.70
            ],
            key=lambda i: (
                float(_topology_asymmetry_score(plans[i])),
                -float(_admission_escape_score(plans[i], "0")),
                -float(plans[i].p_draw),
                i,
            ),
        )
        used_matches: set[int] = set()
        for target in escape_targets:
            ti = int(target["ticket_idx"])
            if ti >= len(tickets):
                continue
            for mi in escape_pool:
                if mi in used_matches:
                    continue
                before = tickets[ti][mi]
                if before == str(target["symbol"]):
                    continue
                cand = list(tickets[ti])
                cand[mi] = str(target["symbol"])
                cand_key = _ticket_key(cand)
                if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                    continue
                tickets[ti] = cand
                used_matches.add(mi)
                flip_descs[ti] = (
                    f"{flip_descs[ti]}; {target['label']}:M{plans[mi].match_no:02d}:{before}->{target['symbol']}"
                )
                changes.append(
                    f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->{target['symbol']} score={float(_admission_escape_score(plans[mi], '0')):.3f}"
                )
                break

        home_escape_pool = sorted(
            [
                i
                for i in ok_indices
                if str(plans[i].base_pick) == "0"
                and float(_admission_escape_score(plans[i], "1")) >= 0.30
                and float(_admission_escape_score(plans[i], "1")) >= float(_admission_escape_score(plans[i], "2"))
            ],
            key=lambda i: (
                -float(_admission_escape_score(plans[i], "1")),
                float(_topology_asymmetry_score(plans[i])),
                i,
            ),
        )
        for ti in [6, 9]:
            if ti >= len(tickets):
                continue
            for mi in home_escape_pool:
                if mi in used_matches:
                    continue
                before = tickets[ti][mi]
                if before == "1":
                    continue
                cand = list(tickets[ti])
                cand[mi] = "1"
                cand_key = _ticket_key(cand)
                if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                    continue
                tickets[ti] = cand
                used_matches.add(mi)
                flip_descs[ti] = f"{flip_descs[ti]}; escape_home:M{plans[mi].match_no:02d}:{before}->1"
                changes.append(
                    f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->1 score={float(_admission_escape_score(plans[mi], '1')):.3f}"
                )
                break

        away_escape_pool = sorted(
            [
                i
                for i in ok_indices
                if str(plans[i].base_pick) == "0"
                and (
                    float(_admission_escape_score(plans[i], "2")) >= 0.22
                    or (
                        float(plans[i].p_away) >= 0.24
                        and (
                            float(plans[i].shape_swing_instability) >= 0.34
                            or float(plans[i].lab_flip_weight) >= 0.34
                        )
                    )
                )
            ],
            key=lambda i: (
                -float(_admission_escape_score(plans[i], "2")),
                -float(plans[i].p_away),
                -float(plans[i].lab_flip_weight),
                i,
            ),
        )
        for ti in [9]:
            if ti >= len(tickets):
                continue
            for mi in away_escape_pool:
                before = tickets[ti][mi]
                if before == "2":
                    continue
                cand = list(tickets[ti])
                cand[mi] = "2"
                cand_key = _ticket_key(cand)
                if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                    continue
                tickets[ti] = cand
                used_matches.add(mi)
                flip_descs[ti] = f"{flip_descs[ti]}; escape_away:M{plans[mi].match_no:02d}:{before}->2"
                changes.append(
                    f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->2 score={float(_admission_escape_score(plans[mi], '2')):.3f}"
                )
                break

        draw_home_targets = [
            {
                "ticket_idx": 4,
                "label": "escape_home_draw_core",
                "match_filter": lambda p: (
                    _cluster_label(p) == "draw_core"
                    and str(p.league).upper() == "J1"
                    and 0.10 <= float(_topology_asymmetry_score(p)) <= 0.16
                    and float(_admission_escape_score(p, "1")) >= 0.28
                    and float(_admission_escape_score(p, "1")) >= float(_admission_escape_score(p, "2"))
                    and float(p.p_draw) >= 0.45
                    and abs(float(p.p_home) - float(p.p_away)) <= 0.05
                ),
            },  # ticket05
            {
                "ticket_idx": 9,
                "label": "escape_home_draw_core_secondary",
                "match_filter": lambda p: (
                    _cluster_label(p) == "draw_core"
                    and str(p.league).upper() == "J1"
                    and _normalize_symbol_token(p.best) == "0"
                    and 0.10 <= float(_topology_asymmetry_score(p)) <= 0.18
                    and float(_admission_escape_score(p, "1")) >= 0.24
                    and float(_admission_escape_score(p, "1")) >= (0.88 * float(_admission_escape_score(p, "2")))
                    and float(p.p_draw) >= 0.43
                    and abs(float(p.p_home) - float(p.p_away)) <= 0.06
                ),
            },  # ticket10: draw本命からの低率home rescue
            {
                "ticket_idx": 9,
                "label": "escape_home_low_tempo",
                "match_filter": lambda p: (
                    _cluster_label(p) == "low_tempo_draw"
                    and str(p.league).upper() == "J1"
                    and str(p.base_pick) == "0"
                    and str(getattr(p, "lab_basis_hint", "")) == "flat_draw_trap"
                    and float(_topology_asymmetry_score(p)) <= 0.03
                    and float(_admission_escape_score(p, "1")) >= 0.24
                    and float(_admission_escape_score(p, "1")) >= float(_admission_escape_score(p, "2"))
                    and float(p.p_draw) >= 0.47
                    and abs(float(p.p_home) - float(p.p_away)) <= 0.01
                ),
            },  # ticket10
        ]
        for target in draw_home_targets:
            ti = int(target["ticket_idx"])
            if ti >= len(tickets):
                continue
            draw_home_pool = sorted(
                [i for i in ok_indices if target["match_filter"](plans[i])],
                key=lambda i: (
                    -float(_admission_escape_score(plans[i], "1")),
                    float(_topology_asymmetry_score(plans[i])),
                    -float(plans[i].p_draw),
                    i,
                ),
            )
            for mi in draw_home_pool:
                before = tickets[ti][mi]
                if "secondary" in str(target["label"]):
                    if before == "1":
                        continue
                elif before != "0":
                    continue
                cand = list(tickets[ti])
                cand[mi] = "1"
                cand_key = _ticket_key(cand)
                if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                    continue
                tickets[ti] = cand
                if "secondary" not in str(target["label"]):
                    used_matches.add(mi)
                if cand[mi] == "1":
                    protected_home_rescue_slots.add((ti, mi))
                flip_descs[ti] = f"{flip_descs[ti]}; {target['label']}:M{plans[mi].match_no:02d}:{before}->1"
                changes.append(
                    f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->1 score={float(_admission_escape_score(plans[mi], '1')):.3f}"
                )
                break

        stats["admission_escape_distribution_count"] = int(len(changes))
        stats["admission_escape_distribution_matches"] = "; ".join(changes[:30])
        if changes:
            _warn(
                warnings,
                "admission_escape_distribution="
                f"{stats['admission_escape_distribution_matches']}",
            )

    _apply_escape_distribution()

    def _apply_draw_origin_home_mix() -> None:
        mix_changes: List[str] = []
        # D本命から side へ逃がす draw-core 試合では、A一辺倒を避けて
        # 一部の券に H を混ぜる。対象は J1 の neutral draw-core 帯に限定。
        target_ticket_indices = [4, 9, 2]  # ticket05, ticket10, ticket03
        eligible_matches = sorted(
            [
                i
                for i in ok_indices
                if str(plans[i].league).upper() == "J1"
                and _cluster_label(plans[i]) == "draw_core"
                and _normalize_symbol_token(plans[i].best) == "0"
                and _normalize_symbol_token(plans[i].second) == "2"
                and str(plans[i].lab_pressure_band).lower() == "neutral"
                and str(plans[i].lab_control_band).lower() == "neutral"
                and float(plans[i].p_draw) >= 0.43
                and abs(float(plans[i].p_home) - float(plans[i].p_away)) <= 0.06
                and float(_topology_asymmetry_score(plans[i])) <= 0.18
                and float(_admission_escape_score(plans[i], "1")) >= 0.23
                and float(_admission_escape_score(plans[i], "1")) >= (0.82 * float(_admission_escape_score(plans[i], "2")))
            ],
            key=lambda i: (
                -float(plans[i].p_draw),
                float(_topology_asymmetry_score(plans[i])),
                -float(_admission_escape_score(plans[i], "1")),
                i,
            ),
        )
        for mi in eligible_matches:
            col = [tickets[ti][mi] for ti in range(min(REQUIRED_TICKET_COUNT, len(tickets)))]
            a_count = sum(1 for sym in col if str(sym) == "2")
            h_count = sum(1 for sym in col if str(sym) == "1")
            # D由来Aの 3-4割までは H を許すが、過剰化を避けて 3 本まで。
            target_h_count = min(3, max(h_count, int(math.ceil(a_count * 0.35))))
            if h_count >= target_h_count:
                continue
            for ti in target_ticket_indices:
                if ti >= len(tickets):
                    continue
                before = tickets[ti][mi]
                if before == "1":
                    continue
                if before != "2":
                    continue
                cand = list(tickets[ti])
                cand[mi] = "1"
                cand_key = _ticket_key(cand)
                if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                    continue
                tickets[ti] = cand
                flip_descs[ti] = f"{flip_descs[ti]}; draw_origin_home_mix:M{plans[mi].match_no:02d}:{before}->1"
                protected_home_rescue_slots.add((ti, mi))
                mix_changes.append(
                    f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->1 target_h={target_h_count}"
                )
                h_count += 1
                if h_count >= target_h_count:
                    break
        stats["draw_origin_home_mix_count"] = int(len(mix_changes))
        stats["draw_origin_home_mix_matches"] = "; ".join(mix_changes[:30])
        if mix_changes:
            _warn(warnings, f"draw_origin_home_mix={stats['draw_origin_home_mix_matches']}")

    _apply_draw_origin_home_mix()

    def _ticket_distance(a: List[str], b: List[str]) -> int:
        return int(sum(1 for x, y in zip(a, b) if x != y))

    def _home_rescue_locked(ticket_idx: int, match_idx: int, alt: Optional[str] = None) -> bool:
        if (ticket_idx, match_idx) not in protected_home_rescue_slots:
            return False
        if alt is None:
            return True
        return str(alt) != "1"

    def _candidate_alt_order(plan: MatchPlan, scenario_id: str = "") -> List[str]:
        ticket06_mode_local = str(stats.get("ticket_06_mode", "skip"))
        if str(scenario_id) in {"08", "09"} or str(scenario_id) == "06":
            ordered = [
                "0",
                str(plan.second),
                _best_flip_symbol(plan, prefer_away=False),
                _best_flip_symbol(plan, prefer_away=True),
                str(plan.third),
                str(plan.best),
            ]
        else:
            ordered = [
                _best_flip_symbol(plan, prefer_away=True),
                _best_flip_symbol(plan, prefer_away=False),
                "0",
                str(plan.second),
                str(plan.third),
                str(plan.best),
            ]
        out: List[str] = []
        for sym in ordered:
            if sym in {"0", "1", "2"} and sym not in out:
                out.append(sym)
        return out

    def _allow_post_pass_change(plan: MatchPlan, alt: str, scenario_id: str = "", phase: str = "general") -> bool:
        if _is_protected_profile(plan):
            return False
        cluster_name = _cluster_label(plan)
        if str(scenario_id) == "06":
            if cluster_name not in {"draw_core", "neutral", "low_tempo_draw"}:
                return False
            if str(plan.league).upper() != "J2":
                return False
            if float(plan.shape_draw_compression) < 0.050 or float(plan.p_draw) < 0.34:
                return False
            if str(topology["mode"]) == "balanced":
                if cluster_name == "draw_core" and float(plan.shape_draw_compression) < 0.078:
                    return False
                if cluster_name == "neutral" and float(plan.shape_entropy) < 0.52:
                    return False
        if str(scenario_id) == "08":
            if cluster_name not in {"low_tempo_draw", "neutral", "draw_core"}:
                return False
            if float(plan.shape_draw_compression) < 0.075 or float(plan.p_draw) < 0.28:
                return False
        if str(scenario_id) == "09":
            allow_low_tempo = cluster_name == "low_tempo_draw"
            allow_j2 = str(plan.league).upper() == "J2" and float(plan.shape_draw_compression) >= 0.075
            allow_neutral = cluster_name == "neutral" and float(plan.shape_entropy) >= 1.02 and float(plan.p_draw) >= 0.31
            allow_draw_core = cluster_name == "draw_core" and float(plan.shape_draw_compression) >= 0.082 and float(plan.p_draw) >= 0.32
            if not (allow_low_tempo or allow_j2 or allow_neutral or allow_draw_core):
                return False
        if cluster_name != "overlap":
            return True
        overlap_subtype = _overlap_subtype(plan)
        if overlap_subtype in {"counterflow_away", "counterflow_home"}:
            return False
        if overlap_subtype == "low_tempo_draw":
            return False
        if alt == "0":
            return False
        if int(plan.match_no) not in overlap_guarded_match_nos:
            return False
        return bool(
            float(plan.shape_swing_instability) >= 0.40
            and float(plan.shape_path_imbalance) >= 0.20
            and max(float(plan.shape_directionality), float(plan.shape_entropy)) >= 0.24
        )

    # Ensure uniqueness without introducing new corrective heuristics.
    seen: Dict[str, int] = {}
    for ti, ticket in enumerate(tickets):
        scenario_id = scenario_defs[ti].scenario_id if ti < len(scenario_defs) else f"{ti+1:02d}"
        key = _ticket_key(ticket)
        if key not in seen:
            seen[key] = ti
            continue
        if scenario_id == "06":
            priority_indices = _rank_indices("j2_draw") + j2_ok_indices + neutral_draw_indices + close_draw_indices
        elif scenario_id == "08":
            priority_indices = close_draw_indices + neutral_draw_indices + low_tempo_draw_indices + draw_cluster_indices
        elif scenario_id == "09":
            priority_indices = low_tempo_cover_indices + close_draw_indices + neutral_draw_indices + draw_cluster_indices + [i for i in _rank_indices("j2_draw") if i not in low_tempo_cover_indices]
        else:
            priority_indices = _rank_indices("flip") + _rank_indices("stall") + close_draw_indices
        for mi in priority_indices:
            p = plans[mi]
            alt_order = _candidate_alt_order(p, scenario_id)
            changed = False
            for alt in alt_order:
                if alt not in {"0", "1", "2"} or alt == ticket[mi]:
                    continue
                if _home_rescue_locked(ti, mi, alt):
                    continue
                if alt == "0" and not _draw_ready(p):
                    continue
                if not _allow_post_pass_change(p, alt, scenario_id, phase="dedupe"):
                    continue
                cand = list(ticket)
                cand[mi] = alt
                cand_key = _ticket_key(cand)
                if cand_key in seen:
                    continue
                ticket = cand
                tickets[ti] = ticket
                flip_descs[ti] = f"{flip_descs[ti]}; dedupe:M{p.match_no:02d}:{base_ticket[mi]}->{alt}"
                seen[cand_key] = ti
                changed = True
                break
            if changed:
                break
        seen[_ticket_key(tickets[ti])] = ti

    min_ticket_distance = 2
    for ti in range(1, len(tickets)):
        scenario_id = scenario_defs[ti].scenario_id if ti < len(scenario_defs) else f"{ti+1:02d}"
        nearest = min((_ticket_distance(tickets[ti], tickets[oj]) for oj in range(ti)), default=99)
        if nearest >= min_ticket_distance:
            continue
        if scenario_id == "06":
            priority_indices = _rank_indices("j2_draw") + j2_ok_indices + neutral_draw_indices + close_draw_indices
        elif scenario_id == "08":
            priority_indices = close_draw_indices + neutral_draw_indices + draw_cluster_indices
        elif scenario_id == "09":
            priority_indices = low_tempo_cover_indices + close_draw_indices + neutral_draw_indices + draw_cluster_indices + [i for i in _rank_indices("j2_draw") if i not in low_tempo_cover_indices]
        else:
            priority_indices = swing_cluster_indices + neutral_swing_indices + draw_cluster_indices + _rank_indices("volatility") + hold_cluster_indices
        for mi in priority_indices:
            p = plans[mi]
            alt_order = _candidate_alt_order(p, scenario_id)
            changed = False
            for alt in alt_order:
                if alt not in {"0", "1", "2"} or alt == tickets[ti][mi]:
                    continue
                if _home_rescue_locked(ti, mi, alt):
                    continue
                if alt == "0" and not _draw_ready(p):
                    continue
                if not _allow_post_pass_change(p, alt, scenario_id, phase="spread"):
                    continue
                cand = list(tickets[ti])
                cand[mi] = alt
                if min((_ticket_distance(cand, tickets[oj]) for oj in range(ti)), default=99) < min_ticket_distance:
                    continue
                tickets[ti] = cand
                flip_descs[ti] = f"{flip_descs[ti]}; spread:M{p.match_no:02d}:{base_ticket[mi]}->{alt}"
                changed = True
                break
            if changed:
                break

    def _force_unique_and_spread() -> None:
        attempts = 0
        while attempts < 3:
            attempts += 1
            changed_any = False
            seen_keys: Dict[str, int] = {}
            for ti, ticket in enumerate(tickets):
                scenario_id = scenario_defs[ti].scenario_id if ti < len(scenario_defs) else f"{ti+1:02d}"
                key = _ticket_key(ticket)
                prev = seen_keys.get(key)
                if prev is None:
                    seen_keys[key] = ti
                    continue
                if scenario_id == "06":
                    priority_indices = _rank_indices("j2_draw") + j2_ok_indices + neutral_draw_indices + close_draw_indices + ok_indices
                elif scenario_id == "08":
                    priority_indices = close_draw_indices + neutral_draw_indices + draw_cluster_indices + ok_indices
                elif scenario_id == "09":
                    priority_indices = low_tempo_cover_indices + close_draw_indices + neutral_draw_indices + draw_cluster_indices + [i for i in _rank_indices("j2_draw") if i not in low_tempo_cover_indices] + ok_indices
                else:
                    priority_indices = swing_cluster_indices + neutral_swing_indices + draw_cluster_indices + _rank_indices("volatility") + close_draw_indices + ok_indices
                resolved = False
                for mi in priority_indices:
                    p = plans[mi]
                    for alt in _candidate_alt_order(p, scenario_id):
                        if alt == tickets[ti][mi]:
                            continue
                        if _home_rescue_locked(ti, mi, alt):
                            continue
                        if alt == "0" and not _draw_ready(p):
                            continue
                        if not _allow_post_pass_change(p, alt, scenario_id, phase="force_unique"):
                            continue
                        cand = list(tickets[ti])
                        cand[mi] = alt
                        cand_key = _ticket_key(cand)
                        if cand_key in seen_keys:
                            continue
                        if min((_ticket_distance(cand, tickets[oj]) for oj in range(ti) if oj != ti), default=99) < 1:
                            continue
                        tickets[ti] = cand
                        flip_descs[ti] = f"{flip_descs[ti]}; force_unique:M{p.match_no:02d}:{base_ticket[mi]}->{alt}"
                        seen_keys[cand_key] = ti
                        changed_any = True
                        resolved = True
                        break
                    if resolved:
                        break
                if not resolved:
                    seen_keys[key] = ti
            if not changed_any:
                break

    _force_unique_and_spread()

    def _force_min_pair_distance(target_distance: int = 2, max_rounds: int = 4) -> None:
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            pair = None
            pair_dist = 999
            for i in range(len(tickets)):
                for j in range(i + 1, len(tickets)):
                    dist = _ticket_distance(tickets[i], tickets[j])
                    if dist < pair_dist:
                        pair = (i, j)
                        pair_dist = dist
            if pair is None or pair_dist >= target_distance:
                break
            ti = pair[1]
            scenario_id = scenario_defs[ti].scenario_id if ti < len(scenario_defs) else f"{ti+1:02d}"
            if scenario_id == "06":
                priority_indices = _rank_indices("j2_draw") + j2_ok_indices + neutral_draw_indices + close_draw_indices + ok_indices
            elif scenario_id == "08":
                priority_indices = close_draw_indices + neutral_draw_indices + draw_cluster_indices + ok_indices
            elif scenario_id == "09":
                priority_indices = low_tempo_cover_indices + close_draw_indices + neutral_draw_indices + draw_cluster_indices + [i for i in _rank_indices("j2_draw") if i not in low_tempo_cover_indices] + ok_indices
            else:
                priority_indices = swing_cluster_indices + neutral_swing_indices + draw_cluster_indices + close_draw_indices + _rank_indices("volatility") + ok_indices
            changed = False
            for mi in priority_indices:
                p = plans[mi]
                for alt in _candidate_alt_order(p, scenario_id):
                    if alt == tickets[ti][mi]:
                        continue
                    if _home_rescue_locked(ti, mi, alt):
                        continue
                    if alt == "0" and not _draw_ready(p):
                        continue
                    if not _allow_post_pass_change(p, alt, scenario_id, phase="pair_spread"):
                        continue
                    cand = list(tickets[ti])
                    cand[mi] = alt
                    cand_key = _ticket_key(cand)
                    if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                        continue
                    distances = [_ticket_distance(cand, tickets[k]) for k in range(len(tickets)) if k != ti]
                    if min(distances, default=99) < target_distance:
                        continue
                    tickets[ti] = cand
                    flip_descs[ti] = f"{flip_descs[ti]}; pair_spread:M{p.match_no:02d}:{base_ticket[mi]}->{alt}"
                    changed = True
                    break
                if changed:
                    break
            if not changed:
                break

    _force_min_pair_distance(target_distance=2, max_rounds=6)

    def _portfolio_rescue_separation() -> None:
        rescue_changes: List[str] = []

        def _apply_single_rescue(mi: int, ti: int, alt: str, tag: str) -> bool:
            if ti >= len(tickets):
                return False
            if alt not in {"0", "1", "2"}:
                return False
            if _home_rescue_locked(ti, mi, alt):
                return False
            before = tickets[ti][mi]
            if before == alt:
                return False
            p = plans[mi]
            if _is_protected_profile(p):
                return False
            scenario_id = scenario_defs[ti].scenario_id if ti < len(scenario_defs) else f"{ti+1:02d}"
            relaxed_draw_rescue = tag == "portfolio_draw_rescue"
            relaxed_side_rescue = tag == "portfolio_side_rescue"
            if alt == "0" and not _draw_ready(p) and not relaxed_draw_rescue:
                return False
            if not (relaxed_draw_rescue or relaxed_side_rescue) and not _allow_post_pass_change(
                p, alt, scenario_id, phase="portfolio_rescue"
            ):
                return False
            cand = list(tickets[ti])
            cand[mi] = alt
            cand_key = _ticket_key(cand)
            if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                return False
            tickets[ti] = cand
            flip_descs[ti] = f"{flip_descs[ti]}; {tag}:M{p.match_no:02d}:{before}->{alt}"
            rescue_changes.append(
                f"ticket={ti+1:02d} M{p.match_no:02d} {before}->{alt} tag={tag}"
            )
            return True

        for mi, p in enumerate(plans):
            if p.status != "OK":
                continue
            col = [tickets[ti][mi] for ti in range(min(REQUIRED_TICKET_COUNT, len(tickets)))]
            if not col or len(set(col)) != 1:
                continue
            dominant = str(col[0])
            risk = str(p.context_risk_level or "").lower()
            guidance = str(p.context_ticket_guidance or "").lower()
            cluster_label = _cluster_label(p)

            # Rescue away-lock cards that still carry strong draw topology.
            if (
                dominant == "2"
                and risk == "draw_watch"
                and guidance == "draw_cover"
                and cluster_label in {"hold_core", "neutral", "draw_core", "low_tempo_draw"}
                and float(p.p_draw) >= 0.40
                and float(p.shape_draw_compression) >= 0.68
                and float(_admission_escape_score(p, "0")) >= 0.42
            ):
                for ti in [7, 9, 4, 5]:
                    if _apply_single_rescue(mi, ti, "0", "portfolio_draw_rescue"):
                        break

            # Rescue draw-lock cards that still carry a meaningful side break path.
            if (
                dominant == "0"
                and risk == "draw_watch"
                and guidance == "draw_cover"
                and cluster_label in {"hold_core", "draw_core", "neutral"}
                and (
                    float(p.shape_swing_instability) >= 0.33
                    or float(p.lab_flip_weight) >= 0.32
                    or str(p.lab_matchup_profile).lower() in {"swing_watch", "balanced"}
                )
                and max(float(p.p_home), float(p.p_away)) >= 0.24
            ):
                prefer_away = bool(
                    str(p.lab_pressure_band).lower() == "away"
                    or float(p.lab_away_path_score) > float(p.lab_home_path_score)
                )
                alt = _best_flip_symbol(p, prefer_away=prefer_away)
                if alt in {"0", dominant}:
                    alt = str(p.second) if str(p.second) in {"1", "2"} else str(p.best)
                if float(_admission_escape_score(p, alt)) >= 0.36:
                    for ti in [6, 9, 2]:
                        if _apply_single_rescue(mi, ti, alt, "portfolio_side_rescue"):
                            break

        stats["portfolio_rescue_separation_count"] = int(len(rescue_changes))
        stats["portfolio_rescue_separation_matches"] = "; ".join(rescue_changes[:30])
        if rescue_changes:
            _warn(
                warnings,
                "portfolio_rescue_separation="
                f"{stats['portfolio_rescue_separation_matches']}",
            )

    _portfolio_rescue_separation()

    def _portfolio_role_family(ticket_index: int) -> str:
        if ticket_index in {1, 4, 5, 7, 8}:
            return "draw"
        if ticket_index in {2, 6, 9}:
            return "swing"
        if ticket_index == 0:
            return "base"
        return "other"

    def _ticket_cluster_change_count(ticket_index: int, cluster_name: str) -> int:
        count = 0
        if ticket_index >= len(tickets):
            return count
        for m_idx, base_symbol in enumerate(base_ticket):
            if tickets[ticket_index][m_idx] == base_symbol:
                continue
            if _cluster_label(plans[m_idx]) == cluster_name:
                count += 1
        return count

    def _portfolio_topology_fit(cluster_name: str, role_family: str, scenario_id: str) -> float:
        if cluster_name == "draw_core":
            if role_family == "draw":
                return 1.00
            if scenario_id == "10":
                return 0.25
            return -0.20
        if cluster_name == "low_tempo_draw":
            if role_family == "draw":
                return 1.10
            return -0.25
        if cluster_name == "neutral":
            if role_family == "draw":
                return 0.65
            if scenario_id == "10":
                return 0.15
            return -0.10
        if cluster_name == "overlap":
            if role_family == "swing":
                return 0.55
            if role_family == "draw":
                return 0.20
            return 0.00
        if cluster_name == "hold_core":
            if role_family == "draw":
                return 0.10
            if scenario_id == "10":
                return -0.25
            return -0.10
        return 0.00

    def _evaluate_portfolio_objective(mi: int, cand_ti: int, alt: str) -> Optional[Tuple[float, List[str], List[str], str]]:
        if cand_ti >= len(tickets) or alt not in {"0", "1", "2"}:
            return None
        before = tickets[cand_ti][mi]
        if before == alt:
            return None
        trial = list(tickets[cand_ti])
        trial[mi] = alt
        trial_key = _ticket_key(trial)
        if any(trial_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != cand_ti):
            return None

        old_distances = [_ticket_distance(tickets[cand_ti], tickets[k]) for k in range(len(tickets)) if k != cand_ti]
        new_distances = [_ticket_distance(trial, tickets[k]) for k in range(len(tickets)) if k != cand_ti]
        old_min = min(old_distances, default=99)
        new_min = min(new_distances, default=99)
        old_avg = float(sum(old_distances)) / float(len(old_distances)) if old_distances else 99.0
        new_avg = float(sum(new_distances)) / float(len(new_distances)) if new_distances else 99.0

        plan = plans[mi]
        cluster_name = _cluster_label(plan)
        overlap_subtype = _overlap_subtype(plan)
        role_family = _portfolio_role_family(cand_ti)
        scenario_id = scenario_defs[cand_ti].scenario_id if cand_ti < len(scenario_defs) else f"{cand_ti+1:02d}"
        same_cluster_changes = _ticket_cluster_change_count(cand_ti, cluster_name)
        current_diff_count = int(sum(1 for m_idx, base_symbol in enumerate(base_ticket) if tickets[cand_ti][m_idx] != base_symbol))
        new_diff_count = current_diff_count + (0 if before != base_ticket[mi] else 1)
        current_role_alt = sum(
            1
            for other_ti in range(len(tickets))
            if other_ti != cand_ti
            and _portfolio_role_family(other_ti) == role_family
            and tickets[other_ti][mi] == alt
        )
        current_cluster_role = current_role_alt + (
            1
            if _portfolio_role_family(cand_ti) == role_family and before == alt
            else 0
        )
        same_family_alt = sum(
            1
            for other_ti in range(len(tickets))
            if other_ti != cand_ti
            and _portfolio_role_family(other_ti) == role_family
            and tickets[other_ti][mi] == alt
        )
        new_cluster_role = current_role_alt + 1
        cluster_target = int(
            portfolio_cluster_role_targets.get(cluster_name, {}).get(role_family, 0)
        )
        role_alt_cap = int(portfolio_role_alt_caps.get(role_family, 1))
        scenario_budget = int(_target_draw(scenario_id, 0)) + int(_target_flip(scenario_id, 0))
        coverage_gain = max(0, min(new_cluster_role, cluster_target) - min(current_cluster_role, cluster_target))
        role_saturation_penalty = max(0, new_cluster_role - role_alt_cap)
        ticket_budget_penalty = max(0, new_diff_count - max(1, scenario_budget))
        topology_fit = _portfolio_topology_fit(cluster_name, role_family, scenario_id)
        diversification = (
            float(portfolio_objective_weights.get("diversification_min", 0.55)) * float(new_min - old_min)
            + float(portfolio_objective_weights.get("diversification_avg", 0.18)) * float(new_avg - old_avg)
        )
        cluster_penalty = float(portfolio_objective_weights.get("cluster_penalty", -0.22)) * float(same_cluster_changes)
        correlated_penalty = float(portfolio_objective_weights.get("correlated_penalty", -0.26)) * float(same_family_alt)
        overlap_bonus = (
            float(portfolio_objective_weights.get("overlap_bonus", 0.15))
            if cluster_name == "overlap" and role_family == "swing"
            else 0.0
        )
        rescue_uniqueness = (
            float(portfolio_objective_weights.get("rescue_uniqueness", 0.18))
            if same_family_alt == 0
            else 0.0
        )
        coverage_bonus = float(portfolio_objective_weights.get("coverage_gain", 0.32)) * float(coverage_gain)
        saturation_penalty = float(portfolio_objective_weights.get("saturation_penalty", -0.32)) * float(role_saturation_penalty)
        budget_penalty = float(portfolio_objective_weights.get("ticket_budget_penalty", -0.28)) * float(ticket_budget_penalty)
        score = (
            float(portfolio_objective_weights.get("topology_fit", 1.00)) * topology_fit
            + diversification
            + cluster_penalty
            + correlated_penalty
            + overlap_bonus
            + rescue_uniqueness
            + coverage_bonus
            + saturation_penalty
            + budget_penalty
        )
        reason_parts = [
            f"topology={cluster_name}",
            f"entropy={float(plan.entropy):.3f}",
            f"swing={float(getattr(plan, 'shape_swing_instability', 0.0)):.3f}",
            f"overlap={overlap_subtype}",
            f"diversification=minΔ{new_min-old_min:+.0f}/avgΔ{new_avg-old_avg:+.2f}",
            f"cluster_coverage={same_cluster_changes}",
            f"correlated_fail={same_family_alt}",
            f"role={role_family}",
            f"coverage_gain={coverage_gain}/{cluster_target}",
            f"role_sat={new_cluster_role}/{role_alt_cap}",
            f"ticket_budget={new_diff_count}/{max(1, scenario_budget)}",
            f"topology_fit={topology_fit:+.2f}",
            f"rescue_unique={rescue_uniqueness:+.2f}",
            f"score={score:+.2f}",
        ]
        return score, trial, reason_parts, before

    def _portfolio_final_escape_fix() -> None:
        final_changes: List[str] = []

        def _try_force_change(mi: int, ti: int, alt: str, tag: str) -> bool:
            if ti >= len(tickets) or alt not in {"0", "1", "2"}:
                return False
            if _home_rescue_locked(ti, mi, alt):
                return False
            evaluated = _evaluate_portfolio_objective(mi, ti, alt)
            if evaluated is None:
                return False
            score, cand, reason_parts, before = evaluated
            if score <= 0.0:
                return False
            tickets[ti] = cand
            flip_descs[ti] = f"{flip_descs[ti]}; {tag}:M{plans[mi].match_no:02d}:{before}->{alt}"
            final_changes.append(
                f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->{alt} tag={tag}"
                f" reason={'|'.join(reason_parts)}"
            )
            return True

        def _rank_force_change_candidates(match_index: int, alt: str, candidate_indices: List[int]) -> List[int]:
            ranked: List[Tuple[float, int]] = []
            for ti in candidate_indices:
                evaluated = _evaluate_portfolio_objective(match_index, ti, alt)
                if evaluated is None:
                    continue
                score, _, _, _ = evaluated
                if score <= 0.0:
                    continue
                ranked.append((score, ti))
            ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            return [ti for _, ti in ranked]

        for mi, p in enumerate(plans):
            if p.status != "OK":
                continue
            col = [tickets[ti][mi] for ti in range(min(REQUIRED_TICKET_COUNT, len(tickets)))]
            if not col or len(set(col)) != 1:
                continue
            dominant = str(col[0])
            asym = float(_topology_asymmetry_score(p))
            draw_escape = float(_admission_escape_score(p, "0"))
            if (
                dominant == "2"
                and _cluster_label(p) == "draw_core"
                and asym <= 0.22
                and draw_escape >= 0.70
            ):
                _try_force_change(mi, 7, "0", "final_draw_escape")
            elif (
                dominant == "2"
                and _cluster_label(p) == "draw_core"
                and asym >= 0.28
                and float(p.p_draw) >= 0.34
            ):
                _try_force_change(mi, 3, "0", "final_all_same_draw_diversify")
            elif (
                dominant == "2"
                and _cluster_label(p) == "hold_core"
                and asym >= 0.24
                and float(p.p_draw) >= 0.36
            ):
                _try_force_change(mi, 9, "0", "final_all_same_hold_diversify")
            elif (
                dominant == "2"
                and _cluster_label(p) == "neutral"
                and asym >= 0.50
                and float(p.p_draw) >= 0.20
            ):
                if ticket06_enabled:
                    _try_force_change(mi, 5, "0", "final_all_same_neutral_diversify")
            elif (
                dominant == "0"
                and float(_admission_escape_score(p, "1")) >= 0.30
                and float(_admission_escape_score(p, "1")) >= float(_admission_escape_score(p, "2"))
            ):
                _try_force_change(mi, 6, "1", "final_home_escape")
            elif dominant == "0" and float(_admission_escape_score(p, "2")) >= 0.30:
                _try_force_change(mi, 6, "2", "final_away_escape")

            col_after = [tickets[ti][mi] for ti in range(min(REQUIRED_TICKET_COUNT, len(tickets)))]
            if (
                col_after
                and len(set(col_after)) == 1
                and str(col_after[0]) == "2"
                and float(p.p_draw) >= 0.20
                and asym >= 0.30
            ):
                candidate_tickets: List[int]
                if str(p.league).upper() == "J2" and _cluster_label(p) == "draw_core":
                    candidate_tickets = _exclude_ticket06([ti for ti in range(1, len(tickets)) if ti != 8])
                elif _cluster_label(p) == "neutral":
                    candidate_tickets = _exclude_ticket06([ti for ti in range(1, len(tickets)) if ti != 8])
                else:
                    candidate_tickets = _exclude_ticket06(list(range(1, len(tickets))))
                for ti in _rank_force_change_candidates(mi, "0", candidate_tickets):
                    if _try_force_change(mi, ti, "0", "final_all_same_second_diversify"):
                        break

        stats["portfolio_final_escape_fix_count"] = int(len(final_changes))
        stats["portfolio_final_escape_fix_matches"] = "; ".join(final_changes[:30])
        if final_changes:
            _warn(warnings, f"portfolio_final_escape_fix={stats['portfolio_final_escape_fix_matches']}")

    _portfolio_final_escape_fix()

    def _rebalance_ticket09_escape() -> None:
        rebalance_changes: List[str] = []
        if len(tickets) < 9:
            return

        for mi, p in enumerate(plans):
            if p.status != "OK":
                continue
            if tickets[8][mi] != "0":
                continue

            cluster_name = _cluster_label(p)
            target_candidates: List[int] = []
            if str(p.league).upper() == "J2" and cluster_name == "draw_core":
                target_candidates = _exclude_ticket06([5, 9, 3, 1])  # ticket06 -> ticket10 -> ticket04 -> ticket02
            elif cluster_name == "neutral":
                target_candidates = _exclude_ticket06([3, 9, 5, 1])  # ticket04 -> ticket10 -> ticket06 -> ticket02
            elif cluster_name == "hold_core":
                target_candidates = [9, 3, 1]  # ticket10 -> ticket04 -> ticket02
            elif cluster_name != "low_tempo_draw":
                target_candidates = _exclude_ticket06([9, 3, 1])  # keep ticket09 for genuine low-tempo only, do not degrade base ticket
            else:
                continue

            ranked_candidates: List[Tuple[float, int, List[str], List[str], List[str], str]] = []
            for cand_ti in target_candidates:
                evaluated = _evaluate_portfolio_objective(mi, cand_ti, "0")
                if evaluated is None:
                    continue
                score, cand_target, reason_parts, before_target = evaluated
                ranked_candidates.append((score, cand_ti, cand_target, reason_parts, [], before_target))
            if not ranked_candidates:
                continue
            ranked_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            best_score, target_ti, cand_target, reason_parts, _, before_target = ranked_candidates[0]
            if best_score <= 0.0:
                continue

            before_09 = tickets[8][mi]
            cand_09 = list(tickets[8])
            if _home_rescue_locked(8, mi, "2"):
                continue
            cand_09[mi] = "2"
            cand_09_key = _ticket_key(cand_09)
            if any(cand_09_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != 8):
                continue

            tickets[target_ti] = cand_target
            tickets[8] = cand_09
            flip_descs[target_ti] = f"{flip_descs[target_ti]}; rebalance_from09:M{p.match_no:02d}:{before_target}->0"
            flip_descs[8] = f"{flip_descs[8]}; rebalance_to_core:M{p.match_no:02d}:{before_09}->2"
            rebalance_changes.append(
                f"M{p.match_no:02d} ticket{target_ti+1:02d} {before_target}->0 / ticket09 {before_09}->2"
                f" reason={'|'.join(reason_parts)}"
            )

        stats["ticket09_rebalance_count"] = int(len(rebalance_changes))
        stats["ticket09_rebalance_matches"] = "; ".join(rebalance_changes[:30])
        if rebalance_changes:
            _warn(warnings, f"ticket09_rebalance={stats['ticket09_rebalance_matches']}")

    _rebalance_ticket09_escape()

    def _finalize_draw_origin_home_mix() -> None:
        final_mix_changes: List[str] = []
        target_ticket_indices = [9, 2, 8, 4]  # ticket10, ticket03, ticket09, ticket05

        def _eligible_draw_origin_home_mix(plan: MatchPlan) -> bool:
            primary_sym = _normalize_symbol_token(plan.context_primary_pick) or _normalize_symbol_token(plan.best)
            secondary_sym = _normalize_symbol_token(plan.context_secondary_pick) or _normalize_symbol_token(plan.second)
            cluster_name = _cluster_label(plan)
            asym = float(_topology_asymmetry_score(plan))
            home_escape = float(_admission_escape_score(plan, "1"))
            away_escape = float(_admission_escape_score(plan, "2"))
            pressure = _safe_text(plan.lab_pressure_band, "").lower()
            control = _safe_text(plan.lab_control_band, "").lower()
            basis_hint = _safe_text(getattr(plan, "lab_basis_hint", ""), "").lower()
            side_gap = abs(float(plan.p_home) - float(plan.p_away))
            if str(plan.league).upper() != "J1":
                return False
            if primary_sym != "0" or secondary_sym != "2":
                return False
            if cluster_name == "draw_core":
                if pressure != "neutral" or control != "neutral":
                    return False
                return bool(
                    float(plan.p_draw) >= 0.43
                    and side_gap <= 0.06
                    and asym <= 0.18
                    and home_escape >= 0.24
                    and home_escape >= (0.82 * away_escape)
                )
            if cluster_name == "low_tempo_draw":
                if control not in {"neutral", "home"}:
                    return False
                if pressure not in {"neutral", "home"}:
                    return False
                return bool(
                    (
                        (
                            basis_hint in {"", "flat_draw_trap"}
                            and float(plan.p_draw) >= 0.40
                            and side_gap <= 0.06
                            and asym <= 0.10
                            and home_escape >= 0.24
                            and home_escape >= (0.90 * away_escape)
                        )
                        or (
                            basis_hint in {"", "draw_compressed"}
                            and float(plan.p_draw) >= 0.40
                            and side_gap <= 0.06
                            and asym <= 0.10
                            and home_escape >= 0.25
                            and home_escape >= (0.95 * away_escape)
                        )
                    )
                )
            if cluster_name == "hold_core" and control == "neutral" and pressure == "home":
                return bool(
                    float(plan.p_draw) >= 0.38
                    and side_gap <= 0.02
                    and asym <= 0.12
                    and home_escape >= 0.24
                    and home_escape >= away_escape
                )
            return False

        eligible_matches = sorted(
            [
                i
                for i in ok_indices
                if _eligible_draw_origin_home_mix(plans[i])
            ],
            key=lambda i: (
                -float(plans[i].p_draw),
                float(_topology_asymmetry_score(plans[i])),
                -float(_admission_escape_score(plans[i], "1")),
                i,
            ),
        )

        for mi in eligible_matches:
            col = [tickets[ti][mi] for ti in range(min(REQUIRED_TICKET_COUNT, len(tickets)))]
            a_count = sum(1 for sym in col if str(sym) == "2")
            h_count = sum(1 for sym in col if str(sym) == "1")
            target_h_count = min(3, max(h_count, int(math.ceil(a_count * 0.35))))
            if h_count >= target_h_count:
                continue

            for ti in target_ticket_indices:
                if ti >= len(tickets):
                    continue
                before = tickets[ti][mi]
                if before != "2":
                    continue
                cand = list(tickets[ti])
                cand[mi] = "1"
                cand_key = _ticket_key(cand)
                if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                    continue
                tickets[ti] = cand
                protected_home_rescue_slots.add((ti, mi))
                flip_descs[ti] = f"{flip_descs[ti]}; final_draw_origin_home_mix:M{plans[mi].match_no:02d}:{before}->1"
                final_mix_changes.append(
                    f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->1 target_h={target_h_count}"
                )
                h_count += 1
                if h_count >= target_h_count:
                    break

        stats["final_draw_origin_home_mix_count"] = int(len(final_mix_changes))
        stats["final_draw_origin_home_mix_matches"] = "; ".join(final_mix_changes[:30])
        if final_mix_changes:
            _warn(
                warnings,
                    f"final_draw_origin_home_mix={stats['final_draw_origin_home_mix_matches']}",
                )

    _finalize_draw_origin_home_mix()

    def _portfolio_force_unique_objective(max_rounds: int = 4) -> None:
        repair_changes: List[str] = []
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            changed_any = False
            seen: Dict[str, int] = {}
            duplicate_target: Optional[int] = None
            for ti, ticket in enumerate(tickets):
                key = _ticket_key(ticket)
                if key in seen:
                    duplicate_target = ti
                    break
                seen[key] = ti
            if duplicate_target is None:
                break

            ti = duplicate_target
            scenario_id = scenario_defs[ti].scenario_id if ti < len(scenario_defs) else f"{ti+1:02d}"
            candidate_moves: List[Tuple[float, int, str, str]] = []
            for mi, p in enumerate(plans):
                if p.status != "OK":
                    continue
                for alt in _candidate_alt_order(p, scenario_id):
                    if alt == tickets[ti][mi]:
                        continue
                    if _home_rescue_locked(ti, mi, alt):
                        continue
                    if alt == "0" and not _draw_ready(p):
                        continue
                    relaxed_unique_gate = bool(scenario_id == "06" and not ticket06_enabled)
                    if (not relaxed_unique_gate) and not _allow_post_pass_change(p, alt, scenario_id, phase="force_unique_objective"):
                        continue
                    evaluated = _evaluate_portfolio_objective(mi, ti, alt)
                    if evaluated is None:
                        continue
                    score, cand, _, before = evaluated
                    if score <= 0.0:
                        continue
                    cand_key = _ticket_key(cand)
                    if cand_key in seen:
                        continue
                    candidate_moves.append((score, mi, alt, before))
            if not candidate_moves:
                break
            candidate_moves.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            score, mi, alt, before = candidate_moves[0]
            tickets[ti][mi] = alt
            flip_descs[ti] = f"{flip_descs[ti]}; force_unique_objective:M{plans[mi].match_no:02d}:{before}->{alt}"
            repair_changes.append(
                f"ticket={ti+1:02d} M{plans[mi].match_no:02d} {before}->{alt} score={score:+.2f}"
            )
            changed_any = True
            if not changed_any:
                break

        stats["portfolio_force_unique_count"] = int(len(repair_changes))
        stats["portfolio_force_unique_matches"] = "; ".join(repair_changes[:30])
        if repair_changes:
            _warn(warnings, f"portfolio_force_unique={stats['portfolio_force_unique_matches']}")

    _portfolio_force_unique_objective()

    def _enforce_purchase_context_minimums() -> None:
        context_changes: List[str] = []

        def _target_cover_specs(plan: MatchPlan) -> List[Tuple[str, int, float]]:
            policy = str(getattr(plan, "context_admission_policy", "") or "").strip().lower()
            tier = str(getattr(plan, "context_draw_purchase_tier", "") or "").strip().lower()
            guidance = str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()
            rank1 = _normalize_symbol_token(getattr(plan, "context_rank1_pick", ""))
            rank2 = _normalize_symbol_token(getattr(plan, "context_rank2_pick", ""))
            rank3 = _normalize_symbol_token(getattr(plan, "context_rank3_pick", ""))
            specs: List[Tuple[str, int, float]] = []
            if policy == "rank1_draw_rescue" or tier == "rescue" or guidance == "draw_rescue_split":
                if rank2 and rank2 != rank1:
                    specs.append((rank2, 3, 0.90))
                if rank3 and rank3 not in {rank1, rank2}:
                    specs.append((rank3, 1, 0.78))
                return specs
            if policy == "rank1_draw_watch" or tier == "watch" or guidance == "draw_watch_split":
                if rank2 and rank2 != rank1:
                    specs.append((rank2, 3, 0.95))
                if rank3 and rank3 not in {rank1, rank2}:
                    specs.append((rank3, 1, 0.82))
                return specs
            if policy == "rank1_draw_core":
                if rank2 and rank2 != rank1:
                    specs.append((rank2, 2, 0.82))
                if rank3 and rank3 not in {rank1, rank2}:
                    specs.append((rank3, 1, 0.70))
                return specs
            if guidance == "side_with_cover":
                secondary = _normalize_symbol_token(getattr(plan, "context_secondary_pick", ""))
                primary = _normalize_symbol_token(getattr(plan, "context_primary_pick", "")) or rank1
                if secondary and secondary != primary:
                    specs.append((secondary, 1, 0.75))
            return specs

        for mi, plan in enumerate(plans):
            if plan.status != "OK":
                continue
            candidate_tickets = _exclude_ticket06(list(range(1, len(tickets))))
            for target_sym, target_count, bonus in _target_cover_specs(plan):
                if target_sym not in {"0", "1", "2"}:
                    continue
                current_count = sum(1 for ti in range(len(tickets)) if tickets[ti][mi] == target_sym)
                if current_count >= target_count:
                    continue

                needed = target_count - current_count
                for _ in range(needed):
                    ranked: List[Tuple[float, int, List[str], str]] = []
                    for ti in candidate_tickets:
                        if tickets[ti][mi] == target_sym:
                            continue
                        evaluated = _evaluate_portfolio_objective(mi, ti, target_sym)
                        if evaluated is None:
                            continue
                        score, cand, reason_parts, before = evaluated
                        adjusted = score + bonus
                        if adjusted <= 0.0:
                            continue
                        ranked.append((adjusted, ti, cand, before, "|".join(reason_parts)))
                    if not ranked:
                        break
                    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                    adjusted, ti, cand, before, reason = ranked[0]
                    tickets[ti] = cand
                    flip_descs[ti] = f"{flip_descs[ti]}; context_minimum:M{plan.match_no:02d}:{before}->{target_sym}"
                    context_changes.append(
                        f"ticket={ti+1:02d} M{plan.match_no:02d} {before}->{target_sym} "
                        f"target={target_count} adjusted={adjusted:+.2f} context={plan.context_ticket_guidance} "
                        f"policy={plan.context_admission_policy} reason={reason}"
                    )
                    current_count += 1
                    if current_count >= target_count:
                        break

        stats["context_minimum_count"] = int(len(context_changes))
        stats["context_minimum_matches"] = "; ".join(context_changes[:30])
        if context_changes:
            _warn(warnings, f"context_minimum={stats['context_minimum_matches']}")

    _enforce_purchase_context_minimums()

    def _enforce_per_match_admission_caps() -> None:
        cap_changes: List[str] = []

        def _policy_caps(plan: MatchPlan) -> Optional[Tuple[int, int, int]]:
            policy = str(getattr(plan, "context_admission_policy", "") or "").strip().lower()
            if policy == "rank1_draw_rescue":
                return (6, 2, 1)
            if policy == "rank1_draw_watch":
                return (4, 3, 1)
            if policy == "rank1_draw_core":
                return (7, 2, 0)
            if policy == "rank2_draw_cut":
                return (2, 0, 0)
            return None

        for mi, plan in enumerate(plans):
            if plan.status != "OK":
                continue
            caps = _policy_caps(plan)
            if not caps:
                continue
            draw_cap, rank2_min, rank3_min = caps
            rank1 = _normalize_symbol_token(getattr(plan, "context_rank1_pick", ""))
            rank2 = _normalize_symbol_token(getattr(plan, "context_rank2_pick", ""))
            rank3 = _normalize_symbol_token(getattr(plan, "context_rank3_pick", ""))
            if rank1 != "0":
                continue

            draw_count = sum(1 for ti in range(len(tickets)) if tickets[ti][mi] == "0")
            rank2_count = sum(1 for ti in range(len(tickets)) if rank2 and tickets[ti][mi] == rank2)
            rank3_count = sum(1 for ti in range(len(tickets)) if rank3 and tickets[ti][mi] == rank3)

            def _candidate_tickets() -> List[int]:
                order = [9, 4, 6, 3, 7, 2, 8, 5, 1, 0]
                return [ti for ti in order if ti < len(tickets)]

            for ti in _candidate_tickets():
                if draw_count <= draw_cap and rank2_count >= rank2_min and rank3_count >= rank3_min:
                    break
                if tickets[ti][mi] != "0":
                    continue
                replacement = ""
                if rank2_count < rank2_min and rank2 and rank2 != "0":
                    replacement = rank2
                    rank2_count += 1
                elif rank3_count < rank3_min and rank3 and rank3 not in {"", "0", rank2}:
                    replacement = rank3
                    rank3_count += 1
                elif draw_count > draw_cap and rank2 and rank2 != "0":
                    replacement = rank2
                    rank2_count += 1
                elif draw_count > draw_cap and rank3 and rank3 not in {"", "0", rank2}:
                    replacement = rank3
                    rank3_count += 1
                if replacement not in {"1", "2"}:
                    continue
                before = tickets[ti][mi]
                tickets[ti][mi] = replacement
                draw_count -= 1
                flip_descs[ti] = f"{flip_descs[ti]}; admission_cap:M{plan.match_no:02d}:{before}->{replacement}"
                cap_changes.append(
                    f"ticket={ti+1:02d} M{plan.match_no:02d} {before}->{replacement} "
                    f"policy={plan.context_admission_policy} cap={draw_cap} r2min={rank2_min} r3min={rank3_min}"
                )

        stats["admission_cap_count"] = int(len(cap_changes))
        stats["admission_cap_matches"] = "; ".join(cap_changes[:30])
        if cap_changes:
            _warn(warnings, f"admission_cap={stats['admission_cap_matches']}")

    _enforce_per_match_admission_caps()

    def _enforce_side_watch_hedges() -> None:
        context_changes: List[str] = []
        side_watch_tickets = _exclude_ticket06([1, 2, 4, 6, 7, 8, 9])

        for mi, plan in enumerate(plans):
            if plan.status != "OK":
                continue
            purchase_type = str(getattr(plan, "context_match_purchase_type", "") or "").strip().lower()
            guidance = str(getattr(plan, "context_ticket_guidance", "") or "").strip().lower()
            primary_sym = _normalize_symbol_token(getattr(plan, "context_primary_pick", ""))
            secondary_sym = _normalize_symbol_token(getattr(plan, "context_secondary_pick", ""))
            third_sym = _normalize_symbol_token(getattr(plan, "third", ""))
            base_sym = _normalize_symbol_token(getattr(plan, "base_pick", ""))
            if purchase_type != "chaos_draw_watch" or guidance != "side_single" or primary_sym != "2":
                continue

            hedge_targets: List[str] = []
            if secondary_sym in {"0", "1"} and secondary_sym != primary_sym and secondary_sym != base_sym:
                hedge_targets.append(secondary_sym)
            if third_sym in {"0", "1"} and third_sym != primary_sym and third_sym not in hedge_targets:
                hedge_targets.append(third_sym)
            if not hedge_targets:
                derived_opposite = "1" if primary_sym == "2" else "2"
                if derived_opposite != base_sym:
                    hedge_targets.append(derived_opposite)
            hedge_targets = hedge_targets[:2]
            if not hedge_targets:
                continue

            for target_sym in hedge_targets:
                current_target = sum(1 for ti in range(len(tickets)) if tickets[ti][mi] == target_sym)
                if current_target >= 1:
                    continue
                ranked: List[Tuple[float, int, List[str], str, str]] = []
                for ti in side_watch_tickets:
                    if ti >= len(tickets) or tickets[ti][mi] == target_sym:
                        continue
                    evaluated = _evaluate_portfolio_objective(mi, ti, target_sym)
                    if evaluated is None:
                        continue
                    score, cand, reason_parts, before = evaluated
                    adjusted = score + 0.65
                    if adjusted <= 0.0:
                        continue
                    ranked.append((adjusted, ti, cand, before, "|".join(reason_parts)))
                if not ranked:
                    continue
                ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                adjusted, ti, cand, before, reason = ranked[0]
                tickets[ti] = cand
                flip_descs[ti] = f"{flip_descs[ti]}; side_watch_hedge:M{plan.match_no:02d}:{before}->{target_sym}"
                context_changes.append(
                    f"ticket={ti+1:02d} M{plan.match_no:02d} {before}->{target_sym} "
                    f"target={target_sym} adjusted={adjusted:+.2f} context={plan.context_ticket_guidance} reason={reason}"
                )

        stats["side_watch_hedge_count"] = int(len(context_changes))
        stats["side_watch_hedge_matches"] = "; ".join(context_changes[:30])
        if context_changes:
            _warn(warnings, f"side_watch_hedge={stats['side_watch_hedge_matches']}")

    _enforce_side_watch_hedges()

    def _purchase_type_main_symbol(plan: MatchPlan) -> str:
        primary = _normalize_symbol_token(plan.context_primary_pick)
        if primary in {"1", "2"}:
            return primary
        if str(plan.best) in {"1", "2"}:
            return str(plan.best)
        return "1" if float(plan.p_home) >= float(plan.p_away) else "2"

    def _log_purchase_type_policy(
        policy_logs: List[str],
        plan: MatchPlan,
        policy_action: str,
        base_symbol: str,
        new_symbol: str,
        affected_tickets: List[int],
        change_cap: int,
        reason: str,
    ) -> None:
        if not affected_tickets:
            return
        purchase_type = str(getattr(plan, "context_match_purchase_type", "") or "").strip()
        policy_logs.append(
            f"purchase_type={purchase_type} policy_action={policy_action} "
            f"base_symbol={base_symbol} new_symbol={new_symbol} "
            f"affected_tickets={','.join(f'{ti+1:02d}' for ti in affected_tickets)} "
            f"change_cap={change_cap} reason={reason} "
            f"rescue_count=0 harm_count=0 net_rescue=0"
        )

    def _apply_purchase_type_policies() -> None:
        policy_logs: List[str] = []
        home_reversal_targets = [9, 2, 8, 4]  # ticket10, ticket03, ticket09, ticket05
        j2_draw_tickets = _exclude_ticket06([1, 4, 5, 7, 8])  # 02,05,06,08,09
        draw_trap_tickets = _exclude_ticket06([1, 4, 5, 7, 8])  # draw-family support only
        j2_main_tickets = _exclude_ticket06([1, 3, 4, 6, 7, 8, 9])  # non-base main/support

        for mi, plan in enumerate(plans):
            if plan.status != "OK":
                continue
            purchase_type = str(getattr(plan, "context_match_purchase_type", "") or "").strip().lower()
            if not purchase_type:
                continue

            if purchase_type == "home_reversal_watch":
                if _purchase_type_main_symbol(plan) != "2":
                    continue
                confidence = str(getattr(plan, "context_purchase_type_confidence", "") or "").strip().lower()
                change_cap = 4 if confidence == "high" else 3 if confidence == "medium" else 2
                changed_tickets: List[int] = []
                for ti in home_reversal_targets:
                    if len(changed_tickets) >= change_cap:
                        break
                    if ti >= len(tickets) or ti == 0:
                        continue
                    before = tickets[ti][mi]
                    if before != "2":
                        continue
                    cand = list(tickets[ti])
                    cand[mi] = "1"
                    cand_key = _ticket_key(cand)
                    if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                        continue
                    tickets[ti] = cand
                    protected_home_rescue_slots.add((ti, mi))
                    flip_descs[ti] = f"{flip_descs[ti]}; policy_home_reversal_watch:M{plan.match_no:02d}:{before}->1"
                    changed_tickets.append(ti)
                _log_purchase_type_policy(
                    policy_logs,
                    plan,
                    "hedge_home",
                    "2",
                    "1",
                    changed_tickets,
                    change_cap,
                    "away_overread_guard",
                )

            elif purchase_type == "j2_draw_trap":
                if str(plan.league).upper() != "J2":
                    continue
                main_symbol = _purchase_type_main_symbol(plan)
                changed_tickets: List[int] = []
                draw_count = sum(1 for ti in range(len(tickets)) if tickets[ti][mi] == "0")
                max_draw_keep = 1 if str(plan.base_pick) == "0" else 0
                for ti in j2_draw_tickets:
                    if draw_count <= max_draw_keep:
                        break
                    if ti >= len(tickets) or ti == 0:
                        continue
                    before = tickets[ti][mi]
                    if before != "0":
                        continue
                    cand = list(tickets[ti])
                    cand[mi] = main_symbol
                    cand_key = _ticket_key(cand)
                    if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                        continue
                    tickets[ti] = cand
                    flip_descs[ti] = f"{flip_descs[ti]}; policy_j2_draw_trap:M{plan.match_no:02d}:{before}->{main_symbol}"
                    changed_tickets.append(ti)
                    draw_count -= 1
                _log_purchase_type_policy(
                    policy_logs,
                    plan,
                    "suppress_draw",
                    "0",
                    main_symbol,
                    changed_tickets,
                    max(0, draw_count - max_draw_keep + len(changed_tickets)),
                    "false_draw_suppression",
                )

            elif purchase_type == "j2_directional_stall":
                # Held out for now; current backtest did not show positive net rescue.
                continue

        stats["purchase_type_policy_count"] = int(len(policy_logs))
        stats["purchase_type_policy_matches"] = "; ".join(policy_logs[:30])
        if policy_logs:
            _warn(warnings, f"purchase_type_policy={stats['purchase_type_policy_matches']}")

    _apply_purchase_type_policies()
    _portfolio_force_unique_objective(max_rounds=6)

    # Post-pass uniqueness/spread can change tickets after the initial scenario log.
    # Recompute scenario-facing counters from the final ticket state and emit a final
    # note when the end state diverged from the first-pass scenario output.
    for ti, ticket in enumerate(tickets):
        scenario_idx = ti + 1
        scenario_id = f"{scenario_idx:02d}"
        final_draw_n = _symbol_count(ticket, "0")
        final_diff_n = int(sum(1 for a, b in zip(ticket, base_ticket) if a != b))
        prev_diff_n = int(stats.get(f"ticket_{scenario_id}_diff_count", 0))
        stats[f"ticket_{scenario_id}_draw_count"] = final_draw_n
        stats[f"ticket_{scenario_id}_diff_count"] = final_diff_n
        stats[f"ticket_{scenario_id}_shape_change_count"] = final_diff_n
        stats[f"ticket_{scenario_id}_lab_change_count"] = final_diff_n
        if ti == 0:
            continue
        if final_diff_n == prev_diff_n:
            continue
        final_logs = [
            f"M{plans[mi].match_no:02d}:{base_ticket[mi]}->{ticket[mi]}"
            for mi in range(len(plans))
            if ticket[mi] != base_ticket[mi]
        ]
        _warn(
            warnings,
            f"[BUYPLAN_SCENARIO_TICKET_FINAL] ticket={scenario_id} "
            f"name={scenario_defs[ti].scenario_name if ti < len(scenario_defs) else scenario_id} "
            f"shape_changes={final_diff_n} draw_count={final_draw_n} diff_count={final_diff_n} "
            f"changes={'; '.join(final_logs) if final_logs else 'none'}",
        )

    total_cells = len(tickets) * len(plans) if tickets and plans else 0
    stats["unique_ticket_count"] = len({_ticket_key(t) for t in tickets})
    stats["duplicate_count"] = max(0, len(tickets) - int(stats["unique_ticket_count"]))
    pairwise_distances = [
        _ticket_distance(tickets[i], tickets[j])
        for i in range(len(tickets))
        for j in range(i + 1, len(tickets))
    ]
    stats["min_ticket_distance"] = min(pairwise_distances) if pairwise_distances else 0
    stats["avg_ticket_distance"] = round(float(sum(pairwise_distances)) / float(len(pairwise_distances)), 3) if pairwise_distances else 0.0
    stats["total_one_count"] = sum(_symbol_count(t, "1") for t in tickets)
    stats["total_zero_count"] = sum(_symbol_count(t, "0") for t in tickets)
    stats["total_two_count"] = sum(_symbol_count(t, "2") for t in tickets)
    stats["total_ratio_1"] = _ratio_str(int(stats["total_one_count"]), total_cells)
    stats["total_ratio_0"] = _ratio_str(int(stats["total_zero_count"]), total_cells)
    stats["total_ratio_2"] = _ratio_str(int(stats["total_two_count"]), total_cells)
    stats["min_draw_insert_count"] = 0
    stats["min_draw_insert_logs"] = "not_used_scenario_weights_mode"

    for idx, ticket in enumerate(tickets, start=1):
        draw_n = _symbol_count(ticket, "0")
        diff_n = sum(1 for a, b in zip(ticket, base_ticket) if a != b)
        _warn(warnings, f"ticket{idx:02d}: D本数={draw_n} 差分件数={diff_n}")

    draw_family_ticket_ids = ["02", "05", "06", "08", "09"]
    draw_family_indexes = [int(sid) - 1 for sid in draw_family_ticket_ids if 0 < int(sid) <= len(tickets)]
    if len(draw_family_indexes) >= 2 and len(tickets) >= 6:
        ref_idx = 5  # ticket06
        ref_ticket = tickets[ref_idx]
        family_peers = [idx for idx in draw_family_indexes if idx != ref_idx]
        role_overlap_parts: List[str] = []
        unique_only_count = 0
        for peer_idx in family_peers:
            peer_ticket = tickets[peer_idx]
            same_change = 0
            diff_change = 0
            for m_idx, base_symbol in enumerate(base_ticket):
                ref_symbol = ref_ticket[m_idx]
                peer_symbol = peer_ticket[m_idx]
                ref_changed = ref_symbol != base_symbol
                peer_changed = peer_symbol != base_symbol
                if ref_changed and peer_changed:
                    if ref_symbol == peer_symbol:
                        same_change += 1
                    else:
                        diff_change += 1
            role_overlap_parts.append(
                f"06~{peer_idx+1:02d}:same={same_change},diff={diff_change}"
            )
        for m_idx, base_symbol in enumerate(base_ticket):
            ref_symbol = ref_ticket[m_idx]
            if ref_symbol == base_symbol:
                continue
            if all(tickets[peer_idx][m_idx] == base_symbol for peer_idx in family_peers):
                unique_only_count += 1
        stats["ticket_06_role_unique_only_count"] = int(unique_only_count)
        stats["ticket_06_role_overlap_summary"] = " / ".join(role_overlap_parts)
        _warn(
            warnings,
            "ticket06_role="
            f"unique_only={unique_only_count} / "
            f"{stats['ticket_06_role_overlap_summary']}",
        )

    _warn(warnings, "buyplan mode: scenario_weights_direct")
    _warn(warnings, f"unique_ticket_count={stats['unique_ticket_count']} duplicate_count={stats['duplicate_count']}")
    _warn(warnings, f"ticket_distance min={stats['min_ticket_distance']} avg={stats['avg_ticket_distance']}")
    _warn(warnings, f"0総出現回数={stats['total_zero_count']} 1/0/2比率={stats['total_ratio_1']}/{stats['total_ratio_0']}/{stats['total_ratio_2']}")

    scenario_result_labels: List[List[str]] = []
    for t in tickets:
        scenario_result_labels.append([_result_label_from_symbol(x) for x in t])

    return tickets[:REQUIRED_TICKET_COUNT], flip_descs[:REQUIRED_TICKET_COUNT], scenario_defs[:REQUIRED_TICKET_COUNT], stats, scenario_result_labels


def _generate_tickets_by_world_allocator(
    df: pd.DataFrame,
    plans: List[MatchPlan],
    warnings: List[str],
) -> Tuple[List[List[str]], List[str], List[ScenarioDef], Dict[str, int], List[List[str]]]:
    scenario_defs: List[ScenarioDef] = [
        ScenarioDef("01", "Main Anchor", "main_anchor_world"),
        ScenarioDef("02", "Safe Main", "safe_main_world"),
        ScenarioDef("03", "Draw Pressure", "draw_pressure_world"),
        ScenarioDef("04", "Hold Guard", "hold_guard_world"),
        ScenarioDef("05", "Controlled Upset", "controlled_upset_world"),
        ScenarioDef("06", "J2 Context", "j2_context_world"),
        ScenarioDef("07", "Side Hedge", "side_hedge_world"),
        ScenarioDef("08", "Close Draw", "close_draw_world"),
        ScenarioDef("09", "Upset", "upset_world"),
        ScenarioDef("10", "Wide", "wide_world"),
    ]
    stats: Dict[str, int | float | str] = {"allocator_engine": "world_topology_v1"}
    ok_indices = [i for i, p in enumerate(plans) if p.status == "OK"]

    def _rank_triplet(plan: MatchPlan) -> List[str]:
        ranks = [
            _normalize_symbol_token(plan.context_rank1_pick),
            _normalize_symbol_token(plan.context_rank2_pick),
            _normalize_symbol_token(plan.context_rank3_pick),
        ]
        if len({r for r in ranks if r in SYMBOLS}) == 3:
            return [r for r in ranks if r in SYMBOLS]
        fallback = [str(plan.best), str(plan.second), str(plan.third)]
        ordered: List[str] = []
        for sym in ranks + fallback + SYMBOLS:
            if sym in SYMBOLS and sym not in ordered:
                ordered.append(sym)
        return ordered[:3]

    def _prob_map(plan: MatchPlan) -> Dict[str, float]:
        return {"1": float(plan.p_home), "0": float(plan.p_draw), "2": float(plan.p_away)}

    def _rank_prob(plan: MatchPlan, sym: str) -> float:
        return float(_prob_map(plan).get(str(sym), 0.0))

    def _top_gap(plan: MatchPlan) -> float:
        ranks = _rank_triplet(plan)
        return float(_rank_prob(plan, ranks[0]) - _rank_prob(plan, ranks[1]))

    def _draw_like_score(plan: MatchPlan) -> float:
        ranks = _rank_triplet(plan)
        draw_rank_bonus = 0.20 if ranks[0] == "0" else 0.12 if "0" in ranks[:2] else 0.0
        return min(
            1.0,
            0.42 * float(plan.p_draw)
            + 0.24 * float(plan.lab_draw_tension_score)
            + 0.18 * float(plan.lab_stall_compactness_score)
            + 0.10 * float(plan.lab_mix_draw)
            + 0.06 * float(plan.lab_scenario_entropy_score)
            + draw_rank_bonus,
        )

    def _hold_score(plan: MatchPlan) -> float:
        return min(
            1.0,
            0.34 * float(plan.lab_hold_weight)
            + 0.18 * float(plan.lab_hold_foundation_score)
            + 0.16 * (1.0 - float(plan.lab_scenario_entropy_score))
            + 0.16 * (1.0 - float(plan.lab_volatility_score))
            + 0.16 * max(float(plan.p_home), float(plan.p_away)),
        )

    def _upset_score(plan: MatchPlan) -> float:
        ranks = _rank_triplet(plan)
        first_prob = _rank_prob(plan, ranks[0])
        second_prob = _rank_prob(plan, ranks[1])
        weak_first = 1.0 if 0.45 <= first_prob < 0.60 else 0.55 if 0.60 <= first_prob < 0.67 else 0.15
        return min(
            1.0,
            0.30 * weak_first
            + 0.24 * float(plan.lab_volatility_score)
            + 0.20 * float(plan.lab_scenario_entropy_score)
            + 0.16 * second_prob
            + 0.10 * (1.0 - max(0.0, first_prob - second_prob)),
        )

    def _wide_score(plan: MatchPlan) -> float:
        ranks = _rank_triplet(plan)
        first_prob = _rank_prob(plan, ranks[0])
        second_prob = _rank_prob(plan, ranks[1])
        return min(
            1.0,
            0.28 * float(plan.lab_scenario_entropy_score)
            + 0.22 * float(plan.lab_volatility_score)
            + 0.20 * (1.0 - max(0.0, first_prob - second_prob))
            + 0.18 * float(plan.lab_flip_weight)
            + 0.12 * float(plan.lab_dynamic_swing_score),
        )

    def _side_hedge_score(plan: MatchPlan) -> float:
        ranks = _rank_triplet(plan)
        if ranks[0] not in {"1", "2"}:
            return 0.0
        side_gap = abs(float(plan.p_home) - float(plan.p_away))
        purchase_type = str(plan.context_match_purchase_type or "").strip().lower()
        bonus = 0.14 if purchase_type in {"home_reversal_watch", "chaos_side_watch", "away_reversal_watch"} else 0.0
        return min(
            1.0,
            0.28 * float(plan.lab_scenario_entropy_score)
            + 0.24 * float(plan.lab_volatility_score)
            + 0.18 * float(plan.lab_flip_dislocation_score)
            + 0.18 * (1.0 - side_gap)
            + 0.12 * float(plan.lab_dynamic_swing_score)
            + bonus,
        )

    def _first_second_third_symbol(plan: MatchPlan, rank_idx: int) -> str:
        ranks = _rank_triplet(plan)
        return ranks[min(max(rank_idx, 0), 2)]

    def _preferred_side_hedge_symbol(plan: MatchPlan) -> str:
        ranks = _rank_triplet(plan)
        main = ranks[0]
        for sym in ranks[1:]:
            if main in {"1", "2"} and sym in {"1", "2"} and sym != main:
                return sym
        return ranks[1]

    def _draw_admission_score(plan: MatchPlan) -> float:
        rank1 = _first_second_third_symbol(plan, 0)
        rank2 = _first_second_third_symbol(plan, 1)
        prob_draw = float(plan.p_draw or 0.0)
        draw_core = float(getattr(plan, "context_draw_core_score", 0.0) or 0.0)
        confidence = str(getattr(plan, "context_purchase_type_confidence", "") or "").strip().lower()
        confidence_bonus = 0.05 if confidence == "high" else 0.025 if confidence == "medium" else 0.0
        draw_type = str(getattr(plan, "context_match_purchase_type", "") or "").strip().lower()
        type_bonus = 0.04 if draw_type in {"j1_draw_rescue_watch", "draw_trap"} else 0.02 if draw_type in {"j2_false_draw_watch", "j2_draw_trap"} else 0.0
        top_gap = _top_gap(plan)
        rank_gap = max(0.0, _rank_prob(plan, rank1) - _rank_prob(plan, rank2))
        return float(prob_draw + 0.25 * draw_core + confidence_bonus + type_bonus - 0.15 * top_gap - 0.10 * rank_gap)

    def _apply_pick(
        ticket: List[str],
        plan_idx: int,
        symbol: str,
        reason: str,
        counts: Dict[str, int],
        reason_log: List[str],
        first_prob_breaks: Dict[str, int],
    ) -> None:
        plan = plans[plan_idx]
        ranks = _rank_triplet(plan)
        current = ticket[plan_idx]
        if symbol == current:
            return
        ticket[plan_idx] = symbol
        if symbol == ranks[0]:
            counts["first"] += 1
        elif symbol == ranks[1]:
            counts["second"] += 1
        else:
            counts["third"] += 1
        first_prob = _rank_prob(plan, ranks[0])
        if symbol != ranks[0]:
            if first_prob >= 0.60:
                first_prob_breaks["strong"] += 1
            elif first_prob >= 0.50:
                first_prob_breaks["weak"] += 1
        reason_log.append(f"M{plan.match_no:02d}:{current}->{symbol}:{reason}")

    def _enforce_world_draw_limits(
        ticket: List[str],
        world: str,
        counts: Dict[str, int],
        reason_log: List[str],
        breaks: Dict[str, int],
    ) -> None:
        if world == "main_anchor_world":
            draw_keep_cap = 6
            rank2_min = 3
            rank3_min = 0
        elif world == "safe_main_world":
            draw_keep_cap = 5
            rank2_min = 4
            rank3_min = 0
        elif world == "draw_pressure_world":
            draw_keep_cap = 6
            rank2_min = 2
            rank3_min = 0
        elif world == "close_draw_world":
            draw_keep_cap = 5
            rank2_min = 3
            rank3_min = 1
        elif world == "hold_guard_world":
            draw_keep_cap = 6
            rank2_min = 3
            rank3_min = 0
        elif world == "controlled_upset_world":
            draw_keep_cap = 5
            rank2_min = 4
            rank3_min = 1
        elif world == "side_hedge_world":
            draw_keep_cap = 5
            rank2_min = 3
            rank3_min = 0
        elif world == "upset_world":
            draw_keep_cap = 4
            rank2_min = 4
            rank3_min = 1
        elif world == "j2_context_world":
            draw_keep_cap = 6
            rank2_min = 2
            rank3_min = 0
        elif world == "wide_world":
            draw_keep_cap = 5
            rank2_min = 4
            rank3_min = 1
        else:
            return

        draw_rank1_indices = [
            i for i in ok_indices
            if _first_second_third_symbol(plans[i], 0) == "0"
        ]
        if not draw_rank1_indices:
            return

        current_draw_count = sum(1 for i in draw_rank1_indices if ticket[i] == "0")
        rank2_now = sum(1 for i in draw_rank1_indices if ticket[i] == _first_second_third_symbol(plans[i], 1))
        rank3_now = sum(1 for i in draw_rank1_indices if ticket[i] == _first_second_third_symbol(plans[i], 2))

        candidates = sorted(
            draw_rank1_indices,
            key=lambda i: (
                -_top_gap(plans[i]),
                -_rank_prob(plans[i], _first_second_third_symbol(plans[i], 1)),
                -_rank_prob(plans[i], _first_second_third_symbol(plans[i], 2)),
                i,
            ),
        )

        for mi in candidates:
            if current_draw_count <= draw_keep_cap and rank2_now >= rank2_min and rank3_now >= rank3_min:
                break
            plan = plans[mi]
            ranks = _rank_triplet(plan)
            if ticket[mi] != "0":
                continue
            target_sym = ""
            if rank2_now < rank2_min and ranks[1] != "0":
                target_sym = ranks[1]
                rank2_now += 1
            elif rank3_now < rank3_min and ranks[2] != "0":
                target_sym = ranks[2]
                rank3_now += 1
            elif current_draw_count > draw_keep_cap and ranks[1] != "0":
                target_sym = ranks[1]
                rank2_now += 1
            if not target_sym or target_sym == ticket[mi]:
                continue
            _apply_pick(ticket, mi, target_sym, f"world_draw_limit:{world}", counts, reason_log, breaks)
            current_draw_count -= 1

    def _base_rank_ticket() -> List[str]:
        return [_first_second_third_symbol(p, 0) if p.status == "OK" else "1" for p in plans]

    base_ticket = _base_rank_ticket()
    tickets: List[List[str]] = []
    flip_descs: List[str] = []

    draw_candidates = sorted(
        [i for i in ok_indices if "0" in _rank_triplet(plans[i])[:2]],
        key=lambda i: (-_draw_like_score(plans[i]), _top_gap(plans[i]), i),
    )
    close_draw_candidates = sorted(
        [i for i in draw_candidates if _top_gap(plans[i]) <= 0.10],
        key=lambda i: (-_draw_like_score(plans[i]), _top_gap(plans[i]), i),
    )
    upset_candidates = sorted(
        ok_indices,
        key=lambda i: (-_upset_score(plans[i]), _top_gap(plans[i]), i),
    )
    side_hedge_candidates = sorted(
        [i for i in ok_indices if _preferred_side_hedge_symbol(plans[i]) in {"1", "2"}],
        key=lambda i: (-_side_hedge_score(plans[i]), _top_gap(plans[i]), i),
    )
    hold_candidates = sorted(
        ok_indices,
        key=lambda i: (-_hold_score(plans[i]), _top_gap(plans[i]), i),
    )
    wide_candidates = sorted(
        ok_indices,
        key=lambda i: (-_wide_score(plans[i]), _top_gap(plans[i]), i),
    )
    j2_candidates = sorted(
        [i for i in ok_indices if str(plans[i].league).upper() == "J2"],
        key=lambda i: (-_draw_like_score(plans[i]), -_hold_score(plans[i]), i),
    )

    world_specs = [
        {"id": "01", "name": "Main Anchor", "world": "main_anchor_world"},
        {"id": "02", "name": "Safe Main", "world": "safe_main_world"},
        {"id": "03", "name": "Draw Pressure", "world": "draw_pressure_world"},
        {"id": "04", "name": "Hold Guard", "world": "hold_guard_world"},
        {"id": "05", "name": "Controlled Upset", "world": "controlled_upset_world"},
        {"id": "06", "name": "J2 Context", "world": "j2_context_world"},
        {"id": "07", "name": "Side Hedge", "world": "side_hedge_world"},
        {"id": "08", "name": "Close Draw", "world": "close_draw_world"},
        {"id": "09", "name": "Upset", "world": "upset_world"},
        {"id": "10", "name": "Wide", "world": "wide_world"},
    ]

    for spec in world_specs:
        world = spec["world"]
        ticket = list(base_ticket)
        reason_log: List[str] = []
        counts = {"first": 0, "second": 0, "third": 0}
        breaks = {"strong": 0, "weak": 0}

        if world == "main_anchor_world":
            reason_log.append("main_anchor")
        elif world == "safe_main_world":
            for mi in [i for i in upset_candidates if _top_gap(plans[i]) < 0.055][:1]:
                if _rank_prob(plans[mi], _first_second_third_symbol(plans[mi], 0)) < 0.44:
                    _apply_pick(ticket, mi, _first_second_third_symbol(plans[mi], 1), "safe_second_cover", counts, reason_log, breaks)
        elif world == "draw_pressure_world":
            applied = 0
            for mi in draw_candidates:
                plan = plans[mi]
                if applied >= 4:
                    break
                ranks = _rank_triplet(plan)
                if "0" not in ranks[:2]:
                    continue
                if _draw_like_score(plan) < 0.55:
                    continue
                _apply_pick(ticket, mi, "0", "draw_pressure_keep", counts, reason_log, breaks)
                applied += 1
        elif world == "hold_guard_world":
            for mi in [i for i in hold_candidates if _hold_score(plans[i]) >= 0.52][:2]:
                _apply_pick(ticket, mi, _first_second_third_symbol(plans[mi], 0), "hold_guard_keep", counts, reason_log, breaks)
            for mi in [i for i in upset_candidates if _hold_score(plans[i]) < 0.38 and _top_gap(plans[i]) < 0.04][:1]:
                _apply_pick(ticket, mi, _first_second_third_symbol(plans[mi], 1), "hold_guard_escape", counts, reason_log, breaks)
        elif world == "controlled_upset_world":
            second_applied = 0
            third_applied = 0
            for mi in upset_candidates:
                plan = plans[mi]
                first_prob = _rank_prob(plan, _first_second_third_symbol(plan, 0))
                if first_prob >= 0.60:
                    continue
                if second_applied < 4 and _upset_score(plan) >= 0.44:
                    _apply_pick(ticket, mi, _first_second_third_symbol(plan, 1), "weak_first_break", counts, reason_log, breaks)
                    second_applied += 1
                    continue
                if third_applied < 3 and first_prob < 0.58 and _wide_score(plan) >= 0.48:
                    _apply_pick(ticket, mi, _first_second_third_symbol(plan, 2), "controlled_upset_third", counts, reason_log, breaks)
                    third_applied += 1
        elif world == "j2_context_world":
            for mi in j2_candidates:
                plan = plans[mi]
                ptype = str(plan.context_match_purchase_type or "").strip().lower()
                if ptype == "j2_draw_trap":
                    ranks = _rank_triplet(plan)
                    non_draw = next((sym for sym in ranks if sym in {"1", "2"}), ranks[0])
                    _apply_pick(ticket, mi, non_draw, "j2_draw_trap_context", counts, reason_log, breaks)
                elif ptype == "j2_false_draw_watch" and "0" in _rank_triplet(plan)[:2] and _draw_like_score(plan) >= 0.60:
                    _apply_pick(ticket, mi, _first_second_third_symbol(plan, 1), "j2_false_draw_suppress", counts, reason_log, breaks)
        elif world == "side_hedge_world":
            hedge_applied = 0
            for mi in side_hedge_candidates:
                if hedge_applied >= 3:
                    break
                plan = plans[mi]
                sym = _preferred_side_hedge_symbol(plan)
                if sym == _first_second_third_symbol(plan, 0) or sym == "0":
                    continue
                if _side_hedge_score(plan) < 0.48:
                    continue
                _apply_pick(ticket, mi, sym, "side_hedge", counts, reason_log, breaks)
                hedge_applied += 1
        elif world == "close_draw_world":
            applied = 0
            for mi in close_draw_candidates:
                plan = plans[mi]
                if applied >= 3:
                    break
                if _draw_like_score(plan) < 0.58:
                    continue
                _apply_pick(ticket, mi, "0", "close_draw_keep", counts, reason_log, breaks)
                applied += 1
        elif world == "upset_world":
            second_applied = 0
            third_applied = 0
            for mi in upset_candidates:
                plan = plans[mi]
                first_prob = _rank_prob(plan, _first_second_third_symbol(plan, 0))
                if second_applied < 4 and _upset_score(plan) >= 0.52:
                    _apply_pick(ticket, mi, _first_second_third_symbol(plan, 1), "controlled_upset", counts, reason_log, breaks)
                    second_applied += 1
                    continue
                if third_applied < 3 and first_prob < 0.60 and _wide_score(plan) >= 0.54:
                    _apply_pick(ticket, mi, _first_second_third_symbol(plan, 2), "upset_third", counts, reason_log, breaks)
                    third_applied += 1
        elif world == "wide_world":
            second_applied = 0
            third_applied = 0
            for mi in wide_candidates:
                plan = plans[mi]
                if second_applied < 4 and _wide_score(plan) >= 0.46:
                    _apply_pick(ticket, mi, _first_second_third_symbol(plan, 1), "second_cover", counts, reason_log, breaks)
                    second_applied += 1
                    continue
                if third_applied < 3 and _wide_score(plan) >= 0.52 and _rank_prob(plan, _first_second_third_symbol(plan, 0)) < 0.64:
                    _apply_pick(ticket, mi, _first_second_third_symbol(plan, 2), "wide_third", counts, reason_log, breaks)
                    third_applied += 1

        tickets.append(ticket)
        flip_descs.append("; ".join(reason_log) if reason_log else "base")
        sid = spec["id"]
        stats[f"ticket_{sid}_mode"] = world
        stats[f"ticket_{sid}_first_count"] = int(counts["first"])
        stats[f"ticket_{sid}_second_count"] = int(counts["second"])
        stats[f"ticket_{sid}_third_count"] = int(counts["third"])
        stats[f"ticket_{sid}_strong_first_break_count"] = int(breaks["strong"])
        stats[f"ticket_{sid}_weak_first_break_count"] = int(breaks["weak"])
        stats[f"ticket_{sid}_third_pick_count"] = int(counts["third"])
        stats[f"ticket_{sid}_world"] = world
        _warn(
            warnings,
            f"ticket{sid}_world={world} first={counts['first']} second={counts['second']} third={counts['third']} "
            f"strong_break={breaks['strong']} weak_break={breaks['weak']}"
        )

    def _log_purchase_type_policy(
        policy_logs: List[str],
        plan: MatchPlan,
        policy_action: str,
        base_symbol: str,
        new_symbol: str,
        affected_tickets: List[int],
        change_cap: int,
        reason: str,
    ) -> None:
        if not affected_tickets:
            return
        purchase_type = str(getattr(plan, "context_match_purchase_type", "") or "").strip()
        policy_logs.append(
            f"purchase_type={purchase_type} policy_action={policy_action} "
            f"base_symbol={base_symbol} new_symbol={new_symbol} "
            f"affected_tickets={','.join(f'{ti+1:02d}' for ti in affected_tickets)} "
            f"change_cap={change_cap} reason={reason} rescue_count=0 harm_count=0 net_rescue=0"
        )

    def _apply_formal_policies() -> None:
        policy_logs: List[str] = []
        home_reversal_targets = [6, 8, 9, 4]  # 07,09,10,05
        j2_draw_targets = [2, 5, 7, 8]  # 03,06,08,09
        for mi, plan in enumerate(plans):
            if plan.status != "OK":
                continue
            purchase_type = str(getattr(plan, "context_match_purchase_type", "") or "").strip().lower()
            if purchase_type == "home_reversal_watch":
                confidence = str(getattr(plan, "context_purchase_type_confidence", "") or "").strip().lower()
                change_cap = 4 if confidence == "high" else 3 if confidence == "medium" else 2
                changed: List[int] = []
                for ti in home_reversal_targets:
                    if len(changed) >= change_cap or ti >= len(tickets):
                        break
                    if tickets[ti][mi] != "2":
                        continue
                    tickets[ti][mi] = "1"
                    changed.append(ti)
                    flip_descs[ti] = f"{flip_descs[ti]}; home_reversal_policy:M{plan.match_no:02d}:2->1"
                _log_purchase_type_policy(policy_logs, plan, "hedge_home", "2", "1", changed, change_cap, "away_overread_guard")
            elif purchase_type == "j2_draw_trap" and str(plan.league).upper() == "J2":
                ranks = _rank_triplet(plan)
                main_side = next((sym for sym in ranks if sym in {"1", "2"}), ranks[0])
                changed: List[int] = []
                for ti in j2_draw_targets:
                    if ti >= len(tickets):
                        continue
                    if tickets[ti][mi] != "0":
                        continue
                    tickets[ti][mi] = main_side
                    changed.append(ti)
                    flip_descs[ti] = f"{flip_descs[ti]}; j2_draw_trap_policy:M{plan.match_no:02d}:0->{main_side}"
                _log_purchase_type_policy(policy_logs, plan, "suppress_draw", "0", main_side, changed, len(changed), "false_draw_suppression")
        stats["purchase_type_policy_count"] = int(len(policy_logs))
        stats["purchase_type_policy_matches"] = "; ".join(policy_logs[:30])
        if policy_logs:
            _warn(warnings, f"purchase_type_policy={stats['purchase_type_policy_matches']}")

    _apply_formal_policies()

    def _apply_global_draw_quota() -> None:
        total_cells = len(tickets) * len(ok_indices)
        if total_cells <= 0:
            return
        target_count = int(round(total_cells * BUYPLAN_DRAW_TARGET_RATIO))
        min_count = int(round(total_cells * BUYPLAN_DRAW_MIN_RATIO))
        max_count = int(round(total_cells * BUYPLAN_DRAW_MAX_RATIO))
        current_count = sum(1 for ticket in tickets for mi in ok_indices if ticket[mi] == "0")
        stats["draw_quota_target_count"] = target_count
        stats["draw_quota_min_count"] = min_count
        stats["draw_quota_max_count"] = max_count
        stats["draw_quota_before_count"] = current_count
        if current_count <= max_count:
            stats["draw_quota_after_count"] = current_count
            return

        removable: List[Tuple[float, int, int, str]] = []
        for ti, ticket in enumerate(tickets):
            for mi in ok_indices:
                if ticket[mi] != "0":
                    continue
                plan = plans[mi]
                world = str(stats.get(f"ticket_{ti+1:02d}_world", ""))
                ranks = _rank_triplet(plan)
                replace_sym = ""
                if world in {"main_anchor_world", "safe_main_world", "draw_pressure_world", "close_draw_world", "hold_guard_world"}:
                    replace_sym = ranks[1]
                elif world in {"controlled_upset_world", "wide_world", "side_hedge_world"}:
                    replace_sym = ranks[2] if ranks[2] != "0" else ranks[1]
                elif world == "j2_context_world":
                    replace_sym = ranks[1]
                else:
                    replace_sym = ranks[1]
                if replace_sym == "0":
                    continue
                score = _draw_admission_score(plan)
                removable.append((score, ti, mi, replace_sym))

        removable.sort(key=lambda x: (x[0], x[2], x[1]))
        changed = 0
        for score, ti, mi, replace_sym in removable:
            if current_count <= target_count:
                break
            if tickets[ti][mi] != "0":
                continue
            plan = plans[mi]
            flip_descs[ti] = f"{flip_descs[ti]}; draw_quota:M{plan.match_no:02d}:0->{replace_sym}:{score:.3f}"
            tickets[ti][mi] = replace_sym
            changed += 1
            current_count -= 1
        stats["draw_quota_change_count"] = changed
        stats["draw_quota_after_count"] = current_count
        _warn(
            warnings,
            f"draw_quota target={target_count} range={min_count}-{max_count} before={stats['draw_quota_before_count']} after={current_count} changed={changed}",
        )

    _apply_global_draw_quota()

    def _reapply_admission_caps_after_quota() -> None:
        for mi, plan in enumerate(plans):
            if plan.status != "OK":
                continue
            policy = str(getattr(plan, "context_admission_policy", "") or "").strip().lower()
            if policy == "rank1_draw_rescue":
                draw_cap, rank2_min, rank3_min = 5, 2, 0
            elif policy == "rank1_draw_watch":
                draw_cap, rank2_min, rank3_min = 4, 3, 0
            elif policy == "rank1_draw_core":
                draw_cap, rank2_min, rank3_min = 6, 2, 0
            elif policy == "rank2_draw_cut":
                draw_cap, rank2_min, rank3_min = 2, 0, 0
            else:
                continue
            rank1 = _normalize_symbol_token(getattr(plan, "context_rank1_pick", ""))
            rank2 = _normalize_symbol_token(getattr(plan, "context_rank2_pick", ""))
            rank3 = _normalize_symbol_token(getattr(plan, "context_rank3_pick", ""))
            if rank1 != "0":
                continue
            draw_count = sum(1 for ti in range(len(tickets)) if tickets[ti][mi] == "0")
            rank2_count = sum(1 for ti in range(len(tickets)) if rank2 and tickets[ti][mi] == rank2)
            rank3_count = sum(1 for ti in range(len(tickets)) if rank3 and tickets[ti][mi] == rank3)
            for ti in [9, 4, 6, 3, 7, 2, 8, 5, 1, 0]:
                if ti >= len(tickets):
                    continue
                if draw_count <= draw_cap and rank2_count >= rank2_min and rank3_count >= rank3_min:
                    break
                if tickets[ti][mi] != "0":
                    continue
                replacement = ""
                if rank2_count < rank2_min and rank2 and rank2 != "0":
                    replacement = rank2
                    rank2_count += 1
                elif rank3_count < rank3_min and rank3 and rank3 not in {"", "0", rank2}:
                    replacement = rank3
                    rank3_count += 1
                elif draw_count > draw_cap and rank2 and rank2 != "0":
                    replacement = rank2
                    rank2_count += 1
                elif draw_count > draw_cap and rank3 and rank3 not in {"", "0", rank2}:
                    replacement = rank3
                    rank3_count += 1
                if replacement not in {"1", "2"}:
                    continue
                before = tickets[ti][mi]
                tickets[ti][mi] = replacement
                draw_count -= 1
                flip_descs[ti] = f"{flip_descs[ti]}; admission_cap_post_quota:M{plan.match_no:02d}:{before}->{replacement}"

    def _refill_draw_quota_after_caps() -> None:
        total_cells = len(tickets) * len(ok_indices)
        if total_cells <= 0:
            return
        min_count = int(round(total_cells * BUYPLAN_DRAW_MIN_RATIO))
        target_count = int(round(total_cells * BUYPLAN_DRAW_TARGET_RATIO))
        current_count = sum(1 for ticket in tickets for mi in ok_indices if ticket[mi] == "0")
        if current_count >= min_count:
            stats["draw_quota_refill_count"] = 0
            stats["draw_quota_final_count"] = current_count
            return

        def _policy_draw_cap(plan: MatchPlan) -> Optional[int]:
            policy = str(getattr(plan, "context_admission_policy", "") or "").strip().lower()
            if policy == "rank1_draw_rescue":
                return 5
            if policy == "rank1_draw_watch":
                return 4
            if policy == "rank1_draw_core":
                return 6
            if policy == "rank2_draw_cut":
                return 2
            return None

        world_bonus_map = {
            "close_draw_world": 0.10,
            "draw_pressure_world": 0.08,
            "main_anchor_world": 0.06,
            "safe_main_world": 0.05,
            "j2_context_world": 0.05,
            "hold_guard_world": 0.03,
            "wide_world": 0.01,
            "controlled_upset_world": -0.02,
            "side_hedge_world": -0.03,
            "upset_world": -0.05,
        }

        addable: List[Tuple[float, int, int]] = []
        for ti, ticket in enumerate(tickets):
            world = str(stats.get(f"ticket_{ti+1:02d}_world", ""))
            world_bonus = float(world_bonus_map.get(world, 0.0))
            for mi in ok_indices:
                if ticket[mi] == "0":
                    continue
                plan = plans[mi]
                rank1 = _normalize_symbol_token(getattr(plan, "context_rank1_pick", ""))
                if rank1 != "0":
                    continue
                draw_cap = _policy_draw_cap(plan)
                if draw_cap is None:
                    continue
                draw_count = sum(1 for tj in range(len(tickets)) if tickets[tj][mi] == "0")
                if draw_count >= draw_cap:
                    continue
                current_sym = _normalize_symbol_token(ticket[mi])
                rank2 = _normalize_symbol_token(getattr(plan, "context_rank2_pick", ""))
                rank3 = _normalize_symbol_token(getattr(plan, "context_rank3_pick", ""))
                replace_penalty = 0.00
                if current_sym == rank2:
                    replace_penalty = 0.02
                elif current_sym == rank3:
                    replace_penalty = 0.00
                else:
                    replace_penalty = 0.04
                score = _draw_admission_score(plan) + world_bonus - replace_penalty
                addable.append((score, ti, mi))

        addable.sort(key=lambda x: (x[0], -x[2], -x[1]), reverse=True)
        changed = 0
        for score, ti, mi in addable:
            if current_count >= target_count:
                break
            if tickets[ti][mi] == "0":
                continue
            plan = plans[mi]
            draw_cap = _policy_draw_cap(plan)
            if draw_cap is None:
                continue
            draw_count = sum(1 for tj in range(len(tickets)) if tickets[tj][mi] == "0")
            if draw_count >= draw_cap:
                continue
            before = tickets[ti][mi]
            tickets[ti][mi] = "0"
            current_count += 1
            changed += 1
            flip_descs[ti] = f"{flip_descs[ti]}; draw_quota_refill:M{plan.match_no:02d}:{before}->0:{score:.3f}"

        stats["draw_quota_refill_count"] = changed
        stats["draw_quota_final_count"] = current_count
        if changed:
            _warn(
                warnings,
                f"draw_quota_refill target={target_count} min={min_count} final={current_count} changed={changed}",
            )

    def _ensure_draw_policy_rank3_presence() -> None:
        # Draw weakening now follows rank2 > rank1 > rank3.
        # Keep rank3 sparse and do not proactively inject it.
        stats["draw_rank3_presence_count"] = 0
        stats["draw_rank3_presence_matches"] = ""

    def _diversify_j1_false_draw_watch() -> None:
        stats["j1_false_draw_watch_dir_count"] = 0
        stats["j1_false_draw_watch_dir_matches"] = ""

    def _fix_two_way_residual_draw_watch() -> None:
        target_tickets = [9, 8]  # ticket10, ticket09 only
        changes: List[str] = []
        for mi, plan in enumerate(plans):
            if plan.status != "OK":
                continue
            purchase_type = str(getattr(plan, "context_match_purchase_type", "") or "").strip().lower()
            policy = str(getattr(plan, "context_admission_policy", "") or "").strip().lower()
            if purchase_type != "j1_false_draw_watch" or policy != "rank1_draw_watch":
                continue
            rank1 = _normalize_symbol_token(getattr(plan, "context_rank1_pick", ""))
            rank2 = _normalize_symbol_token(getattr(plan, "context_rank2_pick", ""))
            rank3 = _normalize_symbol_token(getattr(plan, "context_rank3_pick", ""))
            if rank1 != "0" or rank2 not in {"1", "2"} or rank3 not in {"1", "2"} or rank2 == rank3:
                continue
            counts = Counter(tickets[ti][mi] for ti in range(len(tickets)))
            active_syms = [sym for sym in ("1", "0", "2") if counts.get(sym, 0) > 0]
            if len(active_syms) != 2:
                continue
            if counts.get(rank3, 0) > 0:
                continue
            if counts.get(rank2, 0) < 3:
                continue
            side_gap = abs(_rank_prob(plan, rank2) - _rank_prob(plan, rank3))
            top_gap = _top_gap(plan)
            if not (top_gap <= 0.14 or side_gap <= 0.04):
                continue
            changed = False
            for ti in target_tickets:
                if ti >= len(tickets):
                    continue
                current = tickets[ti][mi]
                if current != rank2:
                    continue
                cand = list(tickets[ti])
                cand[mi] = rank3
                cand_key = _ticket_key(cand)
                if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                    continue
                tickets[ti] = cand
                flip_descs[ti] = f"{flip_descs[ti]}; twoway_residual:M{plan.match_no:02d}:{current}->{rank3}"
                changes.append(f"M{plan.match_no:02d} ticket{ti+1:02d} {current}->{rank3}")
                changed = True
                break
            if changed:
                continue
        stats["twoway_residual_count"] = int(len(changes))
        stats["twoway_residual_matches"] = "; ".join(changes[:30])
        if changes:
            _warn(warnings, f"twoway_residual={stats['twoway_residual_matches']}")

    def _sparsify_draw_rank3() -> None:
        changes: List[str] = []
        preferred_ticket_order = [9, 8, 4, 6, 3, 7, 2, 5, 1, 0]
        for mi, plan in enumerate(plans):
            if plan.status != "OK":
                continue
            policy = str(getattr(plan, "context_admission_policy", "") or "").strip().lower()
            if policy not in {"rank1_draw_rescue", "rank1_draw_watch", "rank1_draw_core"}:
                continue
            rank1 = _normalize_symbol_token(getattr(plan, "context_rank1_pick", ""))
            rank2 = _normalize_symbol_token(getattr(plan, "context_rank2_pick", ""))
            rank3 = _normalize_symbol_token(getattr(plan, "context_rank3_pick", ""))
            if rank1 != "0" or rank3 not in {"1", "2"} or rank3 == rank2:
                continue
            rank3_cap = 1
            rank3_count = sum(1 for ti in range(len(tickets)) if tickets[ti][mi] == rank3)
            if rank3_count <= rank3_cap:
                continue
            for ti in preferred_ticket_order:
                if rank3_count <= rank3_cap:
                    break
                if ti >= len(tickets):
                    continue
                current = tickets[ti][mi]
                if current != rank3:
                    continue
                cand = list(tickets[ti])
                cand[mi] = "0"
                cand_key = _ticket_key(cand)
                if any(cand_key == _ticket_key(tickets[k]) for k in range(len(tickets)) if k != ti):
                    continue
                tickets[ti] = cand
                rank3_count -= 1
                flip_descs[ti] = f"{flip_descs[ti]}; draw_rank3_sparse:M{plan.match_no:02d}:{current}->0"
                changes.append(f"M{plan.match_no:02d} ticket{ti+1:02d} {current}->0")
        stats["draw_rank3_sparse_count"] = int(len(changes))
        stats["draw_rank3_sparse_matches"] = "; ".join(changes[:30])
        if changes:
            _warn(warnings, f"draw_rank3_sparse={stats['draw_rank3_sparse_matches']}")

    def _ticket_distance_local(a: List[str], b: List[str]) -> int:
        return sum(1 for x, y in zip(a, b) if x != y)

    def _repair_uniqueness() -> None:
        seen: Dict[str, int] = {}
        for ti, ticket in enumerate(tickets):
            key = _ticket_key(ticket)
            if key not in seen:
                seen[key] = ti
                continue
            if ti == 0:
                continue
            candidate_matches = sorted(
                ok_indices,
                key=lambda i: (-_wide_score(plans[i]), -_upset_score(plans[i]), _top_gap(plans[i]), i),
            )
            repaired = False
            for mi in candidate_matches:
                ranks = _rank_triplet(plans[mi])
                for alt in ranks[1:]:
                    if alt == tickets[ti][mi]:
                        continue
                    cand = list(tickets[ti])
                    cand[mi] = alt
                    cand_key = _ticket_key(cand)
                    if cand_key in seen:
                        continue
                    tickets[ti] = cand
                    flip_descs[ti] = f"{flip_descs[ti]}; unique_repair:M{plans[mi].match_no:02d}:{ticket[mi]}->{alt}"
                    seen[cand_key] = ti
                    repaired = True
                    break
                if repaired:
                    break
            if not repaired:
                seen[key] = ti

    _repair_uniqueness()
    _ensure_draw_policy_rank3_presence()
    _diversify_j1_false_draw_watch()
    _fix_two_way_residual_draw_watch()

    def _repair_min_ticket_distance(min_distance: int = 2) -> None:
        """Repair near-duplicate tickets before the final portfolio guardrail.

        The final draw guardrail changes one or two tickets at a time.  Requiring
        the *whole* portfolio to already satisfy the distance floor made every
        candidate fail when an unrelated pair entered this stage at distance 1.
        Repair those pairs first, accepting only moves that reduce their number.
        """
        changes: List[str] = []

        def bad_pairs(candidate_tickets: List[List[str]]) -> List[Tuple[int, int, int]]:
            return [
                (i, j, _ticket_distance_local(candidate_tickets[i], candidate_tickets[j]))
                for i in range(len(candidate_tickets))
                for j in range(i + 1, len(candidate_tickets))
                if _ticket_distance_local(candidate_tickets[i], candidate_tickets[j]) < min_distance
            ]

        for _ in range(len(tickets) * len(ok_indices) * 2):
            before_bad = bad_pairs(tickets)
            if not before_bad:
                break
            _, right, _ = min(before_bad, key=lambda x: (x[2], x[0], x[1]))
            best_move: Optional[Tuple[int, float, int, str, List[str]]] = None
            for ti in (right, before_bad[0][0]):
                for mi in ok_indices:
                    current = tickets[ti][mi]
                    for alt in _rank_triplet(plans[mi]):
                        if alt == current:
                            continue
                        candidate = list(tickets[ti])
                        candidate[mi] = alt
                        trial = list(tickets)
                        trial[ti] = candidate
                        after_bad = bad_pairs(trial)
                        improvement = len(before_bad) - len(after_bad)
                        if improvement <= 0:
                            continue
                        draw_bonus = _draw_admission_score(plans[mi]) if alt == "0" else 0.0
                        move = (improvement, draw_bonus, ti, alt, candidate)
                        if best_move is None or move[:2] > best_move[:2]:
                            best_move = move
            if best_move is None:
                break
            _, _, ti, alt, candidate = best_move
            changed_mi = next(mi for mi in ok_indices if candidate[mi] != tickets[ti][mi])
            before = tickets[ti][changed_mi]
            tickets[ti] = candidate
            flip_descs[ti] += f"; distance_repair:M{plans[changed_mi].match_no:02d}:{before}->{alt}"
            changes.append(f"ticket{ti+1:02d} M{plans[changed_mi].match_no:02d} {before}->{alt}")

        remaining = bad_pairs(tickets)
        stats["ticket_distance_repair_count"] = len(changes)
        stats["ticket_distance_repair_remaining"] = len(remaining)
        if changes or remaining:
            _warn(
                warnings,
                f"ticket_distance_repair changes={len(changes)} remaining_lt{min_distance}={len(remaining)}"
                + (f" details={'; '.join(changes[:20])}" if changes else ""),
            )

    _repair_min_ticket_distance()

    def _apply_final_ticket_draw_guardrails() -> None:
        """Balance draw placement without changing prediction probabilities.

        The upstream allocation intentionally creates different scenario worlds,
        but the purchase layer must not leave a nominal ticket with zero draws or
        make the main ticket almost all draws.  Keep the portfolio draw total and
        per-match coverage meaningful while retaining the scenario differences.
        """
        if not tickets or not ok_indices:
            return

        desired_by_world = {
            "main_anchor_world": 4,
            "safe_main_world": 5,
            "draw_pressure_world": 6,
            "hold_guard_world": 3,
            "controlled_upset_world": 3,
            "j2_context_world": 4,
            "side_hedge_world": 3,
            "close_draw_world": 5,
            "upset_world": 3,
            "wide_world": 3,
        }
        desired = []
        for ti in range(len(tickets)):
            world = str(stats.get(f"ticket_{ti+1:02d}_world", ""))
            value = int(desired_by_world.get(world, 4))
            desired.append(max(BUYPLAN_TICKET_DRAW_MIN, min(BUYPLAN_TICKET_DRAW_MAX, value)))

        def draw_counts() -> List[int]:
            return [sum(1 for mi in ok_indices if ticket[mi] == "0") for ticket in tickets]

        def match_draw_count(mi: int) -> int:
            return sum(1 for ticket in tickets if ticket[mi] == "0")

        def match_draw_cap(mi: int) -> int:
            """Return the final cap used by the single portfolio adjustment."""
            policy = str(getattr(plans[mi], "context_admission_policy", "") or "").strip().lower()
            policy_cap = {
                "rank1_draw_rescue": 5,
                "rank1_draw_watch": 4,
                "rank1_draw_core": 6,
                "rank2_draw_cut": 2,
            }.get(policy, DRAW_MATCH_CAP)
            return min(DRAW_MATCH_CAP, int(policy_cap))

        def unique_after(changes: Dict[int, List[str]]) -> bool:
            candidate_tickets = []
            for ti, ticket in enumerate(tickets):
                candidate_tickets.append(changes.get(ti, ticket))
            keys = [_ticket_key(ticket) for ticket in candidate_tickets]
            if len(keys) != len(set(keys)):
                return False
            changed_indices = set(changes)
            return all(
                sum(1 for x, y in zip(candidate_tickets[i], candidate_tickets[j]) if x != y) >= 2
                for i in range(len(candidate_tickets))
                for j in range(i + 1, len(candidate_tickets))
                if i in changed_indices or j in changed_indices
            )

        def best_side(plan: MatchPlan, avoid: str = "") -> str:
            for sym in _rank_triplet(plan):
                if sym in {"1", "2"} and sym != avoid:
                    return sym
            return "1" if avoid != "1" else "2"

        changes: List[str] = []

        # Normalize existing cells first.  Previously the policy cap was only
        # consulted while adding draws, so an over-cap input (for example 8/10
        # draws on rank1_draw_watch) survived untouched.
        for mi in ok_indices:
            cap = match_draw_cap(mi)
            while match_draw_count(mi) > cap:
                counts_now = draw_counts()
                candidates = sorted(
                    [ti for ti in range(len(tickets)) if tickets[ti][mi] == "0"],
                    key=lambda ti: (
                        counts_now[ti] <= desired[ti],
                        _draw_admission_score(plans[mi]),
                        ti,
                    ),
                )
                removed = False
                for ti in candidates:
                    candidate = list(tickets[ti])
                    candidate[mi] = best_side(plans[mi])
                    if not unique_after({ti: candidate}):
                        continue
                    tickets[ti] = candidate
                    flip_descs[ti] += f"; final_draw_cap:M{plans[mi].match_no:02d}:0->{candidate[mi]}"
                    changes.append(f"cap ticket{ti+1:02d} M{plans[mi].match_no:02d}")
                    removed = True
                    break
                if not removed:
                    break

        # Transfer draw cells from overfull tickets to underfull tickets on the
        # same match.  This preserves both the portfolio total and match cover.
        for _ in range(len(tickets) * len(ok_indices) * 2):
            counts = draw_counts()
            donors = [ti for ti, n in enumerate(counts) if n > desired[ti]]
            receivers = [ti for ti, n in enumerate(counts) if n < desired[ti]]
            if not donors or not receivers:
                break
            candidates: List[Tuple[float, int, int, int]] = []
            for donor in donors:
                for receiver in receivers:
                    for mi in ok_indices:
                        if tickets[donor][mi] != "0" or tickets[receiver][mi] == "0":
                            continue
                        candidates.append((_draw_admission_score(plans[mi]), donor, receiver, mi))
            candidates.sort(reverse=True)
            moved = False
            for _, donor, receiver, mi in candidates:
                donor_new = list(tickets[donor])
                receiver_new = list(tickets[receiver])
                donor_new[mi] = tickets[receiver][mi]
                receiver_new[mi] = "0"
                if not unique_after({donor: donor_new, receiver: receiver_new}):
                    continue
                tickets[donor] = donor_new
                tickets[receiver] = receiver_new
                flip_descs[donor] += f"; ticket_draw_guard:M{plans[mi].match_no:02d}:0->{donor_new[mi]}"
                flip_descs[receiver] += f"; ticket_draw_guard:M{plans[mi].match_no:02d}:{tickets[donor][mi]}->0"
                changes.append(f"transfer ticket{donor+1:02d}->ticket{receiver+1:02d} M{plans[mi].match_no:02d}")
                moved = True
                break
            if not moved:
                break

        # A high raw draw probability must retain at least a small portfolio
        # cover. Move a draw within one ticket so its per-ticket count is stable.
        for target_mi in ok_indices:
            target_plan = plans[target_mi]
            if float(target_plan.p_draw) < BUYPLAN_HIGH_PD_FLOOR:
                continue
            while match_draw_count(target_mi) < BUYPLAN_HIGH_PD_MIN_COVER:
                moved = False
                for ti in sorted(range(len(tickets)), key=lambda x: draw_counts()[x]):
                    if tickets[ti][target_mi] == "0":
                        continue
                    donor_matches = sorted(
                        [
                            mi for mi in ok_indices
                            if tickets[ti][mi] == "0"
                            and match_draw_count(mi) > (
                                BUYPLAN_HIGH_PD_MIN_COVER
                                if float(plans[mi].p_draw) >= BUYPLAN_HIGH_PD_FLOOR else 0
                            )
                        ],
                        key=lambda mi: (_draw_admission_score(plans[mi]), mi),
                    )
                    for donor_mi in donor_matches:
                        candidate = list(tickets[ti])
                        candidate[target_mi] = "0"
                        candidate[donor_mi] = best_side(plans[donor_mi])
                        if not unique_after({ti: candidate}):
                            continue
                        before = tickets[ti][target_mi]
                        tickets[ti] = candidate
                        flip_descs[ti] += (
                            f"; high_pd_cover:M{target_plan.match_no:02d}:{before}->0"
                            f"/M{plans[donor_mi].match_no:02d}:0->{candidate[donor_mi]}"
                        )
                        changes.append(
                            f"high_pd ticket{ti+1:02d} M{plans[donor_mi].match_no:02d}->M{target_plan.match_no:02d}"
                        )
                        moved = True
                        break
                    if moved:
                        break
                if not moved:
                    break

        # Bring the portfolio to the expected 13 * 30% * 10 ~= 39 draws,
        # preferring tickets still below their scenario target.
        total_target = int(round(len(tickets) * len(ok_indices) * BUYPLAN_DRAW_TARGET_RATIO))
        current_total = sum(draw_counts())
        rejection_counts: Counter = Counter()
        for _ in range(max(0, total_target - current_total)):
            counts = draw_counts()
            candidates: List[Tuple[float, int, int]] = []
            for ti, count in enumerate(counts):
                if count >= desired[ti] or count >= BUYPLAN_TICKET_DRAW_MAX:
                    continue
                for mi in ok_indices:
                    if tickets[ti][mi] == "0" or match_draw_count(mi) >= match_draw_cap(mi):
                        continue
                    candidates.append((_draw_admission_score(plans[mi]), ti, mi))
            candidates.sort(reverse=True)
            added = False
            for _, ti, mi in candidates:
                candidate = list(tickets[ti])
                before = candidate[mi]
                candidate[mi] = "0"
                if not unique_after({ti: candidate}):
                    rejection_counts["distance_or_duplicate"] += 1
                    continue
                tickets[ti] = candidate
                flip_descs[ti] += f"; ticket_draw_target:M{plans[mi].match_no:02d}:{before}->0"
                changes.append(f"add ticket{ti+1:02d} M{plans[mi].match_no:02d}")
                current_total += 1
                added = True
                break
            if not added:
                if not candidates:
                    rejection_counts["no_eligible_candidate"] += 1
                break

        final_counts = draw_counts()
        stats["ticket_draw_guardrail_change_count"] = len(changes)
        stats["ticket_draw_guardrail_target_total"] = total_target
        stats["ticket_draw_guardrail_final_total"] = sum(final_counts)
        stats["ticket_draw_guardrail_counts"] = ",".join(str(x) for x in final_counts)
        stats["ticket_draw_guardrail_shortfall"] = max(0, total_target - sum(final_counts))
        stats["ticket_draw_guardrail_rejections"] = ",".join(
            f"{key}={value}" for key, value in sorted(rejection_counts.items())
        )
        _warn(
            warnings,
            "ticket_draw_guardrail "
            f"target_total={total_target} final_total={sum(final_counts)} "
            f"counts={stats['ticket_draw_guardrail_counts']} changes={len(changes)} "
            f"shortfall={stats['ticket_draw_guardrail_shortfall']} "
            f"rejections={stats['ticket_draw_guardrail_rejections'] or 'none'}",
        )

    _apply_final_ticket_draw_guardrails()

    def _diversify_ambiguous_all_same() -> None:
        """Avoid ten identical picks on genuinely close, high-entropy matches."""
        changes: List[str] = []
        for mi in ok_indices:
            plan = plans[mi]
            symbols = {tickets[ti][mi] for ti in range(len(tickets))}
            if len(symbols) != 1 or _top_gap(plan) > 0.08 or float(plan.entropy) < 1.05:
                continue
            alternatives = [sym for sym in _rank_triplet(plan)[1:] if sym != tickets[0][mi]]
            for alt in alternatives:
                changed_here = 0
                for ti in [9, 6, 4, 7, 2, 5, 8, 3, 1, 0]:
                    if changed_here >= 2 or ti >= len(tickets):
                        break
                    candidate = list(tickets[ti])
                    before = candidate[mi]
                    candidate[mi] = alt
                    trial = [candidate if tj == ti else tickets[tj] for tj in range(len(tickets))]
                    if len({_ticket_key(ticket) for ticket in trial}) != len(trial):
                        continue
                    if any(
                        _ticket_distance_local(candidate, tickets[tj]) < 2
                        for tj in range(len(tickets)) if tj != ti
                    ):
                        continue
                    tickets[ti] = candidate
                    flip_descs[ti] += f"; ambiguous_spread:M{plan.match_no:02d}:{before}->{alt}"
                    changes.append(f"ticket{ti+1:02d} M{plan.match_no:02d} {before}->{alt}")
                    changed_here += 1
                if changed_here:
                    break
        stats["ambiguous_all_same_change_count"] = len(changes)
        stats["ambiguous_all_same_changes"] = "; ".join(changes[:30])
        if changes:
            _warn(warnings, f"ambiguous_all_same_spread changes={len(changes)} details={stats['ambiguous_all_same_changes']}")

    _diversify_ambiguous_all_same()

    total_cells = len(tickets) * len(plans) if tickets and plans else 0
    pairwise_distances = [
        _ticket_distance_local(tickets[i], tickets[j])
        for i in range(len(tickets))
        for j in range(i + 1, len(tickets))
    ]
    stats["unique_ticket_count"] = len({_ticket_key(t) for t in tickets})
    stats["duplicate_count"] = max(0, len(tickets) - int(stats["unique_ticket_count"]))
    stats["min_ticket_distance"] = min(pairwise_distances) if pairwise_distances else 0
    stats["avg_ticket_distance"] = round(float(sum(pairwise_distances)) / float(len(pairwise_distances)), 3) if pairwise_distances else 0.0
    stats["total_one_count"] = sum(_symbol_count(t, "1") for t in tickets)
    stats["total_zero_count"] = sum(_symbol_count(t, "0") for t in tickets)
    stats["total_two_count"] = sum(_symbol_count(t, "2") for t in tickets)
    stats["total_ratio_1"] = _ratio_str(int(stats["total_one_count"]), total_cells)
    stats["total_ratio_0"] = _ratio_str(int(stats["total_zero_count"]), total_cells)
    stats["total_ratio_2"] = _ratio_str(int(stats["total_two_count"]), total_cells)

    # The legacy allocator recorded how many intermediate changes happened in
    # each topology.  For the world allocator report the final, auditable
    # departures from the rank-1/base ticket instead of leaving stale zeros.
    def _final_cluster_label(plan: MatchPlan) -> str:
        profile = str(getattr(plan, "lab_matchup_profile", "") or "").lower()
        if profile == "stall_shape" and str(plan.league).upper() == "J1":
            return "draw_core" if float(plan.shape_draw_compression) >= 0.68 and float(plan.p_draw) >= 0.40 else "hold_core"
        if float(plan.shape_draw_compression) >= 0.08 and float(plan.shape_swing_instability) >= 0.10:
            return "overlap"
        if float(plan.shape_hold_strength) >= 0.12:
            return "hold_core"
        if float(plan.shape_draw_compression) >= 0.08:
            return "draw_core"
        if float(plan.shape_swing_instability) >= 0.10:
            return "swing_core"
        return "neutral"

    for ti, ticket in enumerate(tickets, start=1):
        cluster_counts: Counter = Counter()
        basis_counts: Counter = Counter()
        for mi in ok_indices:
            if ticket[mi] == base_ticket[mi]:
                continue
            cluster_counts[_final_cluster_label(plans[mi])] += 1
            basis = str(getattr(plans[mi], "lab_basis_hint", "") or "").strip().lower() or "basis_balanced"
            basis_counts[basis] += 1
        for cluster in ["draw_core", "overlap", "swing_core", "neutral", "hold_core"]:
            stats[f"ticket_{ti:02d}_cluster_{cluster}_changes"] = int(cluster_counts[cluster])
        for basis in ["side_strong", "draw_compressed", "flat_draw_trap", "split_side", "basis_balanced"]:
            stats[f"ticket_{ti:02d}_basis_{basis}_changes"] = int(basis_counts[basis])

    for idx, ticket in enumerate(tickets, start=1):
        world = str(stats.get(f"ticket_{idx:02d}_world", scenario_defs[idx - 1].scenario_note if idx - 1 < len(scenario_defs) else ""))
        draw_n = _symbol_count(ticket, "0")
        diff_n = sum(1 for a, b in zip(ticket, base_ticket) if a != b)
        side_all_same = 0
        for mi in ok_indices:
            col = [tickets[ti][mi] for ti in range(len(tickets))]
            if len(set(col)) == 1 and col[0] in {"1", "2"}:
                side_all_same += 1
        stats[f"ticket_{idx:02d}_draw_count"] = int(draw_n)
        stats[f"ticket_{idx:02d}_diff_count"] = int(diff_n)
        stats[f"ticket_{idx:02d}_all_same_side_count"] = int(side_all_same)
        _warn(warnings, f"ticket{idx:02d}: world={world} D本数={draw_n} 差分件数={diff_n}")

    _warn(warnings, "buyplan mode: world_topology_allocator_v1")
    _warn(warnings, f"unique_ticket_count={stats['unique_ticket_count']} duplicate_count={stats['duplicate_count']}")
    _warn(warnings, f"ticket_distance min={stats['min_ticket_distance']} avg={stats['avg_ticket_distance']}")
    _warn(warnings, f"0総出現回数={stats['total_zero_count']} 1/0/2比率={stats['total_ratio_1']}/{stats['total_ratio_0']}/{stats['total_ratio_2']}")

    scenario_result_labels: List[List[str]] = []
    for t in tickets:
        scenario_result_labels.append([_result_label_from_symbol(x) for x in t])
    return tickets[:REQUIRED_TICKET_COUNT], flip_descs[:REQUIRED_TICKET_COUNT], scenario_defs[:REQUIRED_TICKET_COUNT], stats, scenario_result_labels


def _write_buyplan_csv(
    plans: List[MatchPlan],
    tickets: List[List[str]],
    out_csv: str,
    scenario_defs: List[ScenarioDef],
    scenario_result_labels: List[List[str]],
) -> None:
    def _draw_pressure_score(plan: MatchPlan) -> float:
        return float(plan.shape_draw_compression)

    def _swing_score(plan: MatchPlan) -> float:
        return float(plan.shape_swing_instability)

    def _hold_lock_score(plan: MatchPlan) -> float:
        return float(plan.shape_hold_strength)

    def _topology_asymmetry_score(plan: MatchPlan) -> float:
        home_path = float(plan.lab_home_path_score)
        away_path = float(plan.lab_away_path_score)
        side_gap = abs(home_path - away_path)
        directional_gap = abs(float(plan.p_home) - float(plan.p_away))
        return min(
            1.0,
            0.38 * side_gap
            + 0.24 * float(plan.shape_directionality)
            + 0.20 * float(plan.shape_path_imbalance)
            + 0.18 * directional_gap,
        )

    def _draw_admission_score(plan: MatchPlan) -> float:
        return min(
            1.0,
            0.64 * float(plan.shape_draw_compression)
            + 0.20 * float(plan.shape_entropy)
            + 0.16 * float(plan.p_draw),
        )

    def _away_admission_score(plan: MatchPlan) -> float:
        return min(
            1.0,
            0.44 * float(plan.shape_swing_instability)
            + 0.30 * float(plan.shape_path_imbalance)
            + 0.14 * float(plan.shape_directionality)
            + 0.12 * float(plan.p_away),
        )

    def _home_escape_score(plan: MatchPlan) -> float:
        return min(
            1.0,
            0.42 * float(plan.shape_swing_instability)
            + 0.24 * float(plan.shape_path_imbalance)
            + 0.18 * float(plan.shape_directionality)
            + 0.16 * float(plan.p_home),
        )

    def _admission_escape_score(plan: MatchPlan, target_symbol: str) -> float:
        target_symbol = str(target_symbol)
        asym = float(_topology_asymmetry_score(plan))
        draw_adm = float(_draw_admission_score(plan))
        away_adm = float(_away_admission_score(plan))
        home_adm = float(_home_escape_score(plan))
        if target_symbol == "0":
            return min(
                1.0,
                0.58 * draw_adm
                + 0.22 * (1.0 - asym)
                + 0.20 * float(plan.lab_draw_tension_score),
            )
        if target_symbol == "2":
            return min(
                1.0,
                0.52 * away_adm
                + 0.26 * asym
                + 0.12 * float(plan.lab_away_path_score)
                + 0.10 * (1.0 - draw_adm),
            )
        if target_symbol == "1":
            return min(
                1.0,
                0.46 * home_adm
                + 0.24 * asym
                + 0.18 * float(plan.lab_home_path_score)
                + 0.12 * (1.0 - float(plan.shape_hold_strength)),
            )
        return 0.0

    def _cluster_label(plan: MatchPlan) -> str:
        profile = str(getattr(plan, "lab_matchup_profile", "")).lower()
        if profile == "low_tempo_draw":
            return "low_tempo_draw"
        if profile == "stall_shape" and str(plan.league).upper() == "J1":
            if float(plan.shape_draw_compression) >= 0.68 and float(plan.p_draw) >= 0.40:
                return "draw_core"
            return "hold_core"
        if profile in {
            "directional_stall",
            "swing_watch",
            "draw_anchor",
            "stall_side_preserve",
            "mid_stall_side_preserve",
            "home_control_watch",
            "j2_draw_stall_anchor",
            "j2_home_stall_preserve",
            "j1_pressure_draw_watch",
            "j1_floor_recovery_watch",
            "j2_flip_conflict_watch",
            "j2_home_pressure_stall_preserve",
            "j2_low_dyn_home_stall_preserve",
            "j2_neutral_flip_draw_watch",
            "j2_tense_stall_watch",
        }:
            return "hold_core"
        draw_score = float(plan.shape_draw_compression)
        swing_score = float(plan.shape_swing_instability)
        hold_score = float(plan.shape_hold_strength)
        if draw_score >= 0.48 and swing_score >= 0.42:
            return "overlap"
        if hold_score >= 0.48 and draw_score < 0.44 and swing_score < 0.34:
            return "hold_core"
        if draw_score >= 0.48:
            return "draw_core"
        if swing_score >= 0.42:
            return "swing_core"
        return "neutral"

    def _overlap_subtype(plan: MatchPlan) -> str:
        if _cluster_label(plan) != "overlap":
            return "non_overlap"
        primary = str(plan.context_primary_pick or plan.buyplan_choice or plan.best)
        control = str(plan.lab_control_band).lower()
        pressure = str(plan.lab_pressure_band).lower()
        primary_prob = float(plan.p_best)
        if primary == "2" and (control == "home" or pressure == "home") and primary_prob <= 0.38:
            return "counterflow_away"
        if primary == "1" and (control == "away" or pressure == "away") and primary_prob <= 0.38:
            return "counterflow_home"
        if (
            str(plan.lab_tempo_band).lower() == "low"
            and float(plan.shape_draw_compression) >= 0.74
            and float(plan.shape_swing_instability) <= 0.34
            and float(plan.lab_flip_weight) <= 0.30
        ):
            return "low_tempo_draw"
        return "general_overlap"

    rows = []
    for i, p in enumerate(plans):
        row = {
            "match_no": p.match_no,
            "home_team": p.home_team,
            "away_team": p.away_team,
            "league": p.league,
            "context_primary_pick": p.context_primary_pick,
            "context_secondary_pick": p.context_secondary_pick,
            "context_admission_policy": p.context_admission_policy,
            "context_rank1_pick": p.context_rank1_pick,
            "context_rank2_pick": p.context_rank2_pick,
            "context_rank3_pick": p.context_rank3_pick,
            "context_rank1_prob": p.context_rank1_prob,
            "context_rank2_prob": p.context_rank2_prob,
            "context_rank3_prob": p.context_rank3_prob,
            "context_risk_level": p.context_risk_level,
            "context_ticket_guidance": p.context_ticket_guidance,
            "context_decision_summary": p.context_decision_summary,
            "context_match_purchase_type": p.context_match_purchase_type,
            "context_match_purchase_subtype": p.context_match_purchase_subtype,
            "context_purchase_type_confidence": p.context_purchase_type_confidence,
            "context_purchase_type_reason": p.context_purchase_type_reason,
            "context_anchor_candidate_flag": p.context_anchor_candidate_flag,
            "context_draw_core_candidate_flag": p.context_draw_core_candidate_flag,
            "context_draw_core_flag": p.context_draw_core_flag,
            "context_draw_purchase_tier": p.context_draw_purchase_tier,
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
            "shape_draw_pressure_score": _draw_pressure_score(p),
            "shape_swing_score": _swing_score(p),
            "shape_hold_lock_score": _hold_lock_score(p),
            "shape_hold_strength": p.shape_hold_strength,
            "shape_draw_compression": p.shape_draw_compression,
            "shape_swing_instability": p.shape_swing_instability,
            "shape_path_imbalance": p.shape_path_imbalance,
            "shape_entropy": p.shape_entropy,
            "shape_directionality": p.shape_directionality,
            "topology_asymmetry_score": _topology_asymmetry_score(p),
            "admission_draw_score": _draw_admission_score(p),
            "admission_escape_draw_score": _admission_escape_score(p, "0"),
            "admission_escape_home_score": _admission_escape_score(p, "1"),
            "admission_escape_away_score": _admission_escape_score(p, "2"),
            "shape_cluster_label": _cluster_label(p),
            "shape_overlap_subtype": _overlap_subtype(p),
            "lab_draw_pressure_score": _draw_pressure_score(p),
            "lab_swing_score": _swing_score(p),
            "lab_hold_lock_score": _hold_lock_score(p),
            "lab_cluster_label": _cluster_label(p),
            "lab_overlap_subtype": _overlap_subtype(p),
            "lab_matchup_profile": p.lab_matchup_profile,
            "lab_hold_weight": p.lab_hold_weight,
            "lab_stall_weight": p.lab_stall_weight,
            "lab_flip_weight": p.lab_flip_weight,
            "lab_volatility_score": p.lab_volatility_score,
            "lab_draw_tension_score": p.lab_draw_tension_score,
            "lab_dynamic_swing_score": p.lab_dynamic_swing_score,
            "lab_hold_foundation_score": p.lab_hold_foundation_score,
            "lab_stall_compactness_score": p.lab_stall_compactness_score,
            "lab_flip_dislocation_score": p.lab_flip_dislocation_score,
            "lab_home_path_score": p.lab_home_path_score,
            "lab_away_path_score": p.lab_away_path_score,
            "lab_scenario_entropy_score": p.lab_scenario_entropy_score,
            "lab_draw_path_score": p.lab_draw_path_score,
            "lab_mix_home": p.lab_mix_home,
            "lab_mix_draw": p.lab_mix_draw,
            "lab_mix_away": p.lab_mix_away,
            "lab_tempo_band": p.lab_tempo_band,
            "lab_control_band": p.lab_control_band,
            "lab_pressure_band": p.lab_pressure_band,
            "lab_state_notes": p.lab_state_notes,
            "lab_basis_hint": p.lab_basis_hint,
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
            f"scenario{i + 1:02d}:{s.scenario_name}（{SCENARIO_JA_BY_ID.get(s.scenario_id, '未定義')}）"
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
    cluster_parts: List[str] = []
    basis_parts: List[str] = []
    for idx in range(1, REQUIRED_TICKET_COUNT + 1):
        cluster_bits = [
            f"draw={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_draw_core_changes', 0))}",
            f"overlap={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_overlap_changes', 0))}",
            f"swing={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_swing_core_changes', 0))}",
            f"neutral={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_neutral_changes', 0))}",
            f"hold={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_hold_core_changes', 0))}",
        ]
        cluster_parts.append(f"候補{idx:02d}={' '.join(cluster_bits)}")
        basis_bits = [
            f"side={int(ticket_stats.get(f'ticket_{idx:02d}_basis_side_strong_changes', 0))}",
            f"drawcmp={int(ticket_stats.get(f'ticket_{idx:02d}_basis_draw_compressed_changes', 0))}",
            f"trap={int(ticket_stats.get(f'ticket_{idx:02d}_basis_flat_draw_trap_changes', 0))}",
            f"split={int(ticket_stats.get(f'ticket_{idx:02d}_basis_split_side_changes', 0))}",
            f"balanced={int(ticket_stats.get(f'ticket_{idx:02d}_basis_basis_balanced_changes', 0))}",
        ]
        basis_parts.append(f"候補{idx:02d}={' '.join(basis_bits)}")
    html.append(f"<div class='note'>候補別クラスタ採用: {' / '.join(cluster_parts)}</div>")
    html.append(f"<div class='note'>候補別basis_hint採用: {' / '.join(basis_parts)}</div>")
    role_overlap_summary = str(ticket_stats.get("ticket_06_role_overlap_summary", "")).strip()
    if role_overlap_summary:
        html.append(
            "<div class='note'>"
            f"候補06 role観測: mode={str(ticket_stats.get('ticket_06_mode', 'none'))} / "
            f"unique_only={int(ticket_stats.get('ticket_06_role_unique_only_count', 0))} / "
            f"{role_overlap_summary}"
            "</div>"
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
        "<th class='left'>context</th><th class='left'>profile/basis</th><th class='left'>best/second</th>"
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
        if any([
            p.context_primary_pick,
            p.context_secondary_pick,
            p.context_admission_policy,
            p.context_draw_purchase_tier,
            p.context_draw_core_flag,
            p.context_risk_level,
            p.context_ticket_guidance,
            p.context_decision_summary,
        ]):
            context_parts = []
            if p.context_primary_pick:
                context_parts.append(f"本命={p.context_primary_pick}")
            if p.context_secondary_pick:
                context_parts.append(f"次点={p.context_secondary_pick}")
            if p.context_rank1_pick or p.context_rank2_pick or p.context_rank3_pick:
                context_parts.append(
                    f"rank={p.context_rank1_pick}>{p.context_rank2_pick}>{p.context_rank3_pick}"
                )
            if p.context_match_purchase_type:
                context_parts.append(f"type={p.context_match_purchase_type}")
            if p.context_match_purchase_subtype:
                context_parts.append(f"sub={p.context_match_purchase_subtype}")
            if p.context_purchase_type_confidence:
                context_parts.append(f"conf={p.context_purchase_type_confidence}")
            if p.context_admission_policy:
                context_parts.append(f"policy={p.context_admission_policy}")
            if p.context_draw_purchase_tier:
                context_parts.append(f"tier={p.context_draw_purchase_tier}")
            if p.context_draw_core_flag:
                context_parts.append("draw_core=1")
            if p.context_risk_level:
                context_parts.append(f"risk={p.context_risk_level}")
            if p.context_ticket_guidance:
                context_parts.append(f"guide={p.context_ticket_guidance}")
            if p.context_decision_summary:
                context_parts.append(p.context_decision_summary)
            context_text = " / ".join(context_parts)
        html.append(f"<td class='left'>{context_text}</td>")
        profile_basis = " / ".join(
            x for x in [
                (f"profile={p.lab_matchup_profile}" if p.lab_matchup_profile else ""),
                (f"basis={p.lab_basis_hint}" if p.lab_basis_hint else ""),
            ] if x
        )
        html.append(f"<td class='left'>{profile_basis}</td>")
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

    html.append("<h3>Scenario Changes</h3>")
    html.append("<table>")
    html.append("<thead><tr><th>scenario</th><th class='left'>scenario_name</th><th class='left'>description</th></tr></thead><tbody>")
    for t_idx in range(REQUIRED_TICKET_COUNT):
        d = flip_descs[t_idx] if t_idx < len(flip_descs) else "（未生成）"
        sname = scenario_defs[t_idx].scenario_name if t_idx < len(scenario_defs) else "N/A"
        sid = scenario_defs[t_idx].scenario_id if t_idx < len(scenario_defs) else ""
        sja = SCENARIO_JA_BY_ID.get(sid, "未定義")
        html.append(
            f"<tr><td>scenario{t_idx + 1:02d}</td><td class='left'>{sname}（{sja}）</td><td class='left'>{d}</td></tr>"
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
    # Accept both symbolic values (H/D/A) and numeric CSV exports (1/0/2, 1.0/0.0/2.0).
    out["actual_symbol"] = out[result_col].apply(_normalize_symbol_token)
    out.loc[out["actual_symbol"] == "", "actual_symbol"] = pd.NA
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
    experiment_root = os.path.join(purchase_dir, "current_predictions_experiment")
    index_path = os.path.join(purchase_dir, "buyplan_simulation.html")
    if (
        outdir_abs == purchase_dir
        or outdir_abs.startswith(backtest_root + os.sep)
        or outdir_abs.startswith(experiment_root + os.sep)
    ):
        return os.path.relpath(index_path, outdir_abs)
    return ""


def _write_buyplan_scored_outputs(
    plans: List[MatchPlan],
    tickets: List[List[str]],
    outdir: str,
    actual_df: pd.DataFrame,
    ticket_stats: Dict[str, int],
    actual_csv_path: str = "",
    output_name: str = "buyplan",
) -> Tuple[str, str]:
    _validate_actual_alignment(plans, actual_df, actual_csv_path=actual_csv_path)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: List[Dict[str, object]] = []
    actual_map = {int(r["match_no"]): (str(r["actual_symbol"]), str(r["actual_result"])) for _, r in actual_df.iterrows()}
    missing_match_nos = [int(p.match_no) for p in plans if int(p.match_no) not in actual_map]
    if missing_match_nos:
        missing_text = ", ".join(f"M{match_no:02d}" for match_no in missing_match_nos)
        print(f"[WARN] actual結果未確定の試合を未採点で残します: {missing_text}")

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
    n = int(scored_df["actual_symbol"].astype(str).ne("").sum()) if "actual_symbol" in scored_df.columns else 0
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
    cluster_label_map: Dict[int, str] = {}
    basis_label_map: Dict[int, str] = {}
    for idx in range(1, REQUIRED_TICKET_COUNT + 1):
        cluster_label_map[idx] = " ".join(
            [
                f"draw={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_draw_core_changes', 0))}",
                f"overlap={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_overlap_changes', 0))}",
                f"swing={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_swing_core_changes', 0))}",
                f"neutral={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_neutral_changes', 0))}",
                f"hold={int(ticket_stats.get(f'ticket_{idx:02d}_cluster_hold_core_changes', 0))}",
            ]
        )
        basis_label_map[idx] = " ".join(
            [
                f"side={int(ticket_stats.get(f'ticket_{idx:02d}_basis_side_strong_changes', 0))}",
                f"drawcmp={int(ticket_stats.get(f'ticket_{idx:02d}_basis_draw_compressed_changes', 0))}",
                f"trap={int(ticket_stats.get(f'ticket_{idx:02d}_basis_flat_draw_trap_changes', 0))}",
                f"split={int(ticket_stats.get(f'ticket_{idx:02d}_basis_split_side_changes', 0))}",
                f"balanced={int(ticket_stats.get(f'ticket_{idx:02d}_basis_basis_balanced_changes', 0))}",
            ]
        )
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
    draw_family_ids = [2, 5, 6, 8, 9]
    swing_family_ids = [3, 7, 10]
    role_family_map: Dict[int, str] = {}
    role_unique_only_map: Dict[int, int] = {}
    role_same_family_map: Dict[int, int] = {}
    role_diff_family_map: Dict[int, int] = {}
    if "ticket01" in scored_df.columns:
        base_series = scored_df["ticket01"].astype(str)
        role_family_specs = {
            "draw": draw_family_ids,
            "swing": swing_family_ids,
        }
        for family_name, family_ids in role_family_specs.items():
            for idx in family_ids:
                role_family_map[idx] = family_name
                ticket_col = f"ticket{idx:02d}"
                if ticket_col not in scored_df.columns:
                    role_unique_only_map[idx] = 0
                    role_same_family_map[idx] = 0
                    role_diff_family_map[idx] = 0
                    continue
                ticket_series = scored_df[ticket_col].astype(str)
                unique_only = 0
                same_family = 0
                diff_family = 0
                peer_ids = [peer for peer in family_ids if peer != idx and f"ticket{peer:02d}" in scored_df.columns]
                for row_i in range(len(scored_df)):
                    base_symbol = str(base_series.iat[row_i])
                    current_symbol = str(ticket_series.iat[row_i])
                    if current_symbol == base_symbol:
                        continue
                    peers_changed = []
                    for peer in peer_ids:
                        peer_symbol = str(scored_df[f"ticket{peer:02d}"].astype(str).iat[row_i])
                        if peer_symbol != base_symbol:
                            peers_changed.append(peer_symbol)
                    if not peers_changed:
                        unique_only += 1
                    elif any(sym == current_symbol for sym in peers_changed):
                        same_family += 1
                    else:
                        diff_family += 1
                role_unique_only_map[idx] = int(unique_only)
                role_same_family_map[idx] = int(same_family)
                role_diff_family_map[idx] = int(diff_family)

    for t_idx in range(REQUIRED_TICKET_COUNT):
        idx = t_idx + 1
        summary_df.loc[t_idx, "draw_count"] = int(draw_count_map.get(idx, 0))
        summary_df.loc[t_idx, "sway_count"] = int(sway_count_map.get(idx, 0))
        summary_df.loc[t_idx, "cluster_changes"] = cluster_label_map.get(idx, "")
        summary_df.loc[t_idx, "basis_changes"] = basis_label_map.get(idx, "")
        summary_df.loc[t_idx, "role_family"] = str(role_family_map.get(idx, "none"))
        summary_df.loc[t_idx, "role_unique_only"] = int(role_unique_only_map.get(idx, 0))
        summary_df.loc[t_idx, "role_same_family"] = int(role_same_family_map.get(idx, 0))
        summary_df.loc[t_idx, "role_diff_family"] = int(role_diff_family_map.get(idx, 0))
        summary_df.loc[t_idx, "role_mode"] = str(ticket_stats.get(f"ticket_{idx:02d}_mode", "none"))
        summary_df.loc[t_idx, "topology_mode"] = str(ticket_stats.get("topology_mode", "unknown"))
        summary_df.loc[t_idx, "topology_draw_cluster_count"] = int(ticket_stats.get("topology_draw_cluster_count", 0) or 0)
        summary_df.loc[t_idx, "topology_swing_cluster_count"] = int(ticket_stats.get("topology_swing_cluster_count", 0) or 0)
        summary_df.loc[t_idx, "topology_hold_cluster_count"] = int(ticket_stats.get("topology_hold_cluster_count", 0) or 0)

    safe_output_name = (output_name or "buyplan").strip() or "buyplan"
    out_scored_csv = os.path.join(outdir, f"{safe_output_name}_scored.csv")
    out_scored_summary_csv = os.path.join(outdir, f"{safe_output_name}_scored_summary.csv")
    outdir_abs = os.path.abspath(outdir)
    purchase_dir = os.path.join(BASE_DIR, "data", "purchase_reference")
    backtest_root = os.path.join(purchase_dir, "backtest")
    experiment_root = os.path.join(purchase_dir, "current_predictions_experiment")
    use_simulation_name = (
        outdir_abs == purchase_dir
        or outdir_abs.startswith(backtest_root + os.sep)
        or outdir_abs.startswith(experiment_root + os.sep)
    )
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
    cluster_count_parts = [f"候補{idx:02d}={cluster_label_map.get(idx, '')}" for idx in range(1, REQUIRED_TICKET_COUNT + 1)]
    html.append(f"<p>候補別クラスタ採用: <b>{' / '.join(cluster_count_parts)}</b></p>")
    basis_count_parts = [f"候補{idx:02d}={basis_label_map.get(idx, '')}" for idx in range(1, REQUIRED_TICKET_COUNT + 1)]
    html.append(f"<p>候補別basis_hint採用: <b>{' / '.join(basis_count_parts)}</b></p>")
    draw_role_parts: List[str] = []
    for idx in draw_family_ids:
        draw_role_parts.append(
            f"候補{idx:02d}=mode:{str(ticket_stats.get(f'ticket_{idx:02d}_mode', 'none'))}"
            f"/unique:{int(role_unique_only_map.get(idx, 0))}"
            f"/same:{int(role_same_family_map.get(idx, 0))}"
            f"/diff:{int(role_diff_family_map.get(idx, 0))}"
        )
    html.append(f"<p>draw系role観測: <b>{' / '.join(draw_role_parts)}</b></p>")
    swing_role_parts: List[str] = []
    for idx in swing_family_ids:
        swing_role_parts.append(
            f"候補{idx:02d}=mode:{str(ticket_stats.get(f'ticket_{idx:02d}_mode', 'none'))}"
            f"/unique:{int(role_unique_only_map.get(idx, 0))}"
            f"/same:{int(role_same_family_map.get(idx, 0))}"
            f"/diff:{int(role_diff_family_map.get(idx, 0))}"
        )
    html.append(f"<p>swing系role観測: <b>{' / '.join(swing_role_parts)}</b></p>")
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
    if BUYPLAN_ALLOCATOR_ENGINE == "legacy_scenario":
        tickets, flip_descs, scenario_defs, ticket_stats, scenario_result_labels = _generate_tickets_by_scenario(df, plans, warnings)
    else:
        tickets, flip_descs, scenario_defs, ticket_stats, scenario_result_labels = _generate_tickets_by_world_allocator(df, plans, warnings)

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
                ticket_stats=ticket_stats,
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
