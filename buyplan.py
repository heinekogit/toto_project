#!/usr/bin/env python3
import argparse
import itertools
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SYMBOLS = ["1", "0", "2"]  # home/draw/away
REQUIRED_MATCH_COUNT = 13
REQUIRED_TICKET_COUNT = 10
UNIQUE_VARIATION_TOPK = int(os.environ.get("BUYPLAN_UNIQUE_TOPK", "8"))
DEBUG_PROBS = os.environ.get("BUYPLAN_DEBUG_PROBS", "0") == "1"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_TOTO_ORDER_CSV = os.path.join(BASE_DIR, "data", "manual", "toto並び順.csv")
LOCK_BY_PBEST_MIN = float(os.environ.get("LOCK_BY_PBEST_MIN", "0.62"))
LOCK_BY_MARGIN_MIN = float(os.environ.get("LOCK_BY_MARGIN_MIN", "0.18"))


@dataclass(frozen=True)
class ScenarioDef:
    scenario_id: str
    scenario_name: str
    scenario_note: str


SCENARIO_DEFS: List[ScenarioDef] = [
    ScenarioDef("01", "Lock 01", "LOCK遵守（基準）"),
    ScenarioDef("02", "Lock 02", "LOCK遵守（best→second）"),
    ScenarioDef("03", "Lock 03", "LOCK遵守（best→second）"),
    ScenarioDef("04", "Prob 01", "確率忠実（基準）"),
    ScenarioDef("05", "Prob 02", "確率忠実（best→second）"),
    ScenarioDef("06", "Prob 03", "確率忠実（best→second）"),
    ScenarioDef("07", "Prob 04", "確率忠実（best→second）"),
    ScenarioDef("08", "Prob 05", "確率忠実（best→second）"),
    ScenarioDef("09", "Exp 01", "実験（best/second/third）"),
    ScenarioDef("10", "Exp 02", "実験（best/second/third）"),
]

SCENARIO_JA_BY_ID: Dict[str, str] = {
    "01": "LOCK遵守",
    "02": "LOCK遵守",
    "03": "LOCK遵守",
    "04": "確率忠実",
    "05": "確率忠実",
    "06": "確率忠実",
    "07": "確率忠実",
    "08": "確率忠実",
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
    s = _safe_text(v, "")
    s = s.replace("　", " ").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("・", "").replace(".", "").replace("･", "")
    return s


def _resolve_toto_order_csv_path(arg_path: str) -> str:
    if arg_path:
        return os.path.abspath(arg_path)
    # 既定ファイルを最優先
    if os.path.exists(DEFAULT_TOTO_ORDER_CSV):
        return DEFAULT_TOTO_ORDER_CSV
    # ファイル名ゆらぎ（結合文字違い）に備えて曖昧探索
    manual_dir = Path(BASE_DIR) / "data" / "manual"
    if manual_dir.exists():
        candidates = sorted(manual_dir.glob("toto*順*.csv"))
        if candidates:
            return str(candidates[0])
    return DEFAULT_TOTO_ORDER_CSV


def _load_toto_order_df(csv_path: str, warnings: List[str]) -> pd.DataFrame:
    if not csv_path or not os.path.exists(csv_path):
        _warn(warnings, f"toto並び順CSVが見つかりません: {csv_path}")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])
    try:
        # ヘッダなし想定: match_no, home_team, vs, away_team
        raw = pd.read_csv(csv_path, header=None, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        _warn(warnings, f"toto並び順CSVを読み込めません: {csv_path} ({e})")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])

    if raw.empty:
        _warn(warnings, f"toto並び順CSVが空です: {csv_path}")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])

    # ヘッダ付きに誤ってなっていても吸収
    if len(raw.columns) >= 4:
        df = raw.iloc[:, :4].copy()
        df.columns = ["match_no", "home_team", "vs", "away_team"]
    else:
        _warn(warnings, f"toto並び順CSVの列数不足: {csv_path} cols={len(raw.columns)}")
        return pd.DataFrame(columns=["match_no", "home_team", "away_team", "_home_key", "_away_key"])

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


def _build_match_plans(df: pd.DataFrame, warnings: List[str]) -> List[MatchPlan]:
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
        probs = [p_h, p_d, p_a]
        rank = sorted(enumerate(probs), key=lambda x: float(x[1]), reverse=True)
        best_idx, best_p = rank[0]
        second_idx, second_p = rank[1]
        third_idx, _ = rank[2]
        best_symbol = _to_symbol(best_idx)
        second_symbol = _to_symbol(second_idx)
        third_symbol = _to_symbol(third_idx)
        # 基準票は predicted_result を優先し、無効時は確率bestへフォールバック
        predicted_symbol = ""
        if "predicted_result" in row_df.columns:
            predicted_symbol = _result_to_symbol(row.get("predicted_result"))
        if not predicted_symbol and "predicted_highest_prob_result" in row_df.columns:
            predicted_symbol = _result_to_symbol(row.get("predicted_highest_prob_result"))
        base_pick = predicted_symbol if predicted_symbol in {"1", "0", "2"} else best_symbol
        base_from_predicted = predicted_symbol in {"1", "0", "2"}
        if DEBUG_PROBS:
            print(
                f"[BUYPLAN_DEBUG] M{m:02d} {home} vs {away} "
                f"probs=[H:{float(probs[0]):.6f}, D:{float(probs[1]):.6f}, A:{float(probs[2]):.6f}] "
                f"base={base_pick} best={best_symbol} second={second_symbol} third={third_symbol} "
                f"using=({ph_col},{pd_col},{pa_col})"
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
                margin=float(best_p - second_p),
                p_home=p_h,
                p_draw=p_d,
                p_away=p_a,
                p_best=float(best_p),
                p_second=float(second_p),
                prob_best_pick=best_symbol,
                prob_second_pick=second_symbol,
                prob_margin=float(best_p - second_p),
                base_from_predicted=base_from_predicted,
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


def _symbol_count(ticket: List[str], sym: str) -> int:
    return sum(1 for x in ticket if x == sym)


def _ratio_str(v: int, total: int) -> str:
    if total <= 0:
        return "0.000"
    return f"{(v / total):.3f}"


def _generate_tickets(plans: List[MatchPlan], warnings: List[str]) -> Tuple[List[List[str]], List[str], Dict[str, int]]:
    tickets: List[List[str]] = []
    descs: List[str] = []
    seen = set()

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
    }

    mode_stats: Dict[str, Dict[str, float]] = {
        "lock_strict": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
        "prob_faithful": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
        "experimental": {"tickets": 0, "zero_count": 0, "flip_count": 0, "margin_sum": 0.0, "margin_n": 0},
    }

    lock_indices = {
        i
        for i, p in enumerate(plans)
        if p.status == "OK"
        and p.prob_margin is not None
        and p.p_best >= LOCK_BY_PBEST_MIN
        and p.prob_margin >= LOCK_BY_MARGIN_MIN
    }
    stats["locked_count"] = len(lock_indices)

    ok_all = [i for i, p in enumerate(plans) if p.status == "OK" and p.prob_margin is not None]
    sorted_all = sorted(ok_all, key=lambda i: plans[i].prob_margin if plans[i].prob_margin is not None else 999.0)
    sorted_non_lock = [i for i in sorted_all if i not in lock_indices]
    stats["second_zero_matches_all_ok"] = sum(1 for i in ok_all if plans[i].second == "0")
    stats["second_zero_matches"] = sum(1 for i in sorted_non_lock if plans[i].second == "0")

    # 10票固定: 01-03(lock strict), 04-08(prob faithful), 09-10(experimental)
    mode_by_ticket = [_mode_for_ticket_index(i) for i in range(REQUIRED_TICKET_COUNT)]

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
                cands.append({src[1]: 1})
                cands.append({src[0]: 1, src[1]: 1})
            if len(src) >= 3:
                cands.append({src[2]: 1})
                cands.append({src[0]: 1, src[2]: 1})
            return cands

        if mode == "prob_faithful":
            # 確率忠実: lock無効、best->secondのみ。
            # flip下限2 / 上限3（third禁止）
            src = sorted_all
            cands: List[Dict[int, int]] = []
            if len(src) >= 2:
                n = len(src)
                # 低margin先頭に偏らないよう、全体に分散して候補を作る
                anchors = sorted(
                    set(
                        [
                            0,
                            max(0, n - 3),      # 高margin側も早めに拾う
                            max(0, n // 4),
                            max(0, n // 2),
                            max(0, (3 * n) // 4),
                            max(0, n - 2),
                        ]
                    ),
                    key=lambda x: (
                        0 if x == 0 else 1 if x == max(0, n - 3) else 2 + x
                    ),
                )
                for a in anchors:
                    b = a + 1 if a + 1 < n else a - 1
                    if b < 0:
                        continue
                    flip2 = {src[a]: 1, src[b]: 1}
                    cands.append(flip2)
                    # 一部は3flipに拡張（上限3）
                    c = a + 2 if a + 2 < n else (a - 2 if a - 2 >= 0 else None)
                    if c is not None:
                        flip3 = {src[a]: 1, src[b]: 1, src[c]: 1}
                        cands.append(flip3)

            # 重複flip mapを除去し、順序を保持
            dedup = []
            seen_keys = set()
            for fm in cands:
                key = tuple(sorted(fm.items()))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                dedup.append(fm)
            cands = dedup
            return cands

        # experimental: third許可、1票あたりthird最大2
        src = sorted_all
        cands = []
        cands.append({})
        if len(src) >= 1:
            cands.append({src[0]: 2})
            cands.append({src[0]: 1})
        if len(src) >= 2:
            cands.append({src[0]: 2, src[1]: 1})
            cands.append({src[0]: 1, src[1]: 2})
            cands.append({src[0]: 2, src[1]: 2})  # third=2件
            cands.append({src[0]: 1, src[1]: 1})
        if len(src) >= 3:
            cands.append({src[0]: 2, src[2]: 1})
            cands.append({src[1]: 2, src[2]: 1})
            cands.append({src[0]: 1, src[1]: 1, src[2]: 2})
        return cands

    # modeごとの候補プールを作成
    pool_by_mode = {
        "lock_strict": candidate_flip_maps("lock_strict"),
        "prob_faithful": candidate_flip_maps("prob_faithful"),
        "experimental": candidate_flip_maps("experimental"),
    }
    pool_pos = {"lock_strict": 0, "prob_faithful": 0, "experimental": 0}

    # ticketごとに mode固定でユニーク生成
    for t_idx in range(REQUIRED_TICKET_COUNT):
        mode = mode_by_ticket[t_idx]
        pool = pool_by_mode[mode]
        chosen_ticket: Optional[List[str]] = None
        chosen_desc = "base"
        chosen_flips: Dict[int, int] = {}

        while pool_pos[mode] < len(pool):
            flips = pool[pool_pos[mode]]
            pool_pos[mode] += 1
            # lock strict は LOCK試合を強制除外
            if mode == "lock_strict" and any(idx in lock_indices for idx in flips.keys()):
                continue
            # experimental は third最大2件
            if mode == "experimental" and sum(1 for v in flips.values() if v == 2) > 2:
                continue

            stats["attempted_candidates"] += 1
            t = _build_ticket_from_flips(plans, flips)
            k = _ticket_key(t)
            if k in seen:
                stats["duplicate_skips"] += 1
                continue
            seen.add(k)
            chosen_ticket = t
            chosen_desc = _flip_desc(plans, flips)
            chosen_flips = flips
            stats["generated"] += 1
            stats["second_zero_applied"] += sum(
                1 for idx, which in flips.items() if which == 1 and plans[idx].second == "0"
            )
            break

        if chosen_ticket is None:
            # プール切れ時は base を使う（重複の場合は最後の手段）
            fallback = _build_ticket_from_flips(plans, {})
            k = _ticket_key(fallback)
            if k in seen:
                # それでも重複なら最初のOK試合を second にして逃がす
                rescue_flips: Dict[int, int] = {}
                if sorted_all:
                    rescue_flips = {sorted_all[min(t_idx, len(sorted_all) - 1)]: 1}
                fallback = _build_ticket_from_flips(plans, rescue_flips)
                k = _ticket_key(fallback)
                if k in seen:
                    stats["duplicate_skips"] += 1
                else:
                    chosen_desc = _flip_desc(plans, rescue_flips)
                    chosen_flips = rescue_flips
                    seen.add(k)
                    chosen_ticket = fallback
                    stats["generated"] += 1
            else:
                seen.add(k)
                chosen_ticket = fallback
                chosen_desc = "base"
                chosen_flips = {}
                stats["generated"] += 1

        if chosen_ticket is None:
            chosen_ticket = [p.base_pick for p in plans]
            chosen_desc = "base"
            chosen_flips = {}

        tickets.append(chosen_ticket)
        descs.append(chosen_desc)
        record_mode_stats(mode, chosen_ticket, chosen_flips)

    if len(tickets) < REQUIRED_TICKET_COUNT:
        _warn(warnings, f"重複なしで {REQUIRED_TICKET_COUNT} 口を作れず {len(tickets)} 口になりました。")

    stats["unique_ticket_count"] = len(tickets)
    stats["duplicate_count"] = max(0, stats["attempted_candidates"] - stats["unique_ticket_count"])

    # mode stats flatten
    for mode in ["lock_strict", "prob_faithful", "experimental"]:
        ms = mode_stats[mode]
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
    return tickets, descs, stats


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
    _ = df
    tickets, flip_descs, stats = _generate_tickets(plans, warnings)
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
        "全体比率: "
        f"0総出現回数={stats.get('total_zero_count', 0)}, "
        f"1/0/2比率={stats.get('total_ratio_1', '0.000')}/"
        f"{stats.get('total_ratio_0', '0.000')}/"
        f"{stats.get('total_ratio_2', '0.000')}",
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
    html.append("<title>BuyPlan（toto 13試合）</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui,-apple-system,sans-serif;margin:20px;line-height:1.45;color:#111;}")
    html.append(".header{border:1px solid #ddd;background:#fafafa;padding:12px 14px;margin-bottom:12px;}")
    html.append(".title{font-size:22px;font-weight:700;margin-bottom:6px;}")
    html.append(".sub{font-size:12px;color:#444;display:flex;flex-wrap:wrap;gap:10px 18px;margin-bottom:6px;}")
    html.append(".note{font-size:12px;color:#555;}")
    html.append(".table-wrap{overflow-x:auto;border:1px solid #ddd;}")
    html.append("table{border-collapse:collapse;width:100%;font-size:12px;min-width:1300px;}")
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
    html.append("<div class='title'>BuyPlan（toto 13試合）</div>")
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
        "<th class='match'>match_no</th><th class='left'>league</th><th class='left'>home_team</th><th class='left'>away_team</th><th class='left'>best/second</th>"
    )
    for t_idx in range(REQUIRED_TICKET_COUNT):
        idx = t_idx + 1
        label = f"候補{idx:02d}"
        if t_idx == 0:
            label += "(基準)"
        sname = scenario_defs[t_idx].scenario_name if t_idx < len(scenario_defs) else "N/A"
        sid = scenario_defs[t_idx].scenario_id if t_idx < len(scenario_defs) else ""
        sja = SCENARIO_JA_BY_ID.get(sid, "未定義")
        html.append(f"<th class='cand-head'>{label}<br><small>{sname}（{sja}）</small></th>")
    html.append("</tr></thead><tbody>")
    for i, p in enumerate(plans):
        html.append("<tr>")
        html.append(f"<td class='match'>{p.match_no}</td>")
        html.append(f"<td class='left'>{p.league}</td>")
        html.append(f"<td class='left'>{p.home_team}</td>")
        html.append(f"<td class='left'>{p.away_team}</td>")
        best_second = (
            f"best={p.prob_best_pick} second={p.prob_second_pick} (base={p.base_pick})"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toto buy plan (single x10) from predictions.csv")
    parser.add_argument("--in", dest="in_csv", required=True, help="input predictions.csv path")
    parser.add_argument("--scenario", default="", help="deprecated: ignored (scenario is fixed per 候補01..10)")
    parser.add_argument("--scenario-file", default="", help="deprecated: ignored (scenario is fixed per 候補01..10)")
    parser.add_argument("--scenario-dir", default="", help="deprecated: ignored (scenario is fixed per 候補01..10)")
    parser.add_argument("--outdir", default="", help="output directory (default: input file dir)")
    parser.add_argument(
        "--toto-order-csv",
        default="",
        help="toto販売サイト順CSV（既定: data/manual/toto並び順.csv）",
    )
    args = parser.parse_args()

    warnings: List[str] = []
    in_csv = os.path.abspath((args.in_csv or "").strip())
    if args.scenario or args.scenario_file or args.scenario_dir:
        _warn(warnings, "--scenario / --scenario-file / --scenario-dir は非推奨です。候補01〜10の内部シナリオを使用します。")
    outdir = args.outdir.strip() if isinstance(args.outdir, str) else ""
    if not outdir:
        outdir = os.path.dirname(os.path.abspath(in_csv)) or "."
    os.makedirs(outdir, exist_ok=True)

    try:
        df = pd.read_csv(in_csv)
    except Exception as e:
        print(f"[ERROR] 入力CSVを読み込めません: {in_csv} ({e})")
        return

    toto_order_csv = _resolve_toto_order_csv_path(args.toto_order_csv.strip() if isinstance(args.toto_order_csv, str) else "")
    order_df = _load_toto_order_df(toto_order_csv, warnings)
    diff_df = _build_toto_diff_report(df, order_df)
    if not diff_df.empty:
        diff_csv = os.path.join(outdir, "buyplan_toto_diff.csv")
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
    plans = _build_match_plans(df, warnings)
    tickets, flip_descs, scenario_defs, ticket_stats, scenario_result_labels = _generate_tickets_by_scenario(df, plans, warnings)

    out_csv = os.path.join(outdir, "buyplan.csv")
    out_html = os.path.join(outdir, "buyplan.html")
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

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_html}")


if __name__ == "__main__":
    main()
