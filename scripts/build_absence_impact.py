#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
欠場管理CSV + 選手マスタCSV から欠場影響を算出してCSV出力する。

入力:
- 欠場管理CSV(例: data/manual/欠場管理リスト.csv)
  - 先頭のメモ行は自動スキップ
  - 必須: season, round_start, team, player_name
  - 推奨: expected_rounds, absence_type(or reason), availability, note(or detail)
- 選手マスタCSV(例: data/manual/jleague_club_players.csv)
  - 必須: team, player_name, minutes(またはappearances), goals
  - 任意: assists, shots, xg, xga, conceded_on_pitch, starts, appearances, season

出力:
- absences_with_impact.csv
  - 欠場行ごとの照合結果 + weight/impact + debug_reason
- team_shares.csv
  - チーム別(かつ round_start 単位)の集計

計算:
- weight_minutes = player_minutes / team_minutes_total
- weight_attack  = player_goals / team_goals_total
- weight_defense = conceded_on_pitch or xga の比率。無ければ weight_minutes
- impact_total   = 0.6*weight_minutes + 0.2*weight_attack + 0.2*weight_defense
- availability が doubt は 0.5倍、returned は 0
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd


ABSENCE_HEADER_MAP = {
    "season": "season",
    "round_start": "round_start",
    "expected_rounds": "expected_rounds",
    "team": "team",
    "player_name": "player_name",
    "absence_type": "absence_type",
    "reason": "absence_type",
    "availability": "availability",
    "note": "note",
    "detail": "note",
}

TEAM_ALIAS_RAW = {
    "鹿島アントラーズ": "鹿島",
    "水戸ホーリーホック": "水戸",
    "浦和レッズ": "浦和",
    "ジェフユナイテッド千葉": "千葉",
    "柏レイソル": "柏",
    "ＦＣ東京": "FC東京",
    "FC東京": "FC東京",
    "東京ヴェルディ": "東京Ｖ",
    "ＦＣ町田ゼルビア": "町田",
    "川崎フロンターレ": "川崎Ｆ",
    "横浜F・マリノス": "横浜FM",
    "横浜Ｆ・マリノス": "横浜FM",
    "清水エスパルス": "清水",
    "名古屋グランパス": "名古屋",
    "京都サンガF.C.": "京都",
    "ガンバ大阪": "Ｇ大阪",
    "セレッソ大阪": "Ｃ大阪",
    "ヴィッセル神戸": "神戸",
    "ファジアーノ岡山": "岡山",
    "サンフレッチェ広島": "広島",
    "アビスパ福岡": "福岡",
    "V・ファーレン長崎": "長崎",
    "Ｖ・ファーレン長崎": "長崎",
    "北海道コンサドーレ札幌": "札幌",
    "ヴァンラーレ八戸": "八戸",
    "ベガルタ仙台": "仙台",
    "ブラウブリッツ秋田": "秋田",
    "モンテディオ山形": "山形",
    "ザスパ群馬": "群馬",
    "いわきＦＣ": "いわき",
    "いわきFC": "いわき",
    "SC相模原": "相模原",
    "ＳＣ相模原": "相模原",
    "FC岐阜": "岐阜",
    "ＦＣ岐阜": "岐阜",
    "AC長野パルセイロ": "長野",
    "ＡＣ長野パルセイロ": "長野",
    "松本山雅FC": "松本",
    "松本山雅ＦＣ": "松本",
    "栃木シティ": "栃木Ｃ",
    "栃木シティFC": "栃木Ｃ",
    "栃木シティＦＣ": "栃木Ｃ",
    "RB大宮アルディージャ": "大宮",
    "ＲＢ大宮アルディージャ": "大宮",
    "横浜ＦＣ": "横浜FC",
    "湘南ベルマーレ": "湘南",
    "ヴァンフォーレ甲府": "甲府",
    "アルビレックス新潟": "新潟",
    "カターレ富山": "富山",
    "ツエーゲン金沢": "金沢",
    "ジュビロ磐田": "磐田",
    "藤枝ＭＹＦＣ": "藤枝",
    "藤枝MYFC": "藤枝",
    "徳島ヴォルティス": "徳島",
    "ＦＣ今治": "今治",
    "FC今治": "今治",
    "ＦＣ琉球": "琉球",
    "FC琉球": "琉球",
    "ギラヴァンツ北九州": "北九州",
    "ガイナーレ鳥取": "鳥取",
    "カマタマーレ讃岐": "讃岐",
    "鹿児島ユナイテッドFC": "鹿児島",
    "鹿児島ユナイテッドＦＣ": "鹿児島",
    "福島ユナイテッドFC": "福島",
    "福島ユナイテッドＦＣ": "福島",
    "高知ユナイテッドSC": "高知",
    "高知ユナイテッドＳＣ": "高知",
    "コンサドーレ札幌": "札幌",
    "サガン鳥栖": "鳥栖",
    "大分トリニータ": "大分",
    "テゲバジャーロ宮崎": "宮崎",
    "レノファ山口FC": "山口",
    "栃木SC": "栃木SC",
}


def _norm_text(v: object) -> str:
    if v is None:
        return ""
    s = unicodedata.normalize("NFKC", str(v))
    s = s.strip()
    s = s.replace("　", " ")
    return s


TEAM_ALIAS_MAP = {
    unicodedata.normalize("NFKC", str(k)).strip(): unicodedata.normalize("NFKC", str(v)).strip()
    for k, v in TEAM_ALIAS_RAW.items()
}


def _norm_key(v: object) -> str:
    s = _norm_text(v)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"（[^）]*）", "", s)
    s = re.sub(r"[・.\-_/]", "", s)
    s = re.sub(r"\s+", "", s)
    return s.upper()


def canonical_team_name(v: object) -> str:
    s = _norm_text(v)
    s = TEAM_ALIAS_MAP.get(s, s)
    return s


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_absences(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    lines = raw.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        ls = line.strip().lstrip("\ufeff")
        if ls.startswith("season,") and "round_start" in ls and "player_name" in ls:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("欠場CSVのヘッダ行を検出できませんでした。")

    payload = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(payload))
    rows: List[Dict[str, object]] = []
    for row in reader:
        if not row:
            continue
        normalized: Dict[str, object] = {}
        for k, v in row.items():
            if k is None:
                continue
            key = _norm_text(k).replace("\n", "").strip()
            key = ABSENCE_HEADER_MAP.get(key, key)
            normalized[key] = _norm_text(v)
        if not normalized.get("team") and not normalized.get("player_name"):
            continue
        rows.append(normalized)

    df = pd.DataFrame(rows)
    for col in ["season", "round_start", "expected_rounds"]:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ["absence_type", "availability", "note"]:
        if col not in df.columns:
            df[col] = ""

    df["season"] = to_num(df["season"]).astype("Int64")
    df["round_start"] = to_num(df["round_start"]).astype("Int64")
    df["expected_rounds"] = to_num(df["expected_rounds"]).astype("Int64")
    df["team"] = df["team"].map(canonical_team_name)
    df["player_name"] = df["player_name"].map(_norm_text)

    # availability未入力なら out 扱い
    df["availability"] = df["availability"].fillna("").astype(str).str.strip().str.lower()
    df.loc[df["availability"] == "", "availability"] = "out"
    # absence_typeは reason から入ってくる想定。無ければ other
    df["absence_type"] = df["absence_type"].fillna("").astype(str).str.strip().str.lower()
    df["absence_type"] = df["absence_type"].replace(
        {
            "ケガ": "injury",
            "怪我": "injury",
            "injury": "injury",
            "負傷": "injury",
            "カード": "card",
            "累積警告": "card",
            "suspension": "card",
            "card": "card",
            "代表招集": "national",
            "national": "national",
        }
    )
    df.loc[df["absence_type"] == "", "absence_type"] = "other"

    return df


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c: c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def load_players(path: Path, default_season: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise RuntimeError("選手マスタCSVが空です。")

    team_col = pick_first_existing(df, ["team", "team_name", "club_name", "club"])
    player_col = pick_first_existing(df, ["player_name", "名前", "name"])
    season_col = pick_first_existing(df, ["season", "year"])
    minutes_col = pick_first_existing(df, ["minutes", "出場時間", "minute"])
    apps_col = pick_first_existing(df, ["appearances", "出場 試合数 ※2", "出場試合数"])
    goals_col = pick_first_existing(df, ["goals", "ゴール 数 ※3", "ゴール数", "得点"])
    starts_col = pick_first_existing(df, ["starts", "先発数"])
    xga_col = pick_first_existing(df, ["xga"])
    conceded_col = pick_first_existing(df, ["conceded_on_pitch"])

    if team_col is None or player_col is None:
        raise RuntimeError("選手マスタCSVに team/player_name 相当列がありません。")
    if goals_col is None:
        raise RuntimeError("選手マスタCSVに goals 相当列がありません。")

    out = pd.DataFrame()
    out["team"] = df[team_col].map(canonical_team_name)
    out["player_name"] = df[player_col].map(_norm_text)
    if season_col:
        out["season"] = to_num(df[season_col]).astype("Int64")
    else:
        out["season"] = pd.Series([default_season] * len(df), dtype="Int64")

    if minutes_col:
        out["minutes"] = to_num(df[minutes_col]).fillna(0.0)
    else:
        out["minutes"] = pd.Series([0.0] * len(df))

    if apps_col:
        out["appearances"] = to_num(df[apps_col]).fillna(0.0)
    else:
        out["appearances"] = pd.Series([0.0] * len(df))

    # minutes が無い場合は appearances*90 を代替
    out.loc[out["minutes"] <= 0, "minutes"] = out.loc[out["minutes"] <= 0, "appearances"] * 90.0

    out["goals"] = to_num(df[goals_col]).fillna(0.0)
    out["starts"] = to_num(df[starts_col]).fillna(0.0) if starts_col else 0.0
    out["xga"] = to_num(df[xga_col]).fillna(0.0) if xga_col else pd.NA
    out["conceded_on_pitch"] = to_num(df[conceded_col]).fillna(0.0) if conceded_col else pd.NA

    out["team_key"] = out["team"].map(_norm_key)
    out["player_key"] = out["player_name"].map(_norm_key)
    return out


def match_player(
    players_team: pd.DataFrame,
    player_name: str,
    threshold: float = 0.85,
) -> Tuple[Optional[pd.Series], str, str]:
    if players_team.empty:
        return None, "not_found", "team not found"

    key = _norm_key(player_name)
    exact = players_team[players_team["player_key"] == key]
    if len(exact) >= 1:
        return exact.iloc[0], "matched", ""

    candidates = players_team["player_key"].dropna().astype(str).unique().tolist()
    if not candidates:
        return None, "not_found", "team has no players"

    best = difflib.get_close_matches(key, candidates, n=1, cutoff=threshold)
    if best:
        row = players_team[players_team["player_key"] == best[0]].iloc[0]
        return row, "fuzzy_matched", f"fuzzy:{row['player_name']}"

    return None, "not_found", "player not found"


def calc_team_totals(players: pd.DataFrame) -> pd.DataFrame:
    grouped = players.groupby(["season", "team"], dropna=False, as_index=False).agg(
        team_minutes_total=("minutes", "sum"),
        team_goals_total=("goals", "sum"),
        team_conceded_total=("conceded_on_pitch", "sum"),
        team_xga_total=("xga", "sum"),
    )
    return grouped


def availability_factor(v: str) -> float:
    s = _norm_text(v).lower()
    if s in {"returned", "return", "fit", "available"}:
        return 0.0
    if s in {"doubt", "questionable", "uncertain"}:
        return 0.5
    return 1.0


def build_impacts(absences: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    team_totals = calc_team_totals(players)
    total_map = {
        (r["season"], r["team"]): r
        for _, r in team_totals.iterrows()
    }

    rows: List[Dict[str, object]] = []
    for _, a in absences.iterrows():
        season = a.get("season")
        team = a.get("team")
        player_name = a.get("player_name")
        season_num = int(season) if pd.notna(season) else None
        team_key = _norm_key(team)

        p = players.copy()
        p = p[p["team_key"] == team_key]
        if season_num is not None:
            p = p[p["season"] == season_num]

        matched, status, reason = match_player(p, str(player_name))
        debug: List[str] = []
        if reason:
            debug.append(reason)

        key = f"{season_num}|{team_key}|{_norm_key(player_name)}"
        wt_m = 0.0
        wt_a = 0.0
        wt_d = 0.0
        impact_a = 0.0
        impact_d = 0.0
        impact_t = 0.0
        regular_flag = False

        totals = total_map.get((season_num, team))
        tm = float(totals["team_minutes_total"]) if totals is not None and pd.notna(totals["team_minutes_total"]) else 0.0
        tg = float(totals["team_goals_total"]) if totals is not None and pd.notna(totals["team_goals_total"]) else 0.0
        tc = float(totals["team_conceded_total"]) if totals is not None and pd.notna(totals["team_conceded_total"]) else 0.0
        tx = float(totals["team_xga_total"]) if totals is not None and pd.notna(totals["team_xga_total"]) else 0.0

        if matched is not None:
            pm = float(matched.get("minutes", 0.0) or 0.0)
            pg = float(matched.get("goals", 0.0) or 0.0)
            pstarts = float(matched.get("starts", 0.0) or 0.0)
            regular_flag = (pm > 0 and tm > 0 and (pm / tm) >= 0.5) or pstarts >= 2

            if tm > 0:
                wt_m = max(0.0, min(1.0, pm / tm))
            else:
                debug.append("team_minutes_total=0")

            if tg > 0:
                wt_a = max(0.0, min(1.0, pg / tg))
            else:
                debug.append("team_goals_total=0")

            # defense proxy: conceded_on_pitch -> xga -> minutes_share
            p_conceded = matched.get("conceded_on_pitch", pd.NA)
            p_xga = matched.get("xga", pd.NA)
            if pd.notna(p_conceded) and tc > 0:
                wt_d = max(0.0, min(1.0, float(p_conceded) / tc))
                debug.append("def_proxy=conceded_on_pitch")
            elif pd.notna(p_xga) and tx > 0:
                wt_d = max(0.0, min(1.0, float(p_xga) / tx))
                debug.append("def_proxy=xga")
            else:
                wt_d = wt_m
                debug.append("def_proxy=minutes_share")
        else:
            debug.append("not found")

        factor = availability_factor(str(a.get("availability", "")))
        if factor == 0.0:
            debug.append("availability=returned => impact=0")

        impact_a = wt_a * factor
        impact_d = wt_d * factor
        impact_t = (0.6 * wt_m + 0.2 * wt_a + 0.2 * wt_d) * factor

        expected_rounds = a.get("expected_rounds")
        impact_per_round = impact_t / float(expected_rounds) if pd.notna(expected_rounds) and float(expected_rounds) > 0 else pd.NA

        rows.append(
            {
                "season": season_num,
                "round_start": a.get("round_start"),
                "expected_rounds": expected_rounds,
                "team": team,
                "player_name": player_name,
                "absence_type": a.get("absence_type", ""),
                "availability": a.get("availability", ""),
                "note": a.get("note", ""),
                "match_player_key": key,
                "match_status": status,
                "weight_minutes": round(float(wt_m), 6),
                "weight_attack": round(float(wt_a), 6),
                "weight_defense": round(float(wt_d), 6),
                "impact_attack": round(float(impact_a), 6),
                "impact_defense": round(float(impact_d), 6),
                "impact_total": round(float(impact_t), 6),
                "impact_total_per_round": round(float(impact_per_round), 6) if pd.notna(impact_per_round) else pd.NA,
                "regular_flag": bool(regular_flag),
                "debug_reason": "; ".join(sorted(set([d for d in debug if d]))),
            }
        )
    return pd.DataFrame(rows)


def build_team_summary(impacts: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    totals = calc_team_totals(players).rename(
        columns={"team_conceded_total": "team_defense_total_conceded", "team_xga_total": "team_defense_total_xga"}
    )
    agg = impacts.groupby(["season", "team", "round_start"], dropna=False, as_index=False).agg(
        absences_count=("player_name", "count"),
        sum_weight_minutes=("weight_minutes", "sum"),
        sum_weight_attack=("weight_attack", "sum"),
        sum_weight_defense=("weight_defense", "sum"),
        sum_impact_attack=("impact_attack", "sum"),
        sum_impact_defense=("impact_defense", "sum"),
        sum_impact_total=("impact_total", "sum"),
    )
    out = agg.merge(totals, on=["season", "team"], how="left")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="欠場リストと選手マスタから影響スコアを算出")
    parser.add_argument("absences_csv", help="欠場管理CSV")
    parser.add_argument("players_csv", help="選手マスタCSV")
    parser.add_argument("--out-dir", default="data/manual", help="出力ディレクトリ")
    parser.add_argument("--season", type=int, default=None, help="players側にseason列が無い場合の既定値")
    parser.add_argument(
        "--asof-date",
        default=os.environ.get("ABSENCE_ASOF_DATE", ""),
        help="スナップショット日付(YYYY-MM-DD)。省略時は当日。",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="",
        help="欠場影響スナップショット保存先(任意)。",
    )
    args = parser.parse_args()

    abs_path = Path(args.absences_csv)
    players_path = Path(args.players_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    absences = load_absences(abs_path)
    players = load_players(players_path, args.season)
    impacts = build_impacts(absences, players)
    summary = build_team_summary(impacts, players)

    out_abs = out_dir / "absences_with_impact.csv"
    out_team = out_dir / "team_shares.csv"
    impacts.to_csv(out_abs, index=False, encoding="utf-8-sig")
    summary.to_csv(out_team, index=False, encoding="utf-8-sig")

    # 任意: as-of付きスナップショットを保存
    asof_key = "".join(ch for ch in str(args.asof_date or "") if ch.isdigit())
    if len(asof_key) < 8:
        asof_key = datetime.now().strftime("%Y%m%d")
    if args.snapshot_dir:
        snap_dir = Path(args.snapshot_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_abs = snap_dir / f"absences_with_impact_asof_{asof_key}.csv"
        snap_team = snap_dir / f"team_shares_asof_{asof_key}.csv"
        impacts.to_csv(snap_abs, index=False, encoding="utf-8-sig")
        summary.to_csv(snap_team, index=False, encoding="utf-8-sig")
        print(f"[OK] absence snapshot: {snap_abs}")
        print(f"[OK] team_shares snapshot: {snap_team}")

    print(f"[OK] absences_with_impact: {out_abs} rows={len(impacts)}")
    print(f"[OK] team_shares: {out_team} rows={len(summary)}")
    print(f"[INFO] matched={int((impacts['match_status']=='matched').sum())}, "
          f"fuzzy={int((impacts['match_status']=='fuzzy_matched').sum())}, "
          f"not_found={int((impacts['match_status']=='not_found').sum())}")


if __name__ == "__main__":
    main()
