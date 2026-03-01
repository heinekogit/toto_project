#!/usr/bin/env python3
"""
build_team_impact_weights.py

Usage:
  python build_team_impact_weights.py --input players.csv --output weights.csv
  python build_team_impact_weights.py --input players.csv --output weights.csv --team-matches 38
  python build_team_impact_weights.py --demo --output /tmp/weights_demo.csv
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = ["team_name", "player_name", "position", "appearances", "goals"]

DEFENSE_WEIGHTS = {
    "GK": 1.0,
    "DF": 0.8,
    "MF": 0.4,
    "FW": 0.2,
}

DEFAULT_POS_MAP = {
    "gk": "GK",
    "goalkeeper": "GK",
    "ゴールキーパー": "GK",
    "キーパー": "GK",
    "df": "DF",
    "defender": "DF",
    "ディフェンダー": "DF",
    "mf": "MF",
    "midfielder": "MF",
    "ミッドフィールダー": "MF",
    "fw": "FW",
    "forward": "FW",
    "フォワード": "FW",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="チーム所属選手一覧から欠場突合用のチーム内構成比を算出します。"
    )
    p.add_argument("--input", default="", help="入力CSV")
    p.add_argument("--output", required=True, help="選手別出力CSV")
    p.add_argument(
        "--team-summary-output",
        default="",
        help="チーム集計出力CSV（未指定時は <output>_team_summary.csv）",
    )
    p.add_argument("--team-matches", type=int, default=0, help="全チーム共通の試合数")
    p.add_argument("--encoding", default="utf-8-sig", help="CSVエンコーディング")
    p.add_argument(
        "--pos-map",
        default="",
        help="追加ポジションマップ。JSONファイルパス or '日本語:GK,英語:DF' 形式。",
    )
    p.add_argument(
        "--include-def-gk-attack-share",
        action="store_true",
        help="指定時、GK/DFも attack_share_raw を goals/team_goals で算出する",
    )
    p.add_argument("--demo", action="store_true", help="デモデータで実行")
    return p.parse_args()


def to_number_series(series: pd.Series, fill_zero: bool = True) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("—", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
    )
    out = pd.to_numeric(s, errors="coerce")
    if fill_zero:
        out = out.fillna(0)
    return out


def clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def normalize_key(text: object) -> str:
    return re.sub(r"\s+", "", str(text).strip().lower())


def parse_pos_map(arg: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not arg:
        return out
    if os.path.exists(arg):
        with open(arg, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("--pos-map JSONは dict 形式で指定してください")
        for k, v in data.items():
            vv = str(v).upper().strip()
            if vv in {"GK", "DF", "MF", "FW"}:
                out[normalize_key(k)] = vv
        return out

    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        vv = str(v).upper().strip()
        if vv in {"GK", "DF", "MF", "FW"}:
            out[normalize_key(k)] = vv
    return out


def normalize_position(value: object, pos_map: Dict[str, str]) -> str:
    raw = str(value).strip()
    if not raw:
        return "UNK"
    key = normalize_key(raw)
    if key in pos_map:
        return pos_map[key]

    # fallback heuristics
    if "gk" in key or "ゴール" in raw or "キーパー" in raw:
        return "GK"
    if "df" in key or "def" in key or "ディフェ" in raw:
        return "DF"
    if "mf" in key or "mid" in key or "ミッド" in raw:
        return "MF"
    if "fw" in key or "forw" in key or "フォワ" in raw:
        return "FW"
    return "UNK"


def resolve_team_matches(df_team: pd.DataFrame, team_matches_cli: int) -> Tuple[int, str]:
    if team_matches_cli and team_matches_cli > 0:
        return int(team_matches_cli), "cli"

    max_app = int(df_team["appearances"].max()) if len(df_team) else 0
    if max_app > 0:
        return max_app, "max_appearances"

    raise ValueError("team_matches を推定できません（--team-matches 指定を推奨）")


def top_players_text(df_team: pd.DataFrame, col: str, n: int = 3) -> str:
    tmp = df_team.sort_values(col, ascending=False).head(n)
    vals = []
    for _, r in tmp.iterrows():
        vals.append(f"{r['player_name']}({float(r[col]):.4f})")
    return " / ".join(vals)


def build_demo_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_name": "TeamA", "player_name": "A_GK", "position": "GK", "appearances": 30, "goals": 0, "minutes": 2700},
            {"team_name": "TeamA", "player_name": "A_DF1", "position": "DF", "appearances": 31, "goals": 2, "minutes": 2790},
            {"team_name": "TeamA", "player_name": "A_MF1", "position": "MF", "appearances": 29, "goals": 6, "minutes": 2500},
            {"team_name": "TeamA", "player_name": "A_FW1", "position": "FW", "appearances": 27, "goals": 12, "minutes": 2200},
            {"team_name": "TeamB", "player_name": "B_GK", "position": "ゴールキーパー", "appearances": 32, "goals": 0, "minutes": 2880},
            {"team_name": "TeamB", "player_name": "B_DF1", "position": "ディフェンダー", "appearances": 25, "goals": 1, "minutes": 2100},
            {"team_name": "TeamB", "player_name": "B_MF1", "position": "MF", "appearances": 30, "goals": 4, "minutes": 2450},
            {"team_name": "TeamB", "player_name": "B_FW1", "position": "FW", "appearances": "—", "goals": 9, "minutes": 1800},
        ]
    )


def load_input(args: argparse.Namespace) -> pd.DataFrame:
    if args.demo:
        print("[INFO] demoモードでサンプルデータを使用します")
        return build_demo_df()
    if not args.input:
        raise ValueError("--input が必要です（--demo 以外）")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"input not found: {args.input}")
    return pd.read_csv(args.input, encoding=args.encoding)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"必須列が不足しています: {missing}")
    if "minutes" not in out.columns:
        out["minutes"] = 0
    if "note" not in out.columns:
        out["note"] = ""
    return out


def compute(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_input(args)
    df = ensure_columns(df)

    pos_map = dict(DEFAULT_POS_MAP)
    pos_map.update(parse_pos_map(args.pos_map))

    df["team_name"] = df["team_name"].astype(str).str.strip()
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["position_norm"] = df["position"].map(lambda x: normalize_position(x, pos_map))
    df["appearances"] = to_number_series(df["appearances"], fill_zero=False)
    df["goals"] = to_number_series(df["goals"], fill_zero=True)
    df["minutes"] = to_number_series(df["minutes"], fill_zero=False)

    player_rows = []
    team_rows = []

    for team_name, g in df.groupby("team_name", sort=True):
        team = g.copy().reset_index(drop=True)
        team["appearances_filled"] = team["appearances"].fillna(0)
        team["minutes_filled"] = team["minutes"].fillna(0)
        team_goals = float(team["goals"].fillna(0).sum())
        team_minutes_max = float(team["minutes_filled"].max()) if len(team) else 0.0
        team_matches, source = resolve_team_matches(team, args.team_matches)

        missing_apps = int(team["appearances"].isna().sum())
        missing_minutes = int(team["minutes"].isna().sum())
        print(
            f"[TEAM] {team_name} team_matches={team_matches} source={source} "
            f"team_goals={team_goals:.0f} missing_apps={missing_apps} missing_minutes={missing_minutes}"
        )

        regular_rate_list = []
        for _, r in team.iterrows():
            apps = r["appearances"]
            mins = r["minutes"]
            if pd.notna(apps):
                rr = float(apps) / float(team_matches)
            elif pd.notna(mins) and team_minutes_max > 0:
                rr = float(mins) / float(team_minutes_max)
            else:
                rr = 0.0
            regular_rate_list.append(rr)
        team["regular_rate"] = clip01(pd.Series(regular_rate_list, index=team.index))

        if team_goals > 0:
            team["attack_share_raw"] = team["goals"].fillna(0) / team_goals
        else:
            team["attack_share_raw"] = 0.0
        if not args.include_def_gk_attack_share:
            team.loc[team["position_norm"].isin(["GK", "DF"]), "attack_share_raw"] = 0.0
        team["attack_share"] = team["regular_rate"] * team["attack_share_raw"]

        team["defense_share_raw"] = team["position_norm"].map(DEFENSE_WEIGHTS).fillna(0.0)
        team["defense_share"] = team["regular_rate"] * team["defense_share_raw"]

        team["team_matches"] = int(team_matches)
        player_rows.append(
            team[
                [
                    "team_name",
                    "player_name",
                    "position_norm",
                    "appearances",
                    "goals",
                    "minutes",
                    "team_matches",
                    "regular_rate",
                    "attack_share_raw",
                    "attack_share",
                    "defense_share_raw",
                    "defense_share",
                ]
            ]
        )

        team_rows.append(
            {
                "team_name": team_name,
                "sum_regular_rate": float(team["regular_rate"].sum()),
                "sum_attack_share": float(team["attack_share"].sum()),
                "sum_defense_share": float(team["defense_share"].sum()),
                "top_attack_players": top_players_text(team, "attack_share", n=3),
                "top_defense_players": top_players_text(team, "defense_share", n=3),
            }
        )

    out_players = pd.concat(player_rows, ignore_index=True) if player_rows else pd.DataFrame()
    out_teams = pd.DataFrame(team_rows).sort_values("team_name").reset_index(drop=True)

    for c in ["appearances", "goals", "minutes", "team_matches"]:
        if c in out_players.columns:
            out_players[c] = pd.to_numeric(out_players[c], errors="coerce").fillna(0).astype(int)

    return out_players, out_teams


def derive_team_summary_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    ext = ext or ".csv"
    return f"{base}_team_summary{ext}"


def main() -> int:
    args = parse_args()
    try:
        out_players, out_teams = compute(args)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    team_summary_output = args.team_summary_output or derive_team_summary_path(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(team_summary_output)) or ".", exist_ok=True)

    out_players.to_csv(args.output, index=False, encoding=args.encoding)
    out_teams.to_csv(team_summary_output, index=False, encoding=args.encoding)

    print(f"[OK] players: {args.output} rows={len(out_players)}")
    print(f"[OK] teams: {team_summary_output} rows={len(out_teams)}")
    if args.demo:
        print("[INFO] demo完了。実データでは --input を指定してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
