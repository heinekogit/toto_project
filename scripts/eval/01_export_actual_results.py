#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import unicodedata

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def normalize_text(v):
    if pd.isna(v):
        return ""
    return unicodedata.normalize("NFKC", str(v)).strip()


def normalize_league(v):
    s = normalize_text(v).upper().replace(" ", "")
    if "J1" in s:
        return "j1"
    if "J2" in s:
        return "j2"
    return s.lower()


def to_result_102(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return pd.NA
    if float(home_score) > float(away_score):
        return "1"
    if float(home_score) < float(away_score):
        return "2"
    return "0"


def parse_args():
    p = argparse.ArgumentParser(description="対象ラウンドの実結果CSVを厳格抽出")
    p.add_argument("--round", required=True, help="round02")
    p.add_argument("--season", required=True, type=int)
    p.add_argument("--snapshot-dir", default=None, help="既定: data/eval/rounds/{round}/snapshot")
    p.add_argument("--out", default=None, help="既定: data/eval/rounds/{round}/actual_results.csv")
    p.add_argument("--python", default=os.path.join(ROOT_DIR, "scripts", ".venv", "bin", "python"))
    return p.parse_args()


def run_update_results(python_cmd, season, league):
    env = os.environ.copy()
    env["SEASON_YEAR"] = str(season)
    env["LEAGUE"] = league
    cmd = [python_cmd, os.path.join(ROOT_DIR, "scripts", "01_update_match_results.py")]
    print(f"[RUN] {' '.join(cmd)} (LEAGUE={league}, SEASON_YEAR={season})")
    cp = subprocess.run(cmd, cwd=ROOT_DIR, env=env)
    if cp.returncode != 0:
        raise RuntimeError(f"01_update_match_results.py failed: league={league} rc={cp.returncode}")


def load_results_csv(season, league):
    cands = [
        os.path.join(ROOT_DIR, "data", f"{league}_{season}_latest_results.csv"),
        os.path.join(ROOT_DIR, "data", f"{league}_{season}_upcoming.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if not df.empty:
                return p, df
    raise FileNotFoundError(f"結果CSVが見つかりません: {cands}")


def prepare_df(df):
    out = df.copy()
    for col in ["home_team", "away_team", "match_id", "datetime", "home_score", "away_score"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["home_team_n"] = out["home_team"].map(normalize_text)
    out["away_team_n"] = out["away_team"].map(normalize_text)
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    out["datetime_n"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out


def resolve_one_match(pred_row, df_results):
    home_n = normalize_text(pred_row["home_team"])
    away_n = normalize_text(pred_row["away_team"])
    dt = pd.to_datetime(pred_row.get("datetime"), errors="coerce")

    candidates = df_results[(df_results["home_team_n"] == home_n) & (df_results["away_team_n"] == away_n)].copy()
    key_used = "home_team+away_team"
    if not pd.isna(dt):
        candidates_dt = candidates[candidates["datetime_n"] == dt].copy()
        if len(candidates_dt) == 1:
            candidates = candidates_dt
            key_used = "home_team+away_team+datetime"
        elif len(candidates_dt) > 1:
            raise RuntimeError(
                f"結果突合が一意になりません（datetime一致で複数）: "
                f"match_no={pred_row.get('match_no')} {pred_row['home_team']} vs {pred_row['away_team']}"
            )

    if len(candidates) == 0:
        raise RuntimeError(
            f"結果突合失敗（0件）: match_no={pred_row.get('match_no')} "
            f"{pred_row['home_team']} vs {pred_row['away_team']}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"結果突合失敗（複数件）: match_no={pred_row.get('match_no')} "
            f"{pred_row['home_team']} vs {pred_row['away_team']}"
        )
    return candidates.iloc[0], key_used


def resolve_one_match_unknown_league(pred_row, results_by_league):
    matches = []
    for lg, df_results in results_by_league.items():
        try:
            matched, key_used = resolve_one_match(pred_row, df_results)
            matches.append((lg, matched, key_used))
        except Exception:
            continue
    if len(matches) == 0:
        raise RuntimeError(
            f"結果突合失敗（league不明/0件）: match_no={pred_row.get('match_no')} "
            f"{pred_row['home_team']} vs {pred_row['away_team']}"
        )
    if len(matches) > 1:
        leagues = ",".join(sorted([x[0].upper() for x in matches]))
        raise RuntimeError(
            f"結果突合失敗（league不明/複数リーグ一致={leagues}）: match_no={pred_row.get('match_no')} "
            f"{pred_row['home_team']} vs {pred_row['away_team']}"
        )
    return matches[0]


def load_upcoming_team_sets(season):
    out = {}
    for lg in ["j1", "j2"]:
        p = os.path.join(ROOT_DIR, "data", f"{lg}_{season}_upcoming.csv")
        teams = set()
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if "home_team" in df.columns:
                    teams |= set(df["home_team"].dropna().map(normalize_text))
                if "away_team" in df.columns:
                    teams |= set(df["away_team"].dropna().map(normalize_text))
            except Exception:
                teams = set()
        out[lg] = teams
    return out


def _team_match_score(team: str, team_set: set) -> int:
    if not team:
        return 0
    if team in team_set:
        return 2
    # 表記ゆれ対策: 部分一致を弱い根拠として扱う
    for t in team_set:
        if not t:
            continue
        if team in t or t in team:
            return 1
    return 0


def infer_league_from_team_sets(pred_row, candidate_leagues, upcoming_team_sets):
    home_n = normalize_text(pred_row.get("home_team"))
    away_n = normalize_text(pred_row.get("away_team"))
    scores = {}
    for lg in candidate_leagues:
        team_set = upcoming_team_sets.get(lg, set())
        score = _team_match_score(home_n, team_set) + _team_match_score(away_n, team_set)
        scores[lg] = score
    if not scores:
        return None
    best = max(scores.values())
    if best <= 0:
        return None
    tied = [lg for lg, sc in scores.items() if sc == best]
    if len(tied) != 1:
        return None
    return tied[0]


def main():
    args = parse_args()
    snapshot_dir = args.snapshot_dir or os.path.join(ROOT_DIR, "data", "eval", "rounds", args.round, "snapshot")
    out_csv = args.out or os.path.join(ROOT_DIR, "data", "eval", "rounds", args.round, "actual_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    pred_path = os.path.join(snapshot_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"predictions.csv not found: {pred_path}")
    pred = pd.read_csv(pred_path)
    for col in ["match_no", "league", "home_team", "away_team"]:
        if col not in pred.columns:
            raise ValueError(f"predictions.csv 必須列不足: {col}")
    pred["league_n"] = pred["league"].map(normalize_league)
    pred = pred.copy()
    if pred.empty:
        raise RuntimeError("predictions.csv が空です")

    # 対象リーグのみ結果更新
    known_leagues = sorted([x for x in pred["league_n"].dropna().unique().tolist() if x in ["j1", "j2"]])
    unknown_league_count = int((~pred["league_n"].isin(["j1", "j2"])).sum())
    if unknown_league_count > 0:
        print(f"[WARN] league不明の予測行が {unknown_league_count} 件あります。J1/J2結果から自動推定します。")
        target_leagues = sorted(set(known_leagues + ["j1", "j2"]))
    else:
        target_leagues = known_leagues
    if not target_leagues:
        raise RuntimeError("predictions.csv に J1/J2 判定可能な試合がありません")

    for lg in target_leagues:
        run_update_results(args.python, args.season, lg)

    results_by_league = {}
    source_by_league = {}
    for lg in target_leagues:
        src, df = load_results_csv(args.season, lg)
        source_by_league[lg] = src
        results_by_league[lg] = prepare_df(df)

    rows = []
    key_logs = []
    upcoming_team_sets = load_upcoming_team_sets(args.season)
    for _, r in pred.sort_values("match_no").iterrows():
        lg = r["league_n"]
        if lg in results_by_league:
            matched, key_used = resolve_one_match(r, results_by_league[lg])
            resolved_lg = lg
        else:
            try:
                resolved_lg, matched, key_used = resolve_one_match_unknown_league(r, results_by_league)
            except RuntimeError as e:
                msg = str(e)
                if "複数リーグ一致" not in msg:
                    raise
                inferred_lg = infer_league_from_team_sets(r, list(results_by_league.keys()), upcoming_team_sets)
                if not inferred_lg:
                    raise
                print(
                    f"[WARN] league不明をチーム所属から推定: match_no={r.get('match_no')} "
                    f"{r.get('home_team')} vs {r.get('away_team')} -> {inferred_lg.upper()}"
                )
                matched, key_used = resolve_one_match(r, results_by_league[inferred_lg])
                resolved_lg = inferred_lg
        result = to_result_102(matched["home_score"], matched["away_score"])
        if pd.isna(result):
            raise RuntimeError(
                f"結果未確定（スコア欠損）: match_no={r['match_no']} {r['home_team']} vs {r['away_team']}"
            )
        key_logs.append(
            f"match_no={r['match_no']} league={resolved_lg.upper()} key={key_used} "
            f"{r['home_team']} vs {r['away_team']}"
        )
        rows.append(
            {
                "match_no": int(r["match_no"]),
                "league": resolved_lg.upper(),
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_score": float(matched["home_score"]),
                "away_score": float(matched["away_score"]),
                "result": result,
                "match_id": matched.get("match_id", pd.NA),
                "datetime": matched.get("datetime", pd.NA),
            }
        )

    out = pd.DataFrame(rows).sort_values("match_no").reset_index(drop=True)
    if len(out) != 13:
        raise RuntimeError(f"対象試合数が13ではありません: {len(out)}")

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] actual results exported: {out_csv}")
    print("[INFO] result source files:")
    for lg, p in source_by_league.items():
        print(f"  {lg.upper()}: {p}")
    print("[INFO] match key logs:")
    for line in key_logs:
        print(" ", line)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
