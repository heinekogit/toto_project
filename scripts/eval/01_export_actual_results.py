#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import unicodedata

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def eval_base_dir(round_id: str) -> str:
    if str(round_id).startswith("toto"):
        return os.path.join(ROOT_DIR, "data", "eval", "toto_rounds", round_id)
    return os.path.join(ROOT_DIR, "data", "eval", "rounds", round_id)


def normalize_text(v):
    if pd.isna(v):
        return ""
    return unicodedata.normalize("NFKC", str(v)).strip()


def normalize_team_text(v):
    s = normalize_text(v)
    s = s.replace("　", " ")
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
    p.add_argument("--round", required=True, help="round02 / toto1608")
    p.add_argument("--season", required=True, type=int)
    p.add_argument("--snapshot-dir", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/snapshot")
    p.add_argument("--out", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/actual_results.csv")
    p.add_argument("--python", default=os.path.join(ROOT_DIR, "scripts", ".venv", "bin", "python"))
    return p.parse_args()


def load_toto_round_context(season, round_id):
    csv_path = os.path.join(ROOT_DIR, "data", "manual", "toto節リスト.csv")
    out = {"toto_round": None, "j1_round": None}
    if not os.path.exists(csv_path):
        return out
    m = re.match(r"^round0*([0-9]+)$", str(round_id).strip(), flags=re.IGNORECASE)
    if not m:
        if str(round_id).startswith("toto"):
            digits = re.sub(r"\D", "", str(round_id))
            if digits:
                out["toto_round"] = digits
        return out
    j1_round = int(m.group(1))
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out
    need = {"season", "toto_round", "J1_round"}
    if not need.issubset(df.columns):
        return out
    work = df.copy()
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["J1_round"] = pd.to_numeric(work["J1_round"], errors="coerce")
    work["toto_round"] = pd.to_numeric(work["toto_round"], errors="coerce")
    sub = work[(work["season"] == int(season)) & (work["J1_round"] == int(j1_round))].dropna(subset=["toto_round"])
    vals = sorted({int(v) for v in sub["toto_round"].tolist()})
    out["j1_round"] = str(j1_round)
    if len(vals) == 1:
        out["toto_round"] = str(vals[0])
    return out


def ensure_match_no(pred_df, snapshot_dir):
    if "match_no" in pred_df.columns:
        return pred_df

    buyplan_path = os.path.join(snapshot_dir, "buyplan.csv")
    work = pred_df.copy()
    if os.path.exists(buyplan_path):
        try:
            bp = pd.read_csv(buyplan_path)
            if {"match_no", "home_team", "away_team"}.issubset(bp.columns):
                lhs = work.copy()
                rhs = bp.copy()
                lhs["_home_n"] = lhs["home_team"].map(normalize_team_text)
                lhs["_away_n"] = lhs["away_team"].map(normalize_team_text)
                rhs["_home_n"] = rhs["home_team"].map(normalize_team_text)
                rhs["_away_n"] = rhs["away_team"].map(normalize_team_text)

                merged = lhs.merge(
                    rhs[["_home_n", "_away_n", "match_no"]],
                    on=["_home_n", "_away_n"],
                    how="left",
                )
                if merged["match_no"].notna().all():
                    merged["match_no"] = pd.to_numeric(merged["match_no"], errors="coerce").astype("Int64")
                    if merged["match_no"].isna().any():
                        raise RuntimeError("buyplan由来match_noが数値化できません")
                    if len(set(merged["match_no"].astype(int).tolist())) != len(merged):
                        raise RuntimeError("buyplan照合でmatch_noが重複しました")
                    print(f"[INFO] match_no補完: buyplan.csv を使用 ({buyplan_path})")
                    return merged.drop(columns=["_home_n", "_away_n"])
        except Exception as e:
            print(f"[WARN] buyplanからのmatch_no補完に失敗: {e}")

    if len(work) > 0:
        work = work.copy()
        work["match_no"] = list(range(1, len(work) + 1))
        print("[WARN] match_no列がないため行順で採番しました（1..N）")
        return work

    raise RuntimeError("predictions.csv が空で match_no を補完できません")


def build_target_matches(snapshot_dir, pred_df):
    buyplan_path = os.path.join(snapshot_dir, "buyplan.csv")
    if not os.path.exists(buyplan_path):
        return ensure_match_no(pred_df, snapshot_dir)

    try:
        buy = pd.read_csv(buyplan_path)
    except Exception as e:
        print(f"[WARN] buyplan.csv 読み込み失敗。predictions.csv を使用します: {e}")
        return ensure_match_no(pred_df, snapshot_dir)

    need_cols = {"match_no", "home_team", "away_team"}
    if not need_cols.issubset(buy.columns):
        print("[WARN] buyplan.csv に必須列がないため predictions.csv を使用します")
        return ensure_match_no(pred_df, snapshot_dir)

    out = buy[["match_no", "home_team", "away_team"]].copy()
    if "league" in buy.columns:
        out["league"] = buy["league"]
    elif "league" in pred_df.columns:
        out["league"] = pred_df["league"]
    else:
        out["league"] = "UNKNOWN"

    out["match_no"] = pd.to_numeric(out["match_no"], errors="coerce").astype("Int64")
    if out["match_no"].isna().any():
        raise RuntimeError("buyplan.csv の match_no に不正値があります")
    if len(set(out["match_no"].astype(int).tolist())) != len(out):
        raise RuntimeError("buyplan.csv の match_no が重複しています")
    print(f"[INFO] 対象試合は buyplan.csv を使用: {buyplan_path} ({len(out)}試合)")
    return out


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
    out["home_team_n"] = out["home_team"].map(normalize_team_text)
    out["away_team_n"] = out["away_team"].map(normalize_team_text)
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    out["datetime_n"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out


def resolve_one_match(pred_row, df_results):
    home_n = normalize_team_text(pred_row["home_team"])
    away_n = normalize_team_text(pred_row["away_team"])
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
                    teams |= set(df["home_team"].dropna().map(normalize_team_text))
                if "away_team" in df.columns:
                    teams |= set(df["away_team"].dropna().map(normalize_team_text))
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
    home_n = normalize_team_text(pred_row.get("home_team"))
    away_n = normalize_team_text(pred_row.get("away_team"))
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
    base_dir = eval_base_dir(args.round)
    snapshot_dir = args.snapshot_dir or os.path.join(base_dir, "snapshot")
    out_csv = args.out or os.path.join(base_dir, "actual_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    pred_path = os.path.join(snapshot_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"predictions.csv not found: {pred_path}")
    pred = pd.read_csv(pred_path)
    pred = build_target_matches(snapshot_dir, pred)
    toto_context = load_toto_round_context(args.season, args.round)
    if toto_context.get("toto_round"):
        print(
            f"[INFO] round整合: {args.round} -> toto{toto_context['toto_round']} "
            f"(J1_round={toto_context.get('j1_round') or '?'})"
        )
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
        try:
            run_update_results(args.python, args.season, lg)
        except Exception as e:
            print(f"[WARN] 結果更新をスキップして既存CSVを使用します: league={lg} reason={e}")

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
        resolved_lg = None
        matched = None
        key_used = ""
        unresolved_reason = ""
        try:
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
        except RuntimeError as e:
            unresolved_reason = str(e)

        row = {
            "match_no": int(r["match_no"]),
            "league": str(resolved_lg or r.get("league") or "").upper(),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_score": pd.NA,
            "away_score": pd.NA,
            "result": pd.NA,
            "match_id": pd.NA,
            "datetime": pd.NA,
            "status": "UNRESOLVED",
            "resolved_league": str(resolved_lg or "").upper(),
            "resolve_note": unresolved_reason,
            "toto_round": toto_context.get("toto_round") or pd.NA,
        }
        if matched is not None:
            result = to_result_102(matched["home_score"], matched["away_score"])
            row["match_id"] = matched.get("match_id", pd.NA)
            row["datetime"] = matched.get("datetime", pd.NA)
            row["resolved_league"] = str(resolved_lg or "").upper()
            row["home_score"] = float(matched["home_score"]) if not pd.isna(matched["home_score"]) else pd.NA
            row["away_score"] = float(matched["away_score"]) if not pd.isna(matched["away_score"]) else pd.NA
            if pd.isna(result):
                row["status"] = "PENDING"
                row["resolve_note"] = "score_missing"
                print(
                    f"[WARN] 結果未確定: match_no={r['match_no']} {r['home_team']} vs {r['away_team']}"
                )
            else:
                row["status"] = "OK"
                row["result"] = result
                row["resolve_note"] = key_used
                key_logs.append(
                    f"match_no={r['match_no']} league={resolved_lg.upper()} key={key_used} "
                    f"{r['home_team']} vs {r['away_team']}"
                )
        else:
            print(
                f"[WARN] 結果未解決: match_no={r['match_no']} {r['home_team']} vs {r['away_team']} "
                f"reason={unresolved_reason}"
            )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("match_no").reset_index(drop=True)
    if len(out) != 13:
        raise RuntimeError(f"対象試合数が13ではありません: {len(out)}")

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] actual results exported: {out_csv}")
    resolved_count = int((out["status"] == "OK").sum()) if "status" in out.columns else 0
    pending_count = int((out["status"] == "PENDING").sum()) if "status" in out.columns else 0
    unresolved_count = int((out["status"] == "UNRESOLVED").sum()) if "status" in out.columns else 0
    print(
        f"[INFO] result coverage: resolved={resolved_count} pending={pending_count} unresolved={unresolved_count}"
    )
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
