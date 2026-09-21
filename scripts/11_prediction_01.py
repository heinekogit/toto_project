#   ■スクリプトの機能概要
#   j1_2024_results.csv：2024年シーズンの全試合（終了分）※任意
#   j1_20xx_upcoming.csv：対象シーズン（終了試合＋未開催試合）
#       → 終了試合は学習素材として、未開催試合に予測を行う。
#   j1_2025_predictions.csv（出力：）
#       → 未開催試合に予測勝敗と確率を付与し、結果をCSV出力。
#
#   ■予測ロジック（Elo基盤 + HDA確率 + lab対戦補正）
#   各チームの基本強さは、前季最終Eloを初期値に当季終了試合で更新して構築。
#   HDA確率はElo差を主軸に multinom / legacy Poisson で推定。
#   Football Lab 比較値とチーム状態ベクトルから matchup 補正を計算し、
#   `lab_mix_*` を確率へ弱くブレンドしたうえで最終ラベルを決定。
#   その後、draw / incentive / away-restore 系 override を後段で適用。
#   未開催の試合のみ予測対象。
#   =================================================



import os
import sys
import math
from acl_schedule import normalize_acl_schedule
from bisect import bisect_left
import pandas as pd
import numpy as np
from fatigue_merge import add_fatigue_time_aliases, validate_prediction_fatigue
_PROJECT_ROOT_FOR_IMPORTS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT_FOR_IMPORTS not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_FOR_IMPORTS)
from weather_rules import STRONG_WIND_THRESHOLD_KMH, WIND_SPEED_UNIT, adverse_weather_penalty
from scipy.stats import poisson
import json
import re
import hashlib
import subprocess
import pickle
from datetime import datetime
import unicodedata

# パスはスクリプト起点で固定
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MANUAL_DIR = os.path.join(DATA_DIR, "manual")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
OUTPUT_SNAPSHOT_DIR = os.environ.get("OUTPUT_SNAPSHOT_DIR", os.path.join(DATA_DIR, "output_snapshots"))
ENABLE_OUTPUT_SNAPSHOT = str(os.environ.get("ENABLE_OUTPUT_SNAPSHOT", "1")).strip() == "1"
OUTPUT_OVERWRITE_GUARD = str(os.environ.get("OUTPUT_OVERWRITE_GUARD", "1")).strip() == "1"
try:
    OUTPUT_GUARD_MIN_ROW_RATIO = float(os.environ.get("OUTPUT_GUARD_MIN_ROW_RATIO", "0.85"))
except Exception:
    OUTPUT_GUARD_MIN_ROW_RATIO = 0.85
try:
    OUTPUT_GUARD_MAX_COMPLETENESS_DROP = float(os.environ.get("OUTPUT_GUARD_MAX_COMPLETENESS_DROP", "0.03"))
except Exception:
    OUTPUT_GUARD_MAX_COMPLETENESS_DROP = 0.03
STATS_SNAPSHOT_DIR = os.path.join(DATA_DIR, "stats_snapshots")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
SEASON_YEAR = int(os.environ.get("SEASON_YEAR", "2025"))
TOTO_ROUND_ID = os.environ.get("TOTO_ROUND_ID", "").strip()
ROUND_NO_ENV = os.environ.get("ROUND_NO", "").strip()
STATS_ASOF_DATE = os.environ.get("STATS_ASOF_DATE", "").strip()
STATS_SNAPSHOT_NAME = os.environ.get("STATS_SNAPSHOT_NAME", "").strip()
WEATHER_ASOF_DATE = os.environ.get("WEATHER_ASOF_DATE", STATS_ASOF_DATE).strip()
WEATHER_SNAPSHOT_NAME = os.environ.get("WEATHER_SNAPSHOT_NAME", "").strip()
WEATHER_SNAPSHOT_DIR = os.environ.get("WEATHER_SNAPSHOT_DIR", os.path.join(DATA_DIR, "weather_snapshots"))
ABSENCE_ASOF_DATE = os.environ.get("ABSENCE_ASOF_DATE", STATS_ASOF_DATE).strip()
ABSENCE_SNAPSHOT_NAME = os.environ.get("ABSENCE_SNAPSHOT_NAME", "").strip()
ABSENCE_SNAPSHOT_DIR = os.environ.get("ABSENCE_SNAPSHOT_DIR", os.path.join(DATA_DIR, "absence_snapshots"))
EXTERNAL_METRICS_DIR = os.path.join(DATA_DIR, "external_metrics")
TOTO_ORDER_CSV = os.environ.get("TOTO_ORDER_CSV", os.path.join(MANUAL_DIR, "toto並び順.csv"))
RAW_CLI_ARGS = sys.argv[1:]
CLI_ARGS = set(RAW_CLI_ARGS)
_FOOTBALL_LAB_COMPARE_CACHE = {}


def _get_env_int(name, default):
    raw = os.environ.get(name, str(default))
    try:
        return int(str(raw).strip())
    except Exception:
        print(f"[CONFIG][WARN] invalid int env {name}={raw!r}; fallback={default}")
        return int(default)


def _env_flag(name, default=0):
    return _get_env_int(name, default) == 1


def _get_cli_int_arg(flag, default):
    try:
        idx = RAW_CLI_ARGS.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(RAW_CLI_ARGS):
        return default
    try:
        return int(str(RAW_CLI_ARGS[idx + 1]).strip())
    except Exception:
        return default


def _get_cli_str_arg(flag, default=""):
    try:
        idx = RAW_CLI_ARGS.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(RAW_CLI_ARGS):
        return default
    return str(RAW_CLI_ARGS[idx + 1]).strip()


FORCE_RECALC = ("--force" in CLI_ARGS) or _env_flag("FORCE_RECALC", 0)
SELF_CHECK_HFA = ("--self-check-hfa" in CLI_ARGS) or _env_flag("SELF_CHECK_HFA", 0)
SKIP_HFA_SELF_CHECK = ("--skip-hfa-self-check" in CLI_ARGS) or _env_flag("SKIP_HFA_SELF_CHECK", 0)
DUMP_DECISION = ("--dump-decision" in CLI_ARGS) or _env_flag("DUMP_DECISION", 0)
STRICT_MODE = _env_flag("STRICT_MODE", 1)
HFA_TRACE_N = max(1, _get_cli_int_arg("--hfa-trace-n", _get_env_int("HFA_TRACE_N", 5)))
SENSITIVITY_SCAN = ("--sensitivity-scan" in CLI_ARGS) or _env_flag("SENSITIVITY_SCAN", 0)
PROFILE_SCAN = ("--profile-scan" in CLI_ARGS) or _env_flag("PROFILE_SCAN", 0)
SENSITIVITY_HFA_VALUES_RAW = _get_cli_str_arg(
    "--sensitivity-hfa-values",
    os.environ.get("SENSITIVITY_HFA_VALUES", "0,10,20,35"),
)
SENSITIVITY_ELO_SCALE_VALUES_RAW = _get_cli_str_arg(
    "--sensitivity-elo-scale-values",
    os.environ.get("SENSITIVITY_ELO_SCALE_VALUES", "0.5,1.0,1.5"),
)
SENSITIVITY_DRAW_ASSIGN_VALUES_RAW = _get_cli_str_arg(
    "--sensitivity-draw-assign-values",
    os.environ.get("SENSITIVITY_DRAW_ASSIGN_VALUES", "0,1"),
)
BACKTEST_DECISION_RULE = _get_cli_str_arg(
    "--backtest-decision-rule",
    os.environ.get("BACKTEST_DECISION_RULE", "both"),
).strip().lower()
if BACKTEST_DECISION_RULE not in {"argmax", "expect", "hybrid", "close_ha_draw", "close_ha_draw_v2", "both", "all"}:
    print(f"[CONFIG][WARN] invalid BACKTEST_DECISION_RULE={BACKTEST_DECISION_RULE!r}; fallback='both'")
    BACKTEST_DECISION_RULE = "both"
BACKTEST_COMPARE_DATASET = os.environ.get("BACKTEST_COMPARE_DATASET", "current").strip().lower()
BACKTEST_COMPARE_CSV = os.environ.get("BACKTEST_COMPARE_CSV", "").strip()
BACKTEST_MARGIN_SCAN = ("--draw-margin-scan" in CLI_ARGS) or _env_flag("BACKTEST_MARGIN_SCAN", 0)
DRAW_MARGIN_GRID_RAW = _get_cli_str_arg(
    "--draw-margin-grid",
    os.environ.get("DRAW_MARGIN_GRID", "+0.05,+0.04,+0.03,+0.02,+0.01,0.00,-0.01,-0.02,-0.03"),
)
DRAW_SCORE_THRESHOLDS_RAW = os.environ.get("DRAW_SCORE_THRESHOLDS", "0.00,0.01,0.02,0.03,0.04,0.05")
DECISION_RULE_DESC = "argmax(prob_home_win, prob_draw, prob_away_win)"
PROFILE_SCAN_DIR = os.path.join(REPORT_DIR, "metrics")
MERGE_QC_DIR = os.path.join(REPORT_DIR, "merge_qc", f"{LEAGUE}_{SEASON_YEAR}")
PREV_SEASON_YEAR = SEASON_YEAR - 1
csv_prev = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_results.csv")
if not os.path.exists(csv_prev):
    csv_prev_latest = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_latest_results.csv")
    if os.path.exists(csv_prev_latest):
        csv_prev = csv_prev_latest
prev_final_elo_csv = os.path.join(DATA_DIR, f"{LEAGUE}_{PREV_SEASON_YEAR}_final_elo.csv")
csv_season = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_upcoming.csv")
csv_season_latest = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_latest_results.csv")
ENABLE_HFA_INT = _get_env_int("ENABLE_HFA", 1)
if ENABLE_HFA_INT not in (0, 1):
    print(f"[CONFIG][WARN] ENABLE_HFA should be 0/1 but got {ENABLE_HFA_INT}; coerced to {1 if ENABLE_HFA_INT else 0}")
    ENABLE_HFA_INT = 1 if ENABLE_HFA_INT else 0
ENABLE_HFA = ENABLE_HFA_INT == 1
hfa_suffix = "hfa_on" if ENABLE_HFA else "hfa_off"
LEGACY_OUTPUT_CSV = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions.csv")
output_csv_default = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions_{hfa_suffix}.csv")
output_csv = os.environ.get("OUTPUT_PRED_CSV", "").strip() or output_csv_default
backtest_output_default = os.path.join(BASE_DIR, f"backtest_{LEAGUE}_{SEASON_YEAR}.csv")
backtest_output_csv = os.environ.get("BACKTEST_OUTPUT_CSV", "").strip() or backtest_output_default
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def _fatal_data_error(stage, message, hints=None, actions=None):
    print(f"[ERROR] {stage}: {message}")
    for h in hints or []:
        print(f"[HINT] {h}")
    for a in actions or []:
        print(f"[ACTION] {a}")
    raise RuntimeError(f"{stage}: {message}")


def _save_output_snapshot(df: pd.DataFrame, source_path: str, kind: str):
    if not ENABLE_OUTPUT_SNAPSHOT:
        return ""
    try:
        league_key = str(LEAGUE).lower()
        season_key = str(int(SEASON_YEAR))
        target_dir = os.path.join(OUTPUT_SNAPSHOT_DIR, f"{league_key}_{season_key}")
        os.makedirs(target_dir, exist_ok=True)
        src_base = os.path.basename(source_path)
        snap_name = f"{RUN_TS}_{kind}_{src_base}"
        snap_path = os.path.join(target_dir, snap_name)
        df.to_csv(snap_path, index=False, encoding="utf-8-sig")
        return snap_path
    except Exception as e:
        print(f"[SNAPSHOT][WARN] failed to save {kind}: {e}")
        return ""


def _mean_notna_ratio(df: pd.DataFrame, cols):
    if df is None or df.empty or not cols:
        return 0.0
    ratios = []
    for c in cols:
        if c in df.columns:
            ratios.append(float(df[c].notna().mean()))
        else:
            ratios.append(0.0)
    return float(np.mean(ratios)) if ratios else 0.0


def _guarded_write_csv(df: pd.DataFrame, path: str, label: str, critical_cols=None):
    result = {
        "written": False,
        "reason": "",
        "rows_new": int(len(df)),
        "rows_old": 0,
        "quality_new": 0.0,
        "quality_old": 0.0,
        "path": path,
    }
    critical_cols = list(critical_cols or [])
    result["quality_new"] = _mean_notna_ratio(df, critical_cols)

    if (not OUTPUT_OVERWRITE_GUARD) or (not os.path.exists(path)):
        df.to_csv(path, index=False, encoding="utf-8-sig")
        result["written"] = True
        result["reason"] = "write_no_guard_or_no_old"
        return result

    try:
        old_df = pd.read_csv(path)
        result["rows_old"] = int(len(old_df))
        result["quality_old"] = _mean_notna_ratio(old_df, critical_cols)
    except Exception as e:
        print(f"[WRITE_GUARD][WARN] {label}: old csv read failed; overwrite allowed ({e})")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        result["written"] = True
        result["reason"] = "write_old_read_failed"
        return result

    rows_ok = True
    if result["rows_old"] > 0:
        rows_ok = result["rows_new"] >= int(np.floor(result["rows_old"] * OUTPUT_GUARD_MIN_ROW_RATIO))
    quality_drop = result["quality_old"] - result["quality_new"]
    quality_ok = quality_drop <= OUTPUT_GUARD_MAX_COMPLETENESS_DROP
    if rows_ok and quality_ok:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        result["written"] = True
        result["reason"] = "write_guard_pass"
        return result

    result["reason"] = "blocked_by_guard"
    print(
        f"[WRITE_GUARD][BLOCK] {label}: keep old file. "
        f"rows_old={result['rows_old']} rows_new={result['rows_new']} "
        f"quality_old={result['quality_old']:.4f} quality_new={result['quality_new']:.4f} "
        f"row_ratio_min={OUTPUT_GUARD_MIN_ROW_RATIO:.2f} max_quality_drop={OUTPUT_GUARD_MAX_COMPLETENESS_DROP:.4f}"
    )
    return result


def _read_csv_guarded(
    path,
    *,
    stage,
    required_cols=None,
    allow_empty=False,
    strict=True,
    hints=None,
    actions=None,
):
    required_cols = list(required_cols or [])
    if not os.path.exists(path):
        msg = f"file not found: {path}"
        if strict:
            _fatal_data_error(stage, msg, hints=hints, actions=actions)
        print(f"[WARN] {stage}: {msg}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        msg = f"failed to read csv: {path} ({e})"
        if strict:
            _fatal_data_error(stage, msg, hints=hints, actions=actions)
        print(f"[WARN] {stage}: {msg}")
        return pd.DataFrame()
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            msg = f"missing required columns={missing} path={path}"
            extra_hints = list(hints or []) + [f"available_columns={list(df.columns)[:40]}"]
            if strict:
                _fatal_data_error(stage, msg, hints=extra_hints, actions=actions)
            print(f"[WARN] {stage}: {msg}")
            return pd.DataFrame()
    if (not allow_empty) and df.empty:
        msg = f"empty csv: {path}"
        if strict:
            _fatal_data_error(stage, msg, hints=hints, actions=actions)
        print(f"[WARN] {stage}: {msg}")
    return df


def _resolve_football_lab_compare_paths(league):
    league_key = str(league).strip().lower()
    if not league_key or (not os.path.isdir(EXTERNAL_METRICS_DIR)):
        return []
    suffix = f"_{league_key}.csv"
    paths = []
    for fn in os.listdir(EXTERNAL_METRICS_DIR):
        if ("football_lab_compare" not in fn) or (not fn.endswith(suffix)):
            continue
        path = os.path.join(EXTERNAL_METRICS_DIR, fn)
        if os.path.isfile(path):
            paths.append(path)
    return sorted(paths, key=lambda p: (os.path.getmtime(p), os.path.basename(p)))


def _load_football_lab_compare_bundle(league):
    league_key = str(league).strip().lower()
    cached = _FOOTBALL_LAB_COMPARE_CACHE.get(league_key)
    if cached is not None:
        return cached.copy()

    paths = _resolve_football_lab_compare_paths(league_key)
    frames = []
    for priority, path in enumerate(paths):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[FOOTBALL_LAB][WARN] read failed: {path} ({e})")
            continue
        if ("match_id" not in df.columns) or df.empty:
            continue
        keep_cols = ["match_id"] + [c for c in df.columns if c.startswith("flab_")]
        part = df[keep_cols].copy()
        flab_cols = [c for c in part.columns if c.startswith("flab_")]
        part["__flab_nonnull"] = part[flab_cols].notna().sum(axis=1) if flab_cols else 0
        part["__flab_mtime"] = float(os.path.getmtime(path))
        part["__flab_priority"] = int(priority)
        part["__flab_source"] = os.path.basename(path)
        frames.append(part)

    if not frames:
        bundle = pd.DataFrame(columns=["match_id"])
        _FOOTBALL_LAB_COMPARE_CACHE[league_key] = bundle
        return bundle.copy()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values(
        ["match_id", "__flab_nonnull", "__flab_mtime", "__flab_priority"]
    )
    merged = merged.drop_duplicates(subset=["match_id"], keep="last")
    _FOOTBALL_LAB_COMPARE_CACHE[league_key] = merged.copy()
    print(
        f"[FOOTBALL_LAB] league={league_key} files={len(paths)} "
        f"rows={len(merged)}"
    )
    return merged.copy()


def merge_football_lab_compare(df, league, scope):
    if df is None or df.empty or ("match_id" not in df.columns):
        return df
    compare = _load_football_lab_compare_bundle(league)
    if compare.empty:
        return df

    out = df.copy()
    flab_cols = [c for c in compare.columns if c.startswith("flab_")]
    drop_cols = [c for c in flab_cols if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    before = set(out.columns)
    out = out.merge(compare[["match_id"] + flab_cols], on="match_id", how="left", validate="one_to_one")
    merged_cols = [c for c in flab_cols if c not in before]
    available_rows = 0
    if merged_cols:
        available_rows = int(out[merged_cols].notna().any(axis=1).sum())
    print(
        f"[FOOTBALL_LAB] scope={scope} league={str(league).lower()} "
        f"merged_cols={len(merged_cols)} available_rows={available_rows}/{len(out)}"
    )
    return out

def pick_non_empty_csv_path(candidates, required_cols=None):
    required_cols = required_cols or []
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"[PATH] 空CSVのためスキップ: {path}")
                continue
            if required_cols and not set(required_cols).issubset(df.columns):
                print(f"[PATH] 必須列不足のためスキップ: {path}")
                continue
            print(f"[PATH] 採用: {path} (rows={len(df)})")
            return path
        except Exception as e:
            print(f"[PATH] 読み込み失敗のためスキップ: {path} ({e})")
            continue
    return candidates[-1] if candidates else None


def _norm_key_text(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v)).replace("　", " ").strip()
    s = s.replace(" ", "").replace("・", "")
    return s.upper()


def _build_match_merge_key(df):
    out = df.copy()
    dt = pd.to_datetime(out.get("datetime"), errors="coerce")
    out["_dt_key"] = dt.dt.strftime("%Y-%m-%d %H:%M")
    out["_home_key"] = out.get("home_team", pd.Series(index=out.index, dtype="object")).map(_norm_key_text)
    out["_away_key"] = out.get("away_team", pd.Series(index=out.index, dtype="object")).map(_norm_key_text)
    out["_match_merge_key"] = out["_dt_key"].fillna("") + "|" + out["_home_key"].fillna("") + "|" + out["_away_key"].fillna("")
    return out


def enrich_scores_from_latest_results(df_season, latest_results_csv):
    if df_season is None or df_season.empty:
        return df_season
    if not os.path.exists(latest_results_csv):
        print(f"[SCORE_ENRICH] skip: latest_results not found ({latest_results_csv})")
        return df_season
    try:
        latest = pd.read_csv(latest_results_csv)
    except Exception as e:
        print(f"[SCORE_ENRICH][WARN] failed to read latest_results: {e}")
        return df_season
    required = {"home_team", "away_team"}
    if not required.issubset(latest.columns):
        print("[SCORE_ENRICH][WARN] latest_results missing required columns")
        return df_season
    work = df_season.copy()
    before_scored = int(
        pd.to_numeric(work.get("home_score"), errors="coerce").notna()
        .mul(pd.to_numeric(work.get("away_score"), errors="coerce").notna())
        .sum()
    )
    for col in ["home_score", "away_score"]:
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = pd.to_numeric(work[col], errors="coerce")
        if col not in latest.columns:
            latest[col] = pd.NA
        latest[col] = pd.to_numeric(latest[col], errors="coerce")

    # 1) match_id優先
    filled_by_match_id = 0
    if "match_id" in work.columns and "match_id" in latest.columns:
        right = latest[["match_id", "home_score", "away_score"]].dropna(subset=["match_id"]).drop_duplicates(
            subset=["match_id"], keep="last"
        )
        m = work.merge(right, on="match_id", how="left", suffixes=("", "__latest"))
        for col in ["home_score", "away_score"]:
            pre_na = m[col].isna()
            m[col] = m[col].fillna(m[f"{col}__latest"])
            filled_by_match_id += int(pre_na.sum() - m[col].isna().sum())
        work = m.drop(columns=["home_score__latest", "away_score__latest"], errors="ignore")

    # 2) datetime+home+awayで補完
    left = _build_match_merge_key(work)
    right = _build_match_merge_key(latest)
    right = right[["_match_merge_key", "home_score", "away_score"]].drop_duplicates(subset=["_match_merge_key"], keep="last")
    m2 = left.merge(right, on="_match_merge_key", how="left", suffixes=("", "__latest2"))
    filled_by_key = 0
    for col in ["home_score", "away_score"]:
        pre_na = m2[col].isna()
        m2[col] = m2[col].fillna(m2[f"{col}__latest2"])
        filled_by_key += int(pre_na.sum() - m2[col].isna().sum())
    work = m2.drop(
        columns=[
            "home_score__latest2",
            "away_score__latest2",
            "_dt_key",
            "_home_key",
            "_away_key",
            "_match_merge_key",
        ],
        errors="ignore",
    )

    after_scored = int(
        pd.to_numeric(work.get("home_score"), errors="coerce").notna()
        .mul(pd.to_numeric(work.get("away_score"), errors="coerce").notna())
        .sum()
    )
    print(
        f"[SCORE_ENRICH] source={latest_results_csv} "
        f"filled_by_match_id={filled_by_match_id} filled_by_key={filled_by_key} "
        f"scored_rows={before_scored}->{after_scored}"
    )
    return work


def merge_missing_matches_from_latest_results(df_season, latest_results_csv):
    if df_season is None or df_season.empty:
        return df_season
    if not os.path.exists(latest_results_csv):
        print(f"[SEASON_MERGE] skip: latest_results not found ({latest_results_csv})")
        return df_season
    try:
        latest = pd.read_csv(latest_results_csv)
    except Exception as e:
        print(f"[SEASON_MERGE][WARN] failed to read latest_results: {e}")
        return df_season

    required = {"home_team", "away_team"}
    if not required.issubset(latest.columns):
        print("[SEASON_MERGE][WARN] latest_results missing required columns")
        return df_season

    season = df_season.copy()
    latest = latest.copy()
    season_cols = list(season.columns)

    if "match_id" in season.columns and "match_id" in latest.columns:
        season_ids = set(season["match_id"].dropna().astype(str))
        add = latest[~latest["match_id"].astype(str).isin(season_ids)].copy()
    else:
        left = _build_match_merge_key(season)
        right = _build_match_merge_key(latest)
        season_keys = set(left["_match_merge_key"].dropna().astype(str))
        add = right[~right["_match_merge_key"].astype(str).isin(season_keys)].copy()
        add = add.drop(columns=["_dt_key", "_home_key", "_away_key", "_match_merge_key"], errors="ignore")

    if add.empty:
        print(f"[SEASON_MERGE] source={latest_results_csv} added_rows=0 total_rows={len(season)}")
        return season

    for col in season_cols:
        if col not in add.columns:
            add[col] = pd.NA
    add = add[season_cols + [c for c in add.columns if c not in season_cols]]
    out = pd.concat([season, add], ignore_index=True, sort=False)
    print(
        f"[SEASON_MERGE] source={latest_results_csv} added_rows={len(add)} "
        f"total_rows={len(season)}->{len(out)}"
    )
    return out


def _asof_key(value):
    if not value:
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def resolve_team_master_stats_csv():
    # 1) 明示指定（ファイル名）
    if STATS_SNAPSHOT_NAME:
        candidate = STATS_SNAPSHOT_NAME
        if not os.path.isabs(candidate):
            candidate = os.path.join(STATS_SNAPSHOT_DIR, candidate)
        if os.path.exists(candidate):
            print(f"[PATH] stats snapshot(明示指定)を採用: {candidate}")
            return candidate, _asof_key(os.path.basename(candidate))
        print(f"[PATH][WARN] 指定snapshotが見つかりません: {candidate}")

    # 2) snapshots から解決（asof指定があれば <= asof の最新）
    if os.path.isdir(STATS_SNAPSHOT_DIR):
        prefix = f"team_master_stats_{LEAGUE}_{SEASON_YEAR}_asof_"
        candidates = []
        for fn in os.listdir(STATS_SNAPSHOT_DIR):
            if not (fn.startswith(prefix) and fn.endswith(".csv")):
                continue
            asof = fn[len(prefix) : -4]
            if asof and asof.isdigit():
                path = os.path.join(STATS_SNAPSHOT_DIR, fn)
                candidates.append((asof, path))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            target_asof = _asof_key(STATS_ASOF_DATE)
            if target_asof:
                le_targets = [x for x in candidates if x[0] <= target_asof]
                if le_targets:
                    chosen = le_targets[-1]
                    print(f"[PATH] stats snapshot(asof={target_asof})を採用: {chosen[1]}")
                    return chosen[1], chosen[0]
                print(f"[PATH][WARN] asof={target_asof} 以下のsnapshotがないため最新を使用します。")
            chosen = candidates[-1]
            print(f"[PATH] stats snapshot(最新)を採用: {chosen[1]}")
            return chosen[1], chosen[0]

    # 3) 従来フォールバック
    fallback = pick_non_empty_csv_path(
        [
            os.path.join(DATA_DIR, f"team_master_stats_{LEAGUE}_{SEASON_YEAR}.csv"),
            os.path.join(DATA_DIR, f"team_master_stats_{SEASON_YEAR}.csv"),
            os.path.join(DATA_DIR, "team_master_stats.csv"),
        ],
        required_cols=["team_name"],
    )
    return fallback, ""


def resolve_weather_cache_csv():
    # 1) 明示指定
    if WEATHER_SNAPSHOT_NAME:
        candidate = WEATHER_SNAPSHOT_NAME
        if not os.path.isabs(candidate):
            candidate = os.path.join(WEATHER_SNAPSHOT_DIR, candidate)
        if os.path.exists(candidate):
            print(f"[PATH] weather snapshot(明示指定)を採用: {candidate}")
            return candidate, _asof_key(os.path.basename(candidate))
        print(f"[PATH][WARN] 指定weather snapshotが見つかりません: {candidate}")

    # 2) snapshots から解決（asof指定があれば <= asof）
    if os.path.isdir(WEATHER_SNAPSHOT_DIR):
        prefix = f"weather_features_{LEAGUE}_{SEASON_YEAR}_asof_"
        candidates = []
        for fn in os.listdir(WEATHER_SNAPSHOT_DIR):
            if not (fn.startswith(prefix) and fn.endswith(".csv")):
                continue
            asof = fn[len(prefix) : -4]
            if asof and asof.isdigit():
                candidates.append((asof, os.path.join(WEATHER_SNAPSHOT_DIR, fn)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            target_asof = _asof_key(WEATHER_ASOF_DATE)
            if target_asof:
                le_targets = [x for x in candidates if x[0] <= target_asof]
                if le_targets:
                    chosen = le_targets[-1]
                    print(f"[PATH] weather snapshot(asof={target_asof})を採用: {chosen[1]}")
                    return chosen[1], chosen[0]
                print(f"[PATH][WARN] weather asof={target_asof} 以下のsnapshotがないため最新を使用します。")
            chosen = candidates[-1]
            print(f"[PATH] weather snapshot(最新)を採用: {chosen[1]}")
            return chosen[1], chosen[0]

    # 3) 従来フォールバック
    fallback = pick_non_empty_csv_path(
        [
            os.path.join(MANUAL_DIR, f"weather_features_{LEAGUE}_{SEASON_YEAR}.csv"),
            os.path.join(DATA_DIR, f"weather_features_{LEAGUE}_{SEASON_YEAR}.csv"),
            os.path.join(MANUAL_DIR, "weather_cache.csv"),
            os.path.join(DATA_DIR, "weather_cache.csv"),
        ],
        required_cols=["match_id"],
    )
    return fallback, ""


def resolve_absence_impact_csv():
    # 1) 明示指定
    if ABSENCE_SNAPSHOT_NAME:
        candidate = ABSENCE_SNAPSHOT_NAME
        if not os.path.isabs(candidate):
            candidate = os.path.join(ABSENCE_SNAPSHOT_DIR, candidate)
        if os.path.exists(candidate):
            print(f"[PATH] absence snapshot(明示指定)を採用: {candidate}")
            return candidate, _asof_key(os.path.basename(candidate))
        print(f"[PATH][WARN] 指定absence snapshotが見つかりません: {candidate}")

    # 2) snapshots から解決
    # 許容ファイル名:
    # - absences_with_impact_asof_YYYYMMDD.csv
    # - absences_with_impact_<season>_asof_YYYYMMDD.csv
    if os.path.isdir(ABSENCE_SNAPSHOT_DIR):
        candidates = []
        pat1 = re.compile(r"^absences_with_impact_asof_(\d{8})\.csv$")
        pat2 = re.compile(r"^absences_with_impact_(\d{4})_asof_(\d{8})\.csv$")
        for fn in os.listdir(ABSENCE_SNAPSHOT_DIR):
            m1 = pat1.match(fn)
            m2 = pat2.match(fn)
            if m1:
                candidates.append((m1.group(1), os.path.join(ABSENCE_SNAPSHOT_DIR, fn)))
                continue
            if m2 and int(m2.group(1)) == int(SEASON_YEAR):
                candidates.append((m2.group(2), os.path.join(ABSENCE_SNAPSHOT_DIR, fn)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            target_asof = _asof_key(ABSENCE_ASOF_DATE)
            if target_asof:
                le_targets = [x for x in candidates if x[0] <= target_asof]
                if le_targets:
                    chosen = le_targets[-1]
                    print(f"[PATH] absence snapshot(asof={target_asof})を採用: {chosen[1]}")
                    return chosen[1], chosen[0]
                print(f"[PATH][WARN] absence asof={target_asof} 以下のsnapshotがないため最新を使用します。")
            chosen = candidates[-1]
            print(f"[PATH] absence snapshot(最新)を採用: {chosen[1]}")
            return chosen[1], chosen[0]

    # 3) 従来フォールバック
    for candidate in [
        os.path.join(MANUAL_DIR, "absences_with_impact.csv"),
        os.path.join(DATA_DIR, "absences_with_impact.csv"),
    ]:
        if not os.path.exists(candidate):
            continue
        try:
            header = set(pd.read_csv(candidate, nrows=0).columns)
        except Exception:
            continue
        legacy_ok = {"team", "round_start", "impact_total"}.issubset(header)
        dated_ok = {"team", "start_date", "expected_return_date", "impact_total"}.issubset(header)
        if legacy_ok or dated_ok:
            return candidate, ""
    return None, ""


# 追加するパス（非空ファイルを優先）
team_master_stats_csv, stats_asof_key = resolve_team_master_stats_csv()
if not stats_asof_key:
    stats_asof_key = _asof_key(STATS_ASOF_DATE) or datetime.now().strftime("%Y%m%d")
STATS_ASOF_LABEL = (
    f"{stats_asof_key[:4]}-{stats_asof_key[4:6]}-{stats_asof_key[6:8]}"
    if len(stats_asof_key) >= 8
    else stats_asof_key
)
team_management_master_csv = pick_non_empty_csv_path(
    [
        os.path.join(MANUAL_DIR, "team_management_master.csv"),
        os.path.join(DATA_DIR, "team_management_master.csv"),
    ],
    required_cols=["team_name"],
)
absence_impact_csv, absence_asof_key = resolve_absence_impact_csv()
team_motivation_csv = pick_non_empty_csv_path(
    [
        os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_motivation.csv"),
        os.path.join(DATA_DIR, "team_motivation_master.csv"),
    ],
    required_cols=["team_name"],
)
derby_master_csv = os.environ.get("DERBY_MASTER_CSV", os.path.join(MANUAL_DIR, "ダービー一覧.csv"))
team_travel_distances_csv = os.path.join(MANUAL_DIR, "team_travel_distances.csv")
if not os.path.exists(team_travel_distances_csv):
    team_travel_distances_csv = os.path.join(DATA_DIR, "team_travel_distances.csv")
team_fatigue_scores_csv = os.path.join(DATA_DIR, f"team_fatigue_scores_{LEAGUE}_{SEASON_YEAR}.csv")
if not os.path.exists(team_fatigue_scores_csv):
    team_fatigue_scores_csv = os.path.join(DATA_DIR, f"team_fatigue_scores_{SEASON_YEAR}.csv")
if not os.path.exists(team_fatigue_scores_csv):
    team_fatigue_scores_csv = os.path.join(DATA_DIR, "team_fatigue_scores.csv")
acl_schedule_csv = os.environ.get("ACL_SCHEDULE_CSV", os.path.join(MANUAL_DIR, "acl_schedule.csv"))
ACL_EFFECTIVE_DAYS = _get_env_int("ACL_EFFECTIVE_DAYS", 5)
ACL_DEBUG = _env_flag("ACL_DEBUG", 0)
ACL_FATIGUE_MULTIPLIER = float(os.environ.get("ACL_FATIGUE_MULTIPLIER", "1.45"))
ACL_FATIGUE_SHORT_REST_BONUS = float(os.environ.get("ACL_FATIGUE_SHORT_REST_BONUS", "0.25"))
ACL_FATIGUE_TRAVEL_AWAY_BONUS = float(os.environ.get("ACL_FATIGUE_TRAVEL_AWAY_BONUS", "0.20"))
ACL_AWAY_TRAVEL_TYPES = {
    "away",
    "international_away",
    "overseas",
    "short",
    "medium",
    "long_away",
    "long",
    "long_haul",
}
ACL_SECOND_WINDOW_DAYS = _get_env_int("ACL_SECOND_WINDOW_DAYS", 14)
ACL_SECOND_WINDOW_DECAY = float(os.environ.get("ACL_SECOND_WINDOW_DECAY", "0.85"))
ACL_DRAW_MIN_FATIGUE = float(os.environ.get("ACL_DRAW_MIN_FATIGUE", "5.0"))
ACL_DRAW_EDGE_BASE = float(os.environ.get("ACL_DRAW_EDGE_BASE", "0.025"))
ACL_DRAW_EDGE_PER_FATIGUE = float(os.environ.get("ACL_DRAW_EDGE_PER_FATIGUE", "0.005"))
ACL_DRAW_EDGE_CAP = float(os.environ.get("ACL_DRAW_EDGE_CAP", "0.120"))
ACL_DRAW_DRAWRISK_BONUS = float(os.environ.get("ACL_DRAW_DRAWRISK_BONUS", "0.015"))
ACL_DRAW_WEATHER_BONUS_SCALE = float(os.environ.get("ACL_DRAW_WEATHER_BONUS_SCALE", "0.012"))
ACL_DRAW_SECOND_WINDOW_BONUS = float(os.environ.get("ACL_DRAW_SECOND_WINDOW_BONUS", "0.000"))

# 天候キャッシュは as-of 付きsnapshotを優先し、なければ従来ファイルへフォールバック
weather_cache_csv, weather_asof_key = resolve_weather_cache_csv()
J2_ALLOWED_TEAMS_CSV = os.environ.get(
    "J2_ALLOWED_TEAMS_CSV",
    os.path.join(MANUAL_DIR, f"j2_allowed_teams_{SEASON_YEAR}.csv"),
)

# Elo-like 初期スコア
INITIAL_ELO = 1500
ELO_UPDATE_HOME_ADVANTAGE = float(os.environ.get("ELO_UPDATE_HOME_ADVANTAGE", "0"))
GOAL_SCALING_FACTOR = 0.01
FATIGUE_GOAL_SCALING = 0.01
AWAY_PROB_MULTIPLIER = float(os.environ.get("AWAY_PROB_MULTIPLIER", "1.05"))
ENABLE_ROUND_TYPE_DRAW_CONTROL = _env_flag("ENABLE_ROUND_TYPE_DRAW_CONTROL", 0)
ROUND_TYPE_DRAW_REL_THRESHOLD = float(os.environ.get("ROUND_TYPE_DRAW_REL_THRESHOLD", "0.008"))
ROUND_TYPE_DRAW_SHARE_THRESHOLD = float(os.environ.get("ROUND_TYPE_DRAW_SHARE_THRESHOLD", "0.30"))
ROUND_TYPE_DRAW_HEAVY_AVG = float(os.environ.get("ROUND_TYPE_DRAW_HEAVY_AVG", "0.335"))
ROUND_TYPE_DRAW_LIGHT_AVG = float(os.environ.get("ROUND_TYPE_DRAW_LIGHT_AVG", "0.325"))
ROUND_TYPE_DRAW_BOOST = float(os.environ.get("ROUND_TYPE_DRAW_BOOST", "0.015"))
ROUND_TYPE_DRAW_CLAMP_MIN = float(os.environ.get("ROUND_TYPE_DRAW_CLAMP_MIN", "0.05"))
ROUND_TYPE_DRAW_CLAMP_MAX = float(os.environ.get("ROUND_TYPE_DRAW_CLAMP_MAX", "0.60"))
RANK_MOTIVATION_GOAL_SCALING = float(os.environ.get("RANK_MOTIVATION_GOAL_SCALING", "0.01"))
ABSENCE_IMPACT_GOAL_SCALING = float(os.environ.get("ABSENCE_IMPACT_GOAL_SCALING", "0.25"))
# 欠場データ欠損時のベースライン（観測バイアス緩和）
ABSENCE_BASELINE_TOTAL = float(os.environ.get("ABSENCE_BASELINE_TOTAL", "0.05"))
ABSENCE_BASELINE_ATTACK = float(os.environ.get("ABSENCE_BASELINE_ATTACK", "0.03"))
ABSENCE_BASELINE_DEFENSE = float(os.environ.get("ABSENCE_BASELINE_DEFENSE", "0.02"))
# 欠場影響の過補正防止
ABSENCE_IMPACT_CAP_TOTAL = float(os.environ.get("ABSENCE_IMPACT_CAP_TOTAL", "0.25"))
# multinom主経路での弱い欠場補正（Elo差の補助項）
ABSENCE_ELO_COEF = float(os.environ.get("ABSENCE_ELO_COEF", "0.10"))
ABSENCE_ELO_ADJUST_CLIP = float(os.environ.get("ABSENCE_ELO_ADJUST_CLIP", "0.03"))
WEATHER_PENALTY_HEAVY_RAIN = 0.15
WEATHER_PENALTY_STRONG_WIND = 0.10
WEATHER_PENALTY_RAIN = 0.05
WEATHER_DEFAULT_TEMPERATURE = float(os.environ.get("WEATHER_DEFAULT_TEMPERATURE", "15.0"))
WEATHER_DEFAULT_WIND_SPEED = float(os.environ.get("WEATHER_DEFAULT_WIND_SPEED", "3.0"))
D_INTERCEPT = -1.2
D_SCALE = 1.5
DRAW_PROB_THRESHOLD = float(os.environ.get("DRAW_PROB_THRESHOLD", "0.24"))
DRAW_BALANCE_THRESHOLD = 0.10
HOME_ADV_ELO_COEF = float(os.environ.get("HOME_ADV_ELO_COEF", "60"))
# J1は既定でHFAを切る。J2は残すが、効き方を弱める。
_DEFAULT_HFA_ELO = "12" if LEAGUE == "j2" else "0"
HFA_ELO = float(os.environ.get("HFA_ELO", _DEFAULT_HFA_ELO))
HOME_ADV_PROFILE_DIFF_CLIP = float(os.environ.get("HOME_ADV_PROFILE_DIFF_CLIP", "0.8"))
# HFAは固定定数（デフォルト35）。試合固有バイアスは別スイッチで分離する。
ENABLE_MATCHUP_BIAS = _env_flag("ENABLE_MATCHUP_BIAS", 0)
MATCHUP_BIAS_COEF = float(os.environ.get("MATCHUP_BIAS_COEF", str(HOME_ADV_ELO_COEF)))
ELO_DIFF_TEMPERATURE = float(os.environ.get("ELO_DIFF_TEMPERATURE", "1.35"))
_DEFAULT_ELO_DIFF_SCALE = "1.15" if LEAGUE == "j2" else "1.10"
ELO_DIFF_SCALE = float(os.environ.get("ELO_DIFF_SCALE", _DEFAULT_ELO_DIFF_SCALE))
# Elo差→勝率変換の感度。400より大きくすると確率変化が緩やかになる。
_DEFAULT_ELO_D_VALUE = "550" if LEAGUE == "j2" else "600"
ELO_D_VALUE = float(os.environ.get("ELO_D_VALUE", _DEFAULT_ELO_D_VALUE))
# HFAは確率入力に対して重み付きで反映（過剰なhome偏重を抑制）
_DEFAULT_HFA_PROB_WEIGHT = "0.35" if LEAGUE == "j2" else "0.00"
HFA_PROB_WEIGHT = float(os.environ.get("HFA_PROB_WEIGHT", _DEFAULT_HFA_PROB_WEIGHT))
_DEFAULT_HOME_ADV_FEATURE_SCALE = "1.0"
if str(LEAGUE).lower() == "j1":
    _DEFAULT_HOME_ADV_FEATURE_SCALE = "1.0"
HOME_ADV_FEATURE_SCALE = float(os.environ.get("HOME_ADV_FEATURE_SCALE", _DEFAULT_HOME_ADV_FEATURE_SCALE))
ELO_DRAW_BASE = float(os.environ.get("ELO_DRAW_BASE", "0.33"))
# base主導で調整できるよう、既定は0（必要時のみ環境変数で有効化）
ELO_DRAW_BUMP = float(os.environ.get("ELO_DRAW_BUMP", "0.00"))
ELO_DRAW_SENSITIVITY = float(os.environ.get("ELO_DRAW_SENSITIVITY", "400"))
ELO_DRAW_DIFF_SCALE = float(os.environ.get("ELO_DRAW_DIFF_SCALE", "1.00"))
ELO_DRAW_MIN = float(os.environ.get("ELO_DRAW_MIN", "0.10"))
ELO_DRAW_MAX = float(os.environ.get("ELO_DRAW_MAX", "0.38"))
DRAW_DECAY_SCALE = float(os.environ.get("DRAW_DECAY_SCALE", "320.0"))
# draw確率はPoisson由来とElo由来をブレンド（1.0=Poissonのみ, 0.0=Eloのみ）
DRAW_BLEND_WEIGHT = float(os.environ.get("DRAW_BLEND_WEIGHT", "0.60"))
DRAW_TWEAK_MODE = os.environ.get("DRAW_TWEAK_MODE", "off").strip().lower()
if DRAW_TWEAK_MODE not in {"off", "on"}:
    print(f"[CONFIG][WARN] invalid DRAW_TWEAK_MODE={DRAW_TWEAK_MODE!r}; fallback='off'")
    DRAW_TWEAK_MODE = "off"
DRAW_TWEAK_ENABLED = DRAW_TWEAK_MODE == "on"
DRAW_ASSIGN_BY_EXPECTATION_RAW = _env_flag("DRAW_ASSIGN_BY_EXPECTATION", 1)
DRAW_ASSIGN_BY_EXPECTATION = bool(DRAW_TWEAK_ENABLED and DRAW_ASSIGN_BY_EXPECTATION_RAW)
# 期待ドロー件数の倍率（確率自体は変更せず、D割当件数のみ調整）
DRAW_EXPECTATION_MULTIPLIER_RAW = float(os.environ.get("DRAW_EXPECTATION_MULTIPLIER", "1.0"))
DRAW_EXPECTATION_MULTIPLIER = float(DRAW_EXPECTATION_MULTIPLIER_RAW) if DRAW_TWEAK_ENABLED else 1.0
if (not DRAW_TWEAK_ENABLED) and BACKTEST_DECISION_RULE != "argmax":
    print(
        f"[CONFIG][WARN] DRAW_TWEAK_MODE=off のため BACKTEST_DECISION_RULE={BACKTEST_DECISION_RULE!r} は使用せず "
        "argmax固定で評価します"
    )
DRAW_ASSIGN_GROUP_MODE = os.environ.get("DRAW_ASSIGN_GROUP_MODE", "toto_contest").strip().lower()
DRAW_MARGIN = float(os.environ.get("DRAW_MARGIN", "0.03"))
CLOSE_HA_GAP = float(os.environ.get("CLOSE_HA_GAP", "0.06"))
CLOSE_HA_MIN_LEVEL = float(os.environ.get("CLOSE_HA_MIN_LEVEL", "0.33"))
CLOSE_HA_GAP_WEIGHT = float(os.environ.get("CLOSE_HA_GAP_WEIGHT", "0.20"))
CLOSE_HA_DRAW_SCORE_MIN = float(os.environ.get("CLOSE_HA_DRAW_SCORE_MIN", "-0.03"))
CLOSE_HA_DRAW_SCORE_MIN_GRID_RAW = os.environ.get("CLOSE_HA_DRAW_SCORE_MIN_GRID", "-0.03,-0.02,-0.01,0.00,0.01")
CLOSE_D_TOP_GAP = float(os.environ.get("CLOSE_D_TOP_GAP", "0.06"))
CLOSE_D_TOP_GAP_GRID = os.environ.get("CLOSE_D_TOP_GAP_GRID", "0.02,0.03,0.04,0.05,0.06,0.07,0.08")
DRAW_CANDIDATE_PROB_MIN = float(os.environ.get("DRAW_CANDIDATE_PROB_MIN", "0.33"))
DRAW_CANDIDATE_GAP_MAX = float(os.environ.get("DRAW_CANDIDATE_GAP_MAX", "0.02"))
ENABLE_NARROW_DRAW_OVERRIDE = _env_flag("ENABLE_NARROW_DRAW_OVERRIDE", 0)
J1_NARROW_DRAW_PROB_MIN = float(os.environ.get("J1_NARROW_DRAW_PROB_MIN", "0.335"))
J1_NARROW_DRAW_GAP_MAX = float(os.environ.get("J1_NARROW_DRAW_GAP_MAX", "0.010"))
J1_NARROW_DRAW_STRONG_PROB_MIN = float(os.environ.get("J1_NARROW_DRAW_STRONG_PROB_MIN", "0.345"))
J1_NARROW_DRAW_STRONG_GAP_MAX = float(os.environ.get("J1_NARROW_DRAW_STRONG_GAP_MAX", "0.000"))
J1_NARROW_DRAW_NEUTRAL_BALANCED_PROB_MIN = float(os.environ.get("J1_NARROW_DRAW_NEUTRAL_BALANCED_PROB_MIN", "0.430"))
J1_NARROW_DRAW_NEUTRAL_BALANCED_GAP_MAX = float(os.environ.get("J1_NARROW_DRAW_NEUTRAL_BALANCED_GAP_MAX", "-0.130"))
J1_NARROW_DRAW_NEUTRAL_BALANCED_SUPPORT_MIN = float(os.environ.get("J1_NARROW_DRAW_NEUTRAL_BALANCED_SUPPORT_MIN", "0.560"))
J2_NARROW_DRAW_PROB_MIN = float(os.environ.get("J2_NARROW_DRAW_PROB_MIN", "0.338"))
J2_NARROW_DRAW_GAP_MAX = float(os.environ.get("J2_NARROW_DRAW_GAP_MAX", "0.005"))
ENABLE_J2_NARROW_DRAW_SIDE_RESTORE = _env_flag("ENABLE_J2_NARROW_DRAW_SIDE_RESTORE", 1)
J2_NARROW_DRAW_AWAY_PREF_A_PROB_MIN = float(
    os.environ.get("J2_NARROW_DRAW_AWAY_PREF_A_PROB_MIN", "0.375")
)
J2_NARROW_DRAW_HOME_PREF_H_PROB_MIN = float(
    os.environ.get("J2_NARROW_DRAW_HOME_PREF_H_PROB_MIN", "0.345")
)
ENABLE_J2_NARROW_DRAW_NEUTRAL_DRAW_COMPRESSED_RESTORE = _env_flag(
    "ENABLE_J2_NARROW_DRAW_NEUTRAL_DRAW_COMPRESSED_RESTORE", 1
)
J2_NARROW_DRAW_NEUTRAL_HOME_RANK_GAP_MAX = float(
    os.environ.get("J2_NARROW_DRAW_NEUTRAL_HOME_RANK_GAP_MAX", "-1.0")
)
J2_NARROW_DRAW_NEUTRAL_AWAY_RANK_GAP_MIN = float(
    os.environ.get("J2_NARROW_DRAW_NEUTRAL_AWAY_RANK_GAP_MIN", "1.0")
)
J2_NARROW_DRAW_NEUTRAL_FATIGUE_EDGE_MIN = float(
    os.environ.get("J2_NARROW_DRAW_NEUTRAL_FATIGUE_EDGE_MIN", "2.0")
)
ENABLE_J2_NEUTRAL_DRAW_AWAY_RESTORE = _env_flag("ENABLE_J2_NEUTRAL_DRAW_AWAY_RESTORE", 0)
J2_NEUTRAL_DRAW_AWAY_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_DRAW_AWAY_RESTORE_DRAW_GAP_MAX", "0.080")
)
J2_NEUTRAL_DRAW_AWAY_RESTORE_FATIGUE_EDGE_MIN = float(
    os.environ.get("J2_NEUTRAL_DRAW_AWAY_RESTORE_FATIGUE_EDGE_MIN", "2.0")
)
J2_NEUTRAL_DRAW_AWAY_RESTORE_RANK_GAP_MIN = float(
    os.environ.get("J2_NEUTRAL_DRAW_AWAY_RESTORE_RANK_GAP_MIN", "1.0")
)
ENABLE_J2_NEUTRAL_DRAW_HOME_RESTORE_V2 = _env_flag("ENABLE_J2_NEUTRAL_DRAW_HOME_RESTORE_V2", 0)
J2_NEUTRAL_DRAW_HOME_RESTORE_V2_DRAW_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_DRAW_HOME_RESTORE_V2_DRAW_GAP_MAX", "0.110")
)
J2_NEUTRAL_DRAW_HOME_RESTORE_V2_FATIGUE_EDGE_MIN = float(
    os.environ.get("J2_NEUTRAL_DRAW_HOME_RESTORE_V2_FATIGUE_EDGE_MIN", "2.0")
)
J2_NEUTRAL_DRAW_HOME_RESTORE_V2_RANK_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_DRAW_HOME_RESTORE_V2_RANK_GAP_MAX", "-1.0")
)
ENABLE_MAIN_NARROW_DRAW_OVERRIDE = _env_flag("ENABLE_MAIN_NARROW_DRAW_OVERRIDE", 0)
J1_MAIN_NARROW_DRAW_PROB_MIN = float(os.environ.get("J1_MAIN_NARROW_DRAW_PROB_MIN", "0.330"))
J1_MAIN_NARROW_DRAW_GAP_MAX = float(os.environ.get("J1_MAIN_NARROW_DRAW_GAP_MAX", "0.025"))
J2_MAIN_NARROW_DRAW_PROB_MIN = float(os.environ.get("J2_MAIN_NARROW_DRAW_PROB_MIN", "0.338"))
J2_MAIN_NARROW_DRAW_GAP_MAX = float(os.environ.get("J2_MAIN_NARROW_DRAW_GAP_MAX", "0.005"))
ENABLE_J2_MAIN_SIGNAL_CONFLICT_BALANCED_AWAY_RESTORE = _env_flag(
    "ENABLE_J2_MAIN_SIGNAL_CONFLICT_BALANCED_AWAY_RESTORE", 0
)
ENABLE_J2_MAIN_NEUTRAL_DRAW_COMPRESSED_AWAY_RESTORE = _env_flag(
    "ENABLE_J2_MAIN_NEUTRAL_DRAW_COMPRESSED_AWAY_RESTORE", 0
)
ENABLE_J1_MAIN_NEUTRAL_SPLIT_SIDE_AWAY_RESTORE = _env_flag(
    "ENABLE_J1_MAIN_NEUTRAL_SPLIT_SIDE_AWAY_RESTORE", 0
)
ENABLE_J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_SYNC = _env_flag(
    "ENABLE_J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_SYNC", 0
)
J1_MAIN_NEUTRAL_SPLIT_SIDE_AWAY_GAP_MAX = float(
    os.environ.get("J1_MAIN_NEUTRAL_SPLIT_SIDE_AWAY_GAP_MAX", "0.020")
)
J1_MAIN_NEUTRAL_SPLIT_SIDE_LAB_EDGE_MAX = float(
    os.environ.get("J1_MAIN_NEUTRAL_SPLIT_SIDE_LAB_EDGE_MAX", "-3.000")
)
J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_DRAW_MIN = float(
    os.environ.get("J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_DRAW_MIN", "0.360")
)
J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_GAP_MIN = float(
    os.environ.get("J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_GAP_MIN", "0.020")
)
BASE_CONFIDENCE_WEIGHT = float(os.environ.get("BASE_CONFIDENCE_WEIGHT", "1.00"))
LAB_CONFIDENCE_WEIGHT = float(os.environ.get("LAB_CONFIDENCE_WEIGHT", "1.00"))
MAIN_CONFIDENCE_WEIGHT = float(os.environ.get("MAIN_CONFIDENCE_WEIGHT", "0.12"))
J1_NEUTRAL_BALANCED_DRAW_MIN = float(os.environ.get("J1_NEUTRAL_BALANCED_DRAW_MIN", "0.390"))
J1_NEUTRAL_BALANCED_DRAW_MAX = float(os.environ.get("J1_NEUTRAL_BALANCED_DRAW_MAX", "0.445"))
J1_NEUTRAL_BALANCED_HOME_READY_DRAW_MIN = float(os.environ.get("J1_NEUTRAL_BALANCED_HOME_READY_DRAW_MIN", "0.350"))
J1_NEUTRAL_BALANCED_HA_GAP_MAX = float(os.environ.get("J1_NEUTRAL_BALANCED_HA_GAP_MAX", "0.050"))
J1_NEUTRAL_BALANCED_DRAW_SHIFT_MAX = float(os.environ.get("J1_NEUTRAL_BALANCED_DRAW_SHIFT_MAX", "0.045"))
J1_HOME_READY_FATIGUE_EDGE = float(os.environ.get("J1_HOME_READY_FATIGUE_EDGE", "1.50"))
J1_HOME_READY_MOTIVATION_EDGE = float(os.environ.get("J1_HOME_READY_MOTIVATION_EDGE", "0.25"))
J1_HOME_READY_LAB_EDGE_MAX = float(os.environ.get("J1_HOME_READY_LAB_EDGE_MAX", "0.015"))
TITLE_RACE_RANK_MAX = int(os.environ.get("TITLE_RACE_RANK_MAX", "3"))
J1_RELEGATION_RISK_BOTTOM_N = int(os.environ.get("J1_RELEGATION_RISK_BOTTOM_N", "4"))
J2_RELEGATION_RISK_BOTTOM_N = int(os.environ.get("J2_RELEGATION_RISK_BOTTOM_N", "4"))
J1_TITLE_PUSH_RANK_MAX = int(os.environ.get("J1_TITLE_PUSH_RANK_MAX", "2"))
J1_ACL_LINE_RANK = int(os.environ.get("J1_ACL_LINE_RANK", "3"))
J1_ACL_RACE_RANK_MAX = int(os.environ.get("J1_ACL_RACE_RANK_MAX", "5"))
J2_PROMOTION_LINE_RANK = int(os.environ.get("J2_PROMOTION_LINE_RANK", "2"))
J2_PROMOTION_RACE_RANK_MAX = int(os.environ.get("J2_PROMOTION_RACE_RANK_MAX", "4"))
J2_PO_LINE_RANK = int(os.environ.get("J2_PO_LINE_RANK", "6"))
J2_PO_RACE_RANK_MAX = int(os.environ.get("J2_PO_RACE_RANK_MAX", "8"))
PRESSURE_DIRECT_POINTS_GAP_MAX = float(os.environ.get("PRESSURE_DIRECT_POINTS_GAP_MAX", "6.0"))
PRESSURE_BORDER_MOTIVATION_MIN = float(os.environ.get("PRESSURE_BORDER_MOTIVATION_MIN", "0.35"))
INCENTIVE_DRAW_SHIFT_ENABLE = _env_flag("INCENTIVE_DRAW_SHIFT_ENABLE", 0)
J1_INCENTIVE_DRAW_PROB_MIN = float(os.environ.get("J1_INCENTIVE_DRAW_PROB_MIN", "0.305"))
J1_INCENTIVE_DRAW_GAP_MAX = float(os.environ.get("J1_INCENTIVE_DRAW_GAP_MAX", "0.050"))
J2_INCENTIVE_DRAW_PROB_MIN = float(os.environ.get("J2_INCENTIVE_DRAW_PROB_MIN", "0.330"))
J2_INCENTIVE_DRAW_GAP_MAX = float(os.environ.get("J2_INCENTIVE_DRAW_GAP_MAX", "0.040"))
INCENTIVE_TITLE_EDGE_MAX = float(os.environ.get("INCENTIVE_TITLE_EDGE_MAX", "0.020"))
ENABLE_J1_AWAY_RESTORE_OVERRIDE = _env_flag("ENABLE_J1_AWAY_RESTORE_OVERRIDE", 0)
J1_AWAY_RESTORE_HOME_GAP_MAX = float(os.environ.get("J1_AWAY_RESTORE_HOME_GAP_MAX", "0.030"))
J1_AWAY_RESTORE_DRAW_GAP_MAX = float(os.environ.get("J1_AWAY_RESTORE_DRAW_GAP_MAX", "0.090"))
J1_AWAY_RESTORE_DRAW_MIN = float(os.environ.get("J1_AWAY_RESTORE_DRAW_MIN", "0.350"))
ENABLE_J1_HOME_RESTORE_OVERRIDE = _env_flag("ENABLE_J1_HOME_RESTORE_OVERRIDE", 0)
J1_HOME_RESTORE_AWAY_GAP_MAX = float(os.environ.get("J1_HOME_RESTORE_AWAY_GAP_MAX", "0.030"))
J1_HOME_RESTORE_DRAW_GAP_MAX = float(os.environ.get("J1_HOME_RESTORE_DRAW_GAP_MAX", "0.090"))
J1_HOME_RESTORE_DRAW_MIN = float(os.environ.get("J1_HOME_RESTORE_DRAW_MIN", "0.345"))
J1_HOME_RESTORE_DRAW_MAX = float(os.environ.get("J1_HOME_RESTORE_DRAW_MAX", "0.372"))
J1_HOME_RESTORE_DRAW_SUPPORT_MAX = float(os.environ.get("J1_HOME_RESTORE_DRAW_SUPPORT_MAX", "0.545"))
J1_HOME_RESTORE_COMPRESSED_DRAW_MAX = float(os.environ.get("J1_HOME_RESTORE_COMPRESSED_DRAW_MAX", "0.455"))
J1_HOME_RESTORE_COMPRESSED_FATIGUE_EDGE = float(os.environ.get("J1_HOME_RESTORE_COMPRESSED_FATIGUE_EDGE", "3.0"))
J1_HOME_RESTORE_COMPRESSED_MOTIVATION_EDGE = float(os.environ.get("J1_HOME_RESTORE_COMPRESSED_MOTIVATION_EDGE", "0.2"))
J1_HOME_RESTORE_COMPRESSED_HOME_GAP_MAX = float(os.environ.get("J1_HOME_RESTORE_COMPRESSED_HOME_GAP_MAX", "0.005"))
ENABLE_J1_SIGNAL_CONFLICT_HOME_RESTORE = _env_flag("ENABLE_J1_SIGNAL_CONFLICT_HOME_RESTORE", 0)
J1_SIGNAL_CONFLICT_HOME_RESTORE_HOME_GAP_MAX = float(
    os.environ.get("J1_SIGNAL_CONFLICT_HOME_RESTORE_HOME_GAP_MAX", "0.020")
)
J1_SIGNAL_CONFLICT_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J1_SIGNAL_CONFLICT_HOME_RESTORE_FATIGUE_EDGE", "3.0")
)
J1_SIGNAL_CONFLICT_HOME_RESTORE_RANK_GAP_MAX = float(
    os.environ.get("J1_SIGNAL_CONFLICT_HOME_RESTORE_RANK_GAP_MAX", "-3.0")
)
ENABLE_J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE = _env_flag(
    "ENABLE_J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE", 0
)
J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_FATIGUE_EDGE", "2.5")
)
J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_MOTIVATION_EDGE = float(
    os.environ.get("J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_MOTIVATION_EDGE", "0.0")
)
J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_HOME_GAP_MIN = float(
    os.environ.get("J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_HOME_GAP_MIN", "-0.030")
)
J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_HOME_GAP_MAX = float(
    os.environ.get("J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_HOME_GAP_MAX", "0.000")
)
ENABLE_J1_SIGNAL_CONFLICT_AWAY_RESTORE = _env_flag("ENABLE_J1_SIGNAL_CONFLICT_AWAY_RESTORE", 0)
J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_GAP_MAX = float(
    os.environ.get("J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_GAP_MAX", "0.010")
)
J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_FATIGUE_EDGE_MAX = float(
    os.environ.get("J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_FATIGUE_EDGE_MAX", "2.5")
)
J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_RANK_GAP_MIN = float(
    os.environ.get("J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_RANK_GAP_MIN", "-3.0")
)
ENABLE_J2_AWAY_STRONG_AWAY_RESTORE = _env_flag("ENABLE_J2_AWAY_STRONG_AWAY_RESTORE", 0)
ENABLE_J2_SIGNAL_CONFLICT_AWAY_RESTORE = _env_flag("ENABLE_J2_SIGNAL_CONFLICT_AWAY_RESTORE", 0)
ENABLE_J2_NEG_HOME_ADV_AWAY_RESTORE = _env_flag("ENABLE_J2_NEG_HOME_ADV_AWAY_RESTORE", 0)
ENABLE_J2_HOME_RESTORE_OVERRIDE = _env_flag("ENABLE_J2_HOME_RESTORE_OVERRIDE", 0)
J2_HOME_RESTORE_DRAW_GAP_MAX = float(os.environ.get("J2_HOME_RESTORE_DRAW_GAP_MAX", "0.050"))
J2_HOME_RESTORE_FATIGUE_EDGE = float(os.environ.get("J2_HOME_RESTORE_FATIGUE_EDGE", "2.5"))
J2_HOME_RESTORE_MOTIVATION_EDGE = float(os.environ.get("J2_HOME_RESTORE_MOTIVATION_EDGE", "0.5"))
J2_HOME_RESTORE_RANK_GAP_MAX = float(os.environ.get("J2_HOME_RESTORE_RANK_GAP_MAX", "-2.0"))
ENABLE_J2_NEUTRAL_HOME_RESTORE = _env_flag("ENABLE_J2_NEUTRAL_HOME_RESTORE", 0)
J2_NEUTRAL_HOME_RESTORE_DRAW_GAP_MAX = float(os.environ.get("J2_NEUTRAL_HOME_RESTORE_DRAW_GAP_MAX", "0.050"))
J2_NEUTRAL_HOME_RESTORE_FATIGUE_EDGE = float(os.environ.get("J2_NEUTRAL_HOME_RESTORE_FATIGUE_EDGE", "2.5"))
J2_NEUTRAL_HOME_RESTORE_MOTIVATION_EDGE = float(os.environ.get("J2_NEUTRAL_HOME_RESTORE_MOTIVATION_EDGE", "0.5"))
J2_NEUTRAL_HOME_RESTORE_HOME_GAP_MAX = float(os.environ.get("J2_NEUTRAL_HOME_RESTORE_HOME_GAP_MAX", "0.090"))
ENABLE_J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE = _env_flag("ENABLE_J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE", 0)
J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_DRAW_GAP_MAX", "0.030")
)
J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_FATIGUE_EDGE", "2.0")
)
J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_MOTIVATION_EDGE = float(
    os.environ.get("J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_MOTIVATION_EDGE", "2.0")
)
J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_RANK_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_RANK_GAP_MAX", "-5.0")
)
ENABLE_J2_HOME_STRONG_BALANCED_RESTORE = _env_flag("ENABLE_J2_HOME_STRONG_BALANCED_RESTORE", 0)
J2_HOME_STRONG_BALANCED_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_BALANCED_RESTORE_DRAW_GAP_MAX", "0.070")
)
J2_HOME_STRONG_BALANCED_RESTORE_HOME_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_BALANCED_RESTORE_HOME_GAP_MAX", "0.030")
)
J2_HOME_STRONG_BALANCED_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_HOME_STRONG_BALANCED_RESTORE_FATIGUE_EDGE", "3.0")
)
J2_HOME_STRONG_BALANCED_RESTORE_MOTIVATION_EDGE = float(
    os.environ.get("J2_HOME_STRONG_BALANCED_RESTORE_MOTIVATION_EDGE", "1.0")
)
J2_HOME_STRONG_BALANCED_RESTORE_RANK_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_BALANCED_RESTORE_RANK_GAP_MAX", "-4.0")
)
ENABLE_J2_SIGNAL_CONFLICT_HOME_RESTORE = _env_flag("ENABLE_J2_SIGNAL_CONFLICT_HOME_RESTORE", 0)
J2_SIGNAL_CONFLICT_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_SIGNAL_CONFLICT_HOME_RESTORE_DRAW_GAP_MAX", "0.050")
)
J2_SIGNAL_CONFLICT_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_SIGNAL_CONFLICT_HOME_RESTORE_FATIGUE_EDGE", "5.0")
)
J2_SIGNAL_CONFLICT_HOME_RESTORE_MOTIVATION_EDGE = float(
    os.environ.get("J2_SIGNAL_CONFLICT_HOME_RESTORE_MOTIVATION_EDGE", "2.0")
)
J2_SIGNAL_CONFLICT_HOME_RESTORE_RANK_GAP_MAX = float(
    os.environ.get("J2_SIGNAL_CONFLICT_HOME_RESTORE_RANK_GAP_MAX", "-3.0")
)
ENABLE_J2_AWAY_STRONG_FLAT_HOME_RESTORE = _env_flag("ENABLE_J2_AWAY_STRONG_FLAT_HOME_RESTORE", 0)
J2_AWAY_STRONG_FLAT_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_AWAY_STRONG_FLAT_HOME_RESTORE_FATIGUE_EDGE", "4.0")
)
J2_AWAY_STRONG_FLAT_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_AWAY_STRONG_FLAT_HOME_RESTORE_DRAW_GAP_MAX", "0.145")
)
ENABLE_J2_CLOSE_FLAT_HOME_RESTORE = _env_flag("ENABLE_J2_CLOSE_FLAT_HOME_RESTORE", 0)
J2_CLOSE_FLAT_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_CLOSE_FLAT_HOME_RESTORE_FATIGUE_EDGE", "4.0")
)
J2_CLOSE_FLAT_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_CLOSE_FLAT_HOME_RESTORE_DRAW_GAP_MAX", "0.125")
)
ENABLE_J2_HOME_STRONG_FLAT_HOME_RESTORE = _env_flag("ENABLE_J2_HOME_STRONG_FLAT_HOME_RESTORE", 0)
J2_HOME_STRONG_FLAT_HOME_RESTORE_MOTIVATION_EDGE = float(
    os.environ.get("J2_HOME_STRONG_FLAT_HOME_RESTORE_MOTIVATION_EDGE", "2.0")
)
J2_HOME_STRONG_FLAT_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_FLAT_HOME_RESTORE_DRAW_GAP_MAX", "0.010")
)
ENABLE_J2_HOME_STRONG_DEEP_HOME_RESTORE = _env_flag("ENABLE_J2_HOME_STRONG_DEEP_HOME_RESTORE", 0)
J2_HOME_STRONG_DEEP_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_DEEP_HOME_RESTORE_DRAW_GAP_MAX", "0.030")
)
J2_HOME_STRONG_DEEP_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_HOME_STRONG_DEEP_HOME_RESTORE_FATIGUE_EDGE", "3.5")
)
J2_HOME_STRONG_DEEP_HOME_RESTORE_RANK_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_DEEP_HOME_RESTORE_RANK_GAP_MAX", "-5.0")
)
ENABLE_J2_HOME_STRONG_ELITE_HOME_RESTORE = _env_flag("ENABLE_J2_HOME_STRONG_ELITE_HOME_RESTORE", 0)
J2_HOME_STRONG_ELITE_HOME_RESTORE_HOME_MIN = float(
    os.environ.get("J2_HOME_STRONG_ELITE_HOME_RESTORE_HOME_MIN", "0.380")
)
J2_HOME_STRONG_ELITE_HOME_RESTORE_ELO_MIN = float(
    os.environ.get("J2_HOME_STRONG_ELITE_HOME_RESTORE_ELO_MIN", "50.0")
)
ENABLE_J2_NEUTRAL_POS_ELO_HOME_RESTORE = _env_flag("ENABLE_J2_NEUTRAL_POS_ELO_HOME_RESTORE", 0)
J2_NEUTRAL_POS_ELO_HOME_RESTORE_HOME_MIN = float(
    os.environ.get("J2_NEUTRAL_POS_ELO_HOME_RESTORE_HOME_MIN", "0.360")
)
J2_NEUTRAL_POS_ELO_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_POS_ELO_HOME_RESTORE_DRAW_GAP_MAX", "0.035")
)
J2_NEUTRAL_POS_ELO_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_NEUTRAL_POS_ELO_HOME_RESTORE_FATIGUE_EDGE", "4.0")
)
J2_NEUTRAL_POS_ELO_HOME_RESTORE_ELO_MIN = float(
    os.environ.get("J2_NEUTRAL_POS_ELO_HOME_RESTORE_ELO_MIN", "50.0")
)
ENABLE_J2_NEUTRAL_NEG_ELO_HOME_RESTORE = _env_flag("ENABLE_J2_NEUTRAL_NEG_ELO_HOME_RESTORE", 0)
J2_NEUTRAL_NEG_ELO_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_NEG_ELO_HOME_RESTORE_DRAW_GAP_MAX", "0.125")
)
J2_NEUTRAL_NEG_ELO_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_NEUTRAL_NEG_ELO_HOME_RESTORE_FATIGUE_EDGE", "4.3")
)
J2_NEUTRAL_NEG_ELO_HOME_RESTORE_MOTIVATION_EDGE = float(
    os.environ.get("J2_NEUTRAL_NEG_ELO_HOME_RESTORE_MOTIVATION_EDGE", "1.0")
)
J2_NEUTRAL_NEG_ELO_HOME_RESTORE_RANK_GAP_MIN = float(
    os.environ.get("J2_NEUTRAL_NEG_ELO_HOME_RESTORE_RANK_GAP_MIN", "2.0")
)
J2_NEUTRAL_NEG_ELO_HOME_RESTORE_ELO_MAX = float(
    os.environ.get("J2_NEUTRAL_NEG_ELO_HOME_RESTORE_ELO_MAX", "-20.0")
)
ENABLE_J2_HOME_STRONG_TIGHT_BALANCED_RESTORE = _env_flag("ENABLE_J2_HOME_STRONG_TIGHT_BALANCED_RESTORE", 0)
J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_DRAW_GAP_MAX", "0.001")
)
J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_FATIGUE_EDGE", "4.0")
)
J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_RANK_GAP_MAX = float(
    os.environ.get("J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_RANK_GAP_MAX", "-3.0")
)
ENABLE_J2_NEUTRAL_TIGHT_BALANCED_RESTORE = _env_flag("ENABLE_J2_NEUTRAL_TIGHT_BALANCED_RESTORE", 0)
J2_NEUTRAL_TIGHT_BALANCED_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_TIGHT_BALANCED_RESTORE_DRAW_GAP_MAX", "0.020")
)
J2_NEUTRAL_TIGHT_BALANCED_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_NEUTRAL_TIGHT_BALANCED_RESTORE_FATIGUE_EDGE", "2.4")
)
J2_NEUTRAL_TIGHT_BALANCED_RESTORE_ELO_MIN = float(
    os.environ.get("J2_NEUTRAL_TIGHT_BALANCED_RESTORE_ELO_MIN", "20.0")
)
ENABLE_J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE = _env_flag("ENABLE_J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE", 0)
J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_DRAW_GAP_MAX", "0.105")
)
J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_FATIGUE_EDGE", "2.5")
)
J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_RANK_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_RANK_GAP_MAX", "-2.0")
)
J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_ELO_MAX = float(
    os.environ.get("J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_ELO_MAX", "-15.0")
)
ENABLE_J2_NEUTRAL_POS_PRESS_HOME_RESTORE = _env_flag("ENABLE_J2_NEUTRAL_POS_PRESS_HOME_RESTORE", 0)
J2_NEUTRAL_POS_PRESS_HOME_RESTORE_HOME_MIN = float(
    os.environ.get("J2_NEUTRAL_POS_PRESS_HOME_RESTORE_HOME_MIN", "0.340")
)
J2_NEUTRAL_POS_PRESS_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_POS_PRESS_HOME_RESTORE_DRAW_GAP_MAX", "0.060")
)
J2_NEUTRAL_POS_PRESS_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_NEUTRAL_POS_PRESS_HOME_RESTORE_FATIGUE_EDGE", "4.0")
)
J2_NEUTRAL_POS_PRESS_HOME_RESTORE_ELO_MIN = float(
    os.environ.get("J2_NEUTRAL_POS_PRESS_HOME_RESTORE_ELO_MIN", "30.0")
)
J2_NEUTRAL_POS_PRESS_HOME_RESTORE_MOTIVATION_MAX = float(
    os.environ.get("J2_NEUTRAL_POS_PRESS_HOME_RESTORE_MOTIVATION_MAX", "-4.0")
)
ENABLE_J2_CLOSE_DRAW_HOME_RESTORE = _env_flag("ENABLE_J2_CLOSE_DRAW_HOME_RESTORE", 0)
J2_CLOSE_DRAW_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_CLOSE_DRAW_HOME_RESTORE_DRAW_GAP_MAX", "0.105")
)
J2_CLOSE_DRAW_HOME_RESTORE_HOME_MAX = float(
    os.environ.get("J2_CLOSE_DRAW_HOME_RESTORE_HOME_MAX", "0.295")
)
J2_CLOSE_DRAW_HOME_RESTORE_ELO_MAX = float(
    os.environ.get("J2_CLOSE_DRAW_HOME_RESTORE_ELO_MAX", "-10.0")
)
J2_CLOSE_DRAW_HOME_RESTORE_MOTIVATION_MIN = float(
    os.environ.get("J2_CLOSE_DRAW_HOME_RESTORE_MOTIVATION_MIN", "3.0")
)
ENABLE_J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE = _env_flag("ENABLE_J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE", 0)
J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_HOME_MIN = float(
    os.environ.get("J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_HOME_MIN", "0.350")
)
J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_DRAW_GAP_MAX = float(
    os.environ.get("J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_DRAW_GAP_MAX", "0.042")
)
J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_FATIGUE_EDGE = float(
    os.environ.get("J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_FATIGUE_EDGE", "5.4")
)
J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_ELO_MIN = float(
    os.environ.get("J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_ELO_MIN", "38.0")
)
ENABLE_J2_AWAY_STRONG_DRAW_HOME_RESTORE = _env_flag("ENABLE_J2_AWAY_STRONG_DRAW_HOME_RESTORE", 0)
J2_AWAY_STRONG_DRAW_HOME_RESTORE_HOME_MAX = float(
    os.environ.get("J2_AWAY_STRONG_DRAW_HOME_RESTORE_HOME_MAX", "0.250")
)
J2_AWAY_STRONG_DRAW_HOME_RESTORE_DRAW_GAP_MIN = float(
    os.environ.get("J2_AWAY_STRONG_DRAW_HOME_RESTORE_DRAW_GAP_MIN", "0.130")
)
J2_AWAY_STRONG_DRAW_HOME_RESTORE_FATIGUE_EDGE_MAX = float(
    os.environ.get("J2_AWAY_STRONG_DRAW_HOME_RESTORE_FATIGUE_EDGE_MAX", "1.0")
)
J2_AWAY_STRONG_DRAW_HOME_RESTORE_MOTIVATION_MIN = float(
    os.environ.get("J2_AWAY_STRONG_DRAW_HOME_RESTORE_MOTIVATION_MIN", "4.5")
)
J2_AWAY_STRONG_DRAW_HOME_RESTORE_ELO_MAX = float(
    os.environ.get("J2_AWAY_STRONG_DRAW_HOME_RESTORE_ELO_MAX", "-45.0")
)
J2_NEG_HOME_ADV_AWAY_RESTORE_DRAW_MIN = float(os.environ.get("J2_NEG_HOME_ADV_AWAY_RESTORE_DRAW_MIN", "0.335"))
J2_NEG_HOME_ADV_AWAY_RESTORE_REQUIRE_DRAWRISK = _env_flag("J2_NEG_HOME_ADV_AWAY_RESTORE_REQUIRE_DRAWRISK", 1)
ENABLE_J2_AWAY_DRAW_RESTORE = _env_flag("ENABLE_J2_AWAY_DRAW_RESTORE", 0)
J2_AWAY_DRAW_RESTORE_DRAW_MIN = float(os.environ.get("J2_AWAY_DRAW_RESTORE_DRAW_MIN", "0.336"))
J2_AWAY_DRAW_RESTORE_AWAY_GAP_MAX = float(os.environ.get("J2_AWAY_DRAW_RESTORE_AWAY_GAP_MAX", "0.008"))
J2_AWAY_DRAW_RESTORE_REQUIRE_DRAWRISK = _env_flag("J2_AWAY_DRAW_RESTORE_REQUIRE_DRAWRISK", 1)
J2_AWAY_DRAW_RESTORE_REQUIRE_LAB = _env_flag("J2_AWAY_DRAW_RESTORE_REQUIRE_LAB", 1)
J2_AWAY_DRAW_RESTORE_LAB_EDGE_MAX = float(os.environ.get("J2_AWAY_DRAW_RESTORE_LAB_EDGE_MAX", "4.0"))
ENABLE_LAB_PROB_BLEND = _env_flag("ENABLE_LAB_PROB_BLEND", 1)
ENABLE_CONFIDENCE_PROB_FUSION = _env_flag("ENABLE_CONFIDENCE_PROB_FUSION", 1)
LAB_PROB_BLEND_ALPHA_BASE = float(os.environ.get("LAB_PROB_BLEND_ALPHA_BASE", "0.10"))
LAB_PROB_BLEND_ALPHA_GAIN = float(os.environ.get("LAB_PROB_BLEND_ALPHA_GAIN", "0.20"))
LAB_PROB_BLEND_ALPHA_MIN = float(os.environ.get("LAB_PROB_BLEND_ALPHA_MIN", "0.05"))
LAB_PROB_BLEND_ALPHA_MAX = float(os.environ.get("LAB_PROB_BLEND_ALPHA_MAX", "0.35"))
J1_TARGET_D_RANGE_RAW = os.environ.get("J1_TARGET_D_RANGE", "2,4")
J2_TARGET_D_RANGE_RAW = os.environ.get("J2_TARGET_D_RANGE", "1,3")
# Poisson格子の打ち切り誤差を抑えるための設定
POISSON_GRID_MIN_K = int(os.environ.get("POISSON_GRID_MIN_K", "10"))
POISSON_GRID_MAX_K = int(os.environ.get("POISSON_GRID_MAX_K", "20"))
POISSON_TAIL_EPS = float(os.environ.get("POISSON_TAIL_EPS", "1e-6"))
MISSING_WARN_THRESHOLD = float(os.environ.get("MISSING_WARN_THRESHOLD", "0.05"))
DEBUG_ELO_PROB = _env_flag("DEBUG_ELO_PROB", 0)
DEBUG_MATCH_ID = os.environ.get("DEBUG_MATCH_ID", "").strip()
J1_WIN_PROB_CAP = float(os.environ.get("J1_WIN_PROB_CAP", "0.68"))
PROB_FALLBACK = (0.397, 0.251, 0.353)
HFA_APPLY_COUNTER = {"applied": 0, "skipped": 0, "reason_counts": {}}
ENFORCE_ELO_SIGN_MONOTONIC = _env_flag("ENFORCE_ELO_SIGN_MONOTONIC", 1)
ELO_SIGN_FIX_COUNTER = {"total": 0, "neg_to_away": 0, "pos_to_home": 0}
MULTINOM_ELO_DIFF_SIGN = 1
MULTINOM_SWAP_HA_OUTPUT = False
HDA_MODEL_MODE = os.environ.get("HDA_MODEL_MODE", "multinom").strip().lower()
if HDA_MODEL_MODE not in {"legacy", "multinom"}:
    print(f"[CONFIG][WARN] invalid HDA_MODEL_MODE={HDA_MODEL_MODE!r}; fallback='legacy'")
    HDA_MODEL_MODE = "legacy"
HDA_FEATURE_PROFILE = os.environ.get("HDA_FEATURE_PROFILE", "").strip()
HDA_ENABLE_EXTENDED_FEATURES = os.environ.get("HDA_ENABLE_EXTENDED_FEATURES", "0").strip() == "1"
_default_hda_feature_profile_by_league = {
    "j1": "j1_draw_v3",
    "j2": "",
}
HDA_FEATURE_PROFILE_EFFECTIVE = HDA_FEATURE_PROFILE or _default_hda_feature_profile_by_league.get(LEAGUE, "")
_hda_model_default_profile = (
    os.path.join(DATA_DIR, "models", f"hda_multinom_train2025_{LEAGUE}__{HDA_FEATURE_PROFILE_EFFECTIVE}.joblib")
    if HDA_FEATURE_PROFILE_EFFECTIVE
    else ""
)
_hda_model_default_calibrated = os.path.join(DATA_DIR, "models", f"hda_multinom_calibrated_{LEAGUE}.joblib")
_hda_model_default_league = os.path.join(DATA_DIR, "models", f"hda_multinom_train2025_{LEAGUE}.joblib")
_hda_model_default_legacy = os.path.join(DATA_DIR, "models", "hda_multinom_train2025.joblib")
HDA_MODEL_PATH = os.environ.get("HDA_MODEL_PATH", "").strip()
HDA_MODEL_BUNDLE = None
HDA_MODEL_MODE_EFFECTIVE = HDA_MODEL_MODE


def _softmax_rows(logits):
    arr = np.asarray(logits, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    z = arr - np.max(arr, axis=1, keepdims=True)
    ez = np.exp(z)
    den = np.sum(ez, axis=1, keepdims=True)
    den = np.where(den <= 0, 1.0, den)
    return ez / den


def _load_hda_model_bundle(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    required = {"type", "classes", "feature_names"}
    if not isinstance(bundle, dict) or not required.issubset(bundle.keys()):
        raise RuntimeError(f"invalid model bundle keys: required={sorted(required)}")
    bundle_type = str(bundle.get("type"))
    if bundle_type not in {"softmax_linear", "constrained_softmax_1d"}:
        raise RuntimeError(f"unsupported model bundle type: {bundle.get('type')!r}")
    classes = [str(c).upper() for c in bundle["classes"]]
    if not {"H", "D", "A"}.issubset(classes):
        raise RuntimeError(f"classes must include H/D/A but got {classes}")
    bundle["classes"] = classes
    bundle["feature_names"] = [str(c) for c in bundle["feature_names"]]
    if "coef" in bundle:
        bundle["coef"] = np.asarray(bundle["coef"], dtype=float)
    if "intercept" in bundle:
        bundle["intercept"] = np.asarray(bundle["intercept"], dtype=float)
    if "feature_mean" in bundle:
        bundle["feature_mean"] = np.asarray(bundle["feature_mean"], dtype=float)
    if "feature_std" in bundle:
        bundle["feature_std"] = np.asarray(bundle["feature_std"], dtype=float)
    if bundle_type == "constrained_softmax_1d":
        cp = bundle.get("constrained_params", {})
        if not all(k in cp for k in ["k", "b0", "h", "bD"]):
            # 互換: coef/intercept から逆算
            if "coef" in bundle and "intercept" in bundle and bundle["coef"].shape[0] >= 3 and len(bundle["intercept"]) >= 3:
                k = float(bundle["coef"][0][0])
                b_h = float(bundle["intercept"][0])
                b_d = float(bundle["intercept"][1])
                b_a = float(bundle["intercept"][2])
                cp = {"k": k, "b0": (b_h + b_a) / 2.0, "h": (b_h - b_a) / 2.0, "bD": b_d}
            else:
                raise RuntimeError("constrained_softmax_1d requires constrained_params or compatible coef/intercept")
        bundle["constrained_params"] = {
            "k": float(cp["k"]),
            "b0": float(cp["b0"]),
            "h": float(cp["h"]),
            "bD": float(cp["bD"]),
        }
    return bundle


def _log_model_config():
    if HDA_MODEL_BUNDLE is None:
        print(
            f"[MODEL_CONFIG] league={LEAGUE} class_weight=unavailable "
            f"alpha=unavailable baseline_eps=unavailable"
        )
        return
    cw = HDA_MODEL_BUNDLE.get("class_weight", "unknown")
    alpha = HDA_MODEL_BUNDLE.get("class_weight_alpha", "unknown")
    baseline_eps = HDA_MODEL_BUNDLE.get("baseline_eps", "unknown")
    print(
        f"[MODEL_CONFIG] league={LEAGUE} class_weight={cw} "
        f"alpha={alpha} baseline_eps={baseline_eps}"
    )


def _init_hda_model():
    global HDA_MODEL_BUNDLE, HDA_MODEL_MODE_EFFECTIVE, MULTINOM_ELO_DIFF_SIGN, MULTINOM_SWAP_HA_OUTPUT
    if HDA_MODEL_MODE != "multinom":
        HDA_MODEL_MODE_EFFECTIVE = "legacy"
        HDA_MODEL_BUNDLE = None
        MULTINOM_ELO_DIFF_SIGN = 1
        MULTINOM_SWAP_HA_OUTPUT = False
        _log_model_config()
        return
    model_candidates = []
    if HDA_MODEL_PATH:
        model_candidates.append(HDA_MODEL_PATH)
    if _hda_model_default_profile and _hda_model_default_profile not in model_candidates:
        model_candidates.append(_hda_model_default_profile)
    if _hda_model_default_calibrated not in model_candidates:
        model_candidates.append(_hda_model_default_calibrated)
    if _hda_model_default_league not in model_candidates:
        model_candidates.append(_hda_model_default_league)
    if _hda_model_default_legacy not in model_candidates:
        model_candidates.append(_hda_model_default_legacy)
    model_path = next((p for p in model_candidates if p and os.path.exists(p)), "")
    if not model_path or not os.path.exists(model_path):
        print(f"[CONFIG][WARN] HDA_MODEL_MODE=multinom ですがモデル未検出のため legacy にフォールバック: {HDA_MODEL_PATH}")
        HDA_MODEL_MODE_EFFECTIVE = "legacy"
        HDA_MODEL_BUNDLE = None
        MULTINOM_ELO_DIFF_SIGN = 1
        MULTINOM_SWAP_HA_OUTPUT = False
        _log_model_config()
        return
    try:
        HDA_MODEL_BUNDLE = _load_hda_model_bundle(model_path)
        globals()["HDA_MODEL_PATH"] = model_path
        HDA_MODEL_MODE_EFFECTIVE = "multinom"
        _detect_and_apply_multinom_sign_correction()
        _log_model_config()
    except Exception as e:
        print(f"[CONFIG][WARN] multinomモデル読み込み失敗のため legacy にフォールバック: {e}")
        HDA_MODEL_MODE_EFFECTIVE = "legacy"
        HDA_MODEL_BUNDLE = None
        MULTINOM_ELO_DIFF_SIGN = 1
        MULTINOM_SWAP_HA_OUTPUT = False
        _log_model_config()


def _predict_hda_multinom_probs_raw(elo_diff_for_prob, feat_overrides=None):
    if HDA_MODEL_BUNDLE is None:
        raise RuntimeError("HDA_MODEL_BUNDLE is not loaded")
    if str(HDA_MODEL_BUNDLE.get("type")) == "constrained_softmax_1d":
        cp = HDA_MODEL_BUNDLE.get("constrained_params", {})
        d = float(elo_diff_for_prob)
        k = float(cp.get("k", 0.0))
        b0 = float(cp.get("b0", 0.0))
        h = float(cp.get("h", 0.0))
        bD = float(cp.get("bD", 0.0))
        logits = np.array([[k * d + b0 + h, bD, -k * d + b0 - h]], dtype=float)
        probs = _softmax_rows(logits)[0]
        return (float(probs[0]), float(probs[1]), float(probs[2])), {
            "elo_diff_for_prob": d,
            "abs_elo_diff_for_prob": abs(d),
            "d_scaled": abs(d),
            "abs_d_scaled": abs(d),
            "constrained_k": k,
            "constrained_b0": b0,
            "constrained_h": h,
            "constrained_bD": bD,
        }
    feat_values = {
        "elo_diff_for_prob": float(elo_diff_for_prob),
        "diff_raw_no_hfa": float(elo_diff_for_prob),
        "abs_elo_diff_for_prob": float(abs(elo_diff_for_prob)),
        "abs_diff_raw_no_hfa": float(abs(elo_diff_for_prob)),
        # multinomモードでは旧draw調整係数を使わず、生の差分由来のみを使う
        "d_scaled": float(abs(elo_diff_for_prob)),
        "abs_d_scaled": float(abs(elo_diff_for_prob)),
    }
    if feat_overrides:
        feat_values.update({k: v for k, v in feat_overrides.items() if v is not None and not pd.isna(v)})
    feature_names = HDA_MODEL_BUNDLE["feature_names"]
    x = np.array([feat_values.get(name, 0.0) for name in feature_names], dtype=float)
    mu = HDA_MODEL_BUNDLE["feature_mean"]
    sigma = HDA_MODEL_BUNDLE["feature_std"]
    sigma = np.where(np.abs(sigma) < 1e-12, 1.0, sigma)
    x_std = (x - mu) / sigma
    logits = x_std.dot(HDA_MODEL_BUNDLE["coef"].T) + HDA_MODEL_BUNDLE["intercept"]
    probs = _softmax_rows(logits)[0]
    cls_to_prob = {c: float(p) for c, p in zip(HDA_MODEL_BUNDLE["classes"], probs)}
    ph = float(cls_to_prob.get("H", 0.0))
    pdw = float(cls_to_prob.get("D", 0.0))
    pa = float(cls_to_prob.get("A", 0.0))
    arr = np.array([ph, pdw, pa], dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0:
        arr = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    else:
        arr = arr / s
    return (float(arr[0]), float(arr[1]), float(arr[2])), feat_values


def _detect_and_apply_multinom_sign_correction():
    global MULTINOM_ELO_DIFF_SIGN, MULTINOM_SWAP_HA_OUTPUT
    if HDA_MODEL_BUNDLE is None:
        MULTINOM_ELO_DIFF_SIGN = 1
        MULTINOM_SWAP_HA_OUTPUT = False
        return
    if str(HDA_MODEL_BUNDLE.get("type")) == "constrained_softmax_1d":
        MULTINOM_ELO_DIFF_SIGN = 1
        MULTINOM_SWAP_HA_OUTPUT = False
        print("[MULTINOM_SIGN_DETECT] disabled_for_constrained_model sign=1 swap_ha=0")
        return
    # 基本判定: diff増加でP(H)が増えるか
    (ph_pos, pd_pos, pa_pos), _ = _predict_hda_multinom_probs_raw(200.0)
    (ph_neg, pd_neg, pa_neg), _ = _predict_hda_multinom_probs_raw(-200.0)
    if ph_pos > ph_neg:
        MULTINOM_ELO_DIFF_SIGN = 1
    else:
        MULTINOM_ELO_DIFF_SIGN = -1
    # diff符号補正後に再判定
    (ph_pos2, pd_pos2, pa_pos2), _ = _predict_hda_multinom_probs_raw(200.0 * MULTINOM_ELO_DIFF_SIGN)
    (ph_neg2, pd_neg2, pa_neg2), _ = _predict_hda_multinom_probs_raw(-200.0 * MULTINOM_ELO_DIFF_SIGN)
    # それでもdiff>0でP(H)<P(A)なら、クラス定義逆の疑いとしてH/Aをswap
    detected_swap = bool(ph_pos2 < pa_pos2)
    trust_class_order = list(HDA_MODEL_BUNDLE.get("classes", [])) == ["H", "D", "A"]
    MULTINOM_SWAP_HA_OUTPUT = False if trust_class_order else detected_swap
    print(
        f"[MULTINOM_SIGN_DETECT] sign={MULTINOM_ELO_DIFF_SIGN} swap_ha={int(MULTINOM_SWAP_HA_OUTPUT)} "
        f"raw_pH(+200)={ph_pos:.4f} raw_pA(+200)={pa_pos:.4f} raw_pH(-200)={ph_neg:.4f} raw_pA(-200)={pa_neg:.4f} "
        f"corr_pH(+200)={ph_pos2:.4f} corr_pA(+200)={pa_pos2:.4f} corr_pH(-200)={ph_neg2:.4f} corr_pA(-200)={pa_neg2:.4f}"
    )
    if trust_class_order and detected_swap:
        print("[MULTINOM_SIGN_DETECT][INFO] classes=['H','D','A'] を信頼して swap_ha を無効化しました")
    # 追加ヘルスチェック: 低diff帯での符号逆転傾向を可視化
    test_diffs = np.array([-120.0, -80.0, -40.0, 40.0, 80.0, 120.0], dtype=float)
    vio = 0
    total = 0
    for d in test_diffs:
        (h, _dw, a), _ = _predict_hda_multinom_probs_raw(float(d) * float(MULTINOM_ELO_DIFF_SIGN))
        if MULTINOM_SWAP_HA_OUTPUT:
            h, a = a, h
        if d < 0 and h > a:
            vio += 1
        if d > 0 and h < a:
            vio += 1
        total += 1
    if vio > 0:
        print(f"[MULTINOM_SIGN_DETECT][WARN] residual_monotonic_violations={vio}/{total} on probe_diffs")


def _predict_hda_multinom_probs(elo_diff_for_prob, feat_overrides=None):
    signed_diff = float(elo_diff_for_prob) * float(MULTINOM_ELO_DIFF_SIGN)
    (ph, pdw, pa), feat_values = _predict_hda_multinom_probs_raw(signed_diff, feat_overrides=feat_overrides)
    if MULTINOM_SWAP_HA_OUTPUT:
        ph, pa = pa, ph
    feat_values = dict(feat_values)
    feat_values["elo_diff_input_raw"] = float(elo_diff_for_prob)
    feat_values["elo_diff_input_signed"] = float(signed_diff)
    feat_values["multinom_sign"] = int(MULTINOM_ELO_DIFF_SIGN)
    feat_values["multinom_swap_ha"] = bool(MULTINOM_SWAP_HA_OUTPUT)
    arr = np.array([ph, pdw, pa], dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0:
        arr = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    else:
        arr = arr / s
    return (float(arr[0]), float(arr[1]), float(arr[2])), feat_values


def _extend_multinom_feat_values(feat_values, row, home_advantage_diff, absence_effective, home_fatigue_score, away_fatigue_score):
    ext = dict(feat_values)
    if LEAGUE == "j1":
        scaled_home_advantage_diff = 0.0
    else:
        scaled_home_advantage_diff = float(_safe_float_value(home_advantage_diff, 0.0)) * float(HOME_ADV_FEATURE_SCALE)
    ext["home_advantage_diff"] = scaled_home_advantage_diff
    ext["abs_home_advantage_diff"] = abs(float(ext["home_advantage_diff"]))
    home_xg = pd.to_numeric(row.get("stats_ゴール期待値_home"), errors="coerce")
    away_xg = pd.to_numeric(row.get("stats_ゴール期待値_away"), errors="coerce")
    if pd.notna(home_xg) and pd.notna(away_xg):
        xg_diff = float(home_xg - away_xg)
        ext["xg_diff_abs"] = abs(xg_diff)
        ext["xg_diff"] = xg_diff
    home_xga = pd.to_numeric(row.get("stats_被ゴール期待値_home"), errors="coerce")
    away_xga = pd.to_numeric(row.get("stats_被ゴール期待値_away"), errors="coerce")
    if pd.notna(home_xga) and pd.notna(away_xga):
        xga_diff = float(home_xga - away_xga)
        ext["xga_diff_abs"] = abs(xga_diff)
        ext["xga_diff"] = xga_diff
    home_rank = pd.to_numeric(row.get("rankmot_rank_latest_home"), errors="coerce")
    away_rank = pd.to_numeric(row.get("rankmot_rank_latest_away"), errors="coerce")
    if pd.notna(home_rank) and pd.notna(away_rank):
        rank_gap = float(home_rank - away_rank)
        ext["rank_gap_abs"] = abs(rank_gap)
        ext["rank_gap"] = rank_gap
    if absence_effective is not None:
        abs_home = pd.to_numeric(absence_effective.get("absence_effective_total_home"), errors="coerce")
        abs_away = pd.to_numeric(absence_effective.get("absence_effective_total_away"), errors="coerce")
        if pd.notna(abs_home) and pd.notna(abs_away):
            absence_diff = float(abs_home - abs_away)
            ext["absence_effective_total_diff_abs"] = abs(absence_diff)
            ext["absence_effective_total_diff"] = absence_diff
    home_fatigue = pd.to_numeric(home_fatigue_score, errors="coerce")
    away_fatigue = pd.to_numeric(away_fatigue_score, errors="coerce")
    if pd.notna(home_fatigue) and pd.notna(away_fatigue):
        fatigue_diff = float(home_fatigue - away_fatigue)
        ext["fatigue_diff_abs"] = abs(fatigue_diff)
        ext["fatigue_diff"] = fatigue_diff
    home_gc = pd.to_numeric(row.get("stats_1試合平均失点数_home"), errors="coerce")
    away_gc = pd.to_numeric(row.get("stats_1試合平均失点数_away"), errors="coerce")
    if pd.notna(home_gc) and pd.notna(away_gc):
        goal_concede_diff = float(home_gc - away_gc)
        ext["goal_concede_diff_abs"] = abs(goal_concede_diff)
        ext["goal_concede_diff"] = goal_concede_diff
    return ext


def _bounded_unit(value, lower, upper, default=0.5):
    x = pd.to_numeric(value, errors="coerce")
    if pd.isna(x):
        return float(default)
    if upper <= lower:
        return float(default)
    return float(np.clip((float(x) - float(lower)) / (float(upper) - float(lower)), 0.0, 1.0))


def _avg_scores(values, default=0.5):
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not cleaned:
        return float(default)
    return float(sum(cleaned) / len(cleaned))


def build_team_state_vector(row, side, team_elo, absence_effective=None, fatigue_score=None):
    suffix = str(side).strip().lower()
    if suffix not in {"home", "away"}:
        raise ValueError(f"invalid side for team state: {side}")

    xg = _safe_float_value(row.get(f"stats_ゴール期待値_{suffix}"), np.nan)
    xga = _safe_float_value(row.get(f"stats_被ゴール期待値_{suffix}"), np.nan)
    goals = _safe_float_value(row.get(f"stats_1試合平均得点数_{suffix}"), np.nan)
    conceded = _safe_float_value(row.get(f"stats_1試合平均失点数_{suffix}"), np.nan)
    shots = _safe_float_value(row.get(f"stats_1試合平均シュート数_{suffix}"), np.nan)
    allowed_shots = _safe_float_value(row.get(f"stats_1試合平均被シュート数_{suffix}"), np.nan)
    on_target = _safe_float_value(row.get(f"stats_1試合平均枠内シュート数_{suffix}"), np.nan)
    possession = _safe_float_value(row.get(f"stats_平均ボール支配率_{suffix}"), np.nan)
    sprint = _safe_float_value(row.get(f"stats_1試合平均スプリント回数_{suffix}"), np.nan)
    rank_latest = _safe_float_value(row.get(f"rankmot_rank_latest_{suffix}"), np.nan)
    points_latest = _safe_float_value(row.get(f"rankmot_points_latest_{suffix}"), np.nan)
    rank_change_3w = _safe_float_value(row.get(f"rankmot_rank_change_3w_{suffix}"), np.nan)
    points_change_3w = _safe_float_value(row.get(f"rankmot_points_change_3w_{suffix}"), np.nan)
    rank_change_5w = _safe_float_value(row.get(f"rankmot_rank_change_5w_{suffix}"), np.nan)
    points_change_5w = _safe_float_value(row.get(f"rankmot_points_change_5w_{suffix}"), np.nan)
    motivation_3w = _safe_float_value(row.get(f"rankmot_motivation_score_3w_{suffix}"), np.nan)
    motivation_5w = _safe_float_value(row.get(f"rankmot_motivation_score_5w_{suffix}"), np.nan)
    management_motivation = _safe_float_value(row.get(f"management_motivation_score_{suffix}"), np.nan)
    stats_missing_ratio = _safe_float_value(row.get(f"stats_stats_missing_ratio_{suffix}"), np.nan)
    weather_tolerance_raw = _safe_float_value(row.get("weather_influence_score"), np.nan)
    dribble = _safe_float_value(row.get(f"stats_1試合平均ドリブル数_{suffix}"), np.nan)
    through_pass = _safe_float_value(row.get(f"stats_1試合平均スルーパス数_{suffix}"), np.nan)
    passes = _safe_float_value(row.get(f"stats_1試合平均パス数_{suffix}"), np.nan)
    clean_sheet = _safe_float_value(row.get(f"stats_クリーンシート総数_{suffix}"), np.nan)
    acl_fatigue = _safe_float_value(row.get(f"{suffix}_acl_fatigue"), np.nan)
    acl_days_since = _safe_float_value(row.get(f"{suffix}_acl_days_since"), np.nan)

    if absence_effective is None:
        absence_effective = compute_effective_absence_impacts(row)
    absence_total = _safe_float_value(absence_effective.get(f"absence_effective_total_{suffix}"), 0.0)
    absence_defense = _safe_float_value(absence_effective.get(f"absence_effective_defense_{suffix}"), 0.0)
    absence_attack = _safe_float_value(absence_effective.get(f"absence_effective_attack_{suffix}"), 0.0)
    fatigue_value = _safe_float_value(fatigue_score if fatigue_score is not None else row.get(f"{suffix}_total_fatigue_score"), np.nan)
    points_per_game_3w = points_change_3w / 3.0 if math.isfinite(points_change_3w) else np.nan
    points_per_game_5w = points_change_5w / 5.0 if math.isfinite(points_change_5w) else np.nan
    rank_per_game_3w = rank_change_3w / 3.0 if math.isfinite(rank_change_3w) else np.nan
    rank_per_game_5w = rank_change_5w / 5.0 if math.isfinite(rank_change_5w) else np.nan
    momentum_delta = (
        points_per_game_3w - points_per_game_5w
        if math.isfinite(points_per_game_3w) and math.isfinite(points_per_game_5w)
        else np.nan
    )
    rank_momentum_delta = (
        (-rank_per_game_3w) - (-rank_per_game_5w)
        if math.isfinite(rank_per_game_3w) and math.isfinite(rank_per_game_5w)
        else np.nan
    )
    motivation_trend_delta = (
        motivation_3w - motivation_5w
        if math.isfinite(motivation_3w) and math.isfinite(motivation_5w)
        else np.nan
    )

    strength_base = _avg_scores(
        [
            _bounded_unit(team_elo, 1300.0, 1700.0),
            _bounded_unit(points_latest, 8.0, 45.0),
            1.0 - _bounded_unit(rank_latest, 1.0, 20.0),
        ]
    )
    attack_level = _avg_scores(
        [
            _bounded_unit(xg, 0.6, 2.2),
            _bounded_unit(goals, 0.4, 2.2),
            _bounded_unit(shots, 6.0, 16.0),
            _bounded_unit(on_target, 2.0, 6.5),
            _bounded_unit(possession, 38.0, 62.0),
        ]
    )
    defense_level = _avg_scores(
        [
            1.0 - _bounded_unit(xga, 0.6, 2.1),
            1.0 - _bounded_unit(conceded, 0.4, 2.0),
            1.0 - _bounded_unit(allowed_shots, 6.0, 16.0),
        ]
    )
    form_level = _avg_scores(
        [
            _bounded_unit(points_change_3w, -6.0, 9.0),
            _bounded_unit(points_change_5w, -9.0, 15.0),
            _bounded_unit(-rank_change_3w, -5.0, 5.0),
            _bounded_unit(-rank_change_5w, -7.0, 7.0),
        ]
    )
    availability_level = 1.0 - _bounded_unit(absence_total, 0.0, 0.25)
    fatigue_level = 1.0 - _bounded_unit(fatigue_value, 0.0, 8.0)
    motivation_level = _avg_scores(
        [
            _bounded_unit(motivation_3w, -0.5, 1.5),
            _bounded_unit(motivation_5w, -0.5, 1.5),
            _bounded_unit(management_motivation, 0.0, 1.0),
        ]
    )
    stability_level = _avg_scores(
        [
            1.0 - _bounded_unit(abs(points_change_3w), 0.0, 9.0),
            1.0 - _bounded_unit(abs(points_change_5w), 0.0, 15.0),
            1.0 - _bounded_unit(abs(rank_change_3w), 0.0, 6.0),
            1.0 - _bounded_unit(abs(rank_change_5w), 0.0, 8.0),
            1.0 - _bounded_unit(stats_missing_ratio, 0.0, 0.45),
        ]
    )
    weather_tolerance = 1.0 - _bounded_unit(weather_tolerance_raw, 0.0, 1.0)
    tempo_level = _avg_scores(
        [
            _bounded_unit(shots, 6.0, 16.0),
            _bounded_unit(sprint, 110.0, 185.0),
            _bounded_unit(possession, 38.0, 62.0),
        ]
    )
    control_level = _avg_scores(
        [
            _bounded_unit(possession, 38.0, 62.0),
            _bounded_unit(passes, 250.0, 700.0),
            _bounded_unit(through_pass, 2.0, 12.0),
        ]
    )
    transition_level = _avg_scores(
        [
            _bounded_unit(sprint, 110.0, 185.0),
            _bounded_unit(dribble, 4.0, 18.0),
            _bounded_unit(through_pass, 2.0, 12.0),
        ]
    )
    momentum_level = _avg_scores(
        [
            _bounded_unit(momentum_delta, -1.8, 1.8),
            _bounded_unit(rank_momentum_delta, -1.2, 1.2),
            _bounded_unit(motivation_trend_delta, -0.6, 0.6),
        ]
    )
    attack_trend = _avg_scores(
        [
            _bounded_unit(points_per_game_3w, -1.0, 2.5),
            _bounded_unit(momentum_delta, -1.8, 1.8),
            _bounded_unit(motivation_trend_delta, -0.6, 0.6),
            attack_level,
        ]
    )
    defense_trend = _avg_scores(
        [
            1.0 - _bounded_unit(conceded, 0.4, 2.0),
            1.0 - _bounded_unit(xga, 0.6, 2.1),
            _bounded_unit(clean_sheet, 0.0, 12.0),
            1.0 - _bounded_unit(absence_defense, 0.0, 0.18),
        ]
    )
    fatigue_pressure = _avg_scores(
        [
            _bounded_unit(fatigue_value, 0.0, 8.0, default=0.0),
            _bounded_unit(acl_fatigue, 0.0, 5.0, default=0.0),
            1.0 - _bounded_unit(acl_days_since, 0.0, 10.0, default=1.0),
        ],
        default=0.0,
    )
    instability_level = _avg_scores(
        [
            1.0 - stability_level,
            _bounded_unit(abs(momentum_delta), 0.0, 1.5, default=0.0),
            _bounded_unit(abs(rank_momentum_delta), 0.0, 1.0, default=0.0),
            fatigue_pressure,
            _bounded_unit(absence_total + absence_attack + absence_defense, 0.0, 0.45, default=0.0),
        ],
        default=0.0,
    )
    trend_persistence = _avg_scores(
        [
            _bounded_unit(momentum_delta * motivation_trend_delta, -0.45, 0.45, default=0.5),
            _bounded_unit(momentum_delta * (-rank_momentum_delta), -1.2, 1.2, default=0.5),
            _bounded_unit(points_per_game_3w - points_per_game_5w, -1.2, 1.2, default=0.5),
            1.0 - _bounded_unit(abs(momentum_delta - motivation_trend_delta), 0.0, 1.2, default=0.5),
        ]
    )
    regression_pressure = _avg_scores(
        [
            _bounded_unit(abs(goals - xg), 0.0, 0.9, default=0.0),
            _bounded_unit(abs(conceded - xga), 0.0, 0.9, default=0.0),
            _bounded_unit(abs(on_target - (goals * 2.6 if math.isfinite(goals) else np.nan)), 0.0, 2.8, default=0.0),
            instability_level,
            _bounded_unit(stats_missing_ratio, 0.0, 0.45, default=0.0),
        ],
        default=0.0,
    )
    recovery_level = _avg_scores(
        [
            availability_level,
            1.0 - fatigue_pressure,
            defense_trend,
            1.0 - _bounded_unit(absence_defense, 0.0, 0.18, default=0.0),
            1.0 - _bounded_unit(acl_fatigue, 0.0, 5.0, default=0.0),
        ]
    )
    ceiling_level = _avg_scores(
        [
            strength_base,
            attack_level,
            attack_trend,
            motivation_level,
            control_level,
        ]
    )
    floor_level = _avg_scores(
        [
            defense_level,
            defense_trend,
            stability_level,
            recovery_level,
            1.0 - regression_pressure,
        ]
    )
    confidence_level = _avg_scores(
        [
            stability_level,
            trend_persistence,
            floor_level,
            1.0 - regression_pressure,
            1.0 - _bounded_unit(stats_missing_ratio, 0.0, 0.45, default=0.0),
        ]
    )

    return {
        "strength_base": strength_base,
        "attack_level": attack_level,
        "defense_level": defense_level,
        "form_level": form_level,
        "availability_level": availability_level,
        "fatigue_level": fatigue_level,
        "motivation_level": motivation_level,
        "stability_level": stability_level,
        "weather_tolerance": weather_tolerance,
        "tempo_level": tempo_level,
        "control_level": control_level,
        "transition_level": transition_level,
        "momentum_level": momentum_level,
        "attack_trend": attack_trend,
        "defense_trend": defense_trend,
        "fatigue_pressure": fatigue_pressure,
        "instability_level": instability_level,
        "trend_persistence": trend_persistence,
        "regression_pressure": regression_pressure,
        "recovery_level": recovery_level,
        "ceiling_level": ceiling_level,
        "floor_level": floor_level,
        "confidence_level": confidence_level,
    }


def simulate_lab_matchup(home_state, away_state, row):
    strength_delta = float(home_state["strength_base"] - away_state["strength_base"])
    attack_home_edge = float(home_state["attack_level"] - away_state["defense_level"])
    attack_away_edge = float(away_state["attack_level"] - home_state["defense_level"])
    form_delta = float(home_state["form_level"] - away_state["form_level"])
    availability_delta = float(home_state["availability_level"] - away_state["availability_level"])
    fatigue_delta = float(home_state["fatigue_level"] - away_state["fatigue_level"])
    motivation_delta = float(home_state["motivation_level"] - away_state["motivation_level"])
    momentum_delta = float(home_state["momentum_level"] - away_state["momentum_level"])
    control_interaction = float(home_state["control_level"] - away_state["control_level"])
    transition_interaction = float(home_state["transition_level"] - away_state["transition_level"])
    attack_trend_interaction = float(home_state["attack_trend"] - away_state["defense_trend"])
    away_attack_trend_interaction = float(away_state["attack_trend"] - home_state["defense_trend"])
    instability_delta = float(home_state["instability_level"] - away_state["instability_level"])
    fatigue_pressure_delta = float(home_state["fatigue_pressure"] - away_state["fatigue_pressure"])
    confidence_delta = float(home_state["confidence_level"] - away_state["confidence_level"])
    floor_delta = float(home_state["floor_level"] - away_state["floor_level"])
    ceiling_delta = float(home_state["ceiling_level"] - away_state["ceiling_level"])
    recovery_delta = float(home_state["recovery_level"] - away_state["recovery_level"])
    regression_delta = float(home_state["regression_pressure"] - away_state["regression_pressure"])
    persistence_delta = float(home_state["trend_persistence"] - away_state["trend_persistence"])
    control_delta = float(
        0.45 * strength_delta
        + 0.20 * form_delta
        + 0.15 * motivation_delta
        + 0.10 * availability_delta
        + 0.10 * fatigue_delta
    )
    pressure_delta = float(
        0.45 * (attack_home_edge - attack_away_edge)
        + 0.20 * (attack_trend_interaction - away_attack_trend_interaction)
        + 0.20 * transition_interaction
        + 0.15 * momentum_delta
    )

    flab_attack = _safe_float_value(row.get("flab_chance_shot_conversion_diff"), 0.0)
    flab_allow = _safe_float_value(row.get("flab_chance_allowed_shot_conversion_diff"), 0.0)
    flab_xgf = _safe_float_value(row.get("flab_expected_for_xg_diff"), 0.0)
    flab_xga = _safe_float_value(row.get("flab_expected_against_xg_diff"), 0.0)
    flab_build = _safe_float_value(row.get("flab_chance_build_rate_diff"), 0.0)
    flab_allow_build = _safe_float_value(row.get("flab_chance_allowed_build_rate_diff"), 0.0)
    flab_poss = _safe_float_value(row.get("flab_possession_rate_diff"), 0.0)
    flab_cbp = _safe_float_value(row.get("flab_possession_attack_cbp_diff"), 0.0)
    flab_style_short = _safe_float_value(row.get("flab_style_short_counter_index_diff"), 0.0)
    flab_style_long = _safe_float_value(row.get("flab_style_long_counter_index_diff"), 0.0)
    flab_style_enemy_poss = _safe_float_value(row.get("flab_style_enemy_possession_index_diff"), 0.0)
    flab_style_own_poss = _safe_float_value(row.get("flab_style_own_possession_index_diff"), 0.0)
    flab_style_left = _safe_float_value(row.get("flab_style_left_attack_index_diff"), 0.0)
    flab_style_center = _safe_float_value(row.get("flab_style_center_attack_index_diff"), 0.0)
    flab_style_right = _safe_float_value(row.get("flab_style_right_attack_index_diff"), 0.0)
    flab_style_short_sr = _safe_float_value(row.get("flab_style_short_counter_shot_rate_diff"), 0.0)
    flab_style_long_sr = _safe_float_value(row.get("flab_style_long_counter_shot_rate_diff"), 0.0)
    flab_style_enemy_sr = _safe_float_value(row.get("flab_style_enemy_possession_shot_rate_diff"), 0.0)
    flab_style_own_sr = _safe_float_value(row.get("flab_style_own_possession_shot_rate_diff"), 0.0)
    flab_style_left_sr = _safe_float_value(row.get("flab_style_left_attack_shot_rate_diff"), 0.0)
    flab_style_center_sr = _safe_float_value(row.get("flab_style_center_attack_shot_rate_diff"), 0.0)
    flab_style_right_sr = _safe_float_value(row.get("flab_style_right_attack_shot_rate_diff"), 0.0)
    flab_cbp_attack_pg = _safe_float_value(row.get("flab_cbp_attack_per_game_diff"), 0.0)
    flab_cbp_pass_pg = _safe_float_value(row.get("flab_cbp_pass_per_game_diff"), 0.0)
    flab_cbp_cross_pg = _safe_float_value(row.get("flab_cbp_cross_per_game_diff"), 0.0)
    flab_cbp_dribble_pg = _safe_float_value(row.get("flab_cbp_dribble_per_game_diff"), 0.0)
    flab_cbp_shot_pg = _safe_float_value(row.get("flab_cbp_shot_per_game_diff"), 0.0)
    flab_cbp_goal_pg = _safe_float_value(row.get("flab_cbp_goal_per_game_diff"), 0.0)
    flab_cbp_gain_pg = _safe_float_value(row.get("flab_cbp_gain_per_game_diff"), 0.0)
    flab_cbp_defense_pg = _safe_float_value(row.get("flab_cbp_defense_per_game_diff"), 0.0)
    flab_cbp_save_pg = _safe_float_value(row.get("flab_cbp_save_per_game_diff"), 0.0)
    flab_agi = _safe_float_value(row.get("flab_agi_score_diff"), 0.0)
    flab_kagi = _safe_float_value(row.get("flab_kagi_score_diff"), 0.0)

    style_direct_delta = float(
        0.40 * flab_style_short
        + 0.30 * flab_style_long
        + 0.30 * flab_style_enemy_poss
    )
    style_control_lab_delta = float(
        0.45 * flab_style_own_poss
        + 0.35 * flab_style_center
        + 0.20 * flab_cbp_pass_pg
    )
    style_width_delta = float(0.5 * flab_style_left + 0.5 * flab_style_right)
    style_finish_delta = float(
        (
            flab_style_short_sr
            + flab_style_long_sr
            + flab_style_enemy_sr
            + flab_style_own_sr
            + flab_style_left_sr
            + flab_style_center_sr
            + flab_style_right_sr
        )
        / 7.0
    )
    cbp_front_delta = float(
        0.35 * flab_cbp_attack_pg
        + 0.25 * flab_cbp_shot_pg
        + 0.25 * flab_cbp_goal_pg
        + 0.15 * flab_cbp_dribble_pg
    )
    cbp_supply_delta = float(
        0.45 * flab_cbp_pass_pg
        + 0.30 * flab_cbp_cross_pg
        + 0.25 * flab_cbp_gain_pg
    )
    cbp_resist_delta = float(
        0.70 * flab_cbp_defense_pg
        + 0.30 * flab_cbp_save_pg
    )
    agi_kagi_edge = float(0.55 * flab_agi + 0.45 * flab_kagi)

    matchup_edge_score = _clip01(
        _avg_scores(
            [
                _bounded_unit(abs(attack_home_edge - attack_away_edge), 0.0, 0.75, default=0.0),
                _bounded_unit(abs(control_interaction), 0.0, 0.40, default=0.0),
                _bounded_unit(abs(transition_interaction), 0.0, 0.45, default=0.0),
                _bounded_unit(abs(attack_trend_interaction) + abs(away_attack_trend_interaction), 0.0, 0.9, default=0.0),
                _bounded_unit(abs(flab_xgf) + abs(flab_xga), 0.0, 0.8, default=0.0),
                _bounded_unit(abs(style_direct_delta) + abs(style_control_lab_delta), 0.0, 35.0, default=0.0),
                _bounded_unit(abs(cbp_front_delta) + abs(cbp_supply_delta) + abs(cbp_resist_delta), 0.0, 8.0, default=0.0),
                _bounded_unit(abs(agi_kagi_edge), 0.0, 12.0, default=0.0),
            ],
            default=0.0,
        )
    )
    style_conflict_score = _clip01(
        _avg_scores(
            [
                (1.0 if flab_attack * flab_allow < 0.0 else 0.0),
                (1.0 if flab_xgf * flab_xga < 0.0 else 0.0),
                _bounded_unit(abs(flab_cbp), 0.0, 8.0, default=0.0),
                _bounded_unit(abs(transition_interaction - control_interaction), 0.0, 0.55, default=0.0),
                (1.0 if style_direct_delta * style_control_lab_delta < 0.0 else 0.0),
                _bounded_unit(abs(style_width_delta - style_control_lab_delta), 0.0, 20.0, default=0.0),
                _bounded_unit(abs(cbp_front_delta - cbp_resist_delta), 0.0, 4.0, default=0.0),
            ],
            default=0.0,
        )
    )
    low_event_score = _clip01(
        _avg_scores(
            [
                1.0 - _bounded_unit(abs(flab_build), 0.0, 8.0, default=0.5),
                1.0 - _bounded_unit(abs(flab_allow_build), 0.0, 8.0, default=0.5),
                1.0 - _bounded_unit(abs(flab_poss), 0.0, 10.0, default=0.5),
                1.0 - _bounded_unit(abs(attack_home_edge) + abs(attack_away_edge), 0.0, 0.9, default=0.5),
                1.0 - _bounded_unit(abs(transition_interaction), 0.0, 0.5, default=0.5),
                1.0 - _bounded_unit(abs(cbp_front_delta), 0.0, 4.0, default=0.5),
                1.0 - _bounded_unit(abs(style_finish_delta), 0.0, 10.0, default=0.5),
                1.0 - _bounded_unit(abs(agi_kagi_edge), 0.0, 10.0, default=0.5),
            ]
        )
    )
    balance_score = _clip01(
        _avg_scores(
            [
                1.0 - min(abs(strength_delta) / 0.45, 1.0),
                1.0 - min(abs(control_delta) / 0.40, 1.0),
                1.0 - min(abs(pressure_delta) / 0.50, 1.0),
                1.0 - min(abs(momentum_delta) / 0.40, 1.0),
            ]
        )
    )
    volatility_score = _clip01(
        _avg_scores(
            [
                style_conflict_score,
                _bounded_unit(abs(form_delta), 0.0, 0.45, default=0.0),
                _bounded_unit(abs(momentum_delta), 0.0, 0.45, default=0.0),
                _bounded_unit(abs(fatigue_pressure_delta), 0.0, 0.45, default=0.0),
                _bounded_unit(abs(instability_delta), 0.0, 0.45, default=0.0),
                home_state["instability_level"],
                away_state["instability_level"],
            ]
        )
    )
    draw_tension_score = _clip01(
        _avg_scores(
            [
                balance_score,
                low_event_score,
                1.0 - min(abs(attack_home_edge - attack_away_edge) / 0.45, 1.0),
                1.0 - min(abs(control_delta) / 0.35, 1.0),
                1.0 - min(abs(momentum_delta) / 0.35, 1.0),
            ]
        )
    )
    dynamic_swing_score = _clip01(
        _avg_scores(
            [
                _bounded_unit(abs(momentum_delta), 0.0, 0.40, default=0.0),
                _bounded_unit(abs(fatigue_pressure_delta), 0.0, 0.40, default=0.0),
                _bounded_unit(abs(attack_trend_interaction - away_attack_trend_interaction), 0.0, 0.70, default=0.0),
                volatility_score,
            ],
            default=0.0,
        )
    )
    hold_foundation_score = _clip01(
        _avg_scores(
            [
                _bounded_unit(max(abs(strength_delta), abs(control_delta)), 0.0, 0.55, default=0.0),
                matchup_edge_score,
                max(home_state["stability_level"], away_state["stability_level"]),
                max(home_state["defense_trend"], away_state["defense_trend"]),
                1.0 - low_event_score,
                1.0 - style_conflict_score,
                _bounded_unit(abs(style_direct_delta) + abs(cbp_front_delta), 0.0, 20.0, default=0.0),
                _bounded_unit(abs(agi_kagi_edge), 0.0, 12.0, default=0.0),
            ],
            default=0.0,
        )
    )
    stall_compactness_score = _clip01(
        _avg_scores(
            [
                low_event_score,
                draw_tension_score,
                balance_score,
                min(home_state["defense_level"], away_state["defense_level"]),
                min(home_state["defense_trend"], away_state["defense_trend"]),
                1.0 - _bounded_unit(abs(style_finish_delta), 0.0, 10.0, default=0.5),
                1.0 - _bounded_unit(abs(cbp_front_delta), 0.0, 4.0, default=0.5),
            ],
            default=0.0,
        )
    )
    flip_dislocation_score = _clip01(
        _avg_scores(
            [
                style_conflict_score,
                volatility_score,
                balance_score,
                _bounded_unit(abs(flab_attack) + abs(flab_allow), 0.0, 10.0, default=0.0),
                dynamic_swing_score,
                _bounded_unit(abs(style_direct_delta - style_control_lab_delta), 0.0, 20.0, default=0.0),
                _bounded_unit(abs(flab_agi - flab_kagi), 0.0, 12.0, default=0.0),
            ],
            default=0.0,
        )
    )
    home_path_score = _clip01(
        _avg_scores(
            [
                _bounded_unit(control_delta, -0.35, 0.35, default=0.5),
                _bounded_unit(pressure_delta, -0.45, 0.45, default=0.5),
                _bounded_unit(attack_home_edge - attack_away_edge, -0.60, 0.60, default=0.5),
                _bounded_unit(ceiling_delta, -0.35, 0.35, default=0.5),
                _bounded_unit(confidence_delta, -0.30, 0.30, default=0.5),
                _bounded_unit(style_direct_delta + 0.60 * style_control_lab_delta, -25.0, 25.0, default=0.5),
                _bounded_unit(cbp_front_delta + 0.55 * cbp_supply_delta + 0.35 * agi_kagi_edge, -6.0, 6.0, default=0.5),
            ],
            default=0.5,
        )
    )
    away_path_score = _clip01(
        _avg_scores(
            [
                1.0 - _bounded_unit(control_delta, -0.35, 0.35, default=0.5),
                1.0 - _bounded_unit(pressure_delta, -0.45, 0.45, default=0.5),
                1.0 - _bounded_unit(attack_home_edge - attack_away_edge, -0.60, 0.60, default=0.5),
                1.0 - _bounded_unit(ceiling_delta, -0.35, 0.35, default=0.5),
                1.0 - _bounded_unit(confidence_delta, -0.30, 0.30, default=0.5),
                1.0 - _bounded_unit(style_direct_delta + 0.60 * style_control_lab_delta, -25.0, 25.0, default=0.5),
                1.0 - _bounded_unit(cbp_front_delta + 0.55 * cbp_supply_delta + 0.35 * agi_kagi_edge, -6.0, 6.0, default=0.5),
            ],
            default=0.5,
        )
    )
    hold_raw = max(
        0.0,
        (
            0.45 * max(abs(strength_delta), abs(control_delta))
            + 0.20 * matchup_edge_score
            + 0.15 * max(home_state["stability_level"], away_state["stability_level"])
            + 0.10 * (1.0 - low_event_score)
            + 0.10 * (1.0 - style_conflict_score)
            + 0.10 * max(home_state["defense_trend"], away_state["defense_trend"])
            - 0.10 * dynamic_swing_score
        ),
    )
    stall_raw = max(
        0.0,
        (
            0.40 * low_event_score
            + 0.35 * draw_tension_score
            + 0.15 * balance_score
            + 0.10 * min(home_state["defense_level"], away_state["defense_level"])
            + 0.10 * min(home_state["defense_trend"], away_state["defense_trend"])
        ),
    )
    flip_raw = max(
        0.0,
        (
            0.40 * style_conflict_score
            + 0.30 * volatility_score
            + 0.20 * balance_score
            + 0.10 * _bounded_unit(abs(flab_attack) + abs(flab_allow), 0.0, 10.0, default=0.0)
            + 0.15 * dynamic_swing_score
        ),
    )
    scenario_entropy_score = _clip01(1.0 - max(hold_raw, stall_raw, flip_raw) / max(hold_raw + stall_raw + flip_raw, 1e-9))
    raw_sum = hold_raw + stall_raw + flip_raw
    if raw_sum <= 1e-9:
        hold_weight = stall_weight = flip_weight = 1.0 / 3.0
    else:
        hold_weight = hold_raw / raw_sum
        stall_weight = stall_raw / raw_sum
        flip_weight = flip_raw / raw_sum
    draw_path_score = _clip01(
        _avg_scores(
            [
                stall_compactness_score,
                draw_tension_score,
                scenario_entropy_score,
                1.0 - abs(home_path_score - away_path_score),
            ],
            default=0.5,
        )
    )
    # Keep the distribution builder compact: use three bases only.
    # 1) side path: which side has the cleaner route
    # 2) draw support: how naturally the match compresses toward D
    # 3) dispersion: how much the scenario is dislocated / split
    side_path_home = _clip01(
        _avg_scores(
            [
                home_path_score,
                0.55 * hold_foundation_score + 0.45 * max(home_path_score - away_path_score, 0.0),
                1.0 - 0.60 * draw_path_score,
            ],
            default=0.5,
        )
    )
    side_path_away = _clip01(
        _avg_scores(
            [
                away_path_score,
                0.55 * hold_foundation_score + 0.45 * max(away_path_score - home_path_score, 0.0),
                1.0 - 0.60 * draw_path_score,
            ],
            default=0.5,
        )
    )
    draw_support_score = _clip01(
        _avg_scores(
            [
                draw_path_score,
                stall_compactness_score,
                scenario_entropy_score,
            ],
            default=0.5,
        )
    )
    side_gap_score = _clip01(abs(side_path_home - side_path_away))
    home_side_edge = _clip01(0.5 + 0.5 * (side_path_home - side_path_away))
    away_side_edge = _clip01(0.5 + 0.5 * (side_path_away - side_path_home))
    dispersion_score = _clip01(
        _avg_scores(
            [
                flip_dislocation_score,
                scenario_entropy_score,
            ],
            default=0.5,
        )
    )

    league_upper = str(row.get("league", "")).upper()
    if league_upper == "J2":
        dispersion_score = _clip01(
            _avg_scores(
                [
                    dispersion_score,
                    dynamic_swing_score,
                    _bounded_unit(abs(control_delta - pressure_delta), 0.0, 0.65, default=0.0),
                ],
                default=dispersion_score,
            )
        )
    if league_upper == "J2":
        mix_home_raw = max(
            1e-9,
            0.62 * hold_weight * side_path_home
            + 0.24 * flip_weight * side_path_home
            + 0.08 * side_gap_score * home_side_edge
            + 0.14 * (1.0 - draw_support_score)
        )
        mix_draw_raw = max(
            1e-9,
            0.68 * stall_weight * draw_support_score
            + 0.12 * scenario_entropy_score
            + 0.06 * (1.0 - side_gap_score)
            + 0.08 * dispersion_score
        )
        mix_away_raw = max(
            1e-9,
            0.62 * hold_weight * side_path_away
            + 0.24 * flip_weight * side_path_away
            + 0.08 * side_gap_score * away_side_edge
            + 0.14 * (1.0 - draw_support_score)
        )
    else:
        mix_home_raw = max(
            1e-9,
            0.58 * hold_weight * side_path_home
            + 0.18 * flip_weight * side_path_home
            + 0.10 * side_gap_score * home_side_edge
            + 0.10 * (1.0 - dispersion_score)
        )
        mix_draw_raw = max(
            1e-9,
            0.74 * stall_weight * draw_support_score
            + 0.12 * scenario_entropy_score
            + 0.06 * (1.0 - side_gap_score)
            + 0.08 * dispersion_score
        )
        mix_away_raw = max(
            1e-9,
            0.58 * hold_weight * side_path_away
            + 0.18 * flip_weight * side_path_away
            + 0.10 * side_gap_score * away_side_edge
            + 0.10 * (1.0 - dispersion_score)
        )
    mix_sum = mix_home_raw + mix_draw_raw + mix_away_raw
    mix_home = float(mix_home_raw / mix_sum)
    mix_draw = float(mix_draw_raw / mix_sum)
    mix_away = float(mix_away_raw / mix_sum)

    side_path_score = float(max(side_path_home, side_path_away))
    if (
        draw_support_score >= 0.60
        and dispersion_score <= 0.34
        and max(mix_home, mix_away) <= 0.28
    ):
        basis_hint = "flat_draw_trap"
    elif side_path_score >= 0.68 and draw_support_score <= 0.48 and dispersion_score <= 0.42:
        basis_hint = "side_strong"
    elif draw_support_score >= 0.56 and dispersion_score <= 0.40 and side_path_score <= 0.58:
        basis_hint = "draw_compressed"
    elif dispersion_score >= 0.42 and abs(side_path_home - side_path_away) <= 0.10:
        basis_hint = "split_side"
    else:
        basis_hint = "basis_balanced"

    avg_tempo = (float(home_state["tempo_level"]) + float(away_state["tempo_level"])) / 2.0
    if avg_tempo <= 0.42:
        tempo_band = "low"
    elif avg_tempo >= 0.62:
        tempo_band = "high"
    else:
        tempo_band = "mid"
    if control_delta >= 0.10:
        control_band = "home"
    elif control_delta <= -0.10:
        control_band = "away"
    else:
        control_band = "neutral"
    if pressure_delta >= 0.10:
        pressure_band = "home"
    elif pressure_delta <= -0.10:
        pressure_band = "away"
    else:
        pressure_band = "neutral"

    matchup_profile = "balanced"
    if (
        tempo_band == "low"
        and draw_tension_score >= 0.74
        and low_event_score >= 0.66
        and abs(control_delta) <= 0.08
        and abs(pressure_delta) <= 0.12
        and volatility_score <= 0.34
        and flip_weight <= 0.30
    ):
        matchup_profile = "low_tempo_draw"
    elif (
        flip_weight >= 0.38
        and volatility_score >= 0.40
        and dynamic_swing_score >= 0.34
        and (
            (control_band == "home" and pressure_band == "away")
            or (control_band == "away" and pressure_band == "home")
        )
    ):
        matchup_profile = "counterflow"
    elif hold_weight >= max(stall_weight, flip_weight) and hold_weight >= 0.42:
        matchup_profile = "hold_shape"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and str(row.get("league", "")).upper() == "J2"
        and mix_draw >= 0.53
        and flip_weight <= 0.22
        and dispersion_score <= 0.38
        and draw_tension_score >= 0.82
        and max(mix_home, mix_away) <= 0.26
    ):
        matchup_profile = "j2_flat_draw_trap"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and str(row.get("league", "")).upper() == "J2"
        and _safe_float_value(row.get("prob_draw"), 0.0)
        >= max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        )
        and draw_tension_score >= 0.78
        and flip_weight <= 0.24
        and dynamic_swing_score <= 0.32
        and pressure_band == "neutral"
        and control_band != "away"
    ):
        matchup_profile = "j2_draw_stall_anchor"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and str(row.get("league", "")).upper() == "J2"
        and _safe_float_value(row.get("prob_home_win"), 0.0)
        > max(
            _safe_float_value(row.get("prob_draw"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        )
        and _safe_float_value(row.get("prob_home_win"), 0.0) >= 0.39
        and (
            _safe_float_value(row.get("prob_home_win"), 0.0)
            - _safe_float_value(row.get("prob_draw"), 0.0)
        ) >= 0.04
        and draw_tension_score >= 0.78
        and flip_weight <= 0.22
        and pressure_band == "neutral"
    ):
        matchup_profile = "j2_home_stall_preserve"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and str(row.get("league", "")).upper() == "J2"
        and tempo_band == "mid"
        and control_band == "home"
        and pressure_band == "away"
        and _safe_float_value(row.get("prob_home_win"), 0.0) >= 0.35
        and (
            _safe_float_value(row.get("prob_home_win"), 0.0)
            - _safe_float_value(row.get("prob_draw"), 0.0)
        ) >= -0.01
        and flip_weight <= 0.22
        and draw_tension_score >= 0.60
    ):
        matchup_profile = "j2_home_pressure_stall_preserve"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and str(row.get("league", "")).upper() == "J2"
        and tempo_band == "mid"
        and control_band == "home"
        and pressure_band == "neutral"
        and dynamic_swing_score <= 0.35
        and draw_tension_score >= 0.68
        and flip_weight <= 0.30
    ):
        matchup_profile = "j2_low_dyn_home_stall_preserve"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and str(row.get("league", "")).upper() == "J2"
        and tempo_band == "mid"
        and control_band == "neutral"
        and pressure_band == "neutral"
        and flip_weight >= 0.36
        and dynamic_swing_score <= 0.22
        and draw_tension_score >= 0.84
    ):
        matchup_profile = "j2_neutral_flip_draw_watch"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and _safe_float_value(row.get("prob_draw"), 0.0)
        >= max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        )
        and draw_tension_score >= 0.64
        and (
            (
                tempo_band == "low"
                and low_event_score >= 0.78
                and abs(control_delta) <= 0.10
                and abs(pressure_delta) <= 0.10
            )
            or (
                low_event_score >= 0.68
                and
                style_conflict_score >= 0.50
                and flip_weight >= 0.34
            )
        )
    ):
        matchup_profile = "draw_anchor"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and tempo_band == "mid"
        and max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        )
        > _safe_float_value(row.get("prob_draw"), 0.0)
        and (
            max(
                _safe_float_value(row.get("prob_home_win"), 0.0),
                _safe_float_value(row.get("prob_away_win"), 0.0),
            )
            - _safe_float_value(row.get("prob_draw"), 0.0)
        ) <= 0.05
        and low_event_score >= 0.68
        and draw_tension_score >= 0.63
        and flip_weight <= 0.24
        and volatility_score <= 0.30
    ):
        matchup_profile = "mid_stall_side_preserve"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        )
        > _safe_float_value(row.get("prob_draw"), 0.0)
        and (
            (
                tempo_band == "low"
                and max(
                    _safe_float_value(row.get("prob_home_win"), 0.0),
                    _safe_float_value(row.get("prob_away_win"), 0.0),
                ) >= 0.39
                and (
                    max(
                        _safe_float_value(row.get("prob_home_win"), 0.0),
                        _safe_float_value(row.get("prob_away_win"), 0.0),
                    )
                    - _safe_float_value(row.get("prob_draw"), 0.0)
                ) >= 0.02
            )
            or (
                control_band == "neutral"
                and pressure_band == "neutral"
                and style_conflict_score <= 0.05
                and volatility_score <= 0.18
                and max(
                    _safe_float_value(row.get("prob_home_win"), 0.0),
                    _safe_float_value(row.get("prob_away_win"), 0.0),
                ) >= 0.38
                and (
                    max(
                        _safe_float_value(row.get("prob_home_win"), 0.0),
                        _safe_float_value(row.get("prob_away_win"), 0.0),
                    )
                    - _safe_float_value(row.get("prob_draw"), 0.0)
                ) >= 0.02
            )
            or (
                max(
                    _safe_float_value(row.get("prob_home_win"), 0.0),
                    _safe_float_value(row.get("prob_away_win"), 0.0),
                ) >= 0.50
            )
        )
    ):
        matchup_profile = "stall_side_preserve"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        ) >= 0.66
        and volatility_score <= 0.16
        and style_conflict_score <= 0.10
    ):
        matchup_profile = "directional_stall"
    elif (
        stall_weight >= max(hold_weight, flip_weight)
        and max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        ) >= 0.40
        and max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        ) > _safe_float_value(row.get("prob_draw"), 0.0)
        and (
            max(
                _safe_float_value(row.get("prob_home_win"), 0.0),
                _safe_float_value(row.get("prob_away_win"), 0.0),
            )
            - _safe_float_value(row.get("prob_draw"), 0.0)
        ) >= 0.06
        and low_event_score >= 0.74
        and volatility_score <= 0.22
        and draw_tension_score <= 0.66
    ):
        matchup_profile = "stall_side_preserve"
    elif (
        str(row.get("league", "")).upper() == "J1"
        and stall_weight >= max(hold_weight, flip_weight)
        and draw_tension_score >= 0.54
        and draw_tension_score <= 0.62
        and floor_delta >= 0.12
        and recovery_delta >= 0.16
        and persistence_delta <= -0.10
        and dynamic_swing_score >= 0.40
    ):
        matchup_profile = "j1_floor_recovery_watch"
    elif (
        control_band == "home"
        and pressure_band == "home"
        and max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        ) >= 0.33
        and max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        )
        > _safe_float_value(row.get("prob_draw"), 0.0)
        and flip_weight >= 0.20
        and dynamic_swing_score >= 0.40
        and draw_tension_score <= 0.50
    ):
        matchup_profile = "home_control_watch"
    elif (
        flip_weight >= 0.32
        and dynamic_swing_score >= 0.30
        and volatility_score <= 0.30
        and (
            control_band == "neutral"
            or pressure_band == "neutral"
        )
    ):
        matchup_profile = "swing_watch"
    elif (
        str(row.get("league", "")).upper() == "J1"
        and stall_weight >= max(hold_weight, flip_weight)
        and _safe_float_value(row.get("prob_draw"), 0.0)
        >= max(
            _safe_float_value(row.get("prob_home_win"), 0.0),
            _safe_float_value(row.get("prob_away_win"), 0.0),
        )
        and dynamic_swing_score >= 0.46
        and pressure_band == "home"
        and tempo_band == "mid"
    ):
        matchup_profile = "j1_pressure_draw_watch"
    elif (
        str(row.get("league", "")).upper() == "J2"
        and flip_weight >= max(hold_weight, stall_weight)
        and flip_weight >= 0.36
        and control_band == "home"
        and pressure_band == "away"
        and _safe_float_value(row.get("prob_home_win"), 0.0)
        > _safe_float_value(row.get("prob_draw"), 0.0)
        and dynamic_swing_score >= 0.26
    ):
        matchup_profile = "j2_flip_conflict_watch"
    elif (
        str(row.get("league", "")).upper() == "J2"
        and stall_weight >= max(hold_weight, flip_weight)
        and draw_tension_score >= 0.82
        and flip_weight <= 0.22
        and volatility_score <= 0.18
        and dynamic_swing_score >= 0.26
        and pressure_band == "neutral"
    ):
        matchup_profile = "j2_tense_stall_watch"
    elif stall_weight >= max(hold_weight, flip_weight) and draw_tension_score >= 0.62:
        matchup_profile = "stall_shape"

    notes = []
    notes.append(f"profile={matchup_profile}")
    if stall_weight >= max(hold_weight, flip_weight):
        notes.append("stall")
    if flip_weight >= 0.34:
        notes.append("flip")
    if hold_weight >= 0.42:
        notes.append("hold")
    if draw_tension_score >= 0.62:
        notes.append("draw_tension_high")
    if volatility_score >= 0.62:
        notes.append("volatility_high")
    if abs(style_direct_delta) >= 10.0:
        notes.append("style_direct_edge")
    if abs(cbp_front_delta) >= 1.0:
        notes.append("cbp_front_edge")
    if abs(agi_kagi_edge) >= 3.0:
        notes.append("agi_kagi_edge")
    if control_band != "neutral":
        notes.append(f"{control_band}_control")
    if dynamic_swing_score >= 0.60:
        notes.append("dynamic_swing")
    if basis_hint != "basis_balanced":
        notes.append(f"basis={basis_hint}")

    return {
        "lab_version": "state_lab_v1",
        "hold_weight": float(hold_weight),
        "stall_weight": float(stall_weight),
        "flip_weight": float(flip_weight),
        "volatility_score": float(volatility_score),
        "draw_tension_score": float(draw_tension_score),
        "tempo_band": tempo_band,
        "control_band": control_band,
        "pressure_band": pressure_band,
        "matchup_profile": matchup_profile,
        "matchup_edge_score": float(matchup_edge_score),
        "style_conflict_score": float(style_conflict_score),
        "low_event_score": float(low_event_score),
        "dynamic_swing_score": float(dynamic_swing_score),
        "hold_foundation_score": float(hold_foundation_score),
        "stall_compactness_score": float(stall_compactness_score),
        "flip_dislocation_score": float(flip_dislocation_score),
        "side_path_home_score": float(side_path_home),
        "side_path_away_score": float(side_path_away),
        "draw_support_score": float(draw_support_score),
        "dispersion_score": float(dispersion_score),
        "basis_hint": basis_hint,
        "style_direct_delta": float(style_direct_delta),
        "style_control_delta": float(style_control_lab_delta),
        "style_width_delta": float(style_width_delta),
        "cbp_front_delta": float(cbp_front_delta),
        "cbp_supply_delta": float(cbp_supply_delta),
        "cbp_resist_delta": float(cbp_resist_delta),
        "agi_kagi_edge": float(agi_kagi_edge),
        "home_path_score": float(home_path_score),
        "away_path_score": float(away_path_score),
        "scenario_entropy_score": float(scenario_entropy_score),
        "draw_path_score": float(draw_path_score),
        "mix_home": float(mix_home),
        "mix_draw": float(mix_draw),
        "mix_away": float(mix_away),
        "confidence_delta": float(confidence_delta),
        "floor_delta": float(floor_delta),
        "ceiling_delta": float(ceiling_delta),
        "recovery_delta": float(recovery_delta),
        "regression_delta": float(regression_delta),
        "persistence_delta": float(persistence_delta),
        "notes": ",".join(notes) if notes else "balanced",
    }


def add_team_state_and_lab_context(df):
    if df is None or df.empty:
        return df
    rows = []
    for _, row in df.iterrows():
        absence_effective = compute_effective_absence_impacts(row)
        home_state = build_team_state_vector(
            row,
            "home",
            team_elo=_safe_float_value(row.get("home_elo"), 1500.0),
            absence_effective=absence_effective,
            fatigue_score=row.get("home_total_fatigue_score", row.get("home_fatigue_score")),
        )
        away_state = build_team_state_vector(
            row,
            "away",
            team_elo=_safe_float_value(row.get("away_elo"), 1500.0),
            absence_effective=absence_effective,
            fatigue_score=row.get("away_total_fatigue_score", row.get("away_fatigue_score")),
        )
        matchup = simulate_lab_matchup(home_state, away_state, row)
        payload = {}
        for key, value in home_state.items():
            payload[f"state_{key}_home"] = value
        for key, value in away_state.items():
            payload[f"state_{key}_away"] = value
        payload.update(
            {
                "lab_model_version": matchup["lab_version"],
                "lab_hold_weight": matchup["hold_weight"],
                "lab_stall_weight": matchup["stall_weight"],
                "lab_flip_weight": matchup["flip_weight"],
                "lab_volatility_score": matchup["volatility_score"],
                "lab_draw_tension_score": matchup["draw_tension_score"],
                "lab_tempo_band": matchup["tempo_band"],
                "lab_control_band": matchup["control_band"],
                "lab_pressure_band": matchup["pressure_band"],
                "lab_matchup_profile": matchup["matchup_profile"],
                "lab_matchup_edge_score": matchup["matchup_edge_score"],
                "lab_style_conflict_score": matchup["style_conflict_score"],
                "lab_low_event_score": matchup["low_event_score"],
                "lab_dynamic_swing_score": matchup["dynamic_swing_score"],
                "lab_hold_foundation_score": matchup["hold_foundation_score"],
                "lab_stall_compactness_score": matchup["stall_compactness_score"],
                "lab_flip_dislocation_score": matchup["flip_dislocation_score"],
                "lab_side_path_home_score": matchup["side_path_home_score"],
                "lab_side_path_away_score": matchup["side_path_away_score"],
                "lab_draw_support_score": matchup["draw_support_score"],
                "lab_dispersion_score": matchup["dispersion_score"],
                "lab_basis_hint": matchup["basis_hint"],
                "lab_style_direct_delta": matchup["style_direct_delta"],
                "lab_style_control_delta": matchup["style_control_delta"],
                "lab_style_width_delta": matchup["style_width_delta"],
                "lab_cbp_front_delta": matchup["cbp_front_delta"],
                "lab_cbp_supply_delta": matchup["cbp_supply_delta"],
                "lab_cbp_resist_delta": matchup["cbp_resist_delta"],
                "lab_agi_kagi_edge": matchup["agi_kagi_edge"],
                "lab_home_path_score": matchup["home_path_score"],
                "lab_away_path_score": matchup["away_path_score"],
                "lab_scenario_entropy_score": matchup["scenario_entropy_score"],
                "lab_draw_path_score": matchup["draw_path_score"],
                "lab_mix_home": matchup["mix_home"],
                "lab_mix_draw": matchup["mix_draw"],
                "lab_mix_away": matchup["mix_away"],
                "lab_confidence_delta": matchup["confidence_delta"],
                "lab_floor_delta": matchup["floor_delta"],
                "lab_ceiling_delta": matchup["ceiling_delta"],
                "lab_recovery_delta": matchup["recovery_delta"],
                "lab_regression_delta": matchup["regression_delta"],
                "lab_persistence_delta": matchup["persistence_delta"],
                "lab_state_notes": matchup["notes"],
            }
        )
        rows.append(payload)
    payload_df = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in payload_df.columns:
        out[col] = payload_df[col]
    return out


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _short_col_list(cols, head=50, tail=20):
    cols = list(cols)
    if len(cols) <= head + tail:
        return cols
    return cols[:head] + ["..."] + cols[-tail:]


def _update_hfa_apply_counter(reason, applied):
    if applied:
        HFA_APPLY_COUNTER["applied"] += 1
    else:
        HFA_APPLY_COUNTER["skipped"] += 1
    rc = HFA_APPLY_COUNTER["reason_counts"]
    rc[reason] = int(rc.get(reason, 0)) + 1


def _detect_probability_columns(df_a, df_b):
    prob_candidates = [
        ("prob_home", "prob_draw", "prob_away"),
        ("prob_home_win", "prob_draw", "prob_away_win"),
        ("p_home", "p_draw", "p_away"),
    ]
    for c_home, c_draw, c_away in prob_candidates:
        if {c_home, c_draw, c_away}.issubset(df_a.columns) and {c_home, c_draw, c_away}.issubset(df_b.columns):
            return [c_home, c_draw, c_away]
    cols_a = _short_col_list(df_a.columns)
    cols_b = _short_col_list(df_b.columns)
    print(
        "[ERROR] probability columns not found. "
        f"available_columns_a={cols_a} available_columns_b={cols_b}"
    )
    raise RuntimeError("probability columns not found")


def _build_hfa_aligned_dataframe(df_a, df_b, value_cols):
    shared_cols = [c for c in value_cols if c in df_a.columns and c in df_b.columns]
    if not shared_cols:
        return pd.DataFrame(), [], "none"
    if "match_id" in df_a.columns and "match_id" in df_b.columns:
        merged = df_a[["match_id"] + shared_cols].merge(
            df_b[["match_id"] + shared_cols], on="match_id", how="inner", suffixes=("_a", "_b")
        )
        return merged, shared_cols, "match_id"
    key_cols = ["datetime", "home_team", "away_team"]
    if set(key_cols).issubset(df_a.columns) and set(key_cols).issubset(df_b.columns):
        merged = df_a[key_cols + shared_cols].merge(
            df_b[key_cols + shared_cols], on=key_cols, how="inner", suffixes=("_a", "_b")
        )
        return merged, shared_cols, "datetime+home+away"
    min_len = min(len(df_a), len(df_b))
    data = {}
    for col in shared_cols:
        data[f"{col}_a"] = df_a[col].head(min_len)
        data[f"{col}_b"] = df_b[col].head(min_len)
    return pd.DataFrame(data), shared_cols, "row_index"


def _compute_max_abs_diff(merged, col_name):
    left = f"{col_name}_a"
    right = f"{col_name}_b"
    if left not in merged.columns or right not in merged.columns:
        return None
    d = (pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(merged[right], errors="coerce")).abs()
    return float(d.max(skipna=True))


def _compute_diff_stats(merged, col_name):
    left = f"{col_name}_a"
    right = f"{col_name}_b"
    if left not in merged.columns or right not in merged.columns:
        return None, None
    d = (pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(merged[right], errors="coerce")).abs()
    max_abs = float(d.max(skipna=True))
    num_diff = int((d > 1e-12).sum())
    return max_abs, num_diff


def _log_hfa_intermediate_trace(df_a, df_b):
    compare_cols = [
        "elo_diff_for_prob",
        "elo_diff_scaled",
        "elo_diff",
        "elo_diff_raw",
        "d_scaled",
        "elo_diff_before_hfa",
        "elo_diff_after_hfa",
    ]
    merged, shared_cols, key_name = _build_hfa_aligned_dataframe(df_a, df_b, compare_cols)
    if merged.empty:
        print("[HFA_TRACE] key=none max_abs_diff=NA num_rows_with_any_diff=0")
        return

    primary = next(
        (c for c in ["elo_diff_for_prob", "elo_diff_scaled", "elo_diff", "elo_diff_raw"] if c in shared_cols),
        None,
    )
    if primary:
        max_abs, num_diff = _compute_diff_stats(merged, primary)
        print(
            f"[HFA_TRACE] key={primary} align={key_name} "
            f"max_abs_diff={max_abs:.6f} num_rows_with_any_diff={num_diff}"
        )
    else:
        print(f"[HFA_TRACE] key=none align={key_name} max_abs_diff=NA num_rows_with_any_diff=0")

    if "d_scaled" in shared_cols:
        max_abs, num_diff = _compute_diff_stats(merged, "d_scaled")
        print(
            f"[HFA_TRACE] key=d_scaled align={key_name} "
            f"max_abs_diff={max_abs:.6f} num_rows_with_any_diff={num_diff}"
        )

    if "elo_diff_before_hfa" in shared_cols and "elo_diff_after_hfa" in shared_cols:
        added_a = (
            pd.to_numeric(merged["elo_diff_after_hfa_a"], errors="coerce")
            - pd.to_numeric(merged["elo_diff_before_hfa_a"], errors="coerce")
        )
        added_b = (
            pd.to_numeric(merged["elo_diff_after_hfa_b"], errors="coerce")
            - pd.to_numeric(merged["elo_diff_before_hfa_b"], errors="coerce")
        )
        d_added = (added_a - added_b).abs()
        max_abs_added = float(d_added.max(skipna=True))
        max_abs_added_on = float(added_a.abs().max(skipna=True))
        max_abs_added_off = float(added_b.abs().max(skipna=True))
        num_added_diff = int((d_added > 1e-12).sum())
        print(
            f"[HFA_TRACE] max_abs_hfa_added={max_abs_added:.6f} "
            f"num_rows_with_any_diff={num_added_diff}"
        )
        print(
            f"[HFA_TRACE] max_abs_hfa_added_on={max_abs_added_on:.6f} "
            f"max_abs_hfa_added_off={max_abs_added_off:.6f}"
        )
        print(
            "[HFA_TRACE] formula=max_abs_hfa_added=max(abs("
            "(elo_diff_after_hfa-elo_diff_before_hfa)_ON - "
            "(elo_diff_after_hfa-elo_diff_before_hfa)_OFF))"
        )

    # Representative per-row trace: sort by |diff(elo_diff_for_prob)| desc and print top-N
    row_cols = [
        "elo_diff_before_hfa",
        "hfa_added_to_diff",
        "elo_diff_after_hfa",
        "elo_diff_for_prob",
        "d_scaled",
    ]
    if "match_id" in df_a.columns and "match_id" in df_b.columns:
        left_cols = ["match_id"] + [c for c in ["home_team", "away_team"] if c in df_a.columns] + [c for c in row_cols if c in df_a.columns]
        right_cols = ["match_id"] + [c for c in row_cols if c in df_b.columns]
        row_df = df_a[left_cols].merge(df_b[right_cols], on="match_id", how="inner", suffixes=("_a", "_b"))
        row_align = "match_id"
    elif set(["datetime", "home_team", "away_team"]).issubset(df_a.columns) and set(["datetime", "home_team", "away_team"]).issubset(df_b.columns):
        key_cols = ["datetime", "home_team", "away_team"]
        left_cols = key_cols + [c for c in row_cols if c in df_a.columns]
        right_cols = key_cols + [c for c in row_cols if c in df_b.columns]
        row_df = df_a[left_cols].merge(df_b[right_cols], on=key_cols, how="inner", suffixes=("_a", "_b"))
        row_align = "datetime+home+away"
    else:
        min_len = min(len(df_a), len(df_b))
        row_align = "row_index"
        row_df = pd.DataFrame({"row_index": np.arange(min_len)})
        if "home_team" in df_a.columns:
            row_df["home_team"] = df_a["home_team"].head(min_len).values
        if "away_team" in df_a.columns:
            row_df["away_team"] = df_a["away_team"].head(min_len).values
        for c in row_cols:
            if c in df_a.columns:
                row_df[f"{c}_a"] = df_a[c].head(min_len).values
            if c in df_b.columns:
                row_df[f"{c}_b"] = df_b[c].head(min_len).values

    if not row_df.empty:
        if "elo_diff_for_prob_a" in row_df.columns and "elo_diff_for_prob_b" in row_df.columns:
            row_df["__diff_elo_for_prob"] = (
                pd.to_numeric(row_df["elo_diff_for_prob_a"], errors="coerce")
                - pd.to_numeric(row_df["elo_diff_for_prob_b"], errors="coerce")
            )
        elif "elo_diff_used_for_prob_a" in row_df.columns and "elo_diff_used_for_prob_b" in row_df.columns:
            row_df["__diff_elo_for_prob"] = (
                pd.to_numeric(row_df["elo_diff_used_for_prob_a"], errors="coerce")
                - pd.to_numeric(row_df["elo_diff_used_for_prob_b"], errors="coerce")
            )
        else:
            row_df["__diff_elo_for_prob"] = 0.0
        if "d_scaled_a" in row_df.columns and "d_scaled_b" in row_df.columns:
            row_df["__diff_d_scaled"] = (
                pd.to_numeric(row_df["d_scaled_a"], errors="coerce")
                - pd.to_numeric(row_df["d_scaled_b"], errors="coerce")
            )
        else:
            row_df["__diff_d_scaled"] = np.nan
        top_rows = row_df.reindex(row_df["__diff_elo_for_prob"].abs().sort_values(ascending=False).index).head(int(HFA_TRACE_N))
        for _, r in top_rows.iterrows():
            match_id = str(r.get("match_id", r.get("row_index", "")))
            home = str(r.get("home_team", ""))
            away = str(r.get("away_team", ""))
            print(
                f"[HFA_TRACE_ROW] align={row_align} match_id={match_id} home={home} away={away} "
                f"elo_before={pd.to_numeric(r.get('elo_diff_before_hfa_a'), errors='coerce'):.4f} "
                f"hfa_added={pd.to_numeric(r.get('hfa_added_to_diff_a'), errors='coerce'):.4f} "
                f"elo_after={pd.to_numeric(r.get('elo_diff_after_hfa_a'), errors='coerce'):.4f} "
                f"elo_for_prob={pd.to_numeric(r.get('elo_diff_for_prob_a'), errors='coerce'):.4f} "
                f"d_scaled={pd.to_numeric(r.get('d_scaled_a'), errors='coerce'):.4f} "
                f"diff_elo_for_prob={pd.to_numeric(r.get('__diff_elo_for_prob'), errors='coerce'):.4f} "
                f"diff_d_scaled={pd.to_numeric(r.get('__diff_d_scaled'), errors='coerce'):.4f}"
            )


def _compare_hfa_probability_files(path_a, path_b, label_a="HFA_ON", label_b="HFA_OFF"):
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    _log_hfa_intermediate_trace(df_a, df_b)

    sha_a = _sha1_file(path_a)
    sha_b = _sha1_file(path_b)
    print(f"[HFA_SELF_CHECK:SHA1] {label_a}={sha_a}")
    print(f"[HFA_SELF_CHECK:SHA1] {label_b}={sha_b}")
    if sha_a == sha_b:
        print("[ERROR] HFA_ON and HFA_OFF outputs are identical (sha1 match)")
        raise RuntimeError("HFA self-check failed: identical outputs by sha1")

    intermediate_candidates = [
        "elo_diff_for_prob",
        "elo_diff",
        "elo_diff_scaled",
        "elo_diff_raw",
    ]
    aligned_intermediate, shared_intermediate, inter_key = _build_hfa_aligned_dataframe(
        df_a, df_b, intermediate_candidates + ["d_scaled"]
    )
    elo_col = next((c for c in intermediate_candidates if c in shared_intermediate), None)
    max_diff_elo = _compute_max_abs_diff(aligned_intermediate, elo_col) if elo_col else None
    max_diff_d_scaled = _compute_max_abs_diff(aligned_intermediate, "d_scaled") if "d_scaled" in shared_intermediate else None
    elo_part = (
        f"max_abs_diff_{elo_col}={max_diff_elo:.6f}" if (elo_col and max_diff_elo is not None)
        else "max_abs_diff_elo_diff_for_prob=NA"
    )
    d_scaled_part = (
        f"max_abs_diff_d_scaled={max_diff_d_scaled:.6f}" if max_diff_d_scaled is not None
        else "max_abs_diff_d_scaled=NA"
    )
    print(f"[HFA_COMPARE_INTERMEDIATE] key={inter_key} {elo_part} {d_scaled_part}")

    prob_cols = _detect_probability_columns(df_a, df_b)
    merged, _, prob_key = _build_hfa_aligned_dataframe(df_a, df_b, prob_cols)
    if merged.empty:
        raise RuntimeError("[ERROR] HFA compare merge returned 0 rows")
    home_col, draw_col, away_col = prob_cols
    d_home = (
        pd.to_numeric(merged[f"{home_col}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{home_col}_b"], errors="coerce")
    ).abs()
    d_draw = (
        pd.to_numeric(merged[f"{draw_col}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{draw_col}_b"], errors="coerce")
    ).abs()
    d_away = (
        pd.to_numeric(merged[f"{away_col}_a"], errors="coerce")
        - pd.to_numeric(merged[f"{away_col}_b"], errors="coerce")
    ).abs()
    any_diff = (d_home > 1e-12) | (d_draw > 1e-12) | (d_away > 1e-12)
    num_rows_with_any_diff = int(any_diff.sum())
    print(
        f"[HFA_SELF_CHECK] key={prob_key} cols={prob_cols} "
        f"max_abs_diff_prob_home={float(d_home.max(skipna=True)):.6f} "
        f"max_abs_diff_prob_draw={float(d_draw.max(skipna=True)):.6f} "
        f"max_abs_diff_prob_away={float(d_away.max(skipna=True)):.6f} "
        f"num_rows_with_any_diff={num_rows_with_any_diff}"
    )
    if num_rows_with_any_diff == 0:
        print("[ERROR] HFA_ON and HFA_OFF prediction CSVs are identical (no probability differences detected)")
        raise RuntimeError("HFA self-check failed: no probability differences")


def _run_hfa_self_check_generation():
    args = [a for a in RAW_CLI_ARGS if a not in {"--self-check-hfa", "--skip-hfa-self-check"}]
    on_path = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions_hfa_on.csv")
    off_path = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions_hfa_off.csv")
    script_path = os.path.abspath(__file__)
    print("[SELF_CHECK] force recalculation active; cache reuse disabled")
    for mode, out_path, label in [(1, on_path, "HFA_ON"), (0, off_path, "HFA_OFF")]:
        env = os.environ.copy()
        env["ENABLE_HFA"] = str(mode)
        env["OUTPUT_PRED_CSV"] = out_path
        env["SKIP_HFA_SELF_CHECK"] = "1"
        cmd = [sys.executable, script_path] + args
        if "--force" not in cmd:
            cmd.append("--force")
        print(f"[HFA_SELF_CHECK] generating {label}: out={out_path}")
        subprocess.run(cmd, check=True, cwd=BASE_DIR, env=env)
        meta_path = f"{out_path}.meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                c = meta.get("hfa_apply_count", {})
                print(
                    f"[HFA_APPLY_COUNT] label={label} applied={int(c.get('applied', 0))} "
                    f"skipped={int(c.get('skipped', 0))} "
                    f"reason_counts={json.dumps(c.get('reason_counts', {}), ensure_ascii=False, sort_keys=True)}"
                )
            except Exception as e:
                print(f"[HFA_APPLY_COUNT][WARN] label={label} meta read failed: {e}")
        else:
            print(f"[HFA_APPLY_COUNT][WARN] label={label} meta not found: {meta_path}")
    _compare_hfa_probability_files(on_path, off_path, label_a="HFA_ON", label_b="HFA_OFF")


def _parse_sensitivity_float_values(raw, default_values):
    vals = []
    for token in str(raw).split(","):
        t = str(token).strip()
        if not t:
            continue
        try:
            vals.append(float(t))
        except Exception:
            print(f"[SENSITIVITY][WARN] invalid float token ignored: {t!r}")
    if not vals:
        vals = list(default_values)
    dedup = []
    seen = set()
    for v in vals:
        k = f"{v:.8f}"
        if k in seen:
            continue
        seen.add(k)
        dedup.append(float(v))
    return dedup


def _parse_sensitivity_int_values(raw, default_values):
    vals = []
    for token in str(raw).split(","):
        t = str(token).strip()
        if not t:
            continue
        try:
            vals.append(int(float(t)))
        except Exception:
            print(f"[SENSITIVITY][WARN] invalid int token ignored: {t!r}")
    if not vals:
        vals = list(default_values)
    dedup = []
    seen = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            dedup.append(int(v))
    return dedup


def _safe_name_num(v):
    s = f"{float(v):.3f}"
    return s.replace("-", "m").replace(".", "p")


def _parse_target_d_range(raw, default_low, default_high, name):
    vals = _parse_sensitivity_float_values(raw, [float(default_low), float(default_high)])
    if len(vals) < 2:
        return float(default_low), float(default_high)
    lo = float(min(vals[0], vals[1]))
    hi = float(max(vals[0], vals[1]))
    if lo == hi:
        hi = lo + 1.0
    if lo < 0:
        print(f"[CONFIG][WARN] {name} lower bound <0; clipped to 0")
        lo = 0.0
    return lo, hi


def log_draw_score_distribution(df, label):
    if df is None or df.empty:
        print(f"[DRAW_SCORE_DIST:{label}] unavailable")
        return
    cols = None
    for c in [("prob_home_win", "prob_draw", "prob_away_win"), ("prob_home", "prob_draw", "prob_away")]:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None:
        print(f"[DRAW_SCORE_DIST:{label}] unavailable")
        return
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    score = pdw - pd.concat([ph, pa], axis=1).max(axis=1)
    score = score.dropna()
    if score.empty:
        print(f"[DRAW_SCORE_DIST:{label}] unavailable")
        return
    q = score.quantile([0.10, 0.50, 0.90])
    print(
        f"[DRAW_SCORE_DIST:{label}] rows={len(score)} "
        f"min={float(score.min()):.6f} p10={float(q.loc[0.10]):.6f} median={float(q.loc[0.50]):.6f} "
        f"p90={float(q.loc[0.90]):.6f} max={float(score.max()):.6f} mean={float(score.mean()):.6f}"
    )
    ths = _parse_sensitivity_float_values(DRAW_SCORE_THRESHOLDS_RAW, [0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ths = sorted(set(float(x) for x in ths if float(x) >= 0.0))
    counts = [f">={t:.2f}:{int((score >= t).sum())}" for t in ths]
    print(f"[DRAW_SCORE_COUNTS:{label}] " + " ".join(counts))


def _calc_hda_dist_for_sensitivity(series):
    s = pd.Series(series, dtype="object").astype(str).str.upper()
    s = s[s.isin(["H", "D", "A"])]
    n = int(len(s))
    h = int((s == "H").sum())
    d = int((s == "D").sum())
    a = int((s == "A").sum())
    return {
        "rows": n,
        "H_cnt": h,
        "D_cnt": d,
        "A_cnt": a,
        "H_pct": (100.0 * h / n) if n > 0 else 0.0,
        "D_pct": (100.0 * d / n) if n > 0 else 0.0,
        "A_pct": (100.0 * a / n) if n > 0 else 0.0,
    }


def _calc_prob_means_for_sensitivity(df):
    candidates = [
        ("prob_home_win", "prob_draw", "prob_away_win"),
        ("prob_home", "prob_draw", "prob_away"),
    ]
    cols = None
    for c in candidates:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None:
        return None
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    valid = ph.notna() & pdw.notna() & pa.notna()
    if int(valid.sum()) <= 0:
        return None
    return {
        "rows": int(valid.sum()),
        "prob_cols": cols,
        "home_mean": float(ph[valid].mean()),
        "draw_mean": float(pdw[valid].mean()),
        "away_mean": float(pa[valid].mean()),
    }


def _calc_multiclass_logloss_from_df(df, eps=1e-15):
    if df is None or df.empty:
        return None
    if "actual_result" not in df.columns:
        return None
    cols = None
    for c in [("prob_home_win", "prob_draw", "prob_away_win"), ("prob_home", "prob_draw", "prob_away")]:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None:
        return None
    actual = df["actual_result"].astype(str).str.upper()
    valid_label = actual.isin(["H", "D", "A"])
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    valid_prob = ph.notna() & pdw.notna() & pa.notna()
    valid = valid_label & valid_prob
    if int(valid.sum()) <= 0:
        return None
    phv = ph[valid].clip(eps, 1.0 - eps)
    pdv = pdw[valid].clip(eps, 1.0 - eps)
    pav = pa[valid].clip(eps, 1.0 - eps)
    y = actual[valid]
    p = np.where(y == "H", phv, np.where(y == "D", pdv, pav))
    return float(-np.mean(np.log(p)))


def _calc_multiclass_brier_from_df(df):
    if df is None or df.empty:
        return None
    if "actual_result" not in df.columns:
        return None
    cols = None
    for c in [("prob_home_win", "prob_draw", "prob_away_win"), ("prob_home", "prob_draw", "prob_away")]:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None:
        return None
    actual = df["actual_result"].astype(str).str.upper()
    valid_label = actual.isin(["H", "D", "A"])
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    valid_prob = ph.notna() & pdw.notna() & pa.notna()
    valid = valid_label & valid_prob
    if int(valid.sum()) <= 0:
        return None
    actual_h = (actual[valid] == "H").astype(float)
    actual_d = (actual[valid] == "D").astype(float)
    actual_a = (actual[valid] == "A").astype(float)
    brier = ((ph[valid] - actual_h) ** 2) + ((pdw[valid] - actual_d) ** 2) + ((pa[valid] - actual_a) ** 2)
    return float(brier.mean())


def _calc_multiclass_logloss_from_df_with_cols(df, cols, eps=1e-15):
    if df is None or df.empty or "actual_result" not in df.columns:
        return None
    if not set(cols).issubset(df.columns):
        return None
    actual = df["actual_result"].astype(str).str.upper()
    valid_label = actual.isin(["H", "D", "A"])
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    valid_prob = ph.notna() & pdw.notna() & pa.notna()
    valid = valid_label & valid_prob
    if int(valid.sum()) <= 0:
        return None
    phv = ph[valid].clip(eps, 1.0 - eps)
    pdv = pdw[valid].clip(eps, 1.0 - eps)
    pav = pa[valid].clip(eps, 1.0 - eps)
    y = actual[valid]
    p = np.where(y == "H", phv, np.where(y == "D", pdv, pav))
    return float(-np.mean(np.log(p)))


def _calc_multiclass_brier_from_df_with_cols(df, cols):
    if df is None or df.empty or "actual_result" not in df.columns:
        return None
    if not set(cols).issubset(df.columns):
        return None
    actual = df["actual_result"].astype(str).str.upper()
    valid_label = actual.isin(["H", "D", "A"])
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    valid_prob = ph.notna() & pdw.notna() & pa.notna()
    valid = valid_label & valid_prob
    if int(valid.sum()) <= 0:
        return None
    actual_h = (actual[valid] == "H").astype(float)
    actual_d = (actual[valid] == "D").astype(float)
    actual_a = (actual[valid] == "A").astype(float)
    brier = ((ph[valid] - actual_h) ** 2) + ((pdw[valid] - actual_d) ** 2) + ((pa[valid] - actual_a) ** 2)
    return float(brier.mean())


def _calc_multiclass_ece_from_df_with_cols(df, cols, n_bins=10):
    if df is None or df.empty or "actual_result" not in df.columns:
        return None
    if not set(cols).issubset(df.columns):
        return None
    actual = df["actual_result"].astype(str).str.upper()
    valid_label = actual.isin(["H", "D", "A"])
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    valid_prob = ph.notna() & pdw.notna() & pa.notna()
    valid = valid_label & valid_prob
    if int(valid.sum()) <= 0:
        return None
    probs = np.stack([ph[valid].to_numpy(), pdw[valid].to_numpy(), pa[valid].to_numpy()], axis=1)
    conf = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    pred = np.where(pred_idx == 0, "H", np.where(pred_idx == 1, "D", "A"))
    corr = (pred == actual[valid].to_numpy()).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(conf)
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count <= 0:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(corr[mask].mean())
        ece += abs(avg_acc - avg_conf) * (count / total)
    return float(ece)


CALIBRATION_DIR = os.path.join(REPORT_DIR, "metrics", "calibration")
CALIBRATION_MIN_T = float(os.environ.get("CALIBRATION_MIN_T", "0.50"))
CALIBRATION_MAX_T = float(os.environ.get("CALIBRATION_MAX_T", "2.50"))
CALIBRATION_T_STEP = float(os.environ.get("CALIBRATION_T_STEP", "0.01"))
CALIBRATION_VAL_RATIO = float(os.environ.get("CALIBRATION_VAL_RATIO", "0.20"))
CALIBRATION_MIN_VAL_ROWS = int(os.environ.get("CALIBRATION_MIN_VAL_ROWS", "20"))
CALIBRATION_MIN_TRAIN_ROWS = int(os.environ.get("CALIBRATION_MIN_TRAIN_ROWS", "60"))
THRESHOLD_MIN = float(os.environ.get("CONFIDENCE_THRESHOLD_MIN", "0.30"))
THRESHOLD_MAX = float(os.environ.get("CONFIDENCE_THRESHOLD_MAX", "0.70"))
THRESHOLD_STEP = float(os.environ.get("CONFIDENCE_THRESHOLD_STEP", "0.01"))
THRESHOLD_MIN_COUNT = int(os.environ.get("CONFIDENCE_THRESHOLD_MIN_COUNT", "10"))
THRESHOLD_MIN_COVERAGE = float(os.environ.get("CONFIDENCE_THRESHOLD_MIN_COVERAGE", "0.20"))
CONFIDENCE_WATCH_MARGIN = float(os.environ.get("CONFIDENCE_WATCH_MARGIN", "0.05"))


def _temperature_grid():
    return np.round(np.arange(CALIBRATION_MIN_T, CALIBRATION_MAX_T + (CALIBRATION_T_STEP * 0.5), CALIBRATION_T_STEP), 6)


def _threshold_grid():
    return np.round(np.arange(THRESHOLD_MIN, THRESHOLD_MAX + (THRESHOLD_STEP * 0.5), THRESHOLD_STEP), 6)


def _apply_temperature_scaling_arrays(ph, pdw, pa, temperature):
    phv = np.clip(np.asarray(ph, dtype=float), 1e-12, 1.0)
    pdv = np.clip(np.asarray(pdw, dtype=float), 1e-12, 1.0)
    pav = np.clip(np.asarray(pa, dtype=float), 1e-12, 1.0)
    t = max(float(temperature), 1e-6)
    logits = np.stack([np.log(phv), np.log(pdv), np.log(pav)], axis=1) / t
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    denom = expv.sum(axis=1, keepdims=True)
    scaled = expv / np.clip(denom, 1e-12, None)
    return scaled[:, 0], scaled[:, 1], scaled[:, 2]


def _fit_temperature_from_df(df):
    result = {
        "temperature_final": 1.0,
        "temperature_train": 1.0,
        "rows_total": 0,
        "rows_train": 0,
        "rows_val": 0,
        "before": {"logloss": None, "brier": None, "ece": None},
        "after": {"logloss": None, "brier": None, "ece": None},
    }
    if df is None or df.empty or "actual_result" not in df.columns:
        return result
    work = df.copy()
    work["__dt_sort"] = pd.to_datetime(work.get("datetime"), errors="coerce")
    work = work.sort_values(["__dt_sort", "match_id"], kind="mergesort", na_position="last")
    actual = work["actual_result"].astype(str).str.upper()
    valid = actual.isin(["H", "D", "A"])
    for col in ["prob_home_win", "prob_draw", "prob_away_win"]:
        valid &= pd.to_numeric(work.get(col), errors="coerce").notna()
    work = work.loc[valid].copy()
    result["rows_total"] = int(len(work))
    if len(work) < max(CALIBRATION_MIN_VAL_ROWS + 5, CALIBRATION_MIN_TRAIN_ROWS):
        return result

    val_rows = max(CALIBRATION_MIN_VAL_ROWS, int(round(len(work) * CALIBRATION_VAL_RATIO)))
    val_rows = min(val_rows, max(1, len(work) - CALIBRATION_MIN_TRAIN_ROWS))
    if val_rows <= 0:
        return result
    train = work.iloc[:-val_rows].copy()
    val = work.iloc[-val_rows:].copy()
    result["rows_train"] = int(len(train))
    result["rows_val"] = int(len(val))
    if len(train) < CALIBRATION_MIN_TRAIN_ROWS or len(val) <= 0:
        return result

    best_t = 1.0
    best_ll = None
    for t in _temperature_grid():
        tph, tpd, tpa = _apply_temperature_scaling_arrays(
            train["prob_home_win"].to_numpy(),
            train["prob_draw"].to_numpy(),
            train["prob_away_win"].to_numpy(),
            t,
        )
        temp_df = train.copy()
        temp_df["__ph"] = tph
        temp_df["__pd"] = tpd
        temp_df["__pa"] = tpa
        ll = _calc_multiclass_logloss_from_df_with_cols(temp_df, ("__ph", "__pd", "__pa"))
        if ll is None:
            continue
        if (best_ll is None) or (ll < best_ll - 1e-12) or (abs(ll - best_ll) <= 1e-12 and float(t) < float(best_t)):
            best_ll = float(ll)
            best_t = float(t)
    result["temperature_train"] = float(best_t)

    result["before"] = {
        "logloss": _calc_multiclass_logloss_from_df_with_cols(val, ("prob_home_win", "prob_draw", "prob_away_win")),
        "brier": _calc_multiclass_brier_from_df_with_cols(val, ("prob_home_win", "prob_draw", "prob_away_win")),
        "ece": _calc_multiclass_ece_from_df_with_cols(val, ("prob_home_win", "prob_draw", "prob_away_win")),
    }
    vph, vpd, vpa = _apply_temperature_scaling_arrays(
        val["prob_home_win"].to_numpy(),
        val["prob_draw"].to_numpy(),
        val["prob_away_win"].to_numpy(),
        best_t,
    )
    val_scaled = val.copy()
    val_scaled["__ph"] = vph
    val_scaled["__pd"] = vpd
    val_scaled["__pa"] = vpa
    result["after"] = {
        "logloss": _calc_multiclass_logloss_from_df_with_cols(val_scaled, ("__ph", "__pd", "__pa")),
        "brier": _calc_multiclass_brier_from_df_with_cols(val_scaled, ("__ph", "__pd", "__pa")),
        "ece": _calc_multiclass_ece_from_df_with_cols(val_scaled, ("__ph", "__pd", "__pa")),
    }

    best_t_all = 1.0
    best_ll_all = None
    for t in _temperature_grid():
        tph, tpd, tpa = _apply_temperature_scaling_arrays(
            work["prob_home_win"].to_numpy(),
            work["prob_draw"].to_numpy(),
            work["prob_away_win"].to_numpy(),
            t,
        )
        temp_df = work.copy()
        temp_df["__ph"] = tph
        temp_df["__pd"] = tpd
        temp_df["__pa"] = tpa
        ll = _calc_multiclass_logloss_from_df_with_cols(temp_df, ("__ph", "__pd", "__pa"))
        if ll is None:
            continue
        if (best_ll_all is None) or (ll < best_ll_all - 1e-12) or (abs(ll - best_ll_all) <= 1e-12 and float(t) < float(best_t_all)):
            best_ll_all = float(ll)
            best_t_all = float(t)
    result["temperature_final"] = float(best_t_all)
    return result


def _argmax_label_from_triplets(ph, pdw, pa):
    return np.where(
        (ph >= pdw) & (ph >= pa),
        "H",
        np.where((pa >= ph) & (pa >= pdw), "A", "D"),
    )


def _apply_temperature_scaling_df(df, temperature):
    out = df.copy()
    required = ["prob_home_win", "prob_draw", "prob_away_win"]
    if out.empty or not set(required).issubset(out.columns):
        for src, dst in [("prob_home_win", "prob_home_win_cal"), ("prob_draw", "prob_draw_cal"), ("prob_away_win", "prob_away_win_cal")]:
            if src in out.columns and dst not in out.columns:
                out[dst] = out[src]
        return out
    ph = pd.to_numeric(out["prob_home_win"], errors="coerce")
    pdw = pd.to_numeric(out["prob_draw"], errors="coerce")
    pa = pd.to_numeric(out["prob_away_win"], errors="coerce")
    valid = ph.notna() & pdw.notna() & pa.notna()
    out["prob_home_win_cal"] = ph
    out["prob_draw_cal"] = pdw
    out["prob_away_win_cal"] = pa
    if int(valid.sum()) > 0:
        sph, spd, spa = _apply_temperature_scaling_arrays(ph[valid].to_numpy(), pdw[valid].to_numpy(), pa[valid].to_numpy(), temperature)
        out.loc[valid, "prob_home_win_cal"] = sph
        out.loc[valid, "prob_draw_cal"] = spd
        out.loc[valid, "prob_away_win_cal"] = spa
    out["max_prob_cal"] = pd.concat(
        [
            pd.to_numeric(out["prob_home_win_cal"], errors="coerce"),
            pd.to_numeric(out["prob_draw_cal"], errors="coerce"),
            pd.to_numeric(out["prob_away_win_cal"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    valid_cal = out["prob_home_win_cal"].notna() & out["prob_draw_cal"].notna() & out["prob_away_win_cal"].notna()
    out["predicted_result_cal"] = out.get("predicted_result", pd.Series(index=out.index, dtype="object"))
    if int(valid_cal.sum()) > 0:
        out.loc[valid_cal, "predicted_result_cal"] = _argmax_label_from_triplets(
            pd.to_numeric(out.loc[valid_cal, "prob_home_win_cal"], errors="coerce").to_numpy(),
            pd.to_numeric(out.loc[valid_cal, "prob_draw_cal"], errors="coerce").to_numpy(),
            pd.to_numeric(out.loc[valid_cal, "prob_away_win_cal"], errors="coerce").to_numpy(),
        )
    out["argmax_max_prob_cal"] = pd.to_numeric(out["max_prob_cal"], errors="coerce")
    out["decision_reason_cal"] = np.where(valid_cal, "CAL_ARGMAX", out.get("decision_reason", ""))
    return out


def _optimize_confidence_thresholds(df):
    thresholds = {}
    rows = []
    if df is None or df.empty:
        return thresholds, pd.DataFrame(rows)
    required = {"predicted_result", "actual_result", "max_prob_cal"}
    if not required.issubset(df.columns):
        return thresholds, pd.DataFrame(rows)
    work = df.copy()
    work["predicted_result"] = work["predicted_result"].astype(str).str.upper()
    work["actual_result"] = work["actual_result"].astype(str).str.upper()
    work["max_prob_cal"] = pd.to_numeric(work["max_prob_cal"], errors="coerce")
    work = work[work["predicted_result"].isin(["H", "D", "A"]) & work["actual_result"].isin(["H", "D", "A"]) & work["max_prob_cal"].notna()].copy()
    for label in ["H", "D", "A"]:
        part = work[work["predicted_result"] == label].copy()
        label_total = int(len(part))
        if label_total <= 0:
            thresholds[label] = None
            continue
        min_count = max(THRESHOLD_MIN_COUNT, int(math.ceil(label_total * THRESHOLD_MIN_COVERAGE)))
        best_row = None
        for t in _threshold_grid():
            sel = part[part["max_prob_cal"] >= float(t)].copy()
            selected = int(len(sel))
            coverage = float(selected / label_total) if label_total > 0 else 0.0
            correct = int((sel["predicted_result"] == sel["actual_result"]).sum()) if selected > 0 else 0
            precision = float(correct / selected) if selected > 0 else None
            eligible = selected >= min_count
            rows.append({
                "label": label,
                "threshold": float(t),
                "selected": selected,
                "label_total": label_total,
                "coverage": coverage,
                "correct": correct,
                "precision": precision,
                "eligible": int(eligible),
            })
            if not eligible or precision is None:
                continue
            candidate = (float(precision), selected, -float(t))
            if best_row is None:
                best_row = (candidate, float(t))
            elif candidate > best_row[0]:
                best_row = (candidate, float(t))
        thresholds[label] = best_row[1] if best_row is not None else None
    return thresholds, pd.DataFrame(rows)


def _apply_confidence_class(df, thresholds_by_label):
    out = df.copy()
    out["max_prob_cal"] = pd.to_numeric(out.get("max_prob_cal"), errors="coerce")
    pred = out.get("predicted_result", pd.Series(index=out.index, dtype="object")).astype(str).str.upper()
    threshold_values = pred.map(lambda x: thresholds_by_label.get(x))
    out["confidence_threshold"] = pd.to_numeric(threshold_values, errors="coerce")
    hit = out["confidence_threshold"].notna() & out["max_prob_cal"].notna() & (out["max_prob_cal"] >= out["confidence_threshold"])
    out["threshold_hit_flag"] = hit
    watch_floor = out["confidence_threshold"] - CONFIDENCE_WATCH_MARGIN
    watch = out["confidence_threshold"].notna() & out["max_prob_cal"].notna() & (~hit) & (out["max_prob_cal"] >= watch_floor)
    out["confidence_class"] = np.where(
        hit,
        "trusted",
        np.where(
            watch,
            "watch",
            np.where(out["confidence_threshold"].notna(), "fragile", "unscored"),
        ),
    )
    out["low_confidence"] = out["confidence_class"].isin(["fragile", "unscored"])
    return out


def _save_calibration_artifacts(df_backtest, df_pred, league, season_year):
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    league_key = str(league).lower()
    work_bt = df_backtest.copy()
    if "league" in work_bt.columns:
        work_bt = work_bt[work_bt["league"].astype(str).str.lower() == league_key].copy()
    temp_meta = _fit_temperature_from_df(work_bt)
    deploy_temperature = float(temp_meta["temperature_final"])
    validation_ok = True
    ll_before = temp_meta["before"]["logloss"]
    ll_after = temp_meta["after"]["logloss"]
    br_before = temp_meta["before"]["brier"]
    br_after = temp_meta["after"]["brier"]
    if ll_before is not None and ll_after is not None and float(ll_after) > float(ll_before) + 1e-12:
        validation_ok = False
    if br_before is not None and br_after is not None and float(br_after) > float(br_before) + 1e-12:
        validation_ok = False
    if not validation_ok:
        deploy_temperature = 1.0
    df_backtest_cal = _apply_temperature_scaling_df(df_backtest, deploy_temperature)
    df_pred_cal = _apply_temperature_scaling_df(df_pred, deploy_temperature)
    thresholds, threshold_scan = _optimize_confidence_thresholds(work_bt.assign(
        prob_home_win_cal=df_backtest_cal.loc[work_bt.index, "prob_home_win_cal"],
        prob_draw_cal=df_backtest_cal.loc[work_bt.index, "prob_draw_cal"],
        prob_away_win_cal=df_backtest_cal.loc[work_bt.index, "prob_away_win_cal"],
        max_prob_cal=df_backtest_cal.loc[work_bt.index, "max_prob_cal"],
    ))
    df_backtest_cal = _apply_confidence_class(df_backtest_cal, thresholds)
    df_pred_cal = _apply_confidence_class(df_pred_cal, thresholds)

    before_cols = ("prob_home_win", "prob_draw", "prob_away_win")
    after_cols = ("prob_home_win_cal", "prob_draw_cal", "prob_away_win_cal")
    val_report = {
        "league": str(league).upper(),
        "season_year": int(season_year),
        "temperature_train": float(temp_meta["temperature_train"]),
        "temperature_final": float(temp_meta["temperature_final"]),
        "rows_total": int(temp_meta["rows_total"]),
        "rows_train": int(temp_meta["rows_train"]),
        "rows_val": int(temp_meta["rows_val"]),
        "validation_sample_warning": int(temp_meta["rows_val"] < THRESHOLD_MIN_COUNT),
        "validation_ok": int(validation_ok),
        "deployed_temperature": float(deploy_temperature),
        "logloss_before": temp_meta["before"]["logloss"],
        "logloss_after": temp_meta["after"]["logloss"],
        "brier_before": temp_meta["before"]["brier"],
        "brier_after": temp_meta["after"]["brier"],
        "ece_before": temp_meta["before"]["ece"],
        "ece_after": temp_meta["after"]["ece"],
        "overall_logloss_before": _calc_multiclass_logloss_from_df_with_cols(work_bt, before_cols),
        "overall_logloss_after": _calc_multiclass_logloss_from_df_with_cols(df_backtest_cal.loc[work_bt.index].copy(), after_cols),
        "overall_brier_before": _calc_multiclass_brier_from_df_with_cols(work_bt, before_cols),
        "overall_brier_after": _calc_multiclass_brier_from_df_with_cols(df_backtest_cal.loc[work_bt.index].copy(), after_cols),
        "overall_ece_before": _calc_multiclass_ece_from_df_with_cols(work_bt, before_cols),
        "overall_ece_after": _calc_multiclass_ece_from_df_with_cols(df_backtest_cal.loc[work_bt.index].copy(), after_cols),
    }

    report_path = os.path.join(CALIBRATION_DIR, f"calibration_report_{league_key}_{season_year}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, ensure_ascii=False, indent=2)
    threshold_path = os.path.join(CALIBRATION_DIR, f"thresholds_{league_key}_{season_year}.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"league": league_key, "season_year": int(season_year), "thresholds": thresholds}, f, ensure_ascii=False, indent=2)
    threshold_scan_path = os.path.join(CALIBRATION_DIR, f"threshold_scan_{league_key}_{season_year}.csv")
    threshold_scan.to_csv(threshold_scan_path, index=False, encoding="utf-8-sig")
    print(
        f"[CALIBRATION] league={league_key} T_train={temp_meta['temperature_train']:.3f} "
        f"T_final={temp_meta['temperature_final']:.3f} deploy_T={deploy_temperature:.3f} "
        f"validation_ok={int(validation_ok)} "
        f"val_logloss={temp_meta['before']['logloss']}->{temp_meta['after']['logloss']} "
        f"val_brier={temp_meta['before']['brier']}->{temp_meta['after']['brier']} "
        f"val_ece={temp_meta['before']['ece']}->{temp_meta['after']['ece']}"
    )
    if int(temp_meta["rows_val"]) < THRESHOLD_MIN_COUNT:
        print(
            f"[CALIBRATION][WARN] league={league_key} validation_rows={int(temp_meta['rows_val'])} "
            f"is small; temperature may be unstable"
        )
    print(f"[CALIBRATION] report={report_path}")
    print(f"[CALIBRATION] thresholds={threshold_path}")
    print(f"[CALIBRATION] threshold_scan={threshold_scan_path}")
    return df_pred_cal, df_backtest_cal, {
        "report_path": report_path,
        "threshold_path": threshold_path,
        "threshold_scan_path": threshold_scan_path,
        "temperature_final": float(temp_meta["temperature_final"]),
        "deployed_temperature": float(deploy_temperature),
        "temperature_train": float(temp_meta["temperature_train"]),
        "validation_ok": int(validation_ok),
        "thresholds": thresholds,
        "metrics": val_report,
    }


def _calc_argmax_result_from_probs(df):
    if df is None or df.empty:
        return pd.Series(dtype="object")
    cols = None
    for c in [("prob_home_win", "prob_draw", "prob_away_win"), ("prob_home", "prob_draw", "prob_away")]:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    valid = ph.notna() & pdw.notna() & pa.notna()
    out.loc[valid] = np.where(
        (ph[valid] >= pdw[valid]) & (ph[valid] >= pa[valid]),
        "H",
        np.where((pa[valid] >= ph[valid]) & (pa[valid] >= pdw[valid]), "A", "D"),
    )
    return out


def _calc_profile_scan_metrics_from_df(df):
    if df is None or df.empty:
        return {
            "matches": 0,
            "hit_rate_argmax": None,
            "logloss": None,
            "brier": None,
            "count_maxp_gt_060": 0,
            "count_maxp_050_060": 0,
            "count_maxp_lt_050": 0,
            "ratio_maxp_gt_060": 0.0,
            "ratio_maxp_050_060": 0.0,
            "ratio_maxp_lt_050": 0.0,
        }
    cols = None
    for c in [("prob_home_win", "prob_draw", "prob_away_win"), ("prob_home", "prob_draw", "prob_away")]:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None:
        return {
            "matches": int(len(df)),
            "hit_rate_argmax": None,
            "logloss": None,
            "brier": None,
            "count_maxp_gt_060": 0,
            "count_maxp_050_060": 0,
            "count_maxp_lt_050": int(len(df)),
            "ratio_maxp_gt_060": 0.0,
            "ratio_maxp_050_060": 0.0,
            "ratio_maxp_lt_050": 1.0 if len(df) > 0 else 0.0,
        }
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    valid_prob = ph.notna() & pdw.notna() & pa.notna()
    work = df.loc[valid_prob].copy()
    if work.empty:
        return {
            "matches": 0,
            "hit_rate_argmax": None,
            "logloss": None,
            "brier": None,
            "count_maxp_gt_060": 0,
            "count_maxp_050_060": 0,
            "count_maxp_lt_050": 0,
            "ratio_maxp_gt_060": 0.0,
            "ratio_maxp_050_060": 0.0,
            "ratio_maxp_lt_050": 0.0,
        }
    ph = pd.to_numeric(work[cols[0]], errors="coerce")
    pdw = pd.to_numeric(work[cols[1]], errors="coerce")
    pa = pd.to_numeric(work[cols[2]], errors="coerce")
    maxp = pd.concat([ph, pdw, pa], axis=1).max(axis=1)
    count_gt = int((maxp > 0.60).sum())
    count_mid = int(((maxp >= 0.50) & (maxp <= 0.60)).sum())
    count_lt = int((maxp < 0.50).sum())
    matches = int(len(work))
    argmax_pred = _calc_argmax_result_from_probs(work).astype(str).str.upper()
    actual = work.get("actual_result", pd.Series(dtype="object")).astype(str).str.upper()
    valid_label = actual.isin(["H", "D", "A"]) & argmax_pred.isin(["H", "D", "A"])
    hit_rate_argmax = float((argmax_pred[valid_label] == actual[valid_label]).mean()) if int(valid_label.sum()) > 0 else None
    logloss = _calc_multiclass_logloss_from_df(work.loc[valid_label].copy()) if int(valid_label.sum()) > 0 else None
    brier = _calc_multiclass_brier_from_df(work.loc[valid_label].copy()) if int(valid_label.sum()) > 0 else None
    return {
        "matches": matches,
        "hit_rate_argmax": hit_rate_argmax,
        "logloss": logloss,
        "brier": brier,
        "count_maxp_gt_060": count_gt,
        "count_maxp_050_060": count_mid,
        "count_maxp_lt_050": count_lt,
        "ratio_maxp_gt_060": float(count_gt / matches) if matches > 0 else 0.0,
        "ratio_maxp_050_060": float(count_mid / matches) if matches > 0 else 0.0,
        "ratio_maxp_lt_050": float(count_lt / matches) if matches > 0 else 0.0,
    }


def _calc_profile_scan_metrics(backtest_path):
    if not backtest_path or not os.path.exists(backtest_path):
        return []
    try:
        df = pd.read_csv(backtest_path)
    except Exception:
        return []
    if df.empty:
        return []
    rows = []
    overall = _calc_profile_scan_metrics_from_df(df)
    rows.append({"league": str(LEAGUE).upper(), **overall})
    if "league" in df.columns:
        for lg, part in df.groupby("league", dropna=False, sort=True):
            metrics = _calc_profile_scan_metrics_from_df(part.copy())
            rows.append({"league": str(lg).upper(), **metrics})
    out = []
    seen = set()
    for row in rows:
        key = str(row.get("league", "")).upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def apply_draw_candidate_flags(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    prob_cols = None
    for c in [("prob_home_win", "prob_draw", "prob_away_win"), ("prob_home", "prob_draw", "prob_away")]:
        if set(c).issubset(out.columns):
            prob_cols = c
            break
    if prob_cols is None:
        out["draw_candidate_flag"] = False
        out["draw_candidate_gap"] = pd.NA
        out["draw_candidate_prob_draw"] = pd.NA
        out["draw_candidate_rule_result"] = pd.NA
        out["draw_candidate_reason"] = ""
        return out
    ph = pd.to_numeric(out[prob_cols[0]], errors="coerce")
    pdw = pd.to_numeric(out[prob_cols[1]], errors="coerce")
    pa = pd.to_numeric(out[prob_cols[2]], errors="coerce")
    max_other = pd.concat([ph, pa], axis=1).max(axis=1)
    gap = max_other - pdw
    flag = pdw.ge(float(DRAW_CANDIDATE_PROB_MIN)) & gap.le(float(DRAW_CANDIDATE_GAP_MAX))
    out["draw_candidate_gap"] = gap
    out["draw_candidate_prob_draw"] = pdw
    out["draw_candidate_flag"] = flag.fillna(False)
    argmax_pred = _calc_argmax_result_from_probs(out)
    out["draw_candidate_rule_result"] = np.where(out["draw_candidate_flag"], "D", argmax_pred)
    out["draw_candidate_reason"] = np.where(
        out["draw_candidate_flag"],
        f"prob_draw>={DRAW_CANDIDATE_PROB_MIN:.2f} & gap<={DRAW_CANDIDATE_GAP_MAX:.2f}",
        "",
    )
    return out


def save_draw_diagnostics(df, league, season_year):
    if df is None or df.empty:
        return None, None
    os.makedirs(PROFILE_SCAN_DIR, exist_ok=True)
    work = apply_draw_candidate_flags(df)
    rows = []
    for lg, part in work.groupby("league", dropna=False, sort=True) if "league" in work.columns else [(str(league).upper(), work)]:
        actual = part.get("actual_result", pd.Series(index=part.index, dtype="object")).astype(str).str.upper()
        pred = part.get("predicted_result", pd.Series(index=part.index, dtype="object")).astype(str).str.upper()
        candidate_pred = pd.Series(part.get("draw_candidate_rule_result", pd.Series(index=part.index, dtype="object"))).astype(str).str.upper()
        dmask = actual.eq("D")
        valid = actual.isin(["H", "D", "A"])
        row = {
            "league": str(lg).upper(),
            "matches": int(len(part)),
            "actual_d_count": int(dmask.sum()),
            "predicted_d_count": int(pred.eq("D").sum()),
            "draw_candidate_count": int(pd.Series(part.get("draw_candidate_flag", False)).fillna(False).astype(bool).sum()),
            "actual_d_hit_argmax": int((dmask & pred.eq("D")).sum()),
            "actual_d_hit_candidate_rule": int((dmask & candidate_pred.eq("D")).sum()),
            "hit_rate_argmax": float((pred[valid] == actual[valid]).mean()) if int(valid.sum()) > 0 else None,
            "hit_rate_candidate_rule": float((candidate_pred[valid] == actual[valid]).mean()) if int(valid.sum()) > 0 else None,
            "avg_prob_draw_on_actual_d": float(pd.to_numeric(part.loc[dmask, "prob_draw"], errors="coerce").mean()) if int(dmask.sum()) > 0 else None,
            "avg_gap_on_actual_d": float(pd.to_numeric(part.loc[dmask, "draw_candidate_gap"], errors="coerce").mean()) if int(dmask.sum()) > 0 else None,
            "draw_candidate_prob_min": float(DRAW_CANDIDATE_PROB_MIN),
            "draw_candidate_gap_max": float(DRAW_CANDIDATE_GAP_MAX),
        }
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(PROFILE_SCAN_DIR, f"draw_diagnostics_{str(league).lower()}_{season_year}.csv")
    detail_cols = [c for c in [
        "match_id", "league", "節", "home_team", "away_team", "actual_result", "predicted_result",
        "draw_candidate_rule_result", "draw_candidate_flag", "draw_candidate_reason",
        "prob_home_win", "prob_draw", "prob_away_win", "draw_candidate_gap",
    ] if c in work.columns]
    detail_df = work.loc[work["draw_candidate_flag"].fillna(False), detail_cols].copy()
    detail_path = os.path.join(PROFILE_SCAN_DIR, f"draw_candidates_{str(league).lower()}_{season_year}.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"[DRAW_DIAG] summary={summary_path} detail={detail_path}")
    return summary_path, detail_path


def apply_match_type_flags(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    pressure_cols_bool = [
        "match_context_title_race_home",
        "match_context_title_race_away",
        "match_context_acl_race_home",
        "match_context_acl_race_away",
        "match_context_acl_border_home",
        "match_context_acl_border_away",
        "match_context_promotion_race_home",
        "match_context_promotion_race_away",
        "match_context_po_race_home",
        "match_context_po_race_away",
        "match_context_survival_race_home",
        "match_context_survival_race_away",
        "match_context_relegation_escape_home",
        "match_context_relegation_escape_away",
        "match_context_top_direct_duel",
        "match_context_bottom_direct_duel",
    ]
    pressure_cols_misc = [
        "match_context_zone_home",
        "match_context_zone_away",
        "match_context_pressure_score_home",
        "match_context_pressure_score_away",
    ]
    required = {
        "elo_diff_for_prob",
        "stats_ゴール期待値_home",
        "stats_ゴール期待値_away",
        "rankmot_rank_latest_home",
        "rankmot_rank_latest_away",
    }
    if not required.issubset(out.columns):
        out["match_type"] = "unknown"
        out["match_type_signal_conflict"] = False
        out["match_type_home_signal_count"] = 0
        out["match_type_away_signal_count"] = 0
        out["match_type_xg_diff"] = pd.NA
        out["match_type_rank_gap"] = pd.NA
        out["match_type_sig_away_elo"] = False
        out["match_type_sig_away_xg"] = False
        out["match_type_sig_away_rank"] = False
        out["match_type_sig_home_elo"] = False
        out["match_type_sig_home_xg"] = False
        out["match_type_sig_home_rank"] = False
        for c in pressure_cols_bool:
            out[c] = False
        out["match_context_zone_home"] = ""
        out["match_context_zone_away"] = ""
        out["match_context_pressure_score_home"] = 0.0
        out["match_context_pressure_score_away"] = 0.0
        return out

    elo = pd.to_numeric(out["elo_diff_for_prob"], errors="coerce")
    xg_diff = pd.to_numeric(out["stats_ゴール期待値_home"], errors="coerce") - pd.to_numeric(out["stats_ゴール期待値_away"], errors="coerce")
    rank_gap = pd.to_numeric(out["rankmot_rank_latest_home"], errors="coerce") - pd.to_numeric(out["rankmot_rank_latest_away"], errors="coerce")

    sig_away_elo = elo <= -40.0
    sig_away_xg = xg_diff <= -1.0
    sig_away_rank = rank_gap >= 3.0
    sig_home_elo = elo >= 40.0
    sig_home_xg = xg_diff >= 1.0
    sig_home_rank = rank_gap <= -3.0

    away_signal_count = (
        pd.concat([sig_away_elo, sig_away_xg, sig_away_rank], axis=1).fillna(False).sum(axis=1).astype(int)
    )
    home_signal_count = (
        pd.concat([sig_home_elo, sig_home_xg, sig_home_rank], axis=1).fillna(False).sum(axis=1).astype(int)
    )

    close_match = elo.abs().le(20.0) & xg_diff.abs().le(1.0) & rank_gap.abs().le(2.0)
    away_strong = away_signal_count.ge(2)
    home_strong = home_signal_count.ge(2)
    signal_conflict = (
        away_signal_count.ge(1) & home_signal_count.ge(1) & ~away_strong & ~home_strong & ~close_match
    )

    out["match_type"] = "neutral"
    out.loc[close_match.fillna(False), "match_type"] = "close_match"
    out.loc[signal_conflict.fillna(False), "match_type"] = "signal_conflict"
    out.loc[away_strong.fillna(False), "match_type"] = "away_strong"
    out.loc[home_strong.fillna(False), "match_type"] = "home_strong"

    out["match_type_signal_conflict"] = signal_conflict.fillna(False)
    out["match_type_home_signal_count"] = home_signal_count
    out["match_type_away_signal_count"] = away_signal_count
    out["match_type_xg_diff"] = xg_diff
    out["match_type_rank_gap"] = rank_gap
    out["match_type_sig_away_elo"] = sig_away_elo.fillna(False)
    out["match_type_sig_away_xg"] = sig_away_xg.fillna(False)
    out["match_type_sig_away_rank"] = sig_away_rank.fillna(False)
    out["match_type_sig_home_elo"] = sig_home_elo.fillna(False)
    out["match_type_sig_home_xg"] = sig_home_xg.fillna(False)
    out["match_type_sig_home_rank"] = sig_home_rank.fillna(False)
    home_rank = pd.to_numeric(out["rankmot_rank_latest_home"], errors="coerce")
    away_rank = pd.to_numeric(out["rankmot_rank_latest_away"], errors="coerce")
    team_count_raw = pd.concat([home_rank, away_rank], axis=0).max()
    team_count = int(team_count_raw) if pd.notna(team_count_raw) else 20
    league_key = str(out.get("league", pd.Series([LEAGUE])).iloc[0]).strip().lower()
    relegation_bottom_n = J2_RELEGATION_RISK_BOTTOM_N if league_key == "j2" else J1_RELEGATION_RISK_BOTTOM_N
    relegation_threshold = max(1, int(team_count - relegation_bottom_n + 1))
    out["match_type_team_count"] = team_count
    out["match_type_title_race_home"] = home_rank.le(TITLE_RACE_RANK_MAX).fillna(False)
    out["match_type_title_race_away"] = away_rank.le(TITLE_RACE_RANK_MAX).fillna(False)
    out["match_type_relegation_risk_home"] = home_rank.ge(relegation_threshold).fillna(False)
    out["match_type_relegation_risk_away"] = away_rank.ge(relegation_threshold).fillna(False)
    out["match_type_midtable_home"] = (~out["match_type_title_race_home"] & ~out["match_type_relegation_risk_home"]).fillna(False)
    out["match_type_midtable_away"] = (~out["match_type_title_race_away"] & ~out["match_type_relegation_risk_away"]).fillna(False)

    home_points = pd.to_numeric(out.get("rankmot_points_latest_home", pd.Series(np.nan, index=out.index)), errors="coerce")
    away_points = pd.to_numeric(out.get("rankmot_points_latest_away", pd.Series(np.nan, index=out.index)), errors="coerce")
    home_motivation = pd.to_numeric(out.get("rankmot_motivation_score_5w_home", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(0.0)
    away_motivation = pd.to_numeric(out.get("rankmot_motivation_score_5w_away", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(0.0)
    points_gap_abs = (home_points - away_points).abs()

    if league_key == "j1":
        title_home = home_rank.le(J1_TITLE_PUSH_RANK_MAX).fillna(False)
        title_away = away_rank.le(J1_TITLE_PUSH_RANK_MAX).fillna(False)
        acl_race_home = home_rank.le(J1_ACL_RACE_RANK_MAX).fillna(False) & ~title_home
        acl_race_away = away_rank.le(J1_ACL_RACE_RANK_MAX).fillna(False) & ~title_away
        acl_border_home = home_rank.sub(J1_ACL_LINE_RANK).abs().le(1.0).fillna(False) & home_motivation.ge(PRESSURE_BORDER_MOTIVATION_MIN)
        acl_border_away = away_rank.sub(J1_ACL_LINE_RANK).abs().le(1.0).fillna(False) & away_motivation.ge(PRESSURE_BORDER_MOTIVATION_MIN)
        promotion_home = pd.Series(False, index=out.index)
        promotion_away = pd.Series(False, index=out.index)
        po_home = pd.Series(False, index=out.index)
        po_away = pd.Series(False, index=out.index)
    else:
        title_home = pd.Series(False, index=out.index)
        title_away = pd.Series(False, index=out.index)
        acl_race_home = pd.Series(False, index=out.index)
        acl_race_away = pd.Series(False, index=out.index)
        acl_border_home = pd.Series(False, index=out.index)
        acl_border_away = pd.Series(False, index=out.index)
        promotion_home = home_rank.le(J2_PROMOTION_RACE_RANK_MAX).fillna(False)
        promotion_away = away_rank.le(J2_PROMOTION_RACE_RANK_MAX).fillna(False)
        po_home = home_rank.ge(J2_PROMOTION_LINE_RANK + 1).fillna(False) & home_rank.le(J2_PO_RACE_RANK_MAX).fillna(False)
        po_away = away_rank.ge(J2_PROMOTION_LINE_RANK + 1).fillna(False) & away_rank.le(J2_PO_RACE_RANK_MAX).fillna(False)

    survival_home = home_rank.ge(max(1, relegation_threshold - 1)).fillna(False)
    survival_away = away_rank.ge(max(1, relegation_threshold - 1)).fillna(False)
    relegation_escape_home = home_rank.sub(relegation_threshold).abs().le(1.0).fillna(False) & home_motivation.ge(PRESSURE_BORDER_MOTIVATION_MIN)
    relegation_escape_away = away_rank.sub(relegation_threshold).abs().le(1.0).fillna(False) & away_motivation.ge(PRESSURE_BORDER_MOTIVATION_MIN)
    top_direct_duel = (
        (
            (title_home & title_away)
            | (acl_race_home & acl_race_away)
            | (promotion_home & promotion_away)
            | (po_home & po_away)
        )
        & rank_gap.abs().le(3.0)
        & points_gap_abs.le(PRESSURE_DIRECT_POINTS_GAP_MAX)
    ).fillna(False)
    bottom_direct_duel = (
        survival_home
        & survival_away
        & rank_gap.abs().le(3.0)
        & points_gap_abs.le(PRESSURE_DIRECT_POINTS_GAP_MAX)
    ).fillna(False)

    out["match_context_title_race_home"] = title_home
    out["match_context_title_race_away"] = title_away
    out["match_context_acl_race_home"] = acl_race_home
    out["match_context_acl_race_away"] = acl_race_away
    out["match_context_acl_border_home"] = acl_border_home
    out["match_context_acl_border_away"] = acl_border_away
    out["match_context_promotion_race_home"] = promotion_home
    out["match_context_promotion_race_away"] = promotion_away
    out["match_context_po_race_home"] = po_home
    out["match_context_po_race_away"] = po_away
    out["match_context_survival_race_home"] = survival_home
    out["match_context_survival_race_away"] = survival_away
    out["match_context_relegation_escape_home"] = relegation_escape_home
    out["match_context_relegation_escape_away"] = relegation_escape_away
    out["match_context_top_direct_duel"] = top_direct_duel
    out["match_context_bottom_direct_duel"] = bottom_direct_duel

    def _pressure_zone(rank_s: pd.Series, title_s: pd.Series, acl_s: pd.Series, acl_border_s: pd.Series,
                       promo_s: pd.Series, po_s: pd.Series, survival_s: pd.Series, rel_escape_s: pd.Series) -> pd.Series:
        zone = pd.Series("midtable", index=out.index, dtype="object")
        zone.loc[survival_s.fillna(False)] = "survival_race"
        zone.loc[rel_escape_s.fillna(False)] = "relegation_escape"
        zone.loc[po_s.fillna(False)] = "po_race"
        zone.loc[promo_s.fillna(False)] = "promotion_race"
        zone.loc[acl_s.fillna(False)] = "acl_race"
        zone.loc[acl_border_s.fillna(False)] = "acl_border"
        zone.loc[title_s.fillna(False)] = "title_race"
        return zone

    out["match_context_zone_home"] = _pressure_zone(home_rank, title_home, acl_race_home, acl_border_home, promotion_home, po_home, survival_home, relegation_escape_home)
    out["match_context_zone_away"] = _pressure_zone(away_rank, title_away, acl_race_away, acl_border_away, promotion_away, po_away, survival_away, relegation_escape_away)

    out["match_context_pressure_score_home"] = (
        title_home.astype(float) * 1.00
        + acl_race_home.astype(float) * 0.80
        + acl_border_home.astype(float) * 0.70
        + promotion_home.astype(float) * 0.90
        + po_home.astype(float) * 0.65
        + survival_home.astype(float) * 0.75
        + relegation_escape_home.astype(float) * 0.85
        + top_direct_duel.astype(float) * 0.20
        + bottom_direct_duel.astype(float) * 0.20
        + home_motivation.clip(lower=0.0) * 0.15
    ).clip(0.0, 2.0)
    out["match_context_pressure_score_away"] = (
        title_away.astype(float) * 1.00
        + acl_race_away.astype(float) * 0.80
        + acl_border_away.astype(float) * 0.70
        + promotion_away.astype(float) * 0.90
        + po_away.astype(float) * 0.65
        + survival_away.astype(float) * 0.75
        + relegation_escape_away.astype(float) * 0.85
        + top_direct_duel.astype(float) * 0.20
        + bottom_direct_duel.astype(float) * 0.20
        + away_motivation.clip(lower=0.0) * 0.15
    ).clip(0.0, 2.0)

    flab_attack = pd.to_numeric(out.get("flab_chance_shot_conversion_diff"), errors="coerce")
    flab_allow = pd.to_numeric(out.get("flab_chance_allowed_shot_conversion_diff"), errors="coerce")
    flab_xgf = pd.to_numeric(out.get("flab_expected_for_xg_diff"), errors="coerce")
    flab_xga = pd.to_numeric(out.get("flab_expected_against_xg_diff"), errors="coerce")
    flab_build = pd.to_numeric(out.get("flab_chance_build_rate_diff"), errors="coerce")
    flab_allow_build = pd.to_numeric(out.get("flab_chance_allowed_build_rate_diff"), errors="coerce")
    flab_poss = pd.to_numeric(out.get("flab_possession_rate_diff"), errors="coerce")
    flab_cbp = pd.to_numeric(out.get("flab_possession_attack_cbp_diff"), errors="coerce")

    flab_available = (
        flab_attack.notna()
        | flab_allow.notna()
        | flab_xgf.notna()
        | flab_xga.notna()
        | flab_build.notna()
        | flab_allow_build.notna()
        | flab_poss.notna()
        | flab_cbp.notna()
    )
    flab_matchup_edge = (
        flab_attack.fillna(0.0)
        + flab_allow.fillna(0.0)
        + (flab_xgf.fillna(0.0) * 2.0)
        + (flab_xga.fillna(0.0) * 1.5)
        + (flab_cbp.fillna(0.0) * 0.25)
    )
    flab_home_matchup = (
        (
            flab_attack.ge(4.0)
            & flab_allow.ge(1.5)
        )
        | (
            flab_xgf.ge(0.20)
            & flab_xga.ge(0.12)
        )
        | flab_matchup_edge.ge(6.0)
    ).fillna(False)
    flab_away_matchup = (
        (
            flab_attack.le(-4.0)
            & flab_allow.le(-1.5)
        )
        | (
            flab_xgf.le(-0.20)
            & flab_xga.le(-0.12)
        )
        | flab_matchup_edge.le(-6.0)
    ).fillna(False)
    flab_style_conflict = (
        (
            flab_attack.ge(4.0)
            & flab_allow.le(-1.5)
        )
        | (
            flab_attack.le(-4.0)
            & flab_allow.ge(1.5)
        )
        | (
            flab_xgf.ge(0.20)
            & flab_xga.le(-0.12)
        )
        | (
            flab_xgf.le(-0.20)
            & flab_xga.ge(0.12)
        )
    ).fillna(False)
    flab_low_event = (
        flab_build.abs().le(3.0)
        & flab_allow_build.abs().le(3.0)
        & flab_poss.abs().le(4.0)
        & flab_xgf.abs().le(0.18)
        & flab_xga.abs().le(0.18)
    ).fillna(False)

    out["match_type_lab_available"] = flab_available.fillna(False)
    out["match_type_lab_matchup_edge"] = flab_matchup_edge.where(flab_available, np.nan)
    out["match_type_lab_home_matchup"] = flab_home_matchup
    out["match_type_lab_away_matchup"] = flab_away_matchup
    out["match_type_lab_style_conflict"] = flab_style_conflict
    out["match_type_lab_low_event"] = flab_low_event
    return out


def _argmax_hda_label(prob_home, prob_draw, prob_away):
    vals = np.array([prob_home, prob_draw, prob_away], dtype=float)
    idx = int(np.nanargmax(vals))
    return ["H", "D", "A"][idx]


def _calc_predicted_result_main(row):
    ph = float(pd.to_numeric(row.get("prob_home_win"), errors="coerce"))
    pdw = float(pd.to_numeric(row.get("prob_draw"), errors="coerce"))
    pa = float(pd.to_numeric(row.get("prob_away_win"), errors="coerce"))
    return _argmax_hda_label(ph, pdw, pa)


def _symbol_result_argmax(prob_home, prob_draw, prob_away):
    vals = np.array([prob_home, prob_draw, prob_away], dtype=float)
    idx = int(np.nanargmax(vals))
    return ["1", "0", "2"][idx]


def _build_match_type_meta(row):
    match_type = str(row.get("match_type", "unknown") or "unknown").strip()
    ph = float(pd.to_numeric(row.get("prob_home_win"), errors="coerce"))
    pdw = float(pd.to_numeric(row.get("prob_draw"), errors="coerce"))
    pa = float(pd.to_numeric(row.get("prob_away_win"), errors="coerce"))
    elo = float(pd.to_numeric(row.get("elo_diff_for_prob"), errors="coerce"))
    xg_diff = float(pd.to_numeric(row.get("match_type_xg_diff"), errors="coerce"))
    rank_gap = float(pd.to_numeric(row.get("match_type_rank_gap"), errors="coerce"))
    draw_gap = float(max(ph, pa) - pdw)
    draw_risk = bool(pdw >= 0.285 and draw_gap <= 0.055)
    flags = []
    if draw_risk:
        flags.append("draw_risk")
    if match_type == "close_match":
        flags.append("close_match")
    if match_type == "away_strong":
        flags.append("away_strong")
    if match_type == "home_strong":
        flags.append("home_strong")
    if match_type == "signal_conflict":
        flags.append("signal_conflict")
    if bool(row.get("match_type_lab_home_matchup", False)):
        flags.append("lab_home_matchup")
    if bool(row.get("match_type_lab_away_matchup", False)):
        flags.append("lab_away_matchup")
    if bool(row.get("match_type_lab_style_conflict", False)):
        flags.append("lab_style_conflict")
    if bool(row.get("match_type_lab_low_event", False)):
        flags.append("lab_low_event")
    if bool(row.get("match_type_title_race_home", False)):
        flags.append("home_title_race")
    if bool(row.get("match_type_title_race_away", False)):
        flags.append("away_title_race")
    if bool(row.get("match_type_relegation_risk_home", False)):
        flags.append("home_relegation_risk")
    if bool(row.get("match_type_relegation_risk_away", False)):
        flags.append("away_relegation_risk")
    metrics = _compute_match_type_pressures(row)
    if metrics["home_adverse_score"] >= 1.20:
        flags.append("home_adverse")
    if metrics["away_adverse_score"] >= 1.20:
        flags.append("away_adverse")
    if metrics["match_intensity_score"] >= 0.60:
        flags.append("high_intensity")
    if metrics["focus_stability_score"] >= 0.62:
        flags.append("stable_focus")
    if bool(metrics["derby_match"]):
        flags.append("derby")
    primary = match_type
    if draw_risk and primary not in {"signal_conflict", "away_strong", "home_strong"}:
        primary = "draw_risk"
    if primary in {"neutral", "draw_risk", "close_match"} and bool(row.get("match_type_lab_style_conflict", False)):
        primary = "lab_style_conflict"
    elif primary in {"neutral", "draw_risk", "close_match"} and bool(row.get("match_type_lab_low_event", False)):
        primary = "lab_low_event"
    elif primary == "neutral" and bool(row.get("match_type_lab_home_matchup", False)):
        primary = "lab_home_matchup"
    elif primary == "neutral" and bool(row.get("match_type_lab_away_matchup", False)):
        primary = "lab_away_matchup"
    lab_edge = pd.to_numeric(row.get("match_type_lab_matchup_edge"), errors="coerce")
    reason = (
        f"type={match_type}; flags={','.join(flags) if flags else 'none'}; "
        f"elo={elo:.3f}; xg_diff={xg_diff:.3f}; rank_gap={rank_gap:.3f}; "
        f"prob_draw={pdw:.3f}; draw_gap={draw_gap:.3f}; "
        f"lab_edge={'' if pd.isna(lab_edge) else f'{float(lab_edge):.3f}'}; "
        f"adv_home={metrics['home_adverse_score']:.2f}; adv_away={metrics['away_adverse_score']:.2f}; "
        f"intensity={metrics['match_intensity_score']:.2f}; focus={metrics['focus_stability_score']:.2f}; "
        f"home_ctx={'title' if bool(row.get('match_type_title_race_home', False)) else ('relegation' if bool(row.get('match_type_relegation_risk_home', False)) else 'mid')}; "
        f"away_ctx={'title' if bool(row.get('match_type_title_race_away', False)) else ('relegation' if bool(row.get('match_type_relegation_risk_away', False)) else 'mid')}"
    )
    return {
        "match_type_primary": primary,
        "match_type_flags": ",".join(flags),
        "match_type_reason": reason,
        "draw_risk_flag": draw_risk,
        "draw_gap": draw_gap,
        "match_intensity_score": metrics["match_intensity_score"],
        "focus_stability_score": metrics["focus_stability_score"],
    }


def _compute_match_type_pressures(row):
    elo = float(pd.to_numeric(row.get("elo_diff_for_prob"), errors="coerce"))
    xg_diff = float(pd.to_numeric(row.get("match_type_xg_diff"), errors="coerce"))
    rank_gap = float(pd.to_numeric(row.get("match_type_rank_gap"), errors="coerce"))
    ph = float(pd.to_numeric(row.get("prob_home_win"), errors="coerce"))
    pdw = float(pd.to_numeric(row.get("prob_draw"), errors="coerce"))
    pa = float(pd.to_numeric(row.get("prob_away_win"), errors="coerce"))
    home_fatigue = float(pd.to_numeric(row.get("home_total_fatigue_score"), errors="coerce"))
    away_fatigue = float(pd.to_numeric(row.get("away_total_fatigue_score"), errors="coerce"))
    home_acl_fatigue = float(pd.to_numeric(row.get("home_acl_fatigue"), errors="coerce"))
    away_acl_fatigue = float(pd.to_numeric(row.get("away_acl_fatigue"), errors="coerce"))
    home_acl_days_since = float(pd.to_numeric(row.get("home_acl_days_since"), errors="coerce"))
    away_acl_days_since = float(pd.to_numeric(row.get("away_acl_days_since"), errors="coerce"))
    home_absence = float(pd.to_numeric(row.get("absence_effective_total_home"), errors="coerce"))
    away_absence = float(pd.to_numeric(row.get("absence_effective_total_away"), errors="coerce"))
    derby_match = bool(row.get("derby_match", False))
    derby_intensity = float(pd.to_numeric(row.get("derby_intensity"), errors="coerce"))
    derby_stall_bias = float(pd.to_numeric(row.get("derby_stall_bias"), errors="coerce"))
    derby_entropy_bias = float(pd.to_numeric(row.get("derby_entropy_bias"), errors="coerce"))
    home_rankmot = float(pd.to_numeric(row.get("rankmot_motivation_score_3w_home"), errors="coerce"))
    away_rankmot = float(pd.to_numeric(row.get("rankmot_motivation_score_3w_away"), errors="coerce"))
    home_mgmt_mot = float(pd.to_numeric(row.get("management_motivation_score_home"), errors="coerce"))
    away_mgmt_mot = float(pd.to_numeric(row.get("management_motivation_score_away"), errors="coerce"))
    is_rain = bool(row.get("is_rain")) if pd.notna(row.get("is_rain")) else False
    is_heavy_rain = bool(row.get("is_heavy_rain")) if pd.notna(row.get("is_heavy_rain")) else False
    is_strong_wind = bool(row.get("is_strong_wind")) if pd.notna(row.get("is_strong_wind")) else False
    weather_penalty = adverse_weather_penalty(is_rain, is_heavy_rain, is_strong_wind)

    def _safe_mean(values, default=np.nan):
        vals = [float(v) for v in values if math.isfinite(v)]
        if not vals:
            return float(default)
        return float(sum(vals) / len(vals))

    home_strength = (
        max(elo, 0.0) / 40.0
        + max(xg_diff, 0.0) / 1.25
        + max(-rank_gap, 0.0) / 3.0
    )
    away_strength = (
        max(-elo, 0.0) / 40.0
        + max(-xg_diff, 0.0) / 1.25
        + max(rank_gap, 0.0) / 3.0
    )
    structural_delta = home_strength - away_strength
    home_adverse_score = max(home_fatigue - away_fatigue, 0.0) * 0.12 + home_absence * 8.0 + weather_penalty
    away_adverse_score = max(away_fatigue - home_fatigue, 0.0) * 0.12 + away_absence * 8.0 + weather_penalty
    home_acl_draw_pressure = 0.0
    away_acl_draw_pressure = 0.0
    if math.isfinite(home_acl_fatigue) and home_acl_fatigue >= ACL_DRAW_MIN_FATIGUE:
        home_acl_draw_pressure = min(
            ACL_DRAW_EDGE_BASE + (home_acl_fatigue * ACL_DRAW_EDGE_PER_FATIGUE),
            ACL_DRAW_EDGE_CAP,
        )
        if math.isfinite(home_acl_days_since) and int(effective_days := ACL_EFFECTIVE_DAYS) < home_acl_days_since <= int(ACL_SECOND_WINDOW_DAYS):
            home_acl_draw_pressure = min(home_acl_draw_pressure + ACL_DRAW_SECOND_WINDOW_BONUS, ACL_DRAW_EDGE_CAP)
    if math.isfinite(away_acl_fatigue) and away_acl_fatigue >= ACL_DRAW_MIN_FATIGUE:
        away_acl_draw_pressure = min(
            ACL_DRAW_EDGE_BASE + (away_acl_fatigue * ACL_DRAW_EDGE_PER_FATIGUE),
            ACL_DRAW_EDGE_CAP,
        )
        if math.isfinite(away_acl_days_since) and int(effective_days := ACL_EFFECTIVE_DAYS) < away_acl_days_since <= int(ACL_SECOND_WINDOW_DAYS):
            away_acl_draw_pressure = min(away_acl_draw_pressure + ACL_DRAW_SECOND_WINDOW_BONUS, ACL_DRAW_EDGE_CAP)
    title_pressure = (
        (0.18 if bool(row.get("match_type_title_race_home", False)) else 0.0)
        + (0.18 if bool(row.get("match_type_title_race_away", False)) else 0.0)
    )
    relegation_pressure = (
        (0.16 if bool(row.get("match_type_relegation_risk_home", False)) else 0.0)
        + (0.16 if bool(row.get("match_type_relegation_risk_away", False)) else 0.0)
    )
    rank_closeness = _score_small_abs(rank_gap, 4.0)
    motivation_level = _safe_mean(
        [home_rankmot, away_rankmot, home_mgmt_mot, away_mgmt_mot],
        default=0.35,
    )
    motivation_score = _clip01((motivation_level + 0.20) / 1.20)
    derby_score = _clip01(derby_intensity) if derby_match and math.isfinite(derby_intensity) else 0.0
    match_intensity_score = _clip01(
        (0.42 * derby_score)
        + (0.20 * title_pressure)
        + (0.18 * relegation_pressure)
        + (0.10 * rank_closeness)
        + (0.10 * motivation_score)
    )
    fatigue_load = _safe_mean([home_fatigue, away_fatigue], default=4.0)
    acl_load = _safe_mean([home_acl_fatigue, away_acl_fatigue], default=0.0)
    absence_load = _safe_mean([home_absence, away_absence], default=0.05)
    focus_stability_score = _clip01(
        1.0
        - (
            0.35 * _clip01(fatigue_load / 10.0)
            + 0.20 * _clip01(acl_load / 8.0)
            + 0.25 * _clip01(absence_load / 0.18)
            + 0.20 * _clip01(weather_penalty / 1.70)
        )
    )
    return {
        "elo": elo,
        "xg_diff": xg_diff,
        "rank_gap": rank_gap,
        "prob_home": ph,
        "prob_draw": pdw,
        "prob_away": pa,
        "home_fatigue": home_fatigue,
        "away_fatigue": away_fatigue,
        "home_acl_fatigue": home_acl_fatigue,
        "away_acl_fatigue": away_acl_fatigue,
        "home_acl_days_since": home_acl_days_since,
        "away_acl_days_since": away_acl_days_since,
        "home_absence": home_absence,
        "away_absence": away_absence,
        "draw_gap": float(pd.to_numeric(row.get("draw_gap"), errors="coerce")),
        "home_strength": home_strength,
        "away_strength": away_strength,
        "structural_delta": structural_delta,
        "home_title_race": bool(row.get("match_type_title_race_home", False)),
        "away_title_race": bool(row.get("match_type_title_race_away", False)),
        "home_relegation_risk": bool(row.get("match_type_relegation_risk_home", False)),
        "away_relegation_risk": bool(row.get("match_type_relegation_risk_away", False)),
        "home_adverse_score": home_adverse_score,
        "away_adverse_score": away_adverse_score,
        "home_acl_draw_pressure": home_acl_draw_pressure,
        "away_acl_draw_pressure": away_acl_draw_pressure,
        "weather_penalty": weather_penalty,
        "derby_match": derby_match,
        "derby_intensity": derby_intensity,
        "derby_stall_bias": derby_stall_bias,
        "derby_entropy_bias": derby_entropy_bias,
        "motivation_level": motivation_level,
        "motivation_score": motivation_score,
        "rank_closeness": rank_closeness,
        "match_intensity_score": match_intensity_score,
        "focus_stability_score": focus_stability_score,
    }


def _compute_prob_shape(ph: float, pdw: float, pa: float):
    vals = [("H", float(ph)), ("D", float(pdw)), ("A", float(pa))]
    ranked = sorted(vals, key=lambda x: x[1], reverse=True)
    total = max(float(ph) + float(pdw) + float(pa), 1e-12)
    probs = [max(float(ph), 1e-12) / total, max(float(pdw), 1e-12) / total, max(float(pa), 1e-12) / total]
    entropy = -sum(p * math.log(p) for p in probs) / math.log(3.0)
    return {
        "best_label": ranked[0][0],
        "second_label": ranked[1][0],
        "third_label": ranked[2][0],
        "top_gap": ranked[0][1] - ranked[1][1],
        "draw_is_second": ranked[1][0] == "D",
        "entropy_norm": entropy,
    }


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _normalize_hda_triplet(prob_home, prob_draw, prob_away, fallback=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)):
    vals = np.array([prob_home, prob_draw, prob_away], dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, None)
    total = float(vals.sum())
    if total <= 1e-12:
        return tuple(float(x) for x in fallback)
    vals = vals / total
    return float(vals[0]), float(vals[1]), float(vals[2])


def apply_lab_probability_blend(df, stage_label="PRED"):
    if df is None or df.empty:
        return df
    out = df.copy()
    required = {"prob_home_win", "prob_draw", "prob_away_win", "lab_mix_home", "lab_mix_draw", "lab_mix_away"}
    if not required.issubset(out.columns):
        return out

    for src, dst in [
        ("prob_home_win", "prob_elo_home"),
        ("prob_draw", "prob_elo_draw"),
        ("prob_away_win", "prob_elo_away"),
        ("lab_mix_home", "prob_lab_home"),
        ("lab_mix_draw", "prob_lab_draw"),
        ("lab_mix_away", "prob_lab_away"),
    ]:
        out[dst] = pd.to_numeric(out[src], errors="coerce")

    base_sum = out[["prob_elo_home", "prob_elo_draw", "prob_elo_away"]].sum(axis=1)
    lab_sum = out[["prob_lab_home", "prob_lab_draw", "prob_lab_away"]].sum(axis=1)
    lab_available = out[["prob_lab_home", "prob_lab_draw", "prob_lab_away"]].notna().all(axis=1) & lab_sum.gt(0.0)
    if not ENABLE_LAB_PROB_BLEND:
        lab_available = pd.Series(False, index=out.index)
    out["lab_prob_blend_enabled"] = bool(ENABLE_LAB_PROB_BLEND)
    out["lab_prob_blend_available"] = lab_available.fillna(False)

    if not bool(np.any(lab_available.to_numpy())):
        out["lab_prob_blend_alpha"] = 0.0
        out["confidence_lab"] = 0.0
        for src, dst in [
            ("prob_elo_home", "prob_final_home"),
            ("prob_elo_draw", "prob_final_draw"),
            ("prob_elo_away", "prob_final_away"),
        ]:
            out[dst] = out[src]
        out["prob_blend_home"] = out["prob_final_home"]
        out["prob_blend_draw"] = out["prob_final_draw"]
        out["prob_blend_away"] = out["prob_final_away"]
        return out

    lab_vals = out[["prob_lab_home", "prob_lab_draw", "prob_lab_away"]].fillna(0.0).to_numpy(dtype=float)
    base_vals = out[["prob_elo_home", "prob_elo_draw", "prob_elo_away"]].fillna(0.0).to_numpy(dtype=float)
    lab_vals = np.clip(lab_vals, 0.0, None)
    base_vals = np.clip(base_vals, 0.0, None)

    lab_ranked = np.sort(lab_vals, axis=1)
    lab_top_gap = lab_ranked[:, 2] - lab_ranked[:, 1]
    lab_clarity = np.clip(lab_top_gap / 0.20, 0.0, 1.0)
    scenario_entropy = pd.to_numeric(
        out.get("lab_scenario_entropy_score", pd.Series(0.5, index=out.index)),
        errors="coerce",
    ).fillna(0.5).to_numpy(dtype=float)
    matchup_edge = pd.to_numeric(
        out.get("lab_matchup_edge_score", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    draw_support = pd.to_numeric(
        out.get("lab_draw_support_score", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    draw_tension_score = pd.to_numeric(
        out.get("lab_draw_tension_score", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    compactness_score = pd.to_numeric(
        out.get("lab_stall_compactness_score", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    lab_mix_draw = pd.to_numeric(
        out.get("lab_mix_draw", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    matchup_profile = out.get("lab_matchup_profile", pd.Series("", index=out.index)).fillna("").astype(str)
    structural_support = np.maximum(matchup_edge, draw_support)
    confidence = np.clip(
        (0.45 * lab_clarity) + (0.35 * (1.0 - np.clip(scenario_entropy, 0.0, 1.0))) + (0.20 * np.clip(structural_support, 0.0, 1.0)),
        0.0,
        1.0,
    )
    alpha = LAB_PROB_BLEND_ALPHA_BASE + (LAB_PROB_BLEND_ALPHA_GAIN * confidence)
    alpha = np.clip(alpha, LAB_PROB_BLEND_ALPHA_MIN, LAB_PROB_BLEND_ALPHA_MAX)
    alpha = np.where(lab_available.to_numpy(dtype=bool), alpha, 0.0)
    lab_best_idx = np.argmax(lab_vals, axis=1)
    lab_draw_argmax = lab_best_idx == 1
    drawish_profile = matchup_profile.isin(["stall_shape", "low_tempo_draw", "balanced"]).to_numpy(dtype=bool)
    drawish_mask = (
        lab_available.to_numpy(dtype=bool)
        & lab_draw_argmax
        & drawish_profile
        & (lab_mix_draw >= 0.50)
        & (draw_tension_score >= 0.58)
        & (compactness_score >= 0.54)
    )
    alpha = np.where(drawish_mask, np.maximum(alpha * 0.55, LAB_PROB_BLEND_ALPHA_MIN), alpha)
    out["lab_prob_blend_alpha"] = alpha
    out["lab_prob_blend_draw_dampen"] = drawish_mask
    out["confidence_lab"] = confidence

    base_den = np.where(base_sum.to_numpy(dtype=float) > 1e-12, base_sum.to_numpy(dtype=float), 1.0)
    lab_den = np.where(lab_sum.to_numpy(dtype=float) > 1e-12, lab_sum.to_numpy(dtype=float), 1.0)
    base_vals = base_vals / base_den[:, None]
    lab_vals = lab_vals / lab_den[:, None]
    final_vals = ((1.0 - alpha)[:, None] * base_vals) + (alpha[:, None] * lab_vals)
    final_den = np.where(final_vals.sum(axis=1) > 1e-12, final_vals.sum(axis=1), 1.0)
    final_vals = final_vals / final_den[:, None]

    out["prob_blend_home"] = final_vals[:, 0]
    out["prob_blend_draw"] = final_vals[:, 1]
    out["prob_blend_away"] = final_vals[:, 2]
    out["prob_final_home"] = out["prob_blend_home"]
    out["prob_final_draw"] = out["prob_blend_draw"]
    out["prob_final_away"] = out["prob_blend_away"]
    print(
        f"[LAB_PROB_BLEND] stage={stage_label} enabled={int(ENABLE_LAB_PROB_BLEND)} "
        f"available={int(lab_available.sum())}/{len(out)} alpha_mean={float(np.mean(alpha)):.3f} "
        f"draw_dampen={int(drawish_mask.sum())}"
    )
    return out


def _score_small_gap(gap: float, limit: float) -> float:
    if not math.isfinite(gap):
        return 0.0
    return _clip01(1.0 - max(float(gap), 0.0) / max(float(limit), 1e-9))


def _score_small_abs(value: float, limit: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return _clip01(1.0 - abs(float(value)) / max(float(limit), 1e-9))


def _compute_d_scores(row):
    ph = float(pd.to_numeric(row.get("prob_home_win"), errors="coerce"))
    pdw = float(pd.to_numeric(row.get("prob_draw"), errors="coerce"))
    pa = float(pd.to_numeric(row.get("prob_away_win"), errors="coerce"))
    if not (math.isfinite(ph) and math.isfinite(pdw) and math.isfinite(pa)):
        return {
            "d_score_close": 0.0,
            "d_score_stall": 0.0,
            "d_score_total": 0.0,
            "d_combo_close_lowevent": 0.0,
            "d_combo_close_conflict": 0.0,
            "d_combo_drawrisk_stall": 0.0,
            "d_flag_close_lowevent": False,
            "d_flag_close_conflict": False,
            "d_flag_drawrisk_stall": False,
            "d_flag_two_of_three": False,
            "d_flag_j1_close_lowevent": False,
            "d_flag_j1_drawrisk_stall": False,
            "d_flag_j2_second_draw_close": False,
            "d_flag_j2_drawrisk_second": False,
        }

    league = str(row.get("league", "")).strip().lower()
    prob_shape = _compute_prob_shape(ph, pdw, pa)
    draw_gap = float(max(ph, pa) - pdw)
    top_gap = float(prob_shape["top_gap"])
    second_label = str(prob_shape["second_label"])
    best_label = str(prob_shape["best_label"])
    entropy_norm = float(prob_shape["entropy_norm"])

    second_draw_score = 1.0 if second_label == "D" else (0.85 if best_label == "D" else 0.0)
    draw_gap_score = _score_small_gap(draw_gap, 0.080)
    top_gap_score = _score_small_gap(top_gap, 0.080)
    entropy_score = _clip01((entropy_norm - 0.90) / 0.10)
    ha_gap_abs = abs(ph - pa)

    d_score_close = _clip01(
        (0.40 * second_draw_score)
        + (0.30 * draw_gap_score)
        + (0.20 * top_gap_score)
        + (0.10 * entropy_score)
    )

    style_conflict = bool(row.get("match_type_lab_style_conflict"))
    low_event = bool(row.get("match_type_lab_low_event"))
    draw_risk = str(row.get("draw_risk_flag", "")).strip().lower() in {"true", "1"}
    lab_edge = float(pd.to_numeric(row.get("match_type_lab_matchup_edge"), errors="coerce"))
    flab_trial_score_raw = pd.to_numeric(row.get("flab_trial_score"), errors="coerce")
    flab_trial_score = 0.0 if pd.isna(flab_trial_score_raw) else float(flab_trial_score_raw)
    match_intensity_score = float(pd.to_numeric(row.get("match_intensity_score"), errors="coerce"))
    focus_stability_score = float(pd.to_numeric(row.get("focus_stability_score"), errors="coerce"))
    derby_stall_bias = float(pd.to_numeric(row.get("derby_stall_bias"), errors="coerce"))
    derby_entropy_bias = float(pd.to_numeric(row.get("derby_entropy_bias"), errors="coerce"))
    prob_draw_score = _clip01((pdw - 0.24) / 0.12)
    lab_edge_score = _score_small_abs(lab_edge, 8.0)
    flab_score = _clip01((flab_trial_score + 8.0) / 16.0)
    intensity_draw_score = _clip01(match_intensity_score * focus_stability_score)

    d_score_stall = _clip01(
        (0.30 if style_conflict else 0.0)
        + (0.25 if low_event else 0.0)
        + (0.25 * lab_edge_score)
        + (0.10 * flab_score)
        + (0.10 * prob_draw_score)
        + (0.08 * intensity_draw_score)
        + (0.07 * _clip01(derby_stall_bias))
        + (0.05 * (_clip01(derby_entropy_bias) * entropy_score))
    )

    d_score_total = _clip01((0.60 * d_score_close) + (0.40 * d_score_stall))
    # Multiplicative diagnostics: emphasize "all conditions align" draw setups.
    d_combo_close_lowevent = _clip01(draw_gap_score * prob_draw_score * (1.0 if low_event else 0.0))
    d_combo_close_conflict = _clip01(draw_gap_score * entropy_score * (1.0 if style_conflict else 0.0))
    d_combo_drawrisk_stall = _clip01(
        (1.0 if draw_risk else 0.0) * d_score_close * d_score_stall
    )
    d_flag_close_lowevent = bool(
        draw_gap_score >= 0.45 and prob_draw_score >= 0.45 and low_event
    )
    d_flag_close_conflict = bool(
        draw_gap_score >= 0.45 and entropy_score >= 0.80 and style_conflict
    )
    d_flag_drawrisk_stall = bool(
        draw_risk and d_score_close >= 0.55 and d_score_stall >= 0.45
    )
    d_flag_two_of_three = bool(
        sum(
            [
                int(d_flag_close_lowevent),
                int(d_flag_close_conflict),
                int(d_flag_drawrisk_stall),
            ]
        ) >= 2
    )
    d_flag_j1_close_lowevent = bool(
        league == "j1"
        and low_event
        and draw_gap_score >= 0.45
        and prob_draw_score >= 0.35
    )
    d_flag_j1_drawrisk_stall = bool(
        league == "j1"
        and draw_risk
        and d_score_close >= 0.50
        and d_score_stall >= 0.40
    )
    d_flag_j2_second_draw_close = bool(
        league == "j2"
        and ha_gap_abs <= 0.040
        and prob_draw_score >= 0.60
    )
    d_flag_j2_drawrisk_second = bool(
        league == "j2"
        and draw_risk
        and ha_gap_abs <= 0.040
        and top_gap_score >= 0.20
    )
    return {
        "d_score_close": d_score_close,
        "d_score_stall": d_score_stall,
        "d_score_total": d_score_total,
        "d_combo_close_lowevent": d_combo_close_lowevent,
        "d_combo_close_conflict": d_combo_close_conflict,
        "d_combo_drawrisk_stall": d_combo_drawrisk_stall,
        "d_flag_close_lowevent": d_flag_close_lowevent,
        "d_flag_close_conflict": d_flag_close_conflict,
        "d_flag_drawrisk_stall": d_flag_drawrisk_stall,
        "d_flag_two_of_three": d_flag_two_of_three,
        "d_flag_j1_close_lowevent": d_flag_j1_close_lowevent,
        "d_flag_j1_drawrisk_stall": d_flag_j1_drawrisk_stall,
        "d_flag_j2_second_draw_close": d_flag_j2_second_draw_close,
        "d_flag_j2_drawrisk_second": d_flag_j2_drawrisk_second,
    }


def _compute_close_split_scores(row):
    ph = float(pd.to_numeric(row.get("prob_home_win"), errors="coerce"))
    pdw = float(pd.to_numeric(row.get("prob_draw"), errors="coerce"))
    pa = float(pd.to_numeric(row.get("prob_away_win"), errors="coerce"))
    if not (math.isfinite(ph) and math.isfinite(pdw) and math.isfinite(pa)):
        return {
            "draw_core_score": 0.0,
            "swing_close_score": 0.0,
            "close_split_profile": "",
        }

    prob_shape = _compute_prob_shape(ph, pdw, pa)
    draw_gap = float(max(ph, pa) - pdw)
    top_gap = float(prob_shape["top_gap"])
    entropy_norm = float(prob_shape["entropy_norm"])
    ha_gap_abs = abs(ph - pa)
    side_max = max(ph, pa)

    lab_draw_support = float(pd.to_numeric(row.get("lab_draw_support_score"), errors="coerce"))
    lab_mix_draw = float(pd.to_numeric(row.get("lab_mix_draw"), errors="coerce"))
    lab_mix_home = float(pd.to_numeric(row.get("lab_mix_home"), errors="coerce"))
    lab_mix_away = float(pd.to_numeric(row.get("lab_mix_away"), errors="coerce"))
    xg_diff = float(pd.to_numeric(row.get("match_type_xg_diff"), errors="coerce"))
    rank_gap = float(pd.to_numeric(row.get("match_type_rank_gap"), errors="coerce"))
    home_fatigue = float(pd.to_numeric(row.get("home_total_fatigue_score"), errors="coerce"))
    away_fatigue = float(pd.to_numeric(row.get("away_total_fatigue_score"), errors="coerce"))
    home_motivation = float(pd.to_numeric(row.get("rankmot_motivation_score_5w_home"), errors="coerce"))
    away_motivation = float(pd.to_numeric(row.get("rankmot_motivation_score_5w_away"), errors="coerce"))
    home_sig_ct = float(pd.to_numeric(row.get("match_type_home_signal_count"), errors="coerce"))
    away_sig_ct = float(pd.to_numeric(row.get("match_type_away_signal_count"), errors="coerce"))

    fatigue_edge = away_fatigue - home_fatigue if math.isfinite(home_fatigue) and math.isfinite(away_fatigue) else 0.0
    motivation_edge = home_motivation - away_motivation if math.isfinite(home_motivation) and math.isfinite(away_motivation) else 0.0
    signal_edge = home_sig_ct - away_sig_ct if math.isfinite(home_sig_ct) and math.isfinite(away_sig_ct) else 0.0
    side_lab_max = max(
        lab_mix_home if math.isfinite(lab_mix_home) else 0.0,
        lab_mix_away if math.isfinite(lab_mix_away) else 0.0,
    )

    draw_core_score = _clip01(
        (0.28 * _clip01((pdw - 0.28) / 0.18))
        + (0.18 * _clip01((lab_draw_support - 0.42) / 0.32))
        + (0.16 * _clip01((lab_mix_draw - 0.34) / 0.20))
        + (0.12 * _score_small_abs(ha_gap_abs, 0.10))
        + (0.10 * _score_small_gap(max(draw_gap, 0.0), 0.09))
        + (0.08 * _score_small_gap(top_gap, 0.10))
        + (0.08 * _clip01((entropy_norm - 0.88) / 0.12))
        + (0.10 * _score_small_abs(signal_edge, 2.0))
    )

    swing_close_score = _clip01(
        (0.16 * _score_small_abs(ha_gap_abs, 0.14))
        + (0.14 * _score_small_gap(top_gap, 0.14))
        + (0.18 * _clip01((side_max - 0.28) / 0.12))
        + (0.14 * _clip01(abs(xg_diff) / 1.8))
        + (0.12 * _clip01(abs(rank_gap) / 6.0))
        + (0.10 * _clip01(abs(fatigue_edge) / 5.0))
        + (0.08 * _clip01(abs(motivation_edge) / 3.5))
        + (0.08 * _clip01((side_lab_max - 0.31) / 0.18))
    )

    if draw_core_score >= 0.62 and swing_close_score < 0.52:
        profile = "draw_core"
    elif swing_close_score >= 0.60 and draw_core_score < 0.58:
        profile = "swing_close"
    elif draw_core_score >= 0.56 and swing_close_score >= 0.56:
        profile = "mixed_close"
    elif draw_core_score >= 0.48:
        profile = "soft_draw"
    elif swing_close_score >= 0.48:
        profile = "soft_swing"
    else:
        profile = "undiff_close"

    return {
        "draw_core_score": draw_core_score,
        "swing_close_score": swing_close_score,
        "close_split_profile": profile,
    }


def _build_buyplan_purchase_context(row):
    ph = float(pd.to_numeric(row.get("prob_home_win"), errors="coerce"))
    pdw = float(pd.to_numeric(row.get("prob_draw"), errors="coerce"))
    pa = float(pd.to_numeric(row.get("prob_away_win"), errors="coerce"))
    if not (math.isfinite(ph) and math.isfinite(pdw) and math.isfinite(pa)):
        return {
            "match_purchase_type": "",
            "match_purchase_subtype": "",
            "purchase_type_confidence": "",
            "purchase_type_reason": "",
            "admission_policy": "",
            "rank1_symbol": "",
            "rank2_symbol": "",
            "rank3_symbol": "",
            "rank1_prob": 0.0,
            "rank2_prob": 0.0,
            "rank3_prob": 0.0,
            "anchor_candidate_flag": False,
            "anchor_candidate_score": 0.0,
            "anchor_candidate_reason": "",
            "draw_core_candidate_flag": False,
            "draw_core_candidate_score": 0.0,
            "draw_core_candidate_reason": "",
            "away_overread_draw_cover_flag": False,
            "away_overread_draw_cover_score": 0.0,
            "away_overread_draw_cover_reason": "",
            "draw_core_flag": False,
            "draw_purchase_tier": "",
            "primary_pick_symbol": "",
            "secondary_pick_symbol": "",
            "risk_level": "",
            "ticket_guidance": "",
            "decision_summary": "",
        }

    pred = str(row.get("predicted_result", "")).strip().upper()
    ranked = sorted([("1", ph), ("0", pdw), ("2", pa)], key=lambda kv: (-kv[1], kv[0]))
    rank1_symbol = str(ranked[0][0])
    rank2_symbol = str(ranked[1][0])
    rank3_symbol = str(ranked[2][0])
    rank1_prob = float(ranked[0][1])
    rank2_prob = float(ranked[1][1])
    rank3_prob = float(ranked[2][1])
    prob_shape = _compute_prob_shape(ph, pdw, pa)
    top_gap = float(prob_shape["top_gap"])
    draw_gap = float(max(ph, pa) - pdw)
    best_label = str(prob_shape["best_label"])
    second_label = str(prob_shape["second_label"])
    draw_core_score = float(pd.to_numeric(row.get("draw_core_score"), errors="coerce"))
    swing_close_score = float(pd.to_numeric(row.get("swing_close_score"), errors="coerce"))
    profile = str(row.get("close_split_profile", "")).strip().lower()

    def _sym(label: str) -> str:
        if label == "H":
            return "1"
        if label == "D":
            return "0"
        if label == "A":
            return "2"
        return ""

    best_side_label = "H" if ph >= pa else "A"
    best_side_symbol = _sym(best_side_label)
    pred_symbol = _sym(pred)
    pred_main_symbol = _sym(str(row.get("predicted_result_main", "")).strip().upper())
    second_symbol = _sym(second_label)
    hold_weight = float(pd.to_numeric(row.get("lab_hold_weight"), errors="coerce"))
    stall_weight = float(pd.to_numeric(row.get("lab_stall_weight"), errors="coerce"))
    flip_weight = float(pd.to_numeric(row.get("lab_flip_weight"), errors="coerce"))
    mix_home = float(pd.to_numeric(row.get("lab_mix_home"), errors="coerce"))
    mix_draw = float(pd.to_numeric(row.get("lab_mix_draw"), errors="coerce"))
    mix_away = float(pd.to_numeric(row.get("lab_mix_away"), errors="coerce"))
    entropy_score = float(pd.to_numeric(row.get("lab_scenario_entropy_score"), errors="coerce"))
    volatility_score = float(pd.to_numeric(row.get("lab_volatility_score"), errors="coerce"))
    draw_tension_score = float(pd.to_numeric(row.get("lab_draw_tension_score"), errors="coerce"))
    compactness_score = float(pd.to_numeric(row.get("lab_stall_compactness_score"), errors="coerce"))
    flip_dislocation_score = float(pd.to_numeric(row.get("lab_flip_dislocation_score"), errors="coerce"))
    home_path_score = float(pd.to_numeric(row.get("lab_home_path_score"), errors="coerce"))
    away_path_score = float(pd.to_numeric(row.get("lab_away_path_score"), errors="coerce"))
    side_gap = abs(ph - pa)
    path_gap = abs(home_path_score - away_path_score) if math.isfinite(home_path_score) and math.isfinite(away_path_score) else 0.0
    league = str(row.get("league", "")).strip().upper()
    top_prob = max(ph, pdw, pa)
    direction_mix = max(mix_home if math.isfinite(mix_home) else 0.0, mix_away if math.isfinite(mix_away) else 0.0)
    mix_map = {
        "H": mix_home if math.isfinite(mix_home) else 0.0,
        "D": mix_draw if math.isfinite(mix_draw) else 0.0,
        "A": mix_away if math.isfinite(mix_away) else 0.0,
    }
    mix_best_label = max(mix_map.items(), key=lambda kv: (kv[1], kv[0]))[0]
    pred_mix = mix_draw if pred == "D" else (mix_home if pred == "H" else mix_away if pred == "A" else 0.0)

    def _clip01(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    def _confidence_label(score: float) -> str:
        if score >= 0.67:
            return "high"
        if score >= 0.52:
            return "medium"
        return "low"

    purchase_type = "chaos_flat"
    purchase_subtype = "flat_watch"
    purchase_conf = 0.38
    purchase_reason = (
        f"fallback top_prob={top_prob:.3f} top_gap={top_gap:.3f} "
        f"entropy={entropy_score:.3f} vol={volatility_score:.3f}"
    )

    draw_core_candidate_score = (
        (0.22 * _clip01((mix_draw - 0.46) / 0.12))
        + (0.20 * _clip01((pdw - 0.38) / 0.08))
        + (0.18 * _clip01((draw_tension_score - 0.62) / 0.18))
        + (0.14 * _clip01((compactness_score - 0.58) / 0.16))
        + (0.14 * _clip01((0.06 - path_gap) / 0.06))
        + (0.06 * _clip01((0.58 - entropy_score) / 0.20))
        + (0.06 * _clip01((0.06 - side_gap) / 0.06))
    )
    draw_core_candidate_flag = bool(
        league == "J1"
        and best_label == "D"
        and pred == "D"
        and mix_draw >= 0.46
        and pdw >= 0.38
        and draw_tension_score >= 0.62
        and compactness_score >= 0.58
        and path_gap <= 0.06
        and entropy_score <= 0.58
        and side_gap <= 0.06
        and top_gap <= 0.035
    )
    draw_core_candidate_reason = (
        f"j1_draw_core_candidate pd={pdw:.3f} mix_draw={mix_draw:.3f} "
        f"draw_tension={draw_tension_score:.3f} compact={compactness_score:.3f} "
        f"path_gap={path_gap:.3f} entropy={entropy_score:.3f}"
    )
    anchor_candidate_score = (
        (0.22 * _clip01((top_prob - 0.36) / 0.12))
        + (0.18 * _clip01((top_gap - 0.025) / 0.08))
        + (0.20 * _clip01((hold_weight - 0.22) / 0.18))
        + (0.16 * _clip01((0.66 - entropy_score) / 0.24))
        + (0.12 * _clip01((0.32 - volatility_score) / 0.18))
        + (0.12 * _clip01((0.48 - draw_tension_score) / 0.18))
    )
    anchor_candidate_flag = bool(
        best_label in {"H", "A"}
        and pred == best_label
        and mix_best_label == best_label
        and top_prob >= 0.36
        and top_gap >= 0.025
        and hold_weight >= 0.22
        and entropy_score <= 0.66
        and volatility_score <= 0.32
        and draw_tension_score <= 0.48
    )
    anchor_candidate_reason = (
        f"anchor_candidate best={best_label} top_prob={top_prob:.3f} top_gap={top_gap:.3f} "
        f"hold={hold_weight:.3f} entropy={entropy_score:.3f} vol={volatility_score:.3f}"
    )
    away_overread_draw_cover_score = (
        (0.16 * _clip01((pdw - 0.34) / 0.08))
        + (0.12 * _clip01((0.43 - pa) / 0.10))
        + (0.14 * _clip01((abs(ph - pa) - 0.14) / 0.14))
        + (0.14 * _clip01((top_gap - 0.05) / 0.12))
        + (0.16 * _clip01((flip_dislocation_score - 0.24) / 0.24))
        + (0.14 * _clip01((entropy_score - 0.50) / 0.18))
        + (0.14 * _clip01((away_path_score - 0.56) / 0.18))
    )
    away_overread_draw_cover_reason = (
        f"away_draw_cover pd={pdw:.3f} pa={pa:.3f} diff_ha={abs(ph-pa):.3f} "
        f"top_gap={top_gap:.3f} flip={flip_dislocation_score:.3f} "
        f"entropy={entropy_score:.3f} away_path={away_path_score:.3f}"
    )

    if (
        league == "J2"
        and mix_draw >= 0.42
        and pdw >= 0.34
        and draw_tension_score >= 0.56
        and compactness_score >= 0.54
        and top_gap <= 0.05
    ):
        if best_label == "D" or pred == "D":
            purchase_type = "j2_draw_trap"
            purchase_subtype = "stall_draw_bias"
            purchase_conf = min(
                0.95,
                (0.26 * _clip01((mix_draw - 0.42) / 0.18))
                + (0.22 * _clip01((pdw - 0.34) / 0.10))
                + (0.20 * _clip01((draw_tension_score - 0.56) / 0.22))
                + (0.18 * _clip01((compactness_score - 0.54) / 0.18))
                + (0.14 * _clip01((0.05 - top_gap) / 0.05))
            )
            purchase_reason = (
                f"j2_draw_bias pd={pdw:.3f} mix_draw={mix_draw:.3f} "
                f"draw_tension={draw_tension_score:.3f} compact={compactness_score:.3f}"
            )
        else:
            purchase_type = "j2_directional_stall"
            purchase_subtype = "home_stall" if best_label == "H" else "away_stall"
            purchase_conf = min(
                0.95,
                (0.24 * _clip01((top_prob - 0.35) / 0.14))
                + (0.20 * _clip01((top_gap - 0.02) / 0.10))
                + (0.22 * _clip01((stall_weight - 0.38) / 0.22))
                + (0.16 * _clip01((compactness_score - 0.54) / 0.18))
                + (0.10 * _clip01((draw_tension_score - 0.56) / 0.18))
                + (0.08 * _clip01((direction_mix - 0.30) / 0.20))
            )
            purchase_reason = (
                f"j2_stall best={best_label} top_prob={top_prob:.3f} top_gap={top_gap:.3f} "
                f"stall={stall_weight:.3f} compact={compactness_score:.3f}"
            )
    elif (
        best_label in {"H", "A"}
        and pred == best_label
        and flip_weight >= 0.30
        and volatility_score >= 0.20
        and path_gap >= 0.10
        and top_gap <= 0.10
        and side_gap <= 0.16
    ):
        purchase_type = "home_reversal_watch" if best_label == "A" else "away_reversal_watch"
        purchase_subtype = "away_overread" if best_label == "A" else "home_overread"
        purchase_conf = min(
            0.95,
            (0.30 * _clip01((flip_weight - 0.30) / 0.24))
            + (0.22 * _clip01((volatility_score - 0.20) / 0.24))
            + (0.24 * _clip01((path_gap - 0.10) / 0.22))
            + (0.12 * _clip01((0.10 - top_gap) / 0.10))
            + (0.12 * _clip01((0.16 - side_gap) / 0.16))
        )
        purchase_reason = (
            f"best={best_label} flip={flip_weight:.3f} vol={volatility_score:.3f} "
            f"path_gap={path_gap:.3f} top_gap={top_gap:.3f}"
        )
    elif (
        pdw >= 0.34
        and pdw >= max(ph, pa) - 0.015
        and top_gap <= 0.03
        and draw_tension_score >= 0.48
        and mix_draw >= 0.30
        and (flip_dislocation_score >= 0.18 or side_gap >= 0.05 or volatility_score >= 0.24)
    ):
        purchase_type = "draw_trap"
        purchase_subtype = "weak_draw_side_bias"
        purchase_conf = min(
            0.95,
            (0.28 * _clip01((0.03 - top_gap) / 0.03))
            + (0.22 * _clip01((draw_tension_score - 0.48) / 0.24))
            + (0.20 * _clip01((mix_draw - 0.30) / 0.18))
            + (0.18 * _clip01(max(flip_dislocation_score, volatility_score) / 0.40))
            + (0.12 * _clip01(side_gap / 0.12))
        )
        purchase_reason = (
            f"pd={pdw:.3f} top_gap={top_gap:.3f} draw_tension={draw_tension_score:.3f} "
            f"mix_draw={mix_draw:.3f} side_gap={side_gap:.3f}"
        )
    elif pdw >= max(ph, pa) - 0.01 and draw_tension_score >= 0.44:
        if league == "J1":
            strong_j1_draw_live = bool(
                pdw >= 0.438
                and top_gap >= 0.158
                and draw_tension_score >= 0.72
                and compactness_score >= 0.69
                and entropy_score <= 0.43
            )
            j1_false_draw_flag = bool(
                not strong_j1_draw_live
                and (
                    pdw < 0.418
                    or top_gap < 0.145
                    or (path_gap >= 0.10 and side_gap <= 0.055)
                    or (volatility_score >= 0.24 and draw_tension_score < 0.78)
                    or (
                        rank2_symbol == "1"
                        and home_path_score >= away_path_score
                        and top_gap <= 0.132
                    )
                )
            )
            if j1_false_draw_flag:
                purchase_type = "j1_false_draw_watch"
                purchase_subtype = "draw_false_j1"
                purchase_conf = min(
                    0.95,
                    (0.26 * _clip01((pdw - 0.34) / 0.10))
                    + (0.18 * _clip01((draw_tension_score - 0.44) / 0.24))
                    + (0.16 * _clip01((entropy_score - 0.40) / 0.20))
                    + (0.16 * _clip01((path_gap - 0.09) / 0.18))
                    + (0.12 * _clip01((0.05 - side_gap) / 0.05))
                    + (0.12 * _clip01((0.16 - top_gap) / 0.16))
                )
                purchase_reason = (
                    f"j1_false_draw pd={pdw:.3f} top_gap={top_gap:.3f} "
                    f"path_gap={path_gap:.3f} side_gap={side_gap:.3f} "
                    f"draw_tension={draw_tension_score:.3f} vol={volatility_score:.3f}"
                )
            else:
                purchase_type = "j1_draw_rescue_watch"
                if strong_j1_draw_live or (
                    pdw >= 0.448
                    and top_gap >= 0.168
                    and draw_tension_score >= 0.80
                    and compactness_score >= 0.71
                ):
                    purchase_subtype = "draw_rescue_live_j1"
                else:
                    purchase_subtype = "draw_rescue_soft_j1"
        elif league == "J2":
            purchase_type = "j2_false_draw_watch"
            purchase_subtype = "draw_live_j2"
        else:
            purchase_type = "chaos_draw_watch"
            purchase_subtype = "draw_live"
        purchase_conf = min(
            0.95,
            (0.32 * _clip01((pdw - 0.32) / 0.10))
            + (0.24 * _clip01((draw_tension_score - 0.44) / 0.26))
            + (0.18 * _clip01((entropy_score - 0.46) / 0.28))
            + (0.14 * _clip01((0.05 - top_gap) / 0.05))
            + (0.12 * _clip01((mix_draw - 0.28) / 0.20))
        )
        purchase_reason = (
            f"pd={pdw:.3f} draw_tension={draw_tension_score:.3f} "
            f"entropy={entropy_score:.3f} top_gap={top_gap:.3f}"
        )
    elif best_label in {"H", "A"} and (side_gap >= 0.05 or path_gap >= 0.08):
        purchase_type = "chaos_side_watch"
        purchase_subtype = "home_watch" if best_label == "H" else "away_watch"
        purchase_conf = min(
            0.95,
            (0.24 * _clip01((top_prob - 0.34) / 0.16))
            + (0.20 * _clip01((entropy_score - 0.42) / 0.30))
            + (0.18 * _clip01((volatility_score - 0.16) / 0.26))
            + (0.18 * _clip01(side_gap / 0.16))
            + (0.20 * _clip01(path_gap / 0.24))
        )
        purchase_reason = (
            f"best={best_label} top_prob={top_prob:.3f} side_gap={side_gap:.3f} "
            f"path_gap={path_gap:.3f} entropy={entropy_score:.3f}"
        )
    else:
        purchase_type = "chaos_flat"
        purchase_subtype = "flat_watch"
        purchase_conf = min(
            0.95,
            (0.26 * _clip01((0.06 - top_gap) / 0.06))
            + (0.24 * _clip01((0.10 - side_gap) / 0.10))
            + (0.22 * _clip01((entropy_score - 0.38) / 0.32))
            + (0.14 * _clip01((volatility_score - 0.14) / 0.24))
            + (0.14 * _clip01((0.10 - path_gap) / 0.10))
        )
        purchase_reason = (
            f"top_prob={top_prob:.3f} top_gap={top_gap:.3f} side_gap={side_gap:.3f} "
            f"entropy={entropy_score:.3f} path_gap={path_gap:.3f}"
        )

    purchase_confidence = _confidence_label(purchase_conf)
    away_overread_draw_cover_flag = bool(
        purchase_type == "chaos_side_watch"
        and purchase_subtype == "away_watch"
        and (pred == "A" or best_label == "A")
        and pdw >= 0.34
        and pa <= 0.43
        and abs(ph - pa) >= 0.14
        and top_gap >= 0.05
        and flip_dislocation_score >= 0.24
        and entropy_score >= 0.50
        and away_path_score >= 0.56
    )

    if pred == "D":
        false_draw_watch_flag = purchase_type in {"j1_false_draw_watch", "j2_false_draw_watch"}
        strong_draw_core_flag = bool(
            not false_draw_watch_flag
            and pdw >= 0.452
            and top_gap >= 0.175
            and draw_core_score >= 0.86
            and profile in {"draw_core", "soft_draw"}
            and entropy_score <= 0.44
            and compactness_score >= 0.71
            and draw_tension_score >= 0.82
        )
        draw_core_flag = bool(
            strong_draw_core_flag
        )
        if league == "J1":
            draw_core_flag = False
        draw_cut_flag = bool(
            (top_gap <= 0.025)
            or (draw_gap >= -0.030)
            or swing_close_score >= 0.58
            or profile in {"soft_swing", "swing_close", "undiff_close"}
        )
        if false_draw_watch_flag:
            tier = "watch"
            admission_policy = "rank1_draw_watch"
            primary_pick_symbol = "0"
            secondary_pick_symbol = best_side_symbol
            risk_level = "draw_watch"
            ticket_guidance = "draw_watch_split"
            decision_summary = f"draw watch split: pd={pdw:.3f} gap={draw_gap:.3f} top_gap={top_gap:.3f}"
        elif draw_core_flag:
            tier = "core"
            admission_policy = "rank1_draw_core"
            primary_pick_symbol = "0"
            secondary_pick_symbol = best_side_symbol
            risk_level = "low"
            ticket_guidance = "draw_core_keep"
            decision_summary = f"draw core keep: pd={pdw:.3f} gap={draw_gap:.3f} top_gap={top_gap:.3f}"
        elif draw_cut_flag:
            tier = "cut"
            admission_policy = "rank2_draw_cut"
            primary_pick_symbol = best_side_symbol
            secondary_pick_symbol = "0"
            risk_level = "high"
            ticket_guidance = "draw_cut_to_side"
            decision_summary = f"weak draw cut: pd={pdw:.3f} gap={draw_gap:.3f} top_gap={top_gap:.3f}"
        else:
            tier = "rescue"
            admission_policy = "rank1_draw_rescue"
            primary_pick_symbol = "0"
            secondary_pick_symbol = best_side_symbol
            risk_level = "medium"
            ticket_guidance = "draw_rescue_split"
            decision_summary = f"draw rescue split: pd={pdw:.3f} gap={draw_gap:.3f} top_gap={top_gap:.3f}"
        return {
            "match_purchase_type": purchase_type,
            "match_purchase_subtype": purchase_subtype,
            "purchase_type_confidence": purchase_confidence,
            "purchase_type_reason": purchase_reason,
            "admission_policy": admission_policy,
            "rank1_symbol": rank1_symbol,
            "rank2_symbol": rank2_symbol,
            "rank3_symbol": rank3_symbol,
            "rank1_prob": rank1_prob,
            "rank2_prob": rank2_prob,
            "rank3_prob": rank3_prob,
            "anchor_candidate_flag": anchor_candidate_flag,
            "anchor_candidate_score": anchor_candidate_score,
            "anchor_candidate_reason": anchor_candidate_reason,
            "draw_core_candidate_flag": draw_core_candidate_flag,
            "draw_core_candidate_score": draw_core_candidate_score,
            "draw_core_candidate_reason": draw_core_candidate_reason,
            "away_overread_draw_cover_flag": away_overread_draw_cover_flag,
            "away_overread_draw_cover_score": away_overread_draw_cover_score,
            "away_overread_draw_cover_reason": away_overread_draw_cover_reason,
            "draw_core_flag": draw_core_flag,
            "draw_purchase_tier": tier,
            "primary_pick_symbol": primary_pick_symbol,
            "secondary_pick_symbol": secondary_pick_symbol,
            "risk_level": risk_level,
            "ticket_guidance": ticket_guidance,
            "decision_summary": decision_summary,
        }

    side_secondary_symbol = next((sym for sym, _ in ranked if sym not in {pred_symbol, _sym(best_label)}), "")
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
    max_prob = max(ph, pdw, pa)
    risk_level = "low" if (max_prob >= 0.45 and top_gap >= 0.08) else "medium"
    # Legacy side_single/side_single_strong tended to kill secondary/third paths and
    # force an answer too early. Keep side admissions under cover semantics instead.
    ticket_guidance = "side_with_cover"
    return {
        "match_purchase_type": purchase_type,
        "match_purchase_subtype": purchase_subtype,
        "purchase_type_confidence": purchase_confidence,
        "purchase_type_reason": purchase_reason,
        "admission_policy": admission_policy,
        "rank1_symbol": rank1_symbol,
        "rank2_symbol": rank2_symbol,
        "rank3_symbol": rank3_symbol,
        "rank1_prob": rank1_prob,
        "rank2_prob": rank2_prob,
        "rank3_prob": rank3_prob,
        "anchor_candidate_flag": anchor_candidate_flag,
        "anchor_candidate_score": anchor_candidate_score,
        "anchor_candidate_reason": anchor_candidate_reason,
        "draw_core_candidate_flag": draw_core_candidate_flag,
        "draw_core_candidate_score": draw_core_candidate_score,
        "draw_core_candidate_reason": draw_core_candidate_reason,
        "away_overread_draw_cover_flag": away_overread_draw_cover_flag,
        "away_overread_draw_cover_score": away_overread_draw_cover_score,
        "away_overread_draw_cover_reason": away_overread_draw_cover_reason,
        "draw_core_flag": False,
        "draw_purchase_tier": "side",
        "primary_pick_symbol": side_primary_symbol,
        "secondary_pick_symbol": side_secondary_symbol,
        "risk_level": risk_level,
        "ticket_guidance": ticket_guidance,
        "decision_summary": (
            f"side ranks r1={rank1_symbol} r2={rank2_symbol} r3={rank3_symbol}: "
            f"top_gap={top_gap:.3f} max_prob={max_prob:.3f}"
        ),
    }


def add_buyplan_purchase_context(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    ctx = out.apply(_build_buyplan_purchase_context, axis=1, result_type="expand")
    out = pd.concat([out, ctx], axis=1)
    return out


def write_buyplan_context_csv(df: pd.DataFrame, output_csv: str):
    if df is None or df.empty or not output_csv:
        return ""
    cols = [
        "match_id",
        "league",
        "home_team",
        "away_team",
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
        "legacy_direct_override_applied",
        "legacy_direct_override_reason",
        "anchor_candidate_flag",
        "anchor_candidate_score",
        "anchor_candidate_reason",
        "draw_core_candidate_flag",
        "draw_core_candidate_score",
        "draw_core_candidate_reason",
        "away_overread_draw_cover_flag",
        "away_overread_draw_cover_score",
        "away_overread_draw_cover_reason",
        "match_context_title_race_home",
        "match_context_title_race_away",
        "match_context_acl_race_home",
        "match_context_acl_race_away",
        "match_context_acl_border_home",
        "match_context_acl_border_away",
        "match_context_promotion_race_home",
        "match_context_promotion_race_away",
        "match_context_po_race_home",
        "match_context_po_race_away",
        "match_context_survival_race_home",
        "match_context_survival_race_away",
        "match_context_relegation_escape_home",
        "match_context_relegation_escape_away",
        "match_context_top_direct_duel",
        "match_context_bottom_direct_duel",
        "match_context_zone_home",
        "match_context_zone_away",
        "match_context_pressure_score_home",
        "match_context_pressure_score_away",
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
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return ""
    out_path = os.path.join(os.path.dirname(os.path.abspath(output_csv)), "predictions_buyplan_context.csv")
    df[keep].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[BUYPLAN_CONTEXT] saved={out_path}")
    return out_path


def _build_main_proxy_triplet(row):
    ph = float(pd.to_numeric(row.get("prob_home_win"), errors="coerce"))
    pdw = float(pd.to_numeric(row.get("prob_draw"), errors="coerce"))
    pa = float(pd.to_numeric(row.get("prob_away_win"), errors="coerce"))
    base_triplet = _normalize_hda_triplet(ph, pdw, pa)
    label = str(row.get("predicted_result_main", "")).strip().upper()
    if label not in {"H", "D", "A"}:
        return base_triplet
    d_score_total = float(pd.to_numeric(row.get("d_score_total"), errors="coerce"))
    match_intensity = float(pd.to_numeric(row.get("match_intensity_score"), errors="coerce"))
    focus_stability = float(pd.to_numeric(row.get("focus_stability_score"), errors="coerce"))
    main_conf = _clip01(
        (0.60 * d_score_total)
        + (0.20 * match_intensity)
        + (0.20 * focus_stability)
    )
    shift = 0.035 + (0.16 * main_conf)
    h, d, a = base_triplet
    if label == "D":
        take_home = min(h * 0.28, shift * 0.5)
        take_away = min(a * 0.28, shift * 0.5)
        h -= take_home
        a -= take_away
        d += take_home + take_away
    elif label == "H":
        take_draw = min(d * 0.35, shift * 0.60)
        take_away = min(a * 0.20, shift * 0.40)
        d -= take_draw
        a -= take_away
        h += take_draw + take_away
    elif label == "A":
        take_draw = min(d * 0.35, shift * 0.60)
        take_home = min(h * 0.20, shift * 0.40)
        d -= take_draw
        h -= take_home
        a += take_draw + take_home
    return _normalize_hda_triplet(h, d, a)


def apply_confidence_probability_fusion(df, stage_label="PRED"):
    if df is None or df.empty:
        return df
    out = df.copy()
    required = {
        "prob_elo_home", "prob_elo_draw", "prob_elo_away",
        "prob_home_win", "prob_draw", "prob_away_win",
        "predicted_result_main",
    }
    if not required.issubset(out.columns):
        return out
    if not ENABLE_CONFIDENCE_PROB_FUSION:
        print(f"[CONF_BLEND] stage={stage_label} enabled=0; lab-blend確率を維持")
        return out

    base_vals = out[["prob_elo_home", "prob_elo_draw", "prob_elo_away"]].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).fillna(0.0).to_numpy(dtype=float)
    if {"prob_lab_home", "prob_lab_draw", "prob_lab_away"}.issubset(out.columns):
        lab_vals = out[["prob_lab_home", "prob_lab_draw", "prob_lab_away"]].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        ).fillna(0.0).to_numpy(dtype=float)
    else:
        lab_vals = np.zeros_like(base_vals)
    main_proxy = out.apply(_build_main_proxy_triplet, axis=1, result_type="expand")
    main_proxy.columns = ["prob_main_home", "prob_main_draw", "prob_main_away"]
    out = pd.concat([out, main_proxy], axis=1)
    main_vals = out[["prob_main_home", "prob_main_draw", "prob_main_away"]].to_numpy(dtype=float)

    base_ranked = np.sort(base_vals, axis=1)
    base_top_gap = base_ranked[:, 2] - base_ranked[:, 1]
    base_clarity = np.clip(base_top_gap / 0.18, 0.0, 1.0)
    elo = pd.to_numeric(out.get("elo_diff_for_prob", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    xg = pd.to_numeric(out.get("match_type_xg_diff", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rank_gap = pd.to_numeric(out.get("match_type_rank_gap", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    structural_alignment = (
        (np.sign(elo) == np.sign(xg)).astype(float)
        + (np.sign(elo) == np.sign(-rank_gap)).astype(float)
    ) / 2.0
    weather_penalty = pd.to_numeric(out.get("weather_penalty", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_conf = np.clip(
        (0.55 * base_clarity)
        + (0.30 * structural_alignment)
        + (0.15 * (1.0 - np.clip(weather_penalty / 1.7, 0.0, 1.0))),
        0.0,
        1.0,
    )

    lab_conf = pd.to_numeric(out.get("confidence_lab", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    main_conf = (
        0.55 * pd.to_numeric(out.get("d_score_total", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        + 0.20 * pd.to_numeric(out.get("match_intensity_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        + 0.15 * pd.to_numeric(out.get("focus_stability_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        + 0.10 * pd.to_numeric(out.get("lab_draw_support_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    argmax_label = out.apply(
        lambda row: _argmax_hda_label(row.get("prob_home_win"), row.get("prob_draw"), row.get("prob_away_win")),
        axis=1,
    ).astype(str)
    pred_main = out["predicted_result_main"].astype(str)
    main_conf = np.clip(main_conf + np.where(pred_main.to_numpy() != argmax_label.to_numpy(), 0.12, 0.0), 0.0, 1.0)

    out["confidence_base"] = base_conf
    out["confidence_lab"] = lab_conf
    out["confidence_main"] = main_conf

    alpha = pd.to_numeric(out.get("lab_prob_blend_alpha", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_share = np.clip((1.0 - alpha) * BASE_CONFIDENCE_WEIGHT, 0.0, None)
    lab_share = np.clip(alpha * LAB_CONFIDENCE_WEIGHT, 0.0, None)
    non_main_sum = np.where((base_share + lab_share) > 1e-12, base_share + lab_share, 1.0)
    base_share = base_share / non_main_sum
    lab_share = lab_share / non_main_sum
    main_share = np.clip(MAIN_CONFIDENCE_WEIGHT * main_conf, 0.0, 0.18)
    out["weight_main"] = main_share
    out["weight_base"] = base_share * (1.0 - main_share)
    out["weight_lab"] = lab_share * (1.0 - main_share)

    final_vals = (
        out["weight_base"].to_numpy(dtype=float)[:, None] * base_vals
        + out["weight_lab"].to_numpy(dtype=float)[:, None] * lab_vals
        + out["weight_main"].to_numpy(dtype=float)[:, None] * main_vals
    )
    final_den = np.where(final_vals.sum(axis=1) > 1e-12, final_vals.sum(axis=1), 1.0)
    final_vals = final_vals / final_den[:, None]
    match_type = out.get("match_type", pd.Series("", index=out.index)).astype(str).to_numpy(dtype=object)
    basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str).to_numpy(dtype=object)
    prob_draw_now = final_vals[:, 1]
    prob_home_now = final_vals[:, 0]
    prob_away_now = final_vals[:, 2]
    home_fatigue = pd.to_numeric(out.get("home_total_fatigue_score", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=float)
    away_fatigue = pd.to_numeric(out.get("away_total_fatigue_score", pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(dtype=float)
    home_motivation = pd.to_numeric(
        out.get("rankmot_motivation_score_5w_home", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    away_motivation = pd.to_numeric(
        out.get("rankmot_motivation_score_5w_away", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    home_ready_edge_all = (
        ((away_fatigue - home_fatigue) >= J1_HOME_READY_FATIGUE_EDGE)
        & ((home_motivation - away_motivation) >= J1_HOME_READY_MOTIVATION_EDGE)
        & ((prob_home_now - prob_away_now) >= -J1_HOME_READY_LAB_EDGE_MAX)
    )
    neutral_balanced_draw = (
        (match_type == "neutral")
        & (basis_hint == "basis_balanced")
        & (
            (prob_draw_now >= J1_NEUTRAL_BALANCED_DRAW_MIN)
            | (home_ready_edge_all & (prob_draw_now >= J1_NEUTRAL_BALANCED_HOME_READY_DRAW_MIN))
        )
        & (prob_draw_now <= J1_NEUTRAL_BALANCED_DRAW_MAX)
        & (np.abs(prob_home_now - prob_away_now) <= J1_NEUTRAL_BALANCED_HA_GAP_MAX)
    )
    if np.any(neutral_balanced_draw):
        shift = np.minimum(prob_draw_now[neutral_balanced_draw] - 0.360, J1_NEUTRAL_BALANCED_DRAW_SHIFT_MAX)
        shift = np.clip(shift, 0.0, None)
        home_ready_edge = home_ready_edge_all[neutral_balanced_draw]
        home_ge_away = prob_home_now[neutral_balanced_draw] >= prob_away_now[neutral_balanced_draw]
        restore_home = home_ready_edge | home_ge_away
        final_vals[neutral_balanced_draw, 1] -= shift
        final_vals[neutral_balanced_draw, 0] += np.where(restore_home, shift, shift * 0.40)
        final_vals[neutral_balanced_draw, 2] += np.where(restore_home, shift * 0.40, shift)
        final_den = np.where(final_vals.sum(axis=1) > 1e-12, final_vals.sum(axis=1), 1.0)
        final_vals = final_vals / final_den[:, None]
    base_label = np.array([_argmax_hda_label(*vals) for vals in base_vals], dtype=object)
    lab_label = np.array([_argmax_hda_label(*vals) for vals in lab_vals], dtype=object)
    main_label = out["predicted_result_main"].astype(str).str.upper().to_numpy(dtype=object)
    dominant_model = np.array(["BASE"] * len(out), dtype=object)
    dominant_model = np.where(out["weight_lab"].to_numpy(dtype=float) > out["weight_base"].to_numpy(dtype=float), "LAB", dominant_model)
    dominant_model = np.where(out["weight_main"].to_numpy(dtype=float) > np.maximum(out["weight_base"].to_numpy(dtype=float), out["weight_lab"].to_numpy(dtype=float)), "MAIN", dominant_model)
    final_label = np.array([_argmax_hda_label(*vals) for vals in final_vals], dtype=object)
    fusion_reason = np.array([f"{m}_FUSION" for m in dominant_model], dtype=object)
    fusion_reason = np.where(
        (dominant_model == "MAIN") & (final_label == main_label),
        np.char.add(np.char.add(dominant_model.astype(str), "_"), np.char.add(main_label.astype(str), "_FUSION")),
        fusion_reason,
    )
    out["adopted_model"] = dominant_model
    out["fusion_reason"] = fusion_reason
    out["fusion_detail"] = [
        (
            f"adopt={dominant_model[i]}; final={final_label[i]}; "
            f"w=({out['weight_base'].iat[i]:.3f},{out['weight_lab'].iat[i]:.3f},{out['weight_main'].iat[i]:.3f}); "
            f"c=({out['confidence_base'].iat[i]:.3f},{out['confidence_lab'].iat[i]:.3f},{out['confidence_main'].iat[i]:.3f}); "
            f"labels=({base_label[i]},{lab_label[i]},{main_label[i]})"
        )
        for i in range(len(out))
    ]
    out["prob_final_home"] = final_vals[:, 0]
    out["prob_final_draw"] = final_vals[:, 1]
    out["prob_final_away"] = final_vals[:, 2]
    print(
        f"[CONF_BLEND] stage={stage_label} "
        f"w_base_mean={float(out['weight_base'].mean()):.3f} "
        f"w_lab_mean={float(out['weight_lab'].mean()):.3f} "
        f"w_main_mean={float(out['weight_main'].mean()):.3f}"
    )
    return out


def add_main_prediction_columns(df):
    out = df.copy()
    required = {"prob_home_win", "prob_draw", "prob_away_win"}
    if out.empty or (not required.issubset(out.columns)):
        return out

    out = apply_lab_probability_blend(out)
    if {"prob_final_home", "prob_final_draw", "prob_final_away"}.issubset(out.columns):
        out["prob_home_win"] = pd.to_numeric(out["prob_final_home"], errors="coerce")
        out["prob_draw"] = pd.to_numeric(out["prob_final_draw"], errors="coerce")
        out["prob_away_win"] = pd.to_numeric(out["prob_final_away"], errors="coerce")
        out["prob_home"] = out["prob_home_win"]
        out["prob_away"] = out["prob_away_win"]

    meta = out.apply(_build_match_type_meta, axis=1, result_type="expand")
    out = pd.concat([out, meta], axis=1)
    d_scores = out.apply(_compute_d_scores, axis=1, result_type="expand")
    out = pd.concat([out, d_scores], axis=1)
    close_split_scores = out.apply(_compute_close_split_scores, axis=1, result_type="expand")
    out = pd.concat([out, close_split_scores], axis=1)
    out["predicted_result_main"] = out.apply(_calc_predicted_result_main, axis=1)
    out["main_rule_applied"] = "BASE_FUSION"
    if ENABLE_MAIN_NARROW_DRAW_OVERRIDE:
        ph = pd.to_numeric(out["prob_home_win"], errors="coerce")
        pdw = pd.to_numeric(out["prob_draw"], errors="coerce")
        pa = pd.to_numeric(out["prob_away_win"], errors="coerce")
        max_other = pd.concat([ph, pa], axis=1).max(axis=1)
        draw_edge = pdw - max_other
        match_type = out.get("match_type", pd.Series("", index=out.index)).astype(str)
        basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str)
        draw_support = pd.to_numeric(
            out.get("lab_draw_support_score", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        strong_type = match_type.isin(["home_strong", "away_strong"])
        pred_main_draw = out["predicted_result_main"].astype(str).eq("D")
        if str(LEAGUE).lower() == "j1":
            prob_min = float(J1_MAIN_NARROW_DRAW_PROB_MIN)
            gap_max = float(J1_MAIN_NARROW_DRAW_GAP_MAX)
            relaxed_draw_ok = (
                basis_hint.isin(["draw_compressed", "flat_draw_trap"])
                & draw_support.ge(0.58)
                & draw_edge.le(gap_max + 0.015)
            )
            keep_draw = draw_edge.le(gap_max) | relaxed_draw_ok
            restore_mask = pred_main_draw & strong_type & pdw.ge(prob_min) & (~keep_draw.fillna(False))
            restore_label = np.where(ph >= pa, "H", "A")
            if restore_mask.any():
                out.loc[restore_mask, "predicted_result_main"] = restore_label[restore_mask.to_numpy()]
                out.loc[restore_mask, "main_rule_applied"] = f"J1_MAIN_DRAW_GUARD(draw_edge>{gap_max:.3f})"
                out.loc[restore_mask, "type_adjust_note"] = (
                    out.get("type_adjust_note", pd.Series("", index=out.index)).astype(str)
                    + f"; J1_MAIN_DRAW_GUARD(draw_edge>{gap_max:.3f})"
                )
        elif str(LEAGUE).lower() == "j2":
            prob_min = float(J2_MAIN_NARROW_DRAW_PROB_MIN)
            gap_max = float(J2_MAIN_NARROW_DRAW_GAP_MAX)
            keep_draw = draw_edge.le(gap_max)
            restore_mask = pred_main_draw & strong_type & pdw.ge(prob_min) & (~keep_draw.fillna(False))
            restore_label = np.where(ph >= pa, "H", "A")
            if restore_mask.any():
                out.loc[restore_mask, "predicted_result_main"] = restore_label[restore_mask.to_numpy()]
                out.loc[restore_mask, "main_rule_applied"] = f"J2_MAIN_DRAW_GUARD(draw_edge>{gap_max:.3f})"
                out.loc[restore_mask, "type_adjust_note"] = (
                    out.get("type_adjust_note", pd.Series("", index=out.index)).astype(str)
                    + f"; J2_MAIN_DRAW_GUARD(draw_edge>{gap_max:.3f})"
                )
    if str(LEAGUE).lower() == "j2" and ENABLE_J2_MAIN_SIGNAL_CONFLICT_BALANCED_AWAY_RESTORE:
        pred_main_draw = out["predicted_result_main"].astype(str).eq("D")
        match_type = out.get("match_type", pd.Series("", index=out.index)).astype(str)
        basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str)
        away_xg = out.get("match_type_sig_away_xg", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        away_rank = out.get("match_type_sig_away_rank", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        home_xg = out.get("match_type_sig_home_xg", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        restore_mask = (
            pred_main_draw
            & match_type.eq("signal_conflict")
            & basis_hint.eq("basis_balanced")
            & (away_xg | away_rank)
            & (~home_xg)
        )
        if restore_mask.any():
            out.loc[restore_mask, "predicted_result_main"] = "A"
            out.loc[restore_mask, "main_rule_applied"] = "J2_MAIN_SIGNAL_CONFLICT_BALANCED_AWAY"
            out.loc[restore_mask, "type_adjust_note"] = (
                out.get("type_adjust_note", pd.Series("", index=out.index)).astype(str)
                + "; J2_MAIN_SIGNAL_CONFLICT_BALANCED_AWAY"
            )
    if str(LEAGUE).lower() == "j2" and ENABLE_J2_MAIN_NEUTRAL_DRAW_COMPRESSED_AWAY_RESTORE:
        pred_main_draw = out["predicted_result_main"].astype(str).eq("D")
        match_type = out.get("match_type", pd.Series("", index=out.index)).astype(str)
        basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str)
        home_sig_ct = pd.to_numeric(
            out.get("match_type_home_signal_count", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0.0)
        away_sig_ct = pd.to_numeric(
            out.get("match_type_away_signal_count", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0.0)
        away_xg = out.get("match_type_sig_away_xg", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        away_rank = out.get("match_type_sig_away_rank", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        restore_mask = (
            pred_main_draw
            & match_type.eq("neutral")
            & basis_hint.eq("draw_compressed")
            & home_sig_ct.le(0.0)
            & away_sig_ct.ge(1.0)
            & (away_xg | away_rank)
        )
        if restore_mask.any():
            out.loc[restore_mask, "predicted_result_main"] = "A"
            out.loc[restore_mask, "main_rule_applied"] = "J2_MAIN_NEUTRAL_DRAW_COMPRESSED_AWAY"
            out.loc[restore_mask, "type_adjust_note"] = (
                out.get("type_adjust_note", pd.Series("", index=out.index)).astype(str)
                + "; J2_MAIN_NEUTRAL_DRAW_COMPRESSED_AWAY"
            )
    if str(LEAGUE).lower() == "j1" and ENABLE_J1_MAIN_NEUTRAL_SPLIT_SIDE_AWAY_RESTORE:
        pred_main_draw = out["predicted_result_main"].astype(str).eq("D")
        match_type = out.get("match_type", pd.Series("", index=out.index)).astype(str)
        basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str)
        lab_edge = pd.to_numeric(
            out.get("match_type_lab_matchup_edge", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        restore_mask = (
            pred_main_draw
            & match_type.eq("neutral")
            & basis_hint.eq("split_side")
            & (lab_edge <= J1_MAIN_NEUTRAL_SPLIT_SIDE_LAB_EDGE_MAX)
            & (
                pd.to_numeric(out["prob_away_win"], errors="coerce")
                >= pd.to_numeric(out["prob_home_win"], errors="coerce")
                - J1_MAIN_NEUTRAL_SPLIT_SIDE_AWAY_GAP_MAX
            )
        )
        if restore_mask.any():
            out.loc[restore_mask, "predicted_result_main"] = "A"
            out.loc[restore_mask, "main_rule_applied"] = "J1_MAIN_NEUTRAL_SPLIT_SIDE_AWAY"
            out.loc[restore_mask, "type_adjust_note"] = (
                out.get("type_adjust_note", pd.Series("", index=out.index)).astype(str)
                + "; J1_MAIN_NEUTRAL_SPLIT_SIDE_AWAY"
            )
    out["predicted_result_main_prefusion"] = out["predicted_result_main"]
    out = apply_confidence_probability_fusion(out, "MAIN")
    if {"prob_final_home", "prob_final_draw", "prob_final_away"}.issubset(out.columns):
        out["prob_home_win"] = pd.to_numeric(out["prob_final_home"], errors="coerce")
        out["prob_draw"] = pd.to_numeric(out["prob_final_draw"], errors="coerce")
        out["prob_away_win"] = pd.to_numeric(out["prob_final_away"], errors="coerce")
        out["prob_home"] = out["prob_home_win"]
        out["prob_away"] = out["prob_away_win"]
    out["predicted_result_main_postfusion_argmax"] = out.apply(_calc_predicted_result_main, axis=1)
    if str(LEAGUE).lower() == "j1" and ENABLE_J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_SYNC:
        match_type = out.get("match_type", pd.Series("", index=out.index)).astype(str)
        basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str)
        prob_draw = pd.to_numeric(out.get("prob_draw", pd.Series(np.nan, index=out.index)), errors="coerce")
        prob_home = pd.to_numeric(out.get("prob_home_win", pd.Series(np.nan, index=out.index)), errors="coerce")
        prob_away = pd.to_numeric(out.get("prob_away_win", pd.Series(np.nan, index=out.index)), errors="coerce")
        top_gap = (
            pd.concat([prob_home, prob_draw, prob_away], axis=1)
            .apply(
                lambda r: (
                    (lambda vals: vals[0] - vals[1] if len(vals) >= 2 else np.nan)(
                        sorted([x for x in r.tolist() if pd.notna(x)], reverse=True)
                    )
                ),
                axis=1,
            )
            .astype(float)
        )
        sync_mask = (
            match_type.eq("home_strong")
            & basis_hint.eq("draw_compressed")
            & out["predicted_result_main_prefusion"].astype(str).isin(["H", "A"])
            & out["predicted_result_main_postfusion_argmax"].astype(str).eq("D")
            & prob_draw.ge(J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_DRAW_MIN)
            & top_gap.ge(J1_MAIN_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION_GAP_MIN)
        )
        if sync_mask.any():
            out.loc[sync_mask, "predicted_result_main"] = "D"
            out.loc[sync_mask, "main_rule_applied"] = "J1_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION"
            out.loc[sync_mask, "type_adjust_note"] = (
                out.get("type_adjust_note", pd.Series("", index=out.index)).astype(str)
                + "; J1_HOME_STRONG_DRAW_COMPRESSED_POSTFUSION"
            )
    out["predicted_result_main_symbol"] = out.apply(
        lambda r: _symbol_result_argmax(r["prob_home_win"], r["prob_draw"], r["prob_away_win"]),
        axis=1,
    )
    type_adjust_note = out.get("type_adjust_note", pd.Series("", index=out.index)).astype(str)
    match_type_reason = out.get("match_type_reason", pd.Series("", index=out.index)).astype(str)
    out["type_adjust_note"] = np.where(
        type_adjust_note.str.len() > 0,
        type_adjust_note.str.lstrip("; ").str.strip(),
        match_type_reason,
    )
    return out


def save_match_type_diagnostics(df, league, season_year):
    if df is None or df.empty:
        return None, None
    os.makedirs(PROFILE_SCAN_DIR, exist_ok=True)
    work = apply_match_type_flags(df)
    elo_series = work.get("elo_diff_for_prob", pd.Series(index=work.index, dtype="float64"))
    xg_diff_series = work.get("match_type_xg_diff", pd.Series(index=work.index, dtype="float64"))
    rank_gap_series = work.get("match_type_rank_gap", pd.Series(index=work.index, dtype="float64"))
    rows = []
    for lg, part in work.groupby("league", dropna=False, sort=True) if "league" in work.columns else [(str(league).upper(), work)]:
        actual = part.get("actual_result", pd.Series(index=part.index, dtype="object")).astype(str).str.upper()
        pred = part.get("predicted_result", pd.Series(index=part.index, dtype="object")).astype(str).str.upper()
        for match_type, sub in part.groupby("match_type", dropna=False, sort=True):
            act = actual.loc[sub.index]
            pr = pred.loc[sub.index]
            valid = act.isin(["H", "D", "A"]) & pr.isin(["H", "D", "A"])
            rows.append(
                {
                    "league": str(lg).upper(),
                    "match_type": str(match_type),
                    "matches": int(len(sub)),
                    "hits": int((act[valid] == pr[valid]).sum()) if int(valid.sum()) > 0 else 0,
                    "hit_rate": float((act[valid] == pr[valid]).mean()) if int(valid.sum()) > 0 else None,
                    "pred_H": int((pr == "H").sum()),
                    "pred_D": int((pr == "D").sum()),
                    "pred_A": int((pr == "A").sum()),
                    "act_H": int((act == "H").sum()),
                    "act_D": int((act == "D").sum()),
                    "act_A": int((act == "A").sum()),
                    "elo_diff_mean": float(pd.to_numeric(elo_series.loc[sub.index], errors="coerce").mean()),
                    "xg_diff_mean": float(pd.to_numeric(xg_diff_series.loc[sub.index], errors="coerce").mean()),
                    "rank_gap_mean": float(pd.to_numeric(rank_gap_series.loc[sub.index], errors="coerce").mean()),
                }
            )
    summary_df = pd.DataFrame(rows).sort_values(["league", "matches", "hit_rate"], ascending=[True, False, False])
    summary_path = os.path.join(PROFILE_SCAN_DIR, f"match_type_summary_{str(league).lower()}_{season_year}.csv")
    basis_rows = []
    for lg, part in work.groupby("league", dropna=False, sort=True) if "league" in work.columns else [(str(league).upper(), work)]:
        actual = part.get("actual_result", pd.Series(index=part.index, dtype="object")).astype(str).str.upper()
        pred = part.get("predicted_result", pd.Series(index=part.index, dtype="object")).astype(str).str.upper()
        basis_series = part.get("lab_basis_hint", pd.Series("basis_balanced", index=part.index)).fillna("basis_balanced").astype(str)
        for basis_hint, sub in part.groupby(basis_series, dropna=False, sort=True):
            act = actual.loc[sub.index]
            pr = pred.loc[sub.index]
            valid = act.isin(["H", "D", "A"]) & pr.isin(["H", "D", "A"])
            basis_rows.append(
                {
                    "league": str(lg).upper(),
                    "basis_hint": str(basis_hint),
                    "matches": int(len(sub)),
                    "hits": int((act[valid] == pr[valid]).sum()) if int(valid.sum()) > 0 else 0,
                    "hit_rate": float((act[valid] == pr[valid]).mean()) if int(valid.sum()) > 0 else None,
                    "pred_H": int((pr == "H").sum()),
                    "pred_D": int((pr == "D").sum()),
                    "pred_A": int((pr == "A").sum()),
                    "act_H": int((act == "H").sum()),
                    "act_D": int((act == "D").sum()),
                    "act_A": int((act == "A").sum()),
                    "side_path_mean": float(
                        pd.to_numeric(
                            pd.concat(
                                [
                                    part.loc[sub.index, "lab_side_path_home_score"] if "lab_side_path_home_score" in part.columns else pd.Series(index=sub.index, dtype="float64"),
                                    part.loc[sub.index, "lab_side_path_away_score"] if "lab_side_path_away_score" in part.columns else pd.Series(index=sub.index, dtype="float64"),
                                ],
                                axis=1,
                            ).max(axis=1),
                            errors="coerce",
                        ).mean()
                    ),
                    "draw_support_mean": float(pd.to_numeric(part.loc[sub.index, "lab_draw_support_score"], errors="coerce").mean()) if "lab_draw_support_score" in part.columns else None,
                    "dispersion_mean": float(pd.to_numeric(part.loc[sub.index, "lab_dispersion_score"], errors="coerce").mean()) if "lab_dispersion_score" in part.columns else None,
                    "mix_draw_mean": float(pd.to_numeric(part.loc[sub.index, "lab_mix_draw"], errors="coerce").mean()) if "lab_mix_draw" in part.columns else None,
                }
            )
    basis_summary_df = pd.DataFrame(basis_rows).sort_values(["league", "matches", "hit_rate"], ascending=[True, False, False])
    basis_summary_path = os.path.join(PROFILE_SCAN_DIR, f"match_type_basis_summary_{str(league).lower()}_{season_year}.csv")
    detail_cols = [c for c in [
        "match_id", "league", "節", "home_team", "away_team", "actual_result", "predicted_result",
        "match_type", "match_type_signal_conflict", "match_type_home_signal_count", "match_type_away_signal_count",
        "elo_diff_for_prob", "match_type_xg_diff", "match_type_rank_gap",
        "match_type_sig_away_elo", "match_type_sig_away_xg", "match_type_sig_away_rank",
        "match_type_sig_home_elo", "match_type_sig_home_xg", "match_type_sig_home_rank",
        "prob_home_win", "prob_draw", "prob_away_win", "draw_risk_flag", "draw_gap",
        "d_score_close", "d_score_stall", "d_score_total",
        "draw_core_score", "swing_close_score", "close_split_profile",
        "d_combo_close_lowevent", "d_combo_close_conflict", "d_combo_drawrisk_stall",
        "d_flag_close_lowevent", "d_flag_close_conflict", "d_flag_drawrisk_stall", "d_flag_two_of_three",
        "d_flag_j1_close_lowevent", "d_flag_j1_drawrisk_stall",
        "d_flag_j2_second_draw_close", "d_flag_j2_drawrisk_second",
        "lab_hold_weight", "lab_stall_weight", "lab_flip_weight",
        "lab_volatility_score", "lab_draw_tension_score",
        "lab_tempo_band", "lab_control_band", "lab_pressure_band",
        "lab_matchup_profile",
        "lab_matchup_edge_score", "lab_style_conflict_score", "lab_low_event_score",
        "lab_dynamic_swing_score",
        "lab_hold_foundation_score", "lab_stall_compactness_score", "lab_flip_dislocation_score",
        "lab_side_path_home_score", "lab_side_path_away_score",
        "lab_draw_support_score", "lab_dispersion_score", "lab_basis_hint",
        "lab_home_path_score", "lab_away_path_score", "lab_scenario_entropy_score",
        "lab_draw_path_score", "lab_mix_home", "lab_mix_draw", "lab_mix_away",
        "lab_confidence_delta", "lab_floor_delta", "lab_ceiling_delta",
        "lab_recovery_delta", "lab_regression_delta", "lab_persistence_delta",
        "lab_state_notes",
        "decision_reason",
    ] if c in work.columns]
    detail_df = work[detail_cols].copy()
    detail_path = os.path.join(PROFILE_SCAN_DIR, f"match_type_detail_{str(league).lower()}_{season_year}.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    basis_summary_df.to_csv(basis_summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"[MATCH_TYPE_DIAG] summary={summary_path} basis={basis_summary_path} detail={detail_path}")
    return summary_path, detail_path


def apply_main_prediction_result(df, stage_label="PRED"):
    if df is None or df.empty or "predicted_result_main" not in df.columns:
        return df
    out = df.copy()
    main = out["predicted_result_main"].astype(str).str.upper()
    out["predicted_result_source"] = "main_rule"
    out["predicted_result_prob_ref"] = out.get(
        "predicted_result_main_postfusion_argmax",
        pd.Series("", index=out.index),
    ).astype(str).str.upper()
    cur = out.get("predicted_result", pd.Series("", index=out.index)).astype(str).str.upper()
    changed = main.isin(["H", "D", "A"]) & (main != cur)
    if "fusion_reason" in out.columns:
        out["decision_reason_base"] = out["fusion_reason"].astype(str)
        if "decision_reason" in out.columns:
            cur_reason = out["decision_reason"].astype(str)
            replace_mask = cur_reason.eq("") | cur_reason.eq("ARGMAX") | cur_reason.eq("MATCH_TYPE_MAIN_OVERRIDE")
            out.loc[replace_mask, "decision_reason"] = out.loc[replace_mask, "fusion_reason"].astype(str)
    if not changed.any():
        return out
    out.loc[changed, "predicted_result"] = main.loc[changed]
    if "final_result" in out.columns:
        out.loc[changed, "final_result"] = main.loc[changed]
    if "decision_reason" in out.columns:
        out.loc[changed, "decision_reason"] = out.loc[changed, "fusion_reason"].astype(str) if "fusion_reason" in out.columns else "MATCH_TYPE_MAIN_OVERRIDE"
    print(f"[MATCH_TYPE_MAIN_APPLY] stage={stage_label} changed={int(changed.sum())}")
    return out


def apply_connected_prediction_overrides(df, league, stage_label="PRED"):
    if df is None or df.empty:
        return df
    out = df.copy()
    before = out.get("predicted_result", pd.Series("", index=out.index)).astype(str).str.upper()
    for fn in [
        apply_narrow_draw_override,
        apply_incentive_rank_context_override,
        apply_j1_home_restore_override,
        apply_j1_away_restore_override,
        apply_j1_signal_conflict_home_restore,
        apply_j1_signal_conflict_away_restore,
        apply_j2_away_restore_overrides,
    ]:
        out = fn(out, league)
    if "decision_reason" in out.columns:
        base_reason = out.get("decision_reason_base", out["decision_reason"]).astype(str)
        final_reason = out["decision_reason"].astype(str)
        fusion_reason = out.get("fusion_reason", pd.Series("", index=out.index)).astype(str)
        out["decision_reason_detail"] = np.where(
            final_reason.eq(base_reason) | final_reason.eq(fusion_reason) | final_reason.eq(""),
            out.get("fusion_detail", pd.Series("", index=out.index)).astype(str),
            final_reason + " | " + out.get("fusion_detail", pd.Series("", index=out.index)).astype(str),
        )
        if "main_rule_applied" in out.columns:
            main_rule = out["main_rule_applied"].astype(str)
            sync_mask = (
                main_rule.eq("BASE_FUSION")
                & final_reason.ne("")
                & final_reason.ne("ARGMAX")
                & final_reason.ne("MATCH_TYPE_MAIN_OVERRIDE")
                & final_reason.ne(base_reason)
                & final_reason.ne(fusion_reason)
            )
            out.loc[sync_mask, "main_rule_applied"] = final_reason.loc[sync_mask]
    after = out.get("predicted_result", pd.Series("", index=out.index)).astype(str).str.upper()
    changed = before.isin(["H", "D", "A"]) & after.isin(["H", "D", "A"]) & before.ne(after)
    print(f"[CONNECTED_OVERRIDES] stage={stage_label} changed={int(changed.sum())}")
    return out


def save_argmax_diagnostics(df, league, season_year):
    if df is None or df.empty:
        return None
    os.makedirs(PROFILE_SCAN_DIR, exist_ok=True)
    work = df.copy()
    if not {"prob_home_win", "prob_draw", "prob_away_win", "actual_result"}.issubset(work.columns):
        return None

    work["season_year"] = int(season_year)
    if "試合日" not in work.columns:
        dt = pd.to_datetime(work.get("datetime"), errors="coerce")
        work["試合日"] = dt.dt.strftime("%Y-%m-%d")

    work["pred_argmax"] = _calc_argmax_result_from_probs(work).astype(str).str.upper()
    if "predicted_result" in work.columns:
        work["pred_final_label"] = work["predicted_result"].astype(str).str.upper()
    if "predicted_result_main_postfusion_argmax" in work.columns:
        work["pred_postfusion_main_argmax"] = work["predicted_result_main_postfusion_argmax"].astype(str).str.upper()
    actual = work["actual_result"].astype(str).str.upper()
    valid = actual.isin(["H", "D", "A"]) & work["pred_argmax"].isin(["H", "D", "A"])
    work["is_hit_argmax"] = np.where(valid, (work["pred_argmax"] == actual).astype(int), pd.NA)
    if {"pred_final_label", "actual_result"}.issubset(work.columns):
        valid_final = actual.isin(["H", "D", "A"]) & work["pred_final_label"].isin(["H", "D", "A"])
        work["is_hit_final_label"] = np.where(valid_final, (work["pred_final_label"] == actual).astype(int), pd.NA)
    if {"pred_postfusion_main_argmax", "actual_result"}.issubset(work.columns):
        valid_postfusion = actual.isin(["H", "D", "A"]) & work["pred_postfusion_main_argmax"].isin(["H", "D", "A"])
        work["is_hit_postfusion_main_argmax"] = np.where(
            valid_postfusion, (work["pred_postfusion_main_argmax"] == actual).astype(int), pd.NA
        )
    work["max_prob"] = pd.concat(
        [
            pd.to_numeric(work["prob_home_win"], errors="coerce"),
            pd.to_numeric(work["prob_draw"], errors="coerce"),
            pd.to_numeric(work["prob_away_win"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)

    if "hfa_added_to_diff" in work.columns and "applied_hfa_for_prob" not in work.columns:
        work["applied_hfa_for_prob"] = pd.to_numeric(work["hfa_added_to_diff"], errors="coerce")
    if "elo_diff_effective_for_multinom" in work.columns and "effective_diff_for_multinom" not in work.columns:
        work["effective_diff_for_multinom"] = pd.to_numeric(work["elo_diff_effective_for_multinom"], errors="coerce")
    if "absence_adjust_for_prob" in work.columns and "absence_adjust" not in work.columns:
        work["absence_adjust"] = pd.to_numeric(work["absence_adjust_for_prob"], errors="coerce")

    detail_cols = [
        "league",
        "season_year",
        "match_id",
        "節",
        "試合日",
        "datetime",
        "home_team",
        "away_team",
        "actual_result",
        "pred_final_label",
        "pred_postfusion_main_argmax",
        "pred_argmax",
        "is_hit_final_label",
        "is_hit_postfusion_main_argmax",
        "is_hit_argmax",
        "max_prob",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "elo_diff_for_prob",
        "effective_diff_for_multinom",
        "applied_hfa_for_prob",
        "absence_adjust",
    ]
    detail_cols = [c for c in detail_cols if c in work.columns]
    detail_path = os.path.join(PROFILE_SCAN_DIR, f"{str(league).lower()}_{season_year}_argmax_diagnostics.csv")
    work[detail_cols].to_csv(detail_path, index=False, encoding="utf-8-sig")

    actual_labels = ["H", "D", "A"]
    pred_labels = ["H", "D", "A"]
    confusion = pd.crosstab(
        work["pred_argmax"].astype(str).str.upper(),
        actual,
        dropna=False,
    ).reindex(index=pred_labels, columns=actual_labels, fill_value=0)
    confusion = confusion.rename_axis("pred_argmax").reset_index()
    confusion_path = os.path.join(PROFILE_SCAN_DIR, f"{str(league).lower()}_{season_year}_argmax_confusion.csv")
    confusion.to_csv(confusion_path, index=False, encoding="utf-8-sig")

    miss = work.loc[valid & (work["pred_argmax"] != actual)].copy()
    if miss.empty:
        miss_breakdown = pd.DataFrame(columns=["miss_pattern", "count", "ratio"])
    else:
        miss["miss_pattern"] = miss["pred_argmax"].astype(str) + "→" + actual.loc[miss.index].astype(str)
        miss_breakdown = (
            miss.groupby("miss_pattern", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "miss_pattern"], ascending=[False, True])
            .reset_index(drop=True)
        )
        miss_breakdown["ratio"] = miss_breakdown["count"] / int(len(miss))
    miss_path = os.path.join(PROFILE_SCAN_DIR, f"{str(league).lower()}_{season_year}_argmax_miss_breakdown.csv")
    miss_breakdown.to_csv(miss_path, index=False, encoding="utf-8-sig")

    maxp = pd.to_numeric(work["max_prob"], errors="coerce")
    band = pd.Series("0.50以上", index=work.index, dtype="object")
    band.loc[maxp < 0.50] = "0.45以上0.50未満"
    band.loc[maxp < 0.45] = "<0.45"
    work["max_prob_band"] = band
    band_rows = []
    for band_name in ["<0.45", "0.45以上0.50未満", "0.50以上"]:
        part = work.loc[work["max_prob_band"] == band_name].copy()
        part_valid = part["actual_result"].astype(str).str.upper().isin(["H", "D", "A"])
        band_rows.append(
            {
                "max_prob_band": band_name,
                "matches": int(len(part)),
                "hit_argmax": int(pd.to_numeric(part.loc[part_valid, "is_hit_argmax"], errors="coerce").fillna(0).sum()),
                "hit_rate_argmax": float(pd.to_numeric(part.loc[part_valid, "is_hit_argmax"], errors="coerce").mean()) if int(part_valid.sum()) > 0 else None,
                "avg_prob_home_win": float(pd.to_numeric(part["prob_home_win"], errors="coerce").mean()) if len(part) > 0 else None,
                "avg_prob_draw": float(pd.to_numeric(part["prob_draw"], errors="coerce").mean()) if len(part) > 0 else None,
                "avg_prob_away_win": float(pd.to_numeric(part["prob_away_win"], errors="coerce").mean()) if len(part) > 0 else None,
            }
        )
    band_df = pd.DataFrame(band_rows)
    band_path = os.path.join(PROFILE_SCAN_DIR, f"{str(league).lower()}_{season_year}_argmax_by_maxprob_band.csv")
    band_df.to_csv(band_path, index=False, encoding="utf-8-sig")

    round_col = "節" if "節" in work.columns else None
    round_rows = []
    if round_col is not None:
        for rnd, part in work.groupby(round_col, dropna=False, sort=False):
            act_part = part["actual_result"].astype(str).str.upper()
            pred_part = part["pred_argmax"].astype(str).str.upper()
            round_rows.append(
                {
                    "節": rnd,
                    "matches": int(len(part)),
                    "hit_argmax": int(pd.to_numeric(part["is_hit_argmax"], errors="coerce").fillna(0).sum()),
                    "hit_rate_argmax": float(pd.to_numeric(part["is_hit_argmax"], errors="coerce").mean()) if len(part) > 0 else None,
                    "avg_max_prob": float(pd.to_numeric(part["max_prob"], errors="coerce").mean()) if len(part) > 0 else None,
                    "pred_H_count": int((pred_part == "H").sum()),
                    "pred_D_count": int((pred_part == "D").sum()),
                    "pred_A_count": int((pred_part == "A").sum()),
                    "actual_H_count": int((act_part == "H").sum()),
                    "actual_D_count": int((act_part == "D").sum()),
                    "actual_A_count": int((act_part == "A").sum()),
                }
            )
    round_df = pd.DataFrame(round_rows)
    round_path = os.path.join(PROFILE_SCAN_DIR, f"{str(league).lower()}_{season_year}_argmax_by_round.csv")
    round_df.to_csv(round_path, index=False, encoding="utf-8-sig")

    top_miss = ""
    if not miss_breakdown.empty:
        top = miss_breakdown.iloc[0]
        top_miss = f" top_miss={top['miss_pattern']}({int(top['count'])})"
    print(
        f"[ARGMAX_DIAG] league={str(league).upper()} season={season_year} "
        f"matches={int(valid.sum())} hit_rate={float(pd.to_numeric(work.loc[valid, 'is_hit_argmax'], errors='coerce').mean()):.4f}"
        f"{top_miss}"
    )
    print(
        f"[ARGMAX_DIAG_SAVE] detail={detail_path} confusion={confusion_path} "
        f"miss={miss_path} band={band_path} round={round_path}"
    )
    return {
        "detail": detail_path,
        "confusion": confusion_path,
        "miss": miss_path,
        "band": band_path,
        "round": round_path,
    }


def save_draw_threshold_scan(df, league, season_year):
    if df is None or df.empty:
        return None
    os.makedirs(PROFILE_SCAN_DIR, exist_ok=True)
    work = df.copy()
    if not {"prob_home_win", "prob_draw", "prob_away_win"}.issubset(work.columns):
        return None
    actual = work.get("actual_result", pd.Series(index=work.index, dtype="object")).astype(str).str.upper()
    argmax_pred = _calc_argmax_result_from_probs(work)
    valid = actual.isin(["H", "D", "A"])
    dmask = actual.eq("D")
    ph = pd.to_numeric(work["prob_home_win"], errors="coerce")
    pdw = pd.to_numeric(work["prob_draw"], errors="coerce")
    pa = pd.to_numeric(work["prob_away_win"], errors="coerce")
    gap = pd.concat([ph, pa], axis=1).max(axis=1) - pdw

    league_key = str(league).strip().lower()
    if league_key == "j1":
        prob_grid = [0.33, 0.3325, 0.335, 0.3375, 0.34]
        gap_grid = [0.0, 0.005, 0.01, 0.015, 0.02]
    else:
        prob_grid = [0.335, 0.3375, 0.34, 0.3425, 0.345]
        gap_grid = [0.0, 0.005, 0.01, 0.015, 0.02]

    rows = []
    baseline_hit = float((argmax_pred[valid] == actual[valid]).mean()) if int(valid.sum()) > 0 else None
    baseline_d_hit = int((dmask & argmax_pred.eq("D")).sum())
    rows.append({
        "league": str(league).upper(),
        "rule_name": "argmax_baseline",
        "matches": int(len(work)),
        "actual_d_count": int(dmask.sum()),
        "candidate_count": 0,
        "prob_draw_min": None,
        "gap_max": None,
        "hit_rate": baseline_hit,
        "hit_rate_delta_vs_argmax": 0.0 if baseline_hit is not None else None,
        "actual_d_hit": baseline_d_hit,
        "actual_d_hit_delta_vs_argmax": 0,
    })
    for prob_min in prob_grid:
        for gap_max in gap_grid:
            flag = pdw.ge(prob_min) & gap.le(gap_max)
            pred = pd.Series(np.where(flag.fillna(False), "D", argmax_pred), index=work.index, dtype="object")
            hit_rate = float((pred[valid] == actual[valid]).mean()) if int(valid.sum()) > 0 else None
            actual_d_hit = int((dmask & pred.eq("D")).sum())
            rows.append({
                "league": str(league).upper(),
                "rule_name": f"draw_p{prob_min:.4f}_g{gap_max:.4f}",
                "matches": int(len(work)),
                "actual_d_count": int(dmask.sum()),
                "candidate_count": int(flag.fillna(False).sum()),
                "prob_draw_min": float(prob_min),
                "gap_max": float(gap_max),
                "hit_rate": hit_rate,
                "hit_rate_delta_vs_argmax": (hit_rate - baseline_hit) if (hit_rate is not None and baseline_hit is not None) else None,
                "actual_d_hit": actual_d_hit,
                "actual_d_hit_delta_vs_argmax": int(actual_d_hit - baseline_d_hit),
            })
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(PROFILE_SCAN_DIR, f"draw_threshold_scan_{league_key}_{season_year}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[DRAW_SCAN] saved={out_path}")
    return out_path


def apply_narrow_draw_override(df, league):
    if df is None or df.empty:
        return df
    out = df.copy()
    league_key = str(league).strip().lower()
    if not ENABLE_NARROW_DRAW_OVERRIDE:
        out["narrow_draw_override_applied"] = False
        out["narrow_draw_override_reason"] = ""
        return out
    if league_key == "j1":
        prob_min = float(J1_NARROW_DRAW_PROB_MIN)
        gap_max = float(J1_NARROW_DRAW_GAP_MAX)
    elif league_key == "j2":
        prob_min = float(J2_NARROW_DRAW_PROB_MIN)
        gap_max = float(J2_NARROW_DRAW_GAP_MAX)
    else:
        out["narrow_draw_override_applied"] = False
        out["narrow_draw_override_reason"] = ""
        return out
    if not {"prob_home_win", "prob_draw", "prob_away_win"}.issubset(out.columns):
        out["narrow_draw_override_applied"] = False
        out["narrow_draw_override_reason"] = ""
        return out
    ph = pd.to_numeric(out["prob_home_win"], errors="coerce")
    pdw = pd.to_numeric(out["prob_draw"], errors="coerce")
    pa = pd.to_numeric(out["prob_away_win"], errors="coerce")
    gap = pd.concat([ph, pa], axis=1).max(axis=1) - pdw
    flag = pdw.ge(prob_min) & gap.le(gap_max)
    reason = pd.Series(
        np.where(
            flag.fillna(False),
            f"prob_draw>={prob_min:.3f} & gap<={gap_max:.3f}",
            "",
        ),
        index=out.index,
        dtype="object",
    )
    if league_key == "j1" and "match_type" in out.columns:
        match_type = out["match_type"].astype(str)
        basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str)
        pred_main = out.get("predicted_result_main", pd.Series("", index=out.index)).astype(str).str.upper()
        lab_edge = pd.to_numeric(
            out.get("match_type_lab_matchup_edge", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        draw_support = pd.to_numeric(
            out.get("lab_draw_support_score", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        home_strong = match_type.eq("home_strong")
        away_strong = match_type.eq("away_strong")
        strong_type = home_strong | away_strong
        strong_prob_min = float(J1_NARROW_DRAW_STRONG_PROB_MIN)
        strong_gap_max = float(J1_NARROW_DRAW_STRONG_GAP_MAX)
        strong_relaxed_ok = (
            basis_hint.isin(["draw_compressed", "flat_draw_trap"])
            & draw_support.ge(0.60)
            & gap.le(strong_gap_max + 0.010)
        )
        strong_flag = away_strong & pdw.ge(strong_prob_min) & (gap.le(strong_gap_max) | strong_relaxed_ok)
        flag = flag & ~strong_type
        flag = flag | strong_flag.fillna(False)
        if home_strong.any():
            reason.loc[home_strong.fillna(False)] = ""
        if away_strong.any():
            reason.loc[away_strong.fillna(False)] = np.where(
                strong_flag.loc[away_strong.fillna(False)].fillna(False),
                (
                    f"away_strong prob_draw>={strong_prob_min:.3f} & "
                    f"(gap<={strong_gap_max:.3f} or lab_draw_support)"
                ),
                "",
            )
        neutral = match_type.eq("neutral")
        neutral_balanced = neutral & basis_hint.isin(["basis_balanced", "split_side"])
        if neutral_balanced.any():
            neutral_prob_min = float(J1_NARROW_DRAW_NEUTRAL_BALANCED_PROB_MIN)
            neutral_gap_max = float(J1_NARROW_DRAW_NEUTRAL_BALANCED_GAP_MAX)
            neutral_support_min = float(J1_NARROW_DRAW_NEUTRAL_BALANCED_SUPPORT_MIN)
            neutral_balanced_flag = (
                pdw.ge(neutral_prob_min)
                & gap.le(neutral_gap_max)
                & draw_support.ge(neutral_support_min)
            )
            flag = flag & ~neutral_balanced
            flag = flag | (neutral_balanced & neutral_balanced_flag).fillna(False)
            reason.loc[neutral_balanced.fillna(False)] = np.where(
                neutral_balanced_flag.loc[neutral_balanced.fillna(False)].fillna(False),
                (
                    f"neutral_balanced prob_draw>={neutral_prob_min:.3f} & "
                    f"gap<={neutral_gap_max:.3f} & draw_support>={neutral_support_min:.3f}"
                ),
                "",
            )
        skip_split_side_away = (
            match_type.eq("neutral")
            & basis_hint.eq("split_side")
            & pred_main.eq("A")
            & (lab_edge <= J1_MAIN_NEUTRAL_SPLIT_SIDE_LAB_EDGE_MAX)
        )
        flag = flag & (~skip_split_side_away.fillna(False))
        reason.loc[skip_split_side_away.fillna(False)] = ""
    elif league_key == "j2" and "match_type" in out.columns:
        match_type = out["match_type"].astype(str)
        basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str)
        pred_main = out.get("predicted_result_main", pd.Series("", index=out.index)).astype(str).str.upper()
        pred_pref = out.get("predicted_result_main_prefusion", pd.Series("", index=out.index)).astype(str).str.upper()
        skip_signal_conflict_balanced = (
            match_type.eq("signal_conflict")
            & basis_hint.eq("basis_balanced")
            & pred_main.eq("A")
        )
        home_sig_ct = pd.to_numeric(
            out.get("match_type_home_signal_count", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0.0)
        away_sig_ct = pd.to_numeric(
            out.get("match_type_away_signal_count", pd.Series(0, index=out.index)),
            errors="coerce",
        ).fillna(0.0)
        rank_gap = pd.to_numeric(
            out.get("match_type_rank_gap", out.get("rank_gap", pd.Series(np.nan, index=out.index))),
            errors="coerce",
        )
        home_fatigue = pd.to_numeric(
            out.get("home_total_fatigue_score", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        away_fatigue = pd.to_numeric(
            out.get("away_total_fatigue_score", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        fatigue_edge = away_fatigue - home_fatigue
        skip_neutral_draw_compressed = (
            match_type.eq("neutral")
            & basis_hint.eq("draw_compressed")
            & pred_main.eq("A")
            & home_sig_ct.le(0.0)
            & away_sig_ct.ge(1.0)
        )
        skip_away_pref_a = (
            ENABLE_J2_NARROW_DRAW_SIDE_RESTORE
            & match_type.eq("away_strong")
            & basis_hint.isin(["draw_compressed", "basis_balanced"])
            & pred_pref.eq("A")
            & pa.ge(J2_NARROW_DRAW_AWAY_PREF_A_PROB_MIN)
            & home_sig_ct.le(1.0)
            & away_sig_ct.ge(2.0)
        )
        skip_home_pref_h = (
            ENABLE_J2_NARROW_DRAW_SIDE_RESTORE
            & match_type.eq("home_strong")
            & basis_hint.isin(["draw_compressed", "basis_balanced"])
            & pred_pref.eq("H")
            & ph.ge(J2_NARROW_DRAW_HOME_PREF_H_PROB_MIN)
            & home_sig_ct.ge(2.0)
            & away_sig_ct.le(0.0)
        )
        skip_neutral_pref_d_home = (
            ENABLE_J2_NARROW_DRAW_NEUTRAL_DRAW_COMPRESSED_RESTORE
            & match_type.eq("neutral")
            & basis_hint.eq("draw_compressed")
            & pred_pref.eq("D")
            & rank_gap.le(J2_NARROW_DRAW_NEUTRAL_HOME_RANK_GAP_MAX)
            & fatigue_edge.ge(J2_NARROW_DRAW_NEUTRAL_FATIGUE_EDGE_MIN)
            & home_sig_ct.ge(away_sig_ct)
        )
        skip_neutral_pref_d_away = (
            ENABLE_J2_NARROW_DRAW_NEUTRAL_DRAW_COMPRESSED_RESTORE
            & match_type.eq("neutral")
            & basis_hint.eq("draw_compressed")
            & pred_pref.eq("D")
            & rank_gap.ge(J2_NARROW_DRAW_NEUTRAL_AWAY_RANK_GAP_MIN)
            & fatigue_edge.ge(J2_NARROW_DRAW_NEUTRAL_FATIGUE_EDGE_MIN)
            & pa.ge(ph)
        )
        skip_mask = (
            skip_signal_conflict_balanced
            | skip_neutral_draw_compressed
            | skip_away_pref_a
            | skip_home_pref_h
            | skip_neutral_pref_d_home
            | skip_neutral_pref_d_away
        )
        flag = flag & (~skip_mask.fillna(False))
        reason.loc[skip_mask.fillna(False)] = ""
    flag = flag.fillna(False)
    out["narrow_draw_override_applied"] = flag
    out["narrow_draw_override_reason"] = reason.where(flag, "")
    for col in ["predicted_result", "final_result"]:
        if col in out.columns:
            out.loc[flag, col] = "D"
    if "decision_reason" in out.columns:
        out.loc[flag, "decision_reason"] = "NARROW_DRAW_OVERRIDE"
    return out


def apply_incentive_rank_context_override(df, league):
    if df is None or df.empty or not INCENTIVE_DRAW_SHIFT_ENABLE:
        return df
    out = df.copy()
    required = {
        "predicted_result",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "match_type_title_race_home",
        "match_type_title_race_away",
        "match_type_relegation_risk_home",
        "match_type_relegation_risk_away",
        "home_total_fatigue_score",
        "away_total_fatigue_score",
        "absence_effective_total_home",
        "absence_effective_total_away",
    }
    if not required.issubset(out.columns):
        return out
    league_key = str(league).strip().lower()
    if league_key == "j1":
        draw_prob_min = J1_INCENTIVE_DRAW_PROB_MIN
        draw_gap_max = J1_INCENTIVE_DRAW_GAP_MAX
    elif league_key == "j2":
        draw_prob_min = J2_INCENTIVE_DRAW_PROB_MIN
        draw_gap_max = J2_INCENTIVE_DRAW_GAP_MAX
    else:
        return out
    if "incentive_rank_context_applied" not in out.columns:
        out["incentive_rank_context_applied"] = False
    if "incentive_rank_context_reason" not in out.columns:
        out["incentive_rank_context_reason"] = ""
    pred = out["predicted_result"].astype(str)
    ph = pd.to_numeric(out["prob_home_win"], errors="coerce")
    pdw = pd.to_numeric(out["prob_draw"], errors="coerce")
    pa = pd.to_numeric(out["prob_away_win"], errors="coerce")
    gap = pd.concat([ph, pa], axis=1).max(axis=1) - pdw
    draw_risk = pdw.ge(draw_prob_min) & gap.le(draw_gap_max)
    home_title = out["match_type_title_race_home"].astype(str).str.lower().isin(["true", "1"])
    away_title = out["match_type_title_race_away"].astype(str).str.lower().isin(["true", "1"])
    home_relegation = out["match_type_relegation_risk_home"].astype(str).str.lower().isin(["true", "1"])
    away_relegation = out["match_type_relegation_risk_away"].astype(str).str.lower().isin(["true", "1"])
    home_fatigue = pd.to_numeric(out["home_total_fatigue_score"], errors="coerce").fillna(0.0)
    away_fatigue = pd.to_numeric(out["away_total_fatigue_score"], errors="coerce").fillna(0.0)
    home_absence = pd.to_numeric(out["absence_effective_total_home"], errors="coerce").fillna(0.0)
    away_absence = pd.to_numeric(out["absence_effective_total_away"], errors="coerce").fillna(0.0)
    is_rain = out.get("is_rain", pd.Series(False, index=out.index)).astype(str).str.lower().isin(["true", "1"])
    is_heavy_rain = out.get("is_heavy_rain", pd.Series(False, index=out.index)).astype(str).str.lower().isin(["true", "1"])
    is_strong_wind = out.get("is_strong_wind", pd.Series(False, index=out.index)).astype(str).str.lower().isin(["true", "1"])
    weather_penalty = pd.Series(
        adverse_weather_penalty(is_rain, is_heavy_rain, is_strong_wind), index=out.index
    )
    home_adverse = ((home_fatigue - away_fatigue).clip(lower=0.0) * 0.12) + (home_absence * 8.0) + weather_penalty
    away_adverse = ((away_fatigue - home_fatigue).clip(lower=0.0) * 0.12) + (away_absence * 8.0) + weather_penalty

    home_survival_draw = draw_risk & pred.eq("A") & home_relegation & ~away_title & pa.le(ph + 0.060) & home_adverse.ge(1.20)
    away_survival_draw = draw_risk & pred.eq("H") & away_relegation & ~home_title & ph.le(pa + 0.060) & away_adverse.ge(1.20)
    home_title_push = draw_risk & pred.eq("D") & home_title & ~away_title & ph.ge(pdw - INCENTIVE_TITLE_EDGE_MAX) & away_adverse.ge(1.20)
    away_title_push = draw_risk & pred.eq("D") & away_title & ~home_title & pa.ge(pdw - INCENTIVE_TITLE_EDGE_MAX) & home_adverse.ge(1.20)

    if home_survival_draw.any():
        out.loc[home_survival_draw, ["predicted_result", "final_result"]] = "D"
        out.loc[home_survival_draw, "incentive_rank_context_applied"] = True
        out.loc[home_survival_draw, "incentive_rank_context_reason"] = "home_relegation_draw_hold"
        out.loc[home_survival_draw, "legacy_direct_override_applied"] = True
        out.loc[home_survival_draw, "legacy_direct_override_reason"] = "INCENTIVE_HOME_RELEGATION_DRAW"
        if "decision_reason" in out.columns:
            out.loc[home_survival_draw, "decision_reason"] = "INCENTIVE_HOME_RELEGATION_DRAW"
    if away_survival_draw.any():
        out.loc[away_survival_draw, ["predicted_result", "final_result"]] = "D"
        out.loc[away_survival_draw, "incentive_rank_context_applied"] = True
        out.loc[away_survival_draw, "incentive_rank_context_reason"] = "away_relegation_draw_hold"
        out.loc[away_survival_draw, "legacy_direct_override_applied"] = True
        out.loc[away_survival_draw, "legacy_direct_override_reason"] = "INCENTIVE_AWAY_RELEGATION_DRAW"
        if "decision_reason" in out.columns:
            out.loc[away_survival_draw, "decision_reason"] = "INCENTIVE_AWAY_RELEGATION_DRAW"
    if home_title_push.any():
        out.loc[home_title_push, ["predicted_result", "final_result"]] = "H"
        out.loc[home_title_push, "incentive_rank_context_applied"] = True
        out.loc[home_title_push, "incentive_rank_context_reason"] = "home_title_tiebreak"
        out.loc[home_title_push, "legacy_direct_override_applied"] = True
        out.loc[home_title_push, "legacy_direct_override_reason"] = "INCENTIVE_HOME_TITLE_TIEBREAK"
        if "decision_reason" in out.columns:
            out.loc[home_title_push, "decision_reason"] = "INCENTIVE_HOME_TITLE_TIEBREAK"
    if away_title_push.any():
        out.loc[away_title_push, ["predicted_result", "final_result"]] = "A"
        out.loc[away_title_push, "incentive_rank_context_applied"] = True
        out.loc[away_title_push, "incentive_rank_context_reason"] = "away_title_tiebreak"
        out.loc[away_title_push, "legacy_direct_override_applied"] = True
        out.loc[away_title_push, "legacy_direct_override_reason"] = "INCENTIVE_AWAY_TITLE_TIEBREAK"
        if "decision_reason" in out.columns:
            out.loc[away_title_push, "decision_reason"] = "INCENTIVE_AWAY_TITLE_TIEBREAK"
    return out


def apply_j1_away_restore_override(df, league):
    if not ENABLE_J1_AWAY_RESTORE_OVERRIDE or str(league).lower() != "j1":
        return df
    if df is None or df.empty:
        return df
    required = {"prob_home_win", "prob_draw", "prob_away_win", "predicted_result", "match_type"}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    if "legacy_direct_override_applied" not in out.columns:
        out["legacy_direct_override_applied"] = False
    if "legacy_direct_override_reason" not in out.columns:
        out["legacy_direct_override_reason"] = ""
    if "j1_away_restore_override_applied" not in out.columns:
        out["j1_away_restore_override_applied"] = False
    if "j1_away_restore_override_reason" not in out.columns:
        out["j1_away_restore_override_reason"] = ""

    match_type = out["match_type"].astype(str)
    cond = (
        out["predicted_result"].astype(str).eq("D")
        & ~match_type.eq("home_strong")
        & ~match_type.eq("close_match")
        & out["prob_away_win"].notna()
        & out["prob_home_win"].notna()
        & out["prob_draw"].notna()
        & (out["prob_away_win"] >= out["prob_home_win"] - J1_AWAY_RESTORE_HOME_GAP_MAX)
        & (out["prob_draw"] <= out["prob_away_win"] + J1_AWAY_RESTORE_DRAW_GAP_MAX)
        & (out["prob_draw"] >= J1_AWAY_RESTORE_DRAW_MIN)
    )
    if not cond.any():
        return out

    out.loc[cond, "predicted_result"] = "A"
    if "final_result" in out.columns:
        out.loc[cond, "final_result"] = "A"
    if "decision_reason" in out.columns:
        out.loc[cond, "decision_reason"] = "J1_AWAY_RESTORE_OVERRIDE"
    out.loc[cond, "j1_away_restore_override_applied"] = True
    out.loc[cond, "j1_away_restore_override_reason"] = (
        "pred=D and away~home while draw only slightly above away; exclude home_strong/close_match"
    )
    out.loc[cond, "legacy_direct_override_applied"] = True
    out.loc[cond, "legacy_direct_override_reason"] = "J1_AWAY_RESTORE_OVERRIDE"
    return out


def apply_j1_home_restore_override(df, league):
    if not ENABLE_J1_HOME_RESTORE_OVERRIDE or str(league).lower() != "j1":
        return df
    if df is None or df.empty:
        return df
    required = {
        "prob_home_win", "prob_draw", "prob_away_win", "predicted_result", "match_type",
        "lab_basis_hint", "home_total_fatigue_score", "away_total_fatigue_score",
        "rankmot_motivation_score_5w_home", "rankmot_motivation_score_5w_away",
    }
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    if "legacy_direct_override_applied" not in out.columns:
        out["legacy_direct_override_applied"] = False
    if "legacy_direct_override_reason" not in out.columns:
        out["legacy_direct_override_reason"] = ""
    if "j1_home_restore_override_applied" not in out.columns:
        out["j1_home_restore_override_applied"] = False
    if "j1_home_restore_override_reason" not in out.columns:
        out["j1_home_restore_override_reason"] = ""

    pred = out["predicted_result"].astype(str)
    match_type = out["match_type"].astype(str)
    basis_hint = out["lab_basis_hint"].astype(str)
    home_fatigue = pd.to_numeric(out["home_total_fatigue_score"], errors="coerce")
    away_fatigue = pd.to_numeric(out["away_total_fatigue_score"], errors="coerce")
    home_motivation = pd.to_numeric(out["rankmot_motivation_score_5w_home"], errors="coerce")
    away_motivation = pd.to_numeric(out["rankmot_motivation_score_5w_away"], errors="coerce")
    draw_support = pd.to_numeric(out.get("lab_draw_support_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    home_ready_edge = (
        (away_fatigue - home_fatigue >= J1_HOME_READY_FATIGUE_EDGE)
        & (home_motivation - away_motivation >= J1_HOME_READY_MOTIVATION_EDGE)
        & (out["prob_home_win"] >= out["prob_away_win"] - J1_HOME_READY_LAB_EDGE_MAX)
    )
    cond = (
        pred.isin(["D", "A"])
        & match_type.eq("neutral")
        & basis_hint.eq("basis_balanced")
        & home_ready_edge.fillna(False)
        & out["prob_home_win"].notna()
        & out["prob_away_win"].notna()
        & out["prob_draw"].notna()
        & (out["prob_home_win"] >= out["prob_away_win"] - J1_HOME_RESTORE_AWAY_GAP_MAX)
        & (out["prob_draw"] <= out["prob_home_win"] + J1_HOME_RESTORE_DRAW_GAP_MAX)
        & (out["prob_draw"] >= J1_HOME_RESTORE_DRAW_MIN)
        & (
            (out["prob_draw"] <= J1_HOME_RESTORE_DRAW_MAX)
            | (draw_support <= J1_HOME_RESTORE_DRAW_SUPPORT_MAX)
        )
    )
    compressed_cond = (
        pred.eq("D")
        & match_type.eq("neutral")
        & basis_hint.eq("draw_compressed")
        & (away_fatigue - home_fatigue >= J1_HOME_RESTORE_COMPRESSED_FATIGUE_EDGE)
        & (home_motivation - away_motivation >= J1_HOME_RESTORE_COMPRESSED_MOTIVATION_EDGE)
        & (pd.to_numeric(out.get("match_type_rank_gap", pd.Series(np.nan, index=out.index)), errors="coerce") <= -1.0)
        & (pd.to_numeric(out.get("elo_diff_for_prob", pd.Series(np.nan, index=out.index)), errors="coerce") <= 0.0)
        & (out["prob_home_win"] >= out["prob_away_win"] - J1_HOME_RESTORE_COMPRESSED_HOME_GAP_MAX)
        & (out["prob_draw"] <= J1_HOME_RESTORE_COMPRESSED_DRAW_MAX)
    )
    cond = cond | compressed_cond.fillna(False)
    if not cond.any():
        return out

    out.loc[cond, "predicted_result"] = "H"
    if "final_result" in out.columns:
        out.loc[cond, "final_result"] = "H"
    if "decision_reason" in out.columns:
        out.loc[cond, "decision_reason"] = "J1_HOME_RESTORE_OVERRIDE"
    out.loc[cond, "j1_home_restore_override_applied"] = True
    out.loc[cond, "j1_home_restore_override_reason"] = (
        "pred in {D,A} and neutral/basis_balanced with clear home_ready_edge; restore to H"
    )
    out.loc[cond, "legacy_direct_override_applied"] = True
    out.loc[cond, "legacy_direct_override_reason"] = "J1_HOME_RESTORE_OVERRIDE"
    return out


def apply_j1_signal_conflict_away_restore(df, league):
    if not ENABLE_J1_SIGNAL_CONFLICT_AWAY_RESTORE or str(league).lower() != "j1":
        return df
    if df is None or df.empty:
        return df
    required = {"prob_home_win", "prob_away_win", "predicted_result", "match_type"}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    if "legacy_direct_override_applied" not in out.columns:
        out["legacy_direct_override_applied"] = False
    if "legacy_direct_override_reason" not in out.columns:
        out["legacy_direct_override_reason"] = ""
    if "j1_signal_conflict_away_restore_applied" not in out.columns:
        out["j1_signal_conflict_away_restore_applied"] = False
    if "j1_signal_conflict_away_restore_reason" not in out.columns:
        out["j1_signal_conflict_away_restore_reason"] = ""

    home_fatigue = pd.to_numeric(out.get("home_total_fatigue_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    away_fatigue = pd.to_numeric(out.get("away_total_fatigue_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    rank_gap = pd.to_numeric(out.get("match_type_rank_gap", pd.Series(np.nan, index=out.index)), errors="coerce")
    basis_hint = out.get("lab_basis_hint", pd.Series("", index=out.index)).astype(str)
    home_motivation = pd.to_numeric(
        out.get("rankmot_motivation_score_5w_home", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    away_motivation = pd.to_numeric(
        out.get("rankmot_motivation_score_5w_away", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    home_relegation = out.get("match_type_relegation_risk_home", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    away_relegation = out.get("match_type_relegation_risk_away", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    home_gap = pd.to_numeric(out["prob_home_win"], errors="coerce") - pd.to_numeric(out["prob_away_win"], errors="coerce")
    home_ready_block = (
        basis_hint.eq("draw_compressed")
        & ((away_fatigue - home_fatigue) >= J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_FATIGUE_EDGE_MAX)
        & (rank_gap <= J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_RANK_GAP_MIN)
    )
    balanced_home_block = (
        ENABLE_J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE
        & basis_hint.eq("basis_balanced")
        & away_relegation
        & (~home_relegation)
        & ((away_fatigue - home_fatigue) >= J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_FATIGUE_EDGE)
        & ((home_motivation - away_motivation) >= J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_MOTIVATION_EDGE)
        & home_gap.ge(J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_HOME_GAP_MIN)
        & home_gap.le(J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_HOME_GAP_MAX)
    )
    cond = (
        out["match_type"].astype(str).eq("signal_conflict")
        & out["predicted_result"].astype(str).eq("H")
        & out["prob_away_win"].notna()
        & out["prob_home_win"].notna()
        & (out["prob_away_win"] >= out["prob_home_win"] - J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_GAP_MAX)
        & (~home_ready_block.fillna(False))
        & (~balanced_home_block.fillna(False))
    )
    if not cond.any():
        return out

    out.loc[cond, "predicted_result"] = "A"
    if "final_result" in out.columns:
        out.loc[cond, "final_result"] = "A"
    if "decision_reason" in out.columns:
        out.loc[cond, "decision_reason"] = "J1_SIGNAL_CONFLICT_AWAY_RESTORE"
    out.loc[cond, "j1_signal_conflict_away_restore_applied"] = True
    out.loc[cond, "j1_signal_conflict_away_restore_reason"] = (
        f"signal_conflict and away>=home-{J1_SIGNAL_CONFLICT_AWAY_RESTORE_HOME_GAP_MAX:.3f}"
    )
    out.loc[cond, "legacy_direct_override_applied"] = True
    out.loc[cond, "legacy_direct_override_reason"] = "J1_SIGNAL_CONFLICT_AWAY_RESTORE"
    return out


def apply_j1_signal_conflict_home_restore(df, league):
    if not ENABLE_J1_SIGNAL_CONFLICT_HOME_RESTORE or str(league).lower() != "j1":
        return df
    if df is None or df.empty:
        return df
    required = {
        "prob_home_win", "prob_away_win", "predicted_result", "match_type", "lab_basis_hint",
        "home_total_fatigue_score", "away_total_fatigue_score", "match_type_rank_gap",
    }
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    if "legacy_direct_override_applied" not in out.columns:
        out["legacy_direct_override_applied"] = False
    if "legacy_direct_override_reason" not in out.columns:
        out["legacy_direct_override_reason"] = ""
    if "j1_signal_conflict_home_restore_applied" not in out.columns:
        out["j1_signal_conflict_home_restore_applied"] = False
    if "j1_signal_conflict_home_restore_reason" not in out.columns:
        out["j1_signal_conflict_home_restore_reason"] = ""

    home_fatigue = pd.to_numeric(out["home_total_fatigue_score"], errors="coerce")
    away_fatigue = pd.to_numeric(out["away_total_fatigue_score"], errors="coerce")
    rank_gap = pd.to_numeric(out["match_type_rank_gap"], errors="coerce")
    home_motivation = pd.to_numeric(
        out.get("rankmot_motivation_score_5w_home", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    away_motivation = pd.to_numeric(
        out.get("rankmot_motivation_score_5w_away", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    home_relegation = out.get("match_type_relegation_risk_home", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    away_relegation = out.get("match_type_relegation_risk_away", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    home_gap = pd.to_numeric(out["prob_home_win"], errors="coerce") - pd.to_numeric(out["prob_away_win"], errors="coerce")
    cond = (
        out["match_type"].astype(str).eq("signal_conflict")
        & out["lab_basis_hint"].astype(str).eq("draw_compressed")
        & out["predicted_result"].astype(str).eq("D")
        & (away_fatigue - home_fatigue >= J1_SIGNAL_CONFLICT_HOME_RESTORE_FATIGUE_EDGE)
        & (out["prob_home_win"] >= out["prob_away_win"] - J1_SIGNAL_CONFLICT_HOME_RESTORE_HOME_GAP_MAX)
        & (rank_gap <= J1_SIGNAL_CONFLICT_HOME_RESTORE_RANK_GAP_MAX)
    )
    balanced_cond = (
        ENABLE_J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE
        & out["match_type"].astype(str).eq("signal_conflict")
        & out["lab_basis_hint"].astype(str).eq("basis_balanced")
        & out["predicted_result"].astype(str).eq("D")
        & away_relegation
        & (~home_relegation)
        & (away_fatigue - home_fatigue >= J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_FATIGUE_EDGE)
        & ((home_motivation - away_motivation) >= J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_MOTIVATION_EDGE)
        & home_gap.ge(J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_HOME_GAP_MIN)
        & home_gap.le(J1_SIGNAL_CONFLICT_BALANCED_HOME_RESTORE_HOME_GAP_MAX)
    )
    cond = cond | balanced_cond.fillna(False)
    if not cond.any():
        return out

    out.loc[cond, "predicted_result"] = "H"
    if "final_result" in out.columns:
        out.loc[cond, "final_result"] = "H"
    if "decision_reason" in out.columns:
        out.loc[cond, "decision_reason"] = "J1_SIGNAL_CONFLICT_HOME_RESTORE"
    out.loc[cond, "j1_signal_conflict_home_restore_applied"] = True
    out.loc[cond, "j1_signal_conflict_home_restore_reason"] = (
        "signal_conflict draw_compressed with strong home fatigue/rank edge and home~away probs"
    )
    out.loc[cond, "legacy_direct_override_applied"] = True
    out.loc[cond, "legacy_direct_override_reason"] = "J1_SIGNAL_CONFLICT_HOME_RESTORE"
    return out


def _clip_unit(series, low, high):
    series = pd.to_numeric(series, errors="coerce")
    width = max(float(high) - float(low), 1e-9)
    return ((series - float(low)) / width).clip(0.0, 1.0)


def _prepare_j2_restore_context(out):
    idx = out.index
    ctx = {}
    ctx["pred_h"] = out["predicted_result"].astype(str).eq("H")
    ctx["pred_d"] = out["predicted_result"].astype(str).eq("D")
    ctx["match_type"] = out["match_type"].astype(str)
    ctx["basis_hint"] = out.get("lab_basis_hint", pd.Series("", index=idx)).astype(str)
    ctx["prob_home"] = pd.to_numeric(out.get("prob_home_win", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["prob_draw"] = pd.to_numeric(out.get("prob_draw", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["prob_away"] = pd.to_numeric(out.get("prob_away_win", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["home_fatigue"] = pd.to_numeric(out.get("home_total_fatigue_score", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["away_fatigue"] = pd.to_numeric(out.get("away_total_fatigue_score", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["home_motivation"] = pd.to_numeric(out.get("rankmot_motivation_score_5w_home", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["away_motivation"] = pd.to_numeric(out.get("rankmot_motivation_score_5w_away", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["rank_gap"] = pd.to_numeric(out.get("match_type_rank_gap", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["elo_diff"] = pd.to_numeric(out.get("elo_diff_for_prob", pd.Series(np.nan, index=idx)), errors="coerce")
    ctx["fatigue_edge"] = ctx["away_fatigue"] - ctx["home_fatigue"]
    ctx["motivation_edge"] = ctx["home_motivation"] - ctx["away_motivation"]
    ctx["draw_home_gap"] = ctx["prob_draw"] - ctx["prob_home"]
    ctx["home_away_gap"] = ctx["prob_home"] - ctx["prob_away"]
    ctx["prob_ready"] = ctx["prob_home"].notna() & ctx["prob_draw"].notna() & ctx["prob_away"].notna()
    ctx["score_home_ready"] = (
        0.45 * _clip_unit(ctx["fatigue_edge"], 0.0, 6.0)
        + 0.35 * _clip_unit(ctx["motivation_edge"], -1.0, 4.0)
        + 0.20 * _clip_unit(-ctx["rank_gap"], -2.0, 6.0)
    ).clip(0.0, 1.0)
    ctx["score_draw_escape"] = (
        0.45 * (1.0 - _clip_unit(ctx["draw_home_gap"], 0.0, 0.14))
        + 0.30 * _clip_unit(ctx["prob_home"], 0.24, 0.39)
        + 0.25 * _clip_unit(ctx["home_away_gap"], -0.06, 0.12)
    ).clip(0.0, 1.0)
    ctx["score_press"] = (
        0.50 * _clip_unit(ctx["elo_diff"], -50.0, 55.0)
        + 0.30 * _clip_unit(ctx["fatigue_edge"], 0.0, 5.5)
        + 0.20 * (1.0 - _clip_unit(ctx["draw_home_gap"], 0.0, 0.12))
    ).clip(0.0, 1.0)
    ctx["cluster_neutral_balanced"] = ctx["match_type"].eq("neutral") & ctx["basis_hint"].eq("basis_balanced")
    ctx["cluster_neutral_draw"] = ctx["match_type"].eq("neutral") & ctx["basis_hint"].eq("draw_compressed")
    ctx["cluster_home_balanced"] = ctx["match_type"].eq("home_strong") & ctx["basis_hint"].eq("basis_balanced")
    ctx["cluster_home_draw"] = ctx["match_type"].eq("home_strong") & ctx["basis_hint"].eq("draw_compressed")
    ctx["cluster_home_flat"] = ctx["match_type"].eq("home_strong") & ctx["basis_hint"].eq("flat_draw_trap")
    ctx["cluster_away_draw"] = ctx["match_type"].eq("away_strong") & ctx["basis_hint"].eq("draw_compressed")
    ctx["cluster_away_flat"] = ctx["match_type"].eq("away_strong") & ctx["basis_hint"].eq("flat_draw_trap")
    ctx["cluster_close_draw"] = ctx["match_type"].eq("close_match") & ctx["basis_hint"].eq("draw_compressed")
    ctx["cluster_close_flat"] = ctx["match_type"].eq("close_match") & ctx["basis_hint"].eq("flat_draw_trap")
    ctx["cluster_signal_split"] = ctx["match_type"].eq("signal_conflict") & ctx["basis_hint"].eq("split_side")
    return ctx


def _apply_j2_home_restore(out, cond, reason, cluster, ctx, reason_group=None):
    if not cond.any():
        return out
    out.loc[cond, "predicted_result"] = "H"
    if "final_result" in out.columns:
        out.loc[cond, "final_result"] = "H"
    if "decision_reason" in out.columns:
        out.loc[cond, "decision_reason"] = reason
    out.loc[cond, "j2_restore_cluster"] = cluster
    if "j2_restore_reason_group" in out.columns:
        out.loc[cond, "j2_restore_reason_group"] = str(reason_group or cluster)
    out.loc[cond, "j2_restore_home_ready_score"] = ctx["score_home_ready"][cond]
    out.loc[cond, "j2_restore_draw_escape_score"] = ctx["score_draw_escape"][cond]
    out.loc[cond, "j2_restore_press_score"] = ctx["score_press"][cond]
    return out


def _apply_j2_away_restore(out, cond, reason, cluster, ctx, reason_group=None):
    if not cond.any():
        return out
    out.loc[cond, "predicted_result"] = "A"
    if "final_result" in out.columns:
        out.loc[cond, "final_result"] = "A"
    if "decision_reason" in out.columns:
        out.loc[cond, "decision_reason"] = reason
    out.loc[cond, "j2_restore_cluster"] = cluster
    if "j2_restore_reason_group" in out.columns:
        out.loc[cond, "j2_restore_reason_group"] = str(reason_group or cluster)
    out.loc[cond, "j2_restore_home_ready_score"] = ctx["score_home_ready"][cond]
    out.loc[cond, "j2_restore_draw_escape_score"] = ctx["score_draw_escape"][cond]
    out.loc[cond, "j2_restore_press_score"] = ctx["score_press"][cond]
    return out


def apply_j2_away_restore_overrides(df, league):
    if str(league).lower() != "j2":
        return df
    if df is None or df.empty:
        return df
    required = {"predicted_result", "match_type"}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    for col in [
        "j2_away_strong_away_restore_applied",
        "j2_signal_conflict_away_restore_applied",
        "j2_neg_home_adv_away_restore_applied",
        "j2_away_draw_restore_applied",
    ]:
        if col not in out.columns:
            out[col] = False
    for col in [
        "j2_away_strong_away_restore_reason",
        "j2_signal_conflict_away_restore_reason",
        "j2_neg_home_adv_away_restore_reason",
        "j2_away_draw_restore_reason",
    ]:
        if col not in out.columns:
            out[col] = ""
    for col in [
        "j2_restore_cluster",
        "j2_restore_reason_group",
        "j2_restore_home_ready_score",
        "j2_restore_draw_escape_score",
        "j2_restore_press_score",
    ]:
        if col not in out.columns:
            out[col] = np.nan if col not in {"j2_restore_cluster", "j2_restore_reason_group"} else ""

    ctx = _prepare_j2_restore_context(out)
    pred_h = ctx["pred_h"]
    pred_d = ctx["pred_d"]
    match_type = ctx["match_type"]
    basis_hint = ctx["basis_hint"]
    prob_home = ctx["prob_home"]
    prob_draw = ctx["prob_draw"]
    prob_away = ctx["prob_away"]
    home_fatigue = ctx["home_fatigue"]
    away_fatigue = ctx["away_fatigue"]
    home_motivation = ctx["home_motivation"]
    away_motivation = ctx["away_motivation"]
    rank_gap = ctx["rank_gap"]
    elo_diff = ctx["elo_diff"]
    fatigue_edge = ctx["fatigue_edge"]
    motivation_edge = ctx["motivation_edge"]
    draw_home_gap = ctx["draw_home_gap"]
    draw_away_gap = prob_draw - prob_away
    home_sig_ct = pd.to_numeric(
        out.get("match_type_home_signal_count", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    away_sig_ct = pd.to_numeric(
        out.get("match_type_away_signal_count", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)

    home_balanced_cond = pd.Series(False, index=out.index)
    if ENABLE_J2_HOME_RESTORE_OVERRIDE:
        home_balanced_cond |= (
            pred_d
            & ctx["cluster_home_balanced"]
            & ctx["prob_ready"]
            & (prob_home >= prob_away)
            & (prob_draw <= prob_home + J2_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_HOME_RESTORE_FATIGUE_EDGE)
            & (motivation_edge >= J2_HOME_RESTORE_MOTIVATION_EDGE)
            & (rank_gap <= J2_HOME_RESTORE_RANK_GAP_MAX)
        )

    if ENABLE_J2_NEUTRAL_HOME_RESTORE:
        neutral_base = (
            pred_d
            & (ctx["cluster_neutral_balanced"] | ctx["cluster_neutral_draw"])
            & ctx["prob_ready"]
            & (fatigue_edge >= J2_NEUTRAL_HOME_RESTORE_FATIGUE_EDGE)
            & (motivation_edge >= J2_NEUTRAL_HOME_RESTORE_MOTIVATION_EDGE)
            & (rank_gap <= -1.0)
            & (prob_home >= (prob_away - J2_NEUTRAL_HOME_RESTORE_HOME_GAP_MAX))
        )
        cond_balanced = (
            neutral_base & ctx["cluster_neutral_balanced"]
            & (prob_draw <= prob_home + J2_NEUTRAL_HOME_RESTORE_DRAW_GAP_MAX)
        )
        cond_compressed = (
            neutral_base & ctx["cluster_neutral_draw"]
            & elo_diff.notna()
            & (elo_diff <= 0.0)
            & (prob_draw <= prob_home + 0.14)
        )
        cond = cond_balanced | cond_compressed
        out = _apply_j2_home_restore(
            out, cond, "J2_NEUTRAL_HOME_RESTORE", "neutral_base", ctx, "neutral_base"
        )

    if ENABLE_J2_HOME_STRONG_BALANCED_RESTORE:
        home_balanced_cond |= (
            pred_d
            & ctx["cluster_home_balanced"]
            & ctx["prob_ready"]
            & (prob_draw <= prob_home + J2_HOME_STRONG_BALANCED_RESTORE_DRAW_GAP_MAX)
            & (prob_home >= (prob_away - J2_HOME_STRONG_BALANCED_RESTORE_HOME_GAP_MAX))
            & (fatigue_edge >= J2_HOME_STRONG_BALANCED_RESTORE_FATIGUE_EDGE)
            & (motivation_edge >= J2_HOME_STRONG_BALANCED_RESTORE_MOTIVATION_EDGE)
            & (rank_gap <= J2_HOME_STRONG_BALANCED_RESTORE_RANK_GAP_MAX)
            & elo_diff.notna()
            & (elo_diff <= 0.0)
        )

    if ENABLE_J2_SIGNAL_CONFLICT_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_signal_split"]
            & prob_home.notna() & prob_draw.notna()
            & (prob_draw <= prob_home + J2_SIGNAL_CONFLICT_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_SIGNAL_CONFLICT_HOME_RESTORE_FATIGUE_EDGE)
            & (motivation_edge >= J2_SIGNAL_CONFLICT_HOME_RESTORE_MOTIVATION_EDGE)
            & (rank_gap <= J2_SIGNAL_CONFLICT_HOME_RESTORE_RANK_GAP_MAX)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_SIGNAL_CONFLICT_HOME_RESTORE", "signal_split", ctx, "signal_split"
        )

    if ENABLE_J2_AWAY_STRONG_FLAT_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_away_flat"]
            & prob_home.notna() & prob_draw.notna()
            & (prob_draw <= prob_home + J2_AWAY_STRONG_FLAT_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_AWAY_STRONG_FLAT_HOME_RESTORE_FATIGUE_EDGE)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_AWAY_STRONG_FLAT_HOME_RESTORE", "away_flat", ctx, "away_flat"
        )

    if ENABLE_J2_CLOSE_FLAT_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_close_flat"]
            & prob_home.notna() & prob_draw.notna()
            & (prob_draw <= prob_home + J2_CLOSE_FLAT_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_CLOSE_FLAT_HOME_RESTORE_FATIGUE_EDGE)
            & (rank_gap <= -1.0)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_CLOSE_FLAT_HOME_RESTORE", "close_flat", ctx, "close_flat"
        )

    if ENABLE_J2_HOME_STRONG_FLAT_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_home_flat"]
            & prob_home.notna() & prob_draw.notna()
            & (prob_draw <= prob_home + J2_HOME_STRONG_FLAT_HOME_RESTORE_DRAW_GAP_MAX)
            & (motivation_edge >= J2_HOME_STRONG_FLAT_HOME_RESTORE_MOTIVATION_EDGE)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_HOME_STRONG_FLAT_HOME_RESTORE", "home_flat", ctx, "home_flat"
        )

    home_draw_cond = pd.Series(False, index=out.index)
    if ENABLE_J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE:
        home_draw_cond |= (
            pred_d
            & ctx["cluster_home_draw"]
            & ctx["prob_ready"]
            & (prob_draw <= prob_home + J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_FATIGUE_EDGE)
            & (motivation_edge >= J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_MOTIVATION_EDGE)
            & (rank_gap <= J2_HOME_STRONG_DRAW_COMPRESSED_RESTORE_RANK_GAP_MAX)
        )
    if ENABLE_J2_HOME_STRONG_DEEP_HOME_RESTORE:
        home_draw_cond |= (
            pred_d
            & ctx["cluster_home_draw"]
            & prob_home.notna() & prob_draw.notna()
            & (prob_draw <= prob_home + J2_HOME_STRONG_DEEP_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_HOME_STRONG_DEEP_HOME_RESTORE_FATIGUE_EDGE)
            & (rank_gap <= J2_HOME_STRONG_DEEP_HOME_RESTORE_RANK_GAP_MAX)
        )
    if ENABLE_J2_HOME_STRONG_ELITE_HOME_RESTORE:
        home_draw_cond |= (
            pred_d
            & ctx["cluster_home_draw"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (prob_home >= J2_HOME_STRONG_ELITE_HOME_RESTORE_HOME_MIN)
            & (elo_diff >= J2_HOME_STRONG_ELITE_HOME_RESTORE_ELO_MIN)
            & (prob_draw <= prob_home)
        )
    out = _apply_j2_home_restore(
        out, home_draw_cond, "J2_HOME_DRAW_RESTORE", "home_draw", ctx, "home_draw"
    )

    if ENABLE_J2_NEUTRAL_POS_ELO_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_neutral_draw"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (prob_home >= J2_NEUTRAL_POS_ELO_HOME_RESTORE_HOME_MIN)
            & (draw_home_gap <= J2_NEUTRAL_POS_ELO_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_NEUTRAL_POS_ELO_HOME_RESTORE_FATIGUE_EDGE)
            & (elo_diff >= J2_NEUTRAL_POS_ELO_HOME_RESTORE_ELO_MIN)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_NEUTRAL_POS_ELO_HOME_RESTORE", "neutral_draw", ctx, "neutral_draw"
        )

    if ENABLE_J2_NEUTRAL_NEG_ELO_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_neutral_draw"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (draw_home_gap <= J2_NEUTRAL_NEG_ELO_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_NEUTRAL_NEG_ELO_HOME_RESTORE_FATIGUE_EDGE)
            & (motivation_edge >= J2_NEUTRAL_NEG_ELO_HOME_RESTORE_MOTIVATION_EDGE)
            & (rank_gap >= J2_NEUTRAL_NEG_ELO_HOME_RESTORE_RANK_GAP_MIN)
            & (elo_diff <= J2_NEUTRAL_NEG_ELO_HOME_RESTORE_ELO_MAX)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_NEUTRAL_NEG_ELO_HOME_RESTORE", "neutral_draw", ctx, "neutral_draw"
        )

    if ENABLE_J2_HOME_STRONG_TIGHT_BALANCED_RESTORE:
        home_balanced_cond |= (
            pred_d
            & ctx["cluster_home_balanced"]
            & prob_home.notna() & prob_draw.notna()
            & (draw_home_gap <= J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_FATIGUE_EDGE)
            & (rank_gap <= J2_HOME_STRONG_TIGHT_BALANCED_RESTORE_RANK_GAP_MAX)
        )
    out = _apply_j2_home_restore(
        out, home_balanced_cond, "J2_HOME_BALANCED_RESTORE", "home_balanced", ctx, "home_balanced"
    )

    if ENABLE_J2_NEUTRAL_TIGHT_BALANCED_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_neutral_balanced"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (draw_home_gap <= J2_NEUTRAL_TIGHT_BALANCED_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_NEUTRAL_TIGHT_BALANCED_RESTORE_FATIGUE_EDGE)
            & (elo_diff >= J2_NEUTRAL_TIGHT_BALANCED_RESTORE_ELO_MIN)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_NEUTRAL_TIGHT_BALANCED_RESTORE", "neutral_balanced", ctx, "neutral_balanced"
        )

    if ENABLE_J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_neutral_draw"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (draw_home_gap <= J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_FATIGUE_EDGE)
            & (rank_gap <= J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_RANK_GAP_MAX)
            & (elo_diff <= J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE_ELO_MAX)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_NEUTRAL_DEEP_DRAW_HOME_RESTORE", "neutral_draw", ctx, "neutral_draw"
        )

    if ENABLE_J2_NEUTRAL_POS_PRESS_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_neutral_draw"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (prob_home >= J2_NEUTRAL_POS_PRESS_HOME_RESTORE_HOME_MIN)
            & (draw_home_gap <= J2_NEUTRAL_POS_PRESS_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_NEUTRAL_POS_PRESS_HOME_RESTORE_FATIGUE_EDGE)
            & (elo_diff >= J2_NEUTRAL_POS_PRESS_HOME_RESTORE_ELO_MIN)
            & (motivation_edge <= J2_NEUTRAL_POS_PRESS_HOME_RESTORE_MOTIVATION_MAX)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_NEUTRAL_POS_PRESS_HOME_RESTORE", "neutral_draw", ctx, "neutral_draw"
        )

    if ENABLE_J2_NEUTRAL_DRAW_HOME_RESTORE_V2:
        cond = (
            pred_d
            & ctx["cluster_neutral_draw"]
            & prob_home.notna() & prob_draw.notna()
            & (draw_home_gap <= J2_NEUTRAL_DRAW_HOME_RESTORE_V2_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_NEUTRAL_DRAW_HOME_RESTORE_V2_FATIGUE_EDGE_MIN)
            & (rank_gap <= J2_NEUTRAL_DRAW_HOME_RESTORE_V2_RANK_GAP_MAX)
            & (home_sig_ct >= away_sig_ct)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_NEUTRAL_DRAW_HOME_RESTORE_V2", "neutral_draw", ctx, "neutral_draw"
        )

    if ENABLE_J2_NEUTRAL_DRAW_AWAY_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_neutral_draw"]
            & prob_away.notna() & prob_draw.notna()
            & (draw_away_gap <= J2_NEUTRAL_DRAW_AWAY_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_NEUTRAL_DRAW_AWAY_RESTORE_FATIGUE_EDGE_MIN)
            & (rank_gap >= J2_NEUTRAL_DRAW_AWAY_RESTORE_RANK_GAP_MIN)
            & (prob_away >= prob_home)
        )
        out = _apply_j2_away_restore(
            out, cond, "J2_NEUTRAL_DRAW_AWAY_RESTORE", "neutral_draw", ctx, "neutral_draw"
        )

    if ENABLE_J2_CLOSE_DRAW_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_close_draw"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (draw_home_gap <= J2_CLOSE_DRAW_HOME_RESTORE_DRAW_GAP_MAX)
            & (prob_home <= J2_CLOSE_DRAW_HOME_RESTORE_HOME_MAX)
            & (elo_diff <= J2_CLOSE_DRAW_HOME_RESTORE_ELO_MAX)
            & (motivation_edge >= J2_CLOSE_DRAW_HOME_RESTORE_MOTIVATION_MIN)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_CLOSE_DRAW_HOME_RESTORE", "close_draw", ctx, "close_draw"
        )

    if ENABLE_J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_neutral_balanced"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (prob_home >= J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_HOME_MIN)
            & (draw_home_gap <= J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_DRAW_GAP_MAX)
            & (fatigue_edge >= J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_FATIGUE_EDGE)
            & (elo_diff >= J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE_ELO_MIN)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_NEUTRAL_BALANCED_PRESS_HOME_RESTORE", "neutral_balanced", ctx, "neutral_balanced"
        )

    if ENABLE_J2_AWAY_STRONG_DRAW_HOME_RESTORE:
        cond = (
            pred_d
            & ctx["cluster_away_draw"]
            & prob_home.notna() & prob_draw.notna() & elo_diff.notna()
            & (prob_home <= J2_AWAY_STRONG_DRAW_HOME_RESTORE_HOME_MAX)
            & (draw_home_gap >= J2_AWAY_STRONG_DRAW_HOME_RESTORE_DRAW_GAP_MIN)
            & (fatigue_edge <= J2_AWAY_STRONG_DRAW_HOME_RESTORE_FATIGUE_EDGE_MAX)
            & (motivation_edge >= J2_AWAY_STRONG_DRAW_HOME_RESTORE_MOTIVATION_MIN)
            & (elo_diff <= J2_AWAY_STRONG_DRAW_HOME_RESTORE_ELO_MAX)
        )
        out = _apply_j2_home_restore(
            out, cond, "J2_AWAY_STRONG_DRAW_HOME_RESTORE", "away_draw", ctx, "away_draw"
        )

    if ENABLE_J2_AWAY_STRONG_AWAY_RESTORE:
        cond = pred_h & match_type.eq("away_strong")
        if cond.any():
            out.loc[cond, "predicted_result"] = "A"
            if "final_result" in out.columns:
                out.loc[cond, "final_result"] = "A"
            if "decision_reason" in out.columns:
                out.loc[cond, "decision_reason"] = "J2_AWAY_STRONG_AWAY_RESTORE"
            out.loc[cond, "j2_restore_cluster"] = "away_strong"
            out.loc[cond, "j2_restore_reason_group"] = "away_strong"
            out.loc[cond, "j2_away_strong_away_restore_applied"] = True
            out.loc[cond, "j2_away_strong_away_restore_reason"] = "match_type=away_strong and pred=H"

    if ENABLE_J2_SIGNAL_CONFLICT_AWAY_RESTORE:
        cond = pred_h & match_type.eq("signal_conflict")
        if cond.any():
            out.loc[cond, "predicted_result"] = "A"
            if "final_result" in out.columns:
                out.loc[cond, "final_result"] = "A"
            if "decision_reason" in out.columns:
                out.loc[cond, "decision_reason"] = "J2_SIGNAL_CONFLICT_AWAY_RESTORE"
            out.loc[cond, "j2_restore_cluster"] = "signal_conflict"
            out.loc[cond, "j2_restore_reason_group"] = "signal_conflict"
            out.loc[cond, "j2_signal_conflict_away_restore_applied"] = True
            out.loc[cond, "j2_signal_conflict_away_restore_reason"] = "match_type=signal_conflict and pred=H"

    if ENABLE_J2_NEG_HOME_ADV_AWAY_RESTORE and "home_advantage_diff" in out.columns:
        home_adv = pd.to_numeric(out["home_advantage_diff"], errors="coerce")
        cond = pred_h & home_adv.lt(0)
        if J2_NEG_HOME_ADV_AWAY_RESTORE_REQUIRE_DRAWRISK and "draw_risk_flag" in out.columns:
            draw_risk = out["draw_risk_flag"].astype(str).str.lower().isin(["true", "1"])
            cond &= draw_risk
        if "prob_draw" in out.columns:
            cond &= pd.to_numeric(out["prob_draw"], errors="coerce").ge(J2_NEG_HOME_ADV_AWAY_RESTORE_DRAW_MIN)
        if cond.any():
            out.loc[cond, "predicted_result"] = "A"
            if "final_result" in out.columns:
                out.loc[cond, "final_result"] = "A"
            if "decision_reason" in out.columns:
                out.loc[cond, "decision_reason"] = "J2_NEG_HOME_ADV_AWAY_RESTORE"
            out.loc[cond, "j2_restore_cluster"] = "neg_home_adv"
            out.loc[cond, "j2_restore_reason_group"] = "neg_home_adv"
            out.loc[cond, "j2_neg_home_adv_away_restore_applied"] = True
            out.loc[cond, "j2_neg_home_adv_away_restore_reason"] = (
                "home_advantage_diff<0"
                f"; require_drawrisk={int(J2_NEG_HOME_ADV_AWAY_RESTORE_REQUIRE_DRAWRISK)}"
                f"; prob_draw>={J2_NEG_HOME_ADV_AWAY_RESTORE_DRAW_MIN:.3f}"
            )

    if ENABLE_J2_AWAY_DRAW_RESTORE and {"prob_away_win", "prob_draw"}.issubset(out.columns):
        pred_a = out["predicted_result"].astype(str).eq("A")
        prob_away = pd.to_numeric(out["prob_away_win"], errors="coerce")
        prob_draw = pd.to_numeric(out["prob_draw"], errors="coerce")
        cond = (
            pred_a
            & prob_draw.ge(J2_AWAY_DRAW_RESTORE_DRAW_MIN)
            & (prob_away - prob_draw).le(J2_AWAY_DRAW_RESTORE_AWAY_GAP_MAX)
        )
        if J2_AWAY_DRAW_RESTORE_REQUIRE_DRAWRISK and "draw_risk_flag" in out.columns:
            draw_risk = out["draw_risk_flag"].astype(str).str.lower().isin(["true", "1"])
            cond &= draw_risk
        if J2_AWAY_DRAW_RESTORE_REQUIRE_LAB:
            lab_available = out.get("match_type_lab_available", pd.Series(False, index=out.index))
            lab_available = lab_available.astype(str).str.lower().isin(["true", "1"])
            lab_style_conflict = out.get("match_type_lab_style_conflict", pd.Series(False, index=out.index))
            lab_style_conflict = lab_style_conflict.astype(str).str.lower().isin(["true", "1"])
            lab_low_event = out.get("match_type_lab_low_event", pd.Series(False, index=out.index))
            lab_low_event = lab_low_event.astype(str).str.lower().isin(["true", "1"])
            lab_edge = pd.to_numeric(
                out.get("match_type_lab_matchup_edge", pd.Series(np.nan, index=out.index)),
                errors="coerce",
            )
            lab_gate = lab_style_conflict | lab_low_event | lab_edge.abs().le(J2_AWAY_DRAW_RESTORE_LAB_EDGE_MAX)
            cond &= lab_available & lab_gate
        if cond.any():
            out.loc[cond, "predicted_result"] = "D"
            if "final_result" in out.columns:
                out.loc[cond, "final_result"] = "D"
            if "decision_reason" in out.columns:
                out.loc[cond, "decision_reason"] = "J2_AWAY_DRAW_RESTORE"
            out.loc[cond, "j2_restore_cluster"] = "away_draw_restore"
            out.loc[cond, "j2_restore_reason_group"] = "away_draw_restore"
            out.loc[cond, "j2_away_draw_restore_applied"] = True
            out.loc[cond, "j2_away_draw_restore_reason"] = (
                f"pred=A; prob_draw>={J2_AWAY_DRAW_RESTORE_DRAW_MIN:.3f}"
                f"; away_draw_gap<={J2_AWAY_DRAW_RESTORE_AWAY_GAP_MAX:.3f}"
                f"; require_drawrisk={int(J2_AWAY_DRAW_RESTORE_REQUIRE_DRAWRISK)}"
                f"; require_lab={int(J2_AWAY_DRAW_RESTORE_REQUIRE_LAB)}"
                f"; lab_edge<={J2_AWAY_DRAW_RESTORE_LAB_EDGE_MAX:.1f}"
            )
    return out


def _calc_backtest_metrics_for_sensitivity(backtest_path):
    if not backtest_path or not os.path.exists(backtest_path):
        return {"backtest_rows": 0, "acc": None, "logloss": None}
    try:
        df = pd.read_csv(backtest_path)
    except Exception:
        return {"backtest_rows": 0, "acc": None, "logloss": None}
    if df.empty:
        return {"backtest_rows": 0, "acc": None, "logloss": None}
    pred_col = "final_result" if "final_result" in df.columns else ("predicted_result" if "predicted_result" in df.columns else None)
    if pred_col is None or "actual_result" not in df.columns:
        return {"backtest_rows": int(len(df)), "acc": None, "logloss": None}
    actual = df["actual_result"].astype(str).str.upper()
    pred = df[pred_col].astype(str).str.upper()
    valid = actual.isin(["H", "D", "A"]) & pred.isin(["H", "D", "A"])
    acc = float((pred[valid] == actual[valid]).mean()) if int(valid.sum()) > 0 else None
    ll = _calc_multiclass_logloss_from_df(df.loc[valid].copy()) if int(valid.sum()) > 0 else None
    return {"backtest_rows": int(valid.sum()), "acc": acc, "logloss": ll}


def _summarize_prediction_for_sensitivity(path):
    df = pd.read_csv(path)
    if df.empty:
        return {"rows": 0}
    final_col = "final_result" if "final_result" in df.columns else "predicted_result"
    argmax_col = "argmax_result" if "argmax_result" in df.columns else None
    dist_final = _calc_hda_dist_for_sensitivity(df[final_col]) if final_col in df.columns else {"rows": 0, "H_cnt": 0, "D_cnt": 0, "A_cnt": 0, "H_pct": 0.0, "D_pct": 0.0, "A_pct": 0.0}
    dist_arg = _calc_hda_dist_for_sensitivity(df[argmax_col]) if (argmax_col and argmax_col in df.columns) else {"rows": 0, "H_cnt": 0, "D_cnt": 0, "A_cnt": 0, "H_pct": 0.0, "D_pct": 0.0, "A_pct": 0.0}
    means = _calc_prob_means_for_sensitivity(df) or {"rows": 0, "home_mean": float("nan"), "draw_mean": float("nan"), "away_mean": float("nan")}
    force_cnt = int(df.get("decision_reason", pd.Series("", index=df.index)).astype(str).str.contains("FORCE_DRAW", na=False).sum())
    out = {
        "rows": int(len(df)),
        "final": dist_final,
        "argmax": dist_arg,
        "means": means,
        "force_draw_count": force_cnt,
    }
    by_league = {}
    if "league" in df.columns:
        for lg, part in df.groupby("league", dropna=False, sort=True):
            lg_label = str(lg)
            by_league[lg_label] = {
                "rows": int(len(part)),
                "final": _calc_hda_dist_for_sensitivity(part[final_col]) if final_col in part.columns else {"rows": 0, "H_cnt": 0, "D_cnt": 0, "A_cnt": 0, "H_pct": 0.0, "D_pct": 0.0, "A_pct": 0.0},
                "argmax": _calc_hda_dist_for_sensitivity(part[argmax_col]) if (argmax_col and argmax_col in part.columns) else {"rows": 0, "H_cnt": 0, "D_cnt": 0, "A_cnt": 0, "H_pct": 0.0, "D_pct": 0.0, "A_pct": 0.0},
                "means": _calc_prob_means_for_sensitivity(part) or {"rows": 0, "home_mean": float("nan"), "draw_mean": float("nan"), "away_mean": float("nan")},
                "force_draw_count": int(part.get("decision_reason", pd.Series("", index=part.index)).astype(str).str.contains("FORCE_DRAW", na=False).sum()),
            }
    out["by_league"] = by_league
    return out


def _score_sensitivity_candidate(row, league):
    true_draw_target = 0.255 if str(league).lower() == "j1" else 0.284
    rate_h = float(row.get("rate_H_final", 0.0))
    rate_d = float(row.get("rate_D_final", 0.0))
    rate_a = float(row.get("rate_A_final", 0.0))
    draw_gap = abs((rate_d / 100.0) - true_draw_target)
    away_floor = 15.0
    away_penalty = max(0.0, away_floor - rate_a) / 100.0
    symmetry_gap = abs(rate_h - rate_a) / 100.0
    logloss = row.get("logloss")
    if logloss is None or (isinstance(logloss, float) and np.isnan(logloss)):
        logloss_term = 0.0
    else:
        logloss_term = float(logloss)
    score = (away_penalty * 5.0) + (draw_gap * 2.0) + (symmetry_gap * 0.8) + (logloss_term * 0.1)
    reason = (
        f"score={score:.4f} away_penalty={away_penalty:.4f} "
        f"draw_gap={draw_gap:.4f} symmetry_gap={symmetry_gap:.4f} "
        f"logloss_term={logloss_term:.4f} target_draw={true_draw_target:.3f}"
    )
    return score, reason


def _log_sensitivity_recommendations(rows, league, season_year):
    if not rows:
        print(f"[SENSITIVITY_RECOMMEND] league={league} season={season_year} unavailable")
        return
    scored = []
    for r in rows:
        score, reason = _score_sensitivity_candidate(r, league)
        rr = dict(r)
        rr["_score"] = score
        rr["_reason"] = reason
        scored.append(rr)
    scored.sort(key=lambda x: x["_score"])
    top3 = scored[:3]
    top_parts = []
    for i, t in enumerate(top3, 1):
        top_parts.append(
            f"#{i}(hfa={t['hfa_elo']:.1f},elo_scale={t['elo_diff_scale']:.2f},draw_assign={t['draw_assign_by_expectation']},"
            f"H/D/A={t['pred_H_final']}/{t['pred_D_final']}/{t['pred_A_final']},score={t['_score']:.4f})"
        )
    print(f"[SENSITIVITY_TOP3] league={league} season={season_year} " + " ".join(top_parts))
    best = top3[0]
    print(
        f"[SENSITIVITY_RECOMMEND] league={league} season={season_year} "
        f"hfa={best['hfa_elo']:.1f} elo_scale={best['elo_diff_scale']:.2f} "
        f"draw_assign={best['draw_assign_by_expectation']} reason={best['_reason']}"
    )


def _run_sensitivity_scan_generation():
    hfa_values = _parse_sensitivity_float_values(SENSITIVITY_HFA_VALUES_RAW, [0.0, 10.0, 20.0, 35.0])
    scale_values = _parse_sensitivity_float_values(SENSITIVITY_ELO_SCALE_VALUES_RAW, [0.5, 1.0, 1.5])
    draw_assign_values = [1 if int(v) != 0 else 0 for v in _parse_sensitivity_int_values(SENSITIVITY_DRAW_ASSIGN_VALUES_RAW, [0, 1])]
    args = [a for a in RAW_CLI_ARGS if a not in {
        "--sensitivity-scan",
        "--self-check-hfa",
        "--skip-hfa-self-check",
        "--sensitivity-hfa-values",
        "--sensitivity-elo-scale-values",
        "--sensitivity-draw-assign-values",
    }]
    for flag in ["--sensitivity-hfa-values", "--sensitivity-elo-scale-values", "--sensitivity-draw-assign-values"]:
        if flag in args:
            idx = args.index(flag)
            del args[idx: idx + 2]

    script_path = os.path.abspath(__file__)
    rows = []
    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(
        f"[SENSITIVITY] start league={LEAGUE} season={SEASON_YEAR} "
        f"hfa_values={hfa_values} elo_scale_values={scale_values} draw_assign_values={draw_assign_values}"
    )

    for hfa_val in hfa_values:
        for scale_val in scale_values:
            for draw_assign in draw_assign_values:
                tag = f"hfa{_safe_name_num(hfa_val)}_s{_safe_name_num(scale_val)}_draw{draw_assign}"
                out_path = os.path.join(BASE_DIR, f"{LEAGUE}_{SEASON_YEAR}_predictions_sensitivity_{tag}.csv")
                env = os.environ.copy()
                env["OUTPUT_PRED_CSV"] = out_path
                env["HFA_ELO"] = str(hfa_val)
                env["ELO_DIFF_SCALE"] = str(scale_val)
                env["DRAW_ASSIGN_BY_EXPECTATION"] = str(draw_assign)
                env["SENSITIVITY_SCAN"] = "0"
                env["SKIP_HFA_SELF_CHECK"] = "1"
                cmd = [sys.executable, script_path] + args
                if "--force" not in cmd:
                    cmd.append("--force")
                print(
                    f"[SENSITIVITY_RUN] tag={tag} HFA_ELO={hfa_val:.3f} "
                    f"ELO_DIFF_SCALE={scale_val:.3f} DRAW_ASSIGN_BY_EXPECTATION={draw_assign} out={out_path}"
                )
                subprocess.run(cmd, check=True, cwd=BASE_DIR, env=env)
                summary = _summarize_prediction_for_sensitivity(out_path)
                backtest_metrics = _calc_backtest_metrics_for_sensitivity(backtest_output_csv)
                row = {
                    "league": LEAGUE,
                    "season_year": int(SEASON_YEAR),
                    "tag": tag,
                    "hfa_elo": float(hfa_val),
                    "elo_diff_scale": float(scale_val),
                    "draw_assign_by_expectation": int(draw_assign),
                    "rows": int(summary.get("rows", 0)),
                    "pred_count_total": int(summary.get("rows", 0)),
                    "force_draw_count": int(summary.get("force_draw_count", 0)),
                    "final_H_cnt": int(summary["final"]["H_cnt"]),
                    "final_D_cnt": int(summary["final"]["D_cnt"]),
                    "final_A_cnt": int(summary["final"]["A_cnt"]),
                    "final_H_pct": float(summary["final"]["H_pct"]),
                    "final_D_pct": float(summary["final"]["D_pct"]),
                    "final_A_pct": float(summary["final"]["A_pct"]),
                    "pred_H_final": int(summary["final"]["H_cnt"]),
                    "pred_D_final": int(summary["final"]["D_cnt"]),
                    "pred_A_final": int(summary["final"]["A_cnt"]),
                    "rate_H_final": float(summary["final"]["H_pct"]),
                    "rate_D_final": float(summary["final"]["D_pct"]),
                    "rate_A_final": float(summary["final"]["A_pct"]),
                    "argmax_H_cnt": int(summary["argmax"]["H_cnt"]),
                    "argmax_D_cnt": int(summary["argmax"]["D_cnt"]),
                    "argmax_A_cnt": int(summary["argmax"]["A_cnt"]),
                    "argmax_H_pct": float(summary["argmax"]["H_pct"]),
                    "argmax_D_pct": float(summary["argmax"]["D_pct"]),
                    "argmax_A_pct": float(summary["argmax"]["A_pct"]),
                    "pred_H_argmax": int(summary["argmax"]["H_cnt"]),
                    "pred_D_argmax": int(summary["argmax"]["D_cnt"]),
                    "pred_A_argmax": int(summary["argmax"]["A_cnt"]),
                    "rate_H_argmax": float(summary["argmax"]["H_pct"]),
                    "rate_D_argmax": float(summary["argmax"]["D_pct"]),
                    "rate_A_argmax": float(summary["argmax"]["A_pct"]),
                    "mean_prob_home": float(summary["means"]["home_mean"]),
                    "mean_prob_draw": float(summary["means"]["draw_mean"]),
                    "mean_prob_away": float(summary["means"]["away_mean"]),
                    "prob_mean_home": float(summary["means"]["home_mean"]),
                    "prob_mean_draw": float(summary["means"]["draw_mean"]),
                    "prob_mean_away": float(summary["means"]["away_mean"]),
                    "acc": backtest_metrics.get("acc"),
                    "logloss": backtest_metrics.get("logloss"),
                    "backtest_rows": int(backtest_metrics.get("backtest_rows", 0)),
                }
                rows.append(row)
                print(
                    f"[SENSITIVITY_RESULT] league={LEAGUE} tag={tag} rows={row['rows']} "
                    f"FINAL(H/D/A)={row['final_H_cnt']}/{row['final_D_cnt']}/{row['final_A_cnt']} "
                    f"ARGMAX(H/D/A)={row['argmax_H_cnt']}/{row['argmax_D_cnt']}/{row['argmax_A_cnt']} "
                    f"PROB_MEAN(H/D/A)={row['mean_prob_home']:.4f}/{row['mean_prob_draw']:.4f}/{row['mean_prob_away']:.4f} "
                    f"FORCE_DRAW={row['force_draw_count']}"
                )
                for lg, part in summary.get("by_league", {}).items():
                    print(
                        f"[SENSITIVITY_RESULT:LEAGUE] league={lg} tag={tag} rows={part['rows']} "
                        f"FINAL(H/D/A)={part['final']['H_cnt']}/{part['final']['D_cnt']}/{part['final']['A_cnt']} "
                        f"ARGMAX(H/D/A)={part['argmax']['H_cnt']}/{part['argmax']['D_cnt']}/{part['argmax']['A_cnt']} "
                        f"PROB_MEAN(H/D/A)={part['means']['home_mean']:.4f}/{part['means']['draw_mean']:.4f}/{part['means']['away_mean']:.4f} "
                        f"FORCE_DRAW={part['force_draw_count']}"
                    )

    table_path = os.path.join(logs_dir, f"hda_sensitivity_{LEAGUE}_{SEASON_YEAR}.csv")
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"[SENSITIVITY_TABLE] rows={len(rows)} saved={table_path}")
    _log_sensitivity_recommendations(rows, LEAGUE, int(SEASON_YEAR))


def _profile_scan_env_float(name, default):
    raw = os.environ.get(name, "")
    if str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        print(f"[PROFILE_SCAN][WARN] invalid float env {name}={raw!r}; fallback={default}")
        return float(default)


def _build_profile_scan_presets():
    # multinom本線で効く差分レバーを対象にする。
    presets = [
        {
            "profile": "baseline",
            "description": "current env baseline",
            "params": {},
        },
        {
            "profile": "light",
            "description": "HFA/Elo差の効きを少し強める",
            "params": {
                "HFA_ELO": _profile_scan_env_float("PROFILE_SCAN_LIGHT_HFA_ELO", 40.0),
                "ELO_DIFF_SCALE": _profile_scan_env_float("PROFILE_SCAN_LIGHT_ELO_DIFF_SCALE", 1.10),
                "ELO_D_VALUE": _profile_scan_env_float("PROFILE_SCAN_LIGHT_ELO_D_VALUE", 600.0),
                "HFA_PROB_WEIGHT": _profile_scan_env_float("PROFILE_SCAN_LIGHT_HFA_PROB_WEIGHT", 0.65),
            },
        },
        {
            "profile": "medium",
            "description": "HFA/Elo差の効きを中程度に強める",
            "params": {
                "HFA_ELO": _profile_scan_env_float("PROFILE_SCAN_MEDIUM_HFA_ELO", 45.0),
                "ELO_DIFF_SCALE": _profile_scan_env_float("PROFILE_SCAN_MEDIUM_ELO_DIFF_SCALE", 1.20),
                "ELO_D_VALUE": _profile_scan_env_float("PROFILE_SCAN_MEDIUM_ELO_D_VALUE", 500.0),
                "HFA_PROB_WEIGHT": _profile_scan_env_float("PROFILE_SCAN_MEDIUM_HFA_PROB_WEIGHT", 0.70),
            },
        },
        {
            "profile": "strong",
            "description": "HFA/Elo差の効きをやや強める",
            "params": {
                "HFA_ELO": _profile_scan_env_float("PROFILE_SCAN_STRONG_HFA_ELO", 50.0),
                "ELO_DIFF_SCALE": _profile_scan_env_float("PROFILE_SCAN_STRONG_ELO_DIFF_SCALE", 1.30),
                "ELO_D_VALUE": _profile_scan_env_float("PROFILE_SCAN_STRONG_ELO_D_VALUE", 400.0),
                "HFA_PROB_WEIGHT": _profile_scan_env_float("PROFILE_SCAN_STRONG_HFA_PROB_WEIGHT", 0.75),
            },
        },
    ]
    if str(LEAGUE).lower() == "j2":
        presets.append(
            {
                "profile": "j2_mid",
                "description": "J2向けの light-medium 中間",
                "params": {
                    "HFA_ELO": _profile_scan_env_float("PROFILE_SCAN_J2_MID_HFA_ELO", 42.5),
                    "ELO_DIFF_SCALE": _profile_scan_env_float("PROFILE_SCAN_J2_MID_ELO_DIFF_SCALE", 1.15),
                    "ELO_D_VALUE": _profile_scan_env_float("PROFILE_SCAN_J2_MID_ELO_D_VALUE", 550.0),
                    "HFA_PROB_WEIGHT": _profile_scan_env_float("PROFILE_SCAN_J2_MID_HFA_PROB_WEIGHT", 0.675),
                },
            }
        )
    presets.extend(
        [
            {
                "profile": "baseline_signfix",
                "description": "baseline + elo sign monotonic fix",
                "params": {
                    "ENFORCE_ELO_SIGN_MONOTONIC": 1,
                },
            },
            {
                "profile": "light_signfix",
                "description": "light + elo sign monotonic fix",
                "params": {
                    "HFA_ELO": _profile_scan_env_float("PROFILE_SCAN_LIGHT_HFA_ELO", 40.0),
                    "ELO_DIFF_SCALE": _profile_scan_env_float("PROFILE_SCAN_LIGHT_ELO_DIFF_SCALE", 1.10),
                    "ELO_D_VALUE": _profile_scan_env_float("PROFILE_SCAN_LIGHT_ELO_D_VALUE", 600.0),
                    "HFA_PROB_WEIGHT": _profile_scan_env_float("PROFILE_SCAN_LIGHT_HFA_PROB_WEIGHT", 0.65),
                    "ENFORCE_ELO_SIGN_MONOTONIC": 1,
                },
            },
        ]
    )
    if str(LEAGUE).lower() == "j2":
        presets.append(
            {
                "profile": "j2_mid_signfix",
                "description": "J2中間案 + elo sign monotonic fix",
                "params": {
                    "HFA_ELO": _profile_scan_env_float("PROFILE_SCAN_J2_MID_HFA_ELO", 42.5),
                    "ELO_DIFF_SCALE": _profile_scan_env_float("PROFILE_SCAN_J2_MID_ELO_DIFF_SCALE", 1.15),
                    "ELO_D_VALUE": _profile_scan_env_float("PROFILE_SCAN_J2_MID_ELO_D_VALUE", 550.0),
                    "HFA_PROB_WEIGHT": _profile_scan_env_float("PROFILE_SCAN_J2_MID_HFA_PROB_WEIGHT", 0.675),
                    "ENFORCE_ELO_SIGN_MONOTONIC": 1,
                },
            }
        )
    return presets


def _score_profile_scan_candidate(row):
    hit = row.get("hit_rate_argmax")
    logloss = row.get("logloss")
    brier = row.get("brier")
    maxp_gt = row.get("ratio_maxp_gt_060")
    maxp_lt = row.get("ratio_maxp_lt_050")
    hit_term = 1.0 - float(hit) if hit is not None and not pd.isna(hit) else 1.0
    logloss_term = float(logloss) if logloss is not None and not pd.isna(logloss) else 1.0
    brier_term = float(brier) if brier is not None and not pd.isna(brier) else 1.0
    gt_bonus = float(maxp_gt) if maxp_gt is not None and not pd.isna(maxp_gt) else 0.0
    lt_penalty = float(maxp_lt) if maxp_lt is not None and not pd.isna(maxp_lt) else 1.0
    return (hit_term * 2.0) + (logloss_term * 0.8) + (brier_term * 0.6) + (lt_penalty * 0.5) - (gt_bonus * 0.3)


def _pick_profile_scan_recommendation(league_rows):
    if league_rows is None or league_rows.empty:
        return None, "unavailable"
    baseline_part = league_rows[league_rows["profile"] == "baseline"]
    if baseline_part.empty:
        ranked = league_rows.copy()
        ranked["_score"] = ranked.apply(_score_profile_scan_candidate, axis=1)
        ranked = ranked.sort_values(["_score", "profile"]).reset_index(drop=True)
        return ranked.iloc[0].to_dict(), "score_only"
    baseline = baseline_part.iloc[0]
    base_ll = float(baseline["logloss"])
    base_br = float(baseline["brier"])
    tol_logloss = 0.01
    tol_brier = 0.01
    safe = league_rows[
        (pd.to_numeric(league_rows["logloss"], errors="coerce") <= base_ll + tol_logloss) &
        (pd.to_numeric(league_rows["brier"], errors="coerce") <= base_br + tol_brier)
    ].copy()
    if safe.empty:
        return baseline.to_dict(), "baseline_guard"
    safe = safe.sort_values(
        ["hit_rate_argmax", "logloss", "brier", "ratio_maxp_gt_060"],
        ascending=[False, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return safe.iloc[0].to_dict(), "safe_window"


def _run_profile_scan_generation():
    profiles = _build_profile_scan_presets()
    args = [a for a in RAW_CLI_ARGS if a not in {"--profile-scan"}]
    script_path = os.path.abspath(__file__)
    rows = []
    os.makedirs(PROFILE_SCAN_DIR, exist_ok=True)
    run_dir = os.path.join(PROFILE_SCAN_DIR, f"profile_scan_runs_{LEAGUE}_{SEASON_YEAR}")
    os.makedirs(run_dir, exist_ok=True)
    print(
        f"[PROFILE_SCAN] start league={LEAGUE} season={SEASON_YEAR} "
        f"decision_rule=argmax(prob_home_win,prob_draw,prob_away_win) profiles={[p['profile'] for p in profiles]}"
    )
    for profile in profiles:
        tag = str(profile["profile"])
        env = os.environ.copy()
        env["PROFILE_SCAN"] = "0"
        env["SKIP_HFA_SELF_CHECK"] = "1"
        env["OUTPUT_PRED_CSV"] = os.path.join(run_dir, f"{LEAGUE}_{SEASON_YEAR}_predictions_{tag}.csv")
        env["BACKTEST_OUTPUT_CSV"] = os.path.join(run_dir, f"backtest_{LEAGUE}_{SEASON_YEAR}_{tag}.csv")
        for key, value in profile.get("params", {}).items():
            env[str(key)] = str(value)
        cmd = [sys.executable, script_path] + args
        if "--force" not in cmd:
            cmd.append("--force")
        print(
            f"[PROFILE_SCAN_RUN] profile={tag} "
            f"HFA_ELO={env.get('HFA_ELO', HFA_ELO)} "
            f"ELO_DIFF_SCALE={env.get('ELO_DIFF_SCALE', ELO_DIFF_SCALE)} "
            f"ELO_D_VALUE={env.get('ELO_D_VALUE', ELO_D_VALUE)} "
            f"HFA_PROB_WEIGHT={env.get('HFA_PROB_WEIGHT', HFA_PROB_WEIGHT)} "
            f"backtest={env['BACKTEST_OUTPUT_CSV']}"
        )
        subprocess.run(cmd, check=True, cwd=BASE_DIR, env=env)
        metric_rows = _calc_profile_scan_metrics(env["BACKTEST_OUTPUT_CSV"])
        for metric in metric_rows:
            row = {
                "profile": tag,
                "profile_desc": profile.get("description", ""),
                "decision_rule": DECISION_RULE_DESC,
                "league": metric.get("league", str(LEAGUE).upper()),
                "matches": int(metric.get("matches", 0)),
                "hit_rate_argmax": metric.get("hit_rate_argmax"),
                "logloss": metric.get("logloss"),
                "brier": metric.get("brier"),
                "count_maxp_gt_060": int(metric.get("count_maxp_gt_060", 0)),
                "count_maxp_050_060": int(metric.get("count_maxp_050_060", 0)),
                "count_maxp_lt_050": int(metric.get("count_maxp_lt_050", 0)),
                "ratio_maxp_gt_060": float(metric.get("ratio_maxp_gt_060", 0.0)),
                "ratio_maxp_050_060": float(metric.get("ratio_maxp_050_060", 0.0)),
                "ratio_maxp_lt_050": float(metric.get("ratio_maxp_lt_050", 0.0)),
                "hfa_elo": float(env.get("HFA_ELO", HFA_ELO)),
                "elo_diff_scale": float(env.get("ELO_DIFF_SCALE", ELO_DIFF_SCALE)),
                "elo_d_value": float(env.get("ELO_D_VALUE", ELO_D_VALUE)),
                "hfa_prob_weight": float(env.get("HFA_PROB_WEIGHT", HFA_PROB_WEIGHT)),
                "enforce_elo_sign_monotonic": int(str(env.get("ENFORCE_ELO_SIGN_MONOTONIC", 0)).strip() == "1"),
            }
            rows.append(row)
            print(
                f"[PROFILE_SCAN_RESULT] profile={row['profile']} league={row['league']} matches={row['matches']} "
                f"maxP>0.60={row['count_maxp_gt_060']}({row['ratio_maxp_gt_060']:.1%}) "
                f"maxP0.50-0.60={row['count_maxp_050_060']}({row['ratio_maxp_050_060']:.1%}) "
                f"maxP<0.50={row['count_maxp_lt_050']}({row['ratio_maxp_lt_050']:.1%}) "
                f"hit_argmax={row['hit_rate_argmax']:.4f} "
                f"logloss={row['logloss']:.4f} brier={row['brier']:.4f}"
            )
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["league", "profile"]).reset_index(drop=True)
    out_path = os.path.join(PROFILE_SCAN_DIR, f"profile_scan_{LEAGUE}_{SEASON_YEAR}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[PROFILE_SCAN_TABLE] rows={len(out_df)} saved={out_path}")
    league_rows = out_df[out_df["league"].astype(str).str.upper() == str(LEAGUE).upper()].copy()
    if not league_rows.empty:
        baseline_row = league_rows[league_rows["profile"] == "baseline"]
        baseline_row = baseline_row.iloc[0].to_dict() if not baseline_row.empty else None
        best, recommend_mode = _pick_profile_scan_recommendation(league_rows)
        print(
            f"[PROFILE_SCAN_RECOMMEND] league={LEAGUE.upper()} season={SEASON_YEAR} "
            f"best={best['profile']} hit_argmax={best['hit_rate_argmax']:.4f} "
            f"logloss={best['logloss']:.4f} brier={best['brier']:.4f} "
            f"maxP>0.60={best['ratio_maxp_gt_060']:.1%} mode={recommend_mode}"
        )
        if baseline_row:
            print(
                f"[PROFILE_SCAN_BASELINE] league={LEAGUE.upper()} season={SEASON_YEAR} "
                f"hit_argmax={baseline_row['hit_rate_argmax']:.4f} "
                f"logloss={baseline_row['logloss']:.4f} brier={baseline_row['brier']:.4f} "
                f"maxP>0.60={baseline_row['ratio_maxp_gt_060']:.1%} "
                f"maxP<0.50={baseline_row['ratio_maxp_lt_050']:.1%}"
            )


def log_run_config():
    print(
        "[CONFIG] "
        f"HDA_MODEL_MODE={HDA_MODEL_MODE} HDA_FEATURE_PROFILE={HDA_FEATURE_PROFILE_EFFECTIVE or 'default'} "
        f"HDA_MODEL_EFFECTIVE={HDA_MODEL_MODE_EFFECTIVE} HDA_MODEL_PATH={HDA_MODEL_PATH} "
        f"ENABLE_HFA={ENABLE_HFA_INT} HFA_ELO={HFA_ELO:.2f} HFA_BASE_APPLIED={HFA_ELO if ENABLE_HFA else 0.0:.2f} "
        f"ENABLE_MATCHUP_BIAS={int(ENABLE_MATCHUP_BIAS)} MATCHUP_BIAS_COEF={MATCHUP_BIAS_COEF:.3f} "
        f"OUT={output_csv} FORCE={int(FORCE_RECALC)} STRICT_MODE={int(STRICT_MODE)} "
        f"ELO_DIFF_SCALE={ELO_DIFF_SCALE:.3f} ELO_D_VALUE={ELO_D_VALUE:.1f} "
        f"HFA_PROB_WEIGHT={HFA_PROB_WEIGHT:.3f} "
        f"MULTINOM_ELO_DIFF_SIGN={int(MULTINOM_ELO_DIFF_SIGN)} "
        f"MULTINOM_SWAP_HA_OUTPUT={int(MULTINOM_SWAP_HA_OUTPUT)} "
        f"DRAW_DECAY_SCALE={DRAW_DECAY_SCALE:.1f} "
        f"DRAW_BLEND_WEIGHT={DRAW_BLEND_WEIGHT:.3f} ELO_DRAW_MIN={ELO_DRAW_MIN:.3f} "
        f"ELO_DRAW_MAX={ELO_DRAW_MAX:.3f} ELO_DRAW_BASE={ELO_DRAW_BASE:.3f} "
        f"ELO_DRAW_BUMP={ELO_DRAW_BUMP:.3f} ELO_DRAW_DIFF_SCALE={ELO_DRAW_DIFF_SCALE:.3f} "
        f"ENFORCE_ELO_SIGN_MONOTONIC={int(ENFORCE_ELO_SIGN_MONOTONIC)} "
        f"DRAW_TWEAK_MODE={DRAW_TWEAK_MODE} DRAW_TWEAK_ENABLED={int(DRAW_TWEAK_ENABLED)} "
        f"DRAW_ASSIGN_BY_EXPECTATION_RAW={int(DRAW_ASSIGN_BY_EXPECTATION_RAW)} "
        f"DRAW_ASSIGN_BY_EXPECTATION={int(DRAW_ASSIGN_BY_EXPECTATION)} "
        f"DRAW_EXPECTATION_MULTIPLIER_RAW={DRAW_EXPECTATION_MULTIPLIER_RAW:.3f} "
        f"DRAW_EXPECTATION_MULTIPLIER={DRAW_EXPECTATION_MULTIPLIER:.3f} "
        f"DRAW_MARGIN={DRAW_MARGIN:.3f} "
        f"CLOSE_HA_GAP={CLOSE_HA_GAP:.3f} CLOSE_HA_MIN_LEVEL={CLOSE_HA_MIN_LEVEL:.3f} "
        f"CLOSE_HA_GAP_WEIGHT={CLOSE_HA_GAP_WEIGHT:.3f} "
        f"CLOSE_HA_DRAW_SCORE_MIN={CLOSE_HA_DRAW_SCORE_MIN:.3f} "
        f"CLOSE_HA_DRAW_SCORE_MIN_GRID={CLOSE_HA_DRAW_SCORE_MIN_GRID_RAW} "
        f"CLOSE_D_TOP_GAP={CLOSE_D_TOP_GAP:.3f} "
        f"CLOSE_D_TOP_GAP_GRID={CLOSE_D_TOP_GAP_GRID} "
        f"ABSENCE_ELO_COEF={ABSENCE_ELO_COEF:.4f} ABSENCE_ELO_ADJUST_CLIP={ABSENCE_ELO_ADJUST_CLIP:.4f} "
        f"DRAW_ASSIGN_GROUP_MODE={DRAW_ASSIGN_GROUP_MODE} "
        f"BACKTEST_DECISION_RULE={BACKTEST_DECISION_RULE} "
        f"BACKTEST_COMPARE_DATASET={BACKTEST_COMPARE_DATASET} "
        f"BACKTEST_COMPARE_CSV={BACKTEST_COMPARE_CSV or '-'} "
        f"BACKTEST_MARGIN_SCAN={int(BACKTEST_MARGIN_SCAN)} "
        f"DRAW_MARGIN_GRID={DRAW_MARGIN_GRID_RAW}"
    )


def log_hfa_apply_path():
    print("[HFA_APPLY_PATH] active_path=compute_probabilities_and_result:elo_diff_for_prob (single source of HFA addition)")


_init_hda_model()
log_run_config()
log_hfa_apply_path()
if SELF_CHECK_HFA and (not SKIP_HFA_SELF_CHECK):
    _run_hfa_self_check_generation()
    sys.exit(0)
if PROFILE_SCAN:
    _run_profile_scan_generation()
    sys.exit(0)
if SENSITIVITY_SCAN:
    _run_sensitivity_scan_generation()
    sys.exit(0)

TEAM_NAME_ALIAS_RAW_MAP = {
    "G大阪": "G大阪",
    "ガンバ大阪": "G大阪",
    "C大阪": "C大阪",
    "セレッソ大阪": "C大阪",
    "横浜FM": "横浜FM",
    "横浜Fマリノス": "横浜FM",
    "横浜F・マリノス": "横浜FM",
    "横浜FC": "横浜FC",
    "FC東京": "FC東京",
    "FCTOKYO": "FC東京",
    "川崎F": "川崎F",
    "川崎フロンターレ": "川崎F",
    "東京V": "東京V",
    "東京ヴェルディ": "東京V",
    "湘南": "湘南",
    "湘南ベルマーレ": "湘南",
    "神戸": "神戸",
    "ヴィッセル神戸": "神戸",
    "名古屋": "名古屋",
    "名古屋グランパス": "名古屋",
    "浦和": "浦和",
    "浦和レッズ": "浦和",
    "広島": "広島",
    "サンフレッチェ広島": "広島",
    "福岡": "福岡",
    "アビスパ福岡": "福岡",
    "清水": "清水",
    "清水エスパルス": "清水",
    "新潟": "新潟",
    "アルビレックス新潟": "新潟",
    "千葉": "千葉",
    "ジェフユナイテッド千葉": "千葉",
    "鹿島": "鹿島",
    "鹿島アントラーズ": "鹿島",
    "柏": "柏",
    "柏レイソル": "柏",
    "水戸": "水戸",
    "水戸ホーリーホック": "水戸",
    "長崎": "長崎",
    "V・ファーレン長崎": "長崎",
    "Ｖ・ファーレン長崎": "長崎",
    "町田": "町田",
    "FC町田ゼルビア": "町田",
    "岡山": "岡山",
    "ファジアーノ岡山": "岡山",
    "京都": "京都",
    "京都サンガFC": "京都",
    "京都サンガF.C.": "京都",
    "鳥栖": "鳥栖",
    "サガン鳥栖": "鳥栖",
    "仙台": "仙台",
    "ベガルタ仙台": "仙台",
    "秋田": "秋田",
    "ブラウブリッツ秋田": "秋田",
    "山形": "山形",
    "モンテディオ山形": "山形",
    "群馬": "群馬",
    "ザスパ群馬": "群馬",
    "いわき": "いわき",
    "いわきFC": "いわき",
    "いわきＦＣ": "いわき",
    "大宮": "大宮",
    "RB大宮アルディージャ": "大宮",
    "ＲＢ大宮アルディージャ": "大宮",
    "甲府": "甲府",
    "ヴァンフォーレ甲府": "甲府",
    "札幌": "札幌",
    "北海道コンサドーレ札幌": "札幌",
    "八戸": "八戸",
    "ヴァンラーレ八戸": "八戸",
    "磐田": "磐田",
    "ジュビロ磐田": "磐田",
    "藤枝": "藤枝",
    "藤枝MYFC": "藤枝",
    "藤枝ＭＹＦＣ": "藤枝",
    "栃木C": "栃木C",
    "栃木Ｃ": "栃木C",
    "栃木SC": "栃木SC",
    "栃木ＳＣ": "栃木SC",
    "栃木シティ": "栃木C",
    "栃木シティFC": "栃木C",
    "栃木シティＦＣ": "栃木C",
    "相模原": "相模原",
    "SC相模原": "相模原",
    "ＳＣ相模原": "相模原",
    "岐阜": "岐阜",
    "FC岐阜": "岐阜",
    "ＦＣ岐阜": "岐阜",
    "長野": "長野",
    "AC長野パルセイロ": "長野",
    "ＡＣ長野パルセイロ": "長野",
    "松本": "松本",
    "松本山雅FC": "松本",
    "松本山雅ＦＣ": "松本",
    "八戸": "八戸",
    "ヴァンラーレ八戸": "八戸",
    "富山": "富山",
    "カターレ富山": "富山",
    "金沢": "金沢",
    "ツエーゲン金沢": "金沢",
    "今治": "今治",
    "FC今治": "今治",
    "ＦＣ今治": "今治",
    "徳島": "徳島",
    "徳島ヴォルティス": "徳島",
    "山口": "山口",
    "レノファ山口FC": "山口",
    "レノファ山口ＦＣ": "山口",
    "鳥取": "鳥取",
    "ガイナーレ鳥取": "鳥取",
    "讃岐": "讃岐",
    "カマタマーレ讃岐": "讃岐",
    "北九州": "北九州",
    "ギラヴァンツ北九州": "北九州",
    "鹿児島": "鹿児島",
    "鹿児島ユナイテッドFC": "鹿児島",
    "鹿児島ユナイテッドＦＣ": "鹿児島",
    "FC大阪": "FC大阪",
    "ＦＣ大阪": "FC大阪",
    "熊本": "熊本",
    "ロアッソ熊本": "熊本",
    "大分": "大分",
    "大分トリニータ": "大分",
    "宮崎": "宮崎",
    "テゲバジャーロ宮崎": "宮崎",
    "福島": "福島",
    "福島ユナイテッドFC": "福島",
    "福島ユナイテッドＦＣ": "福島",
    "高知": "高知",
    "高知ユナイテッドSC": "高知",
    "高知ユナイテッドＳＣ": "高知",
    "愛媛FC": "愛媛",
    "愛媛ＦＣ": "愛媛",
    "愛媛": "愛媛",
    "FC琉球": "琉球",
    "ＦＣ琉球": "琉球",
    "琉球": "琉球",
    "札幌": "札幌",
    "コンサドーレ札幌": "札幌",
}


def _normalize_team_text(text):
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("　", " ").strip()
    s = s.replace("Ｆ", "F").replace("Ｃ", "C").replace("Ｖ", "V")
    s = s.upper()
    s = s.replace(" ", "").replace("・", "").replace(".", "")
    return s


TEAM_NAME_ALIAS_MAP = {
    _normalize_team_text(k): _normalize_team_text(v)
    for k, v in TEAM_NAME_ALIAS_RAW_MAP.items()
}

# 公式順位表は正式名と略称を同じセルに含む。既知の同一チームの
# 表記の組み合わせだけを許可し、末尾や部分一致による誤結合を避ける。
TEAM_NAME_ALIAS_MAP.update({
    _normalize_team_text(full + " " + short): _normalize_team_text(team)
    for full, team in TEAM_NAME_ALIAS_RAW_MAP.items()
    for short, short_team in TEAM_NAME_ALIAS_RAW_MAP.items()
    if team == short_team
})

# J2(2026特別大会)では未公開が続くため、予測入力から除外するフィジカル系指標
J2_EXCLUDED_STATS_BASE_NAMES = [
    "1試合平均走行距離",
    "1試合平均スプリント回数",
    "1試合平均Atスプリント回数",
    "1試合平均Mtスプリント回数",
    "1試合平均Dtスプリント回数",
    "1試合平均ポゼッション時の走行距離",
    "1試合平均ポゼッション時のスプリント回数",
]

# 勝敗判定
def get_result(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return "H"
    elif home_score < away_score:
        return "A"
    else:
        return "D"

# Eloスコアから平均得点期待値を算出
def calculate_expected_goals(
    elo_diff,
    home_xg_stats=None,
    away_xg_stats=None,
    home_travel_distance=0,
    away_travel_distance=0,
    home_fatigue_score=None,
    away_fatigue_score=None,
    home_rank_motivation_score=None,
    away_rank_motivation_score=None,
    home_absence_impact=None,
    away_absence_impact=None,
    weather_flags=None,
):
    # 確率変換と同じ elo_diff を利用して期待得点を算出する
    temp = max(1e-6, float(ELO_DIFF_TEMPERATURE))
    adjusted_elo_diff = float(elo_diff) / temp
    elo_home_expected_goals = 1.5 + adjusted_elo_diff * GOAL_SCALING_FACTOR
    elo_away_expected_goals = 1.5 - adjusted_elo_diff * GOAL_SCALING_FACTOR
    
    # デフォルト値を設定
    home_hybrid_expected_goals = elo_home_expected_goals
    away_hybrid_expected_goals = elo_away_expected_goals

    # xGスタッツが利用可能であればハイブリッド評価を適用（NaNは未指定扱い）
    if pd.notna(home_xg_stats) and pd.notna(away_xg_stats):
        home_hybrid_expected_goals = (elo_home_expected_goals * 0.7) + (home_xg_stats * 0.3)
        away_hybrid_expected_goals = (elo_away_expected_goals * 0.7) + (away_xg_stats * 0.3)
    
    # 疲労度スコアがあれば控えめに期待値を補正（NaNは未指定扱い）
    if pd.notna(home_fatigue_score):
        home_hybrid_expected_goals -= home_fatigue_score * FATIGUE_GOAL_SCALING
    if pd.notna(away_fatigue_score):
        away_hybrid_expected_goals -= away_fatigue_score * FATIGUE_GOAL_SCALING

    # 順位推移由来のモチベーションを控えめに反映（NaNは未指定扱い）
    if pd.notna(home_rank_motivation_score):
        home_hybrid_expected_goals += home_rank_motivation_score * RANK_MOTIVATION_GOAL_SCALING
    if pd.notna(away_rank_motivation_score):
        away_hybrid_expected_goals += away_rank_motivation_score * RANK_MOTIVATION_GOAL_SCALING

    # 欠場影響（チーム内重みの合算）を控えめに減点
    if pd.notna(home_absence_impact):
        home_hybrid_expected_goals -= float(home_absence_impact) * ABSENCE_IMPACT_GOAL_SCALING
    if pd.notna(away_absence_impact):
        away_hybrid_expected_goals -= float(away_absence_impact) * ABSENCE_IMPACT_GOAL_SCALING

    # 天候フラグによる控えめな補正（両チームに同率で適用）
    if weather_flags:
        penalty = 0.0
        if weather_flags.get("is_heavy_rain"):
            penalty += WEATHER_PENALTY_HEAVY_RAIN
        elif weather_flags.get("is_rain"):
            penalty += WEATHER_PENALTY_RAIN
        if weather_flags.get("is_strong_wind"):
            penalty += WEATHER_PENALTY_STRONG_WIND
        if penalty > 0:
            home_hybrid_expected_goals -= penalty
            away_hybrid_expected_goals -= penalty
    
    # 負の得点期待値にならないように調整
    home_hybrid_expected_goals = max(0.1, home_hybrid_expected_goals)
    away_hybrid_expected_goals = max(0.1, away_hybrid_expected_goals)
    
    return home_hybrid_expected_goals, away_hybrid_expected_goals


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _elo_to_home_expectation(elo_diff):
    d = max(1e-6, float(ELO_D_VALUE))
    return 1.0 / (1.0 + 10.0 ** (-float(elo_diff) / d))


def _elo_scale_for_probability():
    d = max(1e-6, float(ELO_D_VALUE))
    return (400.0 / d) * float(ELO_DIFF_SCALE)


def apply_draw_separation(prob_home_win, prob_draw, prob_away_win):
    # Poisson由来のraw draw確率を分離補正し、D過大を抑える。
    # Home/Awayの比率はPoissonの相対関係をそのまま保持する。
    pD_poisson = prob_draw
    pD = _sigmoid(D_INTERCEPT + D_SCALE * pD_poisson)

    ha_sum = prob_home_win + prob_away_win
    if ha_sum > 0:
        pH = (1.0 - pD) * (prob_home_win / ha_sum)
        pA = (1.0 - pD) * (prob_away_win / ha_sum)
    else:
        pH = (1.0 - pD) * 0.5
        pA = (1.0 - pD) * 0.5

    # 1試合分の計算例:
    # before(Poisson正規化後): pH=0.24, pD_poisson=0.53, pA=0.23
    # pD=sigmoid(-1.2 + 1.5*0.53)=0.400
    # after: pH=(1-0.400)*(0.24/0.47)=0.306, pA=(1-0.400)*(0.23/0.47)=0.294
    # sum=1.000
    return pH, pD, pA


def _resolve_draw_sensitivity(sensitivity):
    # 旧設定(<=1)は互換のため逆数変換して実効スケールを確保する。
    # 例: 0.0002 -> 5000
    s = float(sensitivity)
    if s <= 0:
        return 120.0
    if s <= 1.0:
        return 1.0 / s
    return s


def calibrate_draw_probability(prob_home_win, prob_draw, prob_away_win, elo_diff):
    # drawは「Eloで上書き」せず、Poisson由来とElo由来をブレンドする
    d_scaled = abs(float(elo_diff)) * float(ELO_DRAW_DIFF_SCALE)
    decay_scale = float(DRAW_DECAY_SCALE) if float(DRAW_DECAY_SCALE) > 0 else 120.0
    p_draw_elo = float(ELO_DRAW_MIN) + (float(ELO_DRAW_MAX) - float(ELO_DRAW_MIN)) * float(
        np.exp(-d_scaled / decay_scale)
    )
    p_draw_elo = float(np.clip(p_draw_elo, ELO_DRAW_MIN, ELO_DRAW_MAX))
    p_draw_poi = float(np.clip(float(prob_draw), 0.0, 1.0))
    blend_w = float(np.clip(DRAW_BLEND_WEIGHT, 0.0, 1.0))
    p_draw_raw = float(np.clip((blend_w * p_draw_poi) + ((1.0 - blend_w) * p_draw_elo), 0.0, 1.0))

    ha_sum = float(prob_home_win) + float(prob_away_win)
    if ha_sum > 0:
        p_home = (1.0 - p_draw_raw) * (float(prob_home_win) / ha_sum)
        p_away = (1.0 - p_draw_raw) * (float(prob_away_win) / ha_sum)
    else:
        p_home = (1.0 - p_draw_raw) * 0.5
        p_away = (1.0 - p_draw_raw) * 0.5

    p_home, p_draw_raw, p_away = _normalize_probs(p_home, p_draw_raw, p_away)
    return p_home, p_draw_raw, p_away, d_scaled, p_draw_poi, p_draw_elo


def build_elo_context(
    home_elo,
    away_elo,
    home_advantage_diff,
    stats_home_missing=False,
    stats_away_missing=False,
    data_quality_warn=False,
):
    profile_diff_raw = float(home_advantage_diff)
    profile_diff_clipped = float(np.clip(profile_diff_raw, -HOME_ADV_PROFILE_DIFF_CLIP, HOME_ADV_PROFILE_DIFF_CLIP))
    matchup_bias = 0.0
    if ENABLE_MATCHUP_BIAS:
        matchup_bias = float(profile_diff_clipped) * float(MATCHUP_BIAS_COEF)
    elo_diff_before_hfa = float(home_elo) - float(away_elo) + float(matchup_bias)
    base_hfa = float(HFA_ELO) if ENABLE_HFA else 0.0
    hfa_mult = 1.0
    applied_hfa_raw = float(base_hfa)
    applied_hfa_for_prob = float(applied_hfa_raw) * float(HFA_PROB_WEIGHT)
    elo_diff_raw = float(elo_diff_before_hfa) + float(applied_hfa_raw)
    elo_diff_for_prob_raw = float(elo_diff_before_hfa) + float(applied_hfa_for_prob)
    elo_diff = float(elo_diff_for_prob_raw) * float(_elo_scale_for_probability())
    expected_home = _elo_to_home_expectation(elo_diff)

    return {
        "hfa_enabled": bool(ENABLE_HFA),
        "matchup_bias_enabled": bool(ENABLE_MATCHUP_BIAS),
        "matchup_bias_coef": float(MATCHUP_BIAS_COEF),
        "matchup_bias": float(matchup_bias),
        "home_advantage_profile_diff_raw": profile_diff_raw,
        "home_advantage_profile_diff_clipped": profile_diff_clipped,
        "elo_diff_before_hfa": float(elo_diff_before_hfa),
        "elo_diff_after_hfa": float(elo_diff_for_prob_raw),
        "elo_diff_after_hfa_raw": float(elo_diff_raw),
        "base_hfa": float(base_hfa),
        "hfa_mult": float(hfa_mult),
        "applied_hfa": float(applied_hfa_for_prob),
        "applied_hfa_raw": float(applied_hfa_raw),
        "elo_diff_raw": float(elo_diff_raw),
        "elo_diff_for_prob_raw": float(elo_diff_for_prob_raw),
        "elo_diff_scaled": float(elo_diff),
        "elo_diff": float(elo_diff),
        "expected_home": float(expected_home),
    }


def log_prob_summary(df, label):
    required = {"prob_draw", "predicted_result"}
    if not required.issubset(df.columns):
        return
    rows = int(len(df))
    if rows == 0:
        print(f"[{label}] rows=0 ELO_DIFF_SCALE={ELO_DIFF_SCALE:.2f}")
        return
    avg_draw = float(df["prob_draw"].mean())
    sum_draw = float(df["prob_draw"].sum())
    d_count = int((df["predicted_result"].astype(str) == "D").sum())
    actual_d_rate = None
    if "actual_result" in df.columns:
        actual = df["actual_result"].astype(str).str.upper()
        valid = actual.isin(["H", "D", "A"])
        if int(valid.sum()) > 0:
            actual_d_rate = float((actual[valid] == "D").mean())
    elif {"home_score", "away_score"}.issubset(df.columns):
        hs = pd.to_numeric(df["home_score"], errors="coerce")
        aw = pd.to_numeric(df["away_score"], errors="coerce")
        valid = hs.notna() & aw.notna()
        if int(valid.sum()) > 0:
            actual_d_rate = float((hs[valid] == aw[valid]).mean())
    draw_diff_text = ""
    if actual_d_rate is not None:
        draw_diff_pp = (avg_draw - float(actual_d_rate)) * 100.0
        draw_diff_text = f" actual_D_rate={actual_d_rate:.3f} draw_diff_pp={draw_diff_pp:.2f}"
    if HDA_MODEL_MODE_EFFECTIVE == "multinom":
        print(
            f"[{label}] rows={rows} avg_prob_draw={avg_draw:.3f} "
            f"sum_prob_draw={sum_draw:.3f} predicted_D_count={d_count} "
            f"ELO_DIFF_SCALE={ELO_DIFF_SCALE:.2f}{draw_diff_text} "
            "legacy_draw_adjustment=disabled"
        )
    else:
        print(
            f"[{label}] rows={rows} avg_prob_draw={avg_draw:.3f} "
            f"sum_prob_draw={sum_draw:.3f} predicted_D_count={d_count} "
            f"ELO_DIFF_SCALE={ELO_DIFF_SCALE:.2f}{draw_diff_text} "
            f"DRAW_DECAY_SCALE={DRAW_DECAY_SCALE:.1f} "
            f"ELO_DRAW_DIFF_SCALE={ELO_DRAW_DIFF_SCALE:.3f} "
            f"ELO_DRAW_MIN={ELO_DRAW_MIN:.3f} ELO_DRAW_MAX={ELO_DRAW_MAX:.3f} "
            f"DRAW_BLEND_WEIGHT={DRAW_BLEND_WEIGHT:.3f}"
        )


def log_prob_draw_distribution(df, label):
    if "prob_draw" not in df.columns or df.empty:
        print(f"[PROB_DRAW_DIST:{label}] unavailable")
        return
    s = pd.to_numeric(df["prob_draw"], errors="coerce").dropna()
    if s.empty:
        print(f"[PROB_DRAW_DIST:{label}] unavailable")
        return
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"[PROB_DRAW_DIST:{label}] rows={len(s)} min={float(s.min()):.3f} "
        f"p05={float(q.loc[0.05]):.3f} p25={float(q.loc[0.25]):.3f} "
        f"p50={float(q.loc[0.5]):.3f} p75={float(q.loc[0.75]):.3f} "
        f"p95={float(q.loc[0.95]):.3f} max={float(s.max()):.3f}"
    )


def log_prob_distribution(df, label, col):
    if col not in df.columns or df.empty:
        print(f"[PROB_DIST:{label}] col={col} unavailable")
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        print(f"[PROB_DIST:{label}] col={col} unavailable")
        return
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"[PROB_DIST:{label}] col={col} rows={len(s)} "
        f"min={float(s.min()):.3f} p05={float(q.loc[0.05]):.3f} p25={float(q.loc[0.25]):.3f} "
        f"p50={float(q.loc[0.5]):.3f} p75={float(q.loc[0.75]):.3f} p95={float(q.loc[0.95]):.3f} "
        f"max={float(s.max()):.3f}"
    )


def log_max_prob_distribution(df, label):
    candidates = [
        ("prob_home", "prob_draw", "prob_away"),
        ("prob_home_win", "prob_draw", "prob_away_win"),
    ]
    cols = None
    for c in candidates:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None or df.empty:
        print(f"[MAX_PROB_DIST:{label}] unavailable")
        return
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    mx = pd.concat([ph, pdw, pa], axis=1).max(axis=1).dropna()
    if mx.empty:
        print(f"[MAX_PROB_DIST:{label}] unavailable")
        return
    q = mx.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(
        f"[MAX_PROB_DIST:{label}] rows={len(mx)} "
        f"min={float(mx.min()):.3f} p05={float(q.loc[0.05]):.3f} p25={float(q.loc[0.25]):.3f} "
        f"p50={float(q.loc[0.5]):.3f} p75={float(q.loc[0.75]):.3f} p95={float(q.loc[0.95]):.3f} "
        f"max={float(mx.max()):.3f}"
    )


def _feature_series_for_name(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    if name == "abs_elo_diff_for_prob" and "elo_diff_for_prob" in df.columns:
        return pd.to_numeric(df["elo_diff_for_prob"], errors="coerce").abs()
    if name == "abs_home_advantage_diff" and "home_advantage_diff" in df.columns:
        return pd.to_numeric(df["home_advantage_diff"], errors="coerce").abs()
    if name == "xg_diff" and {"stats_ゴール期待値_home", "stats_ゴール期待値_away"}.issubset(df.columns):
        return pd.to_numeric(df["stats_ゴール期待値_home"], errors="coerce") - pd.to_numeric(df["stats_ゴール期待値_away"], errors="coerce")
    if name == "xg_diff_abs" and {"stats_ゴール期待値_home", "stats_ゴール期待値_away"}.issubset(df.columns):
        base = pd.to_numeric(df["stats_ゴール期待値_home"], errors="coerce") - pd.to_numeric(df["stats_ゴール期待値_away"], errors="coerce")
        return base.abs()
    if name == "rank_gap" and {"rankmot_rank_latest_home", "rankmot_rank_latest_away"}.issubset(df.columns):
        return pd.to_numeric(df["rankmot_rank_latest_home"], errors="coerce") - pd.to_numeric(df["rankmot_rank_latest_away"], errors="coerce")
    if name == "rank_gap_abs" and {"rankmot_rank_latest_home", "rankmot_rank_latest_away"}.issubset(df.columns):
        base = pd.to_numeric(df["rankmot_rank_latest_home"], errors="coerce") - pd.to_numeric(df["rankmot_rank_latest_away"], errors="coerce")
        return base.abs()
    if name in {"d_scaled", "abs_d_scaled"} and "elo_diff_for_prob" in df.columns:
        base = pd.to_numeric(df["elo_diff_for_prob"], errors="coerce").abs()
        return base
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")


def log_multinom_feature_distribution(df, label):
    if HDA_MODEL_MODE_EFFECTIVE != "multinom" or HDA_MODEL_BUNDLE is None:
        return
    if df is None or df.empty:
        print(f"[FEATURE_DIST:{label}] unavailable")
        return
    names = list(HDA_MODEL_BUNDLE.get("feature_names", []))
    mu = np.asarray(HDA_MODEL_BUNDLE.get("feature_mean", []), dtype=float)
    sigma = np.asarray(HDA_MODEL_BUNDLE.get("feature_std", []), dtype=float)
    for i, name in enumerate(names):
        s = _feature_series_for_name(df, name)
        rows = int(len(s))
        missing = int(s.isna().sum())
        valid = s.dropna()
        if valid.empty:
            print(f"[FEATURE_DIST:{label}] col={name} rows={rows} missing={missing} unique=0 unavailable")
            continue
        q = valid.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(
            f"[FEATURE_DIST:{label}] col={name} rows={rows} missing={missing} unique={int(valid.nunique())} "
            f"min={float(valid.min()):.3f} p05={float(q.loc[0.05]):.3f} p25={float(q.loc[0.25]):.3f} "
            f"p50={float(q.loc[0.5]):.3f} p75={float(q.loc[0.75]):.3f} p95={float(q.loc[0.95]):.3f} "
            f"max={float(valid.max()):.3f}"
        )
        if i < len(mu) and i < len(sigma):
            den = 1.0 if abs(float(sigma[i])) < 1e-12 else float(sigma[i])
            s_std = (valid - float(mu[i])) / den
            q2 = s_std.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            print(
                f"[FEATURE_DIST_STD:{label}] col={name} rows={int(len(s_std))} "
                f"min={float(s_std.min()):.3f} p05={float(q2.loc[0.05]):.3f} p25={float(q2.loc[0.25]):.3f} "
                f"p50={float(q2.loc[0.5]):.3f} p75={float(q2.loc[0.75]):.3f} p95={float(q2.loc[0.95]):.3f} "
                f"max={float(s_std.max()):.3f}"
            )


def log_actual_hda_ratio(df, label):
    if df is None or df.empty:
        print(f"[ACTUAL_HDA:{label}] unavailable")
        return
    if "actual_result" in df.columns:
        actual = df["actual_result"].astype(str).str.upper()
    elif {"home_score", "away_score"}.issubset(df.columns):
        hs = pd.to_numeric(df["home_score"], errors="coerce")
        aw = pd.to_numeric(df["away_score"], errors="coerce")
        actual = pd.Series(np.where(hs > aw, "H", np.where(hs < aw, "A", "D")), index=df.index)
        actual = actual.where(hs.notna() & aw.notna(), pd.NA).astype("object")
    else:
        print(f"[ACTUAL_HDA:{label}] unavailable")
        return
    actual = actual[actual.isin(["H", "D", "A"])]
    if actual.empty:
        print(f"[ACTUAL_HDA:{label}] unavailable")
        return
    total = int(len(actual))
    h = int((actual == "H").sum())
    d = int((actual == "D").sum())
    a = int((actual == "A").sum())
    print(
        f"[ACTUAL_HDA:{label}] rows={total} H={100.0*h/total:.1f}% ({h}) "
        f"D={100.0*d/total:.1f}% ({d}) A={100.0*a/total:.1f}% ({a})"
    )


def _calc_hda_dist_from_series(series):
    s = pd.Series(series, dtype="object").astype(str).str.upper()
    s = s[s.isin(["H", "D", "A"])]
    n = int(len(s))
    h = int((s == "H").sum())
    d = int((s == "D").sum())
    a = int((s == "A").sum())
    hp = (100.0 * h / n) if n > 0 else 0.0
    dp = (100.0 * d / n) if n > 0 else 0.0
    ap = (100.0 * a / n) if n > 0 else 0.0
    return {"rows": n, "H_cnt": h, "D_cnt": d, "A_cnt": a, "H_pct": hp, "D_pct": dp, "A_pct": ap}


def _calc_prob_means(df):
    candidates = [
        ("prob_home_win", "prob_draw", "prob_away_win"),
        ("prob_home", "prob_draw", "prob_away"),
    ]
    cols = None
    for c in candidates:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None:
        return None
    ph = pd.to_numeric(df[cols[0]], errors="coerce")
    pdw = pd.to_numeric(df[cols[1]], errors="coerce")
    pa = pd.to_numeric(df[cols[2]], errors="coerce")
    valid = ph.notna() & pdw.notna() & pa.notna()
    if int(valid.sum()) <= 0:
        return None
    return {
        "rows": int(valid.sum()),
        "prob_cols": cols,
        "home_mean": float(ph[valid].mean()),
        "draw_mean": float(pdw[valid].mean()),
        "away_mean": float(pa[valid].mean()),
    }


def _add_force_draw_flag(df):
    out = df.copy()
    if "decision_reason" in out.columns:
        out["force_draw_applied"] = out["decision_reason"].astype(str).str.contains("FORCE_DRAW", na=False)
    else:
        out["force_draw_applied"] = False
    return out


def log_hda_diagnostics(df, label):
    if df is None or df.empty:
        print(f"[PRED_DIST_FINAL:{label}] unavailable")
        print(f"[PRED_DIST_ARGMAX:{label}] unavailable")
        print(f"[PROB_MEAN:{label}] unavailable")
        print(f"[FORCE_DRAW_COUNT:{label}] unavailable")
        return

    final_col = "final_result" if "final_result" in df.columns else ("predicted_result" if "predicted_result" in df.columns else None)
    if final_col:
        dist = _calc_hda_dist_from_series(df[final_col])
        print(
            f"[PRED_DIST_FINAL:{label}] rows={dist['rows']} "
            f"H={dist['H_pct']:.1f}% ({dist['H_cnt']}) D={dist['D_pct']:.1f}% ({dist['D_cnt']}) "
            f"A={dist['A_pct']:.1f}% ({dist['A_cnt']}) col={final_col}"
        )
    else:
        print(f"[PRED_DIST_FINAL:{label}] unavailable")

    if "argmax_result" in df.columns:
        dist_a = _calc_hda_dist_from_series(df["argmax_result"])
        print(
            f"[PRED_DIST_ARGMAX:{label}] rows={dist_a['rows']} "
            f"H={dist_a['H_pct']:.1f}% ({dist_a['H_cnt']}) D={dist_a['D_pct']:.1f}% ({dist_a['D_cnt']}) "
            f"A={dist_a['A_pct']:.1f}% ({dist_a['A_cnt']})"
        )
    else:
        print(f"[PRED_DIST_ARGMAX:{label}] unavailable")

    means = _calc_prob_means(df)
    if means is None:
        print(f"[PROB_MEAN:{label}] unavailable")
    else:
        print(
            f"[PROB_MEAN:{label}] rows={means['rows']} cols={means['prob_cols']} "
            f"home={means['home_mean']:.4f} draw={means['draw_mean']:.4f} away={means['away_mean']:.4f}"
        )
        if set(["prob_home_win", "prob_away_win"]).issubset(df.columns):
            ph = pd.to_numeric(df["prob_home_win"], errors="coerce")
            pa = pd.to_numeric(df["prob_away_win"], errors="coerce")
            gap = (ph - pa).dropna()
            if len(gap) > 0:
                q = gap.quantile([0.10, 0.50, 0.90])
                print(
                    f"[PH_PA_STATS:{label}] rows={len(gap)} "
                    f"min={float(gap.min()):.6f} p10={float(q.loc[0.10]):.6f} median={float(q.loc[0.50]):.6f} "
                    f"p90={float(q.loc[0.90]):.6f} max={float(gap.max()):.6f} mean={float(gap.mean()):.6f}"
                )

    force_all = int(df.get("force_draw_applied", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum())
    print(f"[FORCE_DRAW_COUNT:{label}] rows={int(len(df))} force_draw={force_all}")

    if "league" not in df.columns:
        return
    for lg, part in df.groupby("league", dropna=False, sort=True):
        lg_label = str(lg)
        if final_col:
            d = _calc_hda_dist_from_series(part[final_col])
            print(
                f"[PRED_DIST_FINAL:{label}:league={lg_label}] rows={d['rows']} "
                f"H={d['H_pct']:.1f}% ({d['H_cnt']}) D={d['D_pct']:.1f}% ({d['D_cnt']}) "
                f"A={d['A_pct']:.1f}% ({d['A_cnt']})"
            )
        if "argmax_result" in part.columns:
            da = _calc_hda_dist_from_series(part["argmax_result"])
            print(
                f"[PRED_DIST_ARGMAX:{label}:league={lg_label}] rows={da['rows']} "
                f"H={da['H_pct']:.1f}% ({da['H_cnt']}) D={da['D_pct']:.1f}% ({da['D_cnt']}) "
                f"A={da['A_pct']:.1f}% ({da['A_cnt']})"
            )
        m = _calc_prob_means(part)
        if m is not None:
            print(
                f"[PROB_MEAN:{label}:league={lg_label}] rows={m['rows']} "
                f"home={m['home_mean']:.4f} draw={m['draw_mean']:.4f} away={m['away_mean']:.4f}"
            )
        force_cnt = int(part.get("force_draw_applied", pd.Series(False, index=part.index)).fillna(False).astype(bool).sum())
        print(f"[FORCE_DRAW_COUNT:{label}:league={lg_label}] rows={int(len(part))} force_draw={force_cnt}")


def log_pred_dist(df, label, scope="all"):
    if df is None or df.empty:
        print(f"[PRED_DIST:{label}] scope={scope} unavailable")
        return
    col = "final_result" if "final_result" in df.columns else "predicted_result"
    if col not in df.columns:
        print(f"[PRED_DIST:{label}] scope={scope} unavailable")
        return
    dist = _calc_hda_dist_from_series(df[col])
    if dist["rows"] <= 0:
        print(f"[PRED_DIST:{label}] scope={scope} unavailable")
        return
    print(
        f"[PRED_DIST:{label}] scope={scope} rows={dist['rows']} "
        f"H={dist['H_pct']:.1f}% ({dist['H_cnt']}) "
        f"D={dist['D_pct']:.1f}% ({dist['D_cnt']}) "
        f"A={dist['A_pct']:.1f}% ({dist['A_cnt']})"
    )


def log_pred_dist_expect(df, label, scope="all"):
    if df is None or df.empty:
        print(f"[PRED_DIST_EXPECT:{label}] scope={scope} unavailable")
        return
    col = "final_result" if "final_result" in df.columns else None
    if col is None:
        print(f"[PRED_DIST_EXPECT:{label}] scope={scope} unavailable")
        return
    dist = _calc_hda_dist_from_series(df[col])
    if dist["rows"] <= 0:
        print(f"[PRED_DIST_EXPECT:{label}] scope={scope} unavailable")
        return
    print(
        f"[PRED_DIST_EXPECT:{label}] scope={scope} rows={dist['rows']} "
        f"H={dist['H_pct']:.1f}% ({dist['H_cnt']}) "
        f"D={dist['D_pct']:.1f}% ({dist['D_cnt']}) "
        f"A={dist['A_pct']:.1f}% ({dist['A_cnt']})"
    )


def log_pred_dist_argmax_expect_diff(df, label, scope="all"):
    if df is None or df.empty:
        return
    if "argmax_result" not in df.columns or "final_result" not in df.columns:
        return
    da = _calc_hda_dist_from_series(df["argmax_result"])
    de = _calc_hda_dist_from_series(df["final_result"])
    if da["rows"] <= 0 or de["rows"] <= 0:
        return
    print(
        f"[PRED_DIST_DIFF_ARGMAX_EXPECT:{label}] scope={scope} "
        f"H={de['H_pct']-da['H_pct']:+.1f}pp D={de['D_pct']-da['D_pct']:+.1f}pp A={de['A_pct']-da['A_pct']:+.1f}pp"
    )


def log_draw_argmax_stats(df, label, threshold=0.23):
    candidates = [
        ("prob_home", "prob_draw", "prob_away"),
        ("prob_home_win", "prob_draw", "prob_away_win"),
    ]
    cols = None
    for c in candidates:
        if set(c).issubset(df.columns):
            cols = c
            break
    if cols is None or df.empty:
        print(f"[DRAW_ARGMAX:{label}] unavailable")
        return
    ch, cd, ca = cols
    ph = pd.to_numeric(df[ch], errors="coerce")
    pdw = pd.to_numeric(df[cd], errors="coerce")
    pa = pd.to_numeric(df[ca], errors="coerce")
    valid = ph.notna() & pdw.notna() & pa.notna()
    if int(valid.sum()) == 0:
        print(f"[DRAW_ARGMAX:{label}] unavailable")
        return
    cnt_argmax = int(((pdw >= ph) & (pdw >= pa) & valid).sum())
    cnt_threshold = int(((pdw >= float(threshold)) & valid).sum())
    print(
        f"[DRAW_ARGMAX:{label}] rows={int(valid.sum())} "
        f"prob_draw_argmax_count={cnt_argmax} prob_draw_ge_{threshold:.2f}_count={cnt_threshold}"
    )


def log_elo_sign_check(df, label, diff_col="elo_diff_for_prob", home_col="prob_home_win_raw", away_col="prob_away_win_raw", tol=1e-12):
    if df is None or df.empty:
        print(f"[ELO_SIGN_CHECK:{label}] unavailable")
        return
    if diff_col not in df.columns or home_col not in df.columns or away_col not in df.columns:
        print(
            f"[ELO_SIGN_CHECK:{label}] unavailable "
            f"(required={diff_col},{home_col},{away_col})"
        )
        return
    d = pd.to_numeric(df[diff_col], errors="coerce")
    ph = pd.to_numeric(df[home_col], errors="coerce")
    pa = pd.to_numeric(df[away_col], errors="coerce")
    valid = d.notna() & ph.notna() & pa.notna()
    if int(valid.sum()) <= 0:
        print(f"[ELO_SIGN_CHECK:{label}] unavailable")
        return
    neg = valid & (d < 0)
    pos = valid & (d > 0)
    violated_neg = neg & (ph > (pa + float(tol)))
    violated_pos = pos & (ph < (pa - float(tol)))
    n_neg = int(neg.sum())
    n_pos = int(pos.sum())
    v_neg = int(violated_neg.sum())
    v_pos = int(violated_pos.sum())
    rate_neg = (100.0 * v_neg / n_neg) if n_neg > 0 else 0.0
    rate_pos = (100.0 * v_pos / n_pos) if n_pos > 0 else 0.0
    print(
        f"[ELO_SIGN_CHECK:{label}] diff_col={diff_col} prob_cols=({home_col},{away_col}) "
        f"neg_diff_cases={n_neg} violated={v_neg} rate={rate_neg:.2f}% "
        f"pos_diff_cases={n_pos} violated={v_pos} rate={rate_pos:.2f}%"
    )
    if n_neg > 0:
        d_neg = d[neg]
        gap_neg = (ph[neg] - pa[neg])
        qd = d_neg.quantile([0.10, 0.50, 0.90])
        qg = gap_neg.quantile([0.10, 0.50, 0.90])
        print(
            f"[NEG_DIFF_STATS] label={label} metric={diff_col} "
            f"min={float(d_neg.min()):.4f} p10={float(qd.loc[0.10]):.4f} median={float(qd.loc[0.50]):.4f} "
            f"p90={float(qd.loc[0.90]):.4f} max={float(d_neg.max()):.4f} mean={float(d_neg.mean()):.4f}"
        )
        print(
            f"[NEG_DIFF_STATS] label={label} metric=prob_home_minus_away "
            f"min={float(gap_neg.min()):.6f} p10={float(qg.loc[0.10]):.6f} median={float(qg.loc[0.50]):.6f} "
            f"p90={float(qg.loc[0.90]):.6f} max={float(gap_neg.max()):.6f} mean={float(gap_neg.mean()):.6f}"
        )


def dump_elo_sign_violations(df, label, top_n=50, diff_col="elo_diff_for_prob", home_col="prob_home_win_raw", away_col="prob_away_win_raw"):
    if df is None or df.empty:
        return
    if not {diff_col, home_col, away_col}.issubset(df.columns):
        return
    os.makedirs(MERGE_QC_DIR, exist_ok=True)
    work = df.copy()
    d = pd.to_numeric(work[diff_col], errors="coerce")
    ph = pd.to_numeric(work[home_col], errors="coerce")
    pa = pd.to_numeric(work[away_col], errors="coerce")
    mask = d.notna() & ph.notna() & pa.notna() & (d < 0) & (ph > pa)
    vio = work.loc[mask].copy()
    if vio.empty:
        print(f"[ELO_SIGN_VIOLATIONS:{label}] matched=0")
        return
    vio["sign_gap"] = ph[mask] - pa[mask]
    keep_cols = [c for c in [
        "league", "節", "match_id", "match_no", "datetime", "home_team", "away_team",
        "home_elo", "away_elo", "home_elo_at_prediction", "away_elo_at_prediction",
        "elo_diff_before_hfa", "hfa_added_to_diff", "elo_diff_after_hfa", diff_col,
        home_col, "prob_draw_raw", away_col, "argmax_result", "final_result", "predicted_result", "decision_reason"
    ] if c in vio.columns]
    out = vio.sort_values(["sign_gap", diff_col], ascending=[True, True]).head(int(top_n))
    out_path = os.path.join(MERGE_QC_DIR, f"elo_sign_violations_{str(label).lower()}_top{int(top_n)}.csv")
    out[keep_cols + ["sign_gap"]].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[ELO_SIGN_VIOLATIONS:{label}] matched={len(vio)} saved={out_path}")


def _preferred_diff_col_for_sign_check(df):
    if df is None:
        return "elo_diff_for_prob"
    if "diff_raw_no_hfa" in df.columns and pd.to_numeric(df["diff_raw_no_hfa"], errors="coerce").notna().any():
        return "diff_raw_no_hfa"
    return "elo_diff_for_prob"


def _parse_round_no_env(value):
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"([0-9]+)", unicodedata.normalize("NFKC", s))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _resolve_round_filter(df):
    if df is None or df.empty:
        return pd.Series([], dtype=bool), "empty_df", True

    if TOTO_ROUND_ID:
        if "toto_round_id" in df.columns:
            mask = df["toto_round_id"].astype(str).str.strip() == TOTO_ROUND_ID
            return mask, f"toto_round_id={TOTO_ROUND_ID}", False
        print(f"[WARN] TOTO_ROUND_IDが指定されていますが 'toto_round_id' 列がありません: {TOTO_ROUND_ID}")

    round_no = _parse_round_no_env(ROUND_NO_ENV)
    if round_no is not None:
        if "節" in df.columns:
            mask = df["節"].map(extract_round_number).astype("Int64") == int(round_no)
            return mask.fillna(False), f"round_no={int(round_no)} (from 節)", False
        if "round" in df.columns:
            mask = df["round"].map(extract_round_number).astype("Int64") == int(round_no)
            return mask.fillna(False), f"round_no={int(round_no)} (from round)", False
        print(f"[WARN] ROUND_NOが指定されていますが '節'/'round' 列がありません: {ROUND_NO_ENV}")

    print("[WARN] フィルタ未指定のため全件集計")
    return pd.Series([True] * len(df), index=df.index), "ALL", True


def _load_toto_targets():
    if not os.path.exists(TOTO_ORDER_CSV):
        return pd.DataFrame()
    try:
        src = pd.read_csv(TOTO_ORDER_CSV, header=None, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] toto並び順CSVの読み込み失敗: {TOTO_ORDER_CSV} ({e})")
        return pd.DataFrame()
    if src.empty:
        return pd.DataFrame()
    # 想定: col0=match_no, col1=home_team, col2='vs', col3=away_team
    if src.shape[1] < 4:
        print(f"[WARN] toto並び順CSVの列数が不足: {TOTO_ORDER_CSV}")
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "match_no": pd.to_numeric(src.iloc[:, 0], errors="coerce"),
            "home_team": src.iloc[:, 1].astype(str),
            "away_team": src.iloc[:, 3].astype(str),
        }
    )
    out = out.dropna(subset=["match_no"]).copy()
    out["match_no"] = out["match_no"].astype(int)
    out["_home_key"] = normalize_team_series(out["home_team"])
    out["_away_key"] = normalize_team_series(out["away_team"])
    out = out.dropna(subset=["_home_key", "_away_key"])
    out["_pair_key"] = out["_home_key"].astype(str) + "||" + out["_away_key"].astype(str)
    out = out.drop_duplicates(subset=["_pair_key"], keep="first")
    return out


def _resolve_toto_target_filter(df):
    if df is None or df.empty:
        return None, None
    targets = _load_toto_targets()
    if targets.empty:
        return None, None
    if not {"home_team", "away_team"}.issubset(df.columns):
        print("[WARN] toto並び順フィルタを適用できません（home_team/away_team列不足）")
        return None, None
    work = df.copy()
    work["_home_key"] = normalize_team_series(work["home_team"])
    work["_away_key"] = normalize_team_series(work["away_team"])
    work["_pair_key"] = work["_home_key"].astype(str) + "||" + work["_away_key"].astype(str)
    target_keys = set(targets["_pair_key"].astype(str).tolist())
    mask = work["_pair_key"].isin(target_keys)
    return mask, f"toto_order_csv={os.path.basename(TOTO_ORDER_CSV)}"


def calc_hda_ratio(series_of_HDA) -> dict:
    if series_of_HDA is None:
        return {
            "H": {"count": 0, "pct": 0.0},
            "D": {"count": 0, "pct": 0.0},
            "A": {"count": 0, "pct": 0.0},
            "total": 0,
        }
    s = pd.Series(series_of_HDA).astype(str).str.upper().str.strip()
    valid = s[s.isin(["H", "D", "A"])]
    total = int(len(valid))
    h = int((valid == "H").sum())
    d = int((valid == "D").sum())
    a = int((valid == "A").sum())
    denom = total if total > 0 else 1
    return {
        "H": {"count": h, "pct": (h * 100.0 / denom) if total > 0 else 0.0},
        "D": {"count": d, "pct": (d * 100.0 / denom) if total > 0 else 0.0},
        "A": {"count": a, "pct": (a * 100.0 / denom) if total > 0 else 0.0},
        "total": total,
    }


def _enrich_scores_from_results(df_pred_filtered, df_results):
    if df_results is None or df_results.empty or df_pred_filtered.empty:
        return df_pred_filtered

    out = df_pred_filtered.copy()
    res = df_results.copy()
    for col in ["home_score", "away_score"]:
        if col not in out.columns:
            out[col] = pd.NA

    # 1) match_id があれば最優先で突合
    if "match_id" in out.columns and "match_id" in res.columns:
        right = res[["match_id", "home_score", "away_score"]].copy()
        right = right.dropna(subset=["match_id"]).drop_duplicates(subset=["match_id"], keep="last")
        merged = out.merge(right, on="match_id", how="left", suffixes=("", "__res"))
        for col in ["home_score", "away_score"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(
                pd.to_numeric(merged[f"{col}__res"], errors="coerce")
            )
        return merged.drop(columns=["home_score__res", "away_score__res"], errors="ignore")

    # 2) datetime + home_team + away_team
    key_cols = {"datetime", "home_team", "away_team"}
    if key_cols.issubset(set(out.columns)) and key_cols.issubset(set(res.columns)):
        left = out.copy()
        right = res.copy()
        left["_dt_key"] = pd.to_datetime(left["datetime"], errors="coerce")
        right["_dt_key"] = pd.to_datetime(right["datetime"], errors="coerce")
        left["_home_key"] = normalize_team_series(left["home_team"]) if "normalize_team_series" in globals() else left["home_team"].astype(str)
        left["_away_key"] = normalize_team_series(left["away_team"]) if "normalize_team_series" in globals() else left["away_team"].astype(str)
        right["_home_key"] = normalize_team_series(right["home_team"]) if "normalize_team_series" in globals() else right["home_team"].astype(str)
        right["_away_key"] = normalize_team_series(right["away_team"]) if "normalize_team_series" in globals() else right["away_team"].astype(str)
        right = right[["_dt_key", "_home_key", "_away_key", "home_score", "away_score"]].drop_duplicates(
            subset=["_dt_key", "_home_key", "_away_key"], keep="last"
        )
        merged = left.merge(right, on=["_dt_key", "_home_key", "_away_key"], how="left", suffixes=("", "__res"))
        for col in ["home_score", "away_score"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(
                pd.to_numeric(merged[f"{col}__res"], errors="coerce")
            )
        return merged.drop(
            columns=["home_score__res", "away_score__res", "_dt_key", "_home_key", "_away_key"],
            errors="ignore",
        )

    return out


def _write_round_summary_csv(filter_label, pred_ratio, actual_ratio):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(filter_label)).strip("._")
    if not safe:
        safe = "all"
    out_path = os.path.join(MERGE_QC_DIR, f"round_summary_{safe}.csv")
    os.makedirs(MERGE_QC_DIR, exist_ok=True)
    rows = [
        {
            "kind": "pred",
            "H_cnt": pred_ratio["H"]["count"],
            "D_cnt": pred_ratio["D"]["count"],
            "A_cnt": pred_ratio["A"]["count"],
            "total": pred_ratio["total"],
            "H_pct": pred_ratio["H"]["pct"],
            "D_pct": pred_ratio["D"]["pct"],
            "A_pct": pred_ratio["A"]["pct"],
            "filter": filter_label,
        }
    ]
    if actual_ratio is not None:
        rows.append(
            {
                "kind": "actual",
                "H_cnt": actual_ratio["H"]["count"],
                "D_cnt": actual_ratio["D"]["count"],
                "A_cnt": actual_ratio["A"]["count"],
                "total": actual_ratio["total"],
                "H_pct": actual_ratio["H"]["pct"],
                "D_pct": actual_ratio["D"]["pct"],
                "A_pct": actual_ratio["A"]["pct"],
                "filter": filter_label,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[ROUND_SUMMARY_CSV] saved={out_path}")


def summarize_round_hda(df_pred, df_results=None, round_filter_label="auto"):
    if df_pred is None or df_pred.empty:
        print("[ROUND_SUMMARY] filter=empty rows=0")
        print("[PRED_RATIO] H=0.0% (0) D=0.0% (0) A=0.0% (0)")
        print("[ACTUAL_RATIO] unavailable (scores not found or not finished)")
        return

    mask, filter_label = _resolve_toto_target_filter(df_pred)
    if mask is None:
        mask, filter_label, _ = _resolve_round_filter(df_pred)
    pred_filtered = df_pred.loc[mask].copy()
    rows = int(len(pred_filtered))
    if filter_label.startswith("toto_order_csv=") and rows != 13:
        print(f"[WARN] toto対象の抽出件数が13ではありません: rows={rows} (LEAGUE={LEAGUE})")
    print(f"[ROUND_SUMMARY] filter={filter_label} rows={rows}")

    pred_ratio = calc_hda_ratio(pred_filtered.get("predicted_result", pd.Series(dtype="object")))
    print(
        f"[PRED_RATIO] H={pred_ratio['H']['pct']:.1f}% ({pred_ratio['H']['count']}) "
        f"D={pred_ratio['D']['pct']:.1f}% ({pred_ratio['D']['count']}) "
        f"A={pred_ratio['A']['pct']:.1f}% ({pred_ratio['A']['count']})"
    )

    if rows == 0:
        print("[ACTUAL_RATIO] unavailable (scores not found or not finished)")
        _write_round_summary_csv(filter_label, pred_ratio, None)
        return

    actual_source = _enrich_scores_from_results(pred_filtered, df_results)
    hs = pd.to_numeric(actual_source.get("home_score"), errors="coerce")
    aw = pd.to_numeric(actual_source.get("away_score"), errors="coerce")
    actual_result = pd.Series(
        [get_result(h, a) for h, a in zip(hs.tolist(), aw.tolist())],
        index=actual_source.index,
        dtype="object",
    )
    resolved_count = int(actual_result.notna().sum())
    if resolved_count != rows:
        print("[ACTUAL_RATIO] unavailable (scores not found or not finished)")
        _write_round_summary_csv(filter_label, pred_ratio, None)
        return

    actual_ratio = calc_hda_ratio(actual_result)
    print(
        f"[ACTUAL_RATIO] H={actual_ratio['H']['pct']:.1f}% ({actual_ratio['H']['count']}) "
        f"D={actual_ratio['D']['pct']:.1f}% ({actual_ratio['D']['count']}) "
        f"A={actual_ratio['A']['pct']:.1f}% ({actual_ratio['A']['count']})"
    )
    _write_round_summary_csv(filter_label, pred_ratio, actual_ratio)


def log_decision_rule_once():
    if getattr(log_decision_rule_once, "_done", False):
        return
    if not DRAW_TWEAK_ENABLED:
        rule_desc = "ARGMAX (DRAW_TWEAK_MODE=off; probability tweaks disabled)"
    elif DRAW_ASSIGN_BY_EXPECTATION:
        rule_desc = "FORCE_DRAW_BY_EXPECTATION_ASSIGN > ARGMAX"
    else:
        rule_desc = "ARGMAX"
    print(f"[DECISION_RULE] {rule_desc}")
    log_decision_rule_once._done = True


def dump_decision_artifacts(df, label="pred", threshold=0.25):
    if not DUMP_DECISION:
        return
    required = {"prob_home_win", "prob_draw", "prob_away_win", "predicted_result"}
    if not required.issubset(df.columns):
        print(f"[DUMP_DECISION][WARN] required_columns_missing label={label} need={sorted(required)}")
        return

    os.makedirs(MERGE_QC_DIR, exist_ok=True)
    work = df.copy()
    work["prob_home_win"] = pd.to_numeric(work["prob_home_win"], errors="coerce")
    work["prob_draw"] = pd.to_numeric(work["prob_draw"], errors="coerce")
    work["prob_away_win"] = pd.to_numeric(work["prob_away_win"], errors="coerce")
    work["draw_gap"] = work[["prob_home_win", "prob_away_win"]].max(axis=1) - work["prob_draw"]

    keep_cols = [c for c in ["match_id", "match_no", "league", "節", "home_team", "away_team"] if c in work.columns]
    keep_cols += ["prob_home_win", "prob_draw", "prob_away_win", "predicted_result", "draw_gap"]
    for c in [
        "final_result",
        "decision_reason",
        "force_draw_applied",
        "argmax_result",
        "argmax_raw_result",
        "argmax_max_prob",
        "argmax_raw_max_prob",
        "elo_diff_before_hfa",
        "hfa_added_to_diff",
        "elo_diff_after_hfa",
        "elo_diff_scaled",
        "d_scaled",
        "elo_diff_for_prob",
        "decision_draw_expectation_multiplier",
        "decision_draw_assign_enabled",
    ]:
        if c in work.columns:
            keep_cols.append(c)

    full_csv = os.path.join(MERGE_QC_DIR, f"decision_scores_{label}.csv")
    work[keep_cols].to_csv(full_csv, index=False, encoding="utf-8-sig")

    cond = (work["prob_draw"] >= float(threshold)) & (work["predicted_result"].astype(str).str.upper() != "D")
    top50 = work.loc[cond, keep_cols].sort_values(["draw_gap", "prob_draw"], ascending=[True, False]).head(50)
    top50_csv = os.path.join(MERGE_QC_DIR, f"decision_draw_candidates_{label}_top50.csv")
    top50.to_csv(top50_csv, index=False, encoding="utf-8-sig")

    sign_home_col = "prob_home_win_before_signfix" if "prob_home_win_before_signfix" in work.columns else "prob_home_win_raw"
    sign_away_col = "prob_away_win_before_signfix" if "prob_away_win_before_signfix" in work.columns else "prob_away_win_raw"
    sign_cols = [c for c in [
        "match_id", "match_no", "league", "節", "home_team", "away_team",
        "elo_diff_for_prob", sign_home_col, "prob_draw", sign_away_col,
        "prob_home_win_raw", "prob_away_win_raw", "elo_sign_fix_applied", "elo_sign_fix_reason",
        "argmax_result", "final_result", "predicted_result", "decision_reason",
    ] if c in work.columns]
    if {"elo_diff_for_prob", sign_home_col, sign_away_col}.issubset(work.columns):
        d = pd.to_numeric(work["elo_diff_for_prob"], errors="coerce")
        phs = pd.to_numeric(work[sign_home_col], errors="coerce")
        pas = pd.to_numeric(work[sign_away_col], errors="coerce")
        vio = work[(d < 0) & phs.notna() & pas.notna() & (phs > pas)].copy()
        vio = vio.assign(sign_gap=(phs - pas)).sort_values(["sign_gap"], ascending=[True]).head(50)
        vio_csv = os.path.join(MERGE_QC_DIR, f"decision_elo_sign_violations_{label}_top50.csv")
        vio[sign_cols + (["sign_gap"] if "sign_gap" in vio.columns else [])].to_csv(vio_csv, index=False, encoding="utf-8-sig")
        print(
            f"[ELO_SIGN_DUMP] label={label} "
            f"cond=(elo_diff_for_prob<0 and {sign_home_col}>{sign_away_col}) matched={len(vio)} saved={vio_csv}"
        )

    print(
        f"[DUMP_DECISION] label={label} total={len(work)} "
        f"cond=(prob_draw>={threshold} and predicted_result!=D) matched={int(cond.sum())} "
        f"saved_full={full_csv} saved_top50={top50_csv}"
    )


def compute_row_quality_flags(row):
    stats_home_missing = pd.isna(row.get("stats_ゴール期待値_home"))
    stats_away_missing = pd.isna(row.get("stats_ゴール期待値_away"))
    mgmt_home_col = _pick_first_non_na_value(
        row,
        [
            "management_recent_injuries_suspensions_count_home",
            "management_recent_injuries_suspensions_count",
        ],
    )
    mgmt_away_col = _pick_first_non_na_value(row, ["management_recent_injuries_suspensions_count_away"])
    management_missing = pd.isna(mgmt_home_col) or pd.isna(mgmt_away_col)
    weather_missing = bool(row.get("weather_missing")) if pd.notna(row.get("weather_missing")) else False
    data_quality_warn = bool(weather_missing or stats_home_missing or stats_away_missing or management_missing)
    return {
        "weather_missing": bool(weather_missing),
        "stats_home_missing": bool(stats_home_missing),
        "stats_away_missing": bool(stats_away_missing),
        "management_missing": bool(management_missing),
        "data_quality_warn": bool(data_quality_warn),
    }


def _bool_like(v):
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _safe_float_value(v, default=0.0):
    n = pd.to_numeric(v, errors="coerce")
    if pd.isna(n):
        return float(default)
    return float(n)


def _resolve_absence_scores_for_multinom(row, absence_effective):
    # 優先: home_absence_score / away_absence_score（存在時）
    # fallback: effective_total（既存計算済み）
    def _pick_score(obj, key):
        if obj is None:
            return None
        try:
            val = pd.to_numeric(obj.get(key), errors="coerce")
        except Exception:
            val = np.nan
        if pd.isna(val):
            return None
        return float(val)

    home_score = _pick_score(row, "home_absence_score")
    away_score = _pick_score(row, "away_absence_score")
    if home_score is None:
        home_score = _pick_score(absence_effective, "absence_effective_total_home")
    if away_score is None:
        away_score = _pick_score(absence_effective, "absence_effective_total_away")
    if home_score is None:
        home_score = 0.0
    if away_score is None:
        away_score = 0.0
    return float(home_score), float(away_score)


def compute_effective_absence_impacts(row):
    # raw値（欠損は0扱い）
    raw_total_home = _safe_float_value(row.get("absence_impact_total_home"), 0.0)
    raw_attack_home = _safe_float_value(row.get("absence_impact_attack_home"), 0.0)
    raw_defense_home = _safe_float_value(row.get("absence_impact_defense_home"), 0.0)
    raw_players_home = _safe_float_value(row.get("absence_players_count_home"), 0.0)

    raw_total_away = _safe_float_value(row.get("absence_impact_total_away"), 0.0)
    raw_attack_away = _safe_float_value(row.get("absence_impact_attack_away"), 0.0)
    raw_defense_away = _safe_float_value(row.get("absence_impact_defense_away"), 0.0)
    raw_players_away = _safe_float_value(row.get("absence_players_count_away"), 0.0)

    # 欠損判定（指定要件: count=0 かつ total=0、かつ既存欠損フラグをORで加味）
    missing_home = (raw_players_home <= 0.0 and raw_total_home <= 0.0)
    missing_away = (raw_players_away <= 0.0 and raw_total_away <= 0.0)

    if "absence_missing_home" in row.index:
        missing_home = bool(missing_home or _bool_like(row.get("absence_missing_home")))
    if "absence_missing_away" in row.index:
        missing_away = bool(missing_away or _bool_like(row.get("absence_missing_away")))

    # 既存フラグ（例: management_missing / data_quality_warn）を補助的にOR
    for flag_col in ["management_missing", "data_quality_warn"]:
        if flag_col in row.index:
            flag_val = _bool_like(row.get(flag_col))
            missing_home = bool(missing_home or flag_val)
            missing_away = bool(missing_away or flag_val)

    # 欠損ならベースライン、そうでなければrawを使用
    eff_total_home = ABSENCE_BASELINE_TOTAL if missing_home else raw_total_home
    eff_attack_home = ABSENCE_BASELINE_ATTACK if missing_home else raw_attack_home
    eff_defense_home = ABSENCE_BASELINE_DEFENSE if missing_home else raw_defense_home

    eff_total_away = ABSENCE_BASELINE_TOTAL if missing_away else raw_total_away
    eff_attack_away = ABSENCE_BASELINE_ATTACK if missing_away else raw_attack_away
    eff_defense_away = ABSENCE_BASELINE_DEFENSE if missing_away else raw_defense_away

    # 過補正防止のcap（totalのみ）
    cap_applied_home = eff_total_home > ABSENCE_IMPACT_CAP_TOTAL
    cap_applied_away = eff_total_away > ABSENCE_IMPACT_CAP_TOTAL
    eff_total_home = min(eff_total_home, ABSENCE_IMPACT_CAP_TOTAL)
    eff_total_away = min(eff_total_away, ABSENCE_IMPACT_CAP_TOTAL)

    # 念のため下限
    eff_total_home = max(0.0, eff_total_home)
    eff_total_away = max(0.0, eff_total_away)
    eff_attack_home = max(0.0, eff_attack_home)
    eff_attack_away = max(0.0, eff_attack_away)
    eff_defense_home = max(0.0, eff_defense_home)
    eff_defense_away = max(0.0, eff_defense_away)

    return {
        "absence_missing_home": bool(missing_home),
        "absence_missing_away": bool(missing_away),
        "absence_effective_total_home": float(eff_total_home),
        "absence_effective_attack_home": float(eff_attack_home),
        "absence_effective_defense_home": float(eff_defense_home),
        "absence_effective_total_away": float(eff_total_away),
        "absence_effective_attack_away": float(eff_attack_away),
        "absence_effective_defense_away": float(eff_defense_away),
        "absence_cap_applied_home": bool(cap_applied_home),
        "absence_cap_applied_away": bool(cap_applied_away),
    }


def log_absence_effective_summary(df, label):
    required = {
        "absence_missing_home",
        "absence_missing_away",
        "absence_effective_total_home",
        "absence_effective_total_away",
    }
    if not required.issubset(set(df.columns)):
        return
    rows = int(len(df))
    if rows == 0:
        return
    miss_home = int(pd.Series(df["absence_missing_home"]).fillna(False).astype(bool).sum())
    miss_away = int(pd.Series(df["absence_missing_away"]).fillna(False).astype(bool).sum())
    miss_any = int(
        (pd.Series(df["absence_missing_home"]).fillna(False).astype(bool) |
         pd.Series(df["absence_missing_away"]).fillna(False).astype(bool)).sum()
    )
    eff_home = pd.to_numeric(df["absence_effective_total_home"], errors="coerce").fillna(0.0)
    eff_away = pd.to_numeric(df["absence_effective_total_away"], errors="coerce").fillna(0.0)
    cap_home = int(pd.Series(df.get("absence_cap_applied_home", False)).fillna(False).astype(bool).sum())
    cap_away = int(pd.Series(df.get("absence_cap_applied_away", False)).fillna(False).astype(bool).sum())
    print(
        f"[ABSENCE_EFFECTIVE][{label}] rows={rows} "
        f"missing_home_count={miss_home} missing_away_count={miss_away} missing_any_row_count={miss_any} "
        f"effective_total_home_avg={eff_home.mean():.4f} effective_total_home_max={eff_home.max():.4f} "
        f"effective_total_away_avg={eff_away.mean():.4f} effective_total_away_max={eff_away.max():.4f} "
        f"cap_applied_home_count={cap_home} cap_applied_away_count={cap_away} "
        f"baseline_total={ABSENCE_BASELINE_TOTAL:.4f} cap_total={ABSENCE_IMPACT_CAP_TOTAL:.4f}"
    )

# ポアソン分布を用いて勝敗確率を計算
def predict_poisson_probabilities(
    elo_diff,
    home_xg_stats=None,
    away_xg_stats=None,
    home_travel_distance=0,
    away_travel_distance=0,
    home_fatigue_score=None,
    away_fatigue_score=None,
    home_rank_motivation_score=None,
    away_rank_motivation_score=None,
    home_absence_impact=None,
    away_absence_impact=None,
    weather_flags=None,
    max_goals=10,
):
    home_expected_goals, away_expected_goals = calculate_expected_goals(
        elo_diff,
        home_xg_stats,
        away_xg_stats,
        home_travel_distance,
        away_travel_distance,
        home_fatigue_score,
        away_fatigue_score,
        home_rank_motivation_score,
        away_rank_motivation_score,
        home_absence_impact,
        away_absence_impact,
        weather_flags,
    )

    # 打ち切りによる歪みを減らすため、十分な格子上限Kを動的に決める
    k = max(int(max_goals), int(POISSON_GRID_MIN_K))
    max_k_cap = max(k, int(POISSON_GRID_MAX_K))
    tail_eps = max(1e-12, float(POISSON_TAIL_EPS))
    while k < max_k_cap:
        in_grid_mass = float(poisson.cdf(k, home_expected_goals)) * float(poisson.cdf(k, away_expected_goals))
        tail_mass = 1.0 - in_grid_mass
        if tail_mass <= tail_eps:
            break
        k += 1

    prob_home_win = 0.0
    prob_draw = 0.0
    prob_away_win = 0.0

    for i in range(k + 1):  # ホームチームの得点
        for j in range(k + 1):  # アウェイチームの得点
            prob = poisson.pmf(i, home_expected_goals) * poisson.pmf(j, away_expected_goals)
            if i > j:
                prob_home_win += prob
            elif i == j:
                prob_draw += prob
            else:
                prob_away_win += prob
    
    # 確率の合計が1になるように正規化（max_goalsを小さくした場合に必要）
    total_prob = prob_home_win + prob_draw + prob_away_win
    if total_prob > 0:
        prob_home_win /= total_prob
        prob_draw /= total_prob
        prob_away_win /= total_prob

    return prob_home_win, prob_draw, prob_away_win


def compute_probabilities_and_result(
    match_id,
    home_elo,
    away_elo,
    home_advantage_diff,
    home_xg_stats=None,
    away_xg_stats=None,
    home_travel_distance=0,
    away_travel_distance=0,
    home_fatigue_score=None,
    away_fatigue_score=None,
    home_rank_motivation_score=None,
    away_rank_motivation_score=None,
    home_absence_impact=None,
    away_absence_impact=None,
    home_absence_score=None,
    away_absence_score=None,
    weather_flags=None,
    stats_home_missing=False,
    stats_away_missing=False,
    data_quality_warn=False,
    row=None,
    absence_effective=None,
):
    """予想/バックテスト共通: 確率算出→丸め→結果判定を一元化する。"""
    elo_ctx = build_elo_context(
        home_elo=home_elo,
        away_elo=away_elo,
        home_advantage_diff=home_advantage_diff,
        stats_home_missing=stats_home_missing,
        stats_away_missing=stats_away_missing,
        data_quality_warn=data_quality_warn,
    )

    # HFAの単一適用点: 旧ロジック向け差分（elo_diff_for_prob）と、calibrated向け生差分をここで確定
    elo_diff_before_hfa = float(elo_ctx["elo_diff_before_hfa"])
    applied_hfa = float(elo_ctx["applied_hfa"]) if ENABLE_HFA else 0.0
    elo_diff_after_hfa = float(elo_ctx["elo_diff_after_hfa"])
    elo_diff_for_prob = float(elo_ctx["elo_diff_scaled"])
    diff_raw_no_hfa = float(home_elo) - float(away_elo)
    home_abs_score = _safe_float_value(home_absence_score, _safe_float_value(home_absence_impact, 0.0))
    away_abs_score = _safe_float_value(away_absence_score, _safe_float_value(away_absence_impact, 0.0))
    absence_diff = float(away_abs_score) - float(home_abs_score)
    absence_adjust = float(ABSENCE_ELO_COEF) * float(absence_diff)
    if float(ABSENCE_ELO_ADJUST_CLIP) > 0.0:
        absence_adjust = float(np.clip(absence_adjust, -float(ABSENCE_ELO_ADJUST_CLIP), float(ABSENCE_ELO_ADJUST_CLIP)))
    effective_diff_for_multinom = float(elo_diff_for_prob) + float(absence_adjust)

    if not ENABLE_HFA:
        _update_hfa_apply_counter("ENABLE_HFA=0", applied=False)
    elif float(HFA_ELO) <= 0.0:
        _update_hfa_apply_counter("HFA_ELO<=0", applied=False)
    elif abs(applied_hfa) > 1e-12:
        _update_hfa_apply_counter("applied", applied=True)
    else:
        _update_hfa_apply_counter("applied_hfa_zero_other", applied=False)

    if HDA_MODEL_MODE_EFFECTIVE == "multinom" and HDA_MODEL_BUNDLE is not None:
        model_type = str(HDA_MODEL_BUNDLE.get("type", ""))
        # multinom主経路のみ、Elo差に弱い欠場補正を加える（legacy/poissonは不変）
        model_input_diff = effective_diff_for_multinom
        feat_overrides = _extend_multinom_feat_values(
            {},
            row=row if row is not None else pd.Series(dtype="object"),
            home_advantage_diff=home_advantage_diff,
            absence_effective=absence_effective if absence_effective is not None else {},
            home_fatigue_score=home_fatigue_score,
            away_fatigue_score=away_fatigue_score,
        )
        (prob_home_win, prob_draw, prob_away_win), model_feats = _predict_hda_multinom_probs(
            model_input_diff,
            feat_overrides=feat_overrides,
        )
        draw_model_input = float(model_feats.get("d_scaled", abs(elo_diff_for_prob)))
        draw_poi = float("nan")
        draw_elo = float("nan")
    else:
        prob_home_win, prob_draw, prob_away_win = predict_poisson_probabilities(
            elo_diff_for_prob,
            home_xg_stats,
            away_xg_stats,
            home_travel_distance,
            away_travel_distance,
            home_fatigue_score,
            away_fatigue_score,
            home_rank_motivation_score,
            away_rank_motivation_score,
            home_absence_impact,
            away_absence_impact,
            weather_flags,
        )
        if DRAW_TWEAK_ENABLED:
            prob_home_win, prob_draw, prob_away_win, draw_model_input, draw_poi, draw_elo = calibrate_draw_probability(
                prob_home_win,
                prob_draw,
                prob_away_win,
                elo_diff_for_prob,
            )
        else:
            prob_home_win, prob_draw, prob_away_win = _normalize_probs(prob_home_win, prob_draw, prob_away_win)
            draw_model_input = float(abs(elo_diff_for_prob))
            draw_poi = float(prob_draw)
            draw_elo = float("nan")
    prob_home_win_before_signfix = float(prob_home_win)
    prob_away_win_before_signfix = float(prob_away_win)
    prob_home_win, prob_draw, prob_away_win, sign_fix_reason = enforce_elo_sign_monotonic(
        prob_home_win,
        prob_draw,
        prob_away_win,
        elo_diff_for_prob,
    )
    sum_before_round = prob_home_win + prob_draw + prob_away_win
    if not np.isclose(sum_before_round, 1.0, atol=1e-6):
        print(
            f"[PROB_QC][WARN] match_id={match_id} prob_sum={sum_before_round:.9f} "
            f"(home={prob_home_win:.6f}, draw={prob_draw:.6f}, away={prob_away_win:.6f})"
        )

    if elo_diff_for_prob > 0 and prob_home_win < prob_away_win:
        print(
            f"[PROB_QC][WARN] match_id={match_id} elo_diff_for_prob={elo_diff_for_prob:.4f} "
            f"なのに prob_home({prob_home_win:.4f}) < prob_away({prob_away_win:.4f})"
        )

    predicted_result, decision_reason, decision_metrics = decide_result(
        prob_home_win, prob_draw, prob_away_win
    )

    expected_home_for_prob = _elo_to_home_expectation(elo_diff_for_prob)

    debug_row = {
        "match_id": match_id,
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "home_advantage_diff_input": float(home_advantage_diff),
        "hfa_enabled": elo_ctx["hfa_enabled"],
        "matchup_bias_enabled": elo_ctx["matchup_bias_enabled"],
        "matchup_bias_coef": elo_ctx["matchup_bias_coef"],
        "matchup_bias": elo_ctx["matchup_bias"],
        "home_advantage_profile_diff_raw": elo_ctx["home_advantage_profile_diff_raw"],
        "home_advantage_profile_diff_clipped": elo_ctx["home_advantage_profile_diff_clipped"],
        "HFA_base": elo_ctx["base_hfa"],
        "HFA_multiplier": elo_ctx["hfa_mult"],
        "hfa_clip_min": float("nan"),
        "hfa_clip_max": float("nan"),
        "elo_diff_scale_factor": float(ELO_DIFF_SCALE),
        "hfa_added_to_diff": applied_hfa,
        "hfa_added_to_diff_raw": float(elo_ctx.get("applied_hfa_raw", applied_hfa)),
        "HFA_applied": applied_hfa,
        "elo_diff_before_hfa": elo_diff_before_hfa,
        "elo_diff_after_hfa": elo_diff_after_hfa,
        "diff_raw_no_hfa": diff_raw_no_hfa,
        "elo_diff_raw": elo_diff_after_hfa,
        "elo_diff_scaled": elo_diff_for_prob,
        "elo_diff_for_prob": elo_diff_for_prob,
        "elo_diff": elo_diff_for_prob,
        "expected_home": expected_home_for_prob,
        "draw_model_input": draw_model_input,
        "draw_model_output": prob_draw,
        "draw_model_poi": draw_poi,
        "draw_model_elo": draw_elo,
        "absence_home_score": float(home_abs_score),
        "absence_away_score": float(away_abs_score),
        "absence_diff_for_prob": float(absence_diff),
        "absence_adjust_for_prob": float(absence_adjust),
        "elo_diff_effective_for_multinom": float(effective_diff_for_multinom),
        "hda_model_mode_effective": HDA_MODEL_MODE_EFFECTIVE,
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "prob_home_win_before_signfix": prob_home_win_before_signfix,
        "prob_away_win_before_signfix": prob_away_win_before_signfix,
        "elo_sign_fix_applied": bool(sign_fix_reason is not None),
        "elo_sign_fix_reason": sign_fix_reason or "",
        "predicted_result": predicted_result,
        "decision_reason": decision_reason,
        "argmax_result": decision_metrics.get("argmax_result"),
        "argmax_max_prob": decision_metrics.get("argmax_max_prob"),
    }

    if DEBUG_ELO_PROB or (DEBUG_MATCH_ID and str(match_id) == DEBUG_MATCH_ID):
        print(
            "[ELO_DEBUG] "
            f"match_id={debug_row['match_id']} "
            f"home_elo={debug_row['home_elo']:.2f} away_elo={debug_row['away_elo']:.2f} "
            f"HFA={debug_row['HFA_applied']:.4f} "
            f"elo_diff_raw={debug_row['elo_diff_raw']:.4f} elo_diff={debug_row['elo_diff']:.4f} "
            f"expected_home={debug_row['expected_home']:.4f} "
            f"draw(poi/elo/blend)=({debug_row['draw_model_poi']:.3f}/{debug_row['draw_model_elo']:.3f}/{debug_row['draw_model_output']:.3f}) "
            f"probs=({prob_home_win:.3f},{prob_draw:.3f},{prob_away_win:.3f}) "
            f"result={predicted_result}"
        )

    return prob_home_win, prob_draw, prob_away_win, predicted_result, debug_row


def decide_result(
    prob_home_win,
    prob_draw,
    prob_away_win,
    force_draw=False,
    force_reason=None,
):
    if pd.isna(prob_home_win) or pd.isna(prob_draw) or pd.isna(prob_away_win):
        return None, "UNDECIDED_NAN", {"argmax_result": None, "argmax_max_prob": None}
    ph = float(prob_home_win)
    pdw = float(prob_draw)
    pa = float(prob_away_win)
    argmax_max = max(ph, pdw, pa)
    # Base rule: calibrated H/D/A probabilities の argmax
    if ph >= pdw and ph >= pa:
        argmax_result = "H"
    elif pa >= ph and pa >= pdw:
        argmax_result = "A"
    else:
        argmax_result = "D"

    if force_draw:
        reason = force_reason or "FORCE_DRAW_BY_RULE"
        return "D", reason, {"argmax_result": argmax_result, "argmax_max_prob": float(argmax_max)}
    return argmax_result, "ARGMAX", {"argmax_result": argmax_result, "argmax_max_prob": float(argmax_max)}


def decide_predicted_result(
    prob_home_win,
    prob_draw,
    prob_away_win,
):
    # 互換ラッパー: 決定ロジックは decide_result() に集約
    decided, _, _ = decide_result(prob_home_win, prob_draw, prob_away_win)
    return decided


def _build_provisional_toto13_group(df, verbose=True):
    if df is None or df.empty:
        return None, "none"
    work = df.copy()

    # 最優先: toto並び順13カードと完全突合できる場合は1開催回として扱う
    if {"home_team", "away_team"}.issubset(work.columns):
        targets = _load_toto_targets()
        if not targets.empty and len(work) == 13:
            wk = work.copy()
            wk["_home_key"] = normalize_team_series(wk["home_team"])
            wk["_away_key"] = normalize_team_series(wk["away_team"])
            wk["_pair_key"] = wk["_home_key"].astype(str) + "||" + wk["_away_key"].astype(str)
            target_keys = set(targets["_pair_key"].astype(str).tolist())
            matched = int(wk["_pair_key"].isin(target_keys).sum())
            if matched == 13:
                if verbose:
                    print(
                        f"[CONTEST_GROUP] source=toto_order_csv_exact13 rows=13 groups=1 "
                        f"non13_groups=0 file={os.path.basename(TOTO_ORDER_CSV)}"
                    )
                return pd.Series(["toto_order_csv_exact13"] * len(work), index=work.index), "toto_order_csv_exact13"

    # フォールバック: 時系列順に13件ずつ束ねる
    sort_cols = []
    if "datetime" in work.columns:
        work["__dt_sort"] = pd.to_datetime(work["datetime"], errors="coerce")
        sort_cols.append("__dt_sort")
    if "league" in work.columns:
        sort_cols.append("league")
    if "節" in work.columns:
        work["__round_sort"] = work["節"].map(extract_round_number)
        sort_cols.append("__round_sort")
    for c in ["home_team", "away_team", "match_id"]:
        if c in work.columns:
            sort_cols.append(c)
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort").copy()

    work["__tmp_idx"] = np.arange(len(work))
    work["__provisional_gid"] = (work["__tmp_idx"] // 13).astype(int) + 1
    series = work["__provisional_gid"].map(lambda x: f"prov13_{int(x):05d}")
    series.index = work.index
    series = series.reindex(df.index)

    gsize = series.value_counts(dropna=False)
    non13 = int((gsize != 13).sum())
    if verbose:
        print(
            f"[CONTEST_GROUP] source=provisional_chunk13 rows={len(df)} groups={len(gsize)} "
            f"size_min={int(gsize.min()) if len(gsize)>0 else 0} size_max={int(gsize.max()) if len(gsize)>0 else 0} "
            f"non13_groups={non13}"
        )
        if non13 > 0:
            print(f"[WARN] provisional contest grouping has non-13 groups: {gsize[gsize != 13].to_dict()}")
    return series, "provisional_chunk13"


def _resolve_draw_assign_group(df, verbose=True):
    if df is None or df.empty:
        return None, "none"
    if DRAW_ASSIGN_GROUP_MODE == "round":
        ordered = ["round", "節"]
    else:
        ordered = [
            "contest_id",
            "toto_contest_id",
            "toto_round_id",
            "開催回",
            "holding_round",
            "round_id",
        ]
    for col in ordered:
        if col in df.columns and df[col].notna().any():
            if col == "節":
                s = df[col].map(extract_round_number)
                if s.notna().any():
                    return s.astype("Int64"), "節->round_number"
                continue
            return df[col], col
    if DRAW_ASSIGN_GROUP_MODE != "round":
        return _build_provisional_toto13_group(df, verbose=verbose)
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if dt.notna().any():
            return dt.dt.strftime("%Y-%m-%d"), "datetime(date)"
    return None, "none"


def _target_d_range_for_block(block):
    lg = str(LEAGUE).lower()
    if "league" in block.columns and block["league"].notna().any():
        lg = str(block["league"].dropna().astype(str).iloc[0]).lower()
    if lg == "j1":
        lo, hi = _parse_target_d_range(J1_TARGET_D_RANGE_RAW, 2.0, 4.0, "J1_TARGET_D_RANGE")
    else:
        lo, hi = _parse_target_d_range(J2_TARGET_D_RANGE_RAW, 1.0, 3.0, "J2_TARGET_D_RANGE")
    return float(lo), float(hi), lg


def log_close_ha_candidate_stats(df, label):
    need = {"__close_ha_candidate", "__close_ha_gap", "__close_ha_draw_score", "__contest_group_for_close"}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"[CLOSE_HA_CAND_STATS:{label}] unavailable")
        return
    cand_mask = df["__close_ha_candidate"].fillna(False).astype(bool)
    group_cnt = (
        df.assign(__cand=cand_mask)
        .groupby("__contest_group_for_close", dropna=False)["__cand"]
        .sum()
        .astype(float)
    )
    if group_cnt.empty:
        print(f"[CLOSE_HA_CAND_STATS:{label}] unavailable")
        return
    gq = group_cnt.quantile([0.10, 0.50, 0.90])
    gap = pd.to_numeric(df.loc[cand_mask, "__close_ha_gap"], errors="coerce").dropna()
    scr = pd.to_numeric(df.loc[cand_mask, "__close_ha_draw_score"], errors="coerce").dropna()
    if gap.empty or scr.empty:
        print(
            f"[CLOSE_HA_CAND_STATS:{label}] groups={len(group_cnt)} "
            f"cand_per_group_min={float(group_cnt.min()):.1f} p10={float(gq.loc[0.10]):.1f} "
            f"median={float(gq.loc[0.50]):.1f} p90={float(gq.loc[0.90]):.1f} max={float(group_cnt.max()):.1f} mean={float(group_cnt.mean()):.2f} "
            f"candidate_rows=0"
        )
        return
    gap_q = gap.quantile([0.10, 0.50, 0.90])
    scr_q = scr.quantile([0.10, 0.50, 0.90])
    print(
        f"[CLOSE_HA_CAND_STATS:{label}] groups={len(group_cnt)} "
        f"cand_per_group_min={float(group_cnt.min()):.1f} p10={float(gq.loc[0.10]):.1f} median={float(gq.loc[0.50]):.1f} "
        f"p90={float(gq.loc[0.90]):.1f} max={float(group_cnt.max()):.1f} mean={float(group_cnt.mean()):.2f} "
        f"ha_gap_min={float(gap.min()):.6f} p10={float(gap_q.loc[0.10]):.6f} median={float(gap_q.loc[0.50]):.6f} "
        f"p90={float(gap_q.loc[0.90]):.6f} max={float(gap.max()):.6f} mean={float(gap.mean()):.6f} "
        f"draw_score_min={float(scr.min()):.6f} p10={float(scr_q.loc[0.10]):.6f} median={float(scr_q.loc[0.50]):.6f} "
        f"p90={float(scr_q.loc[0.90]):.6f} max={float(scr.max()):.6f} mean={float(scr.mean()):.6f}"
    )


def log_close_ha_v2_candidate_stats(df, label):
    need = {"__close_ha_v2_candidate", "__close_ha_v2_top_minus_pd", "__contest_group_for_close_v2"}
    if df is None or df.empty or not need.issubset(df.columns):
        print(f"[CLOSE_HA_V2_CAND_STATS:{label}] unavailable")
        return
    cand_mask = df["__close_ha_v2_candidate"].fillna(False).astype(bool)
    group_cnt = (
        df.assign(__cand=cand_mask)
        .groupby("__contest_group_for_close_v2", dropna=False)["__cand"]
        .sum()
        .astype(float)
    )
    if group_cnt.empty:
        print(f"[CLOSE_HA_V2_CAND_STATS:{label}] unavailable")
        return
    gq = group_cnt.quantile([0.10, 0.50, 0.90])
    tmd = pd.to_numeric(df.loc[cand_mask, "__close_ha_v2_top_minus_pd"], errors="coerce").dropna()
    if tmd.empty:
        print(
            f"[CLOSE_HA_V2_CAND_STATS:{label}] groups={len(group_cnt)} "
            f"cand_per_group_min={float(group_cnt.min()):.1f} p10={float(gq.loc[0.10]):.1f} "
            f"median={float(gq.loc[0.50]):.1f} p90={float(gq.loc[0.90]):.1f} max={float(group_cnt.max()):.1f} mean={float(group_cnt.mean()):.2f} "
            f"candidate_rows=0"
        )
        return
    tq = tmd.quantile([0.10, 0.50, 0.90])
    print(
        f"[CLOSE_HA_V2_CAND_STATS:{label}] groups={len(group_cnt)} "
        f"cand_per_group_min={float(group_cnt.min()):.1f} p10={float(gq.loc[0.10]):.1f} median={float(gq.loc[0.50]):.1f} "
        f"p90={float(gq.loc[0.90]):.1f} max={float(group_cnt.max()):.1f} mean={float(group_cnt.mean()):.2f} "
        f"top_minus_pd_min={float(tmd.min()):.6f} p10={float(tq.loc[0.10]):.6f} median={float(tq.loc[0.50]):.6f} "
        f"p90={float(tq.loc[0.90]):.6f} max={float(tmd.max()):.6f} mean={float(tmd.mean()):.6f}"
    )


def assign_draw_results_by_close_ha(df, output_col="predicted_result", verbose=True, draw_score_min_override=None):
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if df is None or df.empty or not required_cols.issubset(df.columns):
        return df

    out = df.copy()
    out["__ha_argmax_pred"] = out.apply(
        lambda r: "H"
        if pd.notna(r.get("prob_home_win")) and pd.notna(r.get("prob_away_win")) and float(r["prob_home_win"]) >= float(r["prob_away_win"])
        else ("A" if pd.notna(r.get("prob_home_win")) and pd.notna(r.get("prob_away_win")) else None),
        axis=1,
    )
    out["__close_ha_candidate"] = False
    out["__close_ha_gap"] = np.nan
    out["__close_ha_level"] = np.nan
    out["__close_ha_draw_score"] = np.nan

    group_series, group_key_src = _resolve_draw_assign_group(out, verbose=verbose)
    if group_series is None:
        out[output_col] = out["__ha_argmax_pred"]
        out["decision_reason"] = "ARGMAX_HA_ONLY_CLOSE_HA"
        out["force_draw_applied"] = False
        return out
    out["__contest_group_for_close"] = group_series.astype(str)

    pieces = []
    draw_score_min = float(CLOSE_HA_DRAW_SCORE_MIN if draw_score_min_override is None else draw_score_min_override)
    for g, block in out.groupby("__contest_group_for_close", dropna=False, sort=False):
        b = block.copy()
        ph = pd.to_numeric(b["prob_home_win"], errors="coerce")
        pdw = pd.to_numeric(b["prob_draw"], errors="coerce")
        pa = pd.to_numeric(b["prob_away_win"], errors="coerce")
        valid = ph.notna() & pdw.notna() & pa.notna()
        ha_gap = (ph - pa).abs()
        ha_level = pd.concat([ph, pa], axis=1).max(axis=1)
        draw_score = (pdw - ha_level) - float(CLOSE_HA_GAP_WEIGHT) * ha_gap
        cand = (
            valid
            & (ha_gap <= float(CLOSE_HA_GAP))
            & (ha_level >= float(CLOSE_HA_MIN_LEVEL))
            & (draw_score >= float(draw_score_min))
        )
        b["__close_ha_candidate"] = cand.fillna(False)
        b["__close_ha_gap"] = ha_gap
        b["__close_ha_level"] = ha_level
        b["__close_ha_draw_score"] = draw_score

        lo, hi, lg = _target_d_range_for_block(b)
        expected_draws = float(pdw[valid].sum()) if int(valid.sum()) > 0 else 0.0
        target_d = int(round(expected_draws))
        target_d = int(max(int(lo), min(int(hi), target_d)))
        n_cand = int(cand.sum())
        target_d = min(target_d, n_cand)
        if n_cand <= 0:
            target_d = 0
        draw_idx = (
            b.loc[cand]
            .sort_values("__close_ha_draw_score", ascending=False)
            .head(target_d)
            .index
        )
        draw_idx_set = set(draw_idx.tolist())
        b[output_col] = b.index.map(lambda idx: "D" if idx in draw_idx_set else b.at[idx, "__ha_argmax_pred"])
        b["decision_reason"] = b.index.map(
            lambda idx: "FORCE_DRAW_BY_CLOSE_HA" if idx in draw_idx_set else "ARGMAX_HA_ONLY_CLOSE_HA"
        )
        b["force_draw_applied"] = b.index.map(lambda idx: idx in draw_idx_set)
        if verbose:
            print(
                f"[CLOSE_HA_DRAW] group_key={group_key_src} group={g} league={lg} "
                f"matches={len(b)} candidates={n_cand} target_D={target_d} assigned_D={int((b[output_col]=='D').sum())} "
                f"gap_thr={CLOSE_HA_GAP:.3f} min_level={CLOSE_HA_MIN_LEVEL:.3f} draw_score_min={draw_score_min:.3f}"
            )
            if len(b) != 13:
                print(f"[WARN] close_ha_draw group size!=13: group={g} size={len(b)} source={group_key_src}")
        pieces.append(b)

    out = pd.concat(pieces, axis=0).sort_index()
    if output_col != "predicted_result":
        out["predicted_result"] = out[output_col]
    return out


def assign_draw_results_by_close_ha_v2(df, output_col="predicted_result", verbose=True, close_d_top_gap_override=None):
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if df is None or df.empty or not required_cols.issubset(df.columns):
        return df

    out = df.copy()
    out["__ha_argmax_pred"] = out.apply(
        lambda r: "H"
        if pd.notna(r.get("prob_home_win")) and pd.notna(r.get("prob_away_win")) and float(r["prob_home_win"]) >= float(r["prob_away_win"])
        else ("A" if pd.notna(r.get("prob_home_win")) and pd.notna(r.get("prob_away_win")) else None),
        axis=1,
    )
    out["__close_ha_v2_candidate"] = False
    out["__close_ha_v2_ha_gap"] = np.nan
    out["__close_ha_v2_top_minus_pd"] = np.nan

    group_series, group_key_src = _resolve_draw_assign_group(out, verbose=verbose)
    if group_series is None:
        out[output_col] = out["__ha_argmax_pred"]
        out["decision_reason"] = "ARGMAX_HA_ONLY_CLOSE_HA_V2"
        out["force_draw_applied"] = False
        return out
    out["__contest_group_for_close_v2"] = group_series.astype(str)

    top_gap_thr = float(CLOSE_D_TOP_GAP if close_d_top_gap_override is None else close_d_top_gap_override)
    pieces = []
    for g, block in out.groupby("__contest_group_for_close_v2", dropna=False, sort=False):
        b = block.copy()
        ph = pd.to_numeric(b["prob_home_win"], errors="coerce")
        pdw = pd.to_numeric(b["prob_draw"], errors="coerce")
        pa = pd.to_numeric(b["prob_away_win"], errors="coerce")
        valid = ph.notna() & pdw.notna() & pa.notna()
        ha_gap = (ph - pa).abs()
        top = pd.concat([ph, pdw, pa], axis=1).max(axis=1)
        top_minus_pd = top - pdw
        ha_level = pd.concat([ph, pa], axis=1).max(axis=1)
        cand = (
            valid
            & (ha_gap <= float(CLOSE_HA_GAP))
            & (ha_level >= float(CLOSE_HA_MIN_LEVEL))
            & (top_minus_pd <= top_gap_thr)
        )
        b["__close_ha_v2_candidate"] = cand.fillna(False)
        b["__close_ha_v2_ha_gap"] = ha_gap
        b["__close_ha_v2_top_minus_pd"] = top_minus_pd

        lo, hi, lg = _target_d_range_for_block(b)
        expected_draws = float(pdw[valid].sum()) if int(valid.sum()) > 0 else 0.0
        target_d = int(round(expected_draws))
        target_d = int(max(int(lo), min(int(hi), target_d)))
        n_cand = int(cand.sum())
        target_d = min(target_d, n_cand)
        if n_cand <= 0:
            target_d = 0

        # v2 は top_minus_pd が小さい順（Dに近い順）
        draw_idx = (
            b.loc[cand]
            .sort_values("__close_ha_v2_top_minus_pd", ascending=True)
            .head(target_d)
            .index
        )
        draw_idx_set = set(draw_idx.tolist())
        b[output_col] = b.index.map(lambda idx: "D" if idx in draw_idx_set else b.at[idx, "__ha_argmax_pred"])
        b["decision_reason"] = b.index.map(
            lambda idx: "FORCE_DRAW_BY_CLOSE_HA_V2" if idx in draw_idx_set else "ARGMAX_HA_ONLY_CLOSE_HA_V2"
        )
        b["force_draw_applied"] = b.index.map(lambda idx: idx in draw_idx_set)
        if verbose:
            print(
                f"[CLOSE_HA_DRAW_V2] group_key={group_key_src} group={g} league={lg} "
                f"matches={len(b)} candidates={n_cand} target_D={target_d} assigned_D={int((b[output_col]=='D').sum())} "
                f"gap_thr={CLOSE_HA_GAP:.3f} min_level={CLOSE_HA_MIN_LEVEL:.3f} top_gap={top_gap_thr:.3f}"
            )
            if len(b) != 13:
                print(f"[WARN] close_ha_draw_v2 group size!=13: group={g} size={len(b)} source={group_key_src}")
        pieces.append(b)
    out = pd.concat(pieces, axis=0).sort_index()
    if output_col != "predicted_result":
        out["predicted_result"] = out[output_col]
    return out


def _calc_confusion_named(actual_series, pred_series):
    labels = ["H", "D", "A"]
    out = {}
    for t in labels:
        for p in labels:
            out[f"{t}->{p}"] = int(((actual_series == t) & (pred_series == p)).sum())
    return out


def _print_backtest_rule_metrics(df, rule_label, dataset_label="current"):
    empty_stats = {
        "rows": 0,
        "accuracy": float("nan"),
        "confusion": {f"{t}->{p}": 0 for t in ["H", "D", "A"] for p in ["H", "D", "A"]},
        "hit_dist": {k: 0 for k in range(14)},
        "mean_hits": float("nan"),
        "assigned_d_mean": float("nan"),
        "assigned_d_var": float("nan"),
        "groups_total": 0,
        "groups_size13": 0,
        "group_source": "none",
    }
    if df is None or df.empty or "actual_result" not in df.columns or "predicted_result" not in df.columns:
        print(f"[BACKTEST_RULE] dataset={dataset_label} rule={rule_label} unavailable")
        return empty_stats
    actual = df["actual_result"].astype(str).str.upper()
    pred = df["predicted_result"].astype(str).str.upper()
    valid = actual.isin(["H", "D", "A"]) & pred.isin(["H", "D", "A"])
    n = int(valid.sum())
    if n <= 0:
        print(f"[BACKTEST_RULE] dataset={dataset_label} rule={rule_label} unavailable")
        return empty_stats
    acc = float((actual[valid] == pred[valid]).mean())
    cm = _calc_confusion_named(actual[valid], pred[valid])
    print(
        f"[BACKTEST_RULE] dataset={dataset_label} rule={rule_label} rows={n} "
        f"accuracy={acc*100:.2f}% ({int((actual[valid] == pred[valid]).sum())}/{n})"
    )
    print(
        f"[BACKTEST_CONFUSION:{dataset_label}:{rule_label}] "
        f"H->H={cm['H->H']} H->D={cm['H->D']} H->A={cm['H->A']} "
        f"D->H={cm['D->H']} D->D={cm['D->D']} D->A={cm['D->A']} "
        f"A->H={cm['A->H']} A->D={cm['A->D']} A->A={cm['A->A']}"
    )

    group_series, group_src = _resolve_draw_assign_group(df.loc[valid].copy())
    if group_series is None:
        print(f"[BACKTEST_TOTO13] dataset={dataset_label} rule={rule_label} unavailable (group_key not found)")
        st = empty_stats.copy()
        st.update({"rows": n, "accuracy": acc, "confusion": cm})
        return st
    work = df.loc[valid].copy()
    work["__contest_group"] = group_series.loc[work.index].astype(str)
    grp = work.groupby("__contest_group", dropna=False, sort=False)
    block = grp["is_correct"].agg(["sum", "count"]).rename(columns={"sum": "hits", "count": "matches"}).reset_index()
    if block.empty:
        print(f"[BACKTEST_TOTO13] dataset={dataset_label} rule={rule_label} unavailable (no grouped rows)")
        st = empty_stats.copy()
        st.update({"rows": n, "accuracy": acc, "confusion": cm})
        return st
    block["hits"] = pd.to_numeric(block["hits"], errors="coerce").fillna(0).astype(int)
    block["matches"] = pd.to_numeric(block["matches"], errors="coerce").fillna(0).astype(int)
    d_counts = work.groupby("__contest_group")["predicted_result"].apply(lambda s: int((s.astype(str).str.upper() == "D").sum()))
    d_counts = d_counts.reset_index(name="assigned_d")
    block = block.merge(d_counts, on="__contest_group", how="left")
    block["assigned_d"] = pd.to_numeric(block["assigned_d"], errors="coerce").fillna(0).astype(int)
    block13 = block[block["matches"] == 13].copy()
    source = "size13"
    if block13.empty:
        block13 = block.copy()
        source = "all_groups"
    dist = {k: 0 for k in range(14)}
    for v in block13["hits"].tolist():
        if 0 <= int(v) <= 13:
            dist[int(v)] += 1
    dist_txt = ",".join([f"{k}:{dist[k]}" for k in range(14)])
    mean_hits = float(block13["hits"].mean()) if len(block13) > 0 else 0.0
    assigned_d_mean = float(block13["assigned_d"].mean()) if len(block13) > 0 else float("nan")
    assigned_d_var = float(block13["assigned_d"].var(ddof=0)) if len(block13) > 0 else float("nan")
    print(
        f"[BACKTEST_TOTO13] dataset={dataset_label} rule={rule_label} group_key={group_src} "
        f"groups_total={len(block)} groups_size13={len(block[block['matches']==13])} "
        f"source={source} mean_hits={mean_hits:.3f} hit_dist_0_13={dist_txt}"
    )
    print(
        f"[BACKTEST_ASSIGNED_D] dataset={dataset_label} rule={rule_label} "
        f"source={source} mean={assigned_d_mean:.3f} var={assigned_d_var:.3f}"
    )
    return {
        "rows": n,
        "accuracy": acc,
        "confusion": cm,
        "hit_dist": dist,
        "mean_hits": mean_hits,
        "assigned_d_mean": assigned_d_mean,
        "assigned_d_var": assigned_d_var,
        "groups_total": int(len(block)),
        "groups_size13": int(len(block[block["matches"] == 13])),
        "group_source": source,
    }


def _apply_backtest_decision_rule(
    df,
    rule,
    draw_margin_override=None,
    close_ha_draw_score_min_override=None,
    close_d_top_gap_override=None,
    verbose=True,
):
    out = df.copy()
    if out.empty:
        if "final_result" not in out.columns:
            out["final_result"] = pd.Series(dtype="object")
        if "predicted_result" not in out.columns:
            out["predicted_result"] = pd.Series(dtype="object")
        if "decision_reason" not in out.columns:
            out["decision_reason"] = pd.Series(dtype="object")
        if "force_draw_applied" not in out.columns:
            out["force_draw_applied"] = pd.Series(dtype="bool")
        if "is_correct" not in out.columns:
            out["is_correct"] = pd.Series(dtype="bool")
        return out
    if rule == "expect":
        out = assign_draw_results_by_expectation(out, "final_result", verbose=verbose)
    elif rule == "hybrid":
        margin = float(DRAW_MARGIN if draw_margin_override is None else draw_margin_override)
        ph = pd.to_numeric(out.get("prob_home_win"), errors="coerce")
        pdw = pd.to_numeric(out.get("prob_draw"), errors="coerce")
        pa = pd.to_numeric(out.get("prob_away_win"), errors="coerce")
        ha_argmax = np.where(ph >= pa, "H", "A")
        draw_mask = pdw >= (pd.concat([ph, pa], axis=1).max(axis=1) + margin)
        out["final_result"] = np.where(draw_mask, "D", ha_argmax)
        out["decision_reason"] = np.where(draw_mask, "FORCE_DRAW_BY_HYBRID_MARGIN", "ARGMAX_HA_ONLY_HYBRID")
        out["force_draw_applied"] = draw_mask.fillna(False).astype(bool)
    elif rule == "close_ha_draw":
        out = assign_draw_results_by_close_ha(
            out,
            "final_result",
            verbose=verbose,
            draw_score_min_override=close_ha_draw_score_min_override,
        )
    elif rule == "close_ha_draw_v2":
        out = assign_draw_results_by_close_ha_v2(
            out,
            "final_result",
            verbose=verbose,
            close_d_top_gap_override=close_d_top_gap_override,
        )
    else:
        if "argmax_result" in out.columns:
            out["final_result"] = out["argmax_result"]
            out["decision_reason"] = "ARGMAX"
            out["force_draw_applied"] = False
        else:
            out = recalculate_predicted_result(out, "final_result")
    if "final_result" not in out.columns:
        out["final_result"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    out["predicted_result"] = out["final_result"]
    if "actual_result" in out.columns:
        out["is_correct"] = out["actual_result"].astype(str).str.upper() == out["predicted_result"].astype(str).str.upper()
    return out


def _collect_backtest_rule_stats(
    df,
    rule,
    draw_margin_override=None,
    close_ha_draw_score_min_override=None,
    close_d_top_gap_override=None,
):
    ruled = _apply_backtest_decision_rule(
        df,
        rule,
        draw_margin_override=draw_margin_override,
        close_ha_draw_score_min_override=close_ha_draw_score_min_override,
        close_d_top_gap_override=close_d_top_gap_override,
        verbose=False,
    )
    actual = ruled.get("actual_result", pd.Series(dtype="object")).astype(str).str.upper()
    pred = ruled.get("predicted_result", pd.Series(dtype="object")).astype(str).str.upper()
    valid = actual.isin(["H", "D", "A"]) & pred.isin(["H", "D", "A"])
    n = int(valid.sum())
    if n <= 0:
        return {
            "rows": 0,
            "accuracy": float("nan"),
            "confusion": {f"{t}->{p}": 0 for t in ["H", "D", "A"] for p in ["H", "D", "A"]},
            "hit_dist": {k: 0 for k in range(14)},
            "mean_hits": float("nan"),
            "assigned_d_mean": float("nan"),
            "assigned_d_var": float("nan"),
        }
    confusion = _calc_confusion_named(actual[valid], pred[valid])
    work = ruled.loc[valid].copy()
    group_series, group_src = _resolve_draw_assign_group(work, verbose=False)
    if group_series is None:
        return {
            "rows": n,
            "accuracy": float((actual[valid] == pred[valid]).mean()),
            "confusion": confusion,
            "hit_dist": {k: 0 for k in range(14)},
            "mean_hits": float("nan"),
            "assigned_d_mean": float("nan"),
            "assigned_d_var": float("nan"),
            "group_key": "none",
        }
    work["__contest_group"] = group_series.loc[work.index].astype(str)
    block = (
        work.groupby("__contest_group", dropna=False, sort=False)["is_correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "hits", "count": "matches"})
        .reset_index()
    )
    d_counts = work.groupby("__contest_group")["predicted_result"].apply(lambda s: int((s.astype(str).str.upper() == "D").sum()))
    d_counts = d_counts.reset_index(name="assigned_d")
    block = block.merge(d_counts, on="__contest_group", how="left")
    block["hits"] = pd.to_numeric(block["hits"], errors="coerce").fillna(0).astype(int)
    block["matches"] = pd.to_numeric(block["matches"], errors="coerce").fillna(0).astype(int)
    block["assigned_d"] = pd.to_numeric(block["assigned_d"], errors="coerce").fillna(0).astype(int)
    block13 = block[block["matches"] == 13].copy()
    if block13.empty:
        block13 = block.copy()
    dist = {k: 0 for k in range(14)}
    for v in block13["hits"].tolist():
        if 0 <= int(v) <= 13:
            dist[int(v)] += 1
    return {
        "rows": n,
        "accuracy": float((actual[valid] == pred[valid]).mean()),
        "confusion": confusion,
        "hit_dist": dist,
        "mean_hits": float(block13["hits"].mean()) if len(block13) > 0 else float("nan"),
        "assigned_d_mean": float(block13["assigned_d"].mean()) if len(block13) > 0 else float("nan"),
        "assigned_d_var": float(block13["assigned_d"].var(ddof=0)) if len(block13) > 0 else float("nan"),
        "group_key": group_src,
    }


def _load_backtest_compare_dataset(df_current):
    mode = BACKTEST_COMPARE_DATASET
    if mode in {"", "current", "in_season"}:
        print(f"[BACKTEST_ROWS] dataset=current rows={int(len(df_current))}")
        return df_current.copy(), "current"

    if BACKTEST_COMPARE_CSV:
        candidate = BACKTEST_COMPARE_CSV
    elif mode in {"full_2025", "full_2025_rounds", "2025_rounds"}:
        candidate = os.path.join(BASE_DIR, f"backtest_{LEAGUE}_2025_rounds.csv")
    else:
        candidate = os.path.join(BASE_DIR, mode)

    if not os.path.isabs(candidate):
        candidate = os.path.join(BASE_DIR, candidate)
    if not os.path.exists(candidate):
        print(f"[WARN] BACKTEST_COMPARE_DATASET not found: mode={mode} path={candidate}; fallback=current")
        print(f"[BACKTEST_ROWS] dataset=current rows={int(len(df_current))}")
        return df_current.copy(), "current"

    try:
        ext = pd.read_csv(candidate)
    except Exception as e:
        print(f"[WARN] failed to read BACKTEST_COMPARE dataset: {candidate} ({e}); fallback=current")
        print(f"[BACKTEST_ROWS] dataset=current rows={int(len(df_current))}")
        return df_current.copy(), "current"
    if ext.empty:
        print(f"[WARN] BACKTEST_COMPARE dataset empty: {candidate}; fallback=current")
        print(f"[BACKTEST_ROWS] dataset=current rows={int(len(df_current))}")
        return df_current.copy(), "current"

    for col in ["prob_home_win", "prob_draw", "prob_away_win"]:
        if col not in ext.columns:
            raise RuntimeError(f"BACKTEST_COMPARE dataset missing required probability column: {col}")
    if "actual_result" not in ext.columns:
        hs = pd.to_numeric(ext.get("home_score"), errors="coerce")
        aw = pd.to_numeric(ext.get("away_score"), errors="coerce")
        ext["actual_result"] = [get_result(h, a) for h, a in zip(hs.tolist(), aw.tolist())]
    if "argmax_result" not in ext.columns:
        ext["argmax_result"] = ext.apply(
            lambda r: decide_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])[2].get("argmax_result"),
            axis=1,
        )
    if "is_correct" not in ext.columns:
        ext["is_correct"] = False
    label = os.path.basename(candidate)
    print(f"[BACKTEST_ROWS] dataset={label} rows={int(len(ext))}")
    return ext.copy(), label


def run_backtest_decision_rule_compare(df_base):
    if df_base is None or df_base.empty:
        return
    eval_df, dataset_label = _load_backtest_compare_dataset(df_base)
    if eval_df is None or eval_df.empty:
        return
    log_draw_score_distribution(eval_df, f"{dataset_label}")
    if BACKTEST_DECISION_RULE in {"both", "all"}:
        targets = ["argmax", "expect", "hybrid", "close_ha_draw", "close_ha_draw_v2"]
    else:
        targets = [BACKTEST_DECISION_RULE]
    metrics = {}
    details = {}
    for rule in targets:
        ruled = _apply_backtest_decision_rule(eval_df, rule, verbose=True)
        details[rule] = _print_backtest_rule_metrics(ruled, rule, dataset_label=dataset_label)
        if rule == "close_ha_draw":
            log_close_ha_candidate_stats(ruled, dataset_label)
        if rule == "close_ha_draw_v2":
            log_close_ha_v2_candidate_stats(ruled, dataset_label)
        actual = ruled.get("actual_result", pd.Series(dtype="object")).astype(str).str.upper()
        pred = ruled.get("predicted_result", pd.Series(dtype="object")).astype(str).str.upper()
        valid = actual.isin(["H", "D", "A"]) & pred.isin(["H", "D", "A"])
        metrics[rule] = float((actual[valid] == pred[valid]).mean()) if int(valid.sum()) > 0 else float("nan")

    if "argmax" in details and "expect" in details:
        d_arg = details["argmax"]
        d_exp = details["expect"]
        hit10_13_arg = int(sum(d_arg["hit_dist"].get(k, 0) for k in [10, 11, 12, 13]))
        hit10_13_exp = int(sum(d_exp["hit_dist"].get(k, 0) for k in [10, 11, 12, 13]))
        print(
            f"[BACKTEST_HITDIST_DIFF] dataset={dataset_label} expect-argmax "
            f"bin10_13={hit10_13_exp-hit10_13_arg:+d} "
            f"bin0_3={sum(d_exp['hit_dist'].get(k,0) for k in [0,1,2,3]) - sum(d_arg['hit_dist'].get(k,0) for k in [0,1,2,3]):+d}"
        )
        cm_a = d_arg["confusion"]
        cm_e = d_exp["confusion"]
        print(
            f"[BACKTEST_CONFUSION_DIFF] dataset={dataset_label} expect-argmax "
            f"H->A={cm_e['H->A']-cm_a['H->A']:+d} A->H={cm_e['A->H']-cm_a['A->H']:+d} "
            f"H->D={cm_e['H->D']-cm_a['H->D']:+d} A->D={cm_e['A->D']-cm_a['A->D']:+d} D->D={cm_e['D->D']-cm_a['D->D']:+d}"
        )
        if str(LEAGUE).lower() == "j2":
            d_over = (d_exp["assigned_d_mean"] - d_arg["assigned_d_mean"]) > 0.8
            ha_flip_inc = (cm_e["H->A"] + cm_e["A->H"]) > (cm_a["H->A"] + cm_a["A->H"])
            acc_drop = metrics.get("expect", float("nan")) < metrics.get("argmax", float("nan"))
            if acc_drop and d_over:
                reason = "D_OVER_ASSIGNMENT"
            elif acc_drop and ha_flip_inc:
                reason = "HA_FLIP_INCREASE"
            elif acc_drop:
                reason = "MIXED_OR_OTHER"
            else:
                reason = "NO_DEGRADATION"
            print(
                f"[J2_DEGRADE_REASON] dataset={dataset_label} reason={reason} "
                f"acc_diff_pp={(metrics.get('expect', float('nan'))-metrics.get('argmax', float('nan')))*100.0:+.2f} "
                f"assigned_d_mean_diff={d_exp['assigned_d_mean']-d_arg['assigned_d_mean']:+.3f} "
                f"ha_flip_diff={(cm_e['H->A']+cm_e['A->H'])-(cm_a['H->A']+cm_a['A->H']):+d}"
            )

    if "argmax" in metrics and "expect" in metrics and not np.isnan(metrics["argmax"]) and not np.isnan(metrics["expect"]):
        diff_pp = (metrics["expect"] - metrics["argmax"]) * 100.0
        print(
            f"[BACKTEST_RULE_DIFF] dataset={dataset_label} expect_minus_argmax={diff_pp:+.2f}pp "
            f"(expect={metrics['expect']*100:.2f}% argmax={metrics['argmax']*100:.2f}%)"
        )
    if "argmax" in metrics and "hybrid" in metrics and not np.isnan(metrics["argmax"]) and not np.isnan(metrics["hybrid"]):
        diff_pp = (metrics["hybrid"] - metrics["argmax"]) * 100.0
        print(
            f"[BACKTEST_RULE_DIFF] dataset={dataset_label} hybrid_minus_argmax={diff_pp:+.2f}pp "
            f"(hybrid={metrics['hybrid']*100:.2f}% argmax={metrics['argmax']*100:.2f}%) "
            f"draw_margin={DRAW_MARGIN:.3f}"
        )
    if "expect" in metrics and "hybrid" in metrics and not np.isnan(metrics["expect"]) and not np.isnan(metrics["hybrid"]):
        diff_pp = (metrics["hybrid"] - metrics["expect"]) * 100.0
        print(
            f"[BACKTEST_RULE_DIFF] dataset={dataset_label} hybrid_minus_expect={diff_pp:+.2f}pp "
            f"(hybrid={metrics['hybrid']*100:.2f}% expect={metrics['expect']*100:.2f}%) "
            f"draw_margin={DRAW_MARGIN:.3f}"
        )
    if "argmax" in metrics and "close_ha_draw" in metrics and not np.isnan(metrics["argmax"]) and not np.isnan(metrics["close_ha_draw"]):
        diff_pp = (metrics["close_ha_draw"] - metrics["argmax"]) * 100.0
        print(
            f"[BACKTEST_RULE_DIFF] dataset={dataset_label} close_ha_draw_minus_argmax={diff_pp:+.2f}pp "
            f"(close_ha_draw={metrics['close_ha_draw']*100:.2f}% argmax={metrics['argmax']*100:.2f}%)"
        )
    if "argmax" in metrics and "close_ha_draw_v2" in metrics and not np.isnan(metrics["argmax"]) and not np.isnan(metrics["close_ha_draw_v2"]):
        diff_pp = (metrics["close_ha_draw_v2"] - metrics["argmax"]) * 100.0
        print(
            f"[BACKTEST_RULE_DIFF] dataset={dataset_label} close_ha_draw_v2_minus_argmax={diff_pp:+.2f}pp "
            f"(close_ha_draw_v2={metrics['close_ha_draw_v2']*100:.2f}% argmax={metrics['argmax']*100:.2f}%)"
        )

    do_scan = bool(BACKTEST_MARGIN_SCAN or BACKTEST_DECISION_RULE == "all")
    if not do_scan:
        return
    margin_grid = _parse_sensitivity_float_values(
        DRAW_MARGIN_GRID_RAW,
        [0.05, 0.04, 0.03, 0.02, 0.01, 0.00, -0.01, -0.02, -0.03],
    )
    base_arg = _collect_backtest_rule_stats(eval_df, "argmax")
    base_exp = _collect_backtest_rule_stats(eval_df, "expect")
    scan_rows = []
    for margin in margin_grid:
        st_h = _collect_backtest_rule_stats(eval_df, "hybrid", draw_margin_override=margin)
        h1013 = int(sum(st_h["hit_dist"].get(k, 0) for k in [10, 11, 12, 13]))
        a1013 = int(sum(base_arg["hit_dist"].get(k, 0) for k in [10, 11, 12, 13]))
        print(
            f"[DRAW_MARGIN_SCAN] dataset={dataset_label} margin={margin:+.3f} "
            f"acc_argmax={base_arg['accuracy']*100.0:.2f}% acc_expect={base_exp['accuracy']*100.0:.2f}% acc_hybrid={st_h['accuracy']*100.0:.2f}% "
            f"assignedD_mean={st_h['assigned_d_mean']:.3f} assignedD_var={st_h['assigned_d_var']:.3f} "
            f"hit10_13_delta_vs_argmax={h1013-a1013:+d}"
        )
        scan_rows.append(
            {
                "margin": float(margin),
                "acc_hybrid": float(st_h["accuracy"]),
                "assigned_d_mean": float(st_h["assigned_d_mean"]),
                "assigned_d_var": float(st_h["assigned_d_var"]),
                "hit10_13_delta": int(h1013 - a1013),
            }
        )
    if not scan_rows:
        return
    if str(LEAGUE).lower() == "j1":
        tgt_lo, tgt_hi = _parse_target_d_range(J1_TARGET_D_RANGE_RAW, 2.0, 4.0, "J1_TARGET_D_RANGE")
    else:
        tgt_lo, tgt_hi = _parse_target_d_range(J2_TARGET_D_RANGE_RAW, 1.0, 3.0, "J2_TARGET_D_RANGE")
    tgt_center = (tgt_lo + tgt_hi) / 2.0

    def _d_range_penalty(x):
        if np.isnan(x):
            return 999.0
        if x < tgt_lo:
            return float(tgt_lo - x)
        if x > tgt_hi:
            return float(x - tgt_hi)
        return abs(float(x) - tgt_center) * 0.1

    scan_rows_sorted = sorted(
        scan_rows,
        key=lambda r: (-float(r["acc_hybrid"]), _d_range_penalty(r["assigned_d_mean"]), abs(float(r["margin"]))),
    )
    best = scan_rows_sorted[0]
    print(
        f"[DRAW_MARGIN_RECOMMEND] league={LEAGUE} dataset={dataset_label} "
        f"margin={best['margin']:+.3f} acc_hybrid={best['acc_hybrid']*100.0:.2f}% "
        f"assignedD_mean={best['assigned_d_mean']:.3f} assignedD_var={best['assigned_d_var']:.3f} "
        f"target_d_range={tgt_lo:.1f}-{tgt_hi:.1f}"
    )

    close_min_grid = _parse_sensitivity_float_values(CLOSE_HA_DRAW_SCORE_MIN_GRID_RAW, [-0.03, -0.02, -0.01, 0.0, 0.01])
    base_arg2 = _collect_backtest_rule_stats(eval_df, "argmax")
    close_rows = []
    for smin in close_min_grid:
        st_c = _collect_backtest_rule_stats(eval_df, "close_ha_draw", close_ha_draw_score_min_override=float(smin))
        c1013 = int(sum(st_c["hit_dist"].get(k, 0) for k in [10, 11, 12, 13]))
        a1013 = int(sum(base_arg2["hit_dist"].get(k, 0) for k in [10, 11, 12, 13]))
        print(
            f"[CLOSE_HA_SCOREMIN_SCAN] dataset={dataset_label} score_min={float(smin):+.3f} "
            f"acc_close_ha={st_c['accuracy']*100.0:.2f}% acc_argmax={base_arg2['accuracy']*100.0:.2f}% "
            f"assignedD_mean={st_c['assigned_d_mean']:.3f} assignedD_var={st_c['assigned_d_var']:.3f} "
            f"hit10_13_delta_vs_argmax={c1013-a1013:+d}"
        )
        close_rows.append(
            {
                "score_min": float(smin),
                "acc_close_ha": float(st_c["accuracy"]),
                "assigned_d_mean": float(st_c["assigned_d_mean"]),
                "assigned_d_var": float(st_c["assigned_d_var"]),
                "hit10_13_delta": int(c1013 - a1013),
            }
        )
    if close_rows:
        close_rows_sorted = sorted(
            close_rows,
            key=lambda r: (-float(r["acc_close_ha"]), _d_range_penalty(r["assigned_d_mean"]), abs(float(r["score_min"]))),
        )
        best_c = close_rows_sorted[0]
        print(
            f"[CLOSE_HA_SCOREMIN_RECOMMEND] league={LEAGUE} dataset={dataset_label} "
            f"score_min={best_c['score_min']:+.3f} acc_close_ha={best_c['acc_close_ha']*100.0:.2f}% "
            f"assignedD_mean={best_c['assigned_d_mean']:.3f} assignedD_var={best_c['assigned_d_var']:.3f} "
            f"target_d_range={tgt_lo:.1f}-{tgt_hi:.1f}"
        )

    v2_gap_grid = _parse_sensitivity_float_values(CLOSE_D_TOP_GAP_GRID, [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    base_arg3 = _collect_backtest_rule_stats(eval_df, "argmax")
    v2_rows = []
    for gap in v2_gap_grid:
        st_v2 = _collect_backtest_rule_stats(eval_df, "close_ha_draw_v2", close_d_top_gap_override=float(gap))
        v1013 = int(sum(st_v2["hit_dist"].get(k, 0) for k in [10, 11, 12, 13]))
        a1013 = int(sum(base_arg3["hit_dist"].get(k, 0) for k in [10, 11, 12, 13]))
        cm_a = base_arg3["confusion"]
        cm_v = st_v2["confusion"]
        print(
            f"[CLOSE_HA_V2_GAP_SCAN] dataset={dataset_label} top_gap={float(gap):.3f} "
            f"acc_close_ha_v2={st_v2['accuracy']*100.0:.2f}% acc_argmax={base_arg3['accuracy']*100.0:.2f}% "
            f"assignedD_mean={st_v2['assigned_d_mean']:.3f} assignedD_var={st_v2['assigned_d_var']:.3f} "
            f"hit10_13_delta_vs_argmax={v1013-a1013:+d} "
            f"conf(H->D)={cm_v['H->D']-cm_a['H->D']:+d} conf(A->D)={cm_v['A->D']-cm_a['A->D']:+d} "
            f"conf(D->D)={cm_v['D->D']-cm_a['D->D']:+d}"
        )
        v2_rows.append(
            {
                "top_gap": float(gap),
                "acc_close_ha_v2": float(st_v2["accuracy"]),
                "assigned_d_mean": float(st_v2["assigned_d_mean"]),
                "assigned_d_var": float(st_v2["assigned_d_var"]),
                "hit10_13_delta": int(v1013 - a1013),
            }
        )
    if v2_rows:
        v2_rows_sorted = sorted(
            v2_rows,
            key=lambda r: (-float(r["acc_close_ha_v2"]), _d_range_penalty(r["assigned_d_mean"]), abs(float(r["top_gap"] - CLOSE_D_TOP_GAP))),
        )
        best_v2 = v2_rows_sorted[0]
        print(
            f"[CLOSE_HA_V2_GAP_RECOMMEND] league={LEAGUE} dataset={dataset_label} "
            f"top_gap={best_v2['top_gap']:.3f} acc_close_ha_v2={best_v2['acc_close_ha_v2']*100.0:.2f}% "
            f"assignedD_mean={best_v2['assigned_d_mean']:.3f} assignedD_var={best_v2['assigned_d_var']:.3f} "
            f"target_d_range={tgt_lo:.1f}-{tgt_hi:.1f}"
        )


def assign_draw_results_by_expectation(df, output_col="predicted_result", verbose=True):
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if not required_cols.issubset(df.columns):
        return df

    out = df.copy()
    out["__base_pred"] = out.apply(
        lambda r: decide_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])[0],
        axis=1,
    )
    out["__ha_argmax_pred"] = out.apply(
        lambda r: "H"
        if pd.notna(r.get("prob_home_win")) and pd.notna(r.get("prob_away_win")) and float(r["prob_home_win"]) >= float(r["prob_away_win"])
        else ("A" if pd.notna(r.get("prob_home_win")) and pd.notna(r.get("prob_away_win")) else None),
        axis=1,
    )
    out["decision_reason"] = out.apply(
        lambda r: decide_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])[1],
        axis=1,
    )

    def _assign_block(block: pd.DataFrame, group_label: str, group_key_src: str) -> pd.DataFrame:
        b = block.copy()
        valid = b["prob_draw"].notna() & b["prob_home_win"].notna() & b["prob_away_win"].notna()
        valid_count = int(valid.sum())
        block_count = int(len(b))
        if valid_count == 0:
            b[output_col] = b["__base_pred"]
            if verbose:
                print(
                    f"[DRAW_ASSIGN] group={group_label} group_key={group_key_src} matches={block_count} "
                    f"Expected_draws_raw=0.00 Expected_draws_scaled=0.00 "
                    f"target_draw_count=0 Assigned_D=0 overwrite_targets=[]"
                )
            return b

        expected_draws_raw = float(b.loc[valid, "prob_draw"].sum())
        expected_draws_scaled = expected_draws_raw * DRAW_EXPECTATION_MULTIPLIER
        target_draw_count = int(round(expected_draws_scaled))
        target_draw_count = max(0, min(target_draw_count, valid_count))

        b.loc[valid, "__draw_priority"] = (
            pd.to_numeric(b.loc[valid, "prob_draw"], errors="coerce")
            - pd.concat(
                [
                    pd.to_numeric(b.loc[valid, "prob_home_win"], errors="coerce"),
                    pd.to_numeric(b.loc[valid, "prob_away_win"], errors="coerce"),
                ],
                axis=1,
            ).max(axis=1)
        )
        top_draw = b.loc[valid].sort_values("__draw_priority", ascending=False).head(target_draw_count)
        draw_idx = top_draw.index
        draw_target_col = "match_no" if "match_no" in b.columns else ("match_id" if "match_id" in b.columns else None)
        if draw_target_col:
            draw_targets = top_draw[draw_target_col].astype(str).tolist()
        else:
            draw_targets = [str(i) for i in draw_idx.tolist()]
        draw_targets_txt = "[" + ",".join(draw_targets) + "]"

        draw_idx_set = set(draw_idx.tolist())
        b[output_col] = b.index.map(lambda idx: "D" if idx in draw_idx_set else b.at[idx, "__ha_argmax_pred"])
        b["decision_reason"] = b.index.map(
            lambda idx: "FORCE_DRAW_BY_EXPECTATION_ASSIGN" if idx in draw_idx_set else "ARGMAX_HA_ONLY_AFTER_EXPECTATION_ASSIGN"
        )
        assigned_d = int((b.loc[valid, output_col] == "D").sum())
        if verbose:
            print(
                f"[DRAW_ASSIGN] group={group_label} group_key={group_key_src} matches={block_count} "
                f"Expected_draws_raw={expected_draws_raw:.2f} Expected_draws_scaled={expected_draws_scaled:.2f} "
                f"target_draw_count={target_draw_count} "
                f"Assigned_D={assigned_d} overwrite_targets={draw_targets_txt} "
                f"score_rule=PD-max(PH,PA)"
            )
            if block_count != 13:
                print(
                    f"[DRAW_ASSIGN][WARN] group={group_label} size={block_count} "
                    f"(toto13 expected 13; source={group_key_src})"
                )
        return b

    # toto開催回(13試合)を優先して割当。列がない場合のみ round/date へフォールバック。
    group_series, group_key_src = _resolve_draw_assign_group(out, verbose=verbose)
    if group_series is None:
        out = _assign_block(out, "ALL", group_key_src)
    else:
        out["__draw_group"] = group_series
        pieces = []
        for g, block in out.groupby("__draw_group", dropna=False, sort=False):
            label = str(g) if pd.notna(g) and str(g).strip() else "NA"
            pieces.append(_assign_block(block, label, group_key_src))
        out = pd.concat(pieces, axis=0).sort_index()

    # 最終結果列を output_col に一本化しつつ、互換列 predicted_result も同期する
    if output_col in out.columns and output_col != "predicted_result":
        out["predicted_result"] = out[output_col]
    out["force_draw_applied"] = out.get("decision_reason", pd.Series("", index=out.index)).astype(str).str.contains("FORCE_DRAW", na=False)
    out = out.drop(columns=["__base_pred", "__ha_argmax_pred", "__draw_priority", "__round_group", "__draw_group"], errors="ignore")
    return out


def recalculate_predicted_result(df, output_col="predicted_result"):
    out = df.copy()
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win"}
    if not required_cols.issubset(out.columns):
        return out
    def _decide_triplet(row):
        return decide_result(row["prob_home_win"], row["prob_draw"], row["prob_away_win"])

    decided = out.apply(_decide_triplet, axis=1)
    out[output_col] = decided.map(lambda x: x[0])
    out["final_result"] = out[output_col]
    out["argmax_result"] = decided.map(lambda x: x[2].get("argmax_result"))
    out["argmax_max_prob"] = decided.map(lambda x: x[2].get("argmax_max_prob"))
    if "decision_reason" not in out.columns:
        out["decision_reason"] = decided.map(lambda x: x[1])
    if "force_draw_applied" not in out.columns:
        out["force_draw_applied"] = out["decision_reason"].astype(str).str.contains("FORCE_DRAW", na=False)
    return out


def recalculate_predicted_highest_prob_result(df, output_col="predicted_highest_prob_result"):
    out = df.copy()
    required_cols = {"prob_home_win_raw", "prob_draw_raw", "prob_away_win_raw"}
    if not required_cols.issubset(out.columns):
        return out
    out[output_col] = out.apply(
        lambda r: decide_result(r["prob_home_win_raw"], r["prob_draw_raw"], r["prob_away_win_raw"])[0],
        axis=1,
    )
    out["argmax_raw_result"] = out[output_col]
    out["argmax_raw_max_prob"] = out[["prob_home_win_raw", "prob_draw_raw", "prob_away_win_raw"]].max(axis=1)
    return out


def apply_round_type_draw_control(df, label):
    out = df.copy()
    required = {"prob_home_win", "prob_draw", "prob_away_win"}
    if (not ENABLE_ROUND_TYPE_DRAW_CONTROL) or out.empty or (not required.issubset(out.columns)):
        return out

    draw_probs = pd.to_numeric(out["prob_draw"], errors="coerce")
    if int(draw_probs.notna().sum()) == 0:
        return out

    median_draw = float(draw_probs.median())
    avg_draw = float(draw_probs.mean())
    rel = draw_probs - median_draw
    count_high = int((rel > ROUND_TYPE_DRAW_REL_THRESHOLD).sum())
    count_low = int((rel < -ROUND_TYPE_DRAW_REL_THRESHOLD).sum())
    share_high = count_high / max(len(out), 1)
    share_low = count_low / max(len(out), 1)

    round_type = "NORMAL"
    if avg_draw >= ROUND_TYPE_DRAW_HEAVY_AVG and share_high >= ROUND_TYPE_DRAW_SHARE_THRESHOLD:
        round_type = "DRAW_HEAVY"
    elif avg_draw <= ROUND_TYPE_DRAW_LIGHT_AVG and share_low >= ROUND_TYPE_DRAW_SHARE_THRESHOLD:
        round_type = "DRAW_LIGHT"

    out["draw_rel"] = rel
    out["round_type"] = round_type
    out["round_median_draw"] = median_draw
    out["round_avg_draw"] = avg_draw

    out["prob_draw_before_round_type"] = out["prob_draw"]
    out["prob_home_before_round_type"] = out["prob_home_win"]
    out["prob_away_before_round_type"] = out["prob_away_win"]
    out["prob_draw_adj"] = out["prob_draw"]
    out["round_type_adjust_applied"] = False

    if round_type != "NORMAL":
        boost = ROUND_TYPE_DRAW_BOOST if round_type == "DRAW_HEAVY" else -ROUND_TYPE_DRAW_BOOST
        if round_type == "DRAW_HEAVY":
            target_mask = draw_probs >= median_draw
        else:
            target_mask = draw_probs <= median_draw

        for idx in out.index[target_mask.fillna(False)]:
            ph = float(out.at[idx, "prob_home_win"])
            pdw = float(out.at[idx, "prob_draw"])
            pa = float(out.at[idx, "prob_away_win"])
            adj_draw = min(ROUND_TYPE_DRAW_CLAMP_MAX, max(ROUND_TYPE_DRAW_CLAMP_MIN, pdw + boost))
            delta = float(adj_draw - pdw)
            ha_sum = ph + pa
            if abs(delta) > 1e-12 and ha_sum > 1e-12:
                ph -= delta * (ph / ha_sum)
                pa -= delta * (pa / ha_sum)
            ph, pdw, pa = _normalize_probs(ph, adj_draw, pa)
            out.at[idx, "prob_home_win"] = ph
            out.at[idx, "prob_draw"] = pdw
            out.at[idx, "prob_away_win"] = pa
            out.at[idx, "prob_draw_adj"] = pdw
            out.at[idx, "round_type_adjust_applied"] = True

        if "final_prob_home" in out.columns:
            out["final_prob_home"] = out["prob_home_win"]
        if "final_prob_draw" in out.columns:
            out["final_prob_draw"] = out["prob_draw"]
        if "final_prob_away" in out.columns:
            out["final_prob_away"] = out["prob_away_win"]
        if "prob_delta_home" in out.columns and "base_prob_home" in out.columns:
            out["prob_delta_home"] = pd.to_numeric(out["prob_home_win"], errors="coerce") - pd.to_numeric(out["base_prob_home"], errors="coerce")
        if "prob_delta_draw" in out.columns and "base_prob_draw" in out.columns:
            out["prob_delta_draw"] = pd.to_numeric(out["prob_draw"], errors="coerce") - pd.to_numeric(out["base_prob_draw"], errors="coerce")
        if "prob_delta_away" in out.columns and "base_prob_away" in out.columns:
            out["prob_delta_away"] = pd.to_numeric(out["prob_away_win"], errors="coerce") - pd.to_numeric(out["base_prob_away"], errors="coerce")

    adjusted_rows = int(pd.Series(out["round_type_adjust_applied"]).fillna(False).astype(bool).sum())
    print(
        f"[ROUND_TYPE:{label}] round_type={round_type} median_draw={median_draw:.4f} avg_draw={avg_draw:.4f} "
        f"count_high={count_high} count_low={count_low} adjusted_rows={adjusted_rows}"
    )
    return out


def sync_and_validate_prediction_results(df, label, raise_on_error=True):
    out = df.copy()
    if "final_result" not in out.columns and "predicted_result" in out.columns:
        out["final_result"] = out["predicted_result"]
    if "predicted_result" not in out.columns and "final_result" in out.columns:
        out["predicted_result"] = out["final_result"]
    if "predicted_result" in out.columns and "final_result" in out.columns:
        mismatch = out["predicted_result"].astype(str) != out["final_result"].astype(str)
        m = int(mismatch.sum())
        if m > 0:
            sample_cols = [c for c in ["match_id", "match_no", "home_team", "away_team", "predicted_result", "final_result"] if c in out.columns]
            sample = out.loc[mismatch, sample_cols].head(10).to_dict(orient="records")
            msg = f"[CONSISTENCY][ERROR:{label}] predicted_result!=final_result rows={m} sample={sample}"
            print(msg)
            if raise_on_error:
                raise RuntimeError(msg)
        out["predicted_result"] = out["final_result"]

    if "decision_reason" in out.columns and "final_result" in out.columns:
        force_mask = out["decision_reason"].astype(str).str.contains("FORCE_DRAW", na=False)
        bad_force = force_mask & (out["final_result"].astype(str) != "D")
        bad = int(bad_force.sum())
        if bad > 0:
            sample_cols = [c for c in ["match_id", "match_no", "home_team", "away_team", "decision_reason", "final_result"] if c in out.columns]
            sample = out.loc[bad_force, sample_cols].head(10).to_dict(orient="records")
            msg = f"[CONSISTENCY][ERROR:{label}] FORCE_DRAW reason but final_result!=D rows={bad} sample={sample}"
            print(msg)
            if raise_on_error:
                raise RuntimeError(msg)

    if "argmax_raw_result" in out.columns and "final_result" in out.columns:
        diff = int((out["argmax_raw_result"].astype(str) != out["final_result"].astype(str)).sum())
        print(f"[CONSISTENCY:{label}] raw_vs_final_diff_rows={diff}")
    out["force_draw_applied"] = out.get("decision_reason", pd.Series("", index=out.index)).astype(str).str.contains("FORCE_DRAW", na=False)
    return out


def log_prediction_consistency(df, label):
    required = {
        "prob_home_win_raw",
        "prob_draw_raw",
        "prob_away_win_raw",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "predicted_result",
        "predicted_highest_prob_result",
    }
    if not required.issubset(df.columns):
        print(f"[PRED_CHECK:{label}] required_columns_missing")
        return

    work = df.copy()
    work["_raw_argmax"] = work.apply(
        lambda r: decide_result(r["prob_home_win_raw"], r["prob_draw_raw"], r["prob_away_win_raw"])[0],
        axis=1,
    )
    work["_cal_argmax"] = work.apply(
        lambda r: decide_result(r["prob_home_win"], r["prob_draw"], r["prob_away_win"])[0],
        axis=1,
    )

    rows = len(work)
    if rows == 0:
        print(f"[PRED_CHECK:{label}] rows=0")
        return

    match_id_col = "match_id" if "match_id" in work.columns else None
    raw_vs_cal_match = (work["_raw_argmax"] == work["_cal_argmax"])
    pred_vs_highest_match = (work["predicted_result"] == work["predicted_highest_prob_result"])
    delta_draw = pd.to_numeric(work["prob_draw"], errors="coerce") - pd.to_numeric(work["prob_draw_raw"], errors="coerce")
    postfusion_col = "predicted_result_main_postfusion_argmax"
    has_postfusion = postfusion_col in work.columns
    if has_postfusion:
        postfusion = work[postfusion_col].astype(str).str.upper()
        pred_vs_postfusion_match = (work["predicted_result"].astype(str).str.upper() == postfusion)
    else:
        postfusion = pd.Series("", index=work.index)
        pred_vs_postfusion_match = pd.Series(False, index=work.index)

    print(
        f"[PRED_CHECK:{label}] raw_argmax_vs_cal_argmax_match_rate={raw_vs_cal_match.mean()*100:.1f}% "
        f"pred_vs_highest_match_rate={pred_vs_highest_match.mean()*100:.1f}%"
    )
    if has_postfusion:
        print(
            f"[PRED_CHECK:{label}] pred_vs_postfusion_main_argmax_match_rate="
            f"{pred_vs_postfusion_match.mean()*100:.1f}% "
            f"mismatch_count={int((~pred_vs_postfusion_match).sum())}"
        )
    print(
        f"[PRED_CHECK:{label}] draw_delta(mean/max/plus_count)="
        f"{delta_draw.mean(skipna=True):.6f}/{delta_draw.max(skipna=True):.6f}/{int((delta_draw > 0).sum())}"
    )

    pred = work["predicted_result"].astype(str)
    h_rate = (pred == "H").mean() * 100
    d_rate = (pred == "D").mean() * 100
    a_rate = (pred == "A").mean() * 100
    print(f"[PRED_CHECK:{label}] predicted_result_ratio(H/D/A)={h_rate:.1f}%/{d_rate:.1f}%/{a_rate:.1f}%")
    if "predicted_result_source" in work.columns:
        source_counts = work["predicted_result_source"].astype(str).value_counts(dropna=False).to_dict()
        print(f"[PRED_CHECK:{label}] predicted_result_source_counts={source_counts}")

    mismatch_highest = work[work["predicted_highest_prob_result"] != work["_raw_argmax"]]
    mismatch_pred = work[work["predicted_result"] != work["_cal_argmax"]]
    mismatch_postfusion = work[work["predicted_result"].astype(str).str.upper() != postfusion] if has_postfusion else pd.DataFrame()
    if len(mismatch_highest) > 0:
        mids = mismatch_highest[match_id_col].astype(str).tolist() if match_id_col else mismatch_highest.index.astype(str).tolist()
        print(f"[PRED_CHECK:{label}][WARN] highest_vs_raw_argmax_mismatch={len(mismatch_highest)} match_ids={mids}")
    else:
        print(f"[PRED_CHECK:{label}] highest_vs_raw_argmax_mismatch=0")
    if len(mismatch_pred) > 0:
        mids = mismatch_pred[match_id_col].astype(str).tolist() if match_id_col else mismatch_pred.index.astype(str).tolist()
        print(f"[PRED_CHECK:{label}][WARN] predicted_vs_cal_argmax_mismatch={len(mismatch_pred)} match_ids={mids}")
    else:
        print(f"[PRED_CHECK:{label}] predicted_vs_cal_argmax_mismatch=0")
    if has_postfusion:
        if len(mismatch_postfusion) > 0:
            mids = mismatch_postfusion[match_id_col].astype(str).tolist() if match_id_col else mismatch_postfusion.index.astype(str).tolist()
            print(f"[PRED_CHECK:{label}][WARN] predicted_vs_postfusion_main_argmax_mismatch={len(mismatch_postfusion)} match_ids={mids}")
        else:
            print(f"[PRED_CHECK:{label}] predicted_vs_postfusion_main_argmax_mismatch=0")

    if "actual_result" in work.columns and has_postfusion:
        actual = work["actual_result"].astype(str).str.upper()
        valid_actual = actual.isin(["H", "D", "A"])
        pred_final = work["predicted_result"].astype(str).str.upper()
        valid_pred = pred_final.isin(["H", "D", "A"])
        valid_postfusion = postfusion.isin(["H", "D", "A"])
        final_mask = valid_actual & valid_pred
        postfusion_mask = valid_actual & valid_postfusion
        final_acc = float((pred_final[final_mask] == actual[final_mask]).mean()) if int(final_mask.sum()) > 0 else float("nan")
        postfusion_acc = float((postfusion[postfusion_mask] == actual[postfusion_mask]).mean()) if int(postfusion_mask.sum()) > 0 else float("nan")
        delta_acc = final_acc - postfusion_acc if pd.notna(final_acc) and pd.notna(postfusion_acc) else float("nan")
        print(
            f"[PRED_CHECK:{label}] final_vs_postfusion_accuracy="
            f"{final_acc:.4f}/{postfusion_acc:.4f} delta={delta_acc:+.4f}"
        )


def _normalize_probs(ph, pdw, pa):
    arr = np.array([ph, pdw, pa], dtype=float)
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    arr = arr / s
    return float(arr[0]), float(arr[1]), float(arr[2])


def apply_away_prob_multiplier(p_home, p_draw, p_away, league):
    ph, pdw, pa = _normalize_probs(p_home, p_draw, p_away)
    if float(AWAY_PROB_MULTIPLIER) <= 1.0:
        return ph, pdw, pa
    lg = str(league if pd.notna(league) and str(league).strip() else LEAGUE).strip().upper()
    if lg != "J1":
        return ph, pdw, pa
    pa *= float(AWAY_PROB_MULTIPLIER)
    return _normalize_probs(ph, pdw, pa)


def _apply_prob_pipeline(raw_home, raw_draw, raw_away, league):
    base_home, base_draw, base_away = _normalize_probs(raw_home, raw_draw, raw_away)
    base_home, base_draw, base_away = apply_away_prob_multiplier(base_home, base_draw, base_away, league)
    if DRAW_TWEAK_ENABLED:
        if HDA_MODEL_MODE_EFFECTIVE == "multinom":
            final_home, final_draw, final_away = _normalize_probs(base_home, base_draw, base_away)
        else:
            final_home, final_draw, final_away = calibrate_probabilities(
                base_home,
                base_draw,
                base_away,
                league,
            )
        final_home, final_draw, final_away = _normalize_probs(final_home, final_draw, final_away)
    else:
        final_home, final_draw, final_away = base_home, base_draw, base_away

    d_home = float(final_home - base_home)
    d_draw = float(final_draw - base_draw)
    d_away = float(final_away - base_away)
    residue = bool(max(abs(d_home), abs(d_draw), abs(d_away)) > 1e-12)
    if (not DRAW_TWEAK_ENABLED) and residue:
        raise RuntimeError(
            "[RESIDUE_DETECTED] DRAW_TWEAK_MODE=off but final_probs differ from base_probs "
            f"(dH={d_home:.12f} dD={d_draw:.12f} dA={d_away:.12f})"
        )
    return {
        "base_home": float(base_home),
        "base_draw": float(base_draw),
        "base_away": float(base_away),
        "final_home": float(final_home),
        "final_draw": float(final_draw),
        "final_away": float(final_away),
        "delta_home": d_home,
        "delta_draw": d_draw,
        "delta_away": d_away,
        "residue_detected": residue,
    }


def log_draw_tweak_audit(df, label):
    required = {
        "base_prob_home",
        "base_prob_draw",
        "base_prob_away",
        "final_prob_home",
        "final_prob_draw",
        "final_prob_away",
        "decision_reason",
    }
    if df is None or df.empty or not required.issubset(set(df.columns)):
        print(f"[DRAW_TWEAK_AUDIT:{label}] unavailable")
        return

    def _mean(cols):
        arr = [pd.to_numeric(df[c], errors="coerce") for c in cols]
        return [float(s.mean()) if int(s.notna().sum()) > 0 else float("nan") for s in arr]

    b_h, b_d, b_a = _mean(["base_prob_home", "base_prob_draw", "base_prob_away"])
    f_h, f_d, f_a = _mean(["final_prob_home", "final_prob_draw", "final_prob_away"])
    d_h = f_h - b_h
    d_d = f_d - b_d
    d_a = f_a - b_a
    residue = int(pd.Series(df.get("residue_detected", False)).fillna(False).astype(bool).sum())
    argmax_only = int((df["decision_reason"].astype(str) == "ARGMAX").sum())
    print(
        f"[DRAW_TWEAK_AUDIT:{label}] mode={DRAW_TWEAK_MODE} rows={int(len(df))} "
        f"base(H/D/A)={b_h:.6f}/{b_d:.6f}/{b_a:.6f} "
        f"final(H/D/A)={f_h:.6f}/{f_d:.6f}/{f_a:.6f} "
        f"delta(H/D/A)={d_h:+.12f}/{d_d:+.12f}/{d_a:+.12f} "
        f"residue_detected={residue} decision_reason_ARGMAX={argmax_only}/{int(len(df))}"
    )

    if "league" in df.columns:
        for lg, part in df.groupby("league", dropna=False, sort=True):
            pb_h, pb_d, pb_a = [float(pd.to_numeric(part[c], errors="coerce").mean()) for c in ["base_prob_home", "base_prob_draw", "base_prob_away"]]
            pf_h, pf_d, pf_a = [float(pd.to_numeric(part[c], errors="coerce").mean()) for c in ["final_prob_home", "final_prob_draw", "final_prob_away"]]
            r_cnt = int(pd.Series(part.get("residue_detected", False)).fillna(False).astype(bool).sum())
            print(
                f"[DRAW_TWEAK_AUDIT:{label}:league={lg}] mode={DRAW_TWEAK_MODE} rows={int(len(part))} "
                f"base(H/D/A)={pb_h:.6f}/{pb_d:.6f}/{pb_a:.6f} "
                f"final(H/D/A)={pf_h:.6f}/{pf_d:.6f}/{pf_a:.6f} "
                f"delta(H/D/A)={pf_h-pb_h:+.12f}/{pf_d-pb_d:+.12f}/{pf_a-pb_a:+.12f} "
                f"residue_detected={r_cnt}"
            )

def enforce_elo_sign_monotonic(prob_home_win, prob_draw, prob_away_win, elo_diff_for_prob):
    ph, pdw, pa = _normalize_probs(prob_home_win, prob_draw, prob_away_win)
    if not ENFORCE_ELO_SIGN_MONOTONIC:
        return ph, pdw, pa, None
    d = float(elo_diff_for_prob)
    fix_reason = None
    if d < 0 and ph > pa:
        # away優勢（diff<0）なのにhome>awayの場合は、H/Aを入れ替えて単調性を担保
        ph, pa = pa, ph
        fix_reason = "neg_diff_home_gt_away_swap"
        ELO_SIGN_FIX_COUNTER["total"] += 1
        ELO_SIGN_FIX_COUNTER["neg_to_away"] += 1
    elif d > 0 and ph < pa:
        # home優勢（diff>0）なのにaway>homeの場合は、H/Aを入れ替える
        ph, pa = pa, ph
        fix_reason = "pos_diff_home_lt_away_swap"
        ELO_SIGN_FIX_COUNTER["total"] += 1
        ELO_SIGN_FIX_COUNTER["pos_to_home"] += 1
    ph, pdw, pa = _normalize_probs(ph, pdw, pa)
    return ph, pdw, pa, fix_reason


def enforce_final_elo_sign_monotonic(df, stage_label="FINAL"):
    """Enforce H/A direction after every probability blend, before final labeling."""
    if df is None or df.empty or not ENFORCE_ELO_SIGN_MONOTONIC:
        return df
    required = {"elo_diff_for_prob", "prob_home_win", "prob_away_win"}
    if not required.issubset(df.columns):
        return df
    out = df.copy()
    diff = pd.to_numeric(out["elo_diff_for_prob"], errors="coerce")
    ph = pd.to_numeric(out["prob_home_win"], errors="coerce")
    pa = pd.to_numeric(out["prob_away_win"], errors="coerce")
    pos_fix = diff.gt(0) & ph.lt(pa)
    neg_fix = diff.lt(0) & ph.gt(pa)
    fix_mask = (pos_fix | neg_fix).fillna(False)
    for home_col, away_col in [
        ("prob_home_win", "prob_away_win"),
        ("prob_home", "prob_away"),
        ("prob_final_home", "prob_final_away"),
        ("prob_blend_home", "prob_blend_away"),
    ]:
        if home_col not in out.columns or away_col not in out.columns:
            continue
        old_home = out.loc[fix_mask, home_col].copy()
        out.loc[fix_mask, home_col] = out.loc[fix_mask, away_col].to_numpy()
        out.loc[fix_mask, away_col] = old_home.to_numpy()
    out["final_elo_sign_fix_applied"] = fix_mask
    out["final_elo_sign_fix_reason"] = np.where(
        pos_fix,
        "pos_diff_home_lt_away_swap",
        np.where(neg_fix, "neg_diff_home_gt_away_swap", ""),
    )
    final_argmax = out.apply(
        lambda row: _argmax_hda_label(
            row.get("prob_home_win"), row.get("prob_draw"), row.get("prob_away_win")
        ),
        axis=1,
    )
    out["predicted_result_main"] = final_argmax
    out["predicted_result_main_postfusion_argmax"] = final_argmax
    print(
        f"[FINAL_ELO_SIGN_FIX] stage={stage_label} total={int(fix_mask.sum())} "
        f"pos={int(pos_fix.sum())} neg={int(neg_fix.sum())}"
    )
    return out


def sanitize_prob_triplet(ph, pdw, pa, fallback=PROB_FALLBACK):
    arr = np.array([ph, pdw, pa], dtype=float)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return _normalize_probs(*fallback)
    arr = np.clip(arr, 0.0, 1.0)
    s = arr.sum()
    if s <= 0:
        return _normalize_probs(*fallback)
    return float(arr[0] / s), float(arr[1] / s), float(arr[2] / s)


def calibrate_probabilities(p_home, p_draw, p_away, league, cap_j1=J1_WIN_PROB_CAP):
    p_home, p_draw, p_away = sanitize_prob_triplet(p_home, p_draw, p_away)
    if pd.isna(league) or str(league).strip() == "":
        lg = str(LEAGUE).strip().upper()
    else:
        lg = str(league).strip().upper()
    if lg != "J1":
        return p_home, p_draw, p_away

    # J1のみ、勝ち確率（H/A）の過剰確信を抑える
    if p_home > cap_j1:
        delta = p_home - cap_j1
        p_home = cap_j1
        p_draw += delta
    if p_away > cap_j1:
        delta = p_away - cap_j1
        p_away = cap_j1
        p_draw += delta

    p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)
    # 念のため cap を再適用（数値誤差吸収）
    p_home = min(p_home, cap_j1)
    p_away = min(p_away, cap_j1)
    p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)
    return p_home, p_draw, p_away


def predict_elo_probabilities_with_home_advantage(
    home_elo,
    away_elo,
    home_advantage_diff,
    home_adv_coef=HOME_ADV_ELO_COEF,
    draw_base=ELO_DRAW_BASE,
    draw_sensitivity=ELO_DRAW_SENSITIVITY,
    draw_min=ELO_DRAW_MIN,
    draw_max=ELO_DRAW_MAX,
):
    # home_advantage_diff を Elo に反映して有効Elo差を作る
    elo_home_eff = float(home_elo) + float(home_advantage_diff) * float(home_adv_coef)
    elo_away_eff = float(away_elo)
    elo_diff = elo_home_eff - elo_away_eff

    # Elo期待値（2値）を先に計算
    p_home_two_way = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_away_two_way = 1.0 - p_home_two_way

    # Elo差が大きいほど draw を下げる簡易モデル
    p_draw = float(draw_base) - abs(float(elo_diff)) * float(draw_sensitivity)
    p_draw = max(float(draw_min), min(float(draw_max), p_draw))

    # 残りを home/away に按分し、最後に正規化
    remaining = 1.0 - p_draw
    p_home = remaining * p_home_two_way
    p_away = remaining * p_away_two_way
    return _normalize_probs(p_home, p_draw, p_away)


def build_home_away_profile_map(results_df):
    finished = results_df.dropna(subset=["home_score", "away_score"]).copy()
    if finished.empty:
        return {}, {}

    home = pd.DataFrame(
        {
            "team": finished["home_team"],
            "venue": "home",
            "gf": pd.to_numeric(finished["home_score"], errors="coerce"),
            "ga": pd.to_numeric(finished["away_score"], errors="coerce"),
        }
    )
    away = pd.DataFrame(
        {
            "team": finished["away_team"],
            "venue": "away",
            "gf": pd.to_numeric(finished["away_score"], errors="coerce"),
            "ga": pd.to_numeric(finished["home_score"], errors="coerce"),
        }
    )
    rows = pd.concat([home, away], ignore_index=True).dropna(subset=["gf", "ga"])
    rows["is_win"] = (rows["gf"] > rows["ga"]).astype(int)
    rows["is_draw"] = (rows["gf"] == rows["ga"]).astype(int)
    rows["points"] = rows["is_win"] * 3 + rows["is_draw"]

    g = rows.groupby(["team", "venue"], as_index=False).agg(matches=("team", "size"), points=("points", "sum"))
    g["ppm"] = g["points"] / g["matches"]

    home_map = g[g["venue"] == "home"].set_index("team")["ppm"].to_dict()
    away_map = g[g["venue"] == "away"].set_index("team")["ppm"].to_dict()
    return home_map, away_map


def calc_home_advantage_diff(home_team, away_team, home_ppm_map, away_ppm_map):
    if LEAGUE == "j1":
        return 0.0, False
    home_ppm = float(home_ppm_map.get(home_team, 0.0))
    away_ppm = float(away_ppm_map.get(away_team, 0.0))
    diff = home_ppm - away_ppm
    return diff, diff > 0


def load_j2_allowed_teams():
    if LEAGUE != "j2":
        return None
    results_csv_for_filter = csv_season_latest if os.path.exists(csv_season_latest) else csv_season

    # 1) 明示ファイル優先
    if os.path.exists(J2_ALLOWED_TEAMS_CSV):
        try:
            df = pd.read_csv(J2_ALLOWED_TEAMS_CSV)
            col = "team_name" if "team_name" in df.columns else df.columns[0]
            teams = set(canonical_team_name(v) for v in df[col].dropna().astype(str))
            teams = {t for t in teams if t}
            if teams:
                # 2026特別大会では実カード側の参加チーム数を下回る許可リストは不整合とみなし無効化
                if SEASON_YEAR >= 2026 and os.path.exists(results_csv_for_filter):
                    try:
                        cur = pd.read_csv(results_csv_for_filter)
                        cur_teams = set(canonical_team_name(v) for v in cur.get("home_team", pd.Series(dtype="object")).dropna().astype(str))
                        cur_teams |= set(canonical_team_name(v) for v in cur.get("away_team", pd.Series(dtype="object")).dropna().astype(str))
                        cur_teams = {t for t in cur_teams if t}
                        if cur_teams and len(teams) < len(cur_teams) and (not _env_flag("ENABLE_J2_STRICT_FILTER_FORCE", 0)):
                            print(
                                f"警告: J2許可チームCSV({len(teams)}チーム)が実カードのチーム数({len(cur_teams)}チーム)より少ないため、"
                                "2026特別大会モードとしてフィルタを無効化します。"
                            )
                            return None
                    except Exception as e:
                        print(f"警告: J2許可チーム整合性チェックに失敗しました: {e}")
                print(f"J2許可チームを {J2_ALLOWED_TEAMS_CSV} から読み込みました: {len(teams)}")
                return teams
        except Exception as e:
            print(f"警告: J2許可チームCSVの読み込みに失敗しました: {e}")

    # 2) フォールバック: 前年J2結果から推定
    fallback_csv = os.path.join(DATA_DIR, f"j2_{PREV_SEASON_YEAR}_latest_results.csv")
    if os.path.exists(fallback_csv):
        try:
            df = pd.read_csv(fallback_csv)
            teams = set(canonical_team_name(v) for v in df["home_team"].dropna().astype(str)) | set(
                canonical_team_name(v) for v in df["away_team"].dropna().astype(str)
            )
            teams = {t for t in teams if t}
            if teams:
                print(f"J2許可チームを前年データから推定しました: {len(teams)} ({fallback_csv})")
                return teams
        except Exception as e:
            print(f"警告: 前年J2結果からの許可チーム推定に失敗しました: {e}")

    print("警告: J2許可チームを特定できませんでした。全カードを予測対象にします。")
    return None


def canonical_team_name(name):
    if pd.isna(name):
        return None
    text = _normalize_team_text(name)
    # 代表表記へ寄せる
    text = TEAM_NAME_ALIAS_MAP.get(text, text)
    return text


def normalize_team_series(series):
    return series.map(canonical_team_name)


def build_unordered_pair_key(home, away):
    home_key = canonical_team_name(home)
    away_key = canonical_team_name(away)
    if not home_key or not away_key:
        return None
    a_key, b_key = sorted([str(home_key), str(away_key)])
    return f"{a_key}||{b_key}"


def load_derby_table(csv_path):
    default_cols = [
        "home_team", "away_team", "derby_name", "derby_type", "derby_intensity",
        "stall_bias", "flip_bias", "entropy_bias", "volatility_bias", "notes",
    ]
    if not csv_path or (not os.path.exists(csv_path)):
        print(f"[DERBY] csv not found: {csv_path}")
        return pd.DataFrame(columns=default_cols + ["home_canon", "away_canon", "derby_pair_key"])
    try:
        src = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"[DERBY][WARN] csv load failed: {csv_path} ({e})")
        return pd.DataFrame(columns=default_cols + ["home_canon", "away_canon", "derby_pair_key"])
    required = {"home_team", "away_team", "derby_name"}
    if src.empty or not required.issubset(src.columns):
        print(f"[DERBY][WARN] required columns missing: path={csv_path}")
        return pd.DataFrame(columns=default_cols + ["home_canon", "away_canon", "derby_pair_key"])
    out = src.copy()
    out["home_canon"] = normalize_team_series(out["home_team"])
    out["away_canon"] = normalize_team_series(out["away_team"])
    out["derby_pair_key"] = [
        build_unordered_pair_key(h, a)
        for h, a in zip(out["home_team"], out["away_team"])
    ]
    out = out.dropna(subset=["home_canon", "away_canon", "derby_pair_key"]).copy()
    dup_count = int(out.duplicated(subset=["derby_pair_key"]).sum())
    if dup_count > 0:
        print(f"[DERBY][WARN] duplicate unordered pairs detected={dup_count}; keep first")
        out = out.drop_duplicates(subset=["derby_pair_key"], keep="first")
    print(f"[DERBY] loaded={len(out)} path={csv_path}")
    return out


def merge_derby_context(df, derby_df, stage_label=""):
    if df is None or df.empty:
        return df
    out = df.copy()
    if not {"home_team", "away_team"}.issubset(out.columns):
        return out
    if derby_df is None or derby_df.empty:
        out["derby_match"] = False
        out["derby_name"] = ""
        out["derby_type"] = ""
        out["derby_intensity"] = np.nan
        out["derby_stall_bias"] = np.nan
        out["derby_flip_bias"] = np.nan
        out["derby_entropy_bias"] = np.nan
        out["derby_volatility_bias"] = np.nan
        out["derby_notes"] = ""
        print(f"[DERBY] scope={stage_label or '-'} skipped(empty master)")
        return out
    work = derby_df.copy()
    keep_cols = [
        "derby_pair_key", "derby_name", "derby_type", "derby_intensity",
        "stall_bias", "flip_bias", "entropy_bias", "volatility_bias", "notes",
    ]
    keep_cols = [c for c in keep_cols if c in work.columns]
    work = work[keep_cols].drop_duplicates(subset=["derby_pair_key"], keep="first").copy()
    out["derby_pair_key"] = [
        build_unordered_pair_key(h, a)
        for h, a in zip(out["home_team"], out["away_team"])
    ]
    out = out.merge(work, on="derby_pair_key", how="left")
    matched = out["derby_name"].notna()
    out["derby_match"] = matched.fillna(False)
    out["derby_name"] = out["derby_name"].fillna("")
    out["derby_type"] = out["derby_type"].fillna("")
    out["derby_intensity"] = pd.to_numeric(out["derby_intensity"], errors="coerce")
    out["derby_stall_bias"] = pd.to_numeric(out.get("stall_bias", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["derby_flip_bias"] = pd.to_numeric(out.get("flip_bias", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["derby_entropy_bias"] = pd.to_numeric(out.get("entropy_bias", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["derby_volatility_bias"] = pd.to_numeric(out.get("volatility_bias", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["derby_notes"] = out.get("notes", pd.Series("", index=out.index)).fillna("")
    out = out.drop(columns=["stall_bias", "flip_bias", "entropy_bias", "volatility_bias", "notes", "derby_pair_key"], errors="ignore")
    print(f"[DERBY] scope={stage_label or '-'} matched={int(matched.fillna(False).sum())}/{len(out)}")
    return out


def drop_j2_excluded_stats_columns(df):
    out = df.copy()
    if LEAGUE != "j2":
        return out
    drop_cols = []
    for base_name in J2_EXCLUDED_STATS_BASE_NAMES:
        drop_cols.append(f"stats_{base_name}_home")
        drop_cols.append(f"stats_{base_name}_away")
    existing = [c for c in drop_cols if c in out.columns]
    if existing:
        out = out.drop(columns=existing, errors="ignore")
        print(f"[INFO] J2除外指標を予測入力から削除: cols={len(existing)}")
    return out


def extract_round_number(v):
    if pd.isna(v):
        return pd.NA
    s = str(v)
    s = unicodedata.normalize("NFKC", s)
    m = re.search(r"第\s*([0-9]+)\s*節", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return pd.NA
    m2 = re.search(r"([0-9]+)", s)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return pd.NA
    return pd.NA


def _safe_numeric(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _absence_uncertainty_weeks(expected_weeks):
    """Return an automatic fade width without adding manual CSV columns."""
    weeks = _safe_float_value(expected_weeks, 0.0)
    if weeks <= 4:
        return 1.0
    if weeks <= 12:
        return 2.0
    if weeks <= 24:
        return 3.0
    return 4.0


def _absence_date_factor(match_date, start_date, expected_return_date, expected_weeks):
    match_ts = pd.to_datetime(match_date, errors="coerce")
    start_ts = pd.to_datetime(start_date, errors="coerce")
    return_ts = pd.to_datetime(expected_return_date, errors="coerce")
    if pd.isna(match_ts) or pd.isna(start_ts) or pd.isna(return_ts):
        return 0.0
    match_ts = match_ts.normalize()
    start_ts = start_ts.normalize()
    return_ts = return_ts.normalize()
    if match_ts < start_ts:
        return 0.0

    duration_weeks = _safe_float_value(expected_weeks, 0.0)
    if duration_weeks <= 0:
        duration_weeks = max(1.0, (return_ts - start_ts).days / 7.0)
    width_days = _absence_uncertainty_weeks(duration_weeks) * 7.0
    fade_start = return_ts - pd.Timedelta(days=width_days)
    fade_end = return_ts + pd.Timedelta(days=width_days)
    if match_ts <= fade_start:
        return 1.0
    if match_ts >= fade_end:
        return 0.0
    return float((fade_end - match_ts).days / max((fade_end - fade_start).days, 1))


def _build_absence_match_fixtures(match_frames):
    parts = []
    for frame in match_frames or []:
        if frame is None or frame.empty or "節" not in frame.columns:
            continue
        work = frame.copy()
        work["round_no"] = work["節"].map(extract_round_number).astype("Int64")
        work["match_date"] = pd.to_datetime(work.get("datetime"), errors="coerce").dt.normalize()
        for col in ["home_team", "away_team"]:
            if col not in work.columns:
                continue
            side = work[["round_no", "match_date", col]].rename(columns={col: "team"})
            side["team"] = normalize_team_series(side["team"])
            parts.append(side)
    if not parts:
        return pd.DataFrame(columns=["round_no", "match_date", "team"])
    fixtures = pd.concat(parts, ignore_index=True).dropna(subset=["round_no", "match_date", "team"])
    return fixtures.drop_duplicates(["round_no", "match_date", "team"])


def load_absence_impact_team_round_map(absence_csv_path, match_frames):
    if (not absence_csv_path) or (not os.path.exists(absence_csv_path)):
        print("[ABSENCE] 欠場影響CSVが見つからないためスキップします。")
        return pd.DataFrame()
    try:
        src = pd.read_csv(absence_csv_path)
    except Exception as e:
        print(f"[ABSENCE][WARN] 欠場影響CSVの読み込みに失敗: {e}")
        return pd.DataFrame()

    if src.empty:
        print("[ABSENCE] 欠場影響CSVが空のためスキップします。")
        return pd.DataFrame()

    columns = set(src.columns)
    dated_mode = {"team", "start_date", "expected_return_date"}.issubset(columns)
    legacy_mode = {"team", "round_start"}.issubset(columns)
    if not dated_mode and not legacy_mode:
        print(
            "[ABSENCE][WARN] 必須列不足: "
            "need=team + (start_date/expected_return_date or round_start) "
            f"have={columns}"
        )
        return pd.DataFrame()

    work = src.copy()
    if "season" not in work.columns:
        work["season"] = int(SEASON_YEAR)
    work["season"] = _safe_numeric(work["season"], default=int(SEASON_YEAR)).astype("Int64")
    if "round_start" not in work.columns:
        work["round_start"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
    else:
        work["round_start"] = pd.to_numeric(work["round_start"], errors="coerce").astype("Int64")
    work["expected_rounds"] = _safe_numeric(work.get("expected_rounds", 1), default=1).astype("Int64")
    work.loc[work["expected_rounds"] <= 0, "expected_rounds"] = 1
    start_values = work["start_date"] if "start_date" in work.columns else pd.Series(pd.NaT, index=work.index)
    return_values = (
        work["expected_return_date"]
        if "expected_return_date" in work.columns
        else pd.Series(pd.NaT, index=work.index)
    )
    week_values = (
        work["expected_weeks"]
        if "expected_weeks" in work.columns
        else pd.Series(float("nan"), index=work.index)
    )
    work["start_date"] = pd.to_datetime(start_values, errors="coerce", yearfirst=True)
    work["expected_return_date"] = pd.to_datetime(
        return_values, errors="coerce", yearfirst=True
    )
    work["expected_weeks"] = pd.to_numeric(week_values, errors="coerce")
    calculated_days = work["expected_weeks"].fillna(0.0) * 7.0
    calculated_return = work["start_date"] + pd.to_timedelta(calculated_days, unit="D")
    calculated_return = calculated_return.where(work["expected_weeks"].notna())
    work["expected_return_date"] = work["expected_return_date"].fillna(calculated_return)

    # 影響列が無い場合は weight から代用
    if "impact_total" not in work.columns:
        wm = _safe_numeric(work.get("weight_minutes", 0))
        wa = _safe_numeric(work.get("weight_attack", 0))
        wd = _safe_numeric(work.get("weight_defense", 0))
        work["impact_total"] = 0.6 * wm + 0.2 * wa + 0.2 * wd
    if "impact_attack" not in work.columns:
        work["impact_attack"] = _safe_numeric(work.get("weight_attack", 0))
    if "impact_defense" not in work.columns:
        work["impact_defense"] = _safe_numeric(work.get("weight_defense", 0))

    work["impact_total"] = _safe_numeric(work["impact_total"])
    work["impact_attack"] = _safe_numeric(work["impact_attack"])
    work["impact_defense"] = _safe_numeric(work["impact_defense"])
    if "availability" in work.columns:
        # returned は影響0扱い
        av = work["availability"].astype(str).str.lower().str.strip()
        work.loc[av.isin(["returned", "return", "available", "fit"]), ["impact_total", "impact_attack", "impact_defense"]] = 0.0

    work["team_name"] = normalize_team_series(work["team"].astype(str))
    work["_merge_team_name"] = normalize_team_series(work["team_name"])

    fixtures = _build_absence_match_fixtures(match_frames)
    if fixtures.empty:
        print("[ABSENCE][WARN] 対象試合日時が特定できないため欠場影響を無効化します。")
        return pd.DataFrame()

    expanded_rows = []
    for _, r in work.iterrows():
        if pd.isna(r["season"]) or pd.isna(r["_merge_team_name"]):
            continue
        team_fixtures = fixtures[fixtures["team"].eq(r["_merge_team_name"])]
        for _, fixture in team_fixtures.iterrows():
            if pd.notna(r["start_date"]) and pd.notna(r["expected_return_date"]):
                factor = _absence_date_factor(
                    fixture["match_date"], r["start_date"], r["expected_return_date"], r["expected_weeks"]
                )
            elif pd.notna(r["round_start"]):
                start_r = int(r["round_start"])
                span = int(r["expected_rounds"]) if pd.notna(r["expected_rounds"]) else 1
                end_r = start_r + max(span, 1) - 1
                factor = 1.0 if start_r <= int(fixture["round_no"]) <= end_r else 0.0
            else:
                factor = 0.0
            if factor <= 0.0:
                continue
            expanded_rows.append(
                {
                    "season": int(r["season"]),
                    "_merge_team_name": r["_merge_team_name"],
                    "round_no": int(fixture["round_no"]),
                    "match_date": fixture["match_date"],
                    "absence_impact_total": float(r["impact_total"]) * factor,
                    "absence_impact_attack": float(r["impact_attack"]) * factor,
                    "absence_impact_defense": float(r["impact_defense"]) * factor,
                    "absence_players_count": 1,
                }
            )

    if not expanded_rows:
        print("[ABSENCE] 対象節に有効な欠場行がありません。")
        return pd.DataFrame()

    out = pd.DataFrame(expanded_rows)
    out = (
        out.groupby(["season", "_merge_team_name", "round_no", "match_date"], as_index=False)
        .agg(
            absence_impact_total=("absence_impact_total", "sum"),
            absence_impact_attack=("absence_impact_attack", "sum"),
            absence_impact_defense=("absence_impact_defense", "sum"),
            absence_players_count=("absence_players_count", "sum"),
        )
    )
    print(
        f"[ABSENCE] 取り込み完了: mode={'date' if dated_mode else 'round'} "
        f"src_rows={len(work)}, expanded={len(expanded_rows)}, team_match_rows={len(out)}"
    )
    return out


def merge_absence_impacts(df, absence_map_df, stage_label):
    if absence_map_df is None or absence_map_df.empty:
        out = df.copy()
        for c in [
            "absence_impact_total_home", "absence_impact_attack_home", "absence_impact_defense_home", "absence_players_count_home",
            "absence_impact_total_away", "absence_impact_attack_away", "absence_impact_defense_away", "absence_players_count_away",
        ]:
            if c not in out.columns:
                out[c] = 0.0
        return out

    out = df.copy()
    out["_round_no"] = out["節"].map(extract_round_number).astype("Int64")
    out["_match_date"] = pd.to_datetime(out.get("datetime"), errors="coerce").dt.normalize()
    out["_season"] = int(SEASON_YEAR)
    out["_merge_home_team"] = normalize_team_series(out["home_team"])
    out["_merge_away_team"] = normalize_team_series(out["away_team"])

    home_map = absence_map_df.rename(
        columns={
            "absence_impact_total": "absence_impact_total_home",
            "absence_impact_attack": "absence_impact_attack_home",
            "absence_impact_defense": "absence_impact_defense_home",
            "absence_players_count": "absence_players_count_home",
        }
    )
    away_map = absence_map_df.rename(
        columns={
            "absence_impact_total": "absence_impact_total_away",
            "absence_impact_attack": "absence_impact_attack_away",
            "absence_impact_defense": "absence_impact_defense_away",
            "absence_players_count": "absence_players_count_away",
        }
    )

    out = audited_left_merge(
        out,
        home_map[
            ["season", "_merge_team_name", "round_no", "match_date", "absence_impact_total_home", "absence_impact_attack_home", "absence_impact_defense_home", "absence_players_count_home"]
        ],
        stage=f"{stage_label}_home",
        left_on=["_season", "_merge_home_team", "_round_no", "_match_date"],
        right_on=["season", "_merge_team_name", "round_no", "match_date"],
        validate="many_to_one",
    )
    out = out.drop(columns=["season", "_merge_team_name", "round_no", "match_date"], errors="ignore")

    out = audited_left_merge(
        out,
        away_map[
            ["season", "_merge_team_name", "round_no", "match_date", "absence_impact_total_away", "absence_impact_attack_away", "absence_impact_defense_away", "absence_players_count_away"]
        ],
        stage=f"{stage_label}_away",
        left_on=["_season", "_merge_away_team", "_round_no", "_match_date"],
        right_on=["season", "_merge_team_name", "round_no", "match_date"],
        validate="many_to_one",
    )
    out = out.drop(columns=["season", "_merge_team_name", "round_no", "match_date"], errors="ignore")

    for c in [
        "absence_impact_total_home", "absence_impact_attack_home", "absence_impact_defense_home", "absence_players_count_home",
        "absence_impact_total_away", "absence_impact_attack_away", "absence_impact_defense_away", "absence_players_count_away",
    ]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out = out.drop(columns=["_merge_home_team", "_merge_away_team", "_round_no", "_match_date", "_season"], errors="ignore")
    return out


def normalize_travel_distance_matrix(df):
    if df.empty:
        return df
    out = df.copy()
    out.index = [canonical_team_name(x) for x in out.index]
    out.columns = [canonical_team_name(x) for x in out.columns]
    out = out.groupby(level=0).mean(numeric_only=True)
    out = out.T.groupby(level=0).mean(numeric_only=True).T
    return out


def _weather_match_fallback_key(match_id):
    """Build a match key that ignores kickoff HHMM but keeps date and matchup."""
    text = str(match_id or "").strip()
    match = re.match(r"^([^_]+)_(\d{4})_(\d{4})\d{4}_(.+)$", text)
    if not match:
        return ""
    return f"{match.group(1)}_{match.group(2)}_{match.group(3)}_{match.group(4)}"


def _add_weather_match_id_aliases(df, weather_cache_df):
    """Alias weather rows to schedule/result IDs when only kickoff HHMM differs."""
    if df.empty or weather_cache_df.empty or "match_id" not in df.columns or "match_id" not in weather_cache_df.columns:
        return weather_cache_df, 0

    weather = weather_cache_df.copy()
    exact_ids = set(weather["match_id"].dropna().astype(str).str.strip())
    left = df[["match_id"]].copy()
    left["match_id"] = left["match_id"].astype(str).str.strip()
    left = left[~left["match_id"].isin(exact_ids)].drop_duplicates("match_id")
    left["__weather_fallback_key"] = left["match_id"].map(_weather_match_fallback_key)
    left = left[left["__weather_fallback_key"].ne("")]

    right = weather.reset_index(names="__weather_source_index")
    right["__weather_fallback_key"] = right["match_id"].map(_weather_match_fallback_key)
    right = right[right["__weather_fallback_key"].ne("")]
    if "__weather_asof" in right.columns:
        right = right.sort_values(["__weather_fallback_key", "__weather_asof"], kind="mergesort")
    right = right.drop_duplicates("__weather_fallback_key", keep="last")

    aliases = left.merge(
        right[["__weather_fallback_key", "__weather_source_index"]],
        on="__weather_fallback_key",
        how="inner",
        validate="many_to_one",
    )
    if aliases.empty:
        return weather, 0

    alias_rows = weather.loc[aliases["__weather_source_index"].to_numpy()].copy().reset_index(drop=True)
    alias_rows["match_id"] = aliases["match_id"].to_numpy()
    weather = pd.concat([weather, alias_rows], ignore_index=True, sort=False)
    return weather, len(alias_rows)


def merge_weather_cache(df, weather_cache_df, stage):
    _ensure_merge_qc_dir()
    weather_cache_df, fallback_matches = _add_weather_match_id_aliases(df, weather_cache_df)
    if fallback_matches:
        print(
            f"[WEATHER_MATCH_FALLBACK] stage={stage} matched={fallback_matches} "
            "key=league+season+date+home+away (kickoff_hhmm ignored)"
        )
    _log_df_key_health(stage, "left_before", df, ["match_id"])
    _log_df_key_health(stage, "right", weather_cache_df, ["match_id"])

    merged = pd.merge(
        df,
        weather_cache_df,
        how="left",
        on="match_id",
        validate="many_to_one",
        indicator="_merge_weather",
        suffixes=("", "_weather"),
    )
    counts = merged["_merge_weather"].value_counts(dropna=False).to_dict()
    print(f"[MERGE_QC] {stage} indicator={counts}")

    left_only = merged[merged["_merge_weather"] == "left_only"].copy()
    if not left_only.empty:
        left_only_path = os.path.join(MERGE_QC_DIR, f"{stage}_left_only.csv")
        left_only.to_csv(left_only_path, index=False, encoding="utf-8-sig")
        print(f"[MERGE_QC][WARN] {stage}: left_only={len(left_only)} -> {left_only_path}")
        show_cols = [c for c in ["match_id", "home_team", "away_team"] if c in left_only.columns]
        if show_cols:
            print(left_only[show_cols].head(10).to_string(index=False))
    else:
        print(f"[MERGE_QC] {stage}: left_only=0")

    weather_cols = ["is_rain", "is_heavy_rain", "is_strong_wind"]
    for col in weather_cols:
        if col not in merged:
            merged[col] = pd.NA

    merged["weather_missing"] = (merged["_merge_weather"] == "left_only") | merged[weather_cols].isna().any(axis=1)
    if "weather_fetch_ok" in merged:
        merged["weather_missing"] |= pd.to_numeric(merged["weather_fetch_ok"], errors="coerce").eq(0)
    if "weather_quality_status" in merged:
        merged["weather_missing"] |= merged["weather_quality_status"].isin(["partial_match_window", "outside_match_window"])
    acquired = pd.to_datetime(merged.get("last_updated_at", pd.Series(pd.NaT, index=merged.index)), errors="coerce", utc=True)
    now_utc = pd.Timestamp.now(tz="UTC")
    merged["weather_age_hours"] = (now_utc - acquired).dt.total_seconds() / 3600
    kickoff_local = pd.to_datetime(merged["datetime"], errors="coerce")
    if kickoff_local.dt.tz is None:
        kickoff_local = kickoff_local.dt.tz_localize("Asia/Tokyo")
    upcoming_weather = kickoff_local.gt(now_utc)
    merged["weather_freshness_status"] = np.where(
        acquired.isna(), "acquisition_unknown",
        np.where(merged["weather_age_hours"].between(0, float(os.environ.get("WEATHER_MAX_FORECAST_AGE_HOURS", "24"))), "fresh", "stale")
    )
    # A failed refresh can leave an old CSV in place: do not silently apply it
    # to a future fixture. Historical snapshots retain their recorded evidence.
    merged["weather_missing"] |= upcoming_weather & merged["weather_freshness_status"].ne("fresh")
    for col in weather_cols:
        merged[col] = merged[col].map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"} if pd.notna(v) else False)
        merged.loc[merged["weather_missing"], col] = False
    # False補完は「晴天」の断定ではなく、欠損時に補正を加えない中立処理。
    merged["weather_adjustment_status"] = np.where(
        merged["weather_missing"], "unknown_no_adjustment", "available_flags"
    )

    # 数値天候の欠損は補完（欠損事実は weather_missing で保持）
    for col, default_val in [("temperature", WEATHER_DEFAULT_TEMPERATURE), ("wind_speed", WEATHER_DEFAULT_WIND_SPEED)]:
        if col in merged.columns:
            num = pd.to_numeric(merged[col], errors="coerce")
            med = num.median(skipna=True)
            fill_val = float(med) if pd.notna(med) else float(default_val)
            merged[col] = num.fillna(fill_val)

    return merged.drop(columns=["_merge_weather"], errors="ignore")


def normalize_weather_cache_columns(df):
    """新旧の天候CSV列名を予測側の期待列へ正規化する。"""
    out = df.copy()
    # キックオフ時刻 / スタジアム名
    if "datetime" not in out.columns and "kickoff_jst" in out.columns:
        out["datetime"] = out["kickoff_jst"]
    if "stadium" not in out.columns and "stadium_name" in out.columns:
        out["stadium"] = out["stadium_name"]

    # 気温 / 風速
    if "temperature" not in out.columns and "temp_kickoff" in out.columns:
        out["temperature"] = out["temp_kickoff"]
    if "wind_speed" not in out.columns and "wind_kickoff" in out.columns:
        out["wind_speed"] = out["wind_kickoff"]
    if "wind_speed" in out.columns:
        wind_kmh = pd.to_numeric(out["wind_speed"], errors="coerce")
        # Open-Meteoの既定値と保存済みスナップショットはkm/h。旧8km/h判定を読み込み時に是正する。
        units = out.get("wind_speed_unit", pd.Series("km/h", index=out.index)).fillna("km/h")
        factors = units.map({"km/h": 1.0, "kmh": 1.0, "m/s": 3.6, "mph": 1.609344, "kn": 1.852})
        wind_kmh *= factors
        out["wind_speed"] = wind_kmh
        out["is_strong_wind"] = wind_kmh.ge(STRONG_WIND_THRESHOLD_KMH).where(wind_kmh.notna(), pd.NA)
        out["wind_speed_unit"] = WIND_SPEED_UNIT

    # 取得時刻（なければ空列を作る）
    if "last_updated_at" not in out.columns:
        out["last_updated_at"] = pd.NA

    return out


def _extract_weather_asof_key(path_str):
    name = os.path.basename(str(path_str))
    m = re.search(r"_asof_(\d{8})\.csv$", name)
    if m:
        return m.group(1)
    return "00000000"


def load_weather_union_dataframe(primary_weather_csv):
    """天候データを複数ソースから束ね、match_id単位で最新を採用する。"""
    paths = []
    if primary_weather_csv and os.path.exists(primary_weather_csv):
        paths.append(primary_weather_csv)

    # 補助: 共有キャッシュ（過去分を含みやすい）
    global_cache = os.path.join(DATA_DIR, "weather_cache.csv")
    if os.path.exists(global_cache):
        paths.append(global_cache)

    # 補助: 同リーグ同年の全スナップショット
    if os.path.isdir(WEATHER_SNAPSHOT_DIR):
        pat = re.compile(rf"^weather_features_{LEAGUE}_{SEASON_YEAR}_asof_(\d{{8}})\.csv$")
        for fn in os.listdir(WEATHER_SNAPSHOT_DIR):
            if pat.match(fn):
                paths.append(os.path.join(WEATHER_SNAPSHOT_DIR, fn))

    # 順序を安定化（重複除去）
    seen = set()
    ordered_paths = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        ordered_paths.append(p)

    frames = []
    for p in ordered_paths:
        try:
            wdf = pd.read_csv(p)
        except Exception as e:
            print(f"[WEATHER][WARN] 読み込み失敗をスキップ: {p} err={e}")
            continue
        if wdf.empty:
            continue
        wdf = normalize_weather_cache_columns(wdf)
        if "match_id" not in wdf.columns:
            continue
        wdf["match_id"] = wdf["match_id"].astype(str).str.strip()
        wdf["__weather_asof"] = _extract_weather_asof_key(p)
        wdf["__weather_source"] = os.path.basename(p)
        frames.append(wdf)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["__weather_asof"] = merged["__weather_asof"].fillna("00000000").astype(str)
    merged = merged.sort_values(["match_id", "__weather_asof"], kind="mergesort")
    merged = merged.drop_duplicates(subset=["match_id"], keep="last")

    keep_cols = [
        c
        for c in [
            "match_id",
            "datetime",
            "stadium",
            "is_rain",
            "is_heavy_rain",
            "is_strong_wind",
            "temperature",
            "wind_speed",
            "wind_speed_unit",
            "last_updated_at",
            "weather_fetch_ok", "weather_quality_status", "weather_data_kind",
            "weather_window_hours", "precip_kickoff", "precip_max_match",
            "precip_sum_match", "wind_max_match", "adverse_weather_during_match",
            "__weather_asof",
            "__weather_source",
        ]
        if c in merged.columns
    ]
    out = merged[keep_cols].copy()
    print(
        f"[WEATHER] union_sources={len(ordered_paths)} union_rows={len(merged)} "
        f"primary={os.path.basename(primary_weather_csv) if primary_weather_csv else '-'}"
    )
    return out


def load_allowed_teams():
    env_allowed_csv = os.environ.get("ALLOWED_TEAMS_CSV")
    csv_candidates = []
    if env_allowed_csv:
        csv_candidates.append(env_allowed_csv)
    csv_candidates.append(os.path.join(MANUAL_DIR, f"{LEAGUE}_allowed_teams_{SEASON_YEAR}.csv"))
    csv_candidates.append(os.path.join(MANUAL_DIR, f"{LEAGUE}_allowed_teams.csv"))

    for allowed_csv in csv_candidates:
        if not allowed_csv or not os.path.exists(allowed_csv):
            continue
        try:
            df = pd.read_csv(allowed_csv)
            col = "team_name" if "team_name" in df.columns else df.columns[0]
            teams = set(canonical_team_name(v) for v in df[col].dropna().astype(str))
            teams = {t for t in teams if t}
            if teams:
                print(f"{LEAGUE.upper()}許可チームを {allowed_csv} から読み込みました: {len(teams)}")
                return teams
        except Exception as e:
            print(f"警告: 許可チームCSVの読み込みに失敗しました ({allowed_csv}): {e}")

    # J1は team_master_stats を優先して許可チームを推定する
    if LEAGUE == "j1":
        def estimate_top_teams_from_results(results_csv, top_n=3):
            try:
                df = pd.read_csv(results_csv)
            except Exception:
                return set()
            required = {"home_team", "away_team", "home_score", "away_score"}
            if not required.issubset(df.columns):
                return set()
            df = df.dropna(subset=["home_score", "away_score"])
            if df.empty:
                return set()
            teams = sorted(set(df["home_team"].astype(str)) | set(df["away_team"].astype(str)))
            pts = {t: 0 for t in teams}
            gd = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            for _, r in df.iterrows():
                h = str(r["home_team"]).strip()
                a = str(r["away_team"]).strip()
                hs = int(r["home_score"])
                aw = int(r["away_score"])
                gf[h] += hs
                gf[a] += aw
                gd[h] += hs - aw
                gd[a] += aw - hs
                if hs > aw:
                    pts[h] += 3
                elif hs < aw:
                    pts[a] += 3
                else:
                    pts[h] += 1
                    pts[a] += 1
            rank = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
            return set(canonical_team_name(t) for t in rank[:top_n])

        stats_candidates = [team_master_stats_csv, os.path.join(DATA_DIR, "team_master_stats.csv")]
        for stats_csv in stats_candidates:
            if not os.path.exists(stats_csv):
                continue
            try:
                stats_df = pd.read_csv(stats_csv)
                if "team_name" not in stats_df.columns:
                    continue
                teams = set(canonical_team_name(v) for v in stats_df["team_name"].dropna().astype(str))
                teams = {t for t in teams if t}
                if teams:
                    j2_prev_csv = os.path.join(DATA_DIR, f"j2_{PREV_SEASON_YEAR}_latest_results.csv")
                    promoted = estimate_top_teams_from_results(j2_prev_csv, top_n=3)
                    if promoted:
                        teams |= promoted
                        print(
                            f"J1許可チームを team_master_stats + 前年J2上位から推定しました: "
                            f"{len(teams)} ({stats_csv})"
                        )
                    else:
                        print(f"J1許可チームを team_master_stats から推定しました: {len(teams)} ({stats_csv})")
                    return teams
            except Exception as e:
                print(f"警告: team_master_stats からの許可チーム推定に失敗しました ({stats_csv}): {e}")

    if LEAGUE == "j2":
        # 2026特別大会（J2/J3混在）ではリーグ外除外を無効化し、日程側の定義に従う
        if SEASON_YEAR >= 2026 and (not _env_flag("ENABLE_J2_STRICT_FILTER", 0)):
            print("J2許可チームフィルタを無効化します（2026特別大会モード）。")
            return None
        return load_j2_allowed_teams()

    print(f"警告: {LEAGUE.upper()}許可チームを特定できませんでした。全カードを予測対象にします。")
    return None

# Elo更新（簡易式）
def update_elo(elo_home, elo_away, result, k=20):
    # predict_eloはここでは使わないが、Elo更新のために仮のp_homeを計算
    elo_diff = elo_home + ELO_UPDATE_HOME_ADVANTAGE - elo_away
    p_home = 1 / (1 + 10 ** (-elo_diff / 400))
    s_home = {"H": 1, "D": 0.5, "A": 0}[result]
    delta = k * (s_home - p_home)
    return elo_home + delta, elo_away - delta


def sort_results_for_elo(df):
    if "datetime" in df.columns:
        out = df.copy()
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        sort_cols = ["datetime"]
        if "match_id" in out.columns:
            sort_cols.append("match_id")
        return out.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return df.reset_index(drop=True)


def compute_elo_map_from_results(results_df, base_elo_map=None):
    elo_map = {} if base_elo_map is None else {k: float(v) for k, v in base_elo_map.items()}
    ordered = sort_results_for_elo(results_df)
    for _, row in ordered.iterrows():
        home = row.get("home_team")
        away = row.get("away_team")
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if pd.isna(home) or pd.isna(away):
            continue
        home = str(home).strip()
        away = str(away).strip()
        if home not in elo_map:
            elo_map[home] = INITIAL_ELO
        if away not in elo_map:
            elo_map[away] = INITIAL_ELO
        result = get_result(hs, as_)
        if result:
            elo_map[home], elo_map[away] = update_elo(elo_map[home], elo_map[away], result)
    return elo_map


def load_or_build_prev_final_elo(df_prev_results):
    if FORCE_RECALC:
        print(f"[FORCE] 前年最終ELOキャッシュを再利用しません: {prev_final_elo_csv}")
    elif os.path.exists(prev_final_elo_csv):
        try:
            prev_elo_df = pd.read_csv(prev_final_elo_csv)
            if {"team", "elo"}.issubset(prev_elo_df.columns):
                elo_map = {
                    str(r["team"]).strip(): float(r["elo"])
                    for _, r in prev_elo_df[["team", "elo"]].dropna(subset=["team"]).iterrows()
                }
                if elo_map:
                    print(f"前年最終ELOを読み込みました: {prev_final_elo_csv} ({len(elo_map)}チーム)")
                    return elo_map
        except Exception as e:
            print(f"警告: 前年最終ELOの読み込みに失敗しました: {e}")

    if df_prev_results.empty:
        print("前年最終ELOは未作成（前年結果データなし）。")
        return {}

    elo_map = compute_elo_map_from_results(df_prev_results)
    try:
        out_df = pd.DataFrame(
            [{"team": team, "elo": round(score, 6)} for team, score in sorted(elo_map.items(), key=lambda x: x[0])]
        )
        out_df.to_csv(prev_final_elo_csv, index=False, encoding="utf-8-sig")
        print(f"前年最終ELOを作成しました: {prev_final_elo_csv} ({len(out_df)}チーム)")
    except Exception as e:
        print(f"警告: 前年最終ELOの保存に失敗しました: {e}")
    return elo_map


def _ensure_merge_qc_dir():
    os.makedirs(MERGE_QC_DIR, exist_ok=True)


def _to_key_list(on=None, left_on=None):
    if on is not None:
        return [on] if isinstance(on, str) else list(on)
    if left_on is not None:
        return [left_on] if isinstance(left_on, str) else list(left_on)
    return []


def _key_stats(df, keys):
    if not keys or any(k not in df.columns for k in keys):
        return {"rows": len(df), "unique_keys": None, "duplicate_keys": None}
    subset = df[keys]
    return {
        "rows": len(df),
        "unique_keys": int(subset.drop_duplicates().shape[0]),
        "duplicate_keys": int(subset.duplicated().sum()),
    }


def _log_df_key_health(stage, side, df, keys):
    stats = _key_stats(df, keys)
    if stats["unique_keys"] is None:
        print(f"[MERGE_QC] {stage} {side}: rows={stats['rows']} keys={keys} (missing key columns)")
        return
    print(
        f"[MERGE_QC] {stage} {side}: rows={stats['rows']} "
        f"unique_keys={stats['unique_keys']} duplicate_keys={stats['duplicate_keys']} keys={keys}"
    )


def audited_left_merge(
    left_df,
    right_df,
    stage,
    on=None,
    left_on=None,
    right_on=None,
    validate=None,
    suffixes=("", "_r"),
):
    _ensure_merge_qc_dir()
    left_keys = _to_key_list(on=on, left_on=left_on)
    right_keys = _to_key_list(on=on, left_on=right_on)

    _log_df_key_health(stage, "left_before", left_df, left_keys)
    _log_df_key_health(stage, "right", right_df, right_keys)
    before_rows = len(left_df)

    missing_left_keys = [k for k in left_keys if k not in left_df.columns]
    missing_right_keys = [k for k in right_keys if k not in right_df.columns]
    if missing_left_keys or missing_right_keys:
        print(
            f"[MERGE_QC][WARN] {stage}: mergeキー不足のため結合をスキップ "
            f"(missing_left={missing_left_keys}, missing_right={missing_right_keys})"
        )
        fallback = left_df.copy()
        for c in right_df.columns:
            if c not in fallback.columns and c not in right_keys:
                fallback[c] = pd.NA
        return fallback

    indicator_col = f"_merge_{stage}"
    merged = pd.merge(
        left_df,
        right_df,
        how="left",
        on=on,
        left_on=left_on,
        right_on=right_on,
        validate=validate,
        indicator=indicator_col,
        suffixes=suffixes,
    )

    after_rows = len(merged)
    counts = merged[indicator_col].value_counts(dropna=False).to_dict()
    print(
        f"[MERGE_QC] {stage} result: rows_before={before_rows} rows_after={after_rows} "
        f"delta={after_rows - before_rows} validate={validate} indicator={counts}"
    )
    if after_rows > before_rows:
        print(f"[MERGE_QC][WARN] {stage}: row増殖を検知しました（右側キー重複の可能性）")
    if after_rows < before_rows:
        print(f"[MERGE_QC][WARN] {stage}: row減少を検知しました（merge条件を要確認）")

    left_only = merged[merged[indicator_col] == "left_only"].copy()
    if not left_only.empty:
        print(f"[MERGE_QC][WARN] {stage}: left_only={len(left_only)}")
        show_cols = [c for c in ["match_id", "home_team", "away_team"] if c in left_only.columns]
        show_cols += [c for c in left_keys if c in left_only.columns and c not in show_cols]
        show_cols += [c for c in right_keys if c in left_only.columns and c not in show_cols]
        preview_cols = show_cols[:10] if show_cols else left_only.columns[:10].tolist()
        print(left_only[preview_cols].head(10).to_string(index=False))
        left_only_path = os.path.join(MERGE_QC_DIR, f"{stage}_left_only.csv")
        left_only.to_csv(left_only_path, index=False, encoding="utf-8-sig")
        print(f"[MERGE_QC] {stage}: left_only CSV保存 -> {left_only_path}")
    else:
        print(f"[MERGE_QC] {stage}: left_only=0")

    return merged.drop(columns=[indicator_col], errors="ignore")


def report_missing_rates(df, stage, threshold=MISSING_WARN_THRESHOLD):
    if len(df) == 0:
        print(f"[MISSING_QC] {stage}: rows=0 のため欠損率をスキップ")
        return
    target_groups = {
        "weather": ["is_rain", "is_heavy_rain", "is_strong_wind", "weather_missing"],
        "stats": ["stats_ゴール期待値_home", "stats_ゴール期待値_away", "stats_home_missing", "stats_away_missing"],
        "management": [
            "management_recent_injuries_suspensions_count_home",
            "management_recent_injuries_suspensions_count",
            "management_recent_injuries_suspensions_count_away",
            "management_missing",
        ],
        "quality": ["data_quality_warn"],
    }
    existing = []
    for cols in target_groups.values():
        existing.extend([c for c in cols if c in df.columns and c not in existing])
    if not existing:
        print(f"[MISSING_QC] {stage}: 対象カラムなし")
        return
    for col in existing:
        miss_rate = float(df[col].isna().mean())
        level = "WARN" if miss_rate > threshold else "INFO"
        print(f"[MISSING_QC][{level}] {stage} {col}: missing_rate={miss_rate:.2%}")


def export_team_name_diff(matches_df, stats_csv_path):
    _ensure_merge_qc_dir()
    try:
        stats_df = pd.read_csv(stats_csv_path)
    except Exception as e:
        print(f"[MERGE_QC][WARN] チーム名差分の算出失敗: {e}")
        return
    if "team_name" not in stats_df.columns:
        print("[MERGE_QC][WARN] stats側に team_name 列がないため差分をスキップ")
        return

    match_teams = set(normalize_team_series(matches_df["home_team"].dropna())) | set(
        normalize_team_series(matches_df["away_team"].dropna())
    )
    stats_teams = set(normalize_team_series(stats_df["team_name"].dropna()))
    only_matches = sorted(match_teams - stats_teams)

    diff_df = pd.DataFrame({"team_name_in_matches_only": only_matches})
    diff_path = os.path.join(MERGE_QC_DIR, "team_name_diff_matches_vs_stats.csv")
    diff_df.to_csv(diff_path, index=False, encoding="utf-8-sig")
    print(
        f"[MERGE_QC] チーム名差分: matches_only={len(only_matches)} "
        f"CSV保存 -> {diff_path}"
    )

    only_matches_norm = sorted(only_matches)
    diff_norm_df = pd.DataFrame({"team_name_in_matches_only_after_canonical": only_matches_norm})
    diff_norm_path = os.path.join(MERGE_QC_DIR, "team_name_diff_matches_vs_stats_after_canonical.csv")
    diff_norm_df.to_csv(diff_norm_path, index=False, encoding="utf-8-sig")
    print(
        f"[MERGE_QC] 正規化後チーム名差分: matches_only={len(only_matches_norm)} "
        f"CSV保存 -> {diff_norm_path}"
    )


def _pick_first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _pick_first_non_na_value(row, candidates):
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None


def add_data_quality_flags(df):
    out = df.copy()

    if "weather_missing" not in out.columns:
        weather_cols = [c for c in ["is_rain", "is_heavy_rain", "is_strong_wind"] if c in out.columns]
        if weather_cols:
            out["weather_missing"] = out[weather_cols].isna().all(axis=1)
        else:
            out["weather_missing"] = True

    home_stats_col = _pick_first_existing(out, ["stats_ゴール期待値_home", "stats_ゴール期待値"])
    away_stats_col = _pick_first_existing(out, ["stats_ゴール期待値_away"])
    out["stats_home_missing"] = out[home_stats_col].isna() if home_stats_col else True
    out["stats_away_missing"] = out[away_stats_col].isna() if away_stats_col else True

    mgmt_home_col = _pick_first_existing(
        out,
        ["management_recent_injuries_suspensions_count_home", "management_recent_injuries_suspensions_count"],
    )
    mgmt_away_col = _pick_first_existing(out, ["management_recent_injuries_suspensions_count_away"])
    if mgmt_home_col and mgmt_away_col:
        out["management_missing"] = out[[mgmt_home_col, mgmt_away_col]].isna().any(axis=1)
    elif mgmt_home_col:
        out["management_missing"] = out[mgmt_home_col].isna()
    elif mgmt_away_col:
        out["management_missing"] = out[mgmt_away_col].isna()
    else:
        out["management_missing"] = True

    out["data_quality_warn"] = out[
        ["weather_missing", "stats_home_missing", "stats_away_missing", "management_missing"]
    ].any(axis=1)
    return out


def fill_management_default_values(df):
    """management値の欠損は0埋め（欠損事実は management_missing で保持）。"""
    out = df.copy()
    target_cols = [
        "management_recent_injuries_suspensions_count_home",
        "management_recent_injuries_suspensions_count_away",
        "management_recent_injuries_suspensions_count",
    ]
    for c in target_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def drop_internal_output_columns(df):
    out = df.copy()
    drop_exact = {
        "team_name",
        "_merge_team_name",
        "datetime_r",
        "datetime_x",
        "datetime_y",
        "_merge_home_team",
        "_merge_away_team",
    }
    drop_cols = [c for c in out.columns if c in drop_exact or c.startswith("_merge_")]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    return out


def load_acl_events(path):
    required_cols = ["match_date", "team", "fatigue_grade"]
    if not path or not os.path.exists(path):
        print(f"[ACL] schedule csv not found: {path}")
        return pd.DataFrame(columns=required_cols)

    out = normalize_acl_schedule(path)
    out["_team_key"] = normalize_team_series(out["team"])
    out = out.sort_values(["_team_key", "match_date"], kind="mergesort").reset_index(drop=True)
    return out


def build_acl_event_index(df_acl):
    index = {}
    if df_acl is None or df_acl.empty:
        return index
    for team_key, group in df_acl.groupby("_team_key", sort=False):
        dates = group["match_date"].tolist()
        rows = group.to_dict("records")
        index[str(team_key)] = {"dates": dates, "rows": rows}
    return index


def get_latest_acl_event(team, target_date, acl_event_index):
    if pd.isna(target_date):
        return None
    team_key = canonical_team_name(team)
    payload = acl_event_index.get(team_key)
    if not payload:
        return None
    dates = payload["dates"]
    pos = bisect_left(dates, pd.Timestamp(target_date))
    if pos <= 0:
        return None
    return payload["rows"][pos - 1]


def get_acl_fatigue(team, target_date, acl_event_index, effective_days):
    event = get_latest_acl_event(team, target_date, acl_event_index)
    if not event:
        return {
            "acl_fatigue": 0.0,
            "acl_last_date": pd.NaT,
            "acl_days_since": pd.NA,
            "acl_travel_type": "",
        }

    target_ts = pd.Timestamp(target_date)
    event_ts = pd.Timestamp(event["match_date"])
    days_since = (target_ts.normalize() - event_ts.normalize()).days
    if days_since > int(ACL_SECOND_WINDOW_DAYS):
        return {
            "acl_fatigue": 0.0,
            "acl_last_date": pd.NaT,
            "acl_days_since": pd.NA,
            "acl_travel_type": "",
        }
    fatigue_value = 0.0
    if 1 <= days_since <= int(effective_days):
        fatigue_value = float(event.get("fatigue_grade", 0.0) or 0.0)
        fatigue_value *= ACL_FATIGUE_MULTIPLIER
        if days_since <= 3:
            fatigue_value *= (1.0 + ACL_FATIGUE_SHORT_REST_BONUS)
        travel_type = str(event.get("travel_type", "") or "").strip().lower()
        if travel_type in ACL_AWAY_TRAVEL_TYPES:
            fatigue_value *= (1.0 + ACL_FATIGUE_TRAVEL_AWAY_BONUS)
    elif int(effective_days) < days_since <= int(ACL_SECOND_WINDOW_DAYS):
        fatigue_value = float(event.get("fatigue_grade", 0.0) or 0.0)
        fatigue_value *= ACL_FATIGUE_MULTIPLIER
        travel_type = str(event.get("travel_type", "") or "").strip().lower()
        if travel_type in ACL_AWAY_TRAVEL_TYPES:
            fatigue_value *= (1.0 + ACL_FATIGUE_TRAVEL_AWAY_BONUS)
        fatigue_value *= ACL_SECOND_WINDOW_DECAY

    return {
        "acl_fatigue": fatigue_value,
        "acl_last_date": event_ts,
        "acl_days_since": days_since,
        "acl_travel_type": event.get("travel_type", "") or "",
    }


def attach_acl_fatigue_to_matches(df_matches, acl_event_index, effective_days, stage_label="matches"):
    out = df_matches.copy()
    if "datetime" not in out.columns:
        out["home_acl_fatigue"] = 0.0
        out["away_acl_fatigue"] = 0.0
        out["home_total_fatigue_score"] = pd.to_numeric(out.get("home_fatigue_score"), errors="coerce").fillna(0.0)
        out["away_total_fatigue_score"] = pd.to_numeric(out.get("away_fatigue_score"), errors="coerce").fillna(0.0)
        return out

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    home_infos = out.apply(
        lambda row: get_acl_fatigue(row.get("home_team"), row.get("datetime"), acl_event_index, effective_days),
        axis=1,
    )
    away_infos = out.apply(
        lambda row: get_acl_fatigue(row.get("away_team"), row.get("datetime"), acl_event_index, effective_days),
        axis=1,
    )

    out["home_acl_fatigue"] = home_infos.map(lambda x: x["acl_fatigue"]).astype(float)
    out["away_acl_fatigue"] = away_infos.map(lambda x: x["acl_fatigue"]).astype(float)
    out["home_acl_last_date"] = home_infos.map(lambda x: x["acl_last_date"])
    out["away_acl_last_date"] = away_infos.map(lambda x: x["acl_last_date"])
    out["home_acl_days_since"] = home_infos.map(lambda x: x["acl_days_since"])
    out["away_acl_days_since"] = away_infos.map(lambda x: x["acl_days_since"])
    out["home_acl_travel_type"] = home_infos.map(lambda x: x["acl_travel_type"])
    out["away_acl_travel_type"] = away_infos.map(lambda x: x["acl_travel_type"])

    base_home = pd.to_numeric(out.get("home_fatigue_score"), errors="coerce").fillna(0.0)
    base_away = pd.to_numeric(out.get("away_fatigue_score"), errors="coerce").fillna(0.0)
    out["home_total_fatigue_score"] = base_home + out["home_acl_fatigue"]
    out["away_total_fatigue_score"] = base_away + out["away_acl_fatigue"]

    if ACL_DEBUG:
        applied = int(((out["home_acl_fatigue"] > 0) | (out["away_acl_fatigue"] > 0)).sum())
        print(
            f"[ACL_DEBUG] stage={stage_label} rows={len(out)} applied={applied} "
            f"effective_days={effective_days}"
        )
    return out

# 外部スタッツデータをマージする関数
def merge_external_stats(
    df,
    stats_csv_path,
    team_name_col="team_name",
    merge_col_prefix="",
    stage_label="external",
):
    try:
        external_stats = pd.read_csv(stats_csv_path)

        if team_name_col != "team_name":
            external_stats = external_stats.rename(columns={team_name_col: "team_name"})

        external_stats["team_name"] = external_stats["team_name"].astype(str).str.strip()
        external_stats["_merge_team_name"] = normalize_team_series(external_stats["team_name"])

        target_team_keys = set(normalize_team_series(df["home_team"]).dropna())
        target_team_keys.update(normalize_team_series(df["away_team"]).dropna())
        external_team_keys = set(external_stats["_merge_team_name"].dropna())
        overlap = target_team_keys & external_team_keys
        if merge_col_prefix == "rankmot_" and target_team_keys - external_team_keys:
            print(
                f"[MERGE_QC][WARN] {stage_label}: 順位情報の未結合 "
                f"missing_teams={sorted(target_team_keys - external_team_keys)} "
                f"matched={len(overlap)}/{len(target_team_keys)} source={stats_csv_path}"
            )
        if target_team_keys and not overlap:
            print(
                f"[MERGE_QC][INFO] {stage_label}: 対象リーグとのチーム一致なし "
                f"(target_teams={len(target_team_keys)}, external_teams={len(external_team_keys)}); "
                "対象外データとしてスキップ"
            )
            return df

        # マージ対象カラム抽出
        stats_cols = [
            col for col in external_stats.columns
            if col not in ["team_name", "_merge_team_name", "match_date"]
        ]

        # 正規化キー重複を解消（例: 栃木SC / 栃木Ｃ / 栃木シティ -> 栃木C）
        dup_count = int(external_stats.duplicated(subset=["_merge_team_name"]).sum())
        if dup_count:
            print(
                f"[MERGE_QC][WARN] {stage_label}: 右側重複キー={dup_count} "
                "-> 情報量が多い行を優先して重複排除"
            )
            # 非欠損が多い行を優先し、同率は後勝ち（最新行）にする
            external_stats["_non_na_score"] = external_stats[stats_cols].notna().sum(axis=1)
            external_stats = (
                external_stats
                .sort_values(by=["_merge_team_name", "_non_na_score"], ascending=[True, False], kind="mergesort")
                .drop_duplicates(subset=["_merge_team_name"], keep="first")
                .drop(columns=["_non_na_score"], errors="ignore")
            )

        # HOME 用データ作成
        home_stats = external_stats.copy()
        home_rename = {
            col: f"{merge_col_prefix}{col}_home"
            for col in stats_cols
        }
        home_stats = home_stats.rename(columns=home_rename)

        # AWAY 用データ作成
        away_stats = external_stats.copy()
        away_rename = {
            col: f"{merge_col_prefix}{col}_away"
            for col in stats_cols
        }
        away_stats = away_stats.rename(columns=away_rename)

        out = df.copy()
        out["_merge_home_team"] = normalize_team_series(out["home_team"])
        out["_merge_away_team"] = normalize_team_series(out["away_team"])

        # HOME merge
        out = audited_left_merge(
            out,
            home_stats[["_merge_team_name"] + list(home_rename.values())],
            stage=f"{stage_label}_home",
            left_on="_merge_home_team",
            right_on="_merge_team_name",
            validate="many_to_one",
        )
        out = out.drop(columns=["_merge_team_name"], errors="ignore")

        # AWAY merge
        out = audited_left_merge(
            out,
            away_stats[["_merge_team_name"] + list(away_rename.values())],
            stage=f"{stage_label}_away",
            left_on="_merge_away_team",
            right_on="_merge_team_name",
            validate="many_to_one",
        )
        out = out.drop(columns=["_merge_team_name"], errors="ignore")

        out = out.drop(columns=["_merge_home_team", "_merge_away_team"], errors="ignore")
        return out

    except FileNotFoundError:
        if merge_col_prefix == "rankmot_":
            print(f"[MERGE_QC][WARN] {stage_label}: 順位情報の未結合 source={stats_csv_path} (ファイルなし)")
        print(f"警告: 外部スタッツファイル '{stats_csv_path}' が見つかりません。スキップ。")
        return df

    except Exception as e:
        print(f"エラー: 外部スタッツのマージ中にエラー発生: {e}")
        raise

# 前年結果読み込み（任意）
if os.path.exists(csv_prev):
    df_prev = _read_csv_guarded(
        csv_prev,
        stage="prev_results",
        required_cols=["home_team", "away_team"],
        allow_empty=True,
        strict=False,
    )
    print(f"前年データを読み込みました: {csv_prev}")
else:
    df_prev = pd.DataFrame(columns=["home_team", "away_team", "home_score", "away_score"])
    print("前年データは未使用（ファイルなし）。")

required_season_cols = ["home_team", "away_team"]
df_2025 = _read_csv_guarded(
    csv_season,
    stage="season_upcoming",
    required_cols=required_season_cols,
    allow_empty=True,
    strict=False,
)
if df_2025.empty:
    print(
        f"[WARN] season_upcoming: empty or unreadable -> trying latest_results fallback ({csv_season_latest})"
    )
    df_2025 = _read_csv_guarded(
        csv_season_latest,
        stage="season_latest_fallback",
        required_cols=required_season_cols,
        allow_empty=False,
        strict=STRICT_MODE,
        hints=[
            f"primary source has no usable rows: {csv_season}",
            "upcoming 取得が未完了の可能性があります",
        ],
        actions=[
            f"SEASON_YEAR={SEASON_YEAR} LEAGUE={LEAGUE} ./scripts/run_batch_weekly.sh",
            f"SEASON_YEAR={SEASON_YEAR} LEAGUE={LEAGUE} ./scripts/run_batch_matchday.sh",
        ],
    )
    if not df_2025.empty:
        print(
            f"[PATH] {os.path.basename(csv_season)} が空のため latest_results を採用: "
            f"{csv_season_latest} (rows={len(df_2025)})"
        )

if df_2025.empty:
    _fatal_data_error(
        "season_source",
        f"no usable rows from upcoming/latest for {LEAGUE}-{SEASON_YEAR}",
        hints=[f"upcoming={csv_season}", f"latest={csv_season_latest}"],
        actions=[f"SEASON_YEAR={SEASON_YEAR} LEAGUE={LEAGUE} ./scripts/run_batch_weekly.sh"],
    )

df_2025 = merge_missing_matches_from_latest_results(df_2025, csv_season_latest)
df_2025 = enrich_scores_from_latest_results(df_2025, csv_season_latest)

# datetime 列の補完（date列がある場合）
if "datetime" not in df_2025.columns and "date" in df_2025.columns:
    df_2025["datetime"] = pd.to_datetime(df_2025["date"], errors="coerce")
if "datetime" in df_2025.columns:
    df_2025["datetime"] = pd.to_datetime(df_2025["datetime"], errors="coerce")

# スコア付き行のみ抽出（素材化）
df_prev = df_prev.dropna(subset=["home_score", "away_score"])
df_2025_finished = df_2025.dropna(subset=["home_score", "away_score"])
df_2025_future = df_2025[df_2025["home_score"].isna()]
_ensure_merge_qc_dir()
export_team_name_diff(df_2025, team_master_stats_csv)

# 疲労度スコア（試合単位）をマージ
try:
    fatigue_scores_df = pd.read_csv(team_fatigue_scores_csv)
    if "datetime" in fatigue_scores_df.columns:
        fatigue_scores_df["datetime"] = pd.to_datetime(fatigue_scores_df["datetime"], errors="coerce")

    merge_keys = ["datetime", "home_team", "away_team"]
    fatigue_detail_cols = [
        "home_rest_fatigue", "away_rest_fatigue",
        "home_recent_load_carry", "away_recent_load_carry",
        "home_travel_fatigue", "away_travel_fatigue",
        "home_away_condition_penalty", "away_away_condition_penalty",
        "away_travel_distance_km", "travel_lookup_status",
        "fatigue_measurement_version", "external_schedule_coverage", "travel_distance_basis",
        "home_last_match_at", "away_last_match_at", "home_rest_hours", "away_rest_hours",
        "home_matches_last7d", "away_matches_last7d", "home_matches_last14d", "away_matches_last14d",
        "home_external_matches_last14d", "away_external_matches_last14d",
        "home_external_load_unknown", "away_external_load_unknown",
        "home_external_events_json", "away_external_events_json",
    ]
    fatigue_cols = ["home_fatigue_score", "away_fatigue_score"] + [
        c for c in fatigue_detail_cols if c in fatigue_scores_df.columns
    ]
    fatigue_merge_df = fatigue_scores_df[merge_keys + fatigue_cols].copy()
    dup = int(fatigue_merge_df.duplicated(subset=merge_keys).sum())
    if dup:
        print(f"[MERGE_QC][WARN] fatigue: 右側重複キー={dup} -> 最後の行を採用して重複排除")
        fatigue_merge_df = fatigue_merge_df.drop_duplicates(subset=merge_keys, keep="last")

    fatigue_future_merge_df, future_rescued = add_fatigue_time_aliases(
        df_2025_future, fatigue_merge_df, merge_keys
    )
    fatigue_finished_merge_df, finished_rescued = add_fatigue_time_aliases(
        df_2025_finished, fatigue_merge_df, merge_keys
    )
    if future_rescued or finished_rescued:
        print(
            f"[FATIGUE_TIME_FALLBACK] tolerance=10min "
            f"future={future_rescued} finished={finished_rescued}"
        )

    df_2025_future = audited_left_merge(
        df_2025_future,
        fatigue_future_merge_df,
        stage="fatigue_future",
        on=merge_keys,
        validate="one_to_one",
    )
    df_2025_finished = audited_left_merge(
        df_2025_finished,
        fatigue_finished_merge_df,
        stage="fatigue_finished",
        on=merge_keys,
        validate="one_to_one",
    )
    report_missing_rates(df_2025_future, "after_fatigue_future")
    report_missing_rates(df_2025_finished, "after_fatigue_finished")
    validate_prediction_fatigue(df_2025_future)
    validate_prediction_fatigue(df_2025_finished)
except FileNotFoundError:
    raise RuntimeError(f"FATIGUE_INPUT: 疲労度ファイル '{team_fatigue_scores_csv}' がありません。予測を停止します。")
except Exception as e:
    print(f"エラー: 疲労度のマージ中にエラーが発生しました: {e}")
    raise

# ACL疲労イベント（直前1試合のみ有効）を通常疲労に加算
try:
    acl_events_df = load_acl_events(acl_schedule_csv)
    acl_event_index = build_acl_event_index(acl_events_df)
    df_2025_future = attach_acl_fatigue_to_matches(
        df_2025_future,
        acl_event_index,
        ACL_EFFECTIVE_DAYS,
        stage_label="future",
    )
    df_2025_finished = attach_acl_fatigue_to_matches(
        df_2025_finished,
        acl_event_index,
        ACL_EFFECTIVE_DAYS,
        stage_label="finished",
    )
    if not acl_events_df.empty:
        print(
            f"[ACL] loaded={len(acl_events_df)} path={acl_schedule_csv} "
            f"effective_days={ACL_EFFECTIVE_DAYS}"
        )
    report_missing_rates(df_2025_future, "after_acl_future")
    report_missing_rates(df_2025_finished, "after_acl_finished")
except FileNotFoundError:
    print(f"[ACL] schedule csv not found: {acl_schedule_csv}")
except Exception as e:
    print(f"エラー: ACL疲労の付与中にエラーが発生しました: {e}")
    raise

# 天候キャッシュ（match_idキー）をマージ
if os.path.exists(weather_cache_csv) or os.path.isdir(WEATHER_SNAPSHOT_DIR):
    try:
        weather_merge_df = load_weather_union_dataframe(weather_cache_csv)
        required_weather_cols = ["match_id", "is_rain", "is_heavy_rain", "is_strong_wind"]
        for c in required_weather_cols:
            if c not in weather_merge_df.columns:
                weather_merge_df[c] = pd.NA

        if weather_merge_df.empty:
            raise RuntimeError("weather union dataframe is empty")

        weather_dup = int(weather_merge_df.duplicated(subset=["match_id"]).sum())
        if weather_dup:
            print(f"[MERGE_QC][WARN] weather_union: 右側重複キー={weather_dup} -> 最後の行を採用")
            weather_merge_df = weather_merge_df.drop_duplicates(subset=["match_id"], keep="last")

        df_2025_future = merge_weather_cache(df_2025_future, weather_merge_df, stage="weather_cache_future")
        df_2025_finished = merge_weather_cache(df_2025_finished, weather_merge_df, stage="weather_cache_finished")
        report_missing_rates(df_2025_future, "after_weather_cache_future")
        report_missing_rates(df_2025_finished, "after_weather_cache_finished")
        print(f"天候データを union ローダーで読み込みました。primary={weather_cache_csv}")
    except Exception as e:
        print(f"エラー: 天候キャッシュのマージ中にエラーが発生しました: {e}")
        raise
else:
    print(f"天候キャッシュが見つかりません: {weather_cache_csv}")
    df_2025_future["is_rain"] = False
    df_2025_future["is_heavy_rain"] = False
    df_2025_future["is_strong_wind"] = False
    df_2025_future["weather_missing"] = True
    df_2025_future["weather_adjustment_status"] = "unknown_no_adjustment"
    df_2025_finished["is_rain"] = False
    df_2025_finished["is_heavy_rain"] = False
    df_2025_finished["is_strong_wind"] = False
    df_2025_finished["weather_missing"] = True
    df_2025_finished["weather_adjustment_status"] = "unknown_no_adjustment"

# team_master_stats.csv をマージ
df_2025_future = merge_external_stats(
    df_2025_future,
    team_master_stats_csv,
    team_name_col="team_name",
    merge_col_prefix="stats_",
    stage_label="stats_future",
)
df_2025_finished = merge_external_stats(
    df_2025_finished,
    team_master_stats_csv,
    team_name_col="team_name",
    merge_col_prefix="stats_",
    stage_label="stats_finished",
)
df_2025_future = drop_j2_excluded_stats_columns(df_2025_future)
df_2025_finished = drop_j2_excluded_stats_columns(df_2025_finished)
report_missing_rates(df_2025_future, "after_stats_future")
report_missing_rates(df_2025_finished, "after_stats_finished")

# team_management_master.csv をマージ
df_2025_future = merge_external_stats(
    df_2025_future,
    team_management_master_csv,
    team_name_col="team_name",
    merge_col_prefix="management_",
    stage_label="management_future",
)
df_2025_finished = merge_external_stats(
    df_2025_finished,
    team_management_master_csv,
    team_name_col="team_name",
    merge_col_prefix="management_",
    stage_label="management_finished",
)
report_missing_rates(df_2025_future, "after_management_future")
report_missing_rates(df_2025_finished, "after_management_finished")

# ランキング推移由来モチベーションをマージ
df_2025_future = merge_external_stats(
    df_2025_future,
    team_motivation_csv,
    team_name_col="team_name",
    merge_col_prefix="rankmot_",
    stage_label="rankmot_future",
)
df_2025_finished = merge_external_stats(
    df_2025_finished,
    team_motivation_csv,
    team_name_col="team_name",
    merge_col_prefix="rankmot_",
    stage_label="rankmot_finished",
)
report_missing_rates(df_2025_future, "after_rankmot_future")
report_missing_rates(df_2025_finished, "after_rankmot_finished")

# 欠場影響（absences_with_impact.csv）を試合日×チームでマージ。
# 新形式は expected_weeks から自動幅を付けて減衰し、旧形式は節指定へフォールバックする。
absence_map_df = load_absence_impact_team_round_map(
    absence_impact_csv,
    [df_2025_future, df_2025_finished],
)
df_2025_future = merge_absence_impacts(df_2025_future, absence_map_df, stage_label="absence_future")
df_2025_finished = merge_absence_impacts(df_2025_finished, absence_map_df, stage_label="absence_finished")
report_missing_rates(df_2025_future, "after_absence_future")
report_missing_rates(df_2025_finished, "after_absence_finished")
derby_master_df = load_derby_table(derby_master_csv)

# team_travel_distances.csv を読み込み、データフレームとして準備 (行列形式)
# これはルックアップテーブルとして使用
try:
    travel_distances_df = pd.read_csv(team_travel_distances_csv, sep='	')
    travel_distances_df = travel_distances_df.set_index('ホーム　／　アウェイ')
    travel_distances_df = normalize_travel_distance_matrix(travel_distances_df)
    print(f"移動距離データを {team_travel_distances_csv} から読み込みました。")
except FileNotFoundError:
    print(f"警告: 移動距離ファイル '{team_travel_distances_csv}' が見つかりませんでした。移動距離データは使用されません。")
    travel_distances_df = pd.DataFrame() # ファイルがない場合は空のDataFrameを設定
except Exception as e:
    print(f"警告: 移動距離データの読み込み中にエラーが発生しました: {e}。移動距離データは使用されません。")
    travel_distances_df = pd.DataFrame()


# Elo初期化
all_teams = set(df_2025["home_team"].tolist() + df_2025["away_team"].tolist())
if not df_prev.empty:
    all_teams |= set(df_prev["home_team"].tolist() + df_prev["away_team"].tolist())

# 前年最終ELOを初期値として使用（未存在チームのみ1500）
prev_final_elo_map = load_or_build_prev_final_elo(df_prev)
elo_base = {team: float(prev_final_elo_map.get(team, INITIAL_ELO)) for team in all_teams}

# 予測用Elo（当年終了済みを反映）
elo_for_prediction = dict(elo_base)
df_2025_finished_for_elo = sort_results_for_elo(df_2025_finished)
for _, row in df_2025_finished_for_elo.iterrows():
    home, away = row["home_team"], row["away_team"]
    hs, as_ = row["home_score"], row["away_score"]
    if home not in elo_for_prediction:
        elo_for_prediction[home] = INITIAL_ELO
    if away not in elo_for_prediction:
        elo_for_prediction[away] = INITIAL_ELO
    result = get_result(hs, as_)
    if result:
        elo_for_prediction[home], elo_for_prediction[away] = update_elo(
            elo_for_prediction[home], elo_for_prediction[away], result
        )

df_all_results = pd.concat([df_prev, df_2025_finished], ignore_index=True)

# STEP1: チーム別ホーム/アウェイ成績を作成
home_ppm_map, away_ppm_map = build_home_away_profile_map(df_all_results)
allowed_teams = load_allowed_teams()

# 予測
predictions = []
elo_debug_rows = []
for _, row in df_2025_future.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    if home not in elo_for_prediction:
        elo_for_prediction[home] = INITIAL_ELO
    if away not in elo_for_prediction:
        elo_for_prediction[away] = INITIAL_ELO
    home_advantage_profile_diff, _ = calc_home_advantage_diff(
        home, away, home_ppm_map, away_ppm_map
    )

    home_canon = canonical_team_name(home)
    away_canon = canonical_team_name(away)
    if allowed_teams is not None and (home_canon not in allowed_teams or away_canon not in allowed_teams):
        # 対象リーグ外カードは出力対象外
        continue

    # team_master_stats からゴール期待値（xG）を取得。存在しない場合はNone
    home_xg_stats = row.get("stats_ゴール期待値_home")
    away_xg_stats = row.get("stats_ゴール期待値_away")

    # 移動距離を取得。存在しない場合は0
    home_travel_distance = 0
    away_travel_distance = 0
    if not travel_distances_df.empty:
        # travel_distances_dfは行列形式なので、locで直接アクセス
        home_key = canonical_team_name(home)
        away_key = canonical_team_name(away)
        if home_key in travel_distances_df.index and away_key in travel_distances_df.columns:
            home_travel_distance = travel_distances_df.loc[home_key, away_key]
        if away_key in travel_distances_df.index and home_key in travel_distances_df.columns:
            away_travel_distance = travel_distances_df.loc[away_key, home_key]

    # 疲労度スコアを取得。存在しない場合はNone
    home_fatigue_score = row.get("home_total_fatigue_score", row.get("home_fatigue_score"))
    away_fatigue_score = row.get("away_total_fatigue_score", row.get("away_fatigue_score"))
    home_rank_motivation_score = _pick_first_non_na_value(
        row,
        ["rankmot_motivation_score_3w", "rankmot_motivation_score_5w"],
    )
    away_rank_motivation_score = _pick_first_non_na_value(
        row,
        ["rankmot_motivation_score_3w_away", "rankmot_motivation_score_5w_away"],
    )
    absence_effective = compute_effective_absence_impacts(row)
    home_absence_impact = absence_effective["absence_effective_total_home"]
    away_absence_impact = absence_effective["absence_effective_total_away"]
    home_absence_score, away_absence_score = _resolve_absence_scores_for_multinom(row, absence_effective)

    # 天候フラグを取得
    weather_flags = {
        "is_rain": bool(row.get("is_rain")) if pd.notna(row.get("is_rain")) else False,
        "is_heavy_rain": bool(row.get("is_heavy_rain")) if pd.notna(row.get("is_heavy_rain")) else False,
        "is_strong_wind": bool(row.get("is_strong_wind")) if pd.notna(row.get("is_strong_wind")) else False,
    }

    quality_flags = compute_row_quality_flags(row)

    # 共通ロジックで確率計算（raw）
    prob_home_win_raw, prob_draw_raw, prob_away_win_raw, _, debug_row = compute_probabilities_and_result(
        row.get("match_id"),
        elo_for_prediction[home],
        elo_for_prediction[away],
        home_advantage_profile_diff,
        home_xg_stats,
        away_xg_stats,
        home_travel_distance,
        away_travel_distance,
        home_fatigue_score,
        away_fatigue_score,
        home_rank_motivation_score,
        away_rank_motivation_score,
        home_absence_impact,
        away_absence_impact,
        home_absence_score,
        away_absence_score,
        weather_flags,
        quality_flags["stats_home_missing"],
        quality_flags["stats_away_missing"],
        quality_flags["data_quality_warn"],
        row=row,
        absence_effective=absence_effective,
    )
    prob_ctx = _apply_prob_pipeline(
        prob_home_win_raw,
        prob_draw_raw,
        prob_away_win_raw,
        row.get("league", LEAGUE),
    )
    prob_home_win = prob_ctx["final_home"]
    prob_draw = prob_ctx["final_draw"]
    prob_away_win = prob_ctx["final_away"]
    predicted_result, decision_reason, decision_metrics = decide_result(prob_home_win, prob_draw, prob_away_win)
    argmax_result = decision_metrics.get("argmax_result")
    argmax_max_prob = decision_metrics.get("argmax_max_prob")
    if DRAW_TWEAK_ENABLED:
        predicted_highest_prob_result, _, raw_decision_metrics = decide_result(
            prob_home_win_raw, prob_draw_raw, prob_away_win_raw
        )
    else:
        predicted_highest_prob_result, _, raw_decision_metrics = decide_result(
            prob_home_win, prob_draw, prob_away_win
        )
    argmax_max_prob_raw = raw_decision_metrics.get("argmax_max_prob")
    if not np.isclose(prob_home_win + prob_draw + prob_away_win, 1.0, atol=1e-6):
        print(
            f"[PROB_QC][WARN] match_id={row.get('match_id')} calibrated_prob_sum="
            f"{(prob_home_win + prob_draw + prob_away_win):.9f}"
        )

    debug_row.update(
        {
            "prob_home_win_raw": prob_home_win_raw,
            "prob_draw_raw": prob_draw_raw,
            "prob_away_win_raw": prob_away_win_raw,
            "base_prob_home": prob_ctx["base_home"],
            "base_prob_draw": prob_ctx["base_draw"],
            "base_prob_away": prob_ctx["base_away"],
            "prob_home_win_cal": prob_home_win,
            "prob_draw_cal": prob_draw,
            "prob_away_win_cal": prob_away_win,
            "final_prob_home": prob_ctx["final_home"],
            "final_prob_draw": prob_ctx["final_draw"],
            "final_prob_away": prob_ctx["final_away"],
            "delta_prob_home": prob_ctx["delta_home"],
            "delta_prob_draw": prob_ctx["delta_draw"],
            "delta_prob_away": prob_ctx["delta_away"],
            "residue_detected": bool(prob_ctx["residue_detected"]),
            "predicted_result_cal": predicted_result,
            "decision_reason_cal": decision_reason,
            "argmax_result_cal": argmax_result,
            "argmax_max_prob_cal": argmax_max_prob,
        }
    )
    elo_debug_rows.append({**debug_row, "phase": "prediction"})

    # profile差分と確率入力elo差分は別管理する（上書きしない）
    home_advantage_diff = home_advantage_profile_diff
    is_home_advantage_positive = home_advantage_diff > 0

    predictions.append({
        **row.to_dict(),
        **absence_effective,
        "league": row.get("league", LEAGUE),
        "stats_asof": STATS_ASOF_LABEL,
        "stats_source_csv": os.path.basename(team_master_stats_csv) if team_master_stats_csv else "",
        "home_elo": round(elo_for_prediction[home]),
        "away_elo": round(elo_for_prediction[away]),
        "home_advantage_diff": round(home_advantage_diff, 4),
        "home_advantage_profile_diff": round(home_advantage_profile_diff, 4),
        "hfa_applied_elo": round(debug_row["HFA_applied"], 4),
        "hfa_added_to_diff": round(debug_row["hfa_added_to_diff"], 4),
        "hfa_clip_min": round(debug_row["hfa_clip_min"], 4),
        "hfa_clip_max": round(debug_row["hfa_clip_max"], 4),
        "elo_diff_scale_factor": round(debug_row["elo_diff_scale_factor"], 4),
        "elo_diff_before_hfa": round(debug_row["elo_diff_before_hfa"], 4),
        "elo_diff_after_hfa": round(debug_row["elo_diff_after_hfa"], 4),
        "diff_raw_no_hfa": round(debug_row.get("diff_raw_no_hfa", np.nan), 4),
        "elo_diff_scaled": round(debug_row["elo_diff_scaled"], 4),
        "elo_diff_for_prob": round(debug_row["elo_diff"], 4),
        "elo_diff_used_for_prob": round(debug_row["elo_diff_for_prob"], 4),
        "expected_home_two_way": round(debug_row["expected_home"], 4),
        "is_home_advantage_positive": bool(is_home_advantage_positive),
        "prob_home_win_raw": prob_home_win_raw,
        "prob_draw_raw": prob_draw_raw,
        "prob_away_win_raw": prob_away_win_raw,
        "draw_tweak_mode": DRAW_TWEAK_MODE,
        "base_prob_home": prob_ctx["base_home"],
        "base_prob_draw": prob_ctx["base_draw"],
        "base_prob_away": prob_ctx["base_away"],
        "prob_home_win_before_signfix": debug_row.get("prob_home_win_before_signfix"),
        "prob_away_win_before_signfix": debug_row.get("prob_away_win_before_signfix"),
        "elo_sign_fix_applied": bool(debug_row.get("elo_sign_fix_applied", False)),
        "elo_sign_fix_reason": debug_row.get("elo_sign_fix_reason", ""),
        "prob_home_raw": prob_home_win_raw,
        "prob_away_raw": prob_away_win_raw,
        "final_prob_home": prob_home_win,
        "final_prob_draw": prob_draw,
        "final_prob_away": prob_away_win,
        "prob_delta_home": prob_ctx["delta_home"],
        "prob_delta_draw": prob_ctx["delta_draw"],
        "prob_delta_away": prob_ctx["delta_away"],
        "residue_detected": bool(prob_ctx["residue_detected"]),
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "prob_home": prob_home_win,
        "prob_away": prob_away_win,
        "final_result": predicted_result,
        "predicted_result": predicted_result,
        "decision_reason": decision_reason,
        "force_draw_applied": str(decision_reason).startswith("FORCE_DRAW"),
        "argmax_result": argmax_result,
        "argmax_max_prob": argmax_max_prob,
        "argmax_raw_result": predicted_highest_prob_result,
        "argmax_raw_max_prob": argmax_max_prob_raw,
        "d_scaled": debug_row.get("draw_model_input"),
        "decision_draw_expectation_multiplier": 1.0 if not DRAW_TWEAK_ENABLED else DRAW_EXPECTATION_MULTIPLIER,
        "decision_draw_assign_enabled": bool(DRAW_ASSIGN_BY_EXPECTATION if DRAW_TWEAK_ENABLED else False),
        "predicted_highest_prob_result": predicted_highest_prob_result,
    })

# 保存
df_pred = pd.DataFrame(predictions)
df_pred = apply_round_type_draw_control(df_pred, "PRED")
df_pred = add_data_quality_flags(df_pred)
df_pred = fill_management_default_values(df_pred)
df_pred = merge_derby_context(df_pred, derby_master_df, "PRED")
df_pred = recalculate_predicted_result(df_pred, "predicted_result_prefusion_argmax")
df_pred = recalculate_predicted_highest_prob_result(df_pred, "predicted_highest_prob_result")
if not DRAW_TWEAK_ENABLED and not df_pred.empty:
    df_pred["predicted_highest_prob_result"] = df_pred["predicted_result_prefusion_argmax"]
    if "argmax_raw_result" in df_pred.columns:
        df_pred["argmax_raw_result"] = df_pred["predicted_result_prefusion_argmax"]
if DRAW_TWEAK_ENABLED and DRAW_ASSIGN_BY_EXPECTATION:
    # 最終ラベルは「調整後確率」をベースに、節単位の期待ドロー数へ合わせてDを付与する
    df_pred = assign_draw_results_by_expectation(df_pred, "final_result")
else:
    print(f"[DRAW_ASSIGN] disabled (DRAW_TWEAK_MODE={DRAW_TWEAK_MODE}, DRAW_ASSIGN_BY_EXPECTATION={int(DRAW_ASSIGN_BY_EXPECTATION)})")
df_pred = sync_and_validate_prediction_results(df_pred, "PRED", raise_on_error=True)
df_pred = _add_force_draw_flag(df_pred)
df_pred = apply_draw_candidate_flags(df_pred)
df_pred = merge_football_lab_compare(df_pred, LEAGUE, "PRED")
df_pred = add_team_state_and_lab_context(df_pred)
df_pred = apply_match_type_flags(df_pred)
df_pred = add_main_prediction_columns(df_pred)
df_pred = enforce_final_elo_sign_monotonic(df_pred, "PRED_POST_BLEND")
df_pred = apply_main_prediction_result(df_pred, "PRED_FINAL")
df_pred = apply_connected_prediction_overrides(df_pred, LEAGUE, "PRED_FINAL")
df_pred = add_buyplan_purchase_context(df_pred)
log_draw_tweak_audit(df_pred, "PRED")
log_decision_rule_once()
log_hda_diagnostics(df_pred, "PRED")
log_pred_dist(df_pred, "PRED", scope="all")
log_pred_dist_expect(df_pred, "PRED", scope="all")
log_pred_dist_argmax_expect_diff(df_pred, "PRED", scope="all")
try:
    round_mask_pred, round_label_pred, _ = _resolve_round_filter(df_pred)
    if len(round_mask_pred) == len(df_pred) and int(round_mask_pred.sum()) > 0:
        log_pred_dist(df_pred.loc[round_mask_pred], "PRED", scope=f"round:{round_label_pred}")
        log_pred_dist_expect(df_pred.loc[round_mask_pred], "PRED", scope=f"round:{round_label_pred}")
        log_pred_dist_argmax_expect_diff(df_pred.loc[round_mask_pred], "PRED", scope=f"round:{round_label_pred}")
except Exception as e:
    print(f"[PRED_DIST:PRED][WARN] round scope unavailable: {e}")
log_prediction_consistency(df_pred, "PRED")
log_prob_summary(df_pred, "PRED_SUMMARY")
pred_sign_diff_col = _preferred_diff_col_for_sign_check(df_pred)
log_elo_sign_check(
    df_pred,
    "PRED_RAW",
    diff_col=pred_sign_diff_col,
    home_col="prob_home_win_before_signfix" if "prob_home_win_before_signfix" in df_pred.columns else "prob_home_win_raw",
    away_col="prob_away_win_before_signfix" if "prob_away_win_before_signfix" in df_pred.columns else "prob_away_win_raw",
)
dump_elo_sign_violations(
    df_pred,
    "PRED_RAW",
    diff_col=pred_sign_diff_col,
    home_col="prob_home_win_before_signfix" if "prob_home_win_before_signfix" in df_pred.columns else "prob_home_win_raw",
    away_col="prob_away_win_before_signfix" if "prob_away_win_before_signfix" in df_pred.columns else "prob_away_win_raw",
)
log_elo_sign_check(df_pred, "PRED_POST", diff_col=pred_sign_diff_col, home_col="prob_home_win_raw", away_col="prob_away_win_raw")
log_prob_draw_distribution(df_pred, "PRED")
log_draw_argmax_stats(df_pred, "PRED")
log_actual_hda_ratio(df_pred, "PRED")
log_absence_effective_summary(df_pred, "PRED")
summarize_round_hda(df_pred, df_results=df_2025, round_filter_label="auto")
dump_decision_artifacts(df_pred, label="pred", threshold=0.25)
df_pred = drop_internal_output_columns(df_pred)
report_missing_rates(df_pred, "final_predictions_df")
pred_snapshot = _save_output_snapshot(df_pred, output_csv, "predictions_candidate")
if pred_snapshot:
    print(f"[SNAPSHOT] predictions candidate saved: {pred_snapshot}")
pred_write_guard = _guarded_write_csv(
    df_pred,
    output_csv,
    "predictions_primary",
    critical_cols=["match_id", "datetime", "home_team", "away_team", "prob_home_win", "prob_draw", "prob_away_win"],
)
if pred_write_guard["written"]:
    print(f"予測結果を {output_csv} に出力しました。")
    write_buyplan_context_csv(df_pred, output_csv)
else:
    print(f"[WRITE_GUARD] 予測結果の本体上書きをスキップしました: {output_csv}")
if output_csv != LEGACY_OUTPUT_CSV:
    if pred_write_guard["written"]:
        df_pred.to_csv(LEGACY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[LEGACY_ALIAS] 互換出力を更新しました: {LEGACY_OUTPUT_CSV}")
        legacy_snapshot = _save_output_snapshot(df_pred, LEGACY_OUTPUT_CSV, "predictions_legacy")
        if legacy_snapshot:
            print(f"[SNAPSHOT] predictions_legacy saved: {legacy_snapshot}")
    else:
        print(f"[LEGACY_ALIAS] 本体保護に連動して更新をスキップ: {LEGACY_OUTPUT_CSV}")

# --- 2025年終了済み試合のバックテストと的中率計算 ---
backtest_results = []
correct_predictions = 0
total_finished_games = 0

# バックテストはリーク防止：時系列で「予測→採点→Elo更新」
elo_for_backtest = dict(elo_base)
for _, row in sort_results_for_elo(df_2025_finished).iterrows():
    home, away = row["home_team"], row["away_team"]
    hs, as_ = row["home_score"], row["away_score"]
    if home not in elo_for_backtest:
        elo_for_backtest[home] = INITIAL_ELO
    if away not in elo_for_backtest:
        elo_for_backtest[away] = INITIAL_ELO
    
    # 実際の試合結果
    actual_result = get_result(hs, as_)
    home_advantage_profile_diff, _ = calc_home_advantage_diff(
        home, away, home_ppm_map, away_ppm_map
    )

    home_canon = canonical_team_name(home)
    away_canon = canonical_team_name(away)
    if allowed_teams is not None and (home_canon not in allowed_teams or away_canon not in allowed_teams):
        # 対象リーグ外カードは出力対象外
        continue

    # team_master_stats からゴール期待値（xG）を取得。存在しない場合はNone
    home_xg_stats = row.get("stats_ゴール期待値_home")
    away_xg_stats = row.get("stats_ゴール期待値_away")

    # 移動距離を取得。存在しない場合は0
    home_travel_distance = 0
    away_travel_distance = 0
    if not travel_distances_df.empty:
        home_key = canonical_team_name(home)
        away_key = canonical_team_name(away)
        if home_key in travel_distances_df.index and away_key in travel_distances_df.columns:
            home_travel_distance = travel_distances_df.loc[home_key, away_key]
        if away_key in travel_distances_df.index and home_key in travel_distances_df.columns:
            away_travel_distance = travel_distances_df.loc[away_key, home_key]

    # 疲労度スコアを取得。存在しない場合はNone
    home_fatigue_score = row.get("home_total_fatigue_score", row.get("home_fatigue_score"))
    away_fatigue_score = row.get("away_total_fatigue_score", row.get("away_fatigue_score"))
    home_rank_motivation_score = _pick_first_non_na_value(
        row,
        ["rankmot_motivation_score_3w", "rankmot_motivation_score_5w"],
    )
    away_rank_motivation_score = _pick_first_non_na_value(
        row,
        ["rankmot_motivation_score_3w_away", "rankmot_motivation_score_5w_away"],
    )
    absence_effective = compute_effective_absence_impacts(row)
    home_absence_impact = absence_effective["absence_effective_total_home"]
    away_absence_impact = absence_effective["absence_effective_total_away"]
    home_absence_score, away_absence_score = _resolve_absence_scores_for_multinom(row, absence_effective)

    # 天候フラグを取得
    weather_flags = {
        "is_rain": bool(row.get("is_rain")) if pd.notna(row.get("is_rain")) else False,
        "is_heavy_rain": bool(row.get("is_heavy_rain")) if pd.notna(row.get("is_heavy_rain")) else False,
        "is_strong_wind": bool(row.get("is_strong_wind")) if pd.notna(row.get("is_strong_wind")) else False,
    }

    quality_flags = compute_row_quality_flags(row)

    # 共通ロジックで確率計算（raw）
    prob_home_win_raw, prob_draw_raw, prob_away_win_raw, _, debug_row = compute_probabilities_and_result(
        row.get("match_id"),
        elo_for_backtest[home],
        elo_for_backtest[away],
        home_advantage_profile_diff,
        home_xg_stats,
        away_xg_stats,
        home_travel_distance,
        away_travel_distance,
        home_fatigue_score,
        away_fatigue_score,
        home_rank_motivation_score,
        away_rank_motivation_score,
        home_absence_impact,
        away_absence_impact,
        home_absence_score,
        away_absence_score,
        weather_flags,
        quality_flags["stats_home_missing"],
        quality_flags["stats_away_missing"],
        quality_flags["data_quality_warn"],
        row=row,
        absence_effective=absence_effective,
    )
    prob_ctx = _apply_prob_pipeline(
        prob_home_win_raw,
        prob_draw_raw,
        prob_away_win_raw,
        row.get("league", LEAGUE),
    )
    prob_home_win = prob_ctx["final_home"]
    prob_draw = prob_ctx["final_draw"]
    prob_away_win = prob_ctx["final_away"]
    predicted_label, decision_reason_bt, decision_metrics_bt = decide_result(prob_home_win, prob_draw, prob_away_win)
    argmax_result_bt = decision_metrics_bt.get("argmax_result")
    argmax_max_prob_bt = decision_metrics_bt.get("argmax_max_prob")
    if DRAW_TWEAK_ENABLED:
        predicted_highest_prob_result, _, raw_decision_metrics_bt = decide_result(
            prob_home_win_raw, prob_draw_raw, prob_away_win_raw
        )
    else:
        predicted_highest_prob_result, _, raw_decision_metrics_bt = decide_result(
            prob_home_win, prob_draw, prob_away_win
        )
    argmax_max_prob_raw_bt = raw_decision_metrics_bt.get("argmax_max_prob")
    if not np.isclose(prob_home_win + prob_draw + prob_away_win, 1.0, atol=1e-6):
        print(
            f"[PROB_QC][WARN] match_id={row.get('match_id')} calibrated_prob_sum="
            f"{(prob_home_win + prob_draw + prob_away_win):.9f}"
        )
    debug_row.update(
        {
            "prob_home_win_raw": prob_home_win_raw,
            "prob_draw_raw": prob_draw_raw,
            "prob_away_win_raw": prob_away_win_raw,
            "base_prob_home": prob_ctx["base_home"],
            "base_prob_draw": prob_ctx["base_draw"],
            "base_prob_away": prob_ctx["base_away"],
            "prob_home_win_cal": prob_home_win,
            "prob_draw_cal": prob_draw,
            "prob_away_win_cal": prob_away_win,
            "final_prob_home": prob_ctx["final_home"],
            "final_prob_draw": prob_ctx["final_draw"],
            "final_prob_away": prob_ctx["final_away"],
            "delta_prob_home": prob_ctx["delta_home"],
            "delta_prob_draw": prob_ctx["delta_draw"],
            "delta_prob_away": prob_ctx["delta_away"],
            "residue_detected": bool(prob_ctx["residue_detected"]),
            "predicted_result_cal": predicted_label,
            "decision_reason_cal": decision_reason_bt,
            "argmax_result_cal": argmax_result_bt,
            "argmax_max_prob_cal": argmax_max_prob_bt,
        }
    )
    elo_debug_rows.append({**debug_row, "phase": "backtest"})
    home_advantage_diff = home_advantage_profile_diff
    is_home_advantage_positive = home_advantage_diff > 0
    is_correct = (predicted_label == actual_result) if actual_result else False
    
    backtest_results.append({
        **row.to_dict(),
        **absence_effective,
        "league": row.get("league", LEAGUE),
        "stats_asof": STATS_ASOF_LABEL,
        "stats_source_csv": os.path.basename(team_master_stats_csv) if team_master_stats_csv else "",
        "home_elo_at_prediction": round(elo_for_backtest[home]), # 予測時のEloスコアを記録
        "away_elo_at_prediction": round(elo_for_backtest[away]),
        "home_advantage_diff": round(home_advantage_diff, 4),
        "home_advantage_profile_diff": round(home_advantage_profile_diff, 4),
        "hfa_applied_elo": round(debug_row["HFA_applied"], 4),
        "hfa_added_to_diff": round(debug_row["hfa_added_to_diff"], 4),
        "hfa_clip_min": round(debug_row["hfa_clip_min"], 4),
        "hfa_clip_max": round(debug_row["hfa_clip_max"], 4),
        "elo_diff_scale_factor": round(debug_row["elo_diff_scale_factor"], 4),
        "elo_diff_before_hfa": round(debug_row["elo_diff_before_hfa"], 4),
        "elo_diff_after_hfa": round(debug_row["elo_diff_after_hfa"], 4),
        "diff_raw_no_hfa": round(debug_row.get("diff_raw_no_hfa", np.nan), 4),
        "elo_diff_scaled": round(debug_row["elo_diff_scaled"], 4),
        "elo_diff_for_prob": round(debug_row["elo_diff"], 4),
        "elo_diff_used_for_prob": round(debug_row["elo_diff_for_prob"], 4),
        "expected_home_two_way": round(debug_row["expected_home"], 4),
        "is_home_advantage_positive": bool(is_home_advantage_positive),
        "prob_home_win_raw": prob_home_win_raw,
        "prob_draw_raw": prob_draw_raw,
        "prob_away_win_raw": prob_away_win_raw,
        "draw_tweak_mode": DRAW_TWEAK_MODE,
        "base_prob_home": prob_ctx["base_home"],
        "base_prob_draw": prob_ctx["base_draw"],
        "base_prob_away": prob_ctx["base_away"],
        "prob_home_win_before_signfix": debug_row.get("prob_home_win_before_signfix"),
        "prob_away_win_before_signfix": debug_row.get("prob_away_win_before_signfix"),
        "elo_sign_fix_applied": bool(debug_row.get("elo_sign_fix_applied", False)),
        "elo_sign_fix_reason": debug_row.get("elo_sign_fix_reason", ""),
        "prob_home_raw": prob_home_win_raw,
        "prob_away_raw": prob_away_win_raw,
        "final_prob_home": prob_home_win,
        "final_prob_draw": prob_draw,
        "final_prob_away": prob_away_win,
        "prob_delta_home": prob_ctx["delta_home"],
        "prob_delta_draw": prob_ctx["delta_draw"],
        "prob_delta_away": prob_ctx["delta_away"],
        "residue_detected": bool(prob_ctx["residue_detected"]),
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
        "prob_home": prob_home_win,
        "prob_away": prob_away_win,
        "final_result": predicted_label,
        "predicted_result": predicted_label,
        "decision_reason": decision_reason_bt,
        "force_draw_applied": str(decision_reason_bt).startswith("FORCE_DRAW"),
        "argmax_result": argmax_result_bt,
        "argmax_max_prob": argmax_max_prob_bt,
        "argmax_raw_result": predicted_highest_prob_result,
        "argmax_raw_max_prob": argmax_max_prob_raw_bt,
        "d_scaled": debug_row.get("draw_model_input"),
        "decision_draw_expectation_multiplier": 1.0 if not DRAW_TWEAK_ENABLED else DRAW_EXPECTATION_MULTIPLIER,
        "decision_draw_assign_enabled": bool(DRAW_ASSIGN_BY_EXPECTATION if DRAW_TWEAK_ENABLED else False),
        "predicted_highest_prob_result": predicted_highest_prob_result,
        "actual_result": actual_result,
        "is_correct": is_correct
    })

    if actual_result: # 実際の試合結果がある場合のみ的中率に含める
        total_finished_games += 1
        if is_correct:
            correct_predictions += 1
        elo_for_backtest[home], elo_for_backtest[away] = update_elo(
            elo_for_backtest[home], elo_for_backtest[away], actual_result
        )

df_backtest = pd.DataFrame(backtest_results)
if "stats_asof" not in df_backtest.columns:
    df_backtest["stats_asof"] = pd.Series(dtype="object")
if "stats_source_csv" not in df_backtest.columns:
    df_backtest["stats_source_csv"] = pd.Series(dtype="object")
df_backtest = apply_round_type_draw_control(df_backtest, "BACKTEST")
df_backtest = add_data_quality_flags(df_backtest)
df_backtest = fill_management_default_values(df_backtest)
df_backtest = merge_derby_context(df_backtest, derby_master_df, "BACKTEST")
df_backtest = recalculate_predicted_result(df_backtest, "predicted_result_prefusion_argmax")
df_backtest = recalculate_predicted_highest_prob_result(df_backtest, "predicted_highest_prob_result")
if not DRAW_TWEAK_ENABLED and not df_backtest.empty:
    df_backtest["predicted_highest_prob_result"] = df_backtest["predicted_result_prefusion_argmax"]
    if "argmax_raw_result" in df_backtest.columns:
        df_backtest["argmax_raw_result"] = df_backtest["predicted_result_prefusion_argmax"]
df_backtest_argmax = _apply_backtest_decision_rule(df_backtest.copy(), "argmax")
if DRAW_TWEAK_ENABLED and (DRAW_ASSIGN_BY_EXPECTATION or BACKTEST_DECISION_RULE in {"expect", "both"}):
    df_backtest_expect = _apply_backtest_decision_rule(df_backtest.copy(), "expect")
else:
    df_backtest_expect = None
if DRAW_TWEAK_ENABLED:
    run_backtest_decision_rule_compare(df_backtest.copy())
else:
    print(f"[BACKTEST_RULE] compare skipped (DRAW_TWEAK_MODE={DRAW_TWEAK_MODE})")
if DRAW_TWEAK_ENABLED and DRAW_ASSIGN_BY_EXPECTATION:
    df_backtest = df_backtest_expect.copy() if df_backtest_expect is not None else _apply_backtest_decision_rule(df_backtest.copy(), "expect")
else:
    df_backtest = df_backtest_argmax.copy()
df_backtest = sync_and_validate_prediction_results(df_backtest, "BACKTEST", raise_on_error=True)
df_backtest = _add_force_draw_flag(df_backtest)
log_draw_tweak_audit(df_backtest, "BACKTEST")
log_hda_diagnostics(df_backtest, "BACKTEST")
log_pred_dist(df_backtest, "BACKTEST", scope="all")
log_pred_dist_expect(df_backtest, "BACKTEST", scope="all")
log_pred_dist_argmax_expect_diff(df_backtest, "BACKTEST", scope="all")
log_prediction_consistency(df_backtest, "BACKTEST")
log_prob_summary(df_backtest, "BACKTEST_SUMMARY")
bt_sign_diff_col = _preferred_diff_col_for_sign_check(df_backtest)
log_elo_sign_check(
    df_backtest,
    "BACKTEST_RAW",
    diff_col=bt_sign_diff_col,
    home_col="prob_home_win_before_signfix" if "prob_home_win_before_signfix" in df_backtest.columns else "prob_home_win_raw",
    away_col="prob_away_win_before_signfix" if "prob_away_win_before_signfix" in df_backtest.columns else "prob_away_win_raw",
)
dump_elo_sign_violations(
    df_backtest,
    "BACKTEST_RAW",
    diff_col=bt_sign_diff_col,
    home_col="prob_home_win_before_signfix" if "prob_home_win_before_signfix" in df_backtest.columns else "prob_home_win_raw",
    away_col="prob_away_win_before_signfix" if "prob_away_win_before_signfix" in df_backtest.columns else "prob_away_win_raw",
)
log_elo_sign_check(df_backtest, "BACKTEST_POST", diff_col=bt_sign_diff_col, home_col="prob_home_win_raw", away_col="prob_away_win_raw")
log_multinom_feature_distribution(df_backtest, "BACKTEST")
log_prob_draw_distribution(df_backtest, "BACKTEST")
log_prob_distribution(df_backtest, "BACKTEST", "prob_home")
log_prob_distribution(df_backtest, "BACKTEST", "prob_draw")
log_prob_distribution(df_backtest, "BACKTEST", "prob_away")
log_max_prob_distribution(df_backtest, "BACKTEST")
log_draw_argmax_stats(df_backtest, "BACKTEST")
log_actual_hda_ratio(df_backtest, "BACKTEST")
log_absence_effective_summary(df_backtest, "BACKTEST")
if "actual_result" in df_backtest.columns and "predicted_result" in df_backtest.columns:
    df_backtest["is_correct"] = df_backtest["actual_result"] == df_backtest["predicted_result"]
    finished_mask = df_backtest["actual_result"].notna()
    total_finished_games = int(finished_mask.sum())
    correct_predictions = int(df_backtest.loc[finished_mask, "is_correct"].sum())
df_backtest = apply_draw_candidate_flags(df_backtest)
save_draw_diagnostics(df_backtest, LEAGUE, SEASON_YEAR)
save_draw_threshold_scan(df_backtest, LEAGUE, SEASON_YEAR)
df_backtest = merge_football_lab_compare(df_backtest, LEAGUE, "BACKTEST")
df_backtest = add_team_state_and_lab_context(df_backtest)
df_backtest = apply_match_type_flags(df_backtest)
df_backtest = add_main_prediction_columns(df_backtest)
df_backtest = enforce_final_elo_sign_monotonic(df_backtest, "BACKTEST_POST_BLEND")
df_backtest = apply_main_prediction_result(df_backtest, "BACKTEST_FINAL")
df_backtest = apply_connected_prediction_overrides(df_backtest, LEAGUE, "BACKTEST_FINAL")
df_backtest = add_buyplan_purchase_context(df_backtest)
if "actual_result" in df_backtest.columns and "predicted_result" in df_backtest.columns:
    df_backtest["is_correct"] = (
        df_backtest["actual_result"].astype(str).str.upper()
        == df_backtest["predicted_result"].astype(str).str.upper()
    )
    finished_mask = df_backtest["actual_result"].notna()
    total_finished_games = int(finished_mask.sum())
    correct_predictions = int(df_backtest.loc[finished_mask, "is_correct"].sum())
calibration_meta = {}
try:
    df_pred, df_backtest, calibration_meta = _save_calibration_artifacts(df_backtest, df_pred, LEAGUE, SEASON_YEAR)
except Exception as e:
    calibration_meta = {"error": str(e)}
    print(f"[CALIBRATION][WARN] skipped due to error: {e}")
save_match_type_diagnostics(df_backtest, LEAGUE, SEASON_YEAR)
save_argmax_diagnostics(df_backtest, LEAGUE, SEASON_YEAR)
df_backtest = drop_internal_output_columns(df_backtest)
report_missing_rates(df_backtest, "final_backtest_df")
bt_snapshot = _save_output_snapshot(df_backtest, backtest_output_csv, "backtest_candidate")
if bt_snapshot:
    print(f"[SNAPSHOT] backtest candidate saved: {bt_snapshot}")
bt_write_guard = _guarded_write_csv(
    df_backtest,
    backtest_output_csv,
    "backtest_primary",
    critical_cols=[
        "match_id",
        "datetime",
        "home_team",
        "away_team",
        "actual_result",
        "predicted_result",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "home_fatigue_score",
        "away_fatigue_score",
        "home_acl_fatigue",
        "away_acl_fatigue",
        "home_total_fatigue_score",
        "away_total_fatigue_score",
        "temperature",
        "wind_speed",
    ],
)
if bt_write_guard["written"]:
    print(f"2025年終了済み試合のバックテスト結果を {backtest_output_csv} に出力しました。")
else:
    print(f"[WRITE_GUARD] バックテスト本体の上書きをスキップしました: {backtest_output_csv}")
pred_write_guard = _guarded_write_csv(
    df_pred,
    output_csv,
    "predictions_primary_post_calibration",
    critical_cols=["match_id", "datetime", "home_team", "away_team", "prob_home_win", "prob_draw", "prob_away_win"],
)
if pred_write_guard["written"]:
    print(f"[CALIBRATION] 予測結果を calibration 列付きで再出力しました: {output_csv}")
else:
    print(f"[CALIBRATION][WRITE_GUARD] 予測結果の再出力をスキップしました: {output_csv}")
if output_csv != LEGACY_OUTPUT_CSV:
    if pred_write_guard["written"]:
        df_pred.to_csv(LEGACY_OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[CALIBRATION][LEGACY_ALIAS] 互換出力を更新しました: {LEGACY_OUTPUT_CSV}")
    else:
        print(f"[CALIBRATION][LEGACY_ALIAS] 再出力スキップに連動して更新をスキップ: {LEGACY_OUTPUT_CSV}")
print(
    f"[HFA_APPLY_COUNT] applied={HFA_APPLY_COUNTER['applied']} "
    f"skipped={HFA_APPLY_COUNTER['skipped']} "
    f"reason_counts={json.dumps(HFA_APPLY_COUNTER['reason_counts'], ensure_ascii=False, sort_keys=True)}"
)
print(
    f"[ELO_SIGN_FIX_COUNT] total={int(ELO_SIGN_FIX_COUNTER['total'])} "
    f"neg_to_away={int(ELO_SIGN_FIX_COUNTER['neg_to_away'])} "
    f"pos_to_home={int(ELO_SIGN_FIX_COUNTER['pos_to_home'])} "
    f"enabled={int(ENFORCE_ELO_SIGN_MONOTONIC)}"
)
if ENABLE_HFA and float(HFA_ELO) > 0:
    evaluated = int(HFA_APPLY_COUNTER["applied"]) + int(HFA_APPLY_COUNTER["skipped"])
    if evaluated <= 0:
        print("[ERROR] HFA apply counter has zero evaluated rows under ENABLE_HFA=1 and HFA_ELO>0")
        raise RuntimeError("HFA apply counter invalid: no evaluated rows")

try:
    run_meta = {
        "output_csv": output_csv,
        "output_snapshot_enabled": int(ENABLE_OUTPUT_SNAPSHOT),
        "output_snapshot_dir": OUTPUT_SNAPSHOT_DIR,
        "output_snapshot_pred": pred_snapshot if "pred_snapshot" in locals() else "",
        "output_snapshot_backtest": bt_snapshot if "bt_snapshot" in locals() else "",
        "output_overwrite_guard": int(OUTPUT_OVERWRITE_GUARD),
        "output_guard_min_row_ratio": float(OUTPUT_GUARD_MIN_ROW_RATIO),
        "output_guard_max_completeness_drop": float(OUTPUT_GUARD_MAX_COMPLETENESS_DROP),
        "pred_write_guard_written": int(pred_write_guard.get("written", False)) if "pred_write_guard" in locals() else 0,
        "pred_write_guard_reason": pred_write_guard.get("reason", "") if "pred_write_guard" in locals() else "",
        "pred_write_guard_rows_old": int(pred_write_guard.get("rows_old", 0)) if "pred_write_guard" in locals() else 0,
        "pred_write_guard_rows_new": int(pred_write_guard.get("rows_new", 0)) if "pred_write_guard" in locals() else 0,
        "pred_write_guard_quality_old": float(pred_write_guard.get("quality_old", 0.0)) if "pred_write_guard" in locals() else 0.0,
        "pred_write_guard_quality_new": float(pred_write_guard.get("quality_new", 0.0)) if "pred_write_guard" in locals() else 0.0,
        "bt_write_guard_written": int(bt_write_guard.get("written", False)) if "bt_write_guard" in locals() else 0,
        "bt_write_guard_reason": bt_write_guard.get("reason", "") if "bt_write_guard" in locals() else "",
        "bt_write_guard_rows_old": int(bt_write_guard.get("rows_old", 0)) if "bt_write_guard" in locals() else 0,
        "bt_write_guard_rows_new": int(bt_write_guard.get("rows_new", 0)) if "bt_write_guard" in locals() else 0,
        "bt_write_guard_quality_old": float(bt_write_guard.get("quality_old", 0.0)) if "bt_write_guard" in locals() else 0.0,
        "bt_write_guard_quality_new": float(bt_write_guard.get("quality_new", 0.0)) if "bt_write_guard" in locals() else 0.0,
        "league": LEAGUE,
        "season_year": int(SEASON_YEAR),
        "enable_hfa": int(ENABLE_HFA_INT),
        "hfa_elo": float(HFA_ELO),
        "draw_tweak_mode": DRAW_TWEAK_MODE,
        "draw_tweak_enabled": int(DRAW_TWEAK_ENABLED),
        "draw_assign_by_expectation_raw": int(DRAW_ASSIGN_BY_EXPECTATION_RAW),
        "draw_assign_by_expectation_effective": int(DRAW_ASSIGN_BY_EXPECTATION),
        "draw_expectation_multiplier_raw": float(DRAW_EXPECTATION_MULTIPLIER_RAW),
        "draw_expectation_multiplier_effective": float(DRAW_EXPECTATION_MULTIPLIER),
        "backtest_decision_rule": BACKTEST_DECISION_RULE,
        "draw_assign_group_mode": DRAW_ASSIGN_GROUP_MODE,
        "backtest_compare_dataset": BACKTEST_COMPARE_DATASET,
        "backtest_compare_csv": BACKTEST_COMPARE_CSV,
        "draw_margin": float(DRAW_MARGIN),
        "close_ha_gap": float(CLOSE_HA_GAP),
        "close_ha_min_level": float(CLOSE_HA_MIN_LEVEL),
        "close_ha_gap_weight": float(CLOSE_HA_GAP_WEIGHT),
        "close_ha_draw_score_min": float(CLOSE_HA_DRAW_SCORE_MIN),
        "close_ha_draw_score_min_grid_raw": CLOSE_HA_DRAW_SCORE_MIN_GRID_RAW,
        "close_d_top_gap": float(CLOSE_D_TOP_GAP),
        "close_d_top_gap_grid": CLOSE_D_TOP_GAP_GRID,
        "backtest_margin_scan": int(BACKTEST_MARGIN_SCAN),
        "draw_margin_grid_raw": DRAW_MARGIN_GRID_RAW,
        "j1_target_d_range_raw": J1_TARGET_D_RANGE_RAW,
        "j2_target_d_range_raw": J2_TARGET_D_RANGE_RAW,
        "acl_schedule_csv": acl_schedule_csv if os.path.exists(acl_schedule_csv) else None,
        "acl_effective_days": int(ACL_EFFECTIVE_DAYS),
        "pred_rows": int(len(df_pred)) if "df_pred" in globals() else 0,
        "hfa_apply_count": {
            "applied": int(HFA_APPLY_COUNTER["applied"]),
            "skipped": int(HFA_APPLY_COUNTER["skipped"]),
            "reason_counts": HFA_APPLY_COUNTER["reason_counts"],
        },
        "elo_sign_fix_count": {
            "enabled": int(ENFORCE_ELO_SIGN_MONOTONIC),
            "total": int(ELO_SIGN_FIX_COUNTER["total"]),
            "neg_to_away": int(ELO_SIGN_FIX_COUNTER["neg_to_away"]),
            "pos_to_home": int(ELO_SIGN_FIX_COUNTER["pos_to_home"]),
        },
        "multinom_sign_detect": {
            "sign": int(MULTINOM_ELO_DIFF_SIGN),
            "swap_ha": int(MULTINOM_SWAP_HA_OUTPUT),
        },
        "calibration": calibration_meta if "calibration_meta" in globals() else {},
    }
    meta_path = f"{output_csv}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[RUN_META] saved={meta_path}")
except Exception as e:
    print(f"[RUN_META][WARN] failed to write meta: {e}")

if elo_debug_rows:
    debug_df = pd.DataFrame(elo_debug_rows)
    debug_csv = os.path.join(REPORT_DIR, f"elo_prob_debug_{LEAGUE}_{SEASON_YEAR}.csv")
    debug_df.to_csv(debug_csv, index=False, encoding="utf-8-sig")
    print(f"Elo確率デバッグCSVを出力しました: {debug_csv}")

# 的中率の表示
if total_finished_games > 0:
    accuracy = (correct_predictions / total_finished_games) * 100
    print(f"2025年シーズン現時点までの累計的中率: {accuracy:.2f}% ({correct_predictions}/{total_finished_games})")
else:
    print("2025年シーズンの終了済み試合がありません。")


# --- 解析内訳レポート出力 ---
def build_report():
    global df_pred, df_backtest
    if "df_pred" not in globals():
        df_pred = pd.DataFrame()
    if "df_backtest" not in globals():
        df_backtest = pd.DataFrame()
    inputs = {
        "csv_prev": csv_prev if os.path.exists(csv_prev) else None,
        "prev_final_elo_csv": prev_final_elo_csv if os.path.exists(prev_final_elo_csv) else None,
        "csv_season": csv_season,
        "team_master_stats_csv": team_master_stats_csv,
        "stats_asof": STATS_ASOF_LABEL,
        "absence_impact_csv": absence_impact_csv if absence_impact_csv and os.path.exists(absence_impact_csv) else None,
        "absence_asof": absence_asof_key if absence_asof_key else None,
        "team_management_master_csv": team_management_master_csv if os.path.exists(team_management_master_csv) else None,
        "team_motivation_csv": team_motivation_csv if os.path.exists(team_motivation_csv) else None,
        "team_travel_distances_csv": team_travel_distances_csv if os.path.exists(team_travel_distances_csv) else None,
        "team_fatigue_scores_csv": team_fatigue_scores_csv if os.path.exists(team_fatigue_scores_csv) else None,
        "acl_schedule_csv": acl_schedule_csv if os.path.exists(acl_schedule_csv) else None,
        "weather_cache_csv": weather_cache_csv if os.path.exists(weather_cache_csv) else None,
        "weather_asof": weather_asof_key if weather_asof_key else None,
    }

    params = {
        "INITIAL_ELO": INITIAL_ELO,
        "ELO_UPDATE_HOME_ADVANTAGE": ELO_UPDATE_HOME_ADVANTAGE,
        "HFA_ELO": HFA_ELO,
        "ENABLE_HFA": ENABLE_HFA_INT,
        "ENABLE_MATCHUP_BIAS": int(ENABLE_MATCHUP_BIAS),
        "MATCHUP_BIAS_COEF": MATCHUP_BIAS_COEF,
        "HOME_ADV_ELO_COEF": HOME_ADV_ELO_COEF,
        "HOME_ADV_PROFILE_DIFF_CLIP": HOME_ADV_PROFILE_DIFF_CLIP,
        "ELO_DIFF_TEMPERATURE": ELO_DIFF_TEMPERATURE,
        "ELO_D_VALUE": ELO_D_VALUE,
        "HFA_PROB_WEIGHT": HFA_PROB_WEIGHT,
        "MULTINOM_ELO_DIFF_SIGN": int(MULTINOM_ELO_DIFF_SIGN),
        "MULTINOM_SWAP_HA_OUTPUT": int(MULTINOM_SWAP_HA_OUTPUT),
        "J1_WIN_PROB_CAP": J1_WIN_PROB_CAP,
        "GOAL_SCALING_FACTOR": GOAL_SCALING_FACTOR,
        "FATIGUE_GOAL_SCALING": FATIGUE_GOAL_SCALING,
        "AWAY_PROB_MULTIPLIER": AWAY_PROB_MULTIPLIER,
        "ACL_EFFECTIVE_DAYS": ACL_EFFECTIVE_DAYS,
        "ENABLE_ROUND_TYPE_DRAW_CONTROL": int(ENABLE_ROUND_TYPE_DRAW_CONTROL),
        "ROUND_TYPE_DRAW_REL_THRESHOLD": ROUND_TYPE_DRAW_REL_THRESHOLD,
        "ROUND_TYPE_DRAW_SHARE_THRESHOLD": ROUND_TYPE_DRAW_SHARE_THRESHOLD,
        "ROUND_TYPE_DRAW_HEAVY_AVG": ROUND_TYPE_DRAW_HEAVY_AVG,
        "ROUND_TYPE_DRAW_LIGHT_AVG": ROUND_TYPE_DRAW_LIGHT_AVG,
        "ROUND_TYPE_DRAW_BOOST": ROUND_TYPE_DRAW_BOOST,
        "RANK_MOTIVATION_GOAL_SCALING": RANK_MOTIVATION_GOAL_SCALING,
        "WEATHER_PENALTY_HEAVY_RAIN": WEATHER_PENALTY_HEAVY_RAIN,
        "WEATHER_PENALTY_RAIN": WEATHER_PENALTY_RAIN,
        "WEATHER_PENALTY_STRONG_WIND": WEATHER_PENALTY_STRONG_WIND,
        "STATS_ASOF_DATE": STATS_ASOF_DATE,
        "STATS_SNAPSHOT_NAME": STATS_SNAPSHOT_NAME,
    }

    summary = {
        "league": LEAGUE,
        "season_year": SEASON_YEAR,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "all_teams": len(all_teams),
            "results_rows": len(df_all_results),
            "future_matches": len(df_2025_future),
            "finished_matches": len(df_2025_finished),
        },
        "accuracy": {
            "finished_games": total_finished_games,
            "correct": correct_predictions,
            "accuracy_pct": round(accuracy, 2) if total_finished_games > 0 else None,
        },
    }

    desired_pred_cols = [
        "match_id",
        "league",
        "節",
        "stadium",
        "home_team",
        "away_team",
        "stats_asof",
        "elo_diff_before_hfa",
        "hfa_added_to_diff",
        "elo_diff_after_hfa",
        "elo_diff_scaled",
        "elo_diff_for_prob",
        "prob_elo_home",
        "prob_elo_draw",
        "prob_elo_away",
        "prob_lab_home",
        "prob_lab_draw",
        "prob_lab_away",
        "lab_prob_blend_alpha",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "draw_core_score",
        "swing_close_score",
        "close_split_profile",
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
        "anchor_candidate_score",
        "anchor_candidate_reason",
        "draw_core_candidate_flag",
        "draw_core_candidate_score",
        "draw_core_candidate_reason",
        "away_overread_draw_cover_flag",
        "away_overread_draw_cover_score",
        "away_overread_draw_cover_reason",
        "match_context_title_race_home",
        "match_context_title_race_away",
        "match_context_acl_race_home",
        "match_context_acl_race_away",
        "match_context_acl_border_home",
        "match_context_acl_border_away",
        "match_context_promotion_race_home",
        "match_context_promotion_race_away",
        "match_context_po_race_home",
        "match_context_po_race_away",
        "match_context_survival_race_home",
        "match_context_survival_race_away",
        "match_context_relegation_escape_home",
        "match_context_relegation_escape_away",
        "match_context_top_direct_duel",
        "match_context_bottom_direct_duel",
        "match_context_zone_home",
        "match_context_zone_away",
        "match_context_pressure_score_home",
        "match_context_pressure_score_away",
        "draw_core_flag",
        "draw_purchase_tier",
        "primary_pick_symbol",
        "secondary_pick_symbol",
        "risk_level",
        "ticket_guidance",
        "decision_summary",
        "predicted_result_prefusion_argmax",
        "predicted_result_main_prefusion",
        "predicted_result_main",
        "main_rule_applied",
        "predicted_result_main_postfusion_argmax",
        "predicted_result_main_symbol",
        "final_result",
        "predicted_result",
        "predicted_result_source",
        "predicted_result_prob_ref",
        "decision_reason",
        "argmax_result",
        "force_draw_applied",
        "predicted_highest_prob_result",
        "argmax_raw_result",
    ]
    if "predicted_result" not in df_pred.columns:
        df_pred = apply_main_prediction_result(df_pred, "PRED_EXPORT")
    if "predicted_highest_prob_result" not in df_pred.columns:
        df_pred = recalculate_predicted_highest_prob_result(df_pred, "predicted_highest_prob_result")
    pred_cols = [c for c in desired_pred_cols if c in df_pred.columns]
    pred_list = df_pred[pred_cols].to_dict(orient="records") if not df_pred.empty else []

    desired_backtest_cols = [
        "match_id",
        "league",
        "節",
        "stadium",
        "home_team",
        "away_team",
        "elo_diff_before_hfa",
        "hfa_added_to_diff",
        "elo_diff_after_hfa",
        "elo_diff_scaled",
        "elo_diff_for_prob",
        "prob_elo_home",
        "prob_elo_draw",
        "prob_elo_away",
        "prob_lab_home",
        "prob_lab_draw",
        "prob_lab_away",
        "lab_prob_blend_alpha",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "draw_core_score",
        "swing_close_score",
        "close_split_profile",
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
        "anchor_candidate_score",
        "anchor_candidate_reason",
        "draw_core_candidate_flag",
        "draw_core_candidate_score",
        "draw_core_candidate_reason",
        "away_overread_draw_cover_flag",
        "away_overread_draw_cover_score",
        "away_overread_draw_cover_reason",
        "match_context_title_race_home",
        "match_context_title_race_away",
        "match_context_acl_race_home",
        "match_context_acl_race_away",
        "match_context_acl_border_home",
        "match_context_acl_border_away",
        "match_context_promotion_race_home",
        "match_context_promotion_race_away",
        "match_context_po_race_home",
        "match_context_po_race_away",
        "match_context_survival_race_home",
        "match_context_survival_race_away",
        "match_context_relegation_escape_home",
        "match_context_relegation_escape_away",
        "match_context_top_direct_duel",
        "match_context_bottom_direct_duel",
        "match_context_zone_home",
        "match_context_zone_away",
        "match_context_pressure_score_home",
        "match_context_pressure_score_away",
        "draw_core_flag",
        "draw_purchase_tier",
        "primary_pick_symbol",
        "secondary_pick_symbol",
        "risk_level",
        "ticket_guidance",
        "decision_summary",
        "predicted_result_prefusion_argmax",
        "predicted_result_main_prefusion",
        "predicted_result_main",
        "main_rule_applied",
        "predicted_result_main_postfusion_argmax",
        "predicted_result_main_symbol",
        "final_result",
        "predicted_result",
        "predicted_result_source",
        "predicted_result_prob_ref",
        "decision_reason",
        "argmax_result",
        "force_draw_applied",
        "predicted_highest_prob_result",
        "argmax_raw_result",
        "actual_result",
        "is_correct",
    ]
    if "predicted_result" not in df_backtest.columns:
        df_backtest = apply_main_prediction_result(df_backtest, "BACKTEST_EXPORT")
    if "predicted_highest_prob_result" not in df_backtest.columns:
        df_backtest = recalculate_predicted_highest_prob_result(df_backtest, "predicted_highest_prob_result")
    backtest_cols = [c for c in desired_backtest_cols if c in df_backtest.columns]
    backtest_list = df_backtest[backtest_cols].to_dict(orient="records") if not df_backtest.empty else []

    report = {
        "inputs": inputs,
        "parameters": params,
        "summary": summary,
        "predictions": pred_list,
        "backtest": backtest_list,
    }
    return report


def write_report():
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = build_report()
    report_json = os.path.join(REPORT_DIR, f"report_{LEAGUE}_{SEASON_YEAR}.json")
    report_md = os.path.join(REPORT_DIR, f"report_{LEAGUE}_{SEASON_YEAR}.md")

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"# 解析内訳レポート ({LEAGUE} {SEASON_YEAR})")
    lines.append("")
    lines.append("## 入力ファイル")
    for k, v in report["inputs"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## パラメータ")
    for k, v in report["parameters"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## サマリ")
    summary = report["summary"]
    lines.append(f"- league: {summary['league']}")
    lines.append(f"- season_year: {summary['season_year']}")
    lines.append(f"- generated_at: {summary['generated_at']}")
    counts = summary["counts"]
    lines.append(f"- all_teams: {counts['all_teams']}")
    lines.append(f"- results_rows: {counts['results_rows']}")
    lines.append(f"- future_matches: {counts['future_matches']}")
    lines.append(f"- finished_matches: {counts['finished_matches']}")
    acc = summary["accuracy"]
    lines.append(f"- accuracy: {acc['accuracy_pct']}% ({acc['correct']}/{acc['finished_games']})")
    lines.append("")
    lines.append("## 出力")
    lines.append(f"- predictions: {output_csv}")
    lines.append(f"- backtest: {backtest_output_csv}")
    lines.append(f"- report_json: {report_json}")
    lines.append("")

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"解析内訳レポートを出力しました: {report_json}")


write_report()
