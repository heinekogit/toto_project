#!/usr/bin/env python3
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
EVAL_ROOT = ROOT_DIR / "data" / "eval" / "rounds"
BACKTEST_FILES = [
    ROOT_DIR / "backtest_j1_2026.csv",
    ROOT_DIR / "backtest_j2_2026.csv",
]


def _norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _norm_league(value: object) -> str:
    text = _norm_text(value).upper()
    if text in {"J1", "J2", "J3"}:
        return text
    return ""


def _round_no_from_text(value: object) -> int | None:
    text = _norm_text(value)
    m = re.search(r"第\s*([0-9]+)\s*節", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_backtest_master() -> pd.DataFrame:
    frames = []
    for path in BACKTEST_FILES:
        if not path.exists():
            raise FileNotFoundError(f"backtest file not found: {path}")
        df = pd.read_csv(path, keep_default_na=False)
        df["league_norm"] = df.get("league", pd.Series(dtype=str)).map(_norm_league)
        df["home_key"] = df["home_team"].map(_norm_text)
        df["away_key"] = df["away_team"].map(_norm_text)
        df["round_no"] = df.get("節", pd.Series(dtype=str)).map(_round_no_from_text)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["league_norm", "round_no", "match_id", "home_team", "away_team"]).reset_index(drop=True)
    return out


def build_round_snapshot(master: pd.DataFrame, actual_csv: Path, round_no: int) -> pd.DataFrame:
    actual = pd.read_csv(actual_csv, keep_default_na=False)
    actual["league_norm"] = actual.get("league", pd.Series(dtype=str)).map(_norm_league)
    actual["home_key"] = actual["home_team"].map(_norm_text)
    actual["away_key"] = actual["away_team"].map(_norm_text)
    actual["match_id_key"] = actual.get("match_id", pd.Series(dtype=str)).map(_norm_text)
    actual["__order"] = range(len(actual))

    master_work = master.copy()
    master_work["match_id_key"] = master_work.get("match_id", pd.Series(dtype=str)).map(_norm_text)

    merged = actual.merge(
        master_work,
        on=["match_id_key"],
        how="left",
        suffixes=("_actual", ""),
    )

    missing_mask = merged.get("match_id", pd.Series(dtype=str)).isna() | merged.get("match_id", pd.Series(dtype=str)).astype(str).eq("")
    if bool(missing_mask.any()):
        fallback_actual = merged.loc[missing_mask, [c for c in actual.columns if c in merged.columns]].copy()
        fallback_known = fallback_actual[fallback_actual["league_norm"] != ""].merge(
            master_work,
            on=["league_norm", "home_key", "away_key"],
            how="left",
            suffixes=("_actual", ""),
        )
        fallback_unknown = fallback_actual[fallback_actual["league_norm"] == ""].merge(
            master_work,
            on=["home_key", "away_key"],
            how="left",
            suffixes=("_actual", ""),
        )
        fallback = pd.concat([fallback_known, fallback_unknown], ignore_index=True)
        if not fallback.empty:
            fallback = fallback.set_index("__order")
            merged = merged.set_index("__order")
            for idx, row in fallback.iterrows():
                for col, value in row.items():
                    if col not in merged.columns:
                        merged[col] = ""
                    if pd.isna(merged.at[idx, col]) or str(merged.at[idx, col]) == "":
                        merged.at[idx, col] = value
            merged = merged.reset_index()

    missing = merged["match_id"].astype(str).eq("").sum() + merged["match_id"].isna().sum()
    if missing:
        missing_rows = merged[merged.get("match_id").isna() | merged.get("match_id").astype(str).eq("")]
        detail = missing_rows[["league_actual", "home_team_actual", "away_team_actual"]] if "league_actual" in missing_rows.columns else missing_rows[["home_team_actual", "away_team_actual"]]
        raise RuntimeError(f"missing regenerated rows for round{round_no:02d}: {detail.to_dict(orient='records')}")

    merged = merged.sort_values("__order").reset_index(drop=True)

    prefer_cols = [c for c in merged.columns if not c.endswith("_actual")]
    out = merged[prefer_cols].copy()
    drop_cols = ["league_norm", "home_key", "away_key", "round_no", "__order"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    return out


def main() -> None:
    master = load_backtest_master()
    round_dirs = sorted([p for p in EVAL_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"round\d+", p.name)])
    updated = []
    skipped = []
    for round_dir in round_dirs:
        round_no = int(round_dir.name.replace("round", ""))
        if round_no < 2:
            continue
        actual_csv = round_dir / "actual_results.csv"
        snapshot_dir = round_dir / "snapshot"
        snapshot_csv = snapshot_dir / "predictions.csv"
        if not actual_csv.exists():
            continue
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            out = build_round_snapshot(master, actual_csv, round_no)
        except RuntimeError as exc:
            skipped.append((round_dir.name, str(exc)))
            continue
        if out.empty:
            continue
        out.to_csv(snapshot_csv, index=False, encoding="utf-8-sig")
        updated.append((round_dir.name, len(out)))

    for name, rows in updated:
        print(f"[OK] {name} rows={rows}")
    for name, reason in skipped:
        print(f"[SKIP] {name} {reason}")
    print(f"[DONE] updated_rounds={len(updated)} skipped_rounds={len(skipped)}")


if __name__ == "__main__":
    main()
