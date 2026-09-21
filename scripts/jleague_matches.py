"""Shared normalization helpers for J.League schedule/result rows."""

import re
import unicodedata

import pandas as pd


def parse_match_datetimes(match_dates, kickoff_times):
    """Parse exact kickoff timestamps, falling back to midnight when only a date exists."""
    dates = match_dates.astype(str).str.replace(r"\s*\(.+\)\s*", "", regex=True).str.strip()
    kickoff = kickoff_times.astype(str).str.extract(r"(\d{1,2}:\d{2})")[0]
    datetimes = pd.to_datetime(
        dates + " " + kickoff.fillna(""),
        format="%y/%m/%d %H:%M",
        errors="coerce",
    )
    date_only = pd.to_datetime(dates, format="%y/%m/%d", errors="coerce")
    fallback = datetimes.isna() & date_only.notna()
    datetimes.loc[fallback] = date_only.loc[fallback]
    return datetimes


def _identity_text(value):
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "_", text)


def build_match_id(league, season, round_name, match_datetime, home_team, away_team):
    """Build a stable non-empty ID even when the official date is still undecided."""
    if pd.notna(match_datetime):
        identity = pd.Timestamp(match_datetime).strftime("%m%d%H%M")
    else:
        round_key = _identity_text(round_name) or "round_unknown"
        identity = f"round_{round_key}"
    parts = [league, season, identity, home_team, away_team]
    return _identity_text("_".join(str(part) for part in parts))


def validate_match_identity(df, *, label):
    """Reject blank or duplicate match IDs; report truthful unknown datetimes."""
    if "match_id" not in df.columns:
        raise RuntimeError(f"{label}: match_id列がありません。")
    ids = df["match_id"].fillna("").astype(str).str.strip()
    blank_count = int(ids.eq("").sum())
    duplicate_count = int(ids.duplicated(keep=False).sum())
    if blank_count or duplicate_count:
        duplicate_samples = ids[ids.duplicated(keep=False)].head(10).tolist()
        raise RuntimeError(
            f"{label}: match_id検証失敗 blank={blank_count} "
            f"duplicate_rows={duplicate_count} samples={duplicate_samples}"
        )
    datetime_missing = int(df["datetime"].isna().sum()) if "datetime" in df.columns else len(df)
    if datetime_missing:
        print(
            f"[MATCH_ID][WARN] {label}: 公式日程未定のためdatetime欠損={datetime_missing}; "
            "match_idは節・対戦カードから生成"
        )
    print(
        f"[MATCH_ID] {label}: rows={len(df)} blank=0 duplicate_rows=0 "
        f"datetime_missing={datetime_missing}"
    )
