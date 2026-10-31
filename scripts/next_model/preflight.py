"""Fail-closed checks for live BuyPlan inputs (before any output is replaced)."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


def timestamp(value):
    text = str(value).strip()
    if re.fullmatch(r"\d{8}(?:\.0)?", text):
        text = text[:8]
        result = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        result = pd.to_datetime(text, errors="coerce")
    if pd.isna(result):
        raise ValueError(f"invalid date: {value!r}")
    return result.tz_localize(JST) if result.tzinfo is None else result.tz_convert(JST)


def check_inputs(selected, order, round_id, *, as_of=None, max_age_days=7):
    if max_age_days < 0:
        raise ValueError("max_age_days must be nonnegative")
    now = timestamp(as_of if as_of is not None else datetime.now(JST))
    errors = []
    dates = {"stats_asof": [], "flab_snapshot_date": []}
    fields = [
        "stats_1試合平均得点数", "stats_1試合平均失点数", "stats_ゴール期待値",
        "stats_被ゴール期待値", "stats_1試合平均被ゴール期待値",
        "flab_expected_for_xg", "flab_expected_against_xg",
    ]
    ordered = order.set_index("match_no")
    for _, row in selected.iterrows():
        label = f"No.{row['match_no']} {row['home_team']}-{row['away_team']}"
        try:
            kickoff = timestamp(row.get("datetime"))
            if not now < kickoff <= now + pd.Timedelta(days=7):
                errors.append(f"{label}: 開催日時が実行後7日以内ではありません ({kickoff})")
            expected = unicodedata.normalize("NFKC", str(ordered.loc[int(row['match_no']), 'match_date'])).strip()
            match = re.fullmatch(r"(\d{1,2})月(\d{1,2})日", expected)
            if match:
                matches = (kickoff.month, kickoff.day) == tuple(map(int, match.groups()))
            else:
                matches = kickoff.date() == timestamp(expected).date()
            if not matches:
                errors.append(f"{label}: toto節リストと開催日が不一致")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{label}: 開催日を確認できません: {exc}")
        for column in dates:
            try:
                date = timestamp(row.get(column)).normalize()
                age = (now.normalize() - date).days
                if not 0 <= age <= max_age_days:
                    errors.append(f"{label}: {column} が未来日または期限超過 ({age}日前)")
                dates[column].append(date.date().isoformat())
            except (ValueError, TypeError) as exc:
                errors.append(f"{label}: {column} を確認できません: {exc}")
        for side in ("home", "away"):
            for field in fields:
                column = f"{field}_{side}"
                value = pd.to_numeric(row.get(column), errors="coerce")
                # The current lambda implementation substitutes nonpositive rates.
                if pd.isna(value) or not np.isfinite(value) or value <= 0:
                    errors.append(f"{label}: {column} が欠損・非数値・非正値 ({row.get(column)!r})")
            total = pd.to_numeric(row.get(f"stats_被ゴール期待値_{side}"), errors="coerce")
            rate = pd.to_numeric(row.get(f"stats_1試合平均被ゴール期待値_{side}"), errors="coerce")
            if pd.notna(total) and pd.notna(rate) and rate > 0 and total / rate < 0.5:
                errors.append(f"{label}: {side} のxGAから試合数を復元できません")
    if errors:
        raise ValueError("事前チェック失敗。入力を更新・修正してください。\n" + "\n".join(errors))
    return {
        "status": "passed", "round_id": round_id, "checked_at": now.isoformat(),
        "max_age_days": max_age_days, "matches": len(selected),
        "source_dates": {key: sorted(set(values)) for key, values in dates.items()},
        "checks": ["kickoff_in_next_7_days", "order_match_date", "source_freshness", "positive_finite_model_inputs"],
    }
