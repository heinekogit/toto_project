"""Parser for the current J.LEAGUE official standings pages."""

import unicodedata

import pandas as pd
from bs4 import BeautifulSoup


CANONICAL_COLUMNS = [
    "順位", "チーム", "勝点", "試合", "勝", "分", "負",
    "得点", "失点", "得失点差", "直近試合の勝敗",
]


def _header_name(value):
    text = unicodedata.normalize("NFKC", str(value or "")).replace(" ", "")
    checks = [
        ("順位", "順位"), ("クラブ", "チーム"), ("チーム", "チーム"),
        ("勝点", "勝点"), ("直近", "直近試合の勝敗"), ("試合", "試合"),
        ("得失", "得失点差"), ("得点", "得点"), ("失点", "失点"),
    ]
    for token, canonical in checks:
        if token in text:
            return canonical
    if text.startswith("勝"):
        return "勝"
    if text.startswith("分"):
        return "分"
    if text.startswith("負"):
        return "負"
    return text


def parse_official_standings(html, *, season, league, fetched_date):
    soup = BeautifulSoup(html, "lxml")
    table = None
    for candidate in soup.find_all("table"):
        headers = [_header_name(th.get_text(" ", strip=True)) for th in candidate.find_all("th")]
        if "順位" in headers and "チーム" in headers and "勝点" in headers:
            table = candidate
            break
    if table is None:
        raise RuntimeError("公式順位ページに順位表テーブルがありません。")

    headers = [_header_name(th.get_text(" ", strip=True)) for th in table.find_all("th")]
    records = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells or all(cell.name == "th" for cell in cells):
            continue
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        row = dict(zip(headers, values[:len(headers)]))
        team = str(row.get("チーム", "")).strip()
        if team:
            records.append(row)
    if not records:
        raise RuntimeError("公式順位ページからクラブ行を取得できませんでした。")

    df = pd.DataFrame(records)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    numeric_cols = ["順位", "勝点", "試合", "勝", "分", "負", "得点", "失点", "得失点差"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].replace({"-": None, "−": None, "": None}), errors="coerce")

    # 全クラブの試合数が未公開なら、失敗ではなく開幕前として扱う。
    preseason = bool(df["試合"].isna().all() or df["試合"].fillna(0).eq(0).all())
    if preseason:
        df["勝点"] = 0
        for col in ["試合", "勝", "分", "負", "得点", "失点", "得失点差"]:
            df[col] = 0

    df["season"] = str(season)
    df["round"] = "開幕前" if preseason else "最新節"
    df["fetched_date"] = str(fetched_date)
    df["competition_key"] = f"regular_{season}_27" if str(season) == "2026" else f"regular_{season}"
    df["preseason"] = int(preseason)
    return df[CANONICAL_COLUMNS + ["season", "round", "fetched_date", "competition_key", "preseason"]]
