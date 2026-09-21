#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

from http_retry import get_with_retry


BASE_URL = "https://www.jleague.jp"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
CLUB_TOP_URL = f"{BASE_URL}/club/"
AJAX_PLAYER_URL = f"{BASE_URL}/club/ajax_player/"
DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


@dataclass
class FetchResult:
    club_key: str
    club_name: str
    ok: bool
    row_count: int
    warning: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="J.LEAGUEクラブの選手一覧を、robots配慮付きで取得します。"
    )
    p.add_argument("--club-keys", default="", help="カンマ区切りクラブキー（例: kashima,urawa）")
    p.add_argument("--out", default="data/manual/jleague_club_players.csv", help="出力CSV")
    p.add_argument("--summary-out", default="data/manual/jleague_club_players_summary.json", help="実行サマリJSON")
    p.add_argument("--sleep-sec", type=float, default=1.5, help="リクエスト間隔（秒）")
    p.add_argument("--timeout-sec", type=float, default=20.0, help="HTTPタイムアウト（秒）")
    p.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent")
    p.add_argument("--max-clubs", type=int, default=0, help="先頭Nクラブのみ取得（0=全件）")
    p.add_argument(
        "--allow-disallowed",
        action="store_true",
        help="robots.txtで非許可でも処理続行（通常は非推奨）",
    )
    return p.parse_args()


def normalize_text(v: object) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def normalize_club_name(v: str) -> str:
    s = normalize_text(v)
    if not s:
        return s
    m = re.match(r"^(.+?)\s+\1$", s)
    if m:
        return m.group(1)
    parts = s.split(" ")
    if len(parts) == 2 and parts[0] == parts[1]:
        return parts[0]
    half = len(s) // 2
    if len(s) % 2 == 0 and s[:half] == s[half:]:
        return s[:half]
    return s


def load_robots(user_agent: str, timeout_sec: float, headers: Dict[str, str]) -> RobotFileParser:
    rp = RobotFileParser()
    resp = get_with_retry(
        ROBOTS_URL,
        headers=headers,
        timeout=(5, timeout_sec),
        max_retries=3,
        backoff_base=1.0,
    )
    rp.parse(resp.text.splitlines())
    return rp


def ensure_robot_allowed(
    rp: RobotFileParser,
    user_agent: str,
    target_url: str,
    allow_disallowed: bool,
) -> None:
    allowed = rp.can_fetch(user_agent, target_url)
    if allowed:
        print(f"[ROBOTS] OK: {target_url}")
        return
    msg = f"[ROBOTS] DISALLOW: {target_url}"
    if allow_disallowed:
        print(f"{msg} (allow-disallowed=True のため続行)")
        return
    raise RuntimeError(f"{msg} / --allow-disallowed 指定なし")


def discover_club_keys(headers: Dict[str, str], timeout_sec: float) -> List[Tuple[str, str]]:
    resp = get_with_retry(
        CLUB_TOP_URL,
        headers=headers,
        timeout=(5, timeout_sec),
        max_retries=3,
        backoff_base=1.0,
    )
    soup = BeautifulSoup(resp.text, "lxml")
    pairs: List[Tuple[str, str]] = []
    seen = set()
    select = soup.find("select", attrs={"name": "clubName"})
    if select is not None:
        for opt in select.find_all("option"):
            key = normalize_text(opt.get("value", ""))
            name = normalize_text(opt.get_text(" ", strip=True))
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key, name or key))

    # /club/<slug>/day/ 形式のリンクから抽出（現行クラブ一覧）
    if not pairs:
        for a in soup.find_all("a", href=True):
            href = normalize_text(a.get("href", ""))
            m = re.match(r"^/club/([^/]+)/day/?$", href)
            if not m:
                continue
            key = normalize_text(m.group(1))
            if not key or key in seen:
                continue
            seen.add(key)
            name = normalize_club_name(normalize_text(a.get_text(" ", strip=True))) or key
            pairs.append((key, name))

    if not pairs:
        raise RuntimeError("クラブキーを抽出できませんでした")
    return pairs


def parse_player_table(fragment_html: str) -> pd.DataFrame:
    soup = BeautifulSoup(fragment_html, "lxml")
    table = None

    for t in soup.find_all("table"):
        text = normalize_text(t.get_text(" ", strip=True))
        if "背番号" in text and ("選手名" in text or "名前" in text):
            table = t
            break
    if table is None:
        table = soup.find("table")
    if table is None:
        raise RuntimeError("選手一覧テーブルを検出できませんでした")

    header_cells = table.find("thead")
    headers: List[str] = []
    if header_cells:
        headers = [normalize_text(th.get_text(" ", strip=True)) for th in header_cells.find_all(["th", "td"])]

    body = table.find("tbody") or table
    rows: List[List[str]] = []
    for tr in body.find_all("tr"):
        cells = [normalize_text(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        if not cells:
            continue
        if headers and len(cells) > len(headers):
            cells = cells[: len(headers)]
        rows.append(cells)

    if not rows:
        raise RuntimeError("選手行データを抽出できませんでした")

    max_len = max(len(r) for r in rows)
    if not headers:
        headers = [f"col_{i+1:02d}" for i in range(max_len)]
    if len(headers) < max_len:
        headers = headers + [f"extra_{i+1:02d}" for i in range(max_len - len(headers))]

    aligned = [r + [""] * (len(headers) - len(r)) for r in rows]
    out = pd.DataFrame(aligned, columns=headers)
    return out


def pick_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lowered = [(c, normalize_text(c).lower()) for c in cols]
    for p in patterns:
        for c, lc in lowered:
            if p in lc:
                return c
    return None


def canonicalize_player_df(df: pd.DataFrame) -> pd.DataFrame:
    no_col = pick_column(df, ["背番号", "no"])
    pos_col = pick_column(df, ["pos", "ポジション"])
    name_col = pick_column(df, ["選手名", "名前", "name"])
    birth_col = pick_column(df, ["生年月日", "生年月", "birthday"])
    hw_col = pick_column(df, ["身長", "体重", "height", "weight"])

    out = df.copy()
    out["player_no"] = out[no_col] if no_col else ""
    out["position"] = out[pos_col] if pos_col else ""
    out["player_name"] = out[name_col] if name_col else ""
    out["birth"] = out[birth_col] if birth_col else ""
    out["height_weight"] = out[hw_col] if hw_col else ""
    return out


def parse_current_player_page(page_html: str) -> pd.DataFrame:
    soup = BeautifulSoup(page_html, "lxml")
    rows = []
    for tr in soup.select("table tbody tr"):
        link = tr.select_one("a.o-table__player-name-link")
        if link is None:
            continue
        def cell(suffix):
            node = tr.select_one(f".o-table__cell--{suffix}")
            if node is None:
                raise RuntimeError(f"選手一覧の必須列がありません: {suffix}")
            return node.get_text(" ", strip=True)
        position = tr.select_one(".o-table__player-position")
        if position is None:
            raise RuntimeError("選手のポジション・背番号がありません")
        parts = position.get_text(" ", strip=True).split(maxsplit=1)
        rows.append({
            "player_name": link.get_text(" ", strip=True),
            "position": parts[0], "player_no": parts[1] if len(parts) > 1 else "",
            "birth": cell("date-of-birth"), "height_weight": cell("height-weight"),
            "出場 試合数 ※2": cell("number-of-games-played"),
            "ゴール 数 ※3": cell("goals-scored"),
        })
    if not rows:
        raise RuntimeError("公式選手一覧ページから選手を取得できませんでした")
    return pd.DataFrame(rows).drop_duplicates(["player_name", "birth"])


def fetch_one_club(
    club_key: str,
    club_name: str,
    headers: Dict[str, str],
    timeout_sec: float,
) -> Tuple[pd.DataFrame, str]:
    url = f"{BASE_URL}/club/{club_key}/player/"
    resp = get_with_retry(url, headers=headers, timeout=(5, timeout_sec),
                          max_retries=3, backoff_base=1.0)
    canon = parse_current_player_page(resp.text)
    soup = BeautifulSoup(resp.text, "lxml")
    heading = soup.select_one(".p-club-details-header__team-name-ja")
    if heading is None:
        raise RuntimeError("クラブ名を取得できませんでした")
    canon["club_key"] = club_key
    canon["club_name"] = heading.get_text(" ", strip=True)
    canon["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    canon["source_url"] = url
    return canon, ""


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    headers = {"User-Agent": args.user_agent}
    warnings: List[str] = []
    results: List[FetchResult] = []

    try:
        rp = load_robots(args.user_agent, args.timeout_sec, headers)
        ensure_robot_allowed(rp, args.user_agent, CLUB_TOP_URL, args.allow_disallowed)

    except Exception as e:
        print(f"[ERROR] robots確認失敗: {e}")
        return 1

    if args.club_keys.strip():
        keys = [normalize_text(x) for x in args.club_keys.split(",") if normalize_text(x)]
        club_pairs = [(k, k) for k in keys]
    else:
        try:
            club_pairs = discover_club_keys(headers, args.timeout_sec)
        except Exception as e:
            print(f"[ERROR] クラブ一覧取得失敗: {e}")
            return 1

    if args.max_clubs and args.max_clubs > 0:
        club_pairs = club_pairs[: args.max_clubs]

    all_rows: List[pd.DataFrame] = []
    for idx, (club_key, club_name) in enumerate(club_pairs, start=1):
        try:
            ensure_robot_allowed(rp, args.user_agent, f"{BASE_URL}/club/{club_key}/player/", args.allow_disallowed)
            df_one, warn = fetch_one_club(club_key, club_name, headers, args.timeout_sec)
            if warn:
                warnings.append(f"{club_key}: {warn}")
            all_rows.append(df_one)
            results.append(FetchResult(club_key=club_key, club_name=club_name, ok=True, row_count=len(df_one)))
            print(f"[OK] {idx}/{len(club_pairs)} {club_key} rows={len(df_one)}")
        except Exception as e:
            msg = str(e)
            results.append(FetchResult(club_key=club_key, club_name=club_name, ok=False, row_count=0, error=msg))
            warnings.append(f"{club_key}: {msg}")
            print(f"[WARN] {idx}/{len(club_pairs)} {club_key} failed: {msg}")

        if idx < len(club_pairs):
            time.sleep(max(0.0, args.sleep_sec))

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
    else:
        out_df = pd.DataFrame()

    if any(not r.ok for r in results) or out_df.empty:
        print("[ERROR] 選手取得に失敗したため既存CSVを更新しません。")
        return 1
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "robots_url": ROBOTS_URL,
        "club_top_url": CLUB_TOP_URL,
        "ajax_player_url": AJAX_PLAYER_URL,
        "user_agent": args.user_agent,
        "sleep_sec": args.sleep_sec,
        "requested_clubs": len(club_pairs),
        "success_clubs": sum(1 for r in results if r.ok),
        "failed_clubs": sum(1 for r in results if not r.ok),
        "rows": int(len(out_df)),
        "warnings": warnings,
        "results": [r.__dict__ for r in results],
        "notice": "本サイトの利用規約・著作権規定に従って利用してください。無断転載等は行わないでください。",
    }
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] csv: {args.out}")
    print(f"[OK] summary: {args.summary_out}")
    if warnings:
        print(f"[WARN] warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted")
        raise SystemExit(130)
