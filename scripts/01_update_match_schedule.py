import os
import time
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.parse import urlparse, parse_qs
from http_retry import get_with_retry
from jleague_competition import resolve_competition_ids
from jleague_matches import build_match_id, parse_match_datetimes, validate_match_identity


SEASON_YEAR = os.environ.get("SEASON_YEAR", "2025")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
COMPETITION_YEARS = os.environ.get("COMPETITION_YEARS", SEASON_YEAR)

DEFAULT_FRAME_ID = "1"
if LEAGUE == "j2":
    DEFAULT_FRAME_ID = "2"
elif LEAGUE == "j3":
    DEFAULT_FRAME_ID = "3"
COMPETITION_FRAME_IDS = os.environ.get("COMPETITION_FRAME_IDS", DEFAULT_FRAME_ID)
COMPETITION_IDS_ENV = os.environ.get("COMPETITION_IDS")
competition_id = ""
TARGET_URL = "https://data.j-league.or.jp/SFMS01/search"

OUTPUT_CSV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUT_CSV_PATH = os.path.join(OUTPUT_CSV_DIR, f"{LEAGUE}_{SEASON_YEAR}_upcoming.csv")
TEMP_HTML_PATH = os.path.abspath(os.path.join(OUTPUT_CSV_DIR, "temp_match_schedule.html"))


def _parse_id_set(text):
    return {v.strip() for v in str(text).split(",") if v.strip()}


EXPECTED_COMPETITION_IDS = set()
EXPECTED_COMPETITION_YEARS = str(COMPETITION_YEARS).strip()
EXPECTED_FRAME_IDS = _parse_id_set(COMPETITION_FRAME_IDS)


def _configure_competition():
    global competition_id, TARGET_URL, EXPECTED_COMPETITION_IDS
    competition_id = resolve_competition_ids(
        COMPETITION_YEARS,
        COMPETITION_FRAME_IDS,
        COMPETITION_IDS_ENV,
    )
    EXPECTED_COMPETITION_IDS = _parse_id_set(competition_id)
    TARGET_URL = (
        "https://data.j-league.or.jp/SFMS01/search?"
        f"competition_years={COMPETITION_YEARS}&competition_frame_ids={COMPETITION_FRAME_IDS}"
        f"&competition_ids={competition_id}&tv_relay_station_name="
    )


def _find_match_table(soup):
    return soup.select_one("table.table-base00.search-table") or soup.select_one("table.search-table")


def _extract_tab_urls(soup, base_url, allowed_frame_ids=None):
    urls = []
    seen = set()
    for a in soup.select("ul.search-link a.tab[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        if "jleague.jp" in href:
            continue
        if "/SFMS01/search" not in href:
            continue
        full_url = urljoin(base_url, href)
        try:
            qs = parse_qs(urlparse(full_url).query)
            years = (qs.get("competition_years") or [""])[0].strip()
            frame_id = (qs.get("competition_frame_ids") or [""])[0].strip()
            if years != EXPECTED_COMPETITION_YEARS:
                continue
            if allowed_frame_ids and frame_id and frame_id not in allowed_frame_ids:
                continue
            ids = _parse_id_set((qs.get("competition_ids") or [""])[0])
            if not ids or not (ids & EXPECTED_COMPETITION_IDS):
                continue
        except Exception:
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        urls.append(full_url)
    return urls


def _extract_frame_id(url):
    try:
        qs = parse_qs(urlparse(url).query)
        values = qs.get("competition_frame_ids")
        if values:
            return values[0]
    except Exception:
        return None
    return None


def _sort_tab_urls_by_preference(urls, preferred_frame_id):
    if not preferred_frame_id:
        return urls
    preferred = []
    others = []
    for u in urls:
        if _extract_frame_id(u) == str(preferred_frame_id):
            preferred.append(u)
        else:
            others.append(u)
    return preferred + others


def _build_candidate_urls(soup, base_url, preferred_frame_id):
    fallback_frames = set(EXPECTED_FRAME_IDS)
    if preferred_frame_id:
        fallback_frames.add(str(preferred_frame_id))
    tab_urls = _extract_tab_urls(soup, base_url, allowed_frame_ids=fallback_frames)
    tab_urls = _sort_tab_urls_by_preference(tab_urls, preferred_frame_id)

    manual_urls = []
    for frame_id in sorted(fallback_frames):
        manual_urls.append(
            "https://data.j-league.or.jp/SFMS01/search?"
            f"competition_years={EXPECTED_COMPETITION_YEARS}"
            f"&competition_frame_ids={frame_id}"
            f"&competition_ids={competition_id}"
            "&tv_relay_station_name="
        )

    candidates = [base_url] + tab_urls + manual_urls
    unique = []
    seen = set()
    for u in candidates:
        if u in seen:
            continue
        seen.add(u)
        unique.append(u)
    return unique


def _extract_table_rows(match_table):
    header_row = match_table.find("thead").find("tr") if match_table.find("thead") else None
    headers = [th.text.strip() for th in header_row.find_all("th")] if header_row else []
    table_data = []

    tbody = match_table.find("tbody")
    if not tbody:
        return headers, table_data

    for row in tbody.find_all("tr"):
        row_data = []
        for cell in row.find_all(["th", "td"]):
            link = cell.find("a")
            row_data.append(link.text.strip() if link else cell.text.strip())
        table_data.append(row_data)
    return headers, table_data


def _load_ranked_teams(results_csv_path):
    if not os.path.exists(results_csv_path):
        return []
    try:
        df = pd.read_csv(results_csv_path)
    except Exception:
        return []
    required = {"home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(df.columns):
        return []
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return []
    teams = sorted(set(df["home_team"].astype(str).str.strip()) | set(df["away_team"].astype(str).str.strip()))
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
    return sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)


def _estimate_allowed_teams_for_league():
    if LEAGUE not in {"j1", "j2"}:
        return None
    try:
        prev_year = str(int(SEASON_YEAR) - 1)
    except Exception:
        return None
    j1_prev = os.path.join(OUTPUT_CSV_DIR, f"j1_{prev_year}_latest_results.csv")
    j2_prev = os.path.join(OUTPUT_CSV_DIR, f"j2_{prev_year}_latest_results.csv")
    j1_rank = _load_ranked_teams(j1_prev)
    j2_rank = _load_ranked_teams(j2_prev)
    if not j1_rank or not j2_rank:
        return None
    promotions = set(j2_rank[:3])
    relegations = set(j1_rank[-3:])
    j1_set = set(j1_rank)
    j2_set = set(j2_rank)
    if LEAGUE == "j1":
        return (j1_set - relegations) | promotions
    return (j2_set - promotions) | relegations


def scrape_match_schedule():
    _configure_competition()
    print(f"試合日程をスクレイピング中: {TARGET_URL}")
    print(f"出力CSVパス: {OUTPUT_CSV_PATH}")

    if not os.path.exists(OUTPUT_CSV_DIR):
        os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

    try:
        response = get_with_retry(TARGET_URL, timeout=(5, 20), max_retries=3)
        response.raise_for_status()
        root_html_text = response.text
        root_soup = BeautifulSoup(root_html_text, "lxml")

        preferred_fallback_frame_id = os.environ.get("FALLBACK_FRAME_ID", DEFAULT_FRAME_ID)
        candidate_urls = _build_candidate_urls(root_soup, TARGET_URL, preferred_fallback_frame_id)
        tables = []

        for idx, url in enumerate(candidate_urls):
            if idx == 0:
                html_text = root_html_text
                soup = root_soup
            else:
                print(f"タブURLを取得します: {url}")
                tab_resp = get_with_retry(url, timeout=(5, 20), max_retries=3)
                tab_resp.raise_for_status()
                html_text = tab_resp.text
                soup = BeautifulSoup(html_text, "lxml")

            match_table = _find_match_table(soup)
            if not match_table:
                continue
            headers, table_data = _extract_table_rows(match_table)
            if not headers:
                if table_data:
                    headers = table_data[0]
                    table_data = table_data[1:]
                else:
                    continue
            tables.append(pd.DataFrame(table_data, columns=headers))

        with open(TEMP_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(root_html_text)

        if not tables:
            raise RuntimeError("試合日程テーブルが見つかりませんでした。")

        df = pd.concat(tables, ignore_index=True, sort=False)

        # 日付整形（K/O時刻が「未定」「-」「空」などで欠損するケースがあるため、日付のみでも補完）
        # 例: K/O時刻 に「未定」等が入ると to_datetime が NaT になり match_id が空になる
        df["datetime"] = parse_match_datetimes(df["試合日"], df["K/O時刻"])

        # 以降は不要なので落とす
        df = df.drop(columns=["試合日", "K/O時刻"])

        df = df.rename(
            columns={
                "節": "節",
                "ホーム": "home_team",
                "スコア": "score_full",
                "アウェイ": "away_team",
                "スタジアム": "stadium",
            }
        )

        # スコア分解（未開催はNaN）
        score_parts = df["score_full"].astype(str).str.extract(r"^\s*(\d+)\s*[-－]\s*(\d+)\s*$")
        df["home_score"] = pd.to_numeric(score_parts[0], errors="coerce")
        df["away_score"] = pd.to_numeric(score_parts[1], errors="coerce")
        df = df.drop(columns=["score_full"])

        df = df.drop_duplicates(subset=["節", "datetime", "home_team", "away_team"], keep="first")

        # competition_year/frame/competition_idで公式側がリーグを確定済み。
        # 前年順位からの所属推定は昇降格・シーズン移行時に正規カードを削るため適用しない。
        print(f"[LEAGUE_SCOPE] official competition filterを採用: league={LEAGUE} rows={len(df)}")

        # match_id生成
        df["match_id"] = df.apply(
            lambda row: build_match_id(
                LEAGUE, SEASON_YEAR, row.get("節"), row.get("datetime"),
                row.get("home_team"), row.get("away_team"),
            ),
            axis=1,
        )
        validate_match_identity(df, label=f"{LEAGUE}_{SEASON_YEAR}_schedule")

        # 未開催試合のみ抽出
        df_upcoming = df[df["home_score"].isna() & df["away_score"].isna()].copy()

        # カラム順序の整形
        desired_columns = ["節", "match_id", "datetime", "stadium", "home_team", "away_team", "home_score", "away_score"]
        df_upcoming = df_upcoming[[c for c in desired_columns if c in df_upcoming.columns]]

        if df_upcoming.empty:
            raise RuntimeError("取得データが0件です。")

        df_upcoming.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        if (not os.path.exists(OUTPUT_CSV_PATH)) or os.path.getsize(OUTPUT_CSV_PATH) == 0:
            raise RuntimeError("CSV書き込みに失敗しました（ファイル未生成/空）。")
        try:
            df_written = pd.read_csv(OUTPUT_CSV_PATH)
        except Exception as e:
            raise RuntimeError(f"CSV書き込み後の検証に失敗しました: {e}") from e
        if df_written.empty:
            raise RuntimeError("CSV書き込み後の行数が0件です。")
        print(f"SUCCESS: 試合日程データを {OUTPUT_CSV_PATH} に保存しました。 rows={len(df_written)}")
        return len(df_written)
    finally:
        time.sleep(1)


if __name__ == "__main__":
    try:
        scrape_match_schedule()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: ネットワークエラー {TARGET_URL}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: エラー発生 {TARGET_URL}: {e}")
        sys.exit(1)
