import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import os
import sys
import traceback
import unicodedata
from urllib.parse import urljoin
from urllib.parse import urlparse, parse_qs
from http_retry import get_with_retry

SEASON_YEAR = os.environ.get("SEASON_YEAR", "2025")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
_competition_years_env = os.environ.get("COMPETITION_YEARS")
_special_mode_auto = _competition_years_env is None and LEAGUE in {"j1", "j2"} and int(SEASON_YEAR) >= 2026
COMPETITION_YEARS = _competition_years_env or (f"{SEASON_YEAR}1" if _special_mode_auto else SEASON_YEAR)
TRANSITION_2026_MODE = str(COMPETITION_YEARS).endswith("1") and len(str(COMPETITION_YEARS)) == 5 and LEAGUE in {"j1", "j2"}

DEFAULT_FRAME_ID = "1"
if LEAGUE == "j2":
    DEFAULT_FRAME_ID = "2"
elif LEAGUE == "j3":
    DEFAULT_FRAME_ID = "3"
if TRANSITION_2026_MODE:
    DEFAULT_FRAME_ID = "35" if LEAGUE == "j1" else "36"
COMPETITION_FRAME_IDS = os.environ.get("COMPETITION_FRAME_IDS", DEFAULT_FRAME_ID)
COMPETITION_IDS_ENV = os.environ.get("COMPETITION_IDS")
COMPETITION_ID_MAP = {
    "j1": "651",
    "j2": "655",
}
competition_id = COMPETITION_IDS_ENV or COMPETITION_ID_MAP.get(LEAGUE, "651")
if TRANSITION_2026_MODE:
    TARGET_URL = (
        f"https://data.j-league.or.jp/SFMS01/search?"
        f"competition_years={COMPETITION_YEARS}&competition_frame_ids={COMPETITION_FRAME_IDS}"
    )
else:
    TARGET_URL = (
        f"https://data.j-league.or.jp/SFMS01/search?"
        f"competition_years={COMPETITION_YEARS}&competition_frame_ids={COMPETITION_FRAME_IDS}"
        f"&competition_ids={competition_id}&tv_relay_station_name="
    )
OUTPUT_CSV_FILENAME = f"{LEAGUE}_{SEASON_YEAR}_latest_results.csv"
OUTPUT_CSV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUT_CSV_PATH = os.path.join(OUTPUT_CSV_DIR, OUTPUT_CSV_FILENAME)

TEMP_HTML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "temp_match_results.html")) # 一時HTMLファイルパス

FALLBACK_FRAME_ID_BY_LEAGUE = {
    "j1": "35",
    "j2": "36",
    "j3": "36",
}


def _parse_id_set(text):
    return {v.strip() for v in str(text).split(",") if v.strip()}


EXPECTED_COMPETITION_IDS = _parse_id_set(competition_id)
EXPECTED_COMPETITION_YEARS = str(COMPETITION_YEARS).strip()
EXPECTED_FRAME_IDS = _parse_id_set(COMPETITION_FRAME_IDS)


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
            # 2026移行モードは未知フレームが追加される可能性があるため、
            # frame_id で候補を絞り込みすぎない。
            if (not TRANSITION_2026_MODE) and allowed_frame_ids and frame_id and frame_id not in allowed_frame_ids:
                continue
            ids = _parse_id_set((qs.get("competition_ids") or [""])[0])
            if years != EXPECTED_COMPETITION_YEARS:
                continue
            # 2026移行モードは competition_ids が不定（分割テーブル/タブ差し替えあり）なため
            # 年度一致のみで候補化し、後段で重複除去する。
            if TRANSITION_2026_MODE:
                # 特殊モードでも competition_ids が付いている場合はリーグ一致を優先
                if ids and not (ids & EXPECTED_COMPETITION_IDS):
                    continue
            else:
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
    if LEAGUE == "j1":
        # 2026移行モードではJ1が複数フレームに分割されるため両方候補化
        fallback_frames.update({"35", "36"} if TRANSITION_2026_MODE else {"35"})
    elif LEAGUE == "j2":
        # 同様にJ2側も取りこぼし回避のため両方候補化
        fallback_frames.update({"36", "35"} if TRANSITION_2026_MODE else {"36"})

    tab_urls = _extract_tab_urls(soup, base_url, allowed_frame_ids=fallback_frames)
    tab_urls = _sort_tab_urls_by_preference(tab_urls, preferred_frame_id)

    manual_urls = []
    for frame_id in sorted(fallback_frames):
        if TRANSITION_2026_MODE:
            manual_urls.append(
                "https://data.j-league.or.jp/SFMS01/search?"
                f"competition_years={EXPECTED_COMPETITION_YEARS}"
                f"&competition_frame_ids={frame_id}"
            )
        else:
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
    headers = []
    header_row = match_table.find("thead").find("tr") if match_table.find("thead") else None
    if header_row:
        headers = [th.text.strip() for th in header_row.find_all("th")]

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


def _parse_toto_score(score_value):
    """
    Toto向けに90分スコアのみを抽出する。
    - PK情報は無視（例: 1-1(PK4-3) -> 1,1）
    - 延長注記などは無視
    """
    s = unicodedata.normalize("NFKC", str(score_value)).replace("\u3000", " ").strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return None, None

    # PK以降は勝敗情報として不要なので切り落とす
    s = re.split(r"\bPK\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    s = re.sub(r"（\s*PK.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\(\s*PK.*$", "", s, flags=re.IGNORECASE)
    s = s.strip()

    # 基本形: 2-1 / 2－1 / 2:1
    m = re.search(r"(\d+)\s*[-－:：]\s*(\d+)", s)
    if m:
        return m.group(1), m.group(2)

    # フォールバック: 文字列中に2つ以上の数値がある場合は先頭2つを採用
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return nums[0], nums[1]

    return None, None


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


def scrape_match_results():
    print(f"試合結果をスクレイピング中: {TARGET_URL}")
    print(f"出力CSVパス: {OUTPUT_CSV_PATH}")

    if not os.path.exists(OUTPUT_CSV_DIR):
        print(f"出力ディレクトリ {OUTPUT_CSV_DIR} が存在しないため作成します。")
        os.makedirs(OUTPUT_CSV_DIR)

    try:
        response = get_with_retry(TARGET_URL, timeout=(5, 20), max_retries=3)
        response.raise_for_status()
        root_html_text = response.text
        root_soup = BeautifulSoup(root_html_text, 'lxml')

        preferred_fallback_frame_id = os.environ.get(
            "FALLBACK_FRAME_ID",
            FALLBACK_FRAME_ID_BY_LEAGUE.get(LEAGUE),
        )
        candidate_urls = _build_candidate_urls(root_soup, TARGET_URL, preferred_fallback_frame_id)
        tables = []

        for idx, url in enumerate(candidate_urls):
            if idx == 0:
                soup = root_soup
            else:
                print(f"タブURLを取得します: {url}")
                tab_resp = get_with_retry(url, timeout=(5, 20), max_retries=3)
                tab_resp.raise_for_status()
                soup = BeautifulSoup(tab_resp.text, "lxml")

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

        with open(TEMP_HTML_PATH, 'w', encoding='utf-8') as f:
            f.write(root_html_text)
        print(f"HTMLコンテンツを一時ファイル {TEMP_HTML_PATH} に保存しました。")

        if tables:
            print("BeautifulSoupで試合結果テーブルを正常に検出しました。手動でデータを抽出します。")
            df_results = pd.concat(tables, ignore_index=True, sort=False)
            print("BeautifulSoupでテーブルデータを正常に抽出しました。")
        else:
            raise RuntimeError("試合結果テーブルが見つかりませんでした。")

        # 日付整形ロジックを修正
        # 曜日表記と括弧内の文字、前後の空白を削除
        df_results['試合日'] = df_results['試合日'].str.replace(r'\s*\(.+\)\s*', '', regex=True) 
        df_results['datetime'] = pd.to_datetime(df_results['試合日'] + ' ' + df_results['K/O時刻'], format='%y/%m/%d %H:%M', errors='coerce')
        df_results = df_results.drop(columns=['試合日', 'K/O時刻'])

        # 以下既存のデータ整形ロジックを継続
        df_results = df_results.rename(columns={
            '節': '節',
            'ホーム': 'home_team',
            'スコア': 'score_full',
            'アウェイ': 'away_team',
            'スタジアム': 'stadium'
        })
        
        # スコア抽出（toto向けにPK情報は無視）
        parsed_scores = df_results['score_full'].apply(_parse_toto_score)
        parsed_df = pd.DataFrame(parsed_scores.tolist(), columns=['home_score_raw', 'away_score_raw'], index=df_results.index)
        df_results['home_score'] = pd.to_numeric(parsed_df['home_score_raw'], errors='coerce')
        df_results['away_score'] = pd.to_numeric(parsed_df['away_score_raw'], errors='coerce')
        # デバッグ: 非空スコアなのに抽出失敗した値を先頭だけ表示
        score_text_non_empty = df_results['score_full'].astype(str).str.strip().replace("nan", "")
        unparsed_mask = score_text_non_empty.ne("") & (df_results['home_score'].isna() | df_results['away_score'].isna())
        if unparsed_mask.any():
            samples = (
                df_results.loc[unparsed_mask, ['節', 'home_team', 'away_team', 'score_full']]
                .head(10)
                .to_dict(orient='records')
            )
            print(f"警告: スコア抽出失敗 {unparsed_mask.sum()} 件（先頭10件）: {samples}")
        df_results = df_results.drop(columns=['score_full'])
        # 2026移行モードでは同一カードが複数フレームに重複することがある。
        # その際、スコアあり行を優先して採用する。
        dedup_keys = ['節', 'datetime', 'home_team', 'away_team']
        df_results['__score_present'] = (
            df_results['home_score'].notna() & df_results['away_score'].notna()
        ).astype(int)
        df_results = (
            df_results
            .sort_values(by=['__score_present'], ascending=False)
            .drop_duplicates(subset=dedup_keys, keep='first')
            .drop(columns=['__score_present'])
        )

        # 所属推定フィルタを適用（2026移行モードの混在カード除去にも使う）
        # ただし2026移行モードでは前年J1/J2だけでは昇格チームを取りこぼすため既定で無効化。
        strict_league_filter = os.environ.get("STRICT_LEAGUE_FILTER", "0") == "1"
        if LEAGUE in {"j1", "j2"} and (not TRANSITION_2026_MODE or strict_league_filter):
            allowed_teams = _estimate_allowed_teams_for_league()
            if allowed_teams:
                before = len(df_results)
                df_results = df_results[
                    df_results['home_team'].isin(allowed_teams) & df_results['away_team'].isin(allowed_teams)
                ].copy()
                print(f"リーグ所属フィルタを適用: {before} -> {len(df_results)}")
        elif LEAGUE in {"j1", "j2"} and TRANSITION_2026_MODE:
            print("リーグ所属フィルタをスキップ: TRANSITION_2026_MODE")

        df_results['match_id'] = df_results.apply(
            lambda row: f"{LEAGUE}_{SEASON_YEAR}_{row['datetime'].strftime('%m%d%H%M')}_{row['home_team']}_{row['away_team']}" if pd.notna(row['datetime']) else "",
            axis=1
        )
        df_results['match_id'] = df_results['match_id'].str.replace(r'[^\w\s-]', '', regex=True).str.replace(r'\s+', '_', regex=True)


        try:
            df_existing_upcoming = pd.read_csv(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", f"{LEAGUE}_{SEASON_YEAR}_upcoming.csv")),
                nrows=0
            )
            desired_columns = df_existing_upcoming.columns.tolist()
            new_columns_order = ['節', 'match_id', 'datetime', 'stadium', 'home_team', 'away_team', 'home_score', 'away_score']
            
            for col in new_columns_order:
                if col not in desired_columns:
                    desired_columns.append(col)
            
            final_columns_to_use = [col for col in desired_columns if col in df_results.columns]
            df_results = df_results[final_columns_to_use]

        except FileNotFoundError:
            print("警告: j1_2025_upcoming.csv が見つかりませんでした。デフォルトのカラム順序を使用します。")
            df_results = df_results[[
                '節', 'match_id', 'datetime', 'stadium', 'home_team', 'away_team', 'home_score', 'away_score'
            ]]
        except Exception as e:
            print(f"警告: j1_2025_upcoming.csv のカラム読み込み中にエラーが発生しました: {e}。デフォルトのカラム順序を使用します。")
            df_results = df_results[[
                '節', 'match_id', 'datetime', 'stadium', 'home_team', 'away_team', 'home_score', 'away_score'
            ]]

        if df_results.empty:
            raise RuntimeError("取得データが0件です。")

        print(f"データをCSVに保存しようとしています: {OUTPUT_CSV_PATH}")
        df_results.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        if (not os.path.exists(OUTPUT_CSV_PATH)) or os.path.getsize(OUTPUT_CSV_PATH) == 0:
            raise RuntimeError("CSV書き込みに失敗しました（ファイル未生成/空）。")

        try:
            df_written = pd.read_csv(OUTPUT_CSV_PATH)
        except Exception as e:
            raise RuntimeError(f"CSV書き込み後の検証に失敗しました: {e}") from e
        if df_written.empty:
            raise RuntimeError("CSV書き込み後の行数が0件です。")

        print(f"SUCCESS: 試合結果データを {OUTPUT_CSV_PATH} に保存しました。 rows={len(df_written)}")
        return len(df_written)
    finally:
        time.sleep(1)

if __name__ == "__main__":
    try:
        scrape_match_results()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: 01_update_match_results.py network failure {TARGET_URL}: {repr(e)}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 01_update_match_results.py failed {TARGET_URL}: {repr(e)}")
        traceback.print_exc()
        sys.exit(1)
