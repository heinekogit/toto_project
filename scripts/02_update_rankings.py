import os
import time
from datetime import datetime, timezone
import re
import traceback
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from http_retry import get_with_retry


SEASON_YEAR = os.environ.get("SEASON_YEAR", "2025")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
COMPETITION_NAME = os.environ.get("COMPETITION_NAME")
COMPETITION_ID = os.environ.get("COMPETITION_ID")
COMPETITION_IDS = [v.strip() for v in os.environ.get("COMPETITION_IDS", "").split(",") if v.strip()]
ROUND_VALUE = os.environ.get("ROUND_VALUE")  # 例: "36" / "0"（最新節）
ALT_RANKING_URLS = [u.strip() for u in os.environ.get("ALT_RANKING_URLS", "").split(",") if u.strip()]
RANKING_YEAR_ID = os.environ.get("RANKING_YEAR_ID")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
FETCH_DATE = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d")
OUTPUT_CSV = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_rankings_{FETCH_DATE}.csv")

SFRT01_URL = "https://data.j-league.or.jp/SFRT01/"
SFRT01_COMP_URL = "https://data.j-league.or.jp/SFRT01/competition"
SFRT01_SECTION_URL = "https://data.j-league.or.jp/SFRT01/competitionSection"


_FULLWIDTH_ASCII = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９　",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ",
)


def _normalize_text(value):
    text = (value or "").translate(_FULLWIDTH_ASCII).lower()
    text = re.sub(r"\s+", "", text)
    return text


def _build_year_candidates(year):
    candidates = []
    if RANKING_YEAR_ID:
        candidates.append(str(RANKING_YEAR_ID))
    year_str = str(year)
    candidates.append(year_str)
    if len(year_str) == 4 and year_str.isdigit():
        candidates.append(f"{year_str}1")
    # 順序を維持した重複除去
    return list(dict.fromkeys(candidates))


def build_driver(headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")

    driver_path = os.path.join(BASE_DIR, "chromedriver")
    if os.path.exists(driver_path):
        from selenium.webdriver.chrome.service import Service
        return webdriver.Chrome(service=Service(executable_path=driver_path), options=options)

    return webdriver.Chrome(options=options)


def get_competition_ids(year, competition_name):
    year_candidates = _build_year_candidates(year)

    if COMPETITION_IDS:
        return list(dict.fromkeys(COMPETITION_IDS)), year_candidates[0]

    if COMPETITION_ID:
        return [COMPETITION_ID], year_candidates[0]

    last_error = None
    for year_id in year_candidates:
        resp = get_with_retry(
            SFRT01_COMP_URL,
            params={"yearId": year_id},
            timeout=(5, 20),
            max_retries=3,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        options = []
        for opt in soup.find_all("option"):
            label = opt.get_text(strip=True)
            value = opt.get("value")
            if not (label and value):
                continue
            options.append((value, label))

        if not options:
            last_error = f"competitionId候補が空: yearId={year_id}"
            continue

        normalized_target = _normalize_text(competition_name)
        explicit_matches = []
        for value, label in options:
            if normalized_target and normalized_target in _normalize_text(label):
                explicit_matches.append((value, label))

        if explicit_matches:
            ids = [v for v, _ in explicit_matches]
            labels = " / ".join([f"{lbl}({val})" for val, lbl in explicit_matches[:10]])
            print(f"[INFO] competitionIdを名称一致で選択: {labels} yearId={year_id}")
            return ids, year_id

        league_keywords = {
            "j1": ["j1", "j1league", "jリーグdivision1", "jleaguedivision1"],
            "j2": ["j2", "j2league", "jリーグdivision2", "jleaguedivision2"],
            "j3": ["j3", "j3league", "jリーグdivision3", "jleaguedivision3"],
        }
        keywords = league_keywords.get(LEAGUE, [LEAGUE])

        fallback_candidates = []
        for value, label in options:
            normalized_label = _normalize_text(label)
            if any(k in normalized_label for k in keywords):
                fallback_candidates.append((value, label))

        if fallback_candidates:
            ids = [v for v, _ in fallback_candidates]
            labels = " / ".join([f"{lbl}({val})" for val, lbl in fallback_candidates[:10]])
            if len(fallback_candidates) == 1:
                print(f"[INFO] competitionIdをフォールバック選択しました: {labels} yearId={year_id}")
            else:
                print(f"[INFO] competitionIdを複数候補で選択しました: {labels} yearId={year_id}")
            return ids, year_id

        if len(options) == 1:
            value, label = options[0]
            print(f"[INFO] competitionId候補が1件のみのため採用: {label} ({value}) yearId={year_id}")
            return [value], year_id

        option_labels = ", ".join([f"{lbl}({val})" for val, lbl in options[:10]])
        last_error = f"competitionId未決定: yearId={year_id}, options={option_labels}"

    raise ValueError(
        f"competitionIdが見つかりません: season={year}, tried_yearIds={year_candidates}, detail={last_error}"
    )


def get_sections(competition_id):
    resp = get_with_retry(
        SFRT01_SECTION_URL,
        params={"competitionId": competition_id},
        timeout=(5, 20),
        max_retries=3,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    sections = []
    for opt in soup.find_all("option"):
        value = opt.get("value")
        label = opt.get_text(strip=True)
        if value and label and value.isdigit():
            sections.append((value, label))
    return sections


def parse_ranking_table(html):
    """
    順位表テーブルを抽出する。
    特殊期間では同一ページ内に順位表が複数テーブルで分割されるため、
    条件に合うテーブルをすべて連結して返す。
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    matched = []
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers:
            continue
        if any("順位" in h for h in headers) and any("チーム" in h for h in headers):
            rows = []
            for tr in table.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if not cells:
                    continue
                values = [c.get_text(" ", strip=True) for c in cells]
                rows.append(values)
            matched.append((headers, rows))

    if not matched:
        return None, None

    base_headers = matched[0][0]
    merged_rows = []
    for headers, rows in matched:
        # 見出し構成が違う場合でも、順位/チームが入っていれば追加対象にする。
        if headers != base_headers:
            if not (any("順位" in h for h in headers) and any("チーム" in h for h in headers)):
                continue
        merged_rows.extend(rows)

    print(f"[INFO] 順位表テーブル検出: {len(matched)}件 / 連結行数={len(merged_rows)}")
    return base_headers, merged_rows


def normalize_table(headers, rows):
    if not headers or not rows:
        return []

    # ヘッダ行を除外（重複行・表記ゆれ行）
    data_rows = []
    norm_header_tokens = {_normalize_text(h) for h in headers if h}
    for row in rows:
        if row == headers:
            continue
        # 先頭行が「順位」「チーム」等の見出し語だけで構成される行は除外
        row_tokens = {_normalize_text(v) for v in row if str(v).strip()}
        if row_tokens and ("順位" in "".join(row) or "チーム" in "".join(row)):
            if len(row_tokens & norm_header_tokens) >= 2:
                continue
        data_rows.append(row)

    # 不要列（グラフ等）を除外
    drop_indices = [i for i, h in enumerate(headers) if h in ["", "グラフ"]]
    norm_headers = [h for i, h in enumerate(headers) if i not in drop_indices]

    cleaned = []
    for row in data_rows:
        values = [v for i, v in enumerate(row) if i not in drop_indices]
        if len(values) < len(norm_headers):
            values += [""] * (len(norm_headers) - len(values))
        cleaned.append(dict(zip(norm_headers, values[: len(norm_headers)])))
    return norm_headers, cleaned


def fetch_rankings_from_data_site():
    os.makedirs(DATA_DIR, exist_ok=True)
    comp_name = COMPETITION_NAME
    if not comp_name:
        comp_name = "明治安田Ｊ１リーグ" if LEAGUE == "j1" else "明治安田Ｊ２リーグ"
    comp_ids, ranking_year_id = get_competition_ids(SEASON_YEAR, comp_name)

    headless = os.environ.get("HEADLESS", "1") != "0"
    driver = build_driver(headless=headless)
    all_records = []

    try:
        for comp_id in comp_ids:
            try:
                # competitionIdごとにページを開き直し、DOM差異で<select>が崩れるケースを回避
                driver.get(SFRT01_URL)
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.NAME, "yearId")))
                Select(driver.find_element(By.NAME, "yearId")).select_by_value(str(ranking_year_id))
                time.sleep(2)

                sections = get_sections(comp_id)
                if ROUND_VALUE:
                    sections = [s for s in sections if s[0] == str(ROUND_VALUE)]
                if not ROUND_VALUE:
                    sections = [("0", "最新節")]

                comp_elem = driver.find_element(By.NAME, "competitionId")
                if comp_elem.tag_name.lower() != "select":
                    raise RuntimeError(f"competitionId要素が<select>ではありません: tag={comp_elem.tag_name}")
                comp_select = Select(comp_elem)
                try:
                    comp_select.select_by_value(str(comp_id))
                except NoSuchElementException:
                    # UIの選択肢がAPI値とズレるケース向けフォールバック
                    options = [(opt.get_attribute("value"), opt.text.strip()) for opt in comp_select.options]
                    print(f"[WARN] competitionId={comp_id} がUIに存在しません。options={options[:8]}")
                    league_keywords = {
                        "j1": ["j1", "j1league", "jリーグdivision1", "jleaguedivision1"],
                        "j2": ["j2", "j2league", "jリーグdivision2", "jleaguedivision2"],
                        "j3": ["j3", "j3league", "jリーグdivision3", "jleaguedivision3"],
                    }
                    keywords = league_keywords.get(LEAGUE, [LEAGUE])
                    fallback = None
                    for val, label in options:
                        nlabel = _normalize_text(label)
                        if any(k in nlabel for k in keywords):
                            fallback = (val, label)
                            break
                    if fallback:
                        comp_select.select_by_value(str(fallback[0]))
                        comp_id = fallback[0]
                        print(f"[INFO] competitionIdをUIフォールバック選択: {fallback[1]} ({fallback[0]})")
                    elif options:
                        comp_select.select_by_value(str(options[0][0]))
                        comp_id = options[0][0]
                        print(f"[INFO] competitionIdをUI先頭で選択: {options[0][1]} ({options[0][0]})")
                    else:
                        raise
                time.sleep(2)

                for value, label in sections:
                    try:
                        # セレクトを毎回取得（画面再描画でstaleになるため）
                        try:
                            section_select = Select(driver.find_element(By.NAME, "competitionSectionId"))
                        except Exception:
                            section_select = None

                        if section_select is None:
                            print("[警告] 節セレクトが見つからないため、最新節のみ取得します。")
                            if str(value) != "0":
                                continue
                        else:
                            section_select.select_by_value(str(value))
                        time.sleep(1)
                        driver.execute_script("document.forms[0].submit()")
                        time.sleep(3)

                        headers, rows = parse_ranking_table(driver.page_source)
                        if not headers:
                            print(f"[警告] 順位表が取得できませんでした: comp={comp_id}, {label}")
                            continue

                        norm_headers, cleaned = normalize_table(headers, rows)
                        for row in cleaned:
                            row["season"] = SEASON_YEAR
                            row["round"] = label
                            row["fetched_date"] = FETCH_DATE
                            all_records.append(row)
                        print(f"[OK] comp={comp_id} {label} を取得しました。")
                    except TimeoutException:
                        print(f"[警告] comp={comp_id} {label} の取得でタイムアウトしました。")
                    except WebDriverException as e:
                        print(f"[警告] comp={comp_id} {label} の取得でエラー: {e}")
            except Exception as e:
                print(f"[WARN] competitionId={comp_id} の処理をスキップします: {e}")
                continue
    finally:
        driver.quit()

    if not all_records:
        raise RuntimeError("順位表を取得できませんでした。")

    df = pd.DataFrame(all_records)
    # 複数competitionId連結時の重複排除
    dedup_keys = [k for k in ["season", "round", "チーム", "順位"] if k in df.columns]
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
    elif "チーム" in df.columns:
        df = df.drop_duplicates(subset=["チーム"], keep="first")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"出力: {OUTPUT_CSV}")


def fetch_from_alt_urls():
    if not ALT_RANKING_URLS:
        return False
    all_records = []
    for url in ALT_RANKING_URLS:
        try:
            resp = get_with_retry(url, timeout=(5, 20), max_retries=3)
            resp.raise_for_status()
            headers, rows = parse_ranking_table(resp.text)
            if not headers:
                print(f"[警告] テーブルが見つかりません: {url}")
                continue
            norm_headers, cleaned = normalize_table(headers, rows)
            for row in cleaned:
                row["season"] = SEASON_YEAR
                row["round"] = "latest"
                row["fetched_date"] = FETCH_DATE
                all_records.append(row)
        except Exception as e:
            print(f"[警告] {url}: {e}")

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"出力: {OUTPUT_CSV}")
        return True
    return False


if __name__ == "__main__":
    try:
        fetch_rankings_from_data_site()
    except (WebDriverException, RuntimeError) as e:
        print(f"[警告] データサイトから取得できませんでした: {e}")
        if not fetch_from_alt_urls():
            print(f"[ERROR] 02_update_rankings.py failed: {repr(e)}")
            traceback.print_exc()
            raise
    except Exception as e:
        print(f"[ERROR] 02_update_rankings.py unexpected failure: {repr(e)}")
        traceback.print_exc()
        raise
