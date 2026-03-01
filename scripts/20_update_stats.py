import pandas as pd
import requests # 再度インポート
from bs4 import BeautifulSoup
import time # time.sleep(1) のために必要
import re # 数値抽出のために正規表現を使用
import os # os.path, os.makedirsのために必要
import sys
import unicodedata
from datetime import datetime
from http_retry import get_with_retry
# from selenium import webdriver # Selenium関連のインポートを削除
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.common.exceptions import WebDriverException
# from selenium.webdriver.common.by import By # 必要に応じて要素検索に使用

# URLリスト（z_old_script/02_save_html_teamstats_01.pyから引用）
SEASON_YEAR = os.environ.get("SEASON_YEAR", "2025")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
STATS_SOURCE = os.environ.get("STATS_SOURCE", "").strip().lower()
if not STATS_SOURCE:
    STATS_SOURCE = LEAGUE
    if LEAGUE == "j2" and int(SEASON_YEAR) >= 2026:
        STATS_SOURCE = "j2j3"

TARGET_METRICS = [
    ("shoot_per_game", "1試合平均シュート数"),
    ("shoot_on_target_per_game", "1試合平均枠内シュート数"),
    ("shoot_rate", "シュート決定率"),
    ("score_per_game", "1試合平均得点数"),
    ("pass_count_per_game", "1試合平均パス数"),
    ("pass_rate", "パス成功率"),
    ("dribble_count_per_game", "1試合平均ドリブル数"),
    ("dribble_rate", "ドリブル成功率"),
    ("through_pass_count_per_game", "1試合平均スルーパス数"),
    ("through_pass_rate", "スルーパス成功率"),
    ("cross_count_per_game", "1試合平均クロス数"),
    ("air_battle_win_count_per_game", "1試合平均空中戦勝利数"),
    ("air_battle_win_rate", "空中戦勝率"),
    ("ball_rate", "平均ボール支配率"),
    ("chance_create_per_game", "1試合平均チャンスクリエイト数"),
    ("one_on_one_per_game", "1試合平均1vs1勝利数"),
    ("recovery_count_per_game", "1試合平均こぼれ球奪取数"),
    ("expected_goals", "ゴール期待値"),
    ("distance_per_game", "1試合平均走行距離"),
    ("sprint_per_game", "1試合平均スプリント回数"),
    ("at_sprint_per_game", "1試合平均Atスプリント回数"),
    ("mt_sprint_per_game", "1試合平均Mtスプリント回数"),
    ("dt_sprint_per_game", "1試合平均Dtスプリント回数"),
    ("possession_distance_per_game", "1試合平均ポゼッション時の走行距離"),
    ("possession_sprint_per_game", "1試合平均ポゼッション時のスプリント回数"),
    ("suffer_shoot_per_game", "1試合平均被シュート数"),
    ("suffer_shoot_on_target_per_game", "1試合平均被枠内シュート数"),
    ("lost_per_game", "1試合平均失点数"),
    ("clear_count_per_game", "1試合平均クリア数"),
    ("tackle_count_per_game", "1試合平均タックル数"),
    ("tackle_rate", "タックル成功率"),
    ("block_count_per_game", "1試合平均ブロック数"),
    ("intercept_count_per_game", "1試合平均インターセプト数"),
    ("expected_goals_against", "被ゴール期待値"),
    ("expected_goals_against_per_game", "1試合平均被ゴール期待値"),
    ("expected_goals_against_excl_pk", "被ゴール期待値 ※PKを除く"),
    ("clean_sheet", "クリーンシート総数"),
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "data", f"team_master_stats_{LEAGUE}_{SEASON_YEAR}.csv")
TEMP_HTML_DIR = os.path.join(BASE_DIR, "data", "temp_stats_html") # 新しいディレクトリ
STATS_SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "stats_snapshots")
STATS_ASOF_DATE = os.environ.get("STATS_ASOF_DATE", "").strip()
try:
    STATS_SUCCESS_THRESHOLD = float(os.environ.get("STATS_SUCCESS_THRESHOLD", "0.7"))
except ValueError:
    print("[WARN] STATS_SUCCESS_THRESHOLD が不正なため 0.7 を使用します。")
    STATS_SUCCESS_THRESHOLD = 0.7
STATS_SUCCESS_THRESHOLD = min(max(STATS_SUCCESS_THRESHOLD, 0.0), 1.0)
DEBUG_LOG = False

PLACEHOLDER_VALUES = {"", "-", "-%", "—", "－", "N/A", "n/a"}
NON_STATS_COLUMNS = {"team_name", "team_id", "league", "season", "round", "fetched_date"}

# jleague.jpの正式名称と、予測側CSVの短縮名称を寄せるための最小マップ
TEAM_ALIAS_RAW_MAP = {
    "北海道コンサドーレ札幌": "札幌",
    "ベガルタ仙台": "仙台",
    "ブラウブリッツ秋田": "秋田",
    "モンテディオ山形": "山形",
    "いわきＦＣ": "いわき",
    "ＲＢ大宮アルディージャ": "大宮",
    "横浜ＦＣ": "横浜FC",
    "湘南ベルマーレ": "湘南",
    "ヴァンラーレ八戸": "八戸",
    "ヴァンフォーレ甲府": "甲府",
    "アルビレックス新潟": "新潟",
    "栃木シティ": "栃木C",
    "栃木SC": "栃木C",
    "栃木ＳＣ": "栃木C",
    "カターレ富山": "富山",
    "ジュビロ磐田": "磐田",
    "藤枝ＭＹＦＣ": "藤枝",
    "徳島ヴォルティス": "徳島",
    "ＦＣ今治": "今治",
    "サガン鳥栖": "鳥栖",
    "大分トリニータ": "大分",
    "テゲバジャーロ宮崎": "宮崎",
    "レノファ山口ＦＣ": "山口",
    "ロアッソ熊本": "熊本",
    "愛媛ＦＣ": "愛媛",
    # NFKC後(半角FC/RB)の表記ゆれも吸収
    "いわきFC": "いわき",
    "FC今治": "今治",
    "愛媛FC": "愛媛",
    "レノファ山口FC": "山口",
    "RB大宮アルディージャ": "大宮",
}

TEAM_ALIAS_MAP = {
    unicodedata.normalize("NFKC", str(k)).strip(): v
    for k, v in TEAM_ALIAS_RAW_MAP.items()
}


def parse_stat_value(stat_value_text):
    text = (stat_value_text or "").strip()
    if text in PLACEHOLDER_VALUES:
        return None, "placeholder"

    stat_value_clean = re.sub(r"[^0-9.]", "", text)
    if not stat_value_clean:
        return None, "invalid"
    try:
        return float(stat_value_clean), None
    except ValueError:
        return None, "invalid"


def normalize_team_text(v):
    if pd.isna(v):
        return ""
    text = unicodedata.normalize("NFKC", str(v)).strip()
    text = TEAM_ALIAS_MAP.get(text, text)
    return text.upper().replace(" ", "")


def canonical_team_name(v):
    if pd.isna(v):
        return ""
    text = unicodedata.normalize("NFKC", str(v)).strip()
    return TEAM_ALIAS_MAP.get(text, text)


def build_metric_urls(source_league, season, metric_key):
    # j2j3 は /club/ なし構成が使われるため、両方試す。
    return [
        f"https://www.jleague.jp/stats/{source_league}/club/{season}/{metric_key}/",
        f"https://www.jleague.jp/stats/{source_league}/{season}/{metric_key}/",
    ]


def load_j2_target_team_keys():
    # j2j3混在ページからJ2だけ残すためのチーム集合
    cands = [
        os.path.join(BASE_DIR, "data", f"j2_{SEASON_YEAR}_upcoming.csv"),
        os.path.join(BASE_DIR, "data", f"j2_{SEASON_YEAR}_latest_results.csv"),
    ]
    teams = set()
    labels = set()
    for path in cands:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for col in ("home_team", "away_team"):
            if col in df.columns:
                vals = df[col].dropna().astype(str)
                teams.update(normalize_team_text(v) for v in vals if str(v).strip())
                labels.update(canonical_team_name(v) for v in vals if str(v).strip())
        if teams:
            return teams, labels, path
    return teams, labels, None


def scrape_jleague_stats():
    all_team_stats_data = []
    attempted_metrics = len(TARGET_METRICS)
    succeeded_metrics = 0
    print(f"[INFO] stats source: league={LEAGUE}, source={STATS_SOURCE}, season={SEASON_YEAR}")

    print(f"TEMP_HTML_DIRのパス: {TEMP_HTML_DIR}")
    if not os.path.exists(TEMP_HTML_DIR):
        os.makedirs(TEMP_HTML_DIR, exist_ok=True)
        print(f"一時HTML保存ディレクトリ {TEMP_HTML_DIR} を作成しました。(成功)")
    if not os.path.exists(STATS_SNAPSHOT_DIR):
        os.makedirs(STATS_SNAPSHOT_DIR, exist_ok=True)

    # Chromeオプションを設定（ヘッドレスモード） - Seleniumを使用しないため不要
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--no-sandbox")
    # chrome_options.add_argument("--disable-dev-shm-usage")

    # driver = None # Seleniumを使用しないため不要
    # print("Chrome WebDriverを初期化しています...") # Seleniumを使用しないため不要
    # driver = webdriver.Chrome(executable_path='chromedriver', options=chrome_options) # Seleniumを使用しないため不要
    # print("Chrome WebDriverの初期化に成功しました。") # Seleniumを使用しないため不要

    for i, (metric_key, stat_name) in enumerate(TARGET_METRICS):
        urls = build_metric_urls(STATS_SOURCE, SEASON_YEAR, metric_key)
        print(f"スクレイピング中: {stat_name} from {urls[0]}")

        temp_html_path = os.path.join(TEMP_HTML_DIR, f"temp_stats_{i}_{stat_name.replace(' ', '_').replace('/', '_')}.html")

        page_source = None
        used_url = None
        try:
            for url in urls:
                try:
                    response = get_with_retry(url, timeout=(5, 20), max_retries=3)
                    response.raise_for_status()
                    page_source = response.text
                    used_url = url
                    break
                except requests.exceptions.RequestException as e:
                    print(f"[WARN] URL候補失敗 {stat_name} from {url}: {e}")
            if page_source is None:
                raise requests.exceptions.RequestException(
                    f"all url candidates failed for {stat_name}"
                )

            if used_url and used_url != urls[0]:
                print(f"[INFO] fallback URL採用: {used_url}")

            with open(temp_html_path, 'w', encoding='utf-8') as f:
                f.write(page_source)
            if DEBUG_LOG:
                print(f"HTMLコンテンツを一時ファイル {temp_html_path} に保存しました。")

            soup = BeautifulSoup(page_source, 'lxml') # 取得したHTMLをBeautifulSoupでパース
            
            ranking_list = soup.find('ul', class_='ranking_list')
            if DEBUG_LOG:
                print(f"ランキングリスト ('ul.ranking_list') 検出結果: {ranking_list is not None}")
            
            if ranking_list:
                stats_for_current_page = []
                list_items = ranking_list.find_all('li')
                placeholder_count = 0
                invalid_count = 0
                if DEBUG_LOG:
                    print(f"リストアイテム数: {len(list_items)}")

                for li in list_items:
                    team_name_tag = li.find('p', class_='team')
                    stat_value_div_tag = li.find('div', class_=re.compile(r'ranking_stats'))
                    
                    if team_name_tag and stat_value_div_tag:
                        team_name_span = team_name_tag.find('span', class_=re.compile(r'embM'))
                        if team_name_span and team_name_span.next_sibling:
                            team_name = team_name_span.next_sibling.strip()
                            if DEBUG_LOG:
                                print(f"抽出チーム名 (next_sibling): {team_name}")
                        else:
                            team_name = team_name_tag.text.strip()
                            if DEBUG_LOG:
                                print(f"抽出チーム名 (text.strip): {team_name}")
                        
                        stat_value_p_tag = stat_value_div_tag.find('p')
                        stat_value = None
                        if stat_value_p_tag:
                            stat_value_text = stat_value_p_tag.text.strip()
                            stat_value, parse_error = parse_stat_value(stat_value_text)
                            if parse_error == "placeholder":
                                placeholder_count += 1
                                if DEBUG_LOG:
                                    print(f"未公開値: {stat_name}='{stat_value_text}'")
                            elif parse_error == "invalid":
                                invalid_count += 1
                                if DEBUG_LOG:
                                    print(f"警告: {stat_name} のスタッツ値 '{stat_value_text}' を数値に変換できませんでした。")
                            elif DEBUG_LOG:
                                print(f"抽出スタッツ値: {stat_name}={stat_value}")
                        
                        if team_name and stat_value is not None:
                            stats_for_current_page.append({
                                'team_name': team_name,
                                stat_name: stat_value
                            })
                    else:
                        if DEBUG_LOG:
                            print("警告: チーム名タグまたはスタッツ値タグが見つかりませんでした。")
                
                print(
                    f"現在のページで抽出されたスタッツ数: {len(stats_for_current_page)} "
                    f"(未公開値={placeholder_count}, 変換失敗={invalid_count}, 対象行={len(list_items)})"
                )
                if stats_for_current_page:
                    df_stats = pd.DataFrame(stats_for_current_page)
                    # ページ/メトリクスごとの表記揺れを吸収してから結合する
                    df_stats["team_name"] = df_stats["team_name"].map(canonical_team_name)
                    value_cols = [c for c in df_stats.columns if c != "team_name"]
                    if value_cols:
                        df_stats = (
                            df_stats.groupby("team_name", as_index=False)[value_cols[0]]
                            .mean()
                        )
                    all_team_stats_data.append(df_stats)
                    succeeded_metrics += 1
                else:
                    if list_items and placeholder_count == len(list_items):
                        print(f"警告: {stat_name} は全チーム未公開（'-'）のためスキップします。")
                    else:
                        print(f"警告: {stat_name} のデータが抽出できませんでした。スキップします。")
            else:
                print(f"警告: {stat_name} のランキングリスト ('ul.ranking_list') が見つかりませんでした。スキップします。")

        except requests.exceptions.RequestException as e:
            print(f"ネットワークエラー {stat_name}: {e}")
        except Exception as e:
            print(f"スクレイピング中にエラー発生 {stat_name}: {e}") # 例外の種類をrequests.exceptions.RequestExceptionとGeneric Exceptionに分割
        
        time.sleep(1) # サイトに負荷をかけないように1秒待機

    # finally: # Seleniumを使用しないため不要
    #     if driver:
    #         driver.quit()
    #         print("Chrome WebDriverを閉じました。")
            
    success_rate = (succeeded_metrics / attempted_metrics) if attempted_metrics else 0.0
    print(
        f"[METRIC_QC] attempted_metrics={attempted_metrics}, "
        f"succeeded_metrics={succeeded_metrics}, success_rate={success_rate:.2%}"
    )

    if not all_team_stats_data:
        raise RuntimeError(
            f"取得できるスタッツデータがありませんでした。 "
            f"(attempted_metrics={attempted_metrics}, succeeded_metrics={succeeded_metrics})"
        )

    if success_rate < STATS_SUCCESS_THRESHOLD:
        raise RuntimeError(
            "スタッツ取得成功率が閾値未満です。 "
            f"(attempted_metrics={attempted_metrics}, "
            f"succeeded_metrics={succeeded_metrics}, "
            f"success_rate={success_rate:.2%}, threshold={STATS_SUCCESS_THRESHOLD:.2%})"
        )

    if all_team_stats_data:
        final_df = all_team_stats_data[0]
        for i in range(1, len(all_team_stats_data)):
            # 先頭メトリクス未掲載チームが落ちないよう outer で統合
            final_df = pd.merge(final_df, all_team_stats_data[i], on='team_name', how='outer')

        if LEAGUE == "j2" and STATS_SOURCE == "j2j3":
            team_keys, team_labels, team_src = load_j2_target_team_keys()
            if team_keys:
                before = len(final_df)
                final_df["_team_key"] = final_df["team_name"].map(normalize_team_text)
                final_df = final_df[final_df["_team_key"].isin(team_keys)].copy()
                final_df = final_df.drop(columns=["_team_key"], errors="ignore")
                if team_labels:
                    missing_labels = sorted(set(team_labels) - set(final_df["team_name"].dropna().astype(str)))
                    if missing_labels:
                        # J2全チーム行を保証（値が未取得ならNaNで残す）
                        missing_df = pd.DataFrame({"team_name": missing_labels})
                        final_df = pd.concat([final_df, missing_df], ignore_index=True, sort=False)
                        print(
                            f"[FILTER][WARN] 欠落チームを補完追加: {len(missing_labels)} "
                            f"({', '.join(missing_labels)})"
                        )
                print(
                    f"[FILTER] j2 teams only: {before} -> {len(final_df)} "
                    f"(source={team_src})"
                )
            else:
                print("[WARN] j2チーム集合が作れないため、j2j3混在データをそのまま使用します。")

        stat_columns = [c for c in final_df.columns if c not in NON_STATS_COLUMNS]
        if stat_columns:
            missing_mask = final_df[stat_columns].isna()
            empty_mask = final_df[stat_columns].apply(lambda s: s.astype(str).str.strip().eq(""))
            missing_mask = missing_mask | empty_mask
            final_df["stats_missing_ratio"] = (
                missing_mask.sum(axis=1) / float(len(stat_columns))
            ).round(3)

            total_cells = len(final_df) * len(stat_columns)
            filled_cells = int((~missing_mask).sum().sum())
            fill_rate = (filled_cells / total_cells) if total_cells else 0.0
            print(
                f"[VALUE_QC] teams={len(final_df)}, stat_columns={len(stat_columns)}, "
                f"filled_cells={filled_cells}/{total_cells}, fill_rate={fill_rate:.2%}"
            )
        else:
            final_df["stats_missing_ratio"] = 1.0
            print("[VALUE_QC][WARN] スタッツ列を検出できないため stats_missing_ratio=1.0 を設定しました。")
        
        print(f"最終的なデータフレームの行数: {len(final_df)}")
        final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        if (not os.path.exists(OUTPUT_CSV_PATH)) or os.path.getsize(OUTPUT_CSV_PATH) == 0:
            raise RuntimeError("CSV書き込みに失敗しました（ファイル未生成/空）。")
        if STATS_ASOF_DATE:
            asof = STATS_ASOF_DATE
        else:
            asof = datetime.now().strftime("%Y-%m-%d")
        asof_key = asof.replace("-", "")
        snapshot_name = f"team_master_stats_{LEAGUE}_{SEASON_YEAR}_asof_{asof_key}.csv"
        snapshot_path = os.path.join(STATS_SNAPSHOT_DIR, snapshot_name)
        final_df.to_csv(snapshot_path, index=False, encoding="utf-8-sig")
        print(f"[SNAPSHOT] stats snapshot saved: {snapshot_path}")
        print(f"SUCCESS: すべてのチームスタッツを {OUTPUT_CSV_PATH} に集約しました。")
        return len(final_df)
    else:
        raise RuntimeError("取得できるスタッツデータがありませんでした。")

if __name__ == "__main__":
    try:
        scrape_jleague_stats()
    except Exception as e:
        print(f"ERROR: 20_update_stats.py failed: {e}")
        sys.exit(1)
