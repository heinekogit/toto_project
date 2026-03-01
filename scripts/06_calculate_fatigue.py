import pandas as pd
import numpy as np
import os
import math
from datetime import timedelta

# ファイルパスの設定
DATA_DIR = "/Users/dev_tomo/Desktop/tt_prj_restart/data" # 変更
MANUAL_DIR = os.path.join(DATA_DIR, "manual")
SEASON_YEAR = os.environ.get("SEASON_YEAR", "2025")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
UPCOMING_CSV = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_upcoming.csv")
LATEST_RESULTS_CSV = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_latest_results.csv")
TEAM_TRAVEL_DISTANCES_CSV = os.path.join(MANUAL_DIR, "team_travel_distances.csv")
if not os.path.exists(TEAM_TRAVEL_DISTANCES_CSV):
    TEAM_TRAVEL_DISTANCES_CSV = os.path.join(DATA_DIR, "team_travel_distances.csv")
HOME_STADIUMS_XLSX = os.path.join(MANUAL_DIR, "ホーム所在地一覧.xlsx")
OUTPUT_FATIGUE_CSV = os.path.join(DATA_DIR, f"team_fatigue_scores_{LEAGUE}_{SEASON_YEAR}.csv")

# 疲労度計算の係数 (調整可能)
DAYS_SINCE_LAST_MATCH_WEIGHT = 0.5 # 試合間隔が短いほど疲労度が高い
TRAVEL_DISTANCE_WEIGHT = 0.005     # 移動距離が長いほど疲労度が高い
AWAY_GAME_PENALTY = 10             # アウェイ戦のペナルティ

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def load_stadiums(path):
    df = pd.read_excel(path)
    # 列名のゆるい対応
    col_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in ["team", "チーム", "チーム名"]:
            col_map[col] = "team"
        elif key in ["stadium_name", "stadium", "スタジアム", "スタジアム名", "ホーム"]:
            col_map[col] = "stadium_name"
        elif key in ["lat", "latitude", "緯度"]:
            col_map[col] = "lat"
        elif key in ["lon", "lng", "longitude", "経度"]:
            col_map[col] = "lon"
    df = df.rename(columns=col_map)
    for col in ["team", "stadium_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df = df.dropna(subset=["team", "lat", "lon"])
    return df


def build_distance_matrix(stadiums_df):
    teams = stadiums_df["team"].tolist()
    matrix = pd.DataFrame(index=teams, columns=teams, dtype=float)
    coords = stadiums_df.set_index("team")[["lat", "lon"]].to_dict("index")
    for home in teams:
        for away in teams:
            if home == away:
                matrix.loc[away, home] = 0.0
                continue
            lat1, lon1 = coords[away]["lat"], coords[away]["lon"]
            lat2, lon2 = coords[home]["lat"], coords[home]["lon"]
            matrix.loc[away, home] = round(haversine_km(lat1, lon1, lat2, lon2), 1)
    matrix.index.name = "ホーム　／　アウェイ"
    return matrix


def load_travel_distances():
    try:
        travel_distances_df = pd.read_csv(TEAM_TRAVEL_DISTANCES_CSV, sep="	")
        travel_distances_df = travel_distances_df.set_index("ホーム　／　アウェイ")
        print(f"移動距離データを {TEAM_TRAVEL_DISTANCES_CSV} から読み込みました。")
        return travel_distances_df
    except FileNotFoundError:
        print(f"警告: 移動距離ファイル '{TEAM_TRAVEL_DISTANCES_CSV}' が見つかりませんでした。")
    except Exception as e:
        print(f"警告: 移動距離データの読み込み中にエラーが発生しました: {e}")

    if os.path.exists(HOME_STADIUMS_XLSX):
        try:
            stadiums_df = load_stadiums(HOME_STADIUMS_XLSX)
            travel_distances_df = build_distance_matrix(stadiums_df)
            os.makedirs(MANUAL_DIR, exist_ok=True)
            travel_distances_df.to_csv(TEAM_TRAVEL_DISTANCES_CSV, sep="	", encoding="utf-8-sig")
            print(f"ホーム所在地一覧から移動距離を生成しました: {TEAM_TRAVEL_DISTANCES_CSV}")
            return travel_distances_df
        except Exception as e:
            print(f"警告: ホーム所在地一覧から移動距離を生成できませんでした: {e}")

    return pd.DataFrame()

def calculate_fatigue(df_matches, travel_distances_df):
    fatigue_scores = []
    
    # チームごとの最新の試合日と累積移動距離を追跡
    team_last_match_date = {}

    # 試合を日付順にソート
    df_matches = df_matches.sort_values(by='datetime').reset_index(drop=True)

    for index, row in df_matches.iterrows():
        match_datetime = row['datetime']
        home_team = row['home_team']
        away_team = row['away_team']

        # ホームチームの疲労度計算
        home_fatigue = 0
        if home_team in team_last_match_date:
            days_since_last_match = (match_datetime - team_last_match_date[home_team]).days
            if days_since_last_match <= 3: # 例: 3日以内の試合は高疲労
                home_fatigue += (3 - days_since_last_match) * DAYS_SINCE_LAST_MATCH_WEIGHT
        
        # ホームチームの移動距離（アウェイチームがホームに来る距離）
        # このスクリプトではhome_teamは移動しないので、これは計算に含めない
        
        # アウェイチームの疲労度計算
        away_fatigue = 0
        if away_team in team_last_match_date:
            days_since_last_match = (match_datetime - team_last_match_date[away_team]).days
            if days_since_last_match <= 3:
                away_fatigue += (3 - days_since_last_match) * DAYS_SINCE_LAST_MATCH_WEIGHT

        # アウェイチームの移動距離（アウェイチームがホームまで移動する距離）
        if not travel_distances_df.empty and away_team in travel_distances_df.index and home_team in travel_distances_df.columns:
            travel_distance = travel_distances_df.loc[away_team, home_team]
            away_fatigue += travel_distance * TRAVEL_DISTANCE_WEIGHT
        
        # アウェイゲームのペナルティ
        away_fatigue += AWAY_GAME_PENALTY

        fatigue_scores.append({
            'match_id': row['match_id'],
            'datetime': match_datetime,
            'home_team': home_team,
            'away_team': away_team,
            'home_fatigue_score': round(home_fatigue, 2),
            'away_fatigue_score': round(away_fatigue, 2),
        })

        # 試合日を更新
        team_last_match_date[home_team] = match_datetime
        team_last_match_date[away_team] = match_datetime
            
    return pd.DataFrame(fatigue_scores)

def main():
    print(f"DATA_DIRのパス: {DATA_DIR}") # 追加
    print(f"UPCOMING_CSVのパス: {UPCOMING_CSV}") # 追加
    print(f"LATEST_RESULTS_CSVのパス: {LATEST_RESULTS_CSV}") # 追加
    print(f"TEAM_TRAVEL_DISTANCES_CSVのパス: {TEAM_TRAVEL_DISTANCES_CSV}") # 追加

    # 試合結果と今後の試合日程を読み込む
    try:
        df_upcoming = pd.read_csv(UPCOMING_CSV)
        df_latest_results = pd.read_csv(LATEST_RESULTS_CSV)
        df_matches = pd.concat([df_upcoming, df_latest_results], ignore_index=True)
        print(f"試合日程データを {UPCOMING_CSV} および {LATEST_RESULTS_CSV} から読み込みました。")
    except FileNotFoundError as e:
        print(f"エラー: 試合日程ファイルが見つかりません。{e}")
        return
    except Exception as e:
        print(f"エラー: 試合日程データの読み込み中にエラーが発生しました。{e}")
        return

    # 'datetime' カラムをdatetime型に変換
    df_matches['datetime'] = pd.to_datetime(df_matches['datetime'], errors='coerce')
    df_matches = df_matches.dropna(subset=['datetime']) # 無効な日付を削除

    # 移動距離データを読み込む（無ければホーム所在地一覧から生成）
    travel_distances_df = load_travel_distances()

    if df_matches.empty:
        print("処理する試合データがありません。")
        return

    # 疲労度を計算
    df_fatigue = calculate_fatigue(df_matches, travel_distances_df)

    # 結果をCSVに保存
    if not df_fatigue.empty:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        df_fatigue.to_csv(OUTPUT_FATIGUE_CSV, index=False, encoding="utf-8-sig")
        print(f"疲労度スコアを {OUTPUT_FATIGUE_CSV} に保存しました。")
    else:
        print("計算された疲労度スコアがありません。")

if __name__ == "__main__":
    main()
