import pandas as pd
import numpy as np
import os
import math

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
ACL_SCHEDULE_CSV = os.environ.get("ACL_SCHEDULE_CSV", os.path.join(MANUAL_DIR, "acl_schedule.csv"))

# 疲労度計算の係数 (調整可能)
DAYS_SINCE_LAST_MATCH_WEIGHT = 0.5 # 試合間隔が短いほど疲労度が高い
TRAVEL_DISTANCE_WEIGHT = 0.005     # 移動距離が長いほど疲労度が高い
AWAY_GAME_PENALTY = 4              # 地域リーグ期間は固定アウェイ罰則を弱める
REST_FATIGUE_EFFECTIVE_DAYS = 4    # 中3日は軽疲労として残す
RECENT_MATCH_CARRY_WEIGHTS = [0.35, 0.20, 0.10]  # 直近3試合の蓄積を減衰で持つ

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


def dedupe_matches(df_matches):
    if df_matches.empty:
        return df_matches

    df = df_matches.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    for col in ["home_team", "away_team", "match_id"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    score_cols = [c for c in ["home_score", "away_score"] if c in df.columns]
    if score_cols:
        df["__has_score"] = df[score_cols].notna().all(axis=1).astype(int)
    else:
        df["__has_score"] = 0

    match_id = df["match_id"] if "match_id" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    fallback_key = (
        df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + "|"
        + df.get("home_team", pd.Series("", index=df.index)).astype(str)
        + "|"
        + df.get("away_team", pd.Series("", index=df.index)).astype(str)
    )
    df["__dedupe_key"] = match_id.where(match_id.notna() & (match_id != ""), fallback_key)

    before_rows = len(df)
    before_keys = int(df["__dedupe_key"].duplicated().sum())

    sort_cols = ["__dedupe_key", "__has_score"]
    ascending = [True, False]
    if "__source_priority" in df.columns:
        sort_cols.append("__source_priority")
        ascending.append(False)

    df = (
        df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .drop_duplicates(subset=["__dedupe_key"], keep="first")
        .drop(columns=["__has_score", "__dedupe_key"], errors="ignore")
        .reset_index(drop=True)
    )

    removed_rows = before_rows - len(df)
    print(f"試合重複を整理: 重複キー={before_keys}, 削除行数={removed_rows}, 残件数={len(df)}")
    return df


def load_acl_team_events(path):
    required_cols = {"match_date", "team"}
    if not path or not os.path.exists(path):
        print(f"[INFO] ACL日程ファイルなし: {path}")
        return pd.DataFrame(columns=["datetime", "team", "event_type", "event_load"])
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] ACL日程の読込失敗: {path} error={e}")
        return pd.DataFrame(columns=["datetime", "team", "event_type", "event_load"])

    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols - set(df.columns))
        print(f"[WARN] ACL日程の必須列不足: missing={missing} path={path}")
        return pd.DataFrame(columns=["datetime", "team", "event_type", "event_load"])

    out = df.copy().dropna(how="all")
    out["team"] = out["team"].astype(str).str.strip()
    out["datetime"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["event_load"] = pd.to_numeric(out.get("fatigue_grade"), errors="coerce").fillna(0.0)
    out = out.dropna(subset=["datetime"])
    out = out[out["team"].ne("")]
    if out.empty:
        return pd.DataFrame(columns=["datetime", "team", "event_type", "event_load"])
    # 日付だけの入力を正午に寄せ、同日のリーグ戦より前の公式戦イベントとして扱う。
    out["datetime"] = out["datetime"].dt.normalize() + pd.to_timedelta(12, unit="h")
    out["event_type"] = "acl_event"
    out = out[["datetime", "team", "event_type", "event_load"]].drop_duplicates().sort_values("datetime", kind="mergesort")
    print(f"[INFO] ACLイベントを読込: rows={len(out)} path={path}")
    return out.reset_index(drop=True)


def calc_rest_fatigue(days_since_last_match):
    if pd.isna(days_since_last_match):
        return 0.0
    days = int(days_since_last_match)
    if days <= 0:
        return 0.0
    if days > REST_FATIGUE_EFFECTIVE_DAYS:
        return 0.0
    return max(0.0, float(REST_FATIGUE_EFFECTIVE_DAYS - days) * DAYS_SINCE_LAST_MATCH_WEIGHT)


def calc_recent_carry(recent_loads):
    if not recent_loads:
        return 0.0
    total = 0.0
    for idx, load in enumerate(recent_loads[: len(RECENT_MATCH_CARRY_WEIGHTS)]):
        total += float(load) * float(RECENT_MATCH_CARRY_WEIGHTS[idx])
    return total


def push_recent_load(team_recent_loads, team, load):
    if not team:
        return
    history = list(team_recent_loads.get(team, []))
    history.insert(0, float(max(0.0, load)))
    team_recent_loads[team] = history[: len(RECENT_MATCH_CARRY_WEIGHTS)]


def calculate_fatigue(df_matches, travel_distances_df, external_team_events=None):
    fatigue_scores = []

    team_last_match_date = {}
    team_recent_loads = {}
    df_matches = df_matches.sort_values(by='datetime').reset_index(drop=True)
    event_rows = []
    for _, row in df_matches.iterrows():
        event_rows.append(("league_match", row["datetime"], row))
    if external_team_events is not None and not external_team_events.empty:
        for _, row in external_team_events.sort_values("datetime", kind="mergesort").iterrows():
            event_rows.append(("external_team_event", row["datetime"], row))
    event_rows.sort(key=lambda item: (item[1], 0 if item[0] == "external_team_event" else 1))

    for event_type, _, row in event_rows:
        if event_type == "external_team_event":
            team = str(row["team"]).strip()
            if team:
                team_last_match_date[team] = row["datetime"]
                push_recent_load(team_recent_loads, team, row.get("event_load", 0.0))
            continue

        match_datetime = row['datetime']
        home_team = row['home_team']
        away_team = row['away_team']

        home_fatigue = 0
        if home_team in team_last_match_date:
            days_since_last_match = (match_datetime - team_last_match_date[home_team]).days
            home_fatigue += calc_rest_fatigue(days_since_last_match)
        home_fatigue += calc_recent_carry(team_recent_loads.get(home_team, []))

        away_fatigue = 0
        if away_team in team_last_match_date:
            days_since_last_match = (match_datetime - team_last_match_date[away_team]).days
            away_fatigue += calc_rest_fatigue(days_since_last_match)
        away_fatigue += calc_recent_carry(team_recent_loads.get(away_team, []))

        if not travel_distances_df.empty and away_team in travel_distances_df.index and home_team in travel_distances_df.columns:
            travel_distance = travel_distances_df.loc[away_team, home_team]
            away_fatigue += travel_distance * TRAVEL_DISTANCE_WEIGHT

        away_fatigue += AWAY_GAME_PENALTY

        home_event_load = home_fatigue
        away_event_load = away_fatigue

        fatigue_scores.append({
            'match_id': row['match_id'],
            'datetime': match_datetime,
            'home_team': home_team,
            'away_team': away_team,
            'home_fatigue_score': round(home_fatigue, 2),
            'away_fatigue_score': round(away_fatigue, 2),
        })

        team_last_match_date[home_team] = match_datetime
        team_last_match_date[away_team] = match_datetime
        push_recent_load(team_recent_loads, home_team, home_event_load)
        push_recent_load(team_recent_loads, away_team, away_event_load)
            
    return pd.DataFrame(fatigue_scores)

def main():
    print(f"DATA_DIRのパス: {DATA_DIR}") # 追加
    print(f"UPCOMING_CSVのパス: {UPCOMING_CSV}") # 追加
    print(f"LATEST_RESULTS_CSVのパス: {LATEST_RESULTS_CSV}") # 追加
    print(f"TEAM_TRAVEL_DISTANCES_CSVのパス: {TEAM_TRAVEL_DISTANCES_CSV}") # 追加
    print(f"ACL_SCHEDULE_CSVのパス: {ACL_SCHEDULE_CSV}") # 追加

    # 試合結果と今後の試合日程を読み込む
    try:
        df_upcoming = pd.read_csv(UPCOMING_CSV)
        df_latest_results = pd.read_csv(LATEST_RESULTS_CSV)
        df_upcoming["__source_priority"] = 0
        df_latest_results["__source_priority"] = 1
        df_matches = pd.concat([df_upcoming, df_latest_results], ignore_index=True)
        print(f"試合日程データを {UPCOMING_CSV} および {LATEST_RESULTS_CSV} から読み込みました。")
    except FileNotFoundError as e:
        print(f"エラー: 試合日程ファイルが見つかりません。{e}")
        return
    except Exception as e:
        print(f"エラー: 試合日程データの読み込み中にエラーが発生しました。{e}")
        return

    # 同一試合の二重投入を避ける
    df_matches = dedupe_matches(df_matches)
    df_matches = df_matches.drop(columns=["__source_priority"], errors="ignore")

    # 移動距離データを読み込む（無ければホーム所在地一覧から生成）
    travel_distances_df = load_travel_distances()
    acl_team_events_df = load_acl_team_events(ACL_SCHEDULE_CSV)

    if df_matches.empty:
        print("処理する試合データがありません。")
        return

    # 疲労度を計算
    df_fatigue = calculate_fatigue(df_matches, travel_distances_df, acl_team_events_df)

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
