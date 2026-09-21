import pandas as pd
import numpy as np
import os
import math
import json
from pathlib import Path
from jleague_team_names import canonical_team_name
from acl_schedule import normalize_acl_schedule

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
EXTERNAL_MATCH_EVENTS_CSV = os.environ.get("EXTERNAL_MATCH_EVENTS_CSV", os.path.join(MANUAL_DIR, "external_match_events.csv"))

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
        travel_distances_df = normalize_distance_matrix(travel_distances_df)
        supplemental = os.path.join(MANUAL_DIR, "team_travel_distances_normalized_all.csv")
        if os.path.exists(supplemental):
            extra = normalize_distance_matrix(pd.read_csv(supplemental, sep="\t", index_col=0))
            travel_distances_df = travel_distances_df.combine_first(extra)
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


def normalize_distance_matrix(frame):
    out = frame.copy()
    out.index = [canonical_team_name(v) for v in out.index]
    out.columns = [canonical_team_name(v) for v in out.columns]
    if out.index.duplicated().any() or out.columns.duplicated().any():
        raise ValueError("duplicate teams after travel matrix normalization")
    return out.apply(pd.to_numeric, errors="coerce")


def load_external_team_events(path):
    """Verified non-league appearances; missing workload stays unknown, not zero evidence.

    One row per team/fixture. Expected columns: datetime, team, competition,
    source, status=confirmed. Optional: event_load, minutes_played, core_usage.
    Never import rotation hypotheses as actual appearances.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    out = pd.read_csv(path)
    required = {"datetime", "team", "competition", "source", "status"}
    if not required.issubset(out.columns):
        raise ValueError(f"external events missing columns: {sorted(required - set(out.columns))}")
    out = out[out.status.eq("confirmed")].copy()
    out["datetime"] = pd.to_datetime(out.datetime, errors="coerce")
    if out.datetime.isna().any() or out[["team", "competition", "source"]].isna().any().any():
        raise ValueError("confirmed external events need timestamp, team, competition and source")
    out["team"] = out.team.map(canonical_team_name)
    out["event_type"] = out.competition
    for col in ["event_load", "minutes_played", "core_usage", "extra_time_minutes"]:
        out[col] = pd.to_numeric(out.get(col, pd.Series(index=out.index, dtype=float)), errors="coerce")
    if (out.event_load.dropna() < 0).any() or (out.minutes_played.dropna() < 0).any() or (out.extra_time_minutes.dropna() < 0).any() or not out.core_usage.dropna().between(0, 1).all():
        raise ValueError("invalid external event workload")
    if out.duplicated(["team", "datetime"]).any():
        raise ValueError("duplicate external team event")
    return out


def load_confirmed_cup_events(data_dir):
    """Use local scored cup fixtures as appearances, never as assumed workloads."""
    root = Path(data_dir)
    events = []
    for manifest_path in sorted((root / 'toto_round_inputs').glob('toto*_matches.csv')):
        actual_path = root / 'eval/toto_rounds' / manifest_path.stem.removesuffix('_matches') / 'actual_results.csv'
        if not actual_path.exists():
            continue
        manifest = pd.read_csv(manifest_path)
        actual = pd.read_csv(actual_path)
        if 'competition' not in manifest or not {'home_score', 'away_score', 'status'}.issubset(actual):
            continue
        for _, match in manifest[manifest.competition.isin(['league_cup', 'emperor_cup', 'cup'])].iterrows():
            hit = actual[pd.to_numeric(actual.match_no, errors='coerce').eq(match.match_no)]
            if len(hit) != 1:
                continue
            row = hit.iloc[0]
            if row.status != 'OK' or pd.isna(row.home_score) or pd.isna(row.away_score):
                continue
            when = pd.to_datetime(row.datetime, errors='coerce')
            planned = pd.to_datetime(match.datetime, errors='coerce')
            if pd.isna(when) or pd.isna(planned) or when.date() != planned.date():
                continue
            if any(canonical_team_name(row[side]) != canonical_team_name(match[side]) for side in ['home_team', 'away_team']):
                continue
            for side in ['home_team', 'away_team']:
                events.append({'datetime': when, 'team': canonical_team_name(row[side]),
                               'event_type': match.competition, 'event_load': np.nan,
                               'minutes_played': np.nan, 'core_usage': np.nan,
                               'source': str(actual_path), 'status': 'confirmed'})
    return pd.DataFrame(events).drop_duplicates(['team', 'datetime']) if events else pd.DataFrame()


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

    # Scheduled 19:00 and actual 19:03 can have different IDs. They are one
    # fixture, otherwise the phantom first row resets rest to zero minutes.
    fallback_key = (
        df["datetime"].dt.strftime("%Y-%m-%d")
        + "|"
        + df.get("home_team", pd.Series("", index=df.index)).map(canonical_team_name)
        + "|"
        + df.get("away_team", pd.Series("", index=df.index)).map(canonical_team_name)
    )
    df["__dedupe_key"] = fallback_key

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
    empty_events = pd.DataFrame({
        "datetime": pd.Series(dtype="datetime64[ns]"),
        "team": pd.Series(dtype="object"),
        "event_type": pd.Series(dtype="object"),
        "event_load": pd.Series(dtype="float64"),
    })
    if not path or not os.path.exists(path):
        print(f"[INFO] ACL日程ファイルなし: {path}")
        return empty_events
    try:
        df = normalize_acl_schedule(path)
    except Exception as e:
        print(f"[WARN] ACL日程の読込失敗: {path} error={e}")
        return empty_events

    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols - set(df.columns))
        print(f"[WARN] ACL日程の必須列不足: missing={missing} path={path}")
        return empty_events

    out = df.copy().dropna(how="all")
    out["team"] = out["team"].astype(str).str.strip()
    out["datetime"] = pd.to_datetime(out["match_date"], errors="coerce")
    # The prediction stage applies the ACL grade once. Here the ACL row exists
    # only to shorten rest time and record the additional fixture.
    out["event_load"] = 0.0
    out = out.dropna(subset=["datetime"])
    out = out[out["team"].ne("")]
    if out.empty:
        return empty_events
    # 日付だけの入力を正午に寄せ、同日のリーグ戦より前の公式戦イベントとして扱う。
    out["datetime"] = out["datetime"].dt.normalize() + pd.to_timedelta(12, unit="h")
    out["event_type"] = "acl_event"
    out = out[["datetime", "team", "event_type", "event_load"]].drop_duplicates().sort_values("datetime", kind="mergesort")
    print(f"[INFO] ACLイベントを読込: rows={len(out)} path={path}")
    return out.reset_index(drop=True)


def calc_rest_fatigue(days_since_last_match):
    if pd.isna(days_since_last_match):
        return 0.0
    days = float(days_since_last_match)
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
    team_events = {}
    team_external_details = {}
    travel_distances_df = normalize_distance_matrix(travel_distances_df)
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
            team = canonical_team_name(row["team"])
            if team:
                # Do not count a supplemental fixture already in the league feed.
                same_day = df_matches[df_matches.datetime.dt.normalize().eq(row["datetime"].normalize())]
                if any(canonical_team_name(v) == team for c in ["home_team", "away_team"] for v in same_day[c]):
                    continue
                team_last_match_date[team] = row["datetime"]
                load = row.get("event_load", np.nan)
                push_recent_load(team_recent_loads, team, load if pd.notna(load) else 0.0)
                team_events.setdefault(team, []).append((row["datetime"], str(row.get("event_type", "external")), pd.notna(load)))
                team_external_details.setdefault(team, []).append({
                    'datetime': row['datetime'].isoformat(), 'competition': str(row.get('event_type', 'external')),
                    'minutes_played': float(row['minutes_played']) if pd.notna(row.get('minutes_played')) else None,
                    'core_usage': float(row['core_usage']) if pd.notna(row.get('core_usage')) else None,
                    'source': str(row.get('source', ACL_SCHEDULE_CSV)),
                    'event_load': float(load) if pd.notna(load) else None,
                    'participation_status': 'confirmed' if row.get('status') == 'confirmed' else 'scheduled',
                    'minutes_status': 'confirmed' if pd.notna(row.get('minutes_played')) else 'unknown',
                    'rotation_status': 'confirmed' if pd.notna(row.get('core_usage')) else 'unknown',
                    'extra_time_minutes': float(row['extra_time_minutes']) if pd.notna(row.get('extra_time_minutes')) else (
                        max(0.0, float(row['minutes_played']) - 90) if pd.notna(row.get('minutes_played')) and float(row['minutes_played']) >= 90 else None),
                    'extra_time_status': 'confirmed' if pd.notna(row.get('extra_time_minutes')) or (
                        pd.notna(row.get('minutes_played')) and float(row['minutes_played']) >= 90) else 'unknown',
                })
            continue

        match_datetime = row['datetime']
        home_team = canonical_team_name(row['home_team'])
        away_team = canonical_team_name(row['away_team'])
        # Do not carry an old fixture's travel load indefinitely across a break.
        for team in [home_team, away_team]:
            if team in team_last_match_date and match_datetime - team_last_match_date[team] > pd.Timedelta(days=14):
                team_recent_loads[team] = []

        home_rest_fatigue = 0.0
        if home_team in team_last_match_date:
            days_since_last_match = (match_datetime - team_last_match_date[home_team]).total_seconds() / 86400
            home_rest_fatigue = calc_rest_fatigue(days_since_last_match)
        home_recent_load_carry = calc_recent_carry(team_recent_loads.get(home_team, []))
        home_travel_fatigue = 0.0
        home_away_condition_penalty = 0.0
        home_fatigue = home_rest_fatigue + home_recent_load_carry

        away_rest_fatigue = 0.0
        if away_team in team_last_match_date:
            days_since_last_match = (match_datetime - team_last_match_date[away_team]).total_seconds() / 86400
            away_rest_fatigue = calc_rest_fatigue(days_since_last_match)
        away_recent_load_carry = calc_recent_carry(team_recent_loads.get(away_team, []))
        away_travel_fatigue = 0.0
        away_travel_distance_km = np.nan
        travel_lookup_status = "matrix_empty" if travel_distances_df.empty else "team_not_found"

        if not travel_distances_df.empty and away_team in travel_distances_df.index and home_team in travel_distances_df.columns:
            travel_distance = pd.to_numeric(travel_distances_df.loc[away_team, home_team], errors="coerce")
            if pd.notna(travel_distance):
                away_travel_distance_km = float(travel_distance)
                away_travel_fatigue = away_travel_distance_km * TRAVEL_DISTANCE_WEIGHT
                travel_lookup_status = "hit"
            else:
                travel_lookup_status = "distance_missing"

        away_away_condition_penalty = float(AWAY_GAME_PENALTY)
        away_fatigue = (
            away_rest_fatigue
            + away_recent_load_carry
            + away_travel_fatigue
            + away_away_condition_penalty
        )

        home_event_load = home_fatigue
        away_event_load = away_fatigue

        fatigue_scores.append({
            'match_id': row['match_id'],
            'datetime': match_datetime,
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_rest_fatigue': round(home_rest_fatigue, 4),
            'away_rest_fatigue': round(away_rest_fatigue, 4),
            'home_recent_load_carry': round(home_recent_load_carry, 4),
            'away_recent_load_carry': round(away_recent_load_carry, 4),
            'home_travel_fatigue': round(home_travel_fatigue, 4),
            'away_travel_fatigue': round(away_travel_fatigue, 4),
            'home_away_condition_penalty': round(home_away_condition_penalty, 4),
            'away_away_condition_penalty': round(away_away_condition_penalty, 4),
            'away_travel_distance_km': (
                round(away_travel_distance_km, 1) if pd.notna(away_travel_distance_km) else np.nan
            ),
            'travel_lookup_status': travel_lookup_status,
            'home_fatigue_score': round(home_fatigue, 2),
            'away_fatigue_score': round(away_fatigue, 2),
            'fatigue_measurement_version': 'conditions_v2',
            'external_schedule_coverage': 'partial' if external_team_events is not None and not external_team_events.empty else 'unknown',
            'travel_distance_basis': 'home_base_to_opponent_base_proxy',
        })
        for side, team in [('home', home_team), ('away', away_team)]:
            past = [(dt, kind, known) for dt, kind, known in team_events.get(team, []) if dt < match_datetime]
            recent = [(dt, kind, known) for dt, kind, known in past if match_datetime - dt <= pd.Timedelta(days=14)]
            last = team_last_match_date.get(team)
            fatigue_scores[-1].update({
                f'{side}_last_match_at': last,
                f'{side}_rest_hours': (match_datetime - last).total_seconds() / 3600 if last is not None else np.nan,
                f'{side}_matches_last7d': sum(match_datetime - dt <= pd.Timedelta(days=7) for dt, _, _ in recent),
                f'{side}_matches_last14d': len(recent),
                f'{side}_external_matches_last14d': sum(kind != 'league' for _, kind, _ in recent),
                f'{side}_external_load_unknown': any(kind != 'league' and not known for _, kind, known in recent),
                f'{side}_external_events_json': json.dumps([e for e in team_external_details.get(team, [])
                    if pd.Timedelta(0) < match_datetime - pd.Timestamp(e['datetime']) <= pd.Timedelta(days=14)], ensure_ascii=False),
            })
            team_events.setdefault(team, []).append((match_datetime, 'league', True))

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
        raise
    except Exception as e:
        print(f"エラー: 試合日程データの読み込み中にエラーが発生しました。{e}")
        raise

    # 同一試合の二重投入を避ける
    df_matches = dedupe_matches(df_matches)
    df_matches = df_matches.drop(columns=["__source_priority"], errors="ignore")

    # 移動距離データを読み込む（無ければホーム所在地一覧から生成）
    travel_distances_df = load_travel_distances()
    acl_team_events_df = load_acl_team_events(ACL_SCHEDULE_CSV)
    confirmed_cup_events = load_confirmed_cup_events(DATA_DIR)
    supplemental_events = load_external_team_events(EXTERNAL_MATCH_EVENTS_CSV)
    extra_frames = [f for f in [confirmed_cup_events, supplemental_events] if not f.empty]
    if extra_frames:
        acl_team_events_df = pd.concat([acl_team_events_df] + extra_frames, ignore_index=True)
        acl_team_events_df['team'] = acl_team_events_df.team.map(canonical_team_name)
        acl_team_events_df['datetime'] = pd.to_datetime(acl_team_events_df['datetime'], errors='coerce')
        acl_team_events_df = acl_team_events_df.dropna(subset=['datetime', 'team'])
        # One appearance per team/calendar day; supplemental verified data wins.
        acl_team_events_df['__day'] = acl_team_events_df.datetime.dt.normalize()
        acl_team_events_df = acl_team_events_df.drop_duplicates(['team', '__day'], keep='last').drop(columns='__day')

    if df_matches.empty:
        raise ValueError("処理する試合データがありません。")

    # 疲労度を計算
    df_fatigue = calculate_fatigue(df_matches, travel_distances_df, acl_team_events_df)
    validate_fatigue_output(df_fatigue)

    # 結果をCSVに保存
    if not df_fatigue.empty:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        df_fatigue.to_csv(OUTPUT_FATIGUE_CSV, index=False, encoding="utf-8-sig")
        print(f"疲労度スコアを {OUTPUT_FATIGUE_CSV} に保存しました。")
    else:
        print("計算された疲労度スコアがありません。")

def validate_fatigue_output(frame):
    """Fail before publishing a partial/legacy fatigue CSV."""
    required = {'travel_lookup_status', 'away_travel_distance_km', 'home_rest_hours', 'away_rest_hours',
                'home_fatigue_score', 'away_fatigue_score', 'fatigue_measurement_version'}
    if frame.empty or not required.issubset(frame):
        raise ValueError('FATIGUE_QC: empty or legacy output; regenerate fatigue data')
    bad = ~frame.travel_lookup_status.eq('hit')
    distance = pd.to_numeric(frame.away_travel_distance_km, errors='coerce')
    bad |= distance.isna() | ~np.isfinite(distance) | distance.lt(0)
    for col in ['home_fatigue_score', 'away_fatigue_score']:
        values = pd.to_numeric(frame[col], errors='coerce')
        bad |= values.isna() | ~np.isfinite(values) | values.lt(0)
    bad |= ~frame.fatigue_measurement_version.eq('conditions_v2')
    if bad.any():
        examples = frame.loc[bad, ['home_team', 'away_team', 'travel_lookup_status']].head(8).to_dict('records')
        raise ValueError(f'FATIGUE_QC: {int(bad.sum())}/{len(frame)} invalid rows: {examples}')
    print(f'FATIGUE_QC: PASS travel={len(frame)}/{len(frame)} version=conditions_v2')


if __name__ == "__main__":
    main()
