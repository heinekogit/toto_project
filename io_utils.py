import os
import pandas as pd
import numpy as np
import unicodedata


STADIUM_ALIASES = {
    "league": ["league", "リーグ", "category"],
    "team": ["team", "チーム"],
    "stadium_name": ["stadium_name", "stadium", "スタジアム", "スタジアム名"],
    "address": ["address", "住所"],
    "lat": ["lat", "latitude", "緯度"],
    "lon": ["lon", "lng", "longitude", "経度"],
    "stadium_id": ["stadium_id", "stadiumid", "スタジアムid", "スタジアムID", "会場ID"],
}

MATCH_ALIASES = {
    "match_id": ["match_id", "matchid", "試合ID"],
    "kickoff_jst": ["kickoff_jst", "kickoff", "キックオフ", "日時", "datetime"],
    "home_team": ["home_team", "ホーム", "home"],
    "away_team": ["away_team", "アウェイ", "away"],
    "stadium_name": ["stadium_name", "stadium", "スタジアム", "スタジアム名"],
    "stadium_id": ["stadium_id", "stadiumid", "スタジアムid", "スタジアムID", "会場ID"],
}


TEAM_NAME_ALIAS_RAW_MAP = {
    "横浜FM": "横浜FM",
    "横浜F・マリノス": "横浜FM",
    "横浜Fマリノス": "横浜FM",
    "川崎Ｆ": "川崎F",
    "川崎フロンターレ": "川崎F",
    "東京Ｖ": "東京V",
    "東京ヴェルディ": "東京V",
    "Ｇ大阪": "G大阪",
    "ガンバ大阪": "G大阪",
    "Ｃ大阪": "C大阪",
    "セレッソ大阪": "C大阪",
    "ＦＣ町田ゼルビア": "町田",
    "町田": "町田",
    "ヴィッセル神戸": "神戸",
    "ジェフ千葉": "千葉",
    "京都サンガ": "京都",
    "京都サンガFC": "京都",
    "ファジアーノ岡山": "岡山",
    "浦和レッズ": "浦和",
    "水戸ホーリーホック": "水戸",
    "清水エスパルス": "清水",
    "アビスパ福岡": "福岡",
    "鹿島アントラーズ": "鹿島",
    "柏レイソル": "柏",
    "名古屋グランパス": "名古屋",
    "サンフレッチェ広島": "広島",
    "V・ファーレン長崎": "長崎",
    "Ｖ・ファーレン長崎": "長崎",
    "ベガルタ仙台": "仙台",
    "ブラウブリッツ秋田": "秋田",
    "モンテディオ山形": "山形",
    "いわきFC": "いわき",
    "RB大宮アルディージャ": "大宮",
    "横浜FC": "横浜FC",
    "湘南ベルマーレ": "湘南",
    "ヴァンフォーレ甲府": "甲府",
    "ジュビロ磐田": "磐田",
    "ヴァンラーレ八戸": "八戸",
    "テゲバジャーロ宮崎": "宮崎",
    "北海道コンサドーレ札幌": "札幌",
    "FC今治": "今治",
    "愛媛FC": "愛媛",
    "愛媛ＦＣ": "愛媛",
    "ロアッソ熊本": "熊本",
    "レノファ山口FC": "山口",
    "レノファ山口ＦＣ": "山口",
    "アルビレックス新潟": "新潟",
    "カターレ富山": "富山",
    "徳島ヴォルティス": "徳島",
    "大分トリニータ": "大分",
    "サガン鳥栖": "鳥栖",
    "藤枝MYFC": "藤枝",
    "藤枝ＭＹＦＣ": "藤枝",
    "栃木シティ": "栃木C",
    "栃木Ｃ": "栃木C",
    "未定": "未定",
}


def _normalize_team_text(text):
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("　", " ").strip()
    s = s.replace("Ｆ", "F").replace("Ｃ", "C").replace("Ｖ", "V")
    s = s.upper()
    s = s.replace(" ", "").replace("・", "").replace(".", "")
    return s


TEAM_NAME_ALIAS_MAP = {
    _normalize_team_text(k): _normalize_team_text(v)
    for k, v in TEAM_NAME_ALIAS_RAW_MAP.items()
}


def canonical_team_name(name):
    if pd.isna(name):
        return None
    text = _normalize_team_text(name)
    return TEAM_NAME_ALIAS_MAP.get(text, text)


def _read_file(path, sheet=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def _normalize_columns(df, aliases):
    mapping = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for target, keys in aliases.items():
        for key in keys:
            key_lower = key.lower()
            if key_lower in lower_cols:
                mapping[lower_cols[key_lower]] = target
                break
    return df.rename(columns=mapping)


def _coerce_lat_lon(df):
    if "lat" not in df.columns:
        return df
    if "lon" not in df.columns:
        df["lon"] = np.nan

    lat_num = pd.to_numeric(df["lat"], errors="coerce")
    lon_num = pd.to_numeric(df["lon"], errors="coerce")

    pair = (
        df["lat"]
        .astype(str)
        .str.extract(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$")
    )
    lat_from_pair = pd.to_numeric(pair[0], errors="coerce")
    lon_from_pair = pd.to_numeric(pair[1], errors="coerce")

    df["lat"] = lat_num.fillna(lat_from_pair)
    df["lon"] = lon_num.fillna(lon_from_pair)
    return df


def load_stadiums(path, sheet="stadiums"):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        try:
            df = _read_file(path, sheet=sheet)
        except ValueError as e:
            msg = str(e)
            if "Worksheet named" in msg:
                xls = pd.ExcelFile(path)
                fallback_sheet = None
                if "studiums" in xls.sheet_names:
                    fallback_sheet = "studiums"
                elif len(xls.sheet_names) > 0:
                    fallback_sheet = xls.sheet_names[0]
                if fallback_sheet is None:
                    raise
                print(f"[WARN] stadiums sheet '{sheet}' が見つからないため '{fallback_sheet}' を使用します。")
                df = _read_file(path, sheet=fallback_sheet)
            else:
                raise
    else:
        df = _read_file(path, sheet=sheet)
    df = _normalize_columns(df, STADIUM_ALIASES)
    df = _coerce_lat_lon(df)
    for col in ["league", "team", "stadium_name", "address", "stadium_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ["lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_matches(path, sheet=None):
    df = _read_file(path, sheet=sheet)
    df = _normalize_columns(df, MATCH_ALIASES)
    for col in ["match_id", "home_team", "away_team", "stadium_name", "stadium_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "kickoff_jst" in df.columns:
        df["kickoff_jst"] = pd.to_datetime(df["kickoff_jst"], errors="coerce")
    return df


def merge_matches_with_stadiums(matches_df, stadiums_df):
    key = None
    if "stadium_id" in matches_df.columns and "stadium_id" in stadiums_df.columns:
        key = "stadium_id"
    elif "stadium_name" in matches_df.columns and "stadium_name" in stadiums_df.columns:
        key = "stadium_name"

    if not key:
        raise ValueError("stadium_id または stadium_name が一致する列にありません。")

    merged = pd.merge(matches_df, stadiums_df, on=key, how="left", suffixes=("", "_stadium"))

    # フォールバック: スタジアム略称差で未結合の場合はホームチームで補完
    if "lat" in merged.columns and "lon" in merged.columns and "home_team" in merged.columns and "team" in stadiums_df.columns:
        missing_mask = merged["lat"].isna() | merged["lon"].isna()
        if missing_mask.any():
            team_lookup = stadiums_df.copy()
            team_lookup["_team_key"] = team_lookup["team"].map(canonical_team_name)
            team_lookup = team_lookup.dropna(subset=["_team_key"])
            team_lookup = team_lookup.drop_duplicates(subset=["_team_key"], keep="first")
            team_lookup = team_lookup.set_index("_team_key")

            merged["_home_team_key"] = merged["home_team"].map(canonical_team_name)

            fill_lat = merged["_home_team_key"].map(team_lookup["lat"]) if "lat" in team_lookup.columns else None
            fill_lon = merged["_home_team_key"].map(team_lookup["lon"]) if "lon" in team_lookup.columns else None

            if fill_lat is not None:
                merged.loc[missing_mask, "lat"] = merged.loc[missing_mask, "lat"].fillna(fill_lat[missing_mask])
            if fill_lon is not None:
                merged.loc[missing_mask, "lon"] = merged.loc[missing_mask, "lon"].fillna(fill_lon[missing_mask])
            merged = merged.drop(columns=["_home_team_key"], errors="ignore")

    return merged, key
