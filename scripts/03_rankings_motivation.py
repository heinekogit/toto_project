import os
import re
import glob
from datetime import datetime
import difflib
import unicodedata
import pandas as pd


SEASON_YEAR = os.environ.get("SEASON_YEAR", "2025")
LEAGUE = os.environ.get("LEAGUE", "j1").lower()
WINDOW_SIZES = os.environ.get("WINDOW_SIZES", "3,5")
POINTS_WEIGHT = float(os.environ.get("POINTS_WEIGHT", "0.1"))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_motivation.csv")
FUZZY_CUTOFF = float(os.environ.get("MOTIVATION_TEAM_FUZZY_CUTOFF", "0.62"))


def extract_date_from_filename(path):
    m = re.search(r"rankings_(\d{8})\.csv$", path)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()


def find_column(cols, keywords):
    for col in cols:
        for kw in keywords:
            if kw in col:
                return col
    return None


def normalize_team_text(value):
    s = str(value or "").strip()
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", "", s)
    # よくある装飾・記号を除去
    s = s.replace("・", "").replace(".", "").replace("’", "'").replace("'", "")
    s = s.replace("f.c", "fc").replace("ｆｃ", "fc")
    return s


def build_team_name_mapping(latest_names, start_names):
    """最新順位のチーム名を、開始時点順位のチーム名へ寄せるマッピングを作る。"""
    start_norm = {name: normalize_team_text(name) for name in start_names}
    latest_norm = {name: normalize_team_text(name) for name in latest_names}
    used_start = set()
    mapping = {}

    # 1) 完全一致（正規化後）
    reverse_start = {}
    for s_name, s_norm in start_norm.items():
        reverse_start.setdefault(s_norm, []).append(s_name)
    for l_name, l_norm in latest_norm.items():
        cands = reverse_start.get(l_norm, [])
        if len(cands) == 1:
            mapping[l_name] = cands[0]
            used_start.add(cands[0])

    # 2) 部分一致（短縮名: 福岡 -> アビスパ福岡 など）
    for l_name, l_norm in latest_norm.items():
        if l_name in mapping:
            continue
        cands = []
        for s_name, s_norm in start_norm.items():
            if s_name in used_start:
                continue
            if not l_norm or not s_norm:
                continue
            if l_norm in s_norm or s_norm in l_norm:
                # 長さ差が小さい候補を優先
                cands.append((abs(len(s_norm) - len(l_norm)), s_name))
        if len(cands) == 1:
            mapping[l_name] = cands[0][1]
            used_start.add(cands[0][1])
        elif len(cands) > 1:
            cands.sort(key=lambda x: x[0])
            mapping[l_name] = cands[0][1]
            used_start.add(cands[0][1])

    # 3) fuzzy（最終手段）
    start_norm_to_name = {v: k for k, v in start_norm.items() if k not in used_start}
    start_norm_list = list(start_norm_to_name.keys())
    for l_name, l_norm in latest_norm.items():
        if l_name in mapping:
            continue
        if not l_norm or not start_norm_list:
            continue
        cand = difflib.get_close_matches(l_norm, start_norm_list, n=1, cutoff=FUZZY_CUTOFF)
        if cand:
            s_name = start_norm_to_name[cand[0]]
            mapping[l_name] = s_name
            used_start.add(s_name)
            # 1対1維持
            start_norm_list.remove(cand[0])

    return mapping


def load_rankings(path):
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    team_col = find_column(cols, ["チーム", "クラブ"])
    rank_col = find_column(cols, ["順位"])
    points_col = find_column(cols, ["勝点", "勝ち点"])

    if not team_col or not rank_col or not points_col:
        raise ValueError(f"必要な列が見つかりません: {path}")

    out = df[[team_col, rank_col, points_col]].copy()
    out.columns = ["team_name", "rank", "points"]

    out["team_name"] = out["team_name"].astype(str).str.strip()
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["points"] = pd.to_numeric(out["points"], errors="coerce")
    out = out.dropna(subset=["team_name", "rank", "points"])
    return out


def compute_window(files, window_size):
    if len(files) < 2:
        return None

    use_files = files[-window_size:] if len(files) >= window_size else files
    start_path = use_files[0]
    end_path = use_files[-1]

    start_df = load_rankings(start_path)
    end_df = load_rankings(end_path)

    # チーム名表記が時点で変わるため、開始時点の名称へマップして結合する
    name_map = build_team_name_mapping(end_df["team_name"].tolist(), start_df["team_name"].tolist())
    end_df = end_df.copy()
    end_df["_team_name_start"] = end_df["team_name"].map(name_map).fillna(end_df["team_name"])
    start_df = start_df.copy()
    start_df = start_df.rename(columns={"team_name": "_team_name_start"})

    merged = pd.merge(
        end_df,
        start_df,
        on="_team_name_start",
        how="left",
        suffixes=("_latest", "_start"),
    )
    matched = merged["rank_start"].notna().sum()
    print(f"[INFO] motivation window={window_size}w team_match={matched}/{len(merged)}")

    merged[f"rank_change_{window_size}w"] = merged["rank_start"] - merged["rank_latest"]
    merged[f"points_change_{window_size}w"] = merged["points_latest"] - merged["points_start"]
    # 履歴不足（startが無い）時は「変化なし=0」で扱い、下流の欠損を防ぐ
    merged[f"rank_change_{window_size}w"] = merged[f"rank_change_{window_size}w"].fillna(0)
    merged[f"points_change_{window_size}w"] = merged[f"points_change_{window_size}w"].fillna(0)
    merged[f"motivation_score_{window_size}w"] = (
        merged[f"rank_change_{window_size}w"] + merged[f"points_change_{window_size}w"] * POINTS_WEIGHT
    )

    keep_cols = [
        "team_name",
        "rank_latest",
        "points_latest",
        f"rank_change_{window_size}w",
        f"points_change_{window_size}w",
        f"motivation_score_{window_size}w",
    ]
    return merged[keep_cols]


def main():
    pattern = os.path.join(DATA_DIR, f"{LEAGUE}_{SEASON_YEAR}_rankings_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"順位表ファイルが見つかりません: {pattern}")

    files_with_date = [(f, extract_date_from_filename(f)) for f in files]
    files_with_date = [x for x in files_with_date if x[1] is not None]
    files_with_date.sort(key=lambda x: x[1])
    files_sorted = [f for f, _ in files_with_date]

    window_sizes = [int(x.strip()) for x in WINDOW_SIZES.split(",") if x.strip().isdigit()]
    if not window_sizes:
        window_sizes = [3, 5]

    # ファイルが1つしか無い場合は最新順位のみ出力
    if len(files_sorted) == 1:
        latest = load_rankings(files_sorted[-1])
        latest = latest.rename(columns={"rank": "rank_latest", "points": "points_latest"})
        latest_date = files_with_date[-1][1].strftime("%Y%m%d")
        latest["season"] = SEASON_YEAR
        latest["fetched_date"] = latest_date
        os.makedirs(DATA_DIR, exist_ok=True)
        latest.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"出力: {OUTPUT_CSV}")
        return

    merged = None
    for w in window_sizes:
        part = compute_window(files_sorted, w)
        if part is None:
            continue
        if merged is None:
            merged = part
        else:
            merged = pd.merge(merged, part, on=["team_name", "rank_latest", "points_latest"], how="outer")

    if merged is None:
        raise RuntimeError("十分な順位表ファイルがありません。")

    latest_date = files_with_date[-1][1].strftime("%Y%m%d")
    merged["season"] = SEASON_YEAR
    merged["fetched_date"] = latest_date

    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"出力: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
