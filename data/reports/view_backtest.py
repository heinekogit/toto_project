import os
import re
import pandas as pd
from report_view_utils import list_files, write_html_table_grouped


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HTML_DIR = os.path.join(os.path.dirname(__file__), "html")
PATTERNS = [
    os.path.join(ROOT_DIR, "backtest_*_*.csv"),
]

DESC_MAP = {
    "match_id": "試合ID",
    "datetime": "試合日時",
    "stadium": "スタジアム",
    "home_team": "ホーム",
    "away_team": "アウェイ",
    "home_score": "ホーム得点",
    "away_score": "アウェイ得点",
    "prob_home_win": "ホーム勝ち確率",
    "prob_draw": "引き分け確率",
    "prob_away_win": "アウェイ勝ち確率",
    "predicted_highest_prob_result": "予測結果",
    "actual_result": "実際の結果",
    "is_correct": "的中",
}


def main():
    files = list_files(PATTERNS)
    if not files:
        print("対象ファイルが見つかりません。")
        return

    targets = []
    for path in files:
        name = os.path.basename(path)
        m = re.match(r"^backtest_(j[123])_(\d{4})\.csv$", name, flags=re.IGNORECASE)
        if not m:
            continue
        league = m.group(1).upper()
        year = int(m.group(2))
        targets.append((year, league, path))

    if not targets:
        print("対象バックテストCSVが見つかりません。")
        return

    latest_year = max(year for year, _, _ in targets)
    latest_targets = [(league, path) for year, league, path in sorted(targets) if year == latest_year]

    parts = []
    source_names = []
    for league, path in latest_targets:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"読み込み失敗: {path} ({e})")
            continue
        if df.empty:
            continue
        if "league" not in df.columns:
            df.insert(0, "league", league)
        else:
            df["league"] = league
        parts.append(df)
        source_names.append(os.path.basename(path))

    if not parts:
        print("表示対象のバックテストCSVを読み込めませんでした。")
        return

    merged = pd.concat(parts, ignore_index=True)
    if "datetime" in merged.columns:
        merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce")
        merged = merged.sort_values(["league", "datetime"], na_position="last").reset_index(drop=True)
    else:
        merged = merged.sort_values(["league"]).reset_index(drop=True)

    title = f"Backtest {latest_year}: {', '.join(source_names)}"
    os.makedirs(HTML_DIR, exist_ok=True)
    output_path = os.path.join(HTML_DIR, "backtest_view.html")
    write_html_table_grouped(merged, title, DESC_MAP, output_path, group_col="league", group_order=["J1", "J2", "J3"])
    print(f"HTML出力: {output_path}（元ファイル: {', '.join(source_names)}）")


if __name__ == "__main__":
    main()
