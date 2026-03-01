import os
import pandas as pd
from report_view_utils import list_files, choose_file, write_html_table


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
    path = choose_file(files)
    if not path:
        return
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"読み込み失敗: {path} ({e})")
        return
    title = f"Backtest: {os.path.basename(path)}"
    os.makedirs(HTML_DIR, exist_ok=True)
    output_path = os.path.join(HTML_DIR, "backtest_view.html")
    write_html_table(df, title, DESC_MAP, output_path)
    print(f"HTML出力: {output_path}（元ファイル: {path}）")


if __name__ == "__main__":
    main()
