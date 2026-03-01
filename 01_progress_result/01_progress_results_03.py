import os
from progress_utils import (
    BASE_DIR,
    DATA_DIR,
    build_driver,
    fetch_team_stats,
    write_csv,
)

# チーム情報リスト（チームコード, チーム名）: 2025 J1 20クラブ
teams = [
    ("kashima", "鹿島アントラーズ"),
    ("urawa", "浦和レッズ"),
    ("kashiwa", "柏レイソル"),
    ("ftokyo", "ＦＣ東京"),
    ("tokyov", "東京ヴェルディ"),
    ("machida", "ＦＣ町田ゼルビア"),
    ("kawasakif", "川崎フロンターレ"),
    ("yokohamafm", "横浜F・マリノス"),
    ("yokohamafc", "横浜ＦＣ"),
    ("shonan", "湘南ベルマーレ"),
    ("niigata", "アルビレックス新潟"),
    ("shimizu", "清水エスパルス"),
    ("nagoya", "名古屋グランパス"),
    ("kyoto", "京都サンガＦ.Ｃ."),
    ("gosaka", "ガンバ大阪"),
    ("cosaka", "セレッソ大阪"),
    ("kobe", "ヴィッセル神戸"),
    ("okayama", "ファジアーノ岡山"),
    ("hiroshima", "サンフレッチェ広島"),
    ("fukuoka", "アビスパ福岡"),
]

OUTPUT_CSV = os.path.join(DATA_DIR, "club_team_results.csv")


def main():
    headless = os.environ.get("HEADLESS", "1") != "0"
    driver = build_driver(headless=headless)

    all_rows = []
    header = None

    try:
        for team_code, team_name in teams:
            print(f"{team_name}のデータを取得中...")
            headers, data_rows = fetch_team_stats(driver, team_code, team_name, wait_between=2)
            if not header:
                header = ["team_name"] + headers

            for row in data_rows:
                all_rows.append([team_name] + row)
            print(f"{team_name}のデータを取得しました。")
    finally:
        driver.quit()

    if not header or not all_rows:
        print("取得できるデータがありませんでした。")
        return

    write_csv(OUTPUT_CSV, header, all_rows)
    print(f"データを '{OUTPUT_CSV}' に保存しました。")


if __name__ == "__main__":
    main()
