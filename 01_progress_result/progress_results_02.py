import os
from progress_utils import (
    DATA_DIR,
    build_driver,
    fetch_team_stats,
    write_csv,
)


TEAM_CODE = "kashima"
TEAM_NAME = "鹿島アントラーズ"
OUTPUT_CSV = os.path.join(DATA_DIR, "club_team_results_single.csv")


def main():
    headless = os.environ.get("HEADLESS", "1") != "0"
    driver = build_driver(headless=headless)

    try:
        headers, data_rows = fetch_team_stats(driver, TEAM_CODE, TEAM_NAME, wait_between=1)
    finally:
        driver.quit()

    if not headers or not data_rows:
        print("取得できるデータがありませんでした。")
        return

    write_csv(OUTPUT_CSV, ["team_name"] + headers, [[TEAM_NAME] + row for row in data_rows])
    print(f"データを '{OUTPUT_CSV}' に保存しました。")


if __name__ == "__main__":
    main()
