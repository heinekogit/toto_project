from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv

# ドライバの設定
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # ヘッドレスモード
driver = webdriver.Chrome(options=options)

url = "https://spaia.jp/football/jleague/j1/stats/team"

# ページを開く
driver.get(url)

# ページが完全に読み込まれるのを待機（最大10秒）
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#stats-team")))

# スタッツの取得
team_stats = []
stats_elements = driver.find_elements(By.CSS_SELECTOR, "#stats-team > div.main > div.team-stats-pane > div.main > div.stats-table")

# スタッツとチーム名を関連付けて取得
for stats_element in stats_elements:
    team_name = stats_element.find_element(By.CSS_SELECTOR, ".team-name").text
    stats = stats_element.text.split("\n")  # 各行を分割してリストに変換
    if len(stats) >= 3:  # 必要なスタッツが含まれているかチェック
        team_stat = {
            "team": team_name,  # チーム名
            "goals": stats[0],  # 得点
            "conceded": stats[1],  # 失点
            "matches": stats[2]   # 試合数
        }
        team_stats.append(team_stat)

# 結果をCSVファイルに保存
output_file = "team_stats_with_data.csv"
with open(output_file, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(["Team Name", "Goals", "Conceded", "Matches"])  # ヘッダー行
    for stats in team_stats:
        writer.writerow([stats["team"], stats["goals"], stats["conceded"], stats["matches"]])

print(f"データが {output_file} に保存されました。")

# ドライバを終了
driver.quit()
