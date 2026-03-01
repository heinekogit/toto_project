from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# ヘッドレスモードを無効にするための設定
options = Options()
# options.add_argument('--headless')  # ヘッドレスモードを無効にするため、コメントアウト

# ドライバーをセットアップ
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ターゲットのURLにアクセス
url = 'https://www.jleague.jp/stats/j1/club/2025/score_per_game/'
driver.get(url)

# ページが完全に読み込まれるまで待つ（適宜待機時間を調整）
time.sleep(5)

# ページのHTMLソースを取得
html = driver.page_source

# BeautifulSoupでHTMLを解析
soup = BeautifulSoup(html, 'html.parser')

# データを抽出
ranking_list = soup.select('ul.ranking_list')

# 結果を表示
for rank in ranking_list:
    rows = rank.find_all('li')
    for row in rows:
        team_name = row.find('a').get_text(strip=True) if row.find('a') else None
        stats = row.find('span').get_text(strip=True) if row.find('span') else None
        print(f"チーム名: {team_name}, スタッツ: {stats}")

# ブラウザを閉じる
driver.quit()
