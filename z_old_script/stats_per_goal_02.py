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

# メタ行を表示
print("順位,チーム名,スタッツ")

# ランキングリストを取得
ranking_list = soup.select('ul.ranking_list')  # 正しいセレクタを確認

# 結果を表示
for rank in ranking_list:
    rows = rank.find_all('li')  # 各ランキングアイテムを取得
    for row in rows:
        # 順位、チーム名、スタッツを取得
        rank_position = row.find('span', class_='rank-position').get_text(strip=True) if row.find('span', class_='rank-position') else None
        team_name = row.find('a').get_text(strip=True) if row.find('a') else None
        stats_text = row.find('span', class_='stats').get_text(strip=True) if row.find('span', class_='stats') else None

        # statsが取得できた場合に処理
        if team_name and stats_text:
            try:
                # スタッツの数値部分を取得（文字列の分割）
                stats = stats_text.split()[0]  # 数字部分のみを取得
            except IndexError:
                stats = "不明"
        else:
            stats = "不明"  # statsがない場合は "不明" を設定

        # データ行を表示（順位, チーム名, スタッツ）
        if rank_position and team_name:
            print(f"{rank_position},{team_name},{stats}")

# ブラウザを閉じる
driver.quit()
