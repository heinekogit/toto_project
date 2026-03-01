from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# ドライバーのパスは自動取得してくれると仮定（webdriver-manager を使ってる場合）
from webdriver_manager.chrome import ChromeDriverManager

# ブラウザ起動
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://www.jleague.jp/match/")

# 読み込みをちょっと待つ（JSの描画待ち）
time.sleep(3)

# HTMLを取得
html = driver.page_source

# BeautifulSoupで解析するなら
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'lxml')

# 例：見出しを全部取得
for h in soup.find_all('h3'):
    print(h.text)

# 最後にブラウザを閉じる
driver.quit()
