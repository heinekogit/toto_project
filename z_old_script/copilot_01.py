from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Seleniumのセットアップ
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
url = "https://spaia.jp/football/jleague/j1/stats/team"
driver.get(url)

# ページの読み込みを待つ
driver.implicitly_wait(10)

# ページソースを取得
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

# 必要なデータを確認
print(soup.prettify())

# ブラウザを閉じる
driver.quit()