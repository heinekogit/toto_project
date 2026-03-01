from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# WebDriverのオプションを設定
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # ヘッドレスモード
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# ドライバの自動インストールと設定
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# スクレイピングしたいURLにアクセス
driver.get("https://spaia.jp/football/jleague/j1/stats/team")

# 特定の要素を取得
title = driver.find_element(By.TAG_NAME, "h1")
print("Title of the page:", title.text)

# 必要な操作を行った後
driver.quit()
