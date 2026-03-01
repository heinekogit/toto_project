from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException  # ここでTimeoutExceptionをインポート

# WebDriverの設定
driver = webdriver.Chrome()

# 対象ページにアクセス
driver.get("https://www.jleague.jp/club/kashima/day/#result")  # 正しいURLに置き換えてください

# ページが完全に読み込まれるまで待機
WebDriverWait(driver, 30).until(
    lambda driver: driver.execute_script('return document.readyState') == 'complete'
)

# 成績・データボタン（リンク）のクリックが可能になるまで待機
WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.XPATH, "/html/body/div[7]/div[1]/section/nav/ul/li[4]/a"))
)

# 成績・データリンクをクリック
result_link = driver.find_element(By.XPATH, "/html/body/div[7]/div[1]/section/nav/ul/li[4]/a")
result_link.click()

# 成績・データページが表示されるのを待機（別の条件で）
try:
    WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.clubResult"))  # 要素が可視化されるまで待機
    )
    print("成績・データページに遷移しました。")
except TimeoutException:
    print("成績・データページの表示に失敗しました。ページを再確認してください。")
