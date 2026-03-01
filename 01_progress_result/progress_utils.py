import csv
import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")


def build_driver(headless=True):
    cache_dir = os.path.join(DATA_DIR, "selenium_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["SELENIUM_MANAGER_CACHE_PATH"] = cache_dir

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--no-first-run")

    profile_dir = os.path.join(DATA_DIR, "selenium_profile", str(int(time.time() * 1000)))
    os.makedirs(profile_dir, exist_ok=True)
    options.add_argument(f"--user-data-dir={profile_dir}")

    driver_path = os.path.join(BASE_DIR, "chromedriver")
    if os.path.exists(driver_path) and os.access(driver_path, os.X_OK):
        service = Service(executable_path=driver_path)
        return webdriver.Chrome(service=service, options=options)

    return webdriver.Chrome(options=options)


def wait_for_ready_state(driver, timeout=20):
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def wait_for_result_table(driver, timeout=20):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#clubTeamDateTab01 .dataTable"))
    )


def open_result_table(driver, team_code, team_name):
    url = f"https://www.jleague.jp/club/{team_code}/day/#result"
    driver.get(url)
    wait_for_ready_state(driver, timeout=30)

    try:
        return wait_for_result_table(driver, timeout=20)
    except TimeoutException:
        pass

    try:
        result_link = driver.find_element(By.CSS_SELECTOR, "a[href*='#result']")
        result_link.click()
        return wait_for_result_table(driver, timeout=20)
    except Exception:
        raise TimeoutException(f"{team_name}の成績テーブルが取得できませんでした。")


def parse_table(table):
    rows = table.find_elements(By.TAG_NAME, "tr")
    header_cells = rows[0].find_elements(By.TAG_NAME, "th") if rows else []
    headers = [cell.text.replace("\n", " ").strip() for cell in header_cells]

    data_rows = []
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if not cols:
            continue
        values = [col.text.replace("\n", " ").strip() for col in cols]
        data_rows.append(values)

    if not headers and data_rows:
        headers = [f"col_{i+1}" for i in range(len(data_rows[0]))]

    return headers, data_rows


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            writer.writerow(row[: len(header)])


def fetch_team_stats(driver, team_code, team_name, wait_between=2):
    table = open_result_table(driver, team_code, team_name)
    headers, data_rows = parse_table(table)
    time.sleep(wait_between)
    return headers, data_rows
