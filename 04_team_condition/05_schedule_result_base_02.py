from bs4 import BeautifulSoup
import csv
import re

with open("日程_結果1.txt", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

rows = []
current_date = None
current_section = None

# 全体を順に走査しながら、節・日付・試合を紐付け
for tag in soup.body.descendants:
    if tag.name == "h3" and "contentsHeadline04" in tag.get("class", []):
        current_section = tag.get_text(strip=True)

    elif tag.name == "h4" and "leftRedTit" in tag.get("class", []):
        date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', tag.text)
        if date_match:
            y, m, d = map(int, date_match.groups())
            current_date = f"{y:04d}-{m:02d}-{d:02d}"

    elif tag.name == "table" and "matchTable" in tag.get("class", []):
        for tr in tag.select("tbody > tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            time_stadium = tds[0]
            time = time_stadium.contents[0].strip()
            stadium_tag = time_stadium.find("a")
            stadium = stadium_tag.text.strip() if stadium_tag else ""

            game_table = tds[1].find("table", class_="gameTable")
            if not game_table:
                continue
            cols = game_table.find_all("td")
            if len(cols) < 5:
                continue

            home_team = cols[0].text.strip()
            home_score = cols[1].text.strip()
            status = cols[2].text.strip()
            away_score = cols[3].text.strip()
            away_team = cols[4].text.strip()

            # URLとmatch_id
            a_tag = tds[1].find("a", href=True)
            match_code = ""
            match_id = ""
            match_url = comment_url_player = comment_url_coach = tracking_data_url = photo_url = recap_url = ""
            if a_tag:
                href = a_tag["href"]
                match_code_match = re.search(r'/match/j1/2024/(\d+)/', href)
                if match_code_match:
                    match_code = match_code_match.group(1)
                    match_id = f"j1_2024_{match_code}"
                    match_url = f"/match/j1/2024/{match_code}/live/"
                    comment_url_player = f"/match/j1/2024/{match_code}/player/"
                    comment_url_coach = f"/match/j1/2024/{match_code}/coach/"
                    tracking_data_url = f"/match/j1/2024/{match_code}/trackingdata/"
                    photo_url = f"/match/j1/2024/{match_code}/photo/"
                    recap_url = f"/match/j1/2024/{match_code}/recap/"

            datetime = f"{current_date} {time}"

            rows.append([
                current_section, current_date, match_id, datetime, stadium,
                home_team, away_team, home_score, away_score, status,
                match_url, comment_url_player, comment_url_coach,
                tracking_data_url, photo_url, recap_url
            ])

# CSV出力
with open("J1_2024_schedule_by_structure.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "第●節", "試合日", "match_id", "datetime", "stadium",
        "home_team", "away_team", "home_score", "away_score", "status",
        "match_url", "comment_url_player", "comment_url_coach",
        "tracking_data_url", "photo_url", "recap_url"
    ])
    writer.writerows(rows)
