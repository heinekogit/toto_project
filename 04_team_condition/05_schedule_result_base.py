from bs4 import BeautifulSoup
import csv
import re

with open("data_list/日程_結果1.txt", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

results = []
current_section = ""
current_date = ""

# HTML要素を順番にたどる（h3, h4, table）
for tag in soup.body.descendants:
    if tag.name == "h3" and "section" in tag.get("class", []):
        current_section = tag.get_text(strip=True)

    elif tag.name == "h4" and "leftRedTit" in tag.get("class", []):
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", tag.get_text())
        if m:
            current_date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    elif tag.name == "table" and "matchTable" in tag.get("class", []):
        try:
            td_stadium = tag.find("td", class_="stadium")
            kickoff = td_stadium.contents[0].strip()
            stadium = td_stadium.find("a").text.strip()

            home_team = tag.select_one("td.clubName.leftside").text.strip()
            away_team = tag.select_one("td.clubName.rightside").text.strip()
            home_score = tag.select_one("td.point.leftside").text.strip()
            away_score = tag.select_one("td.point.rightside").text.strip()
            status = tag.select_one("td.status span").text.strip()

            match_a = tag.find("a", href=re.compile(r"/match/j1/2025/\d+/live/"))
            match_id_part = re.search(r"/match/j1/2025/(\d+)/", match_a["href"]).group(1)
            match_id = f"j1_2025_{match_id_part}"
            base_url = f"/match/j1/2025/{match_id_part}"

            result = {
                "第●節": current_section,
                "試合日": current_date,
                "match_id": match_id,
                "datetime": f"{current_date} {kickoff}",
                "stadium": stadium,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "status": status,
                "match_url": base_url + "/live/",
                "comment_url_player": base_url + "/player/",
                "comment_url_coach": base_url + "/coach/",
                "tracking_data_url": base_url + "/trackingdata/",
                "photo_url": base_url + "/photo/",
                "recap_url": base_url + "/recap/",
            }

            results.append(result)

        except Exception as e:
            print("エラー:", e)
            continue

# CSV出力（ファイル名を変更する場合はここ）
with open("J1_2025_parsed.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "第●節", "試合日", "match_id", "datetime", "stadium",
        "home_team", "away_team", "home_score", "away_score", "status",
        "match_url", "comment_url_player", "comment_url_coach",
        "tracking_data_url", "photo_url", "recap_url"
    ])
    writer.writeheader()
    writer.writerows(results)
