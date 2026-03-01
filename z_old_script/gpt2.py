import requests
from bs4 import BeautifulSoup

# 対象の節（例：12節）
section = 12
url = f"https://www.jleague.jp/match/section/j1/{section}/"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# ファイル準備
filename = f"J1_第{section}節.txt"
with open(filename, "w", encoding="utf-8") as f:
    # 節タイトル
    round_title = soup.select_one("div.leagAccTit h5")
    if round_title:
        f.write("■ " + round_title.text.strip() + "\n")
        print("■ " + round_title.text.strip())

    # 各試合を取得
    matches = soup.select("table.matchTable tbody tr")
    for match in matches:
        stadium_td = match.select_one("td.stadium")
        match_td = match.select_one("td.match")

        if not stadium_td or not match_td:
            continue

        # キックオフ時刻とスタジアム名
        time_and_stadium = stadium_td.get_text(separator=" ", strip=True).split()
        kickoff = time_and_stadium[0]
        stadium = time_and_stadium[1] if len(time_and_stadium) > 1 else ""

        # チーム名とスコア
        team1 = match_td.select_one(".clubName.leftside")
        team2 = match_td.select_one(".clubName.rightside")
        score1 = match_td.select_one(".point.leftside")
        score2 = match_td.select_one(".point.rightside")

        team1_name = team1.text.strip() if team1 else "?"
        team2_name = team2.text.strip() if team2 else "?"
        score1_val = score1.text.strip() if score1 else "-"
        score2_val = score2.text.strip() if score2 else "-"

        # 出力
        line = f"{team1_name} {score1_val} vs {score2_val} {team2_name} | {kickoff} | {stadium}"
        print(line)
        f.write(line + "\n")

print(f"\n✅ 出力完了: {filename}")
