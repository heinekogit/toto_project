import os
from bs4 import BeautifulSoup
import csv

# チームID対応表
team_id_map = {
    "鹿島アントラーズ": "001",
    "川崎フロンターレ": "002",
    "セレッソ大阪": "003",
    "清水エスパルス": "004",
    "京都サンガF.C.": "005",
    "浦和レッズ": "006",
    "名古屋グランパス": "007",
    "柏レイソル": "008",
    "ＦＣ町田ゼルビア": "009",
    "ガンバ大阪": "010",
    "アビスパ福岡": "011",
    "アルビレックス新潟": "012",
    "サンフレッチェ広島": "013",
    "横浜Ｆ・マリノス": "014",
    "ヴィッセル神戸": "015",
    "ファジアーノ岡山": "016",
    "湘南ベルマーレ": "017",
    "ＦＣ東京": "018",
    "東京ヴェルディ": "019",
    "横浜ＦＣ": "020",
}

def extract_data(html_file, output_csv, league_class="J1"):
    with open(html_file, encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    output_rows = []

    # スタッツ名（日本語）をHTMLから抽出
    title_tag = soup.find('p', class_='ranking_txt_none')
    if title_tag:
        stat_label = title_tag.text.strip().split('　')[0]  # 全角スペースで分割（例：「平均ボール支配率　2025シーズン」）
    else:
        stat_label = 'value'  # fallback

    # 各チームのデータ抽出
    ranking_list = soup.find_all('li')
    for li in ranking_list:
        rank_tag = li.find('p', class_='number')
        team_tag = li.find('p', class_='team')
        value_tag = li.find('div', class_='ranking_stats') or \
                    li.find('div', class_='ranking_stats_1') or \
                    li.find('div', class_='ranking_stats_2') or \
                    li.find('div', class_='ranking_stats_3')

        if rank_tag and team_tag and value_tag:
            rank = rank_tag.text.strip()
            team_name = team_tag.text.strip()
            value = value_tag.find('p').text.strip()
            team_id = team_id_map.get(team_name, "")

            output_rows.append([team_id, league_class, team_name, rank, value])

    # team_idで昇順ソート
    output_rows.sort(key=lambda x: x[0])

    # 出力先フォルダ作成
    output_folder = 'stats_listed_csv'
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, output_csv)

    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダー（5列目がスタッツ名）
        writer.writerow(["team_id", "class", "team_name", "rank", stat_label])
        writer.writerows(output_rows)

# フォルダ内のHTMLファイルをすべて処理
input_folder = 'downloaded_html'
for filename in os.listdir(input_folder):
    if filename.endswith('.html'):
        html_path = os.path.join(input_folder, filename)
        csv_filename = filename.replace('.html', '.csv')
        extract_data(html_path, csv_filename)
