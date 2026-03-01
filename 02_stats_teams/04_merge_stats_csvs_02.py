import os
import csv

input_folder = 'stats_listed_csv'
output_file = 'compiled_stats.csv'

team_data = {}
stat_columns = []

# 入力フォルダのすべてのCSVファイルを処理
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # スタッツ名を取得（value以外の5列目）
            if headers[4] == 'value':
                # ファイル名から取得（例: '2025_shoot.csv' → '2025_shoot' → スタッツ名）
                stat_label = filename.replace('.csv', '')
            else:
                stat_label = headers[4]  # 正常なラベルがある場合

            if stat_label not in stat_columns:
                stat_columns.append(stat_label)

            for row in reader:
                tid = row['team_id']
                if tid not in team_data:
                    team_data[tid] = {
                        'team_id': tid,
                        'class': row['class'],
                        'team_name': row['team_name']
                    }
                team_data[tid][stat_label] = row[headers[4]]  # 5列目の値

# 昇順でチームIDを並べる
sorted_teams = sorted(team_data.values(), key=lambda x: x['team_id'])

# 出力
with open(output_file, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["team_id", "class", "team_name"] + stat_columns)
    for team in sorted_teams:
        row = [team['team_id'], team['class'], team['team_name']]
        for stat in stat_columns:
            row.append(team.get(stat, ""))
        writer.writerow(row)

print(f"[OK] 統合ファイルを出力しました: {output_file}")
