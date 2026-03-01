import requests
import os

# 保存先フォルダ
save_folder = 'downloaded_html'
os.makedirs(save_folder, exist_ok=True)

# ダウンロード対象のURLと保存ファイル名のリスト
targets = [
    ["https://www.jleague.jp/stats/j1/club/2025/score_per_game/", "2025_score_average.html"],
    ["https://www.jleague.jp/stats/j1/club/2025/shoot_per_game/", "2025_shoot_average.html"],
]

for url, filename in targets:
    try:
        response = requests.get(url)
        response.raise_for_status()  # エラーチェック

        file_path = os.path.join(save_folder, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Saved: {file_path}")

    except Exception as e:
        print(f"Error downloading {url}: {e}")
