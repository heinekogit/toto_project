#!/usr/bin/env python3

import os
import requests
from time import sleep

# 保存先ディレクトリ
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '00_raw_html')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# J1クラブ一覧（クラブIDを手動で確認・調整可）
teams = {
    'sapporo': '北海道コンサドーレ札幌',
    'kashima': '鹿島アントラーズ',
    'urawa':    '浦和レッズ',
    'kashiwa': '柏レイソル',
    'ftokyo':  'FC東京',
    'tokyov': '東京ヴェルディ',
    'machida': '町田ゼルビア',
    'yokohamafm': '横浜F・マリノス',
    'yokohamafc': '横浜FC',
    'shonan': '湘南ベルマーレ',
    'niigata': 'アルビレックス新潟',
    'nagoya': '名古屋グランパス',
    'kyoto': '京都サンガF.C.',
    'gosaka': 'ガンバ大阪',
    'cosaka': 'セレッソ大阪',
    'kobe': 'ヴィッセル神戸',
    'hiroshima': 'サンフレッチェ広島',
    'fukuoka': 'アビスパ福岡',
    'okayama': 'ファジアーノ岡山',
    'shimizu': '清水エスパルス'
}

BASE_URL = 'https://www.jleague.jp/club/{}/profile/#player'

for club_id, club_name in teams.items():
    url = BASE_URL.format(club_id)
    try:
        response = requests.get(url)
        response.encoding = response.apparent_encoding
        if response.status_code == 200:
            output_file = os.path.join(OUTPUT_DIR, f'{club_id}.html')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f'Saved: {club_id} ({club_name})')
        else:
            print(f'Failed: {club_id} - Status {response.status_code}')
    except Exception as e:
        print(f'Error with {club_id}: {e}')
    sleep(1)  # サーバー負荷を考慮して待機
