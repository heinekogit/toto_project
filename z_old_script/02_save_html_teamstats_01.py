import requests
import os

# 保存先フォルダ
save_folder = 'downloaded_html'
os.makedirs(save_folder, exist_ok=True)

# ダウンロード対象のURLと保存ファイル名のリスト
targets = [
["https://www.jleague.jp/stats/j1/club/2025/shoot_per_game/", "2025_shoot_per_game.html"],  #1試合平均シュート数
["https://www.jleague.jp/stats/j1/club/2025/shoot_on_target_per_game/", "2025_shoot_on_target_per_game.html"],  #1試合平均枠内シュート数
["https://www.jleague.jp/stats/j1/club/2025/shoot_rate/", "2025_shoot_rate.html"],  #シュート決定率
["https://www.jleague.jp/stats/j1/club/2025/score_per_game/", "2025_score_per_game.html"],  #1試合平均得点数
["https://www.jleague.jp/stats/j1/club/2025/pass_count_per_game/", "2025_pass_count_per_game.html"],  #1試合平均パス数
["https://www.jleague.jp/stats/j1/club/2025/pass_rate/", "2025_pass_rate.html"],  #パス成功率
["https://www.jleague.jp/stats/j1/club/2025/dribble_count_per_game/", "2025_dribble_count_per_game.html"],  #1試合平均ドリブル数
["https://www.jleague.jp/stats/j1/club/2025/dribble_rate/", "2025_dribble_rate.html"],  #ドリブル成功率
["https://www.jleague.jp/stats/j1/club/2025/through_pass_count_per_game/", "2025_through_pass_count_per_game.html"],  #1試合平均スルーパス数
["https://www.jleague.jp/stats/j1/club/2025/through_pass_rate/", "2025_through_pass_rate.html"],  #スルーパス成功率
["https://www.jleague.jp/stats/j1/club/2025/cross_count_per_game/", "2025_cross_count_per_game.html"],  #1試合平均クロス数
["https://www.jleague.jp/stats/j1/club/2025/air_battle_win_count_per_game/", "2025_air_battle_win_count_per_game.html"],  #1試合平均空中戦勝利数
["https://www.jleague.jp/stats/j1/club/2025/air_battle_win_rate/", "2025_air_battle_win_rate.html"],  #空中戦勝率
["https://www.jleague.jp/stats/j1/club/2025/ball_rate/", "2025_ball_rate.html"],  #平均ボール支配率
["https://www.jleague.jp/stats/j1/club/2025/chance_create_per_game/", "2025_chance_create_per_game.html"],  #1試合平均チャンスクリエイト数
["https://www.jleague.jp/stats/j1/club/2025/one_on_one_per_game/", "2025_one_on_one_per_game.html"],  #1試合平均1vs1勝利数
["https://www.jleague.jp/stats/j1/club/2025/recovery_count_per_game/", "2025_recovery_count_per_game.html"],  #1試合平均こぼれ球奪取数
["https://www.jleague.jp/stats/j1/club/2025/expected_goals/", "2025_expected_goals.html"],  #ゴール期待値
["https://www.jleague.jp/stats/j1/club/2025/distance_per_game/", "2025_distance_per_game.html"],  #1試合平均走行距離
["https://www.jleague.jp/stats/j1/club/2025/sprint_per_game/", "2025_sprint_per_game.html"],  #1試合平均スプリント回数
["https://www.jleague.jp/stats/j1/club/2025/at_sprint_per_game/", "2025_at_sprint_per_game.html"],  #1試合平均Atスプリント回数
["https://www.jleague.jp/stats/j1/club/2025/mt_sprint_per_game/", "2025_mt_sprint_per_game.html"],  #1試合平均Mtスプリント回数
["https://www.jleague.jp/stats/j1/club/2025/dt_sprint_per_game/", "2025_dt_sprint_per_game.html"],  #1試合平均Dtスプリント回数
["https://www.jleague.jp/stats/j1/club/2025/possession_distance_per_game/", "2025_possession_distance_per_game.html"],  #1試合平均ポゼッション時の走行距離
["https://www.jleague.jp/stats/j1/club/2025/possession_sprint_per_game/", "2025_possession_sprint_per_game.html"],  #1試合平均ポゼッション時のスプリント回数
["https://www.jleague.jp/stats/j1/club/2025/suffer_shoot_per_game/", "2025_suffer_shoot_per_game.html"],  #1試合平均被シュート数
["https://www.jleague.jp/stats/j1/club/2025/suffer_shoot_on_target_per_game/", "2025_suffer_shoot_on_target_per_game.html"],  #1試合平均被枠内シュート数
["https://www.jleague.jp/stats/j1/club/2025/lost_per_game/", "2025_lost_per_game.html"],  #1試合平均失点数
["https://www.jleague.jp/stats/j1/club/2025/clear_count_per_game/", "2025_clear_count_per_game.html"],  #1試合平均クリア数
["https://www.jleague.jp/stats/j1/club/2025/tackle_count_per_game/", "2025_tackle_count_per_game.html"],  #1試合平均タックル数
["https://www.jleague.jp/stats/j1/club/2025/tackle_rate/", "2025_tackle_rate.html"],  #タックル成功率
["https://www.jleague.jp/stats/j1/club/2025/block_count_per_game/", "2025_block_count_per_game.html"],  #1試合平均ブロック数
["https://www.jleague.jp/stats/j1/club/2025/intercept_count_per_game/", "2025_intercept_count_per_game.html"],  #1試合平均インターセプト数
["https://www.jleague.jp/stats/j1/club/2025/expected_goals_against/", "2025_expected_goals_against.html"],  #被ゴール期待値
["https://www.jleague.jp/stats/j1/club/2025/expected_goals_against_per_game/", "2025_expected_goals_against_per_game.html"],  #1試合平均被ゴール期待値
["https://www.jleague.jp/stats/j1/club/2025/expected_goals_against_excl_pk/", "2025_expected_goals_against_excl_pk.html"],  #被ゴール期待値 ※PKを除く
["https://www.jleague.jp/stats/j1/club/2025/clean_sheet/", "2025_clean_sheet.html"]  #クリーンシート総数
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
