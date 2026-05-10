# Weather Features Fetcher (Open-Meteo)

## 概要
スタジアム座標と試合日程から、Open-Meteoの時間別予報を取得し、試合単位の天候特徴量を出力します。

## 入力
### スタジアム一覧（Excel/CSV）
最低限必要な列（列名は可変対応）:
- `league`
- `team`
- `stadium_name`
- `address`
- `lat`
- `lon`
- （任意）`stadium_id`

### 試合日程（Excel/CSV）
最低限必要な列:
- `match_id`
- `kickoff_jst`（例: `2026-03-14 14:00`）
- `home_team`
- `away_team`
- `stadium_name` または `stadium_id`

## 取得対象（Open-Meteo）
Hourly項目:
- `temperature_2m`
- `precipitation`
- `wind_speed_10m`
- `weather_code`

`timezone=Asia/Tokyo` を指定しています。

## 出力
### 生データ（キャッシュ）
`weather_raw/` に日付・座標単位で保存されます。

### 特徴量CSV
`weather_features.csv`（デフォルト）

主な列:
- `match_id`
- `kickoff_jst`
- `stadium_name`
- `lat`, `lon`
- `temp_kickoff`
- `precip_kickoff`
- `wind_kickoff`
- `code_kickoff`
- `temp_avg_pm1h`
- `precip_sum_pm1h`
- `wind_max_pm1h`
- `is_rain`
- `is_heavy_rain`
- `is_strong_wind`

## 実行例
```
python fetch_weather_features.py \
  --stadiums stadiums_2026.xlsx \
  --stadiums-sheet stadiums \
  --matches matches_2026.csv \
  --out weather_features.csv
```

## ログ
`logs/weather_fetch.log` に取得失敗の記録を残します。

## 依存ライブラリ
- pandas
- openpyxl（Excel読み込み用）
- requests
