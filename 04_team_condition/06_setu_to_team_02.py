import pandas as pd

# CSVやExcelの読み込み
#   df = pd.read_csv("match_schedule.csv")  
# Excelなら 
df = pd.read_excel("J1_2025_schedule_by_structure.xlsx")

# チーム別に展開
records = []

for _, row in df.iterrows():
    round_ = row["節"]
    date = row["試合日"]
    home = row["home_team"]
    away = row["away_team"]
    home_score = row["home_score"]
    away_score = row["away_score"]

    # ホーム側
    records.append({
        "チーム名": home,
        "節": round_,
        "試合日": date,
        "対戦相手": away,
        "H/A": "H",
        "スコア": f"{home_score}-{away_score}"
    })
    # アウェイ側
    records.append({
        "チーム名": away,
        "節": round_,
        "試合日": date,
        "対戦相手": home,
        "H/A": "A",
        "スコア": f"{away_score}-{home_score}"
    })

# DataFrame化して保存
team_df = pd.DataFrame(records)
team_df.to_csv("team_schedule.csv", index=False)
