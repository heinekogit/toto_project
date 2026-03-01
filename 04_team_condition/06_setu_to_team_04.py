import pandas as pd
from collections import defaultdict

# === ファイル名 ===
input_file = "J1_2025_schedule_by_structure.xlsx"
output_file = "J1_2025_team_schedule_with_opponent.xlsx"

# === データ読み込み ===
df = pd.read_excel(input_file)

# === チーム別行に変換（H/A, 試合日, 対戦相手） ===
records = []
for _, row in df.iterrows():
    round_ = row["節"]
    date = row["試合日"]
    home = row["home_team"]
    away = row["away_team"]

    # ホーム側
    records.append({
        "チーム名": home,
        "節": round_,
        "H/A": "ホーム",
        "試合日": date,
        "相手": away
    })
    # アウェイ側
    records.append({
        "チーム名": away,
        "節": round_,
        "H/A": "アウェイ",
        "試合日": date,
        "相手": home
    })

df_team = pd.DataFrame(records)

# === ピボット処理：節ごとに「H/A」「試合日」「相手」を展開 ===
pivot = df_team.pivot_table(index="チーム名", columns="節", values=["H/A", "試合日", "相手"], aggfunc="first")

# 列名を整形（例: H/A_第1節）
pivot.columns = [f"{v}_{k}" for v, k in pivot.columns]
pivot = pivot.reset_index()

# === 節単位に「H/A → 試合日 → 相手」順に並べ替え ===
cols = [col for col in pivot.columns if col != "チーム名"]
col_groups = defaultdict(dict)

for col in cols:
    if "_" in col:
        kind, round_ = col.split("_", 1)
        col_groups[round_][kind] = col

ordered_cols = ["チーム名"]
for round_ in sorted(col_groups.keys(), key=lambda x: int(x.replace("第", "").replace("節", ""))):
    for kind in ["H/A", "試合日", "相手"]:
        if kind in col_groups[round_]:
            ordered_cols.append(col_groups[round_][kind])

pivot = pivot[ordered_cols]

# === 出力 ===
pivot.to_excel(output_file, index=False)
print(f"✅ 出力完了: {output_file}")
