import pandas as pd
from collections import defaultdict

# === ファイル名 ===
input_file = "J1_2025_schedule_by_structure.xlsx"
output_file = "J1_2025_team_schedule_pivot_ordered.xlsx"

# === データ読み込み ===
df = pd.read_excel(input_file)

# === チームごとにH/Aと日付を記録（縦持ち形式）
records = []
for _, row in df.iterrows():
    round_ = row["節"]
    date = row["試合日"]
    home = row["home_team"]
    away = row["away_team"]
    
    records.append({"チーム名": home, "節": round_, "H/A": "ホーム", "試合日": date})
    records.append({"チーム名": away, "節": round_, "H/A": "アウェイ", "試合日": date})

df_team = pd.DataFrame(records)

# === ピボット処理（節ごとに H/A と 試合日 を展開） ===
pivot = df_team.pivot_table(index="チーム名", columns="節", values=["H/A", "試合日"], aggfunc="first")

# 列名を「H/A_第●節」「試合日_第●節」に変換
pivot.columns = [f"{v}_{k}" for v, k in pivot.columns]
pivot = pivot.reset_index()

# === 節ごとに列を並び替え（H/A → 試合日 の順） ===
cols = [col for col in pivot.columns if col != "チーム名"]
col_pairs = defaultdict(dict)

for col in cols:
    if "_" in col:
        kind, round_ = col.split("_", 1)
        col_pairs[round_][kind] = col

# 節を「第○節」の数値順にソートして列を交互に並べる
ordered_cols = ["チーム名"]
for round_ in sorted(col_pairs.keys(), key=lambda x: int(x.replace("第", "").replace("節", ""))):
    if "H/A" in col_pairs[round_]:
        ordered_cols.append(col_pairs[round_]["H/A"])
    if "試合日" in col_pairs[round_]:
        ordered_cols.append(col_pairs[round_]["試合日"])

# 列順を適用
pivot = pivot[ordered_cols]

# === Excelに出力 ===
pivot.to_excel(output_file, index=False)

print(f"✅ 出力完了: {output_file}")
