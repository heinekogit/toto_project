# round18 manual override note

- date: 2026-05-23
- scope: J1+J2 第18節 buyplan manual review

## trigger

- `data/reports/html/predictions_round_all_2026_第１８節.html` review
- J1 predicted distribution was heavily draw-biased: `A1 / D9 / H0`

## manual interpretation

- `D core`
  - 町田-浦和
- `D rescue`
  - 広島-名古屋
  - 京都-長崎
  - 岡山-C大阪
  - 水戸-川崎F
  - 東京V-横浜FM
- `D cut`
  - 鹿島-FC東京
  - 柏-千葉
  - 清水-G大阪

## rationale

- avoid J1 draw saturation in purchase portfolio
- do not treat all J1 draw picks as equal-quality draw spots
- preserve only one clear draw core and spread rescue exposure

## operational caution

- `data/purchase_reference/predictions.csv` was still the previous round snapshot at review time
- file rows/date range:
  - rows: `14`
  - range: `2026-05-16 14:00:00` to `2026-05-17 19:00:00`
- therefore `buyplan.py --in data/purchase_reference/predictions.csv` would not represent Round 18 unless the input CSV is refreshed first

## round18 schedule reviewed

- J1
  - 町田-浦和
  - 広島-名古屋
  - 福岡-神戸
  - 鹿島-FC東京
  - 柏-千葉
  - 京都-長崎
  - 岡山-C大阪
  - 水戸-川崎F
  - 東京V-横浜FM
  - 清水-G大阪
- J2
  - 仙台-横浜FC
  - 札幌-磐田
  - 藤枝-いわき
  - 秋田-栃木C
  - 山形-湘南
  - 徳島-今治
