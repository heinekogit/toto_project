# round18 D解釈レビュー

対象は `data/purchase_reference/predictions.csv` で `predicted_result = D` の12試合。
ここではモデル本体を変えず、**Dをどう解釈するか** の人間基準ラベルを暫定で付ける。

分類定義:

- `draw_core`
  D本線として厚く持ってよい。`prob_draw` が十分高く、2位との gap もあり、stall/entropy の暴れも小さめ。
- `draw_rescue`
  Dは主軸だが、side を必ず混ぜるべき。
- `draw_watch`
  D最大だが、本線 label としては弱い。監視対象であり、買いの主軸にはしない。
- `false_d`
  D最大でも suppress/cut 寄りで扱う。

## 暫定ラベル

### `draw_core`

- `広島-名古屋`
  - 現在: `chaos_draw_watch / rescue / rank1_draw_rescue`
  - 指標: `pD=0.380`, `pH=0.318`, `pA=0.302`, `balanced`, `draw_core_score=0.818`
  - コメント: D最大だが gap は薄い。厳密には core より rescue 寄り。現状の `rescue` は妥当。

### `draw_rescue`

- `鹿島-FC東京`
  - 現在: `chaos_draw_watch / rescue / rank1_draw_rescue`
  - 指標: `pD=0.381`, `pA=0.332`, `pH=0.287`, `stall_shape`, `draw_tension=0.823`
  - コメント: D最大だが H/A が近い。`D本線 + side救済` が自然。

- `京都-長崎`
  - 現在: `chaos_draw_watch / rescue / rank1_draw_rescue`
  - 指標: `pD=0.417`, `pA=0.308`, `pH=0.275`, `stall_shape`
  - コメント: D最大は明確。ただし side も弱くない。`rescue` 妥当。

- `岡山-C大阪`
  - 現在: `chaos_draw_watch / rescue / rank1_draw_rescue`
  - 指標: `pD=0.422`, `pA=0.313`, `pH=0.265`, `stall_shape`, `draw_tension=0.851`
  - コメント: D最大は明確。`rescue` 妥当。

- `東京V-横浜FM`
  - 現在: `chaos_draw_watch / rescue / rank1_draw_rescue`
  - 指標: `pD=0.393`, `pA=0.308`, `pH=0.299`, `balanced`
  - コメント: D最大だが三者近い。`rescue` 妥当。

- `水戸-川崎F`
  - 現在: `chaos_draw_watch / rescue / rank1_draw_rescue`
  - 指標: `pD=0.416`, `pA=0.318`, `pH=0.265`, `low_tempo_draw`, `draw_tension=0.850`
  - コメント: D最大は明確。以前 core だったが、今の `rescue` の方が自然。

- `清水-G大阪`
  - 現在: `chaos_draw_watch / rescue / rank1_draw_rescue`
  - 指標: `pD=0.426`, `pA=0.318`, `pH=0.256`, `stall_shape`
  - コメント: D本線だが side 救済は必要。`rescue` 妥当。

### `draw_watch`

- `札幌-磐田`
  - 現在: `chaos_draw_watch / cut / rank2_draw_cut`
  - 指標: `pD=0.385`, `pH=0.311`, `pA=0.305`, `balanced`, J2
  - コメント: J2で D最大だが false D の臭いが強い。watch/cut 寄り。現状 `cut` は妥当。

- `徳島-今治`
  - 現在: `draw_trap / cut / rank2_draw_cut`
  - 指標: `pD=0.379`, `pH=0.365`, `pA=0.256`, `balanced`, J2
  - コメント: D最大だが H が近すぎる。watch か false_d に近い。現状 `cut` 寄りで妥当。

### `false_d`

- `柏-千葉`
  - 現在: `chaos_flat / cut / rank2_draw_cut`
  - 指標: `pD=0.369`, `pA=0.318`, `pH=0.314`, `swing_close`, `entropy=0.600`
  - コメント: D最大でも 3択が詰まりすぎ。false D 寄り。現状 `cut` は妥当。

- `藤枝-いわき` (2行)
  - 現在: `j2_draw_trap / cut / rank2_draw_cut`
  - 指標: `pD=0.376`, `pA=0.344`, `pH=0.280`, `stall_draw_bias`
  - コメント: 既に `j2_draw_trap` として切れている。false D 寄り。

- `山形-湘南`
  - 現在: `j2_draw_trap / cut / rank2_draw_cut`
  - 指標: `pD=0.370`, `pA=0.347`, `pH=0.283`, `stall_draw_bias`
  - コメント: false D 寄り。現状 `cut` 妥当。

## ズレの整理

大きなズレは少ない。現状の `tier/type` と人間基準でズレるのは主に次。

- `広島-名古屋`
  - 人間基準では `draw_rescue`
  - 現状も `rescue`
  - ずれなし

- `札幌-磐田`
  - `chaos_draw_watch` だが実感は J2 false D に近い
  - `match_purchase_type` 側の粒度が甘い

- `徳島-今治`
  - `draw_trap` はむしろ自然
  - policy 接続は不要だが観測ラベルとしては残す価値あり

## 暫定結論

- J1 の D多発帯は、今の round18 に限れば `draw_rescue` が主役
- `draw_core` は今節ほぼ無し
- J2 の D は `draw_watch/false_d` が多く、`cut` 寄りでよい
- 次に粒度を上げる本命は
  - `chaos_draw_watch` を J1 `draw_rescue_watch` と J2 `false_draw_watch` に分けること
  - `draw_core` をさらに狭く保つこと
