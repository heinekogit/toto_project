# v2.5配券スコア型シャドー（暫定試作）

正式ルートから呼び出さない独立実装。予測確率・固定枚数・候補00を変更しない。

```python
from score_placement_v25_shadow import Config, generate, diagnostics
tickets, audit = generate(probabilities, counts=fixed_counts,
                          config=Config(draw_coefficient=1.2))
rows = diagnostics(probabilities, tickets, audit)
```

`probabilities`、`counts`は13×3のH,D,A順。返す購入券は10×13、記号はH=0/D=1/A=2。
`counts`が渡された場合、その値を使用する。省略時のみv2.4の配分を利用。
候補00は元確率の最大記号（同率はH,D,A順）で、返す10券には含めない。
`draw_coefficients`で13試合別の係数も指定可能。特徴量から係数を算出する処理は未実装。

## 暫定選択

- D係数は1.0/1.1/1.2/1.4を比較。デフォルト1.2は未学習の仮値。
- 交換数目標は広がり4の線形目標から、総交換数と実現可能性を満たす最も近い整数列をDPで生成。
- 初期配置は二部グラフの次数列構成で決定。全係数で同一初期配置。
- 目的関数は `100 E + L + .03 C + .2 U + .01 R`。正規化式はレポートに記載。
- Eは交換数目標誤差と逆転、Lは若い候補を重くした交換損失、CはDの後半集中、Uは重複、Rは3連続窓。
- 優先度は有限の重みで表現。設定を変えると目標誤差を許す場合がある。固定枚数・10券・00は常に厳守。
- 同一試合内の単一交換と、交換数を相殺する2試合分の交換を全探索し、最も改善する操作を最大200回。
- 局所解。完全重複の回避は目的関数の項であり保証ではない。残存を隠さず報告する。
- 配置後の確率順ソートは行わない。

## 実行

```sh
./scripts/.venv/bin/python scripts/next_model/test_score_placement_v25_shadow.py
./scripts/.venv/bin/python scripts/next_model/simulate_v25_score_shadow.py
```

出力は `data/reports/v25_score_shadow/index.html` と開催・係数別の診断。
各設定の `buyplan.csv` は候補01〜10のみ、既存方式と同じH=1/D=0/A=2表記。
候補00を含む診断は `candidates00_10.csv`。

比較時、当時の正式版がv2.4の開催は保存券を使い、それ以外は現行v2.4を同入力から再生成。
さらにseed 0〜19の分布を併記。過去の厳密集計は入力ハッシュ一致時のみ再利用。
候補生成に実結果は渡さない。実結果は生成後の採点だけに使用。

全結果空間での確率は試合間独立を仮定するモデル内評価。試合前保存の不足と5開催という限界があり、独立した実戦改善の実証ではない。

## 通常実行への接続（2026-11-04）

通常の `run_next_buyplan.py` 実行で、試合別D係数版を購入対象外のシャドーとして生成する。
正式v2.4が実際に出した配分枚数を入力に使用し、正式券を変更しない。
正規化基準は `data/models/shadow_v25_match_reference_v1.json` に固定し、自動再学習しない。
出力は `shadow_buyplan_v25.csv/.html` と `shadow_v25_features.csv`、
`shadow_v25_candidates.csv`、`shadow_v25_audit.json`、`shadow_v25_reference.json`。
生成manifestにハッシュを記録し、購入時スナップショット・観測アーカイブにも保存する。
開催別・累積検証ページへv2.5の成績を追加する。
過去の正式出力は再生成しない。スコアの欠損・不整合時は係数を捏造せずエラーとする。

## C95への標準シャドー切替（2026-11-08）

ユーザー指定により、通常実行のv2.5シャドーはC95となった。上記の試合別D係数方式は
`shadow_v25_legacy.py` に保存し、比較用の再現を維持する。正式版v2.4は変更しない。

- 固定枚数は正式v2.4の実出力と同じ。
- 元予測による13的中確率の和を最大化するBを同入力から探索。
- Bの丸め前13的中確率の95%以上を維持し、D発生数カバー率を最大化。
- 同カバー率なら13的中確率を優先。候補00は枚数を消費しない。
- Bは従来比較と同じ24初期値、C95は3固定シード×32初期値。設定と履歴を監査JSONに保存。
- 旧D係数版はB初期配置の再現にのみ使用。C95目的関数にD係数やQは使用しない。
- 低確率券やD10/13枚などが残る限界は維持される。品質改善Q3への切替ではない。
- 出力名は従来の`shadow_buyplan_v25.html/.csv`等を継続。featuresは元確率と固定枚数、referenceはC95探索設定。
- 通常の`run_buyplan_engine.py`→`run_next_buyplan.py`から自動生成される。

保存済み予測と正式券を変更せず、シャドーだけ更新する場合：

```sh
./scripts/.venv/bin/python scripts/next_model/refresh_v25_c95.py
```

旧シャドー6ファイルと旧manifestは出力先の`shadow_archive/`へ退避し、manifestの
v2.5情報・ハッシュだけ更新する。過去の購入スナップショットは変更しない。
