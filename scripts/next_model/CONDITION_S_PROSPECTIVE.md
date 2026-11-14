# 条件Sの将来検証ログ

条件Sは、同じtoto開催の13試合で保存済み得点行列の `P(0-0)` が中央値以下、かつ `P(1-1)` が中央値以上、と固定する。同率は含む。定義版は `condition_S_p00_le_median_p11_ge_median_v1`。この定義は過去78試合を見て発見されたため、その78試合の成績を独立検証として扱わない。

`run_next_buyplan.py` が予測と配券の保存後、試合前時刻・ハッシュ・得点行列との整合性を既存 `forecast_registry.py` で検証し、`data/forecast_registry/condition_s_forecasts.jsonl` に13行を追記する。記録実行時点も最初の試合開始前でなければ拒否する。失敗はログに表示して `condition_s_failures.jsonl` に開催・段階・理由を追記するが、予測値・配券・購入処理は変更しない。保存するのは未丸めの得点確率、開催内中央値、該当判定、元のfinal H/D/A、保存λ、mix、予測時刻、入力ファイルのハッシュ、スナップショットID、モデル版・コード版・補正設定である。同じスナップショットの再実行では重複しない。

週次バッチでリーグ結果CSV更新後、既存 `results.jsonl` へ試合結果を取り込み、`condition_s_results.jsonl` へ条件Sの試合前記録と結果の対応を追記する。結果の修正は新しいイベントとして残す。研究用の照合失敗は警告と失敗履歴に残し、週次予測の成否に加えない。手動再同期は次のコマンドを使う。

```sh
scripts/.venv/bin/python scripts/next_model/condition_s_registry.py sync-results
scripts/.venv/bin/python scripts/next_model/condition_s_registry.py status
```

購入締切時刻は現在の予測ファイルとtoto節リストに保存されていない。試合開始時刻を代用せず、締切が未登録の開催は `purchase_eligibility=unknown` と表示する。公式情報など出典のある時刻を取得したら、次のように開催別に追記する。時刻はタイムゾーン付きで指定する。矛盾する時刻の再登録はエラーとする。

```sh
scripts/.venv/bin/python scripts/next_model/condition_s_registry.py record-deadline \
  --round totoXXXX --at 'YYYY-MM-DDTHH:MM:SS+09:00' --source '確認した出典'
```

複数の試合前スナップショットが同じ開催にある場合、記録はすべて保持する。状態表示の評価では開催ごとに最初の完全な13試合スナップショットを一組として選び、試合ごとの混合を禁止する。モデル版・コード版・条件S該当有無・購入締切前確認状況を分け、実D件数、元のP(D)の合計、D対非DのBrier Score、H/D/AのLog Lossを表示する。締切不明の数値は購入判断に利用できた予測として確定扱いしない。少数標本での数値は診断値であり、効果の確定には使わない。条件Sの効果がゼロ・弱・強の場合の配券判断は研究用仮説としてのみ扱い、実測の購入改善と区別する。

この接続は条件Sの予測力を証明しない。新しい開催結果が蓄積するまで係数やDオッズを確定しない。条件S以外のDとH/A逆転も検討対象として残す。
