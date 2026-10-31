# toto事後検証

通常の結果取得・採点・検証:

```bash
SEASON_YEAR=2027 ROUND=toto1660 ./scripts/run_evaluate_round.sh
```

結果取得済みで、検証HTMLだけ更新する場合（通信・予測再生成なし）:

```bash
ROUND=toto1660 ./scripts/run_validate_round.sh
```

表示:

- 累積一覧: data/eval/growth_validation/index.html
- 開催別: data/eval/toto_rounds/toto1660/growth_validation.html

背景データ:

- records/totoNNNN.json: 入力ハッシュ、保存時の版、確率、券、実結果、正式・シャドーの採点
- matches.csv: 開催＋試合番号で一意の確率・券・実結果
- rounds.csv: 開催ごとの正式成績

同じ開催の再実行はその開催を更新する。元の購入snapshot・予測・実結果・BuyPlan生成方式は変更しない。途中結果は確定試合だけを採点。版不明の旧開催は参考履歴として掲載し、版別累積評価には混ぜない。

過去の保存済み開催を再構築:

```bash
./scripts/.venv/bin/python scripts/eval/20_build_growth_report.py --round toto1660 --rebuild-all
```
