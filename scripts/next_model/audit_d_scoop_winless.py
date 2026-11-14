"""Audit whether archived inputs support a prematch 'failure to win' study."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from compare_d_scoop_uncertainty import ROOT, ROUNDS, read_round

OUT = ROOT / 'data/reports/d_scoop_winless'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    provenance = []
    for round_id in ROUNDS:
        diagnostic, source = read_round(round_id)
        provenance.append(source)
        frame = pd.read_csv(ROOT / 'data/eval/toto_rounds' / round_id /
                            'snapshot/next_predictions.csv').sort_values('match_no')
        for _, row in frame.iterrows():
            match_no = int(row.match_no)
            observed = diagnostic.loc[diagnostic.match_no.eq(match_no)].iloc[0]
            rows.append(dict(round=round_id, match_no=match_no, match_id=row.match_id,
                             kickoff=row.datetime, home_team=row.home_team, away_team=row.away_team,
                             final_P_H=row.next_prob_home, final_P_D=row.next_prob_draw,
                             final_P_A=row.next_prob_away, actual_result=observed.actual_symbol,
                             prematch_team_cumulative_xg_home=row['stats_ゴール期待値_home'],
                             prematch_team_cumulative_xg_away=row['stats_ゴール期待値_away'],
                             prematch_lab_team_xg_rate_home=row.flab_expected_for_xg_home,
                             prematch_lab_team_xg_rate_away=row.flab_expected_for_xg_away,
                             historical_match_xg_home=float('nan'),
                             historical_match_xg_away=float('nan'),
                             winless_5_home=float('nan'), winless_5_away=float('nan'),
                             winless_10_home=float('nan'), winless_10_away=float('nan'),
                             status='試合別xG履歴なし・指標未計算'))
    detail = pd.DataFrame(rows)
    if len(detail) != 78 or detail.actual_result.eq('D').sum() != 18:
        raise AssertionError('Unexpected archived study population')
    detail.to_csv(OUT / 'prematch_indicator_availability.csv', index=False, encoding='utf-8-sig')
    coverage = []
    for league in ('j1', 'j2'):
        path = ROOT / 'data' / f'{league}_2026_latest_results.csv'
        results = pd.read_csv(path)
        coverage.append(dict(source=str(path.relative_to(ROOT)), league=league,
                             rows=len(results), completed_score_rows=int(results.home_score.notna().sum()),
                             match_xg_rows=0, match_xga_rows=0,
                             note='結果CSVは得失点を保存。試合別xG列なし'))
    coverage.extend([
        dict(source='data/eval/toto_rounds/*/snapshot/next_predictions.csv', league='J1/J2',
             rows=78, completed_score_rows=78, match_xg_rows=0, match_xga_rows=0,
             note='6開催の試合前予測。xG列はチーム累積・平均値であり過去試合別値ではない'),
        dict(source='data/external_metrics/football_lab_team_metrics_*.csv', league='J1/J2',
             rows=None, completed_score_rows=None, match_xg_rows=0, match_xga_rows=0,
             note='チーム集計スナップショット。試合別xGの履歴ではない')])
    pd.DataFrame(coverage).to_csv(OUT / 'data_coverage.csv', index=False, encoding='utf-8-sig')
    audit = dict(rounds=list(ROUNDS), forecast_matches=78, actual_draws=18,
                 completed_j1=coverage[0]['completed_score_rows'],
                 completed_j2=coverage[1]['completed_score_rows'],
                 historical_match_xg_coverage=0, eligible_indicator_rows=0,
                 shadow_training_performed=False, official_model_changed=False,
                 buyplan_changed=False, provenance=provenance)
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    report = '''# d-scoop「勝ちきれない傾向」データ監査

## 結論

既存データだけでは、依頼された試合別xG・xGAに基づく「勝ちきれない」指標を、試合前時点で再現できない。したがって次戦Dとの関係や既存P(D)への追加予測力は未検証。係数、シャドー予測、正式ロジックは作成・変更していない。

## 確認した範囲

- 6開催78試合の試合前Next Model予測と確定結果は照合できる。実Dは18試合。予測は保存時刻とハッシュを既存Phase 1診断処理で検証した。
- 2026年J1の結果CSVには確定スコア140試合、J2には160試合がある。
- 保存予測の `stats_ゴール期待値_*` はチーム累積xG、`stats_被ゴール期待値_*` はチーム累積xGA、`flab_expected_for_xg_*` と `flab_expected_against_xg_*` はチームの試合平均レート。対象試合の前に存在した値だが、個々の過去試合のxGではない。
- 現行Next Modelは得失点率とxG/xGA率を組み合わせて基礎λを計算し、Football LABのチームxG/xGA率による対戦補正を行う。試合別xG履歴から「勝ちきれない」残差を学習する処理はない。

## 指標の定義と欠損理由

各チームの対象試合より前の直近5・10試合について、(A)実得点−試合別xG、(B)実失点−試合別xGA、(C)試合別xG/xGAからの期待勝点−実勝点、(D)xG優勢試合に限った勝/D/敗の頻度を計算する想定。対象試合自身とそれ以後を履歴に含めない。5・10試合未満は欠損とし、全指標について生値と既存の得失点外れ値調整後の値を別記する設計となる。

しかし、現存の結果CSVには試合別xG・xGA列がなく、Football LABファイルもチーム集計のスナップショットである。累積値の差分を試合別xGとみなすには、連続したスナップショット間に対象チームの試合が1つだけであること、時刻・対象試合・指標定義が一致することの確認が必要。現段階でその保証はないため、差分や予測λを代用品として埋めていない。

## 追加予測力の検証状態

有効な試合前指標が0行のため、D/非D別の比較、P(D)近接試合の比較、敗戦との切り分け、時系列学習・後続評価、D Brier・H/D/A Log Lossは計算しない。欠損をゼロとみなした数値は意味を持たない。`prematch_indicator_availability.csv` に78試合の予測・結果と指標欠損を明示した。

## 次のデータ条件

試合ID、キックオフ、両チーム、実得失点、ホーム・アウェイ双方の**その試合の**xG・xGAを持つ履歴が必要。xGAは相手のxGと定義が一致することを確認する。試合前に利用可能になった日時も記録する。揃えば先に各チームの過去だけで5・10試合指標を構築し、時間順に分けて既存P(D)への追加予測力を検証する。
'''
    (OUT / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('forecast_matches', 'actual_draws', 'historical_match_xg_coverage', 'eligible_indicator_rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
