#!/usr/bin/env python3
"""Separate pre-match condition evidence, result surprise and portfolio omissions.

Reads frozen observation payloads only. Never substitutes current weather or a
recomputed fatigue score for what was known at purchase time.
"""
from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def flag(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).lower().strip()
    return True if text in {"true", "1", "1.0"} else False if text in {"false", "0", "0.0"} else None


def analyze_row(row):
    pred = json.loads(row['prediction_payload_json'])
    buy = json.loads(row['buyplan_payload_json'])
    probabilities = [number(buy.get(k)) for k in ['p_draw', 'p_home', 'p_away']]
    if any(p is None or not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities) - 1) > .001:
        raise ValueError(f"invalid purchase probabilities: {row['round_id']} / {row['match_no']}")
    result = int(row['actual_result'])
    votes = [sum(number(buy.get(f'ticket{i:02}')) == symbol for i in range(1, 11)) for symbol in range(3)]
    if sum(votes) != 10:
        raise ValueError('expected ten valid single-pick tickets')
    top = max(probabilities)
    ranked = sorted(probabilities, reverse=True)
    result_rank = 1 + sum(p > probabilities[result] + 1e-9 for p in probabilities)
    rest = [number(pred.get(f'{side}_rest_hours')) for side in ['home', 'away']]
    rest_proxy = [number(pred.get(f'{side}_rest_fatigue')) for side in ['home', 'away']]
    if all(v is not None for v in rest):
        short_rest = any(v <= 96 for v in rest)
    else:
        short_rest = None
    weather_missing = flag(pred.get('weather_missing'))
    weather = [flag(pred.get(k)) for k in ['is_heavy_rain', 'is_strong_wind', 'adverse_weather_during_match']]
    adverse = None if weather_missing is not False else (True if True in weather else False if all(v is False for v in weather[:2]) else None)
    fetched = pred.get('last_updated_at')
    quality = []
    if pred.get('travel_lookup_status') != 'hit':
        quality.append('travel_unknown')
    if any(v is None for v in rest):
        quality.append('rest_hours_unknown')
    if pred.get('external_schedule_coverage') != 'complete':
        quality.append('external_schedule_incomplete')
    if not fetched:
        quality.append('weather_acquisition_unknown')
    if weather_missing is not False:
        quality.append('weather_missing')
    pre_flags = []
    if short_rest:
        pre_flags.append('short_rest_96h')
    if adverse:
        pre_flags.append('adverse_weather_forecast')
    if ranked[0] - ranked[1] <= .05:
        pre_flags.append('close_top2')
    if any(number(pred.get(f'{side}_external_matches_last14d')) not in (None, 0) for side in ['home', 'away']):
        pre_flags.append('recent_external_match')
    omitted = sum(p for p, count in zip(probabilities, votes) if count == 0)
    return {
        **{k: row[k] for k in ['round_id', 'match_no', 'match_id', 'datetime', 'league', 'home_team', 'away_team']},
        'actual_result': result, 'actual_probability': probabilities[result], 'actual_probability_rank': result_rank,
        'top_probability': top, 'argmax_hit': probabilities[result] == top,
        'low_probability_result_25pct': probabilities[result] <= .25,
        'favorite_defeat_50pct': top >= .5 and probabilities[result] < top,
        'log_loss': -math.log(max(probabilities[result], 1e-15)),
        'brier_sum': sum((p - int(i == result)) ** 2 for i, p in enumerate(probabilities)),
        'portfolio_hits': votes[result], 'portfolio_uncovered': votes[result] == 0,
        'omitted_probability_mass': omitted,
        'omitted_option_20pct': any(p >= .2 and count == 0 for p, count in zip(probabilities, votes)),
        'nonstrong_unanimous': max(votes) == 10 and top < .6,
        'home_votes': votes[1], 'draw_votes': votes[0], 'away_votes': votes[2],
        'home_rest_hours': rest[0], 'away_rest_hours': rest[1],
        'legacy_rest_penalty_present': any(v is not None and v > 0 for v in rest_proxy),
        'short_rest_96h': short_rest, 'adverse_weather_forecast': adverse,
        'weather_fetched_at': fetched, 'weather_data_kind': pred.get('weather_data_kind', 'unknown'),
        'travel_lookup_status': pred.get('travel_lookup_status'),
        'pre_match_flags': '|'.join(pre_flags), 'measurement_issues': '|'.join(quality),
    }


def build(history_dir, output_dir):
    history = pd.read_csv(history_dir / 'matches.csv', encoding='utf-8-sig')
    rows = pd.DataFrame([analyze_row(row) for _, row in history.iterrows()])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / 'condition_feedback_matches.csv', index=False, encoding='utf-8-sig')
    # Counts below are unique fixtures, not ten correlated candidate outcomes.
    unique = rows.sort_values(['datetime', 'round_id']).drop_duplicates(['league', 'datetime', 'home_team', 'away_team'])
    summary = []
    for col in ['short_rest_96h', 'adverse_weather_forecast', 'omitted_option_20pct', 'nonstrong_unanimous']:
        for value, label in [(True, 'yes'), (False, 'no'), (None, 'unknown')]:
            subset = unique[unique[col].isna() if value is None else unique[col].eq(value)]
            summary.append({'condition': col, 'group': label, 'matches': len(subset),
                            'argmax_hit_rate': subset.argmax_hit.mean(),
                            'uncovered_rate': subset.portfolio_uncovered.mean(),
                            'mean_brier_sum': subset.brier_sum.mean(),
                            'mean_log_loss': subset.log_loss.mean()})
    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / 'condition_feedback_summary.csv', index=False, encoding='utf-8-sig')
    tables = []
    for round_id, subset in rows.groupby('round_id', sort=True):
        display = subset[['match_no', 'home_team', 'away_team', 'actual_probability', 'actual_probability_rank',
                          'portfolio_hits', 'pre_match_flags', 'measurement_issues']].copy()
        display['actual_probability'] = display.actual_probability.map(lambda p: f'{p:.1%}')
        display.columns = ['No.', 'ホーム', 'アウェイ', '結果の事前確率', '確率順位', '的中候補数', '事前条件', '計測上の未確認']
        tables.append(f'<h2>{escape(round_id)}</h2>' + display.to_html(index=False, escape=True))
    html = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>条件・波乱フィードバック</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;color:#172033}table{border-collapse:collapse;font-size:12px;width:100%}th,td{border:1px solid #ccd5df;padding:7px;text-align:left}th{background:#edf2f7}.note{padding:16px;background:#f3f6fa;line-height:1.7}h2{margin-top:30px}</style>
<h1>疲労・天候・波乱の振り返り</h1><div class="note">
購入時に保存した値だけで集計。事前条件は結果を参照せず抽出し、結果の意外性と購入漏れを別評価します。<br>
低確率結果＝事前25%以下。本命敗退＝最大確率50%以上の選択肢が外れ。確率3位でも約30%なら強い波乱とは呼びません。<br>
短間隔＝96時間以下。雨風＝大雨・強風の予測フラグ。実測や敗因の証明ではありません。<br>
unknownは通常条件に含めません。旧版の疲労点から休養時間は逆算しません。<br>
確率20%以上の未採用・60%未満の全候補固定は検証用の暫定監査基準です。<br>
改善案は次回の試合前に固定し、同じ10候補数で比較。救済と新規取りこぼし、最高的中、Brier/log lossを追跡し、複数回の結果で採否を決めます。条件別集計は記述統計で、因果効果ではありません。
</div><h2>条件別集計（重複試合を除外）</h2>'''
    html += summary.to_html(index=False, escape=True, float_format=lambda x: f'{x:.3f}') + ''.join(tables) + '</html>'
    path = output_dir / 'condition_feedback.html'
    path.write_text(html, encoding='utf-8')
    return rows, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-dir', type=Path, default=ROOT / 'data/eval/observation_history')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    _, path = build(args.history_dir, args.output_dir or args.history_dir / 'reports/conditions')
    print(f'OK: {path}')


if __name__ == '__main__':
    main()
