"""d-scoop Phase 2: compare raw and uncertainty-adjusted P(D) top-four ranks."""
from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from d_scoop import diagnose

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/reports/d_scoop_phase2_uncertainty'
PHASE1 = ROOT / 'data/reports/d_scoop_phase1'
ROUNDS = ('toto1660', 'toto1663', 'toto1664', 'toto1665', 'toto1668', 'toto1670')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    """Descending probability, ascending match number for an exact tie."""
    order = np.lexsort((numbers, -values))
    result = np.empty(len(order), dtype=int)
    result[order] = np.arange(1, len(order) + 1)
    return result


def read_round(round_id: str) -> tuple[pd.DataFrame, dict]:
    base = ROOT / 'data/eval/toto_rounds' / round_id
    source = base / 'snapshot/next_predictions.csv'
    manifest_path = base / 'snapshot/buyplan_engine_manifest.json'
    actual_path = base / 'actual_results.csv'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('round_id') != round_id:
        raise ValueError(f'{round_id}: manifest round mismatch')
    if manifest['output_sha256']['next_predictions.csv'] != sha(source):
        raise ValueError(f'{round_id}: saved forecast checksum mismatch')
    forecast = pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
    kickoff = pd.to_datetime(forecast.datetime)
    if kickoff.dt.tz is None:
        kickoff = kickoff.dt.tz_localize('Asia/Tokyo')
    if not (kickoff > pd.Timestamp(manifest['created_at'])).all():
        raise ValueError(f'{round_id}: forecast generated after kickoff')
    actual = pd.read_csv(actual_path).sort_values('match_no').reset_index(drop=True)
    if len(actual) != 13 or not actual.result.isin([0, 1, 2]).all():
        raise ValueError(f'{round_id}: 13 finalized results required')
    for col in ('match_no', 'home_team', 'away_team'):
        if forecast[col].tolist() != actual[col].tolist():
            raise ValueError(f'{round_id}: forecast/result match identity mismatch')
    diagnostic, _ = diagnose(forecast)
    phase1_path = PHASE1 / round_id / 'diagnostics.csv'
    phase1 = pd.read_csv(phase1_path).sort_values('match_no').reset_index(drop=True)
    for col in ('match_no', 'home_team', 'away_team'):
        if diagnostic[col].tolist() != phase1[col].tolist():
            raise ValueError(f'{round_id}: Phase 1 identity mismatch')
    for col in ('score_P_D', 'P_D', 'uncertainty_mix'):
        np.testing.assert_allclose(diagnostic[col], phase1[col], rtol=0, atol=1e-12)
    diagnostic['round'] = round_id
    diagnostic['actual_symbol'] = actual.result.map({1: 'H', 0: 'D', 2: 'A'}).to_numpy()
    diagnostic['actual_D'] = actual.result.eq(0).to_numpy()
    provenance = dict(round=round_id, forecast_sha256=sha(source), manifest_sha256=sha(manifest_path),
                      results_sha256=sha(actual_path), phase1_diagnostics_sha256=sha(phase1_path),
                      forecast_created_at=manifest['created_at'],
                      prediction_model=manifest.get('prediction_model', 'not recorded'))
    return diagnostic, provenance


def compare_round(round_id: str) -> tuple[pd.DataFrame, dict, list[dict], dict]:
    diagnostic, provenance = read_round(round_id)
    n = diagnostic.match_no.to_numpy(dtype=int)
    pre = diagnostic.score_P_D.to_numpy(dtype=float)
    post = diagnostic.P_D.to_numpy(dtype=float)
    mix = diagnostic.uncertainty_mix.to_numpy(dtype=float)
    np.testing.assert_allclose(post, (1 - mix) * pre + mix / 3, rtol=0, atol=1e-10)
    pre_rank, post_rank = ranks(pre, n), ranks(post, n)
    pre_selected, post_selected = pre_rank <= 4, post_rank <= 4
    result = diagnostic[['round', 'match_no', 'home_team', 'away_team', 'actual_symbol', 'actual_D']].copy()
    result['score_P_D'] = pre
    result['next_prob_draw'] = post
    result['mix'] = mix
    result['P_D_change'] = post - pre
    result['rank_before'] = pre_rank
    result['rank_after'] = post_rank
    result['rank_change_up_positive'] = pre_rank - post_rank
    result['A_selected'] = post_selected
    result['B_selected'] = pre_selected
    result['selection_change'] = np.select(
        [post_selected & ~pre_selected, pre_selected & ~post_selected],
        ['A_only', 'B_only'], default='unchanged')
    result = result.sort_values('match_no').reset_index(drop=True)
    a_only = result.loc[result.selection_change.eq('A_only')]
    b_only = result.loc[result.selection_change.eq('B_only')]
    if len(a_only) != len(b_only):
        raise AssertionError('Selection exchanges are not balanced')
    if not np.any(mix) and not np.array_equal(pre_selected, post_selected):
        raise AssertionError('No-mix control changed selection')
    a_draw = int(result.loc[result.A_selected, 'actual_D'].sum())
    b_draw = int(result.loc[result.B_selected, 'actual_D'].sum())
    newly = int(b_only.actual_D.sum())
    lost = int(a_only.actual_D.sum())
    if b_draw - a_draw != newly - lost:
        raise AssertionError('Capture decomposition failed')
    summary = dict(round=round_id, actual_D=int(result.actual_D.sum()),
                   A_selected=','.join(map(str, result.loc[result.A_selected, 'match_no'].tolist())),
                   B_selected=','.join(map(str, result.loc[result.B_selected, 'match_no'].tolist())),
                   common_selected=int((post_selected & pre_selected).sum()),
                   A_only=','.join(map(str, a_only.match_no.tolist())),
                   B_only=','.join(map(str, b_only.match_no.tolist())),
                   A_captured_D=a_draw, B_captured_D=b_draw, capture_delta_B_minus_A=b_draw - a_draw,
                   replaced_matches=len(a_only), newly_captured_by_B=newly, lost_from_A=lost,
                   mix_min=float(mix.min()), mix_max=float(mix.max()),
                   max_absolute_rank_change=int(np.abs(pre_rank - post_rank).max()),
                   rank_changed_matches=int(np.count_nonzero(pre_rank != post_rank)))
    changes = result.loc[result.selection_change.ne('unchanged')].to_dict('records')
    return result, summary, changes, provenance


def main() -> None:
    protected = {str(p): sha(p) for p in (ROOT / 'data/purchase_reference').glob('*') if p.is_file()}
    phase1_protected = {str(p): sha(p) for p in PHASE1.rglob('*') if p.is_file()}
    rounds, summaries, changes, sources = [], [], [], []
    for round_id in ROUNDS:
        detail, summary, exchange, source = compare_round(round_id)
        rounds.append(detail)
        summaries.append(summary)
        changes.extend(exchange)
        sources.append(source)
    all_rounds = pd.concat(rounds, ignore_index=True)
    if len(all_rounds) != 78:
        raise AssertionError('Expected 78 matches')
    summary = pd.DataFrame(summaries)
    exchange = pd.DataFrame(changes, columns=list(all_rounds.columns))
    with_mix = summary.loc[summary.mix_max.gt(0)]
    totals = dict(rounds=6, matches=78, actual_D=int(summary.actual_D.sum()),
                  A_captured_D=int(summary.A_captured_D.sum()),
                  B_captured_D=int(summary.B_captured_D.sum()),
                  with_mix_rounds=int(len(with_mix)),
                  A_captured_D_with_mix=int(with_mix.A_captured_D.sum()),
                  B_captured_D_with_mix=int(with_mix.B_captured_D.sum()),
                  replaced_matches=int(summary.replaced_matches.sum()),
                  newly_captured_by_B=int(summary.newly_captured_by_B.sum()),
                  lost_from_A=int(summary.lost_from_A.sum()),
                  net_capture_change=int(summary.capture_delta_B_minus_A.sum()),
                  rank_changed_matches=int(summary.rank_changed_matches.sum()),
                  rank_changed_matches_with_mix=int(with_mix.rank_changed_matches.sum()),
                  max_absolute_rank_change=int(summary.max_absolute_rank_change.max()))
    if totals['replaced_matches'] * 2 != len(exchange):
        raise AssertionError('Each replacement must have one A-only and one B-only match')
    if totals['net_capture_change'] != totals['newly_captured_by_B'] - totals['lost_from_A']:
        raise AssertionError('Overall capture reconciliation failed')
    if any(sha(Path(path)) != digest for path, digest in protected.items()):
        raise AssertionError('Operational files changed')
    if any(sha(Path(path)) != digest for path, digest in phase1_protected.items()):
        raise AssertionError('Phase 1 files changed')
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / 'round_comparison.csv', index=False)
    all_rounds.to_csv(OUT / 'rank_comparison_78.csv', index=False)
    exchange.to_csv(OUT / 'selection_exchanges.csv', index=False)
    (OUT / 'audit.json').write_text(json.dumps(dict(
        comparison='A=final next_prob_draw, B=raw score_P_D',
        tie_break='larger P(D) first, then smaller match_no',
        phase1_reconstructed=True, sources=sources, totals=totals,
        score_matrix_after_uncertainty='not defined or reconstructed',
        forecast_changed=False, buyplan_changed=False,
        caveat='Six historical rounds have changing forecast models; comparison is paired and descriptive, not a causal test.'
    ), ensure_ascii=False, indent=2))
    display = summary[['round', 'actual_D', 'A_selected', 'B_selected', 'common_selected',
                       'A_only', 'B_only', 'A_captured_D', 'B_captured_D',
                       'capture_delta_B_minus_A', 'replaced_matches']]
    def table(frame: pd.DataFrame) -> str:
        return '<div class="scroll">' + frame.to_html(index=False, escape=True,
                     float_format=lambda x: f'{x:.6f}') + '</div>'
    html = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>d-scoop Phase 2：補正前後のD選出</title>
<style>body{font:16px/1.8 system-ui;margin:28px;color:#203147}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:6px;border:1px solid #cbd4df}th{background:#edf2f8}</style>
<h1>d-scoop Phase 2：不確実性補正前後のP(D)上位4比較</h1>
<p>Aは最終next_prob_draw、Bは得点分布のscore_P_D。どちらも13試合中4試合を選択。P(D)の完全同値は試合番号の小さい方を優先する。予測・購入券・補正率は変更していない。</p>
<p>最終P(D)＝(1−mix)×補正前P(D)＋mix/3。補正後のスコア行列は作成していない。Phase 1の再現値と照合し、試合前予測の時刻・ハッシュ・対戦情報を検査した。</p>'''
    html += f'''<p>6開催のD捕捉：A <b>{totals['A_captured_D']}</b>件、B <b>{totals['B_captured_D']}</b>件。補正あり3開催：A {totals['A_captured_D_with_mix']}件、B {totals['B_captured_D_with_mix']}件。置換した試合数 {totals['replaced_matches']}件、Bで新たに拾ったD {totals['newly_captured_by_B']}件、Aで拾えてBで失ったD {totals['lost_from_A']}件、純増減 {totals['net_capture_change']:+d}件。</p>'''
    html += '<h2>開催別比較</h2>' + table(display)
    html += '<h2>採用が入れ替わった試合</h2>' + (table(exchange) if len(exchange) else '<p>入れ替わりなし。</p>')
    for round_id, frame in all_rounds.groupby('round', sort=False):
        if not (frame['mix'] > 0).any():
            continue
        html += '<h2>' + escape(round_id) + '：全13試合の順位変動</h2>'
        html += table(frame[['match_no','home_team','away_team','score_P_D','next_prob_draw',
                             'mix','P_D_change','rank_before','rank_after','rank_change_up_positive',
                             'A_selected','B_selected','selection_change','actual_symbol']])
    html += '''<p>順位変化は「補正前順位−補正後順位」で、正なら補正後に上がったことを示す。置換数はAのみ選出の試合数で数え、Bのみ選出を足して二重計上しない。</p>
<p>この比較は保存された6開催の記述的検証。前半3開催はmix=0の対照群であり、開催間で予測モデルも異なる。補正が選出差の原因となる計算上の関係は示せるが、将来のD捕捉を改善する因果的・一般的な効果は判定できない。得点構造による新スコアの作成は今回行っていない。</p>
<p><a href="round_comparison.csv">開催別CSV</a> ／ <a href="rank_comparison_78.csv">78試合の順位CSV</a> ／ <a href="selection_exchanges.csv">入れ替わり詳細CSV</a> ／ <a href="audit.json">監査</a></p></html>'''
    (OUT / 'index.html').write_text(html)
    print(summary.to_string(index=False))
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(exchange[['round','match_no','home_team','away_team','selection_change','score_P_D',
                    'next_prob_draw','mix','rank_before','rank_after','actual_symbol']].to_string(index=False))
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
