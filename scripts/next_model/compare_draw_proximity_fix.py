"""Compare old and corrected structural scores using identical saved inputs."""
import json
from pathlib import Path

import pandas as pd

from common import ROOT
from simulate_free_structural_draw import run_round, sha, table, page

OUT = ROOT / "data/reports/draw_proximity_fix"
MODES = ("legacy", "near_only", "corrected")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [p for p in (ROOT / "data/purchase_reference").glob('*') if p.is_file()]
    protected += [ROOT / 'data/eval/toto_rounds/toto1664/snapshot/next_predictions.csv',
                  ROOT / 'data/eval/toto_rounds/toto1664/actual_results.csv']
    hashes = {str(p): sha(p) for p in protected}
    rows = []
    for mode in MODES:
        for rid in ('toto1670', 'toto1664'):
            rows += run_round(rid, OUT / mode / rid, mode)
            print('[OK]', mode, rid, flush=True)
    for p, digest in hashes.items():
        if sha(p) != digest:
            raise AssertionError('Source changed during comparison: ' + p)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    main_rows = metrics[metrics.method.str.startswith('structure_')]
    display = main_rows[['round', 'structure_mode', 'strength', 'D_total', 'D_counts', 'P_any13',
                         'actual_D_hit_slots', 'actual_best', 'duplicate_pairs']].copy()
    display['P_any13'] = display.P_any13.map(lambda v: f'{v*100:.6f}%')
    for mode in MODES:
        body = '<p><a href="../index.html">旧式・修正式の比較へ</a></p><ul>'
        for rid in ('toto1670', 'toto1664'):
            body += f'<li><a href="{rid}/index.html">{rid}</a></li>'
        (OUT / mode / 'index.html').write_text(page('D構造評価 / ' + mode, body + '</ul>'))
    changes = []
    for rid in ('toto1670', 'toto1664'):
        for strength in (.5, 1., 1.5, 2.):
            old = pd.read_csv(OUT / 'legacy' / rid / f'structure_{strength:.1f}/allocation.csv')
            for mode in ('near_only', 'corrected'):
                new = pd.read_csv(OUT / mode / rid / f'structure_{strength:.1f}/allocation.csv')
                for i, row in new.iterrows():
                    old_d = [j for j in range(1, 11) if old.iloc[i][f'candidate{j:02d}'] == 'D']
                    new_d = [j for j in range(1, 11) if row[f'candidate{j:02d}'] == 'D']
                    changes.append({'round': rid, 'mode': mode, 'strength': strength,
                                    'match_no': int(row.match_no), 'home_team': row.home_team,
                                    'away_team': row.away_team, 'old_D_count': len(old_d),
                                    'new_D_count': len(new_d), 'old_D_candidates': str(old_d),
                                    'new_D_candidates': str(new_d)})
    pd.DataFrame(changes).to_csv(OUT / 'D_changes.csv', index=False)
    config = {'status': 'non-purchase experiment', 'strengths': [0,.5,1,1.5,2],
              'legacy': 'Q=L*B_onegoal', 'near_only': 'Q=L*N_favourite',
              'corrected': 'Q=L*(B_full+(1-B_full)*N_favourite)',
              'N_favourite': 'one-goal wins / all wins for the favoured H/A side',
              'B_full': '2*min(P_H_score,P_A_score)/(P_H_score+P_A_score)',
              'L': 'P(total goals <= 2)',
              'combination': 'bounded OR of ratings, provisional, not a probability union',
              'selection': 'unchanged exact top ten placement-score products',
              'in_play_transition_estimated': False,
              'forecast_probabilities_changed': False, 'counts_fixed': False,
              'results_used_for_selection': False, 'protected_sha256': hashes}
    (OUT / 'audit.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    body = '''<p>購入対象外。以前の「左右の1点差勝ちの釣り合い」は、片側優勢でも1点差ならDに近いという意図を表していませんでした。</p>
<p>優勢側の勝ちが1点差に集中する割合Nを独立した条件として追加。拮抗Bは得点分布全体で測定。予測確率は変更せず、Dへの交換損失に使う配券評価だけを変えました。</p>
<p>legacy: Q=L×B旧。near_only: Q=L×N。corrected: Q=L×[B全体+(1−B全体)×N]。correctedでは片側優勢によってNの寄与を消しません。Lは総得点2点以下の確率。0–0を含む低得点構造です。</p>
<p>これは得点分布上の近さであり、時間経過や残り時間を使った「1点差から同点になる確率」ではありません。結合式とαは暫定。元の確率・候補生成条件・探索条件・参照券は同一です。</p>'''
    body += table(display)
    body += '<p><a href="D_changes.csv">各試合のD採用候補の差分</a> ／ <a href="metrics.csv">集計CSV</a> ／ <a href="audit.json">監査</a></p><ul>'
    for mode in MODES:
        for rid in ('toto1670', 'toto1664'):
            body += f'<li><a href="{mode}/{rid}/index.html">{rid} / {mode}：全配置と試合別の評価</a></li>'
    body += '</ul><p>near_onlyは低得点条件に優勢側の1点差割合だけを掛けた対照です。各式の尺度を揃える再正規化は行っていないため、同じαでも平均の優遇量は同じではありません。試合別Q値を併記しています。</p>'
    body += '<p>旧Qの検証結果を、新しいQの有効性の証拠として流用しません。標準設定への反映は行っていません。</p>'
    q_report = ROOT / 'data/reports/structural_draw_q_audit_corrected/index.html'
    if q_report.exists():
        body += '<p><a href="../structural_draw_q_audit_corrected/index.html">修正Qの65試合検証・後続26試合検査</a></p>'
    (OUT / 'index.html').write_text(page('1点差のD評価：誤りの修正と再シミュレーション', body))
    # Keep historical results intact, but prevent the obsolete definition from
    # being mistaken for the current interpretation of one-goal proximity.
    for name in ('free_structural_draw', 'structural_draw_q_audit'):
        report = ROOT / 'data/reports' / name / 'index.html'
        if report.exists():
            html = report.read_text()
            marker = '<!-- proximity-definition-correction -->'
            if marker not in html:
                pos = html.find('</h1>') + len('</h1>')
                notice = marker + '<p><b>旧定義の参考記録：</b>1点差のD評価を「左右の釣り合い」に置き換えていた解釈を修正しました。<a href="../draw_proximity_fix/index.html">修正後の比較結果はこちら</a>。以下の数値は旧式の記録です。</p>'
                report.write_text(html[:pos] + notice + html[pos:])
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
