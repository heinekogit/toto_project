"""Phase 1/2 score-grid structure analysis; no forecast or ticket changes."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from compare_d_scoop_uncertainty import ROOT, ROUNDS, sha
from d_scoop import diagnose

OUT = ROOT / 'data/reports/score_structure_phase12'


def score_grid(home_lambda: float, away_lambda: float) -> tuple[np.ndarray, float]:
    """Same 0..10 Poisson product and truncation normalization as d_scoop.diagnose."""
    g = np.arange(11)
    raw = np.outer(poisson.pmf(g, home_lambda), poisson.pmf(g, away_lambda))
    retained = float(raw.sum())
    return raw / retained, retained


def structure(matrix: np.ndarray) -> dict:
    if matrix.shape != (11, 11) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('Invalid 11x11 score matrix')
    if not np.isclose(matrix.sum(), 1, atol=1e-12):
        raise ValueError('Score matrix must sum to one')
    i, j = np.indices(matrix.shape)
    diff = i - j
    diag = np.diag(matrix)
    result = {f'P_{k}_{k}': float(diag[k]) for k in range(4)}
    result.update(P_draw_4plus=float(diag[4:].sum()), P_draw=float(diag.sum()),
                  P_home_1=float(matrix[diff == 1].sum()),
                  P_away_1=float(matrix[diff == -1].sum()),
                  P_home_2=float(matrix[diff == 2].sum()),
                  P_away_2=float(matrix[diff == -2].sum()),
                  P_home_3plus=float(matrix[diff >= 3].sum()),
                  P_away_3plus=float(matrix[diff <= -3].sum()))
    result['P_margin_1'] = result['P_home_1'] + result['P_away_1']
    result['P_margin_2plus'] = sum(result[k] for k in
                                   ('P_home_2', 'P_away_2', 'P_home_3plus', 'P_away_3plus'))
    result['P_draw_2plus'] = float(diag[2:].sum())
    result['P_1or0_share_draw'] = (result['P_0_0'] + result['P_1_1']) / result['P_draw']
    neighbors=np.unique(np.concatenate([np.minimum(i[abs(diff)==1],j[abs(diff)==1]),
                                        np.maximum(i[abs(diff)==1],j[abs(diff)==1])]))
    result['P_draw_neighbor_union'] = float(diag[neighbors].sum())
    # All 1-goal victory cells are adjacent to a diagonal cell; the union is
    # counted once, while edge probabilities remain separately available.
    assert np.isclose(result['P_draw_neighbor_union'], result['P_draw'])
    assert np.isclose(result['P_draw'] + result['P_margin_1'] + result['P_margin_2plus'], 1)
    return result


def adjacency(matrix: np.ndarray, round_id: str, match_no: int) -> pd.DataFrame:
    rows = []
    for h in range(11):
        for a in range(11):
            if abs(h-a) != 1:
                continue
            lower, upper = min(h,a), max(h,a)
            rows.append(dict(round=round_id, match_no=match_no, home_goals=h, away_goals=a,
                             side='H' if h>a else 'A', victory_probability=float(matrix[h,a]),
                             adjacent_lower_draw=f'{lower}-{lower}',
                             lower_draw_probability=float(matrix[lower,lower]),
                             adjacent_upper_draw=f'{upper}-{upper}',
                             upper_draw_probability=float(matrix[upper,upper]),
                             neighbor_draw_sum_for_this_edge=float(matrix[lower,lower]+matrix[upper,upper]),
                             minimum_goal_changes_to_draw=1))
    return pd.DataFrame(rows)


def distance_rows(matrix: np.ndarray, round_id: str, match_no: int) -> pd.DataFrame:
    i,j = np.indices(matrix.shape)
    return pd.DataFrame([dict(round=round_id,match_no=match_no,margin=k,side='D' if k==0 else 'H' if k>0 else 'A',
                              minimum_goal_changes_to_draw=abs(k),
                              probability=float(matrix[(i-j)==k].sum())) for k in range(-10,11)])


def svg_heatmap(matrix: np.ndarray, title: str) -> str:
    cells=[];step=35;left=75;top=68
    for h in range(11):
        for a in range(11):
            p=float(matrix[h,a]);intensity=min(1,p/.18)
            if h==a: fill=f'rgb({int(255-130*intensity)},{int(245-100*intensity)},{int(205-40*intensity)})'
            elif abs(h-a)==1: fill=f'rgb({int(250-85*intensity)},{int(250-75*intensity)},{int(255-20*intensity)})'
            else: fill=f'rgb({int(255-65*intensity)},{int(255-60*intensity)},{int(255-30*intensity)})'
            cells.append(f'<rect x="{left+a*step}" y="{top+h*step}" width="{step}" height="{step}" fill="{fill}" stroke="#ccd4df"/>')
            if p>=.005: cells.append(f'<text x="{left+a*step+step/2}" y="{top+h*step+22}" text-anchor="middle" font-size="9">{p*100:.1f}</text>')
    ticks=''.join(f'<text x="{left+k*step+step/2}" y="{top-8}" text-anchor="middle" font-size="12">{k}</text><text x="{left-12}" y="{top+k*step+22}" text-anchor="end" font-size="12">{k}</text>' for k in range(11))
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="490" height="505" viewBox="0 0 490 505"><rect width="490" height="505" fill="white"/><text x="18" y="24" font-size="16">{escape(title)}</text><text x="220" y="48" font-size="12">アウェイ得点 →</text><text x="5" y="270" font-size="12" transform="rotate(-90 5 270)">ホーム得点 →</text>{"".join(cells)}{ticks}<text x="75" y="478" font-size="11">緑：同点　青：1点差　灰：2点差以上　数値：%</text></svg>'


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    details=[];edges=[];distances=[];cells=[];sources=[];matrices={}
    for rid in ROUNDS:
        base=ROOT/'data/eval/toto_rounds'/rid
        source=base/'snapshot/next_predictions.csv'
        manifest=base/'snapshot/buyplan_engine_manifest.json'
        m=json.loads(manifest.read_text())
        if m['output_sha256']['next_predictions.csv']!=sha(source):raise ValueError('Forecast checksum mismatch')
        f=pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
        kickoff=pd.to_datetime(f.datetime).dt.tz_localize('Asia/Tokyo')
        if not (kickoff>pd.Timestamp(m['created_at'])).all():raise ValueError('Non-prematch archive')
        diagnostic, grids=diagnose(f)
        for ix,(row,matrix) in enumerate(zip(f.itertuples(index=False),grids)):
            n=int(row.match_no); s=structure(matrix); key=(rid,n); matrices[key]=matrix
            record=dict(round=rid,match_no=n,home_team=row.home_team,away_team=row.away_team,
                        lambda_home=float(row.final_lambda_home),lambda_away=float(row.final_lambda_away),
                        lambda_total=float(row.final_lambda_home+row.final_lambda_away),
                        lambda_abs_diff=float(abs(row.final_lambda_home-row.final_lambda_away)),
                        final_P_D=float(row.next_prob_draw),raw_P_H=float(diagnostic.iloc[ix].score_P_H),
                        raw_P_D=float(diagnostic.iloc[ix].score_P_D),raw_P_A=float(diagnostic.iloc[ix].score_P_A),
                        uncertainty_mix=float(diagnostic.iloc[ix].uncertainty_mix),
                        omitted_unconditioned_mass=float(diagnostic.iloc[ix].omitted_tail_mass),**s)
            np.testing.assert_allclose([s['P_draw'],s['P_home_1']+s['P_home_2']+s['P_home_3plus'],
                                         s['P_away_1']+s['P_away_2']+s['P_away_3plus']],
                                        [diagnostic.iloc[ix].score_P_D,diagnostic.iloc[ix].score_P_H,
                                         diagnostic.iloc[ix].score_P_A],atol=1e-12,rtol=0)
            details.append(record);edges.append(adjacency(matrix,rid,n));distances.append(distance_rows(matrix,rid,n))
            cells.extend(dict(round=rid,match_no=n,home_goals=h,away_goals=a,
                              probability=float(matrix[h,a]),margin=h-a,
                              region='D' if h==a else 'one_goal' if abs(h-a)==1 else 'two_plus')
                         for h in range(11) for a in range(11))
        sources.append(dict(round=rid,forecast_sha256=sha(source),manifest_sha256=sha(manifest),
                            created_at=m['created_at'],matches=len(f)))
    detail=pd.DataFrame(details);edge=pd.concat(edges,ignore_index=True);distance=pd.concat(distances,ignore_index=True)
    if len(detail)!=78 or len(edge)!=78*20:raise AssertionError('Study population or adjacency count')
    detail.to_csv(OUT/'match_structure_78.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cells).to_csv(OUT/'score_matrix_cells_78.csv',index=False,encoding='utf-8-sig')
    edge.to_csv(OUT/'one_goal_adjacency.csv',index=False,encoding='utf-8-sig')
    distance.to_csv(OUT/'goal_difference_mass.csv',index=False,encoding='utf-8-sig')
    cases=[(.8,.8),(1.5,1.5),(2.5,2.5),(2.,.8)]
    theory=pd.DataFrame([dict(lambda_home=h,lambda_away=a,lambda_total=h+a,lambda_abs_diff=abs(h-a),**structure(score_grid(h,a)[0])) for h,a in cases])
    theory.to_csv(OUT/'theoretical_cases.csv',index=False,encoding='utf-8-sig')
    picks=[]
    for label,col in [('0-0型','P_0_0'),('1点差型','P_margin_1'),('2点差以上型','P_margin_2plus'),
                      ('高得点型','lambda_total'),('片側優勢型','lambda_abs_diff')]:
        chosen=detail.sort_values(col,ascending=False).iloc[0].copy();chosen['case_type']=label;picks.append(chosen)
    # Compare raw score distributions on the same scale, before H/D/A mixing.
    pairs=[]
    for i in range(len(detail)):
        for j in range(i+1,len(detail)):
            x,y=detail.iloc[i],detail.iloc[j]
            delta=abs(x.raw_P_D-y.raw_P_D)
            if delta<=.005:pairs.append((abs(x.P_0_0-y.P_0_0),-delta,i,j))
    if pairs:
        _,_,i,j=max(pairs);a=detail.iloc[i].copy();b=detail.iloc[j].copy()
        a['case_type']='行列P(D)近接・構造差A';b['case_type']='行列P(D)近接・構造差B';picks.extend([a,b])
    representative=pd.DataFrame(picks);representative.to_csv(OUT/'representative_matches.csv',index=False,encoding='utf-8-sig')
    example_key=(str(representative.iloc[0]['round']),int(representative.iloc[0].match_no))
    example_edges=edge.loc[(edge['round']==example_key[0]) &
                           (edge.match_no.eq(example_key[1])) &
                           (((edge.home_goals.eq(1)) & (edge.away_goals.eq(0))) |
                            ((edge.home_goals.eq(0)) & (edge.away_goals.eq(1))) |
                            ((edge.home_goals.eq(2)) & (edge.away_goals.eq(1))) |
                            ((edge.home_goals.eq(1)) & (edge.away_goals.eq(2))))].copy()
    (OUT/'heatmaps').mkdir(exist_ok=True)
    for _,r in representative.iterrows():
        key=(r['round'],int(r.match_no));title=f'{r["round"]} No.{int(r.match_no)} {r.home_team}－{r.away_team}'
        (OUT/'heatmaps'/f'{key[0]}_{key[1]:02}.svg').write_text(svg_heatmap(matrices[key],title))
    audit=dict(rounds=list(ROUNDS),matches=len(detail),score_grid='independent Poisson 0..10; normalize retained grid',
               matrix_tail_max=float(detail.omitted_unconditioned_mass.max()),
               final_minus_raw_PD_max=float(abs(detail.final_P_D-detail.raw_P_D).max()),
               saved_score_error_max=float(max(diagnose(pd.read_csv(ROOT/'data/eval/toto_rounds'/rid/'snapshot/next_predictions.csv'))[0].score_reconstruction_error.max() for rid in ROUNDS)),
               source_hashes=sources,actual_results_used=False,model_changed=False,buyplan_changed=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    def tab(df,cols):return '<div class="scroll">'+df[cols].to_html(index=False,escape=True,float_format=lambda x:f'{x:.4f}')+'</div>'
    cols=['round','match_no','home_team','away_team','lambda_home','lambda_away','final_P_D','raw_P_D','P_0_0','P_1_1','P_draw_2plus','P_margin_1','P_margin_2plus']
    tcols=['lambda_home','lambda_away','P_draw','P_0_0','P_1_1','P_draw_2plus','P_margin_1','P_margin_2plus']
    imgs=''.join(f'<figure><img width="490" src="heatmaps/{r["round"]}_{int(r.match_no):02}.svg"><figcaption>{escape(r.case_type)}</figcaption></figure>' for _,r in representative.iterrows())
    report=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>得点構造 Phase 1・2</title>
<style>body{{font:16px/1.7 system-ui;margin:26px;color:#203044}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px}}td,th{{border:1px solid #ccd4df;padding:5px}}th{{background:#edf3f8}}figure{{display:inline-block;vertical-align:top;margin:8px}}</style>
<h1>新BuyPlan構想：得点構造 Phase 1・2</h1>
<h2>Phase 1　再現条件</h2><p>6開催78試合の試合前保存λから、既存 <code>d_scoop.diagnose</code> の11×11行列を再利用。各セルは独立Poisson(λH)×Poisson(λA)を0～10得点で切り、残存確率で正規化した条件付き質量。切り捨てた無条件質量の最大は{audit['matrix_tail_max']:.8f}。保存0-0/1-1/2-2との最大誤差は{audit['saved_score_error_max']:.3g}。H/D/Aの最終値は疲労・天候等の不確実性mixによる一様混合後で、得点行列そのものではない。Dの最終と行列値の最大差は{audit['final_minus_raw_PD_max']:.4f}。</p>
<h2>Phase 2　勝敗と同点の距離</h2><p>1点差スコアは1得点の増減で低側・高側の同点スコアに隣接する。2点差以上は最少2得点以上の変化が必要。ただし隣接は予測分布上の幾何学的関係で、試合中にDへ移る確率ではない。<a href="one_goal_adjacency.csv">全1,560隣接行</a>は勝敗セルごとに2つの同点セルと確率を示す。同じ同点セルを含む行を足し合わせず、同点質量は対角線から一度だけ集計した。</p>
<h3>具体例：{escape(example_key[0])} No.{example_key[1]} {escape(str(representative.iloc[0].home_team))}－{escape(str(representative.iloc[0].away_team))}</h3>{tab(example_edges,['home_goals','away_goals','side','victory_probability','adjacent_lower_draw','lower_draw_probability','adjacent_upper_draw','upper_draw_probability'])}<p>隣接する同点スコアの確率は勝敗スコアの確率と独立のセル質量。隣接確率を勝敗確率へ足してDの発生率と解釈しない。</p>
<h2>理論λ比較（過去結果による調整なし）</h2>{tab(theory,tcols)}
<h2>代表試合</h2>{tab(representative,cols)}
<p>行列由来P(D)が近い2試合は差0.5ポイント以内から、0-0確率の差が最大のペアを選んだ。最終P(D)は参考表示にとどめる。不確実性mixは開催回ごと・試合ごとに異なり得るため、最終P(D)だけを揃えて元の得点構造を比較しない。得失点の突出試合に対する外れ値調整はλ算出前の別処理であり、今回の1660～1668保存予測にはその調整の監査列がない。これは構造比較であり、結果への適合やDの優劣ではない。</p>
<h2>得点行列ヒートマップ</h2>{imgs}
<h2>次のPhase 3へ引き継ぐ指標</h2><p>同点帯別質量（0-0、1-1、2-2以上）、H/A別1点差質量、H/A別2点差以上質量、同点への最少得点変化数、λ合計・差、1点差セルに隣接する同点セルの個別確率を残す。λ変動時にこれらがどう動くかを測ることで、隣接性を変化確率と混同せずに感応度を検討できる。Phase 3は実装していない。</p>
<p><a href="match_structure_78.csv">78試合CSV</a> ／ <a href="score_matrix_cells_78.csv">全9,438行の得点行列CSV</a> ／ <a href="goal_difference_mass.csv">得点差別CSV</a> ／ <a href="one_goal_adjacency.csv">隣接CSV</a> ／ <a href="theoretical_cases.csv">理論比較CSV</a> ／ <a href="audit.json">監査JSON</a></p></html>'''
    (OUT/'index.html').write_text(report)
    print(json.dumps({k:audit[k] for k in ('matches','matrix_tail_max','final_minus_raw_PD_max','saved_score_error_max')},ensure_ascii=False))
    print(theory[tcols].to_string(index=False))
    print(representative[cols+['case_type']].to_string(index=False))


if __name__=='__main__':main()
