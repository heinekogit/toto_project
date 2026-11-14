"""Audit threshold resolution and boundary margins without tuning a D rule."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/reports/score_structure_phase35/diagnostics_78.csv'
OUT=ROOT/'data/reports/score_structure_phase35_threshold_audit'


def run():
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    d=pd.read_csv(SOURCE)
    assert len(d)==78 and d.groupby('round').size().eq(13).all()
    d['median_P00']=d.groupby('round').P_0_0.transform('median')
    d['median_P11']=d.groupby('round').P_1_1.transform('median')
    d['margin_P00_pp']=100*(d.median_P00-d.P_0_0)
    d['margin_P11_pp']=100*(d.P_1_1-d.median_P11)
    d['rank_P00_ascending']=d.groupby('round').P_0_0.rank(method='average')
    d['rank_P11_ascending']=d.groupby('round').P_1_1.rank(method='average')
    for source,margin in [('P_0_0','margin_P00_pp'),('P_1_1','margin_P11_pp')]:
        iqr=d.groupby('round')[source].transform(lambda x:x.quantile(.75)-x.quantile(.25))
        d[margin.replace('_pp','_IQR')]=d[margin]/100/iqr.replace(0,np.nan)
    d['original_condition']=(d.margin_P00_pp>=0)&(d.margin_P11_pp>0)
    assert int(d.original_condition.sum())==5
    def evaluate(q00,q11):
        t00=d.groupby('round').P_0_0.transform(lambda x:x.quantile(q00))
        t11=d.groupby('round').P_1_1.transform(lambda x:x.quantile(q11))
        selected=(d.P_0_0<=t00)&(d.P_1_1>t11)
        s=d[selected]
        return dict(P00_quantile=q00,P11_quantile=q11,n=len(s),D=int(s.actual_symbol.eq('D').sum()),
                    H=int(s.actual_symbol.eq('H').sum()),A=int(s.actual_symbol.eq('A').sum()),
                    expected_D=float(s.baseline_P_D.sum()),
                    selected_ids=';'.join(f'{r.round}:{r.match_no}' for r in s.itertuples()))
    qdiag=[.4,.425,.45,.475,.499999,.5,.500001,.525,.55,.575,.6]
    diagonal=pd.DataFrame([evaluate(q,1-q) for q in qdiag])
    qgrid=[.4,.425,.45,.475,.5,.525,.55,.575,.6]
    grid=pd.DataFrame([evaluate(x,y) for x in qgrid for y in qgrid])
    tie_rows=[]
    for inc00 in (False,True):
        for inc11 in (False,True):
            a=(d.P_0_0<=d.median_P00) if inc00 else (d.P_0_0<d.median_P00)
            b=(d.P_1_1>=d.median_P11) if inc11 else (d.P_1_1>d.median_P11)
            s=d[a&b]
            tie_rows.append(dict(P00_rule='<=' if inc00 else '<',P11_rule='>=' if inc11 else '>',n=len(s),
                                 D=int(s.actual_symbol.eq('D').sum()),nonD=int(s.actual_symbol.ne('D').sum())))
    ties=pd.DataFrame(tie_rows)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in [('threshold_diagonal.csv',diagonal),('threshold_grid.csv',grid),
                       ('boundary_margins_78.csv',d),('boundary_inclusion_rules.csv',ties)]:
        frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    audit=dict(matches=78,per_round=13,source_sha256=before,thresholds_chosen_without_result_optimization=True,
               new_selection_score_created=False,selected_optimal_threshold=False,
               independent_validation=False,explanation='Existing development sample; resolution audit only')
    assert before==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return d,diagonal,grid,ties


def render(d,diagonal,grid,ties):
    def md(frame):
        cols=list(frame.columns)
        def cell(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
        return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
                         ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
    selected=d[d.original_condition]
    simple=diagonal[['P00_quantile','P11_quantile','n','D','H','A']]
    margins=selected[['round','match_no','home_team','away_team','actual_symbol','margin_P00_pp','margin_P11_pp']]
    report='\n\n'.join([
        '# Phase 3.5：閾値の刻み幅と境界処理の再点検',
        '前回の「40/60・50/50・60/40分位で1・5・19試合」という3点比較だけでは、条件そのものの不安定性を判断できない。13試合の開催内順位を中央値で切った際の離散性と、<=・>の境界処理を調べた。配券と予測は変更していない。',
        '## 分位点を細かく動かした結果',md(simple),
        '13件の線形補間分位点は、40%で5.8位、50%で7位、60%で8.2位に当たる。40→60%は順位境界を複数またぐ変化であり、連続量のわずかな変化と同じに扱えない。45〜49.9999%は同じ4試合、50.0001〜57.5%は同じ8試合で、結果が変わらない区間もある。',
        '## 中央値と等しい試合の扱い',md(ties),
        '元の条件はP(0-0)を中央値以下、P(1-1)を中央値より上とする非対称な定義だった。中央値は13件中の実在する7番目の値なので、等号をどちらに付けるかで試合が増減する。実結果の良い境界処理を選ぶことはしていない。',
        '## 該当5試合の境界からの距離',md(margins),
        'margin_P00_ppは中央値−P(0-0)、margin_P11_ppはP(1-1)−中央値。単位は確率のパーセントポイント。神戸－名古屋はP(0-0)の差がちょうど0で、P(1-1)の差も0.0179ポイント。神戸－町田のP(1-1)差は0.0070ポイントにすぎない。境界のすぐ両側を断絶した別集団として扱う設計は粗い。',
        '## 修正方針',
        '① 40/50/60%の3点だけから頑健性を否定した説明を撤回する。② 閾値の細分化だけで成績の良い点を探さず、各試合の2方向の連続的な距離を保持する。③ 二値の採用・不採用の前に、距離の近い試合同士のH/D/A実績と予測を比較する。④ 条件の滑らかな重み付けを試す場合、尺度や幅は後続の未使用データで検証する。今回、2距離の合算Dスコアや新しい購入条件は作っていない。',
        '4/4、6/8などの探索結果は、工夫を続ける理由になる。一方、この条件を発見した同じ78試合だけでは将来の予測力を確認できない点は、閾値設計の粗さとは別問題として残る。',
    ])+'\n'
    (OUT/'report.md').write_text(report,encoding='utf-8')
    # Two continuous margins, without merging them into a D-selection coefficient.
    xmin,xmax=float(d.margin_P00_pp.min()),float(d.margin_P00_pp.max())
    ymin,ymax=float(d.margin_P11_pp.min()),float(d.margin_P11_pp.max())
    def X(v):return 70+660*(v-xmin)/(xmax-xmin)
    def Y(v):return 440-350*(v-ymin)/(ymax-ymin)
    svg=['<svg viewBox="0 0 820 510" role="img" aria-label="中央値からの距離と実結果">',
         f'<rect x="{X(0):.2f}" y="90" width="{730-X(0):.2f}" height="{Y(0)-90:.2f}" fill="#f2f6f9"/>',
         f'<path d="M70 {Y(0):.2f}H730 M{X(0):.2f} 90V440" stroke="#7b8a99" stroke-dasharray="4 4"/>']
    for r in d.itertuples():
        color='#c64b45' if r.actual_symbol=='D' else '#3d78a0'
        title=f'{r.round} No.{r.match_no} {r.home_team}－{r.away_team}: {r.actual_symbol}; {r.margin_P00_pp:.4f}, {r.margin_P11_pp:.4f} pp'
        svg.append(f'<circle cx="{X(r.margin_P00_pp):.2f}" cy="{Y(r.margin_P11_pp):.2f}" r="5" fill="{color}" fill-opacity=".75"><title>{html.escape(title)}</title></circle>')
    svg+=['<text x="70" y="35" font-size="18">中央値からの連続的な距離（赤：実D、青：非D）</text>',
          '<text x="220" y="490" font-size="14">横：中央値 − P(0-0)（右ほど0-0が低い）</text>',
          '<text x="70" y="65" font-size="14">縦：P(1-1) − 中央値（上ほど1-1が高い）</text>','</svg>']
    (OUT/'boundary_margins.svg').write_text(''.join(svg),encoding='utf-8')
    body='<h1>閾値の刻み幅と境界処理の再点検</h1><p>1・5・19試合の3点だけでは条件の不安定性を判断できません。13件の順位境界と等号の扱いが影響しています。</p>'
    body+=''.join(svg)
    for title,frame in [('細かい分位点での比較',simple),('中央値の等号',ties),('元の5試合の境界距離（pp）',margins)]:
        body+='<h2>'+title+'</h2>'+frame.to_html(index=False,float_format=lambda v:f'{v:.6f}')
    body+='<p>2方向の距離を合算せず保持しました。最良閾値の選択、購入券の変更はしていません。</p><p><a href="report.md">詳細</a> · <a href="boundary_margins_78.csv">78試合の連続距離</a> · <a href="threshold_grid.csv">2軸の独立変更</a></p>'
    (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 3.5 閾値再点検</title><style>body{font:15px/1.6 system-ui;max-width:1100px;margin:28px auto;color:#20364a}table{border-collapse:collapse}th,td{border:1px solid #ccd5dd;padding:6px}th{background:#eef4f7}svg{max-width:900px}</style>'+body+'</html>',encoding='utf-8')


if __name__=='__main__':
    results=run();render(*results)
    print(results[1][['P00_quantile','P11_quantile','n','D']].to_string(index=False))
