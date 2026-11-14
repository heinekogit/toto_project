"""Read-only, exploratory audit of actual draws in six prematch Toto rounds."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/reports/buyplan_v40_draw_audit'
ROUNDS = ('toto1660', 'toto1663', 'toto1664', 'toto1665', 'toto1668', 'toto1670')
KEY = ['round', 'match_no']
SENS = ROOT / 'data/reports/score_structure_phase3/match_sensitivity_78.csv'
STRUCT = ROOT / 'data/reports/score_structure_phase12/match_structure_78.csv'
BASE = ROOT / 'data/reports/score_structure_phase4a_stage1/match_diagnostics.csv'
PROV = ROOT / 'data/reports/score_structure_phase4a_audit/forecast_result_join.csv'
MATRIX = ROOT / 'data/reports/score_structure_phase12/score_matrix_cells_78.csv'


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding='utf-8-sig')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(values: np.ndarray, labels: np.ndarray) -> float:
    pos = values[labels]
    neg = values[~labels]
    if not len(pos) or not len(neg):
        return float('nan')
    return float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


def analyze() -> dict:
    provenance = read(PROV)
    originals = provenance[(provenance['round'].isin(ROUNDS)) & (provenance.provenance_class == 'A')].copy()
    assert len(originals) == 78 and not originals.duplicated(KEY).any()
    assert originals.prematch_hash_verified.all() and originals.actual_join_verified.all()
    sources = [SENS, STRUCT, BASE, PROV, MATRIX]
    sources += [ROOT / s for s in originals.source_file.unique()]
    before = {str(s.relative_to(ROOT)): digest(s) for s in sources}
    for source_file, records in originals.groupby('source_file'):
        assert records.source_sha256.nunique() == 1
        assert digest(ROOT / source_file) == records.source_sha256.iloc[0]
    base = read(BASE)
    structure = read(STRUCT)
    sensitivity = read(SENS)
    assert all(len(x) == 78 and not x.duplicated(KEY).any() for x in [base, structure, sensitivity])
    keep = KEY + ['P_0_0', 'P_1_1', 'P_2_2', 'P_3_3', 'P_draw_4plus', 'P_draw',
                  'P_home_1', 'P_away_1', 'P_margin_1', 'P_margin_2plus',
                  'P_draw_2plus', 'lambda_total', 'lambda_abs_diff']
    senscols = [c for c in sensitivity.columns if c.startswith('delta_draw_') and c.endswith('_0.2')]
    df = base.merge(structure[keep], on=KEY, validate='one_to_one').merge(sensitivity[KEY + senscols], on=KEY, validate='one_to_one')
    df = df.merge(originals[KEY + ['source_file', 'source_sha256', 'provenance_class']], on=KEY, validate='one_to_one')
    assert len(df) == 78 and df.actual_result.eq(1).sum() == 18
    assert np.allclose(df.raw_P_D, df.P_draw, atol=1e-10)
    df['final_D_rank'] = df.groupby('round').final_P_D.rank(method='first', ascending=False).astype(int)
    df['top_prediction'] = df[['final_P_H', 'final_P_D', 'final_P_A']].to_numpy().argmax(axis=1)
    df['top_prediction'] = df.top_prediction.map({0: 'H', 1: 'D', 2: 'A'})
    df['actual_D'] = df.actual_result.eq(1)
    df['actual_score_probability'] = np.nan
    cells = read(MATRIX)
    score_cols = list(cells.columns)
    # Phase 4-A has already computed the exact observed-score likelihood from the 11x11 grid.
    # Preserve that prematch prediction diagnostic; do not derive a post-result feature.
    if 'actual_score_model_probability' in read(ROOT / 'data/reports/score_structure_phase4a_stage2/actual_draws_18.csv').columns:
        dd = read(ROOT / 'data/reports/score_structure_phase4a_stage2/actual_draws_18.csv')
        df = df.drop(columns='actual_score_probability').merge(dd[KEY + ['actual_score_model_probability']], on=KEY, how='left', validate='one_to_one')
    df['sensitivity_upside_02'] = df[senscols].max(axis=1)
    df['sensitivity_downside_02'] = df[senscols].min(axis=1)
    df['sensitivity_range_02'] = df.sensitivity_upside_02 - df.sensitivity_downside_02
    df['sensitivity_absmax_02'] = df[senscols].abs().max(axis=1)
    df['balance_closing_02'] = np.where(df.lambda_home >= df.lambda_away,
        df['delta_draw_H低下・A上昇_0.2'], df['delta_draw_H上昇・A低下_0.2'])
    df['both_lower_02'] = df['delta_draw_双方低下_0.2']
    df['both_higher_02'] = df['delta_draw_双方上昇_0.2']
    df['lambda_signed_gap'] = df.lambda_home - df.lambda_away
    df['actual_minus_lambda_home'] = df.actual_home - df.lambda_home
    df['actual_minus_lambda_away'] = df.actual_away - df.lambda_away
    df['draw_score_band'] = np.where(~df.actual_D, '非D', np.where(df.actual_home == 0, '0-0', np.where(df.actual_home == 1, '1-1', '2-2以上')))
    df['expected_actual_draw_score_probability'] = np.select(
        [df.draw_score_band.eq('0-0'), df.draw_score_band.eq('1-1'), df.draw_score_band.eq('2-2以上')],
        [df.P_0_0, df.P_1_1, df.P_draw_2plus], default=np.nan)
    # Prove the single-row data match the A-only originals, including final probabilities and scores.
    checks = df.merge(originals[KEY + ['next_prob_home', 'next_prob_draw', 'next_prob_away', 'actual_home_score', 'actual_away_score']], on=KEY)
    for a, b in [('final_P_H', 'next_prob_home'), ('final_P_D', 'next_prob_draw'), ('final_P_A', 'next_prob_away'), ('actual_home', 'actual_home_score'), ('actual_away', 'actual_away_score')]:
        assert np.allclose(checks[a], checks[b], atol=1e-10)

    # A small, prespecified feature list. No direction optimization, coefficient fitting, or selection rule.
    metrics = ['final_P_D', 'raw_P_D', 'P_0_0', 'P_1_1', 'P_draw_2plus', 'P_margin_1',
               'P_margin_2plus', 'lambda_total', 'lambda_abs_diff', 'sensitivity_upside_02',
               'sensitivity_absmax_02', 'balance_closing_02', 'both_lower_02', 'both_higher_02']
    rows = []
    for metric in metrics:
        yes = df.loc[df.actual_D, metric].to_numpy(float)
        no = df.loc[~df.actual_D, metric].to_numpy(float)
        rows.append(dict(metric=metric, D_n=len(yes), nonD_n=len(no), D_mean=yes.mean(), nonD_mean=no.mean(),
                         D_median=np.median(yes), nonD_median=np.median(no), D_min=yes.min(), D_max=yes.max(),
                         nonD_min=no.min(), nonD_max=no.max(), AUC_higher_means_D=auc(df[metric].to_numpy(float), df.actual_D.to_numpy(bool))))
    comparison = pd.DataFrame(rows)

    # Match each D once to the nearest non-D from the same model generation, within 1 P(D) point.
    # This is diagnostic matching, not a balanced causal design or an independent holdout.
    used: set[tuple] = set()
    pairs = []
    for _, d in df[df.actual_D].sort_values(['round', 'final_P_D', 'match_no']).iterrows():
        candidates = df[(~df.actual_D) & (df.model_signature == d.model_signature)].copy()
        candidates['pd_gap'] = (candidates.final_P_D - d.final_P_D).abs()
        candidates = candidates[candidates.pd_gap <= .01].sort_values(['pd_gap', 'round', 'match_no'])
        candidates = candidates[[ (r['round'], int(r.match_no)) not in used for _, r in candidates.iterrows()]]
        if candidates.empty:
            continue
        n = candidates.iloc[0]
        used.add((n['round'], int(n.match_no)))
        row = dict(D_round=d['round'], D_match_no=int(d.match_no), D_fixture=f"{d.home_team}－{d.away_team}",
                   nonD_round=n['round'], nonD_match_no=int(n.match_no), nonD_fixture=f"{n.home_team}－{n.away_team}",
                   model_signature=d.model_signature, P_D_gap=abs(d.final_P_D-n.final_P_D))
        for metric in metrics:
            row['D_'+metric] = d[metric]
            row['nonD_'+metric] = n[metric]
            row['diff_'+metric] = d[metric] - n[metric]
        pairs.append(row)
    pairs = pd.DataFrame(pairs)
    pair_stats = pd.DataFrame([dict(metric=m, pairs=len(pairs), D_higher=int((pairs['diff_'+m]>0).sum()),
                                    D_lower=int((pairs['diff_'+m]<0).sum()), mean_diff=pairs['diff_'+m].mean(),
                                    median_diff=pairs['diff_'+m].median()) for m in metrics])

    # Audit extra prematch fields only. Availability is not evidence of additional predictive power.
    categories = {
      '疲労・間隔': ['home_fatigue_score','away_fatigue_score','home_rest_hours','away_rest_hours'],
      '天候': ['stadium_weather','weather_fetch_ok','adverse_weather_during_match'],
      '欠場': ['absence_impact_total_home','absence_impact_total_away','absence_players_count_home','absence_players_count_away'],
      '順位・勝点': ['rankmot_rank_latest_home','rankmot_rank_latest_away','rankmot_points_latest_home','rankmot_points_latest_away'],
      'Football LAB': ['flab_expected_for_xg_home','flab_expected_for_xg_away','flab_expected_against_xg_home','flab_expected_against_xg_away','flab_possession_rate_home','flab_possession_rate_away'],
    }
    avail=[]
    for rnd in ROUNDS:
        path=ROOT / originals.loc[originals['round']==rnd,'source_file'].iloc[0]
        snap=read(path)
        assert len(snap)==13 and not snap.match_no.duplicated().any()
        for group, fields in categories.items():
            for field in fields:
                present=field in snap
                count=int(snap[field].notna().sum()) if present else 0
                avail.append(dict(round=rnd, category=group, field=field, column_present=present, nonmissing=count,
                                  nonmissing_fraction=count/13, source_file=str(path.relative_to(ROOT))))
    availability=pd.DataFrame(avail)
    external=[]
    numeric_fields=[f for group in categories.values() for f in group if f not in ('stadium_weather','weather_fetch_ok','adverse_weather_during_match')]
    for rnd in ROUNDS:
        path=ROOT / originals.loc[originals['round']==rnd,'source_file'].iloc[0]
        snap=read(path)
        fields=[f for f in numeric_fields if f in snap]
        part=snap[['match_no']+fields].copy();part.insert(0,'round',rnd)
        external.append(part)
    ext=pd.concat(external,ignore_index=True).merge(df[KEY+['actual_D','model_signature','final_P_D']],on=KEY,validate='one_to_one')
    ext_metrics=[]
    for field in numeric_fields:
        if field not in ext: continue
        val=pd.to_numeric(ext[field],errors='coerce');valid=val.notna();usable=ext[valid]
        if len(usable)<10: continue
        yes=val[valid & ext.actual_D];no=val[valid & ~ext.actual_D]
        ext_metrics.append(dict(field=field,observed=int(valid.sum()),distinct=int(val.nunique(dropna=True)),
                                min=val.min(),max=val.max(),D_mean=yes.mean(),nonD_mean=no.mean(),
                                AUC_higher_means_D=auc(val[valid].to_numpy(float),usable.actual_D.to_numpy(bool)) if len(yes) and len(no) else np.nan,
                                note='exploratory only; possible model input and version confounding'))
    ext_summary=pd.DataFrame(ext_metrics)
    external_pairs=pairs[['D_round','D_match_no','D_fixture','nonD_round','nonD_match_no','nonD_fixture','P_D_gap','model_signature']].copy()
    for field in numeric_fields:
        if field not in ext: continue
        lookup=ext.set_index(KEY)[field]
        external_pairs['D_'+field]=[lookup.get((r,int(m)),np.nan) for r,m in zip(external_pairs.D_round,external_pairs.D_match_no)]
        external_pairs['nonD_'+field]=[lookup.get((r,int(m)),np.nan) for r,m in zip(external_pairs.nonD_round,external_pairs.nonD_match_no)]
        external_pairs['diff_'+field]=pd.to_numeric(external_pairs['D_'+field],errors='coerce')-pd.to_numeric(external_pairs['nonD_'+field],errors='coerce')

    OUT.mkdir(parents=True,exist_ok=True)
    draws=df[df.actual_D].copy().sort_values(['round','match_no'])
    for name, frame in [('all_78.csv',df),('actual_draws_18.csv',draws),('draw_non_draw_features.csv',comparison),
                        ('near_PD_matched_pairs.csv',pairs),('matched_pair_summary.csv',pair_stats),('prematch_field_availability.csv',availability),
                        ('prematch_external_78.csv',ext),('external_feature_diagnostics.csv',ext_summary),('matched_external_pairs.csv',external_pairs)]:
        frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    after={str(s.relative_to(ROOT)): digest(s) for s in sources}
    assert before==after
    audit=dict(matches=len(df), actual_draws=len(draws), matched_pairs=len(pairs), source_hashes=before,
               originals_unchanged=True, no_model_or_buyplan_change=True, PD_match_caliper=.01,
               match_same_model_signature=True, results_used_for_feature_selection=False,
               source_note='Phase 4-A provenance A only; historical score and scenario analyses reused')
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(df=df,draws=draws,comparison=comparison,pairs=pairs,pair_stats=pair_stats,availability=availability,external_summary=ext_summary,audit=audit)


def render(x: dict) -> None:
    df=x['df']; draws=x['draws']; comp=x['comparison']; pairs=x['pairs']; ps=x['pair_stats']; av=x['availability']; ex=x['external_summary']
    def pct(v): return f'{v*100:.2f}%'
    def table(frame):
        cols=list(frame.columns)
        def cell(v):
            if pd.isna(v): return ''
            if isinstance(v,(float,np.floating)): return f'{v:.4f}'
            return str(v).replace('|','/')
        return '\n'.join(['| '+' | '.join(cols)+' |', '| '+' | '.join(['---']*len(cols))+' |']+
                         ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
    top=draws.sort_values('final_P_D').head(6)
    keymetrics=['sensitivity_upside_02','sensitivity_absmax_02','balance_closing_02','both_lower_02','P_0_0','P_1_1','P_margin_1']
    paragraphs=[
      '# BuyPlan v4.0：実際のDから予測の見落とし候補を調べる',
      '## 結論',
      f"試合前保存原本Aの78試合（実D18、非D60）を監査した。最終P(D)のAUCは{comp.set_index('metric').loc['final_P_D','AUC_higher_means_D']:.3f}。既存Phase 4-A第2段階と同じく、この標本で強いD識別力は確認できない。今回追加した同一モデル・P(D)差1ポイント以内の1対1対応比較は{len(pairs)}組。λ感応度や得点構造に、既存P(D)を上回る安定した識別根拠は確認できない。**新しいD選別規則を採用する根拠は得られていない。**",
      '実際にDになったことだけでP(D)約25%を誤予測と判定しない。λ−実得点の差は偶然性を含む得点残差であり、真のλ誤差ではない。6開催は複数モデル版が混在し、18件では新しい係数を検証できない。',
      '## 既存研究との区別',
      '既存：Phase 1・2は11×11得点分布、同点帯、1点差・2点差以上を再現。Phase 3はλ±0.1/0.2の数学的感応度を計算。Phase 4-Aは試合前原本Aの照合、P(D)の校正・識別、D/非Dの単純比較と近接ペア列挙を実施。v4.0試作は17状態から10口を生成したが、6開催で13的中なし。',
      '今回新規：**同じモデル版でP(D)が近いDと非Dを重複なしで1対1対応**させ、事前に指定した感応度・構造指標の差を確認。さらに、保存原本の疲労・天候・欠場・順位・LAB列の存在と欠損を監査した。',
      '## D18試合の個別診断',
      '下表の予測確率は最終値。rawとmix、λ、同点帯、勝敗点差、全8方向の感応度、得点残差は `actual_draws_18.csv` に記録。実スコアの予測確率は事後評価専用で、特徴選択には使わない。',
      table(draws[['round','match_no','home_team','away_team','actual_home','actual_away','lambda_home','lambda_away','final_P_H','final_P_D','final_P_A','raw_P_D','final_D_rank','top_prediction','P_0_0','P_1_1','P_2_2','P_margin_1','P_margin_2plus','sensitivity_upside_02','balance_closing_02']]),
      '### 最終P(D)が低かった実D',
      table(top[['round','match_no','home_team','away_team','actual_category','actual_home','actual_away','final_P_D','final_D_rank','lambda_home','lambda_away','P_0_0','P_1_1','sensitivity_upside_02']]),
      '具体例：1664回・京都－鹿島（2-2）はP(D)23.43%、開催内12位で、A50.37%を最有力とした。事前行列のP(2-2)は5.82%、λは1.21対1.76。1664回・FC東京－柏（0-0）はP(D)24.47%、10位で、H46.02%を最有力とした。事前行列のP(0-0)は5.69%、λは1.62対1.25。いずれもDをゼロと予測したわけではなく、個別の実現だけでは系統的な見落としを証明しない。',
      '低いP(D)のDにも、同程度のP(D)で決着した非Dの反例がある。順位だけから特定の計算工程の欠陥を断定できない。0-0・1-1・高得点Dの区分は実スコアに基づく**事後の説明用**であり、試合前の分類規則ではない。',
      '### 実スコア帯別の事後診断',
      table(draws.groupby('draw_score_band').agg(matches=('match_no','size'),mean_lambda_total=('lambda_total','mean'),mean_lambda_abs_diff=('lambda_abs_diff','mean'),mean_final_PD=('final_P_D','mean'),mean_actual_score_probability=('actual_score_model_probability','mean'),mean_both_lower_sensitivity=('both_lower_02','mean')).reset_index()),
      '実スコア別の得点期待値との差は `actual_draws_18.csv` の `actual_minus_lambda_home/away` に残した。0-0の得点残差は必ず負になるため、それをもって両側λの系統的過大評価とは判断できない。',
      '## D対非DとP(D)近接比較',
      '方向を後から反転せず、値が高いほどDとするAUCを記載。各指標は主として保存λから派生するため、同じ情報を重複して見ている。多数の指標を並べた探索的診断であり、有意性や将来の予測力を主張しない。',
      table(comp[['metric','D_mean','nonD_mean','D_median','nonD_median','AUC_higher_means_D']]),
      f"対応比較は同一model_signature・最終P(D)差≤1ポイント、非Dを重複使用せず最近傍で選んだ{len(pairs)}組。開催内一致は要求せず、モデル版を跨がない。対応不能なDは比較から外れる。対応は結果を使う診断であり、独立検証ではない。",
      table(ps[ps.metric.isin(keymetrics)]),
      '## 得点構造と感応度の解釈',
      '感応度は±0.2点の仮定による差分で、現実のλ誤差の方向・頻度ではない。`sensitivity_upside_02` は8方向中でDを最も押し上げる値であるため、実際のDに最も都合のよい方向を事後選択したものではないが、現実にその方向が起きた証拠もない。`balance_closing_02` は高い側λを下げ低い側を上げる方向を試合前λだけで決めたもの。`both_lower_02` と `both_higher_02` は低・高得点型の反応を分ける。これらはP(D)の計算に既に関わるλ2個から決定され、独立情報ではない。',
      '## 得点構造以外の試合前データ',
      '各原本の列充足を `prematch_field_availability.csv` に記録。欠場、天候、疲労、順位、LABの「列が存在する」と「13件で値がある」は異なる。値があっても、現行λ/P(D)に既に取り込まれた可能性がある。今回これらの値で係数や個別結果の因果説明は作っていない。取得時点・バージョン・適用段階の監査と、より長い同一モデル期の予測履歴が必要。試合後の退場・PK・展開は予測入力として使用不可。',
      table(av.groupby(['category','field']).agg(rounds_with_column=('column_present','sum'), nonmissing_of_78=('nonmissing','sum')).reset_index()),
      '数値列については全78試合での欠損数・異なる値の数・D/非D平均・AUCを追加診断した。ただし、モデル入力との重複と開催ごとのバージョン差があるため、AUCの大小で見落とし要因と判断しない。最も目立つのはホーム側Football LAB被xGのAUC 0.638だが、P(D)近接・同一モデル18組では実D側が高いのは11組、低いのは7組。C大阪－横浜FMの1.886対町田－福岡の1.186は支持例、FC東京－柏の0.997対川崎F－神戸の1.404は反例。これは現行λに組み込まれているxGAとの重複が強く疑われ、追加予測力の証拠ではない。',
      table(ex[['field','observed','distinct','D_mean','nonD_mean','AUC_higher_means_D']]),
      '## 要因の分類と次段階',
      'A（既存P(D)に含まれる）：λ合計・差、同点帯、1点差、Phase 3の感応度。補正後P(D)への単純加点は二重評価となる。',
      'B（P(D)で表し切れない可能性）：同一P(D)でも同点帯と感応度は異なる。ただし今回の対応比較だけでは**実際のDとの再現性ある関係は未確認**。将来の同一モデル版・後続開催で検証する候補。',
      'C（追加または適用確認が必要）：欠場、天候、疲労、順位、チームスタイル、相性。保存原本に一部列はあるが、品質・試合前時点・既存モデルへの反映・独立の追加情報を確認してから評価する。',
      'D（事後のみ）：実スコア、退場、PK、当日の試合展開。今回の個別解説・残差には使えても購入前の判断には使えない。',
      'Next Model候補は、同一モデル版での時系列外部検証と入力監査。BuyPlan候補は、検証済みの予測分布・不確実性を10口の13的中目的で比較すること。D最低本数、固定ダブル、上位4選出は導入しない。今回の結果だけでは、Dになった試合に共通し、**現在のP(D)にない予測可能な特徴を発見したとは言えない**。',
      '## 再現性と制限',
      '入力はPhase 4-A監査のA原本78件のみ。既存のPhase 1〜3とPhase 4-AのCSVを再利用。保存原本のSHA-256は実行前後で一致。予測モデル・BuyPlan・購入処理は変更なし。比較は探索的で、結果を見てからの係数学習や新規配券はしていない。',
    ]
    report='\n\n'.join(paragraphs)+'\n'
    (OUT/'report.md').write_text(report,encoding='utf-8')
    # Standalone HTML, with the complete individual and comparison tables accessible in browser.
    import re
    lines=report.splitlines(); parts=[]; i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('| '):
            block=[]
            while i<len(lines) and lines[i].startswith('| '): block.append(lines[i]); i+=1
            rows=[[html.escape(v.strip()) for v in l.strip('|').split('|')] for l in block]
            parts.append('<table><thead><tr>'+''.join('<th>'+v+'</th>' for v in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows[2:])+'</tbody></table>')
            continue
        if line.startswith('# '): parts.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '): parts.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('### '): parts.append('<h3>'+html.escape(line[4:])+'</h3>')
        elif line: parts.append('<p>'+html.escape(line).replace('**','')+'</p>')
        i+=1
    body='\n'.join(parts)
    css='body{font:15px/1.65 system-ui;max-width:1400px;margin:30px auto;padding:0 18px;color:#193047}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{border:1px solid #cbd5df;padding:5px 7px}th{background:#edf3f8}table{display:block;overflow:auto}h1,h2{color:#153b5d}a{color:#1762a3}'
    links='<p><a href="actual_draws_18.csv">D18試合CSV</a> · <a href="near_PD_matched_pairs.csv">近接ペアCSV</a> · <a href="all_78.csv">全78試合CSV</a> · <a href="prematch_field_availability.csv">入力列監査CSV</a></p>'
    chart_rows=comp[comp.metric.isin(['final_P_D','P_0_0','P_1_1','P_margin_1','sensitivity_upside_02','balance_closing_02'])]
    chart=['<svg viewBox="0 0 650 275" role="img" aria-label="事前指標のD対非D AUC。0.5は偶然水準">',
           '<line x1="350" y1="10" x2="350" y2="258" stroke="#aab" stroke-dasharray="4 4"/><text x="355" y="14" font-size="12">0.5</text>']
    for i,r in enumerate(chart_rows.itertuples()):
        y=35+i*37; width=float(r.AUC_higher_means_D)*400
        chart.append(f'<text x="4" y="{y+15}" font-size="13">{html.escape(r.metric)}</text><rect x="150" y="{y}" width="{width:.1f}" height="20" fill="#3b7caa"/><text x="{160+width:.1f}" y="{y+15}" font-size="12">{r.AUC_higher_means_D:.3f}</text>')
    chart.append('</svg>')
    (OUT/'index.html').write_text(f'<!doctype html><html lang="ja"><meta charset="utf-8"><title>v4.0 実D監査</title><style>{css}</style><main>{links}<h2>事前指標の識別力（探索的）</h2>{"".join(chart)}{body}</main></html>',encoding='utf-8')


if __name__=='__main__':
    result=analyze();render(result)
    print(json.dumps({'matches':result['audit']['matches'],'D':result['audit']['actual_draws'],'matched_pairs':result['audit']['matched_pairs'],'output':str(OUT)},ensure_ascii=False))
