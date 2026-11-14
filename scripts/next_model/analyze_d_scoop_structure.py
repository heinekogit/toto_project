"""Exploratory score-structure audit; does not fit or deploy a D selector."""
from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / 'data/reports/d_scoop_phase1'
OUT = ROOT / 'data/reports/d_scoop_phase2_structure'
CALIPER = .01  # Absolute probability: one percentage point, fixed before inspection.
FEATURES = (
    'P_00', 'P_11', 'P_22', 'P_33plus',
    'draw_share00', 'draw_share11', 'draw_share22plus',
    'lambda_total', 'lambda_abs_difference', 'P_home_zero', 'P_away_zero',
    'P_total_le2', 'P_margin1', 'draw_share_within_zero_or_one',
    'HA_probability_gap', 'marginal_overlap', 'goal_difference_sd',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc_unoriented(y: np.ndarray, x: np.ndarray) -> float:
    """AUC for high feature values ranking D ahead of non-D; no direction fitting."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return float('nan')
    return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summary_by_result(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame.actual_D.to_numpy(dtype=bool)
    rows = []
    for field in ('P_D',) + FEATURES:
        x = frame[field].to_numpy(dtype=float)
        d = x[y]; non = x[~y]
        lo, hi = max(d.min(), non.min()), min(d.max(), non.max())
        rows.append(dict(feature=field, D_count=len(d), nonD_count=len(non),
                         D_mean=float(d.mean()), nonD_mean=float(non.mean()),
                         D_median=float(np.median(d)), nonD_median=float(np.median(non)),
                         D_q25=float(np.quantile(d,.25)), D_q75=float(np.quantile(d,.75)),
                         nonD_q25=float(np.quantile(non,.25)), nonD_q75=float(np.quantile(non,.75)),
                         D_min=float(d.min()), D_max=float(d.max()),
                         nonD_min=float(non.min()), nonD_max=float(non.max()),
                         range_overlap_low=float(lo) if lo <= hi else np.nan,
                         range_overlap_high=float(hi) if lo <= hi else np.nan,
                         D_fraction_in_overlap=float(((d>=lo)&(d<=hi)).mean()) if lo <= hi else 0.,
                         nonD_fraction_in_overlap=float(((non>=lo)&(non<=hi)).mean()) if lo <= hi else 0.,
                         descriptive_AUC_high_means_D=auc_unoriented(y,x)))
    return pd.DataFrame(rows)


def pair_close(frame: pd.DataFrame, within_round: bool) -> pd.DataFrame:
    d = frame.loc[frame.actual_D].reset_index(drop=True)
    n = frame.loc[~frame.actual_D].reset_index(drop=True)
    gaps = abs(d.P_D.to_numpy()[:,None] - n.P_D.to_numpy()[None,:])
    eligible = gaps <= CALIPER + 1e-15
    if within_round:
        eligible &= d['round'].to_numpy()[:,None] == n['round'].to_numpy()[None,:]
    # First maximize matched count, then minimize total P(D) distance.
    cost = np.c_[np.where(eligible,gaps,1e6),np.ones((len(d),len(d)))]
    ii,jj = linear_sum_assignment(cost)
    rows = []
    for i,j in zip(ii,jj):
        if j >= len(n):
            continue
        a,b = d.iloc[i],n.iloc[j]
        row = dict(D_round=a['round'],D_match_no=int(a.match_no),
                   D_match=f'{a.home_team}－{a.away_team}',D_actual_score=a.actual_score,
                   nonD_round=b['round'],nonD_match_no=int(b.match_no),
                   nonD_match=f'{b.home_team}－{b.away_team}',nonD_actual_score=b.actual_score,
                   D_P_D=float(a.P_D),nonD_P_D=float(b.P_D),P_D_gap=float(abs(a.P_D-b.P_D)))
        for feature in FEATURES:
            row[f'D_{feature}'] = float(a[feature])
            row[f'nonD_{feature}'] = float(b[feature])
            row[f'delta_{feature}'] = float(a[feature]-b[feature])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(pairs: pd.DataFrame, scope: str) -> pd.DataFrame:
    records=[]
    for feature in FEATURES:
        delta=pairs[f'delta_{feature}'].to_numpy(dtype=float) if len(pairs) else np.array([])
        records.append(dict(scope=scope,feature=feature,pairs=len(delta),
                            D_higher=int((delta>1e-12).sum()),D_lower=int((delta < -1e-12).sum()),
                            tied=int((abs(delta)<=1e-12).sum()),
                            median_D_minus_nonD=float(np.median(delta)) if len(delta) else np.nan,
                            mean_D_minus_nonD=float(delta.mean()) if len(delta) else np.nan))
    return pd.DataFrame(records)


def score_type(actual_score: str) -> str:
    h,a=(int(v) for v in str(actual_score).replace('-', '–').split('–'))
    if h!=a: return '非D'
    return '0–0' if h==0 else '1–1' if h==1 else '2–2' if h==2 else '3–3以上'


def boundary_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for rid,group in frame.groupby('round',sort=True):
        selected=group.loc[group.PD_rank.le(4) & ~group.actual_D]
        missed=group.loc[group.PD_rank.between(5,6) & group.actual_D]
        for _,a in missed.iterrows():
            for _,b in selected.iterrows():
                row=dict(round=rid,missed_D_no=int(a.match_no),missed_D_rank=int(a.PD_rank),
                         missed_D_match=f'{a.home_team}－{a.away_team}',
                         missed_D_score=a.actual_score,selected_nonD_no=int(b.match_no),
                         selected_nonD_rank=int(b.PD_rank),selected_nonD_match=f'{b.home_team}－{b.away_team}',
                         selected_nonD_score=b.actual_score,
                         P_D_gap=float(b.P_D-a.P_D))
                for feature in FEATURES:
                    row[f'delta_{feature}']=float(a[feature]-b[feature])
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    phase1=PHASE1/'all_diagnostics.csv'
    audit=json.loads((PHASE1/'audit.json').read_text())
    input_hashes={str(p):sha(p) for p in PHASE1.rglob('*') if p.is_file()}
    protected={str(p):sha(p) for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()}
    if len(audit['source_audits'])!=6:
        raise ValueError('Phase 1 source audit does not contain six rounds')
    for rec in audit['source_audits']:
        rid=rec['round'];base=ROOT/'data/eval/toto_rounds'/rid
        if sha(base/'snapshot/next_predictions.csv')!=rec['forecast_sha256']:
            raise ValueError(f'{rid}: forecast changed since Phase 1')
        if sha(base/'actual_results.csv')!=rec['result_sha256']:
            raise ValueError(f'{rid}: result changed since Phase 1')
    data=pd.read_csv(phase1).sort_values(['round','match_no']).reset_index(drop=True)
    if len(data)!=78 or data.match_id.duplicated().any() or int(data.actual_D.sum())!=18:
        raise ValueError('Expected 78 unique matches and 18 draws')
    if any(c not in data for c in FEATURES):
        raise ValueError('Required Phase 1 feature missing')
    if not np.isfinite(data[list(FEATURES)+['P_D']].to_numpy(dtype=float)).all():
        raise ValueError('Missing or nonfinite diagnostic')
    data['score_type']=data.actual_score.map(score_type)
    if not (data.actual_D == data.score_type.ne('非D')).all():
        raise ValueError('Actual score/result inconsistency')
    # Rank and selection must be the saved Phase 1 rules, not retrospectively edited.
    for rid,g in data.groupby('round'):
        order=np.lexsort((g.match_no.to_numpy(),-g.P_D.to_numpy()))
        rank=np.empty(13,dtype=int);rank[order]=np.arange(1,14)
        if not np.array_equal(rank,g.PD_rank.to_numpy(dtype=int)):
            raise ValueError(f'{rid}: Phase 1 P(D) rank mismatch')
        if not np.array_equal(rank<=4,g.baseline_selected.to_numpy(dtype=bool)):
            raise ValueError(f'{rid}: Phase 1 selection mismatch')
    missed=data.loc[data.actual_D & ~data.baseline_selected].copy()
    if len(missed)!=12:raise ValueError('Expected 12 missed draws')
    compare=summary_by_result(data)
    within=pair_close(data,True)
    across=pair_close(data,False)
    near_summary=pd.concat([paired_summary(within,'same_round_primary'),paired_summary(across,'cross_round_sensitivity')],ignore_index=True)
    boundary=data.loc[data.PD_rank.between(3,6)].copy()
    boundary_compare=boundary_pairs(data)
    type_summary=data.loc[data.actual_D].groupby(['score_type','baseline_selected']).size().rename('matches').reset_index()
    missed_type=missed.score_type.value_counts()
    near_share11=near_summary.loc[(near_summary.scope.eq('same_round_primary')) & (near_summary.feature.eq('draw_share11'))].iloc[0]
    near_zero=near_summary.loc[(near_summary.scope.eq('same_round_primary')) & (near_summary.feature.eq('P_00'))].iloc[0]
    eligible_missed=missed.loc[missed.PD_rank.between(5,6)]
    boundary_rounds=sorted(eligible_missed['round'].unique().tolist())
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in [('all_78_diagnostics.csv',data),('draw_vs_non_draw.csv',compare),
                       ('missed_12_draws.csv',missed),('near_PD_pairs_same_round.csv',within),
                       ('near_PD_pairs_cross_round.csv',across),('near_PD_feature_summary.csv',near_summary),
                       ('boundary_rank3_to6.csv',boundary),('boundary_exchange_pairs.csv',boundary_compare),
                       ('draw_score_type_counts.csv',type_summary)]:
        frame.to_csv(OUT/name,index=False)
    source_audit=dict(scope='d-scoop Phase 2 structure exploratory',rounds=6,matches=78,draws=18,
        missed_draws=12,caliper_absolute_P_D=CALIPER,
        pairing='one-to-one maximum-cardinality D/non-D matching; then minimum total P(D) gap',
        primary_pairing='within the same toto round',sensitivity_pairing='across rounds; model versions differ',
        feature_directions='Unoriented; descriptive AUC >0.5 means larger feature in D; no threshold learned',
        model_changes='none',scoop_score_created=False,new_coefficients_created=False,
        phase1_file_sha256=input_hashes,source_audits=audit['source_audits'],
        same_round_pairs=len(within),cross_round_pairs=len(across),boundary_exchange_pairs=len(boundary_compare),
        caveat='Selection of interesting patterns after observing outcomes is exploratory, not independent validation.')
    if any(sha(Path(path))!=digest for path,digest in input_hashes.items()):raise AssertionError('Phase 1 changed')
    if any(sha(Path(path))!=digest for path,digest in protected.items()):raise AssertionError('Production changed')
    (OUT/'audit.json').write_text(json.dumps(source_audit,ensure_ascii=False,indent=2))
    def table(frame: pd.DataFrame) -> str:
        return '<div class="scroll">'+frame.to_html(index=False,escape=True,float_format=lambda v:f'{v:.5f}')+'</div>'
    display_cols=['feature','D_median','nonD_median','D_min','D_max','nonD_min','nonD_max',
                  'D_fraction_in_overlap','nonD_fraction_in_overlap','descriptive_AUC_high_means_D']
    missed_cols=['round','match_no','home_team','away_team','actual_score','P_D','PD_rank','score_P_D',
                 'P_00','P_11','P_22','P_33plus','draw_share00','draw_share11','draw_share22plus',
                 'lambda_home','lambda_away','lambda_total','lambda_abs_difference','P_margin1']
    bcols=['round','match_no','home_team','away_team','PD_rank','baseline_selected','P_D','P_00','P_11',
           'P_22','P_33plus','lambda_total','lambda_abs_difference','P_margin1','actual_score']
    tcounts=type_summary.pivot(index='score_type',columns='baseline_selected',values='matches').fillna(0).astype(int)
    tcounts.columns=['4位外','4位内'] if False in tcounts.columns and True in tcounts.columns else [str(c) for c in tcounts.columns]
    tcounts=tcounts.reset_index()
    h='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>d-scoop Phase 2・第2段階</title>
<style>body{font:16px/1.8 system-ui;margin:28px;color:#203147}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:6px;border:1px solid #ccd3df}th{background:#edf2f8}</style>
<h1>d-scoop Phase 2・第2段階：得点構造とD捕捉漏れ</h1>
<p>保存済みの6開催・78試合、実際のD18試合・非D60試合を対象。Phase 1の11×11得点行列から再現済みの指標を使用。予測と結果の照合・Phase 1のハッシュを再確認。選考式・係数・総合スコアは作成していない。</p>
<p>次の表は結果を見た後の探索的な比較であり、指標に将来の予測力がある証明ではない。0–0等は同じλ2個から生じるため、互いに相関する。Dのスコア型は実結果の分類にだけ使い、指標の作成や選考には使わない。</p>'''
    h+='<h2>実際のDのスコア型</h2>'+table(tcounts)
    h+='<h2>A：D18試合と非D60試合</h2><p>中央値・範囲・両群の値の重なる範囲に入る割合を表示。全数値は<a href="draw_vs_non_draw.csv">比較CSV</a>に平均・四分位数も記録。AUCは大きい値がDになりやすい向きで測定し、方向を最適化していない。</p>'+table(compare[display_cols])
    h+='<h2>B：捕捉漏れのD12試合</h2>'+table(missed[missed_cols])
    h+=f'<h2>C：P(D)が近いD/非Dの比較</h2><p>許容差は絶対確率{CALIPER:.2f}（1パーセントポイント）。Dと非Dを一対一で対応付け、まず組数を最大化、次にP(D)差の総和を最小化。同一開催内を主分析とする：{len(within)}組。開催間の対応はモデル版が異なるため感度確認のみ：{len(across)}組。条件を満たさないDは無理に対応付けない。</p>'
    h+=table(near_summary.loc[near_summary.scope.eq('same_round_primary'),['feature','pairs','D_higher','D_lower','tied','median_D_minus_nonD']])
    h+='<p>主分析の全組合せは<a href="near_PD_pairs_same_round.csv">同一開催ペアCSV</a>。各特徴量について両試合の値と差を保存した。<a href="near_PD_pairs_cross_round.csv">開催間ペアCSV</a>は感度確認。</p>'
    h+='<h2>D：各開催のP(D)順位3〜6位</h2>'+table(boundary[bcols])
    h+=f'<p>5〜6位の実際のDと、同じ開催の上位4に入った非Dを全組合せで比較：{len(boundary_compare)}組。結果を見て組を作った診断であり、事前の交換規則ではない。<a href="boundary_exchange_pairs.csv">境界ペアCSV</a>に特徴量差を記録。</p>'
    h+=f'''<h2>5つの結論</h2><ol>
<li>取り逃がし12試合の実結果は0–0が{int(missed_type.get('0–0',0))}、1–1が{int(missed_type.get('1–1',0))}、2–2が{int(missed_type.get('2–2',0))}、3–3以上が{int(missed_type.get('3–3以上',0))}。実際の0–0は全6試合が取り逃がし。ただし取り逃がし群の予測P00中央値は{missed.P_00.median():.4f}で、捕捉群の{data.loc[data.actual_D & data.baseline_selected,'P_00'].median():.4f}より低い。事前に使える0–0確率の共通高値は確認できない。</li>
<li>同一開催かつP(D)差1ポイント以内でDと非Dを対応させた{len(within)}組では、P00がD側で高い組は{int(near_zero.D_higher)}/{len(within)}、同点内1–1比率が高い組は{int(near_share11.D_higher)}/{len(within)}。得点構造の値は違うが、D側に一貫した方向の差は見えない。対応組は同一開催内の一対一で、P(D)中央値差は{within.P_D_gap.median():.4f}。</li>
<li>境界3〜6位にいる取り逃がしDは{len(eligible_missed)}試合、対象開催は{', '.join(boundary_rounds)}。同じ開催の上位4非Dとの{len(boundary_compare)}比較では、取り逃がしD側の予測P00・P11は全比較で低く、λ合計は全比較で高い。これらを単純に優遇しても交換を支持しない。比較には同じ試合が複数回現れる。</li>
<li>境界の取り逃がしは3開催にまたがるが、共通の事前指標の方向が確認できない。全78試合でも各指標の値域は大きく重なり、モデル版も開催ごとに異なる。</li>
<li>この78試合では独自の総合スコアを試作して選出方式を替えるだけの根拠は得られない。仮説を少数に固定し、後続の新しい開催でP(D)近接比較と4位境界の入れ替えを検証する段階に留める。</li>
</ol><p>1–1比率単独の全試合AUCは{float(compare.loc[compare.feature.eq('draw_share11'),'descriptive_AUC_high_means_D'].iloc[0]):.3f}だが、同じ6開催を見てから注目した探索値で、採用基準ではない。1点差確率も試合中の同点化確率ではない。</p>
<p>次段階の判断には、少数の候補指標を事前に固定し、新しい試合前予測と後続の実結果でP(D)上位4との選考差を確かめる必要がある。D捕捉の増加が13的中を保証するわけでもない。</p>
<p><a href="all_78_diagnostics.csv">全78試合CSV</a> ／ <a href="missed_12_draws.csv">捕捉漏れ12試合CSV</a> ／ <a href="boundary_rank3_to6.csv">各開催の選出境界CSV</a> ／ <a href="audit.json">再現性監査</a></p></html>'''
    (OUT/'index.html').write_text(h)
    print('D score types:');print(tcounts.to_string(index=False))
    print('Same-round pairs:',len(within),'Cross-round pairs:',len(across),'Boundary exchange pairs:',len(boundary_compare))
    print('Feature comparison:');print(compare[['feature','D_median','nonD_median','D_fraction_in_overlap','nonD_fraction_in_overlap','descriptive_AUC_high_means_D']].to_string(index=False))
    print('Near-pair summary:');print(near_summary.loc[near_summary.scope.eq('same_round_primary'),['feature','D_higher','D_lower','median_D_minus_nonD']].to_string(index=False))
    print(OUT/'index.html')


if __name__=='__main__':
    main()
