"""Exploratory draw discrimination and score-structure audit of 78 prematch forecasts."""
from __future__ import annotations
import json
from html import escape
from pathlib import Path
from math import comb
import numpy as np
import pandas as pd
from common import ROOT,sha256
from diagnose_score_structure_phase4a import diagnostics
from analyze_score_structure import score_grid
from forecast_registry import DEFAULT_ROOT,read_log,append_unique,digest

OUT=ROOT/'data/reports/score_structure_phase4a_stage2'
STRUCT=ROOT/'data/reports/score_structure_phase12/match_structure_78.csv'
SENS=ROOT/'data/reports/score_structure_phase3/match_sensitivity_78.csv'
ROUNDS=('toto1660','toto1663','toto1664','toto1665','toto1668','toto1670')


def discrimination(g,score):
    y=g.actual_result.eq(1).to_numpy(int);p=g[score].to_numpy(float)
    pos=int(y.sum());neg=len(y)-pos
    auc=float(np.mean([float(a>b)+.5*float(a==b) for a in p[y==1] for b in p[y==0]])) if pos and neg else np.nan
    order=np.argsort(-p,kind='stable');ordered=y[order]
    ap=float(np.mean([ordered[:i+1].mean() for i in np.flatnonzero(ordered)])) if pos else np.nan
    return dict(matches=len(g),actual_D=pos,mean_P_D=float(p.mean()),actual_D_rate=float(y.mean()),
                D_brier=float(np.mean((p-y)**2)),ROC_AUC=auc,average_precision=ap,
                median_P_D_actual_D=float(np.median(p[y==1])) if pos else np.nan,
                median_P_D_non_D=float(np.median(p[y==0])) if neg else np.nan,
                min_P_D_actual_D=float(np.min(p[y==1])) if pos else np.nan,
                max_P_D_actual_D=float(np.max(p[y==1])) if pos else np.nan,
                min_P_D_non_D=float(np.min(p[y==0])) if neg else np.nan,
                max_P_D_non_D=float(np.max(p[y==0])) if neg else np.nan)


def ranked(base):
    d=base.copy();d['actual_D']=d.actual_result.eq(1)
    for stage in ('raw','final'):
        d[f'{stage}_rank']=d.groupby('round')[f'{stage}_P_D'].rank(method='first',ascending=False).astype(int)
        # Exact ties are resolved by match number, rather than input order.
        for rid,g in d.groupby('round'):
            ix=g.sort_values([f'{stage}_P_D','match_no'],ascending=[False,True]).index
            d.loc[ix,f'{stage}_rank']=np.arange(1,len(ix)+1)
    return d.sort_values(['round','match_no']).reset_index(drop=True)


def hypergeom_null(draw_counts,k):
    dist=np.array([1.])
    for draws in draw_counts:
        local=np.array([comb(draws,x)*comb(13-draws,k-x)/comb(13,k) if 0<=x<=draws and 0<=k-x<=13-draws else 0 for x in range(k+1)])
        dist=np.convolve(dist,local)
    return dist


def topk(d):
    rows=[];null=[];draw_counts=[int(g.actual_D.sum()) for _,g in d.groupby('round')]
    for stage in ('raw','final'):
        for k in range(1,14):
            for rid,g in d.groupby('round'):
                actual=int(g.actual_D.sum());capt=int(g.loc[g[f'{stage}_rank']<=k,'actual_D'].sum())
                rows.append(dict(stage=stage,round=rid,k=k,selected=k,actual_D=actual,captured_D=capt,
                    recall=capt/actual if actual else np.nan,precision=capt/k,random_expected=k*actual/13,
                    excess=capt-k*actual/13))
            total=sum(r['captured_D'] for r in rows if r['stage']==stage and r['k']==k)
            expected=k*sum(draw_counts)/13;dist=hypergeom_null(draw_counts,k)
            rows.append(dict(stage=stage,round='全6開催',k=k,selected=6*k,actual_D=sum(draw_counts),captured_D=total,
                recall=total/sum(draw_counts),precision=total/(6*k),random_expected=expected,excess=total-expected))
            null.extend(dict(stage=stage,k=k,captured_D=x,probability=float(p),observed=total,
                p_at_least_observed=float(dist[total:].sum()),p_at_most_observed=float(dist[:total+1].sum()),
                expected=expected) for x,p in enumerate(dist))
    return pd.DataFrame(rows),pd.DataFrame(null)


def group_stats(d,columns,group_col):
    rows=[]
    for name,g in d.groupby(group_col,dropna=False):
        for col in columns:
            vals=g[col].dropna().astype(float)
            rows.append(dict(group=str(name),metric=col,n=len(vals),mean=float(vals.mean()),
                median=float(vals.median()),minimum=float(vals.min()),maximum=float(vals.max())))
    return pd.DataFrame(rows)


def near_pairs(d):
    rows=[]
    for i in range(len(d)):
        a=d.iloc[i]
        for j in range(i+1,len(d)):
            b=d.iloc[j];gap=abs(a.raw_P_D-b.raw_P_D)
            if gap>.01:continue
            rows.append(dict(round_A=a['round'],match_no_A=int(a.match_no),match_A=f'{a.home_team}－{a.away_team}',
                round_B=b['round'],match_no_B=int(b.match_no),match_B=f'{b.home_team}－{b.away_team}',
                same_round=a['round']==b['round'],mixed_result=bool(a.actual_D!=b.actual_D),
                raw_PD_A=a.raw_P_D,raw_PD_B=b.raw_P_D,gap_raw_PD=gap,within_0_005=gap<=.005,
                actual_D_A=bool(a.actual_D),actual_D_B=bool(b.actual_D),
                gap_P00=abs(a.P_0_0-b.P_0_0),gap_P11=abs(a.P_1_1-b.P_1_1),
                gap_P22plus=abs(a.P_draw_2plus-b.P_draw_2plus),
                gap_lambda_total=abs(a.lambda_total-b.lambda_total),
                gap_lambda_abs_diff=abs(a.lambda_abs_diff-b.lambda_abs_diff),
                gap_home1=abs(a.P_home_1-b.P_home_1),gap_away1=abs(a.P_away_1-b.P_away_1),
                gap_Hlow_delta_D=abs(a['delta_draw_H低下_0.1']-b['delta_draw_H低下_0.1']),
                gap_Alow_delta_D=abs(a['delta_draw_A低下_0.1']-b['delta_draw_A低下_0.1'])))
    return pd.DataFrame(rows)


def links():
    """Backfill provenance links only; never edit archived plan/forecast files."""
    events=read_log(DEFAULT_ROOT/'forecasts.jsonl');out=[]
    for rid in ROUNDS:
        matches=[e for e in events if e.get('round_id')==rid]
        snapshot_ids={e['output_sha256'] for e in matches}
        if len(matches)!=13 or len(snapshot_ids)!=1:raise ValueError(f'{rid}: ambiguous registry snapshot')
        directory=ROOT/'data/eval/toto_rounds'/rid/'snapshot'
        manifest_path=directory/'buyplan_engine_manifest.json'
        manifest=json.loads(manifest_path.read_text())
        hashes=manifest.get('output_sha256',{})
        if hashes.get('next_predictions.csv')!=next(iter(snapshot_ids)):
            raise ValueError(f'{rid}: forecast hash differs from plan input')
        plan=directory/'buyplan.csv'
        if sha256(plan)!=hashes.get('buyplan.csv'):raise ValueError(f'{rid}: plan hash mismatch')
        core=dict(event_type='portfolio_link',round_id=rid,forecast_snapshot_sha256=next(iter(snapshot_ids)),
            buyplan_sha256=sha256(plan),buyplan_version=manifest.get('official_version'),
            generated_at=manifest['created_at'],manifest_sha256=sha256(manifest_path),
            manifest_path=str(manifest_path),buyplan_path=str(plan))
        out.append(dict(core,event_id=digest(core)))
    added=append_unique(DEFAULT_ROOT,'portfolio_links.jsonl',out)
    return added,out


def markdown_table(frame):
    cols=list(frame.columns)
    def cell(value):
        return '' if pd.isna(value) else f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_hash={str(p):sha256(p) for p in (DEFAULT_ROOT/'forecasts.jsonl',DEFAULT_ROOT/'results.jsonl',STRUCT,SENS)}
    base=diagnostics()
    s=pd.read_csv(STRUCT,encoding='utf-8-sig').drop(columns=['home_team','away_team','lambda_home','lambda_away','final_P_D','raw_P_H','raw_P_D','raw_P_A','uncertainty_mix'])
    sensitivity=pd.read_csv(SENS,encoding='utf-8-sig').drop(columns=['home_team','away_team'])
    d=ranked(base.merge(s,on=['round','match_no'],validate='one_to_one').merge(sensitivity,on=['round','match_no'],validate='one_to_one'))
    assert len(d)==78 and d.actual_D.sum()==18 and set(d['round'])==set(ROUNDS)
    assert abs(d.raw_P_D.sum()-19.82046066)<.0001
    np.testing.assert_allclose(d.raw_P_D,d.P_0_0+d.P_1_1+d.P_draw_2plus,atol=1e-10)
    d['D_selection']='捕捉D' ;d.loc[d.actual_D & (d.final_rank>4),'D_selection']='漏れD'
    d.loc[~d.actual_D,'D_selection']='非D'
    score_prob=[]
    for _,r in d.iterrows():
        matrix,_=score_grid(r.lambda_home,r.lambda_away)
        score_prob.append(float(matrix[int(r.actual_home),int(r.actual_away)]) if r.actual_home<=10 and r.actual_away<=10 else np.nan)
    d['actual_score_model_probability']=score_prob
    d.to_csv(OUT/'ranked_matches_78.csv',index=False,encoding='utf-8-sig')
    top,null=topk(d);top.to_csv(OUT/'topk_capture.csv',index=False,encoding='utf-8-sig')
    null.to_csv(OUT/'hypergeometric_null.csv',index=False,encoding='utf-8-sig')
    assert top[(top['round']=='全6開催')&(top.k==13)].captured_D.eq(18).all()
    assert int(top[(top['round']=='全6開催')&(top.k==4)&(top.stage=='final')].captured_D.iloc[0])==6
    metrics=[]
    for stage in ('raw','final'):
        for group,g in [('全体',d)]+[(rid,g) for rid,g in d.groupby('round')]+[(sig,g) for sig,g in d.groupby('model_signature')]:
            metrics.append(dict(stage=stage,group=group,**discrimination(g,f'{stage}_P_D')))
    met=pd.DataFrame(metrics);met.to_csv(OUT/'discrimination.csv',index=False,encoding='utf-8-sig')
    cols=['lambda_total','lambda_abs_diff','P_0_0','P_1_1','P_draw_2plus','P_home_1','P_away_1','P_home_2','P_away_2','P_home_3plus','P_away_3plus','P_margin_1','P_margin_2plus']
    structure=group_stats(d,cols,'D_selection');structure.to_csv(OUT/'captured_missed_structure.csv',index=False,encoding='utf-8-sig')
    d[d.actual_D].to_csv(OUT/'actual_draws_18.csv',index=False,encoding='utf-8-sig')
    senscols=[c for c in d if c.startswith('delta_draw_')]
    sens=group_stats(d,senscols,'D_selection');sens.to_csv(OUT/'sensitivity_comparison.csv',index=False,encoding='utf-8-sig')
    pairs=near_pairs(d);pairs.to_csv(OUT/'near_probability_pairs.csv',index=False,encoding='utf-8-sig')
    added,linkrows=links()
    after={p:sha256(Path(p)) for p in source_hash}
    assert after==source_hash
    null4=null[(null.stage=='final')&(null.k==4)].iloc[0]
    audit=dict(matches=78,rounds=list(ROUNDS),draws=18,expected_raw_D=float(d.raw_P_D.sum()),
        model_counts=d.model_signature.value_counts().to_dict(),source_hashes=source_hash,
        plan_links=len(linkrows),new_plan_links=added,source_unchanged=True,official_logic_changed=False,
        rank_tie_rule='P(D) descending, then match_no ascending',null='per-round hypergeometric conditioned on observed D count; six-round convolution')
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    t4=top[(top['round']=='全6開催')&(top.k==4)]
    close=pairs.groupby(['within_0_005','same_round']).agg(pairs=('mixed_result','size'),mixed=('mixed_result','sum'),mean_gap_P00=('gap_P00','mean')).reset_index()
    maincols=['round','match_no','home_team','away_team','raw_P_D','raw_rank','final_P_D','final_rank','actual_home','actual_away','actual_D','P_0_0','P_1_1','P_draw_2plus']
    table=lambda x,cols:x[cols].to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}')
    p_ge=float(null4.p_at_least_observed)
    raw4=int(t4[t4.stage=='raw'].captured_D.iloc[0]);final4=int(t4[t4.stage=='final'].captured_D.iloc[0])
    report=f'''# Phase 4-A 第2段階：Dの識別力と得点構造

> **設計上の位置付け：P(D)上位4での捕捉／漏れDという分類は、廃棄済みv3.0構想の履歴・比較対象である。v4.0のD採用数や配券条件ではない。上位k（k=1〜13）の表は順位識別の診断に限り、k=4を既定値にしない。現在の方針は `BUYPLAN_V4_0.md` を参照。**

## 結論

試合前保存Aの6開催78試合（J1/J2、D18試合）だけを使用した。raw P(D)の期待D数は{d.raw_P_D.sum():.3f}、実際は18。最終P(D)上位4は{final4}/18、raw上位4は{raw4}/18を捕捉した。各開催の実際のD数を固定した無作為4試合選出の期待値は{18*4/13:.3f}で、最終方式の6件以上となる片側確率は{p_ge:.3f}。6件だけでは、偶然より優れた識別力を確認できない。

## 順位と識別力

ROC-AUCはD試合と非D試合のP(D)の順序比較、APはDを上位に並べる精度。Dが0件の群のAP、Dか非Dだけの群のAUCは未定義として欠損。開催回ごとのAUCは標本が小さい。混合モデル3系統を統合した結果に因果的な意味を与えない。

{markdown_table(met)}

## 上位4と偶然選出

{markdown_table(t4)}

無作為基準は各開催のD実数に条件付けた超幾何分布。開催間の捕捉数分布を畳み込んだ。予測確率の校正や試合間独立を仮定しない順位検証だが、試合を交換可能とみなす帰無基準であり、各開催の標本は13試合に限られる。全kの分布と両側の位置はCSVに保存。

## 捕捉D・漏れDの構造

最終P(D)の上位4で捕捉したDは6、漏れDは12。構造指標と0.1/0.2点の感応度は各群の平均・中央値・範囲と個別試合のCSVに記録した。これらは保存λから決定論的に計算され、新しい独立情報ではない。結果を見た後の群比較であり、D選考の新しい係数や規則は作らない。

{markdown_table(structure[structure.metric.isin(['lambda_total','lambda_abs_diff','P_0_0','P_1_1','P_draw_2plus','P_margin_1','P_margin_2plus'])])}

## P(D)近接ペア

raw P(D)差0.5ポイント以内は{int(pairs.within_0_005.sum())}組、1ポイント以内は{len(pairs)}組。1ポイント以内でDと非Dの組は{int(pairs.mixed_result.sum())}組。全ペアを列挙し、同一開催と開催横断を区別した。試合が複数ペアに現れるため独立標本ではない。近接ペアに構造差があっても、それが次試合のDを追加識別する証拠にはならない。

{markdown_table(close)}

## 13的中との関係

同一のH/D/A確率が正しく、試合間を独立と仮定すると、相異なる10券の13的中確率は各券の13試合確率積の和。固定確率の下では高確率の相異なる10組がこの和を最大化する。Dの捕捉件数や得点構造が過去78試合で良く見えても、それだけでは13的中確率の改善にならない。得点構造を配券に使う論拠には、予測確率の誤差や不確実性を未使用の後続試合で実証する必要がある。今回は係数学習、配券生成、Next Model変更を行っていない。

## 監査と成果物

同率は試合番号の昇順。行列由来D確率は0-0、1-1、2-2以上の和と一致。k=13は全D18件を捕捉。予測・結果・Phase 1/2/3原本のハッシュは実行前後で一致。保存された6開催の正式BuyPlanと予測スナップショットをハッシュで照合し、独立した追記ログ `data/forecast_registry/portfolio_links.jsonl` に生成時刻・版・plan hashを関連付けた。既存ファイルは変更していない。

[試合別順位](ranked_matches_78.csv) ／ [上位k](topk_capture.csv) ／ [無作為分布](hypergeometric_null.csv) ／ [識別・校正](discrimination.csv) ／ [実際のD18試合](actual_draws_18.csv) ／ [構造群比較](captured_missed_structure.csv) ／ [感応度](sensitivity_comparison.csv) ／ [近接ペア全件](near_probability_pairs.csv) ／ [監査](audit.json)
'''
    (OUT/'report.md').write_text(report)
    h=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 4-A 第2段階</title><style>body{{font:16px/1.65 system-ui;margin:28px;color:#203047}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{border:1px solid #ccd4df;padding:5px}}th{{background:#edf3f8}}.scroll{{overflow:auto}}section{{margin:24px 0}}a{{color:#245c94}}</style><h1>Phase 4-A 第2段階：Dの識別力と得点構造</h1><p><strong>設計上の位置付け：上位4の捕捉／漏れDは廃棄済みv3.0構想の履歴・比較対象。v4.0のD採用条件ではない。上位kは順位診断であり、k=4を既定値にしない。現行方針は <a href="../../../BUYPLAN_V4_0.md">BUYPLAN_V4_0.md</a>。</strong></p><p>試合前保存A：6開催78試合、実際のD18件。raw P(D)期待数 {d.raw_P_D.sum():.3f}。</p><section><h2>上位4のD捕捉</h2><div class="scroll">{table(t4,list(t4.columns))}</div><p>最終方式6件。開催ごとの実D数を固定した無作為選出の期待値5.538件、6件以上の確率{p_ge:.3f}。明確な識別優位とは言えない。</p></section><section><h2>順位別捕捉</h2><div class="scroll">{table(top[top['round']=='全6開催'],['stage','k','captured_D','recall','precision','random_expected','excess'])}</div></section><section><h2>識別・校正</h2><div class="scroll">{table(met,list(met.columns))}</div></section><section><h2>捕捉D・漏れDの構造</h2><div class="scroll">{table(structure[structure.metric.isin(['lambda_total','lambda_abs_diff','P_0_0','P_1_1','P_draw_2plus','P_margin_1','P_margin_2plus'])],list(structure.columns))}</div></section><section><h2>78試合の順位・結果</h2><div class="scroll">{table(d,maincols)}</div></section><section><h2>解釈</h2><p>近接ペア1ポイント以内{len(pairs)}組、D/非D混合{int(pairs.mixed_result.sum())}組。構造と感応度はλ由来で独立情報ではない。13的中改善には予測確率の不確実性を後続データで検証する必要がある。新スコア、配券、Next Modelの変更は行っていない。</p></section><p><a href="report.md">詳細レポート</a> ／ <a href="ranked_matches_78.csv">78試合</a> ／ <a href="topk_capture.csv">上位k</a> ／ <a href="hypergeometric_null.csv">無作為分布</a> ／ <a href="discrimination.csv">識別</a> ／ <a href="actual_draws_18.csv">D18試合</a> ／ <a href="captured_missed_structure.csv">構造</a> ／ <a href="sensitivity_comparison.csv">感応度</a> ／ <a href="near_probability_pairs.csv">近接ペア</a> ／ <a href="audit.json">監査</a></p></html>'''
    (OUT/'index.html').write_text(h)
    print(json.dumps(dict(raw4=raw4,final4=final4,expected=18*4/13,p_ge=p_ge,near005=int(pairs.within_0_005.sum()),near01=len(pairs),links_added=added),ensure_ascii=False))

if __name__=='__main__':main()
