"""Operational v2.5 C95 shadow; official v2.4 remains unchanged."""
import json
from html import escape
import numpy as np
import pandas as pd
from shadow_v25_legacy import ROOT,REFERENCE,FILES,build as build_legacy
from fixed_exact13_shadow import optimize_b,mass,validate,coverage
from draw_count_cover_shadow import optimize,draw_distribution,lookup_table,signature,better
from audit_draw_portfolio import ExactPortfolio


def build(frame,official_plan,reference_path=REFERENCE):
    # Preserve the previous study's initial layout and B search exactly.
    legacy=build_legacy(frame,official_plan,reference_path)
    frame=frame.sort_values('match_no').reset_index(drop=True)
    p=frame[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy()
    counts=np.array(legacy[3]['counts']);raw=legacy[0][[f'ticket{i:02}' for i in range(1,11)]].to_numpy().T
    initial=np.where(raw==1,0,np.where(raw==0,1,2))
    b,b_audit=optimize_b(p,initial,starts=24,cap=160,seed=2501670)
    bp=float(mass(p,b).sum());dist=draw_distribution(p);lookup=lookup_table(dist)
    best=b.copy();bc=float(lookup[signature(b)]);bestp=bp;runs=[]
    for seed in [951670,1951670,2951670]:
        t,audit,_=optimize(p,b,.95,starts=32,cap=240,seed=seed)
        c=float(lookup[signature(t)]);v=float(mass(p,t).sum());runs.append(audit)
        if better(c,v,bc,bestp):best=t.copy();bc=c;bestp=v
    best=best[np.argsort(-mass(p,best),kind='stable')];validate(best,counts)
    if bestp<.95*bp-1e-16:raise AssertionError('C95 exact13 floor violated')
    evaluator=ExactPortfolio(p);met=evaluator.metrics(best)
    for k in [11,12]:np.testing.assert_allclose(met[f'p{k}'],coverage(evaluator.mass,best,13-k),rtol=1e-12)
    anchor=p.argmax(1);pm=mass(p,best)
    diag=[]
    for j,t in enumerate(np.vstack([anchor,best])):
        prob=float(np.prod(p[np.arange(13),t]));diag.append(dict(candidate=f'{j:02}',purchase=j>0,reference_only=j==0,symbols=''.join('HDA'[s] for s in t),D_count=int((t==1).sum()),exchanges=int((t!=anchor).sum()),joint_probability=prob,contribution=None if j==0 else prob/pm.sum()))
    plan=frame[['match_no','home_team','away_team','next_prob_home','next_prob_draw','next_prob_away']].copy()
    for j,t in enumerate(best,1):plan[f'ticket{j:02}']=np.where(t==0,1,np.where(t==1,0,2))
    features=frame[['match_no','home_team','away_team','next_prob_home','next_prob_draw','next_prob_away']].copy()
    for s,label in enumerate('HDA'):features[f'fixed_count_{label}']=counts[:,s]
    config=dict(mode='C95',retention=.95,B_starts=24,B_cap=160,B_seed=2501670,C_starts_per_seed=32,C_cap=240,C_seeds=[951670,1951670,2951670],objective='maximize D-count probability coverage; tie-break exact13',legacy_reference=json.loads(legacy[4]),legacy_reference_role='initialization only; not the C95 objective')
    audit=dict(status='non-purchase shadow',generator='v25_c95_draw_count_cover_v1',mode='C95',anchor=anchor.tolist(),counts=counts.tolist(),counts_source='actual official v2.4 tickets',D_distribution=dist.tolist(),expected_D=float(p[:,1].sum()),covered_D_counts=sorted(set(map(int,(best==1).sum(1)))),D_count_coverage=bc,baseline_B_p13=bp,p13_floor=.95*bp,p13=float(pm.sum()),p12=met['p12'],p11=met['p11'],retention=float(pm.sum()/bp),B_audit=b_audit,C_seed_audits=runs,config=config,duplicate_pairs=[],counts_preserved=True,probabilities_changed=False,official_tickets_unchanged=True,candidate00_consumes_counts=False,global_optimum_guaranteed=False,output_files=list(FILES))
    return plan,pd.DataFrame(diag),features,audit,json.dumps(config,ensure_ascii=False,indent=2)


def write(bundle,outdir,round_id):
    from pathlib import Path
    plan,candidates,features,audit,reference=bundle;outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    for data,name in [(plan,FILES[0]),(features,FILES[2]),(candidates,FILES[3])]:data.to_csv(outdir/name,index=False,encoding='utf-8-sig')
    (outdir/FILES[4]).write_text(json.dumps(audit,ensure_ascii=False,indent=2));(outdir/FILES[5]).write_text(reference)
    view=plan[['match_no','home_team','away_team']].copy();view['候補00']=['HDA'[s] for s in audit['anchor']]
    for i in range(1,11):view[f'候補{i:02}']=plan[f'ticket{i:02}'].map({1:'H',0:'D',2:'A'})
    diag=candidates.copy();diag.joint_probability=diag.joint_probability.map(lambda z:f'{100*z:.8f}%');diag.contribution=diag.contribution.map(lambda z:'—' if pd.isna(z) else f'{100*z:.4f}%')
    dist=pd.DataFrame({'D発生数':range(14),'モデル確率':[f'{v*100:.6f}%' for v in audit['D_distribution']],'対応券あり':[k in audit['covered_D_counts'] for k in range(14)]})
    html=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>{escape(round_id)} v2.5 C95シャドー</title>
<style>body{{font:16px/1.8 system-ui;margin:24px;color:#203147}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{border:1px solid #ccd3dc;padding:7px}}th{{background:#eef3f8}}</style>
<h1>{escape(round_id)} v2.5：C95シャドー</h1><p>00は購入対象外の参照券。01〜10は比較用10口です。正式版v2.4の購入候補とは別のシャドー出力で、同じ予測確率と試合別H/D/A枚数を使います。</p>
<p>13的中確率を同一入力のB基準の95%以上に保ち、D発生数カバー率を最大化します。同率なら13的中確率を優先。Dカバー率は本数の一致だけを表し、13的中確率ではありません。</p>
<p>13的中：<b>{audit['p13']*100:.6f}%</b> ／ B比：{audit['retention']*100:.4f}% ／ 12以上：{audit['p12']*100:.6f}% ／ 11以上：{audit['p11']*100:.6f}% ／ Dカバー率：<b>{audit['D_count_coverage']*100:.4f}%</b></p>
<p>全て試合間独立を仮定したモデル値。D10枚・13枚など低確率券が残る場合があります。局所探索で大域最適は保証しません。</p>
<h2>候補00〜10</h2><div class="scroll">{view.to_html(index=False)}</div>
<h2>候補別確率・寄与率</h2><div class="scroll">{diag.to_html(index=False)}</div>
<h2>試合別固定枚数</h2><div class="scroll">{features.to_html(index=False)}</div>
<h2>D発生数分布とカバー範囲</h2>{dist.to_html(index=False)}
<p><a href="shadow_buyplan_v25.csv">配券CSV</a> ／ <a href="shadow_v25_audit.json">監査記録</a> ／ <a href="shadow_v25_reference.json">C95設定</a></p></html>'''
    (outdir/FILES[1]).write_text(html)
