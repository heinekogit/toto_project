"""Operational non-purchase v2.5 shadow. Frozen reference, no automatic refit."""
import hashlib
import json
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
from match_draw_coefficient_shadow import extract,coefficients
from score_placement_v25_shadow import Config,generate,diagnostics

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'data/models/shadow_v25_match_reference_v1.json'
FILES=('shadow_buyplan_v25.csv','shadow_buyplan_v25.html','shadow_v25_features.csv',
       'shadow_v25_candidates.csv','shadow_v25_audit.json','shadow_v25_reference.json')


def build(frame,official_plan,reference_path=REFERENCE):
    frame=frame.sort_values('match_no').reset_index(drop=True)
    official=official_plan.sort_values('match_no').reset_index(drop=True)
    for c in ['match_no','home_team','away_team']:
        if frame[c].tolist()!=official[c].tolist():raise ValueError('Official/shadow match order differs')
    raw=official[[f'ticket{i:02d}' for i in range(1,11)]].to_numpy()
    if not np.isin(raw,[0,1,2]).all():raise ValueError('Invalid official symbols')
    counts=np.stack([(raw==s).sum(axis=1) for s in [1,0,2]],axis=1)
    reference_text=Path(reference_path).read_text();ref=json.loads(reference_text)
    features=coefficients(extract(frame),ref)
    p=frame[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy()
    tickets,audit=generate(p,counts,Config(draw_coefficient=1.0),draw_coefficients=features.K_D.to_numpy())
    audit.update(status='non-purchase shadow',generator='v25_match_draw_score_shadow_v1',
                 coefficient_mode='match_structure',normalization_refit=False,
                 reference_sha256=hashlib.sha256(reference_text.encode()).hexdigest(),
                 feature_config=ref['config'],counts_source='actual official v2.4 tickets',
                 official_tickets_unchanged=True,output_files=list(FILES))
    plan=frame[['match_no','home_team','away_team','next_prob_home','next_prob_draw','next_prob_away']].copy()
    for j,t in enumerate(tickets):plan[f'ticket{j+1:02d}']=np.where(t==0,1,np.where(t==1,0,2))
    candidates=pd.DataFrame(diagnostics(p,tickets,audit))
    features=pd.concat([frame[['match_no','home_team','away_team']],features],axis=1)
    return plan,candidates,features,audit,reference_text


def write(bundle,outdir,round_id):
    plan,candidates,features,audit,reference=bundle;outdir=Path(outdir)
    plan.to_csv(outdir/FILES[0],index=False,encoding='utf-8-sig')
    features.to_csv(outdir/FILES[2],index=False,encoding='utf-8-sig')
    candidates.to_csv(outdir/FILES[3],index=False,encoding='utf-8-sig')
    (outdir/FILES[4]).write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    (outdir/FILES[5]).write_text(reference)
    view=plan[['match_no','home_team','away_team']].copy()
    view['候補00']=['HDA'[s] for s in audit['anchor']]
    for i in range(1,11):view[f'候補{i:02d}']=plan[f'ticket{i:02d}'].map({1:'H',0:'D',2:'A'})
    html=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>{escape(round_id)} v2.5試合別D係数シャドー</title>
<style>body{{font:16px/1.8 system-ui;margin:24px;color:#203147}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{border:1px solid #ccd3dc;padding:7px}}th{{background:#eef3f8}}</style>
<h1>{escape(round_id)} v2.5：試合別D係数シャドー</h1><p><b>全候補が購入対象外です。</b>00は最高予測確率の参照券、01〜10は比較用。正式版とは別の番号です。固定枚数は正式券から取得し、予測確率は変更していません。正規化基準は固定・自動再学習なし。</p>
<h2>候補00〜10</h2><div class="scroll">{view.to_html(index=False)}</div>
<h2>候補診断</h2><p>joint_probabilityは元確率による13的中確率。placement_valueは配券評価値。purchase欄は試作内の10券を示し、実購入の許可ではありません。</p><div class="scroll">{candidates.to_html(index=False)}</div>
<h2>試合別D評価</h2><div class="scroll">{features.to_html(index=False)}</div>
<p><a href="shadow_buyplan_v25.csv">配券CSV</a> ／ <a href="shadow_v25_audit.json">監査記録</a> ／ <a href="shadow_v25_reference.json">固定基準</a></p></html>'''
    (outdir/FILES[1]).write_text(html)
