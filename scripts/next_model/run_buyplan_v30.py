"""Replay the discontinued four-double experiment from frozen forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from html import escape
import pandas as pd
from buyplan_v30 import build

ROOT=Path(__file__).resolve().parents[2]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in',dest='source',type=Path,required=True)
    parser.add_argument('--round',required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--outdir',type=Path)
    parser.add_argument('--historical-replay',action='store_true',
                        help='Explicitly replay the discontinued four-double experiment')
    args=parser.parse_args(); source=args.source.resolve()
    if not args.historical_replay:
        parser.error('4ダブル・16口方式はv3.0候補として中止しました。履歴再現だけは --historical-replay を指定してください。')
    manifest=args.manifest
    if manifest is None:
        candidates=[source.parent/'buyplan_engine_manifest.json',source.parent/'manifest.json']
        manifest=next((p for p in candidates if p.exists() and 'output_sha256' in json.loads(p.read_text())),None)
    if manifest is None: raise ValueError('Saved forecast manifest required')
    m=json.loads(manifest.read_text())
    if m.get('round_id')!=args.round: raise ValueError('Requested round differs from source manifest')
    if m.get('output_sha256',{}).get(source.name)!=digest(source): raise ValueError('Forecast checksum mismatch')
    out=(args.outdir or ROOT/'data/reports/buyplan_v30_candidate'/args.round).resolve()
    if out==source.parent or out==ROOT/'data/purchase_reference': raise ValueError('Use a separate candidate directory')
    protected={str(p):digest(p) for p in (ROOT/'data/purchase_reference').glob('*') if p.is_file()}
    f=pd.read_csv(source)
    if 'toto_round' in f:
        rounds=f.toto_round.dropna().astype(str).str.replace(r'\.0$','',regex=True).str.removeprefix('toto')
        if len(rounds) and not rounds.eq(args.round.removeprefix('toto')).all(): raise ValueError('Forecast round mismatch')
    if 'datetime' not in f: raise ValueError('Kickoff times required for saved prediction provenance')
    kickoff=pd.to_datetime(f.datetime)
    if kickoff.dt.tz is None: kickoff=kickoff.dt.tz_localize('Asia/Tokyo')
    if not (kickoff>pd.Timestamp(m['created_at'])).all(): raise ValueError('Forecast must have been generated before every kickoff')
    plan,diag,audit=build(f)
    audit.update(status='discontinued_historical_replay',round_id=args.round,forecast=str(source),forecast_sha256=digest(source),
        forecast_manifest=str(manifest),forecast_created_at=m['created_at'],
        generated_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),prematch_source_verified=True)
    out.mkdir(parents=True,exist_ok=True)
    plan.to_csv(out/'buyplan_v30.csv',index=False,encoding='utf-8-sig')
    diag.to_csv(out/'candidates01_16.csv',index=False,encoding='utf-8-sig')
    plan[['match_no','home_team','away_team','candidate00','purchase_selection','D_rank','double_selected']].to_csv(out/'purchase_specification.csv',index=False,encoding='utf-8-sig')
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    view=plan[['match_no','home_team','away_team','candidate00','purchase_selection','next_prob_draw','D_rank']].copy()
    view.candidate00=view.candidate00.map({1:'H',0:'D',2:'A'});view.next_prob_draw=view.next_prob_draw.map(lambda x:f'{100*x:.2f}%')
    view.columns=['No.','ホーム','アウェー','候補00','購入指定','D予測確率','D順位']
    expanded=plan[['match_no','home_team','away_team']+[f'ticket{i:02}' for i in range(1,17)]].copy()
    for c in expanded.columns[3:]: expanded[c]=expanded[c].map({1:'H',0:'D',2:'A'})
    dv=diag.copy();dv.joint_probability=dv.joint_probability.map(lambda x:f'{100*x:.8f}%')
    html=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>{escape(args.round)} 旧4ダブル方式・履歴再現</title>
<style>body{{font:16px/1.8 system-ui;margin:28px;color:#203147}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px}}td,th{{padding:7px;border:1px solid #ccd3dc}}th{{background:#eef3f8}}</style>
<h1>{escape(args.round)} 旧4ダブル方式・履歴再現</h1>
<p><strong>この方式はv3.0候補として中止済みです。購入候補ではありません。</strong> H/Aの高確率側を候補00とし、P(D)上位4試合にDを追加した旧実験（4ダブル＝16口・1,600円）の再現結果です。</p>
<p>候補00は参照表示ですが、その内容は展開券01として16口に含まれます。別途17口目を購入するものではありません。候補番号は展開順であり、強さの順位ではありません。</p>
<p>16口のうち少なくとも1口が13的中するモデル確率：<b>{audit['model_P13']*100:.6f}%</b>。試合間独立を仮定した保存予測上の値で、実際の的中率ではありません。</p>
<p>選出4試合はH/DまたはA/D、残り9試合は固定です。固定試合を1つ外すと全16口が13的中を失います。ダブル試合でも未採用側のH/Aが勝てば全16口が13的中を失います。</p>
<h2>購入指定と候補00</h2>{view.to_html(index=False)}
<h2>展開後16口</h2><div class="scroll">{expanded.to_html(index=False)}</div>
<h2>各券のモデル確率</h2>{dv.to_html(index=False)}
<p>予測生成：{escape(m['created_at'])}。保存予測のハッシュ一致と試合前生成を確認。実結果は使用していません。</p>
<p><a href="buyplan_v30.csv">全配置CSV</a> ／ <a href="purchase_specification.csv">購入指定CSV</a> ／ <a href="audit.json">設定・監査</a></p></html>'''
    (out/'buyplan_v30.html').write_text(html)
    assert all(digest(Path(p))==h for p,h in protected.items())
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(out/'buyplan_v30.html')

if __name__=='__main__':main()
