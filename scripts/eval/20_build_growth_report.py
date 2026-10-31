#!/usr/bin/env python3
"""Evaluate frozen purchases, store versioned evidence, and build readable HTML."""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from jleague_team_names import canonical_team_name

STYLE = '''body{font-family:system-ui,sans-serif;margin:0;background:#f4f6fa;color:#172033}
main{max-width:1150px;margin:auto;padding:28px}h1{font-size:26px}h2{margin-top:30px;font-size:20px}
.card{background:white;border:1px solid #dce2eb;border-radius:10px;padding:18px;margin:14px 0;overflow:auto}
.note{background:#fff6dc;padding:14px;border-left:4px solid #d69c27}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:9px;border-bottom:1px solid #e0e5ee;text-align:left;white-space:nowrap}th{background:#edf2f8}
a{color:#175ea8}.muted{color:#53657a}'''


def page(title, body):
    return f'<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{STYLE}</style><main><h1>{escape(title)}</h1>{body}</main></html>'


def table(rows):
    if not rows:
        return '<p class="muted">評価できるデータがありません。</p>'
    return pd.DataFrame(rows).to_html(index=False, escape=True, border=0)


def symbol(value):
    if pd.isna(value): return None
    if str(value).strip().upper() in {'H','D','A'}: return str(value).strip().upper()
    try:
        num = float(value)
        return {1:'H',0:'D',2:'A'}.get(num)
    except (ValueError,TypeError): return None


def ticket_columns(frame):
    return sorted([c for c in frame if re.fullmatch(r'ticket\d+', str(c))],key=lambda c:int(c[6:]))


def probabilities(row):
    for keys in [('next_prob_home','next_prob_draw','next_prob_away'),('p_home','p_draw','p_away')]:
        if all(k in row for k in keys):
            values=pd.to_numeric(pd.Series([row[k] for k in keys]),errors='coerce').to_numpy(float)
            if np.isfinite(values).all() and (values>=0).all() and (values<=1).all() and abs(values.sum()-1)<1e-5:
                return values.tolist()
    return None


def export_records(records):
    rows=[]
    for record in records:
        row={k:v for k,v in record.items() if k not in {'probabilities','picks'}}
        for index,symbol in enumerate('HDA'):
            row[f'p_{symbol}']=record['probabilities'][index] if record['probabilities'] else None
        for index,value in enumerate(record['picks'],1):row[f'ticket{index:02}']=value
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(round_id, round_dir):
    snap=round_dir/'snapshot'
    paths={'buyplan':snap/'buyplan.csv','actual':round_dir/'actual_results.csv'}
    if any(not p.exists() for p in paths.values()): raise ValueError('購入snapshotまたは実結果がありません')
    buy=pd.read_csv(paths['buyplan']);actual=pd.read_csv(paths['actual'])
    for label,frame in [('buyplan',buy),('actual',actual)]:
        if 'match_no' not in frame or frame.match_no.isna().any() or frame.match_no.duplicated().any():
            raise ValueError(f'{label}: 試合番号が欠損・重複しています')
    if len(buy)!=13 or set(buy.match_no)!=set(range(1,14)): raise ValueError('13試合の購入案が必要です')
    if not set(actual.match_no).issubset(set(buy.match_no)): raise ValueError('実結果に別試合番号が含まれています')
    buy=buy.sort_values('match_no').reset_index(drop=True)
    columns=ticket_columns(buy)
    if len(columns)!=10: raise ValueError('正式10口が必要です')
    manifest_path=snap/'buyplan_engine_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    strategy=manifest.get('generator',{}).get('generator','unknown')
    version={'coverage_v1':'v2.0','probability_allocation_v1':'v2.1','probability_allocation_ge10_v1':'v2.2','relative_draw_graded_v23':'v2.3','random_remaining_v24':'v2.4'}.get(strategy,'版不明（旧保存形式）')
    if manifest_path.exists():paths['engine_manifest']=manifest_path
    matched=buy.merge(actual,on='match_no',how='left',suffixes=('','__actual'),validate='one_to_one')
    records=[]
    for _,row in matched.iterrows():
        for side in ['home_team','away_team']:
            value=row.get(side+'__actual')
            if pd.notna(value) and canonical_team_name(row[side])!=canonical_team_name(value):
                raise ValueError(f'実結果カード不一致: {int(row.match_no)}')
        result=symbol(row.get('result__actual',row.get('result')))
        status=row.get('status__actual',row.get('status'))
        if pd.notna(status) and status!='OK':result=None
        picks=[symbol(row[c]) for c in columns]
        if None in picks: raise ValueError(f'正式券に無効な値: {int(row.match_no)}')
        p=probabilities(row)
        records.append(dict(round_id=round_id,version=version,match_no=int(row.match_no),
                            home_team=str(row.home_team),away_team=str(row.away_team),result=result,
                            probabilities=p,picks=picks))
    outcomes=[r['result'] for r in records]
    resolved=sum(v is not None for v in outcomes)
    def score(frame, cols):
        if not cols:return None
        frame=frame.sort_values('match_no').reset_index(drop=True)
        if frame.match_no.tolist()!=buy.match_no.tolist():raise ValueError('シャドー試合番号不一致')
        for side in ['home_team','away_team']:
            if not frame[side].map(canonical_team_name).equals(buy[side].map(canonical_team_name)):
                raise ValueError('シャドーカード不一致')
        tickets=[[symbol(v) for v in frame[c]] for c in cols]
        if any(None in t for t in tickets):raise ValueError('券に無効な結果が含まれます')
        hits=[sum(y is not None and x==y for x,y in zip(t,outcomes)) for t in tickets]
        return dict(ticket_count=len(cols),ticket_names=cols,ticket_hits=hits,best_hits=max(hits),
                    mean_hits=float(np.mean(hits)),accuracy=float(np.mean(hits)/resolved) if resolved else None,
                    draw_cells=sum(t.count('D') for t in tickets),
                    ge10=sum(h>=10 for h in hits),ge11=sum(h>=11 for h in hits),ge12=sum(h>=12 for h in hits),ge13=sum(h==13 for h in hits),
                    draw_pair_coverage=sum(all(y!='D' or x=='D' for x,y in zip(t,outcomes)) for t in tickets) if resolved==13 and 'D' in outcomes else None)
    official=score(buy,columns)
    shadow=None;shadow_version='未保存'
    shadow_path=snap/'shadow_buyplan.csv'
    if shadow_path.exists():
        sf=pd.read_csv(shadow_path);cols=ticket_columns(sf)
        if cols:
            shadow=score(sf,cols);paths['shadow']=shadow_path
            sv=manifest.get('shadow_observation',{}).get('generator')
            shadow_version={'coverage_v1':'v2.0','probability_allocation_v1':'確率配分方式','probability_allocation_ge10_v1':'A：v2.2'}.get(sv,'方式不明')
    shadow_b=None;shadow_b_version='未保存'
    shadow_b_path=snap/'shadow_buyplan_b.csv'
    if shadow_b_path.exists():
        sf=pd.read_csv(shadow_b_path);cols=ticket_columns(sf)
        if cols:
            shadow_b=score(sf,cols);paths['shadow_b']=shadow_b_path
            sv=manifest.get('shadow_b_observation',{}).get('generator')
            shadow_b_version={'anchor_graded_shadow':'B：Anchor段階型'}.get(sv,'方式不明')
    shadow_c=None;shadow_c_version='未保存'
    shadow_c_path=snap/'shadow_buyplan_c.csv'
    if shadow_c_path.exists():
        sf=pd.read_csv(shadow_c_path);cols=ticket_columns(sf)
        if cols:
            shadow_c=score(sf,cols);paths['shadow_c']=shadow_c_path
            sv=manifest.get('shadow_c_observation',{}).get('generator')
            shadow_c_version={'free_coverage_shadow_c_v1':'C：採用数自由・Dカバー型 v1',
                              'free_coverage_shadow_c_v2':'C：採用数自由・Dカバー型 v2',
                              'free_coverage_shadow_c_v3':'C：採用数自由・Dカバー型 v3',
                              'free_coverage_shadow_c_v4':'C：採用数自由・Dカバー型 v4'}.get(sv,'方式不明')
    valid=[r for r in records if r['result'] is not None and r['probabilities'] is not None]
    model=None
    if valid:
        p=np.array([r['probabilities'] for r in valid]);y=np.array(['HDA'.index(r['result']) for r in valid]);top=p.argmax(axis=1)
        model=dict(matches=len(valid),top_hits=int((top==y).sum()),top_accuracy=float((top==y).mean()),
                   brier=float(((p-np.eye(3)[y])**2).sum(axis=1).mean()),
                   outcome_rows=[{'結果':s,'平均予測確率':f'{p[:,i].mean():.1%}','実際の割合':f'{(y==i).mean():.1%}','実数':int((y==i).sum())} for i,s in enumerate('HDA')])
    provenance={k:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
    return dict(round_id=round_id,version=version,strategy=strategy,resolved=resolved,pending=13-resolved,
                model_version=manifest.get('prediction_model','base_lambda_v1 + lab_xg_v0' if strategy!='unknown' else '不明'),
                actual_counts={s:outcomes.count(s) for s in 'HDA'},official=official,shadow=shadow,
                shadow_version=shadow_version,shadow_b=shadow_b,shadow_b_version=shadow_b_version,
                shadow_c=shadow_c,shadow_c_version=shadow_c_version,
                model=model,records=records,sources=provenance,
                created_at=datetime.now().astimezone().isoformat(),
                comparison_fair=shadow is not None and shadow['ticket_count']==10 and (shadow_b is None or shadow_b['ticket_count']==10) and (shadow_c is None or shadow_c['ticket_count']==10))


def render_round(report):
    r=report;resolved=r['resolved']
    body=f'<p><a href="../../growth_validation/index.html">累積検証一覧へ</a> · <a href="buyplan_scored.html">従来の採点表</a></p>'
    body+=f'<p>保存時の正式版：<b>{escape(r["version"])}</b> ／ 確定 {resolved}/13試合 ／ 未確定 {r["pending"]}試合</p>'
    body+='<p class="note">予測の精度と券配置の効果を分けて評価します。平均的中率は当せん確率・収益率ではありません。単一開催だけで優劣は確定しません。</p>'
    if r['pending']:body+='<p class="note">途中結果です。未確定試合を外れに数えず、確定試合のみで集計しています。</p>'
    body+='<h2>1. 購入案の成績</h2><div class="card">'
    rows=[]
    for label,score in [(f'正式 {r["version"]}',r['official']),(f'シャドー {r["shadow_version"]}',r['shadow']),
                        (f'シャドー {r.get("shadow_b_version", "未保存")}',r.get('shadow_b')),
                        (f'シャドー {r.get("shadow_c_version", "未保存")}',r.get('shadow_c'))]:
        if score:
            rows.append({'対象':label,'口数':score['ticket_count'],'最高的中数':score['best_hits'],
                         '平均的中数':round(score['mean_hits'],2),'平均的中率':f'{score["accuracy"]:.1%}' if score['accuracy'] is not None else '未確定',
                         '10以上の口':score['ge10'],'11以上':score['ge11'],'12以上':score['ge12'],'13的中':score['ge13'],'D採用枠':score['draw_cells']})
    body+=table(rows)+'</div>'
    if r['shadow'] and not r['comparison_fair']:body+='<p class="note">シャドーの口数が異なります。最高的中数を同条件の比較として扱いません。</p>'
    body+='<div class="card">'+table([{'口':name,'的中数':hits} for name,hits in zip(r['official']['ticket_names'],r['official']['ticket_hits'])])+'</div>'
    if r['shadow']:
        body+='<h3>シャドー各口</h3><div class="card">'+table([{'口':name,'的中数':hits} for name,hits in zip(r['shadow']['ticket_names'],r['shadow']['ticket_hits'])])+'</div>'
    if r.get('shadow_b'):
        body+='<h3>シャドーB各口</h3><div class="card">'+table([{'口':name,'的中数':hits} for name,hits in zip(r['shadow_b']['ticket_names'],r['shadow_b']['ticket_hits'])])+'</div>'
    if r.get('shadow_c'):
        body+='<h3>シャドーC各口</h3><div class="card">'+table([{'口':name,'的中数':hits} for name,hits in zip(r['shadow_c']['ticket_names'],r['shadow_c']['ticket_hits'])])+'</div>'
    body+='<h2>2. 予測確率そのものの精度</h2>'
    if r['model']:
        m=r['model'];body+=f'<p>確率を評価できる {m["matches"]}試合中、最有力結果は {m["top_hits"]}的中（{m["top_accuracy"]:.1%}）。確率の誤差（Brier）は {m["brier"]:.3f}。小さいほど良く、一律1/3なら0.667です。</p><div class="card">'+table(m['outcome_rows'])+'</div>'
    else:body+='<p>保存された購入案に評価可能な確率列がありません。確率の評価は保留します。</p>'
    body+='<h2>3. 試合ごとの配分と結果</h2><div class="card">'
    rows=[]
    for item in r['records']:
        probs=item['probabilities'];result=item['result']
        rows.append({'No.':item['match_no'],'対戦':item['home_team']+'－'+item['away_team'],'実結果':result or '未確定',
                     '確率 H/D/A':' / '.join(f'{p:.0%}' for p in probs) if probs else '不明',
                     '採用 H/D/A':' / '.join(str(item['picks'].count(s)) for s in 'HDA'),
                     '的中した口数':item['picks'].count(result) if result else '未確定'})
    body+=table(rows)+'</div><h2>4. 今回の読み方</h2>'
    body+=f'<p>実結果は H{r["actual_counts"]["H"]}・D{r["actual_counts"]["D"]}・A{r["actual_counts"]["A"]}。正式版の最高は{r["official"]["best_hits"]}的中です。</p>'
    dcover=r['official']['draw_pair_coverage']
    if dcover is not None:body+=f'<p>実際にDだった全試合を同時にDとして持つ正式券は{dcover}口でした。これは配置の検証材料であり、結果を見て券を作り直す指示ではありません。</p>'
    body+='<p>モデルやBuyPlanを自動変更しません。CSV・JSONに保存した試合前確率、券、実結果、版、元ファイルのハッシュを次の分析に使います。</p>'
    body+='<p><a href="growth_validation.json">この開催の検証JSON</a> · <a href="growth_matches.csv">再分析用CSV</a></p>'
    return page(r['round_id']+' 開催別検証',body)


def render_index(history):
    reports=[json.loads(p.read_text()) for p in sorted(history.glob('toto*.json'))]
    body='<p class="note">保存時の版で評価します。異なるモデル・BuyPlanの版をまとめて優劣を判断しません。再実行は同じ開催を更新し、二重計上しません。</p>'
    body+='<h2>開催別の結果</h2><div class="card"><table><tr><th>開催</th><th>正式版</th><th>確定試合</th><th>最高的中</th><th>平均的中率</th><th>シャドーA版／口数</th><th>A最高</th><th>シャドーB版／口数</th><th>B最高</th><th>シャドーC版／口数</th><th>C最高</th></tr>'
    for r in reversed(reports):
        s=r['shadow'];b=r.get('shadow_b');c=r.get('shadow_c');o=r['official'];rate=f'{o["accuracy"]:.1%}' if o['accuracy'] is not None else '未確定'
        body+=f'<tr><td><a href="../toto_rounds/{escape(r["round_id"])}/growth_validation.html">{escape(r["round_id"])}</a></td><td>{escape(r["version"])}</td><td>{r["resolved"]}/13</td><td>{o["best_hits"]}</td><td>{rate}</td><td>{escape(r["shadow_version"])}／{s["ticket_count"] if s else "—"}</td><td>{s["best_hits"] if s else "—"}</td><td>{escape(r.get("shadow_b_version", "未保存"))}／{b["ticket_count"] if b else "—"}</td><td>{b["best_hits"] if b else "—"}</td><td>{escape(r.get("shadow_c_version", "未保存"))}／{c["ticket_count"] if c else "—"}</td><td>{c["best_hits"] if c else "—"}</td></tr>'
    body+='</table></div><h2>正式版ごとの累積成績</h2><div class="card">'
    summaries=[]
    for version in sorted(set(r['version'] for r in reports)):
        if version.startswith('版不明'):continue
        population=[r for r in reports if r['version']==version and r['resolved']==13]
        if not population:continue
        summaries.append({'版':version,'完了開催数':len(population),'最高的中数の平均':round(np.mean([r['official']['best_hits'] for r in population]),2),
                          '全口の平均的中率':f'{np.mean([r["official"]["accuracy"] for r in population]):.1%}',
                          '10以上の口':sum(r['official']['ge10'] for r in population)})
    body+=table(summaries)+'</div><h2>予測確率の累積検証</h2>'
    for version in sorted(set(r.get('model_version','不明') for r in reports)):
        if version=='不明':continue
        records=[x for r in reports if r.get('model_version','不明')==version for x in r['records'] if x['result'] and x['probabilities']]
        if not records:continue
        rows=[]
        for low,high in [(0,.4),(.4,.5),(.5,.6),(.6,1.00001)]:
            subset=[x for x in records if low<=max(x['probabilities'])<high]
            if subset:rows.append({'最有力確率帯':f'{low:.0%}～{min(high,1):.0%}','試合数':len(subset),'平均予測確率':f'{np.mean([max(x["probabilities"]) for x in subset]):.1%}',
                                   '実際の的中率':f'{np.mean(["HDA"[int(np.argmax(x["probabilities"]))]==x["result"] for x in subset]):.1%}'})
        body+=f'<h3>{escape(version)}（{len(records)}試合）</h3><div class="card">'+table(rows)+'</div>'
        rate_rows=[]
        for i,s in enumerate('HDA'):
            rate_rows.append({'結果':s,'平均予測確率':f'{np.mean([x["probabilities"][i] for x in records]):.1%}',
                              '実際の割合':f'{np.mean([x["result"]==s for x in records]):.1%}'})
        body+='<div class="card">'+table(rate_rows)+'</div>'
    body+='<p>予測モデルの集計は同じモデルの保存確率だけを使用します。版が不明な旧保存形式は開催別の参考履歴に残し、版別累積・確率評価へ混ぜません。</p><h2>継続して確認する課題</h2><p>高い確率ほど当たるか、Dの予測確率と発生率が対応するか、券配置が最高的中数へ寄与するかを別々に確認します。少数開催では判断保留です。履歴の再集計は改善の証明や自動学習ではありません。</p>'
    body+='<p>開催別JSON：records/ ／ 再分析用CSV：matches.csv ／ 集計：rounds.csv</p>'
    return page('toto 成長検証・累積一覧',body),reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',required=True)
    parser.add_argument('--eval-root',type=Path,default=ROOT/'data/eval')
    parser.add_argument('--rebuild-all',action='store_true',help='保存済みの過去開催も再集計')
    args=parser.parse_args()
    if not re.fullmatch(r'toto\d+',args.round):raise ValueError('toto開催IDを指定してください')
    directory=args.eval_root/'toto_rounds'/args.round
    report=analyze(args.round,directory)
    history=args.eval_root/'growth_validation';records=history/'records';records.mkdir(parents=True,exist_ok=True)
    if args.rebuild_all:
        for past in sorted((args.eval_root/'toto_rounds').glob('toto*')):
            if not (past/'actual_results.csv').exists() or past.name==args.round:continue
            try:
                previous=analyze(past.name,past)
            except (ValueError,KeyError,FileNotFoundError) as error:
                print(f'[WARN] 過去開催の検証を保留: {past.name}: {error}')
                continue
            text=json.dumps(previous,ensure_ascii=False,indent=2,allow_nan=False)
            (records/f'{past.name}.json').write_text(text,encoding='utf-8')
            (past/'growth_validation.json').write_text(text,encoding='utf-8')
            (past/'growth_validation.html').write_text(render_round(previous),encoding='utf-8')
            export_records(previous['records']).to_csv(past/'growth_matches.csv',index=False,encoding='utf-8-sig')
    encoded=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    (records/f'{args.round}.json').write_text(encoded,encoding='utf-8')
    (directory/'growth_validation.json').write_text(encoded,encoding='utf-8')
    export_records(report['records']).to_csv(directory/'growth_matches.csv',index=False,encoding='utf-8-sig')
    (directory/'growth_validation.html').write_text(render_round(report),encoding='utf-8')
    html,reports=render_index(records)
    (history/'index.html').write_text(html,encoding='utf-8')
    export_records([x for r in reports for x in r['records']]).to_csv(history/'matches.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'round_id':r['round_id'],'version':r['version'],'resolved':r['resolved'],**{k:v for k,v in r['official'].items() if not isinstance(v,list)}} for r in reports]).to_csv(history/'rounds.csv',index=False,encoding='utf-8-sig')
    print(f'[OK] 開催別検証: {directory / "growth_validation.html"}')
    print(f'[OK] 累積検証: {history / "index.html"}')


if __name__=='__main__':main()
