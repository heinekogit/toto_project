"""Match archived summary variants to published team overview; no production adoption."""
import json,re,sys
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import pandas as pd
from bs4 import BeautifulSoup
from common import ROOT,normalize_text,sha256
sys.path.insert(0,str(ROOT/'scripts'))
from build_football_lab_snapshot import parse_js_array_rows
OUT=ROOT/'data/reports/style_value_investigation'
raw=pd.read_csv(ROOT/'data/reports/full_style_replay_audit/row_variants.csv')
raw=raw[raw.league.eq('j1')]
summary=BeautifulSoup((ROOT/'external_contents/20261103/j1/ハイブロック.html').read_text(),'html.parser')
names={urlsplit(a['href']).path.strip('/'):normalize_text(a.select_one('.sp').text) for a in summary.select('td.tName a')}
codes={'45':('ラインブレイクラン','味方シュート率'),'66':('ハイプレッシング','守備成功率'),'67':('ミドルプレッシング','守備成功率'),'69':('カウンタープレス','守備成功率'),'81':('ハイブロック','攻撃獲得率'),'82':('ミドルブロック','攻撃獲得率'),'83':('ローブロック','攻撃獲得率')}
rows=[];selected=[];annual=[]
for source in json.loads((OUT/'sources/team_index.json').read_text()):
 p=OUT/'sources'/source['file'];text=p.read_text();s=BeautifulSoup(text,'html.parser');team=names[source['slug']]
 menu={}
 for a in s.select('a[href]'):
  code=parse_qs(urlsplit(a['href']).query).get('s',[''])[0]
  if code not in codes or not a.select_one('.graphPie'):continue
  name,rate=codes[code];idx=float(a.select_one('.numL').text);pct=float(a.select_one('.graphPie').text.strip('%'))
  menu[code]=(idx,pct)
  group=raw[raw.team.eq(team)&raw['style'].eq(name)]
  matches=[]
  for _,r in group.iterrows():
   values=dict(zip(json.loads(r.headers),json.loads(r['values'])))
   if float(values['指数'])==idx and float(values[rate].strip('%'))==pct:matches.append(values)
  rows.append(dict(team=team,style=name,overview_index=idx,overview_rate=pct,source_variants=len(group),matching_variants=len(matches),url=source['url'],source_sha256=sha256(p)))
  if len(matches)==1:
   selected.append(dict(team=team,style=name,status='overview_aligned_candidate_not_adopted',**matches[0]))
 assert len(menu)==7
 for m in re.finditer(r'arrayToDataTable\(\[(.*?)\]\);',text,re.S):
  table=parse_js_array_rows(m[1]);current=[r for r in table if r and r[0]=='2026/27']
  if current:
   assert len(current[0])==3
   annual.append(dict(team=team,records=len(current),first_index=current[0][1],first_rate=current[0][2],last_index=current[-1][1],last_rate=current[-1][2],overview_index=menu['81'][0],overview_rate=menu['81'][1],overview_matches_last=(int(current[-1][1]+.5)==menu['81'][0] and current[-1][2]==menu['81'][1]),sequence=json.dumps(current,ensure_ascii=False)))
pd.DataFrame(rows).to_csv(OUT/'overview_alignment.csv',index=False)
pd.DataFrame(selected).to_csv(OUT/'candidate_values.csv',index=False)
pd.DataFrame(annual).to_csv(OUT/'annual_sequences.csv',index=False)
print('overview matches',pd.Series([r['matching_variants'] for r in rows]).value_counts().to_dict())
print('annual last matches',sum(r['overview_matches_last'] for r in annual),'/',len(annual))
print('unresolved',[(r['team'],r['style'],r['matching_variants']) for r in rows if r['matching_variants']!=1])
# Resolve rounded-index collisions using the final same-season chart record.
from build_football_lab_snapshot import extract_team_tables, extract_chart_rows
resolved=[]
for row in rows:
 if row['matching_variants']==1:continue
 team=row['team'];code=next(c for c,(n,_) in codes.items() if n==row['style'])
 slug=next(k for k,v in names.items() if v==team)
 text=(OUT/'sources'/f'{slug}_{code}.html').read_text()
 sequences=[]
 for m in re.finditer(r'arrayToDataTable\(\[(.*?)\]\);',text,re.S):
  table=parse_js_array_rows(m[1]);current=[r for r in table if r and r[0]=='2026/27']
  if current:sequences.append(current)
 assert len(sequences)==1
 last=sequences[0][-1]
 assert int(last[1]+.5)==row['overview_index'] and last[2]==row['overview_rate']
 original=(ROOT/'external_contents/20261103/j1'/f'{row["style"]}.html').read_text()
 meta=dict(page_label=row['style']+' (チームスタイル指標)')
 table=[r for r in extract_team_tables(original,meta) if r['section_title']==row['style']+'のサマリー']
 chart=extract_chart_rows(original,meta)
 assert len(table)==len(chart)==100
 matches={}
 for tr,cr in zip(table,chart):
  assert tr['team_key']==cr['team_key']
  v=dict(zip(json.loads(tr['headers_json']),json.loads(tr['values_json'])))
  assert abs(float(v['指数'])-float(cr['metric_x_value'])) <= .500001
  assert float(v[codes[code][1]].strip('%')) == float(cr['metric_y_value'])
  if normalize_text(tr['team_key'])==team and float(cr['metric_x_value'])==last[1] and float(cr['metric_y_value'])==last[2]:
   matches[tr['values_json']]=v
 resolved.append(dict(team=team,style=row['style'],precise_index=last[1],overview_rate=last[2],matching_variants=len(matches)))
 if len(matches)==1:selected.append(dict(team=team,style=row['style'],status='overview_and_final_chart_aligned_candidate_not_adopted',**next(iter(matches.values()))))
pd.DataFrame(selected).to_csv(OUT/'candidate_values.csv',index=False)
pd.DataFrame(resolved).to_csv(OUT/'precision_resolution.csv',index=False)
print('precision resolution',resolved)
print('final unique candidate rows',len(selected))
result={'overview_pairs':len(rows),'unique_full_row_candidates':len(selected),'unresolved_full_rows':[r for r in resolved if r['matching_variants']!=1],'high_block_overview_matches_final_chart':sum(r['overview_matches_last'] for r in annual),'historical_asof_resolved':False,'production_adopted':False}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
example=pd.DataFrame([{'表示箇所':'横浜FM 上部一覧','指数':'55（グラフ54.6）','攻撃獲得率':'73.2%'},{'表示箇所':'横浜FM 詳細欄・シーズン先頭','指数':'グラフ49.8','攻撃獲得率':'72.3%'},{'表示箇所':'川崎F 上部一覧','指数':'51（グラフ51.1）','攻撃獲得率':'66.8%'},{'表示箇所':'川崎F 詳細欄・シーズン先頭','指数':'グラフ50.8','攻撃獲得率':'68.7%'}])
document='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>チームスタイルの複数値調査</title><style>body{font-family:system-ui;line-height:1.8;background:#f2f5f8}main{max-width:1100px;margin:auto;padding:25px;background:white}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}.note{background:#fff2d6;padding:16px}a{color:#17649a}</style><main><h1>チームスタイル指標の複数値を調査</h1><p>2026年11月3日 / J1全20チームの上部一覧、取得済み追加7項目、複数チームの詳細ページを照合。</p><p class="note"><b>現在公開されている上部一覧に対応する値は特定できる。</b>追加7項目×20チーム＝140組で一覧の指数・率を確認。うち137組は対応するサマリー詳細行を一意に絞れた。ただし集計日が確定した「最新の真値」を保証したものではない。過去試合前の値の復元と、予測への正式採用は未実施。</p><h2>確かめた事実</h2><ul><li>元HTML内に同一チームの異なる行がある。抽出処理だけで生じた重複ではない。</li><li>シーズン変化グラフには、同じ「2026/27」というラベルで5レコードが並び、個別の日付がない。</li><li>ハイブロックの上部一覧は、20チームすべてで当該シーズンの最後のグラフ値と対応した。</li><li>横浜FM・川崎Fのハイブロック詳細欄は最初のグラフ値側に対応する。横浜FMの年指定なしURLでも同様で、2025指定では上部一覧と詳細欄の攻撃獲得率は73.9%で一致した。</li></ul>'''
document+=example.to_html(index=False)
document+='''<h2>何が原因か</h2><p>複数の集計レコードが同じシーズン名で出力され、ページの部位によって異なるレコードを表示している可能性が高い。ただし、各レコードが月次更新なのか修正版なのか、内部データベースの重複なのかは公開HTMLから断定できない。ホーム・アウェイや直近5試合などを区別するラベルも確認できなかった。</p><p><a href="https://www.football-lab.jp/pages/team_style">公式の定義</a>には月1回程度の更新予定と、2026年夏のブロック定義変更が記載されている。これだけでは5レコードの意味や集計日を特定できず、「最後の値だから最新」とは断定しない。</p><h2>採用候補を絞る手順</h2><ol><li>チーム別ページ上部の一覧から指数と率を読む。</li><li>取得済みサマリー表から、この組と一致する詳細行を抽出。同値の重複だけをまとめる。</li><li>丸められた指数のため候補が残る場合は、同じ上部一覧と対応するグラフ終端の小数付き指数も照合する。表とグラフの行順はチーム名・指数・率で検査。</li><li>一意に決まらない詳細値は未確定のまま保持し、平均や先頭・末尾で埋めない。</li></ol><p>140組中136組が一覧だけで一意になり、福岡のハイブロックは小数付き指数の照合で絞れた。計137組の詳細行を候補としてCSVに保存した。</p><h2>残る3組</h2><ul><li>広島・ハイプレッシング：同じ指数・守備成功率でも、ゲイン率・被シュート率・被シュート成功率が異なる。</li><li>福岡・カウンタープレス：同じ指数・守備成功率でも、ゲイン率・被シュート率・被シュート成功率が異なる。</li><li>柏・ラインブレイクラン：同じ指数・味方シュート率でも、味方ゴール率が1.9%と2.1%に分かれる。</li></ul><h2>予測への扱い</h2><p>追加指標をすべて使用不能とする必要はない。上部一覧の値を使う入力経路と、詳細行の一致確認、曖昧な列だけの欠損処理を分けて構築できる。推奨するのは「上部一覧との整合」を明示した選択方針であり、実際の集計期間を確定したという扱いではない。精度検証は別工程で行う。</p><p>今回の変更は調査・候補値の保存のみ。通常の予測計算にはまだ組み込んでいない。更新日10月2日の表示から各レコードの時点は復元できず、過去評価に投入する根拠にはしていない。</p><p><a href="overview_alignment.csv">140組の一覧照合</a> / <a href="precision_resolution.csv">小数付き指数による追加照合</a> / <a href="candidate_values.csv">137組の候補値</a> / <a href="annual_sequences.csv">20チームのシーズングラフ</a> / <a href="summary.json">結果JSON</a> / <a href="sources/team_index.json">公式ページ原本一覧</a></p></main></html>'''
(OUT/'report.html').write_text(document)
