"""Collect team overview pages and select additional style values on every LAB fetch."""
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
import pandas as pd
from bs4 import BeautifulSoup
from common import normalize_text
from style_observation import archive_styles
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_football_lab_snapshot import extract_team_tables,extract_chart_rows,parse_js_array_rows,parse_asof_date

CODES={'45':('ラインブレイクラン','味方シュート率'),'66':('ハイプレッシング','守備成功率'),
       '67':('ミドルプレッシング','守備成功率'),'69':('カウンタープレス','守備成功率'),
       '81':('ハイブロック','攻撃獲得率'),'82':('ミドルブロック','攻撃獲得率'),'83':('ローブロック','攻撃獲得率')}


def assemble(league_dir, get_page):
    """get_page(slug,code) returns freshly fetched, season-validated HTML path."""
    base=BeautifulSoup((league_dir/'ハイブロック.html').read_text(),'html.parser')
    teams={urlsplit(a['href']).path.strip('/'):normalize_text(a.select_one('.sp').text) for a in base.select('td.tName a')}
    if len(teams)!=20:raise ValueError('expected 20 J1 teams')
    tables={};texts={};sources=[]
    for code,(name,rate) in CODES.items():
        path=league_dir/f'{name}.html';sources.append(path);texts[code]=path.read_text()
        tables[code]=[r for r in extract_team_tables(texts[code],{'page_label':name+' (チームスタイル指標)'}) if r['section_title']==name+'のサマリー']
        if set(normalize_text(r['team_key']) for r in tables[code])!=set(teams.values()):raise ValueError('team coverage mismatch')
    output=[]
    for slug,team in teams.items():
        overview=get_page(slug,'81');sources.append(overview)
        soup=BeautifulSoup(overview.read_text(),'html.parser');menu={}
        for a in soup.select('a[href]'):
            code=parse_qs(urlsplit(a['href']).query).get('s',[''])[0]
            if code in CODES and a.select_one('.graphPie'):
                menu[code]=(float(a.select_one('.numL').text),float(a.select_one('.graphPie').text.strip('%')))
        if set(menu)!=set(CODES):raise ValueError(f'incomplete overview: {team}')
        for code,(name,rate) in CODES.items():
            idx,pct=menu[code];matching={};headers=None
            for row in tables[code]:
                if normalize_text(row['team_key'])!=team:continue
                headers=json.loads(row['headers_json']);values=json.loads(row['values_json']);v=dict(zip(headers,values))
                if float(v['指数'])==idx and float(v[rate].strip('%'))==pct:matching[tuple(values)]=v
            # Disambiguate rounding only when the annual endpoint agrees with overview.
            if len(matching)>1:
                detail=overview if code=='81' else get_page(slug,code)
                if detail not in sources:sources.append(detail)
                sequences=[]
                for m in re.finditer(r'arrayToDataTable\(\[(.*?)\]\);',detail.read_text(),re.S):
                    # Restrict to the season requested by the page title below.
                    season_match=re.search(r'(\d{4}/\d{2})',BeautifulSoup(detail.read_text(),'html.parser').title.text)
                    current=[r for r in parse_js_array_rows(m[1]) if r and season_match and r[0]==season_match[1]]
                    if current:sequences.append(current)
                if len(sequences)==1:
                    last=sequences[0][-1]
                    if abs(float(last[1])-idx)<=.500001 and last[2]==pct:
                        chart=extract_chart_rows(texts[code],{'page_label':name+' (チームスタイル指標)'})
                        tab=tables[code];precise={}
                        if len(chart)!=len(tab):raise ValueError('table/chart size mismatch')
                        for tr,cr in zip(tab,chart):
                            v=dict(zip(json.loads(tr['headers_json']),json.loads(tr['values_json'])))
                            if tr['team_key']!=cr['team_key'] or abs(float(v['指数'])-float(cr['metric_x_value']))>.500001 or float(v[rate].strip('%'))!=float(cr['metric_y_value']):raise ValueError('table/chart alignment mismatch')
                            if normalize_text(tr['team_key'])==team and cr['metric_x_value']==last[1] and cr['metric_y_value']==last[2]:precise[tuple(v.values())]=v
                        if precise:matching=precise
            if not headers:raise ValueError('missing summary row')
            chosen=next(iter(matching.values())) if len(matching)==1 else {}
            for metric in headers:
                unit='%' if '率' in metric else '';value=None;status='missing_ambiguous'
                if metric=='指数':value=idx;status='accepted_overview'
                elif metric==rate:value=pct;status='accepted_overview'
                elif chosen:
                    m=re.fullmatch(r'(-?\d[\d,]*(?:\.\d+)?)(%|人|m|m2)?',normalize_text(chosen[metric]))
                    if not m:raise ValueError('unparsed style value')
                    value=float(m[1].replace(',',''));unit=m[2] or '';status='accepted_aligned_detail'
                output.append(dict(team=team,style=name,metric=metric,value=value,unit=unit,selection_status=status,
                                   publisher_displayed_date=parse_asof_date(texts[code],''),overview_source=str(overview),summary_source=str(league_dir/f'{name}.html')))
    return pd.DataFrame(output),sources


def collect(out_root,snapshot_date,season_year):
    from fetch_football_lab_html import Target,fetch_and_save,write_index_csv
    index=out_root/snapshot_date/'football_lab_fetch_index.csv';rows=[];cache={}
    def get_page(slug,code):
        key=(slug,code)
        if key not in cache:
            target=Target('j1','チームスタイル照合',f'チームスタイル_{slug}_{code}',f'https://www.football-lab.jp/{slug}/style?s={code}')
            row=fetch_and_save(target,out_root,snapshot_date,False,season_year)
            if row['status']=='ok':
                original=Path(row['saved_path'])
                destination=original.parent/'style_observation_sources'/original.name
                destination.parent.mkdir(exist_ok=True)
                original.replace(destination)
                row['saved_path']=str(destination)
            rows.append(row)
            write_index_csv(index,rows)
            if row['status']!='ok':raise RuntimeError(row['error'])
            cache[key]=Path(row['saved_path'])
        return cache[key]
    values,sources=assemble(out_root/snapshot_date/'j1',get_page)
    saved=archive_styles(values,season_year,sources)
    print(f'[STYLE] inputs={saved} accepted={values.value.notna().sum()} missing={values.value.isna().sum()}')
    return saved
