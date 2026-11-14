#!/usr/bin/env python3
"""Audit full-style replay readiness without selecting ambiguous source rows."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd
from common import ROOT, normalize_text, sha256
sys.path.insert(0, str(ROOT / 'scripts'))
from fetch_football_lab_html import load_targets, DEFAULT_XLSX, extract_title
from build_football_lab_snapshot import extract_team_tables, parse_title_info, parse_asof_date


def run(snapshot: str) -> None:
    out = ROOT / 'data/reports/full_style_replay_audit'
    out.mkdir(parents=True, exist_ok=True)
    folder = ROOT / 'external_contents' / snapshot
    fetch = pd.read_csv(folder / 'football_lab_fetch_index.csv')
    targets = [t for t in load_targets(DEFAULT_XLSX) if t.data_name == 'チームスタイル指標']
    pages, cells, variants = [], [], []
    for t in targets:
        path = folder / t.league / f'{t.item_name}.html'
        records = fetch[(fetch.league == t.league) & (fetch.item_name == t.item_name)]
        assert len(records) == 1 and records.iloc[0].status == 'ok' and path.exists(), t.item_name
        text = path.read_text(encoding='utf-8')
        title = extract_title(text)
        season, league, label = parse_title_info(title, t.league)
        assert season == '2026' and league == t.league and t.item_name in title
        asof = parse_asof_date(text, snapshot)
        meta = dict(snapshot_date=snapshot, asof_date=asof, league=league, season=season,
                    source_file=path.name, page_label=label, title=title)
        rows = [r for r in extract_team_tables(text, meta) if r['section_title'] == t.item_name + 'のサマリー']
        assert rows, t.item_name
        groups = {}
        headers = None
        for position, row in enumerate(rows):
            h, v = json.loads(row['headers_json']), json.loads(row['values_json'])
            assert len(h) == len(v) and (headers is None or h == headers)
            headers = h
            team = normalize_text(row['team_key'])
            groups.setdefault(team, set()).add(tuple(v))
            for name, value in zip(h, v):
                number = re.fullmatch(r'(-?\d[\d,]*(?:\.\d+)?)(%|人|m|m2)?', normalize_text(value))
                assert number, (t.item_name, name, value)
                cells.append(dict(league=league, style=t.item_name, team=team, row_position=position,
                                  metric=name, value=float(number[1].replace(',', '')), unit=number[2] or '',
                                  original=value, asof_date=asof, snapshot_date=snapshot))
        assert len(groups) == 20
        for team, vv in groups.items():
            for v in sorted(vv):
                variants.append(dict(league=league, style=t.item_name, team=team, headers=json.dumps(headers,ensure_ascii=False),
                                     values=json.dumps(v,ensure_ascii=False), unique_variants=len(vv)))
        pages.append(dict(league=league, style=t.item_name, asof_date=asof, snapshot_date=snapshot,
                          rows=len(rows), teams=len(groups), fields=len(headers),
                          conflicting_teams=sum(len(v)>1 for v in groups.values()),
                          source_sha256=sha256(path), source_file=str(path.relative_to(ROOT))))
    coverage = pd.DataFrame(pages)
    pd.DataFrame(cells).to_csv(out / 'all_style_values.csv', index=False)
    pd.DataFrame(variants).to_csv(out / 'row_variants.csv', index=False)
    coverage.to_csv(out / 'page_coverage.csv', index=False)
    previous = pd.read_csv(ROOT / 'data/reports/pattern_matchup_validation/matches.csv').drop_duplicates('match_id')
    replay = []
    for _, m in previous.iterrows():
        if m.league == 'j2':
            reason = 'J2 does not publish the seven J1 tracking styles'
            missing = []
        else:
            expected = [t.item_name for t in targets if t.league == 'j1']
            missing = [s for s in expected if not (ROOT / 'external_contents' / str(m.snapshot) / 'j1' / f'{s}.html').exists()]
            assert missing, 'A complete archive now exists: implement replay instead of reporting it unavailable'
            reason = 'missing tracking styles in prematch archive'
        replay.append(dict(match_id=m.match_id, league=m.league, datetime=m.datetime,
                           original_snapshot=m.snapshot, eligible_full_style=False, reason=reason,
                           missing_pages=json.dumps(missing, ensure_ascii=False)))
    pd.DataFrame(replay).to_csv(out / 'replay_eligibility.csv', index=False)
    summary = dict(snapshot=snapshot, successful_fetches=int(fetch.status.eq('ok').sum()),
                   total_fetches=len(fetch), style_pages=len(pages),
                   j1_fields_per_team=int(coverage[coverage.league.eq('j1')].fields.sum()),
                   j2_fields_per_team=int(coverage[coverage.league.eq('j2')].fields.sum()),
                   previous_matches=len(previous), previous_j1_matches=int(previous.league.eq('j1').sum()),
                   full_style_replay_performed=False,
                   reason='No prematch tracking archive; new tracking rows conflict within each team.')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    example = pd.DataFrame(variants)
    example = example[(example['style']=='ハイブロック') & example.team.isin(['横浜FM', '川崎F'])]
    example_rows = []
    for _, r in example.iterrows():
        example_rows.append({'チーム':r.team, **dict(zip(json.loads(r.headers), json.loads(r['values'])))})
    body = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全スタイル追加後の再検証監査</title><style>body{font-family:system-ui;line-height:1.8;background:#f4f6f8}main{max-width:1200px;margin:auto;padding:24px;background:white}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:7px}.notice{padding:15px;background:#fff0cf}section{overflow:auto}</style><main><h1>全スタイル追加後の再検証監査</h1>'''
    body += f'<p>保存日：{snapshot}。取得は{len(fetch)}ページ中{summary["successful_fetches"]}ページ成功。J1の14項目、J2の公開7項目はすべて存在し、各20チームの表を確認。</p>'
    body += '<p class="notice"><b>取得の補完は完了。ただし「全指標を入れた過去検証」は未実施。</b>試合前の追加7ページが保存されていないことと、元HTMLに同一チームの異なる数値が並んでいることが理由。任意の行や平均値を採用して成績を出すことはしない。</p>'
    body += '<h2>全列を抽出して確認</h2><p>J1は1チーム105指標、J2は49指標。生の行・単位・ページ別更新日を保持して検証用CSVに抽出した。複数行の意味が未確認のため、予測に入れる1行には集約していない。既存の計算用CSVは攻撃7項目の指数・シュート率のみであり、取得コマンドの実行だけでは全列利用にはならない。</p><section>'
    body += coverage[['league','style','asof_date','teams','fields','rows','conflicting_teams']].to_html(index=False) + '</section>'
    body += '<h2>同じチームに異なる数値がある具体例</h2><p>以下は元HTMLのサマリー表にある別々の行。単なる同値の重複ではなく、グラフの数値にも相違がある。集計期間や条件を区別するラベルがこの表にはないため、先頭・末尾・平均を勝手に採用できない。</p><section>'
    body += pd.DataFrame(example_rows).to_html(index=False) + '</section>'
    body += '<p>チーム別ページも照合した。横浜FMのハイブロックは上部メニューが指数55・攻撃獲得率73.2%なのに対し、詳細欄は72.3%。シーズン変化グラフにも同じ2026/27ラベルで複数の値があり、月日を特定できなかった。<a href="yokohama_high_block_source.html">照合した原本</a> / <a href="https://www.football-lab.jp/y-fm/style?s=81&amp;year=2026">公開ページ</a>。</p>'
    body += f'<h2>過去の検証期間との関係</h2><p>前回の評価は{previous.datetime.min()}〜{previous.datetime.max()}の{len(previous)}試合（J1 90、J2 100）。追加7ページの通常保存は11月3日分のみ。J1の90試合はいずれも元の試合前保存分に追加7ページがない。J2には対応する公開ページがないので、J1の14項目と同じ構成では再検証できない。</p>'
    body += '<p>今回の攻撃系ページは11月2日更新で、11月1日の横浜FM－川崎F等の結果を含む。代わりに以前の攻撃系ページを使っても、追加ページの試合前保存がない点は解消しない。守備・走行系の表示更新日は10月2日だが、この表示だけでは各行の集計期間や当時の公開値を保証できない。</p><h2>再検証を成立させる条件</h2><ol><li>複数行の集計条件と、採用すべき行を確認する。</li><li>過去の各時点の守備・走行指標を入手できれば、試合前の攻撃指標と合わせて再検証する。</li><li>過去値を復元できなければ、今後の試合前保存を積み上げて検証する。</li></ol><p>今回、改善・悪化の新しい数値は算出していない。正式予測モデルとBuyPlanは変更していない。</p><p><a href="all_style_values.csv">全指標・全行</a> / <a href="row_variants.csv">同一チームの異なる行</a> / <a href="page_coverage.csv">ページ監査</a> / <a href="replay_eligibility.csv">190試合の利用可否</a> / <a href="summary.json">概要</a></p></main></html>'
    (out / 'report.html').write_text(body, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot', default='20261103')
    run(parser.parse_args().snapshot)
