"""Repair historical chance columns using each CSV's original saved HTML.

Other columns and source file modification times are preserved so fixture
snapshot selection does not change. Original CSVs are backed up before writes.
"""
import argparse
import csv
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from build_football_lab_snapshot import (
    BASE_DIR, OUTPUT_DIR, WIDE_TABLE_MAPPINGS, build_match_comparison,
    build_wide_metrics, extract_team_tables, parse_asof_date, parse_title_info,
    strip_tags, validate_chance_metrics, write_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--since', required=True, help='Earliest snapshot to repair, YYYYMMDD')
    args = parser.parse_args()
    datetime.strptime(args.since, '%Y%m%d')
    required = list(WIDE_TABLE_MAPPINGS['チャンス構築率'].values()) + list(WIDE_TABLE_MAPPINGS['被チャンス構築率'].values())
    backup = BASE_DIR / 'data' / 'repair_backups' / ('football_lab_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    staged, cache = [], {}
    for path in sorted(OUTPUT_DIR.glob('*football_lab_compare_*_j[12].csv')):
        match = re.search(r'_football_lab_compare_(\d{8})_(j[12])\.csv$', path.name)
        date, league = match.groups()
        if date < args.since:
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        def valid(value):
            try:
                return float('-inf') < float(value) < float('inf')
            except (ValueError, TypeError):
                return False
        if not rows or all(valid(row.get(f'{col}_{side}')) for row in rows for col in required for side in ('home', 'away')):
            continue
        key = date, league
        if key not in cache:
            html_path = BASE_DIR / 'external_contents' / date / league / 'チャンス構築率.html'
            html = html_path.read_text(encoding='utf-8')
            title = strip_tags(re.search(r'<title>(.*?)</title>', html, re.S).group(1))
            season, parsed_league, label = parse_title_info(title, league)
            meta = dict(snapshot_date=date, asof_date=parse_asof_date(html, date), league=parsed_league,
                        season=season, page_label=label, source_file=html_path.name, title=title)
            metrics = build_wide_metrics(extract_team_tables(html, meta), [])
            validate_chance_metrics(metrics)
            cache[key] = metrics
        repaired = build_match_comparison(rows, cache[key])
        if not all(valid(row.get(f'{col}_{side}')) for row in repaired for col in required for side in ('home', 'away')):
            raise ValueError(f'Original HTML cannot fully repair {path}; no files changed')
        # Only chance metrics may change; keep every other original field.
        for old, new in zip(rows, repaired):
            for column, value in new.items():
                if column.startswith('flab_chance_'):
                    old[column] = value
        staged.append((path, rows))
    for path, rows in staged:
        backup.mkdir(parents=True, exist_ok=True)
        stat = path.stat()
        shutil.copy2(path, backup / path.name)
        temporary = path.with_suffix('.csv.tmp')
        write_csv(temporary, rows)
        os.utime(temporary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        temporary.replace(path)
        print(f'[FOOTBALL_LAB][REPAIR] {path.name} rows={len(rows)}')
    print(f'[FOOTBALL_LAB][REPAIR] files={len(staged)} backup={backup}')


if __name__ == '__main__':
    main()
