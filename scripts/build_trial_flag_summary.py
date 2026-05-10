#!/usr/bin/env python3
import argparse
import csv
import html
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
TRIAL_DIR = BASE_DIR / "data" / "external_metrics" / "trial_flags"
REPORT_DIR = BASE_DIR / "data" / "external_metrics" / "reports"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_round_no(text):
    m = re.search(r"第([0-9０-９]+)節", str(text))
    if not m:
        return 999
    digits = str(m.group(1)).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return int(digits)


def pct(n, d):
    if not d:
        return 0.0
    return 100.0 * n / d


def build_round_summary(path):
    rows = read_rows(path)
    if not rows:
        return None, []
    league = str(rows[0].get("league") or "").strip().upper()
    if not league:
        mid = str(rows[0].get("match_id") or "").strip().lower()
        m = re.match(r"^(j[123])_", mid)
        if m:
            league = m.group(1).upper()
    if not league:
        m = re.match(r".*_(j[123])_20[0-9]{2}_", str(path.name), flags=re.IGNORECASE)
        if m:
            league = m.group(1).upper()
    round_label = str(rows[0].get("節") or "").strip()
    m = re.search(r"(第[0-9０-９]+節)", round_label)
    round_name = m.group(1) if m else round_label
    flags = Counter(str(r.get("flab_trial_flag") or "").strip().upper() for r in rows)
    total = len(rows)
    caution_rows = [
        {
            "league": league,
            "round": round_name,
            "match_id": r.get("match_id", ""),
            "home_team": r.get("home_team", ""),
            "away_team": r.get("away_team", ""),
            "predicted_result": r.get("predicted_result", ""),
            "flab_trial_flag": r.get("flab_trial_flag", ""),
            "flab_trial_score": r.get("flab_trial_score", ""),
            "flab_trial_reason": r.get("flab_trial_reason", ""),
        }
        for r in rows
        if str(r.get("flab_trial_flag") or "").strip().upper() == "CAUTION"
    ]
    summary = {
        "league": league,
        "round": round_name,
        "round_no": parse_round_no(round_name),
        "rows": total,
        "go_count": flags.get("GO", 0),
        "caution_count": flags.get("CAUTION", 0),
        "hold_count": flags.get("HOLD", 0),
        "blank_count": flags.get("", 0),
        "go_rate": pct(flags.get("GO", 0), total),
        "caution_rate": pct(flags.get("CAUTION", 0), total),
        "hold_rate": pct(flags.get("HOLD", 0), total),
    }
    return summary, caution_rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(round_rows, caution_rows):
    total_rows = sum(r["rows"] for r in round_rows)
    total_go = sum(r["go_count"] for r in round_rows)
    total_caution = sum(r["caution_count"] for r in round_rows)
    total_hold = sum(r["hold_count"] for r in round_rows)

    lines = []
    lines.append("# Trial Flag 集計一覧")
    lines.append("")
    lines.append(f"- 対象リーグ: `J1/J2`")
    lines.append(f"- 対象節数: `{len(round_rows)}`")
    lines.append(f"- 対象試合数: `{total_rows}`")
    lines.append(f"- GO: `{total_go}` / CAUTION: `{total_caution}` / HOLD: `{total_hold}`")
    lines.append("")
    lines.append("## 節別サマリー")
    lines.append("")
    lines.append("| League | Round | Matches | GO | CAUTION | HOLD | GO率 | CAUTION率 | HOLD率 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in round_rows:
        lines.append(
            f"| {row['league']} | {row['round']} | {row['rows']} | {row['go_count']} | {row['caution_count']} | {row['hold_count']} | {row['go_rate']:.1f}% | {row['caution_rate']:.1f}% | {row['hold_rate']:.1f}% |"
        )
    lines.append("")
    lines.append("## CAUTION 試合")
    lines.append("")
    if not caution_rows:
        lines.append("- なし")
    else:
        for row in caution_rows:
            lines.append(
                f"- `{row['league']} {row['round']} {row['home_team']} vs {row['away_team']}` 予想=`{row['predicted_result']}` score=`{row['flab_trial_score']}` 理由=`{row['flab_trial_reason']}`"
            )
    return "\n".join(lines)


def build_html(round_rows, caution_rows):
    total_rows = sum(r["rows"] for r in round_rows)
    total_go = sum(r["go_count"] for r in round_rows)
    total_caution = sum(r["caution_count"] for r in round_rows)
    total_hold = sum(r["hold_count"] for r in round_rows)

    def row_class(caution_rate):
        if caution_rate >= 60.0:
            return "danger"
        if caution_rate >= 45.0:
            return "warn"
        return ""

    lines = []
    lines.append("<!doctype html>")
    lines.append("<html lang='ja'>")
    lines.append("<head>")
    lines.append("<meta charset='utf-8'>")
    lines.append("<title>Trial Flag 集計一覧</title>")
    lines.append("<style>")
    lines.append("body{font-family:system-ui,-apple-system,sans-serif;margin:24px;color:#1a1a1a;}")
    lines.append("table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px;}")
    lines.append("th,td{border:1px solid #ddd;padding:8px 10px;text-align:left;vertical-align:top;}")
    lines.append("thead th{background:#f5f5f5;}")
    lines.append("tr.warn td{background:#fff6db;}")
    lines.append("tr.danger td{background:#ffe0e0;}")
    lines.append(".meta{margin:8px 0 16px;font-size:14px;}")
    lines.append(".meta b{color:#b42318;}")
    lines.append(".small{color:#666;font-size:12px;}")
    lines.append("</style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append("<h2>Trial Flag 集計一覧</h2>")
    lines.append(
        f"<div class='meta'>対象リーグ: <b>J1/J2</b> / 対象節数: <b>{len(round_rows)}</b> / "
        f"対象試合数: <b>{total_rows}</b> / GO: <b>{total_go}</b> / "
        f"CAUTION: <b>{total_caution}</b> / HOLD: <b>{total_hold}</b></div>"
    )
    lines.append("<div class='small'>CAUTION率 60%以上は赤、45%以上は黄で強調表示。</div>")
    lines.append("<h3>節別サマリー</h3>")
    lines.append("<table>")
    lines.append("<thead><tr><th>League</th><th>Round</th><th>Matches</th><th>GO</th><th>CAUTION</th><th>HOLD</th><th>GO率</th><th>CAUTION率</th><th>HOLD率</th></tr></thead>")
    lines.append("<tbody>")
    for row in round_rows:
        klass = row_class(row["caution_rate"])
        cls_attr = f" class='{klass}'" if klass else ""
        lines.append(
            f"<tr{cls_attr}><td>{html.escape(row['league'])}</td><td>{html.escape(row['round'])}</td>"
            f"<td>{row['rows']}</td><td>{row['go_count']}</td><td>{row['caution_count']}</td><td>{row['hold_count']}</td>"
            f"<td>{row['go_rate']:.1f}%</td><td>{row['caution_rate']:.1f}%</td><td>{row['hold_rate']:.1f}%</td></tr>"
        )
    lines.append("</tbody></table>")
    lines.append("<h3>CAUTION 試合</h3>")
    lines.append("<table>")
    lines.append("<thead><tr><th>League</th><th>Round</th><th>Match</th><th>予想</th><th>Score</th><th>理由</th></tr></thead>")
    lines.append("<tbody>")
    if caution_rows:
        for row in caution_rows:
            match_label = f"{row['home_team']} vs {row['away_team']}"
            lines.append(
                f"<tr><td>{html.escape(row['league'])}</td><td>{html.escape(row['round'])}</td>"
                f"<td>{html.escape(match_label)}</td><td>{html.escape(row['predicted_result'])}</td>"
                f"<td>{html.escape(str(row['flab_trial_score']))}</td><td>{html.escape(row['flab_trial_reason'])}</td></tr>"
            )
    else:
        lines.append("<tr><td colspan='6'>CAUTION 試合はありません。</td></tr>")
    lines.append("</tbody></table>")
    lines.append("</body></html>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="J1/J2 の trial flag 集計一覧を作る")
    parser.add_argument("--glob", default="data/external_metrics/trial_flags/predictions_round_j[12]_2026_*_football_lab_trial_flags_*.csv")
    parser.add_argument("--prefix", default="trial_flag_summary_j1_j2_2026")
    args = parser.parse_args()

    paths = sorted(BASE_DIR.glob(args.glob))
    round_rows = []
    caution_rows = []
    for path in paths:
        summary, cautions = build_round_summary(path)
        if summary is None:
            continue
        round_rows.append(summary)
        caution_rows.extend(cautions)

    round_rows.sort(key=lambda r: (r["league"], r["round_no"], r["round"]))
    caution_rows.sort(key=lambda r: (r["league"], parse_round_no(r["round"]), r["match_id"]))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = REPORT_DIR / f"{args.prefix}.csv"
    caution_csv = REPORT_DIR / f"{args.prefix}_cautions.csv"
    summary_md = REPORT_DIR / f"{args.prefix}.md"
    summary_html = REPORT_DIR / f"{args.prefix}.html"

    write_csv(
        summary_csv,
        round_rows,
        [
            "league",
            "round",
            "round_no",
            "rows",
            "go_count",
            "caution_count",
            "hold_count",
            "blank_count",
            "go_rate",
            "caution_rate",
            "hold_rate",
        ],
    )
    write_csv(
        caution_csv,
        caution_rows,
        [
            "league",
            "round",
            "match_id",
            "home_team",
            "away_team",
            "predicted_result",
            "flab_trial_flag",
            "flab_trial_score",
            "flab_trial_reason",
        ],
    )
    summary_md.write_text(build_markdown(round_rows, caution_rows), encoding="utf-8")
    summary_html.write_text(build_html(round_rows, caution_rows), encoding="utf-8")

    print(f"summary_csv={summary_csv}")
    print(f"caution_csv={caution_csv}")
    print(f"summary_md={summary_md}")
    print(f"summary_html={summary_html}")


if __name__ == "__main__":
    main()
