#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
from datetime import datetime
from pathlib import Path


ROUND_RE = re.compile(r"^round(\d+)$")


def collect_scored_pages(rounds_dir: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for child in rounds_dir.iterdir():
        if not child.is_dir():
            continue
        m = ROUND_RE.match(child.name)
        if not m:
            continue
        scored = child / "buyplan_scored.html"
        if scored.exists():
            items.append((int(m.group(1)), scored))
    items.sort(key=lambda x: x[0], reverse=True)
    return items


def build_html(items: list[tuple[int, Path]], limit: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top = items[: max(0, limit)]
    lines: list[str] = []
    lines.append("<!doctype html>")
    lines.append("<html lang='ja'><head><meta charset='utf-8'>")
    lines.append("<title>BuyPlan Scored 履歴</title>")
    lines.append("<style>")
    lines.append("body{font-family:system-ui,-apple-system,sans-serif;margin:24px;color:#111;}")
    lines.append("h2{margin:0 0 12px 0;}")
    lines.append(".meta{font-size:12px;color:#555;margin-bottom:14px;}")
    lines.append("table{border-collapse:collapse;width:100%;max-width:900px;font-size:13px;}")
    lines.append("th,td{border:1px solid #ddd;padding:8px;text-align:left;}")
    lines.append("th{background:#f5f5f5;}")
    lines.append("tbody tr:nth-child(even){background:#fcfcfc;}")
    lines.append("a{text-decoration:none;color:#0b57d0;}")
    lines.append("a:hover{text-decoration:underline;}")
    lines.append("</style></head><body>")
    lines.append("<h2>BuyPlan Scored 履歴リンク（最新20節）</h2>")
    lines.append(f"<div class='meta'>生成日時: {html.escape(now)} / 件数: {len(top)}</div>")
    lines.append("<table><thead><tr><th>節</th><th>リンク</th></tr></thead><tbody>")
    for round_no, scored_path in top:
        rel = f"round{round_no:02d}/buyplan_scored.html"
        lines.append(
            "<tr>"
            f"<td>第{round_no}節</td>"
            f"<td><a href='{html.escape(rel)}'>{html.escape(rel)}</a></td>"
            "</tr>"
        )
    if not top:
        lines.append("<tr><td colspan='2'>buyplan_scored.html が見つかりませんでした</td></tr>")
    lines.append("</tbody></table>")
    lines.append("</body></html>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build index HTML for buyplan_scored pages.")
    parser.add_argument("--rounds-dir", default="data/eval/rounds", help="rounds root directory")
    parser.add_argument("--limit", type=int, default=20, help="max links to include")
    parser.add_argument(
        "--out",
        default="data/eval/rounds/buyplan_scored_index.html",
        help="output html path",
    )
    args = parser.parse_args()

    rounds_dir = Path(args.rounds_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = collect_scored_pages(rounds_dir)
    content = build_html(items, args.limit)
    out_path.write_text(content, encoding="utf-8")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
