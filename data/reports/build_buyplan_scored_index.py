#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import re
from datetime import datetime
from pathlib import Path


ROUND_RE = re.compile(r"^round(\d+)$")
TOTO_RE = re.compile(r"^toto(\d+)$")


def load_toto_rounds_for_season(path: Path, season: str) -> set[int]:
    if not season:
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {
            int(str(row.get("toto_round", "")).strip())
            for row in reader
            if str(row.get("season", "")).strip() == str(season).strip()
            and str(row.get("toto_round", "")).strip().isdigit()
        }


def collect_scored_pages(
    rounds_dir: Path,
    kind: str = "round",
    allowed_ids: set[int] | None = None,
) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    pattern = TOTO_RE if kind == "toto" else ROUND_RE
    if not rounds_dir.exists():
        return items
    for child in rounds_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        item_id = int(m.group(1))
        if allowed_ids is not None and item_id not in allowed_ids:
            continue
        scored = child / "buyplan_scored.html"
        if scored.exists():
            items.append((item_id, scored))
    items.sort(key=lambda x: x[0], reverse=True)
    return items


def build_html(
    items: list[tuple[int, Path]],
    limit: int,
    kind: str = "round",
    season: str = "",
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top = items[: max(0, limit)]
    lines: list[str] = []
    lines.append("<!doctype html>")
    lines.append("<html lang='ja'><head><meta charset='utf-8'>")
    season_text = f"{season}シーズン " if season else ""
    lines.append(f"<title>{html.escape(season_text)}BuyPlan採点履歴</title>")
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
    unit = "開催回" if kind == "toto" else "節"
    lines.append(f"<h2>{html.escape(season_text)}BuyPlan採点履歴（最新{limit}{unit}）</h2>")
    lines.append(f"<div class='meta'>生成日時: {html.escape(now)} / 件数: {len(top)}</div>")
    lines.append("<table><thead><tr><th>節</th><th>リンク</th></tr></thead><tbody>")
    for round_no, scored_path in top:
        dirname = f"toto{round_no}" if kind == "toto" else f"round{round_no:02d}"
        rel = f"{dirname}/buyplan_scored.html"
        label = f"toto第{round_no}回" if kind == "toto" else f"第{round_no}節"
        lines.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
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
    parser.add_argument("--kind", choices=["round", "toto"], default="round")
    parser.add_argument("--season", default="", help="toto節リスト上のシーズン（toto時の絞り込み）")
    parser.add_argument("--toto-order-csv", default="data/manual/toto節リスト.csv")
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

    allowed_ids = None
    if args.kind == "toto" and args.season:
        allowed_ids = load_toto_rounds_for_season(Path(args.toto_order_csv), args.season)
    items = collect_scored_pages(rounds_dir, kind=args.kind, allowed_ids=allowed_ids)
    content = build_html(items, args.limit, kind=args.kind, season=args.season)
    out_path.write_text(content, encoding="utf-8")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
