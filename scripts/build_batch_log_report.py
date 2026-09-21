#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import html
import os
import re
from collections import Counter


RANKMOT_ISSUE_RE = re.compile(
    r"\[MERGE_QC\]\[(?:INFO|WARN|ERROR)\]\s+rankmot_[^:]*:.*"
    r"(?:対象リーグとのチーム一致なし|順位情報の未結合|left_only=[1-9]\d*)", re.I
)

ISSUE_PATTERNS = [
    (
        RANKMOT_ISSUE_RE,
        "順位情報の結合不一致",
        "順位・モチベーション情報が予測に取り込まれていません。チーム名と入力CSVを確認してください。",
    ),
    (
        re.compile(
            r"Network/DNS unavailable|Could not resolve host|Temporary failure in name resolution|"
            r"NameResolutionError|socket\.gaierror|MaxRetryError",
            re.I,
        ),
        "ネットワーク/DNSエラー",
        "外部サイトへ接続できていません。ネットワーク疎通とDNSを確認してください。",
    ),
    (
        re.compile(r"\[PROB_QC\]\[WARN\]|elo_diff_for_prob=.*prob_home.*prob_away", re.I),
        "確率整合性警告",
        "Elo差と H/A 勝率の向きが一致しない試合があります。即停止ではありませんが、モデル整合性の要確認です。",
    ),
    (
        re.compile(r"\[MERGE_QC\]\[WARN\]\s+absence_", re.I),
        "欠場情報の未登録/不一致",
        "欠場管理データに該当するチーム・試合日の行がありません。欠場者なしと未確認の両方を含みます。",
    ),
    (
        re.compile(r"\[MERGE_QC\]\[WARN\]\s+weather_cache_", re.I),
        "天気情報の未取得",
        "試合IDに対応する天気キャッシュがありません。対象試合の日時・会場と天気取得範囲を確認してください。",
    ),
    (
        re.compile(r"\[MERGE_QC\]\[WARN\]\s+fatigue_", re.I),
        "疲労情報の結合不一致",
        "主に試合日時未定などにより疲労データを結合できない試合があります。",
    ),
    (
        re.compile(
            r"\[MISSING_QC\]\[WARN\]|"
            r"\[MERGE_QC\]\[WARN\]\s+(?!(?:absence_|weather_cache_|fatigue_))",
            re.I,
        ),
        "その他のデータ結合/欠損警告",
        "前段データの結合漏れや欠損率上昇があります。入力CSVや結合キーを確認してください。",
    ),
    (
        re.compile(r"command not found", re.I),
        "コマンド未検出",
        "実行コマンド/パスが見つかりません。仮想環境やPATH設定を確認してください。",
    ),
    (
        re.compile(r"No such file or directory|FileNotFoundError", re.I),
        "入力ファイル不足",
        "必要なCSVや入力ファイルが不足しています。前段STEPの出力を確認してください。",
    ),
    (
        re.compile(r"Permission denied", re.I),
        "権限エラー",
        "ファイル読み書き権限を確認してください。",
    ),
    (
        re.compile(r"Traceback \(most recent call last\)|\bException\b|(?:^|\[)ERROR(?::|\])", re.I),
        "実行時エラー",
        "Python処理中に例外が発生しています。該当STEPの詳細ログを確認してください。",
    ),
    (
        re.compile(r"(?:HTTP(?:/\S+)?\s+|status(?: code)?[=: ]+)429\b|Too Many Requests|rate limit(?:ed|ing)?", re.I),
        "レート制限",
        "取得先のアクセス制限に達しています。時間をおいて再実行してください。",
    ),
]


def _escape(s: str) -> str:
    return html.escape(str(s), quote=True)


def parse_log(lines: list[str]) -> dict:
    steps = []
    current_step = None
    current_league = "共通"
    preflight = []
    warnings = []
    errors = []
    result_counts = Counter()
    issue_counter = Counter()
    issue_examples = {}
    merge_details = []

    step_line_re = re.compile(r"^\[STEP\]\s+([^:]+)\s*:\s*(.*)$")
    result_line_re = re.compile(r"^\[RESULT\]\s+([^:]+)\s*:\s*(OK|ERROR)\s*$")
    preflight_re = re.compile(r"^\[PREFLIGHT\]\s+([^:]+)\s*:\s*(OK|ERROR)\s*$")
    league_re = re.compile(r"^===\s+League:\s*([^/\s]+)")
    summary_count_re = re.compile(r"^(OK|ERROR):\s*\d+\s*$", re.I)
    merge_warn_re = re.compile(r"\[MERGE_QC\]\[WARN\]\s+([^:]+):\s+left_only=(\d+)(?:\s*->\s*(.+))?")
    merge_csv_re = re.compile(r"\[MERGE_QC\]\s+([^:]+):\s+left_only CSV保存\s*->\s*(.+)")

    for idx, line in enumerate(lines, start=1):
        raw = line.rstrip("\n")
        if not raw.strip():
            continue

        m = league_re.match(raw)
        if m:
            current_league = m.group(1).strip().lower()
            current_step = None
            continue

        m = step_line_re.match(raw)
        if m:
            step_name = m.group(1).strip()
            purpose = m.group(2).strip()
            current_step = {
                "league": current_league,
                "name": step_name,
                "purpose": purpose,
                "result": "UNKNOWN",
                "line": idx,
            }
            steps.append(current_step)
            continue

        m = result_line_re.match(raw)
        if m:
            step_name = m.group(1).strip()
            result = m.group(2).strip()
            if current_step is not None and current_step["name"] == step_name:
                current_step["result"] = result
            else:
                steps.append(
                    {
                        "league": current_league,
                        "name": step_name,
                        "purpose": "",
                        "result": result,
                        "line": idx,
                    }
                )
            result_counts[result] += 1
            continue

        m = preflight_re.match(raw)
        if m:
            preflight.append({"target": m.group(1).strip(), "status": m.group(2).strip(), "line": idx})
            continue

        is_warn = "[WARN]" in raw or re.search(r"\bWARN\b", raw) or RANKMOT_ISSUE_RE.search(raw)
        normalized_raw = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        summary_count_only = bool(summary_count_re.match(normalized_raw))
        is_error = (
            ("[ERROR]" in raw or "ERROR:" in raw or "FATAL:" in raw or "Traceback" in raw)
            and not summary_count_only
        )

        if is_warn:
            warnings.append((idx, raw))
        if is_error:
            errors.append((idx, raw))

        if summary_count_only:
            continue

        m = merge_warn_re.search(raw)
        if m:
            stage = m.group(1).strip()
            category = (
                "欠場" if stage.startswith("absence_") else
                "天気" if stage.startswith("weather_cache_") else
                "疲労/日程" if stage.startswith("fatigue_") else
                "その他"
            )
            merge_details.append({
                "league": current_league,
                "category": category,
                "stage": stage,
                "count": int(m.group(2)),
                "line": idx,
                "csv_path": (m.group(3) or "").strip(),
            })
        m = merge_csv_re.search(raw)
        if m:
            stage = m.group(1).strip()
            for detail in reversed(merge_details):
                if detail["league"] == current_league and detail["stage"] == stage:
                    detail["csv_path"] = m.group(2).strip()
                    break

        for pattern, issue_name, issue_hint in ISSUE_PATTERNS:
            if pattern.search(raw):
                issue_counter[(issue_name, issue_hint)] += 1
                issue_examples.setdefault((issue_name, issue_hint), []).append((idx, raw))

    if not result_counts:
        # run_batch_matchday.sh のように [RESULT] がないログ向け
        result_counts["ERROR"] = len(errors)
        result_counts["OK"] = 0 if errors else 1

    top_issues = []
    for (name, hint), count in issue_counter.most_common(10):
        top_issues.append(
            {
                "name": name,
                "hint": hint,
                "count": count,
                "example": issue_examples[(name, hint)][0] if issue_examples[(name, hint)] else None,
            }
        )

    return {
        "steps": steps,
        "preflight": preflight,
        "warnings": warnings,
        "errors": errors,
        "result_counts": result_counts,
        "top_issues": top_issues,
        "merge_details": merge_details,
    }


def _csv_examples(path: str, limit: int = 5) -> list[str]:
    if not path or not os.path.isfile(path):
        return []
    preferred = ["match_id", "home_team", "away_team", "datetime", "節"]
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            examples = []
            for row in reader:
                fields = [f"{key}={row.get(key, '')}" for key in preferred if row.get(key, "") not in (None, "")]
                examples.append(" / ".join(fields) if fields else str(row))
                if len(examples) >= limit:
                    break
            return examples
    except Exception:
        return []


def render_html(parsed: dict, log_path: str, title: str) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "ERRORあり" if parsed["errors"] or parsed["result_counts"].get("ERROR", 0) else ("要確認" if parsed["warnings"] or parsed["top_issues"] else "概ね正常")
    color = {"ERRORあり": "#c62828", "要確認": "#b26a00", "概ね正常": "#2e7d32"}[status]

    parts = []
    parts.append("<!doctype html>")
    parts.append("<html lang='ja'><head><meta charset='utf-8'>")
    parts.append(f"<title>{_escape(title)}</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;margin:24px;color:#111;}"
        "h1{margin:0 0 8px;} h2{margin:24px 0 8px;font-size:18px;}"
        ".meta{font-size:13px;color:#555;margin-bottom:8px;}"
        ".badge{display:inline-block;padding:4px 8px;border-radius:6px;font-weight:700;color:white;}"
        "table{border-collapse:collapse;width:100%;font-size:13px;}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top;}"
        "th{background:#f5f5f5;} .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}"
        ".warn{color:#b26a00;} .err{color:#c62828;font-weight:700;}"
        "</style></head><body>"
    )
    parts.append(f"<h1>{_escape(title)}</h1>")
    parts.append(f"<div class='meta'>生成日時: {_escape(now)}</div>")
    parts.append(f"<div class='meta'>ログ: <span class='mono'>{_escape(log_path)}</span></div>")
    parts.append(f"<div class='meta'>状態: <span class='badge' style='background:{color};'>{_escape(status)}</span></div>")

    rc = parsed["result_counts"]
    parts.append("<h2>サマリ</h2>")
    parts.append("<table><tbody>")
    parts.append(f"<tr><th>OKステップ数</th><td>{int(rc.get('OK', 0))}</td></tr>")
    parts.append(f"<tr><th>ERRORステップ数</th><td>{int(rc.get('ERROR', 0))}</td></tr>")
    parts.append(f"<tr><th>警告行数</th><td>{len(parsed['warnings'])}</td></tr>")
    parts.append(f"<tr><th>エラー行数</th><td>{len(parsed['errors'])}</td></tr>")
    parts.append("</tbody></table>")

    if parsed["preflight"]:
        parts.append("<h2>疎通チェック</h2><table><thead><tr><th>対象</th><th>状態</th><th>行</th></tr></thead><tbody>")
        for p in parsed["preflight"]:
            cls = "err" if p["status"] == "ERROR" else ""
            parts.append(
                f"<tr><td>{_escape(p['target'])}</td><td class='{cls}'>{_escape(p['status'])}</td><td>{p['line']}</td></tr>"
            )
        parts.append("</tbody></table>")

    if parsed["steps"]:
        parts.append("<h2>STEP結果</h2><table><thead><tr><th>リーグ</th><th>STEP</th><th>説明</th><th>結果</th><th>定義行</th></tr></thead><tbody>")
        for info in parsed["steps"]:
            name = info["name"]
            cls = "err" if info["result"] == "ERROR" else ""
            parts.append(
                f"<tr><td>{_escape(info['league'])}</td><td class='mono'>{_escape(name)}</td><td>{_escape(info['purpose'])}</td>"
                f"<td class='{cls}'>{_escape(info['result'])}</td><td>{info['line']}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append("<h2>人間向け警告（要確認）</h2>")
    if parsed["top_issues"]:
        parts.append("<table><thead><tr><th>警告分類</th><th>件数</th><th>意味</th><th>代表ログ</th></tr></thead><tbody>")
        for issue in parsed["top_issues"]:
            ex = issue["example"]
            ex_text = f"L{ex[0]}: {ex[1]}" if ex else ""
            parts.append(
                f"<tr><td class='err'>{_escape(issue['name'])}</td><td>{issue['count']}</td>"
                f"<td>{_escape(issue['hint'])}</td><td class='mono'>{_escape(ex_text)}</td></tr>"
            )
        parts.append("</tbody></table>")
    else:
        parts.append("<div>明確な警告パターンは検出されませんでした。</div>")

    if parsed.get("merge_details"):
        parts.append("<h2>結合・欠損警告の詳細</h2>")
        parts.append(
            "<table><thead><tr><th>分類</th><th>リーグ</th><th>処理</th><th>対象件数</th>"
            "<th>対象例（最大5件）</th><th>詳細CSV</th></tr></thead><tbody>"
        )
        for detail in parsed["merge_details"]:
            csv_path = detail.get("csv_path", "")
            examples = _csv_examples(csv_path)
            example_html = "<br>".join(_escape(x) for x in examples) if examples else "—"
            csv_html = (
                f"<a class='mono' href='file://{_escape(csv_path)}'>{_escape(csv_path)}</a>"
                if csv_path else "—"
            )
            parts.append(
                f"<tr><td>{_escape(detail['category'])}</td><td>{_escape(detail['league'])}</td>"
                f"<td class='mono'>{_escape(detail['stage'])}</td><td>{detail['count']}</td>"
                f"<td class='mono'>{example_html}</td><td>{csv_html}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append("<h2>エラー抜粋（先頭20件）</h2>")
    if parsed["errors"]:
        parts.append("<table><thead><tr><th>行</th><th>内容</th></tr></thead><tbody>")
        for ln, text in parsed["errors"][:20]:
            parts.append(f"<tr><td>{ln}</td><td class='mono err'>{_escape(text)}</td></tr>")
        parts.append("</tbody></table>")
    else:
        parts.append("<div>エラー行はありません。</div>")

    parts.append("</body></html>")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batchログを解析して警告HTMLを作成します。")
    ap.add_argument("--input", required=True, help="入力ログファイル")
    ap.add_argument("--output", required=True, help="出力HTMLファイル")
    ap.add_argument("--title", default="Batch Log Report", help="レポートタイトル")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[WARN] log not found: {args.input}")
        return 0

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    parsed = parse_log(lines)
    report = render_html(parsed, args.input, args.title)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[INFO] log report generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
