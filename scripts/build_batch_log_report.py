#!/usr/bin/env python3
import argparse
import datetime as dt
import html
import os
import re
from collections import Counter


ISSUE_PATTERNS = [
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
        re.compile(r"\[MERGE_QC\]\[WARN\]|\[MISSING_QC\]\[WARN\]", re.I),
        "データ結合/欠損警告",
        "前段データの結合漏れや欠損率上昇があります。入力CSVやチーム名対応を確認してください。",
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
        re.compile(r"Traceback \(most recent call last\)|Exception|ERROR:", re.I),
        "実行時エラー",
        "Python処理中に例外が発生しています。該当STEPの詳細ログを確認してください。",
    ),
    (
        re.compile(r"\b429\b|Too Many Requests|rate limit(?:ed|ing)?", re.I),
        "レート制限",
        "取得先のアクセス制限に達しています。時間をおいて再実行してください。",
    ),
]


def _escape(s: str) -> str:
    return html.escape(str(s), quote=True)


def parse_log(lines: list[str]) -> dict:
    steps = {}
    current_step = None
    preflight = []
    warnings = []
    errors = []
    result_counts = Counter()
    issue_counter = Counter()
    issue_examples = {}

    step_line_re = re.compile(r"^\[STEP\]\s+([^:]+)\s*:\s*(.*)$")
    result_line_re = re.compile(r"^\[RESULT\]\s+([^:]+)\s*:\s*(OK|ERROR)\s*$")
    preflight_re = re.compile(r"^\[PREFLIGHT\]\s+([^:]+)\s*:\s*(OK|ERROR)\s*$")

    for idx, line in enumerate(lines, start=1):
        raw = line.rstrip("\n")
        if not raw.strip():
            continue

        m = step_line_re.match(raw)
        if m:
            step_name = m.group(1).strip()
            purpose = m.group(2).strip()
            current_step = step_name
            steps.setdefault(step_name, {"purpose": purpose, "result": "UNKNOWN", "line": idx})
            continue

        m = result_line_re.match(raw)
        if m:
            step_name = m.group(1).strip()
            result = m.group(2).strip()
            steps.setdefault(step_name, {"purpose": "", "result": result, "line": idx})
            steps[step_name]["result"] = result
            result_counts[result] += 1
            continue

        m = preflight_re.match(raw)
        if m:
            preflight.append({"target": m.group(1).strip(), "status": m.group(2).strip(), "line": idx})
            continue

        is_warn = "[WARN]" in raw or re.search(r"\bWARN\b", raw)
        # "ERROR: 0"（終了コード集計）は正常系扱い
        normalized_raw = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        error_zero_only = bool(re.match(r"^ERROR:\s*0$", normalized_raw, re.I))
        is_error = (
            ("[ERROR]" in raw or "ERROR:" in raw or "FATAL:" in raw or "Traceback" in raw)
            and not error_zero_only
        )

        if is_warn:
            warnings.append((idx, raw))
        if is_error:
            errors.append((idx, raw))

        if error_zero_only:
            continue

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
    }


def render_html(parsed: dict, log_path: str, title: str) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "ERRORあり" if parsed["errors"] or parsed["result_counts"].get("ERROR", 0) else "概ね正常"
    color = "#c62828" if status == "ERRORあり" else "#2e7d32"

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
        parts.append("<h2>STEP結果</h2><table><thead><tr><th>STEP</th><th>説明</th><th>結果</th><th>定義行</th></tr></thead><tbody>")
        for name, info in sorted(parsed["steps"].items(), key=lambda x: x[1]["line"]):
            cls = "err" if info["result"] == "ERROR" else ""
            parts.append(
                f"<tr><td class='mono'>{_escape(name)}</td><td>{_escape(info['purpose'])}</td>"
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
