import os
import json
from report_view_utils import list_files, choose_file, load_json


REPORT_DIR = os.path.abspath(os.path.dirname(__file__))
HTML_DIR = os.path.join(REPORT_DIR, "html")
PATTERNS = [
    os.path.join(REPORT_DIR, "report_*.json"),
]


def render_section(title, items):
    html = [f"<h3>{title}</h3><ul>"]
    for k, v in items.items():
        html.append(f"<li><b>{k}</b>: {v}</li>")
    html.append("</ul>")
    return "\n".join(html)


def main():
    files = list_files(PATTERNS)
    path = choose_file(files)
    if not path:
        return

    report = load_json(path)
    title = f"Report: {os.path.basename(path)}"
    os.makedirs(HTML_DIR, exist_ok=True)
    output_path = os.path.join(HTML_DIR, "report_view.html")

    html = []
    html.append("<!doctype html>")
    html.append("<html lang='ja'>")
    html.append("<head>")
    html.append("<meta charset='utf-8'>")
    html.append(f"<title>{title}</title>")
    html.append("<style>body{font-family:system-ui, -apple-system, sans-serif; margin:24px;} ul{line-height:1.6;}</style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<div style='margin-bottom:12px;'><a href='../index.html'>← indexに戻る</a></div>")
    html.append(f"<h2>{title}</h2>")

    html.append(render_section("入力ファイル", report.get("inputs", {})))
    html.append(render_section("パラメータ", report.get("parameters", {})))
    html.append(render_section("サマリ", report.get("summary", {})))

    html.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"HTML出力: {output_path}（元ファイル: {path}）")


if __name__ == "__main__":
    main()
