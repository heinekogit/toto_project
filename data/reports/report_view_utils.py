import os
import glob
import json
import pandas as pd


def list_files(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    return files


def choose_file(files):
    if not files:
        print("対象ファイルが見つかりません。")
        return None
    auto = os.environ.get("AUTO_LATEST", "1") == "1"
    if auto:
        return max(files, key=lambda p: os.path.getmtime(p))
    for i, f in enumerate(files, 1):
        print(f"[{i}] {f}")
    while True:
        choice = input("読み込む番号を選択してください: ").strip()
        if not choice.isdigit():
            print("数字で入力してください。")
            continue
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return files[idx]
        print("範囲内の番号を選択してください。")


def build_description_row(columns, desc_map):
    descs = []
    for col in columns:
        descs.append(desc_map.get(col, ""))
    return descs


def write_html_table(df, title, desc_map, output_path):
    df = df.copy()
    columns = df.columns.tolist()
    desc_row = build_description_row(columns, desc_map)
    has_is_correct = "is_correct" in columns

    def _to_bool_or_none(v):
        if pd.isna(v):
            return None
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
        return None

    hit_summary = None
    if has_is_correct:
        flags = df["is_correct"].map(_to_bool_or_none)
        valid = flags.notna()
        total = int(valid.sum())
        hits = int((flags == True).sum()) if total > 0 else 0  # noqa: E712
        rate = (hits / total * 100.0) if total > 0 else None
        hit_summary = {"hits": hits, "total": total, "rate": rate}

    html = []
    html.append("<!doctype html>")
    html.append("<html lang='ja'>")
    html.append("<head>")
    html.append("<meta charset='utf-8'>")
    html.append(f"<title>{title}</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui, -apple-system, sans-serif; margin:24px;}")
    html.append("table{border-collapse:collapse; width:100%; font-size:12px;}")
    html.append("th,td{border:1px solid #ddd; padding:6px; text-align:left;}")
    html.append("thead th{background:#f5f5f5; position:sticky; top:0;}")
    html.append("tr.desc-row td{background:#fcfcfc; color:#666; font-size:11px;}")
    html.append("td.is-correct-true{color:#c1121f; font-weight:700;}")
    html.append(".summary{margin:8px 0 12px; font-size:13px;}")
    html.append(".summary b{color:#c1121f;}")
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")
    parent = os.path.basename(os.path.dirname(output_path))
    back_link = "../index.html" if parent == "html" else "index.html"
    html.append(f"<div style='margin-bottom:12px;'><a href='{back_link}'>← indexに戻る</a></div>")
    html.append(f"<h2>{title}</h2>")
    if hit_summary is not None:
        if hit_summary["rate"] is None:
            html.append("<div class='summary'>的中数: <b>0/0</b> / 的中率: <b>-</b></div>")
        else:
            html.append(
                f"<div class='summary'>的中数: <b>{hit_summary['hits']}/{hit_summary['total']}</b> / "
                f"的中率: <b>{hit_summary['rate']:.1f}%</b></div>"
            )
    html.append("<table>")
    html.append("<thead><tr>")
    for col in columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead>")
    html.append("<tbody>")
    html.append("<tr class='desc-row'>")
    for desc in desc_row:
        html.append(f"<td>{desc}</td>")
    html.append("</tr>")
    for _, row in df.iterrows():
        html.append("<tr>")
        for col in columns:
            val = row[col]
            text = "" if pd.isna(val) else str(val)
            if col == "is_correct" and _to_bool_or_none(val) is True:
                html.append(f"<td class='is-correct-true'>{text}</td>")
            else:
                html.append(f"<td>{text}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    html.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def write_html_table_grouped(df, title, desc_map, output_path, group_col="league", group_order=None):
    df = df.copy()
    if group_col not in df.columns:
        return write_html_table(df, title, desc_map, output_path)

    def _to_bool_or_none(v):
        if pd.isna(v):
            return None
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
        return None

    if group_order is None:
        group_order = ["J1", "J2", "J3", "ALL", "UNKNOWN"]
    seen = []
    for g in group_order:
        if g in set(df[group_col].astype(str)):
            seen.append(g)
    for g in df[group_col].astype(str):
        if g not in seen:
            seen.append(g)

    html = []
    html.append("<!doctype html>")
    html.append("<html lang='ja'>")
    html.append("<head>")
    html.append("<meta charset='utf-8'>")
    html.append(f"<title>{title}</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui, -apple-system, sans-serif; margin:24px;}")
    html.append("table{border-collapse:collapse; width:100%; font-size:12px; margin-bottom:20px;}")
    html.append("th,td{border:1px solid #ddd; padding:6px; text-align:left;}")
    html.append("thead th{background:#f5f5f5; position:sticky; top:0;}")
    html.append("tr.desc-row td{background:#fcfcfc; color:#666; font-size:11px;}")
    html.append("td.is-correct-true{color:#c1121f; font-weight:700;}")
    html.append(".summary{margin:8px 0 12px; font-size:13px;}")
    html.append(".summary b{color:#c1121f;}")
    html.append("h3{margin:18px 0 8px;}")
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")
    parent = os.path.basename(os.path.dirname(output_path))
    back_link = "../index.html" if parent == "html" else "index.html"
    html.append(f"<div style='margin-bottom:12px;'><a href='{back_link}'>← indexに戻る</a></div>")
    html.append(f"<h2>{title}</h2>")

    for grp in seen:
        sub = df[df[group_col].astype(str) == grp].copy()
        if sub.empty:
            continue
        columns = sub.columns.tolist()
        desc_row = build_description_row(columns, desc_map)
        has_is_correct = "is_correct" in columns

        html.append(f"<h3>{group_col.upper()}: {grp}</h3>")
        if has_is_correct:
            flags = sub["is_correct"].map(_to_bool_or_none)
            valid = flags.notna()
            total = int(valid.sum())
            hits = int((flags == True).sum()) if total > 0 else 0  # noqa: E712
            if total == 0:
                html.append("<div class='summary'>的中数: <b>0/0</b> / 的中率: <b>-</b></div>")
            else:
                rate = hits / total * 100.0
                html.append(f"<div class='summary'>的中数: <b>{hits}/{total}</b> / 的中率: <b>{rate:.1f}%</b></div>")

        html.append("<table>")
        html.append("<thead><tr>")
        for col in columns:
            html.append(f"<th>{col}</th>")
        html.append("</tr></thead>")
        html.append("<tbody>")
        html.append("<tr class='desc-row'>")
        for desc in desc_row:
            html.append(f"<td>{desc}</td>")
        html.append("</tr>")
        for _, row in sub.iterrows():
            html.append("<tr>")
            for col in columns:
                val = row[col]
                text = "" if pd.isna(val) else str(val)
                if col == "is_correct" and _to_bool_or_none(val) is True:
                    html.append(f"<td class='is-correct-true'>{text}</td>")
                else:
                    html.append(f"<td>{text}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")

    html.append("</body></html>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
