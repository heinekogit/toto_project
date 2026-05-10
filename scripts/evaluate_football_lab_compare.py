#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "data" / "external_metrics" / "reports"

DIFF_METRICS = [
    ("flab_expected_for_xg_diff", "攻撃xG差"),
    ("flab_expected_against_xg_diff", "被xG差"),
    ("flab_chance_build_rate_diff", "チャンス構築率差"),
    ("flab_chance_allowed_build_rate_diff", "被チャンス構築率差"),
    ("flab_chance_shot_conversion_diff", "シュート成功率差"),
    ("flab_chance_allowed_shot_conversion_diff", "被シュート成功率差"),
    ("flab_possession_rate_diff", "保持率差"),
    ("flab_possession_attack_cbp_diff", "攻撃CBP差"),
]

POSITIVE_HOME_METRICS = {
    "flab_expected_for_xg_diff",
    "flab_chance_build_rate_diff",
    "flab_chance_shot_conversion_diff",
    "flab_possession_rate_diff",
    "flab_possession_attack_cbp_diff",
}

NEGATIVE_HOME_METRICS = {
    "flab_expected_against_xg_diff",
    "flab_chance_allowed_build_rate_diff",
    "flab_chance_allowed_shot_conversion_diff",
}


def to_float(value):
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt_num(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def expected_sign(metric_key):
    if metric_key in POSITIVE_HOME_METRICS:
        return 1
    if metric_key in NEGATIVE_HOME_METRICS:
        return -1
    raise KeyError(metric_key)


def sign(value, eps=1e-9):
    if value is None:
        return 0
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    total = len(rows)
    correct = sum((row.get("is_correct") or "").lower() == "true" for row in rows)
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
    }

    by_result = {}
    for result in ("H", "D", "A"):
        group = [r for r in rows if r.get("predicted_result") == result]
        if not group:
            continue
        group_correct = sum((r.get("is_correct") or "").lower() == "true" for r in group)
        by_result[result] = {
            "count": len(group),
            "correct": group_correct,
            "accuracy": group_correct / len(group),
        }
    summary["by_result"] = by_result

    correct_rows = [r for r in rows if (r.get("is_correct") or "").lower() == "true"]
    wrong_rows = [r for r in rows if (r.get("is_correct") or "").lower() != "true"]

    metric_stats = []
    for metric_key, metric_label in DIFF_METRICS:
        vals_correct = [to_float(r.get(metric_key)) for r in correct_rows]
        vals_wrong = [to_float(r.get(metric_key)) for r in wrong_rows]
        abs_correct = [abs(v) for v in vals_correct if v is not None]
        abs_wrong = [abs(v) for v in vals_wrong if v is not None]
        metric_stats.append({
            "metric_key": metric_key,
            "metric_label": metric_label,
            "mean_correct": mean(vals_correct),
            "mean_wrong": mean(vals_wrong),
            "mean_abs_correct": mean(abs_correct),
            "mean_abs_wrong": mean(abs_wrong),
        })
    summary["metric_stats"] = metric_stats

    align_rows = [r for r in rows if r.get("predicted_result") in ("H", "A")]
    directional = []
    for metric_key, metric_label in DIFF_METRICS:
        aligned = 0
        decided = 0
        aligned_correct = 0
        aligned_wrong = 0
        for row in align_rows:
            diff = to_float(row.get(metric_key))
            if diff is None:
                continue
            diff_sign = sign(diff) * expected_sign(metric_key)
            if diff_sign == 0:
                continue
            pred_sign = 1 if row.get("predicted_result") == "H" else -1
            is_aligned = diff_sign == pred_sign
            decided += 1
            if is_aligned:
                aligned += 1
                if (row.get("is_correct") or "").lower() == "true":
                    aligned_correct += 1
                else:
                    aligned_wrong += 1
        directional.append({
            "metric_key": metric_key,
            "metric_label": metric_label,
            "decided": decided,
            "aligned": aligned,
            "aligned_rate": (aligned / decided) if decided else None,
            "aligned_correct": aligned_correct,
            "aligned_wrong": aligned_wrong,
        })
    summary["directional"] = directional

    draw_rows = [r for r in rows if r.get("predicted_result") == "D"]
    draw_stats = []
    for metric_key, metric_label in DIFF_METRICS:
        vals = [abs(to_float(r.get(metric_key))) for r in draw_rows if to_float(r.get(metric_key)) is not None]
        vals_correct = [abs(to_float(r.get(metric_key))) for r in draw_rows if (r.get("is_correct") or "").lower() == "true" and to_float(r.get(metric_key)) is not None]
        vals_wrong = [abs(to_float(r.get(metric_key))) for r in draw_rows if (r.get("is_correct") or "").lower() != "true" and to_float(r.get(metric_key)) is not None]
        draw_stats.append({
            "metric_key": metric_key,
            "metric_label": metric_label,
            "mean_abs_draw_all": mean(vals),
            "mean_abs_draw_correct": mean(vals_correct),
            "mean_abs_draw_wrong": mean(vals_wrong),
        })
    summary["draw_stats"] = draw_stats

    contradictions = []
    for row in align_rows:
        contradiction_score = 0.0
        hits = []
        for metric_key, metric_label in DIFF_METRICS:
            diff = to_float(row.get(metric_key))
            if diff is None:
                continue
            normalized_sign = sign(diff) * expected_sign(metric_key)
            if normalized_sign == 0:
                continue
            pred_sign = 1 if row.get("predicted_result") == "H" else -1
            if normalized_sign != pred_sign:
                contradiction_score += abs(diff)
                hits.append(metric_label)
        if hits:
            contradictions.append({
                "match_id": row.get("match_id", ""),
                "datetime": row.get("datetime", ""),
                "home_team": row.get("home_team", ""),
                "away_team": row.get("away_team", ""),
                "predicted_result": row.get("predicted_result", ""),
                "actual_result": row.get("actual_result", ""),
                "is_correct": row.get("is_correct", ""),
                "contradiction_score": contradiction_score,
                "contradiction_metrics": ", ".join(hits),
            })
    contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
    summary["contradictions"] = contradictions[:10]
    return summary


def write_metric_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(compare_path, summary):
    lines = []
    lines.append(f"# Football LAB比較レポート")
    lines.append("")
    lines.append(f"- 入力: `{compare_path}`")
    lines.append(f"- 試合数: `{summary['total']}`")
    lines.append(f"- 的中数: `{summary['correct']}`")
    lines.append(f"- 的中率: `{fmt_num(summary['accuracy'] * 100, 1)}%`" if summary["accuracy"] is not None else "- 的中率: `-`")
    lines.append("")
    lines.append("## 予想結果別")
    lines.append("")
    for result in ("H", "D", "A"):
        item = summary["by_result"].get(result)
        if not item:
            continue
        lines.append(
            f"- `{result}`: {item['correct']}/{item['count']} ({fmt_num(item['accuracy'] * 100, 1)}%)"
        )
    lines.append("")
    lines.append("## 当たり/外れでの外部差分")
    lines.append("")
    lines.append("| 指標 | 正解平均 | 不正解平均 | 正解の絶対差平均 | 不正解の絶対差平均 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in summary["metric_stats"]:
        lines.append(
            f"| {row['metric_label']} | {fmt_num(row['mean_correct'])} | {fmt_num(row['mean_wrong'])} | {fmt_num(row['mean_abs_correct'])} | {fmt_num(row['mean_abs_wrong'])} |"
        )
    lines.append("")
    lines.append("## H/A予想と外部指標の整合率")
    lines.append("")
    lines.append("| 指標 | 判定対象数 | 整合数 | 整合率 | 整合して当たり | 整合して外れ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["directional"]:
        lines.append(
            f"| {row['metric_label']} | {row['decided']} | {row['aligned']} | {fmt_num((row['aligned_rate'] or 0) * 100, 1) if row['aligned_rate'] is not None else '-'}% | {row['aligned_correct']} | {row['aligned_wrong']} |"
        )
    lines.append("")
    lines.append("## D予想の外部差分の小ささ")
    lines.append("")
    lines.append("| 指標 | D全体の絶対差平均 | D的中の絶対差平均 | D外れの絶対差平均 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in summary["draw_stats"]:
        lines.append(
            f"| {row['metric_label']} | {fmt_num(row['mean_abs_draw_all'])} | {fmt_num(row['mean_abs_draw_correct'])} | {fmt_num(row['mean_abs_draw_wrong'])} |"
        )
    lines.append("")
    lines.append("## 外部指標と逆向きだった試合")
    lines.append("")
    for row in summary["contradictions"]:
        lines.append(
            f"- `{row['home_team']} vs {row['away_team']}` 予想=`{row['predicted_result']}` 実績=`{row['actual_result'] or '-'}` 的中=`{row['is_correct']}` 逆行指標=`{row['contradiction_metrics']}` スコア=`{fmt_num(row['contradiction_score'])}`"
        )
    lines.append("")
    lines.append("## 読み方")
    lines.append("")
    lines.append("- `正解の絶対差平均 < 不正解の絶対差平均` なら、その指標は予想の当たりと整合しやすいです。")
    lines.append("- `H/A整合率` が高い指標ほど、従来予想の方向感と噛み合っています。")
    lines.append("- `D的中の絶対差平均` が小さい指標は、引分け判定の補助候補です。")
    lines.append("- `逆向きだった試合` は、従来予想が外部指標と衝突していたカードです。見直し対象に向いています。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Football LAB比較CSVを集計して検証レポートを作る")
    parser.add_argument("--compare-csv", required=True)
    args = parser.parse_args()

    compare_path = Path(args.compare_csv)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(compare_path)
    summary = summarize(rows)

    stem = compare_path.stem
    metric_csv = REPORT_DIR / f"{stem}_metric_summary.csv"
    directional_csv = REPORT_DIR / f"{stem}_directional_summary.csv"
    draw_csv = REPORT_DIR / f"{stem}_draw_summary.csv"
    contradiction_csv = REPORT_DIR / f"{stem}_contradictions.csv"
    md_path = REPORT_DIR / f"{stem}.md"

    write_metric_csv(
        metric_csv,
        summary["metric_stats"],
        ["metric_key", "metric_label", "mean_correct", "mean_wrong", "mean_abs_correct", "mean_abs_wrong"],
    )
    write_metric_csv(
        directional_csv,
        summary["directional"],
        ["metric_key", "metric_label", "decided", "aligned", "aligned_rate", "aligned_correct", "aligned_wrong"],
    )
    write_metric_csv(
        draw_csv,
        summary["draw_stats"],
        ["metric_key", "metric_label", "mean_abs_draw_all", "mean_abs_draw_correct", "mean_abs_draw_wrong"],
    )
    write_metric_csv(
        contradiction_csv,
        summary["contradictions"],
        ["match_id", "datetime", "home_team", "away_team", "predicted_result", "actual_result", "is_correct", "contradiction_score", "contradiction_metrics"],
    )
    md_path.write_text(build_markdown(compare_path, summary), encoding="utf-8")

    print(f"report={md_path}")
    print(f"metric_summary={metric_csv}")
    print(f"directional_summary={directional_csv}")
    print(f"draw_summary={draw_csv}")
    print(f"contradictions={contradiction_csv}")


if __name__ == "__main__":
    main()
