#!/usr/bin/env python3
import argparse
import csv
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


def sign(value, eps=1e-9):
    if value is None:
        return 0
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def expected_sign(metric_key):
    if metric_key in POSITIVE_HOME_METRICS:
        return 1
    if metric_key in NEGATIVE_HOME_METRICS:
        return -1
    raise KeyError(metric_key)


def normalize_score(value, lo, hi):
    if value is None:
        return None
    if hi <= lo:
        return 1.0
    return (value - lo) / (hi - lo)


def fmt_num(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def settled_rows(rows):
    return [r for r in rows if (r.get("actual_result") or "").strip() and (r.get("is_correct") or "").strip()]


def build_metric_stats(rows):
    out = []
    correct_rows = [r for r in rows if (r.get("is_correct") or "").lower() == "true"]
    wrong_rows = [r for r in rows if (r.get("is_correct") or "").lower() != "true"]
    align_rows = [r for r in rows if r.get("predicted_result") in ("H", "A")]
    draw_rows = [r for r in rows if r.get("predicted_result") == "D"]

    for key, label in DIFF_METRICS:
        vals_correct = [to_float(r.get(key)) for r in correct_rows]
        vals_wrong = [to_float(r.get(key)) for r in wrong_rows]
        abs_correct = [abs(v) for v in vals_correct if v is not None]
        abs_wrong = [abs(v) for v in vals_wrong if v is not None]
        mean_abs_correct = mean(abs_correct)
        mean_abs_wrong = mean(abs_wrong)

        decided = 0
        aligned = 0
        for row in align_rows:
            diff = to_float(row.get(key))
            if diff is None:
                continue
            normalized_sign = sign(diff) * expected_sign(key)
            if normalized_sign == 0:
                continue
            pred_sign = 1 if row.get("predicted_result") == "H" else -1
            decided += 1
            if normalized_sign == pred_sign:
                aligned += 1
        aligned_rate = (aligned / decided) if decided else None

        draw_all = [abs(to_float(r.get(key))) for r in draw_rows if to_float(r.get(key)) is not None]
        draw_correct = [
            abs(to_float(r.get(key)))
            for r in draw_rows
            if (r.get("is_correct") or "").lower() == "true" and to_float(r.get(key)) is not None
        ]
        draw_wrong = [
            abs(to_float(r.get(key)))
            for r in draw_rows
            if (r.get("is_correct") or "").lower() != "true" and to_float(r.get(key)) is not None
        ]
        mean_draw_correct = mean(draw_correct)
        mean_draw_wrong = mean(draw_wrong)
        draw_gap = None
        if mean_draw_correct is not None and mean_draw_wrong is not None:
            draw_gap = mean_draw_wrong - mean_draw_correct

        out.append({
            "metric_key": key,
            "metric_label": label,
            "mean_abs_correct": mean_abs_correct,
            "mean_abs_wrong": mean_abs_wrong,
            "separation_gap": None if mean_abs_correct is None or mean_abs_wrong is None else abs(mean_abs_wrong - mean_abs_correct),
            "preferred_pattern": (
                "correct_abs_smaller"
                if mean_abs_correct is not None and mean_abs_wrong is not None and mean_abs_correct < mean_abs_wrong
                else "correct_abs_larger"
            ),
            "aligned_rate": aligned_rate,
            "aligned": aligned,
            "decided": decided,
            "mean_abs_draw_all": mean(draw_all),
            "mean_abs_draw_correct": mean_draw_correct,
            "mean_abs_draw_wrong": mean_draw_wrong,
            "draw_gap": draw_gap,
        })
    return out


def assign_ranks(metric_rows):
    aligned_values = [r["aligned_rate"] for r in metric_rows if r["aligned_rate"] is not None]
    sep_values = [r["separation_gap"] for r in metric_rows if r["separation_gap"] is not None]
    draw_values = [r["draw_gap"] for r in metric_rows if r["draw_gap"] is not None]
    aligned_lo, aligned_hi = (min(aligned_values), max(aligned_values)) if aligned_values else (0.0, 1.0)
    sep_lo, sep_hi = (min(sep_values), max(sep_values)) if sep_values else (0.0, 1.0)
    draw_lo, draw_hi = (min(draw_values), max(draw_values)) if draw_values else (0.0, 1.0)

    for row in metric_rows:
        row["aligned_score"] = normalize_score(row["aligned_rate"], aligned_lo, aligned_hi)
        row["separation_score"] = normalize_score(row["separation_gap"], sep_lo, sep_hi)
        row["draw_score"] = normalize_score(row["draw_gap"], draw_lo, draw_hi) if draw_values else None
        available = [row["aligned_score"], row["separation_score"]]
        if row["draw_score"] is not None:
            available.append(row["draw_score"])
        row["overall_score"] = mean(available)

    metric_rows.sort(key=lambda x: (x["overall_score"] is None, -(x["overall_score"] or -999)))
    n = len(metric_rows)
    for i, row in enumerate(metric_rows, 1):
        row["overall_rank"] = i
        if i <= max(2, n // 3):
            row["tier"] = "A"
        elif i <= max(5, (2 * n) // 3):
            row["tier"] = "B"
        else:
            row["tier"] = "C"
    return metric_rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rounds, total_rows, metric_rows):
    lines = []
    lines.append("# Football LAB指標 暫定ランク")
    lines.append("")
    lines.append(f"- 対象節: `{', '.join(rounds)}`")
    lines.append(f"- 評価対象試合数: `{total_rows}`")
    lines.append(f"- 対象指標数: `{len(metric_rows)}`")
    lines.append("")
    lines.append("## 暫定ランク")
    lines.append("")
    lines.append("| Rank | Tier | 指標 | 総合スコア | 方向整合率 | 分離ギャップ | Dギャップ | 備考 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in metric_rows:
        lines.append(
            f"| {row['overall_rank']} | {row['tier']} | {row['metric_label']} | {fmt_num(row['overall_score'])} | {fmt_num((row['aligned_rate'] or 0) * 100, 1) if row['aligned_rate'] is not None else '-'}% | {fmt_num(row['separation_gap'])} | {fmt_num(row['draw_gap'])} | {row['preferred_pattern']} |"
        )
    lines.append("")
    lines.append("## 解釈")
    lines.append("")
    lines.append("- `方向整合率` は H/A 予想と外部指標の向きが一致した率です。")
    lines.append("- `分離ギャップ` は当たり/外れで外部差分の絶対値がどれだけ分かれたかです。")
    lines.append("- `Dギャップ` は D 的中時の差が D 外れ時より小さいほど高くなります。")
    lines.append("- Tier `A` は暫定採用候補、`B` は保留、`C` は低優先です。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="実績が入った比較CSVだけでFootball LAB指標の暫定ランクを作る")
    parser.add_argument("--glob", default="data/external_metrics/*football_lab_compare_*.csv")
    parser.add_argument("--prefix", default="football_lab_metric_ranking")
    args = parser.parse_args()

    compare_paths = sorted(BASE_DIR.glob(args.glob))
    usable = []
    all_rows = []
    for path in compare_paths:
        rows = settled_rows(read_rows(path))
        if not rows:
            continue
        usable.append(path)
        all_rows.extend(rows)

    metric_rows = assign_ranks(build_metric_stats(all_rows))
    rounds = [p.stem.replace("_football_lab_compare_20260324_j1", "") for p in usable]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / f"{args.prefix}.csv"
    md_path = REPORT_DIR / f"{args.prefix}.md"
    write_csv(
        csv_path,
        metric_rows,
        [
            "overall_rank",
            "tier",
            "metric_key",
            "metric_label",
            "overall_score",
            "aligned_rate",
            "aligned",
            "decided",
            "separation_gap",
            "preferred_pattern",
            "mean_abs_correct",
            "mean_abs_wrong",
            "draw_gap",
            "mean_abs_draw_all",
            "mean_abs_draw_correct",
            "mean_abs_draw_wrong",
            "aligned_score",
            "separation_score",
            "draw_score",
        ],
    )
    md_path.write_text(build_markdown(rounds, len(all_rows), metric_rows), encoding="utf-8")

    print(f"usable_rounds={len(usable)}")
    for path in usable:
        print(f"round={path}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()
