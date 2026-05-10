#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "external_metrics" / "trial_flags"

A_METRICS = [
    {
        "key": "flab_chance_allowed_shot_conversion_diff",
        "label": "被シュート成功率差",
        "home_sign": -1.0,
        "neutral_eps": 0.75,
        "strong_eps": 1.50,
    },
    {
        "key": "flab_chance_shot_conversion_diff",
        "label": "シュート成功率差",
        "home_sign": 1.0,
        "neutral_eps": 0.75,
        "strong_eps": 1.50,
    },
]


def to_float(value):
    text = str(value or "").strip()
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


def fmt_num(value, digits=3):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classify_row(row):
    predicted = str(row.get("predicted_result") or "").strip().upper()
    metrics = []
    for spec in A_METRICS:
        raw_diff = to_float(row.get(spec["key"]))
        if raw_diff is None:
            continue
        edge_home = raw_diff * float(spec["home_sign"])
        abs_edge = abs(edge_home)
        edge_sign = 0 if abs_edge < spec["neutral_eps"] else sign(edge_home)
        metrics.append(
            {
                "spec": spec,
                "raw_diff": raw_diff,
                "edge_home": edge_home,
                "abs_edge": abs_edge,
                "edge_sign": edge_sign,
                "strong": abs_edge >= spec["strong_eps"],
            }
        )

    if not metrics:
        return {
            "flab_trial_flag": "",
            "flab_trial_score": "",
            "flab_trial_reason": "A指標欠損",
            "flab_trial_available_metrics": 0,
            "flab_trial_agree_count": 0,
            "flab_trial_contradict_count": 0,
            "flab_trial_strong_contradict_count": 0,
            "flab_trial_avg_abs_edge": "",
            "flab_trial_home_edge_allowed_shot_conversion": "",
            "flab_trial_home_edge_shot_conversion": "",
        }

    avg_abs = mean([m["abs_edge"] for m in metrics])
    reasons = []

    if predicted in {"H", "A"}:
        pred_sign = 1 if predicted == "H" else -1
        agree = sum(1 for m in metrics if m["edge_sign"] == pred_sign)
        contradict = sum(1 for m in metrics if m["edge_sign"] == -pred_sign)
        strong_contradict = sum(1 for m in metrics if m["edge_sign"] == -pred_sign and m["strong"])
        if strong_contradict >= 1 or (contradict >= 1 and agree == 0):
            flag = "CAUTION"
        elif agree >= 1 and contradict == 0:
            flag = "GO"
        else:
            flag = "HOLD"
        raw_score = mean(
            [
                pred_sign * m["edge_home"]
                for m in metrics
            ]
        )
        for m in metrics:
            direction = "H優位" if m["edge_sign"] > 0 else ("A優位" if m["edge_sign"] < 0 else "拮抗")
            status = "整合" if m["edge_sign"] == pred_sign else ("逆行" if m["edge_sign"] == -pred_sign else "中立")
            reasons.append(f"{m['spec']['label']}={direction}({status},{fmt_num(m['edge_home'], 2)})")
    elif predicted == "D":
        non_neutral = [m for m in metrics if m["edge_sign"] != 0]
        same_dir = len({m["edge_sign"] for m in non_neutral}) == 1 if non_neutral else False
        strong_same_dir = same_dir and len(non_neutral) == len(metrics) and all(m["strong"] for m in non_neutral)
        flag = "CAUTION" if strong_same_dir else "HOLD"
        raw_score = -float(avg_abs or 0.0)
        for m in metrics:
            direction = "H優位" if m["edge_sign"] > 0 else ("A優位" if m["edge_sign"] < 0 else "拮抗")
            reasons.append(f"{m['spec']['label']}={direction}({fmt_num(m['edge_home'], 2)})")
    else:
        flag = ""
        raw_score = None
        reasons.append("predicted_result欠損")
        agree = contradict = strong_contradict = 0

    if predicted not in {"H", "A"}:
        agree = contradict = strong_contradict = 0

    edge_map = {
        "flab_trial_home_edge_allowed_shot_conversion": "",
        "flab_trial_home_edge_shot_conversion": "",
    }
    for m in metrics:
        if m["spec"]["key"] == "flab_chance_allowed_shot_conversion_diff":
            edge_map["flab_trial_home_edge_allowed_shot_conversion"] = fmt_num(m["edge_home"], 3)
        elif m["spec"]["key"] == "flab_chance_shot_conversion_diff":
            edge_map["flab_trial_home_edge_shot_conversion"] = fmt_num(m["edge_home"], 3)

    return {
        "flab_trial_flag": flag,
        "flab_trial_score": fmt_num(raw_score, 3) if raw_score is not None else "",
        "flab_trial_reason": "; ".join(reasons),
        "flab_trial_available_metrics": len(metrics),
        "flab_trial_agree_count": agree,
        "flab_trial_contradict_count": contradict,
        "flab_trial_strong_contradict_count": strong_contradict,
        "flab_trial_avg_abs_edge": fmt_num(avg_abs, 3),
        **edge_map,
    }


def build_rows(compare_rows):
    out = []
    for row in compare_rows:
        base = {}
        for col in ["league", "節", "match_id", "datetime", "home_team", "away_team", "predicted_result"]:
            if col in row:
                base[col] = row.get(col, "")
        base.update(classify_row(row))
        out.append(base)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "league",
        "節",
        "match_id",
        "datetime",
        "home_team",
        "away_team",
        "predicted_result",
        "flab_trial_flag",
        "flab_trial_score",
        "flab_trial_reason",
        "flab_trial_available_metrics",
        "flab_trial_agree_count",
        "flab_trial_contradict_count",
        "flab_trial_strong_contradict_count",
        "flab_trial_avg_abs_edge",
        "flab_trial_home_edge_allowed_shot_conversion",
        "flab_trial_home_edge_shot_conversion",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_path(compare_csv):
    stem = compare_csv.stem.replace("_football_lab_compare", "_football_lab_trial_flags")
    return OUTPUT_DIR / f"{stem}.csv"


def main():
    parser = argparse.ArgumentParser(description="Football LAB比較CSVから trial flag 補足CSVを作る")
    parser.add_argument("--compare-csv", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    compare_csv = Path(args.compare_csv)
    out_path = Path(args.out) if args.out else default_output_path(compare_csv)
    rows = build_rows(read_rows(compare_csv))
    write_csv(out_path, rows)
    print(f"rows={len(rows)}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()
