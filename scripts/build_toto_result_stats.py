"""Validate the archived toto results and build reusable round statistics."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/external_metrics/toto_results"
SOURCE = DIRECTORY / "observed_results_2006_2026_asof_20261017.csv"
DETAIL = DIRECTORY / "round_statistics_2006_2026_asof_20261017.csv"
HISTOGRAM = DIRECTORY / "result_count_distribution_2006_2026_asof_20261017.csv"
YEARLY = DIRECTORY / "year_summary_2006_2026_asof_20261017.csv"
MANIFEST = DIRECTORY / "manifest_2006_2026_asof_20261017.json"
CROSS_CHECK = DIRECTORY / "local_eval_cross_check_2006_2026_asof_20261017.csv"


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))
    rounds = set()
    details = []
    for row in source:
        round_id = row["round_id"]
        code = row["result_code"]
        year = int(row["result_year"])
        if round_id in rounds or not re.fullmatch(r"toto\d+", round_id):
            raise ValueError(f"duplicate/invalid round_id: {round_id}")
        if len(code) != 13 or not set(code) <= {"1", "0", "2", "＊"}:
            raise ValueError(f"invalid 13-match result: {round_id}={code}")
        if row["source_url"] != f"https://totomark.info/toto/year/{year}/":
            raise ValueError(f"source URL/year mismatch: {round_id}")
        rounds.add(round_id)
        counts = Counter(code)
        details.append({
            "round_id": round_id,
            "result_year": year,
            "result_code": code,
            "complete_13": int(counts["＊"] == 0),
            "cancelled_count": counts["＊"],
            "H_count": counts["1"],
            "D_count": counts["0"],
            "A_count": counts["2"],
            **{f"result_{i + 1:02d}": symbol for i, symbol in enumerate(code)},
            "source_url": row["source_url"],
        })
    fields = [
        "round_id", "result_year", "result_code", "complete_13", "cancelled_count",
        "H_count", "D_count", "A_count",
        *[f"result_{i:02d}" for i in range(1, 14)], "source_url",
    ]
    write_csv(DETAIL, fields, details)

    complete = [row for row in details if row["complete_13"]]
    distribution = []
    for symbol, label in (("H", "home_win"), ("D", "draw_or_other"), ("A", "away_win")):
        counts = Counter(row[f"{symbol}_count"] for row in complete)
        for count in range(14):
            distribution.append({
                "symbol": symbol,
                "meaning": label,
                "count_in_round": count,
                "rounds": counts[count],
                "denominator": len(complete),
                "fraction": counts[count] / len(complete),
            })
    write_csv(HISTOGRAM, ["symbol", "meaning", "count_in_round", "rounds", "denominator", "fraction"], distribution)

    by_year = defaultdict(list)
    for row in details:
        by_year[row["result_year"]].append(row)
    yearly = []
    for year, rows in sorted(by_year.items()):
        full = [row for row in rows if row["complete_13"]]
        yearly.append({
            "result_year": year,
            "listed_rounds": len(rows),
            "complete_13_rounds": len(full),
            "cancelled_rounds": len(rows) - len(full),
            "H_zero_rounds": sum(row["H_count"] == 0 for row in full),
            "D_zero_rounds": sum(row["D_count"] == 0 for row in full),
            "A_zero_rounds": sum(row["A_count"] == 0 for row in full),
        })
    write_csv(YEARLY, list(yearly[0]), yearly)

    # Keep an audit trail for the locally available evaluation rounds. A
    # mismatch is reported rather than changing either source automatically.
    by_round = {row["round_id"]: row for row in details}
    cross_checks = []
    for path in sorted((ROOT / "data/eval/toto_rounds").glob("toto*/actual_results.csv")):
        round_id = path.parent.name
        with path.open(newline="", encoding="utf-8-sig") as handle:
            local_rows = list(csv.DictReader(handle))
        local_code = "".join(row["result"] for row in sorted(local_rows, key=lambda row: int(row["match_no"])))
        external = by_round.get(round_id)
        external_code = external["result_code"] if external else ""
        cross_checks.append({
            "round_id": round_id,
            "external_code": external_code,
            "local_eval_code": local_code,
            "status": "match" if external_code == local_code else "mismatch" if external else "not_in_external_snapshot",
            "local_eval_file": str(path.relative_to(ROOT)),
            "external_source_url": external["source_url"] if external else "",
        })
    write_csv(CROSS_CHECK, ["round_id", "external_code", "local_eval_code", "status", "local_eval_file", "external_source_url"], cross_checks)

    manifest = {
        "asof_date": "2026-10-17",
        "source_name": "totomark year-by-year toto results",
        "source_index_url": "https://totomark.info/toto/",
        "source_type": "independent third-party results index",
        "source_years": [min(by_year), max(by_year)],
        "raw_file": str(SOURCE.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "round_statistics_file": str(DETAIL.relative_to(ROOT)),
        "result_count_distribution_file": str(HISTOGRAM.relative_to(ROOT)),
        "year_summary_file": str(YEARLY.relative_to(ROOT)),
        "local_eval_cross_check_file": str(CROSS_CHECK.relative_to(ROOT)),
        "local_eval_cross_check": dict(Counter(row["status"] for row in cross_checks)),
        "listed_rounds": len(details),
        "complete_13_rounds": len(complete),
        "excluded_cancelled_rounds": len(details) - len(complete),
        "zero_rounds": {symbol: sum(row[f"{symbol}_count"] == 0 for row in complete) for symbol in "HDA"},
        "inclusion_rule": "Only complete 13-character results made of 1, 0, and 2 enter frequency denominators; ＊ marks cancelled/unresolved matches.",
        "result_code_legend": {"1": "home win", "0": "draw or other", "2": "away win", "＊": "cancelled/unresolved"},
        "scope_note": "Includes domestic and overseas toto rounds; result_year is the year of the source results index.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("listed_rounds", "complete_13_rounds", "excluded_cancelled_rounds", "zero_rounds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
