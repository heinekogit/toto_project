#!/usr/bin/env python3
"""Read-only inputs; independent v2.4 A/C allocation comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "data/eval/next_model/validation_v2/lab_xg_v0/predictions.csv"
ROUNDS = ROOT / "data/eval/toto_rounds"
RECALCULATED = (1644, 1645, 1647, 1648, 1650, 1653, 1654)
OPERATIONAL = (1660, 1663, 1664, 1665)
SYMBOLS = "HDA"
RESULT = {1: "H", 0: "D", 2: "A"}
JST = timezone(timedelta(hours=9))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked_rows(rows: list[dict], round_id: str) -> list[dict]:
    if len(rows) != 13 or {int(r["match_no"]) for r in rows} != set(range(1, 14)):
        raise ValueError(f"{round_id}: 13 distinct match_no values required")
    rows = sorted(rows, key=lambda r: int(r["match_no"]))
    for row in rows:
        p = [float(row[f"next_prob_{s}"]) for s in ("home", "draw", "away")]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or not math.isclose(sum(p), 1, abs_tol=1e-8):
            raise ValueError(f"{round_id} M{row['match_no']}: invalid probability")
        if int(row["actual_result"]) not in RESULT:
            raise ValueError(f"{round_id} M{row['match_no']}: invalid result")
    return rows


def inputs() -> list[tuple[str, str, bool, str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(VALIDATION):
        grouped[row["round_id"]].append(row)
    output = []
    for number in RECALCULATED:
        rid = f"toto{number}"
        rows = checked_rows(grouped[rid], rid)
        proven = all(r["prematch_snapshot_proven"].lower() == "true" for r in rows)
        output.append((rid, "recalculated", proven, "source_snapshot_flag", rows))
    for number in OPERATIONAL:
        rid = f"toto{number}"
        directory = ROUNDS / rid
        path = directory / "snapshot/next_predictions.csv"
        manifest = json.loads((directory / "snapshot/buyplan_engine_manifest.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["output_sha256"]["next_predictions.csv"]:
            raise ValueError(f"{rid}: generator hash mismatch")
        forecast = read_csv(path)
        actual = read_csv(directory / "actual_results.csv")
        if len(actual) != 13 or {int(r["match_no"]) for r in actual} != set(range(1, 14)):
            raise ValueError(f"{rid}: invalid actual result rows")
        actual_by_no = {int(r["match_no"]): r for r in actual}
        joined = []
        for row in forecast:
            result = actual_by_no[int(row["match_no"])]
            if (row["home_team"], row["away_team"]) != (result["home_team"], result["away_team"]):
                raise ValueError(f"{rid} M{row['match_no']}: team mismatch")
            joined.append({**row, "actual_result": result["result"]})
        rows = checked_rows(joined, rid)
        created = datetime.fromisoformat(manifest["created_at"])
        first_kickoff = min(datetime.fromisoformat(r["datetime"]).replace(tzinfo=JST) for r in actual)
        output.append((rid, "operational", created < first_kickoff, "generator_timestamp_and_csv_hash", rows))
    return output


def allocate(probabilities: list[float]) -> list[int]:
    expected = [p * 10 for p in probabilities]
    counts = [math.floor(x + 1e-10) for x in expected]
    remaining = 10 - sum(counts)
    if remaining < 0 or remaining > 3:
        raise ValueError("Invalid rounding remainder")
    for index in sorted(range(3), key=lambda i: (-(expected[i] - counts[i]), i))[:remaining]:
        counts[index] += 1
    assert sum(counts) == 10
    return counts


def choose(probabilities: list[float], initial: list[int], method: str) -> tuple[str, list[dict]]:
    remaining = initial.copy()
    picks = []
    trace = []
    for candidate in range(1, 11):
        n = 11 - candidate
        eligible = [i for i in range(3) if remaining[i] > 0 and (method != "C" or initial[i] > 0)]
        scores = {i: probabilities[i] * (remaining[i] / (n if method == "A" else initial[i])) for i in eligible}
        best = max(scores.values())
        # Exact equality only; no epsilon makes near-equal floats tie.
        tied = [i for i in eligible if scores[i] == best]
        chosen = min(tied)
        trace.append({"candidate": candidate, "method": method, "initial": initial.copy(),
                      "remaining_before": remaining.copy(), "n": n,
                      "q": [scores.get(i) for i in range(3)], "tie": len(tied) > 1,
                      "selected": SYMBOLS[chosen]})
        picks.append(SYMBOLS[chosen])
        remaining[chosen] -= 1
    assert remaining == [0, 0, 0]
    return "".join(picks), trace


def analyze_round(rid: str, category: str, prematch: bool, proof: str, rows: list[dict]):
    allocations = []
    plans = {"A": [[] for _ in range(10)], "C": [[] for _ in range(10)]}
    decision_rows = []
    match_rows = []
    actual = "".join(RESULT[int(r["actual_result"])] for r in rows)
    anchor = "".join(SYMBOLS[max(range(3), key=lambda i: (float(r[f'next_prob_{("home", "draw", "away")[i]}']), -i))] for r in rows)
    for row in rows:
        match_no = int(row["match_no"])
        p = [float(row[f"next_prob_{s}"]) for s in ("home", "draw", "away")]
        counts = allocate(p)
        allocations.append(counts)
        match_rows.append({"round_id": rid, "data_type": category, "match_no": match_no,
                           "home_team": row["home_team"], "away_team": row["away_team"],
                           "p_H": p[0], "p_D": p[1], "p_A": p[2],
                           "e_H": counts[0], "e_D": counts[1], "e_A": counts[2],
                           "actual": RESULT[int(row["actual_result"])], "candidate00": anchor[match_no - 1]})
        for method in ("A", "C"):
            picks, trace = choose(p, counts, method)
            for candidate, symbol in enumerate(picks):
                plans[method][candidate].append(symbol)
            for item in trace:
                decision_rows.append({"round_id": rid, "data_type": category, "method": method,
                                      "match_no": match_no, "candidate": item["candidate"],
                                      **{f"remaining_{s}": item["remaining_before"][i] for i, s in enumerate(SYMBOLS)},
                                      "n": item["n"],
                                      **{f"q_{s}": item["q"][i] for i, s in enumerate(SYMBOLS)},
                                      "exact_tie": item["tie"], "selected": item["selected"]})
    tickets = {}
    ticket_rows = []
    round_rows = []
    for method in ("A", "C"):
        strings = ["".join(ticket) for ticket in plans[method]]
        tickets[method] = strings
        hits = [sum(a == b for a, b in zip(ticket, actual)) for ticket in strings]
        draw_counts = [ticket.count("D") for ticket in strings]
        duplicates = sum(n * (n - 1) // 2 for n in Counter(strings).values())
        obeyed = all([sum(plans[method][j][i] == s for j in range(10)) for s in SYMBOLS] == allocations[i] for i in range(13))
        for j, ticket in enumerate(strings, 1):
            ticket_rows.append({"round_id": rid, "data_type": category, "prematch_confirmed": prematch,
                                "method": method, "candidate": f"{j:02d}", "symbols": ticket,
                                **{f"M{i:02d}": symbol for i, symbol in enumerate(ticket, 1)},
                                "draws": draw_counts[j - 1], "hits": hits[j - 1],
                                "matches_candidate00": ticket == anchor})
        round_rows.append({"round_id": rid, "data_type": category, "prematch_confirmed": prematch,
                           "prematch_proof": proof, "method": method, "best_hits": max(hits),
                           "average_hits": sum(hits) / 10, "ge10": max(hits) >= 10,
                           "ge11": max(hits) >= 11, "ge12": max(hits) >= 12, "eq13": max(hits) == 13,
                           "zero_draw_tickets": draw_counts.count(0), "draws_01_05": sum(draw_counts[:5]),
                           "draws_06_10": sum(draw_counts[5:]), "duplicate_pairs": duplicates,
                           "unique_tickets": len(set(strings)), "candidate00_matches": strings.count(anchor),
                           "allocation_obeyed": obeyed})
    shared = len(set(tickets["A"]) & set(tickets["C"]))
    best_a, best_c = (r["best_hits"] for r in round_rows)
    for result in round_rows:
        result["shared_unique_tickets"] = shared
        result["A_only_unique_tickets"] = len(set(tickets["A"]) - set(tickets["C"]))
        result["C_only_unique_tickets"] = len(set(tickets["C"]) - set(tickets["A"]))
        result["winner"] = "A" if best_a > best_c else "C" if best_c > best_a else "tie"
    if round_rows[0]["average_hits"] != round_rows[1]["average_hits"] or not all(r["allocation_obeyed"] for r in round_rows):
        raise AssertionError(f"{rid}: invariant failed")
    return match_rows, decision_rows, ticket_rows, round_rows, actual, anchor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data/reports/buyplan_v24_ac_comparison")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Will not overwrite: {args.out}")
    matches, decisions, tickets, rounds, differences = [], [], [], [], []
    source_files = [VALIDATION]
    for rid, category, prematch, proof, source in inputs():
        m, d, t, r, actual, anchor = analyze_round(rid, category, prematch, proof, source)
        matches.extend(m); decisions.extend(d); tickets.extend(t); rounds.extend(r)
        if category == "operational":
            source_files += [ROUNDS / rid / "snapshot/next_predictions.csv", ROUNDS / rid / "snapshot/buyplan_engine_manifest.json", ROUNDS / rid / "actual_results.csv"]
        if r[0]["winner"] != "tie":
            for row in m:
                no = row["match_no"]
                differences.append({"round_id": rid, "match_no": no, "home_team": row["home_team"],
                                    "away_team": row["away_team"], "actual": actual[no - 1],
                                    "A_symbols_01_10": "".join(x["symbols"][no - 1] for x in t if x["method"] == "A"),
                                    "C_symbols_01_10": "".join(x["symbols"][no - 1] for x in t if x["method"] == "C"),
                                    "allocation_HDA": f"{row['e_H']}/{row['e_D']}/{row['e_A']}"})
    args.out.mkdir(parents=True)
    for name, data in (("matches.csv", matches), ("decisions.csv", decisions),
                       ("tickets.csv", tickets), ("round_summary.csv", rounds)):
        write_csv(args.out / name, data)
    if differences:
        write_csv(args.out / "winner_round_match_details.csv", differences)
    groups = []
    for label, subset in (("all", rounds), ("recalculated", [r for r in rounds if r["data_type"] == "recalculated"]),
                          ("operational", [r for r in rounds if r["data_type"] == "operational"]),
                          ("prematch_confirmed", [r for r in rounds if r["prematch_confirmed"]])):
        for method in ("A", "C"):
            selected = [r for r in subset if r["method"] == method]
            n = len(selected)
            item = {"group": label, "method": method, "rounds": n,
                    "A_wins": sum(r["winner"] == "A" for r in selected),
                    "C_wins": sum(r["winner"] == "C" for r in selected),
                    "ties": sum(r["winner"] == "tie" for r in selected),
                    "mean_best_hits": sum(r["best_hits"] for r in selected) / n,
                    "zero_draw_tickets": sum(r["zero_draw_tickets"] for r in selected),
                    "duplicate_pairs": sum(r["duplicate_pairs"] for r in selected),
                    "shared_unique_tickets": sum(r["shared_unique_tickets"] for r in selected),
                    "candidate00_matches": sum(r["candidate00_matches"] for r in selected)}
            for key in ("ge10", "ge11", "ge12", "eq13"):
                item[key + "_rounds"] = sum(bool(r[key]) for r in selected)
                item[key + "_rate"] = item[key + "_rounds"] / n
            groups.append(item)
    write_csv(args.out / "group_summary.csv", groups)
    manifest = {"purpose": "v2.4 A/C independent comparison; no production changes",
                "rounds": [r["round_id"] for r in rounds if r["method"] == "A"],
                "tie_policy": "exact float equality; H then D then A; no epsilon",
                "rounding": "per-match largest remainder; H then D then A for exact fractional ties",
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
