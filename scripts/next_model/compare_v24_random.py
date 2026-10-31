#!/usr/bin/env python3
"""Independent fixed-count random placement experiment for BuyPlan v2.4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from compare_v24_ac import ROOT, SYMBOLS, RESULT, allocate, inputs, write_csv

AC_DIR = ROOT / "data/reports/buyplan_v24_ac_comparison"
DEFAULT_OUT = ROOT / "data/reports/buyplan_v24_random_comparison"
BASE_SEED = 24000000
REPETITIONS = 1000


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def distribution(values: list[int | float]) -> str:
    return json.dumps(dict(sorted(Counter(values).items())), ensure_ascii=False, sort_keys=True)


def mean(values):
    return statistics.mean(values)


def summarize(rows: list[dict], label: str, ac: dict) -> dict:
    n = len(rows)
    best = [int(r["best_hits"]) for r in rows]
    duplicates = [int(r["duplicate_pairs"]) for r in rows]
    zero = [int(r["zero_draw_tickets"]) for r in rows]
    anchor = [int(r["candidate00_matches"]) for r in rows]
    result = {"group": label, "sets": n, "rounds": len({r["round_id"] for r in rows}),
              "mean_best_hits": mean(best), "median_best_hits": statistics.median(best),
              "best_hits_distribution": distribution(best),
              "beats_A_rate": mean(int(int(r["best_hits"]) > ac[r["round_id"]]["A"]) for r in rows),
              "beats_C_rate": mean(int(int(r["best_hits"]) > ac[r["round_id"]]["C"]) for r in rows),
              "ties_A_rate": mean(int(int(r["best_hits"]) == ac[r["round_id"]]["A"]) for r in rows),
              "ties_C_rate": mean(int(int(r["best_hits"]) == ac[r["round_id"]]["C"]) for r in rows),
              "mean_duplicate_pairs": mean(duplicates), "duplicate_pairs_distribution": distribution(duplicates),
              "any_duplicate_rate": mean(int(x > 0) for x in duplicates),
              "mean_zero_draw_tickets": mean(zero), "zero_draw_tickets_distribution": distribution(zero),
              "mean_candidate00_matches": mean(anchor), "candidate00_matches_distribution": distribution(anchor),
              "any_candidate00_match_rate": mean(int(x > 0) for x in anchor),
              "mean_unique_tickets": mean(int(r["unique_tickets"]) for r in rows)}
    for threshold, name in ((10, "ge10"), (11, "ge11"), (12, "ge12"), (13, "eq13")):
        result[name + "_rate"] = mean(int(x >= threshold) for x in best)
        result[name + "_sets"] = sum(x >= threshold for x in best)
    for candidate in range(1, 11):
        result[f"mean_D_candidate{candidate:02d}"] = mean(int(r[f"D_candidate{candidate:02d}"]) for r in rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Will not overwrite: {args.out}")
    prior_path = AC_DIR / "round_summary.csv"
    prior = read_csv(prior_path)
    ac = defaultdict(dict)
    for row in prior:
        ac[row["round_id"]][row["method"]] = int(row["best_hits"])
    all_rows = []
    round_summaries = []
    sources = [prior_path]
    for round_index, (rid, category, prematch, proof, matches) in enumerate(inputs()):
        if set(ac[rid]) != {"A", "C"}:
            raise ValueError(f"{rid}: missing prior A/C result")
        probabilities = [[float(r[f"next_prob_{s}"]) for s in ("home", "draw", "away")] for r in matches]
        allocations = [allocate(p) for p in probabilities]
        actual = "".join(RESULT[int(r["actual_result"])] for r in matches)
        candidate00 = "".join(SYMBOLS[max(range(3), key=lambda i: (p[i], -i))] for p in probabilities)
        if category == "operational":
            directory = ROOT / "data/eval/toto_rounds" / rid
            sources.extend([directory / "snapshot/next_predictions.csv", directory / "actual_results.csv",
                            directory / "snapshot/buyplan_engine_manifest.json"])
        for repetition in range(REPETITIONS):
            seed = BASE_SEED + round_index * REPETITIONS + repetition
            rng = random.Random(seed)
            tickets = [[] for _ in range(10)]
            for counts in allocations:
                # A uniform permutation of a fixed multiset is exactly sequential
                # sampling with probability equal to each symbol's remaining count.
                symbols = [symbol for symbol, count in zip(SYMBOLS, counts) for _ in range(count)]
                rng.shuffle(symbols)
                for candidate, symbol in enumerate(symbols):
                    tickets[candidate].append(symbol)
            strings = ["".join(ticket) for ticket in tickets]
            obeyed = all([sum(ticket[i] == symbol for ticket in strings) for symbol in SYMBOLS] == allocations[i]
                         for i in range(13))
            if not obeyed:
                raise AssertionError(f"{rid}: allocation violation")
            hits = [sum(s == a for s, a in zip(ticket, actual)) for ticket in strings]
            counts = Counter(strings)
            draw_counts = [ticket.count("D") for ticket in strings]
            all_rows.append({"round_id": rid, "data_type": category, "prematch_confirmed": prematch,
                             "prematch_proof": proof, "repetition": repetition + 1, "seed": seed,
                             "best_hits": max(hits), "average_hits": sum(hits) / 10,
                             "ge10": max(hits) >= 10, "ge11": max(hits) >= 11,
                             "ge12": max(hits) >= 12, "eq13": max(hits) == 13,
                             "zero_draw_tickets": draw_counts.count(0),
                             "draws_01_05": sum(draw_counts[:5]), "draws_06_10": sum(draw_counts[5:]),
                             "duplicate_pairs": sum(n * (n - 1) // 2 for n in counts.values()),
                             "unique_tickets": len(counts), "candidate00_matches": strings.count(candidate00),
                             "allocation_obeyed": obeyed,
                             **{f"D_candidate{i:02d}": draw_counts[i - 1] for i in range(1, 11)},
                             **{f"hits_candidate{i:02d}": hits[i - 1] for i in range(1, 11)},
                             **{f"ticket{i:02d}": strings[i - 1] for i in range(1, 11)}})
        selected = all_rows[-REPETITIONS:]
        round_summaries.append(summarize(selected, rid, ac))
    if len(all_rows) != 11 * REPETITIONS:
        raise AssertionError("Unexpected set count")
    args.out.mkdir(parents=True)
    write_csv(args.out / "sets.csv", all_rows)
    write_csv(args.out / "round_summary.csv", round_summaries)
    groups = []
    for label, rows in (("all", all_rows),
                        ("recalculated", [r for r in all_rows if r["data_type"] == "recalculated"]),
                        ("operational", [r for r in all_rows if r["data_type"] == "operational"]),
                        ("prematch_confirmed", [r for r in all_rows if r["prematch_confirmed"]])):
        groups.append(summarize(rows, label, ac))
    write_csv(args.out / "group_summary.csv", groups)
    manifest = {"method": "uniform permutation of each match's fixed H/D/A multiset",
                "equivalent_sampling": "sequential probability = remaining symbol count / remaining candidates",
                "base_seed": BASE_SEED, "repetitions_per_round": REPETITIONS,
                "seed_formula": "base_seed + round_index * 1000 + zero_based_repetition",
                "round_order": [rid for rid, *_ in inputs()],
                "random_generator": "Python random.Random MT19937; shuffle",
                "outcome_selection": "all 1000 sets included; no best-seed selection",
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sources}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
