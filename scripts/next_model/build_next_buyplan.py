#!/usr/bin/env python3
"""Build BuyPlan portfolios from joint H/D/A probabilities.

The maximum-probability ticket/frontier description applies to probability_v0.
Official v2.3 uses proportional counts, then graded placement scores; its
ticket numbers are not forecast-probability ranks. The >=10-hit swaps remain
available only for the v2.2 and Anchor shadow strategies.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT
from draw_filter import classify_draw_v2


DEFAULT_INPUT = ROOT / "data/eval/next_model/validation_v2/lab_xg_v0/predictions.csv"
DEFAULT_OUT = ROOT / "data/eval/next_model/validation_v2/next_buyplan_v0"
SYMBOL_TO_RESULT = {"H": 1, "D": 0, "A": 2}


def ranked_outcomes(row) -> list[tuple[str, float]]:
    values = [
        ("H", float(row["next_prob_home"])),
        ("D", float(row["next_prob_draw"])),
        ("A", float(row["next_prob_away"])),
    ]
    return sorted(values, key=lambda item: (-item[1], "HDA".index(item[0])))


def top_joint_candidates(frame: pd.DataFrame, limit: int = 5000) -> list[dict]:
    ranked = [ranked_outcomes(row) for _, row in frame.iterrows()]

    def candidate(indices: tuple[int, ...]) -> dict:
        symbols = tuple(ranked[i][rank][0] for i, rank in enumerate(indices))
        log_probability = sum(math.log(ranked[i][rank][1]) for i, rank in enumerate(indices))
        return {"indices": indices, "symbols": symbols, "log_probability": log_probability}

    start = tuple(0 for _ in ranked)
    first = candidate(start)
    heap = [(-first["log_probability"], start)]
    seen = {start}
    output = []
    while heap and len(output) < limit:
        _, indices = heapq.heappop(heap)
        current = candidate(indices)
        output.append(current)
        for position, rank in enumerate(indices):
            if rank >= 2:
                continue
            neighbor = list(indices)
            neighbor[position] += 1
            key = tuple(neighbor)
            if key in seen:
                continue
            seen.add(key)
            item = candidate(key)
            heapq.heappush(heap, (-item["log_probability"], key))
    return output


def select_tickets(
    candidates: list[dict], count: int = 10, max_same_picks: int = 11
) -> list[dict]:
    if len(candidates) < count:
        raise ValueError("not enough joint candidates")
    selected = [candidates[0]]
    remaining = candidates[1:]
    while len(selected) < count:
        eligible = [
            item for item in remaining
            if max(
                sum(a == b for a, b in zip(item["symbols"], chosen["symbols"]))
                for chosen in selected
            ) <= max_same_picks
        ]
        if not eligible:
            raise ValueError(
                f"candidate frontier cannot satisfy max_same_picks={max_same_picks}"
            )
        # Candidates are already sorted by joint probability, so diversity is
        # a guard only; it never becomes a separate symbol quota/objective.
        best = eligible[0]
        best = dict(best)
        best["selection_score"] = best["log_probability"]
        selected.append(best)
        remaining.remove(next(item for item in remaining if item["indices"] == best["indices"]))
    selected[0] = dict(selected[0], selection_score=selected[0]["log_probability"])
    return selected


def select_tickets_coverage(
    candidates: list[dict], ranked: list[list[tuple[str, float]]], count: int = 10,
    probability_floor_ratio: float = 0.50, max_same_picks: int = 11,
) -> list[dict]:
    """Maximize probability-weighted new coverage inside a joint-probability floor."""
    selected = [dict(candidates[0], selection_score=candidates[0]["log_probability"])]
    floor = candidates[0]["log_probability"] + math.log(probability_floor_ratio)
    remaining = [item for item in candidates[1:] if item["log_probability"] >= floor]
    covered = [{selected[0]["symbols"][index]} for index in range(len(ranked))]
    probability_lookup = [dict(values) for values in ranked]
    while len(selected) < count:
        eligible = [
            item for item in remaining
            if max(
                sum(a == b for a, b in zip(item["symbols"], chosen["symbols"]))
                for chosen in selected
            ) <= max_same_picks
        ]
        if not eligible:
            raise ValueError(
                "candidate frontier cannot satisfy coverage selection constraints: "
                f"floor_ratio={probability_floor_ratio} max_same_picks={max_same_picks}"
            )

        def coverage_gain(item):
            return sum(
                probability_lookup[index][symbol]
                for index, symbol in enumerate(item["symbols"])
                if symbol not in covered[index]
            )

        best = max(eligible, key=lambda item: (coverage_gain(item), item["log_probability"]))
        best = dict(best, selection_score=coverage_gain(best))
        selected.append(best)
        remaining = [item for item in remaining if item["indices"] != best["indices"]]
        for index, symbol in enumerate(best["symbols"]):
            covered[index].add(symbol)
    return selected


def extend_tickets_coverage(
    candidates: list[dict], ranked: list[list[tuple[str, float]]],
    selected: list[dict], total_count: int,
    probability_floor_ratio: float = 0.35, max_same_picks: int = 11,
) -> list[dict]:
    """Keep official tickets unchanged and append probability-bounded shadow tickets."""
    selected = [dict(item) for item in selected]
    if total_count <= len(selected):
        return selected[:total_count]
    floor = candidates[0]["log_probability"] + math.log(probability_floor_ratio)
    selected_indices = {item["indices"] for item in selected}
    remaining = [
        item for item in candidates
        if item["log_probability"] >= floor and item["indices"] not in selected_indices
    ]
    covered = [{ticket["symbols"][i] for ticket in selected} for i in range(len(ranked))]
    probability_lookup = [dict(values) for values in ranked]
    while len(selected) < total_count:
        eligible = [
            item for item in remaining
            if max(
                sum(a == b for a, b in zip(item["symbols"], chosen["symbols"]))
                for chosen in selected
            ) <= max_same_picks
        ]
        if not eligible:
            raise ValueError(
                "shadow candidate frontier exhausted: "
                f"floor_ratio={probability_floor_ratio} total_count={total_count}"
            )

        def coverage_gain(item):
            return sum(
                probability_lookup[i][symbol]
                for i, symbol in enumerate(item["symbols"])
                if symbol not in covered[i]
            )

        best = max(eligible, key=lambda item: (coverage_gain(item), item["log_probability"]))
        selected.append(dict(best, selection_score=coverage_gain(best)))
        remaining = [item for item in remaining if item["indices"] != best["indices"]]
        for i, symbol in enumerate(best["symbols"]):
            covered[i].add(symbol)
    return selected


def build_shadow_round(
    frame: pd.DataFrame, shadow_count: int = 5, probability_floor_ratio: float = 0.35,
    official_plan: pd.DataFrame | None = None,
    strategy: str = "coverage_v1",
) -> tuple[pd.DataFrame, dict]:
    """Build non-purchase tickets 11..15 without changing official tickets 01..10."""
    frame = frame.sort_values("match_no").reset_index(drop=True)
    if len(frame) != 13 or set(frame["match_no"].astype(int)) != set(range(1, 14)):
        raise ValueError(f"requires complete 13-match round, got {len(frame)}")
    if strategy == "probability_allocation_v1":
        from proportional_buyplan import build_proportional_tickets
        if official_plan is None:
            raise ValueError("Proportional shadow requires official plan")
        plan = official_plan.sort_values("match_no").reset_index(drop=True)
        if not plan["match_no"].equals(frame["match_no"]):
            raise ValueError("Shadow official plan does not match round")
        result_to_index = {1:0, 0:1, 2:2}
        fixed = [[result_to_index[int(v)] for v in plan[f"ticket{i:02d}"]] for i in range(1,11)]
        probabilities = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy()
        tickets, allocation = build_proportional_tickets(probabilities, ticket_count=shadow_count,
                                                        fixed_tickets=fixed, seed=20261011)
        if not allocation["diversity_constraint_satisfied"]:
            raise ValueError("Cannot diversify proportional shadow without changing allocations")
        output = frame[["match_no", "home_team", "away_team", "league", "next_prob_home", "next_prob_draw", "next_prob_away"]].copy()
        output["draw_filter_class"] = [classify_draw_v2(row).draw_class for _, row in frame.iterrows()]
        for i, ticket in enumerate(tickets,11):
            output[f"ticket{i:02d}"] = [SYMBOL_TO_RESULT["HDA"[int(v)]] for v in ticket]
        audit = {"status":"non-purchase shadow observation", "generator":strategy,
                 "official_ticket_count":10,"shadow_ticket_count":shadow_count,
                 "official_tickets_unchanged":True,"shadow_probability_floor_ratio":None,
                 "allocation":allocation,"minimum_pairwise_hamming_distance":allocation["minimum_pairwise_hamming_distance"],
                 "shadow_ticket_joint_probabilities":[math.exp(sum(math.log(probabilities[j,int(v)]) for j,v in enumerate(ticket))) for ticket in tickets]}
        return output, audit
    if strategy != "coverage_v1":
        raise ValueError(f"Unknown shadow strategy: {strategy}")
    candidates = top_joint_candidates(frame)
    ranked = [ranked_outcomes(row) for _, row in frame.iterrows()]
    official = select_tickets_coverage(
        candidates, ranked, count=10, probability_floor_ratio=0.50, max_same_picks=11
    ) if official_plan is None else []
    if official_plan is not None:
        plan = official_plan.sort_values("match_no").reset_index(drop=True)
        if not plan["match_no"].equals(frame["match_no"]):
            raise ValueError("Shadow official plan does not match round")
        result_to_symbol = {value: symbol for symbol, value in SYMBOL_TO_RESULT.items()}
        official = []
        for index in range(1, 11):
            symbols = tuple(result_to_symbol[int(v)] for v in plan[f"ticket{index:02d}"])
            indices = tuple(next(j for j, (s, _) in enumerate(ranked[i]) if s == symbol) for i, symbol in enumerate(symbols))
            official.append({"symbols": symbols, "indices": indices,
                             "log_probability": sum(math.log(dict(ranked[i])[s]) for i, s in enumerate(symbols))})
    # The observation floor always refers to the highest-probability candidate,
    # independent of how official ticket01 was arranged.
    reference = candidates[0]["log_probability"]
    combined = extend_tickets_coverage(
        candidates, ranked, official, total_count=10 + shadow_count,
        probability_floor_ratio=probability_floor_ratio, max_same_picks=11,
    )
    shadow = combined[10:]
    output = frame[["match_no", "home_team", "away_team", "league"]].copy()
    assessments = [classify_draw_v2(row) for _, row in frame.iterrows()]
    output["next_prob_home"] = frame["next_prob_home"]
    output["next_prob_draw"] = frame["next_prob_draw"]
    output["next_prob_away"] = frame["next_prob_away"]
    output["draw_filter_class"] = [item.draw_class for item in assessments]
    for offset, ticket in enumerate(shadow, 11):
        output[f"ticket{offset:02d}"] = [SYMBOL_TO_RESULT[symbol] for symbol in ticket["symbols"]]
    all_tickets = official + shadow
    pairwise = [
        sum(a == b for a, b in zip(all_tickets[left]["symbols"], all_tickets[right]["symbols"]))
        for left in range(len(all_tickets)) for right in range(left + 1, len(all_tickets))
    ]
    audit = {
        "status": "non-purchase shadow observation",
        "official_ticket_count": 10,
        "shadow_ticket_count": shadow_count,
        "shadow_probability_floor_ratio": probability_floor_ratio,
        "official_tickets_unchanged": True,
        "minimum_pairwise_hamming_distance": 13 - max(pairwise),
        "shadow_ticket_joint_probabilities": [math.exp(item["log_probability"]) for item in shadow],
        "shadow_ticket_probability_ratios": [
            math.exp(item["log_probability"] - reference) for item in shadow
        ],
    }
    return output, audit


def build_round(frame: pd.DataFrame, strategy: str = "probability_v0", seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    frame = frame.sort_values("match_no").reset_index(drop=True)
    if len(frame) != 13 or set(frame["match_no"].astype(int)) != set(range(1, 14)):
        raise ValueError(f"requires complete 13-match round, got {len(frame)}")
    allocation_audit = None
    allocation_strategies = {"probability_allocation_v1", "probability_allocation_ge10_v1", "anchor_graded_shadow", "relative_draw_graded_v23", "random_remaining_v24"}
    candidates = [] if strategy in allocation_strategies else top_joint_candidates(frame)
    ranked = [ranked_outcomes(row) for _, row in frame.iterrows()]
    placement_audit = None
    if strategy in allocation_strategies:
        from proportional_buyplan import build_proportional_tickets
        probabilities = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy()
        if strategy == "random_remaining_v24":
            from random_remaining_v24 import generate
            if seed is None:
                raise ValueError("v2.4 requires an explicit reproducible seed")
            arrangements, allocation_audit = generate(probabilities, seed)
        elif strategy == "relative_draw_graded_v23":
            from deterministic_v23 import build_deterministic_tickets
            arrangements, allocation_audit = build_deterministic_tickets(probabilities)
        else:
            arrangements, allocation_audit = build_proportional_tickets(probabilities)
        if strategy != "random_remaining_v24" and not allocation_audit["diversity_constraint_satisfied"]:
            raise ValueError("Cannot satisfy ticket diversity while preserving probability allocations; allocations were not relaxed")
        if strategy in {"probability_allocation_ge10_v1", "anchor_graded_shadow"}:
            from ticket_coverage_optimizer import optimize_ten_hit_coverage
            arrangements, optimizer_audit = optimize_ten_hit_coverage(arrangements, probabilities)
            allocation_audit["optimizer"] = optimizer_audit
            allocation_audit["minimum_pairwise_hamming_distance"] = optimizer_audit["minimum_pairwise_hamming_distance"]
        if strategy == "anchor_graded_shadow":
            from anchor_arrangement import arrange_from_anchor
            arrangements, placement_audit = arrange_from_anchor(arrangements, probabilities, mode="graded")
            placement_audit.update(
                strength=0.0, placement_scores_HDA=probabilities.tolist(),
                ticket_display_order="original_probability_joint_descending",
                original_probabilities_unchanged=True, actual_results_used=False,
            )
            allocation_audit["minimum_pairwise_hamming_distance"] = placement_audit["minimum_pairwise_hamming_distance"]
        if strategy == "relative_draw_graded_v23":
            from deterministic_v23 import arrange_deterministically
            from placement_score import STRENGTH, Z_LIMIT, placement_scores
            scores, standardized, relative = placement_scores(probabilities)
            arrangements, placement_audit = arrange_deterministically(arrangements, scores)
            placement_audit.update(
                strength=STRENGTH, standardization_clip=Z_LIMIT,
                score_formula="logit(qD)=logit(pD)+strength*clip(z,-2,2); qH:qA=pH:pA",
                D_relative_log_ratio=relative.tolist(), D_relative_z=standardized.tolist(),
                placement_scores_HDA=scores.tolist(),
                ticket_display_order="placement_score_joint_descending",
                original_probabilities_used_for_allocations=True,
                original_probabilities_unchanged=True,
                actual_results_used=False,
            )
            for symbol in range(3):
                np.testing.assert_array_equal(
                    (arrangements == symbol).sum(axis=0),
                    np.asarray(allocation_audit["allocation_HDA"])[:, symbol],
                )
            allocation_audit["minimum_pairwise_hamming_distance"] = placement_audit["minimum_pairwise_hamming_distance"]
        tickets = [{"symbols": tuple("HDA"[int(s)] for s in row),
                    "log_probability": sum(math.log(probabilities[i, int(s)]) for i, s in enumerate(row))}
                   for row in arrangements]
    elif strategy == "coverage_v1":
        tickets = select_tickets_coverage(candidates, ranked)
    elif strategy == "probability_v0":
        tickets = select_tickets(candidates)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    # The optimizer preserves ticket identity only as an array index. For the
    # historical v2.2 strategy, make candidate numbers useful to readers.
    # Stable sorting keeps reproducible numbers for equal probabilities.
    probability_order = None
    if strategy == "probability_allocation_ge10_v1":
        probability_order = sorted(
            range(len(tickets)), key=lambda index: (-tickets[index]["log_probability"], index)
        )
        tickets = [tickets[index] for index in probability_order]
    output = frame[["match_no", "home_team", "away_team", "league"]].copy()
    assessments = [classify_draw_v2(row) for _, row in frame.iterrows()]
    output["next_prob_home"] = frame["next_prob_home"]
    output["next_prob_draw"] = frame["next_prob_draw"]
    output["next_prob_away"] = frame["next_prob_away"]
    output["draw_filter_class"] = [item.draw_class for item in assessments]
    for index, ticket in enumerate(tickets, 1):
        output[f"ticket{index:02d}"] = [SYMBOL_TO_RESULT[symbol] for symbol in ticket["symbols"]]

    ticket_columns = [f"ticket{i:02d}" for i in range(1, 11)]
    pairwise = []
    for left in range(10):
        for right in range(left + 1, 10):
            same = int((output[ticket_columns[left]] == output[ticket_columns[right]]).sum())
            pairwise.append(same)
    marginal_rows = []
    for _, row in output.iterrows():
        counts = {symbol: sum(row[col] == value for col in ticket_columns)
                  for symbol, value in SYMBOL_TO_RESULT.items()}
        marginal_rows.append({
            "match_no": int(row["match_no"]),
            "home_team": row["home_team"], "away_team": row["away_team"],
            "count_H": counts["H"], "count_D": counts["D"], "count_A": counts["A"],
            "expected_H": 10 * float(row["next_prob_home"]),
            "expected_D": 10 * float(row["next_prob_draw"]),
            "expected_A": 10 * float(row["next_prob_away"]),
            "draw_filter_class": row["draw_filter_class"],
        })
    metadata = {
        "generator": strategy,
        "fixed_symbol_ratio_used": False,
        "world_forcing_used": False,
        "target_draw_count_used": False,
        "candidate_pool_size": len(candidates),
        "max_same_picks": 11,
        "minimum_pairwise_hamming_distance": 2,
        "joint_probability_floor_ratio": 0.50 if strategy == "coverage_v1" else None,
        "coverage_objective": "probability_weighted_new_symbols" if strategy == "coverage_v1" else None,
        "ticket_log_probabilities": [item["log_probability"] for item in tickets],
        "ticket_joint_probabilities": [math.exp(item["log_probability"]) for item in tickets],
        "pairwise_same_pick_min": min(pairwise),
        "pairwise_same_pick_mean": sum(pairwise) / len(pairwise),
        "pairwise_same_pick_max": max(pairwise),
        "total_draw_cells": int((output[ticket_columns] == 0).sum().sum()),
        "marginals": marginal_rows,
    }
    if allocation_audit is not None:
        metadata["allocation"] = allocation_audit
        metadata["coverage_objective"] = ("random_remaining_count_draw" if strategy == "random_remaining_v24" else
                                          "preserve_allocations_then_maximize_ge10_coverage" if strategy == "probability_allocation_ge10_v1" else
                                          "preserve_probability_allocations_then_diversify")
        metadata["max_same_picks"] = None
        metadata["minimum_pairwise_hamming_distance"] = allocation_audit.get("minimum_pairwise_hamming_distance")
        if strategy == "random_remaining_v24":
            metadata["ticket_display_order"] = "draw_order_01_to_10"
            metadata["duplicate_pairs"] = allocation_audit["duplicate_pairs"]
            metadata["duplicate_count"] = allocation_audit["duplicate_count"]
    if probability_order is not None:
        metadata["ticket_display_order"] = "joint_probability_descending"
        metadata["ticket_original_indices"] = [index + 1 for index in probability_order]
    if placement_audit is not None:
        metadata["placement_score"] = placement_audit
        metadata["ticket_display_order"] = placement_audit["ticket_display_order"]
        metadata["coverage_objective"] = "fixed_allocations_then_graded_relative_D_placement" if strategy == "relative_draw_graded_v23" else "fixed_allocations_then_anchor_graded_placement"
    return output, metadata


def score_round(plan: pd.DataFrame, source: pd.DataFrame) -> dict:
    actual = source.set_index("match_no")["actual_result"].astype(int)
    ticket_columns = [f"ticket{i:02d}" for i in range(1, 11)]
    hits = {column: int((plan.set_index("match_no")[column].astype(int) == actual).sum()) for column in ticket_columns}
    covered = sum(
        any(int(getattr(row, column)) == int(actual.loc[row.match_no]) for column in ticket_columns)
        for row in plan.itertuples()
    )
    return {
        "ticket_hits": hits,
        "best_hits": max(hits.values()),
        "average_hits": sum(hits.values()) / len(hits),
        "ticket01_hits": hits["ticket01"],
        "portfolio_cover": int(covered),
    }


def score_legacy(round_id: str, source: pd.DataFrame) -> dict | None:
    path = ROOT / "data/eval/toto_rounds" / round_id / "snapshot/buyplan.csv"
    if not path.exists():
        return None
    legacy = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    columns = [f"ticket{i:02d}" for i in range(1, 11)]
    if len(legacy) != 13 or not all(column in legacy for column in columns):
        return None
    return score_round(legacy, source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strategy", choices=["probability_v0", "coverage_v1", "probability_allocation_v1", "probability_allocation_ge10_v1", "anchor_graded_shadow", "relative_draw_graded_v23"], default="probability_v0")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    source = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    args.out.mkdir(parents=True)
    summaries = []
    exclusions = []
    for round_id, group in source.groupby("round_id", sort=True):
        if len(group) != 13:
            exclusions.append({"round_id": round_id, "rows": len(group), "reason": "incomplete frozen prediction population"})
            continue
        plan, audit = build_round(group, strategy=args.strategy)
        round_dir = args.out / str(round_id)
        round_dir.mkdir()
        plan.to_csv(round_dir / "buyplan.csv", index=False, encoding="utf-8-sig")
        new_score = score_round(plan, group)
        legacy_score = score_legacy(str(round_id), group)
        report = {"round_id": round_id, "audit": audit, "next": new_score, "legacy": legacy_score}
        (round_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({
            "round_id": round_id, "prematch_snapshot_proven": bool(group["prematch_snapshot_proven"].all()),
            "next_best_hits": new_score["best_hits"], "next_average_hits": new_score["average_hits"],
            "next_ticket01_hits": new_score["ticket01_hits"], "next_portfolio_cover": new_score["portfolio_cover"],
            "next_draw_cells": audit["total_draw_cells"],
            "legacy_best_hits": legacy_score["best_hits"] if legacy_score else None,
            "legacy_average_hits": legacy_score["average_hits"] if legacy_score else None,
            "legacy_ticket01_hits": legacy_score["ticket01_hits"] if legacy_score else None,
            "legacy_portfolio_cover": legacy_score["portfolio_cover"] if legacy_score else None,
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out / "round_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "next official BuyPlan under development; not connected to production",
        "input": str(args.input), "rounds_built": summary["round_id"].tolist(),
        "excluded_rounds": exclusions,
        "fixed_symbol_ratio_used": False, "world_forcing_used": False,
        "target_draw_count_used": False,
        "strategy": args.strategy,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out / "round_summary.csv")


if __name__ == "__main__":
    main()
