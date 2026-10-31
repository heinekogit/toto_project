#!/usr/bin/env python3
"""Freeze the eight-round D-validation population into one auditable table."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common import PREMATCH_PROVEN_ROUNDS, ROOT, VALIDATION_ROUNDS, match_key, sha256


JST = ZoneInfo("Asia/Tokyo")
DEFAULT_OUT = ROOT / "data/eval/next_model/validation_v1"

IDENTITY_COLUMNS = [
    "match_id", "datetime", "league", "home_team", "away_team", "stats_asof",
]
CURRENT_PROBABILITY_COLUMNS = [
    "prob_elo_home", "prob_elo_draw", "prob_elo_away",
    "prob_lab_home", "prob_lab_draw", "prob_lab_away",
    "prob_final_home", "prob_final_draw", "prob_final_away",
    "prob_main_home", "prob_main_draw", "prob_main_away",
    "predicted_result", "predicted_result_main", "predicted_result_main_symbol",
    "rank1_symbol", "rank2_symbol", "rank3_symbol",
]
D_CLASSIFICATION_COLUMNS = [
    "match_purchase_type", "match_purchase_subtype", "purchase_type_confidence",
    "purchase_type_reason", "draw_purchase_tier", "draw_core_flag", "draw_risk_flag",
    "lab_matchup_profile", "lab_basis_hint", "lab_hold_weight", "lab_stall_weight",
    "lab_flip_weight", "lab_volatility_score", "lab_draw_tension_score",
    "lab_stall_compactness_score", "lab_scenario_entropy_score",
    "lab_home_path_score", "lab_away_path_score", "lab_mix_home", "lab_mix_draw",
    "lab_mix_away", "match_type", "match_type_primary", "match_type_flags",
]
BASE_INPUT_COLUMNS = [
    "stats_1試合平均得点数_home", "stats_1試合平均失点数_home",
    "stats_1試合平均得点数_away", "stats_1試合平均失点数_away",
    "stats_ゴール期待値_home", "stats_被ゴール期待値_home",
    "stats_1試合平均被ゴール期待値_home",
    "stats_ゴール期待値_away", "stats_被ゴール期待値_away",
    "stats_1試合平均被ゴール期待値_away",
    "home_fatigue_score", "away_fatigue_score", "home_total_fatigue_score",
    "away_total_fatigue_score", "absence_effective_total_home",
    "absence_effective_total_away", "rankmot_motivation_score_5w_home",
    "rankmot_motivation_score_5w_away", "is_rain", "is_heavy_rain", "is_strong_wind",
]
LAB_SOURCE_COLUMNS = [
    "flab_expected_for_xg_home", "flab_expected_for_xg_away",
    "flab_expected_against_xg_home", "flab_expected_against_xg_away",
    "flab_chance_attack_count_home", "flab_chance_attack_count_away",
    "flab_chance_shots_home", "flab_chance_shots_away",
    "flab_chance_allowed_shots_home", "flab_chance_allowed_shots_away",
    "flab_chance_shot_conversion_home", "flab_chance_shot_conversion_away",
    "flab_agi_score_home", "flab_agi_score_away", "flab_kagi_score_home",
    "flab_kagi_score_away",
]
SELECTED_COLUMNS = list(dict.fromkeys(
    IDENTITY_COLUMNS + CURRENT_PROBABILITY_COLUMNS + D_CLASSIFICATION_COLUMNS
    + BASE_INPUT_COLUMNS + LAB_SOURCE_COLUMNS
))


def select_matches(
    predictions: pd.DataFrame, actual: pd.DataFrame, round_id: str, allow_missing: bool = False
) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["__match_key"] = [
        match_key(home, away) for home, away in zip(predictions["home_team"], predictions["away_team"])
    ]
    if predictions["__match_key"].duplicated().any():
        duplicated = predictions.loc[predictions["__match_key"].duplicated(False), "__match_key"].tolist()
        raise ValueError(f"{round_id}: duplicated prediction pairs: {duplicated}")
    by_key = predictions.set_index("__match_key", drop=False)
    rows = []
    for actual_row in actual.sort_values("match_no").itertuples(index=False):
        key = match_key(actual_row.home_team, actual_row.away_team)
        if key not in by_key.index:
            if allow_missing:
                continue
            raise ValueError(f"{round_id}: prediction missing for {key}")
        source = by_key.loc[key]
        row = {column: source.get(column, pd.NA) for column in SELECTED_COLUMNS}
        row.update({
            "round_id": round_id,
            "match_no": int(actual_row.match_no),
            "actual_home_score": actual_row.home_score,
            "actual_away_score": actual_row.away_score,
            "actual_result": int(actual_row.result),
            "prematch_snapshot_proven": round_id in PREMATCH_PROVEN_ROUNDS,
        })
        rows.append(row)
    if not allow_missing and len(rows) != 13:
        raise ValueError(f"{round_id}: expected 13 matches, got {len(rows)}")
    return pd.DataFrame(rows)


def build(rounds: tuple[str, ...], root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    frames = []
    sources = {}
    exclusions = []
    for round_id in rounds:
        base = root / "data/eval/toto_rounds" / round_id
        prediction_path = base / "snapshot/predictions.csv"
        actual_path = base / "actual_results.csv"
        if not prediction_path.exists() or not actual_path.exists():
            raise FileNotFoundError(f"{round_id}: missing snapshot or actual results")
        predictions = pd.read_csv(prediction_path, encoding="utf-8-sig", low_memory=False)
        actual = pd.read_csv(actual_path, encoding="utf-8-sig")
        selected = select_matches(predictions, actual, round_id, allow_missing=True)
        selected_match_nos = set(selected["match_no"].astype(int))
        missing = actual[~actual["match_no"].astype(int).isin(selected_match_nos)]
        for row in missing.itertuples(index=False):
            exclusions.append({
                "round_id": round_id,
                "match_no": int(row.match_no),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "reason": "purchase prediction row is absent from frozen snapshot",
            })
        frames.append(selected)
        sources[round_id] = {
            "predictions": str(prediction_path.relative_to(root)),
            "predictions_sha256": sha256(prediction_path),
            "actual_results": str(actual_path.relative_to(root)),
            "actual_results_sha256": sha256(actual_path),
            "prematch_snapshot_proven": round_id in PREMATCH_PROVEN_ROUNDS,
        }
    dataset = pd.concat(frames, ignore_index=True)
    manifest = {
        "schema_version": 1,
        "purpose": "frozen acceptance population for next score-model and D-classifier work",
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
        "rounds": list(rounds),
        "rows": len(dataset),
        "prematch_proven_rows": int(dataset["prematch_snapshot_proven"].sum()),
        "excluded_actual_rows": exclusions,
        "sources": sources,
        "column_contract": {
            "identity": IDENTITY_COLUMNS,
            "current_probabilities": CURRENT_PROBABILITY_COLUMNS,
            "legacy_d_classification": D_CLASSIFICATION_COLUMNS,
            "base_inputs": BASE_INPUT_COLUMNS,
            "lab_sources_reserved_for_later": LAB_SOURCE_COLUMNS,
        },
    }
    return dataset, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", nargs="+", default=list(VALIDATION_ROUNDS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"validation output already exists: {out}")
    dataset, manifest = build(tuple(args.rounds))
    out.mkdir(parents=True)
    dataset.to_csv(out / "matches.csv", index=False, encoding="utf-8-sig")
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out / "matches.csv")


if __name__ == "__main__":
    main()
