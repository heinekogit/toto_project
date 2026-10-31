#!/usr/bin/env python3
"""Audit cumulative/per-game relationships used by base lambda v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ROOT


DEFAULT_INPUT = ROOT / "data/eval/next_model/validation_v1/matches.csv"
DEFAULT_OUT = ROOT / "data/eval/next_model/validation_v1/stats_unit_audit.json"


def audit(frame: pd.DataFrame) -> dict:
    sides = {}
    for side in ("home", "away"):
        total = pd.to_numeric(frame[f"stats_被ゴール期待値_{side}"], errors="coerce")
        per_game = pd.to_numeric(frame[f"stats_1試合平均被ゴール期待値_{side}"], errors="coerce")
        ratio = total / per_game
        rounded = ratio.round()
        error = (ratio - rounded).abs()
        attack_total = pd.to_numeric(frame[f"stats_ゴール期待値_{side}"], errors="coerce")
        attack_per_game = attack_total / rounded.clip(lower=1)
        sides[side] = {
            "valid_rows": int(ratio.notna().sum()),
            "ratio_min": float(ratio.min()),
            "ratio_median": float(ratio.median()),
            "ratio_max": float(ratio.max()),
            "integer_rounding_error_max": float(error.max()),
            "rows_with_rounding_error_le_0_55": int(error.le(0.55).sum()),
            "derived_attack_xg_per_game_min": float(attack_per_game.min()),
            "derived_attack_xg_per_game_mean": float(attack_per_game.mean()),
            "derived_attack_xg_per_game_max": float(attack_per_game.max()),
        }
    return {
        "conclusion": (
            "stats_ゴール期待値 and stats_被ゴール期待値 are treated as cumulative; "
            "stats_1試合平均被ゴール期待値 is per-game. Rounded cumulative/per-game xGA "
            "is used only to reconstruct games played inside the same frozen snapshot row."
        ),
        "rows": len(frame),
        "sides": sides,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = audit(pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
