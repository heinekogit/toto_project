#!/usr/bin/env python3
"""Accumulate immutable per-match observations from a completed toto evaluation.

This script is deliberately read-only with respect to prediction and buyplan inputs.
It stores a compact, analysis-friendly row plus JSON payloads containing every source
column, so future investigations are not limited to today's choice of features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "eval" / "observation_history"
TICKET_RE = re.compile(r"^(?:ticket|候補)\s*0*(10|[1-9])$", re.IGNORECASE)
RESULT_LABEL = {0: "D", 1: "H", 2: "A"}


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _payload(row: pd.Series) -> str:
    return json.dumps(
        {str(k): _json_value(v) for k, v in row.items()},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if _json_value(value) is not None and str(value).strip() != "":
                return value
    return None


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _result_number(value: Any) -> int | None:
    number = _number(value)
    if number is not None and int(number) in RESULT_LABEL:
        return int(number)
    text = str(value or "").strip().upper()
    return {"D": 0, "H": 1, "A": 2}.get(text)


def _ticket_columns(df: pd.DataFrame) -> list[str]:
    found: list[tuple[int, str]] = []
    for column in df.columns:
        match = TICKET_RE.match(str(column).strip())
        if match:
            found.append((int(match.group(1)), str(column)))
    return [column for _, column in sorted(found)]


def _join_prediction(buy: pd.Series, predictions: pd.DataFrame) -> pd.Series:
    match_no = _number(buy.get("match_no"))
    if match_no is not None and "match_no" in predictions.columns:
        nums = pd.to_numeric(predictions["match_no"], errors="coerce")
        hit = predictions[nums == int(match_no)]
        if len(hit) == 1:
            return hit.iloc[0]

    required = {"home_team", "away_team"}
    if required.issubset(predictions.columns):
        hit = predictions[
            predictions["home_team"].astype(str).str.strip().eq(str(buy.get("home_team", "")).strip())
            & predictions["away_team"].astype(str).str.strip().eq(str(buy.get("away_team", "")).strip())
        ]
        if len(hit) == 1:
            return hit.iloc[0]
    raise RuntimeError(
        "prediction row could not be matched uniquely: "
        f"match_no={buy.get('match_no')} {buy.get('home_team')}-{buy.get('away_team')}"
    )


def _join_actual(buy: pd.Series, actual: pd.DataFrame) -> pd.Series:
    nums = pd.to_numeric(actual.get("match_no"), errors="coerce")
    hit = actual[nums == int(float(buy.get("match_no")))]
    if len(hit) != 1:
        raise RuntimeError(f"actual row could not be matched uniquely: match_no={buy.get('match_no')}")
    return hit.iloc[0]


def build_rows(round_id: str, predictions: pd.DataFrame, buyplan: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    tickets = _ticket_columns(buyplan)
    if not tickets:
        raise ValueError("buyplan has no ticket columns")

    rows: list[dict[str, Any]] = []
    for _, buy in buyplan.iterrows():
        try:
            pred = _join_prediction(buy, predictions)
        except RuntimeError:
            league = str(buy.get("league") or "").strip().upper()
            if league == "UNKNOWN":
                print(
                    f"[INFO] 正式予測なしのため観測対象外: match_no={buy.get('match_no')} "
                    f"{buy.get('home_team')}-{buy.get('away_team')}"
                )
                continue
            raise
        act = _join_actual(buy, actual)
        result = _result_number(act.get("result"))
        if result is None:
            continue

        picks = [_result_number(buy.get(column)) for column in tickets]
        valid_picks = [pick for pick in picks if pick is not None]
        hits = [int(pick == result) if pick is not None else 0 for pick in picks]
        base_hit = hits[0] if hits else 0

        p_home = _number(_first(pred, ["prob_final_home", "prob_blend_home", "prob_home_win", "p_home"]))
        p_draw = _number(_first(pred, ["prob_final_draw", "prob_blend_draw", "prob_draw", "p_draw"]))
        p_away = _number(_first(pred, ["prob_final_away", "prob_blend_away", "prob_away_win", "p_away"]))

        row: dict[str, Any] = {
            "round_id": round_id,
            "match_no": int(float(buy.get("match_no"))),
            "league": _first(pred, ["league"]),
            "round_label": _first(pred, ["節", "section", "round"]),
            "match_id": _first(pred, ["match_id"]),
            "datetime": _first(pred, ["datetime"]),
            "home_team": buy.get("home_team"),
            "away_team": buy.get("away_team"),
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "predicted_result": _first(
                pred,
                ["predicted_result_main_symbol", "predicted_result_main", "predicted_result", "predicted_result_final"],
            ),
            "rank1_symbol": _first(pred, ["rank1_symbol"]),
            "rank2_symbol": _first(pred, ["rank2_symbol"]),
            "rank3_symbol": _first(pred, ["rank3_symbol"]),
            "match_purchase_type": _first(pred, ["match_purchase_type"]),
            "match_purchase_subtype": _first(pred, ["match_purchase_subtype"]),
            "admission_policy": _first(pred, ["admission_policy"]),
            "draw_purchase_tier": _first(pred, ["draw_purchase_tier"]),
            "draw_risk_flag": _first(pred, ["draw_risk_flag"]),
            "draw_core_flag": _first(pred, ["draw_core_flag"]),
            "match_type_primary": _first(pred, ["match_type_primary", "match_type"]),
            "match_type_flags": _first(pred, ["match_type_flags"]),
            "lab_matchup_profile": _first(pred, ["lab_matchup_profile"]),
            "lab_basis_hint": _first(pred, ["lab_basis_hint"]),
            "lab_hold_weight": _number(_first(pred, ["lab_hold_weight"])),
            "lab_stall_weight": _number(_first(pred, ["lab_stall_weight"])),
            "lab_flip_weight": _number(_first(pred, ["lab_flip_weight"])),
            "lab_volatility_score": _number(_first(pred, ["lab_volatility_score"])),
            "lab_scenario_entropy_score": _number(_first(pred, ["lab_scenario_entropy_score"])),
            "lab_stall_compactness_score": _number(_first(pred, ["lab_stall_compactness_score"])),
            "lab_draw_tension_score": _number(_first(pred, ["lab_draw_tension_score"])),
            "lab_home_path_score": _number(_first(pred, ["lab_home_path_score"])),
            "lab_away_path_score": _number(_first(pred, ["lab_away_path_score"])),
            "lab_mix_home": _number(_first(pred, ["lab_mix_home"])),
            "lab_mix_draw": _number(_first(pred, ["lab_mix_draw"])),
            "lab_mix_away": _number(_first(pred, ["lab_mix_away"])),
            "anchor_candidate_flag": _first(pred, ["anchor_candidate_flag"]),
            "draw_core_candidate_flag": _first(pred, ["draw_core_candidate_flag"]),
            "away_overread_draw_cover_flag": _first(pred, ["away_overread_draw_cover_flag"]),
            "actual_result": result,
            "actual_label": RESULT_LABEL[result],
            "portfolio_hit_count": sum(hits),
            "portfolio_covered": int(any(hits)),
            "all_same_symbol": int(len(set(valid_picks)) == 1) if valid_picks else 0,
            "ticket01_hit": base_hit,
            "rescue_vs_ticket01_count": sum(hits[1:]) if not base_hit else 0,
            "harm_vs_ticket01_count": sum(1 for hit in hits[1:] if not hit) if base_hit else 0,
            "prediction_payload_json": _payload(pred),
            "buyplan_payload_json": _payload(buy),
            "actual_payload_json": _payload(act),
        }
        for index, (column, pick, hit) in enumerate(zip(tickets, picks, hits), 1):
            row[f"ticket{index:02d}_symbol"] = pick
            row[f"ticket{index:02d}_hit"] = hit
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(round_id: str, rows: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "round_id": round_id,
        "matches": len(rows),
        "actual_home": int((rows["actual_result"] == 1).sum()),
        "actual_draw": int((rows["actual_result"] == 0).sum()),
        "actual_away": int((rows["actual_result"] == 2).sum()),
        "portfolio_covered_matches": int(rows["portfolio_covered"].sum()),
        "all_same_matches": int(rows["all_same_symbol"].sum()),
        "ticket01_hits": int(rows["ticket01_hit"].sum()),
        "rescue_vs_ticket01_matches": int((rows["rescue_vs_ticket01_count"] > 0).sum()),
    }
    complete_probs = rows.dropna(subset=["p_home", "p_draw", "p_away"])
    if not complete_probs.empty:
        matrix = complete_probs[["p_draw", "p_home", "p_away"]].astype(float).to_numpy()
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / totals
        actual_idx = complete_probs["actual_result"].astype(int).to_numpy()
        chosen = matrix[range(len(matrix)), actual_idx]
        summary["log_loss_3class"] = float(-pd.Series(chosen).clip(lower=1e-15).map(math.log).mean())
        one_hot = pd.get_dummies(complete_probs["actual_result"]).reindex(columns=[0, 1, 2], fill_value=0).to_numpy()
        summary["brier_3class"] = float(((matrix - one_hot) ** 2).sum(axis=1).mean())
        summary["mean_p_draw"] = float(complete_probs["p_draw"].astype(float).mean())
    return summary


def _upsert(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        old = pd.read_csv(path, low_memory=False)
        combined = pd.concat([old, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows.copy()
    combined = combined.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accumulate post-match observations without changing official logic")
    parser.add_argument("--round", required=True, dest="round_id", help="toto1644 / round01")
    parser.add_argument("--round-dir", default="", help="既定: data/eval/{toto_rounds|rounds}/{round}")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family = "toto_rounds" if args.round_id.startswith("toto") else "rounds"
    round_dir = Path(args.round_dir).resolve() if args.round_dir else ROOT_DIR / "data" / "eval" / family / args.round_id
    snapshot_dir = round_dir / "snapshot"
    paths = {
        "predictions": snapshot_dir / "predictions.csv",
        "buyplan": snapshot_dir / "buyplan.csv",
        "actual": round_dir / "actual_results.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required observation inputs are missing: {missing}")

    predictions = pd.read_csv(paths["predictions"], low_memory=False)
    buyplan = pd.read_csv(paths["buyplan"], low_memory=False)
    actual = pd.read_csv(paths["actual"], low_memory=False)
    rows = build_rows(args.round_id, predictions, buyplan, actual)
    if rows.empty:
        print(
            f"[WARN] 確定済み結果がないため観測履歴の更新をスキップします: "
            f"round={args.round_id}"
        )
        return

    out_dir = Path(args.out_dir).resolve()
    matches_path = out_dir / "matches.csv"
    summaries_path = out_dir / "round_summaries.csv"
    _upsert(matches_path, rows, ["round_id", "match_no"])
    _upsert(summaries_path, pd.DataFrame([build_summary(args.round_id, rows)]), ["round_id"])

    provenance = {
        "round_id": args.round_id,
        "accumulated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": len(rows),
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
    }
    provenance_dir = out_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / f"{args.round_id}.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] observation matches: {matches_path} rows={len(rows)}")
    print(f"[OK] observation summaries: {summaries_path}")
    print(f"[OK] observation provenance: {provenance_dir / f'{args.round_id}.json'}")


if __name__ == "__main__":
    main()
