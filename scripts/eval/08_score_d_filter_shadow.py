#!/usr/bin/env python3
"""Score a frozen pre-match D-filter shadow after results are available."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SHADOW_ROOT = ROOT_DIR / "data" / "eval" / "d_filter_shadow"
DEFAULT_EVAL_ROOT = ROOT_DIR / "data" / "eval"
POLICIES = ("false_draw_only", "false_draw_plus_trap", "topology_guard")


def _latest_shadow(root: Path, logical_season: str, round_id: str) -> Path:
    parent = root / logical_season / round_id
    candidates = sorted(path for path in parent.glob("*_prematch") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"prematch D-filter shadow not found: {parent}")
    return candidates[-1]


def _symbol(value: object) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number) and int(number) in {0, 1, 2}:
        return int(number)
    return {"D": 0, "H": 1, "A": 2}.get(str(value or "").strip().upper())


def _team_key(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("　", " ").strip()


def score(reference: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "d_filter_eligible" in reference.columns:
        eligible = pd.to_numeric(reference["d_filter_eligible"], errors="coerce").fillna(0).eq(1)
        excluded = int((~eligible).sum())
        reference = reference.loc[eligible].copy()
        if excluded:
            print(f"[INFO] D-filter採点対象外: rows={excluded}（正式予測なし）")
    if reference.empty:
        raise RuntimeError("D-filter採点対象がありません")
    actual_cols = ["match_no", "home_team", "away_team", "result"]
    left = reference.copy()
    right = actual[actual_cols].copy()
    for frame in (left, right):
        frame["_match_no_key"] = pd.to_numeric(frame["match_no"], errors="coerce").astype("Int64")
        frame["_home_key"] = frame["home_team"].map(_team_key)
        frame["_away_key"] = frame["away_team"].map(_team_key)
    right = right.drop(columns=["match_no", "home_team", "away_team"])
    merged = left.merge(
        right,
        on=["_match_no_key", "_home_key", "_away_key"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["_match_no_key", "_home_key", "_away_key"])
    if merged["result"].isna().any():
        raise RuntimeError(f"actual result merge failed: missing_rows={int(merged['result'].isna().sum())}")
    merged["actual_symbol"] = merged["result"].map(_symbol)
    merged["baseline_hit"] = merged["baseline_symbol"].map(_symbol).eq(merged["actual_symbol"]).astype(int)
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        symbol_col = f"{policy}_symbol"
        hit_col = f"{policy}_hit"
        merged[hit_col] = merged[symbol_col].map(_symbol).eq(merged["actual_symbol"]).astype(int)
        changed = merged[f"{policy}_changed"].astype(int).eq(1)
        rescue = changed & merged["baseline_hit"].eq(0) & merged[hit_col].eq(1)
        harm = changed & merged["baseline_hit"].eq(1) & merged[hit_col].eq(0)
        rows.append(
            {
                "policy": policy,
                "matches": len(merged),
                "baseline_hits": int(merged["baseline_hit"].sum()),
                "filtered_hits": int(merged[hit_col].sum()),
                "changed_matches": int(changed.sum()),
                "rescue": int(rescue.sum()),
                "harm": int(harm.sum()),
                "net_rescue": int(rescue.sum() - harm.sum()),
                "baseline_hit_rate": float(merged["baseline_hit"].mean()),
                "filtered_hit_rate": float(merged[hit_col].mean()),
            }
        )
    return merged, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score frozen D-filter prematch advice")
    parser.add_argument("--round", required=True, dest="round_id")
    parser.add_argument("--logical-season", default="2027")
    parser.add_argument("--shadow-dir", default="")
    parser.add_argument("--shadow-root", default=str(DEFAULT_SHADOW_ROOT))
    parser.add_argument("--actual", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shadow_dir = Path(args.shadow_dir).resolve() if args.shadow_dir else _latest_shadow(
        Path(args.shadow_root).resolve(), str(args.logical_season), args.round_id
    )
    actual_path = Path(args.actual).resolve() if args.actual else DEFAULT_EVAL_ROOT / (
        "toto_rounds" if args.round_id.startswith("toto") else "rounds"
    ) / args.round_id / "actual_results.csv"
    reference_path = shadow_dir / "buyplan_d_filter_reference.csv"
    if not reference_path.exists() or not actual_path.exists():
        raise FileNotFoundError(f"missing score input: reference={reference_path} actual={actual_path}")
    detail, summary = score(
        pd.read_csv(reference_path, low_memory=False),
        pd.read_csv(actual_path, low_memory=False),
    )
    detail_path = shadow_dir / "d_filter_scored.csv"
    summary_path = shadow_dir / "d_filter_score_summary.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    score_manifest = {
        "round_id": args.round_id,
        "shadow_dir": str(shadow_dir),
        "actual_source": str(actual_path),
        "matches": len(detail),
        "summary": summary.to_dict("records"),
    }
    (shadow_dir / "score_manifest.json").write_text(
        json.dumps(score_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] D-filter shadow score: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
