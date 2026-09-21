#!/usr/bin/env python3
"""Evaluate observation-only D filters without changing official predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = ROOT_DIR / "data" / "eval" / "observation_history"
FALSE_DRAW_TYPES = {"j1_false_draw_watch", "j2_false_draw_watch"}
TRAP_TYPES = {"draw_trap", "j2_draw_trap"}
POLICIES = ("false_draw_only", "false_draw_plus_trap", "topology_guard")


def _number(value: object, default: float = 0.0) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) else float(number)


def _symbol(value: object) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number) and int(number) in {0, 1, 2}:
        return int(number)
    text = str(value or "").strip().upper()
    return {"D": 0, "H": 1, "A": 2}.get(text)


def _side_pick(row: pd.Series) -> tuple[int, str]:
    rank2 = _symbol(row.get("rank2_symbol"))
    if rank2 in {1, 2}:
        return rank2, "rank2_side"
    p_home = _number(row.get("p_home"))
    p_away = _number(row.get("p_away"))
    if abs(p_home - p_away) > 1e-12:
        return (1 if p_home > p_away else 2), "prob_best_side"
    home_path = _number(row.get("lab_home_path_score"), 0.5)
    away_path = _number(row.get("lab_away_path_score"), 0.5)
    return (1 if home_path >= away_path else 2), "path_tiebreak"


def _strong_draw_evidence(row: pd.Series) -> tuple[bool, str]:
    p_draw = _number(row.get("p_draw"))
    side_prob = max(_number(row.get("p_home")), _number(row.get("p_away")))
    draw_gap = p_draw - side_prob
    tension = _number(row.get("lab_draw_tension_score"))
    compact = _number(row.get("lab_stall_compactness_score"))
    stall = _number(row.get("lab_stall_weight"))
    volatility = _number(row.get("lab_volatility_score"), 1.0)
    strong = bool(
        p_draw >= 0.40
        and draw_gap >= 0.025
        and tension >= 0.68
        and compact >= 0.62
        and stall >= 0.42
        and volatility <= 0.30
    )
    reason = (
        f"pD={p_draw:.3f}|gap={draw_gap:.3f}|tension={tension:.3f}|"
        f"compact={compact:.3f}|stall={stall:.3f}|vol={volatility:.3f}"
    )
    return strong, reason


def apply_policy(row: pd.Series, policy: str) -> tuple[int, bool, str]:
    baseline = _symbol(row.get("predicted_result"))
    if baseline is None:
        raise ValueError(f"invalid predicted_result: {row.get('predicted_result')!r}")
    if baseline != 0:
        return baseline, False, "baseline_not_draw"

    purchase_type = str(row.get("match_purchase_type") or "").strip()
    side, side_reason = _side_pick(row)
    if policy == "false_draw_only":
        should_filter = purchase_type in FALSE_DRAW_TYPES
        reason = f"type={purchase_type}|{side_reason}"
    elif policy == "false_draw_plus_trap":
        should_filter = purchase_type in FALSE_DRAW_TYPES | TRAP_TYPES
        reason = f"type={purchase_type}|{side_reason}"
    elif policy == "topology_guard":
        strong_draw, evidence = _strong_draw_evidence(row)
        should_filter = not strong_draw
        reason = f"{'filter_weak_D' if should_filter else 'keep_strong_D'}|{evidence}|{side_reason}"
    else:
        raise ValueError(f"unknown policy: {policy}")
    return (side if should_filter else baseline), should_filter, reason


def build_detail(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, source in matches.iterrows():
        baseline = _symbol(source.get("predicted_result"))
        actual = _symbol(source.get("actual_result"))
        if baseline is None or actual is None:
            continue
        for policy in POLICIES:
            filtered, changed, reason = apply_policy(source, policy)
            baseline_hit = int(baseline == actual)
            filtered_hit = int(filtered == actual)
            rows.append(
                {
                    "round_id": source.get("round_id"),
                    "match_no": int(source.get("match_no")),
                    "league": str(source.get("league") or "").upper(),
                    "home_team": source.get("home_team"),
                    "away_team": source.get("away_team"),
                    "match_purchase_type": source.get("match_purchase_type"),
                    "match_purchase_subtype": source.get("match_purchase_subtype"),
                    "policy": policy,
                    "baseline_symbol": baseline,
                    "filtered_symbol": filtered,
                    "actual_symbol": actual,
                    "changed": int(changed),
                    "baseline_hit": baseline_hit,
                    "filtered_hit": filtered_hit,
                    "rescue": int(changed and not baseline_hit and filtered_hit),
                    "harm": int(changed and baseline_hit and not filtered_hit),
                    "net_rescue": int(filtered_hit - baseline_hit),
                    "reason": reason,
                    "p_home": _number(source.get("p_home")),
                    "p_draw": _number(source.get("p_draw")),
                    "p_away": _number(source.get("p_away")),
                    "lab_hold_weight": _number(source.get("lab_hold_weight")),
                    "lab_stall_weight": _number(source.get("lab_stall_weight")),
                    "lab_flip_weight": _number(source.get("lab_flip_weight")),
                    "lab_volatility_score": _number(source.get("lab_volatility_score")),
                    "lab_draw_tension_score": _number(source.get("lab_draw_tension_score")),
                    "lab_stall_compactness_score": _number(source.get("lab_stall_compactness_score")),
                }
            )
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    def one(group: pd.DataFrame) -> pd.Series:
        changed = group[group["changed"] == 1]
        return pd.Series(
            {
                "matches": len(group),
                "baseline_hits": int(group["baseline_hit"].sum()),
                "filtered_hits": int(group["filtered_hit"].sum()),
                "baseline_hit_rate": float(group["baseline_hit"].mean()),
                "filtered_hit_rate": float(group["filtered_hit"].mean()),
                "changed_matches": int(len(changed)),
                "changed_to_home": int((changed["filtered_symbol"] == 1).sum()),
                "changed_to_away": int((changed["filtered_symbol"] == 2).sum()),
                "rescue": int(group["rescue"].sum()),
                "harm": int(group["harm"].sum()),
                "net_rescue": int(group["net_rescue"].sum()),
                "change_precision": float(group["rescue"].sum() / len(changed)) if len(changed) else float("nan"),
            }
        )

    return detail.groupby(groups, dropna=False).apply(one).reset_index()


def _pct(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "-" if pd.isna(number) else f"{float(number) * 100:.1f}%"


def build_markdown(summary: pd.DataFrame, by_round: pd.DataFrame) -> str:
    lines = [
        "# Dフィルター観測評価",
        "",
        "prediction本体には未接続。確定結果に対するcounterfactual評価のみを行う。",
        "",
        "## 累積",
        "",
        "| policy | 変更 | Hへ/Aへ | rescue | harm | net | 変更精度 | 的中率 before → after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values("policy").iterrows():
        lines.append(
            f"| {row['policy']} | {int(row['changed_matches'])} | {int(row['changed_to_home'])}/{int(row['changed_to_away'])} "
            f"| {int(row['rescue'])} | {int(row['harm'])} | {int(row['net_rescue']):+d} "
            f"| {_pct(row['change_precision'])} | {_pct(row['baseline_hit_rate'])} → {_pct(row['filtered_hit_rate'])} |"
        )
    lines += [
        "",
        "## 節別",
        "",
        "| 開催回 | policy | 変更 | rescue | harm | net | before → after |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_round.sort_values(["round_id", "policy"]).iterrows():
        lines.append(
            f"| {row['round_id']} | {row['policy']} | {int(row['changed_matches'])} | {int(row['rescue'])} "
            f"| {int(row['harm'])} | {int(row['net_rescue']):+d} | {_pct(row['baseline_hit_rate'])} → {_pct(row['filtered_hit_rate'])} |"
        )
    lines += [
        "",
        "## policy定義",
        "",
        "- `false_draw_only`: J1/J2のfalse_draw_watchだけをrank2側へ変更。",
        "- `false_draw_plus_trap`: false_draw_watchにdraw_trap系を加える。",
        "- `topology_guard`: 強いD根拠を満たすものだけDを保持し、それ以外をrank2側へ変更。",
        "",
        "正式採用条件は、複数節で net rescue が正、harmが許容範囲、リーグ別でも再現すること。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate observation-only D filter candidates")
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir).resolve()
    matches_path = history_dir / "matches.csv"
    if not matches_path.exists():
        raise FileNotFoundError(matches_path)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else history_dir / "reports" / "d_filter"
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(matches_path, low_memory=False)
    detail = build_detail(matches)
    if detail.empty:
        raise RuntimeError("no resolved observations")
    summary = summarize(detail, ["policy"])
    by_round = summarize(detail, ["round_id", "policy"])
    by_league = summarize(detail, ["league", "policy"])
    changed = detail[detail["changed"] == 1].copy()
    by_type = summarize(changed, ["match_purchase_type", "policy"]) if not changed.empty else pd.DataFrame()

    outputs = {
        "d_filter_detail.csv": detail,
        "d_filter_summary.csv": summary,
        "d_filter_by_round.csv": by_round,
        "d_filter_by_league.csv": by_league,
        "d_filter_by_type.csv": by_type,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    report_path = out_dir / "d_filter_validation.md"
    report_path.write_text(build_markdown(summary, by_round), encoding="utf-8")
    print(f"[OK] D filter report: {report_path}")
    print(f"[INFO] observations={len(matches)} policies={len(summary)}")


if __name__ == "__main__":
    main()
