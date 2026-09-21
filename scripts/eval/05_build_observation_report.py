#!/usr/bin/env python3
"""Build cross-round reports from immutable regular-season observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = ROOT_DIR / "data" / "eval" / "observation_history"
SYMBOL_LABEL = {0: "D", 1: "H", 2: "A"}


def _symbol(value: object) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number) and int(number) in SYMBOL_LABEL:
        return int(number)
    text = str(value or "").strip().upper()
    return {"D": 0, "H": 1, "A": 2}.get(text)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def prepare(matches: pd.DataFrame) -> pd.DataFrame:
    out = matches.copy()
    out["actual_symbol"] = out["actual_result"].map(_symbol)
    out["prediction_symbol"] = out["predicted_result"].map(_symbol)
    out["prediction_hit"] = (
        out["actual_symbol"].notna()
        & out["prediction_symbol"].notna()
        & out["actual_symbol"].eq(out["prediction_symbol"])
    ).astype(int)
    out["predicted_draw"] = out["prediction_symbol"].eq(0).astype(int)
    out["actual_draw"] = out["actual_symbol"].eq(0).astype(int)
    out["draw_prediction_hit"] = (out["predicted_draw"].eq(1) & out["actual_draw"].eq(1)).astype(int)
    out["portfolio_rescue_match"] = (
        out["prediction_hit"].eq(0) & pd.to_numeric(out["portfolio_covered"], errors="coerce").fillna(0).eq(1)
    ).astype(int)
    out["portfolio_harm_ticket_count"] = pd.to_numeric(
        out.get("harm_vs_ticket01_count", 0), errors="coerce"
    ).fillna(0).astype(int)
    out["portfolio_rescue_ticket_count"] = pd.to_numeric(
        out.get("rescue_vs_ticket01_count", 0), errors="coerce"
    ).fillna(0).astype(int)
    return out


def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    def summarize(group: pd.DataFrame) -> pd.Series:
        predicted_draws = int(group["predicted_draw"].sum())
        actual_draws = int(group["actual_draw"].sum())
        draw_hits = int(group["draw_prediction_hit"].sum())
        return pd.Series(
            {
                "matches": len(group),
                "actual_h": int(group["actual_symbol"].eq(1).sum()),
                "actual_d": actual_draws,
                "actual_a": int(group["actual_symbol"].eq(2).sum()),
                "actual_draw_rate": actual_draws / len(group) if len(group) else 0.0,
                "mean_p_draw": pd.to_numeric(group.get("p_draw"), errors="coerce").mean(),
                "prediction_hits": int(group["prediction_hit"].sum()),
                "prediction_hit_rate": float(group["prediction_hit"].mean()),
                "predicted_draws": predicted_draws,
                "draw_prediction_hits": draw_hits,
                "draw_precision": draw_hits / predicted_draws if predicted_draws else float("nan"),
                "draw_recall": draw_hits / actual_draws if actual_draws else float("nan"),
                "portfolio_covered": int(pd.to_numeric(group["portfolio_covered"], errors="coerce").fillna(0).sum()),
                "portfolio_rescue_matches": int(group["portfolio_rescue_match"].sum()),
                "rescue_ticket_count": int(group["portfolio_rescue_ticket_count"].sum()),
                "harm_ticket_count": int(group["portfolio_harm_ticket_count"].sum()),
                "net_rescue_tickets": int(
                    group["portfolio_rescue_ticket_count"].sum() - group["portfolio_harm_ticket_count"].sum()
                ),
                "all_same_matches": int(pd.to_numeric(group["all_same_symbol"], errors="coerce").fillna(0).sum()),
            }
        )

    if groups:
        return frame.groupby(groups, dropna=False).apply(summarize).reset_index()
    return summarize(frame).to_frame().T


def candidate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for number in range(1, 11):
        symbol_col = f"ticket{number:02d}_symbol"
        if symbol_col not in frame.columns:
            continue
        picks = frame[symbol_col].map(_symbol)
        valid = picks.notna() & frame["actual_symbol"].notna()
        hits = picks.eq(frame["actual_symbol"]) & valid
        draw_picks = picks.eq(0) & valid
        draw_hits = draw_picks & frame["actual_symbol"].eq(0)
        rows.append(
            {
                "candidate_id": f"cand{number:02d}",
                "hits": int(hits.sum()),
                "total": int(valid.sum()),
                "hit_rate": float(hits.sum() / valid.sum()) if valid.sum() else float("nan"),
                "draw_pred_count": int(draw_picks.sum()),
                "draw_hit_count": int(draw_hits.sum()),
                "draw_precision": float(draw_hits.sum() / draw_picks.sum()) if draw_picks.sum() else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["hits", "candidate_id"], ascending=[False, True])


def flag_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for flag in ["anchor_candidate_flag", "draw_core_candidate_flag", "away_overread_draw_cover_flag"]:
        selected = frame[_bool_series(frame, flag)]
        if selected.empty:
            continue
        part = aggregate(selected, ["league"])
        part.insert(0, "flag_name", flag)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pct(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "-" if pd.isna(number) else f"{float(number) * 100:.1f}%"


def build_markdown(overall: pd.DataFrame, rounds: pd.DataFrame, types: pd.DataFrame, candidates: pd.DataFrame) -> str:
    top = overall.iloc[0]
    lines = [
        "# 2026/27 新シーズン横断検証",
        "",
        "このレポートは `data/eval/observation_history/matches.csv` の確定済みtoto対象試合だけを集計する。",
        "J1/J2全試合評価ではなく、購入対象13試合/開催回の検証である。",
        "",
        "## 全体",
        "",
        f"- 開催回: {len(rounds)}",
        f"- 確定試合: {int(top['matches'])}",
        f"- 実結果 H/D/A: {int(top['actual_h'])}/{int(top['actual_d'])}/{int(top['actual_a'])}",
        f"- prediction的中: {int(top['prediction_hits'])}/{int(top['matches'])} ({_pct(top['prediction_hit_rate'])})",
        f"- prediction D予想: {int(top['predicted_draws'])}、D的中: {int(top['draw_prediction_hits'])}",
        f"- D precision / recall: {_pct(top['draw_precision'])} / {_pct(top['draw_recall'])}",
        f"- portfolio cover: {int(top['portfolio_covered'])}/{int(top['matches'])}",
        f"- prediction外れをportfolioが救済: {int(top['portfolio_rescue_matches'])}試合",
        f"- all-same: {int(top['all_same_matches'])}試合",
        "",
        "## 開催回別",
        "",
        "| 開催回 | 試合 | H/D/A | prediction | D予想/的中 | D precision | portfolio cover |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rounds.iterrows():
        lines.append(
            f"| {row['round_id']} | {int(row['matches'])} | {int(row['actual_h'])}/{int(row['actual_d'])}/{int(row['actual_a'])} "
            f"| {int(row['prediction_hits'])}/{int(row['matches'])} | {int(row['predicted_draws'])}/{int(row['draw_prediction_hits'])} "
            f"| {_pct(row['draw_precision'])} | {int(row['portfolio_covered'])}/{int(row['matches'])} |"
        )
    lines += ["", "## D分類・試合型", "", "| type | 試合 | H/D/A | prediction | D予想/的中 | D precision | rescue試合 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in types.sort_values(["matches", "match_purchase_type"], ascending=[False, True]).iterrows():
        lines.append(
            f"| {row['match_purchase_type']} | {int(row['matches'])} | {int(row['actual_h'])}/{int(row['actual_d'])}/{int(row['actual_a'])} "
            f"| {int(row['prediction_hits'])}/{int(row['matches'])} | {int(row['predicted_draws'])}/{int(row['draw_prediction_hits'])} "
            f"| {_pct(row['draw_precision'])} | {int(row['portfolio_rescue_matches'])} |"
        )
    lines += ["", "## 候補累積", "", "| 候補 | 的中 | 的中率 | D予想/的中 | D precision |", "|---|---:|---:|---:|---:|"]
    for _, row in candidates.iterrows():
        lines.append(
            f"| {row['candidate_id']} | {int(row['hits'])}/{int(row['total'])} | {_pct(row['hit_rate'])} "
            f"| {int(row['draw_pred_count'])}/{int(row['draw_hit_count'])} | {_pct(row['draw_precision'])} |"
        )
    lines += [
        "",
        "## 判定上の注意",
        "",
        "- サンプル数が少ない型は正式policyへ昇格させない。",
        "- D分類は precision・recall・rescue・harmを複数節で確認する。",
        "- 分布生成器の学習には、別途J1/J2全試合の確定結果を蓄積する必要がある。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-round regular-season observation reports")
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir).resolve()
    matches_path = history_dir / "matches.csv"
    if not matches_path.exists():
        raise FileNotFoundError(matches_path)
    frame = prepare(pd.read_csv(matches_path, low_memory=False))
    if frame.empty:
        raise RuntimeError("observation history is empty")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else history_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "overall_summary.csv": aggregate(frame, []),
        "round_summary.csv": aggregate(frame, ["round_id"]),
        "league_summary.csv": aggregate(frame, ["league"]),
        "purchase_type_summary.csv": aggregate(frame, ["match_purchase_type"]),
        "purchase_subtype_summary.csv": aggregate(frame, ["match_purchase_type", "match_purchase_subtype"]),
        "draw_tier_summary.csv": aggregate(frame, ["draw_purchase_tier"]),
        "candidate_summary.csv": candidate_summary(frame),
        "observation_flag_summary.csv": flag_summary(frame),
    }
    for name, data in outputs.items():
        data.to_csv(out_dir / name, index=False, encoding="utf-8-sig")

    report = build_markdown(
        outputs["overall_summary.csv"],
        outputs["round_summary.csv"],
        outputs["purchase_type_summary.csv"],
        outputs["candidate_summary.csv"],
    )
    (out_dir / "new_season_validation.md").write_text(report, encoding="utf-8")
    manifest = {
        "source": str(matches_path),
        "rounds": int(frame["round_id"].nunique()),
        "matches": int(len(frame)),
        "outputs": sorted([*outputs, "new_season_validation.md"]),
        "scope": "scored toto purchase matches only",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] observation report: {out_dir / 'new_season_validation.md'}")
    print(f"[INFO] rounds={manifest['rounds']} matches={manifest['matches']} outputs={len(manifest['outputs'])}")


if __name__ == "__main__":
    main()
