"""Offline comparison of v2.5 match-specific D coefficient multipliers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_draw_portfolio import ExactPortfolio
from match_draw_coefficient_shadow import coefficients, extract
from score_placement_v25_shadow import Config, diagnostics, generate
from shadow_v25_operational import REFERENCE, ROOT

MULTIPLIERS = (1.0, 1.1, 1.2)
HISTORICAL = ("toto1660", "toto1663", "toto1664", "toto1665", "toto1668")


def internal_tickets(plan: pd.DataFrame) -> np.ndarray:
    raw = plan[[f"ticket{i:02d}" for i in range(1, 11)]].to_numpy(dtype=int).T
    if not np.isin(raw, [0, 1, 2]).all():
        raise ValueError("Unknown official ticket symbol")
    return np.where(raw == 1, 0, np.where(raw == 0, 1, 2))


def round_inputs(round_id: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    if round_id == "toto1670":
        base = ROOT / "data/purchase_reference"
        frame = pd.read_csv(base / "next_predictions.csv").sort_values("match_no").reset_index(drop=True)
        official = pd.read_csv(base / "buyplan.csv").sort_values("match_no").reset_index(drop=True)
        manifest = json.loads((base / "manifest.json").read_text())
        if manifest["round_id"] != round_id:
            raise ValueError("Current purchase reference belongs to another round")
        actual = None
    else:
        base = ROOT / "data/eval/toto_rounds" / round_id
        frame = pd.read_csv(base / "snapshot/next_predictions.csv").sort_values("match_no").reset_index(drop=True)
        official = pd.read_csv(base / "snapshot/buyplan.csv").sort_values("match_no").reset_index(drop=True)
        results = pd.read_csv(base / "actual_results.csv").sort_values("match_no").reset_index(drop=True)
        for column in ("match_no", "home_team", "away_team"):
            if frame[column].tolist() != results[column].tolist():
                raise ValueError(f"{round_id}: result identity mismatch")
        raw = results.result.to_numpy(dtype=int)
        actual = np.where(raw == 1, 0, np.where(raw == 0, 1, 2))
    for column in ("match_no", "home_team", "away_team"):
        if frame[column].tolist() != official[column].tolist():
            raise ValueError(f"{round_id}: official identity mismatch")
    if len(frame) != 13:
        raise ValueError(f"{round_id}: expected 13 matches")
    return frame, official, actual


def compare_round(round_id: str, reference: dict, outdir: Path) -> list[dict]:
    frame, official, actual = round_inputs(round_id)
    source_dir = (ROOT / "data/purchase_reference" if round_id == "toto1670"
                  else ROOT / "data/eval/toto_rounds" / round_id / "snapshot")
    manifest_path = source_dir / ("manifest.json" if round_id == "toto1670" else "buyplan_engine_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["output_sha256"]["next_predictions.csv"] != hashlib.sha256((source_dir / "next_predictions.csv").read_bytes()).hexdigest():
        raise ValueError("Forecast does not match generator manifest")
    saved_at = pd.Timestamp(manifest["created_at"])
    kickoffs = pd.to_datetime(frame.datetime)
    if kickoffs.dt.tz is None and saved_at.tzinfo is not None:
        kickoffs = kickoffs.dt.tz_localize("Asia/Tokyo")
    saved_before_kickoff = int((kickoffs > saved_at).sum())
    p = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(dtype=float)
    official_tickets = internal_tickets(official)
    counts = np.stack([(official_tickets == s).sum(axis=0) for s in range(3)], axis=1)
    features = coefficients(extract(frame), reference)
    base_k = features.K_D.to_numpy(dtype=float)
    evaluator = ExactPortfolio(p)
    outdir.mkdir(parents=True, exist_ok=True)
    feature_view = frame[["match_no", "home_team", "away_team"]].copy()
    feature_view["base_K_D"] = base_k
    for mult in MULTIPLIERS:
        feature_view[f"final_K_D_x{mult:.1f}"] = base_k * mult
    feature_view.to_csv(outdir / "coefficients.csv", index=False)
    rows, candidate_rows, placements, audits = [], [], {}, {}
    for mult in MULTIPLIERS:
        # No post-multiplier cap: base limits remain in coefficients().
        final_k = base_k * mult
        tickets, audit = generate(p, counts, Config(draw_coefficient=1.0), draw_coefficients=final_k)
        for symbol in range(3):
            np.testing.assert_array_equal((tickets == symbol).sum(axis=0), counts[:, symbol])
        if not np.allclose(p, frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy()):
            raise AssertionError("Original probabilities changed")
        if audit["duplicate_pairs"]:
            raise AssertionError(f"{round_id} x{mult}: duplicate tickets")
        diag = pd.DataFrame(diagnostics(p, tickets, audit))
        target_error = int(np.abs(diag.iloc[1:].exchanges.to_numpy() - diag.iloc[1:].target.to_numpy()).sum())
        metrics = evaluator.metrics(tickets)
        d_counts = (tickets == 1).sum(axis=1)
        row = {"round": round_id, "multiplier": mult,
               "saved_before_kickoff": saved_before_kickoff,
               "D_01_03": int(d_counts[:3].sum()), "D_01_05": int(d_counts[:5].sum()),
               "D_06_10": int(d_counts[5:].sum()), "D_total": int(d_counts.sum()),
               "target_error": target_error, "duplicates": len(audit["duplicate_pairs"]),
               "actual_best": None if actual is None else int((tickets == actual).sum(axis=1).max()),
               "base_K_min": float(base_k.min()), "base_K_max": float(base_k.max()),
               "final_K_min": float(final_k.min()), "final_K_max": float(final_k.max()),
               "accepted_moves": audit["accepted_moves"],
               **{k: metrics[k] for k in ("p10", "p11", "p12", "p13")}}
        rows.append(row)
        diag.insert(0, "multiplier", mult)
        candidate_rows.append(diag)
        placements[mult] = tickets
        audits[f"x{mult:.1f}"] = audit
    pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)
    pd.concat(candidate_rows, ignore_index=True).to_csv(outdir / "candidates.csv", index=False)
    changes = []
    baseline = placements[1.0]
    for mult in MULTIPLIERS[1:]:
        tickets = placements[mult]
        for m, match in frame.iterrows():
            old = np.flatnonzero(baseline[:, m] == 1) + 1
            new = np.flatnonzero(tickets[:, m] == 1) + 1
            changes.append({"round": round_id, "multiplier": mult,
                            "match_no": int(match.match_no), "home_team": match.home_team,
                            "away_team": match.away_team,
                            "D_before": ",".join(f"{x:02d}" for x in old),
                            "D_after": ",".join(f"{x:02d}" for x in new),
                            "D_removed": ",".join(f"{x:02d}" for x in sorted(set(old)-set(new))),
                            "D_added": ",".join(f"{x:02d}" for x in sorted(set(new)-set(old)))})
    pd.DataFrame(changes).to_csv(outdir / "D_movements.csv", index=False)
    (outdir / "audits.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2))
    if round_id == "toto1670":
        saved = pd.read_csv(ROOT / "data/purchase_reference/shadow_buyplan_v25.csv")
        for i in range(1, 11):
            column = f"ticket{i:02d}"
            expected = np.where(baseline[i-1] == 0, 1, np.where(baseline[i-1] == 1, 0, 2))
            np.testing.assert_array_equal(saved[column].to_numpy(), expected)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/reports/v25_match_multiplier")
    parser.add_argument("--include-history", action="store_true")
    args = parser.parse_args()
    reference_bytes = REFERENCE.read_bytes()
    reference = json.loads(reference_bytes)
    rounds = ("toto1670",) + (HISTORICAL if args.include_history else ())
    all_rows = []
    for round_id in rounds:
        all_rows += compare_round(round_id, reference, args.outdir / round_id)
        print(f"[OK] {round_id}", flush=True)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(args.outdir / "metrics.csv", index=False)
    (args.outdir / "config.json").write_text(json.dumps({
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "base_coefficient_bounds": [reference["config"]["lower"], reference["config"]["upper"]],
        "multipliers": MULTIPLIERS, "final_coefficient_bound": None,
        "final_coefficient_policy": "multiply clipped base K; no post-multiplier clipping",
        "official_modified": False, "standard_shadow_modified": False,
        "model_probability_source": "saved next_predictions; independent match outcomes",
    }, indent=2))
    current = metrics[metrics["round"] == "toto1670"].copy()
    for key in ("p10", "p11", "p12", "p13"):
        current[key] = (100 * current[key]).map(lambda x: f"{x:.4f}%")
    all_mean = metrics[metrics["round"] != "toto1670"].groupby("multiplier", as_index=False).mean(numeric_only=True) if args.include_history else None
    body = "<p>購入対象外の比較。試合別元係数は固定基準で算出し、その後に倍率を適用。最終係数を1.20で再制限しない。予測確率、v2.4、標準v2.5は変更しない。高的中確率は13試合の独立結果分布を全列挙し、10口の最高的中数で計算する。</p>"
    body += "<h2>1670回</h2>" + current.to_html(index=False)
    body += '<p><a href="toto1670/candidates.csv">候補別診断</a> ／ <a href="toto1670/D_movements.csv">試合別D移動</a> ／ <a href="toto1670/coefficients.csv">係数</a></p>'
    if all_mean is not None:
        body += "<p>過去5開催は試合前に生成された保存予測の感度比較です。生成記録の時刻とCSVハッシュを照合しています。saved_before_kickoffは評価フォルダへのコピー時刻ではなく予測生成時刻で判定します。ただし今回の係数設計・比較は事後のため独立した将来検証ではありません。</p>"
        body += "<h2>過去5開催の平均</h2>" + all_mean.to_html(index=False)
        body += "<h2>全開催の結果</h2>" + metrics.to_html(index=False)
    body += '<p><a href="metrics.csv">集計CSV</a> ／ <a href="config.json">設定</a></p>'
    (args.outdir / "index.html").write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>v2.5 試合別D係数倍率比較</title><h1>v2.5 試合別D係数倍率比較</h1>' + body + '</html>')
    print(args.outdir / "index.html")


if __name__ == "__main__":
    main()
