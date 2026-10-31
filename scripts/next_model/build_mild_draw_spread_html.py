#!/usr/bin/env python3
"""Build a separate, non-purchase mild draw-spread comparison from v2.2."""

import argparse
import hashlib
import json
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from proportional_buyplan import build_proportional_tickets
from simulate_draw_balance import draw_spread_limit, is_valid, optimize_balanced
from ticket_coverage_optimizer import optimize_ten_hit_coverage


ROOT = Path(__file__).resolve().parents[2]
TICKET_COLUMNS = [f"ticket{i:02d}" for i in range(1, 11)]
PROBABILITY_COLUMNS = ["next_prob_home", "next_prob_draw", "next_prob_away"]
RESULT_TO_INDEX = {1: 0, 0: 1, 2: 2}
INDEX_TO_RESULT = np.array([1, 0, 2])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_indices(frame):
    values = frame[TICKET_COLUMNS].to_numpy(int).T
    if not np.isin(values, list(RESULT_TO_INDEX)).all():
        raise ValueError("Invalid H/D/A ticket values")
    return np.vectorize(RESULT_TO_INDEX.__getitem__)(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/purchase_reference")
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    source = args.source_dir
    official_path = source / "buyplan.csv"
    source_manifest = source / "manifest.json"
    official = pd.read_csv(official_path, encoding="utf-8-sig").sort_values("match_no").reset_index(drop=True)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    round_id = manifest["round_id"]
    if manifest.get("official_version") != "v2.2":
        raise ValueError("Source must be an official v2.2 BuyPlan")
    if len(official) != 13 or official.match_no.astype(int).tolist() != list(range(1, 14)):
        raise ValueError("Source BuyPlan must have match_no 1..13")
    p = official[PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Invalid H/D/A probabilities")

    initial, _ = build_proportional_tickets(p)
    reproduced, _ = optimize_ten_hit_coverage(initial, p)
    if not np.array_equal(reproduced, as_indices(official)):
        raise ValueError("Source official tickets do not match the v2.2 algorithm and probabilities")
    mild = optimize_balanced(initial, p, maximum_draws=7, square_slack=40)
    if not is_valid(mild, draw_spread_limit(mild, 40), 7):
        raise ValueError("Mild draw-spread constraints failed")
    for symbol in range(3):
        np.testing.assert_array_equal((mild == symbol).sum(axis=0), (reproduced == symbol).sum(axis=0))

    outdir = args.outdir or source / "draw_spread_mild" / round_id
    if outdir.exists():
        raise FileExistsError(f"Comparison output already exists: {outdir}")
    outdir.mkdir(parents=True)
    comparison = official.copy()
    for i, column in enumerate(TICKET_COLUMNS):
        comparison[column] = INDEX_TO_RESULT[mild[i]]
    csv_path = outdir / "buyplan_draw_spread_mild.csv"
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")

    labels = {1: "H", 0: "D", 2: "A"}
    view = comparison[["match_no", "home_team", "away_team", "draw_filter_class", *TICKET_COLUMNS]].copy()
    view.insert(1, "対戦", view.home_team.astype(str) + "－" + view.away_team.astype(str))
    view = view.drop(columns=["home_team", "away_team"]).rename(columns={"match_no": "No.", "draw_filter_class": "D分類"})
    for column in TICKET_COLUMNS:
        view[column] = view[column].map(labels)
    official_draws = (as_indices(official) == 1).sum(axis=1).tolist()
    mild_draws = (mild == 1).sum(axis=1).tolist()
    count_table = pd.DataFrame({"候補": range(1, 11), "現行v2.2のD数": official_draws, "弱い制約のD数": mild_draws})
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>{escape(round_id)} BuyPlan 弱いD分散制約</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#172033}}table{{border-collapse:collapse;margin:16px 0}}
th,td{{border:1px solid #ccd5df;padding:7px;text-align:center}}th{{background:#edf2f7}}
td:nth-child(2){{white-space:nowrap}}.note{{background:#fff6df;border-left:5px solid #d99000;padding:12px}}
</style><h1>{escape(round_id)} BuyPlan 弱いD分散制約（比較用・10口）</h1>
<p class="note"><b>購入対象外の比較版です。</b> 現行v2.2とは別に生成しました。各試合のH/D/A採用数を固定し、1口のD数を1～7、D数の二乗和を均等配分の最小値+40以下に制限しています。候補番号は現行券との同一性を示しません。</p>
<p>現行v2.2: {escape(str(sorted(official_draws)))} ／ 弱い制約: {escape(str(sorted(mild_draws)))}</p>
{view.to_html(index=False, escape=True)}
<h2>候補ごとのD数</h2>{count_table.to_html(index=False, escape=True)}
<p>出典: {escape(str(official_path))}</p></html>"""
    html_path = outdir / "buyplan_draw_spread_mild.html"
    html_path.write_text(html, encoding="utf-8")
    record = {
        "round_id": round_id,
        "status": "non-purchase comparison",
        "created_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "source_buyplan": str(official_path.resolve()),
        "source_buyplan_sha256": digest(official_path),
        "source_manifest_sha256": digest(source_manifest),
        "constraints": {"D_per_ticket_min": 1, "D_per_ticket_max": 7, "D_square_slack": 40},
        "official_D_counts": official_draws,
        "mild_D_counts": mild_draws,
        "HDA_per_match_preserved": True,
        "v22_reproduction_verified": True,
        "output_sha256": {path.name: digest(path) for path in (csv_path, html_path)},
    }
    (outdir / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(html_path)


if __name__ == "__main__":
    main()
