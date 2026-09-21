#!/usr/bin/env python3
"""Archive one immutable pre-match buyplan observation with its inputs."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SRCDIR = ROOT_DIR / "data" / "purchase_reference"
DEFAULT_HISTORY = DEFAULT_SRCDIR / "observation_history"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_token(value: object) -> str:
    text = str(value or "").strip()
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="buyplan検証用データを上書きなしで履歴保存")
    p.add_argument("--logical-season", required=True, help="toto上のシーズン。例: 2027")
    p.add_argument("--round", type=int, required=True, dest="round_no", help="Jリーグ節番号")
    p.add_argument("--prediction-season", required=True, help="予測CSV上のシーズン。例: 2026")
    p.add_argument("--srcdir", default=str(DEFAULT_SRCDIR))
    p.add_argument("--history-root", default=str(DEFAULT_HISTORY))
    p.add_argument("--label", default="standard", help="standard / pre_guardrail などの識別名")
    p.add_argument("--buyplan-csv", default="", help="比較用buyplan CSVを明示する場合")
    p.add_argument("--buyplan-html", default="", help="比較用buyplan HTMLを明示する場合")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    srcdir = Path(args.srcdir).resolve()
    prediction_csv = srcdir / "predictions.csv"
    context_csv = srcdir / "predictions_buyplan_context.csv"
    buyplan_csv = Path(args.buyplan_csv).resolve() if args.buyplan_csv else srcdir / "buyplan.csv"
    buyplan_html = Path(args.buyplan_html).resolve() if args.buyplan_html else srcdir / "buyplan.html"
    diff_csv = srcdir / "buyplan_toto_diff.csv"

    required = [prediction_csv, context_csv, buyplan_csv, buyplan_html]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("必要ファイルがありません: " + ", ".join(missing))

    predictions = pd.read_csv(prediction_csv, low_memory=False)
    buyplan = pd.read_csv(buyplan_csv, low_memory=False)
    round_ids = sorted({str(v).strip() for v in predictions.get("toto_round_id", pd.Series(dtype=str)).dropna() if str(v).strip()})
    if len(round_ids) != 1:
        raise RuntimeError(f"toto_round_idを1件に確定できません: {round_ids}")
    toto_round_id = safe_token(round_ids[0])

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    label = safe_token(args.label) or "standard"
    outdir = (
        Path(args.history_root).resolve()
        / safe_token(args.logical_season)
        / toto_round_id
        / f"{timestamp}_{label}"
    )
    outdir.mkdir(parents=True, exist_ok=False)

    sources = {
        "predictions.csv": prediction_csv,
        "predictions_buyplan_context.csv": context_csv,
        "buyplan.csv": buyplan_csv,
        "buyplan.html": buyplan_html,
    }
    if diff_csv.is_file() and not args.buyplan_csv:
        sources["buyplan_toto_diff.csv"] = diff_csv

    copied = {}
    for name, source in sources.items():
        destination = outdir / name
        shutil.copy2(source, destination)
        copied[name] = {
            "source": str(source),
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    ticket_cols = [f"ticket{i:02d}" for i in range(1, 11) if f"ticket{i:02d}" in buyplan.columns]
    draw_counts = {
        col: int(buyplan[col].astype(str).str.strip().eq("0").sum())
        for col in ticket_cols
    }
    matches = [
        {
            "match_no": int(row["match_no"]),
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
        }
        for _, row in buyplan.iterrows()
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "logical_season": str(args.logical_season),
        "prediction_season": str(args.prediction_season),
        "league_round": int(args.round_no),
        "toto_round_id": toto_round_id,
        "label": label,
        "prediction_rows": int(len(predictions)),
        "buyplan_matches": int(len(buyplan)),
        "draw_counts": draw_counts,
        "draw_total": int(sum(draw_counts.values())),
        "matches": matches,
        "buyplan_script": {
            "path": str(ROOT_DIR / "buyplan.py"),
            "sha256": sha256_file(ROOT_DIR / "buyplan.py"),
        },
        "files": copied,
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] buyplan observation archived: {outdir}")
    print(f"[SUMMARY] toto={toto_round_id} matches={len(buyplan)} draw_total={manifest['draw_total']} label={label}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
