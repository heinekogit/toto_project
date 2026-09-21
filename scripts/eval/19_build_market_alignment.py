#!/usr/bin/env python3
"""Build post-match market-alignment observations from a saved toto PDF.

The extracted vote shares are evaluation-only evidence.  This script does not
write to prediction snapshots, buyplans, or any model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY = ROOT / "data" / "eval" / "observation_history"
SYMBOLS = ("1", "0", "2")
PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_pdf(round_id: str, explicit: str = "") -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"toto PDF not found: {path}")
        return path

    number = round_id.removeprefix("toto")
    preferred = [ROOT / f"toto{number}.pdf", ROOT / f"{number}.pdf"]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted(ROOT.glob(f"*toto{number}.pdf"))
    if len(candidates) > 1:
        raise RuntimeError(f"multiple toto PDFs found for {round_id}: {candidates}")
    return candidates[0] if candidates else None


def extract_vote_shares(pdf_path: Path) -> list[dict[str, float]]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    values = [float(value) / 100.0 for value in PERCENT_RE.findall(text)]
    if len(values) != 39:
        raise ValueError(
            f"expected 39 vote-share percentages (13 matches x 3), found {len(values)}: {pdf_path}"
        )

    matches: list[dict[str, float]] = []
    for index in range(13):
        triple = values[index * 3 : index * 3 + 3]
        if not math.isclose(sum(triple), 1.0, abs_tol=0.002):
            raise ValueError(f"vote shares do not sum to 100% at match {index + 1}: {triple}")
        matches.append(dict(zip(SYMBOLS, triple)))
    return matches


def build_match_rows(round_id: str, shares: list[dict[str, float]], actual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, source in actual.iterrows():
        status = str(source.get("status", "OK")).strip().upper()
        result = str(source.get("result", "")).strip().removesuffix(".0")
        if status != "OK" or result not in SYMBOLS:
            continue
        match_no = int(float(source["match_no"]))
        if not 1 <= match_no <= 13:
            raise ValueError(f"match_no out of range: {match_no}")
        vote = shares[match_no - 1]
        favorite = max(SYMBOLS, key=lambda symbol: vote[symbol])
        favorite_share = vote[favorite]
        actual_share = vote[result]
        rows.append(
            {
                "round_id": round_id,
                "match_no": match_no,
                "home_team": source.get("home_team"),
                "away_team": source.get("away_team"),
                "actual_result": result,
                "market_p_home": vote["1"],
                "market_p_draw": vote["0"],
                "market_p_away": vote["2"],
                "market_favorite": favorite,
                "market_favorite_share": favorite_share,
                "actual_market_share": actual_share,
                "market_favorite_hit": int(favorite == result),
                "strong_favorite_60": int(favorite_share >= 0.60),
                "strong_favorite_60_miss": int(favorite_share >= 0.60 and favorite != result),
                "market_surprise_log": -math.log(max(actual_share, 1e-15)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("match_no").reset_index(drop=True)
    if len(frame) != 13:
        raise ValueError(f"expected 13 resolved actual results, found {len(frame)}")
    return frame


def build_round_summary(round_id: str, rows: pd.DataFrame, pdf_path: Path) -> dict[str, object]:
    return {
        "round_id": round_id,
        "matches": len(rows),
        "market_favorite_hits": int(rows["market_favorite_hit"].sum()),
        "market_favorite_hit_rate": float(rows["market_favorite_hit"].mean()),
        "mean_actual_market_share": float(rows["actual_market_share"].mean()),
        "geometric_mean_actual_market_share": float(
            math.exp(rows["actual_market_share"].clip(lower=1e-15).map(math.log).mean())
        ),
        "market_log_loss": float(rows["market_surprise_log"].mean()),
        "strong_favorite_60_count": int(rows["strong_favorite_60"].sum()),
        "strong_favorite_60_misses": int(rows["strong_favorite_60_miss"].sum()),
        "source_pdf": pdf_path.name,
        "source_pdf_sha256": sha256(pdf_path),
    }


def upsert(path: Path, frame: pd.DataFrame, keys: list[str]) -> None:
    old = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    combined = pd.concat([old, frame], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accumulate evaluation-only toto market alignment")
    parser.add_argument("--round", required=True, dest="round_id", help="toto1653")
    parser.add_argument("--pdf", default="", help="Optional explicit toto PDF path")
    parser.add_argument("--actual", default="", help="Optional explicit actual_results.csv path")
    parser.add_argument("--round-dir", default="", help="Optional evaluation round directory")
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.round_id.startswith("toto"):
        print(f"[INFO] market alignment skipped for non-toto round: {args.round_id}")
        return

    pdf_path = find_pdf(args.round_id, args.pdf)
    if pdf_path is None:
        print(f"[INFO] market alignment skipped: PDF not found for {args.round_id}")
        return

    round_dir = (
        Path(args.round_dir).expanduser().resolve()
        if args.round_dir
        else ROOT / "data" / "eval" / "toto_rounds" / args.round_id
    )
    actual_path = Path(args.actual).expanduser().resolve() if args.actual else round_dir / "actual_results.csv"
    if not actual_path.exists():
        raise FileNotFoundError(f"actual results not found: {actual_path}")

    shares = extract_vote_shares(pdf_path)
    actual = pd.read_csv(actual_path, low_memory=False)
    match_rows = build_match_rows(args.round_id, shares, actual)
    summary = build_round_summary(args.round_id, match_rows, pdf_path)

    round_dir.mkdir(parents=True, exist_ok=True)
    detail_path = round_dir / "market_alignment.csv"
    summary_path = round_dir / "market_alignment.json"
    match_rows.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history_dir = Path(args.history_dir).expanduser().resolve()
    upsert(history_dir / "market_alignment_matches.csv", match_rows, ["round_id", "match_no"])
    upsert(history_dir / "market_alignment_rounds.csv", pd.DataFrame([summary]), ["round_id"])

    provenance_dir = history_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = provenance_dir / f"{args.round_id}_market_alignment.json"
    provenance_path.write_text(
        json.dumps(
            {
                "round_id": args.round_id,
                "accumulated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "sources": {
                    "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
                    "actual": {"path": str(actual_path), "sha256": sha256(actual_path)},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] market alignment detail: {detail_path}")
    print(f"[OK] market alignment summary: {summary_path}")
    print(f"[OK] market alignment history: {history_dir / 'market_alignment_rounds.csv'}")


if __name__ == "__main__":
    main()
