"""Append-only prospective condition-S observations, separate from model and BuyPlan."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, sha256
from forecast_registry import DEFAULT_ROOT, JST, append_unique, digest, forecast_events, read_log, now


VERSION = "condition_S_p00_le_median_p11_ge_median_v1"
FORECAST_LOG = "condition_s_forecasts.jsonl"
RESULT_LOG = "condition_s_results.jsonl"
FAILURE_LOG = "condition_s_failures.jsonl"
DEADLINE_LOG = "condition_s_deadlines.jsonl"


def classify(p00: np.ndarray, p11: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Frozen inclusive boundary on raw float64 score probabilities."""
    median00 = float(np.median(p00))
    median11 = float(np.median(p11))
    return (p00 <= median00) & (p11 >= median11), median00, median11


def observations(forecast_path: Path, manifest_path: Path, *, runtime_code_hash: str | None = None) -> list[dict]:
    """Use the already validated prematch bundle; never recompute forecasts later."""
    events = forecast_events(forecast_path, manifest_path, runtime_code_hash=runtime_code_hash)
    if len(events) != 13 or [e["match_no"] for e in events] != list(range(1, 14)):
        raise ValueError("condition S requires an ordered 13-match prematch round")
    # forecast_events verifies saved lambda, score distribution, mix, final H/D/A,
    # manifest hash and creation time before every kickoff.
    # Use the saved scoreline probabilities; no proxy or post-match rebuild.
    frame = pd.read_csv(forecast_path, encoding="utf-8-sig", low_memory=False).sort_values("match_no")
    if not {"next_prob_0_0", "next_prob_1_1"}.issubset(frame.columns):
        raise ValueError("saved P(0-0)/P(1-1) columns are required")
    p00 = frame["next_prob_0_0"].to_numpy(float)
    p11 = frame["next_prob_1_1"].to_numpy(float)
    if not np.isfinite(p00).all() or not np.isfinite(p11).all():
        raise ValueError("condition S scoreline probabilities are missing")
    flags, median00, median11 = classify(p00, p11)
    source_hash = sha256(Path(forecast_path))
    records = []
    for event, a, b, flag in zip(events, p00, p11, flags):
        records.append(dict(event_type="condition_s_forecast", event_id=digest([VERSION, source_hash, event["match_no"]]),
            forecast_event_id=event["event_id"], fixture_id=event["fixture_id"], round_id=event["round_id"],
            match_no=event["match_no"], home_team=event["home_team"], away_team=event["away_team"],
            kickoff=event["kickoff"], forecast_created_at=event["forecast_created_at"],
            snapshot_id=source_hash, source_file=event["source_file"], manifest_sha256=event["manifest_sha256"],
            input_path=event["input_path"], input_sha256=event["input_sha256"],
            model_version=event["model_version"], model_signature=event["model_signature"],
            code_version=event["code_version"], code_version_source=event["code_version_source"],
            correction_settings=event["correction_settings"], uncertainty_mix=event["uncertainty_mix"],
            lambda_home=event["lambda_home"], lambda_away=event["lambda_away"],
            final_prob_home=event["final_prob_home"], final_prob_draw=event["final_prob_draw"],
            final_prob_away=event["final_prob_away"], P_0_0=float(a), P_1_1=float(b),
            round_median_P_0_0=median00, round_median_P_1_1=median11,
            condition_S=bool(flag), condition_version=VERSION,
            provenance_class="A", registered_at=now()))
    return records


def capture(forecast_path: Path, manifest_path: Path, *, root: Path = DEFAULT_ROOT,
            runtime_code_hash: str | None = None) -> dict:
    records = observations(Path(forecast_path), Path(manifest_path), runtime_code_hash=runtime_code_hash)
    captured_at = pd.Timestamp(now())
    if captured_at >= min(pd.Timestamp(r["kickoff"]).tz_localize(JST) if pd.Timestamp(r["kickoff"]).tzinfo is None
                          else pd.Timestamp(r["kickoff"]) for r in records):
        raise ValueError("condition S prospective capture refused after first kickoff")
    return dict(round_id=records[0]["round_id"], total=len(records), S_count=sum(r["condition_S"] for r in records),
                added=append_unique(Path(root), FORECAST_LOG, records))


def sync_results(*, root: Path = DEFAULT_ROOT) -> dict:
    """Link already registered results; corrections append a new event."""
    root = Path(root)
    forecasts = read_log(root / FORECAST_LOG)
    latest = {}
    for result in read_log(root / "results.jsonl"):
        latest[result["fixture_id"]] = result
    records = []
    for forecast in forecasts:
        result = latest.get(forecast["fixture_id"])
        if result is None:
            continue
        actual = "H" if result["result"] == 1 else "D" if result["result"] == 0 else "A"
        records.append(dict(event_type="condition_s_result",
            event_id=digest(["condition_s_result", forecast["event_id"], result["event_id"]]),
            condition_s_event_id=forecast["event_id"], forecast_event_id=forecast["forecast_event_id"],
            fixture_id=forecast["fixture_id"], round_id=forecast["round_id"], match_no=forecast["match_no"],
            snapshot_id=forecast["snapshot_id"], model_version=forecast["model_version"],
            model_signature=forecast["model_signature"], code_version=forecast["code_version"],
            condition_S=forecast["condition_S"], predicted_P_D=forecast["final_prob_draw"],
            actual_symbol=actual, actual_D=actual == "D", home_score=result["home_score"],
            away_score=result["away_score"], result_event_id=result["event_id"],
            result_recorded_at=result["result_recorded_at"], linked_at=now()))
    return dict(known_results=len(latest), matched=len(records), added=append_unique(root, RESULT_LOG, records))


def record_failure(stage: str, reason: str, *, round_id: str = "", source: str = "",
                   root: Path = DEFAULT_ROOT) -> dict:
    timestamp = now()
    event = dict(event_type="condition_s_failure", event_id=digest([timestamp, stage, round_id, source, reason]),
                 recorded_at=timestamp, stage=stage, round_id=round_id, source=source, reason=reason)
    return dict(added=append_unique(Path(root), FAILURE_LOG, [event]))


def record_deadline(round_id: str, deadline_at: str, source: str, *, root: Path = DEFAULT_ROOT) -> dict:
    if not round_id or not source:
        raise ValueError("round and deadline source are required")
    deadline = pd.Timestamp(deadline_at)
    if deadline.tzinfo is None:
        raise ValueError("purchase deadline must include timezone")
    event = dict(event_type="condition_s_deadline", event_id=digest([round_id, deadline.isoformat(), source]),
                 round_id=round_id, purchase_deadline_at=deadline.isoformat(), source=source, recorded_at=now())
    return dict(added=append_unique(Path(root), DEADLINE_LOG, [event]))


def selected_round_snapshots(*, root: Path = DEFAULT_ROOT) -> list[dict]:
    """Choose one complete 13-match snapshot per round, never per-match mixes."""
    root = Path(root)
    snapshots = {}
    for f in read_log(root / FORECAST_LOG):
        snapshots.setdefault((f["round_id"], f["snapshot_id"]), []).append(f)
    deadlines = {}
    for d in read_log(root / DEADLINE_LOG):
        previous = deadlines.get(d["round_id"])
        if previous and previous["purchase_deadline_at"] != d["purchase_deadline_at"]:
            raise ValueError(f"conflicting purchase deadlines: {d['round_id']}")
        deadlines[d["round_id"]] = d
    by_round = {}
    for (round_id, snapshot_id), rows in snapshots.items():
        if len(rows) != 13 or sorted(r["match_no"] for r in rows) != list(range(1, 14)):
            continue
        created = pd.Timestamp(rows[0]["forecast_created_at"])
        deadline = deadlines.get(round_id)
        eligibility = ("unknown" if deadline is None else
                       "before_deadline" if created < pd.Timestamp(deadline["purchase_deadline_at"]) else
                       "after_deadline")
        candidate = dict(round_id=round_id, snapshot_id=snapshot_id, forecast_created_at=created.isoformat(),
                         purchase_deadline_at=deadline["purchase_deadline_at"] if deadline else None,
                         purchase_eligibility=eligibility, records=rows)
        current = by_round.get(round_id)
        if current is None or (candidate["forecast_created_at"], snapshot_id) < (current["forecast_created_at"], current["snapshot_id"]):
            by_round[round_id] = candidate
    return [by_round[key] for key in sorted(by_round)]


def evaluation(*, root: Path = DEFAULT_ROOT) -> list[dict]:
    """Evaluate one earliest complete snapshot per round; flag unknown purchase eligibility."""
    root = Path(root)
    selected = selected_round_snapshots(root=root)
    latest = {}
    for r in read_log(root / RESULT_LOG):
        latest[r["condition_s_event_id"]] = r
    groups = {}
    for snapshot in selected:
        for f in snapshot["records"]:
            r = latest.get(f["event_id"])
            if r is None:
                continue
            key = (f.get("model_version"), f.get("model_signature"), f.get("code_version"),
                   f.get("condition_version"), snapshot["purchase_eligibility"], bool(f["condition_S"]))
            g = groups.setdefault(key, dict(n=0, actual_D=0, expected_D=0.0, brier_D_sum=0.0, log_loss_HDA_sum=0.0))
            p = float(f["final_prob_draw"])
            y = int(r["actual_D"])
            g["n"] += 1
            g["actual_D"] += y
            g["expected_D"] += p
            g["brier_D_sum"] += (p - y) ** 2
            chosen = {"H": "final_prob_home", "D": "final_prob_draw", "A": "final_prob_away"}[r["actual_symbol"]]
            g["log_loss_HDA_sum"] -= math.log(max(float(f[chosen]), 1e-15))
    return [dict(model_version=k[0], model_signature=k[1], code_version=k[2], condition_version=k[3],
                 purchase_eligibility=k[4], condition_S=k[5], n=g["n"], actual_D=g["actual_D"], expected_D=g["expected_D"],
                 brier_D=g["brier_D_sum"] / g["n"], log_loss_HDA=g["log_loss_HDA_sum"] / g["n"])
            for k, g in sorted(groups.items(), key=lambda item: str(item[0]))]


def status(*, root: Path = DEFAULT_ROOT) -> dict:
    root = Path(root)
    forecasts = read_log(root / FORECAST_LOG)
    results = read_log(root / RESULT_LOG)
    by_model = {}
    for record in forecasts:
        key = (record.get("model_version"), record.get("model_signature"), record.get("code_version"),
               record.get("condition_version"))
        item = by_model.setdefault(str(key), dict(matches=0, S=0, mix_min=1.0, mix_max=0.0))
        item["matches"] += 1
        item["S"] += int(record["condition_S"])
        item["mix_min"] = min(item["mix_min"], float(record["uncertainty_mix"]))
        item["mix_max"] = max(item["mix_max"], float(record["uncertainty_mix"]))
    return dict(forecasts=len(forecasts), linked_result_events=len(results),
                failure_events=len(read_log(root / FAILURE_LOG)), groups=by_model,
                selected_snapshots=[{k: v for k, v in s.items() if k != "records"} for s in selected_round_snapshots(root=root)],
                evaluation=evaluation(root=root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--forecast", type=Path, required=True)
    cap.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("sync-results")
    sub.add_parser("status")
    failure = sub.add_parser("record-failure")
    failure.add_argument("--stage", required=True)
    failure.add_argument("--reason", required=True)
    failure.add_argument("--round", default="")
    failure.add_argument("--source", default="")
    deadline = sub.add_parser("record-deadline")
    deadline.add_argument("--round", required=True)
    deadline.add_argument("--at", required=True)
    deadline.add_argument("--source", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        result = capture(args.forecast, args.manifest, root=args.root)
    elif args.command == "sync-results":
        result = sync_results(root=args.root)
    elif args.command == "record-failure":
        result = record_failure(args.stage, args.reason, round_id=args.round, source=args.source, root=args.root)
    elif args.command == "record-deadline":
        result = record_deadline(args.round, args.at, args.source, root=args.root)
    else:
        result = status(root=args.root)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
