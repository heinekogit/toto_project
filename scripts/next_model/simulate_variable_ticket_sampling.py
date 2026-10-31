#!/usr/bin/env python3
"""Read-only simulation of probability-sampled tickets with variable count.

The probability vector for each match is the common trunk. Tickets are
independent draws from those vectors; only duplicate complete tickets are
discarded. Portfolio size is determined by a stated coverage target, not by
an advance ticket count. The targets are examples, not purchase advice.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, sha256


ROUND = "toto1665"
SOURCE = ROOT / "data/purchase_reference"
OUT = ROOT / "data/reports/toto1665_variable_ticket_sampling"
SEED = 16650032
RUNS = 256
CAP = 100
TARGET_MULTIPLIERS = (1, 2, 3)
SYMBOLS = np.array(["H", "D", "A"])
TO_STORED = np.array([1, 0, 2], dtype=np.uint8)


def ticket_mass(ticket: np.ndarray, p: np.ndarray) -> float:
    return float(np.prod(p[np.arange(13), ticket]))


def simulate(p: np.ndarray, thresholds: list[float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw unique tickets until all targets are met or the safety cap applies."""
    tickets: list[np.ndarray] = []
    masses: list[float] = []
    seen: set[tuple[int, ...]] = set()
    cdf = p.cumsum(axis=1)
    reached = np.full(len(thresholds), -1, dtype=int)
    total = 0.0
    attempts = 0
    while len(tickets) < CAP and (reached < 0).any():
        attempts += 1
        if attempts > CAP * 1000:
            raise RuntimeError("unique ticket sampling stalled")
        ticket = (rng.random(13)[:, None] > cdf).sum(axis=1).astype(np.uint8)
        key = tuple(int(x) for x in ticket)
        if key in seen:
            continue
        seen.add(key)
        mass = ticket_mass(ticket, p)
        tickets.append(ticket)
        masses.append(mass)
        total += mass
        for index, threshold in enumerate(thresholds):
            if reached[index] < 0 and total >= threshold:
                reached[index] = len(tickets)
    return np.asarray(tickets, dtype=np.uint8), np.asarray(masses), reached


def main() -> None:
    manifest_path = SOURCE / "manifest.json"
    forecast_path = SOURCE / "next_predictions.csv"
    official_path = SOURCE / "buyplan.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("round_id") != ROUND or manifest.get("official_version") != "v2.3":
        raise ValueError("expected toto1665 official v2.3")
    if manifest.get("output_sha256", {}).get("next_predictions.csv") != sha256(forecast_path):
        raise ValueError("forecast changed after BuyPlan generation")
    frame = pd.read_csv(forecast_path, encoding="utf-8-sig", low_memory=False).sort_values("match_no")
    official = pd.read_csv(official_path, encoding="utf-8-sig").sort_values("match_no")
    if frame.match_no.astype(int).tolist() != list(range(1, 14)) or official.match_no.astype(int).tolist() != list(range(1, 14)):
        raise ValueError("expected match numbers 1..13")
    p = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("invalid next_prob probabilities")
    official_tickets = TO_STORED[official[[f"ticket{i:02d}" for i in range(1, 11)]].to_numpy(dtype=int).T]
    if len({tuple(row) for row in official_tickets}) != 10:
        raise ValueError("official tickets contain duplicates")
    official_mass = sum(ticket_mass(row, p) for row in official_tickets)
    thresholds = [official_mass * scale for scale in TARGET_MULTIPLIERS]

    runs = []
    example_tickets = None
    example_masses = None
    example_reached = None
    for run in range(RUNS):
        tickets, masses, reached = simulate(p, thresholds, np.random.default_rng(SEED + run))
        runs.append(reached)
        if run == 0:
            example_tickets, example_masses, example_reached = tickets, masses, reached
    runs = np.asarray(runs)
    assert example_tickets is not None and example_masses is not None and example_reached is not None

    target_rows = []
    for index, scale in enumerate(TARGET_MULTIPLIERS):
        valid = runs[:, index] >= 0
        n = int(example_reached[index])
        subset = example_tickets[:n] if n >= 0 else example_tickets
        target_rows.append({
            "目標": f"正式v2.3の{scale}倍", "目標13的中確率": f"{thresholds[index]*100:.5f}%",
            "例の必要口数": n if n >= 0 else f"{CAP}口で未達",
            "例の到達確率": f"{example_masses[:n].sum()*100:.5f}%" if n >= 0 else f"{example_masses.sum()*100:.5f}%",
            "例のD入り券": int((subset == 1).any(axis=1).sum()),
            "256回の到達率": f"{valid.mean()*100:.1f}%",
            "到達時口数の中央値": f"{np.median(runs[valid, index]):.0f}" if valid.any() else "—",
            "到達時口数の10–90%": (
                f"{np.quantile(runs[valid, index], .1):.0f}–{np.quantile(runs[valid, index], .9):.0f}"
                if valid.any() else "—"),
        })
    main_n = int(example_reached[0]) if example_reached[0] >= 0 else len(example_tickets)
    shown = example_tickets[:main_n]
    counts = np.stack([(shown == symbol).sum(axis=0) for symbol in range(3)], axis=1)
    match_rows = []
    for index, row in enumerate(frame.itertuples()):
        match_rows.append({"試合": int(row.match_no), "ホーム": row.home_team, "アウェー": row.away_team,
                           "予測H/D/A": "/".join(f"{v:.1%}" for v in p[index]),
                           "実現H/D/A": "/".join(str(int(v)) for v in counts[index])})
    detail = frame[["match_no", "home_team", "away_team"]].copy()
    for index, ticket in enumerate(example_tickets, 1):
        detail[f"ticket{index:03d}"] = TO_STORED[ticket]
    cumulative = np.cumsum(example_masses)
    ticket_rows = [{"口": i + 1, "13的中確率": f"{mass*100:.6f}%",
                    "累積13的中確率": f"{cumulative[i]*100:.6f}%",
                    "D試合数": int((example_tickets[i] == 1).sum()),
                    "組み合わせ": "".join(SYMBOLS[example_tickets[i]]),
                    "目標": ", ".join(f"{s}倍到達" for j, s in enumerate(TARGET_MULTIPLIERS) if example_reached[j] == i + 1)}
                   for i, mass in enumerate(example_masses)]
    summary = {
        "round_id": ROUND, "status": "simulation only; not purchase advice",
        "forecast_sha256": sha256(forecast_path), "official_sha256": sha256(official_path),
        "seed": SEED, "runs": RUNS, "safety_cap_tickets": CAP,
        "sampling": "independent per-match categorical draw from unchanged next_prob_HDA; discard duplicate whole tickets",
        "stopping": "cumulative model probability of any exact 13-hit ticket reaches 1x, 2x or 3x official v2.3; targets are illustrative",
        "official_v23_tickets": 10, "official_v23_model_prob_any_13": official_mass,
        "targets": target_rows, "example_reached": example_reached.tolist(),
        "model_probability_no_draw_in_round": float(np.prod(1 - p[:, 1])),
    }
    if OUT.exists():
        raise FileExistsError(f"simulation output already exists: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(ticket_rows).to_csv(OUT / "ticket_progress.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "ticket_matrix.csv", index=False, encoding="utf-8-sig")
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><title>{ROUND} 確率幹・可変口数シミュレーション</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1450px;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}th,td{{border:1px solid #ccd5df;padding:6px;text-align:center}}th{{background:#edf2f7}}.note{{background:#fff4db;padding:12px;border-left:5px solid #d99000}}.scroll{{overflow-x:auto}}.num{{font-variant-numeric:tabular-nums}}</style>
<h1>{ROUND} 確率幹からの可変口数シミュレーション</h1>
<p class="note">比較用。正式版の予測・配券は変更していません。幹は特定の券ではなく、各試合のH/D/A予測確率です。5/3/2は10口なら50/30/20%を表す例であり、実際の比率は試合ごとの確率を使います。各口を乱数で引き、同一券は除きます。</p>
<p class="note">口数は事前に固定していません。ただし確率だけでは停止点が決まらないため、正式v2.3のモデル上の13的中確率の1・2・3倍を仮の到達目標とし、最大{CAP}口の安全上限を置きました。この目標は収益性や購入推奨を意味しません。実結果は使用していません。</p>
<h2>到達確率と必要口数</h2><p>正式v2.3は10口で{official_mass*100:.5f}%（保存済み next_prob_* で計算）。同じ組み合わせを二度買っても13的中確率は増えないので、重複は除外しています。</p>
{pd.DataFrame(target_rows).to_html(index=False, escape=True)}
<p>256回は乱数の種だけを変えた試算です。モデルが正しい場合の理論上の13的中確率を足し合わせています。実際に的中した割合ではありません。Dなし回のモデル確率は{summary['model_probability_no_draw_in_round']*100:.2f}%です。</p>
<h2>例：1倍の目標に到達した時点の各試合の配分</h2>
{pd.DataFrame(match_rows).to_html(index=False, escape=True)}
<h2>例：券を加えた順と確率の増分</h2><div class="scroll">{pd.DataFrame(ticket_rows).to_html(index=False, escape=True)}</div>
<h2>全券の組み合わせ（1=H、0=D、2=A）</h2><div class="scroll">{detail.to_html(index=False, escape=True)}</div>
<p><a href="ticket_progress.csv">券ごとのCSV</a> · <a href="ticket_matrix.csv">全券のCSV</a> · <a href="comparison.json">計算条件</a></p></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
