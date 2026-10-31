#!/usr/bin/env python3
"""Read-only 13-hit-first Anchor/free-count portfolio simulation.

The Anchor is a reference combination. Candidates are generated from the
unchanged prematch H/D/A probabilities. A deterministic expected-goals
ensemble tests how fragile a ten-ticket portfolio is to lambda error.
"""
from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from build_next_buyplan import top_joint_candidates
from common import ROOT, sha256
from expected_goals import score_distribution


ROUND = "toto1665"
SOURCE = ROOT / "data/purchase_reference"
OUT = ROOT / "data/reports/toto1665_anchor_free_13"
TOP_CANDIDATES = 3000
SAMPLED_CANDIDATES = 5000
MODEL_SCENARIOS = 128
LAMBDA_LOG_SD = 0.15  # provisional, not learned from results
SEED = 16650013
INDEX = {"H": 0, "D": 1, "A": 2}
TO_STORED = np.array([1, 0, 2], dtype=np.uint8)


def candidate_pool(frame: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    pool: dict[tuple[int, ...], None] = {}
    for item in top_joint_candidates(frame, limit=TOP_CANDIDATES):
        pool.setdefault(tuple(INDEX[s] for s in item["symbols"]), None)
    rng = np.random.default_rng(SEED + 1)
    uniform = rng.random((SAMPLED_CANDIDATES, 13))
    sampled = (uniform[:, :, None] > probabilities.cumsum(axis=1)[None, :, :]).sum(axis=2)
    for ticket in sampled:
        pool.setdefault(tuple(int(v) for v in ticket), None)
    return np.asarray(list(pool), dtype=np.uint8)


def scenario_probabilities(frame: pd.DataFrame) -> np.ndarray:
    home = frame["final_lambda_home"].to_numpy(dtype=float)
    away = frame["final_lambda_away"].to_numpy(dtype=float)
    mix = frame["next_uncertainty_calibration_mix"].to_numpy(dtype=float)
    if not np.isfinite(home).all() or not np.isfinite(away).all() or (home <= 0).any() or (away <= 0).any():
        raise ValueError("invalid final expected goals")
    if not np.isfinite(mix).all() or ((mix < 0) | (mix > 1)).any():
        raise ValueError("invalid uncertainty mix")
    rng = np.random.default_rng(SEED + 2)
    shocks = rng.normal(0.0, LAMBDA_LOG_SD, (MODEL_SCENARIOS, 13, 2))
    lambdas = np.stack([home, away], axis=-1)[None, :, :] * np.exp(shocks - LAMBDA_LOG_SD**2 / 2)
    probabilities = np.empty((MODEL_SCENARIOS, 13, 3), dtype=float)
    for scenario in range(MODEL_SCENARIOS):
        for match in range(13):
            result = score_distribution(*lambdas[scenario, match])
            raw = np.array([result[f"prob_{side}"] for side in ("home", "draw", "away")])
            probabilities[scenario, match] = (1 - mix[match]) * raw + mix[match] / 3
    return probabilities


def masses(tickets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    values = np.ones((len(probabilities), len(tickets)), dtype=float)
    for match in range(13):
        values *= probabilities[:, match, :][:, tickets[:, match]]
    return values


def select_tickets(candidates: np.ndarray, scenario_mass: np.ndarray) -> list[int]:
    selected: list[int] = []
    total = np.zeros(len(scenario_mass), dtype=float)
    for _ in range(10):
        trial = total[:, None] + scenario_mass
        # Every term is a probability of 13 hits; no 10/11/12-hit score.
        score = 0.5 * trial.mean(axis=0) + 0.5 * np.quantile(trial, 0.10, axis=0)
        score[selected] = -np.inf
        if selected:
            too_close = (candidates[:, None, :] != candidates[np.asarray(selected)][None, :, :]).sum(axis=2).min(axis=1) < 2
            score[too_close] = -np.inf
        chosen = int(np.argmax(score))
        if not np.isfinite(score[chosen]):
            raise ValueError("candidate pool cannot produce ten distinct tickets")
        selected.append(chosen)
        total += scenario_mass[:, chosen]
    return selected


def main() -> None:
    forecast_path = SOURCE / "next_predictions.csv"
    official_path = SOURCE / "buyplan.csv"
    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("round_id") != ROUND or manifest.get("official_version") != "v2.3":
        raise ValueError("expected current official v2.3 toto1665 input")
    if manifest.get("output_sha256", {}).get("next_predictions.csv") != sha256(forecast_path):
        raise ValueError("forecast changed after BuyPlan generation")
    frame = pd.read_csv(forecast_path, encoding="utf-8-sig", low_memory=False).sort_values("match_no").reset_index(drop=True)
    if len(frame) != 13 or frame.match_no.astype(int).tolist() != list(range(1, 14)):
        raise ValueError("forecast must contain toto match numbers 1..13")
    p = frame[[f"next_prob_{s}" for s in ("home", "draw", "away")]].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("invalid official next probabilities")
    candidates = candidate_pool(frame, p)
    scenarios = scenario_probabilities(frame)
    selected = select_tickets(candidates, masses(candidates, scenarios))
    tickets = candidates[selected]
    baseline = masses(tickets, p[None, :, :])[0]
    order = sorted(range(10), key=lambda i: (-baseline[i], i))
    tickets, baseline = tickets[order], baseline[order]
    selected_mass = masses(tickets, scenarios).sum(axis=1)
    official = pd.read_csv(official_path, encoding="utf-8-sig").sort_values("match_no")
    stored = official[[f"ticket{i:02d}" for i in range(1, 11)]].to_numpy(dtype=np.uint8).T
    official_tickets = TO_STORED[stored]
    official_mass = masses(official_tickets, scenarios).sum(axis=1)
    anchor = p.argmax(axis=1).astype(np.uint8)
    result = frame[["match_no", "home_team", "away_team", "next_prob_home", "next_prob_draw", "next_prob_away"]].copy()
    for number, ticket in enumerate(tickets, 1):
        result[f"candidate{number:02d}"] = TO_STORED[ticket]
    table = []
    for number, ticket in enumerate(tickets, 1):
        table.append({"候補": f"{number:02d}", "13的中モデル確率": f"{baseline[number-1]*100:.6f}%",
                      "中心からの変更試合数": int((ticket != anchor).sum()), "D本数": int((ticket == 1).sum())})
    summary = {
        "round_id": ROUND, "status": "simulation only; not a purchase instruction",
        "forecast_sha256": sha256(forecast_path), "official_sha256": sha256(official_path),
        "method": "unchanged next_prob_*; top joint and model-sampled candidates; greedy 13-hit mean and lower-decile probability under expected-goal uncertainty; minimum pairwise Hamming distance 2",
        "top_candidates": TOP_CANDIDATES, "sampled_candidates": SAMPLED_CANDIDATES,
        "unique_candidates": len(candidates), "model_scenarios": MODEL_SCENARIOS,
        "lambda_log_sd_provisional": LAMBDA_LOG_SD, "seed": SEED,
        "anchor_stored_symbols": TO_STORED[anchor].tolist(),
        "model_probability_no_draw_in_round": float(np.prod(1.0 - p[:, 1])),
        "simulation": {"baseline_prob_any_13": float(baseline.sum()),
                       "ensemble_mean_prob_any_13": float(selected_mass.mean()),
                       "ensemble_p10_prob_any_13": float(np.quantile(selected_mass, 0.10)),
                       "draws_per_ticket": (tickets == 1).sum(axis=1).tolist(),
                       "tickets_with_any_draw": int(((tickets == 1).sum(axis=1) > 0).sum())},
        "official_v23": {"baseline_prob_any_13": float(masses(official_tickets, p[None, :, :]).sum()),
                          "ensemble_mean_prob_any_13": float(official_mass.mean()),
                          "ensemble_p10_prob_any_13": float(np.quantile(official_mass, 0.10))},
    }
    if OUT.exists():
        raise FileExistsError(f"simulation output already exists: {OUT}")
    OUT.mkdir(parents=True)
    shutil.copy2(forecast_path, OUT / "forecast_input.csv")
    shutil.copy2(official_path, OUT / "official_buyplan_input.csv")
    result.to_csv(OUT / "anchor_free_13.csv", index=False, encoding="utf-8-sig")
    (OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><title>{ROUND} 中心＋採用数自由・13的中重視</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5df;padding:6px;text-align:center}}th{{background:#edf2f7}}.note{{background:#fff4db;padding:12px;border-left:5px solid #d99000}}</style>
<h1>{ROUND} 中心＋採用数自由・13的中重視（シミュレーション）</h1>
<p class="note">購入対象外。正式v2.3の試合前確率を変更していません。実結果は使用していません。中心は各試合の最有力目をつないだ参照券で、10口に必ず入れる条件ではありません。各試合のH/D/A採用数やD本数は指定していません。得点期待値の誤差幅15%は未学習の仮設定です。</p>
<p class="note"><b>重要:</b> この試験案は{summary['simulation']['tickets_with_any_draw']}口にDを含みます。元の予測でDなし回となる確率は{summary['model_probability_no_draw_in_round']*100:.2f}%です。Dが1試合でも出た場合、Dなし券は13的中できません。13的中モデル確率が高くても、この集中を採用の根拠にしないでください。</p>
<h2>13的中モデル確率の比較</h2><table><tr><th>方式</th><th>元の確率</th><th>得点期待値を揺らした128条件の平均</th><th>下位10%点</th></tr>
<tr><td>正式v2.3</td><td>{summary['official_v23']['baseline_prob_any_13']*100:.6f}%</td><td>{summary['official_v23']['ensemble_mean_prob_any_13']*100:.6f}%</td><td>{summary['official_v23']['ensemble_p10_prob_any_13']*100:.6f}%</td></tr>
<tr><td>試験案</td><td>{summary['simulation']['baseline_prob_any_13']*100:.6f}%</td><td>{summary['simulation']['ensemble_mean_prob_any_13']*100:.6f}%</td><td>{summary['simulation']['ensemble_p10_prob_any_13']*100:.6f}%</td></tr></table>
<p>上記はモデルに基づく計算値であり、実際の当選率を確認した結果ではありません。選券には13的中確率だけを使用し、11・12的中は評価にも使っていません。</p>
<h2>各口</h2>{pd.DataFrame(table).to_html(index=False, escape=True)}
<h2>13試合の組み合わせ（1=H、0=D、2=A）</h2>{result.to_html(index=False, escape=True)}
<p><a href="anchor_free_13.csv">CSV</a> · <a href="comparison.json">計算条件と元ファイルのハッシュ</a></p></html>"""
    (OUT / "anchor_free_13.html").write_text(html, encoding="utf-8")
    print(OUT / "anchor_free_13.html")


if __name__ == "__main__":
    main()
