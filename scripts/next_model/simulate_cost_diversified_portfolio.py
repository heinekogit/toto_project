#!/usr/bin/env python3
"""Exploratory, read-only ticket selection with cost and exposure diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_next_buyplan import top_joint_candidates
from common import ROOT, sha256


ROUND = "toto1665"
SOURCE = ROOT / "data/purchase_reference"
OUT = ROOT / "data/reports/toto1665_cost_diversified_portfolio"
SEED = 16650033
SAMPLES = 30000
TOP = 3000
CAP = 30  # illustrative spending ceiling: 3,000 yen
PRICE = 100
PAYOUT_SCENARIOS = (10_000_000, 30_000_000, 100_000_000)
BETAS = (0.0, 1.6, 2.0)
TO_STORED = np.array([1, 0, 2], dtype=np.uint8)


def pool(frame: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    names = {"H": 0, "D": 1, "A": 2}
    rows = [tuple(names[s] for s in item["symbols"])
            for item in top_joint_candidates(frame, limit=TOP)]
    rng = np.random.default_rng(SEED)
    sampled = (rng.random((SAMPLES, 13, 1)) > p.cumsum(axis=1)).sum(axis=2)
    rows.extend(map(tuple, sampled.tolist()))
    return np.asarray(list(dict.fromkeys(rows)), dtype=np.uint8)


def ticket_masses(tickets: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.prod(p[np.arange(13)[None, :], tickets], axis=1)


def exposure_tv(exposure: np.ndarray, total: float, p: np.ndarray) -> float:
    if total <= 0:
        return 0.0
    return float(0.5 * np.abs(exposure / total - p).sum() / 13)


def max_ticket_mass_by_draw_count(p: np.ndarray) -> np.ndarray:
    """Exact dynamic program over 13 matches, independent of candidate sampling."""
    best = np.full(14, -np.inf)
    best[0] = 0.0
    for row in p:
        next_best = best + np.log(max(row[0], row[2]))
        next_best[1:] = np.maximum(next_best[1:], best[:-1] + np.log(row[1]))
        best = next_best
    return np.exp(best)


def select(tickets: np.ndarray, masses: np.ndarray, p: np.ndarray, payout: int,
           beta: float) -> tuple[list[int], str, list[dict]]:
    eligible = masses * payout >= PRICE
    exposure = np.zeros((13, 3), dtype=float)
    total = 0.0
    selected = []
    progress = []
    onehot = np.eye(3)[tickets]
    for _ in range(CAP):
        options = np.flatnonzero(eligible)
        if not len(options):
            break
        q = masses[options]
        if beta:
            new_total = total + q
            new_exposure = exposure[None, :, :] + q[:, None, None] * onehot[options]
            tv = 0.5 * np.abs(new_exposure / new_total[:, None, None] - p).sum(axis=(1, 2)) / 13
            score = new_total * (1 - beta * tv)
        else:
            score = q
        chosen = int(options[int(np.argmax(score))])
        eligible[chosen] = False
        selected.append(chosen)
        total += float(masses[chosen])
        exposure[np.arange(13), tickets[chosen]] += masses[chosen]
        progress.append({"口数": len(selected), "券の13的中確率": float(masses[chosen]),
                         "累積13的中確率": total, "費用": PRICE * len(selected),
                         "仮定払戻での13的中期待額": payout * total,
                         "HDA集中度_TV": exposure_tv(exposure, total, p),
                         "券のD試合数": int((tickets[chosen] == 1).sum())})
    reason = "費用基準で停止" if len(selected) < CAP else "仮の費用上限で停止（費用基準では未停止）"
    return selected, reason, progress


def main() -> None:
    forecast_path = SOURCE / "next_predictions.csv"
    official_path = SOURCE / "buyplan.csv"
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("round_id") != ROUND or manifest.get("official_version") != "v2.3":
        raise ValueError("expected toto1665 official v2.3")
    if manifest.get("output_sha256", {}).get("next_predictions.csv") != sha256(forecast_path):
        raise ValueError("forecast changed after BuyPlan generation")
    frame = pd.read_csv(forecast_path, encoding="utf-8-sig", low_memory=False).sort_values("match_no")
    official = pd.read_csv(official_path, encoding="utf-8-sig").sort_values("match_no")
    if frame.match_no.astype(int).tolist() != list(range(1, 14)) or official.match_no.astype(int).tolist() != list(range(1, 14)):
        raise ValueError("expected 13 ordered matches")
    p = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("invalid model probabilities")
    candidates = pool(frame, p)
    masses = ticket_masses(candidates, p)
    official_tickets = TO_STORED[official[[f"ticket{i:02d}" for i in range(1, 11)]].to_numpy(dtype=int).T]
    official_mass = float(ticket_masses(official_tickets, p).sum())
    results = []
    detailed = {}
    for payout in PAYOUT_SCENARIOS:
        for beta in BETAS:
            picked, reason, progress = select(candidates, masses, p, payout, beta)
            chosen = candidates[picked]
            d_counts = (chosen == 1).sum(axis=1) if len(chosen) else np.array([])
            summary = {"想定1等払戻額": payout, "集中抑制係数": beta,
                       "券数": len(picked), "費用": len(picked) * PRICE,
                       "13的中確率": float(masses[picked].sum()),
                       "D入り券": int((d_counts > 0).sum()),
                       "Dが4試合以上の券": int((d_counts >= 4).sum()),
                       "D試合数_最小": int(d_counts.min()) if len(d_counts) else None,
                       "D試合数_最大": int(d_counts.max()) if len(d_counts) else None,
                       "HDA集中度_TV": progress[-1]["HDA集中度_TV"] if progress else None,
                       "停止理由": reason}
            results.append(summary)
            detailed[f"payout_{payout}_beta_{beta}"] = {"summary": summary,
                                                         "ticket_indices": picked,
                                                         "progress": progress}
    primary_key = "payout_30000000_beta_2.0"
    primary = detailed[primary_key]
    primary_tickets = candidates[primary["ticket_indices"]]
    matrix = frame[["match_no", "home_team", "away_team", "next_prob_home", "next_prob_draw", "next_prob_away"]].copy()
    for number, ticket in enumerate(primary_tickets, 1):
        matrix[f"ticket{number:02d}"] = TO_STORED[ticket]
    totals = (primary_tickets[:, :, None] == np.arange(3)).sum(axis=0)
    marginals = pd.DataFrame({"試合": frame.match_no.astype(int).tolist(),
                              "予測H": p[:, 0], "予測D": p[:, 1], "予測A": p[:, 2],
                              "採用H": totals[:, 0], "採用D": totals[:, 1], "採用A": totals[:, 2]})
    table = []
    for row in results:
        table.append({"仮の1等払戻": f"{row['想定1等払戻額']:,}円",
                      "配券": "13的中確率のみ" if row["集中抑制係数"] == 0 else "H/D/A集中を抑制",
                      "口数": row["券数"], "費用": f"{row['費用']:,}円",
                      "モデル上の13的中確率": f"{row['13的中確率']*100:.5f}%",
                      "D入り券": row["D入り券"],
                      "D4以上の券": row["Dが4試合以上の券"],
                      "HDA集中度": f"{row['HDA集中度_TV']:.3f}" if row["HDA集中度_TV"] is not None else "—",
                      "停止": row["停止理由"]})
    draw_count_distribution = np.array([1.0])
    for probability in p[:, 1]:
        draw_count_distribution = np.convolve(draw_count_distribution, [1 - probability, probability])
    draw_count_maxima = max_ticket_mass_by_draw_count(p)
    output = {"round_id": ROUND, "status": "simulation only; not purchase advice",
              "forecast_sha256": sha256(forecast_path), "official_sha256": sha256(official_path),
              "candidate_seed": SEED, "sampled_candidates": SAMPLES, "top_candidates": TOP,
              "unique_candidates": len(candidates), "ticket_price_yen": PRICE,
              "provisional_spend_cap_yen": PRICE * CAP,
              "payout_scenarios_are_hypothetical": list(PAYOUT_SCENARIOS),
              "model_probability_round_draw_count": draw_count_distribution.tolist(),
              "exact_max_ticket_probability_by_draw_count": draw_count_maxima.tolist(),
              "minimum_first_prize_payout_for_a_four_draw_ticket_to_break_even_yen": float(PRICE / draw_count_maxima[4]),
              "cost_stop": "stop when no remaining candidate has model_probability_13 * hypothetical_first_prize_payout >= 100 yen; 2nd/3rd prizes ignored",
              "risk_objective": "maximize M*(1-beta*TV), where M is sum of exact-ticket mass and TV is mean total-variation between the HDA distribution of covered exact outcomes and next_prob for each match; no fixed D quota",
              "official_v23": {"tickets": 10, "model_prob_any_13": official_mass},
              "results": results, "primary": primary_key, "detail": detailed}
    if OUT.exists():
        raise FileExistsError(f"simulation output already exists: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(primary["progress"]).to_csv(OUT / "ticket_progress.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(OUT / "ticket_matrix.csv", index=False, encoding="utf-8-sig")
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><title>{ROUND} 費用と偏りを含む配券試算</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#172033;max-width:1500px}}table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}th,td{{border:1px solid #ccd5df;padding:6px;text-align:center}}th{{background:#edf2f7}}.note{{background:#fff4db;padding:12px;border-left:5px solid #d99000}}.scroll{{overflow-x:auto}}</style>
<h1>{ROUND} 費用と結果の偏りを含む配券試算</h1>
<p class="note">購入対象外。正式v2.3の予測と配券は変更していません。予測確率から{len(candidates):,}件の候補を作り、券群の13的中確率とH/D/A結果範囲への集中を評価して採用しました。D券数の割当は固定していません。</p>
<p class="note">1口{PRICE}円。1等払戻額の1,000万・3,000万・1億円は仮定であり、実際の払戻見込みではありません。停止式は「次の券の13的中確率×仮の1等払戻額＜{PRICE}円」です。2・3等の払戻、税や価格変動、予測誤差は含みません。最大{CAP}口（{CAP*PRICE:,}円）は試算用の上限です。</p>
<h2>停止結果</h2>{pd.DataFrame(table).to_html(index=False, escape=True)}
<p>正式v2.3は10口、モデル上の13的中確率{official_mass*100:.5f}%です。集中度は0に近いほど、的中できる結果のH/D/A内訳が元の予測分布に近くなります。集中抑制係数1.6・2.0は未学習の仮設定です。高払戻の仮定では費用基準が働く前に試算上限に達します。</p>
<p class="note"><b>結果範囲の欠落：</b>今回の予測から計算するとDが4試合以上になる確率は{draw_count_distribution[4:].sum()*100:.2f}%です。表の「D4以上の券」が0なら、その範囲で13的中する確率は0です。費用条件と確率分布だけでは、この欠落を防げません。</p>
<p>全組み合わせを対象とする厳密計算では、Dが4試合の券で最も高い13的中確率は{draw_count_maxima[4]*100:.6f}%です。1等だけの期待額が100円に達するには、仮の1等払戻額が少なくとも{PRICE/draw_count_maxima[4]:,.0f}円必要です。したがって3,000万円の条件では、集中抑制を強めてもD4以上の券は費用条件を通過しません。</p>
<h2>主表示：仮払戻3,000万円・集中抑制係数2.0</h2>
<p>必要払戻額は券ごとに {PRICE}円÷その券の13的中確率です。選択済み券の確率は重複しないため足し算しています。以下の選択順は費用上限までで、実際の購入額を勧めるものではありません。</p>
<div class="scroll">{pd.DataFrame(primary['progress']).to_html(index=False, escape=True)}</div>
<h2>主表示：各試合の採用数</h2>{marginals.to_html(index=False, escape=True)}
<h2>主表示：組み合わせ（1=H、0=D、2=A）</h2><div class="scroll">{matrix.to_html(index=False, escape=True)}</div>
<p><a href="ticket_progress.csv">選択順CSV</a> · <a href="ticket_matrix.csv">券面CSV</a> · <a href="comparison.json">全条件と数値</a></p></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
