"""Hypothetical break-even audit for frozen condition S; no purchase output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_phase35_high_draw_structure import conditions
from analyze_score_structure_phase4b_stage1 import N, POWERS, ROUNDS, decode, top_ten, universe


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/reports/score_structure_phase35/diagnostics_78.csv"
OUT = ROOT / "data/reports/buyplan_v40_step4_break_even"
# These are named counterfactuals, not estimated effects or frequencies.
FACTORS = (1.1, 1.5, 2.0)
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def d_count_on_s(s_indices: np.ndarray) -> np.ndarray:
    codes = np.arange(N, dtype=np.int64)
    count = np.zeros(N, dtype=np.uint8)
    for i in s_indices:
        count += ((codes // POWERS[int(i)]) % 3 == 1).astype(np.uint8)
    return count


def first_d_break_even(base: np.ndarray, alternative: np.ndarray, baseline: np.ndarray,
                       all_d: np.ndarray) -> tuple[float | None, int | None, int | None]:
    """First weight where a D ticket displaces any original top-ten ticket."""
    outsiders = np.flatnonzero(all_d)
    # An incumbent with the lowest mixture mass is the first one displaced.
    thresholds = []
    for incumbent in baseline:
        loss = base[int(incumbent)] - base[outsiders]
        gain = (alternative[outsiders] - base[outsiders]) - (alternative[int(incumbent)] - base[int(incumbent)])
        valid = (loss > 0) & (gain > 0)
        if np.any(valid):
            w = np.divide(loss[valid], gain[valid])
            ix = int(np.argmin(w))
            thresholds.append((float(w[ix]), int(outsiders[valid][ix]), int(incumbent)))
    if not thresholds:
        return None, None, None
    w, entering, leaving = min(thresholds, key=lambda x: (x[0], x[1], x[2]))
    return (w, entering, leaving) if w <= 1 else (None, None, None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE, encoding="utf-8-sig").sort_values(["round", "match_no"])
    data["S"] = conditions(data)["inclusive"]
    codes = np.arange(N, dtype=np.int64)
    all_d = np.zeros(N, dtype=bool)
    for power in POWERS:
        all_d |= (codes // power) % 3 == 1
    summary, grid, tickets = [], [], []
    for rid in ROUNDS:
        group = data[data["round"].eq(rid)].sort_values("match_no")
        if len(group) != 13 or group.match_no.tolist() != list(range(1, 14)):
            raise ValueError(f"{rid}: 13試合が揃っていません")
        p = group[["baseline_P_H", "baseline_P_D", "baseline_P_A"]].to_numpy(float)
        if not np.allclose(p.sum(axis=1), 1, atol=1e-9):
            raise ValueError(f"{rid}: H/D/A確率の合計が1ではありません")
        s = np.flatnonzero(group.S.to_numpy())
        b = universe(p)
        baseline = top_ten(b)
        baseline_mass = float(b[baseline].sum())
        dcounts = d_count_on_s(s)
        for factor in FACTORS:
            # Multiply D odds only on S matches, then normalize each match.
            normalizer = float(np.prod(1 + (factor - 1) * p[s, 1]))
            q = b * np.power(factor, dcounts.astype(np.int16)) / normalizer
            if not np.isclose(q.sum(), 1, atol=1e-9):
                raise AssertionError("counterfactual distribution is not normalized")
            threshold, entering, leaving = first_d_break_even(b, q, baseline, all_d)
            if threshold is not None:
                before = top_ten((1 - max(0, threshold - 1e-6)) * b + max(0, threshold - 1e-6) * q)
                after = top_ten((1 - min(1, threshold + 1e-6)) * b + min(1, threshold + 1e-6) * q)
                if all_d[before].any() or not all_d[after].any() or entering not in set(after):
                    raise AssertionError(f"{rid}: first-D boundary verification failed")
            for weight in WEIGHTS:
                mixture = (1 - weight) * b + weight * q
                selected = top_ten(mixture)
                selected_mass = float(mixture[selected].sum())
                base_mass = float(b[selected].sum())
                scenario_mass = float(q[selected].sum())
                baseline_mixture = float(mixture[baseline].sum())
                if selected_mass + 1e-15 < baseline_mixture:
                    raise AssertionError("selected tickets underperform original tickets in same mixture")
                grid.append(dict(round=rid, S_matches=";".join(map(str, group.iloc[s].match_no)), S_count=len(s),
                                 odds_factor=factor, scenario_weight=weight, interpretation="仮説・未校正",
                                 D_tickets=int(all_d[selected].sum()), changed_tickets=10-len(set(selected)&set(baseline)),
                                 base_final_prob13=base_mass, base_loss=baseline_mass-base_mass,
                                 effect_scenario_prob13=scenario_mass,
                                 mixture_prob13=selected_mass, baseline_mixture_prob13=baseline_mixture,
                                 mixture_gain=selected_mass-baseline_mixture))
                for ticket_no, code in enumerate(selected, 1):
                    symbols="".join("HDA"[i] for i in decode([code])[0])
                    tickets.append(dict(round=rid, odds_factor=factor, scenario_weight=weight,
                                        ticket_no=ticket_no, symbols=symbols, base_probability=float(b[code]),
                                        effect_scenario_probability=float(q[code]), mixture_probability=float(mixture[code])))
            summary.append(dict(round=rid, S_matches=";".join(map(str, group.iloc[s].match_no)), S_count=len(s),
                                odds_factor=factor, first_D_weight=threshold,
                                entering_D_ticket="" if entering is None else "".join("HDA"[i] for i in decode([entering])[0]),
                                replaced_baseline_ticket="" if leaving is None else "".join("HDA"[i] for i in decode([leaving])[0]),
                                baseline_prob13=baseline_mass, baseline_D_tickets=int(all_d[baseline].sum()),
                                no_crossing_at_weight_1=threshold is None))
    summary_df = pd.DataFrame(summary)
    grid_df = pd.DataFrame(grid)
    summary_df.to_csv(OUT / "break_even_by_round.csv", index=False, encoding="utf-8-sig")
    grid_df.to_csv(OUT / "scenario_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tickets).to_csv(OUT / "tickets_by_scenario.csv", index=False, encoding="utf-8-sig")
    audit = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 condition_S="P(0-0)<=round median and P(1-1)>=round median; unchanged",
                 rounds=list(ROUNDS), factor_hypotheses=FACTORS, weight_hypotheses=WEIGHTS,
                 scenario="all S matches have the same hypothetical D-odds effect; independent match outcomes within each state",
                 actual_results_used=False, production_changed=False)
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact=summary_df.copy();compact["first_D_weight"]=(100*compact.first_D_weight).round(2)
    report = ("# v4.0 Step 4：条件Sの仮説効果と10口入替境界\n\n"
              "条件Sを固定した6開催の試合前保存確率のみを使用。実結果は計算に使わない。"
              "倍率1.1/1.5/2.0と効果状態の重み0〜1は**仮説**であり、条件Sの推定効果・成立頻度ではない。"
              "各開催でS該当試合すべてのDオッズを同時に倍率調整した状態Qと、保存final分布Pとの混合(1-w)P+wQを評価する。"
              "状態は同一開催内で共通に成立する仮定で、試合ごとに独立して発生する誤差モデルではない。\n\n"
              "比較は毎回3^13全組から異なる上位10口を厳密に選ぶ。"
              "`first_D_weight`は、初めてD入り券が基準10口の一券を追い越す混合重みの境界。"
              "空欄は重み1でもD入り券が入らないことを意味する。"
              "基準分布Pで失う質量(`base_loss`)と、混合分布で得る質量(`mixture_gain`)を別に保存。"
              "この境界は**設定した倍率が本当に発生する場合だけ**の損益分岐で、現実の購入推奨ではない。\n\n"
              "## 開催別境界（%）\n\n```text\n"+compact.to_string(index=False)+"\n```\n\n"
              "## 判断\n\n条件Sの6/8は発見標本での探索結果。倍率と状態重みをこの結果から推定・採用しない。"
              "将来の試合前保存予測でS該当時のP(D)校正と効果の分布を確認するまで、境界を実際の購入判断へ変換できない。"
              "固定Pが正しければ基準上位10口が最適で、D入り券はP上の機会損失を伴う。"
              "この研究は、どの程度の効果・頻度なら券変更が合理的かを先に示す。\n")
    (OUT / "report.md").write_text(report, encoding="utf-8")
    html = ('<!doctype html><html lang="ja"><meta charset="utf-8"><title>v4.0 Step 4 損益分岐</title>'
            '<style>body{font:16px/1.65 system-ui;max-width:1200px;margin:28px auto;padding:0 16px;color:#24364c}'
            'table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #d6deea;padding:7px}th{background:#e8f0f8}'
            '.scroll{overflow:auto}.note{background:#fff1d8;padding:14px;border-left:4px solid #c98b1f}</style>'
            '<h1>v4.0 Step 4：条件Sの仮説効果と10口の入替境界</h1>'
            '<p class="note">倍率・成立重みは仮定です。実績から推定した値でも、購入推奨でもありません。</p>'
            '<p>条件Sは固定。元のfinal確率Pと、S該当試合のDオッズが変わる仮説状態Qを混合し、全3^13組から10口を選び直します。'
            '「最初のD券が入る重み」は、その仮定下での損益分岐です。</p><h2>開催別の境界</h2><div class="scroll">'
            +compact.to_html(index=False,escape=True,na_rep="なし")+'</div><h2>仮説別の10口評価</h2><div class="scroll">'
            +grid_df.to_html(index=False,escape=True,float_format=lambda x:f"{x:.8g}")+'</div>'
            '<p><a href="report.md">説明</a> / <a href="break_even_by_round.csv">境界CSV</a> / '
            '<a href="scenario_grid.csv">評価CSV</a> / <a href="tickets_by_scenario.csv">券CSV</a> / '
            '<a href="audit.json">監査</a></p></html>')
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
