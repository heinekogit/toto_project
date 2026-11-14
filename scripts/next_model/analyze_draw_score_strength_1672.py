"""Research-only modal-score D strength from saved pre-match lambdas."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from analyze_score_structure import score_grid
from common import sha256


def analyse(source: Path, out: Path):
    f = pd.read_csv(source, encoding="utf-8-sig").sort_values("match_no").reset_index(drop=True)
    if len(f) != 13 or f.match_no.tolist() != list(range(1, 14)):
        raise ValueError("expected the thirteen saved matches")
    out.mkdir(parents=True, exist_ok=True)
    rows, grids = [], {}
    ii, jj = np.indices((11, 11))
    regions = {"H": ii > jj, "D": ii == jj, "A": ii < jj}

    for _, match in f.iterrows():
        lh, la = float(match.final_lambda_home), float(match.final_lambda_away)
        if min(lh, la) <= 0 or not np.isfinite([lh, la]).all():
            raise ValueError("invalid saved lambda")
        old_normalized, retained = score_grid(lh, la)
        raw = np.outer(poisson.pmf(np.arange(11), lh), poisson.pmf(np.arange(11), la))
        if not np.isclose(raw.sum(), retained, rtol=0, atol=1e-12) or not np.allclose(raw / retained, old_normalized, rtol=0, atol=1e-14):
            raise AssertionError("matrix differs from Phase 1 implementation")
        record = dict(match_no=int(match.match_no), home_team=match.home_team, away_team=match.away_team,
                      lambda_home=lh, lambda_away=la, retained_0_to_10=retained,
                      outside_0_to_10=1-retained,
                      final_P_H=float(match.next_prob_home), final_P_D=float(match.next_prob_draw),
                      final_P_A=float(match.next_prob_away),
                      first_choice="HDA"[int(np.argmax([match.next_prob_home, match.next_prob_draw, match.next_prob_away]))])
        for label, mask in regions.items():
            highest = float(raw[mask].max())
            ties = np.argwhere(mask & np.isclose(raw, highest, rtol=1e-12, atol=1e-15))
            record[f"S_{label}"] = highest
            record[f"modal_{label}_score"] = ";".join(f"{i}-{j}" for i,j in ties)
            record[f"modal_{label}_tie_count"] = len(ties)
        record["D_strength"] = record["S_D"] / max(record["S_H"], record["S_A"])
        record["D_vs_H"] = record["S_D"] / record["S_H"]
        record["D_vs_A"] = record["S_D"] / record["S_A"]
        record["D_margin"] = record["S_D"] - max(record["S_H"], record["S_A"])
        rows.append(record)
        grids[int(match.match_no)] = raw
    table = pd.DataFrame(rows).sort_values(["D_strength", "match_no"], ascending=[False, True]).reset_index(drop=True)
    table.insert(0, "D_strength_rank", range(1, 14))
    table.to_csv(out / "D_strength_13.csv", index=False, encoding="utf-8-sig")

    summary = dict(n=13, D_strength_above_one=int((table.D_strength > 1+1e-12).sum()),
                   D_strength_equal_one=int(np.isclose(table.D_strength, 1, atol=1e-12).sum()),
                   D_strength_min=float(table.D_strength.min()), D_strength_max=float(table.D_strength.max()),
                   D_strength_mean=float(table.D_strength.mean()), D_strength_median=float(table.D_strength.median()),
                   D_strength_q25=float(table.D_strength.quantile(.25)), D_strength_q75=float(table.D_strength.quantile(.75)),
                   modal_D_counts=table.modal_D_score.value_counts().to_dict(),
                   outside_mass_max=float(table.outside_0_to_10.max()), outside_mass_sum=float(table.outside_0_to_10.sum()),
                   source_sha256=sha256(source), probabilities_are_raw_0_to_10=True,
                   result_used=False, prediction_or_buyplan_changed=False)
    (out / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    highest = table.iloc[0]
    lowest = table.iloc[-1]
    representatives = [int(highest.match_no), int(lowest.match_no)]
    cells = []
    for no in representatives:
        grid = grids[no]
        match = f.loc[f.match_no.eq(no)].iloc[0]
        name = f"{match.home_team}－{match.away_team}"
        header = "<tr><th>H\\A</th>"+"".join(f"<th>{j}</th>" for j in range(11))+"</tr>"
        body = []
        for i in range(11):
            body.append("<tr><th>"+str(i)+"</th>"+"".join(f"<td class='{('draw' if i==j else 'home' if i>j else 'away')}'>{100*grid[i,j]:.3f}</td>" for j in range(11))+"</tr>")
        cells.append(f"<details><summary>{html.escape(name)}：D強度 {float(table.loc[table.match_no.eq(no),'D_strength'].iloc[0]):.3f}</summary><div class='scroll'><table>{header}{''.join(body)}</table></div></details>")
    shown = table[["D_strength_rank","match_no","home_team","away_team","lambda_home","lambda_away",
                   "modal_H_score","S_H","modal_D_score","S_D","modal_A_score","S_A",
                   "D_strength","D_vs_H","D_vs_A","D_margin","outside_0_to_10","final_P_H","final_P_D","final_P_A"]].copy()
    for col in ("S_H","S_D","S_A","D_margin","outside_0_to_10","final_P_H","final_P_D","final_P_A"):
        shown[col] = shown[col].map(lambda x: f"{100*x:.5f}%")
    for col in ("D_strength","D_vs_H","D_vs_A"):
        shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    same_11 = table[table.modal_D_score.eq("1-1")]
    report = ("# 第1672回 D最頻スコア強度\n\n"
        "保存λとPhase 1の11×11行列生成方式が一致することを確認。主表のS確率は0〜10点内のraw質量で、再正規化していない。"
        "D強度は最頻Dスコアと最頻H/Aスコアの比であり、P(D)ではない。\n\n"
        f"D強度>1：{summary['D_strength_above_one']}/13。最小 {summary['D_strength_min']:.4f}、最大 {summary['D_strength_max']:.4f}、平均 {summary['D_strength_mean']:.4f}。"
        f"1-1が最頻Dの試合：{len(same_11)}、そのD強度範囲：{same_11.D_strength.min():.4f}〜{same_11.D_strength.max():.4f}。"
        f"範囲外質量の最大：{100*summary['outside_mass_max']:.7f}%。\n\n"
        f"最高のC大阪－浦和は1-1が {100*highest.S_D:.3f}% 、非D最頻の1-2が {100*highest.S_A:.3f}% で強度 {highest.D_strength:.4f}。"
        f"最低の富山－徳島は1-1が {100*lowest.S_D:.3f}% 、非D最頻の1-0が {100*lowest.S_H:.3f}% で強度 {lowest.D_strength:.4f}。"
        f"一方、最終P(D)は前者 {100*highest.final_P_D:.3f}% 、後者 {100*lowest.final_P_D:.3f}% と逆順である。"
        "最頻スコアの比とD結果全体の確率は異なる量なので矛盾ではない。\n\n"
        "D強度は既存λから計算した形状指標であり、独立した入力情報ではない。"
        "この指標が同程度のP(D)の試合を追加的に識別できるか、実結果に対して予測力を持つかは未検証。"
        "購入券・予測確率は変更していない。\n")
    (out / "report.md").write_text(report, encoding="utf-8")
    document = ("<!doctype html><html lang='ja'><meta charset='utf-8'><title>第1672回 D強度</title>"
        "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:5px 8px;text-align:center}th{background:#e8eff7}.draw{background:#ffe299;font-weight:700}.home{background:#e9f3ff}.away{background:#f0eafd}.note{background:#fff0d3;padding:12px}details{padding:10px;border-bottom:1px solid #ddd}</style>"
        "<h1>第1672回 ポアソン得点分布のD強度</h1>"
        "<p class='note'>D強度＝最頻Dスコア確率÷最頻H/Aスコア確率。これはD発生確率ではなく、配券やP(D)の変更には使用していません。Sは0〜10点のraw確率で、範囲外の質量を別記します。</p>"
        f"<p>D強度&gt;1：{summary['D_strength_above_one']}/13。最小 {summary['D_strength_min']:.4f}、最大 {summary['D_strength_max']:.4f}、平均 {summary['D_strength_mean']:.4f}。最頻Dが1-1：{len(same_11)}試合、そのD強度範囲 {same_11.D_strength.min():.4f}〜{same_11.D_strength.max():.4f}。</p>"
        f"<p>最高の{html.escape(str(highest.home_team))}－{html.escape(str(highest.away_team))}はD強度 {highest.D_strength:.4f}、最低の{html.escape(str(lowest.home_team))}－{html.escape(str(lowest.away_team))}は {lowest.D_strength:.4f}。最終P(D)の大小とは一致しません。これはλ由来の形状指標で、追加の予測力はまだ未検証です。</p>"
        "<h2>13試合の比較</h2><div class='scroll'>"+shown.to_html(index=False, escape=True)+"</div>"
        "<h2>最高・最低D強度の得点行列（各セルのraw確率、%）</h2>"+"".join(cells)
        +"<p><a href='D_strength_13.csv'>全指標CSV</a>／<a href='report.md'>分析メモ</a>／<a href='audit.json'>再現性記録</a></p></html>")
    (out / "index.html").write_text(document, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyse(args.forecast, args.outdir), ensure_ascii=False))
