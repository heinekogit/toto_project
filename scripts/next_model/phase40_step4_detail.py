"""Translate hypothetical D-branch break-evens into probability lifts and ticket swaps."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_phase35_high_draw_structure import conditions
from analyze_score_structure_phase4b_stage1 import N, POWERS, ROUNDS, decode, top_ten, universe
from phase40_step4_break_even import FACTORS, SOURCE, first_d_break_even


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/reports/buyplan_v40_step4_break_even_detail"


def symbols(code: int) -> str:
    return "".join("HDA"[int(v)] for v in decode([code])[0])


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE, encoding="utf-8-sig").sort_values(["round", "match_no"])
    data["S"] = conditions(data)["inclusive"]
    codes = np.arange(N, dtype=np.int64)
    all_d = np.zeros(N, dtype=bool)
    for power in POWERS:
        all_d |= ((codes // power) % 3 == 1)
    boundaries, match_lifts, swap_rows, coverage_rows, comparison = [], [], [], [], []
    for rid in ROUNDS:
        g = data[data["round"].eq(rid)].sort_values("match_no").reset_index(drop=True)
        p = g[["baseline_P_H", "baseline_P_D", "baseline_P_A"]].to_numpy(float)
        base = universe(p)
        baseline = top_ten(base)
        s_all = np.flatnonzero(g.S.to_numpy())
        modes = [("同時", s_all)] + [(f"No.{i+1}のみ", np.array([i], dtype=int)) for i in s_all] if len(s_all)>1 else [("同時", s_all)]
        for factor in FACTORS:
            for mode, active in modes:
                q_match = p.copy()
                q_match[active, 1] *= factor
                q_match[active] /= q_match[active].sum(axis=1, keepdims=True)
                q = universe(q_match)
                if not np.isclose(q.sum(), 1, atol=1e-9):
                    raise AssertionError(f"{rid}: scenario probability mass")
                weight, entering, leaving = first_d_break_even(base, q, baseline, all_d)
                key = dict(round=rid, mode=mode, active_S=";".join(str(i+1) for i in active), odds_factor=factor)
                boundaries.append(dict(**key, threshold_weight=weight, first_D_ticket="" if entering is None else symbols(entering),
                                       removed_ticket="" if leaving is None else symbols(leaving),
                                       base_prob13=float(base[baseline].sum()), no_crossing=weight is None))
                if weight is None:
                    continue
                mixture_match = (1-weight)*p + weight*q_match
                for i in s_all:
                    match_lifts.append(dict(**key, match_no=int(i+1), card=f"{g.loc[i,'home_team']}－{g.loc[i,'away_team']}",
                                            active=bool(i in active), base_P_D=float(p[i,1]),
                                            effect_state_P_D=float(q_match[i,1]), break_even_P_D=float(mixture_match[i,1]),
                                            required_lift_pp=float(100*(mixture_match[i,1]-p[i,1]))))
                before=max(0.,weight-1e-6);after=min(1.,weight+1e-6)
                chosen_before=top_ten((1-before)*base+before*q)
                chosen_after=top_ten((1-after)*base+after*q)
                if all_d[chosen_before].any() or not all_d[chosen_after].any() or entering not in set(chosen_after):
                    raise AssertionError(f"{rid}: boundary is not first D insertion")
                added=sorted(set(chosen_after)-set(chosen_before))
                removed=sorted(set(chosen_before)-set(chosen_after))
                if len(added)!=1 or len(removed)!=1:
                    raise AssertionError(f"{rid}: expected single first ticket swap")
                entering,leaving=added[0],removed[0]
                e=decode([entering])[0];l=decode([leaving])[0]
                for i,(old,new) in enumerate(zip(l,e)):
                    if old!=new:
                        swap_rows.append(dict(**key, entering_ticket=symbols(entering), removed_ticket=symbols(leaving),
                                              match_no=i+1,card=f"{g.loc[i,'home_team']}－{g.loc[i,'away_team']}",
                                              from_symbol="HDA"[old],to_symbol="HDA"[new],
                                              S_match=bool(g.loc[i,"S"]),base_P_H=float(p[i,0]),base_P_D=float(p[i,1]),base_P_A=float(p[i,2])))
                before_matrix=decode(chosen_before);after_matrix=decode(chosen_after)
                for i in range(13):
                    prior={"HDA"[x] for x in before_matrix[:,i]}
                    later={"HDA"[x] for x in after_matrix[:,i]}
                    coverage_rows.append(dict(**key,match_no=i+1,card=f"{g.loc[i,'home_team']}－{g.loc[i,'away_team']}",
                                              before="".join(sorted(prior)),after="".join(sorted(later)),
                                              lost_HA="".join(sorted((prior-later)&{"H","A"})),
                                              gained_HA="".join(sorted((later-prior)&{"H","A"})),
                                              lost_D="D" in prior-later,gained_D="D" in later-prior))
                comparison.append(dict(**key,base_mass_before=float(base[chosen_before].sum()),
                                       base_mass_after=float(base[chosen_after].sum()),
                                       base_opportunity_loss=float(base[chosen_before].sum()-base[chosen_after].sum()),
                                       effect_state_gain=float(q[chosen_after].sum()-q[chosen_before].sum()),
                                       mixture_gain_just_above=float(((1-after)*base+after*q)[chosen_after].sum()-((1-after)*base+after*q)[chosen_before].sum())))
    frames={"break_even.csv":boundaries,"match_lifts.csv":match_lifts,"ticket_swaps.csv":swap_rows,
            "symbol_coverage.csv":coverage_rows,"opportunity_cost.csv":comparison}
    for name,rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding="utf-8-sig")
    b=pd.DataFrame(boundaries);l=pd.DataFrame(match_lifts);sw=pd.DataFrame(swap_rows);cov=pd.DataFrame(coverage_rows)
    audit=dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               condition_S="P00<=round median and P11>=round median (frozen)",factor_hypotheses=FACTORS,
               scenarios="simultaneous S effect and each individual S effect; no probability assigned to either",
               actual_results_used_in_calculation=False,originals_changed=False,production_changed=False)
    (OUT/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    focus=l[(l["round"]=="toto1670")&(l.odds_factor==1.5)].iloc[0]
    hist=data[data.S]
    report=("# BuyPlan v4.0 Step 4：D分岐の必要上昇幅と券の代償\n\n"
            "条件Sの定義と6開催78試合の試合前保存確率を固定した。実結果は計算に使用していない。"
            "オッズ倍率1.1/1.5/2.0は新しい最適化ではなく、前段階と同じ**仮説**。"
            "共通状態の重みも実測値ではない。2試合Sの開催では、両方同時と各試合のみの効果を別計算し、"
            "相関・独立を決めていない。\n\n"
            f"第1670回No.3福岡－長崎は、元P(D) {100*focus.base_P_D:.3f}%に対し、最初のD券が入る境界が"
            f"{100*focus.break_even_P_D:.3f}%（+{focus.required_lift_pp:.3f}ポイント）。"
            "これはその確率が正しいという推定ではない。\n\n"
            "## 全開催・仮説別の境界\n\n```text\n"+l[["round","mode","odds_factor","match_no","base_P_D","break_even_P_D","required_lift_pp"]].to_string(index=False)+"\n```\n\n"
            "## 入れ替えと失う記号\n\n"+f"入れ替えで変わる試合は{len(sw)}行。"
            f"H/A記号が10口全体から消えるケースは{int(cov.lost_HA.fillna('').ne('').sum())}行。"
            "詳細は `ticket_swaps.csv` と `symbol_coverage.csv`。"
            "個別券の変更と10口全体で記号が消えることは別に記録した。\n\n"
            "## 条件Sの過去実績で境界を支持できるか\n\n"
            f"S該当は{len(hist)}試合で実D{int(hist.actual_symbol.eq('D').sum())}、元P(D)期待数{hist.baseline_P_D.sum():.3f}。"
            "これは条件Sを発見した同じ78試合の記述値であり、必要上昇幅の推定値・独立検証結果ではない。"
            "開催別のS件数は少なく、同一データから試合別効果や同時発生頻度を決められない。"
            "したがって現時点では、どの仮説境界も実購入へ適用する裏付けにはならない。\n\n"
            "次の後続未使用開催では、Sの該当と予測P(D)を試合前に固定保存し、実Dとの差とH/D/A全体の校正を検証する。"
            "BuyPlan側は現行分布での損失と、検証済み効果状態での利得を別々に比較する。\n")
    (OUT/"report.md").write_text(report,encoding="utf-8")
    def table(df:pd.DataFrame)->str:
        return df.to_html(index=False,escape=True,float_format=lambda x:f"{x:.6g}",na_rep="なし")
    page=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>v4.0 Step 4 境界詳細</title>"
          "<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:28px auto;padding:0 16px;color:#24364c}"
          "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #d9e0e8;padding:7px}th{background:#e7eff8}.scroll{overflow:auto}"
          ".notice{background:#fff0d7;border-left:4px solid #c68a1b;padding:14px}</style>"
          "<h1>v4.0 Step 4：D券を入れるための必要上昇幅</h1>"
          "<p class='notice'>すべて仮説上の境界です。条件Sの6/8は発見標本の結果で、購入根拠として校正していません。</p>"
          "<h2>試合別P(D)の境界</h2><div class='scroll'>"+table(l)+"</div>"
          "<h2>同時効果・個別効果</h2><div class='scroll'>"+table(b)+"</div>"
          "<h2>入れ替わる券の試合別記号</h2><div class='scroll'>"+table(sw)+"</div>"
          "<h2>10口全体で失うH/A記号</h2><div class='scroll'>"+table(cov[cov.lost_HA.fillna('').ne('')])+"</div>"
          "<p><a href='report.md'>分析</a> / <a href='match_lifts.csv'>必要上昇幅CSV</a> / <a href='ticket_swaps.csv'>券差分CSV</a> / "
          "<a href='symbol_coverage.csv'>記号カバーCSV</a> / <a href='opportunity_cost.csv'>機会損失CSV</a> / <a href='audit.json'>監査</a></p></html>")
    (OUT/"index.html").write_text(page,encoding="utf-8")
    print(OUT)


if __name__=="__main__":
    run()
