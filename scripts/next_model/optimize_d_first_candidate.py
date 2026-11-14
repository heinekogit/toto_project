"""D-first shadow method comparison and deterministic purchase-candidate proposal."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analyze_score_structure_phase4b_stage1 import top_ten, universe, decode, SYMBOLS
from analyze_score_structure_phase4b_stage3 import draw_distribution
from common import sha256
from forecast_registry import selected_matches
from shadow_d_first_buyplan import (METHODS, data_from_saved, stratified_counts,
                                    sample_masks, optimize_ha, ticket_mass, best_assignments)


ALL_METHODS = ("uniform", "strength", "strength_times_PD", "PD_sequential", "PD_conditional", "PD_conditional_strength")


def conditional_subset(k: int, weights: np.ndarray, rng: np.random.Generator) -> int:
    """Sample exactly k indices with probability proportional to product weights."""
    n = len(weights)
    suffix = np.zeros((n+1, k+1), dtype=float)
    suffix[n, 0] = 1.
    for i in range(n-1, -1, -1):
        suffix[i, 0] = 1.
        for r in range(1, k+1):
            suffix[i, r] = suffix[i+1, r] + weights[i] * suffix[i+1, r-1]
    mask, remaining = 0, k
    for i in range(n):
        if remaining == 0:
            break
        probability = weights[i] * suffix[i+1, remaining-1] / suffix[i, remaining]
        if rng.random() < probability:
            mask |= 1 << i
            remaining -= 1
    if remaining:
        raise AssertionError("conditional subset draw incomplete")
    return mask


def masks_for(method, counts, p, strength, rng, distinct_masks=True):
    if method in METHODS or method == "PD_sequential":
        weights = {"uniform": np.ones(13), "strength": strength,
                   "strength_times_PD": strength*p[:,1], "PD_sequential": p[:,1]}[method]
        if distinct_masks:
            return sample_masks(counts, weights, rng)[0]
        return [sum(1 << i for i in rng.choice(13, int(k), replace=False, p=weights/weights.sum())) for k in counts]
    odds = p[:,1]/(1-p[:,1])
    weights = odds if method == "PD_conditional" else odds*strength
    masks, used = [], set()
    for k in counts:
        for attempt in range(101):
            mask = conditional_subset(int(k), weights, rng)
            if not distinct_masks or mask not in used or attempt == 100:
                break
        masks.append(mask)
        used.add(mask)
    return masks


def mask_mass(mask, pd):
    return float(np.prod([pd[i] if mask & (1 << i) else 1-pd[i] for i in range(13)]))


def evaluate(masks, p, truth):
    tickets = optimize_ha(masks, p)
    individually = np.stack([best_assignments(mask,p,1)[0] for mask in masks])
    unique_individual = np.unique(individually,axis=0)
    unique = set(masks)
    coverage = sum(mask_mass(mask, p[:,1]) for mask in unique)
    value = ticket_mass(tickets, p)
    result = dict(P13=value, D_mask_mass=coverage, unique_D_masks=len(unique),
                  individual_unique_P13=ticket_mass(unique_individual,p),
                  individual_duplicate_tickets=10-len(unique_individual),
                  joint_HA_gain=value-ticket_mass(unique_individual,p),
                  D_total=int((tickets==1).sum()), D_counts=";".join(map(str,(tickets==1).sum(axis=1))))
    if truth is not None:
        actual_mask = sum(1 << i for i in range(13) if truth[i] == 1)
        exact_D = actual_mask in unique
        hits = (tickets==truth).sum(axis=1)
        result.update(D_mask_exact=exact_D, hit13=bool(np.any(hits==13)),
                      hit12=bool(np.any(hits>=12)), best_hit=int(hits.max()),
                      matched_D_mask_tickets=int(masks.count(actual_mask)),
                      H_A_best_hit_given_D_mask=int(max(((tickets[j]==truth).sum() for j,mask in enumerate(masks) if mask==actual_mask), default=0)),
                      failure="13的中" if np.any(hits==13) else "D集合一致・H/A外れ" if exact_D else "D集合不一致")
    return result, tickets


def run_round(frame, round_id, current, seeds, policies):
    p, strength, truth = data_from_saved(frame, current)
    distribution = draw_distribution(p)
    baseline = decode(top_ten(universe(p)))
    baseline_value = ticket_mass(baseline,p)
    rows=[]
    for seed in range(seeds):
        counts = stratified_counts(distribution,np.random.default_rng(seed))
        for method in ALL_METHODS:
            for policy in policies:
                # Paired stream across methods and collision policies.
                rng = np.random.default_rng(np.random.SeedSequence([seed,729]))
                masks = masks_for(method,counts,p,strength,rng,distinct_masks=(policy=="broad"))
                info,_ = evaluate(masks,p,truth)
                rows.append(dict(round=round_id,seed=seed,method=method,policy=policy,
                                 baseline_P13=baseline_value,**info))
    return pd.DataFrame(rows),p,strength,truth,distribution


def deterministic_candidate(source: Path | str, p, strength, draws=64, method="PD_conditional_strength"):
    """One predeclared selection: fixed D counts; maximize unique D-set mass among proposals."""
    digest=source if isinstance(source,str) else sha256(source)
    if len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("expected SHA-256 of prematch forecast")
    if method not in ("PD_conditional", "PD_conditional_strength"):
        raise ValueError("unsupported deterministic method")
    base_seed=int(digest[:16],16)
    counts=stratified_counts(draw_distribution(p),np.random.default_rng(base_seed))
    proposals=[]
    for candidate_id in range(draws):
        rng=np.random.default_rng(np.random.SeedSequence([base_seed & 0xffffffff, base_seed >> 32, candidate_id, 9801]))
        masks=masks_for(method,counts,p,strength,rng,distinct_masks=True)
        info,tickets=evaluate(masks,p,None)
        proposals.append((info["D_mask_mass"],info["P13"],-candidate_id,candidate_id,masks,tickets,info))
    chosen=max(proposals,key=lambda x:x[:3])
    return dict(base_seed=base_seed,proposal_count=draws,proposal_id=chosen[3],D_counts=list(map(int,counts)),
                masks=chosen[4],tickets=chosen[5],metrics=chosen[6],
                rule="same stratified D counts; conditional-set weights chosen by method; among 64 proposals maximize distinct D-mask probability mass; P13 tie break; no result input")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--forecast",type=Path,required=True)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=1000)
    parser.add_argument("--historical-seeds",type=int,default=300)
    args=parser.parse_args()
    out=args.outdir;out.mkdir(parents=True,exist_ok=True)
    current=pd.read_csv(args.forecast,encoding="utf-8-sig").sort_values("match_no").reset_index(drop=True)
    now,p,strength,_,distribution=run_round(current,"toto1672",True,args.seeds,("broad","repeat"))
    adoption={method:np.zeros(13,dtype=int) for method in ALL_METHODS}
    for seed in range(args.seeds):
        counts=stratified_counts(distribution,np.random.default_rng(seed))
        for method in ALL_METHODS:
            masks=masks_for(method,counts,p,strength,
                            np.random.default_rng(np.random.SeedSequence([seed,729])),distinct_masks=True)
            for mask in masks:
                for i in range(13):
                    adoption[method][i]+=bool(mask & (1<<i))
    pd.DataFrame(dict(match_no=current.match_no,home_team=current.home_team,away_team=current.away_team,
        D_strength=strength,P_D=p[:,1],
        **{f"{method}_D_adoption_rate":adoption[method]/(10*args.seeds) for method in ALL_METHODS})
    ).to_csv(out/"D_adoption_by_match.csv",index=False,encoding="utf-8-sig")
    old=selected_matches()
    historical=[]
    diagnostic_rows=[]
    for round_id in ("toto1660","toto1663","toto1664","toto1665","toto1668","toto1670"):
        group=old[old.round_id.eq(round_id)]
        if len(group)!=13:raise ValueError(round_id)
        data,old_p,old_strength,old_truth,_=run_round(group,round_id,False,args.historical_seeds,("broad","repeat"))
        historical.append(data)
        for i in range(13):
            diagnostic_rows.append(dict(round=round_id,match_no=i+1,P_D=float(old_p[i,1]),
                D_strength=float(old_strength[i]),actual_D=bool(old_truth[i]==1)))
    diagnostic=pd.DataFrame(diagnostic_rows)
    diagnostic.to_csv(out/"D_strength_historical_78.csv",index=False,encoding="utf-8-sig")
    def auc(col):
        yes=diagnostic.loc[diagnostic.actual_D,col]
        no=diagnostic.loc[~diagnostic.actual_D,col]
        return float(mannwhitneyu(yes,no).statistic/(len(yes)*len(no)))
    diagnostic_summary=dict(n=78,actual_D=int(diagnostic.actual_D.sum()),
        AUC_D_strength=auc("D_strength"),AUC_P_D=auc("P_D"),
        mean_strength_D=float(diagnostic.loc[diagnostic.actual_D,"D_strength"].mean()),
        mean_strength_nonD=float(diagnostic.loc[~diagnostic.actual_D,"D_strength"].mean()),
        in_sample_exploratory=True)
    (out/"D_strength_diagnostic.json").write_text(json.dumps(diagnostic_summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    past=pd.concat(historical,ignore_index=True)
    now.to_csv(out/"current_seed_comparison.csv",index=False,encoding="utf-8-sig")
    past.to_csv(out/"historical_seed_comparison.csv",index=False,encoding="utf-8-sig")
    summary=[]
    for segment,data in (("current",now),("historical",past)):
        for (method,policy),group in data.groupby(["method","policy"]):
            summary.append(dict(segment=segment,method=method,policy=policy,runs=len(group),
                P13_mean=float(group.P13.mean()),P13_median=float(group.P13.median()),
                joint_HA_gain_mean=float(group.joint_HA_gain.mean()),
                individual_duplicates_mean=float(group.individual_duplicate_tickets.mean()),
                D_mask_mass_mean=float(group.D_mask_mass.mean()),unique_D_masks_mean=float(group.unique_D_masks.mean()),
                D_total_mean=float(group.D_total.mean()),
                D_mask_exact_count=int(group.D_mask_exact.sum()) if segment=="historical" else np.nan,
                hit13_count=int(group.hit13.sum()) if segment=="historical" else np.nan,
                H_A_failure_count=int(group.failure.eq("D集合一致・H/A外れ").sum()) if segment=="historical" else np.nan,
                best_hit_mean=float(group.best_hit.mean()) if segment=="historical" else np.nan))
    summary=pd.DataFrame(summary)
    summary.to_csv(out/"method_summary.csv",index=False,encoding="utf-8-sig")
    past.groupby(["round","method","policy"],as_index=False).agg(
        runs=("seed","size"),D_mask_exact=("D_mask_exact","sum"),hit13=("hit13","sum"),
        H_A_failure=("failure",lambda x:int((x=="D集合一致・H/A外れ").sum())),
        P13_mean=("P13","mean"),D_mask_mass_mean=("D_mask_mass","mean"),best_hit_mean=("best_hit","mean")
    ).to_csv(out/"historical_by_round.csv",index=False,encoding="utf-8-sig")
    candidate_history=[]
    for round_id in ("toto1660","toto1663","toto1664","toto1665","toto1668","toto1670"):
        group=old[old.round_id.eq(round_id)]
        if group.output_sha256.nunique()!=1:
            raise ValueError("ambiguous prematch forecast hash")
        historic_p,historic_strength,historic_truth=data_from_saved(group,False)
        digest=str(group.output_sha256.iloc[0])
        for method in ("PD_conditional","PD_conditional_strength"):
            for proposal_count in (1,64):
                choice=deterministic_candidate(digest,historic_p,historic_strength,proposal_count,method)
                info,_=evaluate(choice["masks"],historic_p,historic_truth)
                candidate_history.append(dict(round=round_id,method=method,proposal_count=proposal_count,
                    selected_proposal_id=choice["proposal_id"],base_seed=choice["base_seed"],**info))
    pd.DataFrame(candidate_history).to_csv(out/"deterministic_historical.csv",index=False,encoding="utf-8-sig")
    # Precommitted purchase-candidate rule: conditional P(D), broad mask sampling.
    candidate=deterministic_candidate(args.forecast,p,strength)
    matrix=candidate.pop("tickets")
    ticket_rows=[]
    for j,ticket in enumerate(matrix,1):
        ticket_rows.append(dict(candidate=f"候補{j:02d}",symbols="".join(SYMBOLS[ticket]),
                                D_count=int((ticket==1).sum()),probability=float(np.prod(p[np.arange(13),ticket])),
                                **{f"match_{i+1:02d}":SYMBOLS[ticket[i]] for i in range(13)}))
    pd.DataFrame(ticket_rows).to_csv(out/"purchase_candidate_10.csv",index=False,encoding="utf-8-sig")
    reference=np.where(p[:,0]>=p[:,2],"H","A")
    pd.DataFrame(dict(match_no=current.match_no,home_team=current.home_team,away_team=current.away_team,
                      candidate00_reference=reference)).to_csv(out/"candidate00_reference.csv",index=False,encoding="utf-8-sig")
    candidate["source_sha256"]=sha256(args.forecast)
    candidate["method"]="PD_conditional_strength"
    candidate["mask_policy"]="broad"
    candidate["lambda_home"]=[float(x) for x in current.final_lambda_home]
    candidate["lambda_away"]=[float(x) for x in current.final_lambda_away]
    candidate["D_strength"]=[float(x) for x in strength]
    candidate["HDA_probabilities"]=p.tolist()
    (out/"purchase_candidate_reproduction.json").write_text(json.dumps(candidate,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    leader=np.where(p[:,0]>=p[:,2],0,2)
    reverse=((matrix!=1)&(matrix!=leader[None,:]))
    ha_rows=[]
    for i,match in current.iterrows():
        ha_rows.append(dict(match_no=i+1,home_team=match.home_team,away_team=match.away_team,
            candidate00_HA=SYMBOLS[leader[i]],D_cells=int((matrix[:,i]==1).sum()),
            H_A_leader_cells=int((matrix[:,i]==leader[i]).sum()),H_A_reverse_cells=int(reverse[:,i].sum()),
            conditional_reverse_probability=float(min(p[i,0],p[i,2])/(p[i,0]+p[i,2]))))
    pd.DataFrame(ha_rows).to_csv(out/"HA_coverage_diagnostic.csv",index=False,encoding="utf-8-sig")
    conditional_all_leader=[]
    for ticket in matrix:
        free=ticket!=1
        conditional_all_leader.append(float(np.prod(np.maximum(p[free,0],p[free,2])/(p[free,0]+p[free,2]))))
    historical_nonD=old[old.round_id.isin(("toto1660","toto1663","toto1664","toto1665","toto1668","toto1670"))].copy()
    historical_leader=np.where(historical_nonD.final_prob_home>=historical_nonD.final_prob_away,1,2)
    nonD=historical_nonD.actual_result.ne(0)
    ha_audit=dict(candidate_reverse_HA_cells=int(reverse.sum()),candidate_nonD_cells=int((matrix!=1).sum()),
        candidate_distinct_D_masks=len(set(candidate["masks"])),
        conditional_all_HA_leader_probability_min=min(conditional_all_leader),
        conditional_all_HA_leader_probability_max=max(conditional_all_leader),
        historical_nonD_matches=int(nonD.sum()),
        historical_HA_reverse_matches=int((nonD & historical_nonD.actual_result.ne(historical_leader)).sum()),
        purchase_ready=False,reason="all non-D cells use candidate00 H/A; H/A reversal not covered")
    (out/"HA_coverage_audit.json").write_text(json.dumps(ha_audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    audit=dict(source_sha256=sha256(args.forecast),seeds=args.seeds,historical_seeds=args.historical_seeds,
        historical_rounds=6,method_definitions={
            "uniform":"sequential uniform without replacement",
            "strength":"sequential D_strength without replacement",
            "strength_times_PD":"sequential D_strength*P(D) without replacement",
            "PD_sequential":"sequential P(D) without replacement",
            "PD_conditional":"exact conditional D set P(S|count=k) proportional to product P(D)/(1-P(D))",
            "PD_conditional_strength":"conditional-set sampling with weights D_strength*P(D)/(1-P(D)); research weight only"},
        same_stratified_counts_across_methods=True,D_removed_after_sampling=False,
        probability_correction=False,actual_results_used_for_selection=False,formal_program_changed=False)
    (out/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    render(out,current,p,strength,distribution,summary,candidate,matrix)
    print(json.dumps({"candidate_method":candidate["method"],"candidate_P13":candidate["metrics"]["P13"],
                      "candidate_D_mask_mass":candidate["metrics"]["D_mask_mass"]},ensure_ascii=False))


def render(out,current,p,strength,distribution,summary,candidate,matrix):
    row_headers="<tr><th>No.</th><th>対戦</th><th>候補00<br>参考・購入外</th>"+"".join(f"<th>候補{i:02d}</th>" for i in range(1,11))+"</tr>"
    matrix_rows=[]
    for i,match in current.iterrows():
        ref="H" if p[i,0]>=p[i,2] else "A"
        cells=[str(i+1),f"{match.home_team}－{match.away_team}",ref]+[str(SYMBOLS[matrix[j,i]]) for j in range(10)]
        matrix_rows.append("<tr>"+"".join(f"<td class='{('draw' if c=='D' else '')}'>{html.escape(c)}</td>" for c in cells)+"</tr>")
    current_summary=summary[summary.segment.eq("current")].copy()
    historic_summary=summary[summary.segment.eq("historical")].copy()
    for table in (current_summary,historic_summary):
        for key in ("P13_mean","P13_median","D_mask_mass_mean"):
            table[key]=table[key].map(lambda x:f"{100*x:.6f}%")
    past_round=pd.read_csv(out/"historical_by_round.csv")
    deterministic_past=pd.read_csv(out/"deterministic_historical.csv")
    diagnostic=json.loads((out/"D_strength_diagnostic.json").read_text())
    ha_audit=json.loads((out/"HA_coverage_audit.json").read_text())
    adoption=pd.read_csv(out/"D_adoption_by_match.csv")
    pd_only=summary[(summary.segment.eq("historical"))&(summary.method.eq("PD_conditional"))&(summary.policy.eq("broad"))].iloc[0]
    strength_added=summary[(summary.segment.eq("historical"))&(summary.method.eq("PD_conditional_strength"))&(summary.policy.eq("broad"))].iloc[0]
    high=adoption.loc[adoption.match_no.eq(2)].iloc[0]
    low=adoption.loc[adoption.match_no.eq(10)].iloc[0]
    dist=pd.DataFrame({"D_count":range(14),"probability":[f"{100*x:.3f}%" for x in distribution]})
    score=pd.DataFrame(dict(match_no=current.match_no,home_team=current.home_team,away_team=current.away_team,
                            D_strength=strength,P_D=p[:,1]))
    adoption=pd.read_csv(out/"D_adoption_by_match.csv")
    report=("# D先行BuyPlan 購入候補化の研究\n\n"
        "D本数は全方式で同じ層化抽選を維持した。抽選済みDをH/Aへ戻さない。"
        "条件付き方式は独立ベルヌーイのD集合をD本数kで条件付けた厳密分布を使用。"
        "D強度は未校正の研究重みで、発生確率に読み替えていない。"
        "D集合の異なる10口が覆う確率質量はP(D)の積で計算。"
        "H/Aは固定D集合ごとの上位候補を同時に選ぶ。D集合が重複した場合は別のH/A券を割り当て、完全重複を禁止。"
        "同じD集合に対する個別最適化は同一券の重複を生むため、そのままでは購入に使えない。\n\n"
        "## 第一候補\n\n"
        "P(D)の条件付きオッズにD強度を掛けてD集合を抽選する方式を、研究用の暫定第一候補とする。"
        "同一入力からハッシュ由来のシードを固定し、D本数を一回だけ層化抽選。"
        "そのD本数を維持した64案から、異なるD集合のモデル確率質量が最大の案を選び、P13を同率時の判定にする。"
        "人間がシードを選ばない。D強度は未校正であり、購入優位は未証明。P(D)のみを対照に残す。"
        "この推奨は運用手順の提案であり、実際の全的中率優位を証明するものではない。"
        f"D強度の反映は採用頻度に見える。第1672回のC大阪－浦和はP(D)条件付き {100*high.PD_conditional_D_adoption_rate:.2f}% → D強度併用 {100*high.PD_conditional_strength_D_adoption_rate:.2f}% 、"
        f"富山－徳島は {100*low.PD_conditional_D_adoption_rate:.2f}% → {100*low.PD_conditional_strength_D_adoption_rate:.2f}% 。"
        f"しかし過去6開催各300シードのD集合完全一致はP(D)のみ {int(pd_only.D_mask_exact_count)}件、D強度併用 {int(strength_added.D_mask_exact_count)}件、双方13的中0件。"
        f"過去78試合内の探索的AUCはD強度 {diagnostic['AUC_D_strength']:.3f}、P(D) {diagnostic['AUC_P_D']:.3f}。"
        "同じ6開催内のシード反復は独立開催ではなく、D強度の効果がないと断定もできない。"
        f"ただし暫定10口の非Dセル {ha_audit['candidate_nonD_cells']} 個にH/A逆側は {ha_audit['candidate_reverse_HA_cells']} 個。"
        f"過去6開催の非D {ha_audit['historical_nonD_matches']} 試合中、H/A逆側は {ha_audit['historical_HA_reverse_matches']} 試合。"
        "この10口を購入可能な完成案とは扱わない。D集合を固定したままH/A分岐を10口全体で検討する必要がある。"
        "6開催でのシード反復は独立した開催実績ではない。"
        "同じハッシュシードから1案を採用した場合と64案から選んだ場合を、過去開催でも事後照合した。\n\n"
        f"再現用base_seed={candidate['base_seed']}、proposal_id={candidate['proposal_id']}、10口P13={100*candidate['metrics']['P13']:.8f}%。\n\n"
        "## 方式比較\n\n```csv\n"+summary.to_csv(index=False)+"```\n")
    (out/"report.md").write_text(report,encoding="utf-8")
    doc=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>D先行BuyPlan第一候補</title>"
         "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.draw{background:#ffe299;font-weight:700}.note{background:#fff0d3;padding:12px}</style>"
         "<h1>第1672回 D先行BuyPlan・H/A再検討が必要な試作</h1>"
         f"<p class='note'>研究用。正式BuyPlan・購入処理は変更していません。D本数は層化抽選で先に確定。現案は非D {ha_audit['candidate_nonD_cells']} セルのH/A逆側を {ha_audit['candidate_reverse_HA_cells']} セルしか含まず、購入用の完成案とは扱いません。</p>"
         f"<p>方式：P(D)の条件付きオッズ×D強度によるD集合抽選／broad。固定入力由来シード {candidate['base_seed']}、64案中{candidate['proposal_id']}番。10口P13 {100*candidate['metrics']['P13']:.8f}%。D集合の捕捉確率質量 {100*candidate['metrics']['D_mask_mass']:.4f}%。D強度は発生確率ではなく未校正の抽選重みです。</p>"
         "<h2>13試合×候補01〜10（候補00は購入外の参考）</h2><div class='scroll'><table>"+row_headers+"".join(matrix_rows)+"</table></div>"
         +"<h2>D発生本数のモデル分布</h2>"+dist.to_html(index=False)
         +"<h2>D強度とP(D)</h2><div class='scroll'>"+score.to_html(index=False)+"</div>"
         +"<h3>試合別D採用頻度（1,000シード）</h3><div class='scroll'>"+adoption.to_html(index=False)+"</div>"
         +"<h3>H/Aの逆側を拾えていない箇所</h3><div class='scroll'>"+pd.read_csv(out/"HA_coverage_diagnostic.csv").to_html(index=False)+"</div>"
         +f"<p>D強度は配置へ反映されたが、過去6開催のD集合完全一致はP(D)のみ {int(pd_only.D_mask_exact_count)}/1800、D強度併用 {int(strength_added.D_mask_exact_count)}/1800。13的中は双方0。この差で予測力は判断できません。</p>"
         +f"<p>過去78試合内の探索的AUC：D強度 {diagnostic['AUC_D_strength']:.3f}、P(D) {diagnostic['AUC_P_D']:.3f}。同じ試合群の診断であり、独立検証ではありません。</p>"
         +"<h2>第1672回の方式比較</h2><div class='scroll'>"+current_summary.to_html(index=False)+"</div>"
         +"<h2>過去6開催の比較</h2><div class='scroll'>"+historic_summary.to_html(index=False)+"</div>"
         +"<h3>開催別D集合一致・13的中</h3><div class='scroll'>"+past_round.to_html(index=False)+"</div>"
         +"<h3>決定的手順の過去開催照合：1案と64案</h3><div class='scroll'>"+deterministic_past.to_html(index=False)+"</div>"
         +"<p>同じD本数で比較。D集合の確率質量、10口P13、実際のD集合一致を混同しません。過去6開催だけでは購入優位を証明できません。</p>"
         +"<p><a href='purchase_candidate_10.csv'>試作の10口CSV</a>／<a href='purchase_candidate_reproduction.json'>再現設定</a>／<a href='HA_coverage_diagnostic.csv'>H/A欠落診断</a>／<a href='D_adoption_by_match.csv'>試合別D採用頻度</a>／<a href='D_strength_historical_78.csv'>78試合の診断</a>／<a href='current_seed_comparison.csv'>全シード比較</a>／<a href='historical_by_round.csv'>開催別検証</a>／<a href='deterministic_historical.csv'>決定的手順の照合</a>／<a href='report.md'>研究レポート</a></p></html>")
    (out/"index.html").write_text(doc,encoding="utf-8")


if __name__=="__main__":main()
