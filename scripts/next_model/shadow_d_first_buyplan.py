"""Research-only D-first stratified BuyPlan simulation; no production writes."""
from __future__ import annotations

import argparse
import heapq
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from analyze_score_structure import score_grid
from analyze_score_structure_phase4b_stage1 import top_ten, universe, decode, SYMBOLS
from analyze_score_structure_phase4b_stage3 import draw_distribution
from common import sha256
from forecast_registry import selected_matches


METHODS = ("uniform", "strength", "strength_times_PD")


def strength_from_lambda(lh, la):
    grid, retained = score_grid(float(lh), float(la))
    raw = grid * retained
    ii, jj = np.indices((11, 11))
    sh = float(raw[ii > jj].max())
    sd = float(raw[ii == jj].max())
    sa = float(raw[ii < jj].max())
    return sd / max(sh, sa)


def data_from_saved(frame, current=False):
    frame = frame.sort_values("match_no").reset_index(drop=True)
    if len(frame) != 13 or frame.match_no.tolist() != list(range(1, 14)):
        raise ValueError("expected 13 ordered matches")
    if current:
        p = frame[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
        lambdas = frame[["final_lambda_home", "final_lambda_away"]].to_numpy(float)
    else:
        if not frame.provenance_class.eq("A").all() or frame.fixture_id.duplicated().any():
            raise ValueError("archive must contain unique prematch originals A")
        p = frame[["final_prob_home", "final_prob_draw", "final_prob_away"]].to_numpy(float)
        lambdas = frame[["lambda_home", "lambda_away"]].to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("invalid final probabilities")
    strength = np.array([strength_from_lambda(*pair) for pair in lambdas])
    if current:
        truth = None
    else:
        # Registry result convention is 1=H, 0=D, 2=A.
        truth = frame.actual_result.map({1: 0, 0: 1, 2: 2}).to_numpy()
        if not np.isfinite(truth).all():
            raise ValueError("missing or invalid archived result")
        truth = truth.astype(np.uint8)
    return p, strength, truth


def stratified_counts(distribution, rng):
    cdf = np.cumsum(distribution)
    counts = np.searchsorted(cdf, (np.arange(10) + rng.random(10)) / 10, side="right")
    rng.shuffle(counts)
    return counts


def sample_masks(counts, weights, rng):
    masks = []
    seen = set()
    repeats = 0
    for count in counts:
        count = int(count)
        unique_possible = count in range(1, 13)
        attempts = 0
        while True:
            chosen = tuple(sorted(rng.choice(13, count, replace=False, p=weights / weights.sum()).tolist()))
            mask = sum(1 << i for i in chosen)
            if not unique_possible or mask not in seen or attempts == 100:
                break
            attempts += 1
        repeats += int(mask in seen)
        masks.append(mask)
        seen.add(mask)
    return masks, repeats


def best_assignments(mask, p, n):
    """Exact top-n H/A strings for one fixed D mask; deterministic tie break."""
    if n > 2 ** (13 - mask.bit_count()):
        raise ValueError("D mask cannot support this many distinct H/A tickets")
    symbols = np.argmax(p[:, [0, 2]], axis=1) * 2
    symbols = symbols.astype(np.uint8)
    active = []
    for i in range(13):
        if mask & (1 << i):
            symbols[i] = 1
        else:
            best = int(symbols[i])
            other = 2 - best
            active.append((i, float(np.log(p[i, best] / p[i, other]))))
    active.sort(key=lambda v: v[0])
    heap = [(0., (), -1)]
    answer = []
    while len(answer) < n:
        cost, flips, last = heapq.heappop(heap)
        ticket = symbols.copy()
        for j in flips:
            idx = active[j][0]
            ticket[idx] = 2 - int(ticket[idx])
        answer.append(ticket)
        for j in range(last + 1, len(active)):
            heapq.heappush(heap, (cost + active[j][1], flips + (j,), j))
    return answer


def optimize_ha(masks, p):
    groups = defaultdict(list)
    for ticket_index, mask in enumerate(masks):
        groups[mask].append(ticket_index)
    output = [None] * 10
    for mask, indices in groups.items():
        for index, ticket in zip(indices, best_assignments(mask, p, len(indices))):
            output[index] = ticket
    result = np.stack(output)
    if len({tuple(x) for x in result}) != 10:
        raise AssertionError("duplicate tickets")
    return result


def random_ha(masks, p, rng):
    output = []
    seen = set()
    for mask in masks:
        for _ in range(500):
            ticket = np.zeros(13, dtype=np.uint8)
            for i in range(13):
                if mask & (1 << i):
                    ticket[i] = 1
                else:
                    ticket[i] = int(rng.choice([0, 2], p=p[i, [0, 2]] / (p[i, 0] + p[i, 2])))
            key = tuple(ticket)
            if key not in seen:
                break
        else:
            raise RuntimeError("could not construct unique random H/A ticket")
        seen.add(key)
        output.append(ticket)
    return np.stack(output)


def ticket_mass(tickets, p):
    return float(np.prod(p[np.arange(13)[None, :], tickets], axis=1).sum())


def metrics(tickets, p, truth):
    d_counts = (tickets == 1).sum(axis=1)
    result = dict(P13=ticket_mass(tickets, p), total_D_cells=int(d_counts.sum()),
                  D_ticket_counts=";".join(map(str, d_counts)),
                  D_masks_unique=len({tuple(np.flatnonzero(row == 1)) for row in tickets}),
                  D_cells_by_match=";".join(map(str, (tickets == 1).sum(axis=0))))
    if truth is not None:
        hits = (tickets == truth).sum(axis=1)
        result.update(best_hit=int(hits.max()), hit13=bool((hits == 13).any()),
                      hit12=bool((hits >= 12).any()),
                      actual_D_covered=int(np.sum(np.any(tickets[:, truth == 1] == 1, axis=0))))
    return result


def simulate(frame, round_id, current, seeds, representative_seed):
    p, strength, truth = data_from_saved(frame, current)
    distribution = draw_distribution(p)
    baseline_codes = top_ten(universe(p))
    baseline = decode(baseline_codes)
    rows, frequency, representative = [], {m: np.zeros(13, dtype=int) for m in METHODS}, {}
    rows.append(dict(round=round_id, method="A_fixed_probability_top10", seed=-1,
                     D_mask_duplicates_before_HA=0, P13_before_HA=np.nan,
                     **metrics(baseline, p, truth)))
    for seed in range(seeds):
        counts = stratified_counts(distribution, np.random.default_rng(seed))
        for method_index, method in enumerate(METHODS):
            weights = {"uniform": np.ones(13), "strength": strength,
                       "strength_times_PD": strength * p[:, 1]}[method]
            rng = np.random.default_rng(np.random.SeedSequence([seed, method_index, 431]))
            masks, duplicates = sample_masks(counts, weights, rng)
            pre = random_ha(masks, p, np.random.default_rng(np.random.SeedSequence([seed, method_index, 682])))
            optimized = optimize_ha(masks, p)
            frequency[method] += (optimized == 1).sum(axis=0)
            row = dict(round=round_id, method=method, seed=seed,
                       D_mask_duplicates_before_HA=duplicates,
                       P13_before_HA=ticket_mass(pre, p), **metrics(optimized, p, truth))
            if row["P13"] + 1e-15 < row["P13_before_HA"]:
                raise AssertionError("H/A optimizer reduced P13")
            rows.append(row)
            if seed == representative_seed:
                representative[method] = optimized
    return pd.DataFrame(rows), frequency, representative, baseline, p, strength, distribution, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--historical-seeds", type=int, default=300)
    parser.add_argument("--representative-seed", type=int, default=42)
    args = parser.parse_args()
    if args.seeds <= args.representative_seed or args.historical_seeds <= 0:
        raise ValueError("invalid seed count")
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(args.forecast, encoding="utf-8-sig")
    now, freq, portfolios, base, p, strength, distribution, _ = simulate(current, "toto1672", True, args.seeds, args.representative_seed)
    history = selected_matches()
    historical = []
    for rid in ("toto1660", "toto1663", "toto1664", "toto1665", "toto1668", "toto1670"):
        frame = history[history.round_id.eq(rid)]
        if len(frame) != 13:
            raise ValueError(f"missing archived prematch round {rid}")
        data, _, _, _, _, _, _, _ = simulate(frame, rid, False, args.historical_seeds, 0)
        historical.append(data)
    past = pd.concat(historical, ignore_index=True)
    now.to_csv(out / "toto1672_seed_results.csv", index=False, encoding="utf-8-sig")
    past.to_csv(out / "historical_seed_results.csv", index=False, encoding="utf-8-sig")
    count_rows = []
    for method, group in now[now.method.isin(METHODS)].groupby("method"):
        ticket_counts = Counter(int(value) for entry in group.D_ticket_counts for value in entry.split(";"))
        for d in range(14):
            count_rows.append(dict(method=method, D_count_per_ticket=d,
                                   ticket_count=ticket_counts[d], fraction=ticket_counts[d]/(10*args.seeds)))
    pd.DataFrame(count_rows).to_csv(out / "D_count_per_ticket_frequency.csv", index=False, encoding="utf-8-sig")
    past.groupby(["round","method"],as_index=False).agg(
        runs=("seed","size"), hit13=("hit13","sum"), hit12=("hit12","sum"),
        best_hit_max=("best_hit","max"), best_hit_mean=("best_hit","mean"),
        P13_mean=("P13","mean"), D_cells_mean=("total_D_cells","mean")
    ).to_csv(out / "historical_by_round.csv", index=False, encoding="utf-8-sig")
    row_list = []
    for i, r in current.sort_values("match_no").reset_index(drop=True).iterrows():
        row_list.append(dict(match_no=i+1, home_team=r.home_team, away_team=r.away_team,
                             D_strength=strength[i], P_D=p[i,1],
                             **{f"{method}_D_frequency":freq[method][i]/args.seeds/10 for method in METHODS}))
    pd.DataFrame(row_list).to_csv(out / "D_adoption_by_match.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"D_count":range(14),"probability":distribution}).to_csv(out / "D_count_distribution.csv", index=False, encoding="utf-8-sig")
    representative = {"A_fixed_probability_top10":base, **portfolios}
    tickets = []
    for method, matrix in representative.items():
        for index, ticket in enumerate(matrix, 1):
            tickets.append(dict(method=method, candidate=index, symbols="".join(SYMBOLS[ticket]),
                                D_count=int((ticket==1).sum()), ticket_probability=float(np.prod(p[np.arange(13),ticket])),
                                **{f"match_{i+1:02d}":SYMBOLS[ticket[i]] for i in range(13)}))
    pd.DataFrame(tickets).to_csv(out / "representative_tickets.csv", index=False, encoding="utf-8-sig")
    metrics_rows = []
    for label, data in (("current", now), ("historical", past)):
        for method, group in data.groupby("method"):
            metrics_rows.append(dict(sample=label, method=method, runs=len(group),
                                     P13_mean=float(group.P13.mean()), P13_median=float(group.P13.median()),
                                     P13_min=float(group.P13.min()), P13_max=float(group.P13.max()),
                                     total_D_cells_mean=float(group.total_D_cells.mean()),
                                     P13_before_HA_mean=float(group.P13_before_HA.mean()) if method != "A_fixed_probability_top10" else np.nan,
                                     hit13_count=int(group.hit13.sum()) if label=="historical" else np.nan,
                                     hit12_count=int(group.hit12.sum()) if label=="historical" else np.nan,
                                     best_hit_mean=float(group.best_hit.mean()) if label=="historical" else np.nan))
    pd.DataFrame(metrics_rows).to_csv(out / "method_summary.csv", index=False, encoding="utf-8-sig")
    audit = dict(forecast_sha256=sha256(args.forecast), current_seeds=args.seeds,
                 historical_seeds=args.historical_seeds, representative_seed=args.representative_seed,
                 methods=list(METHODS), D_weight="D_strength * final_P_D for method D; no probability correction",
                 D_counts="10 stratified quantile draws from Poisson-binomial each seed",
                 H_A="exact top assignments for each fixed D mask, no D removal",
                 historical_rounds=6, archive_prematch_class="A", actual_results_used_for_selection=False,
                 production_changed=False)
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    render_html(out, current.sort_values("match_no").reset_index(drop=True), representative, p, now, past, strength, distribution, args.representative_seed)
    print(json.dumps(audit, ensure_ascii=False))


def render_html(out, matches, portfolios, p, now, past, strength, distribution, seed):
    summary = pd.read_csv(out / "method_summary.csv")
    adoption = pd.read_csv(out / "D_adoption_by_match.csv")
    current_summary = summary[summary["sample"].eq("current")].copy()
    historical_summary = summary[summary["sample"].eq("historical")].copy()
    for table in (current_summary, historical_summary):
        for col in ("P13_mean", "P13_median", "P13_min", "P13_max", "P13_before_HA_mean"):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{100*x:.8f}%")
    charts = []
    colors = {"uniform":"#7587b7", "strength":"#7eac6b", "strength_times_PD":"#d28757"}
    for method in METHODS:
        values = adoption[f"{method}_D_frequency"].to_numpy()
        bars = "".join(f"<tr><td>{i+1:02d} {html.escape(str(matches.iloc[i].home_team))}－{html.escape(str(matches.iloc[i].away_team))}</td><td><div class='bar' style='width:{max(1, v*360):.1f}px;background:{colors[method]}'></div></td><td>{100*v:.1f}%</td></tr>" for i,v in enumerate(values))
        charts.append(f"<details><summary>{method}：試合別D採用頻度</summary><table>{bars}</table></details>")
    plans = []
    for method, tickets in portfolios.items():
        header = "<tr><th>No.</th><th>対戦</th>"+"".join(f"<th>候補{i:02d}</th>" for i in range(1,11))+"</tr>"
        body = []
        for i, match in matches.iterrows():
            cells = [str(i+1),f"{match.home_team}－{match.away_team}"]+[str(SYMBOLS[tickets[j,i]]) for j in range(10)]
            body.append("<tr>"+"".join(f"<td class='{('draw' if value=='D' else '')}'>{html.escape(value)}</td>" for value in cells)+"</tr>")
        row = now[(now.method.eq(method)) & (now.seed.eq(seed if method != "A_fixed_probability_top10" else -1))].iloc[0]
        plans.append(f"<details><summary>{method}：D総枚数 {int(row.total_D_cells)}、P13 {100*row.P13:.8f}%</summary><div class='scroll'><table>{header}{''.join(body)}</table></div></details>")
    by_round = pd.read_csv(out / "historical_by_round.csv")
    current_means = now[now.method.isin(METHODS)].groupby("method").agg(
        before=("P13_before_HA","mean"), after=("P13","mean"),
        D_cells=("total_D_cells","mean"))
    historical_hits = int(past[past.method.isin(METHODS)].hit13.sum())
    dist_rows = "".join(f"<tr><td>{i}</td><td>{100*v:.3f}%</td><td><div class='bar' style='width:{v*1000:.1f}px;background:#87a9c9'></div></td></tr>" for i,v in enumerate(distribution))
    report=("# D先行シャドー比較\n\n"
            "D本数は各開催の保存final P(D)によるポアソン二項分布から層化抽選。"
            "採用試合は一様、D強度、D強度×P(D)の3方式。D配置後にH/Aを厳密最適化し、Dは変更しない。"
            "D強度は新しい発生確率ではなく、追加予測力は未確認。"
            "過去6開催は同じ78試合から特徴の検討が行われているため、独立した方式評価ではない。"
            "13的中が希少であり、少数開催・複数シードを独立試合のように扱わない。"
            "同じ保存P(D)からD本数を層化抽選するため、各方式のD総枚数分布は同じ。"
            "方式間の差はD試合の選び方に限られる。D強度はλ由来で、元P(D)にない情報の発見を意味しない。\n\n"
            f"第1672回のD総枚数平均は全ランダム方式で {current_means.iloc[0].D_cells:.3f}。"
            f"H/A最適化前後の平均P13は一様方式 {100*current_means.loc['uniform','before']:.8f}% → {100*current_means.loc['uniform','after']:.8f}% 、"
            f"D強度×P(D)方式 {100*current_means.loc['strength_times_PD','before']:.8f}% → {100*current_means.loc['strength_times_PD','after']:.8f}% 。"
            "この改善はランダムH/A配置から最有力H/A配置への比較であり、D選出の予測力を示さない。"
            f"過去6開催×各300シードの3方式で13的中は合計 {historical_hits} 件。"
            "シードは同じ試合を繰り返し購入した実績ではなく、仮想配券の反復である。\n\n"
            "## 現行1672\n\n```csv\n"+current_summary.to_csv(index=False)+"```\n\n"
            "## 過去6開催\n\n```csv\n"+historical_summary.to_csv(index=False)+"```\n")
    (out / "report.md").write_text(report, encoding="utf-8")
    doc=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>D先行BuyPlanシャドー</title>"
         "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.draw{background:#ffe299;font-weight:700}details{padding:8px;border-bottom:1px solid #ddd}.note{background:#fff0d3;padding:12px}.bar{height:12px;display:inline-block}</style>"
         "<h1>D先行BuyPlan・研究用シャドー</h1>"
         f"<p class='note'>第1672回、代表シード{seed}。Dを先に抽選・固定してH/Aを最適化。正式BuyPlanやNext Modelは変更していません。D強度の追加予測力は未検証であり、元のモデル確率の大小だけで方式の採否を決めません。</p>"
         "<h2>D発生本数の保存モデル分布</h2><table><tr><th>D数</th><th>確率</th><th>棒</th></tr>"+dist_rows+"</table>"
         "<h2>第1672回・1000シード比較</h2><div class='scroll'>"+current_summary.to_html(index=False)+"</div>"
         "<h2>過去6開催の事後成績</h2><div class='scroll'>"+historical_summary.to_html(index=False)+"</div>"
         "<h3>開催別</h3><div class='scroll'>"+by_round.to_html(index=False)+"</div>"
         "<h2>代表シードの13試合×候補01〜10</h2>"+"".join(plans)
         +"<h2>試合別D採用頻度</h2><div class='scroll'>"+adoption.to_html(index=False)+"</div>"+"".join(charts)
         +"<p><a href='representative_tickets.csv'>代表配券CSV</a>／<a href='toto1672_seed_results.csv'>全シードCSV</a>／<a href='D_count_per_ticket_frequency.csv'>券別D本数</a>／<a href='historical_seed_results.csv'>過去開催CSV</a>／<a href='historical_by_round.csv'>開催別成績</a>／<a href='report.md'>考察</a></p></html>")
    (out / "index.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
