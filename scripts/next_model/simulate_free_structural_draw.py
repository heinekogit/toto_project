"""Offline, free-count ticket selection using a structural D placement score.

Placement scores never replace forecast probabilities. No result enters selection.
All strengths are exploratory; there is no production integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from common import ROOT
from expected_goals import score_distribution
from match_draw_coefficient_shadow import extract
from simulate_free_portfolio_1664 import top_joint_candidates

STRENGTHS = (0.0, 0.5, 1.0, 1.5, 2.0)
PROB_COLUMNS = ["next_prob_home", "next_prob_draw", "next_prob_away"]
LABELS = "HDA"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proximity_components(raw_home, raw_away, home_one, away_one):
    """Score-space proximity, not a forecast of an in-play transition.

    A one-goal favourite does not need a matching one-goal underdog outcome.
    Balance is kept as a separate rating based on all winning scorelines.
    """
    h, a, h1, a1 = map(lambda v: np.asarray(v, dtype=float), (raw_home, raw_away, home_one, away_one))
    if np.any(h <= 0) or np.any(a <= 0) or np.any(h1 < 0) or np.any(a1 < 0) or np.any(h1 > h+1e-10) or np.any(a1 > a+1e-10):
        raise ValueError("Invalid winning/one-goal masses")
    nh, na = np.clip(h1/h, 0, 1), np.clip(a1/a, 0, 1)
    balance = 2*np.minimum(h, a)/(h+a)
    near = np.where(h >= a, nh, na)
    # Bounded OR of two ratings. It is not a probability union or an
    # independence assumption: each can support D without requiring the other.
    combined = balance + (1-balance)*near
    return balance, nh, na, near, combined


def structure(frame, mode="corrected"):
    features = extract(frame).reset_index(drop=True)
    low = poisson.cdf(2, features.lambda_total.to_numpy())
    plus = features.P_home_lead1.to_numpy()
    minus = features.P_away_lead1.to_numpy()
    balance = 2 * np.minimum(plus, minus) / (plus + minus)
    features["low_total_le2"] = low
    features["balanced_one_goal_sides"] = balance
    raw = np.array([[d["prob_home"], d["prob_away"]] for d in (
        score_distribution(h, a) for h, a in zip(features.lambda_home, features.lambda_away))])
    full_balance, nh, na, near, combined = proximity_components(raw[:, 0], raw[:, 1], plus, minus)
    features["score_P_H"] = raw[:, 0]
    features["score_P_A"] = raw[:, 1]
    features["full_win_balance"] = full_balance
    features["one_goal_given_H_win"] = nh
    features["one_goal_given_A_win"] = na
    features["favoured_HA"] = np.where(raw[:, 0] >= raw[:, 1], "H", "A")
    features["favoured_one_goal_share"] = near
    features["balance_or_one_goal_rating"] = combined
    features["Q_legacy"] = low*balance
    features["Q_near_only"] = low*near
    features["Q_corrected"] = low*combined
    if mode not in ("legacy", "near_only", "corrected"):
        raise ValueError("Unknown structural evaluation mode")
    features["D_structure_support"] = features[f"Q_{mode}"]
    features["structure_mode"] = mode
    return features


def placement_scores(p, support, strength):
    if strength < 0 or not np.isfinite(strength):
        raise ValueError("Invalid experimental strength")
    p = np.asarray(p, dtype=float)
    gap = np.maximum(np.maximum(p[:, 0], p[:, 2]) - p[:, 1], 0)
    scores = p.copy()
    scores[:, 1] += strength * support * gap
    return scores


def stored_tickets(frame, start=1):
    raw = frame[[f"ticket{i:02d}" for i in range(start, start + 10)]].to_numpy(dtype=int).T
    if not np.isin(raw, [0, 1, 2]).all():
        raise ValueError("Invalid saved ticket symbols")
    return np.where(raw == 1, 0, np.where(raw == 0, 1, 2))


def metrics(p, tickets, actual):
    if tickets.shape != (10, 13) or len(np.unique(tickets, axis=0)) != 10:
        raise ValueError("Expected ten unique tickets")
    mass = np.prod(p[np.arange(13), tickets], axis=1)
    dcounts = (tickets == 1).sum(axis=1)
    draw_distribution = np.array([1.0])
    for pd_ in p[:, 1]:
        draw_distribution = np.convolve(draw_distribution, [1 - pd_, pd_])
    strata = []
    for k in range(14):
        joint = float(mass[dcounts == k].sum())
        strata.append({"actual_D_count": k, "model_D_count_probability": float(draw_distribution[k]),
                       "tickets_with_this_D_count": int((dcounts == k).sum()),
                       "P_any13_and_D_count": joint,
                       "P_any13_given_D_count": joint / draw_distribution[k]})
    # Distinct exact-result tickets describe mutually exclusive events.
    return {"P_any13": float(mass.sum()), "D_total": int(dcounts.sum()),
            "D_counts": "/".join(map(str, dcounts)),
            "tickets_without_D": int((dcounts == 0).sum()),
            "actual_best": None if actual is None else int((tickets == actual).sum(axis=1).max()),
            "actual_any13": None if actual is None else bool((tickets == actual).all(axis=1).any()),
            "duplicate_pairs": 0}, strata


def candidates(frame, p, tickets, scores, actual):
    anchor = p.argmax(axis=1)
    order = np.argsort(-p, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.tile(np.arange(1, 4), (13, 1)), axis=1)
    rows = []
    for j, ticket in enumerate(np.vstack([anchor, tickets])):
        ranks = rank[np.arange(13), ticket]
        rows.append({"candidate": f"{j:02d}", "reference_only": j == 0,
                     "symbols": "".join(LABELS[v] for v in ticket),
                     "D_count": int((ticket == 1).sum()),
                     "exchanges": int((ticket != anchor).sum()),
                     "rank2": int((ranks == 2).sum()), "rank3": int((ranks == 3).sum()),
                     "original_P13": float(np.prod(p[np.arange(13), ticket])),
                     "placement_product_not_probability": None if scores is None else float(np.prod(scores[np.arange(13), ticket])),
                     "actual_hits": None if actual is None else int((ticket == actual).sum())})
    return pd.DataFrame(rows)


def table(frame):
    return '<div class="scroll">' + frame.to_html(index=False, float_format=lambda v: f"{v:.7g}") + '</div>'


def page(title, body):
    return ('<!doctype html><html lang="ja"><meta charset="utf-8"><title>' + escape(title) + '</title>'
            '<style>body{font:16px/1.7 system-ui;margin:28px;color:#213047}.scroll{overflow:auto}'
            'table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{border:1px solid #ccd3dc;padding:6px}'
            'th{background:#edf3f7}code{background:#eee}</style><h1>' + escape(title) + '</h1>' + body + '</html>')


def run_round(rid, outdir, structure_mode="corrected"):
    current = rid == "toto1670"
    base = ROOT / ("data/purchase_reference" if current else "data/eval/toto_rounds/toto1664/snapshot")
    forecast_path = base / "next_predictions.csv"
    frame = pd.read_csv(forecast_path).sort_values("match_no").reset_index(drop=True)
    if frame.match_no.tolist() != list(range(1, 14)):
        raise ValueError("Invalid match order")
    if current and json.loads((base / "manifest.json").read_text())["round_id"] != rid:
        raise ValueError("Current forecast round mismatch")
    actual = None
    # Results are retained for reporting only, not passed to scoring or selection.
    if not current:
        result = pd.read_csv(base.parent / "actual_results.csv").sort_values("match_no")
        for c in ("match_no", "home_team", "away_team"):
            if frame[c].tolist() != result[c].tolist():
                raise ValueError("Results identity mismatch")
        raw = result.result.to_numpy(dtype=int)
        actual = np.where(raw == 1, 0, np.where(raw == 0, 1, 2))
    p = frame[PROB_COLUMNS].to_numpy(dtype=float)
    p_original = p.copy()
    feat = structure(frame, mode=structure_mode)
    support = feat.D_structure_support.to_numpy()
    experiments = {}
    for strength in STRENGTHS:
        scores = placement_scores(p, support, strength)
        tickets, _ = top_joint_candidates(scores, 10)
        experiments[f"structure_{strength:.1f}"] = (tickets, scores, strength)
    # Feature ablations at the same strength: no result-dependent parameter choice.
    for name, q in [("without_low_total", support/feat.low_total_le2.to_numpy()),
                    ("only_balance", feat.low_total_le2.to_numpy()*feat.full_win_balance.to_numpy())]:
        scores = placement_scores(p, q, 1.0)
        experiments[name] = (*top_joint_candidates(scores, 10)[:1], scores, 1.0)
    if current:
        for name, filename, start in [("saved_C_v4", "shadow_buyplan_c.csv", 31),
                                       ("saved_v25", "shadow_buyplan_v25.csv", 1)]:
            saved = pd.read_csv(base / filename).sort_values("match_no")
            for c in ("match_no", "home_team", "away_team"):
                if saved[c].tolist() != frame[c].tolist():
                    raise ValueError("Saved baseline identity mismatch")
            np.testing.assert_allclose(saved[PROB_COLUMNS].to_numpy(), p, atol=1e-12, rtol=0)
            experiments[name] = (stored_tickets(saved, start), None, None)
    else:
        old_comparison = json.loads((ROOT / "data/reports/toto1664_free_portfolio/comparison.json").read_text())
        if old_comparison["forecast_sha256"] != sha(forecast_path):
            raise ValueError("Historical comparison used a different prediction input")
        saved = pd.read_csv(ROOT / "data/reports/toto1664_free_portfolio/portfolios.csv")
        saved = saved[saved.portfolio.eq("free_coverage_80pct_first")].sort_values("match_no")
        for c in ("match_no", "home_team", "away_team"):
            if saved[c].tolist() != frame[c].tolist():
                raise ValueError("Saved baseline identity mismatch")
        experiments["saved_free_coverage"] = (stored_tickets(saved), None, None)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "forecast_input.csv", index=False)
    feature_table = pd.concat([frame[["match_no", "home_team", "away_team"]], feat], axis=1)
    feature_table.to_csv(outdir / "D_structure_features.csv", index=False)
    rows, allocation_rows = [], []
    for name, (tickets, scores, strength) in experiments.items():
        dest = outdir / name
        dest.mkdir(exist_ok=True)
        metric, strata = metrics(p, tickets, actual)
        metric.update(round=rid, method=name, strength=strength, structure_mode=structure_mode,
                      actual_D_count=None if actual is None else int((actual == 1).sum()),
                      actual_D_hit_slots=None if actual is None else int(((tickets == 1) & (actual == 1)).sum()),
                      D_outranks_alternative_HA=None if scores is None else int((scores[:, 1] > np.minimum(scores[:, 0], scores[:, 2])).sum()))
        rows.append(metric)
        pd.DataFrame(strata).to_csv(dest / "D_count_conditional_P13.csv", index=False)
        diag = candidates(frame, p, tickets, scores, actual)
        diag.to_csv(dest / "candidates.csv", index=False)
        allocation = frame[["match_no", "home_team", "away_team"]].copy()
        if actual is not None:
            allocation["actual"] = [LABELS[s] for s in actual]
        for s, label in enumerate(LABELS):
            allocation[f"P_{label}"] = p[:, s]
            allocation[f"count_{label}"] = (tickets == s).sum(axis=0)
            if scores is not None:
                allocation[f"placement_score_{label}"] = scores[:, s]
        for j, ticket in enumerate(tickets, 1):
            allocation[f"candidate{j:02d}"] = [LABELS[s] for s in ticket]
        allocation.insert(0, "method", name)
        allocation.to_csv(dest / "allocation.csv", index=False)
        allocation_rows.append(allocation)
        note = '<p>候補00は元確率最大の参照券。01〜10が比較用10口。全て購入対象外。</p>'
        note += '<p>original_P13は元モデルの確率。placement_product_not_probabilityは配券評価であり確率ではありません。</p>'
        note += '<p><a href="../index.html">開催比較へ</a></p>'
        body = note + '<h2>配券</h2>' + table(allocation) + '<h2>候補診断</h2>' + table(diag)
        body += '<h2>実際のD数を条件としたモデル上の13的中確率</h2>' + table(pd.DataFrame(strata))
        (dest / "index.html").write_text(page(rid + ' / ' + name, body))
    np.testing.assert_array_equal(p, p_original)
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "metrics.csv", index=False)
    pd.concat(allocation_rows, ignore_index=True).to_csv(outdir / "all_allocations.csv", index=False)
    display = summary.copy()
    display["P_any13"] = display.P_any13.map(lambda v: f"{v*100:.6f}%")
    links = '<ul>' + ''.join(f'<li><a href="{name}/index.html">{name}：候補・配券評価・D数別13的中</a></li>' for name in experiments) + '</ul>'
    body = '<p><a href="../index.html">実験概要へ</a></p>' + table(display) + links
    body += '<h2>試合別のD構造評価</h2>' + table(feature_table[["match_no", "home_team", "away_team", "lambda_total", "P_D", "low_total_le2", "balanced_one_goal_sides", "full_win_balance", "favoured_HA", "one_goal_given_H_win", "one_goal_given_A_win", "Q_legacy", "Q_near_only", "Q_corrected", "D_structure_support"]])
    (outdir / "index.html").write_text(page(rid + ' 非固定式D構造評価', body))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/reports/free_structural_draw_corrected")
    parser.add_argument("--structure-mode", choices=("legacy", "near_only", "corrected"), default="corrected")
    args = parser.parse_args()
    protected = [p for p in (ROOT / "data/purchase_reference").glob('*') if p.is_file()]
    protected += [ROOT / "scripts/next_model" / name for name in ["shadow_free_coverage.py", "score_placement_v25_shadow.py", "shadow_v25_operational.py"]]
    protected += [ROOT / "data/eval/toto_rounds/toto1664/snapshot/next_predictions.csv",
                  ROOT / "data/eval/toto_rounds/toto1664/actual_results.csv"]
    hashes = {str(p): sha(p) for p in protected}
    all_rows = []
    for rid in ("toto1670", "toto1664"):
        all_rows += run_round(rid, args.outdir / rid, args.structure_mode)
        print('[OK]', rid, flush=True)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise AssertionError(f"Protected file changed: {path}")
    combined = pd.DataFrame(all_rows)
    combined.to_csv(args.outdir / "metrics.csv", index=False)
    config = {"status": "offline experimental only", "strengths": STRENGTHS,
              "formula": "D_score=P_D+alpha*Q*max(max(P_H,P_A)-P_D,0); H/A scores unchanged",
              "structure_mode": args.structure_mode,
              "Q_corrected": "L*(B_full+(1-B_full)*N_favourite)",
              "N_favourite": "P(favourite wins by one)/P(favourite wins); not a temporal transition probability",
              "L": "Poisson(lambda_H+lambda_A) probability of total goals <= 2",
              "B_legacy": "2*min(P(H leads by 1),P(A leads by 1))/(P(H leads by 1)+P(A leads by 1))",
              "B_full": "2*min(P(H wins),P(A wins))/(P(H wins)+P(A wins))",
              "selection": "exact ten largest products of placement scores; unique tickets; same scores used throughout",
              "score_is_probability": False, "D_quotas": None, "exchange_targets": None,
              "randomness": False, "outcomes_used_for_selection": False,
              "provisional": "alpha and feature design are not learned; sensitivity analysis, no adoption",
              "timing": "Designed after user reported two draws in 1670; not a prospective validation",
              "protected_sha256": hashes}
    (args.outdir / "audit.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    display = combined[["round", "method", "D_total", "D_counts", "P_any13", "actual_D_count", "actual_D_hit_slots"]].copy()
    display["P_any13"] = display.P_any13.map(lambda v: f"{v*100:.6f}%")
    body = '<h2>比較結果</h2>' + table(display)
    current = combined[combined["round"].eq("toto1670")].set_index("method")
    historical = combined[combined["round"].eq("toto1664")].set_index("method")
    body += '<p>第1670回のD総枠数：' + '、'.join(
        f'α={a:g}で{int(current.loc[f"structure_{a:.1f}", "D_total"])}枠' for a in STRENGTHS) + '。</p>'
    body += '<p>第1664回の実際のDを当てた枠数：' + '、'.join(
        f'α={a:g}で{int(historical.loc[f"structure_{a:.1f}", "actual_D_hit_slots"])}枠' for a in STRENGTHS) + '。対象試合と候補番号は各設定の配券表に記録しています。</p>'
    body += '''<p>選択構造：積の上位10口は、各試合の配券スコアが最大となる組み合わせの周辺へ集中します。αQが1を超えるとDが基準H/Aより高く評価され、その試合のDが多数の券で共有されやすくなります。D枠の指定による現象ではありません。</p>
<p>非固定式にD構造評価を反映できても、強度に対する配置の安定性や実結果での有効性は別に確認する必要があります。正式採用・係数確定は行いません。</p>
<h2>試作仕様と検証の範囲</h2><p>購入対象外。予測H/D/A確率・正式版・現行シャドーは変更していません。</p>
<p>基準H/AからDへ変える配券上の損失を、低得点と双方の僅差構造で軽減します。D本数や試合別枚数は指定しません。</p>
<p><code>L=P(総得点≤2), B=2×min(P(H勝ち),P(A勝ち))÷両者の和, N=優勢側の1点差勝ち確率÷優勢側の勝ち確率</code></p>
<p>修正式は<code>Q=L×[B+(1−B)×N]</code>。拮抗と優勢側の1点差構造を別々の条件として評価します。結合は0〜1の評価値に収める暫定設計で、確率ではありません。得点分布上の近さであり、試合途中から同点へ動く確率を推定したものではありません。旧式はL×左右の1点差勝ちの釣り合い、near_onlyはL×Nです。</p>
<p><code>D配券評価=P(D)+α×Q×max(max(P(H),P(A))−P(D),0)</code>。H/A配券評価は元確率。α=0,0.5,1,1.5,2は暫定の感度設定。αQ&gt;1ならDへの交換損失は負になり、基準H/Aより優先できます。</p>
<p>同じ配券評価の積で候補生成から最終10口まで選択します。全組み合わせの上位10口を決定的に求め、10・11・12的中の報酬や類似度ペナルティは使用しません。元確率の13的中確率は選択後に別途計算します。</p>
<p>α=0は元モデルの13的中確率を最大化する比較基準です。この同じモデルでそれ以上の13的中確率は得られません。構造評価は予測情報の追加ではなく、配券上の優先付けです。</p>
<p>対照実験without_low_totalは低得点条件を1に置き換え、only_balanceは優勢側の1点差評価を外します。現在のCは11・12的中カバーも使う別方式なので、構造の効果は主にα=0との差で判定します。</p>
<p>第1670回は一部結果を聞いた後の設計で、事前検証ではありません。結果は選択に使用していません。第1664回は過去の保存入力による再検証で、独立した将来検証ではありません。</p>
<ul><li><a href="toto1670/index.html">第1670回：比較と全候補</a></li><li><a href="toto1664/index.html">第1664回：比較と全候補</a></li></ul>
<p><a href="metrics.csv">全設定の集計CSV</a> ／ <a href="audit.json">監査・固定した式・入力ハッシュ</a></p>'''
    (args.outdir / "index.html").write_text(page('非固定式：D構造評価のシミュレーション', body))
    print(args.outdir / 'index.html')


if __name__ == '__main__':
    main()
