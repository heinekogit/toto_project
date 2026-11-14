"""Descriptive and retrospective temporal checks of Q, without deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit, logit
from scipy.stats import binomtest

from common import ROOT
from simulate_free_structural_draw import structure, sha, page, table

ROUNDS = ("toto1660", "toto1663", "toto1664", "toto1665", "toto1668")
OUT = ROOT / "data/reports/structural_draw_q_audit"
CALIPER = .01


def load_data(structure_mode="legacy"):
    frames, provenance = [], []
    for rid in ROUNDS:
        base = ROOT / "data/eval/toto_rounds" / rid
        forecast = base / "snapshot/next_predictions.csv"
        generated = base / "snapshot/buyplan_engine_manifest.json"
        archive = base / "snapshot/manifest.json"
        actual_path = base / "actual_results.csv"
        manifest = json.loads(generated.read_text())
        digest = sha(forecast)
        if manifest["output_sha256"]["next_predictions.csv"] != digest:
            raise ValueError(f"{rid}: forecast checksum mismatch")
        f = pd.read_csv(forecast).sort_values("match_no").reset_index(drop=True)
        result = pd.read_csv(actual_path).sort_values("match_no").reset_index(drop=True)
        for c in ("match_no", "home_team", "away_team"):
            if f[c].tolist() != result[c].tolist():
                raise ValueError("Result identity mismatch")
        kickoff = pd.to_datetime(f.datetime).dt.tz_localize("Asia/Tokyo")
        before = kickoff > pd.Timestamp(manifest["created_at"])
        if not before.all():
            raise ValueError("Non-prematch forecast must be excluded")
        # Explicit mode keeps old and corrected Q audits distinct.
        features = structure(f, mode=structure_mode)
        frame = f[["match_no", "match_id", "datetime", "home_team", "away_team"]].copy()
        frame["round"] = rid
        frame["P_D"] = f.next_prob_draw
        frame["Q"] = features.D_structure_support
        frame["low_total"] = features.low_total_le2
        frame["legacy_onegoal_balance"] = features.balanced_one_goal_sides
        frame["full_win_balance"] = features.full_win_balance
        frame["favoured_one_goal_share"] = features.favoured_one_goal_share
        frame["actual_D"] = result.result.eq(0).astype(int)
        frames.append(frame)
        provenance.append({"round": rid, "forecast_sha256": digest,
                           "actual_sha256": sha(actual_path), "manifest_sha256": sha(generated),
                           "generator_created_at": manifest["created_at"],
                           "archive_created_at": json.loads(archive.read_text())["created_at"],
                           "matches": len(frame), "prematch_matches": int(before.sum())})
    data = pd.concat(frames, ignore_index=True)
    if data.match_id.duplicated().any():
        raise ValueError("Duplicate fixture")
    return data, provenance


def wilson(successes, n):
    z = 1.95996398454
    rate = successes / n
    center = (rate + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half


def grouped(frame, group_column):
    rows = []
    for group, f in frame.groupby(group_column, sort=True):
        lo, hi = wilson(int(f.actual_D.sum()), len(f))
        rows.append({"group": str(group), "matches": len(f), "D_count": int(f.actual_D.sum()),
                     "actual_D_rate": float(f.actual_D.mean()), "mean_P_D": float(f.P_D.mean()),
                     "mean_Q": float(f.Q.mean()), "mean_D_residual": float((f.actual_D-f.P_D).mean()),
                     "Wilson95_low": lo, "Wilson95_high": hi})
    return pd.DataFrame(rows)


def matched_pairs(data, within_round):
    high = data[data.Q_high].reset_index(drop=True)
    low = data[~data.Q_high].reset_index(drop=True)
    differences = abs(high.P_D.to_numpy()[:, None] - low.P_D.to_numpy()[None, :])
    allowed = differences <= CALIPER
    if within_round:
        allowed &= high["round"].to_numpy()[:, None] == low["round"].to_numpy()[None, :]
    # Dummy columns permit unmatched high-Q fixtures. No outcome enters costs.
    cost = np.c_[np.where(allowed, differences, 1e6), np.ones((len(high), len(high)))]
    ii, jj = linear_sum_assignment(cost)
    rows = []
    for i, j in zip(ii, jj):
        if j >= len(low):
            continue
        h, l = high.iloc[i], low.iloc[j]
        rows.append({**{f"high_{c}": h[c] for c in ("round", "match_no", "P_D", "Q", "actual_D")},
                     **{f"low_{c}": l[c] for c in ("round", "match_no", "P_D", "Q", "actual_D")},
                     "P_D_gap": abs(float(h.P_D-l.P_D))})
    pairs = pd.DataFrame(rows)
    h, l = pairs.high_actual_D.to_numpy(), pairs.low_actual_D.to_numpy()
    positive, negative = int(((h == 1) & (l == 0)).sum()), int(((h == 0) & (l == 1)).sum())
    summary = {"within_round": within_round, "pairs": len(pairs), "caliper": CALIPER,
               "high_D_count": int(h.sum()), "low_D_count": int(l.sum()),
               "high_D_rate": float(h.mean()), "low_D_rate": float(l.mean()),
               "high_mean_P_D": float(pairs.high_P_D.mean()), "low_mean_P_D": float(pairs.low_P_D.mean()),
               "high_mean_Q": float(pairs.high_Q.mean()), "low_mean_Q": float(pairs.low_Q.mean()),
               "high_only_D_pairs": positive, "low_only_D_pairs": negative,
               "paired_binomial_p_exploratory": binomtest(positive, positive+negative).pvalue if positive+negative else 1.0}
    return pairs, summary


def fit_probe(train, test, include_q):
    # Fixed L2=1, not tuned against evaluation outcomes. Penalize slopes only.
    def inputs(f):
        x = logit(f.P_D.to_numpy())[:, None]
        return np.c_[x, f.Q.to_numpy()] if include_q else x
    x, xt = inputs(train), inputs(test)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    x, xt = np.c_[np.ones(len(x)), (x-center)/scale], np.c_[np.ones(len(xt)), (xt-center)/scale]
    y = train.actual_D.to_numpy()
    def objective(beta):
        z = x @ beta
        loss = np.sum(np.logaddexp(0, z) - y*z) + .5*np.sum(beta[1:]**2)
        gradient = x.T @ (expit(z)-y) + np.r_[0., beta[1:]]
        return loss, gradient
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="BFGS", options={"gtol": 1e-8})
    if not result.success and np.linalg.norm(result.jac) > 1e-5:
        raise ValueError(result.message)
    return expit(xt @ result.x), result.x.tolist()


def predictive_metrics(y, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return {"binary_Brier": float(np.mean((p-y)**2)),
            "binary_LogLoss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure-mode', choices=('legacy', 'near_only', 'corrected'), default='legacy')
    parser.add_argument('--outdir', type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or (OUT if args.structure_mode == 'legacy' else ROOT / f'data/reports/structural_draw_q_audit_{args.structure_mode}')
    data, provenance = load_data(args.structure_mode)
    outdir.mkdir(parents=True, exist_ok=True)
    median = float(data.Q.median())
    data["Q_high"] = data.Q >= median
    data.to_csv(outdir / "matches.csv", index=False)
    groups = grouped(data, "Q_high")
    groups.to_csv(outdir / "Q_groups.csv", index=False)
    grouped(data, "round").to_csv(outdir / "rounds.csv", index=False)
    pair_summaries = []
    for within in (False, True):
        pairs, summary = matched_pairs(data, within)
        pairs.to_csv(outdir / f"pairs_within_round_{within}.csv", index=False)
        pair_summaries.append(summary)
    pd.DataFrame(pair_summaries).to_csv(outdir / "matched_summary.csv", index=False)
    temporal, fits = [], []
    for position in (3, 4):
        train = data[data["round"].isin(ROUNDS[:position])]
        test = data[data["round"].eq(ROUNDS[position])]
        probe_base, beta_base = fit_probe(train, test, False)
        probe_q, beta_q = fit_probe(train, test, True)
        for (_, r), pb, pq in zip(test.iterrows(), probe_base, probe_q):
            temporal.append({"round": r["round"], "match_no": r.match_no, "actual_D": r.actual_D,
                             "raw_P_D": r.P_D, "calibrated_P_D_only": pb, "P_D_plus_Q_probe": pq})
        fits.append({"test_round": ROUNDS[position], "train_rounds": ROUNDS[:position],
                     "train_matches": len(train), "base_beta": beta_base, "Q_beta": beta_q})
    temporal = pd.DataFrame(temporal)
    temporal.to_csv(outdir / "temporal_predictions.csv", index=False)
    temporal_summary = []
    for name in ("raw_P_D", "calibrated_P_D_only", "P_D_plus_Q_probe"):
        temporal_summary.append({"model": name, "test_matches": len(temporal),
                                 **predictive_metrics(temporal.actual_D.to_numpy(), temporal[name].to_numpy())})
    pd.DataFrame(temporal_summary).to_csv(outdir / "temporal_summary.csv", index=False)
    definition = {'legacy': 'Q=L*B_onegoal', 'near_only': 'Q=L*N_favourite', 'corrected': 'Q=L*(B_full+(1-B_full)*N_favourite)'}[args.structure_mode]
    config = {"Q_definition": definition, "structure_mode": args.structure_mode,
              "median_Q_threshold_descriptive": median,
              "pearson_Q_P_D": float(data.Q.corr(data.P_D)),
              "match_caliper_P_D": CALIPER, "source_manifest": provenance,
              "temporal_fits": fits, "L2_penalty": 1.0, "fitted_models_deployed": False,
              "scope": "descriptive retrospective audit; Q was designed after historical results were known",
              "limitations": ["65 matches, five rounds; small and correlated sample",
                              "no independent prospective holdout", "uncertainty formulas are descriptive",
                              "temporal probe standardization uses training data only",
                              "paired comparisons do not establish causality or calibration"]}
    (outdir / "audit.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    body = '<p>予測確率と購入券は変更していません。Qの高低によるD実発生率と、元D確率を考慮した追加的な有効性を調べます。</p>'
    body += f'<p>検証する式：{definition}。Lは総得点2点以下の確率。B_onegoalは左右の1点差勝ちの釣り合い。B_fullは勝敗分布全体の釣り合い。N_favouriteは優勢側が勝つ場合の1点差割合です。選択した定義を全開催に同じ条件で適用します。</p>'
    body += '<p>対象は生成時刻が試合前で、生成記録と保存CSVのハッシュが一致した5開催65試合。Q設計後の事後検証であり、独立した将来検証ではありません。</p>'
    body += f'<p>Qと元D確率の相関：{config["pearson_Q_P_D"]:.4f}。Q高低の境界は全65試合のQ中央値 {median:.6f}（結果を使わず設定）。</p>'
    body += '<h2>Q高低の単純比較</h2>' + table(groups)
    body += '<h2>元D確率が近い試合の比較</h2><p>D確率差1ポイント以内で、Q高群と低群を重複なしで対応。結果を使わず、組数を最大化し確率差を最小化。同一開催内で対応する厳しい比較も併記。</p>' + table(pd.DataFrame(pair_summaries))
    body += '<p>paired_binomial_pはD結果が異なるペアの偏りに関する探索的指標です。少数・同じチームの繰り返しを含むため、単独で採用判断しません。</p>'
    body += '<h2>後続開催への追加予測力（事後の時系列検査）</h2><p>最初の3開催から1665回、最初の4開催から1668回を予測。D確率だけで補正する小さなモデルとQを追加するモデルを、固定L2=1で比較。尺度は学習期間だけから作成。Brier/LogLossは低い方がよい。試験用の出力は正式予測へ反映しません。</p>' + table(pd.DataFrame(temporal_summary))
    body += '<p>同程度のD確率で比較できる組数は少なく、追加的な有効性を確定する材料には不足しています。高低比較と時系列検査の両方を確認し、この結果だけで強度を設定・採用しません。</p>'
    body += '<h2>生成時刻の確認と訂正</h2><p>以前の比較では評価フォルダへコピーした時刻を予測生成時刻と取り違え、事後保存と説明していました。生成記録とハッシュを確認したところ、全65試合が試合前生成でした。今回の比較はこれを訂正しています。</p>' + table(pd.DataFrame(provenance))
    body += '<p><a href="matches.csv">全試合とQ</a> ／ <a href="matched_summary.csv">対応比較</a> ／ <a href="temporal_summary.csv">時系列検査</a> ／ <a href="audit.json">検査条件と履歴</a></p>'
    (outdir / "index.html").write_text(page('D構造評価Q：実結果との検証 / ' + args.structure_mode, body))
    print(groups.to_string(index=False))
    print(pd.DataFrame(pair_summaries).to_string(index=False))
    print(pd.DataFrame(temporal_summary).to_string(index=False))
    print(outdir / 'index.html')


if __name__ == '__main__':
    main()
