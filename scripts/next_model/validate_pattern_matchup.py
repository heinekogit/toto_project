#!/usr/bin/env python3
"""Archived LAB attack/concession profiles: auditable rolling residual test.

Research only. Production forecasts and acquisition settings are not modified.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from common import ROOT, normalize_text
from expected_goals import score_distribution

OUT = ROOT / "data/reports/pattern_matchup_validation"
PATTERNS = ("set_piece", "cross", "through", "short_pass", "long_pass", "dribble", "rebound", "other")
LABELS = ("セットプレー", "クロス", "スルーパス", "30m未満のパス", "30m以上のパス", "ドリブル", "こぼれ球", "その他")
EXPECTED_HEADERS = ["チーム", "ＰＫ", "セットプレー直接", "セットプレーから", "クロスから",
                    "スルーパスから", "30m未満のパスから", "30m以上のパスから", "ドリブルから", "こぼれ球から", "その他"]
PRIORS = (5.0, 10.0, 20.0)
ALPHAS = (0.1, 1.0, 10.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_patterns(path: Path) -> tuple[pd.Timestamp, dict[str, np.ndarray], dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", text, re.S)
    if title is None or "2026/27" not in title.group(1):
        raise ValueError("wrong_season")
    date_match = re.search(r'<em class="uiDate">\s*(\d{4})\.(\d+)\.(\d+)', text)
    if date_match is None:
        raise ValueError("missing_asof")
    asof = pd.Timestamp(*map(int, date_match.groups()))
    arrays = re.findall(r"arrayToDataTable\(\s*(\[.*?\])\s*\)", text, re.S)
    if not arrays:
        raise ValueError("missing_pattern_chart")
    tables = [ast.literal_eval(block) for block in arrays]
    if any(table != tables[0] for table in tables[1:]):
        raise ValueError("count_and_percentage_charts_disagree")
    if tables[0][0] != EXPECTED_HEADERS:
        raise ValueError("unexpected_pattern_headers")
    result, total = {}, {}
    for row in tables[0][1:]:
        key = normalize_text(row[0])
        counts = np.asarray(row[1:], dtype=float)
        if len(counts) != 10 or np.any(counts < 0) or not np.all(counts == np.floor(counts)) or key in result:
            raise ValueError("invalid_pattern_counts")
        total[key] = int(counts.sum())
        # PK is excluded; direct and indirect set-piece goals are combined.
        result[key] = np.r_[counts[1] + counts[2], counts[3:]]
    return asof, result, total


def load_sources() -> tuple[list[dict], dict[str, pd.DataFrame], list[dict]]:
    sources, histories, audit = [], {}, []
    for league in ("j1", "j2"):
        history = pd.read_csv(ROOT / f"data/{league}_2026_latest_results.csv")
        history["datetime"] = pd.to_datetime(history.datetime)
        history = history.dropna(subset=["home_score", "away_score"]).copy()
        history["home_key"] = history.home_team.map(normalize_text)
        history["away_key"] = history.away_team.map(normalize_text)
        histories[league] = history
        for goal_path in sorted((ROOT / "external_contents").glob(f"2026*/{league}/得点パターン.html")):
            date = goal_path.parent.parent.name
            if date < "20260801":
                continue
            loss_path = goal_path.with_name("失点パターン.html")
            metrics_path = ROOT / f"data/external_metrics/football_lab_team_metrics_{date}_{league}.csv"
            if not loss_path.exists() or not metrics_path.exists():
                continue
            try:
                goal_asof, goals, totals_for = read_patterns(goal_path)
                loss_asof, losses, totals_against = read_patterns(loss_path)
            except ValueError as error:
                if str(error) == "wrong_season":
                    audit.append({"snapshot": date, "league": league, "status": "excluded_special_season"})
                    continue
                raise
            snapshot_date = pd.to_datetime(date, format="%Y%m%d")
            metrics = pd.read_csv(metrics_path)
            metrics["team_key"] = metrics.team_key.map(normalize_text)
            metrics_asof = pd.to_datetime(metrics.asof_date.astype(str), format="%Y%m%d")
            if goal_asof != loss_asof or not metrics_asof.eq(goal_asof).all() or goal_asof > snapshot_date:
                raise ValueError(f"{date}/{league}: inconsistent source dates")
            if set(goals) != set(losses) or set(goals) != set(metrics.team_key):
                raise ValueError(f"{date}/{league}: inconsistent team coverage")
            index_path = goal_path.parent.parent / "football_lab_fetch_index.csv"
            fetch = pd.read_csv(index_path)
            required = fetch[(fetch.league == league) & fetch.item_name.isin(["得点パターン", "失点パターン", "ゴール期待値"])]
            if len(required) != 3 or not required.status.eq("ok").all():
                raise ValueError(f"{date}/{league}: missing successful fetch records")
            known = history[history.datetime < goal_asof]
            for team in goals:
                gf = known.loc[known.home_key.eq(team), "home_score"].sum() + known.loc[known.away_key.eq(team), "away_score"].sum()
                ga = known.loc[known.home_key.eq(team), "away_score"].sum() + known.loc[known.away_key.eq(team), "home_score"].sum()
                if totals_for[team] != gf or totals_against[team] != ga:
                    raise ValueError(f"{date}/{league}/{team}: pattern totals do not match results before asof")
            pooled = np.sum(list(goals.values()), axis=0)
            if not np.array_equal(pooled, np.sum(list(losses.values()), axis=0)):
                raise ValueError("league scored/conceded pattern totals disagree")
            # One count per category ensures finite composition for early-season zeros.
            pool = (pooled + 1.0) / (pooled.sum() + len(PATTERNS))
            sources.append({"snapshot": date, "date": snapshot_date, "asof": goal_asof,
                            "league": league, "goals": goals, "losses": losses,
                            "pool": pool, "metrics": metrics.set_index("team_key")})
            audit.append({"snapshot": date, "league": league, "asof": str(goal_asof.date()),
                          "status": "ok", "teams": len(goals), "total_goals": sum(totals_for.values()),
                          "counts_match_results_before_asof": True,
                          "goal_html_sha256": digest(goal_path), "loss_html_sha256": digest(loss_path),
                          "metrics_sha256": digest(metrics_path)})
    return sources, histories, audit


def dataset(sources: list[dict], histories: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
    rows, exclusions = [], []
    for league, history in histories.items():
        for _, game in history.iterrows():
            day = game.datetime.normalize()
            eligible = [s for s in sources if s["league"] == league and s["date"] < day and s["asof"] < day]
            if not eligible:
                exclusions.append({"match_id": game.match_id, "reason": "no_prior_day_regular_season_snapshot"})
                continue
            source = max(eligible, key=lambda s: s["date"])
            for side in ("home", "away"):
                opponent_side = "away" if side == "home" else "home"
                team, opponent = game[f"{side}_key"], game[f"{opponent_side}_key"]
                own, opp = source["metrics"].loc[team], source["metrics"].loc[opponent]
                baseline = math.sqrt(float(own.flab_expected_for_xg) * float(opp.flab_expected_against_xg))
                if not math.isfinite(baseline) or baseline <= 0:
                    raise ValueError("invalid archived xG")
                row = {"match_id": game.match_id, "datetime": game.datetime, "league": league,
                       "week": game.datetime.to_period("W-SUN").start_time, "side": side,
                       "group": (0 if league == "j1" else 2) + (side == "away"),
                       "home_team": game.home_key, "away_team": game.away_key,
                       "y": float(game[f"{side}_score"]), "base_lambda": baseline,
                       "snapshot": source["snapshot"], "asof": str(source["asof"].date()),
                       "snapshot_age_days": (day - source["date"]).days}
                for j, name in enumerate(PATTERNS):
                    row[f"for_{name}"] = source["goals"][team][j]
                    row[f"against_{name}"] = source["losses"][opponent][j]
                    row[f"pool_{name}"] = source["pool"][j]
                rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["datetime", "match_id", "side"]).reset_index(drop=True)
    if frame.duplicated(["match_id", "side"]).any():
        raise ValueError("duplicate match sides")
    return frame, exclusions


def affinity(frame: pd.DataFrame, prior: float) -> np.ndarray:
    f = frame[[f"for_{n}" for n in PATTERNS]].to_numpy(float)
    a = frame[[f"against_{n}" for n in PATTERNS]].to_numpy(float)
    p = frame[[f"pool_{n}" for n in PATTERNS]].to_numpy(float)
    qf = (f + prior * p) / (f.sum(axis=1, keepdims=True) + prior)
    qa = (a + prior * p) / (a.sum(axis=1, keepdims=True) + prior)
    return np.log(np.sum(qf * qa / p, axis=1))


def fit(train: pd.DataFrame, prior: float | None = None, alpha: float = 0.0) -> dict:
    group = train.group.to_numpy(int)
    onehot = np.eye(4)[group]
    y, baseline = train.y.to_numpy(), train.base_lambda.to_numpy()
    initial = np.array([math.log((y[group == i].sum() + 1e-8) / baseline[group == i].sum()) for i in range(4)])
    if prior is None:
        return {"coef": initial, "prior": None, "mean": 0.0, "std": 1.0}
    raw = affinity(train, prior)
    mean, std = raw.mean(), max(raw.std(), 1e-8)
    x = np.column_stack([onehot, (raw - mean) / std])
    offset = np.log(baseline)

    def objective(beta: np.ndarray):
        eta = offset + x @ beta
        rate = np.exp(eta)
        loss = np.mean(rate - y * eta) + alpha * beta[-1] ** 2 / 2
        grad = x.T @ (rate - y) / len(y)
        grad[-1] += alpha * beta[-1]
        return loss, grad

    result = minimize(objective, np.r_[initial, 0.0], method="L-BFGS-B", jac=True,
                      bounds=[(-3, 3)] * 4 + [(-2, 2)])
    if not result.success:
        raise ValueError(result.message)
    return {"coef": result.x, "prior": prior, "mean": mean, "std": std}


def fit_league_only(train: pd.DataFrame) -> dict:
    coef = []
    for league in ("j1", "j2"):
        group = train[train.league == league]
        coefficient = math.log(group.y.sum() / group.base_lambda.sum())
        coef.extend([coefficient, coefficient])
    return {"coef": np.asarray(coef), "prior": None, "mean": 0.0, "std": 1.0}


def predict(frame: pd.DataFrame, model: dict) -> np.ndarray:
    eta = model["coef"][frame.group.to_numpy(int)]
    if model["prior"] is not None:
        eta = eta + model["coef"][-1] * (affinity(frame, model["prior"]) - model["mean"]) / model["std"]
    return frame.base_lambda.to_numpy(float) * np.exp(eta)


def poisson_nll(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return mu - y * np.log(mu) + gammaln(y + 1)


def tune(train: pd.DataFrame) -> tuple[float, float, list[dict]]:
    folds = []
    for week in sorted(train.week.unique()):
        past, future = train[train.week < week], train[train.week == week]
        if len(past) >= 60 and past.groupby("group").size().min() >= 10:
            folds.append((past, future))
    if not folds:
        raise ValueError("no inner time folds")
    scores = []
    # Include the zero-coefficient model so tuning can reject the extra feature.
    for prior, alpha in [(None, 0.0)] + [(p, a) for p in PRIORS for a in ALPHAS]:
        losses = []
        for past, future in folds:
            mu = predict(future, fit(past, prior, alpha))
            losses.extend(poisson_nll(future.y.to_numpy(), mu))
        scores.append({"prior": prior, "alpha": alpha, "goal_nll": float(np.mean(losses))})
    best = min(scores, key=lambda s: s["goal_nll"])
    return best["prior"], best["alpha"], scores


def summarize(detail: pd.DataFrame) -> dict:
    return {"matches": len(detail), **{name: float(detail[name].mean()) for name in
            ("goal_mae", "goal_nll", "log_loss", "brier", "rank1_hit")}}


def evaluate(test: pd.DataFrame, mu: np.ndarray, name: str) -> list[dict]:
    work = test.copy()
    work["mu"] = mu
    records = []
    for _, pair in work.groupby("match_id", sort=False):
        h, a = pair.set_index("side").loc["home"], pair.set_index("side").loc["away"]
        result = 0 if h.y > a.y else 1 if h.y == a.y else 2
        probabilities = score_distribution(h.mu, a.mu, max_goals=40)
        p = np.array([probabilities[f"prob_{s}"] for s in ("home", "draw", "away")])
        if probabilities["grid_mass_before_normalization"] < 1 - 1e-9:
            raise ValueError("insufficient score grid")
        records.append({"model": name, "match_id": h.match_id, "datetime": str(h.datetime),
            "week": str(h.week.date()), "league": h.league, "snapshot": h.snapshot,
            "asof": h["asof"], "home_team": h.home_team, "away_team": h.away_team,
            "lambda_home": h.mu, "lambda_away": a.mu, "actual_home": h.y, "actual_away": a.y,
            **{f"prob_{s}": float(v) for s, v in zip(("home", "draw", "away"), p)},
            "goal_mae": (abs(h.y - h.mu) + abs(a.y - a.mu)) / 2,
            "goal_nll": float(poisson_nll(np.array([h.y, a.y]), np.array([h.mu, a.mu])).mean()),
            "log_loss": -math.log(p[result]), "brier": float(np.sum((p - np.eye(3)[result]) ** 2)),
            "rank1_hit": int(np.argmax(p) == result)})
    return records


def write_report(data: pd.DataFrame, result: pd.DataFrame, summary: dict) -> None:
    labels = {"lab_xg": "LAB xGのみ", "league_calibrated_xg": "＋リーグ平均補正",
              "calibrated_xg": "＋リーグ・ホーム/アウェイ補正（比較基準）",
              "pattern_candidate": "比較基準＋パターン相性を必ず追加",
              "pattern_matchup": "過去検証で相性の追加可否を選択"}

    def metrics_table(league=None):
        rows = []
        for name, label in labels.items():
            group = result[result.model == name]
            if league:
                group = group[group.league == league]
            scores = summarize(group)
            rows.append({"方式": label, "試合数": len(group), "得点MAE": f"{scores['goal_mae']:.4f}",
                         "H/D/A Log Loss": f"{scores['log_loss']:.4f}", "Brier": f"{scores['brier']:.4f}",
                         "確率1位の的中率": f"{scores['rank1_hit']:.1%}"})
        return pd.DataFrame(rows).to_html(index=False, escape=True, border=0)

    match = result[(result.home_team == "横浜FM") & (result.away_team == "川崎F")]
    example = []
    for _, row in match.iterrows():
        example.append({"方式": labels[row.model], "得点期待値 横浜FM–川崎F": f"{row.lambda_home:.3f}–{row.lambda_away:.3f}",
                        "H / D / A": f"{row.prob_home:.1%} / {row.prob_draw:.1%} / {row.prob_away:.1%}"})
    inputs = data[(data.home_team == "横浜FM") & (data.away_team == "川崎F")].set_index("side")
    patterns = []
    for name, label in zip(PATTERNS, LABELS):
        patterns.append({"パターン（PKを除く）": label,
                         "横浜FMの得点": int(inputs.loc["home", f"for_{name}"]),
                         "川崎Fの失点": int(inputs.loc["home", f"against_{name}"]),
                         "川崎Fの得点": int(inputs.loc["away", f"for_{name}"]),
                         "横浜FMの失点": int(inputs.loc["away", f"against_{name}"])})
    unused = sum(f["prior"] is None for f in summary["folds"])
    interval = summary["candidate_log_loss_delta_week_bootstrap_95"]
    coefficients = pd.DataFrame([{ "評価開始週": f["week"], "学習試合": f["train_matches"],
        "評価試合": f["test_matches"], "過去検証の選択": "追加しない" if f["prior"] is None else "相性を追加",
        "追加必須案の係数": f"{f['candidate_coefficient_per_sd']:.5f}"} for f in summary["folds"]])
    body = f"""<p><a href='../index.html'>レポート一覧</a></p>
    <h1>攻撃と相手守備の組み合わせを検証</h1>
    <p>2026年11月3日 / 保存済みFootball LAB情報による追加検証</p>
    <div class='notice'><b>結論：</b>得点・失点パターンは取得済みHTMLにあるが、従来の数値抽出では未利用だった。
    これを対応させた相性補正を検証したところ、過去の評価だけで選ぶ手順は{len(summary['folds'])}期間中{unused}期間で「追加しない」を選んだ。
    攻撃スタイルが無効という結論ではなく、今回のパターン構成による補正には改善の証拠がない。</div>
    <h2>取得情報で分かること・足りないこと</h2>
    <table><tr><th>情報</th><th>現状</th><th>意味と限界</th></tr>
    <tr><td>7種類の攻撃スタイル指数・シュート率</td><td>取得・数値化済み</td><td>攻撃の試行傾向。指数の大小は優劣を意味しない。</td></tr>
    <tr><td>得点パターン・失点パターン</td><td>HTML取得済み。今回、検証用に数値を抽出</td><td>同じ種類の攻撃で得点した数と失点した数を対応可能。ただし被攻撃回数がなく、守備成功率ではない。</td></tr>
    <tr><td>被xG・被シュート・守備CBP等</td><td>取得・数値化済み</td><td>守備全体の情報。カウンターやクロスへの耐性を直接示す値とは限らない。</td></tr>
    <tr><td>ハイ／ミドルプレス、カウンタープレス、守備ブロック</td><td>取得対象外</td><td>カウンター対ハイライン、保持型対ローブロックの検証には不足。</td></tr></table>
    <p>定義：<a href='https://www.football-lab.jp/pages/team_style'>Football LAB チームスタイル指標とは</a>、
    <a href='https://www.football-lab.jp/summary/team_ranking/j1?year=2026&amp;data=lost'>失点パターン</a>。</p>
    <h2>何を、どう比べたか</h2>
    <p>J1・J2の2026/27シーズン。{summary['snapshot_count']}組の保存情報から{summary['dataset_matches']}試合の入力を作成した。
    先頭60試合を初期学習に使い、その後190試合（J1 90、J2 100）を週単位で順番に評価。
    各週の係数・縮約量・追加可否は、その週より前の試合だけで決めた。各試合の入力と正規化に、その試合以後の結果を使っていない。</p>
    <p>試合当日以後に保存されたHTMLは使わない。特別大会の旧シーズンは除外。
    各保存ページの得点・失点合計が、その更新日以前の実スコア合計と全チームで一致することを検査した。
    事前情報がなかった開幕20試合は除外。原本のハッシュと採用日付を保存した。</p>
    <p><b>比較対象は検証用のLAB xGモデル。</b>λ＝√（自チームのLAB xG × 相手のLAB被xG）。
    現行フル予測の欠場・疲労・天候・新しい得失点補正は再現していない。
    正式予測やBuyPlanの成績改善を直接示す検証ではない。</p>
    <details><summary>相性の式と係数の決め方</summary>
    <p>PKを除き、セットプレー・クロス・スルーパス等8分類の得点構成と相手の失点構成を使用。
    小標本の偏りを抑えるため、各構成を同じ時点のリーグ構成へ縮約する。
    相性＝log［Σ（自チームの得点割合 × 相手の失点割合 ÷ リーグの割合）］。
    片方がリーグ平均と同じ構成なら相性は0になるため、一方の数値を単に足す式とは異なる。</p>
    <p>平均得点の補正係数は相性1個。別にリーグ・ホーム／アウェイの基準補正4個を両比較に共通で持たせる。
    Poisson回帰で基準λの残差を説明する。縮約量5・10・20、正則化0.1・1・10と「追加なし」を学習期間内の時系列検証で比較。
    評価基準は得点分布の負の対数尤度。今回の190試合は既知の過去結果による探索的検証で、今後の新規試合での実証ではない。</p></details>
    <h2>評価結果</h2><p>MAE・Log Loss・Brierは低いほど良い。</p>{metrics_table()}
    <p>追加必須案のLog Loss差は比較基準に対して
    {summary['summaries']['pattern_candidate']['log_loss']-summary['summaries']['calibrated_xg']['log_loss']:+.6f}。
    週単位再抽出の参考95%区間は[{interval[0]:+.6f}, {interval[1]:+.6f}]で0を含む。9期間だけなので区間も参考値である。</p>
    <details><summary>J1とJ2を分けた結果</summary><h3>J1</h3>{metrics_table('j1')}<h3>J2</h3>{metrics_table('j2')}</details>
    <details><summary>各週で追加を選んだか</summary>{coefficients.to_html(index=False, escape=True, border=0)}</details>
    <h2>No.10 横浜FM－川崎F</h2>
    <p>実スコア4–2。使用情報は10月31日保存・10月26日更新。試合は11月1日。</p>
    {pd.DataFrame(example).to_html(index=False, escape=True, border=0)}
    <p>この検証用モデルではホーム／アウェイを分けた基準補正でH優勢に変わり、パターン相性を加えた変化は小さい。
    Hへの変化をスタイルの効果として扱ってはいけない。リーグ別の検証結果も異なるため、ホーム補正も正式採用の根拠としては未確定。</p>
    <details><summary>相性計算の元になった得点・失点パターン</summary>
    {pd.DataFrame(patterns).to_html(index=False, escape=True, border=0)}
    <p>得点済み・失点済みの件数であり、クロスを何回受けて何回防いだかは分からない。
    例えばクロス失点の割合だけでは、クロスに弱いのか、クロスを多く受けたのかを分離できない。</p></details>
    <h2>ばらつきと次の判断</h2>
    <p>比較基準の予測誤差から測ったPearson分散比は{summary['out_of_time_pearson_dispersion']:.3f}（Poissonの目安1）。
    全試合の分布を一律に広げる根拠は、この集計からは得られない。特定の相性だけ分布が広がるか、得点が連動するかは未検証。</p>
    <p>守備位置・プレスの指標と、攻撃種類別の被試行数・被シュート率が追加検証に必要。
    保存済みの攻撃指数だけで「相手の弱点」を作ったことにはしない。今回の相性補正は正式予測へ組み込まない。</p>
    <p><a href='inputs.csv'>試合前入力</a> / <a href='matches.csv'>全比較結果</a> / <a href='source_audit.csv'>取得情報の監査</a> /
    <a href='summary.json'>集計</a> / <a href='tuning.json'>学習期間内の係数選択</a> /
    <a href='../../../scripts/next_model/validate_pattern_matchup.py'>再現スクリプト</a></p>"""
    document = """<!doctype html><html lang='ja'><head><meta charset='utf-8'>
    <meta name='viewport' content='width=device-width,initial-scale=1'><title>攻撃と相手守備の組み合わせ検証</title>
    <style>body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#172033;line-height:1.8;margin:0}
    main{max-width:1150px;margin:auto;padding:24px;background:white}h1{font-size:26px}h2{margin-top:32px;font-size:21px}
    table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{padding:9px;border:1px solid #dce2e8;text-align:left}
    th{background:#edf3f8}.notice{background:#fff4d8;border-left:4px solid #bd8700;padding:16px}a{color:#166298}
    details{margin:15px 0;padding:12px;background:#f6f8fa;overflow-x:auto}summary{cursor:pointer;font-weight:600}
    @media(max-width:650px){main{padding:12px}table{font-size:12px}th,td{padding:5px}}</style></head><body><main>"""
    (OUT / "report.html").write_text(document + body + "</main></body></html>", encoding="utf-8")


def main() -> None:
    sources, histories, audit = load_sources()
    data, exclusions = dataset(sources, histories)
    details, folds, tuned = [], [], []
    for week in sorted(data.week.unique()):
        train, test = data[data.week < week], data[data.week == week]
        if len(train) < 120:
            continue
        prior, alpha, cv = tune(train)
        calibrated, matched = fit(train), fit(train, prior, alpha)
        candidate = min((s for s in cv if s["prior"] is not None), key=lambda s: s["goal_nll"])
        forced = fit(train, candidate["prior"], candidate["alpha"])
        for name, mu in (("lab_xg", test.base_lambda.to_numpy()),
                         ("league_calibrated_xg", predict(test, fit_league_only(train))),
                         ("calibrated_xg", predict(test, calibrated)),
                         ("pattern_candidate", predict(test, forced)),
                         ("pattern_matchup", predict(test, matched))):
            details.extend(evaluate(test, mu, name))
        folds.append({"week": str(pd.Timestamp(week).date()), "train_matches": len(train) // 2,
                      "test_matches": len(test) // 2, "prior": prior, "alpha": alpha,
                      "coefficient_per_sd": float(matched["coef"][-1]) if prior is not None else 0.0,
                      "candidate_prior": candidate["prior"], "candidate_alpha": candidate["alpha"],
                      "candidate_coefficient_per_sd": float(forced["coef"][-1]),
                      "calibration_multipliers_j1H_j1A_j2H_j2A": np.exp(calibrated["coef"]).tolist(),
                      "train_last_kickoff": str(train.datetime.max()), "test_first_kickoff": str(test.datetime.min())})
        tuned.append({"week": str(pd.Timestamp(week).date()), "candidates": cv})
    result = pd.DataFrame(details)
    summaries = {name: summarize(g) for name, g in result.groupby("model")}
    by_league = {f"{name}/{league}": summarize(g) for (name, league), g in result.groupby(["model", "league"])}
    comparison = result[result.model == "pattern_candidate"].set_index("match_id")
    reference = result[result.model == "calibrated_xg"].set_index("match_id").loc[comparison.index]
    changes = pd.DataFrame({"week": comparison.week,
                            "log_loss_delta": comparison.log_loss - reference.log_loss,
                            "goal_mae_delta": comparison.goal_mae - reference.goal_mae})
    # Resample whole weeks to acknowledge matches sharing teams/source snapshots.
    blocks = [g.log_loss_delta.to_numpy() for _, g in changes.groupby("week")]
    rng = np.random.default_rng(20261103)
    boot = [float(np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]).mean()) for _ in range(2000)]
    summary = {"dataset_matches": len(data) // 2, "excluded_matches": len(exclusions),
        "snapshot_count": len(sources), "summaries": summaries, "by_league": by_league,
        "folds": folds, "candidate_log_loss_delta_week_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "source_result_hashes": {lg: digest(ROOT / f"data/{lg}_2026_latest_results.csv") for lg in histories},
        "scope": "retrospective exploratory LAB-only study; same-day archives excluded; no full next_prob replay"}
    base_records = result[result.model == "calibrated_xg"]
    mu = np.r_[base_records.lambda_home, base_records.lambda_away]
    y = np.r_[base_records.actual_home, base_records.actual_away]
    summary["out_of_time_pearson_dispersion"] = float(np.mean((y - mu) ** 2 / mu))
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "inputs.csv", index=False)
    result.to_csv(OUT / "matches.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "source_audit.csv", index=False)
    pd.DataFrame(exclusions).to_csv(OUT / "excluded.csv", index=False)
    for filename, content in (("summary.json", summary), ("tuning.json", tuned)):
        (OUT / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(data, result, summary)
    print(json.dumps({"dataset_matches": len(data)//2, "summaries": summaries,
                     "candidate_delta_ci": summary["candidate_log_loss_delta_week_bootstrap_95"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
