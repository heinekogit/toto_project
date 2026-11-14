#!/usr/bin/env python3
"""Human-readable trace of a frozen next-model forecast and actual scores."""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def number(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if pd.notna(value) else math.nan


def fmt(value: float, digits: int = 2) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def average_xg(row: pd.Series, side: str) -> float:
    total = number(row, f"stats_ゴール期待値_{side}")
    against_total = number(row, f"stats_被ゴール期待値_{side}")
    against_average = number(row, f"stats_1試合平均被ゴール期待値_{side}")
    if not all(map(math.isfinite, (total, against_total, against_average))) or against_average <= 0:
        return math.nan
    games = round(against_total / against_average)
    return total / games if games > 0 else math.nan


def likely_score_for_result(home_lambda: float, away_lambda: float, result: str) -> tuple[str, float]:
    """Most likely exact score among scorelines matching the leading H/D/A sign."""
    if not all(map(math.isfinite, (home_lambda, away_lambda))):
        return "—", math.nan
    home_prob = [math.exp(-home_lambda) * home_lambda ** n / math.factorial(n) for n in range(11)]
    away_prob = [math.exp(-away_lambda) * away_lambda ** n / math.factorial(n) for n in range(11)]
    options = ((h, a) for h in range(11) for a in range(11)
               if ("H" if h > a else "A" if h < a else "D") == result)
    h, a = max(options, key=lambda score: home_prob[score[0]] * away_prob[score[1]])
    return f"{h}–{a}", home_prob[h] * away_prob[a]


def build(round_id: str) -> str:
    directory = ROOT / "data/eval/toto_rounds" / round_id
    forecast = pd.read_csv(directory / "snapshot/next_predictions.csv", encoding="utf-8-sig", low_memory=False)
    actual = pd.read_csv(directory / "actual_results.csv", encoding="utf-8-sig")
    if len(forecast) != 13 or set(forecast.match_no.astype(int)) != set(range(1, 14)):
        raise ValueError("expected 13 frozen matches")
    merged = forecast.merge(actual[["match_no", "home_score", "away_score", "result"]],
                            on="match_no", suffixes=("", "_actual"), validate="one_to_one")
    cards = []
    symbols = {1: "H", 0: "D", 2: "A"}
    for _, row in merged.sort_values("match_no").iterrows():
        number_id = int(row.match_no)
        home = html.escape(str(row.home_team))
        away = html.escape(str(row.away_team))
        goals_home = number(row, "home_score_actual")
        goals_away = number(row, "away_score_actual")
        actual_text = f"{fmt(goals_home, 0)}–{fmt(goals_away, 0)}" if math.isfinite(goals_home) and math.isfinite(goals_away) else "未確定"
        p = [number(row, f"next_prob_{side}") for side in ("home", "draw", "away")]
        prediction = ("H", "D", "A")[max(range(3), key=lambda i: p[i])]
        predicted_score, predicted_score_probability = likely_score_for_result(
            number(row, "final_lambda_home"), number(row, "final_lambda_away"), prediction)
        actual_symbol = symbols.get(int(row.result), "?") if pd.notna(row.result) else "?"
        attack_home = 0.5 * (number(row, "stats_1試合平均得点数_home") + average_xg(row, "home"))
        attack_away = 0.5 * (number(row, "stats_1試合平均得点数_away") + average_xg(row, "away"))
        defence_home = 0.5 * (number(row, "stats_1試合平均失点数_home") + number(row, "stats_1試合平均被ゴール期待値_home"))
        defence_away = 0.5 * (number(row, "stats_1試合平均失点数_away") + number(row, "stats_1試合平均被ゴール期待値_away"))
        card_class = "match flag" if prediction != actual_symbol and actual_symbol != "?" else "match"
        rates = ""
        if "goal_rate_adjustment_version" in merged.columns:
            rates = ("<tr><th>得失点率の補正</th><td colspan='3'>"
                     f"H得点 {fmt(number(row, 'goal_rate_for_home_published'))}→{fmt(number(row, 'goal_rate_for_home_used'))}、"
                     f"A失点 {fmt(number(row, 'goal_rate_against_away_published'))}→{fmt(number(row, 'goal_rate_against_away_used'))}。"
                     f"最終λ差 H {fmt(number(row, 'goal_rate_final_lambda_delta_home'), 3)}、"
                     f"A {fmt(number(row, 'goal_rate_final_lambda_delta_away'), 3)}。"
                     f"過去退場記録: {html.escape(str(row.get('goal_rate_historical_red_card_status', '不明')))}"
                     "</td></tr>")
        else:
            rates = "<tr><th>得失点率の補正</th><td colspan='3'>この保存時点では未適用。現在の補正を結果に遡って混ぜていません。</td></tr>"
        cards.append(f"""<details class='{card_class}'><summary><span>No.{number_id} {home}－{away}</span><span>予想 {prediction}（代表{predicted_score}、{predicted_score_probability:.1%}） ／ 実結果 {actual_symbol}（{actual_text}）</span></summary>
        <div class='detail-grid'><div><h3>ホーム {home} の得点計算</h3><table>
        <tr><th>チームの平均得点 / 平均xG</th><td>{fmt(number(row, 'stats_1試合平均得点数_home'))} / {fmt(average_xg(row, 'home'))}</td></tr>
        <tr><th>攻撃指標</th><td>{fmt(attack_home, 3)} = 上の2値の平均</td></tr>
        <tr><th>相手の平均失点 / 平均被xG</th><td>{fmt(number(row, 'stats_1試合平均失点数_away'))} / {fmt(number(row, 'stats_1試合平均被ゴール期待値_away'))}</td></tr>
        <tr><th>相手の守備指標</th><td>{fmt(defence_away, 3)} = 上の2値の平均</td></tr>
        <tr><th>基礎λ</th><td>{fmt(number(row, 'base_lambda_home'), 3)} ≈ √(攻撃指標 × 相手守備指標)</td></tr></table></div>
        <div><h3>アウェー {away} の得点計算</h3><table>
        <tr><th>チームの平均得点 / 平均xG</th><td>{fmt(number(row, 'stats_1試合平均得点数_away'))} / {fmt(average_xg(row, 'away'))}</td></tr>
        <tr><th>攻撃指標</th><td>{fmt(attack_away, 3)} = 上の2値の平均</td></tr>
        <tr><th>相手の平均失点 / 平均被xG</th><td>{fmt(number(row, 'stats_1試合平均失点数_home'))} / {fmt(number(row, 'stats_1試合平均被ゴール期待値_home'))}</td></tr>
        <tr><th>相手の守備指標</th><td>{fmt(defence_home, 3)} = 上の2値の平均</td></tr>
        <tr><th>基礎λ</th><td>{fmt(number(row, 'base_lambda_away'), 3)} ≈ √(攻撃指標 × 相手守備指標)</td></tr></table></div></div>
        <table class='steps'><tr><th>段階</th><th>H得点λ</th><th>A得点λ</th><th>意味</th></tr>
        <tr><td>基礎</td><td>{fmt(number(row, 'base_lambda_home'), 3)}</td><td>{fmt(number(row, 'base_lambda_away'), 3)}</td><td>平均得点・xGと相手守備指標</td></tr>
        <tr><td>Football LAB後</td><td>{fmt(number(row, 'lab_final_lambda_home'), 3)}</td><td>{fmt(number(row, 'lab_final_lambda_away'), 3)}</td><td>Football LAB補正</td></tr>
        <tr><td>欠場補正後</td><td>{fmt(number(row, 'final_lambda_home'), 3)}</td><td>{fmt(number(row, 'final_lambda_away'), 3)}</td><td>得点分布の入力</td></tr>
        {rates}</table>
        <p>得点分布の確率: H {number(row, 'next_prob_pre_uncertainty_home'):.1%} ／ D {number(row, 'next_prob_pre_uncertainty_draw'):.1%} ／ A {number(row, 'next_prob_pre_uncertainty_away'):.1%}。</p>
        <p>疲労・天候による確率調整率 {number(row, 'next_uncertainty_calibration_mix'):.1%} → 最終 <code>next_prob_*</code>: H {p[0]:.1%} ／ D {p[1]:.1%} ／ A {p[2]:.1%}。実スコア {actual_text}。従来の <code>prob_final_*</code> はこの計算に使いません。</p></details>""")
    return f"""<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>{round_id} 予想計算の検証</title><style>
    body{{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f3f6fa;color:#172033}}main{{max-width:1200px;margin:auto;padding:24px}}
    a{{color:#1769aa}}.card,details{{background:#fff;border:1px solid #d6dee8;border-radius:10px;padding:15px;margin:12px 0}}
    .flag{{border-left:5px solid #d99000}}summary{{cursor:pointer;display:flex;justify-content:space-between;gap:12px;font-weight:700}}.detail-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
    table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:7px;border-bottom:1px solid #e0e5ee;text-align:left}}th{{background:#edf2f8}}.steps{{margin-top:14px}}.note{{background:#fff6dc;border-left:4px solid #d99000;padding:12px}}
    @media(max-width:600px){{main{{padding:12px}}summary{{display:block}}}}
    </style><main><p><a href='index.html'>← レポート一覧</a></p><h1>{round_id} 予想計算の検証</h1>
    <div class='card'><p>保存済みの試合前予測を、当時の値のまま表示します。実結果は採点時に後から照合しました。オレンジ線は確率1位と実結果が違った試合です。</p>
    <p>見出しの「代表スコア」は、保存済みの最終λから計算した、予想H/D/Aに一致する中で最も確率の高い得点組です。括弧内の確率はそのスコア単独の発生確率であり、予想H/D/A全体の確率ではありません。保存データに独立したスコア予想列があるわけではありません。</p>
    <p><b>読み方:</b> チーム単独の攻撃指標と、その試合の予想得点λを分けて表示します。基礎λ＝√(自チーム攻撃指標 × 相手守備指標)。Football LABと欠場補正後のλから得点分布を計算し、疲労・天候による確率調整を経て最終H/D/A確率を得ます。</p></div>
    <p class='note'>このページは検証画面の試作です。将来の試合では平均値の対象試合数、突出値と退場情報の有無、補正前後のλ差、実スコアとのずれを継続して表示する設計です。過去の保存値に新補正を遡及適用しません。</p>
    {''.join(cards)}<p><a href='../eval/toto_rounds/{round_id}/growth_validation.html'>開催全体の検証へ</a></p></main></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="toto1668")
    args = parser.parse_args()
    output = ROOT / "data/reports" / f"forecast_trace_{args.round}.html"
    output.write_text(build(args.round), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
