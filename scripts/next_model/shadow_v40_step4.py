"""Independent v4.0 Step 4 hypothesis shadow. Never changes official tickets."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure_phase4b_stage1 import decode, top_ten, universe
from condition_s_registry import VERSION, classify

FILES = ("shadow_buyplan_v40.csv", "shadow_buyplan_v40.html", "shadow_v40_audit.json", "shadow_v40_sensitivity.csv")
TICKETS = [f"ticket{i:02d}" for i in range(1, 11)]
SYMBOLS = np.array(["H", "D", "A"])
# Previously published Step 4 grid point. These values are NOT calibrated effects.
ODDS_FACTOR = 1.5
EFFECT_WEIGHT = 0.25
SENSITIVITY_FACTORS = (1.0, 1.1, 1.5, 2.0)
SENSITIVITY_WEIGHTS = (0.0, 0.25, 0.5, 1.0)


def build(forecast: pd.DataFrame, official: pd.DataFrame):
    f = forecast.sort_values("match_no").reset_index(drop=True)
    o = official.sort_values("match_no").reset_index(drop=True)
    if len(f) != 13 or len(o) != 13 or f.match_no.tolist() != list(range(1, 14)) or o.match_no.tolist() != list(range(1, 14)):
        raise ValueError("v4 shadow requires 13 ordered matches")
    if not (f[["home_team", "away_team"]].to_numpy() == o[["home_team", "away_team"]].to_numpy()).all():
        raise ValueError("official and shadow fixture identities differ")
    p = f[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-9):
        raise ValueError("invalid final H/D/A probabilities")
    flags, med00, med11 = classify(f.next_prob_0_0.to_numpy(float), f.next_prob_1_1.to_numpy(float))
    if not np.isfinite(f[["next_prob_0_0", "next_prob_1_1"]].to_numpy(float)).all():
        raise ValueError("invalid condition S score probabilities")
    q = p.copy()
    q[flags, 1] *= ODDS_FACTOR
    q /= q.sum(axis=1, keepdims=True)
    # Mixture of complete round distributions: the S effect is a common state.
    base_mass = universe(p)
    effect_mass = universe(q)
    mixture = (1 - EFFECT_WEIGHT) * base_mass + EFFECT_WEIGHT * effect_mass
    codes = top_ten(mixture)
    matrix = decode(codes)
    if len(set(codes)) != 10:
        raise AssertionError("duplicate shadow tickets")
    official_values = o[TICKETS].to_numpy(int).T
    if not np.isin(official_values, [0, 1, 2]).all():
        raise ValueError("invalid official ticket values")
    official_matrix = np.where(official_values == 1, 0, np.where(official_values == 0, 1, 2))
    def strings(m):
        return ["".join(SYMBOLS[row]) for row in m]
    shadow_strings, official_strings = strings(matrix), strings(official_matrix)
    def mass(prob, m):
        return float(np.prod(prob[np.arange(13)[None, :], m], axis=1).sum())
    shadow_prob, official_prob = mass(p, matrix), mass(p, official_matrix)
    shadow_q, official_q = mass(q, matrix), mass(q, official_matrix)
    shadow_mix = (1-EFFECT_WEIGHT)*shadow_prob + EFFECT_WEIGHT*shadow_q
    official_mix = (1-EFFECT_WEIGHT)*official_prob + EFFECT_WEIGHT*official_q
    out = f[["match_no", "home_team", "away_team", "next_prob_home", "next_prob_draw", "next_prob_away"]].copy()
    out["condition_S"] = flags
    for i, column in enumerate(TICKETS):
        out[column] = np.array([1, 0, 2])[matrix[i]]
    audit = dict(mode="v4.0 Step 4 research hypothesis; not adopted", condition_S_version=VERSION,
                 condition_S_matches=[int(x) for x in f.loc[flags, "match_no"]],
                 median_P_0_0=med00, median_P_1_1=med11,
                 hypothesis_D_odds_factor=ODDS_FACTOR, hypothesis_effect_state_weight=EFFECT_WEIGHT,
                 state_definition="all S matches affected together; mixture of full 13-match distributions",
                 selection="exact top 10 of hypothesis mixture among 3^13 outcomes",
                 official_final_model_prob13=official_prob, shadow_final_model_prob13=shadow_prob,
                 official_effect_state_prob13=official_q, shadow_effect_state_prob13=shadow_q,
                 official_hypothesis_prob13=official_mix, shadow_hypothesis_prob13=shadow_mix,
                 official_tickets=official_strings, shadow_tickets=shadow_strings,
                 added_tickets=sorted(set(shadow_strings)-set(official_strings)),
                 removed_tickets=sorted(set(official_strings)-set(shadow_strings)),
                 official_unchanged=True, validated_effect=False)
    return out, audit


def sensitivity(forecast: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    f = forecast.sort_values("match_no").reset_index(drop=True)
    p = f[["next_prob_home", "next_prob_draw", "next_prob_away"]].to_numpy(float)
    flags, _, _ = classify(f.next_prob_0_0.to_numpy(float), f.next_prob_1_1.to_numpy(float))
    baseline = universe(p)
    records = []
    for factor in SENSITIVITY_FACTORS:
        q = p.copy()
        q[flags, 1] *= factor
        q /= q.sum(axis=1, keepdims=True)
        effect = universe(q)
        for weight in SENSITIVITY_WEIGHTS:
            codes = top_ten((1-weight)*baseline + weight*effect)
            tickets = ["".join(SYMBOLS[row]) for row in decode(codes)]
            records.append(dict(D_odds_factor=factor, effect_state_weight=weight,
                                P_prob13=float(baseline[codes].sum()), Q_prob13=float(effect[codes].sum()),
                                mixture_prob13=float(((1-weight)*baseline+weight*effect)[codes].sum()),
                                changed_tickets_vs_display=10-len(set(tickets)&set(selected)),
                                D_tickets=sum("D" in ticket for ticket in tickets),
                                tickets=";".join(tickets)))
    return pd.DataFrame(records)


def write(forecast: pd.DataFrame, official: pd.DataFrame, round_id: str, outdir: Path) -> dict:
    out, audit = build(forecast, official)
    outdir = Path(outdir)
    out.to_csv(outdir / FILES[0], index=False, encoding="utf-8-sig")
    grid = sensitivity(forecast, audit["shadow_tickets"])
    grid.to_csv(outdir / FILES[3], index=False, encoding="utf-8-sig")
    audit["sensitivity_factors"] = SENSITIVITY_FACTORS
    audit["sensitivity_weights"] = SENSITIVITY_WEIGHTS
    (outdir / FILES[2]).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    f = forecast.sort_values("match_no").reset_index(drop=True)
    o = official.sort_values("match_no").reset_index(drop=True)
    def table(frame):
        head = "<tr><th>No.</th><th>対戦</th><th>条件S</th>" + "".join(f"<th>{c}</th>" for c in TICKETS) + "</tr>"
        rows = []
        for i in range(13):
            cells = [str(i+1), f"{f.loc[i,'home_team']}－{f.loc[i,'away_team']}", "該当" if out.loc[i,"condition_S"] else "—"]
            cells += [{0:"D",1:"H",2:"A"}[int(frame.loc[i,c])] for c in TICKETS]
            rows.append("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in cells) + "</tr>")
        return "<div class='scroll'><table>" + head + "".join(rows) + "</table></div>"
    delta = audit["shadow_final_model_prob13"] - audit["official_final_model_prob13"]
    comparisons = []
    for label, official_key, shadow_key in (("保存final P", "official_final_model_prob13", "shadow_final_model_prob13"),
                                            ("条件S効果状態 Q（仮説）", "official_effect_state_prob13", "shadow_effect_state_prob13"),
                                            ("混合 (1−w)P+wQ（仮説）", "official_hypothesis_prob13", "shadow_hypothesis_prob13")):
        a, b = audit[official_key], audit[shadow_key]
        comparisons.append(f"<tr><td>{label}</td><td>{a*100:.9f}%</td><td>{b*100:.9f}%</td><td>{(b-a)*100:+.9f}pt</td></tr>")
    sensitivity_view = grid.drop(columns=["tickets"]).copy()
    for col in ("P_prob13", "Q_prob13", "mixture_prob13"):
        sensitivity_view[col] = (sensitivity_view[col]*100).map(lambda x:f"{x:.9f}%")
    document = f"""<!doctype html><html lang='ja'><meta charset='utf-8'><title>{html.escape(round_id)} v4.0研究用シャドー</title>
<style>body{{font:16px/1.6 system-ui;margin:24px;color:#172033}}.warn{{background:#fff0d3;padding:14px;border-left:5px solid #b57800;max-width:1100px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;margin:12px 0 28px}}th,td{{border:1px solid #cbd5df;padding:6px 9px;text-align:center}}th{{background:#e8eff7}}td:nth-child(2){{text-align:left}}code{{white-space:normal}}</style>
<h1>{html.escape(round_id)} 正式v2.4／v4.0 Step 4仮説シャドー</h1>
<p class='warn'>これはv4.0の代表方式や正式配券ではなく、Step 4の<strong>条件S混合方式・未校正仮説の一設定</strong>です。17状態λ変動方式は使用していません。購入推奨ではありません。正式v2.4の券は変更していません。</p>
<h2>確定仕様と仮説設定</h2><p>確定：13試合、相異なる10口、固定枚数・固定D数なし。券の13的中モデル確率は試合別確率の積、10口全体はその和。固定final分布だけなら確率上位10口が数学的基準です。</p>
<p>仮説：条件S該当試合のDオッズを <strong>{ODDS_FACTOR}倍</strong>した効果状態Qを作り、元のfinal分布Pと <strong>{1-EFFECT_WEIGHT:.0%}／{EFFECT_WEIGHT:.0%}</strong> で混合。S該当が複数なら同時に効果が出る共通状態。倍率1.5・重み25%は既存Step 4の試験格子の一点を再現するための表示設定で、推定値でも採用値でもありません。3^13組から混合分布の確率上位10口を選択しました。</p>
<p>条件S：P(0-0) ≤ {audit['median_P_0_0']*100:.4f}% かつ P(1-1) ≥ {audit['median_P_1_1']*100:.4f}%（未丸め値）。該当 No.{', '.join(map(str,audit['condition_S_matches'])) or 'なし'}。</p>
<h2>P・Q・混合での同一10口比較</h2><table><tr><th>評価分布</th><th>正式v2.4</th><th>Step 4仮説シャドー</th><th>差</th></tr>{''.join(comparisons)}</table>
<p>Pの差は機会損失または利得、Qの差は仮説成立時の差、混合の差は仮定した効果状態重みに依存する差です。PとQの混合重み25%を現実の成立頻度と解釈しません。</p>
<p>券集合の入替：{len(audit['added_tickets'])}口。追加 <code>{html.escape(', '.join(audit['added_tickets']) or 'なし')}</code>／除外 <code>{html.escape(', '.join(audit['removed_tickets']) or 'なし')}</code>。</p>
<h2>仮説値を変えた感度</h2><p>Dオッズ倍率1.0/1.1/1.5/2.0、効果状態重み0/25/50/100%の各設定で10口を再選択。変化口数は表示中の1.5倍・25%券との集合差です。個々の券は<a href='{FILES[3]}'>感度CSV</a>に保存しています。</p><div class='scroll'>{sensitivity_view.to_html(index=False,escape=True)}</div>
<h2>正式BuyPlan v2.4：13試合×10口</h2>{table(o)}<h2>独立v4.0研究用シャドー：13試合×10口</h2>{table(out)}
<p>予測確率と正式券は同一生成時点の値を使用。実結果や購入処理は使っていません。<a href='{FILES[0]}'>シャドーCSV</a>／<a href='{FILES[2]}'>設定と監査JSON</a></p></html>"""
    (outdir / FILES[1]).write_text(document, encoding="utf-8")
    return audit
