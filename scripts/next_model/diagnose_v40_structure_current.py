"""Read-only current-round score-structure audit; no probability correction or tickets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure import score_grid, structure
from condition_s_registry import VERSION, classify
from common import sha256


def diagnose(forecast: pd.DataFrame) -> pd.DataFrame:
    f = forecast.sort_values("match_no").reset_index(drop=True)
    if len(f) != 13 or f.match_no.tolist() != list(range(1, 14)):
        raise ValueError("13 ordered matches required")
    flags, median00, median11 = classify(f.next_prob_0_0.to_numpy(float), f.next_prob_1_1.to_numpy(float))
    rows = []
    for i, row in f.iterrows():
        lh, la = float(row.final_lambda_home), float(row.final_lambda_away)
        matrix, retained = score_grid(lh, la)
        s = structure(matrix)
        if not np.isclose(s["P_0_0"], row.next_prob_0_0, atol=1e-9) or not np.isclose(s["P_1_1"], row.next_prob_1_1, atol=1e-9):
            raise ValueError(f"No.{i+1}: saved scoreline probabilities do not match saved lambda")
        rows.append(dict(match_no=i+1, home_team=row.home_team, away_team=row.away_team,
                         lambda_home=lh, lambda_away=la, lambda_total=lh+la, lambda_abs_diff=abs(lh-la),
                         final_P_H=float(row.next_prob_home), final_P_D=float(row.next_prob_draw),
                         final_P_A=float(row.next_prob_away), raw_score_P_D=s["P_draw"],
                         P_0_0=s["P_0_0"], P_1_1=s["P_1_1"], P_2_2=s["P_2_2"],
                         P_3_3=s["P_3_3"], P_draw_4plus=s["P_draw_4plus"],
                         P_home_one_goal=s["P_home_1"], P_away_one_goal=s["P_away_1"],
                         P_two_plus=s["P_margin_2plus"],
                         condition_S=bool(flags[i]), median_P_0_0=median00, median_P_1_1=median11,
                         retained_0_to_10_mass=retained,
                         correction_status="未推定・final H/D/Aは変更しない"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    f = pd.read_csv(args.forecast, encoding="utf-8-sig", low_memory=False)
    current = diagnose(f)
    history_path = Path("data/reports/score_structure_phase35/diagnostics_78.csv")
    h = pd.read_csv(history_path, encoding="utf-8-sig")
    historic_flags = []
    for _, group in h.groupby("round"):
        flags, _, _ = classify(group.P_0_0.to_numpy(float), group.P_1_1.to_numpy(float))
        historic_flags.extend(zip(group.index, flags))
    h["condition_S"] = False
    for index, flag in historic_flags:
        h.at[index, "condition_S"] = bool(flag)
    args.outdir.mkdir(parents=True, exist_ok=True)
    current.to_csv(args.outdir / "match_structure_13.csv", index=False, encoding="utf-8-sig")
    historic = h.groupby("condition_S").agg(matches=("actual_D", "size"), actual_D=("actual_D", "sum"), expected_D=("baseline_P_D", "sum"), mean_P_D=("baseline_P_D", "mean")).reset_index()
    historic["actual_D_rate"] = historic.actual_D / historic.matches
    historic.to_csv(args.outdir / "historic_condition_S_exploratory.csv", index=False, encoding="utf-8-sig")
    audit = dict(round=args.round, forecast_sha256=sha256(args.forecast), history_sha256=sha256(history_path),
                 condition_version=VERSION, history_discovery_sample=True,
                 no_probability_correction=True, no_tickets_generated=True)
    (args.outdir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    display = current[["match_no", "home_team", "away_team", "final_P_H", "final_P_D", "final_P_A",
                       "lambda_total", "lambda_abs_diff", "P_0_0", "P_1_1", "P_2_2", "P_home_one_goal", "P_away_one_goal", "condition_S"]].copy()
    for col in ("final_P_H", "final_P_D", "final_P_A", "P_0_0", "P_1_1", "P_2_2", "P_home_one_goal", "P_away_one_goal"):
        display[col] = (display[col]*100).round(2)
    body = ("# v4.0 得点構造診断："+args.round+"\n\n"
            "条件Sを含む得点構造は、保存λから再現した11×11得点分布の記述です。"
            "現時点では構造別のD予測誤差を独立標本で確認できていないため、試合別P(D)の上方修正値は算出しません。"
            "この診断は10口の選択根拠として確定していません。\n\n"
            "## 13試合\n\n```csv\n"+display.to_csv(index=False)+"```\n\n"
            "## 過去78試合の探索集計\n\n```csv\n"+historic.to_csv(index=False)+"```\n\n"
            "条件Sはこの78試合から発見されました。上表の実D率と期待Dの差は独立検証ではなく、補正幅に変換できません。"
            "得点構造指標は保存λに由来し、既存P(D)と重なる情報です。"
            "将来の試合前スナップショットで、同程度のP(D)における構造別の校正差を確認する必要があります。\n")
    (args.outdir / "report.md").write_text(body, encoding="utf-8")
    (args.outdir / "index.html").write_text("<!doctype html><html lang='ja'><meta charset='utf-8'><title>v4.0得点構造診断</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd5df;padding:6px}th{background:#e8eff7}.scroll{overflow:auto}.note{background:#fff0d3;padding:12px}</style><h1>"+args.round+" 得点構造診断</h1><p class='note'>構造別のD過小評価は未検証。P(D)補正と配券変更は行っていません。</p><div class='scroll'>"+display.to_html(index=False,escape=True)+"</div><h2>過去78試合・探索集計</h2>"+historic.to_html(index=False,escape=True)+"<p><a href='report.md'>詳細</a>／<a href='match_structure_13.csv'>13試合CSV</a></p></html>",encoding="utf-8")
    print(args.outdir)


if __name__ == "__main__":
    main()
