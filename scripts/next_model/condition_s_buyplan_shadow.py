"""Read-only condition-S annotation of the official BuyPlan tickets."""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from condition_s_registry import VERSION, classify


TICKETS = [f"ticket{i:02d}" for i in range(1, 11)]
SYMBOLS = {0: "D", 1: "H", 2: "A"}


def build_view(plan: pd.DataFrame, forecast: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a display copy; never modify the official plan or probabilities."""
    p = plan.sort_values("match_no").reset_index(drop=True)
    f = forecast.sort_values("match_no").reset_index(drop=True)
    if len(p) != 13 or len(f) != 13 or p.match_no.tolist() != list(range(1, 14)) or f.match_no.tolist() != list(range(1, 14)):
        raise ValueError("condition-S BuyPlan shadow requires the same 13 numbered matches")
    if not {"next_prob_0_0", "next_prob_1_1"}.issubset(f.columns):
        raise ValueError("scoreline probabilities are missing")
    if not set(TICKETS).issubset(p.columns):
        raise ValueError("official BuyPlan tickets are missing")
    if not np.isfinite(f[["next_prob_0_0", "next_prob_1_1"]].to_numpy(float)).all():
        raise ValueError("scoreline probabilities are not finite")
    flags, median00, median11 = classify(f.next_prob_0_0.to_numpy(float), f.next_prob_1_1.to_numpy(float))
    rows = []
    for ix, (buy, pred) in enumerate(zip(p.itertuples(index=False), f.itertuples(index=False))):
        if (buy.match_no, buy.home_team, buy.away_team) != (pred.match_no, pred.home_team, pred.away_team):
            raise ValueError(f"match identity mismatch at No.{ix+1}")
        for side in ("home", "draw", "away"):
            if not math.isclose(float(getattr(buy, f"next_prob_{side}")), float(getattr(pred, f"next_prob_{side}")), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"saved probability mismatch at No.{ix+1}")
        row = {"No.": ix + 1, "対戦": f"{buy.home_team}－{buy.away_team}",
               "H%": float(buy.next_prob_home) * 100, "D%": float(buy.next_prob_draw) * 100,
               "A%": float(buy.next_prob_away) * 100,
               "P(0-0)%": float(pred.next_prob_0_0) * 100, "P(1-1)%": float(pred.next_prob_1_1) * 100,
               "条件S": "該当" if flags[ix] else "—"}
        for number, column in enumerate(TICKETS, 1):
            symbol = int(getattr(buy, column))
            if symbol not in SYMBOLS:
                raise ValueError(f"invalid official ticket symbol at No.{ix+1}, candidate{number:02d}")
            row[f"候補{number:02d}"] = SYMBOLS[symbol]
        rows.append(row)
    return pd.DataFrame(rows), dict(version=VERSION, S_count=int(flags.sum()), median_P_0_0=median00,
                                    median_P_1_1=median11, source="official_buyplan_tickets_unchanged")


def write(plan: pd.DataFrame, forecast: pd.DataFrame, round_id: str, outdir: Path) -> dict:
    view, audit = build_view(plan, forecast)
    outdir = Path(outdir)
    csv_path = outdir / "shadow_condition_s_buyplan.csv"
    html_path = outdir / "shadow_condition_s_buyplan.html"
    view.to_csv(csv_path, index=False, encoding="utf-8-sig")
    columns = view.columns.tolist()
    head = "".join(f"<th>{html.escape(name)}</th>" for name in columns)
    body = "".join("<tr" + (' class="selected"' if r["条件S"] == "該当" else "") + ">" +
                   "".join(f"<td>{html.escape(f'{r[name]:.2f}' if name.endswith('%') else str(r[name]))}</td>" for name in columns) +
                   "</tr>" for _, r in view.iterrows())
    ticket_summary = []
    for number in range(1, 11):
        column = f"候補{number:02d}"
        probability = math.prod(float(plan.sort_values("match_no").iloc[ix][{"H": "next_prob_home", "D": "next_prob_draw", "A": "next_prob_away"}[symbol]])
                                for ix, symbol in enumerate(view[column]))
        ticket_summary.append(f"<tr><td>候補{number:02d}</td><td>{int((view[column] == 'D').sum())}</td><td>{probability*100:.8f}%</td></tr>")
    html_path.write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>{html.escape(round_id)} 現行BuyPlanと条件S</title>
<style>body{{font:16px system-ui,sans-serif;margin:24px;color:#172033}}.notice{{background:#fff3cf;padding:12px;border-left:5px solid #c48700;max-width:1000px}}.wrap{{overflow-x:auto}}table{{border-collapse:collapse;white-space:nowrap}}th,td{{border:1px solid #ccd5df;padding:7px 9px;text-align:center}}th{{background:#edf2f7}}td:nth-child(2){{text-align:left}}.selected{{background:#ffedb0}}</style>
<h1>{html.escape(round_id)} 現行BuyPlan ＋ 条件S</h1><p class="notice"><strong>候補01〜10は正式BuyPlan v2.4の出力そのものです。</strong> 条件Sは閲覧用の注記で、券のH/D/Aを変更していません。研究用v4.0配券ではありません。</p>
<p>条件S：P(0-0) ≤ {audit['median_P_0_0']*100:.4f}%、P(1-1) ≥ {audit['median_P_1_1']*100:.4f}%（開催内中央値、未丸め値で判定）。該当 {audit['S_count']} 試合。</p>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<h2>候補別の診断</h2><table><thead><tr><th>候補</th><th>D本数</th><th>単独13的中モデル確率</th></tr></thead><tbody>{''.join(ticket_summary)}</tbody></table>
<p>予測確率と券は正式出力から読み取っています。条件Sの予測力は将来検証中です。</p></html>''', encoding="utf-8")
    # Keep the previous shadow path for existing links, and publish the same
    # current-BuyPlan view under the descriptive name used by the 1670 preview.
    (outdir / "current_buyplan_with_condition_s.csv").write_bytes(csv_path.read_bytes())
    (outdir / "current_buyplan_with_condition_s.html").write_bytes(html_path.read_bytes())
    return audit
