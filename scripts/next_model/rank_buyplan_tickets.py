"""Rank an existing 10-ticket BuyPlan by model joint probability without editing it."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROB_COLUMNS = ['next_prob_home', 'next_prob_draw', 'next_prob_away']
SYMBOL_INDEX = {1: 0, 0: 1, 2: 2}
TICKET_COLUMNS = [f'ticket{i:02}' for i in range(1, 11)]


def rank_tickets(plan: pd.DataFrame, actual: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(plan) != 13 or set(plan.match_no.astype(int)) != set(range(1, 14)):
        raise ValueError('BuyPlan must contain exactly match_no 1..13')
    plan = plan.sort_values('match_no').reset_index(drop=True)
    probabilities = plan[PROB_COLUMNS].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError('Invalid H/D/A probabilities')
    tickets = plan[TICKET_COLUMNS].to_numpy(int).T
    if not np.isin(tickets, [0, 1, 2]).all():
        raise ValueError('Invalid ticket symbols')
    results = None
    if actual is not None:
        if len(actual) != 13 or set(actual.match_no.astype(int)) != set(range(1, 14)):
            raise ValueError('Actual results must contain exactly match_no 1..13')
        results = actual.sort_values('match_no').result.to_numpy(int)
        if not np.isin(results, [0, 1, 2]).all():
            raise ValueError('Unresolved actual results')
    ranking = []
    for index, ticket in enumerate(tickets):
        symbol_indices = np.array([SYMBOL_INDEX[int(value)] for value in ticket])
        joint = float(np.prod(probabilities[np.arange(13), symbol_indices]))
        ranking.append({'original_ticket': TICKET_COLUMNS[index], 'joint_probability': joint,
                        'joint_probability_percent': 100 * joint,
                        'D_count': int((ticket == 0).sum()),
                        'hits': int((ticket == results).sum()) if results is not None else None})
    order = sorted(range(10), key=lambda i: (-ranking[i]['joint_probability'], i))
    summary = pd.DataFrame([{'probability_rank': rank, **ranking[i]} for rank, i in enumerate(order, 1)])
    reordered = plan.drop(columns=TICKET_COLUMNS).copy()
    for rank, i in enumerate(order, 1):
        reordered[f'ticket{rank:02}'] = tickets[i]
    if sorted(map(tuple, reordered[TICKET_COLUMNS].to_numpy().T)) != sorted(map(tuple, tickets)):
        raise AssertionError('Ticket contents changed')
    return summary, reordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--actual', type=Path)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    plan = pd.read_csv(args.plan)
    actual = pd.read_csv(args.actual) if args.actual else None
    summary, reordered = rank_tickets(plan, actual)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.outdir / 'probability_ranking.csv', index=False)
    reordered.to_csv(args.outdir / 'buyplan_probability_order.csv', index=False)
    display = summary.copy()
    display['joint_probability_percent'] = display.joint_probability_percent.map(lambda x: f'{x:.8f}%')
    html = ('<!doctype html><html lang="ja"><meta charset="utf-8"><title>BuyPlan 確率順</title>'
            '<style>body{font-family:system-ui;margin:30px;color:#223}table{border-collapse:collapse}'
            'td,th{border:1px solid #ccd;padding:7px}th{background:#eef3f8}</style>'
            '<h1>BuyPlan 10券のモデル確率順</h1><p>元の候補番号を残して並び替え。'
            '各試合の確率を掛け合わせた値です。試合間の独立を仮定します。'
            '購入済み・保存済みの券や正式版ファイルは変更していません。</p>'
            + display[['probability_rank','original_ticket','joint_probability_percent','D_count','hits']].to_html(index=False)
            + '</html>')
    (args.outdir / 'index.html').write_text(html, encoding='utf-8')

if __name__ == '__main__':
    main()
