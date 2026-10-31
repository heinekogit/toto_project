"""Retrospective accuracy of the latest official and shadow generators."""
import json
import hashlib
from pathlib import Path
import pandas as pd
from build_next_buyplan import build_round, build_shadow_round

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/reports/latest_portfolio_accuracy'


def main():
    source = ROOT/'data/eval/next_model/validation_v2/lab_xg_v0/predictions.csv'
    data = pd.read_csv(source)
    OUT.mkdir(exist_ok=True)
    rows, ticket_rows, exclusions = [], [], []
    for rid, frame in data.groupby('round_id', sort=True):
        frame = frame.sort_values('match_no').reset_index(drop=True)
        if len(frame)!=13 or set(frame.match_no)!=set(range(1,14)) or frame.actual_result.isna().any():
            exclusions.append({'round_id':rid,'rows':len(frame),'reason':'incomplete 13-match predictions/results'})
            continue
        plan, audit = build_round(frame, strategy='probability_allocation_v1')
        shadow, shadow_audit = build_shadow_round(frame, official_plan=plan, strategy='probability_allocation_v1')
        proven = bool(frame.prematch_snapshot_proven.fillna(False).all())
        hits = []
        for number in range(1,16):
            part = plan if number<=10 else shadow
            count = int(part[f'ticket{number:02}'].eq(frame.actual_result).sum())
            hits.append(count)
            ticket_rows.append(dict(round_id=rid,prematch_proven=proven,ticket=number,group='official' if number<=10 else 'shadow',hits=count,accuracy=count/13))
        for group, numbers in [('official',hits[:10]),('shadow',hits[10:]),('combined',hits)]:
            rows.append(dict(round_id=rid,prematch_proven=proven,group=group,tickets=len(numbers),
                             hits=sum(numbers),evaluated_cells=13*len(numbers),accuracy=sum(numbers)/(13*len(numbers)),
                             best_hits=max(numbers),best_accuracy=max(numbers)/13,
                             shadow_improved_best=max(hits)>max(hits[:10]),
                             **{f'tickets_ge{k}':sum(h>=k for h in numbers) for k in [10,11,12,13]}))
        plan.to_csv(OUT/f'{rid}_official.csv',index=False)
        shadow.to_csv(OUT/f'{rid}_shadow.csv',index=False)
        (OUT/f'{rid}_audit.json').write_text(json.dumps({'official':audit,'shadow':shadow_audit},ensure_ascii=False,indent=2))
    rounds = pd.DataFrame(rows)
    rounds.to_csv(OUT/'round_accuracy.csv',index=False)
    pd.DataFrame(ticket_rows).to_csv(OUT/'ticket_accuracy.csv',index=False)
    summaries = []
    for label, population in [('prematch_proven',rounds[rounds.prematch_proven]),('all_complete_reference',rounds)]:
        for group, part in population.groupby('group'):
            summaries.append(dict(population=label,group=group,rounds=len(part),tickets=int(part.tickets.sum()),
                                  accuracy=part.hits.sum()/part.evaluated_cells.sum(),
                                  average_best_hits=part.best_hits.mean(),shadow_improved_rounds=int(part.shadow_improved_best.sum()),
                                  **{f'tickets_ge{k}':int(part[f'tickets_ge{k}'].sum()) for k in [10,11,12,13]}))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'summary.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                                'mode':'retrospective regeneration, not live operating results',
                                                'exclusions':exclusions},indent=2))
    print(summary.to_string(index=False))
    print(rounds[rounds.prematch_proven].to_string(index=False))


if __name__=='__main__': main()
