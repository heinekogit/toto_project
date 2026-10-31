"""Generate reviewable proportional plans and compare frozen historical rounds."""
import json
from pathlib import Path
import pandas as pd
from build_next_buyplan import build_round, score_round

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/reports/buyplan_probability_allocation_v1'


def main():
    OUT.mkdir(exist_ok=True)
    historical = pd.read_csv(ROOT/'data/eval/next_model/validation_v2/lab_xg_v0/predictions.csv')
    summaries = []
    for rid, frame in list(historical.groupby('round_id')) + [('toto1660', pd.read_csv(ROOT/'data/purchase_reference/next_predictions.csv'))]:
        if len(frame)!=13: continue
        for strategy in ['coverage_v1','probability_allocation_v1']:
            plan, audit = build_round(frame, strategy)
            row = dict(round_id=rid, strategy=strategy, draw_cells=audit['total_draw_cells'],
                       prematch_proven=bool(frame.prematch_snapshot_proven.all()) if 'prematch_snapshot_proven' in frame else False,
                       joint_probability_sum=sum(audit['ticket_joint_probabilities']))
            if rid!='toto1660':
                score=score_round(plan,frame)
                row.update(best_hits=score['best_hits'], average_hits=score['average_hits'])
            summaries.append(row)
            plan.to_csv(OUT/f'{rid}_{strategy}.csv',index=False,encoding='utf-8-sig')
            (OUT/f'{rid}_{strategy}.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
            if rid=='toto1660' and strategy=='probability_allocation_v1':
                (OUT/'toto1660_review.html').write_text('<meta charset="utf-8"><h1>確率配分方式・検証用</h1>'+plan.to_html(index=False),encoding='utf-8')
    results=pd.DataFrame(summaries)
    results.to_csv(OUT/'comparison.csv',index=False)
    print(results.to_string(index=False))


if __name__=='__main__': main()
