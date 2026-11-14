"""Offline placement study. Never writes official models or purchase plans.

Exact outcome probabilities under independent fixtures. Local searches optimize
P(max hits >= 10); no claim of globally optimal portfolios or real calibration.
"""
from collections import OrderedDict
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from draw_count_conditioned import generate as quota_generate
from random_remaining_v24 import generate as random_generate
from deterministic_v23 import build_deterministic_tickets, arrange_deterministically
from placement_score import placement_scores

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/reports/draw_portfolio_design_audit'
POWERS = 3**np.arange(12, -1, -1)


class ExactPortfolio:
    def __init__(self, p):
        self.p = np.asarray(p)
        self.outcomes = np.indices((3,)*13, dtype=np.int8).reshape(13, -1).T
        self.mass = np.ones(3**13)
        for j in range(13): self.mass *= self.p[j, self.outcomes[:, j]]
        self.draws = (self.outcomes == 1).sum(axis=1)
        assert np.isclose(self.mass.sum(), 1)

    def metrics(self, tickets):
        tickets = np.asarray(tickets)
        best = np.zeros(len(self.mass), dtype=np.int8)
        for ticket in tickets:
            np.maximum(best, (self.outcomes == ticket).sum(axis=1), out=best)
        result = {f'p{k}': float(self.mass[best >= k].sum()) for k in range(10,14)}
        result['expected_max_hits'] = float(self.mass @ best)
        result['min_distance'] = min(int((a != b).sum()) for a,b in combinations(tickets,2))
        result['mean_distance'] = float(np.mean([int((a != b).sum()) for a,b in combinations(tickets,2)]))
        result['D_total'] = int((tickets == 1).sum())
        result['D_counts'] = (tickets == 1).sum(axis=1).tolist()
        result['D_position_distance'] = float(np.mean([int(((a==1)!=(b==1)).sum()) for a,b in combinations(tickets,2)]))
        # These are conditional model scenarios, not independent observed rounds.
        for label, mask in [('D_low_0_2', self.draws <= 2),('D_mid_3_4',(self.draws>=3)&(self.draws<=4)),('D_high_5plus',self.draws>=5)]:
            prob = float(self.mass[mask].sum())
            result[label+'_mass'] = prob
            result[label+'_p10'] = float(self.mass[mask & (best>=10)].sum()/prob) if prob else None
            result[label+'_expected_max'] = float(self.mass[mask] @ best[mask]/prob) if prob else None
        return result


@lru_cache(maxsize=2048)
def ball(ticket):
    """All outcomes within three errors: exactly 2,627 distinct base-3 codes."""
    ticket = np.asarray(ticket, dtype=int)
    center = int(ticket @ POWERS)
    parts = [np.array([center], dtype=np.int32)]
    for radius in (1,2,3):
        positions = np.array(list(combinations(range(13), radius)))
        offsets = np.array(list(product((1,2), repeat=radius)))
        old = ticket[positions][:,None,:]
        new = (old + offsets[None,:,:]) % 3
        codes = center + ((new-old)*POWERS[positions][:,None,:]).sum(axis=2)
        parts.append(codes.ravel().astype(np.int32))
    return np.sort(np.concatenate(parts))


def optimize_fixed(initial, evaluator, preserve_draw_counts=False, max_steps=20, protect_draw_strata=False):
    """Best-improvement exact union coverage; fixed match marginal counts.

    Both searches share single swaps and D-count-preserving two-match swaps.
    Constrained search rejects single swaps changing per-ticket D counts.
    No distance penalty or constraint. Search budget is recorded explicitly.
    """
    started = time.perf_counter()
    tickets = np.asarray(initial).copy()
    coverage = np.zeros(3**13, dtype=np.int8)
    for ticket in tickets: coverage[ball(tuple(ticket))] += 1
    trace = [float(evaluator.mass[coverage>0].sum())]
    strata = np.bincount(evaluator.draws[coverage>0],weights=evaluator.mass[coverage>0],minlength=14)
    reference_strata = strata.copy()
    for step in range(max_steps):
        best_gain = 1e-14; best_move = None
        for i,j in combinations(range(10),2):
            old_i,old_j = ball(tuple(tickets[i])),ball(tuple(tickets[j]))
            coverage[old_i]-=1; coverage[old_j]-=1
            old_union = np.union1d(old_i,old_j)
            lost = old_union[coverage[old_union]==0]
            loss = float(evaluator.mass[lost].sum())
            if protect_draw_strata:
                lost_strata = np.bincount(evaluator.draws[lost],weights=evaluator.mass[lost],minlength=14)
            different = np.flatnonzero(tickets[i]!=tickets[j])
            d_delta = (tickets[i]==1).astype(int)-(tickets[j]==1).astype(int)
            moves = [(int(m),) for m in different if not preserve_draw_counts or d_delta[m]==0]
            moves += [(int(m),int(n)) for m,n in combinations(different,2) if d_delta[m]*d_delta[n]==-1]
            for match_positions in moves:
                a,b = tickets[i].copy(),tickets[j].copy()
                ix = list(match_positions); a[ix],b[ix] = b[ix].copy(),a[ix].copy()
                new_union = np.union1d(ball(tuple(a)),ball(tuple(b)))
                gained = new_union[coverage[new_union]==0]
                gain = float(evaluator.mass[gained].sum())-loss
                if gain > best_gain:
                    if protect_draw_strata:
                        delta = np.bincount(evaluator.draws[gained],weights=evaluator.mass[gained],minlength=14)-lost_strata
                        if np.any(strata+delta < reference_strata-1e-14): continue
                    best_gain,best_move = gain,(i,j,a,b)
                    if protect_draw_strata: best_strata_delta=delta
            coverage[old_i]+=1; coverage[old_j]+=1
        if best_move is None: break
        i,j,a,b = best_move
        coverage[ball(tuple(tickets[i]))]-=1; coverage[ball(tuple(tickets[j]))]-=1
        tickets[i],tickets[j] = a,b
        if protect_draw_strata: strata += best_strata_delta
        coverage[ball(tuple(a))]+=1; coverage[ball(tuple(b))]+=1
        trace.append(float(evaluator.mass[coverage>0].sum()))
    for s in range(3): np.testing.assert_array_equal((tickets==s).sum(axis=0),(initial==s).sum(axis=0))
    if preserve_draw_counts: np.testing.assert_array_equal((tickets==1).sum(axis=1),(initial==1).sum(axis=1))
    return tickets, {'seconds':time.perf_counter()-started,'steps':len(trace)-1,'cap':max_steps,
                     'stopped_at_local_optimum':best_move is None,'p10_trace':trace,
                     'protect_draw_strata':protect_draw_strata,'initial_K_coverage':reference_strata.tolist(),
                     'final_K_coverage':np.bincount(evaluator.draws[coverage>0],weights=evaluator.mass[coverage>0],minlength=14).tolist()}


def optimize_free(initial, evaluator, max_steps=20):
    """Separate experiment: single-cell search, marginal counts NOT preserved."""
    started = time.perf_counter(); tickets = initial.copy()
    coverage = np.zeros(3**13,dtype=np.int8)
    for ticket in tickets: coverage[ball(tuple(ticket))]+=1
    trace=[float(evaluator.mass[coverage>0].sum())]
    for step in range(max_steps):
        best_gain=1e-14;best_move=None
        for i in range(10):
            old=ball(tuple(tickets[i]));coverage[old]-=1
            loss=float(evaluator.mass[old[coverage[old]==0]].sum())
            for m in range(13):
                for s in range(3):
                    if tickets[i,m]==s:continue
                    a=tickets[i].copy();a[m]=s;new=ball(tuple(a))
                    gain=float(evaluator.mass[new[coverage[new]==0]].sum())-loss
                    if gain>best_gain:best_gain,best_move=gain,(i,a)
            coverage[old]+=1
        if best_move is None:break
        i,a=best_move;coverage[ball(tuple(tickets[i]))]-=1;tickets[i]=a;coverage[ball(tuple(a))]+=1
        trace.append(float(evaluator.mass[coverage>0].sum()))
    return tickets,{'seconds':time.perf_counter()-started,'steps':len(trace)-1,'cap':max_steps,
                   'stopped_at_local_optimum':best_move is None,'p10_trace':trace}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]; portfolios={}; meta=[]
    for source in sorted((ROOT/'data/eval/toto_rounds').glob('*/snapshot/next_predictions.csv')):
        name=source.parent.parent.name
        frame=pd.read_csv(source).sort_values('match_no')
        p=frame[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy()
        manifest=json.loads((source.parent/'manifest.json').read_text())
        dates=pd.to_datetime(frame.datetime)
        if dates.dt.tz is None:dates=dates.dt.tz_localize('Asia/Tokyo')
        saved_before=int((dates>pd.Timestamp(manifest['created_at'])).sum())
        results=pd.read_csv(source.parent.parent/'actual_results.csv').sort_values('match_no')
        for col in ('match_no','home_team','away_team'):assert frame[col].tolist()==results[col].tolist()
        raw=results.result.to_numpy().astype(int);actual=np.where(raw==1,0,np.where(raw==0,1,2))
        evaluator=ExactPortfolio(p)
        info={'round':name,'created_at':manifest['created_at'],'saved_before_kickoff':saved_before,
              'actual_D':int((actual==1).sum()),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'interpretation':'development' if name=='toto1668' else 'historical_input_sensitivity_not_new_holdout'}
        meta.append(info);print('START',name,flush=True)
        portfolios[name]={}
        def record(method,tickets,seconds,seed=None,search=None):
            metrics=evaluator.metrics(tickets)
            metrics.update(round=name,method=method,seed=seed,seconds=seconds,
                           actual_mean=float((tickets==actual).sum(axis=1).mean()),
                           actual_max=int((tickets==actual).sum(axis=1).max()))
            rows.append(metrics)
            if seed in (None,0):portfolios[name][method]={'tickets':tickets.tolist(),'search':search}
        # 20 predeclared seeds, not selected after comparing results.
        for seed in range(20):
            started=time.perf_counter();t,_=random_generate(p,seed)
            record('v24_random',t,time.perf_counter()-started,seed)
        print('RANDOM DONE',name,flush=True)
        started=time.perf_counter();q,qa=quota_generate(p)
        record('quota_prototype',q,time.perf_counter()-started)
        base,_=random_generate(p,0)
        scores,_,_=placement_scores(p)
        started=time.perf_counter();v23_initial,_=build_deterministic_tickets(p)
        v23,_=arrange_deterministically(v23_initial,scores)
        record('v23_current_native_counts',v23,time.perf_counter()-started)
        # Isolate placement score/arrangement from different v2.3 rounding.
        started=time.perf_counter();v23_fixed,_=arrange_deterministically(base,scores)
        record('v23_arrangement_v24_counts_seed0',v23_fixed,time.perf_counter()-started)
        record('v23_initial_diversity_native_counts',v23_initial,0)
        old=pd.read_csv(source.parent/'buyplan.csv').sort_values('match_no')
        raw_t=old[[f'ticket{i:02d}' for i in range(1,11)]].to_numpy().T.astype(int)
        saved=np.where(raw_t==1,0,np.where(raw_t==0,1,2))
        record('historical_saved_tickets_version_varies',saved,0)
        for method,initial,restricted in [('direct_fixed_from_quota',q,False),('direct_fixed_Dquotas_retained',q,True),('direct_fixed_from_v24_seed0',base,False),('direct_fixed_K_coverage_guard',q,False)]:
            t,audit=optimize_fixed(initial,evaluator,preserve_draw_counts=restricted,protect_draw_strata=(method=='direct_fixed_K_coverage_guard'))
            record(method,t,audit['seconds'],search=audit)
            print(name,method,audit['seconds'],audit['steps'],flush=True)
        t,audit=optimize_free(q,evaluator)
        record('direct_FREE_counts_separate',t,audit['seconds'],search=audit)
        pd.DataFrame(rows).to_json(OUT/'metrics.json',orient='records',indent=2,force_ascii=False)
        (OUT/'portfolios.json').write_text(json.dumps(portfolios,ensure_ascii=False,indent=2))
        (OUT/'inputs.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        ball.cache_clear()
        print('DONE',name,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)


if __name__=='__main__':main()
