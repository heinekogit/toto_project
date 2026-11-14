"""Four-double D selection, isolated from production BuyPlan.

A is the operational default of this experimental module. B probes are
past-only offset logistic fits with bounded ranking adjustments, not new HDA.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from simulate_free_structural_draw import structure, PROB_COLUMNS

VARIANTS = {'B1_draw_composition': ['share00', 'share11'],
            'B2_combined': ['share00', 'share11', 'low_balanced', 'near_draw_support']}
MIN_TRAIN = 39
RIDGE = 10.0
MAX_ADJUSTMENT = .01

def features(frame):
    z = structure(frame).copy()
    for c in ['match_no', 'home_team', 'away_team', 'datetime']: z[c] = frame[c].to_numpy()
    p = frame[PROB_COLUMNS].to_numpy()
    z['anchor'] = np.where(p[:,0] >= p[:,2], 0, 2)
    z['unused_HA_probability'] = np.where(z.anchor == 0, p[:,2], p[:,0])
    z['HA_balance'] = 2*np.minimum(p[:,0], p[:,2])/(p[:,0]+p[:,2])
    z['P_22'] = frame.next_prob_2_2.to_numpy()
    z['share00'] = z.P_00/z.score_P_D
    z['share11'] = z.P_11/z.score_P_D
    z['low_balanced'] = z.low_total_le2*z.HA_balance
    z['near_draw_support'] = z.favoured_one_goal_share*(z.P_00+z.P_11)/(z.P_00+z.P_11+z.P_margin1)
    return z

def select(z, score=None):
    # Result columns never participate in sorting.
    if score is None: score = z.P_D.to_numpy()
    return pd.DataFrame({'score': np.asarray(score), 'pd': z.P_D.to_numpy(),
                         'match_no': z.match_no.to_numpy()}).sort_values(
        ['score','pd','match_no'], ascending=[False,False,True]).head(4).index.to_numpy()

def expand(anchor, chosen):
    anchor=np.asarray(anchor,dtype=int); chosen=np.asarray(chosen,dtype=int)
    if len(anchor)!=13 or not np.isin(anchor,[0,2]).all(): raise ValueError('H/A anchor required')
    if len(chosen)!=4 or len(set(chosen))!=4 or min(chosen)<0 or max(chosen)>12: raise ValueError('Four distinct matches required')
    tickets=np.tile(anchor,(16,1))
    for t,bits in zip(tickets,itertools.product([False,True],repeat=4)):
        t[chosen[np.array(bits)]]=1
    assert len(np.unique(tickets,axis=0))==16
    return tickets

def probe(train, test, columns):
    """Fit only preceding rounds; return bounded selection scores and audit."""
    if len(train)<MIN_TRAIN: raise ValueError('Insufficient earlier matches')
    center=float(train.P_D.mean()); scale=max(float(train.P_D.std(ddof=0)),.01)
    def basis(f):
        x=(f.P_D.to_numpy()-center)/scale
        return np.c_[np.ones(len(f)),x,x*x]
    a,b=basis(train),basis(test)
    beta_res=np.linalg.lstsq(a,train[columns].to_numpy(),rcond=None)[0]
    r=train[columns].to_numpy()-a@beta_res
    rt=test[columns].to_numpy()-b@beta_res
    rs=np.maximum(r.std(0),.02)
    x=np.clip(r/rs,-3,3); xt=np.clip(rt/rs,-3,3)
    y=train.actual_D.to_numpy(); offset=logit(train.P_D.to_numpy())
    def objective(beta):
        v=offset+x@beta
        return (np.logaddexp(0,v).sum()-y@v+.5*RIDGE*(beta@beta),
                x.T@(expit(v)-y)+RIDGE*beta)
    fit=minimize(objective,np.zeros(len(columns)),jac=True,method='BFGS')
    if not fit.success and np.linalg.norm(fit.jac)>1e-5: raise ValueError(fit.message)
    delta=np.clip(expit(logit(test.P_D.to_numpy())+xt@fit.x)-test.P_D.to_numpy(),-MAX_ADJUSTMENT,MAX_ADJUSTMENT)
    return test.P_D.to_numpy()+delta,dict(columns=columns,train_rows=len(train),
        train_rounds=sorted(train['round'].unique().tolist()),center=center,scale=scale,
        residual_beta=beta_res.tolist(),residual_scale=rs.tolist(),beta=fit.x.tolist(),
        ridge=RIDGE,max_adjustment=MAX_ADJUSTMENT,selection_score_is_probability=False)
