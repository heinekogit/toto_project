"""Experimental score-structure coefficients, never prediction corrections.
Reference normalization uses input features only; no results are accepted.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass(frozen=True)
class FeatureConfig:
    strength: float = .15
    zero_share_weight: float = .7
    near_balance_weight: float = .3
    lower: float = .85
    upper: float = 1.2
    scale_floor: float = .02


def extract(frame):
    required=['final_lambda_home','final_lambda_away','next_prob_home','next_prob_draw','next_prob_away','next_prob_0_0','next_prob_1_1','next_prob_2_2']
    if any(c not in frame for c in required):raise ValueError('Required saved score inputs missing; no proxy substitution')
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():raise ValueError('Missing/nonfinite score inputs')
    rows=[]
    for _,r in frame.iterrows():
        lh,la=float(r.final_lambda_home),float(r.final_lambda_away)
        if min(lh,la)<=0:raise ValueError('Invalid lambda')
        h=poisson.pmf(np.arange(11),lh);a=poisson.pmf(np.arange(11),la)
        matrix=np.outer(h,a);mass=matrix.sum();matrix/=mass
        d=float(np.trace(matrix));ph=float(np.tril(matrix,-1).sum());pa=float(np.triu(matrix,1).sum())
        saved=np.array([r.next_prob_0_0,r.next_prob_1_1,r.next_prob_2_2],dtype=float)
        err=float(np.max(np.abs(saved-np.diag(matrix)[:3])))
        if err>1e-9:raise ValueError('Saved score probabilities disagree with reconstructed model; stop')
        mix=float(r.get('next_uncertainty_calibration_mix',0))
        if not np.isfinite(mix) or not 0<=mix<=1:raise ValueError('Invalid uncertainty mix')
        final=np.array([r.next_prob_home,r.next_prob_draw,r.next_prob_away],dtype=float)
        probability_error=float(np.max(np.abs(final-((1-mix)*np.array([ph,d,pa])+mix/3))))
        if probability_error>1e-8:raise ValueError('Final HDA cannot be reconciled with saved score model')
        plus=float(np.diag(matrix,-1).sum());minus=float(np.diag(matrix,1).sum())
        rows.append({'P_H':final[0],'P_D':final[1],'P_A':final[2],'score_P_D':d,
                     'P_00':matrix[0,0],'P_11':matrix[1,1],'P_22plus':float(np.diag(matrix)[2:].sum()),
                     'lambda_home':lh,'lambda_away':la,'lambda_total':lh+la,'lambda_abs_difference':abs(lh-la),
                     'marginal_overlap':float(np.minimum(matrix.sum(axis=1),matrix.sum(axis=0)).sum()),
                     'P_home_lead1':plus,'P_away_lead1':minus,'P_margin1':plus+minus,
                     'zero_share_given_draw':matrix[0,0]/d,
                     'balanced_near_given_non_draw':2*min(plus,minus)/(1-d),
                     'uncertainty_mix':mix,'score_grid_mass':mass,'score_reconstruction_error':err,'HDA_reconciliation_error':probability_error})
    return pd.DataFrame(rows,index=frame.index)


def design(features,center,scale):
    x=(features.P_D.to_numpy()-center)/scale
    raw=(features.score_P_D.to_numpy()-center)/scale
    return np.column_stack([np.ones(len(x)),x,x*x,x*x*x,raw,raw*raw,raw*raw*raw,features.uncertainty_mix.to_numpy()])


def fit_reference(features,config=FeatureConfig()):
    """Freeze a descriptive normalizer; no outcome/coefficient tuning."""
    if len(features)<10:raise ValueError('Too few rows for reference')
    center=float(features.P_D.mean());scale=max(float(features.P_D.std(ddof=0)),1e-3)
    x=design(features,center,scale);f=features.zero_share_given_draw.to_numpy()
    beta0=np.linalg.lstsq(x,f,rcond=None)[0];res0=f-x@beta0
    near_x=np.column_stack([x,res0]);n=features.balanced_near_given_non_draw.to_numpy()
    beta1=np.linalg.lstsq(near_x,n,rcond=None)[0];res1=n-near_x@beta1
    return {'version':'match_draw_structure_shadow_v1','config':asdict(config),'center':center,'scale':scale,
            'zero_beta':beta0.tolist(),'near_beta':beta1.tolist(),
            'zero_scale':max(float(res0.std()),config.scale_floor),'near_scale':max(float(res1.std()),config.scale_floor),
            'zero_residual_std':float(res0.std()),'near_residual_std':float(res1.std()),
            'controls':['final_P_D_cubic','score_P_D_cubic','uncertainty_mix'],'reference_rows':len(features),'P_D_range':[float(features.P_D.min()),float(features.P_D.max())],
            'fit_uses_outcomes':False,'normalizer_is_descriptive_not_predictive_training':True}


def coefficients(features,reference,config=None):
    cfg=config or FeatureConfig(**reference['config'])
    vals=np.array(list(asdict(cfg).values()),dtype=float)
    if not np.isfinite(vals).all() or cfg.lower<=0 or not cfg.lower<=1<=cfg.upper or min(cfg.strength,cfg.zero_share_weight,cfg.near_balance_weight)<0 or cfg.scale_floor<=0:raise ValueError('Invalid feature configuration')
    weights=cfg.zero_share_weight+cfg.near_balance_weight
    if weights<=0:raise ValueError('At least one feature weight is required')
    x=design(features,reference['center'],reference['scale'])
    f=features.zero_share_given_draw.to_numpy();n=features.balanced_near_given_non_draw.to_numpy()
    expected0=np.sum(x*np.asarray(reference['zero_beta']),axis=1);r0=f-expected0
    expected1=np.sum(np.column_stack([x,r0])*np.asarray(reference['near_beta']),axis=1);r1=n-expected1
    z0=np.tanh(r0/max(reference['zero_scale'],cfg.scale_floor));z1=np.tanh(r1/max(reference['near_scale'],cfg.scale_floor))
    part0=cfg.strength*cfg.zero_share_weight/weights*z0;part1=cfg.strength*cfg.near_balance_weight/weights*z1
    kd=np.clip(np.exp(part0+part1),cfg.lower,cfg.upper)
    out=features.copy()
    out['expected_zero_share']=expected0;out['zero_residual']=r0;out['near_residual']=r1
    out['zero_log_contribution']=part0;out['near_log_contribution']=part1
    out['K_D']=kd;out['D_placement_value']=out.P_D*kd
    out['outside_reference_D_range']=(out.P_D<reference['P_D_range'][0])|(out.P_D>reference['P_D_range'][1])
    out['reason']=[f'最終D・補正前D・補正量を考慮した0–0構成差 {a:+.4f}（係数対数寄与 {b:+.4f}）、重複除去後の双方1点差 {c:+.4f}（寄与 {d:+.4f}）' for a,b,c,d in zip(r0,part0,r1,part1)]
    return out
