"""d-scoop Phase 1: score-distribution diagnostics, no ranking correction.

Input: saved Next Model outputs. Output: features and raw score matrices.
No result, ticket allocation, model fitting, or purchase generation is used.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

VERSION='d-scoop-v0.1-diagnostics'
REQUIRED=['match_no','home_team','away_team','final_lambda_home','final_lambda_away',
          'next_prob_home','next_prob_draw','next_prob_away','next_prob_0_0','next_prob_1_1','next_prob_2_2']

def diagnose(frame):
    if any(c not in frame for c in REQUIRED): raise ValueError('Missing saved Next Model fields')
    f=frame.sort_values('match_no').reset_index(drop=True)
    if len(f)!=13 or f.match_no.tolist()!=list(range(1,14)):raise ValueError('13 ordered fixtures required')
    g=np.arange(11);i,j=np.meshgrid(g,g,indexing='ij');difference=i-j
    rows=[]; matrices=[]
    for _,r in f.iterrows():
        lh,la=float(r.final_lambda_home),float(r.final_lambda_away)
        if not np.isfinite([lh,la]).all() or min(lh,la)<=0:raise ValueError('Invalid saved lambdas')
        matrix=np.outer(poisson.pmf(g,lh),poisson.pmf(g,la)); retained=float(matrix.sum());matrix/=retained
        raw=np.array([matrix[i>j].sum(),np.trace(matrix),matrix[i<j].sum()])
        saved=np.array([r.next_prob_0_0,r.next_prob_1_1,r.next_prob_2_2],dtype=float)
        score_error=float(np.max(np.abs(saved-np.diag(matrix)[:3])))
        if not np.isfinite(saved).all() or score_error>1e-9:raise ValueError('Saved score distribution mismatch')
        p=np.array([r.next_prob_home,r.next_prob_draw,r.next_prob_away],dtype=float)
        if not np.isfinite(p).all() or min(p)<0 or max(p)>1 or abs(p.sum()-1)>1e-9:raise ValueError('Invalid HDA')
        mix_value=r.get('next_uncertainty_calibration_mix',np.nan)
        if pd.isna(mix_value):
            if np.max(abs(p-raw))>1e-8:raise ValueError('Missing uncertainty mix; final HDA cannot be explained')
            mix=0.; mix_source='absent; raw HDA equality verified'
        else: mix=float(mix_value); mix_source='saved'
        if not 0<=mix<=1:raise ValueError('Invalid uncertainty mix')
        hda_error=float(np.max(abs(p-((1-mix)*raw+mix/3))))
        if hda_error>1e-8:raise ValueError('Final HDA differs from known saved model')
        mh,ma=matrix.sum(1),matrix.sum(0); near_h=float(matrix[difference==1].sum());near_a=float(matrix[difference==-1].sum())
        near=near_h+near_a;mu=float((matrix*difference).sum()); variance=float((matrix*(difference-mu)**2).sum())
        row=dict(match_no=int(r.match_no),home_team=r.home_team,away_team=r.away_team,
            P_H=p[0],P_D=p[1],P_A=p[2],score_P_H=raw[0],score_P_D=raw[1],score_P_A=raw[2],
            uncertainty_mix=mix,mix_source=mix_source,PD_adjustment=float(p[1]-raw[1]),
            lambda_home=lh,lambda_away=la,lambda_total=lh+la,lambda_abs_difference=abs(lh-la),
            P_00=float(matrix[0,0]),P_11=float(matrix[1,1]),P_22=float(matrix[2,2]),P_33plus=float(np.diag(matrix)[3:].sum()),
            P_home_zero=float(mh[0]),P_away_zero=float(ma[0]),P_total_le2=float(matrix[i+j<=2].sum()),
            P_10=float(matrix[1,0]),P_01=float(matrix[0,1]),P_21=float(matrix[2,1]),P_12=float(matrix[1,2]),
            draw_share00=float(matrix[0,0]/raw[1]),draw_share11=float(matrix[1,1]/raw[1]),
            draw_share22plus=float(np.diag(matrix)[2:].sum()/raw[1]),
            HA_probability_gap=float(abs(p[0]-p[2])),marginal_overlap=float(np.minimum(mh,ma).sum()),
            P_margin_plus1=near_h,P_margin_minus1=near_a,P_margin1=near,
            draw_share_within_zero_or_one=float(raw[1]/(raw[1]+near)),
            onegoal_bias=float((near_h-near_a)/near),goal_difference_mean=mu,goal_difference_sd=float(np.sqrt(variance)),
            home_goal_mean=float(g@mh),away_goal_mean=float(g@ma),
            home_goal_variance=float(((g-g@mh)**2)@mh),away_goal_variance=float(((g-g@ma)**2)@ma),
            retained_grid_mass=retained,omitted_tail_mass=1-retained,
            score_reconstruction_error=score_error,HDA_reconstruction_error=hda_error,
            d_scoop_score=np.nan,d_scoop_rank=np.nan,d_scoop_selected=pd.NA,
            evaluation_reason='Phase 1: 指標のみ。総合式未決定のためd-scoopによる採否・加点なし。')
        for d in range(-10,11):row[f'goal_difference_{d:+d}']=float(matrix[difference==d].sum())
        rows.append(row);matrices.append(matrix)
    out=pd.DataFrame(rows)
    order=np.lexsort((out.match_no.to_numpy(),-out.P_D.to_numpy()))
    out['PD_rank']=np.argsort(order)+1;out['baseline_selected']=out.PD_rank<=4
    return out,np.stack(matrices)
