"""Discontinued four-double experiment, retained for historical replay."""
import numpy as np
import pandas as pd
from four_double_selection_shadow import expand

VERSION='v3.0-rc1'
PROBS=['next_prob_home','next_prob_draw','next_prob_away']

def build(frame):
    f=frame.sort_values('match_no').reset_index(drop=True).copy()
    if len(f)!=13 or f.match_no.tolist()!=list(range(1,14)):
        raise ValueError('Exactly 13 matches numbered 1..13 required')
    if f[['home_team','away_team']].isna().any().any() or f[['home_team','away_team']].duplicated().any():
        raise ValueError('Missing or duplicate fixture identity')
    p=f[PROBS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,rtol=0,atol=1e-9):
        raise ValueError('Invalid H/D/A probabilities; no automatic normalization')
    anchor=np.where(p[:,0]>=p[:,2],0,2)
    ranked=np.lexsort((f.match_no.to_numpy(),-p[:,1])); selected=ranked[:4]
    # Expansion ordering is by match number. Ticket01 is the anchor itself.
    tickets=expand(anchor,np.sort(selected))
    plan=f[['match_no','home_team','away_team']+PROBS].copy()
    plan['candidate00']=np.array([1,0,2])[anchor]
    plan['D_rank']=np.argsort(ranked)+1
    plan['double_selected']=np.isin(np.arange(13),selected)
    plan['purchase_selection']=[('HDA'[a]+'/D') if i in selected else 'HDA'[a] for i,a in enumerate(anchor)]
    for j,t in enumerate(tickets,1): plan[f'ticket{j:02}']=np.array([1,0,2])[t]
    mass=p[np.arange(13),tickets].prod(1)
    singles=np.ones(13,dtype=bool);singles[selected]=False
    direct=np.prod(p[np.flatnonzero(singles),anchor[singles]])*np.prod(p[selected,anchor[selected]]+p[selected,1])
    np.testing.assert_allclose(mass.sum(),direct,rtol=1e-12,atol=1e-20)
    diagnostics=pd.DataFrame({'candidate':[f'{i:02}' for i in range(1,17)],
        'symbols':[''.join('HDA'[s] for s in t) for t in tickets],
        'D_count':(tickets==1).sum(1),'joint_probability':mass})
    audit=dict(version=VERSION,status='discontinued_historical_experiment',selection='P(D) descending; ties by match_no ascending',
        anchor='max(P(H),P(A)); ties choose H',selected_match_nos=f.match_no.iloc[selected].astype(int).tolist(),
        ticket_count=16,unit_yen=100,cost_yen=1600,model_P13=float(mass.sum()),
        candidate00_is_reference=True,candidate00_equals_ticket01=True,candidate00_additional_purchase=False,
        probabilities_changed=False,results_used=False,structural_correction_used=False,
        prediction_assumption='independent matches',D_counts=(tickets==1).sum(1).tolist(),
        fixed_nine_matches=f.match_no[singles].astype(int).tolist(),duplicate_tickets=0)
    return plan,diagnostics,audit
