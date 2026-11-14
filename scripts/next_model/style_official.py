"""Official deployment of learned style residuals; untrained means zero correction."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, sha256
from expected_goals import score_distribution
from style_observation import STATE, record_forecast, verify

CONFIG = ROOT / 'data/models/style_residual_deployment_v1.json'


def promote(baseline, comparison, policy):
    """Retain all baseline stage values; replace only final forecasts when enabled."""
    if 'style_official_version' in baseline:
        raise ValueError('style correction already applied')
    if comparison.source_match_id.duplicated().any() or baseline.match_id.duplicated().any():
        raise ValueError('duplicate official fixture')
    c=comparison.set_index('source_match_id').reindex(baseline.match_id)
    if c.shadow_prob_home.isna().any():raise ValueError('incomplete style forecast coverage')
    out=baseline.copy()
    for side in ('home','away'):
        out[f'pre_style_lambda_{side}']=baseline[f'final_lambda_{side}']
    for side in ('home','draw','away'):
        out[f'pre_style_prob_{side}']=baseline[f'next_prob_{side}']
    enabled=bool(policy['enabled'])
    trained=c.style_model_enabled.astype(str).str.lower().eq('true').to_numpy()
    use=trained & enabled
    for side in ('home','away'):
        out[f'final_lambda_{side}']=np.where(use,c[f'shadow_lambda_{side}'].to_numpy(),baseline[f'final_lambda_{side}'].to_numpy())
        out[f'style_lambda_delta_{side}']=out[f'final_lambda_{side}']-out[f'pre_style_lambda_{side}']
    for side in ('home','draw','away'):
        out[f'next_prob_{side}']=np.where(use,c[f'shadow_prob_{side}'].to_numpy(),baseline[f'next_prob_{side}'].to_numpy())
        out[f'style_prob_delta_{side}']=out[f'next_prob_{side}']-out[f'pre_style_prob_{side}']
    out['final_lambda_total']=out.final_lambda_home+out.final_lambda_away
    changed=out.style_lambda_delta_home.ne(0)|out.style_lambda_delta_away.ne(0)
    for position in np.flatnonzero(changed.to_numpy()):
        i=out.index[position];r=out.iloc[position]
        dist=score_distribution(r.final_lambda_home,r.final_lambda_away)
        for side in ('home','draw','away'):
            out.loc[i,f'next_prob_pre_uncertainty_{side}']=dist[f'prob_{side}']
        before=np.array([dist[f'prob_{s}'] for s in ('home','draw','away')])
        after=np.array([r[f'next_prob_{s}'] for s in ('home','draw','away')])
        out.loc[i,'next_uncertainty_entropy_delta']=float(-(after*np.log(np.clip(after,1e-12,1))).sum()+(before*np.log(np.clip(before,1e-12,1))).sum())
        if 'next_legacy_mix' in out:
            legacy_mix=float(r.next_legacy_mix)
            for side in ('home','draw','away'):
                out.loc[i,f'next_prob_legacy_mix_{side}']=(1-legacy_mix)*dist[f'prob_{side}']+legacy_mix/3
        for score in ('0_0','1_1','2_2'):
            out.loc[i,f'next_prob_{score}']=dist[f'prob_{score}']
    out['style_official_version']='style_residual_deployment_v1'
    out['style_official_enabled']=enabled
    out['style_official_model_enabled']=use
    out['style_official_status']=c.style_shadow_status.to_numpy() if enabled else 'disabled_by_policy'
    out['style_model_cutoff']=c.style_model_cutoff.to_numpy()
    out['buyplan_probability_source']=out.get('buyplan_probability_source',pd.Series('next_score_distribution_HDA',index=out.index)).astype(str)+' + style_residual_deployment_v1'
    p=out[[f'next_prob_{s}' for s in ('home','draw','away')]].to_numpy(float)
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):raise ValueError('invalid official probabilities')
    return out


def apply_official(baseline, season, config_path=CONFIG, root=STATE, data_dir=ROOT/'data', at=None):
    policy=json.loads(Path(config_path).read_text())
    if policy.get('version')!='style_residual_deployment_v1' or not isinstance(policy.get('enabled'),bool):
        raise ValueError('invalid style deployment policy')
    observation=record_forecast(baseline,season,root,data_dir,at=at,deployment_policy=policy)
    if observation['status']!='saved':raise ValueError('could not freeze a prematch official prediction')
    folder=Path(observation['path']);verify(folder)
    comparison=pd.read_csv(folder/'comparison.csv',low_memory=False,float_precision='round_trip')
    out=promote(baseline,comparison,policy)
    audit={**observation,'deployment':'official','policy':policy,'policy_sha256':sha256(Path(config_path)),
           'model_sha256':sha256(folder/'model.json'),'corrected_matches':int((out.style_lambda_delta_home.ne(0)|out.style_lambda_delta_away.ne(0)).sum()),
           'model_enabled':bool(out.style_official_model_enabled.any())}
    out['style_observation_path']=str(folder)
    return out,audit
