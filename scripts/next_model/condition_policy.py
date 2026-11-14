"""Production policy: side-preserving evidence, calibrated-only lambda hook, optional mix."""
import json,math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from team_conditions import read_inputs,number,flag,estimate_effects,apply_team_effects,FatigueParameters
from uncertainty_calibration import signals,adjust


def clean_json(value):
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if value is None or (not isinstance(value,(dict,list,tuple)) and pd.isna(value)):return None
    if isinstance(value,np.generic):return value.item()
    return value


def prepare_conditions(row):
    out=row.copy()
    for side in ['home','away']:
        total=number(row.get(side+'_total_fatigue_score',row.get(side+'_fatigue_score')))
        rest=number(row.get(side+'_rest_hours'))
        incomplete=flag(row.get(side+'_external_load_unknown')) is True or str(row.get(side+'_fatigue_status','')) in ['missing','partial','pending']
        # For old archives with unknown coverage retain the reported subtotal separately.
        out[side+'_fatigue_reported']=total
        travel_missing=(side=='away' and ('travel_lookup_status' in row and row.get('travel_lookup_status')!='hit'))
        incomplete=incomplete or travel_missing
        unknown=total is None or rest is None or incomplete
        out[side+'_condition_fatigue_status']='partial' if incomplete else 'missing' if total is None or rest is None else 'known_zero' if total==0 else 'known_load'
        out[side+'_total_fatigue_score']=np.nan if unknown else total
        out[side+'_fatigue_score']=np.nan if unknown else total
        if rest is None:out[side+'_rest_fatigue']=np.nan
        if travel_missing:out[side+'_travel_fatigue']=np.nan
        out[side+'_fatigue_adjustment_status']='held_missing_or_partial' if unknown else 'held_uncalibrated'
        missing=flag(row.get('absence_missing_'+side))
        if missing is not False:
            out['absence_missing_'+side]=True
            for name in ['effective_attack','effective_defense','effective_total','players_count']:
                out['absence_'+name+'_'+side]=np.nan
            out['absence_data_status_'+side]='missing' if missing is True else 'pending'
        else:out['absence_data_status_'+side]='known'
    weather_known=flag(row.get('weather_missing')) is False and all(flag(row.get(k)) is not None for k in ['is_rain','is_heavy_rain','is_strong_wind'])
    out['weather_missing']=not weather_known
    out['condition_weather_status']='known' if weather_known else 'missing_or_partial'
    if not weather_known:
        for k in ['is_rain','is_heavy_rain','is_strong_wind','temperature','wind_speed','temp_kickoff','precip_kickoff','wind_kickoff']:out[k]=np.nan
    return out


def team_lambda(row,home,away,config):
    settings=config.get('team_lambda',{'enabled':False})
    inputs=read_inputs(row,weather_sensitivities=settings.get('weather_sensitivities'))
    params=None;include_weather=False
    if settings.get('enabled'):
        if settings.get('calibration_status')!='validated' or not settings.get('validation_artifact') or not (settings.get('parameters') or settings.get('weather_sensitivities')):
            raise ValueError('team lambda parameters require validated calibration and evidence')
        if not Path(settings['validation_artifact']).is_file():
            raise ValueError('team lambda validation artifact is unavailable')
        if float(config.get('mix_strength',1))!=0:raise ValueError('team lambda and legacy fatigue mix cannot both be enabled')
        params=FatigueParameters(**settings['parameters']) if settings.get('parameters') else None
        include_weather=bool(settings.get('weather_sensitivities'))
    pending=[];effects=[]
    if params is not None or include_weather:
        try:effects,pending=estimate_effects(inputs,fatigue_parameters=params,include_weather=include_weather)
        except ValueError as exc:pending=['held_missing_input: '+str(exc)]
    else:pending=['fatigue/travel/weather coefficients uncalibrated; lambda correction disabled']
    result=apply_team_effects(home,away,effects) if effects else {'lambda_home':home,'lambda_away':away,'effects':[],'clipped':False}
    audit={'team_condition_inputs_json':json.dumps(clean_json(asdict(inputs)),ensure_ascii=False,allow_nan=False,default=str),
       'team_condition_effects_json':json.dumps(result['effects'],ensure_ascii=False),
       'team_condition_status':'applied' if effects else ';'.join(pending),
       'team_fatigue_sum':inputs.fatigue_sum,'team_fatigue_difference':inputs.fatigue_difference,
       'team_lambda_delta_home':result['lambda_home']-home,'team_lambda_delta_away':result['lambda_away']-away,
       'team_condition_clipped':result['clipped']}
    return result['lambda_home'],result['lambda_away'],audit


def final_policy(frame,config):
    out=frame.copy();strength=float(config.get('mix_strength',1.))
    if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('mix_strength must be in [0,1]')
    rows=[]
    for _,r in out.iterrows():
        z=r.to_dict();p=np.array([r['next_prob_'+s] for s in ['home','draw','away']]);assert np.isclose(p.sum(),1)
        fh=number(r.get('home_total_fatigue_score'));fa=number(r.get('away_total_fatigue_score'))
        weather_known=flag(r.get('weather_missing')) is False
        fatigue_known=fh is not None and fa is not None
        # Unknown is retained as NaN. A held component has zero *effect*, not zero observed load.
        f=(min(1.,max(0.,(fh+fa)/16))) if fatigue_known else np.nan
        if weather_known:
            _,w=signals([0],[0],[flag(r.get('is_rain'))],[flag(r.get('is_heavy_rain'))],[flag(r.get('is_strong_wind'))],config.get('rain_signal',.35),config.get('heavy_rain_signal',1.),config.get('strong_wind_signal',.65));w=float(w[0])
        else:w=np.nan
        shadow,m=adjust(p[None,:],[f if fatigue_known else 0.],[w if weather_known else 0.],float(config['fatigue_weight']),float(config['weather_weight']),float(config['max_mix']))
        actual_mix=float(m[0])*strength;final=(1-actual_mix)*p+actual_mix/3
        for i,s in enumerate(['home','draw','away']):
            z['next_prob_pre_uncertainty_'+s]=float(p[i]);z['next_prob_'+s]=float(final[i]);z['next_prob_legacy_mix_'+s]=float(shadow[0,i])
        z['next_uncertainty_entropy_delta']=float(-(final*np.log(np.clip(final,1e-15,1))).sum()+(p*np.log(np.clip(p,1e-15,1))).sum())
        z.update(next_uncertainty_calibration_mix=actual_mix,next_uncertainty_calibration_version=config['version'],next_uncertainty_fatigue_signal=f,next_uncertainty_weather_signal=w,
          next_uncertainty_weather_known=weather_known,next_uncertainty_fatigue_known=fatigue_known,next_legacy_mix=float(m[0]),
          next_legacy_mix_status='complete' if fatigue_known and weather_known else 'partial_components_held',
          next_condition_policy_version=config['version'],next_condition_effective_at=config.get('effective_at','legacy_config'),next_condition_change_reason=config.get('change_reason','legacy_config'),condition_mix_strength=strength)
        rows.append(z)
    return pd.DataFrame(rows,index=frame.index)
