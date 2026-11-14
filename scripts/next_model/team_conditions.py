"""Shared side-preserving conditions. No fitted defaults; activation requires calibration.

Input -> team-specific attack/defence effects -> lambdas -> score matrix ->
optional, separately attributed uncertainty. Absences already in saved final
lambdas are never applied again. Effects are log-rate changes, not probabilities.
"""
from dataclasses import dataclass, field, asdict
import math
from typing import Mapping
import numpy as np
import pandas as pd
from expected_goals import score_distribution, LAMBDA_MIN, LAMBDA_MAX


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def flag(value):
    value = str(value).strip().lower()
    return True if value in {'true','1','1.0'} else False if value in {'false','0','0.0'} else None


def observed_before_kickoff(observed, kickoff):
    """Unknown is not evidence of a prematch observation."""
    try:
        a=pd.Timestamp(observed);b=pd.Timestamp(kickoff)
        if pd.isna(a) or pd.isna(b):return 'unknown'
        if a.tzinfo is None:a=a.tz_localize('Asia/Tokyo')
        if b.tzinfo is None:b=b.tz_localize('Asia/Tokyo')
        return 'prematch' if a<b else 'not_prematch'
    except (ValueError,TypeError):return 'unknown'


@dataclass(frozen=True)
class TeamInput:
    team: str
    fatigue: float | None
    rest_hours: float | None
    travel_km: float | None
    travel_load: float | None
    carry_load: float | None
    external_load_unknown: bool | None
    absence_missing: bool | None
    absence_attack: float | None
    absence_defense: float | None
    absence_observed_at: str | None
    weather_sensitivity: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class MatchInput:
    home: TeamInput
    away: TeamInput
    weather: Mapping
    coverage: Mapping

    @property
    def fatigue_sum(self):
        return None if self.home.fatigue is None or self.away.fatigue is None else self.home.fatigue+self.away.fatigue

    @property
    def fatigue_difference(self):
        return None if self.home.fatigue is None or self.away.fatigue is None else self.home.fatigue-self.away.fatigue


def read_inputs(row, weather_sensitivities=None):
    """Missing stays None. A measured zero is not silently substituted for unknown."""
    sensitivities = weather_sensitivities or {}
    def side(name):
        fatigue=number(row.get(name+'_total_fatigue_score'))
        # Do not fall back when an explicitly supplied total is missing.
        if name+'_total_fatigue_score' not in row:
            fatigue=number(row.get(name+'_fatigue_score'))
        if fatigue is not None and fatigue<0: raise ValueError('negative fatigue')
        return TeamInput(str(row.get(name+'_team','unknown')),fatigue,
            number(row.get(name+'_rest_hours')),number(row.get(name+'_travel_distance_km')),
            number(row.get(name+'_travel_fatigue')),number(row.get(name+'_recent_load_carry')),
            flag(row.get(name+'_external_load_unknown')),flag(row.get('absence_missing_'+name)),
            number(row.get('absence_effective_attack_'+name)),number(row.get('absence_effective_defense_'+name)),
            row.get('absence_observed_at_'+name),sensitivities.get(name,{}))
    weather={k:row.get(k) for k in ['weather_missing','temp_kickoff','precip_kickoff','wind_kickoff','is_rain','is_heavy_rain','is_strong_wind','stadium_name','lat','lon','__weather_asof','weather_data_kind','last_updated_at']}
    weather['known']=flag(row.get('weather_missing')) is False
    return MatchInput(side('home'),side('away'),{**weather},
        {**{k:row.get(k) for k in ['fatigue_measurement_version','external_schedule_coverage','travel_lookup_status','travel_distance_basis']},
         'weather_observation_time_status':observed_before_kickoff(row.get('last_updated_at'),row.get('datetime')),
         **{'absence_'+side+'_observation_time_status':observed_before_kickoff(row.get('absence_observed_at_'+side),row.get('datetime')) for side in ['home','away']}})


@dataclass(frozen=True)
class Effect:
    cause: str
    side: str
    attack_log_change: float
    defense_concession_log_change: float
    evidence: str


@dataclass(frozen=True)
class FatigueParameters:
    """Explicit research parameters only. No default numeric coefficient."""
    attack_loss_per_unit: float
    defense_loss_per_unit: float
    evidence: str

    def __post_init__(self):
        if not self.evidence or not all(math.isfinite(x) and x>=0 for x in [self.attack_loss_per_unit,self.defense_loss_per_unit]):
            raise ValueError('explicit nonnegative parameters and provenance required')


def estimate_effects(inputs, *, fatigue_parameters=None, include_weather=False):
    effects=[];pending=[]
    if fatigue_parameters is None:
        pending.append('fatigue: coefficients uncalibrated; no directional lambda change')
    else:
        for side in ['home','away']:
            team=getattr(inputs,side)
            if team.fatigue is None:raise ValueError(f'{side}: unknown fatigue cannot become zero')
            if team.external_load_unknown is True:
                raise ValueError(f'{side}: known missing external workload; cannot treat total as complete')
            effects.append(Effect('fatigue',side,-fatigue_parameters.attack_loss_per_unit*team.fatigue,
                                  fatigue_parameters.defense_loss_per_unit*team.fatigue,fatigue_parameters.evidence))
    if include_weather:
        if not inputs.weather['known']:raise ValueError('weather unknown')
        if inputs.coverage.get('weather_observation_time_status')=='not_prematch':raise ValueError('postmatch weather cannot be a prematch input')
        # Each supplied slope has units per weather measurement. No invented team contrast.
        for side in ['home','away']:
            team=getattr(inputs,side)
            if not team.weather_sensitivity:raise ValueError(f'{side}: weather sensitivity uncalibrated')
            attack=defense=0.;evidence=[]
            for metric,spec in team.weather_sensitivity.items():
                if metric not in {'temp_kickoff','precip_kickoff','wind_kickoff'}:raise ValueError('unsupported exposure')
                x=number(inputs.weather.get(metric))
                if x is None:raise ValueError('missing weather exposure')
                vals=[number(spec.get(k)) for k in ['reference','attack_slope','defense_slope']]
                if any(v is None for v in vals) or not spec.get('evidence'):raise ValueError('explicit weather coefficients required')
                ref,a,b=vals;attack+=(x-ref)*a;defense+=(x-ref)*b;evidence.append(spec['evidence'])
            effects.append(Effect('weather',side,attack,defense,';'.join(evidence)))
    else:pending.append('weather: common observations retained; side sensitivities not calibrated')
    return effects,pending


def apply_team_effects(home_lambda, away_lambda, effects, *, already_applied=('absence',)):
    """Own attack changes own lambda; own defensive concession changes opponent lambda."""
    if not all(math.isfinite(v) and v>0 for v in [home_lambda,away_lambda]):raise ValueError('invalid baseline lambdas')
    changes={'home':0.,'away':0.};seen=set()
    for e in effects:
        if e.side not in changes or e.cause not in {'fatigue','travel','weather','absence'}:raise ValueError('unknown side/cause')
        if e.cause in already_applied:raise ValueError('cause already included in baseline: '+e.cause)
        if (e.cause,e.side) in seen:raise ValueError('duplicate effect')
        if not e.evidence or not all(math.isfinite(x) for x in [e.attack_log_change,e.defense_concession_log_change]):raise ValueError('invalid effect')
        seen.add((e.cause,e.side));changes[e.side]+=e.attack_log_change
        changes['away' if e.side=='home' else 'home']+=e.defense_concession_log_change
    # Aggregate fatigue already contains travel: reject component double counting.
    if any(c=='fatigue' for c,_ in seen) and any(c=='travel' for c,_ in seen):raise ValueError('travel already in aggregate fatigue')
    requested=[math.log(home_lambda)+changes['home'],math.log(away_lambda)+changes['away']]
    bounded=np.clip(requested,math.log(LAMBDA_MIN),math.log(LAMBDA_MAX));h,a=np.exp(bounded)
    distribution=score_distribution(h,a)
    return dict(lambda_home=float(h),lambda_away=float(a),raw_probabilities=[distribution['prob_home'],distribution['prob_draw'],distribution['prob_away']],
                log_lambda_change=changes,clipped=bool(not np.allclose(requested,bounded)),effects=[asdict(e) for e in effects])


def apply_research_mix(probabilities, *, fatigue_effect=0.,weather_effect=0.,max_mix=.25,strength=1.,directional_causes=()):
    """Legacy comparison only; component effects supplied explicitly, not calibrated here."""
    p=np.asarray(probabilities,float)
    if p.shape!=(3,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):raise ValueError('invalid probabilities')
    if not all(math.isfinite(x) and 0<=x<=1 for x in [fatigue_effect,weather_effect,max_mix,strength]):raise ValueError('invalid mix setting')
    if strength>0:
        if fatigue_effect>0 and set(directional_causes)&{'fatigue','travel'}:raise ValueError('fatigue/travel would be counted in both lambda and mix')
        if weather_effect>0 and 'weather' in directional_causes:raise ValueError('weather would be counted twice')
    mix=strength*min(1-(1-fatigue_effect)*(1-weather_effect),max_mix)
    return (1-mix)*p+mix/3,mix


def research_forecast(row, *, fatigue_parameters=None,weather_sensitivities=None,include_weather=False,mix_strength=0.):
    """Adapter from saved post-absence lambdas. Does not call or modify production."""
    inp=read_inputs(row,weather_sensitivities)
    effects,pending=estimate_effects(inp,fatigue_parameters=fatigue_parameters,include_weather=include_weather)
    result=apply_team_effects(float(row['final_lambda_home']),float(row['final_lambda_away']),effects)
    # Missing archived signal/config is never inferred as a no-weather/no-fatigue observation.
    if mix_strength>0:
        fields=['next_uncertainty_fatigue_signal','next_uncertainty_weather_signal']
        if any(number(row.get(k)) is None for k in fields):raise ValueError('mix comparison requires archived signals')
        fw=number(row.get('condition_fatigue_weight'));ww=number(row.get('condition_weather_weight'));cap=number(row.get('condition_max_mix'))
        if any(v is None for v in [fw,ww,cap]):raise ValueError('explicit comparison config required')
        fe=min(1.,fw*float(row[fields[0]]));we=min(1.,ww*float(row[fields[1]]))
    else:fe=we=0.;cap=.25
    p,mix=apply_research_mix(result['raw_probabilities'],fatigue_effect=fe,weather_effect=we,max_mix=cap,strength=mix_strength,directional_causes=[e.cause for e in effects])
    return {**result,'probabilities':p.tolist(),'mix':mix,'pending':pending,'team_inputs':asdict(inp),'fatigue_sum':inp.fatigue_sum,'fatigue_difference':inp.fatigue_difference,'production_applied':False}
