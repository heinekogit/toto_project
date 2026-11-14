"""Research-only residual Poisson model; never writes production predictions.

All naive timestamps are Japan time. Source dates without acquisition times
become usable the following day. Missing features contribute exactly zero.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from common import normalize_text
from expected_goals import score_distribution, LAMBDA_MIN, LAMBDA_MAX

VERSION = 'style_residual_shadow_v1'


def timestamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError('missing timestamp')
    return t.tz_localize('Asia/Tokyo') if t.tzinfo is None else t.tz_convert('Asia/Tokyo')


def build_examples(predictions, styles):
    required = ['match_id', 'datetime', 'forecast_created_at', 'home_team', 'away_team',
                'final_lambda_home', 'final_lambda_away']
    if any(c not in predictions for c in required):
        raise ValueError(f'predictions require {required}')
    if predictions.match_id.duplicated().any():
        raise ValueError('duplicate prediction match_id')
    styles = styles.copy()
    styles['team'] = styles.team.map(normalize_text)
    styles['usable_at'] = styles.apply(lambda r: timestamp(r.first_observed_at) if pd.notna(r.get('first_observed_at')) and str(r.get('first_observed_at','')).strip() else timestamp(r.first_observed_date).normalize() + pd.Timedelta(days=1), axis=1) if len(styles) else pd.Series(dtype='datetime64[ns, Asia/Tokyo]')
    keys = ['team', 'style', 'metric', 'usable_at']
    if styles.duplicated(keys).any():
        raise ValueError('duplicate style observations')
    metric_keys = sorted(set(zip(styles['style'], styles.metric)))
    rows = []
    for _, game in predictions.iterrows():
        created, kickoff = timestamp(game.forecast_created_at), timestamp(game.datetime)
        if created >= kickoff:
            raise ValueError('prediction must be created before kickoff')
        available = styles[styles.usable_at <= created].sort_values('usable_at').drop_duplicates(['team','style','metric'], keep='last')
        for side, other in [('home', 'away'), ('away', 'home')]:
            lam = float(game[f'final_lambda_{side}'])
            if not np.isfinite(lam) or lam <= 0:
                raise ValueError('invalid baseline lambda')
            row = dict(match_id=game.match_id, side=side, datetime=kickoff.isoformat(),
                       forecast_created_at=created.isoformat(), baseline_lambda=lam,
                       source_available_at=available.usable_at.max().isoformat() if len(available) else '',
                       feature_version=VERSION)
            for role, team_side in [('own',side), ('opp',other)]:
                team = normalize_text(game[f'{team_side}_team'])
                values = available[available.team.eq(team)].set_index(['style','metric'])
                for style, metric in metric_keys:
                    value = np.nan
                    if (style,metric) in values.index:
                        item = values.loc[(style,metric)]
                        if str(item.selection_status).startswith('accepted_'):
                            value = float(item.value)
                    row[f'f|{role}|{style}|{metric}'] = value
            # Attack characteristics already stored with the baseline forecast.
            # Interaction coefficients/signs are learned, never assigned by hand.
            for attack, defence in [('long_counter','ハイブロック'),
                                    ('short_counter','ハイプレッシング'),
                                    ('enemy_possession','ローブロック'),
                                    ('own_possession','カウンタープレス')]:
                a = float(game.get(f'flab_style_{attack}_index_{side}', np.nan))
                d = row.get(f'f|opp|{defence}|指数', np.nan)
                row[f'f|interaction|{attack}|{defence}'] = ((a-50)/10)*((d-50)/10)
            rows.append(row)
    return pd.DataFrame(rows)


def matrix(frame, columns, mean=None, scale=None):
    raw = frame.reindex(columns=columns).to_numpy(float)
    valid = np.isfinite(raw)
    if mean is None:
        count = valid.sum(axis=0)
        mean = np.divide(np.where(valid,raw,0).sum(axis=0), count, out=np.zeros(len(columns)), where=count>0)
        var = np.divide(np.where(valid,(raw-mean)**2,0).sum(axis=0), count, out=np.zeros(len(columns)), where=count>0)
        scale = np.sqrt(var)
        scale[scale < 1e-8] = 1
    return np.where(valid,(raw-mean)/scale,0), mean, scale


def fit_coefficients(frame, columns, alpha):
    x, mean, scale = matrix(frame, columns)
    y, base = frame.actual_goals.to_numpy(float), frame.baseline_lambda.to_numpy(float)
    def objective(beta):
        linear = np.log(base) + x @ beta
        bounded = np.clip(linear, -30, 30)
        mu = np.exp(bounded)
        loss = np.mean(mu-y*bounded) + alpha*np.dot(beta,beta)/2
        grad = x.T @ ((mu-y)*((linear>-30)&(linear<30))) / len(y) + alpha*beta
        return loss, grad
    fitted = minimize(objective, np.zeros(len(columns)), jac=True, method='L-BFGS-B')
    if not fitted.success:
        raise ValueError(f'fit failed: {fitted.message}')
    return dict(columns=columns, mean=mean.tolist(), scale=scale.tolist(), beta=fitted.x.tolist(), alpha=alpha)


def corrected(frame, model):
    if not model or not model.get('enabled',False):
        return frame.baseline_lambda.to_numpy(float), np.zeros((len(frame),0))
    if model['version'] != VERSION or not frame.feature_version.eq(VERSION).all():
        raise ValueError('feature version mismatch')
    cutoff = timestamp(model['cutoff'])
    if any(timestamp(x)<cutoff for x in frame.forecast_created_at):
        raise ValueError('model was fitted after this forecast')
    x, _, _ = matrix(frame, model['columns'], np.asarray(model['mean']), np.asarray(model['scale']))
    terms = x*np.asarray(model['beta'])
    base = frame.baseline_lambda.to_numpy(float)
    delta = terms.sum(axis=1)
    mu = np.clip(base*np.exp(np.clip(delta,-30,30)), LAMBDA_MIN,LAMBDA_MAX)
    # Preserve the baseline exactly when no correction is applied.
    mu[delta==0] = base[delta==0]
    return mu, terms


def nll(y, mu):
    return mu-y*np.log(mu)+gammaln(y+1)


def train(examples, results, cutoff, min_matches=20):
    cutoff = timestamp(cutoff)
    if results.match_id.duplicated().any() or examples.duplicated(['match_id','side']).any():
        raise ValueError('duplicate training match/side')
    if not examples.feature_version.eq(VERSION).all():
        raise ValueError('feature version mismatch')
    joined = examples.merge(results[['match_id','home_score','away_score','result_available_at']],on='match_id',validate='many_to_one')
    joined['actual_goals'] = np.where(joined.side.eq('home'),joined.home_score,joined.away_score).astype(float)
    if not np.isfinite(joined.actual_goals).all() or (joined.actual_goals<0).any() or (joined.actual_goals%1!=0).any():
        raise ValueError('invalid goal observations')
    eligible=[]
    for _, r in joined.iterrows():
        created,kickoff,known=map(timestamp,[r.forecast_created_at,r.datetime,r.result_available_at])
        if not created<kickoff<known:
            raise ValueError('invalid forecast/kickoff/result chronology')
        if r.source_available_at and pd.notna(r.source_available_at) and timestamp(r.source_available_at)>created:
            raise ValueError('future style source')
        eligible.append(known<cutoff)
    joined=joined.loc[eligible].copy()
    columns=sorted(c for c in examples if c.startswith('f|'))
    joined=joined[joined[columns].notna().any(axis=1)]
    model=dict(version=VERSION,cutoff=cutoff.isoformat(),enabled=False,reason='insufficient_prematch_training_data',train_matches=int(joined.match_id.nunique()))
    days=sorted(joined.datetime.map(lambda x:timestamp(x).normalize()).unique())
    if joined.match_id.nunique()<min_matches or len(days)<3:
        return model
    split=days[max(1,int(len(days)*.7))]
    before=joined[joined.datetime.map(timestamp)<split]
    after=joined[joined.datetime.map(timestamp)>=split]
    candidates=[]
    baseline=float(nll(after.actual_goals,after.baseline_lambda).mean())
    for alpha in (.1,1.,10.):
        fitted=fit_coefficients(before,columns,alpha)
        fitted.update(version=VERSION,enabled=True,cutoff=split.isoformat())
        # Validation forecast itself must postdate completion of all training results.
        if any(timestamp(x)>=min(after.forecast_created_at.map(timestamp)) for x in before.result_available_at):
            return {**model,'reason':'no_nonoverlapping_time_holdout'}
        fitted['cutoff']=min(after.forecast_created_at.map(timestamp)).isoformat()
        mu,_=corrected(after,fitted)
        candidates.append(dict(alpha=alpha,goal_nll=float(nll(after.actual_goals,mu).mean())))
    best=min(candidates,key=lambda x:x['goal_nll'])
    model.update(validation=candidates,validation_baseline_nll=baseline)
    if best['goal_nll']>=baseline:
        return {**model,'reason':'time_holdout_prefers_baseline'}
    model.update(fit_coefficients(joined,columns,best['alpha']))
    return {**model,'enabled':True,'reason':'selected_on_time_holdout','last_training_result':max(joined.result_available_at.map(timestamp)).isoformat(),'training_match_ids':sorted(joined.match_id.astype(str).unique().tolist()),'validation_start':split.isoformat()}


def predict(predictions, examples, model=None):
    mu,terms=corrected(examples,model)
    detail=examples.copy();detail['shadow_lambda']=mu
    detail['raw_log_correction']=terms.sum(axis=1)
    detail['effective_log_correction']=np.log(mu/detail.baseline_lambda.to_numpy(float))
    detail['available_feature_count']=examples.filter(regex=r'^f\|').notna().sum(axis=1)
    detail['contributions_json']=[json.dumps(dict(zip(model['columns'],row)),ensure_ascii=False) if model and model.get('enabled') else '{}' for row in terms]
    out=predictions.copy()
    indexed=detail.set_index(['match_id','side'])
    for i,g in out.iterrows():
        pair=[indexed.loc[(g.match_id,s)] for s in ('home','away')]
        for side,r in zip(('home','away'),pair):
            out.loc[i,f'shadow_lambda_{side}']=r.shadow_lambda
            out.loc[i,f'style_lambda_delta_{side}']=r.shadow_lambda-r.baseline_lambda
        changed=any(r.shadow_lambda!=r.baseline_lambda for r in pair)
        if changed:
            mix=float(g['next_uncertainty_calibration_mix'])
            if not 0<=mix<=1:raise ValueError('invalid uncertainty mix')
            dist=score_distribution(pair[0].shadow_lambda,pair[1].shadow_lambda)
        for s in ('home','draw','away'):
            p=float(g[f'next_prob_{s}'])
            q=(1-mix)*dist[f'prob_{s}']+mix/3 if changed else p
            out.loc[i,f'shadow_prob_{s}']=q
            out.loc[i,f'style_prob_delta_{s}']=q-p
    out['style_model_enabled']=bool(model and model.get('enabled'))
    out['style_model_cutoff']=model.get('cutoff','') if model else ''
    out['style_model_version']=VERSION
    out['style_shadow_status']=model.get('reason','untrained') if model else 'untrained'
    return out,detail


def evaluate(comparison, results):
    if comparison.match_id.duplicated().any() or results.match_id.duplicated().any():raise ValueError('duplicate evaluation matches')
    if any(timestamp(r.forecast_created_at)>=timestamp(r.datetime) for _,r in comparison.iterrows()):raise ValueError('postmatch evaluation forecast')
    d=comparison.merge(results[['match_id','home_score','away_score']],on='match_id',suffixes=('','_actual'),validate='one_to_one')
    h=d['home_score_actual'] if 'home_score_actual' in d else d.home_score
    a=d['away_score_actual'] if 'away_score_actual' in d else d.away_score
    d=d[h.notna() & a.notna()].copy();h=h.loc[d.index];a=a.loc[d.index]
    outcome=np.where(h>a,0,np.where(h==a,1,2))
    scores={}
    for label,lam,prob in [('baseline','final_lambda','next_prob'),('shadow','shadow_lambda','shadow_prob')]:
        p=d[[f'{prob}_{s}' for s in ('home','draw','away')]].to_numpy(float)
        if len(d) and (not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1)):raise ValueError('invalid probabilities')
        scores[label]=dict(matches=len(d),goal_mae=float(((d[f'{lam}_home']-h).abs()+(d[f'{lam}_away']-a).abs()).mean()/2) if len(d) else None,log_loss=float(-np.log(np.maximum(p[np.arange(len(d)),outcome],1e-15)).mean()) if len(d) else None,brier=float(((p-np.eye(3)[outcome])**2).sum(axis=1).mean()) if len(d) else None)
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    pred=sub.add_parser('predict');pred.add_argument('--predictions',required=True);pred.add_argument('--styles',required=True);pred.add_argument('--forecast-created-at');pred.add_argument('--model');pred.add_argument('--out-dir',required=True)
    fit=sub.add_parser('train');fit.add_argument('--examples',required=True);fit.add_argument('--results',required=True);fit.add_argument('--cutoff',required=True);fit.add_argument('--model-out',required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--comparison',required=True);ev.add_argument('--results',required=True);ev.add_argument('--out',required=True)
    args=p.parse_args()
    if args.command=='predict':
        frame=pd.read_csv(args.predictions,low_memory=False)
        if args.forecast_created_at:frame['forecast_created_at']=args.forecast_created_at
        examples=build_examples(frame,pd.read_csv(args.styles))
        model=json.loads(Path(args.model).read_text()) if args.model else None
        output,detail=predict(frame,examples,model)
        folder=Path(args.out_dir);folder.mkdir(parents=True,exist_ok=True)
        output.to_csv(folder/'comparison.csv',index=False);detail.to_csv(folder/'examples.csv',index=False)
        print(json.dumps({'matches':len(output),'status':model.get('reason') if model else 'untrained','style_terms_available':int(examples.filter(regex=r'^f\|').notna().sum().sum())}))
    elif args.command=='train':
        model=train(pd.read_csv(args.examples),pd.read_csv(args.results),args.cutoff)
        Path(args.model_out).write_text(json.dumps(model,ensure_ascii=False,indent=2));print(model['reason'])
    else:
        scores=evaluate(pd.read_csv(args.comparison),pd.read_csv(args.results))
        Path(args.out).write_text(json.dumps(scores,ensure_ascii=False,indent=2));print(json.dumps(scores))


if __name__=='__main__':main()
