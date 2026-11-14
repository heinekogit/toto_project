"""Read-only production input audit; outputs only independent Phase39 research artifacts."""
from pathlib import Path
import sys,json,hashlib,unicodedata,itertools,html
import numpy as np
import pandas as pd
from phase38_ha_diagnostics import raw
from expected_goals import base_lambdas_v1
from lab_matchup import apply_lab_matchup
from absence_adjustment import apply_absence_to_lambdas
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from uncertainty_calibration import signals,adjust
OUT=ROOT/'data/reports/score_structure_phase39';PRE=ROOT/'data/reports/score_structure_phase38'
ROUNDS=[1660,1663,1664,1665,1668,1670]
def norm(s):return unicodedata.normalize('NFKC',str(s)).replace(' ','')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def games_candidates(total,rate,mode='nearest'):
 # Sensitivity assumption only: both displayed to 0.1. Not a claim about publisher rounding.
 low,high=(rate-.05,rate+.05) if mode=='nearest' else (rate,rate+.1)
 return [n for n in range(1,61) if (total-.05)/n<high+1e-12 and (total+.05)/n>low-1e-12]
def loss(p,y):return -np.log(p[y]),float(((p-np.eye(3)[y])**2).sum())
def boot_round(df,col):
 v=df.groupby('round')[col].mean().to_numpy();a=np.array([np.mean(v[list(ix)]) for ix in itertools.product(range(len(v)),repeat=len(v))]);return np.quantile(a,[.025,.975]).tolist()
def run():
 OUT.mkdir(parents=True,exist_ok=True)
 hashes={}; protected={str(p):sha(p) for p in list((ROOT/'scripts/next_model').glob('*.py')) if not p.name.startswith('phase39')}
 protected[str(ROOT/'scripts/uncertainty_calibration.py')]=sha(ROOT/'scripts/uncertainty_calibration.py')
 d=pd.read_csv(PRE/'match_diagnostics.csv');stages=pd.read_csv(PRE/'stage_probabilities.csv')
 oldaudit=json.loads((PRE/'audit.json').read_text());assert all(sha(p)==h for p,h in oldaudit['source_hashes'].items())
 qa=[];mix=[];checks=[];corrections=[];sensitivity=[]
 for rnd in ROUNDS:
  path=ROOT/f'data/eval/toto_rounds/toto{rnd}/snapshot/next_predictions.csv';hashes[str(path)]=sha(path);snap=pd.read_csv(path)
  for _,r in snap.iterrows():
   meta=d[(d['round']==f'toto{rnd}')&(d.match_no==r.match_no)].iloc[0];y={'H':0,'D':1,'A':2}[meta.actual_symbol]
   identity=dict(round=f'toto{rnd}',match_no=int(r.match_no),card=meta.card,model_signature=meta.model_signature,actual=meta.actual_symbol)
   date=str(r.stats_asof).replace('-','');league=str(r.match_id)[:2]
   sp=ROOT/'data/stats_snapshots'/str(r.stats_source_csv);rk=ROOT/f'data/{league}_2026_rankings_{date}.csv'
   for p in [sp,rk]:
    if p.exists():hashes[str(p)]=sha(p)
   stats=pd.read_csv(sp);ranking=pd.read_csv(rk);nums={}
   for side in ['home','away']:
    team=r[side+'_team'];rank=ranking[ranking['チーム'].map(lambda s:norm(s.split()[-1])==norm(team))];assert len(rank)==1
    rank=rank.iloc[0];fullname=' '.join(rank['チーム'].split()[:-1]);ss=stats[stats.team_name.map(norm)==norm(fullname)]
    if len(ss)!=1: ss=stats[stats.team_name.map(norm)==norm(team)]
    total=float(r[f'stats_被ゴール期待値_{side}']);rate=float(r[f'stats_1試合平均被ゴール期待値_{side}']);xg=float(r[f'stats_ゴール期待値_{side}']);n=round(total/rate);nrank=int(rank['試合']);nums[side]=nrank
    matchvals={f:bool(len(ss)==1 and np.isclose(float(ss.iloc[0][f]),r[f'stats_{f}_{side}'])) for f in ['ゴール期待値','被ゴール期待値','1試合平均被ゴール期待値']}
    qa.append(dict(**identity,side=side,team=team,league=league,stats_file=str(sp.relative_to(ROOT)),standings_file=str(rk.relative_to(ROOT)),
     stats_asof=r.stats_asof,kickoff=r.datetime,asof_not_after_kickoff=pd.Timestamp(r.stats_asof)<=pd.Timestamp(r.datetime),competition=rank.get('competition_key','unknown'),
     xg_total=xg,xga_total=total,xga_display_rate=rate,inferred_games=n,standings_games=nrank,count_mismatch=n!=nrank,
     nearest_candidates=','.join(map(str,games_candidates(total,rate))),floor_candidates=','.join(map(str,games_candidates(total,rate,'floor'))),
     snapshot_xg_matches=matchvals['ゴール期待値'],snapshot_xga_matches=matchvals['被ゴール期待値'],snapshot_rate_matches=matchvals['1試合平均被ゴール期待値'],
     source_team_matched=len(ss)==1,xga_rate_if_standings=total/nrank,xg_rate_current=xg/n,xg_rate_if_standings=xg/nrank,
     published_goal_rate=r[f'stats_1試合平均得点数_{side}'],standings_goals=rank['得点'],standings_conceded=rank['失点'],
     window_start='season label only; exact xG coverage not saved',window_end='snapshot date; metric cutoff not saved',
     xg_coverage_confirmed=False,corrected_input_adopted=False))
   bh,ba,_=base_lambdas_v1(r);assert np.allclose([bh,ba],[r.base_lambda_home,r.base_lambda_away])
   lab=apply_lab_matchup(r,bh,ba,weight=float(r.get('lab_adjustment_weight',.25)))
   assert np.allclose([lab['final_lambda_home'],lab['final_lambda_away']],[r.get('lab_final_lambda_home',r.final_lambda_home),r.get('lab_final_lambda_away',r.final_lambda_away)])
   checks.append(dict(**identity,check='LAB formula',passed=True))
   if 'absence_adjustment_version' in snap:
    ab=apply_absence_to_lambdas(r,r.get('lab_final_lambda_home',r.final_lambda_home),r.get('lab_final_lambda_away',r.final_lambda_away),goals_per_impact=float(r.get('condition_absence_goals_per_impact',.25)))
    assert np.allclose([ab['final_lambda_home'],ab['final_lambda_away']],[r.final_lambda_home,r.final_lambda_away])
    checks.append(dict(**identity,check='absence formula',passed=True))
   for side in ['home','away']:
    missing=r.get('absence_input_missing_'+side,np.nan);missing_bool=str(missing).lower() in ['true','1']
    corrections.append(dict(**identity,side=side,absence_applied='absence_adjustment_version' in snap,absence_missing=missing,
     absence_status='未適用・不明' if pd.isna(missing) else ('情報未取得' if missing_bool else ('取得済み・人数0' if r.get('absence_players_count_'+side,np.nan)==0 else '取得済み・人数あり/人数不明')),
     absence_count=r.get('absence_players_count_'+side,np.nan),absence_attack=r.get('absence_attack_used_'+side,np.nan),absence_defense=r.get('absence_defense_used_'+side,np.nan),
     absence_lambda_delta=r.get('absence_lambda_delta_'+side,np.nan),absence_asof=r.get('__absence_asof',r.get('absence_asof',np.nan)),
     lab_asof=r.flab_snapshot_date,lab_for=r['flab_expected_for_xg_'+side],lab_against=r['flab_expected_against_xg_'+side],
     goal_rate_version=r.get('goal_rate_adjustment_version',np.nan),goal_rate_lambda_delta=r.get('goal_rate_final_lambda_delta_'+side,np.nan)))
   before=raw(r.final_lambda_home,r.final_lambda_away);after=np.array([r.next_prob_home,r.next_prob_draw,r.next_prob_away]);m=float(meta.uncertainty_mix)
   assert np.allclose(after,(1-m)*before+m/3)
   f=w=np.nan;reproduced=None
   if m>0:
    # Use saved signals to test arithmetic; separately check actual source signal calculation.
    f=float(r.next_uncertainty_fatigue_signal);w=float(r.next_uncertainty_weather_signal)
    ap,mp=adjust(before[None,:],[f],[w],r.get('condition_fatigue_weight',.25),r.get('condition_weather_weight',.25),r.get('condition_max_mix',.25))
    reproduced=bool(np.allclose(ap[0],after) and np.isclose(mp[0],m));assert reproduced
    fp=np.clip((r.home_total_fatigue_score+r.away_total_fatigue_score)/16,0,1);assert np.isclose(fp,f)
   l0,b0=loss(before,y);l1,b1=loss(after,y)
   mix.append(dict(**identity,mix=m,fatigue_signal=f,weather_signal=w,weather_known=r.get('next_uncertainty_weather_known',np.nan),mix_formula_reproduced=reproduced,
    raw_H=before[0],raw_D=before[1],raw_A=before[2],final_H=after[0],final_D=after[1],final_A=after[2],p_actual_before=before[y],p_actual_after=after[y],
    log_loss_before=l0,log_loss_after=l1,delta_log_loss=l1-l0,brier_before=b0,brier_after=b1,delta_brier=b1-b0,
    entropy_before=-sum(before*np.log(before)),entropy_after=-sum(after*np.log(after)),improved=l1<l0-1e-12,worsened=l1>l0+1e-12))
   # Not a corrected forecast: denominator-only sensitivity under explicit unverified same-coverage assumption.
   rates={}
   for side in ['home','away']:
    rates[side]=.5*r[f'stats_1試合平均得点数_{side}']+.5*r[f'stats_ゴール期待値_{side}']/nums[side]
   defense={s:.5*(r[f'stats_1試合平均失点数_{s}']+r[f'stats_1試合平均被ゴール期待値_{s}']) for s in ['home','away']}
   nh=float(np.clip(np.sqrt(rates['home']*defense['away']),.15,4.5));na=float(np.clip(np.sqrt(rates['away']*defense['home']),.15,4.5))
   q=apply_lab_matchup(r,nh,na,weight=float(r.get('lab_adjustment_weight',.25)))
   if 'absence_adjustment_version' in snap:q=apply_absence_to_lambdas(r,q['final_lambda_home'],q['final_lambda_away'],goals_per_impact=float(r.get('condition_absence_goals_per_impact',.25)))
   cp=(1-m)*raw(q['final_lambda_home'],q['final_lambda_away'])+m/3
   sensitivity.append(dict(**identity,status='未確認分母による感度試算・修正予測ではない',base_H_before=bh,base_A_before=ba,base_H_if_count=nh,base_A_if_count=na,
    delta_base_H=nh-bh,delta_base_A=na-ba,final_lambda_H_before=r.final_lambda_home,final_lambda_A_before=r.final_lambda_away,
    final_lambda_H_if_count=q['final_lambda_home'],final_lambda_A_if_count=q['final_lambda_away'],
    delta_P_H=cp[0]-after[0],delta_P_D=cp[1]-after[1],delta_P_A=cp[2]-after[2]))
 q=pd.DataFrame(qa);mx=pd.DataFrame(mix);co=pd.DataFrame(corrections);sen=pd.DataFrame(sensitivity)
 agg=[]
 for dimension in ['round','model_signature','mix_band']:
  if dimension=='mix_band':mx['mix_band']=pd.cut(mx['mix'],[-1e-12,0,.2,.24999999,.25000001],labels=['0','0超〜0.20','0.20超〜0.25未満','0.25'])
  for k,g in mx.groupby(dimension,observed=True):agg.append(dict(dimension=dimension,group=k,n=len(g),mix_mean=g['mix'].mean(),mix_min=g['mix'].min(),mix_max=g['mix'].max(),improved=int(g.improved.sum()),worsened=int(g.worsened.sum()),LL_before=g.log_loss_before.mean(),LL_after=g.log_loss_after.mean(),delta_LL=g.delta_log_loss.mean(),Brier_before=g.brier_before.mean(),Brier_after=g.brier_after.mean(),delta_Brier=g.delta_brier.mean(),entropy_before=g.entropy_before.mean()))
 agg=pd.DataFrame(agg)
 home=[]
 for dim in ['round','model_signature','favorite']:
  for key,g in d.groupby(dim):
   z=dict(dimension=dim,group=key,n=len(g),H_expected=g.lambda_home.sum(),H_actual=g.actual_home.sum(),A_expected=g.lambda_away.sum(),A_actual=g.actual_away.sum(),predicted_goal_difference=(g.lambda_home-g.lambda_away).sum(),actual_goal_difference=(g.actual_home-g.actual_away).sum())
   for symbol in 'HDA':z[symbol+'_expected_results']=g['baseline_P_'+symbol].sum();z[symbol+'_actual_results']=int(g.actual_symbol.eq(symbol).sum())
   home.append(z)
 transitions=[]
 for a,b in [('基礎λ','LAB後'),('LAB後','最終λ・mix前'),('最終λ・mix前','final'),('保存差分から外れ値補正前を復元','最終λ・mix前')]:
  pair=stages[stages.stage==a].merge(stages[stages.stage==b],on=['round','match_no','model_signature'],suffixes=('_before','_after'))
  for _,r in pair.iterrows():
   transitions.append(dict(round=r['round'],match_no=r.match_no,model_signature=r.model_signature,before=a,after=b,
    lambda_H_before=r.lambda_home_before,lambda_A_before=r.lambda_away_before,lambda_H_after=r.lambda_home_after,lambda_A_after=r.lambda_away_after,
    H_before=r.H_before,D_before=r.D_before,A_before=r.A_before,H_after=r.H_after,D_after=r.D_after,A_after=r.A_after,
    delta_LL=r.log_loss_after-r.log_loss_before,delta_Brier=r.brier_after-r.brier_before,delta_goal_NLL=r.goal_nll_after-r.goal_nll_before))
 frames={'input_quality':q,'mix_matches':mx,'mix_summary':agg,'home_away_calibration':pd.DataFrame(home),'correction_inputs':co,'correction_stages':pd.DataFrame(transitions),'formula_checks':pd.DataFrame(checks),'denominator_sensitivity_NOT_corrected':sen}
 for name,frame in frames.items():frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
 pd.read_csv(PRE/'lambda_residual_summary.csv').to_csv(OUT/'goal_uncertainty.csv',index=False,encoding='utf-8-sig')
 pd.read_csv(PRE/'calibration.csv').to_csv(OUT/'HDA_calibration.csv',index=False,encoding='utf-8-sig')
 audit=dict(matches=78,team_occurrences=len(q),unique_team_snapshots=len(q.drop_duplicates(['team','stats_file'])),count_mismatches=int(q.count_mismatch.sum()),
  snapshot_value_mismatches=int((~q[['snapshot_xg_matches','snapshot_xga_matches','snapshot_rate_matches']].all(axis=1)).sum()),
  asof_future=int((~q.asof_not_after_kickoff).sum()),mix_applied=int(mx['mix'].gt(0).sum()),mix_improved=int(mx.improved.sum()),mix_worsened=int(mx.worsened.sum()),
  mix_delta_LL=float(mx.delta_log_loss.mean()),mix_delta_Brier=float(mx.delta_brier.mean()),round_bootstrap95_delta_LL=boot_round(mx,'delta_log_loss'),
  bootstrap_note='全6開催の等重み・全6^6再標本化。3開催mix0を含む。探索的で6群しかなく一般化CIではない。',
  source_hashes=hashes,production_hashes=protected,no_new_coefficients=True,no_new_tickets=True,corrected_inputs_confirmed=0)
 assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in protected.items())
 audit['inputs_and_production_unchanged']=True
 (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 return frames,audit
if __name__=='__main__':
 f,a=run();print(json.dumps({k:v for k,v in a.items() if 'hashes' not in k},ensure_ascii=False,indent=2));print(f['mix_summary'].to_string(index=False))
