"""Prematch-only HA diagnostics; no learned correction, no new tickets."""
from pathlib import Path
import json,hashlib,math,html
import numpy as np
import pandas as pd
from scipy.stats import chi2,poisson
from analyze_phase35_high_draw_structure import conditions
from expected_goals import base_lambdas_v1,_attack_rate_v1,_defence_rate_v1,score_distribution

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/reports/score_structure_phase38'
SOURCE=ROOT/'data/reports/score_structure_phase35/diagnostics_78.csv'
SPEC=dict(version='phase38-diagnostic-1',reversal='actual nonD AND winner != argmax(final H,final A); ties favor H',
 expected='min(H,A)/(H+A) conditional on each observed nonD match; unconditioned all78 sum min(H,A) separately',
 gap_bins=[0,.10,.20,1.],conditional_favorite_bins=[.5,.6,.7,1.],
 numerical_diagnostics=['T','abs_logR','lambda_gap','P00','P11','P22plus','margin1'],
 splits='each continuous feature split at whole78 prematch median, before outcome comparison; no optimizing cuts',
 binary_diagnostics=['S','scenario_HA_flip_10pct','xg_only_disagrees','LAB_reduces_gap','absence_reduces_gap','favorite_more_fatigued','favorite_less_rest'],
 calibration_bins=[0,1/3,2/3,1],small_bin_warning_n=10,
 expected_count_interval='exact Poisson-binomial 2.5%-97.5%, independence assumption, prediction interval not parameter CI',
 matched_diagnostic='case-control only; same model_signature and favorite, conditional reverse probability gap<=.03; nearest, controls reusable',
 residuals='observed goals-lambda are residuals, not observed latent lambda errors',
 new_coefficients=False,correction_model_trained=False,new_portfolios=False)

def pb_interval(q):
 a=np.array([1.])
 for p in q:a=np.convolve(a,[1-p,p])
 c=np.cumsum(a);return float(np.sum(q)),int(np.searchsorted(c,.025)),int(np.searchsorted(c,.975))

def wilson(k,n):
 if not n:return np.nan,np.nan
 z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return mid-half,mid+half

def metrics(p,y):
 p=np.asarray(p,float);y=np.asarray(y,int)
 return dict(log_loss=float(-np.log(p[np.arange(len(y)),y]).mean()),brier=float(((p-np.eye(3)[y])**2).sum(axis=1).mean()),top_accuracy=float((p.argmax(axis=1)==y).mean()))

def raw(h,a):
 s=score_distribution(h,a);return np.array([s['prob_home'],s['prob_draw'],s['prob_away']])

def reverse_summary(g,**meta):
 z=g[g.actual_symbol.ne('D')];n=len(z);k=int(z.reversal.sum());exp,lo,hi=pb_interval(z.expected_reversal.to_numpy());wl,wh=wilson(k,n)
 return dict(**meta,all_matches=len(g),nonD=n,reversals=k,correct=n-k,observed_rate=k/n if n else np.nan,expected_reversals=exp,expected_rate=exp/n if n else np.nan,
             excess=k-exp,count_prediction_lo=lo,count_prediction_hi=hi,rate_wilson_lo=wl,rate_wilson_hi=wh,small_sample=n<10)

def run():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'fixed_spec.json').write_text(json.dumps(SPEC,ensure_ascii=False,indent=2))
 d=pd.read_csv(SOURCE);st=pd.read_csv(ROOT/'data/reports/score_structure_phase12/match_structure_78.csv');sc=pd.read_csv(ROOT/'data/reports/score_structure_phase35/relative_scenarios.csv')
 assert len(d)==78
 d['S']=conditions(d)['inclusive'];d['card']=d.home_team+'－'+d.away_team;d['score']=d.actual_home.astype(str)+'–'+d.actual_away.astype(str)
 d['favorite']=np.where(d.baseline_P_H>=d.baseline_P_A,'H','A');d['reversal']=d.actual_symbol.ne('D')&d.actual_symbol.ne(d.favorite)
 d['expected_reversal']=d[['baseline_P_H','baseline_P_A']].min(axis=1)/(d.baseline_P_H+d.baseline_P_A)
 d['conditional_favorite']=1-d.expected_reversal;d['HA_gap']=abs(d.baseline_P_H-d.baseline_P_A);d['HA_ratio']=d.baseline_P_H/d.baseline_P_A
 d['lambda_gap']=abs(d.lambda_home-d.lambda_away);d['abs_logR']=abs(d.log_R)
 d=d.merge(st[['round','match_no','P_2_2','P_3_3','P_draw_4plus','P_home_2','P_away_2','P_home_3plus','P_away_3plus']],on=['round','match_no'],validate='one_to_one')
 events=[json.loads(x) for x in (ROOT/'data/forecast_registry/forecasts.jsonl').read_text().splitlines()];snapshots={};hashes={};availability=[];inputs=[];stage_rows=[]
 for rnd,g in d.groupby('round'):
  path=ROOT/f'data/eval/toto_rounds/{rnd}/snapshot/next_predictions.csv';h=hashlib.sha256(path.read_bytes()).hexdigest();assert all(e['output_sha256']==h and e['provenance_class']=='A' for e in events if e['round_id']==rnd)
  hashes[str(path)]=h;snapshots[rnd]=pd.read_csv(path)
  snap=snapshots[rnd]
  # Inventory whitelisted prematch inputs; legacy model probability fields are never used.
  names=['base_lambda_home','base_lambda_away','base_lambda_reason','lab_matchup_lambda_home','lab_matchup_lambda_away','lab_final_lambda_home','lab_final_lambda_away','lab_adjustment_reason',
   'absence_lambda_delta_home','absence_lambda_delta_away','absence_input_missing_home','absence_input_missing_away','home_rest_hours','away_rest_hours','home_fatigue_score','away_fatigue_score',
   'goal_rate_adjustment_version','style_official_status','style_lambda_delta_home','style_lambda_delta_away','stats_asof','flab_snapshot_date']
  for side in ['home','away']:
   names += [f'stats_{x}_{side}' for x in ['1試合平均得点数','1試合平均失点数','ゴール期待値','被ゴール期待値','1試合平均被ゴール期待値']]
   names += [f'flab_expected_{x}_{side}' for x in ['for_xg','against_xg']]
  for name in names:availability.append(dict(round=rnd,field=name,present=name in snap,available=int(snap[name].notna().sum()) if name in snap else 0,n=13))
  for _,r in g.iterrows():
   x=snap[snap.match_no==r.match_no].iloc[0];p=np.array([x.next_prob_home,x.next_prob_draw,x.next_prob_away]);assert np.allclose(p,r[['baseline_P_H','baseline_P_D','baseline_P_A']].to_numpy(float),atol=1e-12)
   bh,ba,_=base_lambdas_v1(x);assert np.allclose([bh,ba],[x.base_lambda_home,x.base_lambda_away],atol=1e-12)
   ah,xgh,gamesh=_attack_rate_v1(x,'home');aa,xga,gamesa=_attack_rate_v1(x,'away');dh,xgah=_defence_rate_v1(x,'home');da,xgaa=_defence_rate_v1(x,'away')
   lh=x.get('lab_final_lambda_home',x.final_lambda_home);la=x.get('lab_final_lambda_away',x.final_lambda_away)
   raw_final=raw(r.lambda_home,r.lambda_away);mix=r.uncertainty_mix;assert np.allclose((1-mix)*raw_final+mix/3,p,atol=1e-10)
   xg_h=np.sqrt(xgh*xgaa);xg_a=np.sqrt(xga*xgah)
   inp=dict(round=r['round'],match_no=r.match_no,home_attack=ah,away_attack=aa,home_defense=dh,away_defense=da,home_xgpg=xgh,away_xgpg=xga,home_xgapg=xgah,away_xgapg=xgaa,
    inferred_games_home=gamesh,inferred_games_away=gamesa,xg_only_lambda_home=xg_h,xg_only_lambda_away=xg_a,
    xg_only_disagrees=('H' if xg_h>=xg_a else 'A')!=r.favorite,
    LAB_reduces_gap=abs(lh-la)<abs(bh-ba),absence_reduces_gap=abs(r.lambda_home-r.lambda_away)<abs(lh-la) if 'absence_lambda_delta_home' in snap else np.nan,
    favorite_more_fatigued=(x.home_fatigue_score>x.away_fatigue_score) if r.favorite=='H' else (x.away_fatigue_score>x.home_fatigue_score),
    favorite_less_rest=(x.home_rest_hours<x.away_rest_hours) if r.favorite=='H' else (x.away_rest_hours<x.home_rest_hours),
    base_logR=math.log(bh/ba),lab_logR=math.log(lh/la),final_logR=math.log(r.lambda_home/r.lambda_away),
    **{name:x.get(name,np.nan) for name in names})
   inputs.append(inp)
   stages=[('基礎λ',bh,ba,raw(bh,ba)),('LAB後',lh,la,raw(lh,la)),('最終λ・mix前',r.lambda_home,r.lambda_away,raw_final),('final',r.lambda_home,r.lambda_away,p)]
   if 'goal_rate_final_lambda_delta_home' in snap:
    h0=r.lambda_home-x.goal_rate_final_lambda_delta_home;a0=r.lambda_away-x.goal_rate_final_lambda_delta_away
    stages.append(('保存差分から外れ値補正前を復元',h0,a0,raw(h0,a0)))
   y={'H':0,'D':1,'A':2}[r.actual_symbol]
   for name,hv,av,pv in stages:
    stage_rows.append(dict(round=r['round'],match_no=r.match_no,model_signature=r.model_signature,stage=name,lambda_home=hv,lambda_away=av,H=pv[0],D=pv[1],A=pv[2],
       log_loss=-np.log(pv[y]),brier=float(((pv-np.eye(3)[y])**2).sum()),goal_nll=-poisson.logpmf(r.actual_home,hv)-poisson.logpmf(r.actual_away,av),
       HA_favorite='H' if pv[0]>=pv[2] else 'A',actual=r.actual_symbol))
 inp=pd.DataFrame(inputs);d=d.merge(inp,on=['round','match_no'],validate='one_to_one')
 sen=[]
 for (rnd,no),g in sc.groupby(['round','match_no']):
  base=g[g.scenario=='基準'].iloc[0];b_sign=np.sign(base.final_P_H-base.final_P_A);z=g[g.epsilon.eq(.1)]
  sen.append(dict(round=rnd,match_no=no,scenario_HA_flip_10pct=bool((np.sign(z.final_P_H-z.final_P_A)!=b_sign).any()),
    HA_gap_sensitivity_range=float((z.final_P_H-z.final_P_A).max()-(z.final_P_H-z.final_P_A).min()),
    scenario_PD_min=z.final_P_D.min(),scenario_PD_max=z.final_P_D.max()))
 d=d.merge(pd.DataFrame(sen),on=['round','match_no'],validate='one_to_one')
 d['gap_band']=pd.cut(d.HA_gap,[-1e-12,.1,.2,1.],labels=['0〜10pt','10〜20pt','20pt超'],right=False)
 assert int(d.reversal.sum())==23 and int(d.actual_symbol.ne('D').sum())==60
 rows=[reverse_summary(d,dimension='全体',group='78試合中H/A決着60')]
 for col in ['round','model_signature','favorite','gap_band','S']:
  for key,g in d.groupby(col,observed=True):rows.append(reverse_summary(g,dimension=col,group=str(key)))
 numerical={'T':'T','abs_logR':'abs_logR','lambda_gap':'lambda_gap','P00':'P_0_0','P11':'P_1_1','P22plus':'P_draw_2plus','margin1':'P_onegoal'}
 for label,col in numerical.items():
  cut=d[col].median()
  for flag in [False,True]:rows.append(reverse_summary(d[(d[col]<=cut)==flag],dimension=label,group=f'{"以下" if flag else "超"}中央値 {cut:.8f}'))
 for col in SPEC['binary_diagnostics'][1:]:
  for flag in [False,True]:rows.append(reverse_summary(d[d[col]==flag],dimension=col,group=str(flag)))
 comparisons=pd.DataFrame(rows)
 stages=pd.DataFrame(stage_rows);stage_compare=[]
 for name,g in stages.groupby('stage'):
  keys=pd.MultiIndex.from_frame(g[['round','match_no']]);base=stages[stages.stage=='final'].set_index(['round','match_no']).loc[keys]
  stage_compare.append(dict(stage=name,n=len(g),log_loss=g.log_loss.mean(),brier=g.brier.mean(),goal_nll=g.goal_nll.mean(),same_rows_final_log_loss=base.log_loss.mean(),
    delta_log_loss_vs_final=g.log_loss.mean()-base.log_loss.mean(),delta_brier_vs_final=g.brier.mean()-base.brier.mean()))
 transitions=[]
 for before,after in [('基礎λ','LAB後'),('LAB後','最終λ・mix前'),('最終λ・mix前','final'),('保存差分から外れ値補正前を復元','最終λ・mix前')]:
  a=stages[stages.stage==before];b=stages[stages.stage==after]
  pair=a.merge(b,on=['round','match_no','model_signature'],suffixes=('_before','_after'),validate='one_to_one')
  for rnd,z in [('全体',pair)]+list(pair.groupby('round')):
   transitions.append(dict(before=before,after=after,round=rnd,n=len(z),log_loss_before=z.log_loss_before.mean(),log_loss_after=z.log_loss_after.mean(),
    delta_log_loss=(z.log_loss_after-z.log_loss_before).mean(),delta_brier=(z.brier_after-z.brier_before).mean(),delta_goal_nll=(z.goal_nll_after-z.goal_nll_before).mean(),
    HA_favorite_changes=int((z.HA_favorite_before!=z.HA_favorite_after).sum())))
 stage_version=stages.groupby(['model_signature','stage']).agg(n=('match_no','size'),log_loss=('log_loss','mean'),brier=('brier','mean'),goal_nll=('goal_nll','mean')).reset_index()
 cal=[];metric=[]
 groups=[('全体',d)]+[(f'S={b}',d[d.S==b]) for b in [False,True]]+[(f'版:{v}',g) for v,g in d.groupby('model_signature')]
 for name,g in groups:
  probs=g[['baseline_P_H','baseline_P_D','baseline_P_A']].to_numpy();y=g.actual_symbol.map({'H':0,'D':1,'A':2}).to_numpy()
  metric.append(dict(group=name,n=len(g),**metrics(probs,y)))
  for symbol,col in zip('HDA',['baseline_P_H','baseline_P_D','baseline_P_A']):
   for band in ['全域','0〜1/3','1/3〜2/3','2/3〜1']:
    z=g if band=='全域' else g[pd.cut(g[col],[-1e-12,1/3,2/3,1.000001],labels=['0〜1/3','1/3〜2/3','2/3〜1'],right=False)==band]
    n=len(z);k=int(z.actual_symbol.eq(symbol).sum());wl,wh=wilson(k,n);ex,lo,hi=pb_interval(z[col].to_numpy())
    cal.append(dict(group=name,symbol=symbol,band=band,n=n,expected_count=ex,actual_count=k,mean_probability=z[col].mean(),actual_rate=k/n if n else np.nan,wilson_lo=wl,wilson_hi=wh,
      count_prediction_lo=lo,count_prediction_hi=hi,small_sample=n<10))
 # Goal residuals, no latent lambda 'error' observed.
 residual=[];lambda_groups=[('全体',d)]+[(f'開催:{v}',g) for v,g in d.groupby('round')]+[(f'版:{v}',g) for v,g in d.groupby('model_signature')]+[(f'S={v}',g) for v,g in d.groupby('S')]
 d['residual_H']=d.actual_home-d.lambda_home;d['residual_A']=d.actual_away-d.lambda_away
 d['residual_total']=d.residual_H+d.residual_A;d['residual_difference']=d.residual_H-d.residual_A
 for name,g in lambda_groups:
  # g may precede the residual-column assignment, so calculate from primitives.
  rh=g.actual_home-g.lambda_home;ra=g.actual_away-g.lambda_away
  for side,goal,lam in [('H',g.actual_home,g.lambda_home),('A',g.actual_away,g.lambda_away)]:
   obs=goal.sum();expected=lam.sum();ci_lo=chi2.ppf(.025,2*obs)/2/expected if obs else 0.;ci_hi=chi2.ppf(.975,2*(obs+1))/2/expected
   residual.append(dict(group=name,side=side,n=len(g),predicted_goals=expected,actual_goals=obs,mean_residual=(goal-lam).mean(),actual_to_expected=obs/expected,
      poisson_ratio_ci_lo=ci_lo,poisson_ratio_ci_hi=ci_hi,pearson_mean=((goal-lam)**2/lam).mean(),
      excess_second_moment=((goal-lam)**2-lam).mean(),total_residual_mean=(rh+ra).mean(),difference_residual_mean=(rh-ra).mean(),
      residual_covariance=float(np.cov(rh,ra,ddof=1)[0,1]) if len(g)>1 else np.nan))
 # Matched comparison is explanatory, not a deployed classifier.
 paired=[]
 for _,r in d[d.reversal].iterrows():
  pool=d[d.actual_symbol.ne('D')&~d.reversal&(d.model_signature==r.model_signature)&(d.favorite==r.favorite)].copy()
  pool['distance']=abs(pool.expected_reversal-r.expected_reversal);pool=pool[pool.distance<=.03].sort_values(['distance','round','match_no'])
  row=dict(round=r['round'],match_no=r.match_no,card=r.card,q=r.expected_reversal,matched=bool(len(pool)))
  if len(pool):
   c=pool.iloc[0];row.update(control_round=c['round'],control_no=c.match_no,control_card=c.card,control_q=c.expected_reversal,
     T=r['T'],control_T=c['T'],P11=r.P_1_1,control_P11=c.P_1_1,sensitivity=r.HA_gap_sensitivity_range,control_sensitivity=c.HA_gap_sensitivity_range)
  paired.append(row)
 # Save coupled H/D/A responses for the fixed Phase3.5 perturbations, not corrections.
 sensitivity=sc.merge(d[['round','match_no','S','reversal']],on=['round','match_no'],validate='many_to_one')
 frames={'match_diagnostics':d,'HA_condition_comparison':comparisons,'input_availability':pd.DataFrame(availability),'saved_input_trace':inp,
  'stage_probabilities':stages,'paired_stage_transitions':pd.DataFrame(transitions),'stage_metrics':pd.DataFrame(stage_compare),'stage_version_metrics':stage_version,'probability_metrics':pd.DataFrame(metric),
  'calibration':pd.DataFrame(cal),'lambda_residual_summary':pd.DataFrame(residual),'conditional_probability_matches':pd.DataFrame(paired),'sensitivity_HDA':sensitivity}
 decisions=[]
 for feature in list(numerical)+SPEC['binary_diagnostics']:
  reason='同じλ由来の構造・接戦確率と重複。期待逆転件数を超える独立した識別力を確認できず、補正係数を作らない。'
  if feature=='S':reason='D6/8の探索仮説を保持。H/A決着2試合しかなくH/A補正の根拠にできない。'
  elif feature=='xg_only_disagrees':reason='6決着中4逆転だが期待2.89、予測範囲1〜5。僅差条件と重複。後続診断候補として保持、補正は不採用。'
  elif feature in ['favorite_more_fatigued','favorite_less_rest']:reason='保存済み方向情報として調査したが、小標本の期待件数との差から補正方向・量を確定できない。'
  elif feature in ['LAB_reduces_gap','absence_reduces_gap']:reason='補正による差の縮小自体と試合の逆転原因を同一視できない。段階別確率評価を併記、後付けで処理を削除しない。'
  decisions.append(dict(feature=feature,threshold_optimized=False,status='診断継続・新係数なし',reason=reason))
 frames['feature_decisions']=pd.DataFrame(decisions)
 for name,frame in frames.items():frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
 d[d.reversal].to_csv(OUT/'reversals_23.csv',index=False,encoding='utf-8-sig')
 d[(d['round']=='toto1665')&d.match_no.isin([2,6,13])].to_csv(OUT/'toto1665_three_matches.csv',index=False,encoding='utf-8-sig')
 future=dict(**SPEC,discovery_rounds=sorted(d['round'].unique()),unused_complete_rounds=len(set(e['round_id'] for e in events)-set(d['round'])),
  deployment='diagnostics only; correction=None; baseline tickets unchanged',
  next_protocol=['Freeze hashes and inputs before kickoff; do not rerun history with future team data.','Report all next complete rounds and each model version separately.','Keep same bands and condition S; record log loss/Brier/calibration and conditional reverse expectations.','If later deriving correction, fit on prior records only and reserve later untouched rounds. No same78 claim of independent validation.'])
 (OUT/'future_validation.json').write_text(json.dumps(future,ensure_ascii=False,indent=2))
 audit=dict(original_forecasts=78,nonD=60,reversals=23,source_hashes=hashes,all_base_lambdas_reproduced=True,all_final_probabilities_reproduced=True,
   numerical_feature_splits=len(numerical),binary_feature_diagnostics=len(SPEC['binary_diagnostics']),optimized_thresholds=0,new_coefficients=0,
   original_files_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
   missing_information=['No causal identification of individual lambda error from one realized score','No independent unused complete round','No historical per-opponent strength adjustment in saved base formula','No separate explicit home-advantage term in inspected Next base formula'])
 (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 return frames


def render(f):
 d=f['match_diagnostics'];c=f['HA_condition_comparison'];parts=[]
 def tab(g):return '<div class="scroll">'+g.to_html(index=False,border=0,float_format=lambda v:f'{v:.5f}')+'</div>'
 def add(title,text,frame=None):parts.append('<section><h2>'+title+'</h2><p>'+text+'</p>'+('' if frame is None else tab(frame))+'</section>')
 total=c[c.dimension=='全体'].iloc[0]
 add('全体：逆転23件は多すぎたか',f'H/A決着60試合の逆転は23件。各試合のmin(PH,PA)/(PH+PA)からの期待は{total.expected_reversals:.3f}件、試合間独立のPoisson二項95%予測範囲は{total.count_prediction_lo}〜{total.count_prediction_hi}件。少なくとも総数だけから逆転が想定以上とは言えない。D18試合は条件付き診断の分母からのみ除外し、購入用の確率ではDを消さない。78全体の無条件逆転期待は'+f'{d[["baseline_P_H","baseline_P_A"]].min(axis=1).sum():.3f}件で、別の標本条件に対する値。')
 add('予測優勢側・確率差・モデル版','H/A差0〜10、10〜20、20ポイント超を計算前に固定。少数群は率の振れが大きく、期待逆転数との差を見る。',c[c.dimension.isin(['gap_band','favorite','model_signature','round'])])
 add('23逆転試合の診断','逆転を予測失敗と一律に断定しない。qはH/A決着時の劣勢側勝率。',d[d.reversal][['round','match_no','card','score','favorite','baseline_P_H','baseline_P_D','baseline_P_A','expected_reversal','lambda_home','lambda_away','T','log_R','S','scenario_HA_flip_10pct']])
 add('得点構造・入力の比較','連続量7種類は全78の試合前中央値で二分、二値診断7種類を固定。閾値探索・最良条件の採択・新係数は0。Dを含む全件数とH/A決着数を分け、非逆転例も分母に残した。信頼区間はWilson、件数範囲は独立仮定による予測範囲であり、同一チーム・開催の依存を十分扱えない。',c[~c.dimension.isin(['全体','gap_band','favorite','model_signature','round'])])
 add('検討した条件と採否','14項目の固定診断を記録。独立した14証拠ではなく、例えば|logR|とλ差の中央値分割は今回同じ試合群になる。実績最大化の閾値探索はしていない。',f['feature_decisions'])
 add('同程度の逆転確率の比較','結果を使った事後ケース対照比較。各逆転例に同じモデル版・予測優勢側、q差3ポイント以内の非逆転例を最短距離で対応。対照再使用あり。この対応から予測ルールを学習していない。',f['conditional_probability_matches'])
 focus=d[(d['round']=='toto1665')&d.match_no.isin([2,6,13])]
 add('1665の3試合：入力からλまで','攻撃は得点率とxG率の50/50、守備は失点率とxGA率の50/50。その積の平方根が基礎λ。xG率の試合数は累積xGA/試合平均xGAを丸めて推定しており、保存入力から再現できるが分母・集計期間の品質は独立の監査対象。LABのxG対戦値を対数空間25%で混合し、欠場の攻守影響を加算。保存原本78件の基礎λと最終HDAを再現した。',focus[['card','score','home_attack','away_attack','home_defense','away_defense','home_xgpg','away_xgpg','home_xgapg','away_xgapg','base_lambda_home','base_lambda_away','lab_final_lambda_home','lab_final_lambda_away','absence_lambda_delta_home','absence_lambda_delta_away','lambda_home','lambda_away']])
 add('鹿島－柏','基礎λ1.600対1.511と僅差。LAB後1.567対1.459、欠場補正はH+0.011で最終1.578対1.459。A34.85%は十分な事前可能性を残していた。Aの欠落はPhase3.7の10口選択の機会損失でもあり、この1結果でH優勢予測そのものを誤設計とは判定できない。')
 add('東京V－福岡','基礎攻撃率0.636対1.075、東京V守備側失点期待1.750・福岡1.400がA優勢を作った。LAB後のHは0.967へ上がり、欠場補正も相対差を縮めるが最終0.912対1.310。保存疲労値は東京V2.70・福岡12.67でHを相対的に支持し得る材料。ただし既存mixは一様化で方向を変えず、疲労値差を勝者補正へ使う妥当性は別途必要。得点/xG自体からH優勢を裏付ける入力はこの式には見られない。')
 add('富山－札幌','基礎攻撃率1.883対1.138、富山守備1.000・札幌1.450がH優勢を作った。LABはほぼ同方向、欠場はH−0.084・A+0.016で差を縮めたが最終1.557対1.087。札幌側の疲労値15.85は富山4.00より高く、東京Vの説明をそのまま使うとこちらのA勝利は説明できない。元のA28.22%もゼロではない。')
 add('保存情報の有無と実際の消費','保存された旧prob_*・Elo/LAB混合予測は使用しない。Nextの式に入る得失点率・xG/xGA・LAB xG・欠場と、保存だけされている情報を区別。fatigue/weatherはmix、独立したhome advantage項・対戦相手の過去日程強度補正は今回確認したNext式にはない。相手の守備力は平方根積に入る。直近個別試合の退場などは再取得せず未取得として扱う。1670のstyleはno_prematch_archivesでλ差分0、得失点外れ値補正のみ保存差分で比較可能。',f['input_availability'].groupby('field').agg(available=('available','sum'),total=('n','sum')).reset_index())
 add('補正段階の確率評価','保存λから再集計した既存段階の診断。現在のチーム履歴で再計算していない。各段階と同じ試合集合のfinalを比較し、モデル版も分けた。途中段階の成績が良くても、この78試合だけを理由に補正を削除する指示にはしない。',f['stage_metrics'])
 add('直近成績・外れ値処理の扱い','保存された得失点率を基礎入力とし、今回最新の履歴へ差し替えていない。1670の外れ値処理は、過去各試合時点の自チーム・相手の成績から突出得点を判定し、重みを0.5〜1に抑える既存方式。現行コードの補正差は過去全件の加重平均−通常平均を公開率に加える形で、単純な直近5試合の指数減衰ではない。今回その計算を履歴から再実行せず、保存済み最終λ差分だけで補正前を復元した。以前の開催へ現在の補正を遡及適用していない。')
 add('同じ試合・同じ処理段階での差','λ補正の前後はともにmix前で比較し、mixの効果は別の行とする。外れ値補正前の復元は保存差分のある1670だけ。これらは既存処理の診断で、新規に修正量を学習したものではない。',f['paired_stage_transitions'])
 add('モデル版別・補正段階', '異なるモデル版の差を補正の因果効果と混同しない。',f['stage_version_metrics'])
 add('H/D/Aの校正','Brierは3カテゴリの二乗誤差の和、Log Lossは自然対数。最高確率一致率とは別の指標。',f['probability_metrics'])
 add('H/D/A期待件数と実件数','確率帯は固定の3区分。小標本帯の過度な解釈を避ける。',f['calibration'][f['calibration'].band=='全域'])
 add('λ残差と推定不確実性','実得点−λは残差。Poisson比率区間はλを既知offsetとする独立Poisson仮定の診断に限り、真のλ誤差区間ではない。Pearson二乗残差やH/A残差共分散には平均の偏り・未観測要因・得点の依存も含まれ、それだけから潜在λの分散・相関を同定できない。',f['lambda_residual_summary'])
 add('条件Sとの統合','S8試合はD6、残るH決着2試合はいずれもH優勢予測どおり。H/A決着に限ると標本は2件で、SがH/A逆転を見分けると結論できない。S・T・logRは同じλ由来の情報で、独立証拠として加算しない。相対λ変動ではH/D/Aを得点行列から同時に再計算する必要があり、Phase3.5の全25状態をsensitivity_HDA.csvに保存した。これらは仮定上の反応で現実の誤差分布ではない。')
 add('残った調査候補と反例','xGだけの対戦期待値とfinal優勢側が食い違う9試合のうちH/A決着は6、逆転4（期待2.89）。徳島－今治・いわき－秋田・札幌－秋田・宮崎－大分は逆転したが、柏－浦和・京都－清水はfinal側が勝った。元々q≈0.48の僅差群であり、別の識別情報と確認できない。±10%試験で優勢反転する群は17決着中9逆転（期待8.15）、しない群は43中14（期待16.17）。接戦の表現として有用でも、期待以上の識別力の証明にはならない。')
 add('ホーム側の偏りは調査を継続する価値がある','ホーム得点は予測104.43に対し実122、アウェーは103.22に対し97。特に1665・1668の26試合でH予測35.89対実52だが、1670ではH予測16.96対実13と逆方向。明示的ホーム有利項がない計算構造と合わせて監査候補になるが、ホームλを一律増やす係数の根拠にはまだ不足。得点のPoisson偶然性を除いた潜在λ誤差の分散・相関は、この少数の集計残差から一意に決まらない。')
 add('疲労の方向情報は補正根拠になったか','優勢側の疲労が高い群は26決着中10逆転（期待10.21）、それ以外は34中13（期待14.11）。今回の単純な方向比較では期待以上の逆転増は確認できない。東京V－福岡の疲労差だけを根拠にHへ補正する案は採用しない。')
 add('mixと確率順位の関係','保存mixは全HDAを一様分布へ近づけ、PH−PAを(1−mix)倍する。mix<1ならH/A首位は変わらない。λに方向補正を加える場合にはraw H/D/Aを同じ得点分布から再計算し、mixとの役割を分ける。今回の原本ではこの関係とHDA合計1を確認した。')
 add('修正案と配券を今回作らない理由','逆転総数はモデル期待から逸脱しておらず、条件ごとの違いは小標本・版・ホーム側傾向と重なる。入力に改善を調べる余地はあるが、この結果から修正方向・適用条件・修正量を一貫して確定する根拠は不足。新しいH/A係数、Sの75%置換、修正後10口は作らない。これは現在の予測が完全であるという結論ではない。保存原本と固定指標による後続検証を先に行う。')
 add('今後必要な確認','未使用の完成済み試合前予測はレジストリでは確認できない。future_validation.jsonに指標・区間・S・評価方法を固定。各モデル版で、ホーム側の校正、疲労差の方向情報、xG分母と集計期間、欠場の実測品質を確認する。修正を試す場合は後続未使用期間でLog Loss/Brierを評価し、HDA合計1を保った分布を作ってから10口へ接続する。H/A変動を導入した状態では、Phase3.7のD配置別圧縮は再証明が必要。')
 # Chart: observed vs expected reversal counts by fixed gap bands.
 z=c[c.dimension=='gap_band'];svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 330"><rect width="820" height="330" fill="white"/><g font-family="sans-serif" font-size="15"><text x="25" y="25">H/A確率差別：逆転の実件数と条件付き期待件数</text>']
 scale=210/max(1,z.reversals.max(),z.expected_reversals.max())
 for j,r in enumerate(z.itertuples()):
  x=90+j*240
  for v,dx,color,label in [(r.expected_reversals,0,'#8298aa','期待'),(r.reversals,70,'#207597','実際')]:
   h=v*scale;svg.append(f'<rect x="{x+dx}" y="{265-h}" width="55" height="{h}" fill="{color}"/><text x="{x+dx}" y="{255-h}">{v:.2f}</text><text x="{x+dx}" y="285">{label}</text>')
  svg.append(f'<text x="{x}" y="315">{r.group} n={r.nonD}</text>')
 svg.append('</g></svg>');(OUT/'reversal_counts.svg').write_text(''.join(svg));parts.insert(2,'<img src="reversal_counts.svg" alt="H/A差別の逆転実数と期待">')
 links=' '.join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(OUT.glob('*.csv')))
 (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase3.8 H/A逆転と予測校正</title><style>body{font:16px/1.8 system-ui,sans-serif;color:#24364a;max-width:1250px;margin:30px auto;padding:0 20px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:6px 9px;border-bottom:1px solid #ddd;text-align:right}th{background:#eaf0f5}section{margin:36px 0}img{max-width:100%}a{margin-right:12px}</style><h1>Phase3.8：H/A逆転の構造と予測の校正</h1>'+''.join(parts)+'<h2>CSV・再現仕様</h2>'+links+'<p><a href="future_validation.json">将来検証仕様</a><a href="audit.json">監査</a></p></html>')
 print(c[c.dimension.isin(['全体','gap_band','favorite','S','favorite_more_fatigued','xg_only_disagrees','scenario_HA_flip_10pct'])].to_string(index=False))
 print(f['stage_metrics'].to_string(index=False));print(f['probability_metrics'].to_string(index=False))

if __name__=='__main__':render(run())
