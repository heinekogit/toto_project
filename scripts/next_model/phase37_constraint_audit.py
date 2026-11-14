"""Research-only ablations. Shared scenarios, exact Cartesian enumeration, lossless pruning."""
from pathlib import Path
import json,hashlib,time,itertools,html
import numpy as np
import pandas as pd
from phase36_branch_portfolios import SOURCE,FEATURES,select_candidates,masks_for
from analyze_phase35_high_draw_structure import conditions
from analyze_score_structure_phase4b_stage1 import universe,encode,decode,SYMBOLS,N,POWERS
from prototype_buyplan_v40 import greedy_relative_robust,ticket_probabilities

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/reports/score_structure_phase37'
FULLMASK=(1<<13)-1
CODES=np.arange(N,dtype=np.int64)
DMASK=np.zeros(N,dtype=np.uint16)
for i,power in enumerate(POWERS):DMASK|=(((CODES//power)%3)==1).astype(np.uint16)<<i
CONFIGS=[(mode,n,2.,False) for mode in ['R','U'] for n in [30,100,300,1000,'FULL']]+[(mode,30,f,False) for mode in ['R','U'] for f in [1.,1.5]]+[(mode,30,2.,True) for mode in ['R','U']]
SPEC=dict(version='phase37-frozen-1',counts=[3,4,5],selectors=['A','B'],strengths=[1.,1.5,2.],topN=[30,100,300,1000,'FULL'],
 suites={'legacy':'36 Phase3.6 portfolios reproduced with original states and denominator',
 'focus':'1665 B5 own 32 masks fixed across all ablations; global unrestricted scenario oracle denominator',
 'common':'six rounds A/B k=3/4/5; union of A and B subset masks at the SAME k shared by generation AND objective; same unrestricted oracle denominator'},
 common_assessment='For every plan in a round, evaluate on union masks of A/B all k and strengths 1,1.5,2; identical distributions',
 objective='max min_s(sum_t p_s(t)/O_s); same deterministic greedy + 12 swaps; O_s exact unrestricted top10 mass for focus/common',
 full='Every allowed combination evaluated by dominance: keep top10 base probabilities per D-pattern on active scenario support; sufficient for any optimum of monotone scenario mass objective',
 branch_addition='Union with HA baseline modified by every subset of S allowed in mode; also every 0/1/2 HA flip among nonD slots; result blind, no purchase mandate',
 prune='Same lossless per-pattern top10 pruning applied after generating all finite pools; legacy checks against saved tickets',
 no_new_D_rule=True,no_parameter_tuning=True,all_generation_before_results=True,
 warning='Odds factors uncalibrated; full candidate sufficiency is not proof greedy solver is globally optimal')

def state(p,mask,factor):
 q=p.copy();active=np.array([bool(mask>>i&1) for i in range(13)])
 q[active,1]*=factor;q[active]/=q[active].sum(axis=1,keepdims=True)
 return q

def prune(pool,support,base):
 pool=np.array(sorted(set(map(int,pool))),dtype=np.int64)
 pat=DMASK[pool]&support;order=np.lexsort((pool,-base[pool],pat));ordered=pool[order];pg=pat[order]
 keep=[]
 for pattern in np.unique(pg):keep.extend(ordered[pg==pattern][:10])
 return np.array(sorted(keep),dtype=np.int64)

def branch_pool(p,S,allowedmask):
 anchor=np.where(p[:,0]>=p[:,2],0,2).astype(np.uint8);s=[i for i in S if allowedmask>>i&1];rows=[]
 for dmask in masks_for(s):
  item=anchor.copy()
  for i in range(13):
   if dmask>>i&1:item[i]=1
  others=[i for i in range(13) if item[i]!=1]
  for n in [0,1,2]:
   for flips in itertools.combinations(others,n):
    x=item.copy()
    for i in flips:x[i]=2-x[i]
    rows.append(x)
 return np.unique(encode(np.array(rows)))

class Engine:
 def __init__(self,p):
  self.p=p;self.base=universe(p);self.cache={};self.allowed_cache={};self.full_cache={}
 def allowed(self,mask):
  if mask not in self.allowed_cache:self.allowed_cache[mask]=np.flatnonzero((DMASK&(FULLMASK^mask))==0)
  return self.allowed_cache[mask]
 def mass(self,mask,factor,ids):
  z=np.prod([1+(factor-1)*self.p[i,1] for i in range(13) if mask>>i&1])
  return self.base[ids]*np.power(factor,np.bitwise_count(DMASK[ids]&mask).astype(int))/z
 def top(self,mask,factor,allowedmask):
  key=(mask if factor!=1 else 0,factor,allowedmask)
  if key not in self.cache:
   ids=self.allowed(allowedmask);v=self.mass(mask,factor,ids);n=min(1000,len(ids));cut=np.partition(v,-n)[-n];eligible=ids[v>=cut];ev=self.mass(mask,factor,eligible)
   self.cache[key]=eligible[np.lexsort((eligible,-ev))[:n]]
  return self.cache[key]
 def full(self,allowedmask,support):
  key=(allowedmask,support)
  if key not in self.full_cache:self.full_cache[key]=prune(self.allowed(allowedmask),support,self.base)
  return self.full_cache[key]
 def solve(self,selected,masks,mode,n,factor,extra,S,legacy=False):
  t=time.perf_counter();allowedmask=sum(1<<i for i in selected) if mode=='R' else FULLMASK
  # All masks shared even if factor=1; duplicate states cannot change min/mean ties.
  support=int(np.bitwise_or.reduce(masks));basecodes=self.top(0,1.,allowedmask)[:10]
  if n=='FULL':raw=None;pool=self.full(allowedmask,support);rawsize=len(self.allowed(allowedmask))
  else:
   raw=np.unique(np.concatenate([self.top(m,factor,allowedmask)[:int(n)] for m in masks]+[basecodes]))
   if extra:raw=np.union1d(raw,branch_pool(self.p,S,allowedmask))
   rawsize=len(raw);pool=prune(raw,support,self.base)
  values=np.array([self.mass(m,factor,pool) for m in masks])
  denom_mask=allowedmask if legacy else FULLMASK
  opts=np.array([self.mass(m,factor,self.top(m,factor,denom_mask)[:10]).sum() for m in masks])
  if factor==1:chosen=basecodes.copy()
  else:chosen=greedy_relative_robust(pool,values,opts,basecodes)
  totals=np.array([self.mass(m,factor,chosen).sum() for m in masks])/opts
  return dict(codes=chosen,raw_pool=raw,pool=pool,masks=list(masks),allowedmask=allowedmask,support=support,
              raw_pool_size=rawsize,effective_pool_size=len(pool),allowed_count=len(self.allowed(allowedmask)),
              objective=float(totals.min()),mean_relative=float(totals.mean()),seconds=time.perf_counter()-t)


def generate():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'fixed_spec.json').write_text(json.dumps(SPEC,ensure_ascii=False,indent=2))
 paths=[SOURCE,ROOT/'data/forecast_registry/forecasts.jsonl',ROOT/'data/forecast_registry/results.jsonl',ROOT/'data/reports/score_structure_phase36/shadow_tickets.csv',ROOT/'scripts/next_model/phase36_branch_portfolios.py']
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
 gdata=pd.read_csv(SOURCE,usecols=FEATURES);events=[json.loads(x) for x in paths[1].read_text().splitlines()]
 for ev in events:
  assert ev['provenance_class']=='A';assert hashlib.sha256(Path(ev['source_file']).read_bytes()).hexdigest()==ev['output_sha256']
  row=gdata[(gdata['round']==ev['round_id'])&(gdata.match_no==ev['match_no'])].iloc[0]
  for a,b in [('lambda_home','lambda_home'),('lambda_away','lambda_away'),('baseline_P_H','final_prob_home'),('baseline_P_D','final_prob_draw'),('baseline_P_A','final_prob_away')]:assert abs(row[a]-ev[b])<1e-12
 old=pd.read_csv(paths[3],usecols=['round','method','k','code']) # No results columns
 plans=[];engines={};groups={};commonmasks={};repro=[];sequence=0
 for rnd in ['toto1665','toto1660','toto1663','toto1664','toto1668','toto1670']:
  g=gdata[gdata['round']==rnd].sort_values('match_no');p=g[['baseline_P_H','baseline_P_D','baseline_P_A']].to_numpy();engine=Engine(p);engines[rnd]=engine;groups[rnd]=g
  selection={(m,k):tuple(sorted(map(int,select_candidates(g,m,k)))) for m in 'AB' for k in [3,4,5]}
  S=tuple(np.flatnonzero(conditions(g)['inclusive'].to_numpy()));union=sorted(set(x for s in selection.values() for x in masks_for(s)));commonmasks[rnd]=union
  cache={}
  def make(suite,method,k,mode,n,factor,extra,masks,legacy=False):
   nonlocal sequence
   selected=selection[(method,k)];allowedmask=sum(1<<i for i in selected) if mode=='R' else FULLMASK
   key=(allowedmask,tuple(masks),n,factor,extra,legacy)
   cached=key in cache
   if not cached:cache[key]=engine.solve(selected,masks,mode,n,factor,extra,S,legacy)
   ans=cache[key];sequence+=1
   plans.append(dict(**ans,id=f'E{sequence:04}',suite=suite,round=rnd,method=method,k=k,mode=mode,topN=str(n),factor=factor,extra_S=extra,
                    selected=selected,cache_reused=cached))
   return ans
  for method,k in selection:
   ans=make('legacy',method,k,'R',30,2.,False,masks_for(selection[(method,k)]),True)
   saved=old[(old['round']==rnd)&(old.method==method)&(old.k==k)].code.to_numpy()
   same=np.array_equal(ans['codes'],saved);assert same,(rnd,method,k,'legacy mismatch')
   repro.append(dict(round=rnd,method=method,k=k,exact_order_reproduced=same))
  if rnd=='toto1665':
   masks=masks_for(selection[('B',5)])
   for mode,n,factor,extra in CONFIGS:make('focus','B',5,mode,n,factor,extra,masks)
   print('1665 focus complete',flush=True)
  for k in [3,4,5]:
   masks=sorted(set(masks_for(selection[('A',k)]))|set(masks_for(selection[('B',k)])))
   for method in 'AB':
    for mode,n,factor,extra in CONFIGS:make('common',method,k,mode,n,factor,extra,masks)
  print(f'{rnd}: generated {len(plans)} portfolios so far, common states={len(union)}',flush=True)
 assert all(hashlib.sha256(Path(x).read_bytes()).hexdigest()==h for x,h in hashes.items())
 pd.DataFrame(repro).to_csv(OUT/'phase36_reproduction.csv',index=False,encoding='utf-8-sig')
 (OUT/'source_audit.json').write_text(json.dumps(dict(input_hashes=hashes,legacy_reproduced=len(repro),outputs_generated_without_results=True),indent=2))
 return plans,engines,groups,commonmasks


def diagnose(plans,engines,groups,commonmasks):
 # Only here load actual match results.
 actual=pd.read_csv(SOURCE,usecols=['round','match_no','actual_symbol'])
 rows=[];tickets=[];matchrows=[];opportunity=[];scenario_rows=[];ranks=[];forced=[];state_prob=[]
 code_sets={r:eng.top(0,1.,FULLMASK)[:10] for r,eng in engines.items()}
 for rnd,eng in engines.items():
  for factor in [1.,1.5,2.]:
   for mask in commonmasks[rnd]:
    q=state(eng.p,mask,factor)
    for i in range(13):state_prob.append(dict(round=rnd,factor=factor,mask=mask,match_no=i+1,H=q[i,0],D=q[i,1],A=q[i,2]))
 for plan in plans:
  rnd=plan['round'];eng=engines[rnd];g=groups[rnd];truth=actual[actual['round']==rnd].sort_values('match_no').actual_symbol.map({'H':0,'D':1,'A':2}).to_numpy();tc=int(encode(truth[None,:])[0]);truthD=int(DMASK[tc])
  codes=plan['codes'];t=decode(codes);pool=plan['pool'];raw=plan['raw_pool'];allowed=plan['allowedmask'];base=code_sets[rnd];baseprob=eng.base[base].sum();bcover=(decode(base)==truth).any(axis=0)
  allowed_truth=not bool(truthD&(FULLMASK^allowed));rawtruth=allowed_truth if raw is None else tc in set(raw);effective_truth=tc in set(pool);cover=(t==truth).any(axis=0);hits=(t==truth).sum(axis=1)
  additions=np.setdiff1d(codes,base);removed=np.setdiff1d(base,codes)
  phase='許容段階:D禁止' if not allowed_truth else '探索候補:上位N' if not rawtruth else '探索候補:全状態で優越する10券あり' if not effective_truth else '最終10口選択' if tc not in set(codes) else '13的中'
  row={k:v for k,v in plan.items() if k not in ['codes','raw_pool','pool','masks','selected']}
  row.update(selected=';'.join(str(i+1) for i in plan['selected']),state_count=len(plan['masks']),new_allowed_vs_R=plan['allowed_count']-3**plan['k']*2**(13-plan['k']),
   candidate_D_capture=sum(truth[i]==1 for i in plan['selected']),allowed_truth=allowed_truth,raw_pool_truth=rawtruth,effective_pool_truth=effective_truth,selected_truth=tc in set(codes),loss_stage=phase,
   final_prob13=eng.base[codes].sum(),baseline_retention=eng.base[codes].sum()/baseprob,best_hits=int(hits.max()),exact13=bool(max(hits)==13),D_cells=int((t==1).sum()),D_tickets=int((t==1).any(axis=1).sum()),
   purchased_D_capture=int((cover&(truth==1)).sum()),missing_HA=int((~cover&(truth!=1)).sum()),new_missing_HA_vs_base=int((bcover&~cover&(truth!=1)).sum()),
   correct_symbols=int(cover.sum()),combination_gap=int(cover.sum()-max(hits)),all_symbols_but_no13=bool(cover.all() and max(hits)<13),overlap_base=len(set(codes)&set(base)),
   added_mass=eng.base[additions].sum(),removed_mass=eng.base[removed].sum())
  assert abs(row['added_mass']-row['removed_mass']-(row['final_prob13']-baseprob))<1e-12
  rows.append(row)
  for j,c in enumerate(codes):tickets.append(dict(id=plan['id'],round=rnd,suite=plan['suite'],method=plan['method'],k=plan['k'],mode=plan['mode'],topN=plan['topN'],factor=plan['factor'],extra_S=plan['extra_S'],ticket_no=j+1,code=int(c),symbols=''.join(SYMBOLS[t[j]]),final_probability=eng.base[c],hits=int(hits[j]),**{f'match_{i+1:02}':SYMBOLS[t[j,i]] for i in range(13)}))
  for kind,items in [('added',additions),('removed',removed)]:
   for c in items:opportunity.append(dict(id=plan['id'],round=rnd,change=kind,code=int(c),symbols=''.join(SYMBOLS[decode([c])[0]]),final_mass=eng.base[c]))
  # All plans evaluated on identical union-mask/strength distributions per round.
  for factor in [1.,1.5,2.]:
   for mask in commonmasks[rnd]:
    v=eng.mass(mask,factor,codes).sum();b=eng.mass(mask,factor,base).sum();opt=eng.mass(mask,factor,eng.top(mask,factor,FULLMASK)[:10]).sum()
    scenario_rows.append(dict(id=plan['id'],round=rnd,eval_factor=factor,mask=mask,prob13=v,baseline_prob13=b,relative_to_oracle=v/opt,added_mass=eng.mass(mask,factor,additions).sum(),removed_mass=eng.mass(mask,factor,removed).sum()))
  poolmatrix=decode(pool)
  rawmatrix=decode(raw) if raw is not None else None
  for i,r in enumerate(g.itertuples()):
   possible=truth[i]!=1 or bool(allowed>>i&1)
   rawhas=possible if rawmatrix is None else bool((rawmatrix[:,i]==truth[i]).any())
   effhas=bool((poolmatrix[:,i]==truth[i]).any())
   q=state(eng.p,1<<i,plan['factor']) if i in plan['selected'] else eng.p
   matchrows.append(dict(id=plan['id'],round=rnd,suite=plan['suite'],method=plan['method'],k=plan['k'],mode=plan['mode'],topN=plan['topN'],factor=plan['factor'],extra_S=plan['extra_S'],match_no=i+1,card=r.home_team+'－'+r.away_team,
    actual=SYMBOLS[truth[i]],candidate=i in plan['selected'],allowed=possible,in_raw_pool=rawhas,in_effective_pool=effhas,in_tickets=bool(cover[i]),
    H_count=int((t[:,i]==0).sum()),D_count=int((t[:,i]==1).sum()),A_count=int((t[:,i]==2).sum()),H=eng.p[i,0],D=eng.p[i,1],A=eng.p[i,2],
    own_active_H=q[i,0],own_active_D=q[i,1],own_active_A=q[i,2],actual_HA_ratio_to_best=float(eng.p[i,truth[i]]/max(eng.p[i,0],eng.p[i,2])) if truth[i]!=1 else np.nan))
 # Exact observed-combination ranks are POSTHOC only, never fed back into pools.
 focus=[x for x in plans if x['suite']=='focus' and x['factor']==2 and not x['extra_S']]
 eng=engines['toto1665'];truth=actual[actual['round']=='toto1665'].sort_values('match_no').actual_symbol.map({'H':0,'D':1,'A':2}).to_numpy();tc=int(encode(truth[None,:])[0])
 for mode in ['R','U']:
  plan=next(x for x in focus if x['mode']==mode and x['topN']=='30');ids=eng.allowed(plan['allowedmask'])
  for mask in plan['masks']:
   v=eng.mass(mask,2.,ids);pv=eng.mass(mask,2.,np.array([tc]))[0]
   rank=int((v>pv).sum()+((v==pv)&(ids<tc)).sum()+1)
   ranks.append(dict(round='toto1665',mode=mode,mask=mask,actual_combination_rank=rank,probability=pv))
 # Diagnosis-only actual ticket injection; never include in ordinary simulations.
 for plan in focus:
  if plan['topN'] not in ['30','1000','FULL']:continue
  pool=np.union1d(plan['pool'],[tc]);masks=plan['masks'];values=np.array([eng.mass(m,2.,pool) for m in masks]);opts=np.array([eng.mass(m,2.,eng.top(m,2.,FULLMASK)[:10]).sum() for m in masks]);base=eng.top(0,1.,plan['allowedmask'])[:10]
  chosen=greedy_relative_robust(pool,values,opts,base)
  totals=np.array([eng.mass(m,2.,plan['codes']).sum() for m in masks])/opts
  actualv=np.array([eng.mass(m,2.,np.array([tc]))[0] for m in masks])/opts
  bestforced=max(float((totals-np.array([eng.mass(m,2.,np.array([old]))[0] for m in masks])/opts+actualv).min()) for old in plan['codes'])
  pattern=DMASK[tc]&plan['support'];same=eng.allowed(plan['allowedmask']);same=same[(DMASK[same]&plan['support'])==pattern]
  dominators=int((eng.base[same]>eng.base[tc]).sum())
  forced.append(dict(id=plan['id'],mode=plan['mode'],topN=plan['topN'],diagnostic_only=True,actual_added_to_pool=True,actual_chosen=tc in set(chosen),objective_original=plan['objective'],best_single_forced_swap_objective=bestforced,
    same_pattern_strict_dominators=dominators,dominated_by_at_least_10=dominators>=10,actual_symbols=''.join(SYMBOLS[truth])))
 frames={'evaluation':pd.DataFrame(rows),'tickets':pd.DataFrame(tickets),'match_stage_audit':pd.DataFrame(matchrows),'opportunity_cost':pd.DataFrame(opportunity),
         'scenario_evaluation':pd.DataFrame(scenario_rows),'scenario_probabilities':pd.DataFrame(state_prob),'toto1665_actual_ranks':pd.DataFrame(ranks),'toto1665_injection_diagnostic':pd.DataFrame(forced)}
 # Pairwise same-state factor ablations against R30, within fixed strength.
 comparisons=[];e=frames['evaluation'];tix={x['id']:set(map(int,x['codes'])) for x in plans}
 for _,r in e[e.suite.isin(['focus','common'])].iterrows():
  ref=e[(e.suite==r.suite)&(e['round']==r['round'])&(e.method==r.method)&(e.k==r.k)&(e['mode']=='R')&(e.topN=='30')&(e.factor==r.factor)&~e.extra_S].iloc[0]
  comparisons.append(dict(id=r.id,reference_id=ref.id,round=r['round'],suite=r.suite,method=r.method,k=r.k,mode=r['mode'],topN=r.topN,factor=r.factor,extra_S=r.extra_S,
    changed_tickets=10-len(tix[r.id]&tix[ref.id]),delta_objective=r.objective-ref.objective,delta_final_prob13=r.final_prob13-ref.final_prob13,delta_best_hits=r.best_hits-ref.best_hits,
    delta_missing_HA=r.missing_HA-ref.missing_HA,delta_raw_pool_size=r.raw_pool_size-ref.raw_pool_size))
 frames['paired_changes']=pd.DataFrame(comparisons)
 for name,df in frames.items():df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
 return frames


def render(f):
 e=f['evaluation'];focus=e[e.suite=='focus'];common=e[e.suite=='common'];m=f['match_stage_audit'];parts=[]
 def tab(x):return '<div class="scroll">'+x.to_html(index=False,border=0,float_format=lambda v:f'{v:.8f}')+'</div>'
 def add(title,text,frame=None):parts.append('<section><h2>'+title+'</h2><p>'+text+'</p>'+('' if frame is None else tab(frame))+'</section>')
 add('今回の結論','1665・B5では上位30→100→300→1000→全許容空間で最終10口は変わらず、最高10的中のまま。正解組合せはN1000で候補に入るが、全状態で劣るため選ばれない。D禁止の問題と、H/Aの相対評価が全状態で固定される問題は別だった。628設定を記録したが、同一配券の重複設定も含み、独立した628回の実績ではない。全設定で13的中0。')
 add('検証の範囲','Dの新条件・係数を探索せず、条件SとA/B候補数3/4/5を固定。正式モデル・BuyPlan・原本は変更していない。強度1/1.5/2は感度試験で、現実の確率補正値ではない。結果は全配券の生成終了後に読み、診断用の正解券注入だけを別枠にした。')
 add('再現と共通状態','Phase3.6のA/B×3候補数×6開催=36配券を、券番号まで再現。1665の単独要因実験はB5の同じ32状態を固定。6開催の対照比較は同じ候補数のA/B状態集合の和集合を生成・目的関数とも共通にした。共通状態比較はPhase3.6の方法別状態から変更された別対照実験であり、その変更を制約解除の効果に混ぜない。分母は同じ状態の全HDA許容・最適10口確率。さらに全方式を同一の全候補数・全強度の状態表で再評価した。')
 add('目的関数と全空間の検討','最大化は min_s Σ券 p_s(券)/O_s。O_sはその状態の全HDA許容最適10口の確率和。選択は既存greedy＋最大12回交換で大域最適保証なし。全空間FULLは単なる上位1000より強い監査：変動対象試合のD配置が同じ券は全状態で確率ベクトルが比例し、その配置の基準確率上位10券があれば11位以下を選ぶ最適解は不要。したがって各配置上位10へ圧縮しても最適解の存在を失わない。ただしその候補集合からのgreedy選択が最適という意味ではない。')
 add('1665 B5：制約別比較','R=候補外D禁止、U=全13試合HDA許容。extra_Sは条件SのD分岐専用候補追加。raw_pool_truthとeffective_pool_truthは、上位N絞り込みと全状態の優越関係による圧縮を区別する。',focus[['id','mode','topN','factor','extra_S','allowed_count','raw_pool_size','effective_pool_size','raw_pool_truth','effective_pool_truth','objective','final_prob13','baseline_retention','best_hits','D_cells','missing_HA','loss_stage']])
 add('正解組合せは何位か','許容全空間を走査した事後順位。順位は候補生成には使用していない。',f['toto1665_actual_ranks'].groupby('mode').agg(minimum_rank=('actual_combination_rank','min'),maximum_rank=('actual_combination_rank','max')).reset_index())
 add('正解券を強制的に探索候補へ追加した診断','正解を通常の購入候補生成には使わない。この表だけは診断用。最良の1券強制交換の評価と、全状態で確率が上の同D配置券の数を記録した。10券以上に厳密に劣るなら、未校正状態という前提内でも正解券を選ばない理由がある。',f['toto1665_injection_diagnostic'])
 add('1665の3試合：どこで正解H/Aが消えるか','鹿島－柏、東京V－福岡、富山－札幌について、各段階の正解記号の有無を示す。H/A比はDオッズ操作では変わらない。own_active確率はその試合を優先対象としたときの値。共通状態の全確率はscenario_probabilities.csvを参照。',m[(m.suite=='focus')&m.match_no.isin([2,6,13])][['id','mode','topN','factor','extra_S','match_no','card','actual','H','D','A','own_active_H','own_active_D','own_active_A','actual_HA_ratio_to_best','in_raw_pool','in_effective_pool','in_tickets','H_count','D_count','A_count']])
 totals=common.groupby(['method','k','mode','topN','factor','extra_S']).agg(rounds=('id','size'),exact13=('exact13','sum'),mean_best_hits=('best_hits','mean'),mean_retention=('baseline_retention','mean'),mean_objective=('objective','mean'),D_cells=('D_cells','sum'),missing_HA=('missing_HA','sum'),allowed_truth_rounds=('allowed_truth','sum'),raw_truth_rounds=('raw_pool_truth','sum'),effective_truth_rounds=('effective_pool_truth','sum')).reset_index()
 totals.to_csv(OUT/'common_totals.csv',index=False,encoding='utf-8-sig')
 add('6開催比較','A/B間は同一状態集合・同一分母で評価。Uの通常候補生成はA/Bで同一になる場合がある。共通状態へSの情報は残るが、Rの禁止と分離したA/B独自の連続評価は新設していないためであり、条件Sの無効性を示すものではない。',totals)
 for rnd,rr in common[(common.topN=='30')&~common.extra_S].groupby('round'):
  parts.append('<details><summary>'+rnd+'：共通状態・N30のA/B・強度別実績</summary>'+tab(rr[['method','k','mode','factor','objective','final_prob13','baseline_retention','best_hits','D_cells','missing_HA','allowed_truth']])+'</details>')
 add('制約変更による券・評価の差','各強度のR30を参照。N拡大による目的値の改善がない場合も記録する。局所探索のため、候補拡大で結果が悪化する場合は探索経路の影響であり、許容最適値の悪化ではない。',f['paired_changes'][f['paired_changes'].suite=='focus'])
 add('追加券・除外券の機会損失','券は互いに排他的な13結果。追加券質量−除外券質量が元分布の13的中確率差に一致することを監査。仮定状態にも同じ分解を保存。H/A正解の欠落と、正解が複数券へ分散した損失を別に記録。単に異なる記号へ多様化したことに加点していない。',focus[['id','mode','topN','factor','extra_S','added_mass','removed_mass','correct_symbols','combination_gap','all_symbols_but_no13']])
 add('条件S専用候補の追加','基準H/A券からSのすべての部分集合をDにし、D以外の0・1・2試合のH/A反転を組み合わせて上位Nとは別に探索へ追加。Rでは許容されないDを追加しない。結果非依存で、追加候補の購入は強制しない。')
 bchanges=f['paired_changes']
 bc=bchanges[(bchanges.suite=='common')&(bchanges.method=='B')&(bchanges['mode']=='U')&(bchanges.topN=='30')&(bchanges.factor==2)&~bchanges.extra_S]
 br=bc.groupby('round').agg(changed_settings=('changed_tickets',lambda x:int((x>0).sum())),minimum_changed_tickets=('changed_tickets','min'),maximum_changed_tickets=('changed_tickets','max'),min_hit_change=('delta_best_hits','min'),max_hit_change=('delta_best_hits','max')).reset_index()
 add('D禁止解除：開催ごとの影響','共通状態・B・候補数3/4/5・N30・強度2の比較。Rでは18設定中17設定でD禁止により正解が不可能、Uでは全18設定で許容される。しかし最高的中が上がるとは限らず、1663の4候補では9→8。1665の3候補は7→9、4候補は8→10。これは元のB5単独状態の実験とは状態集合が異なる。',br)
 add('候補拡大の収束と探索経路','共通状態・強度2・S追加なしでは、Rの36設定はN30から全空間まで10口不変。Uの36設定中6設定（A/B重複込み）はN100で券が変わり、うち目的値改善2設定・悪化4設定。N100・300・1000・全空間は同じ10口へ収束し、最高的中数はN30と同じ。局所探索の悪化を候補拡大の不利益とは解釈しない。次段階では旧解を必ず初期解/比較解として保持すれば、この悪化は避けられる。今回、結果に合わせて解を採り直す処理は加えていない。S専用分岐追加は共通比較72設定すべてで購入10口を変えなかった。')
 add('Dオッズ強度の代償：1665 B5','強度1/1.5/2でD枠は0/8/26、元final分布の13的中確率は0.0105465573%/0.0090819528%/0.0054535092%。維持率100%/86.11%/51.71%、最高的中7/8/10、正解H/A欠落2/3/3。Dを増やした直接原因は未校正の状態仮定であり、条件Sの実証的な補正量ではない。異なる強度の最悪相対値は別の状態集合に対する指標なので、そのまま優劣比較しない。全設定を同一状態で評価したCSVを別に保存した。')
 add('H/A欠落の原因','鹿島－柏はH38.75%対A34.85%、A/H比0.899。Aは探索候補と優越関係による圧縮後にも残るが最終10口で0枚。東京V－福岡はH27.82%対A42.69%、H/A比0.652。Dオッズ2倍時はH21.48%・D45.55%・A32.96%になり、H/A比は不変。富山－札幌はH44.47%対A28.22%、A/H比0.635。後2試合の正解H/AはN30の元プールにもあるが、同じD配置の上位10以内に入れず、全空間でも10口最適解に必要な候補にはならない。元Phase3.6には圧縮処理はなかったが、最終選択が同じ劣後候補を選ばなかったことを今回再現した。')
 add('正解券の順位と選ばれない根拠','1665正解HADDHHDDHHHHAは、Rで最良状態395位、Uで716位。N300では入らずN1000では入る。同じ変動対象D配置で全状態の確率が上の券はR78券・U238券あり、その条件のもとでは正解券を含む10口は大域最適になり得ない。診断用に正解券を追加しても選ばれず、既存10口から1券を強制置換した最良の最悪相対値は0.483459→0.447939。この数値は全体再最適化の上界ではなく、1券交換の診断値。')
 add('計算時間の読み方','secondsはキャッシュを利用した各計算の実測。状態ごとの上位1000を最初に求めてN30/100/300で再利用するため、N30の時間には先行計算費用が含まれる。各Nを別プロセスで計測した独立ベンチマークではない。全空間の優越関係監査は1,594,323結果を対象にしており、有限Nを十分と仮定していない。')
 add('次段階への提案','D候補は優先評価の研究対象とし、候補外Dを禁止する根拠とはしない。S専用分岐や全状態優越関係で候補を監査し、上位30を十分とは仮定しない。採用判断では追加・除外券の質量と状態別変化を同時に示す。選択目的は校正された共同予測分布があれば10口確率和、分布不確実性を扱うなら裏付けのある状態集合・重みまたは損失上限を明示する。未校正の2倍と頑健性だけでは現実の優劣を決められない。H/A機会損失を避けるための均等配分や基準券固定は導入しない。')
 for _,r in focus.iterrows():
  tt=f['tickets'][f['tickets'].id==r.id]
  parts.append(f'<details><summary>{r.id} {r["mode"]} N={r.topN} 強度={r.factor} S追加={r.extra_S}：最高{r.best_hits}的中</summary>'+tab(tt[['ticket_no','symbols','final_probability','hits']])+'</details>')
 # Graph: N vs objective and base probability, focused factor2/no branch.
 z=focus[(focus.factor==2)&~focus.extra_S];svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 390"><rect width="900" height="390" fill="white"/><g font-family="sans-serif" font-size="14"><text x="25" y="25">1665 B5：候補拡大と最悪相対評価（0〜1）</text>']
 for mode,color in [('R','#205b87'),('U','#d77c30')]:
  zz=z[z['mode']==mode].set_index('topN');coords=[]
  for i,n in enumerate(['30','100','300','1000','FULL']):
   rr=zz.loc[n];x=90+i*165;y=330-270*rr.objective;coords.append(f'{x},{y}');svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/><text x="{x-22}" y="{y-(12 if mode=="R" else -22)}" fill="{color}">{mode}:{rr.objective:.3f}</text>')
   if mode=='R':svg.append(f'<text x="{x-15}" y="365">{n}</text>')
  svg.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}"/>')
 svg.append('</g></svg>');(OUT/'focus_convergence.svg').write_text(''.join(svg));parts.insert(4,'<img src="focus_convergence.svg" alt="候補数と評価値">')
 links=' '.join(f'<a href="{x.name}">{x.name}</a>' for x in sorted(OUT.glob('*.csv')))
 (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 3.7 配券の制約と機会損失</title><style>body{font:16px/1.8 system-ui,sans-serif;color:#233449;max-width:1250px;margin:30px auto;padding:0 20px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:6px 9px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f6}section{margin:36px 0}details{padding:10px;border:1px solid #ddd;margin:12px 0}img{max-width:100%}a{margin-right:14px}</style><h1>Phase 3.7：制約・候補絞り込み・Dオッズの機会損失</h1>'+''.join(parts)+'<h2>CSV・監査</h2>'+links+'<p><a href="fixed_spec.json">固定仕様</a><a href="source_audit.json">原本監査</a><a href="next_design.md">次段階設計</a></p></html>')
 (OUT/'next_design.md').write_text('''# 次段階の配券設計案\n\nD候補評価、HDA許容、探索候補、10口選択を独立させる。候補外Dは禁止せず、条件Sは未検証の評価仮説として記録する。D首位・D最低枚数・固定ダブル・H/A均等配分を導入しない。\n\n今回のオッズ状態ではD配置ごと上位10券という優越関係を利用すれば、全許容空間の最適解を失わない候補集合を得られる。これは任意のλ変動では成立しないため、状態のH/A比まで変える次のモデルへ無条件に転用しない。一般の上位Nは候補数別の安定性と目的値の上界を併記して監査する。\n\n採用価値は、元final分布の追加/除外確率質量、同じ状態での評価、校正された状態の支持根拠を示して判断する。条件Sが成立したからDオッズを2倍とする根拠は未確認。後続の未使用予測でSの校正、H/Aの誤差、共同変動、状態の強度と頻度を検証してから採用する。\n\n最終目的は校正された共同分布下の10口13的中確率。頑健化を用いるなら状態集合と許容する基準分布の機会損失を明示し、重みのない状態列挙だけで現実の確率改善としない。結果による正解券注入は診断に限定する。候補拡大時・制約解除時には旧解を比較解として保持し、局所探索の経路で目的値が悪化しない設計とする。H/A比を変えないDオッズ状態ではH/A逆転の不確実性を表現できないため、校正されたH/A変動を調べる必要がある。\n''')
 print(focus[['id','mode','topN','factor','extra_S','raw_pool_size','effective_pool_size','objective','best_hits','D_cells','missing_HA','raw_pool_truth','effective_pool_truth']].to_string(index=False))
 print(f['toto1665_actual_ranks'].groupby('mode').actual_combination_rank.agg(['min','max']).to_string())
 print(f['toto1665_injection_diagnostic'].to_string(index=False))

if __name__=='__main__':
 p,e,g,m=generate();f=diagnose(p,e,g,m);render(f)
