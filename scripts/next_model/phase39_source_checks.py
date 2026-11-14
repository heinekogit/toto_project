from phase39_input_audit import *
from jleague_team_names import canonical_team_name

def main():
 rows=[];audit=json.loads((OUT/'audit.json').read_text())
 for rnd in ROUNDS:
  d=pd.read_csv(ROOT/f'data/eval/toto_rounds/toto{rnd}/snapshot/next_predictions.csv')
  for _,r in d.iterrows():
   league=str(r.match_id)[:2];date=str(int(r.flab_snapshot_date));p=ROOT/f'data/external_metrics/football_lab_team_metrics_{date}_{league}.csv'
   if p.exists():
    audit['source_hashes'][str(p)]=sha(p);lab=pd.read_csv(p)
    for side in ['home','away']:
     match=lab[lab.team_name.map(canonical_team_name)==canonical_team_name(r[side+'_team'])]
     for field in ['flab_expected_for_xg','flab_expected_against_xg']:
      same=len(match)==1 and np.isclose(float(match.iloc[0][field]),r[field+'_'+side])
      rows.append(dict(round=rnd,match_no=r.match_no,check='LAB source '+side+' '+field,status='一致' if same else '不一致または照合不能',source=str(p.relative_to(ROOT))))
   else:rows.append(dict(round=rnd,match_no=r.match_no,check='LAB source',status='未保存',source=str(p)))
   if 'next_uncertainty_calibration_mix' in d:
    boolean=lambda x:str(x).lower() in ['true','1']
    known=boolean(r.next_uncertainty_weather_known)
    f,w=signals([r.home_total_fatigue_score],[r.away_total_fatigue_score],[boolean(r.is_rain)],[boolean(r.is_heavy_rain)],[boolean(r.is_strong_wind)],r.get('condition_rain_signal',.35),r.get('condition_heavy_rain_signal',1.),r.get('condition_strong_wind_signal',.65))
    w=w[0] if known else 0
    rows.append(dict(round=rnd,match_no=r.match_no,check='weather signal',status='一致' if np.isclose(w,r.next_uncertainty_weather_signal) else '不一致',source='saved weather flags'))
   if 'goal_rate_adjustment_version' in d:
    for side in ['home','away']:
     for kind,metric in [('for','1試合平均得点数'),('against','1試合平均失点数')]:
      pref=f'goal_rate_{kind}_{side}'
      pub=r[pref+'_published'];expected=max(.05,pub+r[pref+'_weighted']-r[pref+'_ordinary']) if r[pref+'_games']>=8 else pub
      ok=np.isclose(expected,r[pref+'_used']) and np.isclose(expected,r[f'stats_{metric}_{side}'])
      rows.append(dict(round=rnd,match_no=r.match_no,check='outlier saved arithmetic '+side+' '+kind,status='一致' if ok else '不一致',source='saved published/ordinary/weighted/used'))

 result=pd.DataFrame(rows);result.to_csv(OUT/'source_checks.csv',index=False,encoding='utf-8-sig');print(result.groupby(['check','status']).size().to_string())
 audit['source_checks']=result.groupby('status').size().to_dict();(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
