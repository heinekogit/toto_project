"""Replay archived Next forecasts; compare frozen mix strengths, never fit outcomes."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib,html
import pandas as pd
import numpy as np
from team_conditions_research import research_forecast,read_inputs,FatigueParameters
from phase36_branch_portfolios import top_codes,decode,SYMBOLS
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/reports/team_conditions_research'
ROUNDS=[1660,1663,1664,1665,1668,1670]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 prod=[ROOT/'scripts/next_model'/n for n in ['forecast_pipeline.py','expected_goals.py','absence_adjustment.py','goal_rate_adjustment.py','lab_matchup.py','run_next_buyplan.py','build_next_buyplan.py']]+[ROOT/'scripts'/n for n in ['06_calculate_fatigue.py','uncertainty_calibration.py','fatigue_merge.py']]+[ROOT/'data/models/uncertainty_calibration_v1.json']
 hashes={str(p):sha(p) for p in prod}
 old=json.loads((ROOT/'data/reports/score_structure_phase39/audit.json').read_text())['source_hashes']
 diag=pd.read_csv(ROOT/'data/reports/score_structure_phase38/match_diagnostics.csv')
 matches=[];tickets=[];portfolios=[];inputs=[];current_inputs=[]
 config=json.loads((ROOT/'data/models/uncertainty_calibration_v1.json').read_text())
 spec=dict(strengths=[1,.5,0],definition='effective saved mix multiplied by strength; not coefficient refitting',fatigue_lambda_parameters=None,weather_team_parameters=None,history='same prematch A78',selection='exact top10 per evaluation distribution, no restrictions; not adoption',legacy_replay_config=config,production_applied=False)
 (OUT/'spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
 for rnd in ROUNDS:
  p=ROOT/f'data/eval/toto_rounds/toto{rnd}/snapshot/next_predictions.csv';assert sha(p)==old[str(p)];hashes[str(p)]=sha(p)
  d=pd.read_csv(p).sort_values('match_no');actual=diag[diag['round']==f'toto{rnd}'].sort_values('match_no');assert list(d.match_no)==list(actual.match_no)
  qs={};sets={}
  for strength in [1.,.5,0.]:
   values=[]
   for (_,r),(_,truth) in zip(d.iterrows(),actual.iterrows()):
    inp=read_inputs(r)
    if strength==1:
     inputs.append(dict(round=rnd,match_no=r.match_no,card=truth.card,fatigue_home=inp.home.fatigue,fatigue_away=inp.away.fatigue,fatigue_sum=inp.fatigue_sum,fatigue_difference=inp.fatigue_difference,team_inputs_json=json.dumps(asdict(inp),ensure_ascii=False,default=str),directional_status='未校正・未適用'))
    s=r.to_dict();active=float(truth.uncertainty_mix)>0
    if active:
     for name in ['fatigue_weight','weather_weight','max_mix']:
      s.setdefault('condition_'+name,config[name])
    out=research_forecast(s,mix_strength=strength if active else 0.)
    probability=np.array(out['probabilities']);values.append(probability)
    if strength==1:assert np.allclose(probability,[r.next_prob_home,r.next_prob_draw,r.next_prob_away],atol=1e-12)
    y='HDA'.index(truth.actual_symbol)
    matches.append(dict(round=rnd,match_no=r.match_no,card=truth.card,model_signature=truth.model_signature,strength=strength,mix=out['mix'],H=probability[0],D=probability[1],A=probability[2],actual=truth.actual_symbol,LL=-np.log(probability[y]),Brier=((probability-np.eye(3)[y])**2).sum(),lambda_home=out['lambda_home'],lambda_away=out['lambda_away'],directional_status='未校正・未適用'))
   qs[strength]=np.array(values);sets[strength]=decode(top_codes(qs[strength],[range(3)]*13,count=10));assert len({tuple(t) for t in sets[strength]})==10
  for strength,ts in sets.items():
   def mass(prob):return float(np.prod(prob[np.arange(13),ts],axis=1).sum())
   base_set={tuple(t) for t in sets[1.]};new=len({tuple(t) for t in ts}-base_set)
   portfolios.append(dict(round=rnd,strength=strength,probability_own=mass(qs[strength]),probability_original=mass(qs[1.]),probability_no_mix=mass(qs[0.]),replaced_tickets=new))
   for j,t in enumerate(ts):tickets.append(dict(round=rnd,strength=strength,ticket=j+1,symbols=''.join('HDA'[int(k)] for k in t),original_probability=float(np.prod(qs[1.][np.arange(13),t])),own_probability=float(np.prod(qs[strength][np.arange(13),t]))))
 for league in ['j1','j2']:
  p=ROOT/f'data/team_fatigue_scores_{league}_2026.csv';hashes[str(p)]=sha(p);d=pd.read_csv(p)
  current_inputs.append(dict(league=league,rows=len(d),coverage=str(d.external_schedule_coverage.value_counts().to_dict()),travel_missing=int(d.travel_lookup_status.ne('hit').sum()),missing_home_rest=int(d.home_rest_hours.isna().sum()),missing_away_rest=int(d.away_rest_hours.isna().sum()),home_external_unknown=int(d.home_external_load_unknown.sum()),away_external_unknown=int(d.away_external_load_unknown.sum())))
 frames={'match_comparison':pd.DataFrame(matches),'tickets':pd.DataFrame(tickets),'portfolio_comparison':pd.DataFrame(portfolios),'team_inputs':pd.DataFrame(inputs),'current_schedule_coverage':pd.DataFrame(current_inputs)}
 m=frames['match_comparison'];ag=[]
 for dim in ['round','model_signature']:
  for (key,strength),g in m.groupby([dim,'strength']):ag.append(dict(dimension=dim,group=key,strength=strength,n=len(g),LL=g.LL.mean(),Brier=g.Brier.mean()))
 for strength,g in m.groupby('strength'):ag.append(dict(dimension='all',group='all78',strength=strength,n=len(g),LL=g.LL.mean(),Brier=g.Brier.mean()))
 frames['summary']=pd.DataFrame(ag)
 for n,f in frames.items():f.to_csv(OUT/(n+'.csv'),index=False,encoding='utf-8-sig')
 # Synthetic fixtures check wiring only; never used to generate historic forecasts.
 cases=[]
 for h,a in [(14,2),(2,14),(14,14),(14,0)]:
  r=dict(home_team='synthetic_H',away_team='synthetic_A',home_total_fatigue_score=h,away_total_fatigue_score=a,final_lambda_home=1.3,final_lambda_away=1.3,weather_missing=True)
  z=research_forecast(r,fatigue_parameters=FatigueParameters(.01,.005,'unit-test fixture only'))
  cases.append(dict(home_fatigue=h,away_fatigue=a,lambda_H=z['lambda_home'],lambda_A=z['lambda_away'],H=z['probabilities'][0],D=z['probabilities'][1],A=z['probabilities'][2],scope='合成テスト専用・推奨係数ではない'))
 pd.DataFrame(cases).to_csv(OUT/'synthetic_direction_checks.csv',index=False,encoding='utf-8-sig')
 assert all(sha(Path(p))==h for p,h in hashes.items())
 (OUT/'audit.json').write_text(json.dumps(dict(hashes=hashes,unchanged=True,original_reproduced=78,unique_10_each=True,no_directional_fit=True,no_production_import=True),indent=2))
 render(frames,pd.DataFrame(cases))
 print(frames['summary'][frames['summary'].dimension=='all'].to_string(index=False));print(frames['portfolio_comparison'].to_string(index=False))

def render(f,cases):
 parts=[];md=[]
 def add(t,s,frame=None):
  parts.append('<section><h2>'+html.escape(t)+'</h2><p>'+html.escape(s).replace('\n','<br>')+'</p>'+('' if frame is None else '<div class="scroll">'+frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.8f}')+'</div>')+'</section>');md.append('## '+t+'\n\n'+s+'\n'+('' if frame is None else frame.to_string(index=False))+'\n')
 add('実装済み／正式未適用を明確に分ける','研究用team_conditions_research.pyを新設。入力、チーム別攻守影響、λ、得点分布、mixを分離した。本番forecast_pipeline・正式係数・購入処理は変更していない。これは未校正係数とmixの正式変更を禁止する今回の指示に従ったもの。現在の本番mixは従来のまま残る。新しい方向補正を導入済みと誤解しないこと。')
 add('1. 疲労・カップ戦・ACL・移動の現在地','06_calculate_fatigue.pyはリーグの消化/予定日程に、確認済みACL、保存totoのルヴァン/天皇杯等の結果、emperor_cup_eventsと手動external_match_eventsを統合。現在acl_schedule.csvは52行中confirmed12、unverified40で、40行は採用しない。external_match_events.csvは存在しない。外部大会はpartial coverageで全日程を網羅した保証はない。休養は前試合からの実時間、4日以下の間隔に負荷、直近3試合の負荷を0.35/0.20/0.10で持ち越し、14日超の中断で持越しを消す。移動は本拠地間距離×0.005、アウェー固定負荷4も加算される。実際の前会場からの連続遠征経路・帰宅有無・選手出場時間を常に取得しているわけではない。カップ出場は休養時間へ反映されるが、不明な負荷量・ローテーションは未確認。',f['current_schedule_coverage'])
 add('欠損をゼロと誤認する経路','ACL距離は通常、acl_schedule.pyがチーム本拠地と試合地の座標から計算する。現存52行の距離欠損は0、confirmed12行のうち海外遠征6行はすべて正の距離で、実際に負荷へ接続されていた。fillna(0)は下流の防御不足として残るが、現在の遠征が距離不明で0になっている事実は確認されなかった。通常経路の座標未登録は例外。ただし旧memo上書き経路は座標未登録でも初期値0が残り得る。最初の試合で休養履歴が無い場合も休養負荷0。上流use_generic_fatigue_onlyにも欠損を0へ置く経路がある。現行conditions_v2は移動lookup欠損を通常予測前に拒否する一方、外部大会網羅性やACL不明距離の保証にはならない。修正候補はmissing距離/休養/大会coverageを別保存し、不明を既知0へ変換しないこと。新研究入力はNoneを保持し、明示欠損をfallbackしない。既にゼロ化された古い原本から真の負荷は復元しない。')
 add('2・3. 合算箇所とチーム別λ接続','旧Nextのuncertainty_calibration.signalsが(H総疲労+A総疲労)/16を使い、どちらの疲労かを捨てていた。現行Nextには疲労差からλへの方向補正は無い。新研究経路はhome/away各入力を保持し、合計と差を診断用に別出力する。攻撃のlog率変化は自チームλへ、守備の失点log率変化は相手λへ接続する。λH′=λH×exp(H攻撃変化+A守備失点変化)、λA′=λA×exp(A攻撃変化+H守備失点変化)。指数接続も研究上の構造選択で、効果量は未校正。既存範囲0.15〜4.5で制限した場合はclippedを記録し、11×11得点分布からH/D/Aを再計算。')
 add('方向の合成テスト','下表は攻撃係数0.01、守備係数0.005を使用したテスト専用fixture。実データへの採用案・推奨値ではなく、歴史78試合には使っていない。14/2と2/14で方向が逆転、14/14と14/0も区別できることを確認した。',cases)
 add('4. 天候と欠場','天候取得は会場座標とkickoffを照合しOpen-Meteoを使用。現在feature_builderは最寄り時刻が60分以内か確認し、kickoff前後1時間と開始〜終了付近の時間窓、風速単位、取得時刻、forecast等の種別を残す。Nextのmixは雨・強雨・強風の共通signalのみ使用し、気温やチーム戦術によるλ差は使わない。weather_style_shadow.pyの仮説は存在するが本番Nextの校正済み感応度ではない。新構造では共通気象とチーム別感応度を分離し、感応度が無ければ勝手な差を作らずpending。\n欠場は既存の攻守別λ加算を維持。自攻撃欠場は自λを下げ、自守備欠場は相手λを上げる。missingと人数0を区別。過去原本の欠場観測timestampは不足しており、試合前原本Aであることだけから台帳完全性は保証しない。新入力にも観測時刻を独立保存。')
 add('5. 二重評価と欠損を防ぐ実装','保存final_lambdaは既に欠場補正後なので欠場を再適用すると例外。総疲労は移動を含むため、総疲労と移動の同時加算も例外。疲労/移動をλへ適用しながら疲労mixを適用すること、天候をλと天候mixで二重利用することも拒否する。欠損疲労や既知の外部負荷欠損は係数適用時に拒否。weather sensitivityには両チームについて明示係数・参照値・根拠が必要。同じ気象でも影響を同じと決め打ちせず、データ無しで違いも捏造しない。')
 add('6. 未校正のため保留したもの','新疲労攻守係数・チーム別天候係数・疲労成分ごとの係数は設定していない。実データの方向補正は未適用。過去の疲労スコア自体にも固定アウェー罰則や移動の代理値が含まれ、それを実証済み生理負荷と扱わない。今の構造で校正データを渡せるが、Noneが正式設定の代わりにゼロ係数として採用されることはない。')
 add('7. 現行mix・半分・なしの実データ比較','同じ保存λ・欠場・LABを固定し、有効mixを1倍/0.5倍/0倍にした研究比較。初期39試合は元々mixなし。78試合の現行確率を全件再現。新疲労係数による予測改善を主張する実験ではない。LL/Brierは小さいほど良い。',f['summary'])
 pp=f['portfolio_comparison'].copy()
 for col in ['probability_own','probability_original','probability_no_mix']:pp[col]*=100
 add('10口のモデル確率比較（表は%）','各分布の確率上位10組を独立に選び、相異なる10口で13的中となる確率を積の和で厳密計算した。全試合H/D/Aを許可。ownは各方式自身の分布、originalは元final、no_mixはmix前の同じ分布で評価。分布が違うown同士の増加を実際の的中率改善と解釈しない。既存の正式購入券を再生成/置換していない。券一覧はtickets.csv。',pp)
 add('8. 正式導入前の必要事項','外部大会の収録範囲、ACLの日時/移動距離、不明負荷、会場間の実移動、出場時間・ローテーション、欠場台帳の観測/復帰時刻、気象取得時刻と時間窓、チーム特性を試合前に固定保存する。疲労総点の単純投入と、休養・移動・持越しの成分モデルを混在させず比較する。過去期間で係数推定、後続未使用期間で得点NLL・HDAのLL/Brier・校正を検証し、会場効果・既存xG/欠場補正との重複を調べる。今回の39適用試合に合わせた係数を正式採用しない。')
 chart=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 300"><rect width="760" height="300" fill="white"/><text x="20" y="25" font-size="18">78試合 Log Loss：mix強度比較（縦軸0から）</text>']
 for j,r in enumerate(f['summary'][f['summary'].dimension=='all'].sort_values('strength').itertuples()):
  x=90+j*220;height=r.LL*180;chart.append(f'<rect x="{x}" y="{245-height}" width="100" height="{height}" fill="#387d9d"/><text x="{x}" y="{235-height}">{r.LL:.6f}</text><text x="{x}" y="275">mix ×{r.strength:g}</text>')
 chart.append('</svg>');(OUT/'mix_comparison.svg').write_text(''.join(chart))
 (OUT/'report.md').write_text('# チーム別条件反映の実装・監査\n\n'+'\n'.join(md))
 links=' '.join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(OUT.glob('*.csv')))
 (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>チーム別条件反映</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.8 system-ui;color:#24384a}section{margin:32px 0}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{padding:7px;border-bottom:1px solid #ddd}th{background:#eef3f5}img{max-width:100%}a{margin-right:15px}</style><h1>疲労・移動・天候・欠場のチーム別反映</h1><img src="mix_comparison.svg" alt="mix比較">'+''.join(parts)+'<h2>データ</h2>'+links+'<p><a href="spec.json">固定比較条件</a><a href="audit.json">監査</a><a href="report.md">日本語レポート</a></p></html>')
if __name__=='__main__':main()
