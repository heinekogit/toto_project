"""Render the production change audit; never edits model settings."""
from pathlib import Path
import json,html
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/reports/next_conditions_production'
def main():
 parts=[];md=[]
 def add(title,text,frame=None):
  md.extend(['## '+title,'',text,'']);parts.append('<section><h2>'+html.escape(title)+'</h2><p>'+html.escape(text).replace('\n','<br>')+'</p>')
  if frame is not None:
   parts.append('<div class="scroll">'+frame.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'</div>');md.extend([frame.to_csv(index=False),''])
  parts.append('</section>')
 cfg=json.loads((ROOT/'data/models/next_conditions_v2.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
 add('本番へ反映した内容',f"Next Modelの標準設定を{cfg['version']}へ変更。適用開始情報は{cfg['effective_at']}。疲労・天候の一様mix強度を0にした。以後の予測呼出しに反映される。過去予測・購入券は再生成していない。LAB、欠場、直近得失点の外れ値補正、学習済みstyle補正の計算は維持した。")
 add('まだ反映していない内容','疲労・移動の新しい攻守係数、チーム固有の天候係数は未校正のため無効。したがって現在、疲労・天候による直接のλ・H/D/A補正はない。欠場は既存のチーム別λ補正として有効。この暫定状態を、チーム別条件反映の完成とは扱わない。')
 add('計算経路と欠損の扱い','入力→得失点外れ値調整→基礎λ→LAB→欠場→チーム別条件フック→得点分布→mix強度0→既存style処理、の順。自チーム攻撃は自λ、守備の失点増加は相手λに接続する。校正済みの明示指定・検証資料ファイルがなければ有効化を拒否する。欠場の再適用、疲労総点と移動の重複、方向補正と旧mixの同時適用を拒否する。\n未知の疲労・休養・天候・欠場をNaN/未取得として保存。報告済み疲労値はfatigue_reportedに分離。不完全な総点で負荷0を主張しない。既知の成分小計はfatigue_known_subtotalであり総負荷ではない。既存の不足履歴は推測で復元しない。')
 add('日程・移動監査','リーグ日程、確認済みACL、保存totoのカップ戦結果、emperor_cup_events、存在すればmanual external_match_eventsを読み込む。現存ACL52行中12行を確認済みとして使用、未確認40行は除外。海外6件は正の座標由来距離であり0扱いではない。manual external_match_eventsは未配置。外部大会の収録はpartialで網羅保証なし。今回、新しい未確認日程を投入していない。\n休養は前試合との時間差。4日以下の休養負荷、距離×0.005、アウェー固定4、直近3負荷の0.35/0.20/0.10持越しという既存尺度を維持。リーグ移動は本拠地間の代理距離で、実際の前会場→次会場・帰宅・ローテーションを完全には把握していない。ホームの今回移動0はこの代理尺度の定義であり、実移動ゼロの観測ではない。\n新しいconditions_v3では未知の休養・持越し・距離をNaNにする。未知の履歴を含む持越しは保守的に保留し、既存の14日超リセットまで完全値とはみなさない。失われた過去負荷は逆算しない。ACLの不正距離・座標解決エラーは依然明示エラー。通常距離の欠損はpartial＋NaNを条件に保存可能とし、Next側で補正保留する。',pd.read_csv(OUT/'archived_current_schedule_coverage.csv'))
 add('天候・欠場監査','天候は会場座標・キックオフ時刻に対応する予報を取得し、時間窓・取得時刻・予報/実測・鮮度を別記録する。古い/部分的な予報、欠落フラグをFalseへ変換せず、気温・風速の中央値埋めも廃止。共通の気象観測と、各チームの感応係数を分離した。感応係数は未導入。旧11_prediction側の天候処理はNextとは別であり、Nextの直接補正と混同しない。\n欠場の自攻撃低下→自λ減、守備欠損→相手λ増という既存処理を維持。未取得は補正保留。過去原本に取得時刻がなければ「試合前確認済み」とは補完しない。台帳の生成時刻・復帰情報の継続保存が必要。')
 comp=pd.read_csv(OUT/'production_13_comparison.csv');table=comp[['match_no','home_team','away_team']].copy()
 for s in ['home','draw','away']:
  table['旧'+s+'%']=100*comp['old_'+s];table['新'+s+'%']=100*comp['new_'+s];table['差'+s+'pt']=comp['delta_pp_'+s]
 add('本番関数の実行確認：第1670回13試合','実際のforecast_pipeline.build_next_probabilitiesを同じ保存入力で実行した。保存済みの外れ値調整済み特徴量を利用し、過去の外れ値計算を現在データで再実行していない。標準の外れ値処理自体は回帰テストで確認。旧設定と新設定のλは同じ。新finalとmix前の得点分布は全13試合一致。最大変化は約2.8269ポイント。実行結果は研究用ディレクトリだけへ保存。',table)
 summaries=pd.read_csv(OUT/'archived_summary.csv');add('78試合のmix比較','既存の同一保存λの再現比較を再利用。初期39試合は元々mixなし、後期39試合にmixあり。表は全78試合。低いほどよいが、反復使用済み小標本であり、将来の改善を証明しない。',summaries[summaries.dimension.eq('all')][['strength','n','LL','Brier']])
 port=pd.read_csv(OUT/'archived_portfolio_comparison.csv')
 for c in ['probability_own','probability_original','probability_no_mix']:port[c]*=100
 add('10口全的中のモデル確率（%）','既存研究の相異なる確率上位10口による診断を再掲。ownは各方式自身の分布、original/no_mixは同一の評価分布。分布を変えたown同士の増加は実際の全的中率改善ではない。今回、正式購入券を生成・変更していない。',port)
 add('新しい係数の研究：採用しない',f"保存78試合中、明示的な欠損・不完全負荷を除いた{audit['coefficient_eligible_matches']}試合で探索診断。初期4開催の28試合で推定し、後続1668/1670の26試合で評価した。全て過去研究で反復使用済みであり、未使用の独立検証データではない。外部日程のcoverageは全てpartialのままで、54試合も完全網羅を保証しない。\n既存λをoffsetにするPoisson回帰を使用。H/A切片だけ、疲労攻守、休養攻守、移動攻守を個別比較。固定L2=1、係数範囲[-2,2]、結果による強度探索なし。連戦回数や出場時間は少数・欠損・相関のため独立係数を増やさなかった。モデル世代差、チーム差、既存xG/欠場/ホーム補正との交絡を完全には除けない。",pd.read_csv(OUT/'temporal_research_metrics.csv'))
 add('攻撃・守備・相対差の解釈','疲労16点あたりの攻撃損失係数は0.249（正則化Hessian由来SE約0.445）、相手守備失点増加係数は−0.197（同0.414）。攻撃低下を確証できず、守備側は仮説と逆方向で不確実。SEは厳密な校正済み信頼区間ではない。後続26試合のLLは基準1.005957に対し疲労入り1.042024、Brierは0.603454に対し0.628953。得点NLLのみ2.933470→2.916329と小幅改善したが、全指標で改善とはいえない。\n相対差はlogλHに−β攻撃×疲労H＋β守備×疲労A、logλAに逆を入れて検証。疲労14対2と2対14の方向交換は合成テストで確認済み。合成係数を正式設定へ転用していない。研究の負係数も採用していない。',pd.read_csv(OUT/'research_coefficients_NOT_DEPLOYED.csv'))
 matches=pd.read_csv(OUT/'temporal_research_matches.csv');v=matches[matches.subset.eq('evaluation')];base=v[v.model.eq('saved_lambda')][['round','match_no','LL']].rename(columns={'LL':'baseline_LL'});x=v.merge(base,on=['round','match_no']);x['better']=x.LL<x.baseline_LL-1e-12;x['worse']=x.LL>x.baseline_LL+1e-12
 add('評価試合ごとの改善・悪化','LLでの診断。どの試合が改善・悪化したかとλはtemporal_research_matches.csvに保存。研究結果に応じて本番設定を変更していない。',x.groupby('model',as_index=False)[['better','worse']].sum())
 changed=pd.DataFrame([
 ('data/models/next_conditions_v2.json','Next専用標準設定。mix0、未校正λ無効、変更日時/理由'),
 ('scripts/next_model/forecast_pipeline.py','本番経路へ状態保持・チーム別λフック・設定式mixを接続'),
 ('scripts/next_model/condition_policy.py','欠損保持、係数有効化ガード、旧mix並行保存'),
 ('scripts/next_model/team_conditions.py','既存研究の方向計算を共通モジュールへ移設'),
 ('scripts/next_model/team_conditions_research.py','旧研究importの互換アダプタ'),
 ('scripts/next_model/style_official.py','styleでλが変わったとき旧mix比較列のみ追随更新'),
 ('scripts/06_calculate_fatigue.py','未知をNaNに保持、成分小計/状態/生成時刻を追加'),
 ('scripts/fatigue_merge.py','明示partialを許容。不明なのに数値0の入力は拒否'),
 ('scripts/11_prediction_01.py','Next入力の疲労0埋め・天候False/中央値埋めを廃止'),
 ('scripts/uncertainty_calibration.py','上流旧モデルの欠損成分保留オプション。既知入力の旧計算は維持'),
 ('scripts/next_model/audit_conditions_production.py','13試合実行、保存原本監査、時系列研究'),
 ('scripts/next_model/report_conditions_production.py','本レポート再生成'),
 ('scripts/next_model/test_condition_policy.py','本番mix0・復旧・欠損・未校正ガード'),
 ('scripts/test_run_next_buyplan.py','新しい出力ポリシー名の期待値'),
 ('scripts/test_fatigue_components.py','不明値保留・ゼロ偽装拒否の期待値'),
 ('scripts/test_weather_prediction_merge.py','不明な天候をNAとして検証'),
 ],columns=['ファイル','今回の変更']);add('変更ファイル','既存の大きな作業ツリーに対し、今回の変更だけを列挙。Nextの得点計算コア・欠場・LAB・外れ値数式、BuyPlanロジックは変更していない。',changed)
 add('復旧・保存・検証','変更前設定previous_config.jsonと既存uncertainty_calibration_v1.jsonを保持。完全な入力で旧設定をconfig_pathに指定すれば従来mixを再現する。次回以降はmix_strengthを0/0.5/1で比較可能。欠損時は旧方式を無理に完全再現せずnext_legacy_mix_status=partial_components_heldと明示。出力next_prob_legacy_mix_*は研究比較列で購入入力ではない。\n78件の回帰テストが成功。既存予測原本と購入処理・購入出力のハッシュ不変を確認。検査詳細audit.json、実行テストtest_results.txt。Nextの標準設定変更は次回予測実行に適用。既に保存したbuyplan.htmlや購入券を更新していない。')
 add('正式なチーム別補正へ進む条件','外部大会の確認済み日程、実会場間の移動、休養、選手出場時間・延長/ローテーション、欠場台帳の観測/復帰時刻、予報の取得時刻を試合前レジストリへ保存する。新しいmeasurement_v3と過去v2を混ぜず評価し、未知の持越しは保留する。未使用の後続期間を確保し、得点NLL・H/D/A LL/Brier・校正を併せて確認する。fatigue係数を有効化する前に、会場・チーム・xG・欠場と日程の交絡、相関する疲労成分の二重計上を検証する。現段階の係数は不採用。')
 (OUT/'report.md').write_text('# Next Model チーム別条件・本番修正\n\n'+'\n'.join(md))
 links=' '.join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(OUT.glob('*.csv')))
 (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Next Model 本番条件修正</title><style>body{max-width:1280px;margin:32px auto;padding:0 20px;font:16px/1.8 system-ui;color:#21354a}section{margin:35px 0}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{padding:7px;border-bottom:1px solid #ccd5df}th{background:#eef3f7}a{margin-right:14px}</style><h1>Next Model チーム別条件・本番修正</h1><p><strong>mixは本番で暫定無効化。チーム別λ補正は接続済み・係数未適用。</strong></p>'+''.join(parts)+'<h2>CSV</h2>'+links+'<p><a href="audit.json">監査</a><a href="coefficient_spec.json">研究条件</a><a href="test_results.txt">テスト</a></p></html>')
if __name__=='__main__':main()
