"""Render audit findings from Phase39 CSV, with uncertainty and source evidence."""
from phase39_input_audit import OUT,ROOT
import pandas as pd,numpy as np,json,html

def build():
 read=lambda n:pd.read_csv(OUT/(n+'.csv'))
 q=read('input_quality');m=read('mix_matches');a=read('mix_summary');c=read('correction_stages');s=read('denominator_sensitivity_NOT_corrected');h=read('home_away_calibration');ci=read('correction_inputs')
 issues=[
 ('確定・設計上の脆弱性','本番で丸め済みxGAから試合数を逆算','156側入力中47で同日順位表と不一致。正しいxG収録数との同一性は未確定。','収録試合数を取得・保存し、期間一致を確認して明示的分母へ移行する案。今回は変更しない'),
 ('データ不足','集計期間・収録数・指標別取得時刻','snapshotの日付はあるが、各指標の集計始終・xG収録率・ホーム/アウェー内訳なし。','取得URL、competition、指標別timestamp、period、coverage、gamesを保存'),
 ('疑い','平均xGA表示の丸め/更新不整合','順位表試合数で除した値の小数1桁切捨てと152/156が整合。FC東京の4側入力はこの説明から外れる。表示仕様未確認。','過去原HTML・公式の収録対象を確認。表示値の切捨てと断定しない'),
 ('確定・検証の適用範囲','mix係数の学習系列がNext固有ではない','train_uncertainty_calibration.pyは観測履歴p_home等、作成器はprob_final等を優先。Next確率の検証成績とは扱えない。','Next raw/finalを後続未使用開催で対比較。旧方式を予測規則として使わない'),
 ('疑い・校正','ホーム側予測不足','104.43対122だが1670は16.96対13。明示的HFAはなく、全会場平均には歴史的効果が混在。','同一版・リーグ・会場別の校正を蓄積。今は一律係数なし'),
 ('確定・保存不足','欠場入力の時点・ゼロの意味','適用78側中22側はmissing。人数0を欠場なしと読み替えない。原本には欠場情報取得時刻なし。','台帳の観測時刻・復帰・完全性とhashを保存。missingはゼロとは別'),
 ('疑い・統計的重複','LABと基礎xGの情報重複','別取得系列でも両方xG。係数25%の統計的独立性を意味しない。数式再現は78/78。','共通試合・予測時点で両系列の期間と未使用期間の差分を評価'),
 ('確定・比較範囲制約','外れ値補正の再現範囲','保存λ差分がある1670の13試合で比較。現行結果DBから過去入力を上書きしていない。','対象履歴ID/hashと各過去試合の利用可能時刻を保存'),
 ('疑い・運用経路','上流stats解決の未来fallback','11_prediction_01.pyは指定asof以下が無い時最新へfallback。78原本には未来stats日付なし。Next preflightは別途鮮度検査。','過去再現はfail-closedを検討。今回この経路を実行/修正していない')]
 pd.DataFrame(issues,columns=['status','issue','evidence','proposal']).to_csv(OUT/'issues.csv',index=False,encoding='utf-8-sig')
 protocol=dict(version='phase39-fixed-1',scope='research only',freeze_before_kickoff=['source SHA256, model/config SHA256, capture timestamp','all final lambdas, raw HDA, final HDA, actual saved mix and signals','condition S: within13 P00<=median AND P11>=median','xG/xGA totals, rates, directly observed games, coverage, period, metric URL/timestamp','LAB fields, absence missing vs known zero, ledger timestamp/hash, historical rate audit'],
 evaluation=['Join complete outcomes by unique match_id; no probability recomputation from updated team data','paired LL/Brier plus goal NLL, H/A residual and calibration per round/version/league','report all rounds; do not stop at favorable results; round-level uncertainty','S remains discovery hypothesis, not probability correction; no selection of four candidates'],
 adoption=['Verify input semantics independently from results first','Reserve later unused data before any fitting; assess consistent paired gains and calibration with uncertainty','Do not infer latent lambda error from realized residual','No automatic deployment or learned coefficients in these scripts'],
 current_78='exploratory reused sample; not independent validation',implemented_capture='scripts/next_model/phase39_future_check.py capture',implemented_evaluation='scripts/next_model/phase39_future_check.py evaluate')
 (OUT/'future_validation.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
 sections=[];md=[]
 def add(title,text,frame=None):
  sections.append('<section><h2>'+html.escape(title)+'</h2><p>'+html.escape(text).replace('\n','<br>')+'</p>'+('' if frame is None else '<div class="scroll">'+frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')+'</div>')+'</section>')
  md.append('## '+title+'\n\n'+text+'\n'+('' if frame is None else '\n```text\n'+frame.to_string(index=False)+'\n```')+'\n')
 add('結論：入力の確定修正と性能評価を区別する','本番Next Modelの試合数逆算に脆弱性があり、順位表との不一致47/156を確認した。転記されたxG・xGA値自体は保存statsと156/156で一致。xG収録期間・分母の対応が確定していないため、順位表を正しい分母として正式に置換していない。mixは39試合で適用され15改善・24悪化。今回、新係数・修正済みと称する予測・10口・正式変更は0。')
 add('① 本番か研究用か','forecast_pipeline.build_next_probabilities → expected_goals.base_lambdas_v1 → _attack_rate_v1 → _games_from_xga が本番経路。n=round(累積被xG/平均被xG)、攻撃=(得点率+xG総量/n)/2、守備=(失点率+平均被xG)/2、λH=sqrt(H攻撃×A守備)。研究コードは同じ関数を使用。零・欠損ではfallbackがある一方、現行preflightは非正値を拒否する。研究再現だけの問題ではない。')
 add('② 取得・期間・as-ofの監査','20_update_stats.pyはJリーグ公式stats/clubページの表示値をfloatへ変換。J1/J2別、2026は2026-27 URL優先、旧URLfallbackあり。J2は従来j2j3源を設定し現行URLはj2、チーム集合で絞る。数値を自前集計したxGではない。指標ごとの試合数・収録開始終了・取得URLをCSVには残さない。延期/未消化は統計側に委ねており、節番号を消化数と同一視できない。\n対象6開催は2026-10-10〜11-07の保存stats。全156側で同日順位表regular_2026_27の消化数と照合でき、未来stats日付は0、保存値とstatsファイルのxG3項目の不一致0。これは対象試合後の値が使われた証拠が無いという確認であり、指標内部の収録終了時刻まで証明したものではない。試合前原本Aのhashを再確認。最新HTMLは過去の証拠に流用しない。')
 summary=q.groupby(['round','league']).agg(team_inputs=('team','size'),mismatches=('count_mismatch','sum'),games_min=('standings_games','min'),games_max=('standings_games','max')).reset_index()
 add('逆算数と順位表の差', '順位表は直接得られる競技の消化試合数。ただしxGがその全試合を収録している証拠は別途必要。ここで47件をすべて誤分母確定とはしない。',summary)
 add('丸めの危険性','例：被xG14.0、平均1.2なら本番逆算は12試合。しかし同日順位表の柏は11試合で、14/11=1.2727。平均値が小数1桁切捨てなら1.2と両立する。0.1刻み四捨五入を仮定した候補集合と切捨てを仮定した候補集合をCSVで併記した。どちらも表示仕様の確定ではない。152/156は順位表分母と切捨て表示で整合するが、FC東京の4例は別途調査が必要。xG/被xGそれぞれの直接分母は保存されておらず、同一分母も未証明。',q[q.count_mismatch][['round','match_no','team','xga_total','xga_display_rate','inferred_games','standings_games','nearest_candidates','floor_candidates']])
 add('分母不一致のλ影響：未確認の感度試算','同じ収録範囲と仮定し順位表分母だけへ置換した試算。基礎λの絶対差最大はH '+f'{s.delta_base_H.abs().max():.6f}、A {s.delta_base_A.abs().max():.6f}。LAB・欠場・mixを保存条件で固定した確率差最大はH {100*s.delta_P_H.abs().max():.4f}、D {100*s.delta_P_D.abs().max():.4f}、A {100*s.delta_P_A.abs().max():.4f}ポイント。'+'正しい入力が確定した再予測ではないため実績比較で採否を決めず、10口へ接続しない。平均被xGは今回変更していない。',s[['round','match_no','card','delta_base_H','delta_base_A','delta_P_H','delta_P_D','delta_P_A']])
 add('③ mix仕様と実際の効果','fatigue=clip((H総疲労+A総疲労)/16,0,1)。weather=強雨1.0/雨0.35＋強風0.65を上限1、未取得はweather効果0と明示。f_effect=0.25×fatigue、w_effect=0.25×weather、mix=min(1−(1−f_effect)(1−w_effect),0.25)。final=(1−mix)raw+mix/3。確率の一様化であり、特定側やDへの現実の遷移確率ではない。λは変えない。\n1660/1663/1664は適用なし39、1665/1668/1670は39適用。後者のmix範囲0.1459375〜0.25、22試合が上限。保存signalと係数から39/39再現。1665は係数列が未保存だが現行0.25と保存signalから一致（当時設定原本の証明とは分ける）。',a)
 add('1668での悪化理由','13試合中9悪化・4改善。水戸－広島の実Aは66.24→58.15%、Log Loss+0.13031。一方、町田－G大阪の実Dは23.44→25.91%、−0.10036。正解結果のraw確率が1/3より高ければmixはその確率を下げ、Log Lossは悪化する。横浜FM－川崎Fの実Hは最高予測記号ではないがraw34.24%なのでmixによりわずかに悪化する。単なる最高予測の的中/外れとmix損益は一致しない。',m[m['round']=='toto1668'][['card','actual','mix','p_actual_before','p_actual_after','delta_log_loss','delta_brier']])
 add('④ mixの校正と移植の検証','全78のLL差+0.005086、Brier差+0.003652。適用3開催はいずれも平均LL悪化。一方上限mix=.25群22試合のLL差は−0.001579で、強いmixほど悪いという結論ではない。予測entropyと実際の誤差は別。係数学習はtrain_uncertainty_calibration.pyがobservation_historyのp_home等を消費し、その作成器04_accumulate_observations.pyはprob_final等を優先。保存configの旧系列検証改善をNext専用の証拠に流用しない。旧規則で今回の予測は作っていない。\n6開催単位の全6^6再標本化でLL差95%範囲は'+str(json.loads((OUT/'audit.json').read_text())['round_bootstrap95_delta_LL'])+'。6群のみ、版差と無適用3開催を含む探索的集計で、将来への確実な効果とは扱わない。維持/廃止を今決めず後続の同一版raw/finalを対評価する。')
 add('⑤⑥ ホーム効果はどこにあるか','基礎statsとLABの4xG率はチーム全会場合計/平均であり、列名_home/_awayは今回の対戦側への結合を意味し、ホーム専用・アウェー専用成績ではない。過去のホーム効果は全会場平均に混ざるが、次戦会場に条件付けた補正とは異なる。確認したNext経路には独立したHFA乗数はない。欠場の攻守差は会場係数ではなく、疲労/移動の情報はmixに入り方向を保持しない。相手守備力は基礎式に入る。したがって未評価の会場効果が残る可能性はあるが、明示項の欠如だけで過小予測原因と断定しない。',h)
 add('ホーム校正の判定','全78でH104.43対122、A103.22対97。1665/1668を中心とした上振れと1670の下振れがある。元のPoisson変動、チーム構成、版差を切り分ける必要がある。goal_uncertainty.csvに得点比率のPoisson仮定区間と残差を保持。一試合の残差を真のλ誤差とみなさず、新HFA係数は導入しない。HDAの期待件数と実件数、固定確率帯校正も別CSVへ保存。')
 add('⑦ LAB・欠場・外れ値の監査','LAB元スナップショットの4指標×78=312値も一致。天候signal39件、外れ値の保存published/ordinary/weighted/used間の算術52件も一致（過去履歴の完全性を保証する検査とは別）。LABは基礎λとsqrt(LAB攻撃xG×相手被xG)のlog空間25%混合で78/78再現。相関したxG情報の二重利用になり得るが同じ補正を二度実行した計算誤りは見つからない。欠場は欠損flagを確認し、攻守影響上限0.5、相手守備欠場−自攻撃欠場に0.25を掛け、λをclip。適用39/39再現。\n外れ値補正は直近5だけでなく、対象試合より前の全履歴の通常/加重平均差を公開率へ加える。各過去試合より前の自他成績から突出度を測る。得点/失点を対称処理し未来退場を推定しない。期間の異なる公開率に全履歴差を加える整合性は今後監査対象。今回は最新結果履歴で再実行せず1670の保存差分のみ利用。疲労/天候がNextのλにも直接入る重複は、この経路では確認していない。',ci.groupby(['round','absence_status']).size().rename('team_inputs').reset_index())
 add('欠場のゼロと不明','適用対象78側入力のうち22側は情報未取得として方向補正0、56側は取得済み。取得済みで人数0の例は今回なし。未取得を欠場者なしと解釈しない。欠場原本の取得timestampは保存予測にないため、原本Aが試合前であることと台帳完全性・復帰の正しさは別。日程/疲労の保存値は利用可能だが、upstream use_generic_fatigue_onlyには欠損を0埋めする経路があり、今後は欠損フラグを保存する。')
 sm=c.groupby(['before','after']).agg(n=('match_no','size'),delta_LL=('delta_LL','mean'),delta_Brier=('delta_Brier','mean'),delta_goal_NLL=('delta_goal_NLL','mean')).reset_index()
 add('同一試合での補正段階差','LAB→最終λの比較は当該保存経路での欠場差（初期39は同一）。外れ値は最終λ保存差分を戻した13試合を比較し、下流LAB/欠場を含む経路の総影響。段階比較の改善と因果的効果を混同しない。基礎λがすでに外れ値調整済みの1670を未調整版と呼ばない。',sm)
 add('⑧ 不整合・疑い・不足を区別する','実績が良くなるから値を修正することはしない。今回判明した値の不一致、保存不足、校正仮説を分ける。',pd.DataFrame(issues,columns=['区分','項目','根拠','提案']))
 add('⑨ 10口への接続','正しいxG収録分母を独立に確定できておらず、入力修正採用0、研究用10口比較も今回は実施しない。分母の条件付き感度試算は修正分布の精度改善を証明しない。mixを事後的に外して券を選び直すこともしない。Sは保存済み探索仮説として維持。')
 add('⑩ 次の実施順と将来評価','1) 指標別の収録試合数・期間・取得時刻と公式消化数を同時保存。2) 元値と照合し分母だけの修正を研究用再計算。3) 同一Next版でraw/final、H/A校正、Sを未使用開催ごとに保存・事後評価。4) 十分な後続期間で補正の変更を判断。\n独立CLI phase39_future_check.pyを用意。captureは13試合すべて試合前であること、出力未存在を確認して保存し、evaluateは試合IDで後日結果と一対一結合する。operator指定asof単独は時刻証明ではないので原本レジストリのtimestamp/hashも必要。本番から自動呼出しせず、予測/購入処理への影響なし。新しい校正モデルの学習なし。')
 # Simple self-contained bar chart of paired effects.
 z=a[a.dimension=='round'];svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 310"><rect width="900" height="310" fill="white"/><text x="25" y="28" font-size="18">mix後−mix前 Log Loss（正は悪化）</text>']
 for j,r in enumerate(z.itertuples()):
  x=70+j*135;v=max(float(r.delta_LL),0);height=v*7000
  svg.append(f'<rect x="{x}" y="{240-height}" width="70" height="{height}" fill="#bb6656"/><text x="{x}" y="{230-height}" font-size="13">{r.delta_LL:.5f}</text><text x="{x}" y="265" font-size="13">{r.group}</text>')
 svg.append('</svg>');(OUT/'mix_rounds.svg').write_text(''.join(svg))
 links=' '.join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(OUT.glob('*.csv')))
 (OUT/'report.md').write_text('# Phase3.9 入力品質・既存補正監査\n\n'+'\n'.join(md))
 (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase3.9 入力品質監査</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.8 system-ui;color:#25364a}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd}th{background:#eaf0f5}section{margin:36px 0}img{max-width:100%}a{margin-right:12px}</style><h1>Phase3.9：入力品質と既存補正の監査</h1><p>試合前原本A・6開催78試合／正式モデル変更なし</p><img src="mix_rounds.svg" alt="開催別mix前後Log Loss差">'+''.join(sections)+'<h2>CSV・仕様・監査</h2>'+links+'<p><a href="report.md">日本語レポート</a><a href="future_validation.json">将来仕様</a><a href="audit.json">監査</a></p></html>')
if __name__=='__main__':build()
