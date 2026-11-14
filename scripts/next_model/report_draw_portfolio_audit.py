"""Render the offline audit without modifying production configuration."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from audit_draw_portfolio import OUT, ROOT, ExactPortfolio
from draw_count_conditioned import generate, draw_count_distribution

NAMES={
 'v24_random':'v2.4（20シード平均）',
 'v23_current_native_counts':'v2.3 現在ソース・元の丸め',
 'v23_arrangement_v24_counts_seed0':'v2.3 配置処理・v2.4枚数固定',
 'v23_initial_diversity_native_counts':'v2.3 初期配置（相違数優先）',
 'quota_prototype':'当初試作：D本数枠＋券ごと選択',
 'direct_fixed_Dquotas_retained':'D本数枠維持＋全体確率評価',
 'direct_fixed_from_quota':'D本数枠解除＋全体確率評価',
 'direct_fixed_from_v24_seed0':'枠なし・v2.4 seed0から全体評価',
 'direct_fixed_K_coverage_guard':'枠なし・D本数別の成績維持（診断実験）',
 'direct_FREE_counts_separate':'別実験：H/D/A枚数も自由',
 'historical_saved_tickets_version_varies':'当時の保存券（版は開催ごとに異なる）'}
COLUMNS={'p10':'10的中以上 %','p11':'11的中以上 %','p12':'12的中以上 %','p13':'13的中 %',
         'expected_max_hits':'最高的中数の期待値','min_distance':'最小相違数','mean_distance':'平均相違数',
         'D_position_distance':'D位置の平均相違数','D_total':'D総枠','seconds':'生成・探索 秒',
         'D_low_0_2_p10':'実D 0〜2：10以上 %','D_mid_3_4_p10':'実D 3〜4：10以上 %',
         'D_high_5plus_p10':'実D 5以上：10以上 %','actual_mean':'実平均的中','actual_max':'実最高的中'}


def table(frame, cols, index=True):
    f=frame[cols].copy()
    for c in cols:
        if c.startswith('p') or c.endswith('_p10'):f[c]*=100
    f=f.rename(columns=COLUMNS,index=NAMES)
    return '<div class="scroll">'+f.to_html(index=index,float_format=lambda x:f'{x:.5f}',border=0)+'</div>'


def main():
    f=pd.read_json(OUT/'metrics.json')
    inputs=json.loads((OUT/'inputs.json').read_text())
    portfolios=json.loads((OUT/'portfolios.json').read_text())
    for item in inputs:
        source=ROOT/f'data/eval/toto_rounds/{item["round"]}/snapshot/next_predictions.csv'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==item['input_sha256']
        m=source.parent/'buyplan_engine_manifest.json'
        item['historical_version']=json.loads(m.read_text()).get('official_version','未記載') if m.exists() else '未記載'
    means=f.groupby('method').mean(numeric_only=True)
    main_methods=['v23_current_native_counts','v23_arrangement_v24_counts_seed0','v24_random','quota_prototype',
                  'direct_fixed_Dquotas_retained','direct_fixed_from_quota','direct_fixed_K_coverage_guard']
    metrics=['p10','p11','p12','p13','expected_max_hits']
    report='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>配券スコアとDの評価：設計検証</title>
<style>body{font:16px/1.85 system-ui,sans-serif;color:#243347;max-width:1450px;margin:36px auto;padding:0 24px}h1,h2,h3{line-height:1.4}h2{margin-top:42px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap;margin:18px 0}th,td{border:1px solid #ccd4de;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#eaf0f7}.note{background:#fff1cc;border-left:5px solid #c98a00;padding:16px}.finding{background:#e9f3f0;padding:18px}code{background:#edf1f5;padding:2px}a{color:#164b91}</style>
<h1>配券スコアでDをどう評価するか：設計評価と比較検証</h1>
<p>対象：1660・1663・1664・1665・1668回の保存入力。正式版v2.4、予測モデル、購入券は変更していない。</p>
<div class="finding"><b>結論：</b>D本数別の枠を確率に比例させることは、説明しやすい配置規則だが、高的中確率を最大化する原理ではない。
一方、全体の10的中確率だけを最大化すると、Dを一部の弱い券へ集める解が現れた。
今回の目的は「Dを何本入れる義務」ではなく「Dの価値を配券スコアへ反映すること」。本数枠・成績維持条件は比較実験であり、そのまま採用する提案ではない。</div>
<h2>1. 検証条件と限界</h2>
<ul><li>各試合のH/D/A確率は固定し、試合間独立を仮定した。全3¹³＝1,594,323通りを列挙し、少なくとも1券が10・11・12・13的中する確率と最高的中数の期待値を厳密集計した。これは入力モデル内での計算値で、実戦的中率ではない。</li>
<li>v2.4は事前に決めたseed 0〜19の20通り。100個の別開催としては扱わない。各開催のシード平均を取ってから5開催を等重みで集計した。</li>
<li>現在のv2.3は決定的実装で、当時乱数を使っていたv2.3とは違う。現存するgit履歴の該当実装も決定的方式だったため、古い乱数版の完全な再現・複数シード検証は今回できていない。当時の保存券は別表で評価する。</li>
<li>v2.3の元の丸めは総H/D/A枚数も制御するが、v2.4は試合単位で丸める。純粋な配置比較用に、v2.4 seed0と同じ枚数・初期券へv2.3の配置処理を適用した対照も作った。</li>
<li>直接探索は10的中以上を目的とし、各候補交換の確率差を厳密計算。最大20交換の局所探索で、全配置の大域最適性は保証しない。11〜13的中や最高的中数の期待値を同時に最大化するものではない。</li></ul>'''
    provenance=pd.DataFrame(inputs)[['round','created_at','saved_before_kickoff','actual_D','historical_version','interpretation']]
    report+='<div class="scroll">'+provenance.to_html(index=False,border=0)+'</div>'
    report+='''<p class="note">13試合すべてが保存時点で試合開始前だったのは1663回のみ。1668回は開発データ、他4回も既存の過去入力を用いた感度検証であり、新たな独立ホールドアウトではない。実Dが5以上の保存開催はなく、多D回の実戦的な優位性は未検証。</p>
<h2>2. 比較結果：5開催のモデル上の平均</h2><p>各確率は「10口の少なくとも1口」。v2.3元の丸め以外の下表はv2.4と各試合の採用枚数が一致する。</p>'''
    report+=table(means.loc[main_methods],metrics)
    report+='''<p>当初試作は10的中以上を増やしても、最高的中数の期待値はv2.4より下がった。何を改善対象にするかで評価が変わる。
D本数枠を維持したまま全体確率を評価すると当初試作から改善したため、当初の「1券ずつ強いものを取る」選択にも改善余地があった。</p>
<h2>3. Dが少ない／通常／多い場合を分けると見える問題</h2>
<p>以下は実測の開催分類ではなく、各入力モデルの結果空間を実D数で条件付けた確率。0〜2、3〜4、5以上は読みやすい表示区分で、次の成績維持実験はK=0〜13を個別に判定している。</p>'''
    report+=table(means.loc[main_methods],['D_low_0_2_p10','D_mid_3_4_p10','D_high_5plus_p10'])
    report+='''<p class="note">1668回の枠なし直接探索では、D総枠37を維持しながら、候補のD本数が <b>0,0,0,0,0,2,11,2,11,11</b> になった。
Dを11個含む3券へ押し込み、低Dの場合の成功確率を優先している。枚数固定だけでは、実質的にDを重視しない配置を防げない。</p>
<p>「D本数別の成績維持」は、当初試作に対し、すべてのKで P(10以上的中かつ実D数=K) を下げない交換だけを許した診断実験。
D本数枠自体は固定せず、5開催すべてで当初試作の10的中以上を改善できた。ただし参照先が当初試作であること、全Kでの不悪化という強い制約を置くことには設計上の選好がある。係数による評価という依頼意図に対し、これをそのまま正式採用しない。</p>
<h2>4. 7つの懸念への回答</h2>
<ol><li><b>D総数を主軸にすると試合事情を失うか：</b>試作は各試合のH/D/A枚数を保持し、各券の元の同時確率で選択しているため、個々の試合確率を無視してはいない。しかし、先に決めた本数枠が、確率の高い組み合わせを選べなくする場合はある。D数分布だけでは、どの試合にDを置くべきかは決まらない。</li>
<li><b>発生比率を候補数に変換する根拠：</b>分布を代表させる工夫であり、当選確率最大化の根拠ではない。10的中なら3試合まで外せるので、券のD本数と実D数が違っていても成功する。枠固定は探索を狭める。今回は解除で総確率は上昇したが、多D時の性能が低下した。</li>
<li><b>Dを置く位置と10券の関係：</b>試作の各券選択は元の確率と残り枚数を評価するが、他9券との高的中領域の重なりを目的関数にしていなかった。新しい比較実験では10券の成功領域の和集合を直接評価し、同じ結果をカバーしても重複加点しない。</li>
<li><b>二重評価：</b>D数分布は元のD確率から導出される。同じ情報を配置制約に用いることは新しい予測情報ではなく、配置方針の選択である。元の同時確率へP(K)をさらに掛けてはいない。低得点や拮抗が既にD確率に反映されている場合、理由なく再加点してはならない。</li>
<li><b>類似度：</b>似ているだけで罰する必要はない。成功領域の重なりと、その領域の確率質量を見る。今回、最小相違数を大きくする初期配置は高的中確率で劣り、似た券を含む方式が上回った。これは「似た券が常に良い」ことも意味しない。</li>
<li><b>改善効果：</b>5開催・20シード・全結果空間で比較したが、モデル上の配置改善と実戦改善は別。保存時点の問題と開催数の少なさから、実戦優位はまだ証明できない。</li>
<li><b>本数枠は本質か：</b>今回の証拠では本質的な最適化原理ではない。一方、単に枠を外して総成功確率だけを評価することも意図を満たさない。Dの価値をどうスコアへ反映するか、目的関数そのものを設計すべきである。</li></ol>
<h2>5. 「係数でDをどう評価するか」へ戻す改善案</h2>
<p><b>一律D加点の限界：</b>各Dに同じ係数βを掛けるだけなら、券の対数スコアは元の対数確率＋D本数×logβとなる。同じD本数の券の順位は変わらず、Dを置く試合の性質を評価し直すことにはならない。
さらに、各試合のH/D/A枚数を固定すると、10券全体の単純な対数スコア合計はどの配置でも同じ。係数だけを加えて合計を最大化しても配置は決まらない。</p>
<p><b>位置ごとのD価値：</b>あるH/AをDへ置き換えたとき、他9券で届かない結果をどれだけ新たにカバーするか、逆に失うかを評価する。元のpDが係数として入り、他の試合と他の券の構成によって貢献度が変わる。
0–0、低得点、拮抗、僅差の性質はまず予測モデルの得点分布・H/D/A確率の整合性で確認する。配券で同じ情報を重複加点しない。</p>
<p><b>比較対象にするスコア案（係数は未決定・未採用）：</b><br>
<code>Sβ(T) = P(10口の最高的中≥10) − β × ΣK max(0, A_K(参照配置) − A_K(T))</code><br>
ここで <code>A_K(T)=P(最高的中≥10 かつ 実D数=K)</code>。
D本数を義務づける代わりに、Dが出る場合の性能を失う配置に評価上の不利益を与える。β=0は今回の無制約直接評価に相当する。
βや参照配置の決め方は「正しいD確率」ではなく配券の選好を表す設計であり、今回の5回に合わせて正式値を選ばない。参照を当初試作にする必然性もなく、v2.4複数シード等と比較すべき。</p>
<p>候補番号の前半にもDを含めたい、という要求は、券の評価・表示順の設計とも区別する。番号を付け替えて性能が改善したように見せず、各候補の確率とD構成を併記する。</p>
<h2>6. H/D/A枚数自体を自由にした別実験</h2>'''
    report+=table(means.loc[['quota_prototype','direct_fixed_from_quota','direct_FREE_counts_separate']],metrics+['D_total'])
    report+='''<p>この実験は各試合の枚数を変更するため、配置だけの改善とは比較を分ける。単一セル変更を最大20回行った。計算上の総成功確率は上がるが、低D時へ偏る性質は残る。今回の正式採用候補とはしない。</p>
<h2>7. 類似度・D位置・実行コスト</h2>'''
    report+=table(means.loc[main_methods+['v23_initial_diversity_native_counts']],['min_distance','mean_distance','D_position_distance','seconds','p10'])
    source=ROOT/'data/eval/toto_rounds/toto1668/snapshot/next_predictions.csv'
    frame=pd.read_csv(source).sort_values('match_no');p=frame[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy()
    start=time.perf_counter();e=ExactPortfolio(p);build=time.perf_counter()-start
    start=time.perf_counter();e.metrics(np.array(portfolios['toto1668']['quota_prototype']['tickets']));evaluation=time.perf_counter()-start
    report+=f'<p>この環境で、結果空間と確率の構築は約{build:.3f}秒、1組10券の全指標集計は約{evaluation:.3f}秒。上表は生成・探索時間で集計時間を含まない。直接探索はおおむね数秒／開催。13試合・10券での比較は実用的な時間で可能だが、大域最適解を探索したわけではない。</p>'
    searches=[]
    for r,methods in portfolios.items():
        for name,d in methods.items():
            if d['search']:
                a=d['search'];searches.append({'開催':r,'方式':NAMES[name],'交換数':a['steps'],'上限':a['cap'],'局所停止を確認':a['stopped_at_local_optimum']})
    report+='<details><summary>探索の終了理由</summary><div class="scroll">'+pd.DataFrame(searches).to_html(index=False,border=0)+'</div></details>'
    report+='<h2>8. 開催別・シード別の詳細</h2>'
    for r in sorted(f['round'].unique()):
        rf=f[f['round']==r];agg=rf.groupby('method').mean(numeric_only=True)
        report+=f'<h3>{r}'+('（開発データ）' if r=='toto1668' else '（過去入力の感度検証）')+'</h3>'
        report+=table(agg.loc[main_methods],metrics)
        report+='<details><summary>実結果での参考集計（試合前独立検証ではない）</summary>'+table(agg,['actual_mean','actual_max','D_total'])+'</details>'
    stats=[]
    for r,g in f[f.method=='v24_random'].groupby('round'):
        for c in metrics:
            factor=100 if c.startswith('p') else 1
            vals=g[c]*factor
            stats.append({'開催':r,'指標':COLUMNS[c],'平均':vals.mean(),'標準偏差':vals.std(),'最小':vals.min(),'5%点':vals.quantile(.05),'中央値':vals.median(),'95%点':vals.quantile(.95),'最大':vals.max()})
    report+='<h3>v2.4：20シードの分布</h3><div class="scroll">'+pd.DataFrame(stats).to_html(index=False,float_format=lambda x:f'{x:.6f}',border=0)+'</div>'
    pd.DataFrame(stats).to_csv(OUT/'seed_distributions.csv',index=False)
    # User's heterogeneous-D example: no additional optimization or coefficient fit.
    examples=[]
    for label,p0 in [('13試合ともD25%',np.tile([.375,.25,.375],(13,1))),('5試合D40%＋8試合D15%',np.array([[.3,.4,.3]]*5+[[.425,.15,.425]]*8))]:
        t,a=generate(p0)
        examples.append({'例':label,'D期待数':float(p0[:,1].sum()),'D数の分散':float((p0[:,1]*(1-p0[:,1])).sum()),'試合別D採用枚数':str(np.array(a['allocation_HDA'])[:,1].tolist()),'券別D本数':str(a['draw_count_order'])})
    report+='<h3>同程度のD期待数でも試合別の配置は異なる</h3><div class="scroll">'+pd.DataFrame(examples).to_html(index=False,border=0)+'</div>'
    report+='''<p>当初試作でも高D確率の試合へ多くのDを割り当てており、総数だけでは決めていない。ただし10券への試合単位の丸めにも偏りが出る。この表の差には丸めの影響も含まれ、D本数枠の効果と混同しない。</p>
<h2>9. 検証の記録</h2>
<p>元試作5テストと追加5テストが通過。全列挙と局所的な成功領域計算の一致、重複券の二重計上防止、枚数保持、D本数保持、D本数別成績維持、確率増分の一致を確認した。コードも結果も独立した研究用ファイルで、正式ルートへの接続はない。</p>
<p><a href="metrics.csv">全指標CSV</a> ／ <a href="metrics.json">全指標JSON</a> ／ <a href="portfolios.json">各券と探索履歴</a> ／ <a href="seed_distributions.csv">シード分布</a> ／ <a href="inputs.json">入力の保存時点・ハッシュ</a></p>
</html>'''
    (OUT/'index.html').write_text(report)
    (OUT/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
    files=['audit_draw_portfolio.py','report_draw_portfolio_audit.py','test_audit_draw_portfolio.py','draw_count_conditioned.py','placement_score.py','deterministic_v23.py','random_remaining_v24.py','proportional_buyplan.py']
    manifest={n:hashlib.sha256((ROOT/'scripts/next_model'/n).read_bytes()).hexdigest() for n in files}
    (OUT/'code_hashes.json').write_text(json.dumps(manifest,indent=2))
    print(OUT/'index.html')


if __name__=='__main__': main()
