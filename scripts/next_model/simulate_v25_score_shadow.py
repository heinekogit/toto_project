"""Offline multi-round v2.5 simulation. Default report only, no official writes."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import html
import json
import time
import numpy as np
import pandas as pd
from score_placement_v25_shadow import Config, generate, diagnostics
from random_remaining_v24 import allocate_per_match, generate as v24_generate, prediction_fingerprint, seed_from_fingerprint
from audit_draw_portfolio import ExactPortfolio

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/reports/v25_score_shadow'
COEFFICIENTS=(1.0,1.1,1.2,1.4)
CSS='''body{font:16px/1.8 system-ui,sans-serif;color:#203147;max-width:1550px;margin:32px auto;padding:0 24px}h1,h2{line-height:1.4}h2{margin-top:36px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px;margin:16px 0}td,th{padding:7px;border:1px solid #cad3de;text-align:right}td:first-child,th:first-child{text-align:left}thead{background:#e9eff7}.note{padding:16px;background:#fff0ce}.good{padding:16px;background:#e8f2f0}code{background:#eef2f6}a{color:#165296}'''


def page(title,body):return f'<!doctype html><html lang="ja"><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style><h1>{html.escape(title)}</h1>'+body+'</html>'

def table(f):return '<div class="scroll">'+f.to_html(index=False,border=0,float_format=lambda x:f'{x:.7g}')+'</div>'

def metric_table(f):
    fields=['method','actual_max','p10','p11','p12','p13','expected_max_hits','D_centroid','duplicate_pairs','stage_error','seconds']
    d=f[fields].copy()
    for c in ['p10','p11','p12','p13']:d[c]*=100
    return table(d.rename(columns={'method':'方式','actual_max':'実結果の最高的中','p10':'10以上 %','p11':'11以上 %','p12':'12以上 %','p13':'13的中 %','expected_max_hits':'最高的中期待値','D_centroid':'Dの配置重心 0=先頭/1=末尾','duplicate_pairs':'完全重複ペア','stage_error':'交換数目標との絶対差合計','seconds':'生成秒'}))


def save_configuration(path,frame,p,tickets,audit,actual,metrics):
    path.mkdir(parents=True,exist_ok=True)
    rows=diagnostics(p,tickets,audit)
    for row,t in zip(rows,np.vstack([audit['anchor'],tickets])):
        row['actual_hits']=int((t==actual).sum())
        for i,s in enumerate(t):row[f'match{i+1:02d}']='HDA'[s]
    pd.DataFrame(rows).to_csv(path/'candidates00_10.csv',index=False)
    match=frame[['match_no','home_team','away_team']].copy()
    for s,label in enumerate('HDA'):
        match[f'P_{label}']=p[:,s];match[f'count_{label}']=np.array(audit['counts'])[:,s]
        match[f'score_{label}']=np.array(audit['scores_HDA'])[:,s];match[f'loss_{label}']=np.array(audit['losses_HDA'])[:,s]
    match['K_D']=audit['D_coefficients'];match['candidate00']=['HDA'[s] for s in audit['anchor']]
    for j,t in enumerate(tickets):match[f'candidate{j+1:02d}']=['HDA'[s] for s in t]
    match['actual']=['HDA'[s] for s in actual]
    match.to_csv(path/'match_diagnostics.csv',index=False)
    buyplan=frame[['match_no','home_team','away_team','next_prob_home','next_prob_draw','next_prob_away']].copy()
    for j,t in enumerate(tickets):buyplan[f'ticket{j+1:02d}']=np.where(t==0,1,np.where(t==1,0,2))
    buyplan.to_csv(path/'buyplan.csv',index=False)
    (path/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    placement=match[['match_no','home_team','away_team','candidate00']+[f'candidate{j:02d}' for j in range(1,11)]+['actual']]
    summary=pd.DataFrame(rows).drop(columns=[f'match{i:02d}' for i in range(1,14)]+['symbols'])
    summary['joint_probability']*=100
    summary=summary.rename(columns={'candidate':'候補','purchase':'購入対象','exchanges':'交換数','target':'目標','D_count':'D本数','rank2':'2番手','rank3':'3番手','joint_probability':'13的中予測確率 %','placement_value':'配券評価値合計','exchange_loss':'交換損失合計','duplicate_purchase_candidates':'同一購入候補','actual_hits':'実的中'})
    body='<p><a href="../index.html">開催比較へ</a> ／ <a href="../../index.html">全体報告へ</a></p><p>H=ホーム勝ち（1）、D=引き分け（0）、A=アウェー勝ち（2）。候補00は購入対象外。配券評価値は予測確率ではない。</p>'
    body+='<h2>候補00〜10の配置</h2>'+table(placement)+'<h2>候補別診断</h2>'+table(summary)
    body+='<h2>試合別の確率・枚数・係数・交換損失</h2>'+table(match.drop(columns=[f'candidate{j:02d}' for j in range(1,11)]))
    body+='<p>固定枚数保存・候補00の不消費・予測確率不変を検査済み。完全重複が残った場合、全試合が固定記号となる場合以外は「不可避」と断定せず、探索で解消できなかったものとして監査記録へ残す。</p>'
    body+='<p><a href="candidates00_10.csv">候補診断CSV</a> ／ <a href="match_diagnostics.csv">試合診断CSV</a> ／ <a href="buyplan.csv">購入候補01〜10のみの互換CSV</a> ／ <a href="audit.json">設定と探索履歴</a></p>'
    (path/'index.html').write_text(page(f'{path.parent.name}／{path.name} 配券スコア型シャドー',body))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    production_files=[ROOT/'scripts/next_model/random_remaining_v24.py',ROOT/'scripts/next_model/run_next_buyplan.py',ROOT/'scripts/next_model/build_next_buyplan.py']
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in production_files}
    all_metrics=[];input_records=[];effects=[]
    old_audit=ROOT/'data/reports/draw_portfolio_design_audit'
    old_meta={x['round']:x for x in json.loads((old_audit/'inputs.json').read_text())} if (old_audit/'inputs.json').exists() else {}
    old_metrics=pd.read_json(old_audit/'metrics.json') if (old_audit/'metrics.json').exists() else pd.DataFrame()
    for source in sorted((ROOT/'data/eval/toto_rounds').glob('*/snapshot/next_predictions.csv')):
        name=source.parent.parent.name;dest=OUT/name;dest.mkdir(exist_ok=True)
        frame=pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
        p=frame[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy();counts=allocate_per_match(p)
        actual_frame=pd.read_csv(source.parent.parent/'actual_results.csv').sort_values('match_no')
        for col in ('match_no','home_team','away_team'):assert frame[col].tolist()==actual_frame[col].tolist()
        r=actual_frame.result.to_numpy().astype(int);actual=np.where(r==1,0,np.where(r==0,1,2))
        evaluator=ExactPortfolio(p);manifest=json.loads((source.parent/'manifest.json').read_text())
        dt=pd.to_datetime(frame.datetime)
        if dt.dt.tz is None:dt=dt.dt.tz_localize('Asia/Tokyo')
        info={'round':name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'snapshot_created_at':manifest['created_at'],'saved_before_kickoff':int((dt>pd.Timestamp(manifest['created_at'])).sum()),'actual_D':int((actual==1).sum())}
        input_records.append(info)
        seed=seed_from_fingerprint(prediction_fingerprint(frame,name))
        base,ba=v24_generate(p,seed);baseline_label='v2.4 同入力再生成'
        engine_path=source.parent/'buyplan_engine_manifest.json'
        if engine_path.exists() and json.loads(engine_path.read_text()).get('official_version')=='v2.4':
            saved=pd.read_csv(source.parent/'buyplan.csv').sort_values('match_no')
            raw=saved[[f'ticket{j:02d}' for j in range(1,11)]].to_numpy().T.astype(int)
            base=np.where(raw==1,0,np.where(raw==0,1,2));baseline_label='v2.4 当時の保存券'
        for s in range(3):np.testing.assert_array_equal((base==s).sum(axis=0),counts[:,s])
        generated={};round_metrics=[]
        def evaluate(label,t,a,seconds):
            m=evaluator.metrics(t);d=(t==1).sum(axis=1)
            m.update(round=name,method=label,actual_max=int((t==actual).sum(axis=1).max()),actual_mean=float((t==actual).sum(axis=1).mean()),actual_D=info['actual_D'],D_centroid=float(d@np.linspace(0,1,10)/max(d.sum(),1)),duplicate_pairs=sum(np.array_equal(t[i],t[j]) for i in range(10) for j in range(i)),stage_error=0 if a is None else int(np.abs((t!=a['anchor']).sum(axis=1)-a['target_exchanges']).sum()),seconds=seconds)
            round_metrics.append(m);all_metrics.append(m);return m
        baseline=evaluate(baseline_label,base,None,0)
        # Baseline target error is undefined: it does not optimize exchange tiers.
        baseline['stage_error']=None
        for k in COEFFICIENTS:
            cfg=Config(draw_coefficient=k);start=time.perf_counter();t,a=generate(p,counts,cfg);seconds=time.perf_counter()-start
            label=f'v2.5 K_D={k:.1f}';m=evaluate(label,t,a,seconds)
            generated[k]=(t,a,m)
            save_configuration(dest/f'kd_{k:.1f}',frame,p,t,a,actual,m)
            if k!=1:
                reference=generated[1.0][0]
                effects.append({'round':name,'K_D':k,'changed_cells_vs_K1':int((t!=reference).sum()),'changed_D_positions_vs_K1':int(((t==1)!=(reference==1)).sum()),'initial_placement_equal':a['initial_tickets']==generated[1.0][1]['initial_tickets']})
            assert np.isclose(m['actual_mean'],baseline['actual_mean'])
        # Reuse prior exact metrics only after confirming the immutable input hash.
        if name in old_meta and old_meta[name]['input_sha256']==info['source_sha256']:
            seeds=old_metrics[(old_metrics['round']==name)&(old_metrics.method=='v24_random')].copy()
            assert sorted(seeds.seed.astype(int))==list(range(20))
        else:
            seed_rows=[]
            for sn in range(20):
                t,_=v24_generate(p,sn);seed_rows.append(dict(seed=sn,**evaluator.metrics(t)))
            seeds=pd.DataFrame(seed_rows)
        seeds.to_csv(dest/'v24_seed_metrics.csv',index=False)
        ctable=[]
        for label,t in [(baseline_label,base)]+[(f'K_D={k:.1f}',generated[k][0]) for k in COEFFICIENTS]:
            anchor=p.argmax(axis=1)
            ctable.append({'方式':label,'候補01〜10の交換数':str((t!=anchor).sum(axis=1).tolist()),'候補01〜10のD本数':str((t==1).sum(axis=1).tolist())})
        body='<p><a href="../index.html">全体報告へ</a></p>'
        body+=f'<p>実D数：{info["actual_D"]}。保存時点で試合開始前だった試合数：{info["saved_before_kickoff"]}/13。参照する実結果は候補生成に使用していない。</p>'
        body+=metric_table(pd.DataFrame(round_metrics))+table(pd.DataFrame(ctable))
        body+='<p>'+ ' ／ '.join(f'<a href="kd_{k:.1f}/index.html">K_D={k:.1f} 配置表・診断</a>' for k in COEFFICIENTS)+'</p>'
        base_table=frame[['match_no','home_team','away_team']].copy();base_table['candidate00']=['HDA'[s] for s in p.argmax(axis=1)]
        for j,t in enumerate(base):base_table[f'candidate{j+1:02d}']=['HDA'[s] for s in t]
        body+='<h2>v2.4の比較対象配置</h2>'+table(base_table)
        body+='<h2>v2.4・20シードのモデル上の分布</h2>'
        seed_summary=seeds[['p10','p11','p12','p13','expected_max_hits']].agg(['mean','std','min','median','max']).reset_index()
        for c in ['p10','p11','p12','p13']:seed_summary[c]*=100
        body+=table(seed_summary)+'<p>p10〜p13は%表示。シード0〜19。以前の厳密集計は入力ハッシュ一致時に再利用した。</p>'
        (dest/'index.html').write_text(page(name+' v2.5シャドー比較',body))
        np.savez(dest/'input_and_baseline.npz',probabilities=p,counts=counts,baseline=base,actual=actual)
        print(name,[(m['method'],round(m['p10']*100,3),m['actual_max']) for m in round_metrics],flush=True)
    metrics=pd.DataFrame(all_metrics);metrics.to_csv(OUT/'metrics.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(all_metrics,ensure_ascii=False,indent=2))
    (OUT/'inputs.json').write_text(json.dumps(input_records,ensure_ascii=False,indent=2));pd.DataFrame(effects).to_csv(OUT/'coefficient_effects.csv',index=False)
    (OUT/'provisional_config.json').write_text(json.dumps({'default':asdict(Config()),'tested_D_coefficients':COEFFICIENTS,'formal_adoption':False},ensure_ascii=False,indent=2))
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in protected.items())
    (OUT/'production_code_hashes.json').write_text(json.dumps(protected,indent=2))
    render_summary(metrics,input_records,effects)


def render_summary(metrics,inputs,effects):
    body='''<p class="good">独立したシャドー実装。候補00は参照専用、購入対象は01〜10。元の予測確率と各試合の固定H/D/A枚数を保存する。D係数は配券用の評価係数で、予測確率を変更しない。</p>
<p>初期設定は暫定。今回の結果だけで係数や重みを正式採用しない。得点構造から係数を作る機能は未実装で、一律係数の効果を確認する段階。</p>
<h2>実装案と採用した暫定仕様</h2>
<ul><li><b>交換数目標：</b>総交換数Sを計算し、平均S/10の周囲に広がり4の等間隔目標を置く。0〜13へ収め、合計を合わせる。固定枚数から実現できる単調非減少の整数列を、二乗誤差最小の動的計画法で求める。特定の列を埋め込んでいない。</li>
<li><b>初期配置：</b>各試合の非基準記号枚数と候補の交換数を満たす二部グラフの構成法で変更位置を決め、非基準記号を元の確率で決定的に埋める。全D係数で同じ初期配置を用いる。</li>
<li><b>配置スコア：</b>HとAは元の確率、Dは元の確率×K_D。候補00は元の確率の最大記号のまま。交換損失は「基準記号の配券評価値−採用記号の配券評価値」で、負の値も許す。</li>
<li><b>探索：</b>全ての単一試合内交換と、同じ候補ペアについて交換数を相殺する2試合分の交換を調べ、目的関数が最も改善する操作を採用。2試合の操作も固定枚数を保つ同一試合内交換の組み合わせ。最大200回、改善がなくなれば終了。乱数なし。同点は固定走査順。</li>
<li><b>候補順：</b>最終的な予測確率順への並べ替えはしない。00を含めずに10口を評価する。</li></ul>
<h2>目的関数と変更可能な重み</h2>
<p><code>J = 100 E + 1 L + 0.03 C + 0.2 U + 0.01 R</code> を小さくする。全重みは設定値。優先順位を有限の重みで表現した試作であり、最適化条件を厳密な辞書式順序にはしていない。</p>
<ul><li><b>E：</b>交換数目標との平均二乗差÷169 ＋ 隣接候補での交換数逆転量合計÷117。初期配置は目標に一致する。設定変更時は目標から外れることも許し、診断へ記録する。</li>
<li><b>L：</b>候補01から10へ重みを1から0へ線形に下げた、候補別交換損失の加重平均÷13。固定枚数で不変になる単純合計を使わない。</li>
<li><b>C：</b>D配置位置の重心（01=0、10=1）が0.65を超えた分の二乗。後半への極端な偏りだけへの補助項であり、D本数を各候補へ割り当てる制約はない。</li>
<li><b>U：</b>完全重複ペア数÷45。完全重複は避ける方向に評価するが、重み付き探索なので解消を保証しない。残存は明示する。不可避と証明できるのは全試合の採用記号が一種類しかない場合などで、単なる探索失敗と区別する。</li>
<li><b>R：</b>同一試合で同じ記号が3候補続く窓の数÷104。連続2候補まではペナルティなし。許容連続長は設定可能。</li></ul>
<p>デフォルトK_D=1.2。比較は1.0、1.1、1.2、1.4。これらを学習済み・最適値とは扱わない。13試合ごとの係数配列を渡せる入口も用意したが、その係数を得点特徴から作るロジックは次段階。</p>
<h2>開催ごとの比較・候補00〜10</h2>'''
    body+='<ul>'+''.join(f'<li><a href="{x["round"]}/index.html">{x["round"]}：4設定の配置・診断とv2.4比較</a>（実D {x["actual_D"]}試合、保存時に開始前 {x["saved_before_kickoff"]}/13）</li>' for x in inputs)+'</ul>'
    body+='<h2>1668回の比較</h2>'+metric_table(metrics[metrics['round']=='toto1668'])
    seed_frames=[pd.read_csv(OUT/x['round']/'v24_seed_metrics.csv').assign(round=x['round']) for x in inputs]
    seed_means=pd.concat(seed_frames).groupby('round')[['p10','p11','p12','p13','expected_max_hits','D_low_0_2_p10','D_mid_3_4_p10','D_high_5plus_p10']].mean().mean().to_frame().T
    for col in ['p10','p11','p12','p13','D_low_0_2_p10','D_mid_3_4_p10','D_high_5plus_p10']:seed_means[col]*=100
    seed_means.insert(0,'method','v2.4：20シード×5開催平均（確率は%）')
    means=metrics.copy();means['method']=means.method.where(means.method.str.startswith('v2.5'),'v2.4 比較対象')
    aggregate=means.groupby('method',sort=False).mean(numeric_only=True).reset_index()
    body+='<h2>5開催の平均（モデル上の感度検証）</h2>'+metric_table(aggregate)
    body+='<p>上表のv2.4は保存版がv2.4ならその券、それ以外は同入力から現行の指紋・シード生成で再計算した券。各開催ページに20シードの分布も併記。v2.4の交換数目標差は対象外。</p>'
    body+='<h3>v2.4のシード平均も併記</h3>'+table(seed_means)+'<p>上の特定券の平均と、この20シード平均は比較対象が異なる。以前の報告の3.15%はこちらの20シード平均に対応する。購入券をこの平均から選んだわけではない。</p>'
    body+='<p><b>今回の読み取り：</b>係数1.0ではDが後方へ偏り、係数を上げると前方へ移る。一方、10的中以上の総確率や実最高的中が単調に改善するわけではない。係数1.2は配置ロジック確認用の暫定値であり、最適値の選定結果ではない。</p>'
    conditional=aggregate[['method','D_low_0_2_p10','D_mid_3_4_p10','D_high_5plus_p10']].copy()
    for c in conditional.columns[1:]:conditional[c]*=100
    body+='<h2>実D数を条件としたモデル内の10的中以上確率（%）</h2>'+table(conditional)
    body+='<p>0〜2、3〜4、5以上は結果空間を区分したモデル内評価。実開催の的中率ではない。今回の実開催のD数は1〜4で、実Dが5以上の開催での実証はできていない。</p>'
    actual=metrics.pivot(index='round',columns='method',values='actual_max').reset_index()
    body+='<h2>実結果に対する最高的中数（参考値）</h2>'+table(actual)
    body+='<p class="note">1668回は開発中に参照済み。保存時点で全13試合が開始前なのは1663回だけで、他開催の実結果比較は厳密な試合前検証ではない。各試合独立の全3¹³通りを使う推定確率と、将来の実際の改善を区別する。平均的中数は固定枚数が同じなので全方式一致し、主要な改善指標にはしていない。</p>'
    body+='<h2>D係数が配置に反映されたか</h2>'+table(pd.DataFrame(effects))
    body+='<p>changed_cellsはK_D=1.0から変わったセル数、changed_D_positionsはD採用位置が変わったセル数。同じ初期配置・同じ補助重みで比較している。係数を変えても配置が変わらない入力はあり得るが、今回の開催での反映有無はこの表で確認できる。係数が大きいほど全ての指標が単調改善する保証はない。</p>'
    body+='<h2>検証・再現方法</h2><p>6テスト通過：枚数保存、候補00不消費、再現性、係数効果、予測確率不変、負の交換損失、交換数目標の実現可能性、入力異常、重複不可避ケースを確認。正式版3ファイルのハッシュも実行前後で一致。</p><p><code>./scripts/.venv/bin/python scripts/next_model/simulate_v25_score_shadow.py</code></p>'
    body+='<p><a href="provisional_config.json">暫定設定</a> ／ <a href="metrics.csv">全比較指標</a> ／ <a href="coefficient_effects.csv">係数効果</a> ／ <a href="inputs.json">入力情報</a></p>'
    (OUT/'index.html').write_text(page('BuyPlan v2.5候補：配券スコア型シャドー検証',body))


if __name__=='__main__':main()
