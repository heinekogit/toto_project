"""Compare frozen score-structure coefficients without touching official outputs."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from match_draw_coefficient_shadow import extract,fit_reference,coefficients
from score_placement_v25_shadow import generate,Config
from simulate_v25_score_shadow import ROOT,save_configuration,page,table,metric_table
from audit_draw_portfolio import ExactPortfolio

OUT=ROOT/'data/reports/v25_match_draw_shadow'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=sorted((ROOT/'data/eval/toto_rounds').glob('*/snapshot/next_predictions.csv'))
    frames={p.parent.parent.name:pd.read_csv(p).sort_values('match_no').reset_index(drop=True) for p in paths}
    fs={name:extract(f) for name,f in frames.items()}
    reference=fit_reference(pd.concat(list(fs.values()),ignore_index=True))
    reference['inputs']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'normalization_reference.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2))
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/next_model'/n for n in ['random_remaining_v24.py','score_placement_v25_shadow.py','run_next_buyplan.py','build_next_buyplan.py']]}
    allrows=[];allfeatures=[];differences=[];provenance=[]
    for name,frame in frames.items():
        dest=OUT/name;dest.mkdir(exist_ok=True)
        source=ROOT/f'data/eval/toto_rounds/{name}/snapshot'
        original=ROOT/f'data/reports/v25_score_shadow/{name}'
        data=np.load(original/'input_and_baseline.npz');p=data['probabilities'];counts=data['counts'];baseline=data['baseline'];actual=data['actual']
        np.testing.assert_array_equal(p,frame[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy())
        evaluation=ExactPortfolio(p);feature=coefficients(fs[name],reference)
        feature.insert(0,'match_no',frame.match_no);feature.insert(1,'home_team',frame.home_team);feature.insert(2,'away_team',frame.away_team)
        feature.insert(0,'round',name);allfeatures.append(feature)
        feature.to_csv(dest/'match_D_evaluation.csv',index=False)
        variants={};round_rows=[]
        def score(label,t,a,elapsed):
            m=evaluation.metrics(t);d=(t==1).sum(axis=1)
            m.update(round=name,method=label,actual_max=int((t==actual).sum(axis=1).max()),actual_mean=float((t==actual).sum(axis=1).mean()),D_centroid=float(d@np.linspace(0,1,10)/max(d.sum(),1)),duplicate_pairs=sum(np.array_equal(t[i],t[j]) for i in range(10) for j in range(i)),stage_error=None if a is None else int(np.abs((t!=a['anchor']).sum(axis=1)-a['target_exchanges']).sum()),seconds=elapsed)
            allrows.append(m);round_rows.append(m);return m
        engine=json.loads((source/'buyplan_engine_manifest.json').read_text()) if (source/'buyplan_engine_manifest.json').exists() else {}
        version=engine.get('official_version','unknown')
        # Only label actual v2.4 tickets as v2.4 saved tickets.
        score('v2.4保存券' if version=='v2.4' else 'v2.4同入力再生成',baseline,None,0)
        saved=pd.read_csv(source/'buyplan.csv').sort_values('match_no')
        raw=saved[[f'ticket{j:02d}' for j in range(1,11)]].to_numpy().T.astype(int)
        saved_t=np.where(raw==1,0,np.where(raw==0,1,2))
        if version!='v2.4':score(f'当時の保存券・{version}（別配分の場合あり）',saved_t,None,0)
        for mode,k in [('uniform_1.0',1.),('uniform_1.2',1.2),('match_structure',None),('uniform_matched_mean',float(feature.K_D.mean()))]:
            start=time.perf_counter();t,a=generate(p,counts,config=Config(draw_coefficient=1. if k is None else k),draw_coefficients=feature.K_D.to_numpy() if k is None else None)
            elapsed=time.perf_counter()-start;m=score(mode,t,a,elapsed)
            variants[mode]=t;save_configuration(dest/mode,frame,p,t,a,actual,m)
            assert np.isclose(m['actual_mean'],float((baseline==actual).sum(axis=1).mean()))
        for base_mode in ['uniform_1.0','uniform_1.2','uniform_matched_mean']:
            old,new=variants[base_mode],variants['match_structure']
            for j,i in zip(*np.where(old!=new)):
                differences.append({'round':name,'reference':base_mode,'candidate':f'{j+1:02d}','match_no':int(frame.match_no.iloc[i]),'home_team':frame.home_team.iloc[i],'away_team':frame.away_team.iloc[i],'old_symbol':'HDA'[old[j,i]],'new_symbol':'HDA'[new[j,i]],'D_changed':bool((old[j,i]==1)!=(new[j,i]==1)),'K_D':float(feature.K_D.iloc[i])})
        stamp=json.loads((source/'manifest.json').read_text())['created_at'];dates=pd.to_datetime(frame.datetime)
        if dates.dt.tz is None:dates=dates.dt.tz_localize('Asia/Tokyo')
        provenance.append({'round':name,'saved_version':version,'saved_before_kickoff':int((dates>pd.Timestamp(stamp)).sum()),'actual_D':int((actual==1).sum())})
        localdiff=pd.DataFrame([d for d in differences if d['round']==name]);localdiff.to_csv(dest/'placement_changes.csv',index=False)
        columns=['match_no','home_team','away_team','P_H','P_D','P_A','score_P_D','P_00','P_11','P_22plus','lambda_total','lambda_abs_difference','marginal_overlap','P_margin1','P_home_lead1','P_away_lead1','K_D','D_placement_value','reason']
        body='<p><a href="../index.html">全体報告へ</a></p><p>'+ ' ／ '.join(f'<a href="{mode}/index.html">{mode} 候補00〜10</a>' for mode in variants)+'</p>'
        body+='<h2>比較指標</h2>'+metric_table(pd.DataFrame(round_rows))+'<h2>試合別D評価</h2>'+table(feature[columns])
        body+='<p>score_P_Dと同点内訳は不確実性補正前の得点分布。P_Dは最終予測値なので一致しない場合がある。1点差は終局スコアの分布で、試合中の同点への遷移確率ではない。</p>'
        body+='<h2>一律1.2からD配置が変わった位置</h2>'+table(localdiff[(localdiff.reference=='uniform_1.2') & localdiff.D_changed])
        body+='<p><a href="match_D_evaluation.csv">全特徴量と係数の根拠CSV</a> ／ <a href="placement_changes.csv">全比較のセル差分CSV</a></p>'
        (dest/'index.html').write_text(page(name+' 試合別D係数シャドー',body))
        print(name,[(m['method'],round(m['p10']*100,3),m['actual_max']) for m in round_rows],flush=True)
    f=pd.concat(allfeatures,ignore_index=True);m=pd.DataFrame(allrows);diff=pd.DataFrame(differences)
    f.to_csv(OUT/'match_D_evaluation.csv',index=False);m.to_csv(OUT/'metrics.csv',index=False);diff.to_csv(OUT/'placement_changes.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    f.describe().to_csv(OUT/'feature_distribution.csv')
    pairs=[]
    for i in range(len(f)):
        for j in range(i):
            a,b=f.iloc[i],f.iloc[j]
            if abs(a.P_D-b.P_D)<=.001:
                pairs.append({'試合1':f'{a["round"]} {a.home_team}−{a.away_team}','試合2':f'{b["round"]} {b.home_team}−{b.away_team}','D確率1':a.P_D,'D確率2':b.P_D,'D内0–0比率1':a.zero_share_given_draw,'D内0–0比率2':b.zero_share_given_draw,'係数1':a.K_D,'係数2':b.K_D,'係数差':abs(a.K_D-b.K_D)})
    near=pd.DataFrame(pairs).sort_values('係数差',ascending=False);near.to_csv(OUT/'similar_D_pairs.csv',index=False)
    summary=m[~m.method.str.startswith('当時')].copy();summary['method']=summary.method.where(~summary.method.str.startswith('v2.4'),'v2.4比較対象')
    agg=summary.groupby('method').mean(numeric_only=True).reset_index()
    body='''<p>シャドー専用。既存配置アルゴリズム・固定枚数・候補00・元のH/D/A確率は維持。得点特徴から試合別係数を作り、既存の係数配列入力へ渡した。</p>
<h2>取得できたもの／取得していないもの</h2>
<p>全65試合に最終λと0–0・1–1・2–2が保存されている。既存モデルと同じ、両チーム独立Poisson・0〜10点の正規化分布を再構成し、保存スコア確率と1e−9以内で照合した。最終H/D/Aも不確実性混合を含め照合した。欠損や不一致は中止し、代替値で埋めない。</p>
<p>分布の重なりは両チームの得点周辺分布のminの和。1点差はH側・A側を別々に計算。退場・試合時間・試合中の状態遷移は取得しておらず、僅差から同点になる実際の確率は算出していない。</p>
<h2>具体式と暫定パラメータ</h2>
<p><code>c=P(0–0)/P_score(D)</code>：D内の0–0構成比。<br><code>n=2 min(P(Hが1点差勝ち), P(Aが1点差勝ち))/(1−P_score(D))</code>：非D内で両方向に存在する僅差成分。片側だけの1点差集中は加点しない。</p>
<p>65試合の入力特徴だけを使い、最終P(D)と補正前P_score(D)それぞれの3次多項式、および不確実性補正量を説明変数としてcの期待値を近似し、その残差r₀を求める。nは同じ説明変数とr₀で説明できる部分を除き、残差r₁を求める。D確率の水準と、特徴間の重複を減らすための記述的な正規化で、予測モデルや結果を学習したものではない。</p>
<p><code>z₀=tanh(r₀/s₀), z₁=tanh(r₁/s₁)</code><br><code>K_D=clip(exp(0.15 × (0.7 z₀ + 0.3 z₁)), 0.85, 1.20)</code><br>sは基準入力での残差標準偏差（最低0.02）。残差ゼロなら係数1.0。重み・強度・上下限・尺度下限は設定可能。tanhと上下限で過大な係数を抑える。</p>
<p>得点合計・差・分布重なり・同点内訳は診断に表示するが、相関した項目を独立の証拠として加算しない。正規化基準はJSONへ固定した。同じ試合を別の13試合へ組み替えても、同じ基準を使えば係数は変わらない。新しい試合のたびに基準を学習し直さない。</p>
<p class="note">この係数は「Dが本当に当たりやすくなった」ことを意味しない。同じD確率なら0–0と1–1はtotoでは同じDであり、0–0構成を優先する理論的な的中優位は未証明。今回の0.7/0.3と0.15は、予測結果に合わせず設定した比較用の選好。多項式残差化で二重評価を完全に除去できる保証もない。</p>
<h2>入力分布と係数</h2>'''
    body+=table(f[['P_D','P_00','P_11','P_22plus','lambda_total','lambda_abs_difference','marginal_overlap','P_margin1','zero_share_given_draw','balanced_near_given_non_draw','K_D']].describe().reset_index())
    body+='<p>係数の平均が一律1.2とは異なるため、試合別係数の平均と同じ一律係数も対照に加えた。これにより平均的な優遇度と、試合別に変えた効果を分けて見る。</p>'
    body+='<h2>同程度のD確率で係数が異なる例</h2>'+table(near.head(8))
    body+=f'<p>D確率差0.1パーセントポイント以内の組を表示。65試合のうちP(0–0)&gt;P(1–1)は{int((f.P_00>f.P_11).sum())}試合。両タイプが十分あると主張せず、実際にある構成比の違いを比較した。</p>'
    body+='<h2>5開催平均・モデル上の比較</h2>'+metric_table(agg)
    measures=agg.set_index('method')
    body+=f'<p><b>今回の評価：</b>試合別係数の10的中以上は{measures.loc["match_structure","p10"]*100:.4f}%、一律1.0は{measures.loc["uniform_1.0","p10"]*100:.4f}%、平均係数を揃えた対照は{measures.loc["uniform_matched_mean","p10"]*100:.4f}%。差は小さく、試合別化による優位性は確認できない。係数は実際にDの配置位置を変えたが、この暫定式を正式採用する根拠とはしない。</p>'
    conditional=agg[['method','D_low_0_2_p10','D_mid_3_4_p10','D_high_5plus_p10']].copy()
    for col in conditional.columns[1:]:conditional[col]*=100
    body+='<h2>実D数で条件付けた10的中以上の確率（%）</h2>'+table(conditional)
    body+='<h2>実結果の最高的中数</h2>'+table(summary.pivot(index='round',columns='method',values='actual_max').reset_index())
    body+='<h2>配置差分の件数</h2>'+table(diff.groupby(['round','reference']).agg(changed_cells=('candidate','size'),changed_D_positions=('D_changed','sum')).reset_index())
    body+='<h2>各開催の評価表・配置・具体的な変更位置</h2><ul>'+''.join(f'<li><a href="{name}/index.html">{name}</a></li>' for name in frames)+'</ul>'
    body+='<h2>検証の限界</h2>'+table(pd.DataFrame(provenance))
    body+='''<p>v2.4の保存券がある開催だけを「v2.4保存券」と表示。他開催は同じ予測・配分からv2.4を再生成し、当時の別版の保存券は別行に記載した。実Dが5以上の開催はなく、多D評価はモデル内条件付き確率である。</p>
<p>65試合の入力全体で正規化を作っているため、過去の時点で作成済みの係数という扱いはできない。結果を係数の学習・調整に使用していないが、今回全体が試作内の記述的検証であり独立ホールドアウトではない。1668は開発データ、保存時刻が全試合開始前なのは1663だけ。推定確率・実結果の双方を示し、正式優位や予測精度改善とは判定しない。</p>
<p>追加6テストを通過。分布の再現、欠損・不一致時の停止、同一基準下のバッチ独立性、実結果の未使用、D確率・特徴重複の残差化、同一D確率での構造差、係数上限下限を検証した。候補生成は既存v2.5の実装を変更せず使用。</p><p>再実行：<code>./scripts/.venv/bin/python scripts/next_model/simulate_match_draw_shadow.py</code></p>
<p><a href="normalization_reference.json">正規化基準と設定</a> ／ <a href="match_D_evaluation.csv">65試合の全特徴量</a> ／ <a href="placement_changes.csv">候補・試合別の全差分</a> ／ <a href="metrics.csv">比較指標</a></p>'''
    (OUT/'index.html').write_text(page('v2.5 試合別D特性係数：シャドー検証',body))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    (OUT/'unchanged_code_hashes.json').write_text(json.dumps(protected,indent=2))
    print('REPORT',OUT/'index.html')


if __name__=='__main__':main()
