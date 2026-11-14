"""Read-only audit of archived Next Model lambdas for a future error model."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from compare_d_scoop_uncertainty import ROOT, ROUNDS, read_round, sha

OUT=ROOT/'data/reports/score_structure_phase4a_audit'
HISTORY=ROOT/'data/eval/next_model/correction_history_v1/matches.csv'


def model_signature(row, columns):
    if 'goal_rate_adjustment_version' in columns and pd.notna(row.get('goal_rate_adjustment_version')):
        return 'saved_goal_rate_plus_style_policy_1670'
    if 'next_uncertainty_calibration_mix' in columns and pd.notna(row.get('next_uncertainty_calibration_mix')):
        return 'saved_absence_uncertainty_1665_1668'
    return 'saved_early_score_model_1660_1664'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[];sources=[];inventory=[]
    for rid in ROUNDS:
        diagnostic,provenance=read_round(rid)
        folder=ROOT/'data/eval/toto_rounds'/rid
        source=folder/'snapshot/next_predictions.csv'
        f=pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
        actual=pd.read_csv(folder/'actual_results.csv').sort_values('match_no').reset_index(drop=True)
        manifest=json.loads((folder/'snapshot/buyplan_engine_manifest.json').read_text())
        for i,r in f.iterrows():
            a=actual.iloc[i]
            if int(r.match_no)!=int(a.match_no):raise ValueError('Match number mismatch')
            records.append(dict(round=rid,match_no=int(r.match_no),match_id=r.match_id,
                league=str(r.match_id).split('_')[0],season=str(r.match_id).split('_')[1],
                kickoff=r.datetime,home_team=r.home_team,away_team=r.away_team,
                forecast_created_at=manifest['created_at'],
                final_lambda_home=r.final_lambda_home,final_lambda_away=r.final_lambda_away,
                next_prob_home=r.next_prob_home,next_prob_draw=r.next_prob_draw,next_prob_away=r.next_prob_away,
                uncertainty_mix=r.get('next_uncertainty_calibration_mix',0),
                probability_source=r.get('buyplan_probability_source',pd.NA),
                goal_rate_adjustment_version=r.get('goal_rate_adjustment_version',pd.NA),
                style_official_version=r.get('style_official_version',pd.NA),
                model_signature=model_signature(r,f.columns),
                actual_home_score=a.home_score,actual_away_score=a.away_score,
                actual_result=int(a.result),provenance_class='A',
                source_file=str(source.relative_to(ROOT)),source_sha256=sha(source),
                source_kind='archived_prematch_next_predictions',
                prematch_hash_verified=True,actual_join_verified=True,
                duplicate_of_A=False,eligible_for_A_only_model=True,
                exclusion_reason=''))
        sources.append(provenance)
        inventory.append(dict(source=str(source.relative_to(ROOT)),type='A: original Next Model forecast',
                              rows=len(f),actual_scores='yes',lambda_values='yes',
                              provenance='manifest hash and created_at before kickoff verified'))
    direct=pd.DataFrame(records)
    if len(direct)!=78 or direct[['round','match_no']].duplicated().any():raise AssertionError('A population invalid')
    history=pd.read_csv(HISTORY)
    if len(history)!=88:raise AssertionError('Correction history size changed')
    hmanifest=json.loads((HISTORY.parent/'manifest.json').read_text())
    if hmanifest.get('forecast_source')!='recomputed with current build_next_probabilities from frozen prematch predictions':
        raise ValueError('Correction history provenance changed')
    akeys=set(zip(direct['round'],direct.match_no))
    for _,r in history.iterrows():
        duplicate=(r.round_id,int(r.match_no)) in akeys
        records.append(dict(round=r.round_id,match_no=int(r.match_no),match_id=r.match_id,
            league=str(r.match_id).split('_')[0],season=str(r.match_id).split('_')[1],
            kickoff=r.datetime,home_team=r.home_team,away_team=r.away_team,
            forecast_created_at=hmanifest['sources'][r.round_id].get('prematch_snapshot_created_at'),
            final_lambda_home=r.final_lambda_home,final_lambda_away=r.final_lambda_away,
            next_prob_home=r.next_prob_home,next_prob_draw=r.next_prob_draw,next_prob_away=r.next_prob_away,
            uncertainty_mix=r.get('next_uncertainty_calibration_mix',pd.NA),
            probability_source='recomputed_current_pipeline_from_frozen_inputs',
            goal_rate_adjustment_version=pd.NA,style_official_version=pd.NA,
            model_signature='retrospective_correction_history_v1',
            actual_home_score=r.actual_home_score,actual_away_score=r.actual_away_score,
            actual_result=int(r.actual_result),provenance_class='C',
            source_file=str(HISTORY.relative_to(ROOT)),source_sha256=sha(HISTORY),
            source_kind='retrospective_recomputed_lambda_from_frozen_inputs',
            prematch_hash_verified=False,actual_join_verified=True,
            duplicate_of_A=duplicate,eligible_for_A_only_model=False,
            exclusion_reason='recomputed lambda, not the originally saved Next Model lambda'+
                             ('; same fixture as A' if duplicate else '')))
    inventory.append(dict(source=str(HISTORY.relative_to(ROOT)),type='C: retrospective recomputation',
                          rows=len(history),actual_scores='yes',lambda_values='yes',
                          provenance='original inputs frozen; lambdas recomputed later by current pipeline'))
    for source,kind in [
        ('data/eval/next_model/validation_v2/matches.csv','legacy validation population'),
        ('data/reports/next_probability_calibration/matches.csv','legacy probability calibration population'),
        ('data/purchase_reference/next_predictions.csv','current working copy'),
        ('data/eval/next_model/integration/toto1654/next_predictions.csv','post-match integration output')]:
        p=ROOT/source
        if p.exists():
            f=pd.read_csv(p)
            inventory.append(dict(source=source,type=kind,rows=len(f),
                                  actual_scores='yes' if 'actual_home_score' in f else 'separate or unknown',
                                  lambda_values='yes' if 'final_lambda_home' in f else 'no',
                                  provenance='not added to A: legacy/no lambda, mutable copy, or post-match generation'))
    observed=ROOT/'data/purchase_reference/observation_history'
    for p in sorted(observed.rglob('next_predictions.csv')):
        f=pd.read_csv(p)
        inventory.append(dict(source=str(p.relative_to(ROOT)),type='purchased archive duplicate',
                              rows=len(f),actual_scores='separate',lambda_values='yes',
                              provenance='same round forecast already represented in A; not an independent match'))
    for p in sorted((ROOT/'data/repair_backups').glob('*/next_predictions.csv')):
        f=pd.read_csv(p)
        inventory.append(dict(source=str(p.relative_to(ROOT)),type='D: repair backup with unverified creation time',
                              rows=len(f),actual_scores='separate',lambda_values='yes',
                              provenance='historical copy, no matching original forecast manifest in this folder; not counted as A'))
    inventory.append(dict(source='data/selenium_profile/**/*.db',type='browser profile database',rows=None,
                          actual_scores='no',lambda_values='no',provenance='not a prediction database'))
    joined=pd.DataFrame(records)
    joined.to_csv(OUT/'forecast_result_join.csv',index=False,encoding='utf-8-sig')
    quality=joined[['round','match_no','match_id','league','season','provenance_class','model_signature',
                    'source_file','forecast_created_at','prematch_hash_verified','actual_join_verified',
                    'duplicate_of_A','eligible_for_A_only_model','exclusion_reason']].copy()
    quality.to_csv(OUT/'data_quality.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(inventory).to_csv(OUT/'data_inventory.csv',index=False,encoding='utf-8-sig')
    stats=dict(A_forecast_rows=len(direct),A_distinct_matches=direct.match_id.nunique(),
        A_actual_joined=int(direct.actual_home_score.notna().sum()),
        C_recomputed_rows=len(history),C_overlap_A=int(joined.loc[joined.provenance_class.eq('C'),'duplicate_of_A'].sum()),
        C_additional_distinct=int((joined.provenance_class.eq('C')&~joined.duplicate_of_A).sum()),
        A_by_model=direct.model_signature.value_counts().to_dict(),
        A_by_league=direct.league.value_counts().to_dict(),
        A_by_season=direct.season.value_counts().to_dict(),
        A_rounds=direct.groupby('round').size().to_dict(),
        total_distinct_A_plus_C=joined.match_id.nunique(),
        A_earliest=direct.kickoff.min(),A_latest=direct.kickoff.max(),
        input_file_sources=sources,correction_history_sha256=sha(HISTORY))
    (OUT/'audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    report=f'''# Phase 4-A準備：予測誤差モデルのデータ監査

## 結論

**B：共通のNext Model予測誤差モデルを、独立した時系列評価まで含めて試験実装するには、原本Aの蓄積が先に必要。** 厳密な試合前保存を確認でき、実得点まで照合できたのは**{stats['A_distinct_matches']}試合**（6開催）。この期間に予測処理が少なくとも3系統に分かれるため、同一版だけの学習・調整・未使用評価に分けるには小さい。探索的な尤度診断や簡単な過分散チェックは可能だが、真のλ誤差分布を安定推定・検証できるとは言えない。構造型はさらに厳しい。

## 所在と分類

- A：`data/eval/toto_rounds/toto*/snapshot/next_predictions.csv` の6開催、{stats['A_forecast_rows']}予測行＝{stats['A_distinct_matches']}実試合。購入エンジンmanifestの出力ハッシュ、作成時刻＜キックオフ、試合番号・対戦カードと実結果の一致を確認。実得点結合は{stats['A_actual_joined']}行。
- C：`data/eval/next_model/correction_history_v1/matches.csv` に{stats['C_recomputed_rows']}行。うち{stats['C_overlap_A']}行はAの試合と重複、{stats['C_additional_distinct']}行はA以外。入力の元スナップショットは試合前保存だが、λは**後から現行処理で再計算**した値。原本Aと混ぜない。
- B/D候補：修復バックアップ、現在の作業コピー、旧検証データ、購入履歴を棚卸しした。購入履歴のnext_predictionsは既存Aの同試合の複製であり、新しい実試合数ではない。旧検証表はλを保存していないか、別モデルの検証出力。詳細は `data_inventory.csv`。
- 修復バックアップのnext_predictionsは3ファイル各13行。単独では生成時刻の原本manifestがないためD扱いで、Aを増やさない。B（試合前と推定できるが証拠不足）へ昇格できるかは別途出所の確認が必要。
- リポジトリ内の `.db` はブラウザプロファイルのデータで、予測DBではない。

## モデル版と時系列

A原本の内訳は {stats['A_by_model']}。最初の3開催は保存されたmix列がなく、行列H/D/Aとの一致からmix=0を確認。1665・1668は不確実性mixあり、1670は直近得失点の外れ値調整と追加スタイル統合の監査列あり。`model_signature` は**保存列から作った監査用の区分**で、全処理の厳密なソフトウェア版番号ではない。特に1670のスタイル統合は設定が有効でも係数が未学習なら補正ゼロであり、適用の有無を列から再確認する必要がある。

AとCで重なる39試合は同一の実結果を持つが、λの最大差は0.122点。試合前に固定した元予測と、後から再計算したλを無区別に統合してはいけない。

Aのリーグ別は {stats['A_by_league']}、シーズン別は {stats['A_by_season']}。対象期間は {stats['A_earliest']} ～ {stats['A_latest']}。Aの開催順は {stats['A_rounds']}。各系統の試合数が39、26、13のため、系統ごとに推定用・調整用・最終未使用評価用の3期間を確保しにくい。J1/J2を別々に推定するならさらに不足する。

## 統計的な制約

実得点と予測λの差にはポアソンの偶然変動が含まれる。1試合の真のλは観測できない。共通型のホーム・アウェイ偏り、追加分散、相関を同時に推定する場合、少なくとも複数のパラメータが必要で、78試合を版・リーグ・時系列に分けると不確実性が大きい。得点分布の仮定そのものは、保存λからのPoisson対数尤度や予測区間被覆の探索的診断で調べられる。H/D/Aの最終確率にはλ後段のmixがあるので、λ分布の評価とは分ける。

構造型のλ合計・差、0-0、1点差、2点差以上は同じλから導く特徴であり、独立な情報源ではない。現在のA原本だけで複数の交互作用や条件別分散を安定推定する根拠はない。

## 本実装の最初の成果物

最初に**試合前予測原本と実結果の追記専用レジストリ**を作る。試合ID、予測日時、入力・出力ハッシュ、コード版、補正設定、λ、raw H/D/A、mix、final H/D/A、実得点、結果確定日時を保存し、同じ実試合の複数予測時点を別行として紐付ける。学習時は1実試合1観測を選び、予測時点より後の入力を拒否する。過去データの追加復元は原本Aと分離し、作成時刻・入力のas-ofを監査できるものだけ昇格させる。

その後、同一モデル系統が蓄積した段階で、まずPoisson基準の得点対数尤度・平均残差・分散診断を実施。必要な場合に限り少数パラメータの共通混合モデルを時系列ウォークフォワードで比較し、さらに後続の未使用期間で得点対数尤度、rawとfinalそれぞれのH/D/A Log Loss・Brier、校正を確認する。構造型はその先の比較候補とする。今回、推定や係数調整は行っていない。
'''
    (OUT/'report.md').write_text(report)
    print(json.dumps({k:stats[k] for k in ('A_forecast_rows','A_distinct_matches','A_actual_joined','C_recomputed_rows','C_overlap_A','C_additional_distinct','A_by_model','A_by_league')},ensure_ascii=False))


if __name__=='__main__':main()
