"""Exploratory D branches without a D-argmax gate; forward-only shadow tickets."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure_phase4b_stage1 import decode,top_ten,universe

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/reports/score_structure_phase35/diagnostics_78.csv'
OUT=ROOT/'data/reports/score_structure_phase35_draw_recheck'
SYMS=np.array(['H','D','A'])
ROUNDS=['toto1660','toto1663','toto1664','toto1665','toto1668','toto1670']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame):
    cols=list(frame.columns)
    def fmt(v):
        if pd.isna(v):return ''
        if isinstance(v,(float,np.floating)):
            return f'{v:.8f}' if 0<abs(v)<.0001 else f'{v:.4f}'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def condition_rows(d):
    d=d.copy()
    d['rare']=d.low_P00 & d.high_P11
    d['broad']=d.low_P00 & d.high_P2plus
    d['broad_only']=d.broad & ~d.rare
    d['gap_abs']=abs(d.baseline_P_H-d.baseline_P_A)
    d['near_HA']=d.gap_abs<=d.groupby('round').gap_abs.transform('median')
    return d


def analyze_conditions(d):
    rows=[]
    cond={'0-0低・1-1高':'rare','0-0低・2-2以上高':'broad','広域のみ（希少条件を除く）':'broad_only',
          'H/A確率差が開催内中央値以下':'near_HA'}
    for name,col in cond.items():
        yes=d[d[col]];no=d[~d[col]]
        counts=yes.actual_symbol.value_counts()
        exp=yes[['baseline_P_H','baseline_P_D','baseline_P_A']].sum()
        rows.append(dict(condition=name,n=len(yes),actual_H=int(counts.get('H',0)),actual_D=int(counts.get('D',0)),
                         actual_A=int(counts.get('A',0)),nonD=len(yes)-int(counts.get('D',0)),
                         expected_H=exp.baseline_P_H,expected_D=exp.baseline_P_D,expected_A=exp.baseline_P_A,
                         D_rate=float(yes.actual_symbol.eq('D').mean()) if len(yes) else np.nan,
                         noncondition_D_rate=float(no.actual_symbol.eq('D').mean()) if len(no) else np.nan,
                         D_excess=int(counts.get('D',0))-exp.baseline_P_D,
                         baseline_top_hit=int((yes.baseline_symbol==yes.actual_symbol).sum())))
    return pd.DataFrame(rows)


def rank_tickets(prob):
    mass=universe(prob)
    code=top_ten(mass)
    return decode(code),float(mass[code].sum())


def update_probabilities(target,train,kappa,with_HA=False,include_broad=False):
    """Explicitly hypothetical Dirichlet/Beta shrinkage; no target results used."""
    result=target[['baseline_P_H','baseline_P_D','baseline_P_A']].to_numpy(float).copy()
    details=[]
    for i,r in enumerate(target.itertuples(index=False)):
        label='rare' if r.rare else 'broad_only' if (include_broad and r.broad_only) else 'none'
        n=0; counts=np.zeros(3,dtype=int)
        if label!='none':
            group=train[train[label]]
            n=len(group)
            counts=np.array([(group.actual_symbol==s).sum() for s in SYMS],int)
            result[i]=(kappa*result[i]+counts)/(kappa+n)
        ha_n=0;ha_counts=np.zeros(2,dtype=int)
        if with_HA and r.near_HA and label=='none':
            group=train[train.near_HA & ~train.rare & ~train.broad_only & train.actual_symbol.ne('D')]
            ha_n=len(group)
            ha_counts=np.array([(group.actual_symbol==s).sum() for s in ('H','A')],int)
            pD=result[i,1];den=result[i,0]+result[i,2]
            ha=np.array([result[i,0],result[i,2]])/den
            ha=(kappa*ha+ha_counts)/(kappa+ha_n)
            result[i,0]=(1-pD)*ha[0];result[i,2]=(1-pD)*ha[1]
        details.append(dict(round=r.round,match_no=r.match_no,condition=label,prior_rounds=len(train['round'].unique()),
                            train_condition_n=n,train_H=int(counts[0]),train_D=int(counts[1]),train_A=int(counts[2]),
                            near_HA=bool(r.near_HA),HA_train_nonD_n=ha_n,HA_train_H=int(ha_counts[0]),HA_train_A=int(ha_counts[1]),
                            original_P_H=r.baseline_P_H,original_P_D=r.baseline_P_D,original_P_A=r.baseline_P_A,
                            shadow_P_H=result[i,0],shadow_P_D=result[i,1],shadow_P_A=result[i,2]))
    assert np.allclose(result.sum(axis=1),1,atol=1e-12)
    return result,details


def run():
    d=pd.read_csv(SOURCE).sort_values(['round','match_no']).reset_index(drop=True)
    assert len(d)==78 and d['round'].unique().tolist()==ROUNDS and not d.duplicated(['round','match_no']).any()
    d=condition_rows(d)
    sources=[SOURCE,ROOT/'data/reports/score_structure_phase35/branch_candidates.csv',ROOT/'data/reports/score_structure_phase4a_audit/forecast_result_join.csv']
    before={str(p.relative_to(ROOT)):sha(p) for p in sources}
    conditions=analyze_conditions(d)
    band_rows=[]
    for band in ('低','中','高'):
        for flag in (True,False):
            subset=d[d.PD_band.eq(band)&d.rare.eq(flag)]
            band_rows.append(dict(PD_band=band,rare_condition=flag,n=len(subset),actual_D=int(subset.actual_symbol.eq('D').sum()),
                                  D_rate=float(subset.actual_symbol.eq('D').mean()) if len(subset) else np.nan,
                                  mean_P_D=float(subset.baseline_P_D.mean()) if len(subset) else np.nan))
    pd_bands=pd.DataFrame(band_rows)
    # A compact, predeclared threshold sensitivity. Values are diagnostics, not tuned selection rules.
    sensitivity=[]
    for low_q,high_q in ((.4,.6),(.5,.5),(.6,.4)):
        low=d.P_0_0<=d.groupby('round').P_0_0.transform(lambda x:x.quantile(low_q))
        high=d.P_1_1>d.groupby('round').P_1_1.transform(lambda x:x.quantile(high_q))
        subset=d[low&high]
        sensitivity.append(dict(low_P00_quantile=low_q,high_P11_quantile=high_q,n=len(subset),actual_D=int(subset.actual_symbol.eq('D').sum()),
                                actual_H=int(subset.actual_symbol.eq('H').sum()),actual_A=int(subset.actual_symbol.eq('A').sum()),
                                expected_D=float(subset.baseline_P_D.sum())))
    threshold=pd.DataFrame(sensitivity)
    # P(D) and T/log-R comparisons use same model version and near neighbors; no fit.
    rare=d[d.rare];matches=[]
    for r in rare.itertuples(index=False):
        pool=d[(~d.rare)&(d.model_signature==r.model_signature)&(d['round']!=r.round)].copy()
        if pool.empty:continue
        pool['PD_gap']=(pool.baseline_P_D-r.baseline_P_D).abs()
        pool['T_gap']=abs(pool['T']-r.T)
        pool['logR_gap']=abs(abs(pool.log_R)-abs(r.log_R))
        for kind,sortcols in [('near_PD',['PD_gap','T_gap']),('near_T_logR',['T_gap','logR_gap'])]:
            other=pool.sort_values(sortcols+['round','match_no']).iloc[0]
            matches.append(dict(kind=kind,rare_round=r.round,rare_match_no=r.match_no,rare_fixture=f'{r.home_team}－{r.away_team}',
                                rare_actual=r.actual_symbol,comparison_round=other['round'],comparison_match_no=other.match_no,
                                comparison_fixture=f"{other.home_team}－{other.away_team}",comparison_actual=other.actual_symbol,
                                rare_PD=r.baseline_P_D,comparison_PD=other.baseline_P_D,PD_gap=abs(r.baseline_P_D-other.baseline_P_D),
                                rare_T=r.T,comparison_T=other['T'],rare_abs_logR=abs(r.log_R),comparison_abs_logR=abs(other.log_R),
                                rare_P00=r.P_0_0,comparison_P00=other.P_0_0,rare_P11=r.P_1_1,comparison_P11=other.P_1_1))
    matches=pd.DataFrame(matches)
    # Three base methods; kappa is deliberately varied, not optimized on this same dataset.
    tickets=[];summaries=[];parameters=[];scenario_rows=[];allocations=[]
    for kappa in (1.,3.,10.):
        for idx,rnd in enumerate(ROUNDS):
            target=d[d['round']==rnd].sort_values('match_no').reset_index(drop=True)
            train=d[d['round'].isin(ROUNDS[:idx])]
            original=target[['baseline_P_H','baseline_P_D','baseline_P_A']].to_numpy(float)
            proposals={'基準・数学上位10口':(original,[])}
            proposals['希少条件Dシャドー']=update_probabilities(target,train,kappa,False)
            proposals['希少条件D＋広域シャドー']=update_probabilities(target,train,kappa,False,True)
            proposals['希少条件D＋HAシャドー']=update_probabilities(target,train,kappa,True)
            plans={}
            for method,(prob,details) in proposals.items():
                if method!='基準・数学上位10口':
                    for item in details:parameters.append(dict(kappa=kappa,method=method,**item))
                plan,_=rank_tickets(prob)
                plans[method]=plan
                original_mass=float(np.prod(original[np.arange(13)[None,:],plan],axis=1).sum())
                shadow_mass=float(np.prod(prob[np.arange(13)[None,:],plan],axis=1).sum())
                actual=target.actual_symbol.to_numpy()
                ticket_symbols=SYMS[plan]
                hits=(ticket_symbols==actual[None,:]).sum(axis=1)
                d_games=set(np.where((plan==1).any(axis=0))[0].tolist())
                actual_D=set(np.where(actual=='D')[0].tolist())
                actual_missing=[j+1 for j in range(13) if actual[j] not in set(ticket_symbols[:,j])]
                missing_HA=[j for j in actual_missing if actual[j-1]!='D']
                ha_loss=[]
                for j in range(13):
                    if actual[j]=='D':continue
                    original_symbols=set(SYMS[plans['基準・数学上位10口']][:,j])
                    new_symbols=set(ticket_symbols[:,j])
                    if actual[j] in original_symbols and actual[j] not in new_symbols:ha_loss.append(j+1)
                for num,(sym,hit) in enumerate(zip(ticket_symbols,hits),1):
                    row=dict(round=rnd,kappa=kappa,method=method,ticket_no=num,symbols=''.join(sym),D_count=int((sym=='D').sum()),hits=int(hit),
                             original_13_probability=float(np.prod(original[np.arange(13),plan[num-1]])),
                             shadow_13_probability=float(np.prod(prob[np.arange(13),plan[num-1]])))
                    row.update({f'match_{j+1:02d}':str(sym[j]) for j in range(13)})
                    tickets.append(row)
                for j in range(13):
                    allocations.append(dict(round=rnd,kappa=kappa,method=method,match_no=j+1,
                        home_team=target.iloc[j].home_team,away_team=target.iloc[j].away_team,actual_symbol=actual[j],
                        baseline_P_H=original[j,0],baseline_P_D=original[j,1],baseline_P_A=original[j,2],
                        rare_condition=bool(target.iloc[j].rare),broad_condition=bool(target.iloc[j].broad),
                        **{f'ticket_{num+1:02d}':str(ticket_symbols[num,j]) for num in range(10)}))
                summaries.append(dict(round=rnd,kappa=kappa,method=method,original_13_probability=original_mass,
                                      shadow_13_probability=shadow_mass,max_hits=int(hits.max()),exact13=bool((hits==13).any()),
                                      D_adopted_matches=len(d_games),actual_D_captured=len(d_games&actual_D),
                                      D_false_adopted=len(d_games-actual_D),actual_D_missed=len(actual_D-d_games),
                                      actual_HA_unrepresented=len(missing_HA),all_actual_symbols_represented=not actual_missing,
                                      missed_by_combination_only=bool(not actual_missing and not (hits==13).any()),
                                      H_A_actual_symbols_lost_by_match=';'.join(map(str,ha_loss)),
                                      overlap_with_baseline=len(set(map(tuple,plan))&set(map(tuple,plans['基準・数学上位10口'])))))
            baseplan=plans['基準・数学上位10口']
            for method in ('希少条件Dシャドー','希少条件D＋広域シャドー','希少条件D＋HAシャドー'):
                for ticket in plans[method]:
                    if tuple(ticket) in set(map(tuple,baseplan)):continue
                    alt=[j+1 for j in range(13) if ticket[j]==1 and not (baseplan==1).any(axis=0)[j]]
                    distances=(baseplan!=ticket).sum(axis=1)
                    nearest=baseplan[int(np.argmin(distances))]
                    diffs=[f'{j+1}:{SYMS[nearest[j]]}→{SYMS[ticket[j]]}' for j in range(13) if nearest[j]!=ticket[j]]
                    scenario_rows.append(dict(round=rnd,kappa=kappa,method=method,symbols=''.join(SYMS[ticket]),
                                              D_on_new_games=';'.join(map(str,alt)),
                                              nearest_baseline=''.join(SYMS[nearest]),changed_symbols=';'.join(diffs),
                                              branch_conditions=';'.join(f'{j}:{"0-0低・1-1高" if target.iloc[j-1].rare else "0-0低・2-2以上高" if target.iloc[j-1].broad_only else "その他"}' for j in alt),
                                              actual_hits=int((SYMS[ticket]==target.actual_symbol.to_numpy()).sum())))
    OUT.mkdir(parents=True,exist_ok=True)
    outputs={'condition_identification.csv':conditions,'threshold_sensitivity.csv':threshold,'matched_rare_examples.csv':matches,
             'same_PD_band.csv':pd_bands,
             'shadow_tickets.csv':pd.DataFrame(tickets),'shadow_summary.csv':pd.DataFrame(summaries),
             'training_evidence.csv':pd.DataFrame(parameters),'changed_tickets.csv':pd.DataFrame(scenario_rows),
             'allocation_matrix.csv':pd.DataFrame(allocations)}
    candidate=d[['round','match_no','home_team','away_team','baseline_symbol','actual_symbol','baseline_P_H','baseline_P_D','baseline_P_A',
                 'rare','broad','P_0_0','P_1_1','P_draw_2plus','T','log_R']].copy()
    candidate['condition_support']=np.where(candidate.rare,'0-0低・1-1高',np.where(candidate.broad,'0-0低・2-2以上高のみ','未確認'))
    candidate['development_condition_D_rate']=np.where(candidate.rare,4/5,np.where(candidate.broad,6/28,np.nan))
    candidate['development_condition_H_rate']=np.where(candidate.rare,1/5,np.where(candidate.broad,14/28,np.nan))
    candidate['development_condition_A_rate']=np.where(candidate.rare,0.,np.where(candidate.broad,8/28,np.nan))
    candidate['HA_probability_if_D_replaces_one']=candidate[['baseline_P_H','baseline_P_A']].max(axis=1)
    candidate['D_candidate_status']='探索段階・独立検証なし'
    candidate['purchase_decision']='未決定・D首位かどうかで除外しない'
    candidate.to_csv(OUT/'D_purchase_candidates.csv',index=False,encoding='utf-8-sig')
    for name,frame in outputs.items():frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in sources}
    audit=dict(source_hashes=before,rounds=ROUNDS,all_78_are_development_data=True,
               target_results_used_to_generate_target_tickets=False,training_only_prior_rounds=True,
               condition_was_discovered_on_all_78=True,independent_validation=False,
               pseudo_count_sensitivity=[1,3,10],formal_model_or_buyplan_changed=False,
               D_argmax_used_as_branch_gate=False,D_minimum_ticket_quota=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return d,conditions,threshold,matches,outputs['shadow_summary.csv'],outputs['shadow_tickets.csv'],audit


def render(d,c,t,m,s,tickets,audit):
    rare=d[d.rare].copy();rare['actual_score']=rare.actual_home.astype(int).astype(str)+'–'+rare.actual_away.astype(int).astype(str)
    rare['in_broad']=rare.broad
    # Report all results, including unfavorable priors. No selection of a best kappa.
    base=s[s.kappa==3]
    aggregate=s.groupby(['kappa','method']).agg(D_captured=('actual_D_captured','sum'),false_D_adoptions=('D_false_adopted','sum'),
        mean_best_hits=('max_hits','mean'),exact13_rounds=('exact13','sum'),
        model_13_mass_sum=('original_13_probability','sum'),mean_baseline_overlap=('overlap_with_baseline','mean')).reset_index()
    bands=pd.read_csv(OUT/'same_PD_band.csv')
    report='\n\n'.join([
      '# Phase 3.5：D首位ゲート廃止後の条件・購入判断の再検証',
      '**Dが最高確率でなくてもD分岐候補から除外しない。** 旧Phase 3.5のD首位判定は今回は採否に使用しない。試合前保存Aの6開催78試合を開発用標本として使用し、Next Model、正式BuyPlan、購入フローは変更していない。',
      '## 「0-0低・1-1高」の5試合',
      '条件は各開催13試合の事前P(0-0)が中央値以下、P(1-1)が中央値超。中央値は予測分布だけから計算され、実結果を使わない。同一開催内の相対条件であり、開催間の絶対閾値ではない。',
      table(rare[['round','match_no','home_team','away_team','actual_score','actual_symbol','baseline_P_H','baseline_P_D','baseline_P_A','P_0_0','median_P_0_0','P_1_1','median_P_1_1','T','log_R','in_broad']]),
      '該当5試合はD4・H1・A0。基準予測の期待合計はD1.36試合分。神戸－名古屋（H、3–1）が明確な反例。5試合すべて「0-0低・2-2以上高」にも該当し、希少条件が広域条件から独立した情報とは言えない。',
      '### 同じP(D)帯との比較',
      table(bands),
      '## 条件全体とH/D/A実績',
      table(c),
      '「0-0低・2-2以上高」は希少条件を除くと28試合でD6、非D22。基準P(D)の期待数との差だけで有効性を判断しない。各試合の得点構造指標は保存λから決定論的に出るので、TとlogRにない独立情報が増えたわけではない。ただし、非線形な条件がP(D)の要約で失われた関係を取り出す可能性は、後続試合で検証できる。',
      '## 条件閾値の感応度',
      '同じ78試合で閾値を最適化せず、開催内の40/60・50/50・60/40分位を並べた初回診断。後続の細分化監査により、この3点だけから条件の頑健性が弱いとは判断できないと訂正した。13件の順位境界と中央値の等号が件数に影響する。45〜49.9999%では4試合中4D、50.0001〜57.5%では8試合中6Dの安定区間もある。詳細は ../score_structure_phase35_threshold_audit/index.html を参照。',
      table(t),
      '## 近いP(D)・T・得点力比の比較',
      '同一モデル版の別開催から、P(D)が近い試合とT・|logR|が近い試合をそれぞれ探した診断。厳密なマッチングや独立評価ではない。同じ非D試合が複数の比較相手として再利用され得る。1670回は同じモデル版の別開催がなく対応相手を作れなかった。全件をCSVに残す。',
      table(m[['kind','rare_fixture','rare_actual','comparison_fixture','comparison_actual','PD_gap','rare_T','comparison_T','rare_abs_logR','comparison_abs_logR']]),
      '## 研究用シャドー配券',
      '「基準」は保存final確率の数学的上位10口。「希少条件D」は0-0低・1-1高の試合だけ、「希少条件D＋広域」はそれに0-0低・2-2以上高の広域のみ試合を加え、**対象開催より前の開催だけ**のH/D/A件数で条件付き確率を更新する。「希少条件D＋HA」は希少条件に加え、条件外のH/A拮抗試合について過去の非D結果からH/A比率だけを更新する。いずれもD口数の下限・固定ダブル・D首位ゲートはない。',
      '更新は事前確率を中心とするDirichlet/Beta擬似件数で、擬似件数1・3・10をすべて出力。これらは未検証の仮定であり、正式なD確率補正ではない。各開催の結果は、その開催の券生成後にのみ採点。条件自体を78試合全体で発見しているため、前方更新でも独立検証ではない。',
      table(base[['round','method','D_adopted_matches','actual_D_captured','D_false_adopted','actual_D_missed','actual_HA_unrepresented','missed_by_combination_only','H_A_actual_symbols_lost_by_match','max_hits','exact13','original_13_probability','overlap_with_baseline']]),
      '`actual_HA_unrepresented` は、実際のH/A記号がその試合で10口のどれにも含まれない試合数。`H_A_actual_symbols_lost_by_match` は基準10口には含まれていた実H/A記号がシャドーでは消えた試合番号。いずれも13試合の組み合わせが合ったかとは別に診断する。',
      '### 擬似件数を変えた場合',
      table(aggregate),
      '希少条件だけの案は擬似件数1/3で実D3・誤採用1、擬似件数10でも実D3・誤採用1だった。ただし後続の完全未使用データでは未検証。広域条件追加やH/A更新は券を大きく変え、元の保存final確率で測る13的中質量を低下させる。3つの擬似件数での全結果、10口ごとの13記号、元のモデル確率、実的中数はCSVにある。券の入れ替わりだけを成果とせず、どの開催でDを拾い、H/A機会を失い、13的中が成立したかを併記した。',
      '## 判断',
      '5件中4Dは探索上の有望な仮説だが、同じ78試合で見つけた条件を同じ78試合で証明できない。λ由来の得点構造は既存P(D)と情報源が重複する。今の時点で正式なD購入根拠として使える条件はない。次は閾値・条件定義を凍結し、同一モデル版の後続試合でD/H/Aの校正と13的中寄与を評価する。独立した得点内容・欠場等の事前情報があれば、既存P(D)への追加力を別途検証する。',
    ])+'\n'
    (OUT/'report.md').write_text(report,encoding='utf-8')
    css='body{font:15px/1.6 system-ui;max-width:1300px;margin:24px auto;padding:0 18px;color:#20384e}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{border:1px solid #cdd7e0;padding:5px}th{background:#edf3f8}h1,h2{color:#153d60}'
    body='<h1>D首位ゲート廃止後の再検証</h1><p>5件中4Dは探索結果。次の未使用開催で検証が必要です。</p><p><strong>閾値に関する訂正：</strong>40/50/60%の3点だけから不安定と判断できません。<a href="../score_structure_phase35_threshold_audit/index.html">細かい境界・連続距離の追加監査</a>を参照してください。</p>'
    for title,frame in [('5試合の実績',rare[['round','match_no','home_team','away_team','actual_score','baseline_P_H','baseline_P_D','baseline_P_A','P_0_0','P_1_1']]),
                        ('条件別H/D/A',c),('同じP(D)帯',bands),('閾値感応度',t),('シャドー比較：擬似件数3',base),('擬似件数別の総括',aggregate),
                        ('10口の実券：擬似件数3',tickets[tickets.kappa==3])]:
        body+=f'<h2>{html.escape(title)}</h2>'+frame.to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}')
    body+='<p><a href="report.md">詳細</a> · <a href="allocation_matrix.csv">試合名付き配券表</a> · <a href="shadow_tickets.csv">全配券</a> · <a href="D_purchase_candidates.csv">D候補</a> · <a href="changed_tickets.csv">変更券</a> · <a href="condition_identification.csv">条件識別力</a> · <a href="training_evidence.csv">学習根拠</a></p>'
    (OUT/'index.html').write_text(f'<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 3.5 D分岐再評価</title><style>{css}</style><main>{body}</main></html>',encoding='utf-8')


if __name__=='__main__':
    result=run();render(*result)
    print(json.dumps({'output':str(OUT),'shadow_rows':len(result[5]),'rare_matches':int(result[0].rare.sum())},ensure_ascii=False))
