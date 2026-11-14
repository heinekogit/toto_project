"""Phase 3.5 research: score-structure branches under relative T/log-R changes."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure import score_grid, structure

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/reports/score_structure_phase35'
BASE = ROOT/'data/reports/score_structure_phase4a_stage1/match_diagnostics.csv'
STRUCT = ROOT/'data/reports/score_structure_phase12/match_structure_78.csv'
PROV = ROOT/'data/reports/score_structure_phase4a_audit/forecast_result_join.csv'
SYMS=np.array(['H','D','A'])


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(h,a,mix):
    mat,ret=score_grid(h,a)
    raw=np.array([np.tril(mat,-1).sum(),np.trace(mat),np.triu(mat,1).sum()])
    final=(1-mix)*raw+mix/3
    assert np.isclose(final.sum(),1,atol=1e-12)
    return mat,structure(mat),raw,final,ret


def scenarios(h,a):
    """Exploratory 5%/10% relative changes, with no empirical error frequency."""
    total=h+a; logratio=math.log(h/a)
    favored='H' if h>=a else 'A'
    yield '基準','基準',0.0,h,a,1.0,0.0
    for eps in (.05,.10):
        shift=math.log1p(eps)
        toward=0.0 if abs(logratio)<1e-12 else -math.copysign(min(shift,abs(logratio)),logratio)
        away=0.0 if abs(logratio)<1e-12 else math.copysign(shift,logratio)
        cases=[('総得点低下','総得点',1-eps,0.0),('総得点上昇','総得点',1+eps,0.0),
               ('H相対上昇','得点力比',1.0,shift),('A相対上昇','得点力比',1.0,-shift),
               ('総得点低下・拮抗','複合',1-eps,toward),
               ('総得点上昇・拮抗','複合',1+eps,toward),
               ('総得点低下・格差拡大','複合',1-eps,away),
               ('総得点上昇・格差拡大','複合',1+eps,away)]
        # Named single-side changes separate favored and underdog evaluations.
        for side in ('優勢側','劣勢側'):
            for direction in ('低下','上昇'):
                hh,aa=h,a
                factor=1-eps if direction=='低下' else 1+eps
                if (side=='優勢側' and favored=='H') or (side=='劣勢側' and favored=='A'):hh*=factor
                else:aa*=factor
                yield f'{side}{direction}','片側',eps,hh,aa,(hh+aa)/total,math.log(hh/aa)-logratio
        for name,family,t_scale,log_shift in cases:
            tt=total*t_scale; lr=logratio+log_shift
            share=1/(1+math.exp(-lr));hh=tt*share;aa=tt*(1-share)
            yield name,family,eps,hh,aa,t_scale,log_shift


def table(frame):
    cols=list(frame.columns)
    def fmt(v):
        if pd.isna(v):return ''
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def run():
    base=pd.read_csv(BASE); st=pd.read_csv(STRUCT);prov=pd.read_csv(PROV)
    rounds=set(base['round'])
    orig=prov[(prov.provenance_class=='A') & prov['round'].isin(rounds)]
    assert len(base)==len(st)==len(orig)==78 and len(rounds)==6
    assert orig.prematch_hash_verified.all() and orig.actual_join_verified.all()
    assert not any(x.duplicated(['round','match_no']).any() for x in [base,st,orig])
    source_files=[BASE,STRUCT,PROV]+[ROOT/f for f in orig.source_file.unique()]
    before={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    for f,g in orig.groupby('source_file'):
        assert sha(ROOT/f)==g.source_sha256.iloc[0]
    base=base.merge(st[['round','match_no','P_0_0','P_1_1','P_draw_2plus','P_home_1','P_away_1','P_margin_2plus']],on=['round','match_no'],validate='one_to_one')
    rows=[];diag=[]
    for r in base.itertuples(index=False):
        h0,a0,mix=float(r.lambda_home),float(r.lambda_away),float(r.uncertainty_mix)
        matrix,s0,raw0,final0,_=probabilities(h0,a0,mix)
        saved=np.array([r.final_P_H,r.final_P_D,r.final_P_A],float)
        assert np.allclose(final0,saved,atol=1e-10,rtol=0)
        actual='H' if r.actual_home>r.actual_away else 'A' if r.actual_home<r.actual_away else 'D'
        ref=SYMS[final0.argmax()]
        diag.append(dict(round=r.round,match_no=r.match_no,home_team=r.home_team,away_team=r.away_team,
                         model_signature=r.model_signature,actual_symbol=actual,actual_home=r.actual_home,actual_away=r.actual_away,
                         baseline_symbol=ref,lambda_home=h0,lambda_away=a0,T=h0+a0,log_R=math.log(h0/a0),
                         uncertainty_mix=mix,baseline_P_H=final0[0],baseline_P_D=final0[1],baseline_P_A=final0[2],
                         P_0_0=s0['P_0_0'],P_1_1=s0['P_1_1'],P_draw_2plus=s0['P_draw_2plus'],
                         P_home_1=s0['P_home_1'],P_away_1=s0['P_away_1'],P_margin_2plus=s0['P_margin_2plus']))
        for name,family,eps,h,a,T_factor,log_shift in scenarios(h0,a0):
            mat,ss,raw,fin,ret=probabilities(h,a,mix)
            rows.append(dict(round=r.round,match_no=r.match_no,home_team=r.home_team,away_team=r.away_team,
                scenario=name,family=family,epsilon=eps,lambda_home=h,lambda_away=a,T=h+a,
                T_factor=T_factor,log_R=math.log(h/a),log_R_shift=log_shift,retained_mass=ret,
                baseline_symbol=ref,scenario_top=SYMS[fin.argmax()],actual_symbol=actual,
                raw_P_H=raw[0],raw_P_D=raw[1],raw_P_A=raw[2],final_P_H=fin[0],final_P_D=fin[1],final_P_A=fin[2],
                P_0_0=ss['P_0_0'],P_1_1=ss['P_1_1'],P_2_2=ss['P_2_2'],P_3_3=ss['P_3_3'],P_draw_4plus=ss['P_draw_4plus'],
                P_home_1=ss['P_home_1'],P_away_1=ss['P_away_1'],P_home_2=ss['P_home_2'],P_away_2=ss['P_away_2'],
                P_home_3plus=ss['P_home_3plus'],P_away_3plus=ss['P_away_3plus'],P_margin_2plus=ss['P_margin_2plus'],
                delta_P_D=fin[1]-final0[1],delta_P_0_0=ss['P_0_0']-s0['P_0_0'],
                delta_P_1_1=ss['P_1_1']-s0['P_1_1'],delta_P_draw_2plus=ss['P_draw_2plus']-s0['P_draw_2plus']))
    state=pd.DataFrame(rows);d=pd.DataFrame(diag)
    assert len(state)==78*25 and not state.duplicated(['round','match_no','scenario','epsilon']).any()
    # Every baseline is unchanged, all 25 states use the same saved mix.
    baseline=state[state.scenario=='基準'].sort_values(['round','match_no'])
    dd=d.sort_values(['round','match_no'])
    assert np.allclose(baseline[['final_P_H','final_P_D','final_P_A']],dd[['baseline_P_H','baseline_P_D','baseline_P_A']],atol=1e-12)
    branches=[]
    for r in d.itertuples(index=False):
        sub=state[(state['round']==r.round)&(state.match_no==r.match_no)&(state.scenario!='基準')]
        bidx={'H':0,'D':1,'A':2}[r.baseline_symbol]
        basevals=np.array([r.baseline_P_H,r.baseline_P_D,r.baseline_P_A])
        for alt in SYMS:
            if alt==r.baseline_symbol:continue
            ai={'H':0,'D':1,'A':2}[alt]
            gap0=basevals[ai]-basevals[bidx]
            gaps=sub[f'final_P_{alt}'].to_numpy()-sub[f'final_P_{r.baseline_symbol}'].to_numpy()
            k=int(np.argmax(gaps));best=sub.iloc[k]
            top_rows=sub[sub.scenario_top==alt]
            branches.append(dict(round=r.round,match_no=r.match_no,home_team=r.home_team,away_team=r.away_team,
                baseline_symbol=r.baseline_symbol,alternative_symbol=alt,branch_direction=f'{r.baseline_symbol}→{alt}',
                actual_symbol=r.actual_symbol,actual_alternative=(r.actual_symbol==alt),
                condition=best.scenario,condition_family=best.family,exploratory_epsilon=best.epsilon,
                baseline_gap=float(gap0),max_scenario_gap=float(gaps[k]),gap_improvement=float(gaps[k]-gap0),
                scenario_alt_top_count=len(top_rows),alternative_became_top=bool(len(top_rows)),
                baseline_P_alt=basevals[ai],scenario_P_alt=best[f'final_P_{alt}'],
                baseline_P_D=r.baseline_P_D,scenario_P_D=best.final_P_D,
                baseline_P00=r.P_0_0,scenario_P00=best.P_0_0,
                baseline_P11=r.P_1_1,scenario_P11=best.P_1_1,
                baseline_P2plus=r.P_draw_2plus,scenario_P2plus=best.P_2_2+best.P_3_3+best.P_draw_4plus,
                prematch_condition_support='未検証：λ誤差の方向・頻度を示す試合前データなし',
                validation_status='探索段階',research_assessment='探索段階'))
    br=pd.DataFrame(branches)
    d_branch=br[br.alternative_symbol=='D'][['round','match_no','condition','gap_improvement','max_scenario_gap','alternative_became_top']].rename(
        columns={'condition':'D_branch_condition','gap_improvement':'D_gap_improvement',
                 'max_scenario_gap':'D_max_scenario_gap','alternative_became_top':'D_top_in_scenario'})
    h_branch=br[br.alternative_symbol!='D'][['round','match_no','alternative_symbol','condition','alternative_became_top']].rename(
        columns={'alternative_symbol':'HA_alternative','condition':'HA_branch_condition',
                 'alternative_became_top':'HA_alternative_top_in_scenario'})
    d=d.merge(d_branch,on=['round','match_no'],validate='one_to_one').merge(h_branch,on=['round','match_no'],validate='one_to_one')
    # Prespecified, scale-free condition combinations using within-round medians.
    for col in ['T','log_R','P_0_0','P_1_1','P_draw_2plus','P_home_1','P_away_1']:
        d[f'median_{col}']=d.groupby('round')[col].transform('median')
    d['low_T']=d['T']<=d.median_T
    d['near_ratio']=d.log_R.abs()<=d.groupby('round').log_R.transform(lambda x:x.abs().median())
    d['low_P00']=d.P_0_0<=d.median_P_0_0
    d['high_P11']=d.P_1_1>d.median_P_1_1
    d['high_P2plus']=d.P_draw_2plus>d.median_P_draw_2plus
    d['P_onegoal']=d.P_home_1+d.P_away_1
    d['high_onegoal']=d.P_onegoal>d.groupby('round').P_onegoal.transform('median')
    d['actual_D']=d.actual_symbol.eq('D')
    d['HA_reverse']=((d.baseline_symbol=='H')&(d.actual_symbol=='A'))|((d.baseline_symbol=='A')&(d.actual_symbol=='H'))
    conditions={
      '低T・得点力比接近':d.low_T&d.near_ratio,
      '高T・得点力比接近':(~d.low_T)&d.near_ratio,
      '片側優勢・拮抗方向でD首位':(~d.near_ratio)&d.D_top_in_scenario,
      '0-0低・1-1高':d.low_P00&d.high_P11,
      '0-0低・2-2以上高':d.low_P00&d.high_P2plus,
      '1点差高・拮抗方向でD首位':d.high_onegoal&d.D_top_in_scenario,
      '得点力比接近・1-1高':d.near_ratio&d.high_P11,
    }
    # This is a fixed diagnostic set, including unfavorable results; no condition is selected as a rule.
    cond_rows=[]
    for name,mask in conditions.items():
        yes=d[mask];no=d[~mask]
        cond_rows.append(dict(condition=name,matched=len(yes),actual_D=int(yes.actual_D.sum()),nonD=int((~yes.actual_D).sum()),
            actual_HA_reverse=int(yes.HA_reverse.sum()),D_recall=float(yes.actual_D.sum()/18),
            D_rate=float(yes.actual_D.mean()) if len(yes) else np.nan,
            nonmatched_D_rate=float(no.actual_D.mean()) if len(no) else np.nan,
            expected_D=float(yes.baseline_P_D.sum()),excess_D=float(yes.actual_D.sum()-yes.baseline_P_D.sum()),
            baseline_HDA_top_correct=int((yes.baseline_symbol==yes.actual_symbol).sum()),
            validation_status='探索的・同一78試合内、独立検証なし'))
    conditions_df=pd.DataFrame(cond_rows)
    # Predeclared PD strata show whether apparent effects merely track existing P(D).
    d['PD_band']=d.groupby('round').baseline_P_D.transform(lambda x:pd.qcut(x.rank(method='first'),3,labels=['低','中','高']))
    band_rows=[]
    for name,mask in conditions.items():
        for band in ['低','中','高']:
            sel=d.PD_band.eq(band)
            for flag in [True,False]:
                sub=d[sel & (mask==flag)]
                band_rows.append(dict(condition=name,PD_band=band,condition_met=flag,matches=len(sub),actual_D=int(sub.actual_D.sum()),
                                      mean_P_D=float(sub.baseline_P_D.mean()) if len(sub) else np.nan,
                                      D_rate=float(sub.actual_D.mean()) if len(sub) else np.nan))
    bands=pd.DataFrame(band_rows)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in [('relative_scenarios.csv',state),('diagnostics_78.csv',d),('branch_candidates.csv',br),
                       ('conditional_validation.csv',conditions_df),('PD_band_comparison.csv',bands)]:
        frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    after={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    assert before==after
    audit=dict(matches=78,actual_D=18,states_per_match=25,branch_rows=len(br),condition_families=len(conditions),
        source_hashes=before,originals_unchanged=True,results_used_to_set_scenarios_or_conditions=False,
        exploratory_relative_widths=[.05,.10],widths_are_empirical_errors=False,formal_model_or_buyplan_changed=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return d,state,br,conditions_df,bands,audit


def render(d,state,br,conditions,bands,audit):
    draw=d[d.actual_D]
    reverse=d[d.HA_reverse]
    Dbr=br[br.alternative_symbol=='D']
    actual_D_branch=Dbr.merge(draw[['round','match_no']],on=['round','match_no'])
    alt_top=int(actual_D_branch.alternative_became_top.sum())
    nonD_Dbr=Dbr.merge(d[~d.actual_D][['round','match_no']],on=['round','match_no'])
    reverse_br=br[br.actual_alternative].merge(reverse[['round','match_no']],on=['round','match_no'])
    reverse_support=int(reverse_br.alternative_became_top.sum())
    delta_summary=state[(state.scenario!='基準')&(state.epsilon==.10)].groupby('scenario').agg(
        mean_D_delta=('delta_P_D','mean'),D_up=('delta_P_D',lambda x:int((x>0).sum())),
        top_flips=('scenario_top',lambda x:int((x!=state.loc[x.index,'baseline_symbol']).sum()))).reset_index()
    report='\n\n'.join([
      '# Phase 3.5：総得点水準・得点力比からの条件付きH/D/A分岐',
      '研究用の数学的分岐診断。Next Model、正式BuyPlan、購入フローは変更していない。試合前保存原本Aの6開催78試合とPhase 1〜3の11×11得点行列計算を再利用。旧予想システムは使用しない。',
      '## 変動の設計',
      'T=λH+λA、L=log(λH/λA)を軸にする。Tだけを同率±5/10%変更、Tを固定してLを±log(1+5/10%)変更、優勢側・劣勢側だけを相対変更、TとLを複合変更。Tは得点規模、Lは優劣関係を分離する。各試合で基準を含む25状態。すべて同一11×11行列関数で再計算し、保存済みmixを `(1-mix)*raw+mix/3` として適用。変動幅は探索用で、実際のλ誤差分布ではない。',
      'Phase 3の±0.1/0.2点は絶対変動だったが、今回は元の得点規模に対して相対変動を与える。5/10%という選択に実証的根拠はない。得点構造は各シナリオの同点帯、H/A別1点差、2点差以上としてCSVに保持。',
      '## 実D・非D、H/A逆転との関係',
      f'実D18試合のうち基準予測がH/Aだった試合は{int((draw.baseline_symbol!="D").sum())}。そのうち、探索状態のいずれかでDが首位になる試合は{alt_top}。非D60試合でもD首位へ切り替わるのは{int(nonD_Dbr.alternative_became_top.sum())}。Dと基準首位との確率差は全78試合で縮められたが、最大改善の中央値は実D{actual_D_branch.gap_improvement.median():.4f}、非D{nonD_Dbr.gap_improvement.median():.4f}と近い。全試合で最大改善条件が「総得点低下・拮抗」だったため、これを識別規則にしても区別できない。',
      f'H/A逆方向の実結果は{len(reverse)}試合。探索状態で実際の反対側記号が首位になるのは{reverse_support}試合。ただし、H/A逆転でない試合にも反対側が首位になるものが{int(br[(br.alternative_symbol!="D") & (~br.actual_alternative)].alternative_became_top.sum())}試合ある。結果を知ってから状態を選ぶことは禁止し、全状態を全試合に適用した。',
      '### 相対変動10%の方向別診断（全78試合）',
      table(delta_summary),
      '### 条件を組み合わせた探索比較',
      '条件は結果を見る前に、各開催内のT、|L|、同点帯、1点差の中央値とシナリオ首位切替だけで定義した。該当数・実D数・非D数・既存P(D)の期待D数をすべて表示する。`PD_band_comparison.csv` は各開催内P(D)の3帯で同じ条件を比較する。少数セルやD18の標本では偶然差を排除できない。',
      table(conditions),
      '例：`0-0低・1-1高`は5試合中4試合がDで、今治－八戸、神戸－町田、G大阪－京都、福岡－長崎が該当した。一方、神戸－名古屋は同条件でもH（3-1）だった。実D4件という見かけの強さは、該当5件という小ささと、試合結果を見た同一標本の探索である点を踏まえる必要がある。次の未使用開催で同じ中央値定義を固定して検証する候補にとどめる。',
      '`高T・得点力比接近`は18試合中D6・非D12。`低T・得点力比接近`は24試合中D5・非D19。どちらも多数の非Dを含み、単独でD購入を決める条件ではない。',
      '### 保存P(D)の同程度帯での比較',
      '各開催内P(D)を低・中・高の3帯に分けた。条件に合う試合と合わない試合の平均P(D)と実D率を並べる。帯内に2〜3試合しかないセルは偶然変動が大きく、予測力を示さない。',
      table(bands[bands.condition.isin(['高T・得点力比接近','0-0低・1-1高','0-0低・2-2以上高'])]),
      '## 分岐候補の評価',
      '各試合について基準記号以外の2記号を保持。`branch_candidates.csv` には、代替記号との確率差を最も縮めたシナリオ、首位へ切り替わった状態数、0-0/1-1/2-2以上の変化、実結果、研究用評価を記録した。最大の有利状態を記録するのは数学的な到達条件の説明であり、現実の発生確率や購入価値の根拠ではない。独立の予測情報がない現段階では**全候補を探索段階**とし、採用推奨は付けない。',
      '## 結論と限界',
      'TとLの分離により、同じ「D確率上昇」でも低得点化・優劣接近・片側評価変更のどれに由来するか追える。しかし、TとLは元のλ2値の再表現であり、それだけではNext Modelにない情報を追加しない。上記条件と実D/H/A逆転の関係は同じ78試合での探索結果であり、分岐の予測有効性は未検証。旧v3.0の固定ダブル、D本数、H/D/A枚数固定は導入しない。',
      '次の検証には、同一モデル版の後続試合で保存λの誤差方向を統計的に評価できる履歴、試合前に観測できる原因変数、時系列で分離した分岐条件の検証が必要。実得点−λを真のλ誤差としない。試合間の同時分岐も独立を仮定せず、共通要因を別途検証する。',
      '## 13試合10口への接続',
      '別紙 `buyplan_connection.md` を参照。今回、確率上位10口の再計算や購入券生成はしていない。',
    ])+'\n'
    (OUT/'report.md').write_text(report,encoding='utf-8')
    design='''# Phase 3.5 → BuyPlan v4.0 接続設計\n\n1. 試合前の保存final H/D/Aとλ、mix、検証済みの試合前条件を読む。\n2. 条件付き分岐は「代替記号、成立条件、推定した条件発生確率、検証期間、誤差幅」を持つ。今回のCSVは成立条件の数学的形だけで、発生確率が未検証のため購入候補への自動採用を禁止する。\n3. 検証後、条件が共有される試合の同時変動を扱うシナリオを作る。試合ごと独立に最良状態だけをつなぎ合わせない。\n4. 各13試合の結果組み合わせについて、基準分布の13的中質量と検証済みシナリオ群での質量を保存し、どの分岐を拾うか追跡する。\n5. 10口は同一予算・重複なしで一括比較。目的は少なくとも1口が13的中する確率。確率上位10口を固定分布の基準にし、条件の検証外評価での改善がなければ変更しない。D券数・ダブル数・候補番号位置は固定しない。\n6. 評価は時系列の未使用試合で、H/D/Aの校正、13的中確率、実際の13的中、下振れを分けて報告する。6開催78試合では全的中実績が希少で結論できない。\n'''
    (OUT/'buyplan_connection.md').write_text(design,encoding='utf-8')
    # HTML uses native pandas tables; all numbers are from the same saved frames.
    css='body{font:15px/1.6 system-ui;max-width:1350px;margin:24px auto;padding:0 18px;color:#20384e}table{border-collapse:collapse;display:block;overflow:auto;font-size:12px;white-space:nowrap}td,th{border:1px solid #cdd7e0;padding:5px}th{background:#edf3f8}h1,h2{color:#153d60}a{color:#17629c}'
    sections=[('<h1>Phase 3.5：条件付きH/D/A分岐</h1>',None),
      ('<p>探索用の相対λ変動。試合前保存A 78試合。5/10%は現実の予測誤差幅ではありません。</p>',None),
      ('<h2>研究上の結論</h2>',None),
      (f'<p>実D18試合のうち、仮定シナリオでDが首位へ変わるのは{alt_top}試合。非D60試合にも{int(nonD_Dbr.alternative_became_top.sum())}試合あるため、変動可能性だけでは識別できません。実際のH/A逆転{len(reverse)}試合のうち、仮定状態で反対側が首位になるのは{reverse_support}試合。現実のλ変動方向は未検証です。</p>',None),
      ('<h2>変動方向別</h2>',delta_summary),('<h2>条件別比較</h2>',conditions),
      ('<p>0-0低・1-1高は5試合中D4、非D1。ただし小標本の探索結果で、後続試合で未検証です。高T・得点力比接近はD6、非D12、低T・得点力比接近はD5、非D19でした。</p>',None),
      ('<h2>同程度のP(D)帯での比較</h2>',bands[bands.condition.isin(['高T・得点力比接近','0-0低・1-1高','0-0低・2-2以上高'])]),
      ('<h2>実D18試合の分岐</h2>',actual_D_branch[['round','match_no','home_team','away_team','baseline_symbol','alternative_symbol','condition','exploratory_epsilon','baseline_gap','max_scenario_gap','alternative_became_top','actual_symbol']]),
      ('<h2>実H/A逆転の分岐</h2>',reverse_br[['round','match_no','home_team','away_team','baseline_symbol','alternative_symbol','condition','alternative_became_top','actual_symbol']]),
      ('<p><a href="report.md">詳細レポート</a> · <a href="diagnostics_78.csv">78試合CSV</a> · <a href="branch_candidates.csv">分岐候補CSV</a> · <a href="conditional_validation.csv">条件別CSV</a> · <a href="buyplan_connection.md">10口への接続設計</a></p>',None)]
    body=''.join(t+(df.to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}') if df is not None else '') for t,df in sections)
    (OUT/'index.html').write_text(f'<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 3.5</title><style>{css}</style><main>{body}</main></html>',encoding='utf-8')
    return alt_top,int(nonD_Dbr.alternative_became_top.sum()),len(reverse),reverse_support


if __name__=='__main__':
    result=run();numbers=render(*result)
    print(json.dumps({'actual_D_to_Dtop':numbers[0],'nonD_to_Dtop':numbers[1],
                      'HA_reverse':numbers[2],'HA_reverse_to_top':numbers[3],
                      'output':str(OUT)},ensure_ascii=False))
