"""v4.2 score-relation audit and ten-ticket output; research only."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from analyze_score_structure import score_grid, structure
from analyze_score_structure_phase4b_stage1 import decode, top_ten, universe
from condition_s_registry import classify
from common import sha256

ROOT=Path(__file__).resolve().parents[2]
HISTORY_STRUCTURE=ROOT/'data/reports/score_structure_phase12/match_structure_78.csv'
HISTORY_RESULTS=ROOT/'data/reports/score_structure_phase35/diagnostics_78.csv'
TRAIN_ROUNDS=('toto1660','toto1663','toto1664')
HOLDOUT_ROUNDS=('toto1665','toto1668','toto1670')
FILES=('index.html','match_structure_13.csv','historic_relation_78.csv','validation.csv','tickets_10.csv','audit.json','report.md','D_branch_candidates_13.csv','similar_structure_13.csv')


def metrics(y,p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12)
    y=np.asarray(y,float)
    return dict(n=len(y),actual_D=int(y.sum()),expected_D=float(p.sum()),
                d_log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
                d_brier=float(np.mean((y-p)**2)))


def historical():
    s=pd.read_csv(HISTORY_STRUCTURE,encoding='utf-8-sig')
    d=pd.read_csv(HISTORY_RESULTS,encoding='utf-8-sig')
    x=s.merge(d[['round','match_no','actual_D','actual_symbol','baseline_P_H','baseline_P_D','baseline_P_A']],
              on=['round','match_no'],validate='one_to_one')
    if len(x)!=78 or set(x['round'])!=set(TRAIN_ROUNDS+HOLDOUT_ROUNDS):
        raise ValueError('expected exactly the 78 prematch originals')
    x['leader']=np.where(x.baseline_P_H>=x.baseline_P_A,'H','A')
    x['leader_one_goal_mass']=np.where(x.leader.eq('H'),x.P_home_1,x.P_away_1)
    x['leader_two_plus_mass']=np.where(x.leader.eq('H'),x.P_home_2+x.P_home_3plus,x.P_away_2+x.P_away_3plus)
    x['leader_one_goal_share']=x.leader_one_goal_mass/(x.leader_one_goal_mass+x.leader_two_plus_mass)
    x['D_to_one_goal_ratio']=x.P_draw/(x.P_home_1+x.P_away_1)
    x['leader_to_D_gap']=np.maximum(x.baseline_P_H,x.baseline_P_A)-x.baseline_P_D
    flags=[]
    for _,g in x.groupby('round',sort=False):
        values,_,_=classify(g.P_0_0.to_numpy(float),g.P_1_1.to_numpy(float))
        flags.extend(zip(g.index,values))
    x['condition_S']=False
    for idx,value in flags:x.at[idx,'condition_S']=bool(value)
    return x


def validate(x):
    train=x[x['round'].isin(TRAIN_ROUNDS)]
    test=x[x['round'].isin(HOLDOUT_ROUNDS)]
    rows=[]
    def predicted(g,coeff,feature=None,center=0.,scale=1.):
        z=logit(g.baseline_P_D.to_numpy(float))
        if len(coeff):z=z+coeff[0]
        if len(coeff)>1:z=z+coeff[1]*(g[feature].to_numpy(float)-center)/scale
        return expit(z)
    models=[('saved_P_D',0,None),('global_intercept_only',1,None),
            ('leader_one_goal_share',2,'leader_one_goal_share'),
            ('D_to_one_goal_ratio',2,'D_to_one_goal_ratio'),
            ('leader_to_D_gap',2,'leader_to_D_gap')]
    for label,nparam,feature in models:
        center=float(train[feature].mean()) if feature else 0.
        scale=float(train[feature].std()) if feature else 1.
        if scale<=0:raise ValueError('no feature variation')
        if nparam:
            def objective(beta):
                p=np.clip(predicted(train,beta,feature,center,scale),1e-12,1-1e-12)
                y=train.actual_D.to_numpy(int)
                return float(-np.sum(y*np.log(p)+(1-y)*np.log(1-p))+.5*np.sum(beta*beta))
            fit=minimize(objective,np.zeros(nparam),method='BFGS')
            if not fit.success and np.linalg.norm(fit.jac)>1e-4:
                raise RuntimeError(f'fit failed: {fit.message}')
            beta=fit.x
        else:beta=np.array([])
        for segment,g in [('train',train),('later_held_out_rounds',test)]:
            rows.append(dict(model=label,segment=segment,**metrics(g.actual_D,predicted(g,beta,feature,center,scale)),
                             intercept=float(beta[0]) if len(beta) else 0.,
                             feature_coefficient=float(beta[1]) if len(beta)>1 else 0.,
                             feature_train_mean=center,feature_train_sd=scale))
    return pd.DataFrame(rows)


def current(f):
    f=f.sort_values('match_no').reset_index(drop=True)
    if len(f)!=13 or f.match_no.tolist()!=list(range(1,14)):
        raise ValueError('13 ordered matches required')
    flags,_,_=classify(f.next_prob_0_0.to_numpy(float),f.next_prob_1_1.to_numpy(float))
    rows=[]
    for i,r in f.iterrows():
        grid,_=score_grid(float(r.final_lambda_home),float(r.final_lambda_away))
        s=structure(grid)
        if not np.isclose(s['P_0_0'],r.next_prob_0_0,atol=1e-9) or not np.isclose(s['P_1_1'],r.next_prob_1_1,atol=1e-9):
            raise ValueError('saved score probabilities mismatch saved lambda')
        ph,pd_,pa=(float(r.next_prob_home),float(r.next_prob_draw),float(r.next_prob_away))
        leader='H' if ph>=pa else 'A'
        raw_h=s['P_home_1']+s['P_home_2']+s['P_home_3plus']
        raw_a=s['P_away_1']+s['P_away_2']+s['P_away_3plus']
        home_goals,away_goals=np.indices(grid.shape)
        margin=home_goals-away_goals
        max_other_margin=max(float(grid[margin==difference].sum()) for difference in range(-10,11) if difference!=0)
        leading_one=s['P_home_1'] if leader=='H' else s['P_away_1']
        leading_two=(s['P_home_2']+s['P_home_3plus']) if leader=='H' else (s['P_away_2']+s['P_away_3plus'])
        mode=np.unravel_index(int(np.argmax(grid)),grid.shape)
        rows.append(dict(match_no=i+1,home_team=r.home_team,away_team=r.away_team,
                         lambda_home=float(r.final_lambda_home),lambda_away=float(r.final_lambda_away),
                         final_P_H=ph,final_P_D=pd_,final_P_A=pa,first_choice='HDA'[int(np.argmax([ph,pd_,pa]))],
                         raw_score_P_H=raw_h,raw_score_P_D=s['P_draw'],raw_score_P_A=raw_a,
                         raw_draw_vs_home_one_gap=s['P_draw']-s['P_home_1'],
                         raw_draw_vs_away_one_gap=s['P_draw']-s['P_away_1'],
                         draw_largest_goal_difference=bool(s['P_draw']>=max_other_margin),
                         max_other_goal_difference_mass=max_other_margin,
                         P_0_0=s['P_0_0'],P_1_1=s['P_1_1'],P_2_2=s['P_2_2'],
                         modal_score=f'{mode[0]}-{mode[1]}',modal_score_probability=float(grid[mode]),
                         home_one_goal=s['P_home_1'],away_one_goal=s['P_away_1'],
                         home_two_plus=s['P_home_2']+s['P_home_3plus'],
                         away_two_plus=s['P_away_2']+s['P_away_3plus'],
                         D_to_one_goal_ratio=s['P_draw']/(s['P_home_1']+s['P_away_1']),
                         leader=leader,leader_one_goal_share=leading_one/(leading_one+leading_two),
                         leader_to_D_gap=max(ph,pa)-pd_,condition_S=bool(flags[i]),
                         D_probability_changed=False,reason='独立した予測改善を確認できず、保存final P(D)を維持'))
    return pd.DataFrame(rows)


def tickets(f):
    p=f[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
    mass=universe(p)
    codes=top_ten(mass)
    matrix=decode(codes)
    out=[]
    for n,(code,row) in enumerate(zip(codes,matrix),1):
        symbols=''.join('HDA'[v] for v in row)
        out.append(dict(candidate=n,symbols=symbols,D_count=symbols.count('D'),P_ticket_probability=float(mass[code]),
                        **{f'match_{i+1:02d}':symbols[i] for i in range(13)}))
    return pd.DataFrame(out)


def draw_candidates(f):
    """Retain a D option for every match; these are candidates, not forced purchases."""
    p=f[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
    mass=universe(p)
    codes=np.arange(len(mass),dtype=np.int64)
    powers=3**np.arange(12,-1,-1,dtype=np.int64)
    rows=[]
    for i,r in f.iterrows():
        eligible=np.flatnonzero((codes//powers[i])%3==1)
        best=int(eligible[np.argmax(mass[eligible])])
        symbols=''.join('HDA'[v] for v in decode([best])[0])
        expected=10*float(r.next_prob_draw)
        rows.append(dict(match_no=i+1,home_team=r.home_team,away_team=r.away_team,
                         P_D=float(r.next_prob_draw),proportional_D_cells=expected,
                         D_count_floor=int(np.floor(expected)),D_count_ceil=int(np.ceil(expected)),
                         best_D_ticket=symbols,best_D_ticket_P=float(mass[best]),
                         status='候補のみ・10口への採用強制なし'))
    return pd.DataFrame(rows)


def analogs(current_frame,historical_frame):
    """Fixed eight nearest descriptive analogs; never used for correction."""
    fields=[('final_P_D','baseline_P_D'),('leader_one_goal_share','leader_one_goal_share'),
            ('D_to_one_goal_ratio','D_to_one_goal_ratio')]
    scales=[max(float(historical_frame[h].std()),1e-9) for _,h in fields]
    rows=[]
    for _,r in current_frame.iterrows():
        distance=np.zeros(len(historical_frame))
        for (c,h),scale in zip(fields,scales):
            distance+=((historical_frame[h].to_numpy(float)-float(r[c]))/scale)**2
        chosen=historical_frame.iloc[np.argsort(distance,kind='stable')[:8]]
        rows.append(dict(match_no=int(r.match_no),home_team=r.home_team,away_team=r.away_team,
                         analog_count=len(chosen),analog_expected_D=float(chosen.baseline_P_D.sum()),
                         analog_actual_D=int(chosen.actual_D.sum()),
                         analog_rounds=';'.join(chosen['round'].astype(str)),
                         analog_match_numbers=';'.join(chosen.match_no.astype(str)),
                         warning='探索用類似例・モデル版混在・独立検証ではない'))
    return pd.DataFrame(rows)


def run(forecast: Path,outdir: Path,round_id: str):
    f=pd.read_csv(forecast,encoding='utf-8-sig',low_memory=False).sort_values('match_no').reset_index(drop=True)
    h=historical();v=validate(h);c=current(f);t=tickets(f);branches=draw_candidates(f);similar=analogs(c,h)
    outdir.mkdir(parents=True,exist_ok=True)
    c.to_csv(outdir/FILES[1],index=False,encoding='utf-8-sig')
    h.to_csv(outdir/FILES[2],index=False,encoding='utf-8-sig')
    v.to_csv(outdir/FILES[3],index=False,encoding='utf-8-sig')
    t.to_csv(outdir/FILES[4],index=False,encoding='utf-8-sig')
    branches.to_csv(outdir/FILES[7],index=False,encoding='utf-8-sig')
    similar.to_csv(outdir/FILES[8],index=False,encoding='utf-8-sig')
    audit=dict(round=round_id,forecast_sha256=sha256(forecast),history_structure_sha256=sha256(HISTORY_STRUCTURE),
               history_results_sha256=sha256(HISTORY_RESULTS),training_rounds=TRAIN_ROUNDS,held_out_rounds=HOLDOUT_ROUNDS,
               historical_model_versions_differ=True,condition_S_discovered_in_same_78=True,
               correction_applied=False,reason='structural feature tests are tiny, version-confounded and not prospectively validated; no reliable correction magnitude',
               tickets_source='unaltered final H/D/A probabilities',p13=float(t.P_ticket_probability.sum()),
               D_tickets=int((t.D_count>0).sum()),D_branch_candidates_retained=len(branches))
    (outdir/FILES[5]).write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    display=c[['match_no','home_team','away_team','lambda_home','lambda_away','final_P_H','final_P_D','final_P_A','first_choice',
               'modal_score','modal_score_probability','raw_score_P_H','raw_score_P_D','raw_score_P_A',
               'draw_largest_goal_difference','max_other_goal_difference_mass',
               'raw_draw_vs_home_one_gap','raw_draw_vs_away_one_gap',
               'home_one_goal','away_one_goal',
               'home_two_plus','away_two_plus','leader_one_goal_share','leader_to_D_gap','condition_S','D_probability_changed']].copy()
    for col in display.select_dtypes('number').columns:
        if col!='match_no':display[col]=display[col].round(4)
    held=v[v.segment.eq('later_held_out_rounds')]
    note=('予測時点の原本Aは6開催78試合。前半39試合で学習し後半39試合で評価した。'
          '前半・後半はモデル系統が異なり、条件Sも78試合から発見したため独立検証とは扱えない。'
          '第一候補の1点差比率は後半のD Log Loss・Brierを切片のみの補正より改善しなかった。D対1点差比率の差はごく小さい。'
          '元P(D)を変更せず、13試合10口は固定final分布の上位10組となる。'
          'この結果はDを排除すべきという実証ではなく、構造別の補正値をまだ確定できないことを示す。')
    report=f'# {round_id} v4.2 得点差分布とD評価\n\n{note}\n\n2025年などには実スコア履歴があるが、対応する試合前Next Model λ・確率原本Aがないため、構造別の予測誤差学習には混ぜない。17状態の仮想λ変動は一切使わない。\n\nDを実際に配置した10口の比較は [portfolio_comparison/index.html](portfolio_comparison/index.html) に分離する。補正なしでDを組む場合の代償を示すものであり、予測改善の証明ではない。\n\n## 後続39試合の評価\n\n```csv\n{held.to_csv(index=False)}```\n\n## 13試合の得点構造\n\n```csv\n{display.to_csv(index=False)}```\n\n## 類似構造の過去例（記述のみ）\n\n```csv\n{similar.to_csv(index=False)}```\n\n## D分岐候補（採用強制なし）\n\n```csv\n{branches.to_csv(index=False)}```\n\n## 配券\n\n```csv\n{t.to_csv(index=False)}```\n\n'
    (outdir/FILES[6]).write_text(report,encoding='utf-8')
    head='<tr><th>No.</th><th>対戦</th>'+''.join(f'<th>{i:02d}</th>' for i in range(1,11))+'</tr>'
    body=[]
    for _,r in c.iterrows():
        n=int(r.match_no);cells=[str(n),f'{r.home_team}－{r.away_team}']+[str(t.iloc[j][f'match_{n:02d}']) for j in range(10)]
        body.append('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in cells)+'</tr>')
    doc=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>v4.2得点構造・配券</title>"
         "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.note{background:#fff0d3;padding:12px}</style>"
         f'<h1>{html.escape(round_id)} v4.2 得点構造と10口</h1><p class="note">{note}</p>'
         '<h2>13試合の得点構造</h2><div class="scroll">'+display.to_html(index=False,escape=True)+'</div>'
         '<h2>後続39試合の検証</h2>'+held.to_html(index=False,escape=True)
         +'<h2>類似得点構造の過去例</h2><p>P(D)、第一候補の1点差比率、D対1点差比率で近い8試合。実D数は記述のみで、補正率には使いません。モデル版も混在します。</p><div class="scroll">'+similar.to_html(index=False,escape=True)+'</div>'
         +'<h2>全13試合のD分岐候補</h2><p>P(D)×10の下側・上側の整数枚数を候補生成の出発点として記録。各試合にDを含む最も強い券も保存します。これは購入枚数の固定ではありません。</p><div class="scroll">'+branches.to_html(index=False,escape=True)+'</div>'
         +f'<h2>10口：元Pでの13的中モデル確率 {audit["p13"]*100:.9f}%・D入り{audit["D_tickets"]}口</h2>'
         +'<div class="scroll"><table>'+head+''.join(body)+'</table></div>'
         f'<p><a href="portfolio_comparison/index.html">Dを実際に組んだ三方式の比較</a>／<a href="{FILES[1]}">得点構造CSV</a>／<a href="{FILES[2]}">78試合CSV</a>／<a href="{FILES[3]}">検証CSV</a>／<a href="{FILES[8]}">類似例CSV</a>／<a href="{FILES[7]}">D分岐候補CSV</a>／<a href="{FILES[4]}">10口CSV</a>／<a href="{FILES[6]}">詳細</a></p></html>')
    (outdir/FILES[0]).write_text(doc,encoding='utf-8')
    return audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--forecast',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--round',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.forecast,args.outdir,args.round),ensure_ascii=False))


if __name__=='__main__':main()
