"""Hypothetical probability-sensitivity audit of fixed, archived 10-ticket sets."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,sha256
from forecast_registry import selected_matches,DEFAULT_ROOT
from analyze_score_structure_phase4b_stage1 import OUT as STAGE1, ROUNDS, encode, SYMBOLS

OUT=ROOT/'data/reports/score_structure_phase4b_stage2'
SEED=20261111
METHOD_OPT='数学最適10口_final'
METHOD_SHADOW='保存Shadow C'
METHOD_OFFICIAL_PREFIX='保存正式版 '
STRENGTHS=(.01,.03,.05)
TRIALS=100


def shift(q,match_no,recipient,amount):
    """Transfer probability points from the two other symbols, proportional to their mass."""
    q=q.copy();p=q[match_no];other=[j for j in range(3) if j!=recipient]
    amount=max(-float(p[recipient]),min(float(amount),float(p[other].sum())))
    q[match_no,recipient]+=amount
    q[match_no,other]-=amount*p[other]/p[other].sum()
    if np.any(q< -1e-12) or not np.allclose(q.sum(axis=1),1,atol=1e-12):raise AssertionError('invalid perturbed probabilities')
    return np.clip(q,0,1)


def scenario(base,kind,delta,rng=None,subset=None):
    q=base.copy()
    all_matches=range(13) if subset is None else subset
    if kind in ('D上昇','D低下','最高確率過信','最高確率過小評価','H一斉上昇'):
        for i in all_matches:
            if kind.startswith('D'):symbol=1;amount=delta if kind=='D上昇' else -delta
            elif kind=='H一斉上昇':symbol=0;amount=delta
            else:symbol=int(np.argmax(base[i]));amount=-delta if kind=='最高確率過信' else delta
            q=shift(q,i,symbol,amount)
    elif kind in ('独立ランダム','共通方向ランダム'):
        if rng is None:raise ValueError('random generator required')
        shared=int(rng.integers(0,3)) if kind=='共通方向ランダム' else None
        for i in all_matches:
            symbol=shared if shared is not None else int(rng.integers(0,3))
            amount=delta*(1 if rng.random()<.5 else -1) if kind=='独立ランダム' else delta
            q=shift(q,i,symbol,amount)
    else:raise ValueError(kind)
    return q


def portfolio_probability(q,tickets):
    return float(np.prod(q[np.arange(13)[None,:],tickets],axis=1).sum())


def method_data(tickets,rid):
    d=tickets[(tickets['round']==rid)&(tickets.evaluation=='final')]
    plans={}
    for name,g in d.groupby('method',sort=False):
        arr=np.array([[{'H':0,'D':1,'A':2}[ch] for ch in seq] for seq in g.symbols],dtype=np.uint8)
        if arr.shape!=(10,13) or len(set(encode(arr)))!=10:raise ValueError(f'{rid} {name}: not ten distinct tickets')
        plans[name]=arr
    return plans


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=[STAGE1/'ticket_probabilities.csv',STAGE1/'portfolio_comparison.csv',STAGE1/'audit.json',ROOT/'BUYPLAN_V4_0.md',
            DEFAULT_ROOT/'forecasts.jsonl',DEFAULT_ROOT/'results.jsonl']
    hashes={str(p):sha256(p) for p in source}
    stage1_audit=json.loads((STAGE1/'audit.json').read_text())
    if stage1_audit['matches']!=78 or stage1_audit['old_v30_used']:raise ValueError('bad stage 1 provenance')
    reg=selected_matches(DEFAULT_ROOT)
    if len(reg)!=78 or not reg.provenance_class.eq('A').all():raise ValueError('A originals only')
    tickets=pd.read_csv(STAGE1/'ticket_probabilities.csv')
    structure=[];adoption=[];scenario_rows=[];comparisons=[];sensitivity=[];reversals=[];curves=[]
    for rid in ROUNDS:
        g=reg[reg.round_id.eq(rid)].sort_values('match_no')
        p=g[['final_prob_home','final_prob_draw','final_prob_away']].to_numpy(float)
        if len(g)!=13 or not np.allclose(p.sum(axis=1),1):raise ValueError(rid)
        methods=method_data(tickets,rid)
        if METHOD_OPT not in methods or not any(x.startswith(METHOD_OFFICIAL_PREFIX) for x in methods):raise ValueError('missing method')
        base_val={name:portfolio_probability(p,arr) for name,arr in methods.items()}
        actual_draws=int(np.sum(g.actual_home_score.to_numpy()==g.actual_away_score.to_numpy()))
        for name,arr in methods.items():
            counts=np.bincount(arr.ravel(),minlength=3)
            probs=np.prod(p[np.arange(13)[None,:],arr],axis=1)
            structure.append(dict(round=rid,method=name,actual_D=actual_draws,H_cells=int(counts[0]),D_cells=int(counts[1]),
                A_cells=int(counts[2]),D0_tickets=int(np.sum((arr==1).sum(axis=1)==0)),
                D_counts=','.join(map(str,(arr==1).sum(axis=1))),
                min_ticket_probability=float(probs.min()),max_ticket_probability=float(probs.max()),
                median_ticket_probability=float(np.median(probs)),total_probability=base_val[name],
                low_3_share=float(np.sort(probs)[:3].sum()/probs.sum())))
            for i,r in enumerate(g.itertuples()):
                counts=np.bincount(arr[:,i],minlength=3)
                adoption.append(dict(round=rid,method=name,match_no=i+1,home_team=r.home_team,away_team=r.away_team,
                    H=int(counts[0]),D=int(counts[1]),A=int(counts[2]),
                    unused=''.join(SYMBOLS[j] for j in range(3) if counts[j]==0),
                    P_H=p[i,0],P_D=p[i,1],P_A=p[i,2]))
        def record(kind,delta,trial,q,subset_desc='全13試合'):
            sid=f'{rid}|{kind}|{delta:.3f}|{trial:03d}'
            scenario_rows.append(dict(scenario_id=sid,round=rid,kind=kind,delta_pp=delta*100,trial=trial,
                subset=subset_desc,assumption='仮定した確率変動。実測誤差の推定ではない',
                max_cell_shift_pp=float(100*np.max(abs(q-p))),
                mean_cell_shift_pp=float(100*np.mean(abs(q-p))),
                probabilities_json=json.dumps(q.tolist(),ensure_ascii=False,separators=(',',':'))))
            values={name:portfolio_probability(q,arr) for name,arr in methods.items()}
            for name,val in values.items():
                comparisons.append(dict(scenario_id=sid,round=rid,kind=kind,delta_pp=delta*100,trial=trial,
                    method=name,P_probability=base_val[name],Q_probability=val,P_percent=100*base_val[name],Q_percent=100*val,
                    change_percent=100*(val/base_val[name]-1),
                    ratio_to_fixed_optimum=val/values[METHOD_OPT],beats_fixed_optimum=bool(val>values[METHOD_OPT]+1e-18)))
        record('基準',0,0,p)
        # Select three high-confidence fixtures by forecast only; no outcome information.
        subset=np.argsort(-p.max(axis=1),kind='stable')[:3]
        subset_desc='予測最大確率上位3試合:'+','.join(map(str,subset+1))
        for delta in STRENGTHS:
            for kind in ('D上昇','D低下','最高確率過信','最高確率過小評価','H一斉上昇'):
                record(kind,delta,0,scenario(p,kind,delta))
            record('高信頼3試合だけD上昇',delta,0,scenario(p,'D上昇',delta,subset=subset),subset_desc)
            record('高信頼3試合だけ最高確率過信',delta,0,scenario(p,'最高確率過信',delta,subset=subset),subset_desc)
            rng=np.random.default_rng(SEED+int(rid[-4:])*100+int(delta*1000))
            for trial in range(1,TRIALS+1):
                for kind in ('独立ランダム','共通方向ランダム'):
                    record(kind,delta,trial,scenario(p,kind,delta,rng))
        if METHOD_SHADOW in methods:
            for kind in ('D上昇','最高確率過信'):
                for step in range(0,201):
                    delta=step*.001
                    q=scenario(p,kind,delta)
                    curves.append(dict(round=rid,kind=kind,delta_pp=delta*100,
                        fixed_optimum_percent=100*portfolio_probability(q,methods[METHOD_OPT]),
                        shadow_c_percent=100*portfolio_probability(q,methods[METHOD_SHADOW])))
        # Local transfer derivative of the fixed-optimum advantage over each saved portfolio.
        for name,arr in methods.items():
            if name==METHOD_OPT:continue
            for i in range(13):
                for symbol in range(3):
                    for direction in (-1,1):
                        q=shift(p,i,symbol,direction*.01)
                        gap=portfolio_probability(q,methods[METHOD_OPT])-portfolio_probability(q,arr)
                        baseline=base_val[METHOD_OPT]-base_val[name]
                        sensitivity.append(dict(round=rid,method=name,match_no=i+1,home_team=g.iloc[i].home_team,
                            away_team=g.iloc[i].away_team,symbol=SYMBOLS[symbol],direction=direction,
                            delta_pp=1.,baseline_gap=baseline,perturbed_gap=gap,gap_change=gap-baseline,
                            abs_gap_change=abs(gap-baseline),P_symbol=p[i,symbol]))
            for kind in ('D上昇','D低下','最高確率過信','最高確率過小評価','H一斉上昇'):
                found=None
                for step in range(1,201):
                    delta=step*.001
                    q=scenario(p,kind,delta)
                    if portfolio_probability(q,arr)>portfolio_probability(q,methods[METHOD_OPT])+1e-18:
                        found=delta;break
                reversals.append(dict(round=rid,method=name,kind=kind,first_grid_reversal_pp=100*found if found is not None else np.nan,
                    searched_to_pp=20,grid_step_pp=.1,found=found is not None,
                    note='同一方向の仮定変動を全13試合へ適用した格子探索。一般的な最小変動ではない'))
    for name,rows in [('ticket_structure.csv',structure),('match_adoption.csv',adoption),('scenarios.csv',scenario_rows),
                      ('scenario_comparison.csv',comparisons),('reversal_thresholds.csv',reversals),('match_symbol_sensitivity.csv',sensitivity),('stress_curves.csv',curves)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    if any(sha256(Path(k))!=v for k,v in hashes.items()):raise AssertionError('input changed')
    audit=dict(input_hashes=hashes,rounds=list(ROUNDS),matches=78,seed=SEED,random_trials_per_kind_per_delta_per_round=TRIALS,
        strengths_pp=[1,3,5],scenario_count=len(scenario_rows),comparison_count=len(comparisons),
        archived_portfolios_fixed=True,Q_reoptimizes_tickets=False,all_scenarios_hypothetical=True,real_error_model_fitted=False,
        discarded_v30_used=False,production_changed=False,existing_data_unchanged=True,
        random_independent='each match independently chooses a random H/D/A recipient and random transfer sign; fixed generator',
        random_common='all matches receive the same randomly chosen H/D/A recipient; fixed generator',
        top3_selection='three fixtures with largest max(P(H),P(D),P(A)), ties match number')
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    report(pd.DataFrame(structure),pd.DataFrame(comparisons),pd.DataFrame(reversals),pd.DataFrame(sensitivity),pd.DataFrame(adoption),pd.DataFrame(curves),audit)
    print(json.dumps(dict(scenarios=len(scenario_rows),comparisons=len(comparisons),rounds=len(ROUNDS)),ensure_ascii=False))


def curve_svg(g,rid,kind):
    x=g.delta_pp.to_numpy(float);a=g.fixed_optimum_percent.to_numpy(float);b=g.shadow_c_percent.to_numpy(float)
    vmax=max(a.max(),b.max())*1.05
    def points(y):return ' '.join(f'{40+v*18:.1f},{195-w/vmax*160:.1f}' for v,w in zip(x,y))
    return f'<svg viewBox="0 0 445 235" width="445" height="235" role="img" aria-label="{rid} {kind}の感度曲線"><rect width="445" height="235" fill="white"/><text x="40" y="18" font-size="14">{rid} {kind}（仮定）</text><path d="M40 35V195H400" fill="none" stroke="#888"/><polyline points="{points(a)}" fill="none" stroke="#2368a2" stroke-width="2"/><polyline points="{points(b)}" fill="none" stroke="#c7643a" stroke-width="2"/><text x="40" y="220" font-size="11">0</text><text x="355" y="220" font-size="11">20 ポイント</text><text x="225" y="34" font-size="11" fill="#2368a2">固定最適</text><text x="310" y="34" font-size="11" fill="#c7643a">Shadow C</text></svg>'


def report(struct,comp,rev,sens,adopt,curves,audit):
    base=struct[struct.method.isin([METHOD_OPT,METHOD_SHADOW])].copy()
    summary=comp[(comp.kind!='基準')&(comp.method!=METHOD_OPT)].groupby(['method','kind','delta_pp']).agg(
        trials=('beats_fixed_optimum','size'),reversals=('beats_fixed_optimum','sum'),mean_ratio_to_opt=('ratio_to_fixed_optimum','mean')).reset_index()
    shadow=summary[summary.method.eq(METHOD_SHADOW)]
    strongest=sens.sort_values('abs_gap_change',ascending=False).groupby(['round','method']).head(3)
    def table(d,columns):return d[columns].to_html(index=False,escape=True,float_format=lambda x:f'{x:.6f}')
    selected=base[['round','method','actual_D','H_cells','D_cells','A_cells','D0_tickets','D_counts','total_probability']]
    top_reversals=rev[rev.found].sort_values('first_grid_reversal_pp')
    text=f'''# Phase 4-B 第2段階：固定10口の構造と仮定した確率変動への感度\n\n参照：`BUYPLAN_V4_0.md`、Phase 4-B第1段階、Phase 4-Aの原本A。旧v3.0の4試合選出・ダブル展開、およびv2.4の固定枚数を新設計の条件にしていない。数学的最適10口は購入時Pで固定し、以下は同一の変動後Qで各保存券を比較する。\n\n## 構造\n\n{selected.to_string(index=False)}\n\nD本数は特徴量であり、良否の目的指標ではない。各試合で不採用の結果とH/D/A枚数は `match_adoption.csv`、券単位の構成と確率は第1段階に保存した。実D数は説明用で、確率変動・券生成に使っていない。\n\n## 変動試験の定義と限界\n\n保存final確率Pを基準とし、1・3・5パーセントポイントを移す。指定記号へ移す場合、他の2記号から元の比率に応じて引く。逆方向は指定記号から取り、他の2記号へ比例配分。非負・合計1を保つ。D上下、最高確率記号の過信／過小評価、H一斉上昇、予測最大確率が高い3試合だけの変動、独立ランダム、共通方向ランダムを実施。後者2種は各強度・各開催100試行、seed={SEED}。**変動幅と分布はすべて仮定であり、実測された予測誤差ではない。**結果に合わせて選択・調整していない。\n\n## 同一Qでの逆転\n\n{summary.to_string(index=False)}\n\nShadow Cが存在するのは1665・1668・1670回のみ。表の `mean_ratio_to_opt` は変動後のQにおける、固定された数学的最適10口との比。比が1を超えたケースで逆転する。元のPからの低下率が小さいだけで優位とはしない。各シナリオのP/Q確率と券別比較は `scenario_comparison.csv`。\n\n## 同方向変動の逆転幅（格子探索）\n\n{rev.to_string(index=False)}\n\n0.1ポイント刻み、最大20ポイントまで。見つからなかったものは「この範囲で逆転なし」であり、不可能の証明ではない。数値は検討した変動経路に依存し、現実の誤差幅とは比較できない。\n\n## 試合・記号別感応度\n\n{strongest[['round','method','match_no','home_team','away_team','symbol','direction','gap_change','P_symbol']].to_string(index=False)}\n\n各セルを1ポイント移した際の「固定最適券−保存券」の差の変化量。結果を見て対象試合を選んでいない。全試合・全記号・両方向は `match_symbol_sensitivity.csv`。\n\n## 解釈と次段階\n\n固定Pが正しいなら第1段階の最適10口は最適。Qへの変動で優劣が変わっても、Qの発生頻度が不明なため現実に優れる方式は決められない。Phase 4-Aの78試合はモデル処理が3系統で、λ誤差分布や試合間依存を安定推定できない。後続の試合前原本を保存し、H/D/A校正・λ誤差・追加分散・試合間依存を時系列の未使用期間で検証する。集中の有利・不利をD本数だけで判断しない。v4.0正式方式は未決定。\n\n## シャドー運用\n\n`shadow_exact10.py` の `generate` は試合前の保存予測とmanifestのハッシュ・時刻を検証し、予測スナップショットSHAのディレクトリへ券と確率・入力の証跡を不変保存する。`score` は結果確定後に別ディレクトリへ照合結果を保存し、元の券を変更しない。本番配券フローには接続しない。手順は [shadow_runbook.md](shadow_runbook.md)。\n\n[構造](ticket_structure.csv) ／ [試合別H/D/A](match_adoption.csv) ／ [曲線](stress_curves.csv) ／ [シナリオ](scenarios.csv) ／ [同一Q比較](scenario_comparison.csv) ／ [逆転幅](reversal_thresholds.csv) ／ [感応度](match_symbol_sensitivity.csv) ／ [監査](audit.json)\n'''
    (OUT/'report.md').write_text(text)
    chart=''.join(f'<p>変動幅{delta:.0f}ポイントでのShadow Cとの逆転：{int(g.reversals.sum())}/{int(g.trials.sum())}試行・条件</p>' for delta,g in shadow.groupby('delta_pp'))
    chart+=''.join(curve_svg(g,rid,kind) for (rid,kind),g in curves.groupby(['round','kind']))
    html=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 4-B 第2段階</title><style>body{{font:16px/1.65 system-ui;margin:28px;color:#203047}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{border:1px solid #ccd4df;padding:5px}}th{{background:#edf3f8}}.scroll{{overflow:auto}}a{{color:#245b92}}</style><h1>v4.0 Phase 4-B 第2段階</h1><p>試合前原本A・6開催。10口を固定し、仮定した確率変動後の同じQで比較。実測誤差・正式採用の証明ではない。</p><h2>数学的最適10口とShadow Cの構造</h2><div class="scroll">{table(selected,list(selected.columns))}</div><h2>Shadow Cが固定最適10口を逆転した条件</h2>{chart}<div class="scroll">{table(shadow,list(shadow.columns))}</div><h2>同方向変動での逆転幅</h2><div class="scroll">{table(top_reversals,['round','method','kind','first_grid_reversal_pp']) if len(top_reversals) else '探索範囲内の逆転なし'}</div><p>0.1ポイント刻み、最大20ポイント。一般的な最小変動でも、現実の誤差幅でもない。</p><h2>感応度の大きい試合・記号</h2><div class="scroll">{table(strongest,['round','method','match_no','home_team','away_team','symbol','direction','gap_change'])}</div><p>元Pの最適10券はQに合わせて作り直していない。予測モデル・本番BuyPlan・過去原本は変更していない。</p><p><a href="report.md">詳細レポート</a> ／ <a href="ticket_structure.csv">構造</a> ／ <a href="match_adoption.csv">試合別採用</a> ／ <a href="stress_curves.csv">変動曲線</a> ／ <a href="scenarios.csv">確率シナリオ</a> ／ <a href="scenario_comparison.csv">全比較</a> ／ <a href="reversal_thresholds.csv">逆転幅</a> ／ <a href="match_symbol_sensitivity.csv">感応度</a> ／ <a href="shadow_runbook.md">シャドー手順</a> ／ <a href="audit.json">監査</a></p></html>'''
    (OUT/'index.html').write_text(html)

if __name__=='__main__':main()
