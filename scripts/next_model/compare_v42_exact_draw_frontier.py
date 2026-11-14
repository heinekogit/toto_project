"""Exact 0..10 D-ticket frontier under one saved final H/D/A distribution."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_score_structure_phase4b_stage1 import N, POWERS, SYMBOLS, decode, top_ten, universe
from analyze_score_structure_phase4b_stage3 import draw_distribution, top_category
from common import sha256


def compute(forecast: pd.DataFrame):
    f=forecast.sort_values('match_no').reset_index(drop=True)
    if len(f)!=13 or f.match_no.tolist()!=list(range(1,14)):
        raise ValueError('13 ordered matches required')
    p=f[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy(float)
    if not np.isfinite(p).all() or (p<=0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-9):
        raise ValueError('invalid final H/D/A')
    mass=universe(p)
    codes=np.arange(N,dtype=np.int64)
    has_draw=np.zeros(N,dtype=bool)
    for power in POWERS:has_draw|=(codes//power)%3==1
    no=top_category(mass,~has_draw)
    yes=top_category(mass,has_draw)
    baseline=top_ten(mass)
    if set(baseline)!=set(no):raise ValueError('this round baseline is not D-free; adjust exact frontier interpretation')
    dist=draw_distribution(p)
    prob_any_D=float(1-dist[0])
    base_mass=float(mass[baseline].sum())
    rows=[];tickets=[];changes=[];previous=set()
    for k in range(11):
        selected=np.concatenate((no[:10-k],yes[:k]))
        selected=selected[np.lexsort((selected,-mass[selected]))]
        chosen=set(map(int,selected))
        p_all=float(mass[selected].sum())
        p_D=float(mass[yes[:k]].sum())
        strings=[''.join(SYMBOLS[digits]) for digits in decode(selected)]
        d_counts=[s.count('D') for s in strings]
        rows.append(dict(D_ticket_count=k,P13=p_all,P13_percent=p_all*100,
                         P13_ratio_to_unrestricted=p_all/base_mass,
                         absolute_loss=base_mass-p_all,
                         incremental_loss=0. if k==0 else float(mass[no[10-k]]-mass[yes[k-1]]),
                         D_result_probability_mass=p_D,
                         conditional_P13_given_any_D=p_D/prob_any_D,
                         D0_result_probability_mass=p_all-p_D,
                         D_cells_total=sum(d_counts),D_counts_by_ticket=';'.join(map(str,d_counts)),
                         covered_D_count_values=';'.join(map(str,sorted(set(d_counts)-{0}))),
                         distinct_tickets=len(chosen)))
        for number,(code,symbol) in enumerate(zip(selected,strings),1):
            tickets.append(dict(D_ticket_count=k,candidate=number,symbols=symbol,
                                contains_D='D' in symbol,D_count=symbol.count('D'),
                                P_ticket_probability=float(mass[code]),
                                D_match_numbers=';'.join(str(i+1) for i,c in enumerate(symbol) if c=='D'),
                                **{f'match_{i+1:02d}':symbol[i] for i in range(13)}))
        for code in sorted(chosen-previous):
            changes.append(dict(D_ticket_count=k,change='追加',symbols=''.join(SYMBOLS[decode([code])[0]]),
                                P_ticket_probability=float(mass[code])))
        for code in sorted(previous-chosen):
            changes.append(dict(D_ticket_count=k,change='除外',symbols=''.join(SYMBOLS[decode([code])[0]]),
                                P_ticket_probability=float(mass[code])))
        previous=chosen
    summary=pd.DataFrame(rows)
    if not (summary.distinct_tickets==10).all() or not np.all(np.diff(summary.P13)<=1e-15):
        raise AssertionError('frontier ordering/uniqueness failed')
    if not np.allclose(summary.P13, [sum(mass[no[:10-k]])+sum(mass[yes[:k]]) for k in range(11)],atol=1e-15):
        raise AssertionError('exact category optimum failed')
    return summary,pd.DataFrame(tickets),pd.DataFrame(changes),dist,dict(first_D_ticket_rank=int(np.sum(mass>mass[yes[0]])+np.sum((mass==mass[yes[0]])&(codes<yes[0]))+1),
                                                                        P_any_D=prob_any_D,expected_D=float(p[:,1].sum()))


def write(forecast_path: Path,outdir: Path,round_id: str):
    forecast_path=Path(forecast_path);outdir=Path(outdir)
    f=pd.read_csv(forecast_path,encoding='utf-8-sig',low_memory=False).sort_values('match_no').reset_index(drop=True)
    summary,tickets,changes,distribution,extra=compute(f)
    outdir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(outdir/'frontier.csv',index=False,encoding='utf-8-sig')
    tickets.to_csv(outdir/'tickets.csv',index=False,encoding='utf-8-sig')
    changes.to_csv(outdir/'ticket_changes.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'D_count':range(14),'probability':distribution}).to_csv(outdir/'D_count_distribution.csv',index=False,encoding='utf-8-sig')
    audit=dict(round=round_id,source_sha256=sha256(forecast_path),fixed_HDA_probabilities=True,
               probability_correction=False,actual_results_used=False,condition='exactly k of 10 tickets contain at least one D',
               D_total_unrestricted=True,per_match_D_unrestricted=True,per_ticket_D_unrestricted=True,
               search_space=3**13,optimization='exact category top-k selection with deterministic base-3 tie break',
               historical_reference='Phase 4-B stage 3 used the same exact category enumeration for six archived rounds',
               d_scoop_predictive_improvement_claim=False,**extra)
    (outdir/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    view=summary[['D_ticket_count','P13_percent','P13_ratio_to_unrestricted','absolute_loss','incremental_loss',
                  'D_result_probability_mass','conditional_P13_given_any_D','D_cells_total','D_counts_by_ticket','covered_D_count_values']].copy()
    view['P13_percent']=view.P13_percent.map(lambda x:f'{x:.9f}%')
    view['P13_ratio_to_unrestricted']=(100*view.P13_ratio_to_unrestricted).map(lambda x:f'{x:.3f}%')
    for col in ('absolute_loss','incremental_loss','D_result_probability_mass','conditional_P13_given_any_D'):
        view[col]=(100*view[col]).map(lambda x:f'{x:.9f}%')
    cards=[]
    for k in range(11):
        group=tickets[tickets.D_ticket_count.eq(k)]
        head='<tr><th>No.</th><th>対戦</th>'+''.join(f'<th>{i:02d}</th>' for i in range(1,11))+'</tr>'
        body=[]
        for i,r in f.iterrows():
            cells=[str(i+1),f'{r.home_team}－{r.away_team}']+[str(t[f'match_{i+1:02d}']) for _,t in group.iterrows()]
            body.append('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in cells)+'</tr>')
        cards.append(f'<details><summary>D入り{k}口：{summary.iloc[k].P13_percent:.9f}%</summary><div class="scroll"><table>{head}{"".join(body)}</table></div></details>')
    report=(f'# {round_id} D入り券0〜10口の厳密比較\n\n'
            '保存final確率だけを使用。D総枚数、試合別D枚数、券別D枚数の制約なし。'
            '各kについて、Dなし組み合わせ上位10-kとD入り組み合わせ上位kを選ぶ厳密解。'
            'これは仮にD入り券数を固定した場合の機会損失で、購入口数の推奨ではない。'
            '得点構造による予測補正も行っていない。\n\n'
            f'モデルのD発生確率は {extra["P_any_D"]:.6f}、期待D数は {extra["expected_D"]:.6f}。'
            '「D発生時の13的中確率」はDを1試合以上含む結果を条件とする確率であり、'
            'D入り券の無条件の購入価値とは異なる。\n\n'
            '前回の21枚・35枚案にはD本数の範囲と上限の制約があった。'
            '本表はその制約を除いた配券側の比較であり、d-scoopの予測力を検証するものではない。'
            '過去6開催の同種の厳密比較はPhase 4-B第3段階に保存されている。\n\n'
            '```csv\n'+summary.to_csv(index=False)+'```\n')
    (outdir/'report.md').write_text(report,encoding='utf-8')
    document=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>D入り券0〜10口</title>"
              "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.note{background:#fff0d3;padding:12px}details{padding:8px;border-bottom:1px solid #ddd}</style>"
              f'<h1>{html.escape(round_id)} D入り券0〜10口の厳密比較</h1>'
              '<p class="note">前回のD本数1〜6を各1券、1券最大7Dという制約は使いません。D入り券数だけを仮に指定し、各条件で元の保存確率Pの全的中確率を最大化します。これはD予測の改善ではありません。</p>'
              f'<p>モデルの少なくとも1試合Dの確率：{100*extra["P_any_D"]:.3f}%。D入り券の最上位は全3^13組中{extra["first_D_ticket_rank"]}位。</p>'
              '<h2>損失とD発生時の獲得確率</h2><div class="scroll">'+view.to_html(index=False,escape=True)+'</div>'
              '<h2>券の追加・除外</h2><div class="scroll">'+changes.to_html(index=False,escape=True)+'</div>'
              +''.join(cards)+'<p><a href="frontier.csv">比較CSV</a>／<a href="tickets.csv">全10口CSV</a>／<a href="ticket_changes.csv">入替CSV</a>／<a href="report.md">説明</a></p></html>')
    (outdir/'index.html').write_text(document,encoding='utf-8')
    return audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--forecast',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--round',required=True)
    args=parser.parse_args()
    print(json.dumps(write(args.forecast,args.outdir,args.round),ensure_ascii=False))


if __name__=='__main__':main()
