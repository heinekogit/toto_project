"""Frozen toto1670 B/B-D2/G1/G2 comparison."""
import json
import numpy as np
import pandas as pd
from group_draw_shadow import optimize,groups,validate_groups
from fixed_exact13_shadow import mass,coverage,validate
from audit_draw_portfolio import ExactPortfolio
from simulate_free_structural_draw import ROOT,PROB_COLUMNS,sha,page,table

OUT=ROOT/'data/reports/toto1670_draw_groups'


def main():
    OUT.mkdir(exist_ok=True);base=ROOT/'data/reports/toto1670_draw_bounds';old=ROOT/'data/reports/toto1670_fixed_exact13'
    protected=[p for d in [base,old,ROOT/'data/purchase_reference'] for p in d.glob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected}
    f=pd.read_csv(old/'forecast_input.csv').sort_values('match_no').reset_index(drop=True);p=f[PROB_COLUMNS].to_numpy()
    modes={};audits={};oldmetrics=pd.read_csv(base/'comparison.csv')
    for name in ['B','B-D2']:
        g=pd.read_csv(base/f'{name}_tickets.csv')
        for c in ['match_no','home_team','away_team']:assert g[c].tolist()==f[c].tolist()
        modes[name]=np.array([['HDA'.index(s) for s in g[f'{i:02}']] for i in range(1,11)])
        audits[name]=dict(seconds=float(oldmetrics.loc[oldmetrics['mode'].eq(name),'optimization_seconds'].iloc[0]),source='previous frozen result')
    counts=np.stack([(modes['B']==s).sum(0) for s in range(3)],1)
    for name,circular in [('G1',False),('G2',True)]:
        t,audit=optimize(p,counts,circular,starts=64,warm=modes.get('G1') if circular else None)
        modes[name]=t;audits[name]=audit
    evaluator=ExactPortfolio(p);summary=[];details=[];distribution=[];checks=[];group_rows=[];sections=[]
    for name,t in modes.items():
        validate(t,counts);met=evaluator.metrics(t)
        for k in [11,12]:np.testing.assert_allclose(met[f'p{k}'],coverage(evaluator.mass,t,13-k),rtol=1e-12)
        np.testing.assert_allclose(met['p13'],mass(p,t).sum(),rtol=1e-12)
        d=(t==1).sum(1)
        summary.append(dict(mode=name,p13=met['p13'],p12=met['p12'],p11=met['p11'],retention_percent=100*met['p13']/mass(p,modes['B']).sum(),D_counts='/'.join(map(str,d)),D_total=int(d.sum()),D3_tickets=int((d==3).sum()),D4_tickets=int((d==4).sum()),D_std=float(d.std()),D_0_1_tickets=int((d<=1).sum()),D_11_13_tickets=int((d>=11).sum()),duplicates=0,counts_match=True,optimization_seconds=audits[name]['seconds']))
        for k in range(14):distribution.append(dict(mode=name,D_count=k,tickets=int((d==k).sum())))
        g=f[['match_no','home_team','away_team']].copy();g['reference00']=['HDA'[v] for v in p.argmax(1)]
        for j,row in enumerate(t,1):
            g[f'{j:02}']=['HDA'[v] for v in row]
            details.append(dict(mode=name,candidate=f'{j:02}',symbols=''.join('HDA'[v] for v in row),D_count=int((row==1).sum()),p13=float(mass(p,row[None])[0])))
        g.to_csv(OUT/f'{name}_tickets.csv',index=False)
        sections.append('<h2>'+name+'</h2>'+table(g))
        for m in range(13):
            for s,label in enumerate('HDA'):checks.append(dict(mode=name,match_no=m+1,symbol=label,expected=int(counts[m,s]),actual=int((t[:,m]==s).sum())))
        if name in ['G1','G2']:
            shifts=validate_groups(t,counts,name=='G2')
            for m,shift in enumerate(shifts):
                gs=groups(int(counts[m,1]),shift)
                for n,slots in enumerate(gs,1):group_rows.append(dict(mode=name,match_no=m+1,home_team=f.loc[m,'home_team'],away_team=f.loc[m,'away_team'],D_count=int(counts[m,1]),shift=shift,group=n,candidates='/'.join(f'{i+1:02}' for i in slots),D_candidates='/'.join(f'{i+1:02}' for i in slots if t[i,m]==1),group_D_count=int(sum(t[i,m]==1 for i in slots))))
    equivalence=[]
    b_lookup={tuple(row):i+1 for i,row in enumerate(modes['B'])}
    for name in ['G1','G2']:
        for j,row in enumerate(modes[name],1):equivalence.append(dict(mode=name,candidate=j,identical_B_candidate=b_lookup.get(tuple(row))))
    pd.DataFrame(equivalence).to_csv(OUT/'B_ticket_equivalence.csv',index=False)
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'candidate_diagnostics.csv',index=False)
    pd.DataFrame(distribution).to_csv(OUT/'draw_count_distribution.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'counts_audit.csv',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/'group_assignments.csv',index=False)
    stability=[]
    for name in ['G1','G2']:
        for n in [16,32,64]:
            runs=audits[name]['runs'][:n];stability.append(dict(mode=name,starts=n,best_p13=max(r['final'] for r in runs),min_final=min(r['final'] for r in runs),median_final=float(np.median([r['final'] for r in runs])),max_sweeps=max(r['sweeps'] for r in runs),all_local_stop=all(r['local_stop'] for r in runs)))
    pd.DataFrame(stability).to_csv(OUT/'search_stability.csv',index=False)
    shown=summary[['mode','p13','p12','p11','retention_percent','D_counts','D3_tickets','D4_tickets','optimization_seconds']].copy()
    for c in ['p13','p12','p11']:shown[c]=shown[c].map(lambda v:f'{100*v:.6f}%')
    body='<p>第1670回の保存予測、試合別H/D/A枚数、D総数38を固定。B・B-D2は前回保存結果を使用。現行v2.4/v2.5・購入記録を変更していません。00は参照用のみです。</p>'
    body+='<h2>比較結果</h2>'+table(shown)
    body+='<p>13/12/11以上は10口の少なくとも1口が達成するモデル確率。試合間独立の下、全3^13結果を列挙して重複なく集計。11/12以上は別の近傍集合和計算とも照合しています。実績改善を示す検証ではありません。</p>'
    body+='<h2>結論：今回は二極化を解消しない</h2><p>G1/G2の10券は、候補番号を除けば前回Bの10券と完全に同じでした。単に同程度の確率ではなく、券集合そのものが一致します。対応番号をB_ticket_equivalence.csvに記録しました。したがって13的中確率を100%維持できたのは、D分布を改善したからではありません。</p><p>13試合中12試合のD配分が3枚で、残る1試合が2枚です。G1では12試合が同じ01〜03/04〜06/07〜10の区切りを使います。各グループの特定候補へ何度でもDを置けるため、候補01・05・08へDが重なりました。試合内の分散と、複数試合を通じた候補ごとの分散は別です。</p><p>G2で採用された解の循環位置は全13試合で0でした。全循環位置を探索候補に含めても、G1の解を上回る解は見つかりませんでした。循環は選択可能であって、異なる位置を強制する条件ではありません。今回の目的に対してG1/G2の採用は勧めません。試合をまたぐDの重なりを制御する条件が必要ですが、今回それを追加して結果を変えることはしていません。</p>'
    body+='<h2>分割規則と探索</h2><p>D=d枚の境界をfloor(10k/d), k=0..dと定義。d=3なら01〜03/04〜06/07〜10、d=4なら01〜02/03〜05/06〜07/08〜10。サイズ差は最大1です。D0枚は制約なし、D10枚は全候補Dです。</p><p>G1は位置固定。G2は0〜9候補分の循環移動だけを許容します。各試合について、合法な全D配置と全H/A枚数配置を列挙し、最適な列へ置換する探索を行います。G2は全循環位置を候補列集合へ含めるため、循環位置の変更も同時に探索します。複数位置で同じ配置が合法なら最小shiftを表示します。</p><p>64個の決定的初期配置から探索。乱数は使わず、列候補のインデックスを規則的に変えて初期値を生成します。各試合の全列置換（最大10候補を同時変更）に加え、2候補間の1試合交換・2試合同時交換も探索。80反復上限、改善がなくなれば終了。G2にはG1最良解も初期値として与えます。目的は13的中確率の和だけで、D本数上下限・均等化・候補番号別の優先度はありません。大域最適の保証はありません。</p><p>グループ制約は候補番号に依存するため、G1/G2の出力は確率順に並べ替えていません。初期値16/32/64個時点の最良値・中央値・最悪値を記録しています。局所探索の反復回数と多初期値数が異なるため、B/B-D2との実行時間の単純比較には注意してください。</p>'
    body+='<h2>D本数分布</h2>'+table(pd.DataFrame(distribution).pivot(index='D_count',columns='mode',values='tickets').reset_index())
    body+='<h2>初期値による安定性</h2>'+table(pd.DataFrame(stability))
    body+=''.join(sections)
    body+='<h2>試合別グループ・循環位置</h2>'+table(pd.DataFrame(group_rows))
    for path in sorted(OUT.glob('*.csv')):body+=f'<p><a href="{path.name}">{path.name}</a></p>'
    (OUT/'index.html').write_text(page('第1670回：Dグループ分散配置G1・G2',body))
    assert all(sha(path)==h for path,h in hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(modes=audits,protected_sha256=hashes,production_changed=False,random_D_selection=False),ensure_ascii=False,indent=2))
    print(shown.to_string(index=False),flush=True);print(OUT/'index.html',flush=True)

if __name__=='__main__':main()
