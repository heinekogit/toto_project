"""Read-only toto1670 multi-bet comparison at ten exact-result lines (1,000 yen)."""
from itertools import combinations,product
import json,hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from fixed_exact13_shadow import mass,validate
from draw_count_cover_shadow import draw_distribution,lookup_table,signature
from simulate_free_structural_draw import ROOT,page,table
from audit_draw_portfolio import ExactPortfolio

OUT=ROOT/'data/reports/toto1670_multi_draw_shadow'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def expand(base, positions, choices):
    rows=[]
    for selected in product(*choices):
        t=base.copy()
        for m,s in zip(positions,selected):t[m]=s
        rows.append(t)
    return np.array(rows,dtype=int)

def assess(name,t,p,dist,meta):
    if t.shape!=(10,13) or len(np.unique(t,axis=0))!=10:raise AssertionError('Expected ten distinct exact-result lines')
    per=mass(p,t);ks=sorted(set(map(int,(t==1).sum(1))))
    return dict(name=name,lines=10,price_yen=1000,p13=float(per.sum()),D_count_coverage=float(dist[ks].sum()),covered_D_counts='/'.join(map(str,ks)),D_counts='/'.join(map(str,(t==1).sum(1))),min_ticket_p13=float(per.min()),max_ticket_p13=float(per.max()),**meta)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fpath=ROOT/'data/reports/toto1670_fixed_exact13/forecast_input.csv'
    current=ROOT/'data/purchase_reference/shadow_buyplan_v25.csv'
    manifest=ROOT/'data/purchase_reference/manifest.json'
    paths=[fpath,current,manifest];hashes={str(path):digest(path) for path in paths}
    f=pd.read_csv(fpath).sort_values('match_no').reset_index(drop=True)
    saved=pd.read_csv(current).sort_values('match_no').reset_index(drop=True)
    for c in ['match_no','home_team','away_team']:assert f[c].tolist()==saved[c].tolist()
    assert json.loads(manifest.read_text())['round_id']=='toto1670'
    p=f[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy()
    c95=np.array([[{1:0,0:1,2:2}[int(s)] for s in saved[f'ticket{j:02}']] for j in range(1,11)],dtype=int)
    dist=draw_distribution(p);modes={'C95':c95};details={'C95':dict(purchase='10 single lines',chosen_from_saved=True)}
    by_count={k:c95[np.flatnonzero((c95==1).sum(axis=1)==k)[0]] for k in [3,4,5,10]}
    # Three H/D or D/A doubles: all 2^3 exact results, plus saved D4/D5 singles.
    best=None
    for base_no,base in enumerate(c95):
        if (base==1).any():continue
        for cols in combinations(range(13),3):
            e=expand(base,cols,[[int(base[m]),1] for m in cols])
            t=np.vstack([e,by_count[4],by_count[5]])
            if len(np.unique(t,axis=0))!=10:continue
            score=mass(p,t).sum()
            if best is None or score>best[0]:best=(score,t,dict(base_candidate=base_no+1,double_matches=[m+1 for m in cols],single_D_counts=[4,5]))
    if best is None:raise RuntimeError('No 3-double feasible portfolio found')
    modes['3Doubles_plus_2Singles']=best[1];details['3Doubles_plus_2Singles']=best[2]
    # One triple and one double give 3*2=6 lines, plus four saved singles.
    extras=np.vstack([by_count[k] for k in [3,4,5,10]])
    best=None
    for base_no,base in enumerate(c95):
        if (base==1).any():continue
        for triple in range(13):
            for double in range(13):
                if double==triple:continue
                e=expand(base,[triple,double],[list(range(3)),[int(base[double]),1]])
                t=np.vstack([e,extras])
                if len(np.unique(t,axis=0))!=10:continue
                score=mass(p,t).sum()
                if best is None or score>best[0]:best=(score,t,dict(base_candidate=base_no+1,triple_match=triple+1,double_match=double+1,single_D_counts=[3,4,5,10]))
    if best is None:raise RuntimeError('No triple/double feasible portfolio found')
    modes['1Triple_1Double_plus_4Singles']=best[1];details['1Triple_1Double_plus_4Singles']=best[2]
    summaries=[];tickets=[];sections=[];hit_rows=[];allocation=[]
    evaluator=ExactPortfolio(p)
    for name,t in modes.items():
        row=assess(name,t,p,dist,details[name]);summaries.append(row)
        exact=evaluator.metrics(t)
        hit_rows.append(dict(name=name,p13=exact['p13'],p12=exact['p12'],p11=exact['p11'],D_count_coverage=row['D_count_coverage']))
        grid=f[['match_no','home_team','away_team']].copy()
        for j,symbols in enumerate(t,1):grid[f'{j:02}']=['HDA'[v] for v in symbols]
        grid.to_csv(OUT/f'{name}_tickets.csv',index=False)
        sections.append('<h2>'+name+'</h2>'+table(grid))
        for j,symbols in enumerate(t,1):tickets.append(dict(mode=name,line=j,symbols=''.join('HDA'[v] for v in symbols),D_count=int((symbols==1).sum()),p13=float(mass(p,symbols[None])[0])))
        for m in range(13):
            allocation.append(dict(mode=name,match_no=m+1,home_team=f.loc[m,'home_team'],away_team=f.loc[m,'away_team'],count_H=int((t[:,m]==0).sum()),count_D=int((t[:,m]==1).sum()),count_A=int((t[:,m]==2).sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(tickets).to_csv(OUT/'exact_lines.csv',index=False)
    pd.DataFrame(hit_rows).to_csv(OUT/'hit_probabilities.csv',index=False)
    pd.DataFrame(allocation).to_csv(OUT/'match_allocations.csv',index=False)
    display=summary[['name','price_yen','p13','D_count_coverage','covered_D_counts','D_counts']].copy()
    for k in ['p13','D_count_coverage']:display[k]=display[k].map(lambda x:f'{x*100:.6f}%')
    body='<p>第1670回の保存予測と現行C95シャドーによる購入対象外の試算。1行は13試合の結果を1通り指定する100円のtoto券。ダブル・トリプルの入力は全組み合わせへ展開して同じ10行、1,000円で比較します。予測確率や現行の購入券を変更していません。</p>'
    body+='<h2>同額比較</h2>'+table(display)
    body+='<p>3ダブル案はC95のDなし券を土台とし、3試合にDを加えて8行へ展開。残り2行は保存C95のD4枚・5枚券です。3つのD対象試合と土台は、元予測確率での13的中確率合計が最大になる組を事前結果を使わず選択しました。1トリプル＋1ダブル案は6行へ展開し、残り4行を保存C95のD3・4・5・10枚券としました。これらは比較用の具体案であり、全てのマルチ購入形式の最適値ではありません。</p>'
    body+='<p>ダブル1試合なら2行＝200円、トリプル1試合なら3行＝300円、ダブル3試合なら8行＝800円。同じ10行の集合なら、マルチ入力とシングル10行は13的中モデル確率も費用も同じです。D本数カバー率は本数が合う結果の確率で、当選確率ではありません。</p>'
    body+='<p>ダブルではH/DまたはD/Aの二択が結び付いて展開されるため、どちらが起きても残る構造を作れます。一方、ダブル3試合の8行はその3試合以外の10試合が全て同じ記号です。その外で一つ外れれば8行が同時に13的中を失います。Dを選ぶ試合と土台の10固定試合の精度が重要です。</p>'
    body+='<h2>11・12的中以上と試合別枚数</h2>'+table(pd.DataFrame(hit_rows))
    body+='<p>C95とマルチ案では試合別H/D/A枚数が異なります。確率差には、10行の記号配分変更が含まれます。</p>'
    body+=''.join(sections)
    for file in sorted(OUT.glob('*.csv')):body+=f'<p><a href="{file.name}">{file.name}</a></p>'
    (OUT/'index.html').write_text(page('第1670回：Dのダブル・トリプル同額比較',body))
    assert all(digest(Path(path))==value for path,value in hashes.items())
    (OUT/'audit.json').write_text(json.dumps(dict(protected_sha256=hashes,round='toto1670',base='saved C95 v2.5 shadow',ticket_price_yen=100,multi_lines_expanded=True,production_changed=False,details=details),ensure_ascii=False,indent=2))
    print(display.to_string(index=False));print(details);print(OUT/'index.html')

if __name__=='__main__':main()
