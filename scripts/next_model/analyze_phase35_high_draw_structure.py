"""Frozen-condition exploratory audit; no forecast or ticket mutations."""
from pathlib import Path
import hashlib, json, html
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from analyze_score_structure import score_grid, structure

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/reports/score_structure_phase35_high_draw_structure'
BASE=ROOT/'data/reports/score_structure_phase35'

def conditions(d):
    m00=d.groupby('round').P_0_0.transform('median')
    m11=d.groupby('round').P_1_1.transform('median')
    return {'strict':(d.P_0_0<m00)&(d.P_1_1>m11),
            'central':(d.P_0_0<=m00)&(d.P_1_1>m11),
            'inclusive':(d.P_0_0<=m00)&(d.P_1_1>=m11)}

def summary(g,**meta):
    return dict(**meta,n=len(g),H=int(g.actual_symbol.eq('H').sum()),D=int(g.actual_symbol.eq('D').sum()),
                A=int(g.actual_symbol.eq('A').sum()),D_rate=g.actual_symbol.eq('D').mean(),expected_D=g.baseline_P_D.sum(),
                mean_PD=g.baseline_P_D.mean(),mean_HA_gap=g.HA_gap.mean(),mean_T=g['T'].mean(),
                mean_abs_logR=g.log_R.abs().mean(),mean_margin1=g.P_onegoal.mean())

def match_controls(d,mode):
    """Per round one-to-one matching, no outcome-dependent selection. Calipers fixed."""
    rows=[]
    for rnd,g in d.groupby('round'):
        a=g[g.inclusive];b=g[~g.inclusive]
        if not len(a):continue
        pdgap=abs(a.baseline_P_D.to_numpy()[:,None]-b.baseline_P_D.to_numpy())
        hagap=abs(a.HA_gap.to_numpy()[:,None]-b.HA_gap.to_numpy())
        if mode=='PD_HA':
            valid=(pdgap<=.01)&(hagap<=.05)
            cost=pdgap/.01+hagap/.05
        else:
            td=abs(a['T'].to_numpy()[:,None]-b['T'].to_numpy())
            rd=abs(a.log_R.abs().to_numpy()[:,None]-b.log_R.abs().to_numpy())
            valid=(td<=.15)&(rd<=.10)
            cost=td/.15+rd/.10
        # Dummy matches permit unmatched; invalid edges cost more than any valid pairing.
        c=np.concatenate([np.where(valid,cost,1e6),np.full((len(a),len(a)),1000.)],axis=1)
        ii,jj=linear_sum_assignment(c)
        for i,j in zip(ii,jj):
            x=a.iloc[i];ok=j<len(b) and valid[i,j]
            row=dict(mode=mode,round=rnd,target_no=x.match_no,target_card=x.card,matched=bool(ok),target_result=x.actual_symbol,
                     target_PD=x.baseline_P_D,target_HA_gap=x.HA_gap,target_T=x['T'],target_abs_logR=abs(x.log_R))
            if ok:
                y=b.iloc[j]
                row.update(control_no=y.match_no,control_card=y.card,control_result=y.actual_symbol,control_PD=y.baseline_P_D,
                           control_HA_gap=y.HA_gap,control_T=y['T'],control_abs_logR=abs(y.log_R),PD_gap=pdgap[i,j],HA_gap_difference=hagap[i,j],
                           control_P00=y.P_0_0,control_P11=y.P_1_1,target_P00=x.P_0_0,target_P11=x.P_1_1)
            rows.append(row)
    return pd.DataFrame(rows)

def run():
    paths=[BASE/'diagnostics_78.csv',BASE/'relative_scenarios.csv',ROOT/'data/reports/score_structure_phase12/match_structure_78.csv',
           ROOT/'data/reports/score_structure_phase35_threshold_audit/boundary_margins_78.csv',ROOT/'data/forecast_registry/forecasts.jsonl',ROOT/'data/forecast_registry/results.jsonl']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    d=pd.read_csv(paths[3]);st=pd.read_csv(paths[2]);sc=pd.read_csv(paths[1])
    for k,v in conditions(d).items():d[k]=v
    d['card']=d.home_team+'－'+d.away_team
    d['actual_score']=d.actual_home.astype(str)+'–'+d.actual_away.astype(str)
    d['HA_gap']=abs(d.baseline_P_H-d.baseline_P_A)
    d['R']=np.exp(d.log_R)
    d['tier']=np.select([d.strict,d.central,d.inclusive],['厳格4','中央追加1','境界追加3'],default='条件外')
    d['PD_rank']=d.groupby('round').baseline_P_D.rank(ascending=False,method='min')
    cols=['P_2_2','P_3_3','P_draw_4plus','P_home_2','P_away_2','P_home_3plus','P_away_3plus']
    d=d.merge(st[['round','match_no']+cols],on=['round','match_no'],validate='one_to_one')
    d['P_33plus']=d.P_3_3+d.P_draw_4plus;d['P_H2plus']=d.P_home_2+d.P_home_3plus;d['P_A2plus']=d.P_away_2+d.P_away_3plus
    assert [(int(d[k].sum()),int(d.loc[d[k],'actual_D'].sum())) for k in ['strict','central','inclusive']]==[(4,4),(5,4),(8,6)]
    for q,key in [(.45,'strict'),(.5,'central'),(.55,'inclusive')]:
        a=d.groupby('round').P_0_0.transform(lambda x:x.quantile(q));b=d.groupby('round').P_1_1.transform(lambda x:x.quantile(1-q))
        assert ((d.P_0_0<=a)&(d.P_1_1>b)).equals(d[key])
    # Check reconstructed structures and final mixing without changing the model.
    errs=[]
    for r in d.itertuples():
        matrix,_=score_grid(r.lambda_home,r.lambda_away);ss=structure(matrix)
        errs.append(max(abs(ss['P_0_0']-r.P_0_0),abs(ss['P_1_1']-r.P_1_1)))
    assert max(errs)<1e-12
    OUT.mkdir(exist_ok=True,parents=True)
    def save(name,frame):frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    save('match_comparison_78.csv',d);save('selected_8.csv',d[d.inclusive]);save('added_4.csv',d[d.inclusive&~d.strict])
    masks={'厳格4':d.strict,'中央5':d.central,'境界含む8':d.inclusive,'追加4':d.inclusive&~d.strict,
           '0-0低・2-2以上高':d.low_P00&d.high_P2plus,'高T・比接近':~d.low_T&d.near_ratio,'低T・比接近':d.low_T&d.near_ratio}
    rows=[summary(d,name='全体')]
    for name,mask in masks.items():rows.append(summary(d[mask],name=name))
    for name in list(masks)[4:]:
        mask=masks[name]
        for title,m in [('両方',mask&d.inclusive),('8条件のみ',~mask&d.inclusive),('相手条件のみ',mask&~d.inclusive),('どちらも外',~mask&~d.inclusive)]:
            rows.append(summary(d[m],name=name+'：'+title))
    comb=pd.DataFrame(rows);save('condition_combinations.csv',comb)
    rs=[]
    for rnd,g in d.groupby('round'):
        for name,m in [('該当',g.inclusive),('条件外',~g.inclusive)]:rs.append(summary(g[m],round=rnd,group=name))
    rounds=pd.DataFrame(rs);save('round_summary.csv',rounds)
    matches=pd.concat([match_controls(d,m) for m in ['PD_HA','T_ratio']],ignore_index=True);save('matched_controls.csv',matches)
    # Fixed 4 equal-count bins for descriptive continuous plots; not a selection rule.
    continuous=[]
    features=['P_0_0','P_1_1','T','HA_gap','margin_P00_pp','margin_P11_pp','rank_P00_ascending','rank_P11_ascending']
    for col in features:
        groups=pd.qcut(d[col],4,duplicates='drop')
        for bin_,g in d.groupby(groups,observed=True):
            continuous.append(summary(g,feature=col,bin=str(bin_),minimum=g[col].min(),maximum=g[col].max()))
    cont=pd.DataFrame(continuous);save('continuous_analysis.csv',cont)
    joint=[]
    for a in ['負','ゼロ','正']:
        for b in ['負','ゼロ','正']:
            x=np.sign(d.margin_P00_pp)=={'負':-1,'ゼロ':0,'正':1}[a];y=np.sign(d.margin_P11_pp)=={'負':-1,'ゼロ':0,'正':1}[b]
            joint.append(summary(d[x&y],P00_margin=a,P11_margin=b))
    save('joint_margin_analysis.csv',pd.DataFrame(joint))
    # Absolute global median compared with within-round median. This is a diagnostic, not retuning.
    glob=(d.P_0_0<=d.P_0_0.median())&(d.P_1_1>=d.P_1_1.median())
    globaltab=pd.DataFrame([summary(d[glob],name='全78の中央値'),summary(d[d.inclusive],name='開催内中央値'),summary(d[glob&d.inclusive],name='共通')]);save('absolute_relative_comparison.csv',globaltab)
    sc=sc.merge(d[['round','match_no','inclusive','tier','HA_gap']],on=['round','match_no'],validate='many_to_one')
    save('sensitivity_selected_8.csv',sc[sc.inclusive])
    sr=[]
    for (scenario,eps),g in sc[sc.scenario.ne('基準')].groupby(['scenario','epsilon']):
        for name,m in [('該当D',g.inclusive&g.actual_symbol.eq('D')),('該当非D',g.inclusive&g.actual_symbol.ne('D')),('条件外D',~g.inclusive&g.actual_symbol.eq('D')),('条件外非D',~g.inclusive&g.actual_symbol.ne('D'))]:
            z=g[m];sr.append(dict(scenario=scenario,epsilon=eps,group=name,n=len(z),mean_delta_PD_pp=100*z.delta_P_D.mean(),min_delta_PD_pp=100*z.delta_P_D.min(),max_delta_PD_pp=100*z.delta_P_D.max(),mean_delta_P11_pp=100*z.delta_P_1_1.mean()))
    sens=pd.DataFrame(sr);save('sensitivity_summary.csv',sens)
    forecasts=[json.loads(x) for x in paths[4].read_text().splitlines()];results=[json.loads(x) for x in paths[5].read_text().splitlines()]
    known=set(d['round']);inv=[]
    for event in forecasts:
        if event['round_id'] not in known: continue
        assert event['provenance_class']=='A'
        source=Path(event['source_file'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==event['output_sha256']
        row=d[(d['round']==event['round_id'])&(d.match_no==event['match_no'])].iloc[0]
        for a,b in [('lambda_home','lambda_home'),('lambda_away','lambda_away'),('baseline_P_H','final_prob_home'),('baseline_P_D','final_prob_draw'),('baseline_P_A','final_prob_away')]:
            assert abs(row[a]-event[b])<1e-12
    assert not d.duplicated(['round','match_no']).any()
    for rnd in sorted(set(x['round_id'] for x in forecasts)):
        ff=[x for x in forecasts if x['round_id']==rnd];rr=[x for x in results if x['round_id']==rnd]
        inv.append(dict(round=rnd,forecast_events=len(ff),result_events=len(rr),used_discovery=rnd in known,eligible_unused=rnd not in known and len(ff)==13 and len(rr)==13))
    save('unused_data_inventory.csv',pd.DataFrame(inv))
    spec=dict(version='research_frozen_1',condition='P00 <= within_round_median(P00) AND P11 >= within_round_median(P11)',
              score_grid='existing analyze_score_structure.score_grid; normalized independent Poisson 0..10',round_size=13,
              equality='inclusive; unrounded float64 computed with same function; no epsilon or decimal display rounding',
              outcomes_not_used_for_selection=True,discovery_rounds=sorted(known),new_thresholds_searched=0,
              future_protocol=['Save all 13 prematch A lambda/model versions, source hash, features and flags before kickoff.',
                               'Do not recalculate past features from updated team data. Record model versions separately.',
                               'Accumulate every subsequent eligible complete round, not only qualifying or successful rounds.',
                               'Report D/H/A rates, expected D, individual scores, exact same PD/HA matching, and all unmatched matches.',
                               'First descriptive checkpoint at 10 new complete rounds, not a significance or success threshold. No retuning.',
                               'Keep original rule frozen even if results are unfavorable; any revision becomes a new exploratory version.'],
              ticket_generation=False)
    (OUT/'future_validation_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
    audit=dict(input_hashes=hashes,rows=78,draws=18,condition_counts=[4,5,8],draw_counts=[4,4,6],max_reconstruction_error=max(errs),
               new_threshold_searches=0,matching_specs=['same round, one-to-one: |PD gap|<=.01 and |HA gap difference|<=.05; normalized L1 minimum assignment',
               'same round, one-to-one: |T gap|<=.15 and ||logR| gap|<=.10; normalized L1 minimum assignment'],
               independent_new_rounds=sum(x['eligible_unused'] for x in inv),existing_inputs_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()))
    assert audit['existing_inputs_unchanged']
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    return d,comb,matches,cont,sens,rounds,globaltab,audit


def scatter(d,x,y,title,path):
    # Standalone SVG: actual D red, non-D blue; inclusive condition outlined.
    w,h=860,480; xmin,xmax=d[x].min(),d[x].max();ymin,ymax=d[y].min(),d[y].max()
    xp=lambda v:80+700*(v-xmin)/(xmax-xmin);yp=lambda v:405-330*(v-ymin)/(ymax-ymin)
    bits=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13"><text x="30" y="25">{html.escape(title)}</text>']
    for k in range(5):
        xv=xmin+(xmax-xmin)*k/4;yv=ymin+(ymax-ymin)*k/4
        bits.append(f'<path d="M {xp(xv)} 65 V 405 M 80 {yp(yv)} H 780" stroke="#ddd"/><text x="{xp(xv)}" y="427">{xv:.3f}</text><text x="15" y="{yp(yv)}">{yv:.3f}</text>')
    if xmin<=0<=xmax:bits.append(f'<path d="M {xp(0)} 65 V 405" stroke="#333"/>')
    if ymin<=0<=ymax:bits.append(f'<path d="M 80 {yp(0)} H 780" stroke="#333"/>')
    for _,r in d.iterrows():
        color='#bb352e' if r.actual_symbol=='D' else '#3574a7';radius=6 if r.inclusive else 3
        bits.append(f'<circle cx="{xp(r[x])}" cy="{yp(r[y])}" r="{radius}" fill="{color}" stroke="{"#111" if r.inclusive else color}"><title>{r["round"]} #{r.match_no} {html.escape(r.card)} {r.actual_score}</title></circle>')
    bits.append(f'<text x="260" y="458">{x}</text><text x="90" y="52">縦軸: {y} ／ 赤=D 青=非D 黒縁=8条件</text></g></svg>')
    path.write_text(''.join(bits))


def render(d,comb,matches,cont,sens,rounds,globaltab,audit):
    s=d[d.inclusive];non=s[s.actual_symbol.ne('D')]
    def table(f):return '<div class="scroll">'+f.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)+'</div>'
    blocks=[]
    def add(title,text,frame=None):
        blocks.append(f'<section><h2>{title}</h2><p>{text}</p>'+('' if frame is None else table(frame))+'</section>')
    add('結論','開催内の相対条件は8試合・実D6（75%）、モデル期待Dは'+f'{s.baseline_P_D.sum():.3f}試合。条件外70試合はD12（17.1%）。この差は探索データ内で確認できるが、成立原因や将来の識別力は未確定。非D2試合は8試合内で|logR|が大きい上位2例（final H/A差では2位・3位が逆転）だった。これは次に固定検証すべき診断候補であり、今回それを新しい除外閾値にはしない。')
    add('条件の正確な再現','各開催13試合の未丸め確率を使用。qは試合の確率値そのものではなく、開催内13値の線形補間分位点。P00≤Q(q)、P11>Q(1−q)。q=.45で4、.5で5、.55で8。8条件はP00≤中央値かつP11≥中央値と一致。中央値は7番目の実在値。厳格4は両境界を除外、中央5はP00境界のみ含む。距離は中央値−P00、P11−中央値（表は百分率ポイント）。',s[['round','match_no','card','tier','actual_score','actual_symbol','margin_P00_pp','margin_P11_pp','rank_P00_ascending','rank_P11_ascending']])
    add('8試合の予測と構造','確率列は0〜1。P_H2plus/P_A2plusは各側2点差以上。P_33plusは3-3以上。全8試合とも基準首位はH/Aで、D首位は採用条件にしていない。',s[['round','match_no','card','actual_score','baseline_symbol','baseline_P_H','baseline_P_D','baseline_P_A','PD_rank','lambda_home','lambda_away','T','log_R','HA_gap','P_0_0','P_1_1','P_2_2','P_33plus','P_home_1','P_away_1','P_H2plus','P_A2plus']])
    detail=[]
    for r in s.itertuples():
        z=pd.read_csv(OUT/'sensitivity_selected_8.csv')
        z=z[(z['round']==r.round)&(z.match_no==r.match_no)&z.epsilon.eq(.1)]
        low=float(z.loc[z.scenario.eq('総得点低下'),'delta_P_D'].iloc[0])*100
        balance=float(z.loc[z.scenario.eq('総得点低下・拮抗'),'delta_P_D'].iloc[0])*100
        detail.append(f'{r.round} No.{r.match_no} {r.card}（{r.actual_score} {r.actual_symbol}）：基準{r.baseline_symbol}、P(D)={100*r.baseline_P_D:.2f}%・開催内{r.PD_rank:.0f}位。T={r.T:.3f}、logR={r.log_R:.3f}、1点差合計{100*r.P_onegoal:.2f}%。Tを10%下げるとfinal Dは{low:+.3f}ポイント、低下＋拮抗では{balance:+.3f}ポイント。D首位か否かで候補から落とす処理は行わない。')
    add('各試合の感応度と判断材料','<br><br>'.join(detail))
    add('追加4試合と反例','千葉－長崎1–1、名古屋－C大阪1–1、町田－福岡3–0は両距離が0。神戸－名古屋3–1はP00距離だけ0。境界距離だけでは同じ座標にD2・非D1が重なるため識別不能。非D2試合はH優勢が相対的に強いが、1点差・同点の確率は残っており、この条件はH勝利を否定しない。実スコア3–0/3–1は照合にのみ使用し、条件に逆流させていない。',non[['card','baseline_P_H','baseline_P_D','baseline_P_A','HA_gap','log_R','P_home_1','P_away_1','P_H2plus','P_A2plus']])
    add('なぜこの形になるか','独立PoissonではP00=exp(−T)/Z、P11=P00×λHλA。λHλA=T²/(4 cosh²(logR/2))。Zは0〜10の保持質量。低P00は相対的に高いTを要求する一方、高P11は得点積・拮抗を要求する。均衡時P11はT≈2で極大となり、高Tを無制限に優遇しない。今回8試合のTは'+f'{s["T"].min():.3f}〜{s["T"].max():.3f}。これはλの非線形な組合せ・開催内比較であり、新しい独立入力ではない。P(D)一つへ集約すると失われる違いを分類している可能性はある。')
    add('他条件との重複', '既存の低/高T・比接近・P22+条件は変更せず再利用。「両方」「片方のみ」を分離した。n=0群の率は欠損とする。',comb)
    add('開催差とモデル版','6開催それぞれに該当試合がある。複数開催のDを拾っている一方、各開催1〜2試合にすぎない。モデル版別の持続性はこの件数から確定できない。',rounds)
    add('P(D)とH/A差を揃えた比較','結果を使わず同一開催内の条件外試合と一対一で対応。PD差≤1ポイント、HA差の差≤5ポイントを事前固定し、正規化L1距離の総和を最小化。該当不能なら未対応のまま。T/|logR|比較は差≤0.15/0.10。これは真の独立検証でも無作為比較でもない。Tと符号付きlogRが完全一致すれば行列も同一であり、独立の構造効果は識別不能。',matches)
    for mode,g in matches.groupby('mode'):
        z=g[g.matched];add(mode+'の集計',f'対応{len(z)}/8。該当側D {int(z.target_result.eq("D").sum())}/{len(z)}、条件外側D {int(z.control_result.eq("D").sum())}/{len(z)}。同一試合は方式間では再使用され得るが各方式・開催内では重複しない。条件外側の実結果を選択に使っていない。')
    add('連続距離と絶対値','全78試合の各指標を結果非依存の四分位区間で表示。これは新しい条件探索ではない。境界に近い試合と遠い試合を同じ方向へ単純順位付けできる証拠はない。今回の6Dは1–1が3、0–0・2–2・3–3が各1で、1–1だけの現象でもない。',cont)
    add('全78の絶対中央値と開催内相対中央値','全78をプールした中央値による診断は、将来の固定絶対閾値として採用していない。異なるモデル版・開催分布が混ざるため、開催内相対条件と同一視しない。',globaltab)
    add('λ感応度','Phase3.5の相対±5%/10%の既存シナリオを再利用。総得点方向と得点比方向は分離。final確率は既存mix固定で評価し、P11はraw。変動幅は仮定であり実際の誤差分布ではない。各8試合×25状態の全詳細はsensitivity_selected_8.csv。',sens[sens.epsilon.eq(.1)])
    add('感応度で非D2例を除外できるか','町田－福岡はT10%低下でDが+1.687ポイント、低下＋拮抗で+1.979ポイントと、Dだった今治－八戸の+1.650/+1.792より大きい。神戸－名古屋も+1.406/+1.590で、Dだった福岡－長崎の+1.299/+1.300より大きい。したがってDの増え方が大きい順に優遇するだけでは、この反例は除けない。福岡－長崎はもともとほぼ均衡しているためH相対上昇・A相対上昇のどちらもDを低下させる。')
    add('連続量の数値的な結論','P00境界距離の低い四分位からD率は15.0%→27.3%→43.8%→10.0%。P11境界距離は15.0%→22.7%→37.5%→20.0%。二軸の正/正は4/4D、ゼロ/ゼロは2/3D、ゼロ/正は0/1D。同じ距離でDと非Dが重なり、遠いほどDとする加点は支持されない。全78共通の絶対中央値では3試合中1Dに変わり、開催内比較の文脈を落とすと再現しない。')
    add('残る仮説・反例','①中程度Tと比較的均衡した得点積の領域でモデルがDを過小評価した可能性：期待約2に対し実6。ただし同じ条件でH勝利2。②H/A非対称性：非D2例が8例内で最も非対称。結果を見て認識した仮説なので、これで除外して6/6を成績としない。③感応度：数学的反応はλから決まり、反応差がD原因だとはいえない。④開催相対順位の偶然：同じ境界値でD/非Dが重なり、絶対条件とは一致しない。新しい複合選考条件の探索は0件。')
    add('購入判断への接続','この条件を満たす試合は、既存PD順位だけなら見落とすDを候補として再検討する根拠候補になる。ただし確率75%への置換は不可。H/Aも残して共同の13試合結果分布を評価する必要がある。今回の条件内でもH勝利2例なので、基準H/Aを一律に捨てない。同時に複数該当した際の優先順位やD複数券の配分は未確定。感応度・H/A逆転候補と合わせて比較する対象は作れるが、新係数・D最低枚数・全件D購入は導入しない。')
    add('独立検証の可否と固定仕様',f'現レジストリは78予測・78結果、保存next_predictionsスナップショットも同じ6開催。利用可能な未使用の完成13試合開催は{audit["independent_new_rounds"]}。既存監査の事後再計算を独立試合前予測と扱わない。future_validation_spec.jsonに未丸め境界・入力・再検証方法を固定した。将来13試合が揃えば試合前に中央値・対象フラグを計算できる。正式処理への接続はしていない。')
    for x,y,name,title in [('margin_P00_pp','margin_P11_pp','margins.svg','境界距離：同じ0,0にDと非Dが重なる'),('T','log_R','structure.svg','得点規模と得点比'),('P_0_0','P_1_1','absolute.svg','絶対確率の構造')]:
        scatter(d,x,y,title,OUT/name);blocks.insert(3,f'<figure><img src="{name}" alt="{title}"><figcaption>{title}。点を指すと試合を表示。</figcaption></figure>')
    links=' '.join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(OUT.glob('*.csv')))
    doc='<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase3.5 高D率条件の構造分析</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;line-height:1.8;color:#233044}h1,h2{color:#183958}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:6px 10px;border-bottom:1px solid #dde3e9;text-align:right}th{background:#eaf0f5}section{margin:40px 0}img{max-width:100%}a{margin-right:16px}</style><h1>Phase 3.5追加研究：高D率条件の構造と識別力</h1><p>研究用・正式モデル変更なし。新しい配券は生成していません。</p>'+''.join(blocks)+'<h2>CSV・固定仕様</h2>'+links+'<p><a href="future_validation_spec.json">将来検証仕様</a><a href="audit.json">監査</a><a href="purchase_connection.md">購入判断メモ</a></p></html>'
    (OUT/'index.html').write_text(doc)
    (OUT/'purchase_connection.md').write_text('''# 購入判断への接続（研究用）\n\n8条件はDの候補検討を促す探索仮説であり、採用命令や確率補正ではない。D首位を要求しない。基準H/A首位を残す価値は元の予測確率とその感応度から比較し、同条件の非D2試合を反例として残す。\n\n開催内に複数該当しても上位k・D本数・券番号を固定しない。単一D券と複数D券の優劣は13試合共同確率と予測誤差に依存するため、今回の実D率だけから配分を決めない。H/A逆転候補も含めた候補集合の比較材料にするが、今回は生成しない。\n\n非D2例のH/A非対称性は今後の診断仮説として記録する。今回の8試合を分け切る追加閾値は作らない。独立した後続開催で固定条件・期待D対実D・H/A結果・感応度を蓄積してから、校正や共同分布の変更に根拠があるか判断する。\n''')
    print(comb[['name','n','D','expected_D']].to_string(index=False))
    print(matches[['mode','target_card','matched','target_result','control_card','control_result']].to_string(index=False))

if __name__=='__main__':render(*run())
