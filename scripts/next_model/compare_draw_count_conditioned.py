"""Reproduce isolated toto1668 placement comparison; no production writes."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
from draw_count_conditioned import generate
from placement_score import placement_scores

ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / 'data/eval/toto_rounds/toto1668'
OUT = ROOT / 'data/reports/draw_count_conditioned_proposal'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(ROUND/'snapshot/next_predictions.csv').sort_values('match_no')
    old = pd.read_csv(ROUND/'snapshot/buyplan.csv').sort_values('match_no')
    actual = pd.read_csv(ROUND/'actual_results.csv').sort_values('match_no')
    for other in (old, actual):
        for col in ('match_no', 'home_team', 'away_team'):
            assert f[col].tolist() == other[col].tolist(), col
    p = f[['next_prob_home','next_prob_draw','next_prob_away']].to_numpy()
    proposal, audit = generate(p)
    raw = old[[f'ticket{i:02d}' for i in range(1,11)]].to_numpy().T.astype(int)
    old_tickets = np.array([0,1,2])[np.where(raw==1,0,np.where(raw==0,1,2))]
    truth_raw = actual.result.to_numpy().astype(int)
    truth = np.where(truth_raw==1,0,np.where(truth_raw==0,1,2))
    for symbol in range(3):
        np.testing.assert_array_equal((old_tickets==symbol).sum(axis=0),(proposal==symbol).sum(axis=0))
    # Enumerate all 3^13 outcomes, not random samples.
    outcomes = np.indices((3,)*13, dtype=np.int8).reshape(13,-1).T
    mass = np.ones(len(outcomes))
    for j in range(13): mass *= p[j,outcomes[:,j]]
    assert np.isclose(mass.sum(),1)
    comparison = []
    ticket_rows = []
    for label, tickets in [('v2.4保存券',old_tickets),('D本数分布を使う試作',proposal)]:
        max_hits = np.zeros(len(outcomes),dtype=np.int8)
        hits = (tickets==truth).sum(axis=1)
        for i,ticket in enumerate(tickets):
            predicted_hits = (outcomes==ticket).sum(axis=1)
            np.maximum(max_hits,predicted_hits,out=max_hits)
            ticket_rows.append({'方式':label,'候補':f'{i+1:02d}','D本数':int((ticket==1).sum()),
                'モデル上の13的中確率（%）':float(np.prod(p[np.arange(13),ticket])*100),
                '1668実的中数':int(hits[i]),'券':''.join('HDA'[s] for s in ticket)})
        pair_min = min(int((tickets[i]!=tickets[j]).sum()) for i in range(10) for j in range(i))
        comparison.append({'方式':label,'D総枠':int((tickets==1).sum()),'実平均的中':float(hits.mean()),
            '実最大的中':int(hits.max()),'最小券間相違数':pair_min,
            **{f'モデル上の少なくとも1券が{k}的中以上（%）':float(mass[max_hits>=k].sum()*100) for k in (10,11,12,13)}})
    contexts=[]
    for other in ([.7,.15,.15],[.35,.3,.35]):
        q,_,_=placement_scores(np.array([[.4,.25,.35]]+[other]*12))
        contexts.append(float(q[0,1]))
    audit['comparison']=comparison
    audit['unchanged_input_draw_25_percent_old_score_examples']=contexts
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    plan=f[['match_no','home_team','away_team','next_prob_home','next_prob_draw','next_prob_away']].copy()
    for i,row in enumerate(proposal): plan[f'ticket{i+1:02d}']=[1 if s==0 else 0 if s==1 else 2 for s in row]
    plan.to_csv(OUT/'proposal_buyplan.csv',index=False)
    pd.DataFrame(ticket_rows).to_csv(OUT/'ticket_comparison.csv',index=False)
    distribution = pd.DataFrame({'D試合数':range(14),'予測確率（%）':np.array(audit['draw_count_distribution'])*100,
                                 '試作の候補数':audit['draw_count_quotas']})
    report = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>D本数分布を使う配券案</title>
<style>body{font:16px/1.8 sans-serif;max-width:1400px;margin:36px auto;padding:0 24px;color:#203043}table{border-collapse:collapse;margin:20px 0;font-size:14px}td,th{border:1px solid #ccd3db;padding:7px}th{background:#eef3f7}section{overflow:auto}.note{background:#fff3d7;padding:16px}code{background:#eef3f7}</style>
<h1>D本数の分布を使う配券スコア・配置の試作</h1>
<p>正式版v2.4は変更せず、1668回の保存入力で配置のみを比較した。乱数・実結果は候補生成に使用していない。</p>
<h2>修正する考え方</h2><p>Dは1試合ではH/Aより低い確率でも、13試合のうち複数がDとなる確率は高くなり得る。
現行v2.3は log(D/max(H,A)) を13試合内で標準化し、係数0.5でDスコアを変更している。同一のH40%・D25%・A35%でも、他の12試合によりDスコアは約47.5%または10.9%になる。これは較正された予測確率ではない。</p>
<p>試作は①元の予測確率を保持、②試合間独立を仮定してD本数Kの確率分布を計算、③その分布から10候補のD本数枠を決定、④各枠で、残りの採用枚数を満たせる最も確率の高い未使用券を順に選択する。</p>
<p>各試合のH/D/A採用枚数はv2.4と同じ。D本数枠は <code>Σ(候補数[K] − 10×P(K))²</code> を最小化し、合計10券・D総枠一致・配置可能性を満たす整数解を求める。乱数は使わない。</p>
<p>枠内の評価は元の券確率の積で行う。条件付きスコア <code>P(券 | D本数=K) = P(券)/P(K)</code> でも同じK内の順位は一致する。条件付きスコアを無条件の的中確率として表示したり、異なるK間の当たりやすさ比較に使ったりしない。</p>
<p>候補番号は、D本数の出現確率が高い枠から、その枠内で残り枚数の制約を満たす強い券を選んだ順番。番号全体が無条件の強さ順であるとは定義しない。</p>
<h2>Dの分布</h2>'''
    report+=f'<p>予測のD期待試合数は{audit["expected_draw_matches"]:.3f}。各試合を10券へ丸めるとD総枠は{audit["rounded_draw_cells"]}で、1券平均3.7本になる。この丸めによる差も維持した比較である。</p>'
    report+='<section>'+distribution.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</section>'
    report+='<h2>比較結果</h2><section>'+pd.DataFrame(comparison).to_html(index=False,float_format=lambda x:f'{x:.6f}')+'</section>'
    report+='''<p class="note">モデル上の確率は全3¹³通りを列挙した、試合間独立を仮定する計算値。実データでの優位性の実証ではない。1668回の保存スナップショットには試合開始後の保存も含まれるため、この13試合を厳密な試合前検証とは扱わない。実結果は完成した券の集計にのみ使用した。</p>
<p>各試合の採用枚数が同じなら、10券の平均的中数は、どの実結果に対しても必ず同じになる。変わるのは的中数の候補間分布や高的中券の被覆。D本数枠と先頭券の強さを設計した本試作は、高的中確率を全体最適化したものではない。</p>
<h2>候補ごとの比較</h2>'''
    report+='<section>'+pd.DataFrame(ticket_rows).to_html(index=False,float_format=lambda x:f'{x:.7f}')+'</section>'
    report+='''<h2>残る課題</h2><ul><li>近い券に強さが集中し、高的中の対象範囲が狭くなる可能性がある。券間相違数と10〜13的中の確率を併記した。</li>
<li>D本数の枠数は「分布への近さ」を目的にした設計であり、賞金期待値や当選確率を最大化する定理ではない。候補枠の規則自体も比較対象にする。</li>
<li>元のD予測が過大・過小なら、その誤差は配券にも引き継がれる。低D回だけでなく通常・多D回の試合前保存データで検証する。</li>
<li>同じ券の回避を優先するが、残り枚数の制約で避けられない場合は監査記録へ明示する。1668試作に完全重複券はない。</li></ul>
<p>5項目のテストで分布計算、整数枠の最適性、小例での券選択、採用枚数・平均的中の保存、再現性、入力異常を確認済み。</p>
<p><a href="proposal_buyplan.csv">試作配券CSV</a> ／ <a href="audit.json">計算監査JSON</a> ／ <a href="ticket_comparison.csv">候補比較CSV</a></p></html>'''
    (OUT/'index.html').write_text(report)
    print(json.dumps(comparison,ensure_ascii=False,indent=2))
    print(OUT/'index.html')


if __name__=='__main__': main()
