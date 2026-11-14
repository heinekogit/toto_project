"""Compare frozen legacy Elo/LAB probabilities with Next Model on six Toto rounds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
ROUNDS = [1660, 1663, 1664, 1665, 1668, 1670]
OUT = ROOT / 'data/reports/legacy_lab_next_78'
PREMATCH_SNAPSHOTS = {
    1660: ('20261010_091425', '20261010_091501'),
    1663: ('20261017_094216', '20261017_094257'),
    1664: ('20261021_090514', '20261021_090620'),
    1665: ('20261024_093733', '20261024_093814'),
    1668: ('20261031_102503', '20261031_102509'),
    1670: ('20261107_075418', '20261107_075448'),
}
SYMBOLS = np.array(['H', 'D', 'A'])


def metric(y, p):
    n1 = int(y.sum()); n0 = len(y)-n1
    rank = rankdata(p, method='average')
    auc = float((rank[y].sum()-n1*(n1+1)/2)/(n1*n0))
    return auc, float(np.mean((p-y)**2)), float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def main():
    provenance = pd.read_csv(ROOT/'data/reports/score_structure_phase4a_audit/forecast_result_join.csv')
    provenance = provenance[(provenance.provenance_class == 'A') & provenance['round'].isin([f'toto{x}' for x in ROUNDS])]
    assert len(provenance)==78 and provenance.prematch_hash_verified.all()
    frames=[]
    for n in ROUNDS:
        rnd=f'toto{n}'
        folder=ROOT/f'data/eval/toto_rounds/{rnd}/snapshot'
        saved=folder/'predictions.csv'
        manifest=json.loads((folder/'manifest.json').read_text())
        assert hashlib.sha256(saved.read_bytes()).hexdigest()==manifest['files']['predictions.csv']['sha256']
        old=pd.read_csv(saved)
        for lg, stamp in zip(('j1','j2'), PREMATCH_SNAPSHOTS[n]):
            candidates=list((ROOT/f'data/output_snapshots/{lg}_2026').glob(f'{stamp}_predictions_legacy_{lg}_2026_predictions.csv'))
            assert len(candidates)==1
            prematch=pd.read_csv(candidates[0])
            common=old.merge(prematch[['match_id','prob_home_win','prob_draw','prob_away_win']],on='match_id',suffixes=('_saved','_prematch'))
            assert len(common)>0
            for col in ('prob_home_win','prob_draw','prob_away_win'):
                assert np.array_equal(common[col+'_saved'].to_numpy(),common[col+'_prematch'].to_numpy())
        g=provenance[provenance['round']==rnd]
        fields=['match_id','home_team','away_team','prob_elo_home','prob_elo_draw','prob_elo_away',
                'prob_lab_home','prob_lab_draw','prob_lab_away','prob_home_win','prob_draw','prob_away_win','lab_prob_blend_alpha']
        z=g.merge(old[fields],on='match_id',suffixes=('','_old'),validate='one_to_one')
        assert len(z)==13
        frames.append(z)
    d=pd.concat(frames,ignore_index=True)
    y=(d.actual_home_score==d.actual_away_score).to_numpy()
    actual=np.where(d.actual_home_score>d.actual_away_score,'H',np.where(y,'D','A'))
    d['actual_symbol']=actual
    methods={
        'legacy_pre_lab': ['prob_elo_home','prob_elo_draw','prob_elo_away'],
        'legacy_lab_only': ['prob_lab_home','prob_lab_draw','prob_lab_away'],
        'legacy_final': ['prob_home_win','prob_draw','prob_away_win'],
        'next_final': ['next_prob_home','next_prob_draw','next_prob_away'],
    }
    summary=[]
    for name, cols in methods.items():
        arr=d[cols].to_numpy(float)
        assert np.isfinite(arr).all() and np.allclose(arr.sum(axis=1),1,atol=1e-9)
        pred=SYMBOLS[arr.argmax(axis=1)]
        d[name+'_top_symbol']=pred
        pD=arr[:,1]
        auc,brier,loss=metric(y,pD)
        row={'method':name,'matches':78,'actual_D':int(y.sum()),'D_top':int((pred=='D').sum()),
             'D_top_hit':int(((pred=='D')&y).sum()),'sum_P_D':float(pD.sum()),
             'D_AUC':auc,'D_Brier':brier,'D_binary_log_loss':loss,
             'HDA_log_loss':float(-np.log(arr[np.arange(78),np.where(actual=='H',0,np.where(actual=='D',1,2))]).mean())}
        for k in (18,30,45):
            order=np.lexsort((d.match_no.to_numpy(),d['round'].to_numpy(),-pD))
            row[f'top_{k}_D_hits']=int(y[order[:k]].sum())
        row['top_3_each_round_D_hits']=int(sum(y[g.nlargest(3,cols[1]).index].sum() for _,g in d.groupby('round')))
        summary.append(row)
    pre=d.legacy_pre_lab_top_symbol.eq('D')
    post=d.legacy_final_top_symbol.eq('D')
    d['LAB_added_D_top']=(~pre)&post
    d['LAB_removed_D_top']=pre&(~post)
    d['elo_lab_top_disagree']=d.legacy_pre_lab_top_symbol.ne(d.legacy_lab_only_top_symbol)
    d['elo_top_correct']=d.legacy_pre_lab_top_symbol.eq(d.actual_symbol)
    d['lab_top_correct']=d.legacy_lab_only_top_symbol.eq(d.actual_symbol)
    d['elo_PD_brier']=(d.prob_elo_draw-y)**2
    d['lab_PD_brier']=(d.prob_lab_draw-y)**2
    d['elo_HDA_log_loss']=-np.log(d[['prob_elo_home','prob_elo_draw','prob_elo_away']].to_numpy()[np.arange(78),np.where(actual=='H',0,np.where(actual=='D',1,2))])
    d['lab_HDA_log_loss']=-np.log(d[['prob_lab_home','prob_lab_draw','prob_lab_away']].to_numpy()[np.arange(78),np.where(actual=='H',0,np.where(actual=='D',1,2))])
    OUT.mkdir(parents=True,exist_ok=True)
    cols=['round','match_no','match_id','home_team','away_team','actual_home_score','actual_away_score','actual_symbol',
          'prob_elo_home','prob_elo_draw','prob_elo_away','prob_lab_home','prob_lab_draw','prob_lab_away',
          'prob_home_win','prob_draw','prob_away_win','next_prob_home','next_prob_draw','next_prob_away',
          'lab_prob_blend_alpha']+[c for c in d if c.endswith(('_top_symbol','_D_top','_disagree','_correct','_brier','_log_loss'))]
    d[cols].to_csv(OUT/'all_78.csv',index=False,encoding='utf-8-sig')
    d.loc[d.LAB_added_D_top|d.LAB_removed_D_top,cols].to_csv(OUT/'D_top_changes.csv',index=False,encoding='utf-8-sig')
    d.loc[d.elo_lab_top_disagree,cols].to_csv(OUT/'elo_lab_disagreements.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    def table(frame):
        cs=list(frame.columns)
        def fmt(v):
            if isinstance(v,(float,np.floating)): return f'{v:.4f}'
            return str(v)
        return '\n'.join(['| '+' | '.join(cs)+' |','| '+' | '.join(['---']*len(cs))+' |']+
                         ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
    changes=d[d.LAB_added_D_top|d.LAB_removed_D_top].copy()
    changes['actual_score']=changes.actual_home_score.astype(int).astype(str)+'–'+changes.actual_away_score.astype(int).astype(str)
    disagreements=d[d.elo_lab_top_disagree]
    report='\n\n'.join([
        '# 旧予想 LAB混合前後・Next Model：同一78試合のD比較',
        '試合前保存Aの6開催78試合を対象。旧probの開催保存CSVは、各開催の最初のキックオフ前に作成されたJ1/J2別スナップショットと確率3列が全行一致することを確認した。実結果は予測後の採点にのみ使用。',
        '## 主結果',
        table(pd.DataFrame(summary)[['method','D_top','D_top_hit','sum_P_D','D_AUC','D_Brier','D_binary_log_loss','HDA_log_loss','top_18_D_hits','top_30_D_hits','top_45_D_hits']]),
        'AUCはD対非Dの順位識別、D BrierとD binary log lossはDの二値確率評価、HDA log lossは3結果全体の評価。小さいBrier/log lossが良い。`D_top`はH/D/AでDが最高確率の試合数。`top_k`は方法間で選出数を揃えた全78試合中P(D)上位kの実D捕捉数であり、配券条件ではない。同率は開催回・試合番号で固定。',
        '## LAB混合でD首位へ変わった試合',
        f"D首位への追加は{int(d.LAB_added_D_top.sum())}試合、そのうち実Dは{int((d.LAB_added_D_top&y).sum())}試合。D首位から外れた試合は{int(d.LAB_removed_D_top.sum())}。",
        table(changes[['round','match_no','home_team','away_team','prob_elo_draw','prob_lab_draw','prob_draw','actual_score','actual_symbol']]),
        '## Elo系とLAB単独の食い違い',
        f"H/D/A首位が食い違う試合は{len(disagreements)}。実結果の記号と一致したのは混合前Elo系{int(disagreements.elo_top_correct.sum())}試合、LAB単独{int(disagreements.lab_top_correct.sum())}試合、どちらも不一致{int((~disagreements.elo_top_correct & ~disagreements.lab_top_correct).sum())}試合。食い違い40試合でのH/D/A log lossはElo系{disagreements.elo_HDA_log_loss.mean():.4f}、LAB単独{disagreements.lab_HDA_log_loss.mean():.4f}。D二値BrierはElo系{disagreements.elo_PD_brier.mean():.4f}、LAB単独{disagreements.lab_PD_brier.mean():.4f}。LAB単独は運用予測ではなく、食い違いの診断用。全事例は `elo_lab_disagreements.csv`。",
        '## 解釈上の制限',
        '旧モデルとNext Modelは別の設計・補正を持つため、数値差をLAB混合だけの因果効果とみなさない。LAB混合前後の比較だけは同一旧モデルの後段処理差。6開催・D18という小標本であり、同じ78試合で方式を調整して有効性を主張しない。LAB単独D確率の合計は37.25試合分、旧最終は27.68、Nextは20.51、実Dは18。これらは校正上の診断であり、D最高確率判定の的中率とは別。',
    ])+'\n'
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print('added',int(d.LAB_added_D_top.sum()),'added_hit',int((d.LAB_added_D_top&y).sum()),'removed',int(d.LAB_removed_D_top.sum()))
    disagreement=d[d.elo_lab_top_disagree]
    print('elo_lab_disagreements',len(disagreement),'elo_correct',int(disagreement.elo_top_correct.sum()),
          'lab_correct',int(disagreement.lab_top_correct.sum()),'D',int(disagreement.actual_symbol.eq('D').sum()))


if __name__=='__main__': main()
