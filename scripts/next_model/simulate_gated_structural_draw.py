"""Offline eligibility gates for the corrected non-fixed D placement score."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd

from common import ROOT
from simulate_free_portfolio_1664 import top_joint_candidates
from simulate_free_structural_draw import (
    STRENGTHS, PROB_COLUMNS, structure, placement_scores, metrics, candidates,
    sha, table, page,
)

OUT = ROOT / "data/reports/gated_structural_draw"
ROUNDS = ("toto1670", "toto1664")


@dataclass(frozen=True)
class Gate:
    low_total_min: float = .5
    balance_min: float = .9
    favourite_one_goal_min: float = .6


def eligibility(features, gate):
    """No outcomes, desired D counts or candidate positions enter eligibility."""
    # Under the saved independent Poisson score model, lambda <= 1 on both
    # sides means 0-0 is a joint mode (allowing ties at lambda == 1).
    zero_mode = (features.lambda_home <= 1) & (features.lambda_away <= 1)
    low = features.low_total_le2 >= gate.low_total_min
    balanced = features.full_win_balance >= gate.balance_min
    near = features.favoured_one_goal_share >= gate.favourite_one_goal_min
    eligible = zero_mode | (low & (balanced | near))
    reasons = []
    for i in range(len(features)):
        included = []
        if zero_mode.iloc[i]:
            included.append("0–0が最頻スコア（同率を含む）")
        if low.iloc[i] and balanced.iloc[i]:
            included.append("低得点＋H/A拮抗")
        if low.iloc[i] and near.iloc[i]:
            included.append("低得点＋優勢側の1点差割合")
        if not included:
            included.append("低得点条件に未達" if not low.iloc[i]
                            else "拮抗条件・優勢側1点差条件とも未達")
        reasons.append("／".join(included))
    return eligible.to_numpy(dtype=bool), reasons


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gates = {"ungated": None, "gate_near50": Gate(favourite_one_goal_min=.5),
             "gate_near60": Gate()}
    protected = [p for p in (ROOT / "data/purchase_reference").glob('*') if p.is_file()]
    forecasts = {rid: ROOT / f"data/reports/draw_proximity_fix/corrected/{rid}/forecast_input.csv" for rid in ROUNDS}
    actual_path = ROOT / "data/eval/toto_rounds/toto1664/actual_results.csv"
    protected += list(forecasts.values()) + [actual_path]
    protected += [ROOT / 'scripts/next_model' / f for f in (
        'shadow_v25_operational.py', 'shadow_free_coverage.py', 'score_placement_v25_shadow.py')]
    hashes = {str(p): sha(p) for p in protected}
    all_metrics, all_allocations, all_eligibility, movement_rows = [], [], [], []
    for rid in ROUNDS:
        frame = pd.read_csv(forecasts[rid]).sort_values('match_no').reset_index(drop=True)
        if frame.match_no.tolist() != list(range(1,14)):
            raise ValueError('Unexpected match order')
        p = frame[PROB_COLUMNS].to_numpy(dtype=float)
        original_p = p.copy()
        feat = structure(frame, mode='corrected')
        actual = None
        if rid == 'toto1664':
            results = pd.read_csv(actual_path).sort_values('match_no').reset_index(drop=True)
            for c in ('match_no', 'home_team', 'away_team'):
                if frame[c].tolist() != results[c].tolist():
                    raise ValueError('Actual result identity mismatch')
            raw = results.result.to_numpy(dtype=int)
            actual = np.where(raw == 1, 0, np.where(raw == 0, 1, 2))
        round_dir = OUT / rid
        round_dir.mkdir(exist_ok=True)
        baseline = {}
        round_rows = []
        for name, gate in gates.items():
            eligible, reasons = (np.ones(13, dtype=bool), ['全試合に適用']*13) if gate is None else eligibility(feat, gate)
            support = feat.Q_corrected.to_numpy()*eligible
            eval_table = pd.concat([frame[['match_no', 'home_team', 'away_team']], feat], axis=1)
            eval_table['round'] = rid
            eval_table['gate'] = name
            eval_table['eligible'] = eligible
            eval_table['reason'] = reasons
            eval_table['applied_Q'] = support
            if actual is not None:
                eval_table['actual'] = ['HDA'[s] for s in actual]
            all_eligibility.append(eval_table)
            gate_dir = round_dir / name
            gate_dir.mkdir(exist_ok=True)
            eval_table.to_csv(gate_dir/'eligibility.csv', index=False)
            for strength in STRENGTHS:
                scores = placement_scores(p, support, strength)
                np.testing.assert_array_equal(scores[~eligible,1], p[~eligible,1])
                np.testing.assert_array_equal(scores[:,[0,2]], p[:,[0,2]])
                tickets, _ = top_joint_candidates(scores, 10)
                result, strata = metrics(p, tickets, actual)
                dcounts = (tickets == 1).sum(axis=0)
                if name == 'ungated':
                    baseline[strength] = tickets
                result.update(round=rid, gate=name, strength=strength,
                              eligible_matches=int(eligible.sum()),
                              matches_D_in_all10=int((dcounts == 10).sum()),
                              max_D_copies_per_match=int(dcounts.max()),
                              D_in_ineligible_matches=int(dcounts[~eligible].sum()),
                              actual_D_eligible=None if actual is None else int(((actual==1)&eligible).sum()),
                              actual_D_hit_slots=None if actual is None else int(((tickets==1)&(actual==1)).sum()))
                all_metrics.append(result)
                round_rows.append(result)
                out = gate_dir / f'alpha_{strength:.1f}'
                out.mkdir(exist_ok=True)
                diag = candidates(frame,p,tickets,scores,actual)
                diag.to_csv(out/'candidates.csv', index=False)
                allocation = eval_table[['match_no','home_team','away_team','eligible','reason','applied_Q']].copy()
                for s,label in enumerate('HDA'):
                    allocation[f'P_{label}']=p[:,s]
                    allocation[f'score_{label}']=scores[:,s]
                    allocation[f'count_{label}']=(tickets==s).sum(axis=0)
                for i,ticket in enumerate(tickets,1):
                    allocation[f'candidate{i:02d}']=['HDA'[s] for s in ticket]
                if actual is not None:
                    allocation['actual']=['HDA'[s] for s in actual]
                allocation.to_csv(out/'allocation.csv',index=False)
                all_allocations.append(allocation.assign(round=rid,gate=name,strength=strength))
                pd.DataFrame(strata).to_csv(out/'D_count_conditional_P13.csv',index=False)
                for m,match in frame.iterrows():
                    old_d=np.flatnonzero(baseline[strength][:,m]==1)+1
                    new_d=np.flatnonzero(tickets[:,m]==1)+1
                    movement_rows.append({'round':rid,'gate':name,'strength':strength,'match_no':int(match.match_no),
                                          'home_team':match.home_team,'away_team':match.away_team,
                                          'eligible':bool(eligible[m]),'ungated_D_candidates':str(old_d.tolist()),
                                          'gated_D_candidates':str(new_d.tolist()),'D_count':int(dcounts[m])})
                body='<p>比較用・購入対象外。候補00は元確率最大の参照券です。</p><p><a href="../index.html">対象判定と他の強度へ</a></p>'
                body+='<h2>配券</h2>'+table(allocation)+'<h2>候補別診断</h2>'+table(diag)
                body+='<h2>D数を条件とした13的中確率</h2>'+table(pd.DataFrame(strata))
                (out/'index.html').write_text(page(f'{rid} / {name} / α={strength:g}',body))
            display_cols=['match_no','home_team','away_team','low_total_le2','full_win_balance',
                          'favoured_one_goal_share','eligible','reason','Q_corrected','applied_Q']
            body='<p><a href="../index.html">開催比較へ</a></p>'+table(eval_table[display_cols])
            body+='<ul>'+''.join(f'<li><a href="alpha_{a:.1f}/index.html">α={a:g} の配置</a></li>' for a in STRENGTHS)+'</ul>'
            (gate_dir/'index.html').write_text(page(rid+' / '+name+'：D優遇の対象判定',body))
        np.testing.assert_array_equal(p,original_p)
        display=pd.DataFrame(round_rows)
        display['P_any13']=display.P_any13.map(lambda v:f'{v*100:.6f}%')
        body='<p><a href="../index.html">実験全体へ</a></p>'+table(display)
        body+='<ul>'+''.join(f'<li><a href="{name}/index.html">{name}：対象判定と全候補</a></li>' for name in gates)+'</ul>'
        (round_dir/'index.html').write_text(page(rid+'：D優遇の対象選別比較',body))
        print('[OK]',rid,flush=True)
    for path,digest in hashes.items():
        if sha(path)!=digest:
            raise AssertionError('Protected file changed: '+path)
    summary=pd.DataFrame(all_metrics)
    summary.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(all_allocations,ignore_index=True).to_csv(OUT/'all_allocations.csv',index=False)
    pd.concat(all_eligibility,ignore_index=True).to_csv(OUT/'eligibility.csv',index=False)
    pd.DataFrame(movement_rows).to_csv(OUT/'D_movements.csv',index=False)
    # Reproduce the prior all-match application before interpreting gating.
    old=pd.read_csv(ROOT/'data/reports/draw_proximity_fix/metrics.csv')
    old=old[old.structure_mode.eq('corrected')&old.method.str.startswith('structure_')]
    merged=summary[summary.gate.eq('ungated')].merge(old,on=['round','strength'],suffixes=('_now','_old'))
    if len(merged)!=10:
        raise AssertionError('Missing baseline comparisons')
    np.testing.assert_allclose(merged.P_any13_now,merged.P_any13_old,rtol=1e-12,atol=1e-15)
    np.testing.assert_array_equal(merged.D_counts_now,merged.D_counts_old)
    config={'status':'offline exploratory comparison; no adoption',
            'rule':'zero_zero_is_mode OR (low_total >= cutoff AND (balance >= cutoff OR favourite_one_goal_share >= cutoff))',
            'gates':{k:None if v is None else asdict(v) for k,v in gates.items()},
            'strengths':STRENGTHS,'thresholds_learned':False,'thresholds_optimized_on_outcomes':False,
            'noneligible_policy':'Q boost is zero; original D score remains eligible for selection',
            'forecast_probabilities_changed':False,'D_counts_fixed':False,
            'selection':'same exact top ten placement-score products; no randomness',
            'protected_sha256':hashes,'ungated_baseline_reproduced':True}
    (OUT/'audit.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    display=summary[['round','gate','strength','eligible_matches','D_total','D_counts','matches_D_in_all10',
                     'P_any13','actual_D_eligible','actual_D_hit_slots']].copy()
    display['P_any13']=display.P_any13.map(lambda v:f'{v*100:.6f}%')
    focus=summary[summary['round'].eq('toto1670')&summary.strength.eq(2)][
        ['gate','eligible_matches','D_total','D_counts','matches_D_in_all10','P_any13']].copy()
    focus['P_any13']=focus.P_any13.map(lambda v:f'{v*100:.6f}%')
    body='<h2>第1670回・α=2の比較</h2>'+table(focus)
    elig_all=pd.concat(all_eligibility,ignore_index=True)
    selected=elig_all[elig_all.gate.eq('gate_near60')]
    for rid in ROUNDS:
        chosen=selected[selected['round'].eq(rid)]
        near_pass=int(((chosen.low_total_le2>=.5)&(chosen.favoured_one_goal_share>=.6)).sum())
        body+=f'<p>{rid}：優勢側の1点差割合60%以上と低得点条件を満たす試合は{near_pass}試合。対象に入った理由は試合別表で確認できます。</p>'
    old_draws=selected[selected['round'].eq('toto1664')&selected.actual.eq('D')]
    covered=int(old_draws.eligible.sum())
    body+=f'<p>第1664回の実際のD {len(old_draws)}試合のうち、gate_near60で優遇対象に入ったのは{covered}試合。対象を絞ってD総数が減っただけでは、Dの選別能力が改善したとは言えません。</p>'
    body+='''<h2>実験条件</h2><p>購入対象外。修正済みQのD優遇を、対象試合だけに適用する比較です。</p>
<p>対象外は補正Qをゼロにし、元のD予測確率による配券評価を残します。D採用の禁止ではありません。</p>
<p>条件：0–0が最頻スコア（同率を含む）、または「総得点2点以下の確率≥50%」かつ「決着時のH/A比が45:55以内、または優勢側の勝ちのうち1点差が60%以上」。片側優勢でも1点差条件だけで通過できます。各試合の時間経過の予測ではありません。</p>
<p>gate_near60が上記の試案。gate_near50は1点差条件を50%以上にした感度比較。境界は学習済みではなく解釈可能な暫定条件です。望むD枠数へ合わせる調整は行っていません。</p>
<p>Qの式、元の確率、α、上位10口の探索方式は共通です。対象試合のH/D/A枚数やD総数は固定しません。Dが減って元モデルの13的中確率が上がっただけでは、改善成功と判定しません。</p>'''
    body+='<h2>全設定の結果</h2>'+table(display)
    body+='<ul>'+''.join(f'<li><a href="{rid}/index.html">{rid}：対象試合と全候補</a></li>' for rid in ROUNDS)+'</ul>'
    body+='<p><a href="D_movements.csv">Dの採用候補の移動</a> ／ <a href="eligibility.csv">全試合の対象判定</a> ／ <a href="audit.json">条件・監査</a></p>'
    (OUT/'index.html').write_text(page('非固定式：D優遇を適用する試合の選別',body))
    print(OUT/'index.html')


if __name__=='__main__':
    main()
