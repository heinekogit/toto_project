#!/usr/bin/env python3
"""Read-only audit of frozen base/Lab probabilities; print reproducible JSON."""
import ast
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / 'data/eval/observation_history'
STAGES = {
    'base': ['prob_elo_home', 'prob_elo_draw', 'prob_elo_away'],
    'lab': ['prob_lab_home', 'prob_lab_draw', 'prob_lab_away'],
    'blend': ['prob_blend_home', 'prob_blend_draw', 'prob_blend_away'],
    'main_proxy': ['prob_main_home', 'prob_main_draw', 'prob_main_away'],
    'final': ['prob_home_win', 'prob_draw', 'prob_away_win'],
}


def pick(p):
    return max(range(3), key=p.__getitem__)


def metrics(rows, stage):
    pairs = [(r['stages'][stage], r['y']) for r in rows]
    n = len(pairs)
    return dict(n=n, hits=sum(pick(p) == y for p, y in pairs),
                draw_picks=sum(pick(p) == 1 for p, y in pairs),
                mean_draw=sum(p[1] for p, y in pairs) / n,
                brier=sum(sum((v - (i == y)) ** 2 for i, v in enumerate(p)) for p, y in pairs) / n,
                logloss=-sum(math.log(max(p[y], 1e-15)) for p, y in pairs) / n)


def changes(rows, before, after):
    changed = [r for r in rows if pick(r['stages'][before]) != pick(r['stages'][after])]
    return dict(changes=len(changed), rescue=sum(pick(r['stages'][after]) == r['y'] for r in changed),
                harm=sum(pick(r['stages'][before]) == r['y'] for r in changed),
                transitions=dict(Counter('HDA'[pick(r['stages'][before])] + '->' + 'HDA'[pick(r['stages'][after])] for r in changed)))


def main():
    with (HISTORY / 'matches.csv').open(encoding='utf-8-sig') as f:
        history = list(csv.DictReader(f))
    rows = []
    for h in history:
        p = json.loads(h['prediction_payload_json'])
        stages = {name: [float(p[k]) for k in keys] for name, keys in STAGES.items()}
        for name, vals in stages.items():
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in vals), (h['round_id'], name)
            assert abs(sum(vals) - 1) < 1e-6, (h['round_id'], name, vals)
        alpha = float(p['lab_prob_blend_alpha'])
        stages['blend_reconstructed'] = [(1-alpha)*b + alpha*l for b,l in zip(stages['base'],stages['lab'])]
        y = {'1': 0, '0': 1, '2': 2}[h['actual_result']]
        rows.append(dict(h=h, p=p, stages=stages, y=y))
    assert len({(r['h']['round_id'], r['h']['match_no']) for r in rows}) == len(rows)
    result = {'n': len(rows), 'actual': dict(Counter('HDA'[r['y']] for r in rows)),
              'stages': {s: metrics(rows, s) for s in STAGES},
              'changes': {a + '->' + b: changes(rows, a, b) for a, b in [('base', 'blend'), ('base', 'final'), ('blend', 'final')]}}
    result['by_round'] = {}
    result['provenance'] = {}
    for round_id in sorted({r['h']['round_id'] for r in rows}):
        part = [r for r in rows if r['h']['round_id'] == round_id]
        result['by_round'][round_id] = {s: metrics(part, s) for s in STAGES}
        manifest_path = ROOT / 'data/eval/toto_rounds' / round_id / 'snapshot/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        provenance = json.loads((HISTORY / 'provenance' / (round_id + '.json')).read_text())
        source = Path(provenance['sources']['predictions']['path'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open(encoding='utf-8-sig') as f:
            snapshot_rows = {r['match_id']: r for r in csv.DictReader(f)}
        for r in part:
            snap = snapshot_rows[r['h']['match_id']]
            for keys in STAGES.values():
                assert all(abs(float(snap[k]) - float(r['p'][k])) < 1e-10 for k in keys)
        earliest = min(datetime.fromisoformat(r['h']['datetime']) for r in part)
        saved = datetime.fromisoformat(manifest['created_at'])
        result['provenance'][round_id] = dict(created_at=manifest['created_at'],
            before_first_match=saved.replace(tzinfo=None) < earliest,
            hash_matches_provenance=digest == provenance['sources']['predictions']['sha256'],
            hash_matches_manifest=digest == manifest['files']['predictions.csv']['sha256'])
    result['by_league'] = {league: {s: metrics([r for r in rows if r['h']['league'] == league], s) for s in STAGES}
                           for league in sorted({r['h']['league'] for r in rows})}
    strict = [r for r in rows if result['provenance'][r['h']['round_id']]['before_first_match']]
    result['prematch_manifest_subset'] = dict(stages={s: metrics(strict,s) for s in STAGES},
        changes=changes(strict,'base','blend'), actual=dict(Counter('HDA'[r['y']] for r in strict)))
    result['reconstructed_blend'] = metrics(rows,'blend_reconstructed')
    result['blend_reconstruction_differences'] = [dict(round_id=r['h']['round_id'],match_no=r['h']['match_no'],
        sign_fix=r['p'].get('final_elo_sign_fix_applied'),
        max_error=max(abs(a-b) for a,b in zip(r['stages']['blend'],r['stages']['blend_reconstructed'])))
        for r in rows if max(abs(a-b) for a,b in zip(r['stages']['blend'],r['stages']['blend_reconstructed']))>1e-9]
    result['weights'] = {k: dict(mean=sum(float(r['p'][k]) for r in rows) / len(rows),
                               min=min(float(r['p'][k]) for r in rows), max=max(float(r['p'][k]) for r in rows))
                         for k in ['weight_base', 'weight_lab', 'weight_main', 'lab_prob_blend_alpha']}
    result['blend_formula_max_error'] = max(abs(r['stages']['blend'][i] -
        ((1-float(r['p']['lab_prob_blend_alpha']))*r['stages']['base'][i] + float(r['p']['lab_prob_blend_alpha'])*r['stages']['lab'][i]))
        for r in rows for i in range(3))
    result['lab_draw_change'] = dict(increased=sum(r['stages']['lab'][1] > r['stages']['base'][1] for r in rows),
        blend_mean_delta=sum(r['stages']['blend'][1]-r['stages']['base'][1] for r in rows)/len(rows))
    result['examples'] = [{k: r['h'][k] for k in ['round_id','match_no','home_team','away_team']} |
        dict(actual='HDA'[r['y']], base=r['stages']['base'], blend=r['stages']['blend'], final=r['stages']['final'])
        for r in rows if pick(r['stages']['base']) != pick(r['stages']['blend'])]
    # Execute only isolated pure functions, never the prediction module's top-level pipeline.
    import numpy as np
    import pandas as pd
    tree = ast.parse((ROOT / 'scripts/11_prediction_01.py').read_text())
    names = {'simulate_lab_matchup', '_safe_float_value', '_clip01', '_bounded_unit', '_avg_scores'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    namespace = dict(np=np, pd=pd, math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<isolated_lab_functions>', 'exec'), namespace)
    discrepancies = []
    for r in rows:
        states = {side: {k[len('state_'):-len('_'+side)]: v for k, v in r['p'].items()
                        if k.startswith('state_') and k.endswith('_'+side)} for side in ['home','away']}
        r['states'] = states
        out = namespace['simulate_lab_matchup'](states['home'], states['away'], r['p'])
        discrepancies.append(max(abs(out['mix_'+side] - r['stages']['lab'][i]) for i, side in enumerate(['home','draw','away'])))
        neutral = dict(r['p'])
        for k in neutral:
            if k.startswith('flab_') and k.endswith('_diff'):
                neutral[k] = 0.0
        out0 = namespace['simulate_lab_matchup'](states['home'], states['away'], neutral)
        r['stages']['lab_flab_zero'] = [out0['mix_'+s] for s in ['home','draw','away']]
        alpha = float(r['p']['lab_prob_blend_alpha'])
        r['stages']['blend_flab_zero_fixed_alpha'] = [(1-alpha)*b + alpha*l for b,l in zip(r['stages']['base'],r['stages']['lab_flab_zero'])]
    result['current_lab_replay_error'] = dict(max=max(discrepancies), matching_rows=sum(x < 1e-9 for x in discrepancies))
    result['diagnostic_flab_zero'] = {s: metrics(rows,s) for s in ['lab_flab_zero','blend_flab_zero_fixed_alpha']}
    result['feature_availability'] = {}
    used = sorted({node.args[0].value for node in ast.walk(next(n for n in selected if n.name == 'simulate_lab_matchup'))
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get'
                   and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value,str)
                   and node.args[0].value.startswith('flab_')})
    for k in used:
        result['feature_availability'][k] = sum(r['p'].get(k) is not None and not pd.isna(r['p'].get(k)) for r in rows)
    decisive = [r for r in rows if r['y'] != 1]
    result['ha_direction_decisive'] = {s: sum((0 if r['stages'][s][0] >= r['stages'][s][2] else 2) == r['y'] for r in decisive)
                                     for s in ['base','lab','blend_reconstructed','blend','final']}
    result['ha_direction_decisive']['n'] = len(decisive)
    result['ha_direction_changes'] = {}
    for stage in ['lab','blend_reconstructed','blend','final']:
        changed = [r for r in decisive if (r['stages']['base'][0] >= r['stages']['base'][2]) != (r['stages'][stage][0] >= r['stages'][stage][2])]
        result['ha_direction_changes'][stage] = dict(changes=len(changed),
            rescue=sum((0 if r['stages'][stage][0]>=r['stages'][stage][2] else 2)==r['y'] for r in changed),
            harm=sum((0 if r['stages']['base'][0]>=r['stages']['base'][2] else 2)==r['y'] for r in changed))
    # Hold the body/state constant: test whether actual Lab volume reaches this function.
    probe = rows[-1]
    variants = []
    for volume in [5.0,20.0]:
        p = dict(probe['p'])
        for side in ['home','away']:
            p['flab_chance_shots_'+side] = volume
            p['flab_chance_attack_count_'+side] = volume*10
            p['flab_chance_allowed_attack_count_'+side] = volume*10
            p['flab_chance_build_rate_'+side] = volume
        p['flab_chance_shots_diff'] = 0.0
        p['flab_chance_attack_count_diff'] = 0.0
        p['flab_chance_allowed_attack_count_diff'] = 0.0
        p['flab_chance_build_rate_diff'] = 0.0
        out=namespace['simulate_lab_matchup'](probe['states']['home'],probe['states']['away'],p)
        variants.append(dict(shots_per_side=volume, mix=[out['mix_'+s] for s in ['home','draw','away']],low_event=out['low_event_score']))
    result['equal_volume_probe_body_fixed']=variants
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
