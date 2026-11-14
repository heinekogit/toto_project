"""Apply the agreed selection policy to the audited 2026-11-03 sources.

Produces research inputs only. Does not change production forecasts or invent
historical availability from the publisher's displayed update date.
"""
import json
import re

import pandas as pd

from common import ROOT, normalize_text


def main():
    folder = ROOT / 'data/reports/style_value_investigation'
    overview = pd.read_csv(folder / 'overview_alignment.csv')
    candidates = pd.read_csv(folder / 'candidate_values.csv', dtype=str).fillna('')
    variants = pd.read_csv(ROOT / 'data/reports/full_style_replay_audit/row_variants.csv')
    coverage = pd.read_csv(ROOT / 'data/reports/full_style_replay_audit/page_coverage.csv')
    assert not overview.duplicated(['team', 'style']).any()
    assert not candidates.duplicated(['team', 'style']).any()
    rows = []
    for _, source in overview.iterrows():
        group = variants[variants.team.eq(source.team) & variants['style'].eq(source['style'])]
        headers = json.loads(group.iloc[0].headers)
        chosen = candidates[candidates.team.eq(source.team) & candidates['style'].eq(source['style'])]
        rate = ('味方シュート率' if source['style'] == 'ラインブレイクラン' else
                '攻撃獲得率' if 'ブロック' in source['style'] else '守備成功率')
        page = coverage[coverage.league.eq('j1') & coverage['style'].eq(source['style'])].iloc[0]
        assert int(page.snapshot_date) == 20261103
        for metric in headers:
            if metric == '指数':
                value, unit, status = source.overview_index, '', 'accepted_overview'
            elif metric == rate:
                value, unit, status = source.overview_rate, '%', 'accepted_overview'
            elif len(chosen) == 1:
                raw = normalize_text(chosen.iloc[0][metric])
                match = re.fullmatch(r'(-?\d[\d,]*(?:\.\d+)?)(%|人|m|m2)?', raw)
                assert match, (source.team, metric, raw)
                value, unit, status = float(match[1].replace(',', '')), match[2] or '', 'accepted_aligned_detail'
            else:
                value, unit, status = None, '%' if '率' in metric else '', 'missing_ambiguous'
            rows.append(dict(team=source.team, style=source['style'], metric=metric,
                             value=value, unit=unit, selection_status=status,
                             first_observed_date='2026-11-03',
                             publisher_displayed_date=str(page.asof_date),
                             historical_asof_verified=False, source_url=source.url,
                             overview_source_sha256=source.source_sha256,
                             summary_source_sha256=page.source_sha256))
    result = pd.DataFrame(rows)
    assert len(result) == 840 and len(overview) == 140
    assert result.value.isna().sum() == 7
    assert not result.duplicated(['team', 'style', 'metric']).any()
    result.to_csv(folder / 'validation_inputs.csv', index=False)
    policy = {
        'policy': 'overview_aligned_v1', 'scope': 'J1 additional seven styles; validation inputs',
        'accepted_fields': int(result.value.notna().sum()), 'ambiguous_fields': 7,
        'selection': 'overview for index/rate; unique aligned summary row for details',
        'ambiguity': 'null; never choose first/last/mean or substitute zero',
        'availability': 'first observed 2026-11-03; no retrospective use based on displayed date',
        'model_missingness': 'omit the unavailable correction term; retain baseline prediction',
        'production_prediction_changed': False,
        'validation': 'current full prediction baseline vs additional-style shadow; same matches and budget; goal MAE, H/D/A log loss and Brier; record lambda and probability deltas; tune on earlier data only',
    }
    (folder / 'input_policy.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2))
    print(f'Validation input fields: {policy["accepted_fields"]} accepted, 7 ambiguous/null; production unchanged.')


if __name__ == '__main__':
    main()
