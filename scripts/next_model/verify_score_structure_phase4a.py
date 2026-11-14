"""Verify registry backfill and diagnostic isolation against all six originals."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ROOT,sha256
from compare_d_scoop_uncertainty import ROUNDS
from forecast_registry import DEFAULT_ROOT,read_log,register_forecast,register_results,selected_matches
from diagnose_score_structure_phase4a import diagnostics

OUT=ROOT/'data/reports/score_structure_phase4a_stage1'


def main():
    forecast_file=DEFAULT_ROOT/'forecasts.jsonl';result_file=DEFAULT_ROOT/'results.jsonl'
    before=(sha256(forecast_file),sha256(result_file))
    records=selected_matches();assert len(records)==78 and records.fixture_id.nunique()==78
    assert len(read_log(forecast_file))==78 and len(read_log(result_file))==78
    for rid in ROUNDS:
        base=ROOT/'data/eval/toto_rounds'/rid
        source=base/'snapshot/next_predictions.csv';manifest=base/'snapshot/buyplan_engine_manifest.json'
        actual=base/'actual_results.csv';m=json.loads(manifest.read_text())
        assert m['output_sha256']['next_predictions.csv']==sha256(source)
        assert register_forecast(source,manifest)['added']==0
        assert register_results(actual,rid)['added']==0
        f=pd.read_csv(source).sort_values('match_no').reset_index(drop=True)
        a=pd.read_csv(actual).sort_values('match_no').reset_index(drop=True)
        r=records[records.round_id==rid].sort_values('match_no').reset_index(drop=True)
        assert len(r)==13
        for col,registry_col in [('final_lambda_home','lambda_home'),('final_lambda_away','lambda_away'),
                                 ('next_prob_home','final_prob_home'),('next_prob_draw','final_prob_draw'),
                                 ('next_prob_away','final_prob_away')]:
            assert all(float(x)==float(y) for x,y in zip(f[col],r[registry_col])),(rid,col)
        for col,registry_col in [('home_score','actual_home_score'),('away_score','actual_away_score'),
                                 ('result','actual_result')]:
            assert all(float(x)==float(y) for x,y in zip(a[col],r[registry_col])),(rid,col)
        assert r.forecast_created_at.eq(m['created_at']).all()
        assert r.output_sha256.eq(sha256(source)).all()
    detail=diagnostics()
    assert len(detail)==78
    assert np.allclose(detail.raw_P_D,detail.expected_00+detail.expected_11+detail.expected_22plus,atol=1e-12)
    for stage in ('raw','final'):
        assert np.allclose(detail[[f'{stage}_P_{s}' for s in ('H','D','A')]].sum(axis=1),1,atol=1e-12)
    after=(sha256(forecast_file),sha256(result_file))
    assert before==after
    checks=dict(backfilled_forecasts=78,backfilled_results=78,original_numeric_exact=True,
                reimport_added=0,forecast_log_unchanged_after_result_and_diagnosis=True,
                registry_unchanged_on_rerun=True,raw_draw_partition_valid=True,
                hda_normalized=True,manifest_forecast_hash_verified=True,
                historical_forecast_output_hashes_verified=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
