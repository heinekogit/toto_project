"""Read-only audit of archived prematch H/D/A and conditional H/A forecasts."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import ROOT, sha256
from forecast_registry import DEFAULT_ROOT, read_log, selected_matches


ROUNDS=("toto1660","toto1663","toto1664","toto1665","toto1668","toto1670")
HA_BINS=[.5,.55,.60,.70,.80,1.0000001]
HA_LABELS=["50–55%","55–60%","60–70%","70–80%","80%以上"]


def wilson(successes:int,n:int,confidence=.95):
    if n==0:return np.nan,np.nan
    z=float(norm.ppf(.5+confidence/2))
    center=(successes/n+z*z/(2*n))/(1+z*z/n)
    spread=z*np.sqrt((successes/n)*(1-successes/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return float(center-spread),float(center+spread)


def prepare():
    all_forecasts=read_log(DEFAULT_ROOT/"forecasts.jsonl")
    x=selected_matches().sort_values(["round_id","match_no"]).reset_index(drop=True)
    if x.fixture_id.duplicated().any():raise ValueError("selected registry contains duplicate fixtures")
    count_by_fixture=pd.Series([r["fixture_id"] for r in all_forecasts]).value_counts()
    x["forecast_snapshots_for_fixture"]=x.fixture_id.map(count_by_fixture)
    created=pd.to_datetime(x.forecast_created_at,utc=True,errors="coerce")
    kickoff=pd.to_datetime(x.kickoff,errors="coerce").dt.tz_localize("Asia/Tokyo").dt.tz_convert("UTC")
    x["forecast_before_kickoff"]=created.lt(kickoff)
    p=x[["final_prob_home","final_prob_draw","final_prob_away"]].to_numpy(float)
    x["probability_valid"]=np.isfinite(p).all(axis=1)&(p>0).all(axis=1)&np.isclose(p.sum(axis=1),1,atol=1e-8)
    score_result=np.select([x.actual_home_score.gt(x.actual_away_score),
                            x.actual_home_score.eq(x.actual_away_score),
                            x.actual_home_score.lt(x.actual_away_score)],[1,0,2],default=-1)
    x["result_valid"]=x.actual_result.isin([0,1,2])&x.actual_home_score.notna()&x.actual_away_score.notna()&x.actual_result.eq(score_result)
    x["quality"]="C"
    x["quality_reason"]="結果未登録または必要項目不足"
    valid=x.provenance_class.eq("A")&x.forecast_before_kickoff&x.probability_valid
    x.loc[valid&~x.result_valid,"quality_reason"]="試合前予測は保存済み、結果が未登録"
    x.loc[valid&x.result_valid,"quality"]="A"
    x.loc[valid&x.result_valid,"quality_reason"]="試合前予測・確定得点・結果が一致"
    x.loc[~x.forecast_before_kickoff,"quality_reason"]="予測時刻がキックオフ前でない"
    x.loc[~x.probability_valid,"quality_reason"]="H/D/A確率が不正"
    # B is reserved for evidence-backed as-of replay with an incomplete original
    # manifest; the current selected registry does not contain such a case.
    x["model_group"]=x.model_signature.fillna("unknown")
    actual_map={1:"H",0:"D",2:"A"}
    x["actual_symbol"]=x.actual_result.map(actual_map)
    x["P_H"]=x.final_prob_home
    x["P_D"]=x.final_prob_draw
    x["P_A"]=x.final_prob_away
    x["P_H_given_HA"]=x.P_H/(x.P_H+x.P_A)
    x["P_A_given_HA"]=x.P_A/(x.P_H+x.P_A)
    x["HA_leader"]=np.where(x.P_H>=x.P_A,"H","A")
    x["HA_leader_conditional_probability"]=np.maximum(x.P_H_given_HA,x.P_A_given_HA)
    x["HA_gap_absolute"]=abs(x.P_H-x.P_A)
    x["HA_leader_hit"]=x.HA_leader.eq(x.actual_symbol)
    x["HDA_leader"]=np.array(["H","D","A"])[np.argmax(p,axis=1)]
    x["HDA_leader_hit"]=x.HDA_leader.eq(x.actual_symbol)
    x["HA_band"]=pd.cut(x.HA_leader_conditional_probability,HA_BINS,labels=HA_LABELS,
                         right=False,include_lowest=True)
    return x,all_forecasts


def metrics(g,conditional=False):
    if len(g)==0:return dict(n=0,leader_hit=np.nan,leader_hit_rate=np.nan,brier=np.nan,log_loss=np.nan)
    if conditional:
        y=g.actual_symbol.eq("H").to_numpy(int)
        p=g.P_H_given_HA.to_numpy(float)
        brier=float(np.mean((p-y)**2))
        log_loss=float(np.mean(-(y*np.log(p)+(1-y)*np.log1p(-p))))
        hit=int(g.HA_leader_hit.sum())
    else:
        y=g.actual_symbol.map({"H":0,"D":1,"A":2}).to_numpy(int)
        p=g[["P_H","P_D","P_A"]].to_numpy(float)
        target=np.eye(3)[y]
        brier=float(np.mean(np.sum((p-target)**2,axis=1)))
        log_loss=float(np.mean(-np.log(p[np.arange(len(g)),y])))
        hit=int(g.HDA_leader_hit.sum())
    return dict(n=int(len(g)),leader_hit=hit,leader_hit_rate=hit/len(g),
                brier=brier,log_loss=log_loss)


def aggregate(g,group_col):
    rows=[]
    for key,section in g.groupby(group_col,dropna=False,observed=True):
        nonD=section[section.actual_symbol.isin(["H","A"])]
        rows.append(dict(group=str(key),all_n=len(section),actual_D=int(section.actual_symbol.eq("D").sum()),
                         **{f"HDA_{k}":v for k,v in metrics(section).items()},
                         **{f"HA_{k}":v for k,v in metrics(nonD,True).items()},
                         HA_leader_expected=float(nonD.HA_leader_conditional_probability.sum()),
                         HA_leader_actual=int(nonD.HA_leader_hit.sum()),
                         HA_leader_expected_minus_actual=float(nonD.HA_leader_conditional_probability.sum()-nonD.HA_leader_hit.sum())))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--outdir",type=Path,default=ROOT/"data/reports/buyplan_v42_ha_forecast_audit")
    args=parser.parse_args();out=args.outdir;out.mkdir(parents=True,exist_ok=True)
    x,events=prepare()
    a=x[x.quality.eq("A")].copy()
    if len(a)!=78 or set(a.round_id)!=set(ROUNDS):
        raise ValueError("78 prematch result-linked originals expected; investigate before evaluating")
    ha=a[a.actual_symbol.isin(["H","A"])].copy()
    if len(ha)!=60 or int(ha.HA_leader_hit.sum())!=37:
        raise AssertionError("historical 60/37 H/A cross-check failed")
    selected_cols=["round_id","match_no","fixture_id","league","season","kickoff","home_team","away_team",
        "forecast_created_at","source_file","source_input_available_at","input_path","input_sha256",
        "output_sha256","model_version","model_group","provenance_class","forecast_snapshots_for_fixture",
        "forecast_before_kickoff","probability_valid","result_valid","quality","quality_reason",
        "lambda_home","lambda_away","P_H","P_D","P_A","actual_home_score","actual_away_score",
        "actual_symbol","HDA_leader","HDA_leader_hit","P_H_given_HA","P_A_given_HA","HA_gap_absolute",
        "HA_leader","HA_leader_conditional_probability","HA_leader_hit","HA_band"]
    x[selected_cols].to_csv(out/"match_diagnostics.csv",index=False,encoding="utf-8-sig")
    retrospective=pd.read_csv(ROOT/"data/eval/next_model/correction_history_v1/matches.csv",encoding="utf-8-sig")
    known=set(zip(x.round_id,x.match_no))
    additional=retrospective[[ (round_id,match_no) not in known for round_id,match_no in zip(retrospective.round_id,retrospective.match_no)]].copy()
    if len(retrospective)!=88 or len(additional)!=49 or additional.match_id.duplicated().any():
        raise ValueError("retrospective-history overlap changed; inspect before reporting")
    additional["quality"]="C"
    additional["quality_reason"]="当時の保存予測ではなく事後再計算したλ・確率"
    additional["source_file"]="data/eval/next_model/correction_history_v1/matches.csv"
    additional[["round_id","match_no","match_id","datetime","home_team","away_team",
                "final_lambda_home","final_lambda_away","next_prob_home","next_prob_draw","next_prob_away",
                "actual_result","actual_home_score","actual_away_score","quality","quality_reason","source_file"]
    ].to_csv(out/"retrospective_excluded_49.csv",index=False,encoding="utf-8-sig")
    inventory=pd.DataFrame([
        dict(source=str(DEFAULT_ROOT/"forecasts.jsonl"),role="追記型試合前予測イベント",records=len(events),
             unique_fixtures=x.fixture_id.nunique(),quality="A候補。最初の検証済み予測を各実試合に1件採用"),
        dict(source=str(DEFAULT_ROOT/"results.jsonl"),role="追記型結果・訂正イベント",records=len(read_log(DEFAULT_ROOT/"results.jsonl")),
             unique_fixtures=int(x.result_valid.sum()),quality="最新訂正を採用"),
        dict(source="data/eval/next_model/correction_history_v1/matches.csv",role="事後再計算の比較履歴",records=88,
             unique_fixtures=88,quality="39試合はAと重複、49試合は追加の実試合だが当時保存λではないため精度評価から除外"),
        dict(source="data/repair_backups/**/next_predictions.csv",role="修復バックアップ",records=39,
             unique_fixtures=np.nan,quality="生成時点の原本manifestを確認できず追加A件数へ算入しない"),
    ])
    inventory.to_csv(out/"data_inventory.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([dict(quality=k,matches=int(x.quality.eq(k).sum()),rounds=int(x.loc[x.quality.eq(k),"round_id"].nunique()),
                       first_kickoff=x.loc[x.quality.eq(k),"kickoff"].min() if x.quality.eq(k).any() else "",
                       last_kickoff=x.loc[x.quality.eq(k),"kickoff"].max() if x.quality.eq(k).any() else "")
                  for k in ("A","B","C")]).to_csv(out/"quality_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([
        dict(scope="registry unique fixtures",quality="A",matches=78,detail="試合前保存・結果あり"),
        dict(scope="registry unique fixtures",quality="B",matches=0,detail="該当なし"),
        dict(scope="registry unique fixtures",quality="C",matches=13,detail="第1672回、結果待ち"),
        dict(scope="additional distinct retrospective matches",quality="C",matches=49,detail="事後再計算、精度評価から除外"),
        dict(scope="known distinct fixtures total",quality="A/B/C",matches=140,detail="78+13+49。バックアップの重複不明分は算入しない")
    ]).to_csv(out/"quality_summary_extended.csv",index=False,encoding="utf-8-sig")
    overall=aggregate(a.assign(all_matches="全78試合"),"all_matches")
    overall.to_csv(out/"overall_metrics.csv",index=False,encoding="utf-8-sig")
    version=aggregate(a,"model_group");version.to_csv(out/"model_version_metrics.csv",index=False,encoding="utf-8-sig")
    league=aggregate(a,"league");league.to_csv(out/"league_metrics.csv",index=False,encoding="utf-8-sig")
    season=aggregate(a,"season");season.to_csv(out/"season_metrics.csv",index=False,encoding="utf-8-sig")
    bands=[]
    for label in HA_LABELS:
        part=ha[ha.HA_band.eq(label)]
        n=len(part);hit=int(part.HA_leader_hit.sum())
        lo,hi=wilson(hit,n)
        bands.append(dict(HA_band=label,n=n,expected_mean=float(part.HA_leader_conditional_probability.mean()) if n else np.nan,
            actual_leader_hit=hit,actual_rate=hit/n if n else np.nan,
            actual_minus_expected=(hit/n-float(part.HA_leader_conditional_probability.mean())) if n else np.nan,
            wilson95_low=lo,wilson95_high=hi))
    bands=pd.DataFrame(bands);bands.to_csv(out/"HA_competitiveness_bands.csv",index=False,encoding="utf-8-sig")
    calibration=[]
    bins=np.linspace(0,1,6)
    for symbol in ("H","D","A"):
        for j in range(5):
            part=a[(a[f"P_{symbol}"]>=bins[j])&(a[f"P_{symbol}"]<(bins[j+1] if j<4 else 1.0000001))]
            calibration.append(dict(symbol=symbol,bin=f"{int(bins[j]*100)}–{int(bins[j+1]*100)}%",n=len(part),
                mean_forecast=float(part[f"P_{symbol}"].mean()) if len(part) else np.nan,
                observed_rate=float(part.actual_symbol.eq(symbol).mean()) if len(part) else np.nan))
    pd.DataFrame(calibration).to_csv(out/"HDA_calibration.csv",index=False,encoding="utf-8-sig")
    ha_cal=[]
    for label,part in ha.groupby("HA_band",observed=True):
        ha_cal.append(dict(band=str(label),n=len(part),mean_forecast=float(part.HA_leader_conditional_probability.mean()),
            observed_rate=float(part.HA_leader_hit.mean())))
    pd.DataFrame(ha_cal).to_csv(out/"HA_calibration.csv",index=False,encoding="utf-8-sig")
    audit=dict(registry_forecast_sha256=sha256(DEFAULT_ROOT/"forecasts.jsonl"),
               registry_results_sha256=sha256(DEFAULT_ROOT/"results.jsonl"),
               v42_baseline_sha256=sha256(ROOT/"data/reports/d_first_ha_overlay_toto1672/portfolio_comparison.csv"),
               forecast_events=len(events),unique_fixture_count=x.fixture_id.nunique(),
               extra_snapshots=len(events)-x.fixture_id.nunique(),quality_A=len(a),quality_B=int(x.quality.eq("B").sum()),
               quality_C=int(x.quality.eq("C").sum()),HA_decisions=len(ha),HA_leader_hits=int(ha.HA_leader_hit.sum()),
               additional_retrospective_quality_C=len(additional),known_distinct_fixtures_total=len(x)+len(additional),
               forecast_selection="earliest registered verified prematch snapshot per fixture; latest corrected result",
               input_asof_known=int(x.source_input_available_at.notna().sum()),
               full_model_version_known=int(x.model_version.notna().sum()),
               production_changed=False,formal_buyplan_changed=False,v42_changed=False)
    (out/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    render(out,x,a,ha,overall,version,league,season,bands,audit)
    print(json.dumps({k:audit[k] for k in ("forecast_events","unique_fixture_count","quality_A","quality_B","quality_C","HA_decisions","HA_leader_hits")},ensure_ascii=False))


def render(out,x,a,ha,overall,version,league,season,bands,audit):
    row=overall.iloc[0]
    expected=float(ha.HA_leader_conditional_probability.sum())
    actual=int(ha.HA_leader_hit.sum())
    low,high=wilson(actual,len(ha))
    symbols=[]
    for s in ("H","D","A"):
        symbols.append(dict(symbol=s,forecast_mean=float(a[f"P_{s}"].mean()),forecast_sum=float(a[f"P_{s}"].sum()),
                            actual_count=int(a.actual_symbol.eq(s).sum()),actual_rate=float(a.actual_symbol.eq(s).mean())))
    pd.DataFrame(symbols).to_csv(out/"HDA_symbol_summary.csv",index=False,encoding="utf-8-sig")
    calibration=pd.read_csv(out/"HDA_calibration.csv")
    bars="".join(f"<div><span>{html.escape(str(r.symbol))} {html.escape(str(r['bin']))} (n={int(r.n)})</span><div class='track'><i style='width:{100*r.mean_forecast if pd.notna(r.mean_forecast) else 0:.1f}%;background:#456ab0'></i></div><div class='track'><i style='width:{100*r.observed_rate if pd.notna(r.observed_rate) else 0:.1f}%;background:#e49b38'></i></div></div>" for _,r in calibration[calibration.n.gt(0)].iterrows())
    band_bars="".join(f"<div><span>{r.HA_band} (n={int(r.n)})</span><div class='track'><i style='width:{100*r.expected_mean if pd.notna(r.expected_mean) else 0:.1f}%;background:#456ab0'></i></div><div class='track'><i style='width:{100*r.actual_rate if pd.notna(r.actual_rate) else 0:.1f}%;background:#e49b38'></i></div></div>" for _,r in bands[bands.n.gt(0)].iterrows())
    insight=(f"品質Aは{len(a)}試合（6開催）。H/A決着は{len(ha)}試合、優勢側的中{actual}試合（{100*actual/len(ha):.1f}%、Wilson 95%区間 {100*low:.1f}–{100*high:.1f}%）。"
             f"モデルが予測した的中数は{expected:.2f}試合。差は{actual-expected:+.2f}試合。"
             "この標本とモデル版混在から、特定の拮抗度で系統的な過信・過小評価は確定できない。")
    report=("# BuyPlan v4.2 H/A予測データの棚卸しと精度診断\n\n"
        "v4.2暫定配券、Next Model、正式購入処理は変更していない。"
        f"予測イベント{audit['forecast_events']}件、実試合{audit['unique_fixture_count']}件。"
        f"試合前保存値と結果が揃う品質Aは{len(a)}試合。Bは{audit['quality_B']}試合。"
        f"第1672回の13試合は結果未登録のためC（将来の照合待ち）。"
        "別の事後再計算履歴88行のうち39試合はAと重複。追加49試合はCとして別表に記録し、精度評価に混ぜていない。"
        "既知の重複を除いた140実試合ではA=78、B=0、C=62。"
        "入力データ個別の利用可能時刻は保存されておらず、試合前予測時刻の証拠とは区別する。\n\n"
        f"{insight}\n\n"
        f"多クラスBrier = {row.HDA_brier:.6f}、多クラスLog Loss = {row.HDA_log_loss:.6f}。"
        f"H/A条件付きBrier = {row.HA_brier:.6f}、Log Loss = {row.HA_log_loss:.6f}。"
        "どちらも小さいほど良い。多クラスBrierは3結果の確率とone-hot結果の二乗差を各試合で合計し平均。"
        "Log Lossは実現結果に割り当てた確率の負の対数の平均。"
        "H/Aの二値BrierとLog Lossは実際にH/Aで決着した60試合だけに限定。Dを除いた条件付き診断であり、全H/D/A性能の代わりにはならない。\n\n"
        "## H/A拮抗度の診断\n\n```csv\n"+bands.to_csv(index=False)+"```\n\n"
        "拮抗度帯の境界は分析用の仮置き。各帯の件数とWilson区間を確認し、空帯や小標本から校正係数・配券閾値を決めない。"
        "校正誤差とモデルパラメータの不確実性は別物。後者をこの78試合だけで安定推定できない。"
        "同一版内で学習・検証を時系列分割するには件数が乏しい。"
        "新しいH/A分岐や確率補正をv4.2へ導入する根拠はまだ不足する。\n\n"
        "## 棚卸しと利用判断\n\n"
        "品質Aの対象期間は2026年10月10日〜11月8日。6開催78試合で、J1が58試合、J2が20試合。"
        "モデル系統はearly_score_model 39試合、absence_uncertainty 26試合、goal_rate_plus_style_policy 13試合。"
        "これは監査用の系統名であり、完全な実装バージョン番号が全件保存されているわけではない。"
        "予測イベント195件のうち、第1672回は同じ13試合に9スナップショットずつ存在する。"
        "評価では各実試合の最初の検証済み試合前スナップショットを採り、結果は最新訂正を採った。"
        "追加49試合は事後再計算なので通常の精度評価に含めない。\n\n"
        "## v4.2への含意\n\n"
        "50〜55%帯は予測52.85%に対し実績48.0%（25試合）、60〜70%帯は予測62.90%に対し実績72.22%（18試合）。"
        "いずれも信頼区間が広く、過信・過小評価の方向を確定できない。"
        "H/A逆側の採用やD配置の変更をこの集計から決めるべきではない。"
        "同じモデル系統の将来試合で予測時点と入力基準時刻を保存し、開催順を守った校正検証を蓄積する。"
        "確率校正と10口の13的中成績は別々に評価する。\n")
    (out/"report.md").write_text(report,encoding="utf-8")
    page=("<!doctype html><html lang='ja'><meta charset='utf-8'><title>v4.2 H/A予測監査</title>"
          "<style>body{font:16px/1.6 system-ui;margin:24px;color:#172033;max-width:1500px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;margin:12px 0}td,th{border:1px solid #cbd5df;padding:6px 9px;text-align:center}th{background:#e8eff7}.note{background:#fff1d6;padding:12px}.track{width:310px;height:12px;background:#edf0f4;margin:3px}.track i{display:block;height:12px}span{font-size:13px}</style>"
          "<h1>BuyPlan v4.2に向けたH/A予測の棚卸し</h1>"
          f"<p class='note'>{html.escape(insight)} v4.2暫定配券と正式処理は変更していません。</p>"
          "<h2>品質分類とデータ所在</h2>"+pd.read_csv(out/"quality_summary_extended.csv").to_html(index=False)
          +"<div class='scroll'>"+pd.read_csv(out/"data_inventory.csv").to_html(index=False)+"</div>"
          "<h2>H/D/A全体</h2>"+pd.read_csv(out/"HDA_symbol_summary.csv").to_html(index=False)
          +"<div class='scroll'>"+overall.to_html(index=False)+"</div>"
          "<h3>校正：青は平均予測、橙は実際の発生率</h3>"+bars
          +"<div class='scroll'>"+calibration.to_html(index=False)+"</div>"
          "<h2>H/A決着だけの条件付き診断</h2>"+bands.to_html(index=False)
          +"<h3>拮抗度帯の校正：青は予測的中率、橙は実際の的中率</h3>"+band_bars
          +"<h2>モデル系統別・リーグ別・年度別</h2><div class='scroll'>"+version.to_html(index=False)+league.to_html(index=False)+season.to_html(index=False)+"</div>"
          +"<p>同一モデル系統の標本は39、26、13試合。入力データ個別のas-of時刻は未保存。帯ごとの見かけの差を新係数の根拠にはできません。BrierとLog Lossは小さいほど良く、H/A条件付き分析にはD試合を含めていません。</p>"
          +"<p><a href='match_diagnostics.csv'>試合別CSV</a>／<a href='retrospective_excluded_49.csv'>事後再計算49試合</a>／<a href='HA_competitiveness_bands.csv'>拮抗度別CSV</a>／<a href='model_version_metrics.csv'>モデル別CSV</a>／<a href='report.md'>詳細レポート</a>／<a href='audit.json'>監査記録</a></p></html>")
    (out/"index.html").write_text(page,encoding="utf-8")


if __name__=="__main__":main()
