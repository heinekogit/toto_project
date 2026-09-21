#!/usr/bin/env python3
"""Compare stored weather flags with corrected unit-safe rules without rerunning predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weather_rules import STRONG_WIND_THRESHOLD_KMH, adverse_weather_penalty

XG_HEAVY_RAIN = 0.15
XG_RAIN = 0.05
XG_STRONG_WIND = 0.10


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def _xg_penalty(rain, heavy, wind):
    return pd.Series(
        (heavy.astype(float) * XG_HEAVY_RAIN)
        + ((rain & ~heavy).astype(float) * XG_RAIN)
        + (wind.astype(float) * XG_STRONG_WIND),
        index=rain.index,
    )


def build(predictions: Path, output_dir: Path, label: str) -> tuple[Path, Path]:
    data = pd.read_csv(predictions, encoding="utf-8-sig")
    missing = _bool(data.get("weather_missing", pd.Series(False, index=data.index)))
    rain = _bool(data.get("is_rain", pd.Series(False, index=data.index)))
    heavy = _bool(data.get("is_heavy_rain", pd.Series(False, index=data.index)))
    current_wind = _bool(data.get("is_strong_wind", pd.Series(False, index=data.index)))
    wind_kmh = pd.to_numeric(data.get("wind_speed", pd.Series(pd.NA, index=data.index)), errors="coerce")
    corrected_wind = wind_kmh.ge(STRONG_WIND_THRESHOLD_KMH) & wind_kmh.notna() & ~missing

    out = data[[c for c in ["match_id", "datetime", "home_team", "away_team", "predicted_result",
                             "prob_home_win", "prob_draw", "prob_away_win"] if c in data.columns]].copy()
    out["weather_missing"] = missing
    out["wind_speed_kmh"] = wind_kmh
    out["current_strong_wind"] = current_wind
    out["corrected_strong_wind"] = corrected_wind
    out["current_xg_penalty_each"] = _xg_penalty(rain, heavy, current_wind)
    out["corrected_xg_penalty_each"] = _xg_penalty(rain, heavy, corrected_wind)
    out["xg_delta_each"] = out.current_xg_penalty_each - out.corrected_xg_penalty_each
    out["current_adverse_weather_penalty"] = (
        heavy.astype(float) * 0.8 + rain.astype(float) * 0.45 + current_wind.astype(float) * 0.45
    )
    out["corrected_adverse_weather_penalty"] = adverse_weather_penalty(rain, heavy, corrected_wind)
    out["adverse_penalty_delta"] = (
        out.corrected_adverse_weather_penalty - out.current_adverse_weather_penalty
    )
    out["weather_adjustment_status"] = missing.map(
        {True: "unknown_no_adjustment", False: "observed_flags"}
    )
    out["prediction_recheck_required"] = (
        out.current_strong_wind.ne(out.corrected_strong_wind)
        | out.adverse_penalty_delta.abs().gt(1e-12)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{label}_weather_rule_shadow.csv"
    html_path = output_dir / f"{label}_weather_rule_shadow.html"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")
    affected = out[out.prediction_recheck_required].copy()
    affected["対戦"] = affected.home_team + "－" + affected.away_team
    affected["風速"] = affected.wind_speed_kmh.map(lambda v: "—" if pd.isna(v) else f"{v:.1f} km/h")
    affected["強風 現→修"] = affected.apply(
        lambda r: f"{bool(r.current_strong_wind)} → {bool(r.corrected_strong_wind)}", axis=1
    )
    affected["xG戻し/チーム"] = affected.xg_delta_each.map(lambda v: f"{v:+.2f}")
    affected["adverse差"] = affected.adverse_penalty_delta.map(lambda v: f"{v:+.2f}")
    cols = [c for c in ["対戦", "風速", "強風 現→修", "xG戻し/チーム", "adverse差",
                            "predicted_result", "prob_home_win", "prob_draw", "prob_away_win"] if c in affected]
    table = affected[cols].to_html(index=False, escape=True) if not affected.empty else "<p>影響対象なし</p>"
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><title>{label} 天候ルールシャドー</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Hiragino Sans',sans-serif;margin:28px;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7deea;padding:7px}}th{{background:#eaf0f8}}.note{{background:#f2f6fb;padding:14px;border-radius:8px}}</style>
<h1>{label} 天候ルール修正影響</h1><div class="note">対象 {len(out)}試合 / 要再計算 {len(affected)}試合 / 天候不明 {int(missing.sum())}試合 / 修正後強風 {int(corrected_wind.sum())}試合。強風閾値を8m/s相当の28.8km/hへ修正し、大雨は通常雨を包含して二重加算しません。ここでは一次的なxG・adverse差を表示し、保存済み正式予測の確率・記号・buyplanは上書きしていません。</div>{table}
<p>詳細CSV: {csv_path.resolve()}</p></html>"""
    html_path.write_text(html, encoding="utf-8")
    return csv_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/purchase_reference/predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/eval/weather_rule_shadow")
    parser.add_argument("--label", default="current")
    args = parser.parse_args()
    csv_path, html_path = build(args.predictions, args.output_dir, args.label)
    print(f"OK: {csv_path.resolve()}")
    print(f"OK: file://{html_path.resolve()}")
    print("NOTE: 保存済み正式予測・buyplanは変更していません。")


if __name__ == "__main__":
    main()
