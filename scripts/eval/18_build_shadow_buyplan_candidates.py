#!/usr/bin/env python3
"""Build two purchasable shadow BuyPlans from one frozen production reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JST = ZoneInfo("Asia/Tokyo")


def norm(value):
    return unicodedata.normalize("NFKC", str(value)).strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command, cwd, env, log):
    with Path(log).open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def round_order(round_id, logical_season):
    table = pd.read_csv(ROOT / "data/manual/toto節リスト.csv", encoding="utf-8-sig")
    number = int(str(round_id).lower().removeprefix("toto"))
    rows = table[
        pd.to_numeric(table["toto_round"], errors="coerce").eq(number)
        & pd.to_numeric(table["season"], errors="coerce").eq(int(logical_season))
    ].copy()
    if len(rows) != 13:
        raise ValueError(f"{round_id}: toto order must contain 13 rows, got {len(rows)}")
    return rows.sort_values("match_no")


def validate_reference(reference, order, round_id):
    predictions = pd.read_csv(reference / "predictions.csv", encoding="utf-8-sig")
    buyplan = pd.read_csv(reference / "buyplan.csv", encoding="utf-8-sig")
    if len(buyplan) != 13:
        raise ValueError("production reference must contain exactly 13 BuyPlan rows")
    expected = {(norm(r.home_team), norm(r.away_team)) for r in order.itertuples()}
    prediction_pairs = predictions.apply(lambda r: (norm(r.home_team), norm(r.away_team)), axis=1)
    selected_predictions = predictions[prediction_pairs.isin(expected)].copy()
    actual = set(selected_predictions.apply(lambda r: (norm(r.home_team), norm(r.away_team)), axis=1))
    if actual != expected or len(selected_predictions) != 13:
        missing = sorted(expected - actual)
        raise ValueError(f"{round_id}: purchase_reference belongs to another round; missing={missing}")
    plan_pairs = {(norm(r.home_team), norm(r.away_team)) for r in buyplan.itertuples()}
    if plan_pairs != expected:
        raise ValueError(f"{round_id}: production BuyPlan belongs to another round")
    if set(pd.to_numeric(buyplan.match_no, errors="coerce")) != set(range(1, 14)):
        raise ValueError("production BuyPlan match numbers are incomplete")
    return selected_predictions, buyplan


def prepare_workspace(work, component_root, prediction_season):
    shutil.copytree(ROOT / "data", work / "data", ignore=shutil.ignore_patterns("reports", "eval", "__pycache__"))
    shutil.copytree(ROOT / "scripts", work / "scripts", ignore=shutil.ignore_patterns("__pycache__", ".venv"))
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, work / path.name)
    (work / "logs").mkdir()
    (work / "data/reports").mkdir(parents=True, exist_ok=True)
    for league in ("j1", "j2"):
        production_path = work / f"data/team_fatigue_scores_{league}_{prediction_season}.csv"
        production = pd.read_csv(production_path, encoding="utf-8-sig")
        components = pd.read_csv(component_root / league / "fatigue_components_shadow.csv", encoding="utf-8-sig")
        keys = ["match_id", "datetime", "home_team", "away_team"]
        replacement = components[keys + ["home_component_total_legacy_scale", "away_component_total_legacy_scale"]]
        merged = production.merge(replacement, on=keys, how="left", validate="one_to_one")
        if merged.home_component_total_legacy_scale.isna().any() or merged.away_component_total_legacy_scale.isna().any():
            raise ValueError(f"{league}: fatigue component shadow did not cover production rows")
        merged["home_fatigue_score"] = merged.pop("home_component_total_legacy_scale")
        merged["away_fatigue_score"] = merged.pop("away_component_total_legacy_scale")
        merged["fatigue_shadow_variant"] = "remove_fixed_away_and_recursive_carry"
        merged.to_csv(production_path, index=False, encoding="utf-8-sig")


def select_round(predictions, source_predictions, order):
    rows = []
    source_by_pair = {(norm(r.home_team), norm(r.away_team)): r for r in source_predictions.itertuples()}
    for match in order.itertuples():
        source = source_by_pair[(norm(match.home_team), norm(match.away_team))]
        hits = predictions[predictions.match_id.astype(str).eq(str(source.match_id))]
        if len(hits) != 1:
            dt = pd.Timestamp(source.datetime)
            hits = predictions[
                predictions.home_team.map(norm).eq(norm(match.home_team))
                & predictions.away_team.map(norm).eq(norm(match.away_team))
                & pd.to_datetime(predictions.datetime).dt.date.eq(dt.date())
            ]
        if len(hits) != 1:
            raise ValueError(f"cannot uniquely select {match.match_no}: {match.home_team}-{match.away_team}")
        row = hits.iloc[0].copy()
        row["match_no"] = int(match.match_no)
        row["home_team"], row["away_team"] = match.home_team, match.away_team
        rows.append(row)
    return pd.DataFrame(rows).sort_values("match_no").reset_index(drop=True)


def ticket_table(label, plan, current_plan=None):
    columns = [f"ticket{i:02d}" for i in range(1, 11)]
    view = plan[["match_no", "home_team", "away_team"] + columns].copy()
    view[columns] = view[columns].astype(str).replace({"1.0": "1", "0.0": "0", "2.0": "2"})
    changed_count = 0
    if current_plan is not None:
        baseline = current_plan.set_index("match_no")[columns].astype(str).replace(
            {"1.0": "1", "0.0": "0", "2.0": "2"}
        )
        for row_index, row in view.iterrows():
            match_no = row["match_no"]
            for column in columns:
                current_value = baseline.loc[match_no, column]
                if row[column] != current_value:
                    changed_count += 1
                    view.at[row_index, column] = (
                        f'<span class="changed" title="現行: {escape(current_value)}">'
                        f'{escape(row[column])}</span>'
                    )
    view.insert(1, "対戦", view.home_team.astype(str) + "－" + view.away_team.astype(str))
    view = view.drop(columns=["home_team", "away_team"]).rename(columns={"match_no": "No."})
    view = view.rename(columns={column: f"候補{index:02d}" for index, column in enumerate(columns, 1)})
    note = f'<p class="diff-count">現行との差分: {changed_count}セル</p>' if current_plan is not None else ""
    return f"<h2>{escape(label)}</h2>{note}" + view.to_html(index=False, escape=False, classes="tickets")


def build_index(round_id, output, plans, metadata):
    sections = []
    current_plan = pd.read_csv(output / "current" / "buyplan.csv", encoding="utf-8-sig")
    for key, label in (("current", "現行BuyPlan"), ("fatigue_structural", "Shadow A：疲労構造修正"),
                       ("fatigue_weather_weak", "Shadow B：疲労構造修正＋荒天弱補正")):
        plan = pd.read_csv(output / key / "buyplan.csv", encoding="utf-8-sig")
        sections.append(f'<p><a href="{key}/buyplan.html">{escape(label)}の詳細画面</a></p>')
        sections.append(ticket_table(label, plan, None if key == "current" else current_plan))
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><title>{escape(round_id)} BuyPlan候補比較</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#172033}}.note{{background:#fff7df;border-left:5px solid #d88b00;padding:14px;line-height:1.65}}table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:28px}}th,td{{border:1px solid #ccd5df;padding:6px;text-align:center}}th{{background:#edf2f7;position:sticky;top:0}}td:nth-child(2){{text-align:left;white-space:nowrap}}h2{{margin-top:30px}}.changed{{display:block;margin:-6px;padding:6px;background:#ffe08a;color:#8a2d00;font-weight:800;box-shadow:inset 0 0 0 2px #e07a00}}.diff-count{{margin:0 0 8px;color:#8a2d00;font-weight:700}}</style>
<h1>{escape(round_id)} BuyPlan候補比較</h1><div class="note"><b>購入候補は3系列です。</b>現行版を正式基準とし、Shadow A/Bは比較用です。各系列は10口で、相互に混ぜて生成していません。Shadow Bの荒天補正は最終確率へ一度だけ適用しています。</div>
<p>生成日時: {escape(metadata['created_at'])} / as-of: {escape(metadata['asof'])}</p>{''.join(sections)}</html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, help="例: toto1653")
    parser.add_argument("--logical-season", type=int, required=True, help="toto節リストのシーズン")
    parser.add_argument("--prediction-season", type=int, default=2026)
    parser.add_argument("--reference", type=Path, default=ROOT / "data/purchase_reference")
    parser.add_argument("--asof", default=datetime.now(JST).date().isoformat())
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    round_id = "toto" + str(args.round).lower().removeprefix("toto")
    output = args.out or ROOT / "data/eval/shadow_buyplans" / round_id / args.asof.replace("-", "")
    if output.exists():
        raise FileExistsError(f"shadow output already exists: {output}")
    order = round_order(round_id, args.logical_season)
    source_predictions, _ = validate_reference(args.reference, order, round_id)
    output.mkdir(parents=True)

    current = output / "current"
    current.mkdir()
    for name in ("predictions.csv", "predictions_buyplan_context.csv", "buyplan.csv", "buyplan.html"):
        path = args.reference / name
        if path.exists():
            shutil.copy2(path, current / name)

    component_root = Path(tempfile.mkdtemp(prefix=f"{round_id}_fatigue_components_", dir="/private/tmp"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    run_command([sys.executable, str(ROOT / "scripts/eval/17_build_fatigue_components_shadow.py"),
                 "--season", str(args.prediction_season), "--leagues", "j1", "j2", "--out", str(component_root)],
                ROOT, env, output / "fatigue_components.log")

    work = Path(tempfile.mkdtemp(prefix=f"{round_id}_shadow_buyplans_", dir="/private/tmp"))
    prepare_workspace(work, component_root, args.prediction_season)
    base_predictor = (work / "scripts/11_prediction_01.py").read_text(encoding="utf-8")
    variants = {
        "fatigue_structural": False,
        "fatigue_weather_weak": True,
    }
    for label, weather_shadow in variants.items():
        variant_dir = output / label
        variant_dir.mkdir()
        predictor = work / "scripts/11_prediction_01.py"
        text = base_predictor
        if weather_shadow:
            marker = "df_pred = add_buyplan_purchase_context(df_pred)"
            if text.count(marker) != 1:
                raise ValueError("weather shadow hook point is not unique")
            hook = ("from weather_style_shadow import apply_weather_uncertainty_shadow\n"
                    "df_pred = apply_weather_uncertainty_shadow(df_pred, max_shift=0.04, "
                    "style_tilt=0.15, side_favourite_only=True)\n")
            text = text.replace(marker, hook + marker)
        predictor.write_text(text, encoding="utf-8")
        run_env = dict(env, SEASON_YEAR=str(args.prediction_season), STATS_ASOF_DATE=args.asof,
                       WEATHER_ASOF_DATE=args.asof, TOTO_ROUND_ID=round_id,
                       ENABLE_ROUND_TYPE_DRAW_CONTROL="0", SKIP_HFA_SELF_CHECK="1",
                       OUTPUT_SNAPSHOT_DIR=str(work / "data/output_snapshots"))
        for league in ("j1", "j2"):
            run_command([sys.executable, "scripts/11_prediction_01.py"], work,
                        dict(run_env, LEAGUE=league), variant_dir / f"prediction_{league}.log")
        all_predictions = pd.concat([
            pd.read_csv(work / f"{league}_{args.prediction_season}_predictions_hfa_on.csv", encoding="utf-8-sig")
            for league in ("j1", "j2")
        ], ignore_index=True)
        selected = select_round(all_predictions, source_predictions, order)
        selected.to_csv(variant_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        selected.to_csv(variant_dir / "predictions_buyplan_context.csv", index=False, encoding="utf-8-sig")
        run_command([sys.executable, "buyplan.py", "--in", str(variant_dir / "predictions.csv"),
                     "--context-csv", str(variant_dir / "predictions_buyplan_context.csv"),
                     "--outdir", str(variant_dir), "--toto-order-csv", str(work / "data/manual/toto節リスト.csv")],
                    work, run_env, variant_dir / "buyplan.log")

    metadata = {
        "schema_version": 1, "round_id": round_id, "created_at": datetime.now(JST).isoformat(timespec="seconds"),
        "asof": args.asof, "logical_season": args.logical_season, "prediction_season": args.prediction_season,
        "production_reference": str(args.reference.resolve()), "production_changed": False,
        "arms": {
            "current": "frozen production BuyPlan",
            "fatigue_structural": "fixed away 4 removed; recursive carry replaced by finite cause carry",
            "fatigue_weather_weak": "fatigue_structural plus side-favourite-only weak weather uncertainty",
        },
        "not_applied": ["fatigue_gap", "fatigue_total", "fatigue_uncertainty", "D-filter"],
        "source_hashes": {name: sha256(args.reference / name) for name in ("predictions.csv", "buyplan.csv")},
        "workspace": str(work), "component_workspace": str(component_root),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    build_index(round_id, output, variants, metadata)
    print(output / "index.html")


if __name__ == "__main__":
    main()
